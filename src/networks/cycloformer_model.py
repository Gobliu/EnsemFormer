"""CycloFormer: ensemble-of-conformers model for molecular property prediction.

Architecture:
    N conformers -> [shared GNN encoder] -> conformer embeddings (B, N_conf, d_gnn)
                                                     |
                                    [Transformer encoder over conformers]
                                                     |
                                      [CLS or mean-pool] -> [MLP head] -> scalar
"""

import torch
import torch.nn as nn

from src.module import Module
from src.networks.graph_utils import bond_type_to_one_hot
from src.networks.conformer_transformer import ConformerTransformerEncoder, MLPHead
from src.networks.egnn_backbone import EGNNBackbone, get_edges_batch
from src.networks.cpmp_backbone import CPMPBackbone
from src.networks.se3t_backbone import SE3TBackbone
from src.networks.cycloformer_training import CycloFormerTrainingMixin


class CycloFormerModule(CycloFormerTrainingMixin, Module):
    """Full CycloFormer model wrapping a shared GNN encoder + conformer Transformer.

    The GNN encoder is called once for all conformers jointly (efficient batching):
    inputs are reshaped from ``(B, N_conf, N_atoms, ...)`` ->
    ``(B*N_conf, N_atoms, ...)`` before the GNN call, then reshaped back.

    Parameters
    ----------
    gnn_type : str
        'egnn', 'cpmp', or 'se3t'.
    d_atom : int
        Atom feature dimension from featurization.
    d_gnn : int
        GNN encoder output dimension (hidden_nf for EGNN, d_model for CPMP).
    d_model : int
        Conformer Transformer width. A projection is added if d_gnn != d_model.
    n_tf_heads : int
    n_tf_layers : int
    d_ff : int
        Feed-forward dimension inside the conformer Transformer.
    dropout : float
    pooling : str
        'cls' or 'mean'.
    max_conformers : int
    device : torch.device
    local_rank : int
    gnn_kwargs : dict
        Forwarded to EGNNBackbone, CPMPBackbone, or SE3TBackbone.
    """

    def __init__(
        self,
        gnn_type: str,
        d_atom: int,
        d_gnn: int,
        d_model: int,
        n_tf_heads: int,
        n_tf_layers: int,
        d_ff: int,
        dropout: float,
        pooling: str,
        max_conformers: int,
        device,
        local_rank: int,
        mode: str = "ensemble",
        use_bond_type: bool = False,
        **gnn_kwargs,
    ):
        super().__init__(device, local_rank)
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.mode = mode
        self.use_bond_type = use_bond_type

        if gnn_type == "egnn":
            egnn_kwargs = {k: v for k, v in gnn_kwargs.items()
                          if k in ("n_layers", "in_edge_nf", "residual", "attention", "normalize", "tanh")}
            if use_bond_type:
                egnn_kwargs["in_edge_nf"] = 4
            self.gnn_encoder = EGNNBackbone(
                in_node_nf=d_atom,
                hidden_nf=d_gnn,
                **egnn_kwargs,
            )
        elif gnn_type == "cpmp":
            cpmp_kwargs = {k: v for k, v in gnn_kwargs.items()
                          if k in ("N", "h", "dropout", "lambda_attention", "lambda_distance",
                                    "trainable_lambda", "N_dense", "leaky_relu_slope",
                                    "aggregation_type", "dense_output_nonlinearity",
                                    "distance_matrix_kernel", "use_edge_features",
                                    "integrated_distances", "d_edge", "scale_norm", "init_type")}
            if use_bond_type:
                cpmp_kwargs["use_edge_features"] = True
                cpmp_kwargs["d_edge"] = 4
            self.gnn_encoder = CPMPBackbone(
                d_atom=d_atom,
                d_model=d_gnn,
                **cpmp_kwargs,
            )
        elif gnn_type == "se3t":
            se3t_kwargs = {k: v for k, v in gnn_kwargs.items()
                          if k in ("num_layers", "num_channels", "num_degrees",
                                    "num_heads", "channels_div", "norm",
                                    "use_layer_norm", "low_memory")}
            if use_bond_type:
                se3t_kwargs["edge_channels"] = 4
            self.gnn_encoder = SE3TBackbone(
                in_node_nf=d_atom,
                d_gnn=d_gnn,
                **se3t_kwargs,
            )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type!r}. Choose 'egnn', 'cpmp', or 'se3t'.")

        self.proj = nn.Linear(d_gnn, d_model) if d_gnn != d_model else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.conformer_encoder = ConformerTransformerEncoder(
            d_model=d_model,
            n_heads=n_tf_heads,
            n_layers=n_tf_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_conformers=max_conformers,
        )
        self.head = MLPHead(d_model, d_model // 2, dropout)
        self.loss_fn = nn.MSELoss()

        # Collect all sub-modules as a single nn.ModuleList so save/load works
        self.model = nn.ModuleList([self.gnn_encoder, self.conformer_encoder, self.head])

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_conformers_egnn(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with EGNN via a single batched GNN call.

        Expects batch to contain:
            'node_feat'  : Tensor (B, N_conf, N_atoms, F)
            'coords'     : Tensor (B, N_conf, N_atoms, 3)
            'conformer_mask' : BoolTensor (B, N_conf)
            'bond_type'  : LongTensor (B, N_conf, N_atoms, N_atoms) -- optional

        Returns
        -------
        Tensor  (B, N_conf, d_gnn)
        """
        node_feat = batch["node_feat"]       # (B, N_conf, N_atoms, F)
        coords = batch["coords"]             # (B, N_conf, N_atoms, 3)
        B, N_conf, N_atoms, F = node_feat.shape

        # Flatten: treat each (molecule, conformer) pair as one independent graph
        h = node_feat.view(B * N_conf, N_atoms, F)  # (B*N_conf, N_atoms, F)
        x = coords.view(B * N_conf, N_atoms, 3)     # (B*N_conf, N_atoms, 3)

        # Flat atom layout: (B*N_conf * N_atoms, ...)
        h_flat = h.view(B * N_conf * N_atoms, F)
        x_flat = x.view(B * N_conf * N_atoms, 3)

        edges, _ = get_edges_batch(N_atoms, B * N_conf, device=self.device)

        # Build per-edge bond type features if enabled
        edge_attr = None
        if self.use_bond_type and "bond_type" in batch:
            bt = batch["bond_type"]  # (B, N_conf, N_atoms, N_atoms)
            bt_oh = bond_type_to_one_hot(bt)  # (B, N_conf, N_atoms, N_atoms, 4)
            bt_flat = bt_oh.view(B * N_conf, N_atoms, N_atoms, 4)
            row, col = edges
            graph_ids = row // N_atoms
            local_row = row % N_atoms
            local_col = col % N_atoms
            edge_attr = bt_flat[graph_ids, local_row, local_col]  # (n_edges, 4)

        h_flat = self.gnn_encoder(h_flat, x_flat, edges, edge_attr)  # (B*N_conf*N_atoms, hidden_nf)

        # Mean-pool over atoms for each (batch, conformer) graph
        h_graphs = h_flat.view(B * N_conf, N_atoms, -1).mean(dim=1)  # (B*N_conf, hidden_nf)
        return h_graphs.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def _encode_conformers_cpmp(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with CPMP via a single batched GNN call.

        Expects batch to contain:
            'node_feat'      : Tensor (B, N_conf, N_atoms, F)
            'adj'            : Tensor (B, N_conf, N_atoms, N_atoms)
            'dist'           : Tensor (B, N_conf, N_atoms, N_atoms)
            'conformer_mask' : BoolTensor (B, N_conf)
            'atom_mask'      : BoolTensor (B, N_atoms)
            'bond_type'      : LongTensor (B, N_conf, N_atoms, N_atoms) -- optional

        Returns
        -------
        Tensor  (B, N_conf, d_gnn)
        """
        node_feat = batch["node_feat"]   # (B, N_conf, N_atoms, F)
        adj = batch["adj"]               # (B, N_conf, N_atoms, N_atoms)
        dist = batch["dist"]             # (B, N_conf, N_atoms, N_atoms)
        atom_mask = batch["atom_mask"]   # (B, N_atoms)
        B, N_conf, N_atoms, F = node_feat.shape

        # Flatten conformer dimension: (B*N_conf, N_atoms, ...)
        src = node_feat.view(B * N_conf, N_atoms, F)
        adj_flat = adj.view(B * N_conf, N_atoms, N_atoms)
        dist_flat = dist.view(B * N_conf, N_atoms, N_atoms)

        # Repeat atom mask for each conformer
        src_mask = atom_mask.unsqueeze(1).expand(-1, N_conf, -1).reshape(B * N_conf, N_atoms)

        # Build edge attention features from bond types if enabled
        edges_att = None
        if self.use_bond_type and "bond_type" in batch:
            bt = batch["bond_type"]  # (B, N_conf, N_atoms, N_atoms)
            bt_oh = bond_type_to_one_hot(bt)  # (B, N_conf, N_atoms, N_atoms, 4)
            bt_flat = bt_oh.view(B * N_conf, N_atoms, N_atoms, 4)
            edges_att = bt_flat.permute(0, 3, 1, 2)  # (B*N_conf, 4, N_atoms, N_atoms)

        graph_emb = self.gnn_encoder(src, src_mask, adj_flat, dist_flat, edges_att)  # (B*N_conf, d_gnn)
        return graph_emb.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def _encode_conformers_se3t(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with SE3-Transformer via batched calls.

        Expects batch to contain:
            'node_feat'      : Tensor (B, N_conf, N_atoms, F)
            'coords'         : Tensor (B, N_conf, N_atoms, 3)
            'conformer_mask' : BoolTensor (B, N_conf)
            'atom_mask'      : BoolTensor (B, N_atoms) -- optional
            'bond_type'      : LongTensor (B, N_conf, N_atoms, N_atoms) -- optional

        Returns
        -------
        Tensor  (B, N_conf, d_gnn)
        """
        node_feat = batch["node_feat"]   # (B, N_conf, N_atoms, F)
        coords = batch["coords"]         # (B, N_conf, N_atoms, 3)
        atom_mask = batch.get("atom_mask")  # (B, N_atoms) or None
        B, N_conf, N_atoms, F = node_feat.shape

        # Flatten: treat each (molecule, conformer) pair as one graph
        h = node_feat.view(B * N_conf, N_atoms, F)
        x = coords.view(B * N_conf, N_atoms, 3)

        # Expand atom_mask to cover all conformers
        if atom_mask is not None:
            mask = atom_mask.unsqueeze(1).expand(-1, N_conf, -1).reshape(B * N_conf, N_atoms)
        else:
            mask = None

        # Build bond type tensor if enabled
        bt_flat = None
        if self.use_bond_type and "bond_type" in batch:
            bt = batch["bond_type"]  # (B, N_conf, N_atoms, N_atoms)
            bt_flat = bt.view(B * N_conf, N_atoms, N_atoms)

        graph_emb = self.gnn_encoder(h, x, node_mask=mask, bond_type=bt_flat)  # (B*N_conf, d_gnn)
        return graph_emb.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def encode_conformers(self, batch: dict) -> torch.Tensor:
        """Dispatch to the correct encoder and return (B, N_conf, d_model)."""
        if self.gnn_type == "egnn":
            tokens = self._encode_conformers_egnn(batch)
        elif self.gnn_type == "cpmp":
            tokens = self._encode_conformers_cpmp(batch)
        else:
            tokens = self._encode_conformers_se3t(batch)
        return self.proj(tokens)  # (B, N_conf, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extract_features(self, batch: dict) -> torch.Tensor:
        """Return per-conformer embeddings without prediction.

        Parameters
        ----------
        batch : dict
            Same keys as ``forward()``.

        Returns
        -------
        Tensor  (B, N_conf, d_model)
            Per-conformer feature vectors suitable as input to a downstream
            transformer encoder or any other aggregation module.
        """
        return self.encode_conformers(batch)  # (B, N_conf, d_model)

    def forward(self, batch: dict) -> torch.Tensor:
        """Full forward pass.

        Parameters
        ----------
        batch : dict with keys:
            (EGNN)  'node_feat', 'coords', 'conformer_mask'
            (CPMP)  'node_feat', 'adj', 'dist', 'atom_mask', 'conformer_mask'
            (SE3T)  'node_feat', 'coords', 'conformer_mask', optionally 'atom_mask'

        Returns
        -------
        Tensor  (B, 1)
        """
        tokens = self.encode_conformers(batch)  # (B, N_conf, d_model)
        B = tokens.size(0)

        # ---- Standalone mode: skip conformer transformer ----
        if self.mode == "standalone":
            conf_mask = batch.get("conformer_mask")
            if conf_mask is not None:
                mask = conf_mask.unsqueeze(-1).to(dtype=tokens.dtype)  # (B, N_conf, 1)
                pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = tokens.mean(dim=1)
            return self.head(pooled)  # (B, 1)

        # ---- Ensemble mode: conformer transformer ----
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N_conf+1, d_model)

        # Build key_padding_mask: True means IGNORE.
        conf_mask = batch.get("conformer_mask")  # (B, N_conf) -- True where conformer exists
        if conf_mask is not None:
            padding_mask = ~conf_mask  # (B, N_conf)
            cls_col = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            key_padding_mask = torch.cat([cls_col, padding_mask], dim=1)  # (B, N_conf+1)
        else:
            key_padding_mask = None

        out = self.conformer_encoder(tokens, key_padding_mask)  # (B, N_conf+1, d_model)

        if self.pooling == "cls":
            pooled = out[:, 0, :]
        else:
            pooled = out[:, 1:, :].mean(dim=1)

        return self.head(pooled)  # (B, 1)
