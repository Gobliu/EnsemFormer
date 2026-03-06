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

from src.networks.graph_utils import bond_type_to_one_hot
from src.networks.conformer_transformer import ConformerTransformerEncoder, MLPHead
from src.networks.egnn_backbone import EGNNBackbone, get_edges_batch
from src.networks.cpmp_backbone import CPMPBackbone
from src.networks.se3t_backbone import SE3TBackbone
from src.networks.cycloformer_training import CycloFormerTrainingMixin


class CycloFormerCore(nn.Module):
    """All trainable parameters and forward logic of CycloFormer in one nn.Module.

    Collecting every learnable component here means a single DDP wrap of this
    object synchronises all gradients during distributed training.

    Parameters
    ----------
    gnn_type : str
        'egnn', 'cpmp', or 'se3t'.
    d_atom : int
        Atom feature dimension from featurization.
    d_gnn : int
        GNN encoder output dimension.
    d_model : int
        Conformer Transformer width.
    n_tf_heads, n_tf_layers : int
        Conformer Transformer hyperparameters.
    dropout : float
    pooling : str
        'cls' or 'mean'.
    max_conformers : int
    mode : str
        'ensemble' or 'standalone'.
    use_bond_type : bool
    gnn_kwargs : dict
        Forwarded to the selected GNN backbone.
    """

    def __init__(
        self,
        gnn_type: str,
        d_atom: int,
        d_gnn: int,
        d_model: int,
        n_tf_heads: int,
        n_tf_layers: int,
        dropout: float,
        pooling: str,
        max_conformers: int,
        mode: str = "ensemble",
        use_bond_type: bool = False,
        **gnn_kwargs,
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.pooling = pooling
        self.mode = mode
        self.use_bond_type = use_bond_type

        if gnn_type == "egnn":
            egnn_kwargs = {k: v for k, v in gnn_kwargs.items()
                          if k in ("n_layers", "in_edge_nf", "residual", "attention", "normalize", "tanh")}
            if use_bond_type:
                egnn_kwargs["in_edge_nf"] = 4
            self.backbone = EGNNBackbone(
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
                                    "integrated_distances", "d_edge", "scale_norm", "init_type",
                                    "one_hot_formal_charge")}
            if use_bond_type:
                cpmp_kwargs["use_edge_features"] = True
                cpmp_kwargs["d_edge"] = 4
            self.backbone = CPMPBackbone(
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
            self.backbone = SE3TBackbone(
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
            dropout=dropout,
            max_conformers=max_conformers,
        )
        self.head = MLPHead(d_model, d_model // 2, dropout)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_conformers_egnn(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with EGNN via a single batched GNN call.

        Expects batch to contain:
            'node_feat'  : Tensor (B, N_conf, N_atoms, F)
            'coords'     : Tensor (B, N_conf, N_atoms, 3)
            'atom_mask'  : BoolTensor (B, N_atoms) -- strongly recommended for padded batches
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

        edges, _ = get_edges_batch(N_atoms, B * N_conf, device=node_feat.device)

        # Strip edges that touch padded atoms so they cannot corrupt real-atom messages
        atom_mask = batch["atom_mask"]  # (B, N_atoms) BoolTensor
        node_valid = atom_mask.unsqueeze(1).expand(-1, N_conf, -1).reshape(B * N_conf * N_atoms)
        row_e, col_e = edges
        keep = node_valid[row_e] & node_valid[col_e]
        edges = [row_e[keep], col_e[keep]]

        # Build per-edge bond type features if enabled
        edge_attr = None
        if self.use_bond_type and "bond_type" in batch:
            bt = batch["bond_type"]  # (B, N_conf, N_atoms, N_atoms)
            bt_oh = bond_type_to_one_hot(bt)  # (B, N_conf, N_atoms, N_atoms, 4)
            bt_flat = bt_oh.view(B * N_conf, N_atoms, N_atoms, 4)
            row_e, col_e = edges
            graph_ids = row_e // N_atoms
            local_row = row_e % N_atoms
            local_col = col_e % N_atoms
            edge_attr = bt_flat[graph_ids, local_row, local_col]  # (n_edges, 4)

        h_flat = self.backbone(h_flat, x_flat, edges, edge_attr)  # (B*N_conf*N_atoms, hidden_nf)

        # Masked mean-pool: average only over real atoms, not padding
        h_graphs = h_flat.view(B * N_conf, N_atoms, -1)  # (B*N_conf, N_atoms, d)
        mask_bc = atom_mask.unsqueeze(1).expand(-1, N_conf, -1).reshape(B * N_conf, N_atoms, 1).to(dtype=h_graphs.dtype)
        h_pooled = (h_graphs * mask_bc).sum(dim=1) / mask_bc.sum(dim=1)
        return h_pooled.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def _encode_conformers_cpmp(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with CPMP via a single batched GNN call.

        Expects batch to contain:
            'node_feat'      : Tensor (B, N_conf, N_atoms, F)
            'adj'            : Tensor (B, N_conf, N_atoms, N_atoms)
            'dist'           : Tensor (B, N_conf, N_atoms, N_atoms)
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

        graph_emb = self.backbone(src, src_mask, adj_flat, dist_flat, edges_att)  # (B*N_conf, d_gnn)
        return graph_emb.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def _encode_conformers_se3t(self, batch: dict) -> torch.Tensor:
        """Encode all conformers with SE3-Transformer via batched calls.

        Expects batch to contain:
            'node_feat'      : Tensor (B, N_conf, N_atoms, F)
            'coords'         : Tensor (B, N_conf, N_atoms, 3)
            'atom_mask'      : BoolTensor (B, N_atoms) -- optional
            'bond_type'      : LongTensor (B, N_conf, N_atoms, N_atoms) -- optional

        Returns
        -------
        Tensor  (B, N_conf, d_gnn)
        """
        node_feat = batch["node_feat"]   # (B, N_conf, N_atoms, F)
        coords = batch["coords"]         # (B, N_conf, N_atoms, 3)
        atom_mask = batch["atom_mask"]  # (B, N_atoms) BoolTensor
        B, N_conf, N_atoms, F = node_feat.shape

        # Flatten: treat each (molecule, conformer) pair as one graph
        h = node_feat.view(B * N_conf, N_atoms, F)
        x = coords.view(B * N_conf, N_atoms, 3)

        # Expand atom_mask to cover all conformers
        mask = atom_mask.unsqueeze(1).expand(-1, N_conf, -1).reshape(B * N_conf, N_atoms)

        # Build bond type tensor if enabled
        bt_flat = None
        if self.use_bond_type and "bond_type" in batch:
            bt = batch["bond_type"]  # (B, N_conf, N_atoms, N_atoms)
            bt_flat = bt.view(B * N_conf, N_atoms, N_atoms)

        graph_emb = self.backbone(h, x, node_mask=mask, bond_type=bt_flat)  # (B*N_conf, d_gnn)
        return graph_emb.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def _encode_conformers(self, batch: dict) -> torch.Tensor:
        """Dispatch to the correct encoder and return (B, N_conf, d_model)."""
        if self.gnn_type == "egnn":
            tokens = self._encode_conformers_egnn(batch)
        elif self.gnn_type == "cpmp":
            tokens = self._encode_conformers_cpmp(batch)
        else:
            tokens = self._encode_conformers_se3t(batch)
        return self.proj(tokens)  # (B, N_conf, d_model)

    def extract_features(self, batch: dict) -> torch.Tensor:
        """Return per-conformer embeddings without prediction.

        Returns
        -------
        Tensor  (B, N_conf, d_model)
        """
        return self._encode_conformers(batch)

    def forward(self, batch: dict) -> torch.Tensor:
        """Full forward pass.

        Parameters
        ----------
        batch : dict with keys:
            (EGNN)  'node_feat', 'coords', 'atom_mask'
            (CPMP)  'node_feat', 'adj', 'dist', 'atom_mask'
            (SE3T)  'node_feat', 'coords', 'atom_mask'

        Returns
        -------
        Tensor  (B, 1)
        """
        tokens = self._encode_conformers(batch)  # (B, N_conf, d_model)
        B = tokens.size(0)

        # ---- Standalone mode: skip conformer transformer ----
        if self.mode == "standalone":
            preds = self.head(tokens)  # (B, N_conf, 1)
            return preds.mean(dim=1)   # (B, 1)

        # ---- Ensemble mode: conformer transformer ----
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N_conf+1, d_model)

        out = self.conformer_encoder(tokens)  # (B, N_conf+1, d_model)

        if self.pooling == "cls":
            pooled = out[:, 0, :]
        else:
            pooled = out[:, 1:, :].mean(dim=1)

        return self.head(pooled)  # (B, 1)


class CycloFormerModule(CycloFormerTrainingMixin):
    """CycloFormer model: thin shell that holds CycloFormerCore and training state.

    All learnable parameters live in ``self.model`` (a CycloFormerCore instance).
    Wrapping ``self.model`` in DDP is sufficient to synchronise every gradient
    in distributed training.

    Parameters
    ----------
    gnn_type : str
        'egnn', 'cpmp', or 'se3t'.
    d_atom : int
    d_gnn : int
    d_model : int
    n_tf_heads, n_tf_layers : int
    dropout : float
    pooling : str
    max_conformers : int
    device : torch.device
    local_rank : int
    mode : str
    use_bond_type : bool
    gnn_kwargs : dict
    """

    def __init__(
        self,
        gnn_type: str,
        d_atom: int,
        d_gnn: int,
        d_model: int,
        n_tf_heads: int,
        n_tf_layers: int,
        dropout: float,
        pooling: str,
        max_conformers: int,
        device,
        local_rank: int,
        mode: str = "ensemble",
        use_bond_type: bool = False,
        **gnn_kwargs,
    ):
        self.device = device
        self.local_rank = local_rank
        self.optimizer = None
        self.lr_scheduler = None
        self.model = CycloFormerCore(
            gnn_type=gnn_type,
            d_atom=d_atom,
            d_gnn=d_gnn,
            d_model=d_model,
            n_tf_heads=n_tf_heads,
            n_tf_layers=n_tf_layers,
            dropout=dropout,
            pooling=pooling,
            max_conformers=max_conformers,
            mode=mode,
            use_bond_type=use_bond_type,
            **gnn_kwargs,
        )
        self.loss_fn = nn.MSELoss()

    @classmethod
    def from_config(cls, config: dict, d_atom: int, device, local_rank: int) -> "CycloFormerModule":
        """Instantiate CycloFormerModule from a config dict.

        Parameters
        ----------
        config : dict
            Full config (must contain 'gnn' and 'conformer_transformer' keys).
        d_atom : int
            Atom feature dimension from the dataset featurizer.
        device : torch.device
        local_rank : int
        """
        gnn_cfg = config["gnn"]
        tf_cfg = config["conformer_transformer"]

        gnn_type = gnn_cfg["type"]
        backbone_cfg: dict = gnn_cfg[gnn_type]

        if gnn_type == "egnn":
            d_gnn = backbone_cfg["hidden_nf"]
        elif gnn_type == "se3t":
            d_gnn = backbone_cfg["num_degrees"] * backbone_cfg["num_channels"]
        else:
            d_gnn = backbone_cfg["d_model"]

        return cls(
            gnn_type=gnn_type,
            d_atom=d_atom,
            d_gnn=d_gnn,
            d_model=tf_cfg["d_model"],
            n_tf_heads=tf_cfg["n_heads"],
            n_tf_layers=tf_cfg["n_layers"],
            dropout=tf_cfg["dropout"],
            pooling=tf_cfg["pooling"],
            max_conformers=tf_cfg["max_conformers"],
            device=device,
            local_rank=local_rank,
            mode=gnn_cfg["mode"],
            use_bond_type=gnn_cfg["use_bond_type"],
            **backbone_cfg,
        )

    def extract_features(self, batch: dict) -> torch.Tensor:
        """Return per-conformer embeddings (B, N_conf, d_model)."""
        # DDP wraps model in .module; unwrap to access CycloFormerCore methods
        core: CycloFormerCore = getattr(self.model, "module", self.model)  # type: ignore[assignment]
        return core.extract_features(batch)

    def forward(self, batch: dict) -> torch.Tensor:
        """Full forward pass → (B, 1)."""
        return self.model(batch)
