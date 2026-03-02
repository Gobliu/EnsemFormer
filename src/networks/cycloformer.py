"""CycloFormer: ensemble-of-conformers model for molecular property prediction.

Architecture:
    N conformers → [shared GNN encoder] → conformer embeddings (B, N_conf, d_gnn)
                                                     ↓
                                    [Transformer encoder over conformers]
                                                     ↓
                                      [CLS or mean-pool] → [MLP head] → scalar
"""

import math
import types
import logging

import torch
import torch.nn as nn
from tqdm import tqdm

from models.Wrapper import Module
from src.networks.egnn_encoder import EGNNEncoder, get_edges_batch, mean_pool_atoms
from src.networks.cpmp_encoder import CPMPEncoder
from src.utils import to_device


# ---------------------------------------------------------------------------
# Conformer Transformer
# ---------------------------------------------------------------------------

class ConformerTransformerEncoder(nn.Module):
    """Standard BERT-style Transformer encoder over conformer token sequences.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    d_ff : int
        Feed-forward hidden dimension (typically 2–4× d_model).
    dropout : float
    max_conformers : int
        Maximum number of conformers + 1 (for CLS). Sets positional-encoding
        table size.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        max_conformers: int,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Learned positional encoding
        self.pos_emb = nn.Embedding(max_conformers + 1, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : Tensor  (B, seq_len, d_model)
            Sequence of conformer embeddings + CLS token prepended.
        key_padding_mask : BoolTensor or None  (B, seq_len)
            True for positions to **ignore** (padding). The CLS token position
            should always be False (not masked).

        Returns
        -------
        Tensor  (B, seq_len, d_model)
        """
        B, L, _ = tokens.shape
        pos_ids = torch.arange(L, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = self.dropout(tokens + self.pos_emb(pos_ids))
        return self.transformer(tokens, src_key_padding_mask=key_padding_mask)


# ---------------------------------------------------------------------------
# MLP Head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Two-layer MLP regression head.

    Parameters
    ----------
    d_in : int
    d_hidden : int
    dropout : float
    """

    def __init__(self, d_in: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CycloFormerModule
# ---------------------------------------------------------------------------

class CycloFormerModule(Module):
    """Full CycloFormer model wrapping a shared GNN encoder + conformer Transformer.

    The GNN encoder is called once for all conformers jointly (efficient batching):
    inputs are reshaped from ``(B, N_conf, N_atoms, ...)`` →
    ``(B*N_conf, N_atoms, ...)`` before the GNN call, then reshaped back.

    Parameters
    ----------
    gnn_type : str
        'egnn' or 'cpmp'.
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
        Forwarded to EGNNEncoder or CPMPEncoder.
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
        **gnn_kwargs,
    ):
        super().__init__(device, local_rank)
        self.gnn_type = gnn_type
        self.pooling = pooling

        if gnn_type == "egnn":
            self.gnn_encoder = EGNNEncoder(
                in_node_nf=d_atom,
                hidden_nf=d_gnn,
                **{k: v for k, v in gnn_kwargs.items()
                   if k in ("n_layers", "in_edge_nf", "residual", "attention", "normalize", "tanh")},
            )
        elif gnn_type == "cpmp":
            self.gnn_encoder = CPMPEncoder(
                d_atom=d_atom,
                d_model=d_gnn,
                **{k: v for k, v in gnn_kwargs.items()
                   if k in ("N", "h", "dropout", "lambda_attention", "lambda_distance",
                             "trainable_lambda", "N_dense", "leaky_relu_slope",
                             "aggregation_type", "dense_output_nonlinearity",
                             "distance_matrix_kernel", "use_edge_features",
                             "integrated_distances", "scale_norm", "init_type")},
            )
        else:
            raise ValueError(f"Unknown gnn_type: {gnn_type!r}. Choose 'egnn' or 'cpmp'.")

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
            'coords'     : Tensor (B, N_conf, N_atoms, 3)  — 3D positions
            'conformer_mask' : BoolTensor (B, N_conf)

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

        edges, edge_attr = get_edges_batch(N_atoms, B * N_conf, device=self.device)
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
            'atom_mask'      : BoolTensor (B, N_atoms) — True for real atoms

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

        graph_emb = self.gnn_encoder(src, src_mask, adj_flat, dist_flat, None)  # (B*N_conf, d_gnn)
        return graph_emb.view(B, N_conf, -1)  # (B, N_conf, d_gnn)

    def encode_conformers(self, batch: dict) -> torch.Tensor:
        """Dispatch to the correct encoder and return (B, N_conf, d_model)."""
        if self.gnn_type == "egnn":
            tokens = self._encode_conformers_egnn(batch)
        else:
            tokens = self._encode_conformers_cpmp(batch)
        return self.proj(tokens)  # (B, N_conf, d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict) -> torch.Tensor:
        """Full forward pass.

        Parameters
        ----------
        batch : dict with keys:
            (EGNN) 'node_feat', 'coords', 'conformer_mask'
            (CPMP) 'node_feat', 'adj', 'dist', 'atom_mask', 'conformer_mask'

        Returns
        -------
        Tensor  (B, 1)
        """
        tokens = self.encode_conformers(batch)  # (B, N_conf, d_model)
        B = tokens.size(0)

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, N_conf+1, d_model)

        # Build key_padding_mask: True means IGNORE.
        # CLS token (position 0) is never masked.
        conf_mask = batch.get("conformer_mask")  # (B, N_conf) — True where conformer exists
        if conf_mask is not None:
            # Invert: True = real conformer → want False (don't mask); padding → True (mask)
            padding_mask = ~conf_mask  # (B, N_conf)
            cls_col = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            key_padding_mask = torch.cat([cls_col, padding_mask], dim=1)  # (B, N_conf+1)
        else:
            key_padding_mask = None

        out = self.conformer_encoder(tokens, key_padding_mask)  # (B, N_conf+1, d_model)

        if self.pooling == "cls":
            pooled = out[:, 0, :]
        else:
            # Mean over real conformer positions (exclude CLS)
            pooled = out[:, 1:, :].mean(dim=1)

        return self.head(pooled)  # (B, 1)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def configure_optimizers(self, config: dict):
        """Instantiate AdamW optimizer and cosine-annealing LR scheduler."""
        tc = config["training"]
        self.optimizer = torch.optim.AdamW(
            list(self.gnn_encoder.parameters())
            + list(self.conformer_encoder.parameters())
            + list(self.head.parameters())
            + [self.cls_token]
            + (list(self.proj.parameters()) if not isinstance(self.proj, nn.Identity) else []),
            lr=tc["learning_rate"],
            weight_decay=tc["weight_decay"],
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=tc["epochs"],
        )

    def train_one_epoch(
        self,
        train_dataloader,
        epoch_idx: int,
        grad_scaler,
        callbacks: dict,
        config: dict,
    ) -> torch.Tensor:
        args = types.SimpleNamespace(**config["training"])
        self.gnn_encoder.train()
        self.conformer_encoder.train()
        self.head.train()

        loss_acc = torch.zeros((1,), device=self.device)
        for i, batch in self._get_tqdm(
            train_dataloader,
            desc=f"Epoch {epoch_idx}",
            disable=(args.silent or self.local_rank != 0),
        ):
            batch = to_device(batch, self.device)

            for cb in callbacks.values():
                cb.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.forward(batch)
                target = batch["target"]
                loss = self.loss_fn(pred.flatten(), target.flatten()) / args.accumulate_grad_batches

            loss_acc += loss.detach()
            grad_scaler.scale(loss).backward()

            if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(train_dataloader):
                if args.gradient_clip is not None:
                    grad_scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.gnn_encoder.parameters())
                        + list(self.conformer_encoder.parameters())
                        + list(self.head.parameters()),
                        args.gradient_clip,
                    )
                grad_scaler.step(self.optimizer)
                grad_scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        return loss_acc / len(train_dataloader)

    @torch.inference_mode()
    def evaluate_one_epoch(
        self,
        val_dataloader,
        callbacks: dict,
        config: dict,
    ) -> torch.Tensor:
        args = types.SimpleNamespace(**config["training"])
        self.gnn_encoder.eval()
        self.conformer_encoder.eval()
        self.head.eval()

        loss_acc = torch.zeros((1,), device=self.device)
        for _, batch in self._get_tqdm(
            val_dataloader,
            desc="Evaluation",
            disable=(args.silent or self.local_rank != 0),
        ):
            batch = to_device(batch, self.device)

            for cb in callbacks.values():
                cb.on_batch_start()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred = self.forward(batch)
                target = batch["target"]

                for cb in callbacks.values():
                    cb.on_validation_step(None, target, pred)

                loss = self.loss_fn(pred.flatten(), target.flatten()) / args.accumulate_grad_batches

            loss_acc += loss.detach()

        return loss_acc / len(val_dataloader)

    @torch.inference_mode()
    def predict(self, batch: dict) -> torch.Tensor:
        """Run inference and return raw predictions (B, 1)."""
        self.gnn_encoder.eval()
        self.conformer_encoder.eval()
        self.head.eval()
        batch = to_device(batch, self.device)
        return self.forward(batch)
