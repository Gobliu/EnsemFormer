"""Conformer-level Transformer encoder and MLP prediction head."""

import torch
import torch.nn as nn


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
        Feed-forward hidden dimension (typically 2-4x d_model).
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
