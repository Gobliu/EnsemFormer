"""CPMP graph-transformer backbone adapted as a per-conformer encoder.

The original CPMPGraphTransformer pools atom representations to a scalar via
a Generator head. Here we expose the pre-pooled node matrix and apply simple
mean pooling (or dummy-node readout) to produce a fixed-size conformer
embedding suitable as a token for CycloFormer's conformer Transformer.
"""

import torch
import torch.nn as nn

from src.networks.cpmp_layers import INITERS, Encoder, Embeddings


class CPMPBackbone(nn.Module):
    """CPMP graph-transformer backbone that returns a graph-level embedding.

    Runs the atom-level Encoder (N transformer layers) and then mean-pools
    the node matrix ``(B, N_atoms, d_model)`` -> ``(B, d_model)``.
    Alternatively, if ``aggregation_type='dummy_node'``, it returns the
    embedding of the first (dummy) atom position.

    Parameters
    ----------
    d_atom : int
        Input atom feature dimension.
    d_model : int
        Transformer hidden dimension (also the output embedding size).
    N : int
        Number of encoder layers.
    h : int
        Number of attention heads.
    dropout : float
    lambda_attention : float
    lambda_distance : float
    trainable_lambda : bool
    N_dense : int
        FFN sub-layers per encoder block.
    leaky_relu_slope : float
    aggregation_type : str
        'mean', 'sum', or 'dummy_node'.
    dense_output_nonlinearity : str
    distance_matrix_kernel : str
    use_edge_features : bool
    integrated_distances : bool
    d_edge : int or None
        Edge feature dimension. If None, uses the default (11 or 12).
    scale_norm : bool
    init_type : str
    """

    def __init__(
        self,
        d_atom: int,
        d_model: int = 128,
        N: int = 4,
        h: int = 8,
        dropout: float = 0.1,
        lambda_attention: float = 0.3,
        lambda_distance: float = 0.3,
        trainable_lambda: bool = False,
        N_dense: int = 2,
        leaky_relu_slope: float = 0.0,
        aggregation_type: str = "mean",
        dense_output_nonlinearity: str = "relu",
        distance_matrix_kernel: str = "softmax",
        use_edge_features: bool = False,
        integrated_distances: bool = False,
        d_edge: int | None = None,
        scale_norm: bool = False,
        init_type: str = "uniform",
        one_hot_formal_charge: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.aggregation_type = aggregation_type
        self.d_model = d_model
        self._use_dummy = aggregation_type == "dummy_node"
        self._one_hot_fc = one_hot_formal_charge

        # Formal charge: scalar (index 22) -> 3-dim one-hot expands d_atom by 2
        effective_d_atom = d_atom + 2 if one_hot_formal_charge else d_atom

        # When using dummy_node aggregation, the embedding layer needs d_atom+1
        # (extra column for the dummy node indicator bit, prepended at forward time)
        embed_d_atom = effective_d_atom + 1 if self._use_dummy else effective_d_atom

        attention_kwargs = {
            "h": h,
            "d_model": d_model,
            "dropout": dropout,
            "lambda_attention": lambda_attention,
            "lambda_distance": lambda_distance,
            "trainable_lambda": trainable_lambda,
            "distance_matrix_kernel": distance_matrix_kernel,
            "use_edge_features": use_edge_features,
            "integrated_distances": integrated_distances,
            "d_edge": d_edge,
        }
        feedforward_kwargs = {
            "d_model": d_model,
            "N_dense": N_dense,
            "dropout": dropout,
            "leaky_relu_slope": leaky_relu_slope,
            "dense_output_nonlinearity": dense_output_nonlinearity,
        }
        self.src_embed = Embeddings(d_model, embed_d_atom, dropout)
        self.encoder = Encoder(N, d_model, dropout, scale_norm, attention_kwargs, feedforward_kwargs)

        def _init_fn(m):
            fn = INITERS[init_type]
            for p in m.parameters(recurse=False):
                if p.dim() > 1:
                    fn(p)

        self.apply(_init_fn)

    def _expand_formal_charge(self, src: torch.Tensor) -> torch.Tensor:
        """Expand scalar formal charge (index 22) to 3-dim one-hot in-place.

        Input  src : (B, N_atoms, 25)
        Output src : (B, N_atoms, 27)
        """
        charge = src[:, :, 22]  # (B, N_atoms)
        vals = torch.tensor([-1., 0., 1.], device=src.device, dtype=src.dtype)
        charge_oh = (charge.unsqueeze(-1) == vals).float()  # (B, N_atoms, 3)
        return torch.cat([src[:, :, :22], charge_oh, src[:, :, 23:]], dim=-1)

    def _prepend_dummy_node(self, src, src_mask, adj_matrix, distances_matrix, edges_att):
        """Prepend a dummy (CLS) node at position 0 for dummy_node aggregation."""
        B, N, F = src.shape
        device, dtype = src.device, src.dtype

        # Node features: (B, N, F) -> (B, N+1, F+1)
        real = torch.zeros(B, N, F + 1, device=device, dtype=dtype)
        real[:, :, 1:] = src
        dummy_row = torch.zeros(B, 1, F + 1, device=device, dtype=dtype)
        dummy_row[:, :, 0] = 1.0
        src = torch.cat([dummy_row, real], dim=1)

        # Adjacency: (B, N, N) -> (B, N+1, N+1)
        new_adj = torch.zeros(B, N + 1, N + 1, device=device, dtype=adj_matrix.dtype)
        new_adj[:, 1:, 1:] = adj_matrix

        # Distances: (B, N, N) -> (B, N+1, N+1), dummy distance = 1e6
        new_dist = torch.full((B, N + 1, N + 1), 1e6, device=device, dtype=distances_matrix.dtype)
        new_dist[:, 1:, 1:] = distances_matrix

        # Mask: (B, N) -> (B, N+1), dummy is always "real"
        dummy_mask = torch.ones(B, 1, device=device, dtype=src_mask.dtype)
        src_mask = torch.cat([dummy_mask, src_mask], dim=1)

        # Edge features: expand if present
        if edges_att is not None:
            C = edges_att.shape[1]
            new_edges = torch.zeros(B, C, N + 1, N + 1, device=device, dtype=edges_att.dtype)
            new_edges[:, :, 1:, 1:] = edges_att
            edges_att = new_edges

        return src, src_mask, new_adj, new_dist, edges_att

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        adj_matrix: torch.Tensor,
        distances_matrix: torch.Tensor,
        edges_att=None,
    ) -> torch.Tensor:
        """Encode atom features and return a graph-level embedding.

        Parameters
        ----------
        src : Tensor  (B, N_atoms, d_atom)
            Node feature matrix.
        src_mask : BoolTensor  (B, N_atoms)
            True for real atoms, False for padding.
        adj_matrix : Tensor  (B, N_atoms, N_atoms)
        distances_matrix : Tensor  (B, N_atoms, N_atoms)
        edges_att : Tensor or None

        Returns
        -------
        Tensor  (B, d_model)
            Graph-level conformer embedding.
        """
        if self._one_hot_fc:
            src = self._expand_formal_charge(src)

        if self._use_dummy:
            src, src_mask, adj_matrix, distances_matrix, edges_att = (
                self._prepend_dummy_node(src, src_mask, adj_matrix, distances_matrix, edges_att)
            )

        out = self.encoder(
            self.src_embed(src), src_mask, adj_matrix, distances_matrix, edges_att
        )
        # out: (B, N_atoms, d_model)
        mask = src_mask.to(out.device).unsqueeze(-1).to(dtype=out.dtype)
        out_masked = out * mask

        if self.aggregation_type == "mean":
            out_sum = out_masked.sum(dim=1)
            mask_sum = mask.sum(dim=1)
            return out_sum / mask_sum.clamp(min=1e-6)
        elif self.aggregation_type == "sum":
            return out_masked.sum(dim=1)
        elif self.aggregation_type == "dummy_node":
            return out_masked[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation_type: {self.aggregation_type}")
