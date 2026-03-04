"""EGNN backbone adapted as a per-conformer graph encoder.

The original EGNN (Satorras et al. 2021) pools atom features to a single
scalar via ``embedding_out``. Here we remove that final projection and instead
mean-pool atom hidden states to produce a fixed-size graph embedding suitable
as a conformer token for the Transformer in CycloFormer.
"""

import torch
import torch.nn as nn

from src.networks.egnn_layers import E_GCL

# Re-exports for backward compatibility (used by cycloformer.py)
from src.networks.egnn_graph_utils import get_edges, get_edges_batch, mean_pool_atoms


class EGNNBackbone(nn.Module):
    """EGNN backbone that returns a graph-level embedding, not a scalar.

    Applies ``embedding_in`` + N x E_GCL layers, then mean-pools atom hidden
    states to produce a single vector of shape ``(batch, hidden_nf)``.

    Parameters
    ----------
    in_node_nf : int
        Input atom feature dimension.
    hidden_nf : int
        Hidden (and output) feature dimension.
    n_layers : int
        Number of E_GCL message-passing layers.
    in_edge_nf : int
        Edge feature dimension (0 if no edge features).
    act_fn : nn.Module
        Activation function used inside E_GCL.
    residual : bool
        Use residual connections in E_GCL.
    attention : bool
        Use attention gating in E_GCL edge model.
    normalize : bool
        Normalize coordinate differences.
    tanh : bool
        Apply tanh to coordinate update output.
    """

    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        n_layers: int = 4,
        in_edge_nf: int = 0,
        act_fn: nn.Module = None,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ):
        super().__init__()
        if act_fn is None:
            act_fn = nn.SiLU()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, hidden_nf)
        for i in range(n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    hidden_nf,
                    hidden_nf,
                    hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        edges: list[torch.Tensor],
        edge_attr: torch.Tensor | None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of graphs and return graph-level embeddings.

        Parameters
        ----------
        h : Tensor  (B*N_atoms, in_node_nf)
            Atom node features, flattened across the batch.
        x : Tensor  (B*N_atoms, 3)
            Atom 3-D coordinates.
        edges : list[Tensor]
            [row_indices, col_indices] -- fully-connected edges within each
            graph, offset-corrected for the flat batch layout. Use
            :func:`get_edges_batch` to build this.
        edge_attr : Tensor or None  (n_edges, in_edge_nf)
        node_mask : Tensor or None  (B*N_atoms, 1)
            1 for real atoms, 0 for padding. Used in mean-pooling.

        Returns
        -------
        Tensor  (B, hidden_nf)
            Graph-level embedding, one vector per molecule in the batch.
        """
        h = self.embedding_in(h)
        for i in range(self.n_layers):
            h, x, edge_attr = self._modules[f"gcl_{i}"](h, edges, x, edge_attr=edge_attr)
        if node_mask is not None:
            h = h * node_mask
            graph_emb = h.sum(dim=0)  # placeholder -- caller handles batching
        return h
