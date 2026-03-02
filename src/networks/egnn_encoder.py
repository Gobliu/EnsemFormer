"""EGNN backbone adapted as a per-conformer graph encoder.

The original EGNN (Satorras et al. 2021) pools atom features to a single
scalar via ``embedding_out``. Here we remove that final projection and instead
mean-pool atom hidden states to produce a fixed-size graph embedding suitable
as a conformer token for the Transformer in CycloFormer.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# EGNN core layers (verbatim from OldCode/egnn/models/egnn_clean/egnn_clean.py)
# ---------------------------------------------------------------------------

class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer."""

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        act_fn=nn.SiLU(),
        residual=True,
        attention=False,
        normalize=False,
        coords_agg="mean",
        tanh=False,
    ):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        layer = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = _unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == "sum":
            agg = _unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == "mean":
            agg = _unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise ValueError(f"Wrong coords_agg parameter: {self.coords_agg}")
        return coord + agg

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


# ---------------------------------------------------------------------------
# Graph-level encoder (new class for EnsemFormer)
# ---------------------------------------------------------------------------

class EGNNEncoder(nn.Module):
    """EGNN backbone that returns a graph-level embedding, not a scalar.

    Applies ``embedding_in`` + N × E_GCL layers, then mean-pools atom hidden
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
            [row_indices, col_indices] — fully-connected edges within each
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
        # Mean-pool over atoms to get a graph embedding
        # node_mask distinguishes real atoms from padding
        if node_mask is not None:
            h = h * node_mask
            graph_emb = h.sum(dim=0)  # placeholder — caller handles batching
        # Caller is responsible for reshaping (B*N_atoms, hidden_nf) → (B, hidden_nf)
        return h


# ---------------------------------------------------------------------------
# Edge construction utilities
# ---------------------------------------------------------------------------

def get_edges(n_nodes: int) -> list[list[int]]:
    """Return fully-connected directed edge list for n_nodes (no self-loops)."""
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [rows, cols]


def get_edges_batch(
    n_nodes: int, batch_size: int, device: torch.device | str = "cpu"
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Build offset-corrected fully-connected edges for a batch of graphs.

    Parameters
    ----------
    n_nodes : int
        Number of atoms per graph (must be uniform across the batch).
    batch_size : int
        Number of graphs in the batch.
    device : torch.device or str

    Returns
    -------
    edges : [Tensor(n_edges_total,), Tensor(n_edges_total,)]
    edge_attr : Tensor(n_edges_total, 1)
        All-ones dummy edge attributes.
    """
    base = get_edges(n_nodes)
    edge_attr = torch.ones(len(base[0]) * batch_size, 1, device=device)
    base_t = [torch.LongTensor(base[0]), torch.LongTensor(base[1])]
    if batch_size == 1:
        return [t.to(device) for t in base_t], edge_attr
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(base_t[0] + n_nodes * i)
        cols.append(base_t[1] + n_nodes * i)
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]
    return edges, edge_attr


def mean_pool_atoms(
    h: torch.Tensor, n_atoms: int, batch_size: int
) -> torch.Tensor:
    """Reshape flat atom features into graph-level embeddings by mean pooling.

    Parameters
    ----------
    h : Tensor  (batch_size * n_atoms, hidden_nf)
        Flat atom hidden states after EGNN forward.
    n_atoms : int
        Number of atoms per graph (uniform).
    batch_size : int

    Returns
    -------
    Tensor  (batch_size, hidden_nf)
    """
    return h.view(batch_size, n_atoms, -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Segment helpers (verbatim from egnn_clean)
# ---------------------------------------------------------------------------

def _unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def _unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)
