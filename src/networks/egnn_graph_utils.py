"""Edge construction and pooling utilities for EGNN graphs."""

import torch


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
