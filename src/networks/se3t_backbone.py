"""SE(3)-Transformer backbone adapted as a per-conformer graph encoder.

Wraps the NVIDIA SE3Transformer (with graph pooling) to produce a fixed-size
graph-level embedding per conformer, matching the interface of EGNNBackbone and
CPMPBackbone.

The encoder:
1. Receives atom features + 3D coordinates (same as EGNN).
2. Builds a DGL graph internally (fully connected within each molecule).
3. Runs the SE3Transformer backbone with average pooling on type-0 features.
4. Returns a graph-level embedding of shape ``(B, d_gnn)``.
"""

import torch
import torch.nn as nn

import dgl

from src.se3_transformer_lib.transformer import SE3Transformer
from src.se3_transformer_lib.fiber import Fiber


class SE3TBackbone(nn.Module):
    """SE(3)-Transformer backbone that returns a graph-level embedding.

    Parameters
    ----------
    in_node_nf : int
        Input atom feature dimension (will be projected to ``num_channels``
        type-0 features).
    num_layers : int
        Number of SE3 attention layers.
    num_channels : int
        Number of channels per feature type in hidden layers.
    num_degrees : int
        Number of feature types (degrees 0 .. num_degrees-1).
    num_heads : int
        Number of attention heads.
    channels_div : int
        Channel divisor for value projection inside attention.
    d_gnn : int
        Output embedding dimension.  A linear projection is added if
        ``d_gnn`` differs from the SE3Transformer output size.
    norm : bool
        Apply NormSE3 after each attention block.
    use_layer_norm : bool
        Use layer norm inside radial MLPs.
    low_memory : bool
        Trade speed for lower memory in convolutions.
    """

    def __init__(
        self,
        in_node_nf: int,
        num_layers: int = 4,
        num_channels: int = 32,
        num_degrees: int = 4,
        num_heads: int = 8,
        channels_div: int = 2,
        d_gnn: int = 128,
        norm: bool = True,
        use_layer_norm: bool = True,
        low_memory: bool = True,
        edge_channels: int = 0,
    ):
        super().__init__()
        self.d_gnn = d_gnn

        fiber_in = Fiber({0: in_node_nf})
        fiber_hidden = Fiber.create(num_degrees, num_channels)
        fiber_out = Fiber({0: num_degrees * num_channels})
        fiber_edge = Fiber({0: edge_channels}) if edge_channels > 0 else Fiber({})

        self.se3t = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=num_heads,
            channels_div=channels_div,
            fiber_edge=fiber_edge,
            return_type=0,
            pooling="avg",
            norm=norm,
            use_layer_norm=use_layer_norm,
            tensor_cores=False,
            low_memory=low_memory,
        )

        se3t_out_dim = fiber_out.num_features  # num_degrees * num_channels * 1 (degree-0)
        self.proj = nn.Linear(se3t_out_dim, d_gnn) if se3t_out_dim != d_gnn else nn.Identity()

    def forward(
        self,
        h: torch.Tensor,
        x: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        n_atoms: int | None = None,
        bond_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of molecular graphs.

        Parameters
        ----------
        h : Tensor  (B, N_atoms, in_node_nf)
            Atom node features.  **Not** flattened — each row in batch dim
            is one molecular graph.
        x : Tensor  (B, N_atoms, 3)
            Atom 3-D coordinates.
        node_mask : BoolTensor or None  (B, N_atoms)
            True for real atoms, False for padding.
        n_atoms : int or None
            If all molecules have the same number of atoms (no padding), pass
            the atom count here instead of ``node_mask``.
        bond_type : LongTensor or None  (B, N_atoms, N_atoms)
            Integer bond types (0=none, 1=single, 2=double, 3=triple, 4=aromatic).
            When provided, one-hot encoded and passed as type-0 edge features.

        Returns
        -------
        Tensor  (B, d_gnn)
            Graph-level embedding, one vector per molecule.
        """
        B, N, F = h.shape
        device = h.device

        # Build a batch of fully-connected DGL graphs
        graphs = []
        for i in range(B):
            if node_mask is not None:
                n = int(node_mask[i].sum().item())
            elif n_atoms is not None:
                n = n_atoms
            else:
                n = N
            # Fully-connected (no self-loops)
            src_ids = []
            dst_ids = []
            for a in range(n):
                for b in range(n):
                    if a != b:
                        src_ids.append(a)
                        dst_ids.append(b)
            g = dgl.graph((src_ids, dst_ids), num_nodes=n)
            # Relative positions for edges
            xi = x[i, :n, :]  # (n, 3)
            rel_pos = xi[src_ids] - xi[dst_ids]  # (n_edges, 3)
            g.edata['rel_pos'] = rel_pos

            # Bond type edge features
            if bond_type is not None:
                bt_i = bond_type[i, :n, :n]  # (n, n)
                bt_oh = torch.nn.functional.one_hot(bt_i.clamp(0, 4), num_classes=5).float()
                bt_oh = bt_oh[..., 1:]  # drop class 0 → (n, n, 4)
                g.edata['edge_feat'] = bt_oh[src_ids, dst_ids]  # (n_edges, 4)

            graphs.append(g)

        batched_graph = dgl.batch(graphs).to(device)

        # Collect node features for the batched graph
        if node_mask is not None:
            node_feats_list = [h[i, :int(node_mask[i].sum().item()), :] for i in range(B)]
        else:
            n_per = n_atoms if n_atoms is not None else N
            node_feats_list = [h[i, :n_per, :] for i in range(B)]
        all_node_feats = torch.cat(node_feats_list, dim=0)  # (total_nodes, F)

        # SE3Transformer expects node features as {degree: (N, C, 2d+1)}
        # For type-0 (scalar) input: shape (total_nodes, C, 1)
        node_feats_dict = {'0': all_node_feats.unsqueeze(-1)}

        # Build edge features dict if bond types provided
        edge_feats = None
        if bond_type is not None:
            edge_feats = {'0': batched_graph.edata['edge_feat'].unsqueeze(-1)}

        # Forward through SE3Transformer (with built-in avg pooling → (B, out_features))
        pooled = self.se3t(batched_graph, node_feats_dict, edge_feats=edge_feats)  # (B, se3t_out_dim)
        return self.proj(pooled)  # (B, d_gnn)
