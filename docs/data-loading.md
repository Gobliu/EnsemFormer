# Data Loading — Encoder Consumption

This document describes how each GNN encoder consumes the collated batch dict.
It covers the `CycloFormerModule` dispatch layer and per-encoder tensor flow.

> For everything upstream — featurization, trajectory parsing, cache format,
> conformer subsampling, and batch collation — see
> [mol-data-preprocessing.md](mol-data-preprocessing.md).

---

## 1. From Batch Dict to Encoder

After `conformer_collate_fn` (described in `mol-data-preprocessing.md §Batch Collation`),
`CycloFormerModule` receives a single dict of dense padded tensors and dispatches
to the appropriate encoder:

```
batch dict (B, C_max, A_max, …)
        │
        ▼
CycloFormerModule.forward()
        │
        ├─ gnn_type=egnn  ──▶ _encode_conformers_egnn()  ──▶ (B, C_max, d_gnn)
        ├─ gnn_type=cpmp  ──▶ _encode_conformers_cpmp()  ──▶ (B, C_max, d_gnn)
        └─ gnn_type=se3t  ──▶ _encode_conformers_se3t()  ──▶ (B, C_max, d_gnn)
                │
                ▼
        [Transformer over conformers]   (mode=ensemble)
        or direct pool                  (mode=standalone)
                │
                ▼
        [MLP head] → scalar
```

All three paths flatten `(B, C_max)` into a single super-batch before the
backbone and unflatten afterward.

---

## 2. Per-Encoder Details

### 2.1 EGNN (`_encode_conformers_egnn`)

**Required keys:** `node_feat`, `coords`, `atom_mask`

```
node_feat  (B, C_max, A_max, F)  ──┐
coords     (B, C_max, A_max, 3)  ──┤─▶ flatten → (B*C_max, A_max, ...)
                                    │
                                    ▼
                        get_edges_batch()          # fully-connected among real atoms
                        + edge filtering via atom_mask
                                    │
                                    ▼
                           EGNNBackbone.forward()  # E_GCL layers
                                    │              # input: (B*C_max*A_max, F)
                                    ▼
                        masked mean-pool over real atoms
                                    │
                                    ▼
                           (B, C_max, d_gnn)
```

Key details:
- Edges are **fully-connected among real atoms**: `get_edges_batch()` builds a
  uniform `A_max`-node edge list, then edges where either endpoint is a padded
  atom are filtered out via `atom_mask`. This prevents padded nodes from
  corrupting real-atom message passing.
- EGNN updates **coordinates** as well as node hidden states; only node states
  are returned and masked mean-pooled over real atoms.
- `in_edge_nf=0` by default — no edge features are passed (`edge_attr=None`).
  The dummy all-ones tensor returned by `get_edges_batch()` is not used.

---

### 2.2 CPMP (`_encode_conformers_cpmp`)

**Required keys:** `node_feat`, `adj`, `dist`, `atom_mask`, `conformer_mask`

```
node_feat  (B, C_max, A_max, F)                 ──┐
adj        (B, C_max, A_max, A_max)             ──┤
dist       (B, C_max, A_max, A_max)             ──┤─▶ flatten → (B*C_max, ...)
atom_mask  (B, A_max)  expand to (B*C_max, …)  ──┘
                                    │
                                    ▼
                          CPMPBackbone.forward()
                            Embeddings (Linear)
                                    │
                            Encoder (N × EncoderLayer)
                            ┌── MultiHeadedAttention
                            │     weights: λ_attn·p_attn + λ_dist·p_dist + λ_adj·p_adj
                            │     (QK dot-product + softmax(−dist) + adj)
                            └── PositionwiseFeedForward
                                    │
                                    ▼
                            atom-masked aggregation
                            ('mean' | 'sum' | 'dummy_node')
                                    │
                                    ▼
                           (B, C_max, d_gnn)
```

Key details:
- `adj` and `dist` enter the **attention score** directly alongside the learned
  QK dot-product, weighted by `lambda_attention`, `lambda_distance`, and
  `lambda_adjacency`.
- `atom_mask` gates out padding atoms in both attention softmax and pooling.
- The dummy node (index 0) serves as the global readout token when
  `aggregation_type='dummy_node'`; it is prepended inside `CPMPBackbone`, not
  in the cache.

---

### 2.3 SE3-Transformer (`_encode_conformers_se3t`)

**Required keys:** `node_feat`, `coords`, `conformer_mask`; `atom_mask` optional

```
node_feat  (B, C_max, A_max, F)  ──┐
coords     (B, C_max, A_max, 3)  ──┤─▶ flatten → (B*C_max, A_max, ...)
atom_mask  (B, A_max) expand     ──┘
                                    │
                                    ▼
                          SE3TBackbone.forward()
                            for each graph i:
                              n_real = atom_mask[i].sum()
                              build DGL fully-connected graph on n_real nodes
                              edata['rel_pos'] = x[src] − x[dst]   (n_edges, 3)
                            dgl.batch(graphs)
                                    │
                                    ▼
                          SE3Transformer (NVIDIA)
                            fiber_in:     {0: F}         # type-0 scalar features
                            fiber_hidden: {0..d−1: C}    # multi-degree hidden
                            fiber_out:    {0: d·C}       # type-0 output only
                            return_type=0, pooling='avg'
                                    │
                                    ▼
                          Linear projection → (B*C_max, d_gnn)
                                    │
                                    ▼
                           (B, C_max, d_gnn)
```

Key details:
- DGL graphs are built **individually** per graph in Python (no vectorized edge
  construction), with edges restricted to **real** (non-padded) atoms via
  `atom_mask`.
- Relative positions `rel_pos = x[src] − x[dst]` are the geometric input to
  the SE3 convolutions; the model is equivariant to SO(3) rotations.
- The NVIDIA SE3Transformer's built-in **avg pooling** returns one vector per
  graph — no separate pooling step needed.
- `low_memory=True` (default) trades speed for lower VRAM in tensor-product
  convolutions.

---

## 3. Summary Comparison

| | EGNN | CPMP | SE3-Transformer |
|---|---|---|---|
| **Graph repr.** | flat node list + explicit 3-D coords | dense adj + dist matrices | DGL graph + relative 3-D positions |
| **Batch keys used** | `node_feat`, `coords`, `atom_mask` | `node_feat`, `adj`, `dist`, `atom_mask` | `node_feat`, `coords`, `atom_mask` |
| **Edge construction** | fully-connected among real atoms (`get_edges_batch` + `atom_mask` edge filter) | implicit — attention over dense adj/dist | fully-connected DGL graph per real atom |
| **3-D geometry** | raw xyz updated per layer (equivariant coord update) | softmax(−dist) blended into attention | relative position vectors, SO(3)-equivariant conv |
| **Atom pooling** | masked mean over real atoms only (`atom_mask`) | masked mean / sum / dummy-node | SE3T built-in avg pooling over real nodes |
| **External deps** | none (pure PyTorch) | none (pure PyTorch) | DGL + e3nn |
