# Data Loading in EnsemFormer

This document describes the full data-loading pipeline — from raw inputs to the
per-model tensors consumed by the GNN encoders — for all three supported
backbone types: **EGNN**, **CPMP**, and **SE3-Transformer**.

---

## 1. Overview

```
Raw source (SMILES / PDB / trajectory)
        │
        ▼
  featurize_mol()          # per conformer → (node_feat, adj, dist)
        │
        ▼
ConformerEnsembleMolecule  # N conformers + label, one molecule
        │
        ▼
ConformerEnsembleDataset   # PyTorch Dataset of molecules
        │
        ▼
 conformer_collate_fn()    # pads atoms & conformers → batch dict
        │
        ▼
  CycloFormerModule        # dispatches to EGNN / CPMP / SE3T encoder
```

---

## 2. Featurization (`src/featurization.py`)

### 2.1 Atom features — `get_atom_features(atom)`

Each heavy atom is encoded as a 1-D float32 vector of length **25** (or **27**
with `one_hot_formal_charge=True`):

| Feature | Encoding | Size |
|---|---|---|
| Atomic number | one-hot over `[B, C, N, O, F, P, S, Cl, Br, I, other]` | 11 |
| Degree (# neighbors) | one-hot over `[0, 1, 2, 3, 4, 5]` | 6 |
| Total H count | one-hot over `[0, 1, 2, 3, 4]` | 5 |
| Formal charge | raw int **or** one-hot over `[-1, 0, 1]` | 1 or 3 |
| Is in ring | bool | 1 |
| Is aromatic | bool | 1 |

### 2.2 Per-conformer tuple — `featurize_mol(mol)`

Returns a triple `(node_features, adj_matrix, dist_matrix)`:

- **`node_features`** — `(N_atoms, F)` float32 atom feature matrix.
- **`adj_matrix`** — `(N_atoms, N_atoms)` float32, 1 where a bond exists plus
  self-loops on the diagonal.
- **`dist_matrix`** — `(N_atoms, N_atoms)` float32 Euclidean pairwise distances
  computed from the RDKit conformer's 3-D atom positions.

The CPMP encoder internally prepends a virtual CLS node at forward time when
using `aggregation_type='dummy_node'`. This is not part of the cached data.

---

## 3. Conformer Sources (`ConformerEnsembleDataModule`)

Three modes are selected via `conformer_source`:

### `smiles` — RDKit on-the-fly generation

```
SMILES → AddHs → EmbedMultipleConfs (ETKDGv3) → MMFF/UFF optimize → RemoveHs
       → featurize_mol() × N_conformers
```

Node features and adjacency are conformation-independent and computed once;
only `dist_matrix` differs per conformer.

### `pdb` — Precomputed single PDB files

```
{source}_{id}.pdb → Chem.MolFromPDBFile (→ Open Babel fallback)
                  → featurize_mol()
```

Each PDB file becomes one conformer. Multiple paths can be supplied per
molecule to form an ensemble.

### `cycpeptmpdb` — CycPeptMPDB-4D trajectory PDB

```
{source}_{id}_H2O_Traj.pdb  (or Hexane_Traj.pdb)
    │  parsed MODEL / ENDMDL blocks
    ▼
load_frames_from_traj_pdb()
    │  frame_indices:
    │    - standalone: single representative frame from CSV (Water_RepFrame)
    │    - ensemble:   uniformly subsample n_conformers from available frames
    ▼
Chem.MolFromPDBBlock() → featurize_mol() × n_frames
```

---

## 4. Batch Collation — `conformer_collate_fn`

The collate function pads both the conformer and atom dimensions to the
batch-maximum and returns a single dict of dense tensors.

### Padding strategy

| Dimension | Pad value |
|---|---|
| `node_feat` | 0 |
| `adj` | 0 |
| `dist` | `1e6` (large sentinel distance) |

### Output keys

| Key | Shape | Dtype | Description |
|---|---|---|---|
| `node_feat` | `(B, N_conf_max, N_atoms_max, F)` | float32 | Atom feature matrices |
| `adj` | `(B, N_conf_max, N_atoms_max, N_atoms_max)` | float32 | Bond adjacency |
| `dist` | `(B, N_conf_max, N_atoms_max, N_atoms_max)` | float32 | Pairwise distances |
| `atom_mask` | `(B, N_atoms_max)` | bool | `True` for real atoms |
| `conformer_mask` | `(B, N_conf_max)` | bool | `True` for real conformers |
| `target` | `(B, 1)` | float32 | Regression label |

> **Note:** Raw 3-D coordinates are **not** included in the collated batch.
> EGNN and SE3-Transformer both require a `coords` key `(B, N_conf, N_atoms, 3)`.
> This must be added upstream (e.g., store coordinates alongside `dist` during
> featurization) before those encoders can be fully exercised end-to-end.

---

## 5. Per-Model Batch Consumption

All three encoders receive the same batch dict from the DataLoader.
`CycloFormerModule` dispatches to the appropriate method and reshapes tensors
so every `(molecule, conformer)` pair is processed as an independent graph.

### 5.1 EGNN (`_encode_conformers_egnn`)

**Required keys:** `node_feat`, `coords`, `conformer_mask`

```
node_feat  (B, N_conf, N_atoms, F)  ──┐
coords     (B, N_conf, N_atoms, 3)  ──┤─▶ flatten → (B*N_conf, N_atoms, ...)
                                       │
                                       ▼
                           get_edges_batch()          # fully-connected edge list
                                       │
                                       ▼
                              EGNNBackbone.forward()  # E_GCL layers on flat atoms
                                       │              # input: (B*N_conf*N_atoms, F)
                                       ▼
                           mean-pool over atoms
                                       │
                                       ▼
                              (B, N_conf, d_gnn)
```

Key details:
- Edges are **fully-connected** and built on-the-fly by `get_edges_batch()` for
  the flat `(B*N_conf)` super-batch. All graphs must have the same `N_atoms`
  (guaranteed by the padding to `N_atoms_max`).
- EGNN updates **coordinates** as well as node hidden states; only node states
  are returned and mean-pooled.
- `in_edge_nf=0` by default (no edge features, dummy all-ones `edge_attr`).

### 5.2 CPMP (`_encode_conformers_cpmp`)

**Required keys:** `node_feat`, `adj`, `dist`, `atom_mask`, `conformer_mask`

```
node_feat  (B, N_conf, N_atoms, F)                 ──┐
adj        (B, N_conf, N_atoms, N_atoms)            ──┤
dist       (B, N_conf, N_atoms, N_atoms)            ──┤─▶ flatten → (B*N_conf, ...)
atom_mask  (B, N_atoms)  expand to (B*N_conf, ...)  ──┘
                                       │
                                       ▼
                             CPMPBackbone.forward()
                               Embeddings (Linear)
                                       │
                               Encoder (N × EncoderLayer)
                               ┌── MultiHeadedAttention
                               │     mixes: QK-attention + adj + softmax(−dist)
                               │     weights: λ_attn · p_attn + λ_dist · p_dist + λ_adj · p_adj
                               └── PositionwiseFeedForward
                                       │
                                       ▼
                               atom-masked aggregation
                               ('mean' | 'sum' | 'dummy_node')
                                       │
                                       ▼
                              (B, N_conf, d_gnn)
```

Key details:
- `adj` and `dist` enter the **attention score** directly alongside the learned
  QK dot-product, weighted by `lambda_attention`, `lambda_distance`, and
  `lambda_adjacency` (= 1 − the other two).
- `atom_mask` gates out padding atoms in both attention softmax and pooling.
- The dummy node (index 0) can serve as the global readout token when
  `aggregation_type='dummy_node'`.

### 5.3 SE3-Transformer (`_encode_conformers_se3t`)

**Required keys:** `node_feat`, `coords`, `conformer_mask`; `atom_mask` optional

```
node_feat  (B, N_conf, N_atoms, F)  ──┐
coords     (B, N_conf, N_atoms, 3)  ──┤─▶ flatten → (B*N_conf, N_atoms, ...)
atom_mask  (B, N_atoms) expand      ──┘
                                       │
                                       ▼
                             SE3TBackbone.forward()
                               for each graph i:
                                 n_real = atom_mask[i].sum()
                                 build DGL fully-connected graph on n_real nodes
                                 edata['rel_pos'] = xi[src] − xi[dst]   (n_edges, 3)
                               dgl.batch(graphs)
                                       │
                                       ▼
                             SE3Transformer (NVIDIA)
                               fiber_in:  {0: F}          # type-0 scalar features
                               fiber_hidden: {0..d−1: C}  # multi-degree hidden
                               fiber_out: {0: d·C}        # type-0 output only
                               return_type=0, pooling='avg'
                                       │
                                       ▼
                               Linear projection → (B*N_conf, d_gnn)
                                       │
                                       ▼
                              (B, N_conf, d_gnn)
```

Key details:
- DGL graphs are built **individually** per graph in Python (no vectorized edge
  construction), with edges restricted to **real** (non-padded) atoms via
  `atom_mask`.
- Relative positions `rel_pos = x[src] − x[dst]` serve as the geometric
  input to the SE3 convolutions; the model is equivariant to SO(3) rotations.
- The NVIDIA SE3Transformer's built-in **avg pooling** returns one vector per
  graph, so no separate pooling step is needed after the backbone call.
- `low_memory=True` (default) trades speed for lower VRAM in the tensor-product
  convolutions.

---

## 6. Summary Comparison

| | EGNN | CPMP | SE3-Transformer |
|---|---|---|---|
| **Primary graph repr.** | flat node list + explicit 3-D coords | adj + dist matrix (dense) | DGL graph + relative 3-D positions |
| **Batch keys used** | `node_feat`, `coords`* | `node_feat`, `adj`, `dist`, `atom_mask` | `node_feat`, `coords`*, `atom_mask` |
| **Edge construction** | fully-connected, built by `get_edges_batch` (uniform N_atoms) | implicit — attention over dense adj/dist | fully-connected DGL graph per real atom |
| **3-D geometry enters how** | raw xyz updated per layer (equivariant coord update) | softmax(−dist) blended into attention | relative position vectors, SO(3)-equivariant convolution |
| **Atom pooling** | mean over N_atoms_max (all padded slots contribute 0) | masked mean / sum / dummy-node | SE3T built-in avg pooling over real nodes only |
| **External dependency** | none (pure PyTorch) | none (pure PyTorch) | DGL + e3nn |

\* `coords` must be added to the batch dict; see note in §4.
