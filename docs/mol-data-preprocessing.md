# Data Preprocessing Pipeline

## Overview

EnsemFormer's preprocessing converts cyclic peptide trajectory data into cached graph
representations for GNN training. The pipeline runs in two phases:

1. **Offline preprocessing** (`scripts/preprocess_trajectories.py`) — run once, produces `.pt` cache files
2. **Training-time loading** (`src/dataset.py`) — loads cache, subsamples conformers, pads batches

```
Raw Data (CSV + trajectory PDBs)
        │
        ▼
  [preprocess_trajectories.py]  ──►  cache_traj_{env}_{noH|withH}.pt
                                          │
                                          ▼
                              [ConformerEnsembleDataModule]
                                          │
                                          ▼
                               Padded batch tensors → Model
```

---

## Phase 1: Offline Preprocessing

### Input Data

| Source | Description |
|--------|-------------|
| `data/CycPeptMPDB-4D.csv` | 5,160 cyclic peptides — columns: `CycPeptMPDB_ID`, `Source`, `SMILES`, `PAMPA` (target), metadata |
| `{traj_dir}/Water/Trajectories/` | Multi-MODEL PDB trajectory files (one per molecule, ~100+ frames each) |
| `{traj_dir}/Hexane/Trajectories/` | Same, hexane solvent |

**PDB filename pattern:** `{Source}_{CycPeptMPDB_ID}_{H2O_Traj|Hexane_Traj}.pdb`

### Running

```bash
# Single env (default from config)
python scripts/preprocess_trajectories.py

# Multiple envs
python scripts/preprocess_trajectories.py --env water hexane

# Custom config
python scripts/preprocess_trajectories.py --config config/dev.yaml
```

### Pipeline Flow

```
preprocess_trajectories.py
│
└─ for each hydrogen variant (noH, withH):
    │
    └─ mol_featurizer.py :: featurize_all_molecules()
        │
        ├─ read CSV once
        │
        └─ for each molecule in CSV:          ← mol_featurizer.py :: featurize_single_molecule()
            │
            └─ for each env (water / hexane):
                │
                └─ pdb_loader.py :: load_frames_from_traj_pdb()
                    │
                    ├─ parse MODEL/ENDMDL blocks from trajectory PDB
                    └─ for each frame:
                        ├─ RDKit: PDB block → Mol object
                        ├─ graph_builder.py :: mol_to_graph()
                        │     builds: node_feat, adj, dist, coords, bond_types
                        └─ within-traj consistency check vs. frame 0
                │
                └─ cross-env topology check (node_feat, adj, bond_types vs first env)
                      └─ ValueError if any mismatch
    │
    └─ torch.save() → cache_traj_{envs}_{noH|withH}.pt
```

### Notes

- Hydrogens are stripped (or kept) before featurization based on the `remove_h` flag.
- `node_feat`, `adj`, and `bond_types` encode molecular topology — they are invariant across MD frames
  and stored once per molecule. Only `dist` and `coords` vary per frame.

### Per-Frame Featurization

Each frame is featurized into these components (`mol_to_graph` in `graph_builder.py`):

| Matrix | Shape | Description |
|--------|-------|-------------|
| `node_features` | (N, 25) | Atom feature vectors |
| `adj` | (N, N) | Boolean adjacency matrix (True if bonded or self-loop) |
| `dist` | (N, N) | Pairwise Euclidean distance matrix |
| `coords` | (N, 3) | 3D atomic coordinates |
| `bond_types` | (N, N) | Bond type matrix (0=none, 1=single, 2=double, 3=triple, 4=aromatic) |

*The CPMP encoder internally prepends a dummy (CLS) node at forward time when using
`aggregation_type='dummy_node'`. This is not part of the cached data.*

### Atom Features (25-dim)

| Feature | Encoding | Dim | Values |
|---------|----------|-----|--------|
| Atomic number | One-hot | 11 | [5,6,7,8,9,15,16,17,35,53,other] |
| Degree | One-hot | 6 | [0,1,2,3,4,5] |
| Total H count | One-hot | 5 | [0,1,2,3,4] |
| Formal charge | Scalar | 1 | Raw value |
| In ring | Binary | 1 | 0/1 |
| Aromatic | Binary | 1 | 0/1 |

### Output: Cache File

```python
# data/cache_traj_water+hexane_noH.pt
{
    "molecules": [
        {
            # Topology — identical across all frames, stored once per molecule
            "node_feat":  ndarray (N_atoms, 25),      # node features
            "adj":        ndarray (N_atoms, N_atoms),  # adjacency matrix — bool; True if bonded or self-loop
            "bond_types": ndarray (N_atoms, N_atoms),  # bond type matrix — int8; 0=none, 1=single, 2=double, 3=triple, 4=aromatic
            # Per-frame data — only what actually varies across MD frames
            "envs": {
                "water":  [(dist, coords), ...],   # ALL frames stored; (dist, coords) per frame
                "hexane": [(dist, coords), ...],
            },
            "label": -5.1,                       # PAMPA value
            "CycPeptMPDB_ID": "CPMP-0001",       # unique molecule ID
            "SMILES": "CC(=O)...",                # if available in CSV
            "Structurally_Unique_ID": "SU-001",   # if available in CSV
            "rep_frame_idxs": {"water": 42, "hexane": 17}  # from Water_RepFrame / Hexane_RepFrame columns
        },
        ...  # 5,160 molecules
    ],
    "d_atom": 25,  # feature dimension
    "envs": ["hexane", "water"]   # environments stored in this cache
}
```

Two cache variants are produced per env combination:
- `cache_traj_{envs}_noH.pt` — hydrogens removed
- `cache_traj_{envs}_withH.pt` — hydrogens kept

---

## Phase 2: Training-Time Loading

### Conformer Subsampling

The cache stores **all** trajectory frames. At training time, `ConformerEnsembleDataModule`
subsamples based on config:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `env` | `[water]` | Which environment(s) to use from the cache |
| `n_conformers` | 10 | Uniformly sample N frames **per env** via `np.linspace(0, total-1, N)` |
| `rep_frame_only` | false | If true, keep only the representative frame (single conformer) per env |

`n_conformers` is applied per-env: with `env: [water, hexane]` and `n_conformers: 10`,
each env contributes 10 conformers for a total of 20 per molecule.

This lets you experiment with different conformer counts and env combinations without
re-running preprocessing.

### Data Splitting

- If the CSV has `split_0`, `split_1`, ... columns → use predefined splits
- Otherwise → random 80/10/10 train/val/test split (seeded)

### Batch Collation (`conformer_collate_fn`)

Variable-size graphs are padded to the batch maximum:

| Tensor | Shape | Notes |
|--------|-------|-------|
| `node_feat` | (B, C_max, A_max, d_atom) | Zero-padded atom features |
| `adj` | (B, C_max, A_max, A_max) | Zero-padded adjacency |
| `dist` | (B, C_max, A_max, A_max) | Padding filled with 1e6 |
| `coords` | (B, C_max, A_max, 3) | Zero-padded coordinates |
| `bond_type` | (B, C_max, A_max, A_max) | Zero-padded bond types |
| `atom_mask` | (B, A_max) | True for real atoms |
| `conformer_mask` | (B, C_max) | True for real conformers |
| `target` | (B, 1) | Regression label |

Where B = batch size, C_max = max conformers in batch, A_max = max atoms in batch.

---

## Config Reference

```yaml
paths:
  data_dir: data/
  csv_file: CycPeptMPDB-4D.csv
  traj_dir: /path/to/CycPeptMPDB_4D     # trajectory database root
  cache_file: data/cache_traj_water_noH.pt

data:
  target_col: PAMPA               # CSV column to predict
  env: [water]                    # environment(s): [water], [hexane], [water, hexane]
  n_conformers: 10                # frames per env per molecule at train time
  rep_frame_only: false           # single frame mode
  remove_h: true                  # strip hydrogens
```

---

## Key Design Decisions

1. **Two-phase design** — Trajectory parsing is slow (PDB I/O + RDKit). Caching all frames once
   avoids repeated work across training runs.

2. **Store all frames, subsample later** — The cache holds every trajectory frame. `n_conformers`
   is a training-time knob, enabling quick experimentation without re-preprocessing.

3. **Dual hydrogen variants** — Both noH and withH caches are produced. EGNN/CPMP typically use
   noH (fewer atoms, faster); withH is available if needed.

4. **Dummy node (CPMP-internal)** — The CPMP encoder optionally prepends a virtual CLS node at
   forward time for graph-level readout (`aggregation_type='dummy_node'`). This is handled
   entirely inside `CPMPBackbone`, not in preprocessing or the cache.

5. **Mask-based padding** — Variable-size molecules and conformer counts are handled via boolean
   masks (`atom_mask`, `conformer_mask`), keeping batching simple and GPU-friendly.

---

## Sanity Checks

The pipeline enforces several invariants to catch corrupt or malformed data early.

### Single connected fragment (`graph_builder.py :: mol_to_graph`)

Each molecule must be a single connected component — no disconnected fragments. A PDB parsing error or broken bond table can silently produce multiple fragments, which would be meaningless as GNN input. Checked via `rdmolops.GetMolFrags(mol)` before featurization; raises `ValueError` if fragment count != 1.

### Cross-frame topology consistency (`pdb_loader.py :: load_frames_from_traj_pdb`)

Within a single trajectory, `node_feat`, `adj`, and `bond_types` must be identical across all frames. These encode molecular topology (atom types, bonding), which is invariant across MD conformers — only `dist` and `coords` should change. The first frame's topology is used as the reference; any subsequent frame that disagrees raises `ValueError`.

### Cross-environment topology consistency (`mol_featurizer.py :: featurize_single_molecule`)

When loading multiple environments (water, hexane) for the same molecule, topology arrays must also match across environments. The same molecule in different solvents must have the same atom graph. Checked after all envs are loaded; raises `ValueError` on mismatch.

---

## Pitfalls

### RDKit implicit hydrogen counts from PDB files

When loading a PDB with `Chem.MolFromPDBBlock(block, removeHs=True)`, RDKit strips explicit H atoms and then **infers implicit Hs from valence rules** during sanitization. These inferred counts may not match the actual hydrogens in the PDB (e.g. unusual protonation states, non-standard residues). To preserve the true PDB hydrogen counts:

1. Load with `removeHs=False` first
2. Count each heavy atom's H neighbors: `sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)`
3. Store the count as an atom property
4. Call `Chem.RemoveHs(mol)`, then `atom.SetNumExplicitHs(count)` + `atom.SetNoImplicit(True)`

This ensures `atom.GetTotalNumHs()` returns the PDB's actual H count, not RDKit's guess. See `src/featurization/pdb_loader.py` for the implementation.
