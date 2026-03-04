# Data Preprocessing Pipeline

## Overview

EnsemFormer's preprocessing converts cyclic peptide trajectory data into cached graph
representations for GNN training. The pipeline runs in two phases:

1. **Offline preprocessing** (`scripts/traj_preprocess.py`) — run once, produces `.pt` cache files
2. **Training-time loading** (`src/dataset.py`) — loads cache, subsamples conformers, pads batches

```
Raw Data (CSV + trajectory PDBs)
        │
        ▼
  [traj_preprocess.py]  ──────►  cache_traj_{env}_{noH|withH}.pt
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
python scripts/traj_preprocess.py

# Multiple envs
python scripts/traj_preprocess.py --env water hexane

# Custom config
python scripts/traj_preprocess.py --config config/dev.yaml
```

### Pipeline Flow

```
traj_preprocess.py
│
└─ for each hydrogen variant (noH, withH):
    │
    ├─ for each env (water / hexane):           ← pipeline.py :: featurize_molecules()
    │   │
    │   └─ for each molecule in CSV:
    │       │
    │       └─ pdb_loading.py :: load_frames_from_traj_pdb()
    │           │
    │           ├─ parse MODEL/ENDMDL blocks from trajectory PDB
    │           └─ for each frame:
    │               ├─ RDKit: PDB block → Mol object
    │               ├─ atom_features.py :: featurize_mol()
    │               │     builds: nf, adj, dist, coords, bond_types
    │               └─ consistency check vs. frame 0:
    │                     nf == ref?  adj == ref?  bond_types == ref?
    │                     └─ ValueError if any mismatch
    │
    ├─ _merge_env_molecules()                   ← traj_preprocess.py
    │   ├─ align molecules across envs by CycPeptMPDB_ID
    │   ├─ hoist nf / adj / bond_types to molecule level (topology stored once)
    │   └─ cross-env consistency check: same checks as above
    │         └─ ValueError if any mismatch
    │
    └─ torch.save() → cache_traj_{envs}_{noH|withH}.pt
```

### Notes

- Hydrogens are stripped (or kept) before featurization based on the `remove_h` flag.
- `nf`, `adj`, and `bond_types` encode molecular topology — they are invariant across MD frames
  and stored once per molecule. Only `dist` and `coords` vary per frame.

### Per-Frame Featurization

Each frame is featurized into these components (`featurize_mol` in `atom_features.py`):

| Matrix | Shape | Description |
|--------|-------|-------------|
| `node_features` | (N, 25) | Atom feature vectors |
| `adj` | (N, N) | Boolean adjacency matrix (True if bonded or self-loop) |
| `dist` | (N, N) | Pairwise Euclidean distance matrix |
| `coords` | (N, 3) | 3D atomic coordinates |
| `bond_types` | (N, N) | Bond type matrix (0=none, 1=single, 2=double, 3=triple, 4=aromatic) |

*The CPMP encoder internally prepends a dummy (CLS) node at forward time when using
`aggregation_type='dummy_node'`. This is not part of the cached data.*

### Atom Features (25-dim default)

| Feature | Encoding | Dim | Values |
|---------|----------|-----|--------|
| Atomic number | One-hot | 11 | [5,6,7,8,9,15,16,17,35,53,other] |
| Degree | One-hot | 6 | [0,1,2,3,4,5] |
| Total H count | One-hot | 5 | [0,1,2,3,4] |
| Formal charge | Scalar | 1 | Raw value (or 3-dim one-hot if configured) |
| In ring | Binary | 1 | 0/1 |
| Aromatic | Binary | 1 | 0/1 |

### Output: Cache File

```python
# data/cache_traj_water+hexane_noH.pt
{
    "molecules": [
        {
            # Topology — identical across all frames, stored once per molecule
            "nf":         ndarray (N_atoms, 25),      # node features
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
    "d_atom": 25,  # feature dimension (or 27 with one_hot_formal_charge)
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

## Alternative Conformer Sources

The pipeline supports three modes (controlled by `data.conformer_source`):

| Mode | Source | Use Case |
|------|--------|----------|
| `cycpeptmpdb` (default) | Trajectory PDB files | Full MD simulation data |
| `pdb` | Directory of single PDB files | External conformer sets |
| `smiles` | SMILES strings in CSV | RDKit-generated conformers (ETKDGv3 + MMFF/UFF) |

---

## Config Reference

```yaml
paths:
  data_dir: data/
  csv_file: CycPeptMPDB-4D.csv
  traj_dir: /path/to/CycPeptMPDB_4D     # trajectory database root
  cache_file: data/cache_traj_water_noH.pt

data:
  conformer_source: cycpeptmpdb   # cycpeptmpdb | pdb | smiles
  target_col: PAMPA               # CSV column to predict
  env: [water]                    # environment(s): [water], [hexane], [water, hexane]
  n_conformers: 10                # frames per env per molecule at train time
  rep_frame_only: false           # single frame mode
  one_hot_formal_charge: false    # scalar (1-dim) vs one-hot (3-dim) charge
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
