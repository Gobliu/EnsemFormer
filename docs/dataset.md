# CycPeptMPDB-4D Dataset

## Overview

Primary raw dataset root:
- `/home/liuw/GitHub/Data/CycPeptMPDB_4D/`

CycPeptMPDB-4D contains 5,160 cyclic peptides with MD-derived conformer ensembles and PAMPA labels, spanning 5 sources (`2015_Wang`, `2016_Furukawa`, `2018_Naylor`, `2020_Townsend`, `2021_Kelly`).

## Directory Structure (Raw Dataset Root)

```
CycPeptMPDB_4D/
├── CycPeptMPDB-4D.csv
├── Water/
│   ├── Structures/     # 5,160 PDB files
│   ├── Trajectories/   # 5,160 trajectory PDB files
│   └── Logs/           # 5,160 clustering logs
├── Hexane/
│   ├── Structures/     # 5,160 PDB files
│   ├── Trajectories/   # 5,160 trajectory PDB files
│   └── Logs/           # 5,160 clustering logs
└── CHCl3/              # 6 trajectories (not used in current training flow)
```

Local copy size in this environment is about `11G`.

## File Naming Convention

```
{YEAR}_{Author}_{ID}_H2O_Str.pdb
{YEAR}_{Author}_{ID}_H2O_Traj.pdb
{YEAR}_{Author}_{ID}_Hexane_Str.pdb
{YEAR}_{Author}_{ID}_Hexane_Traj.pdb
{YEAR}_{Author}_{ID}_H2O.log
{YEAR}_{Author}_{ID}_Hexane.log
```

`Source` provides `{YEAR}_{Author}` and `CycPeptMPDB_ID` provides `{ID}`.

## CSV Variants

There are two CSV variants in common use:

1. Raw CSV at dataset root (`/home/liuw/GitHub/Data/CycPeptMPDB_4D/CycPeptMPDB-4D.csv`):
- 19 columns (metadata + labels), no representative-frame columns.

2. Prepared CSV used by this repo (`data/CycPeptMPDB-4D.csv`):
- Includes additional columns like `SMILES`, `Water_RepFrame`, and `Hexane_RepFrame`.
- `Water_RepFrame` and `Hexane_RepFrame` are 1-based frame indices derived from log cluster-1 middle times.

Representative-frame conversion logic is implemented in `scripts/prepare_dataset.py`:

```python
frame = round((t_ns - 20.0) / 0.3) + 1
```

## Conformer Ensembles

- Most trajectory PDBs have 100 MODEL frames (20.0 to 49.7 ns at 0.3 ns spacing).
- Four Hexane trajectories are shorter: `5840` (19), `5952` (40), `6051` (41), `6352` (62).

Each valid frame is featurized into:
- `nf` (node features)
- `adj` (bond adjacency + self-loop)
- `bond_types`
- per-frame `(dist, coords)`

Note: graph topology is bond-based, not distance-cutoff-based.

## Usage in EnsemFormer (Current Code)

Current training uses a cache-first workflow:

1. Run preprocessing once (`scripts/traj_preprocess.py`) to create `.pt` cache files.
2. Configure `paths.cache_file` in YAML.
3. `ConformerEnsembleDataModule` loads cached molecules from `.pt` (it does not parse trajectory PDBs during training).
4. At runtime, it applies `env`, `n_conformers`, and `rep_frame_only`, then collates padded tensors.

Important behavior:
- `rep_frame_only=True` requires representative frame indices in cache.
- Missing or out-of-range rep-frame indices now raise errors (no silent clamp/fallback).

## Splits

- If CSV has `split_<k>` columns, those are used.
- Otherwise, code falls back to seeded random `80/10/10` split.
