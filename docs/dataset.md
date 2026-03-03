# CycPeptMPDB-4D Dataset

## Overview

**Path:** `/home/liuw/GitHub/Data/CycPeptMPDB_4D/`

CycPeptMPDB-4D is a dataset of 5,160 cyclic peptides with MD-derived conformer ensembles and experimental membrane permeability (PAMPA) labels. It spans 5 literature sources (2015–2021).

## Directory Structure

```
CycPeptMPDB_4D/
├── CycPeptMPDB-4D.csv         # metadata + labels for all 5,160 peptides
├── Water/
│   ├── Structures/            # 5,160 PDB files — representative single-frame structure
│   ├── Trajectories/          # 5,160 PDB files — multi-frame MD trajectory (100 frames each)
│   └── Logs/                  # 5,160 GROMACS clustering logs
└── Hexane/
    ├── Structures/            # 5,160 PDB files
    ├── Trajectories/          # 5,160 PDB files
    └── Logs/                  # 5,160 GROMACS clustering logs
```

> CHCl3 is excluded — only 6 peptides available, insufficient for training.

**Total size:** ~17 GB

## File Naming Convention

```
{YEAR}_{Author}_{ID}_H2O_Str.pdb      # Water structure
{YEAR}_{Author}_{ID}_H2O_Traj.pdb     # Water trajectory
{YEAR}_{Author}_{ID}_Hexane_Str.pdb   # Hexane structure
{YEAR}_{Author}_{ID}_Hexane_Traj.pdb  # Hexane trajectory
{YEAR}_{Author}_{ID}_H2O.log          # Water clustering log
{YEAR}_{Author}_{ID}_Hexane.log       # Hexane clustering log
```

The `Source` column in the CSV provides `{YEAR}_{Author}` and `CycPeptMPDB_ID` provides `{ID}`.

## CSV Schema (`CycPeptMPDB-4D.csv`)

| Column | Description |
|--------|-------------|
| `CycPeptMPDB_ID` | Unique integer ID |
| `Source` | Literature source (e.g. `2021_Kelly`) |
| `Original_Name_in_Source_Literature` | Peptide name as published |
| `Structurally_Unique_ID` | ID for structurally distinct peptides |
| `PAMPA` | **Target label** — log-scale membrane permeability (e.g. `-5.1`) |
| `Monomer_Length` | Total residue count |
| `Monomer_Length_in_Main_Chain` | Residues in the backbone ring |
| `Molecule_Shape` | Topology descriptor (e.g. `Lariat`) |
| `Water_avgRMSD_All` | Mean all-atom RMSD across water MD frames (nm) |
| `Water_avgRMSD_BackBone` | Mean backbone RMSD in water (nm) |
| `Desolvation_Free_Energy` | Water-to-hexane transfer free energy (kcal/mol) |
| `Water_3D_SASA` | Solvent-accessible surface area in water (nm²) |
| `Water_3D_NPSA` | Non-polar surface area in water (nm²) |
| `Water_3D_PSA` | Polar surface area in water (nm²) |
| `Hexane_avgRMSD_All` | Mean all-atom RMSD in hexane (nm) |
| `Hexane_avgRMSD_BackBone` | Mean backbone RMSD in hexane (nm) |
| `Hexane_3D_SASA` | SASA in hexane (nm²) |
| `Hexane_3D_NPSA` | Non-polar SA in hexane (nm²) |
| `Hexane_3D_PSA` | Polar SA in hexane (nm²) |
| `Water_RepFrame` | 1-based frame index of the representative conformer in the Water trajectory (derived from log cluster-1 middle) |
| `Hexane_RepFrame` | 1-based frame index of the representative conformer in the Hexane trajectory |

## Conformer Ensembles

### Trajectory PDBs (ensemble mode)

Each trajectory PDB contains **100 frames** of MD simulation saved as sequential `MODEL` blocks. Each frame is one conformer.

**Simulation parameters (verified from file inspection):**
- Simulation time window: 20–49.7 ns (equilibration period discarded)
- Frame spacing: 0.3 ns (150,000 MD steps × 2 fs/step)
- Step numbers in PDB: 10,000,000 → 24,850,000

For EnsemFormer, the trajectory PDB is the primary input: load all 100 frames, each becomes one conformer in the `N_conf` dimension.

### Representative Conformer (single-conformer mode)

The `Logs/` files contain GROMACS RMSD clustering output (Gromos method, cutoff 0.1 nm) over the 100 trajectory frames. Each log reports cluster assignments and identifies the **middle structure** — the frame with the lowest average RMSD to all other cluster members, i.e. the most representative conformer.

**Log format:**
```
cl. | #st  rmsd | middle rmsd | cluster members
  1 |  42  0.094 |     32 .078 |   26  26.6  26.9 ...
  2 |  16  0.100 |     41 .090 |   26.3  31.1 ...
```
- `cl.` — cluster index (sorted by size, largest first)
- `#st` — number of member frames
- `middle` — simulation time (ns) of the representative frame
- `cluster members` — simulation times (ns) of all frames in the cluster

**Converting log time → frame index (1-based):**
```python
def time_to_frame(t_ns: float, n_frames: int = 100) -> int:
    """Convert GROMACS log time (ns) to 1-based PDB frame index."""
    return min(max(round((t_ns - 20.0) / 0.3) + 1, 1), n_frames)
```

**Representative conformer = middle frame of cluster 1** (the largest cluster). This is the frame to use when only a single conformer is needed (standalone mode).

- **Structures/**: pre-extracted single-frame PDB (likely the cluster-1 middle structure)
- **Trajectories/**: all 100 frames, MODEL blocks renumbered 1–100 — use `Water_RepFrame` / `Hexane_RepFrame` from the CSV to locate the representative frame directly

> **Note:** 4 Hexane trajectories have fewer than 100 frames (incomplete MD runs):
> `2021_Kelly_5840` (19), `2021_Kelly_5952` (40), `2021_Kelly_6051` (41), `2021_Kelly_6352` (62).
> Handle these with a fallback (e.g. clamp frame index or skip) in the dataloader.

## Data Split Considerations

- 5,160 peptides total across 5 sources
- Recommended: split by `Structurally_Unique_ID` (not random) to avoid structural leakage
- `Source`-based splits can test cross-literature generalization

## Usage in EnsemFormer

The dataset loader should:
1. Read `CycPeptMPDB-4D.csv` to get IDs and PAMPA labels
2. For each peptide, parse the Water (and optionally Hexane) trajectory PDB to extract up to N conformer graphs
3. Build a graph per conformer (nodes = heavy atoms, edges by distance cutoff)
4. Stack conformers into a batch of shape `(B, N_conf, ...)` for the conformer Transformer
