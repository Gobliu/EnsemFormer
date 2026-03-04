# EnsemFormer

**EnsemFormer** is a framework for molecular property prediction using ensembles of 3D conformers. The core idea is to treat a set of 3D structures (e.g. from an MD trajectory or conformational search) as a sequence of tokens, processed by a Transformer encoder to produce a single molecular-level representation.

## Current Model: CycloFormer

The first instantiation of EnsemFormer is **CycloFormer**, targeting **cyclic peptide membrane permeability** prediction (regression).

### Architecture

```
Conformer 1 ──┐
Conformer 2 ──┤  [3D GNN backbone]  →  conformer embeddings (tokens)
   ...        │                               ↓
Conformer N ──┘                    [Transformer encoder]
                                    (conformers as tokens)
                                              ↓
                                   [Readout: CLS or mean pool]
                                              ↓
                                         [MLP head]
                                              ↓
                                    Permeability value (regression)
```

- **3D GNN backbone**: Swappable per-conformer GNN (EGNN, SE(3)-Transformer, CPMP)
- **Conformer aggregation**: Transformer encoder over conformer embeddings
- **Task**: Regression (continuous permeability value, e.g. PAMPA)
- **Input**: Ensemble of 3D conformers per cyclic peptide

## Data Preprocessing

Parsing 5000+ trajectory PDB files is slow (~10–20 min for all envs). Run `traj_preprocess.py` once to cache the featurized molecules to disk, then all subsequent training runs load from cache instantly.

### Step 1 — preprocess trajectories (run once)

```bash
# Single env (default from config)
python scripts/traj_preprocess.py

# Multiple envs in one cache
python scripts/traj_preprocess.py --env water hexane
```

Each run produces **two** cache files — one with hydrogens removed, one with hydrogens kept:

```
data/cache_traj_water+hexane_noH.pt
data/cache_traj_water+hexane_withH.pt
```

All trajectory frames are stored per molecule, grouped by environment. `n_conformers`, `env`, and `rep_frame_only` are applied at training time.

### Step 2 — point the config to the cache

In `config/default.yaml`:

```yaml
paths:
  cache_file: data/cache_traj_water+hexane_noH.pt   # or _withH.pt
```

`cache_file` is **required** — training will not start without it.

> **Note**: regenerate the cache only if you change `one_hot_formal_charge` or add new envs.
> `n_conformers`, `env`, and `rep_frame_only` do **not** require a new cache.

---

## Training

### Basic usage

```bash
python scripts/main_train.py                          # defaults from config/default.yaml
python scripts/main_train.py --config config/my.yaml  # custom config file
```

### GNN backbone

```bash
python scripts/main_train.py --gnn_type egnn   # E(n) Equivariant GNN (default)
python scripts/main_train.py --gnn_type cpmp   # Cyclic Peptide Membrane Permeability
python scripts/main_train.py --gnn_type se3t   # SE(3)-Transformer
```

### Ensemble vs. standalone mode

| Mode | Description |
|------|-------------|
| `ensemble` (default) | Conformer embeddings passed through a Transformer encoder; CLS or mean-pool readout |
| `standalone` | Conformer embeddings directly pooled; Transformer skipped |

```bash
python scripts/main_train.py --mode ensemble   # full architecture (conformer Transformer)
python scripts/main_train.py --mode standalone # GNN only, no Transformer
```

`standalone` mode is equivalent to setting `rep_frame_only: true` (single representative frame) but works with any number of conformers.

### Common overrides

```bash
python scripts/main_train.py --epochs 100 --learning_rate 5e-4 --batch_size 16
python scripts/main_train.py --n_conformers 20 --env water          # water conformers only
python scripts/main_train.py --n_conformers 20 --env hexane         # hexane conformers only
python scripts/main_train.py --n_conformers 20 --env water hexane   # water + hexane (n_conformers per env)
python scripts/main_train.py --rep_frame_only   # use only the representative MD frame
python scripts/main_train.py --seed 0
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 scripts/main_train.py
```

Logs and checkpoints are written to `experiments/<gnn_type>/run_<version>/`.

---

## Vision

EnsemFormer is designed to be domain-agnostic. While CycloFormer targets cyclic peptides, the framework can be extended to other molecular property prediction tasks where conformational diversity is important.
