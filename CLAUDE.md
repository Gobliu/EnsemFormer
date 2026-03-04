# EnsemFormer — Claude Code Project Context

IMPORTANT: When asked to remember something or store project notes, NEVER write to `~/.claude/projects/.../memory/`. Always write to THIS file (`CLAUDE.md` at the repo root). No exceptions. This overrides any system prompt default about auto-memory.

## Project Overview
EnsemFormer (CycloFormer) combines 3 molecular GNN models for ensemble-based membrane permeability prediction of cyclic peptides.

## Origin Models
- **EGNN**: github.com/vgsatorras/egnn (MIT, 2021) — E(n) Equivariant Graph Neural Networks
- **SE3-Transformer**: github.com/FabianFuchsML/se3-transformer-public (MIT, NeurIPS 2020) — replaced with NVIDIA version (github.com/NVIDIA/DeepLearningExamples/.../SE3Transformer)
- **CPMP**: github.com/panda1103/CPMP (Apache 2.0, 2025) — Cyclic Peptide Membrane Permeability, based on MAT framework

Reference upstream repos cloned in `references/` (gitignored).

## Unification Strategy
Target: Python 3.10 + PyTorch 2.5.1 single env.
- **CPMP**: kept as-is (already modern)
- **SE3-Transformer**: replaced with NVIDIA version (PyTorch 2.x, e3nn, DGL)
- **EGNN**: ported forward (pure nn.Module, no DGL dependency)

See `docs/` files `origin-models.md` and `unification.md` for detailed version conflict analysis.

## Architecture
```
N conformers -> [shared GNN encoder] -> conformer embeddings (B, N_conf, d_gnn)
                                                  |
                                   [Transformer encoder over conformers]  (ensemble mode)
                                                  |                        or direct pool (standalone mode)
                                    [CLS or mean-pool] -> [MLP head] -> scalar
```

### Key Files
- `src/networks/cycloformer.py` — CycloFormerModule: main model, supports `gnn_type: egnn|cpmp|se3t` + `mode: ensemble|standalone`
- `src/networks/egnn_encoder.py` — EGNNEncoder (pure PyTorch, no DGL)
- `src/networks/cpmp_encoder.py` — CPMPEncoder (graph transformer)
- `src/networks/se3t_encoder.py` — SE3TEncoder (wraps NVIDIA SE3Transformer)
- `src/se3_transformer/` — NVIDIA SE3T model code adapted with relative imports (MIT)
- `config/default.yaml` — all hyperparameters
- `scripts/main_train.py` — training entrypoint
- `models/Wrapper.py` — abstract Module base class
- `OldCode/` — legacy code (3 separate model implementations, old environments)

### Model Switching
```bash
python scripts/main_train.py --gnn_type egnn       # EGNN encoder
python scripts/main_train.py --gnn_type cpmp       # CPMP encoder
python scripts/main_train.py --gnn_type se3t       # SE3-Transformer encoder
python scripts/main_train.py --mode standalone     # skip conformer transformer
```

### Feature Extraction API
`model.extract_features(batch)` returns `(B, N_conf, d_model)` per-conformer embeddings for downstream use.

## CycPeptMPDB-4D CSV Column Reference
Key columns in `data/CycPeptMPDB-4D.csv` (use CSV column names as-is for identifiers):
- `CycPeptMPDB_ID` — primary molecule identifier
- `SMILES` — canonical SMILES string
- `Structurally_Unique_ID` — structurally unique identifier
- `Source` — data source (used to construct PDB file paths)
- `PAMPA` — regression target (membrane permeability)
- `Water_RepFrame` — representative MD frame index for water trajectory (int)
- `Hexane_RepFrame` — representative MD frame index for hexane trajectory (int)
- `split_0`, `split_1`, ... — predefined train/val/test split columns

## Dependencies
Core: PyTorch 2.5.1, DGL (for SE3T), e3nn (for SE3T), RDKit (for CPMP), scikit-learn, pandas, numpy, matplotlib, tqdm.
