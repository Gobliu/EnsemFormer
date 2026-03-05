# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains core Python modules used by training and inference.
- `src/networks/` holds model backbones and CycloFormer wrappers (`egnn_*`, `cpmp_*`, `se3t_*`, `cycloformer*`).
- `src/featurization/` contains molecule loading and feature pipeline code; `src/dataset.py` and `src/trainer.py` drive data flow and training.
- `scripts/` contains runnable entry points: `main_train.py`, `preprocess_trajectories.py`, and dataset prep utilities.
- `config/` stores YAML configs (`default.yaml`, `dev.yaml`).
- `tests/` contains `pytest` smoke/unit tests for dataset, featurization, and model forward/gradient behavior.
- `data/` stores local datasets/cache files, and `experiments/` stores logs/checkpoints (both should be treated as generated artifacts).

## Build, Test, and Development Commands
- `python scripts/preprocess_trajectories.py --env water hexane`: preprocess trajectory PDBs into cached tensors.
- `python scripts/main_train.py --config config/default.yaml`: run a standard training job.
- `python scripts/main_train.py --gnn_type egnn --mode ensemble`: run with explicit backbone/mode overrides.
- `torchrun --nproc_per_node=2 scripts/main_train.py`: multi-GPU DDP training.
- `python -m pytest tests -v`: run the full test suite.
- `python -m pytest tests/test_dataset.py -v`: run a focused test file during iteration.

## Coding Style & Naming Conventions
- Use Python with PEP 8 conventions and 4-space indentation.
- Use `snake_case` for modules/functions/variables and `PascalCase` for classes.
- Keep reusable logic in `src/`; keep CLI/orchestration in `scripts/`.
- Prefer clear tensor naming (`B`, `N_conf`, `N_atoms`) and preserve batch/conformer dimension ordering.
- Keep config keys lowercase with underscores (for example `n_conformers`, `rep_frame_only`).

## Testing Guidelines
- Framework: `pytest`.
- Add tests as `tests/test_<component>.py`; name cases `test_<behavior>`.
- For model/data changes, verify output shapes, masks, finite values, and backward pass where relevant.
- Use small synthetic tensors/molecules so tests run quickly on CPU.

## Commit & Pull Request Guidelines
- Follow the existing imperative style in history: `Refactor ...`, `Unify ...`, `Integrate ...`.
- Keep subject lines concise and component-focused (for example `Fix dataset conformer mask padding`).
- PRs should include: goal, key code/config changes, exact test commands run, and any relevant artifact paths under `experiments/`.
