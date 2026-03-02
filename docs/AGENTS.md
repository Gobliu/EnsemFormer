# Repository Guidelines

## Project Structure & Module Organization
- `src/` contains importable core code.
- `src/networks/` holds model components (`lufnet.py`, `phi_generator.py`, `psi_generator.py`, `water_radial_basis_function.py`, `icosahedron.py`).
- Top-level modules in `src/` manage model assembly and training flow (`build_model.py`, `trainer.py`, `loss.py`, `optim.py`, `datasets.py`, `checkpoint.py`).
- `src/utils/` provides shared infrastructure (config loading, logging, distributed helpers, periodic boundary tools).
- `scripts/main_train.py` is the main training entry point.
- `tests/` mirrors runtime modules with unit and integration tests.
- `config/default.yaml` is the single experiment config, separated by comment blocks.
- `data/` holds local datasets (gitignored; see `data/README.md`).
- `experiments/` stores checkpoints and run outputs (gitignored).
- `analysis/` and `docs/` are for post-processing and design notes; `archive/` is legacy code.

## Build, Test, and Development Commands
```bash
pip install -e ".[dev]"
python scripts/main_train.py
python -m pytest tests -v --ignore=tests/test_transformer_encoder.py --ignore=tests/check_predict.py
```
- `pip install -e ".[dev]"`: installs package + development tools in editable mode.
- `python scripts/main_train.py`: runs training using `config/default.yaml`.
- `pytest ...`: runs the maintained test suite (excluding currently ignored tests shown above).
- For fast iteration, run targeted tests, for example: `python -m pytest tests/test_loss.py -v`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation.
- Use `snake_case` for modules/functions/variables and `CapWords` for classes.
- Keep modules in `src/` import-safe; place orchestration in `scripts/`.
- Use concise docstrings and add tensor-shape comments where logic is non-obvious.
- If installed locally, run `black src tests` and `flake8 src tests` before PRs.

## Testing Guidelines
- Framework: `pytest` with shared fixtures in `tests/conftest.py` and `tests/helpers.py`.
- Add tests as `tests/test_<module>.py` for each behavior change.
- For numerical/model code, assert shape, finiteness, and gradient behavior; use fixed seeds for reproducibility.
- Run the full test suite before submitting.

## Commit & Pull Request Guidelines
- Recent history uses imperative commit subjects (`Refactor ...`, `Fix ...`, `Add ...`).
- Prefer: `<Verb> <component>: <what changed>` with concise first lines.
- Avoid non-descriptive subjects (`xx`, `..`, `merge`).
- PRs should include objective, key changes, config impact, and exact test commands/results.
- Link related issues; include artifact paths when relevant (for example outputs under `analysis/` or `figures/`).
