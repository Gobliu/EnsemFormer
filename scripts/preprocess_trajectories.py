"""Preprocess CycPeptMPDB trajectory PDB files and save to cache files.

Run this once before training to avoid repeating slow PDB/trajectory parsing
on every run. Produces TWO cache files per env combination: one with hydrogens
removed (noH) and one with hydrogens kept (withH).

ALL trajectory frames are stored per molecule. n_conformers and
rep_frame_only are deferred to training time.

Usage:
    # Single env (default from config)
    python scripts/preprocess_trajectories.py

    # Multiple envs
    python scripts/preprocess_trajectories.py --env water hexane

    # Custom config file
    python scripts/preprocess_trajectories.py --config config/custom.yaml
"""

import argparse
import logging
import pathlib
import sys

import torch
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess trajectory data and save cache")
    parser.add_argument(
        "--config",
        default=str(_REPO_ROOT / "config" / "default.yaml"),
    )
    parser.add_argument("--env", nargs="+", type=str, help="Environments: water hexane (overrides config)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = get_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    data = config["data"]

    # Resolve env list from CLI or config
    if args.env:
        envs = args.env
    else:
        cfg_env = data["env"]
        envs = cfg_env if isinstance(cfg_env, list) else [cfg_env]

    out_dir = _REPO_ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.featurization import featurize_all_molecules

    env_str = "+".join(sorted(envs))

    for remove_h, tag in [(True, "noH"), (False, "withH")]:
        molecules, d_atom = featurize_all_molecules(
            csv_path=_REPO_ROOT / paths["data_dir"] / paths["csv_file"],
            target_col=data["target_col"],
            traj_dir=paths["traj_dir"],
            envs=envs,
            remove_h=remove_h,
        )
        output = out_dir / f"cache_traj_{env_str}_{tag}.pt"
        torch.save({"molecules": molecules, "d_atom": d_atom, "envs": sorted(envs)}, output)
        logging.info(f"Saved {len(molecules)} molecules (d_atom={d_atom}, envs={sorted(envs)}) -> {output}")


if __name__ == "__main__":
    main()
