"""Preprocess CycPeptMPDB trajectory PDB files and save to cache files.

Run this once before training to avoid repeating slow PDB/trajectory parsing
on every run. Produces TWO cache files per env combination: one with hydrogens
removed (noH) and one with hydrogens kept (withH).

ALL trajectory frames are stored per molecule. n_conformers and
rep_frame_only are deferred to training time.

Usage:
    # Single env (default from config)
    python scripts/traj_preprocess.py

    # Multiple envs
    python scripts/traj_preprocess.py --env water hexane

    # Custom config file
    python scripts/traj_preprocess.py --config config/custom.yaml
"""

import argparse
import logging
import pathlib
import sys

import numpy as np
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


def _merge_env_molecules(env_results: dict[str, list[dict]]) -> list[dict]:
    """Merge per-env molecule lists into molecule-level dicts with an envs layer.

    Parameters
    ----------
    env_results : {env_name: [molecule_dicts]} where each molecule_dict has
        keys 'nf', 'adj', 'bond_types', 'conformers', 'label', 'CycPeptMPDB_ID', etc.

    Returns
    -------
    List of merged molecule dicts with structure:
        {"nf", "adj", "bond_types", "envs": {env: [(dist, coords), ...], ...}, "label", ...}
    """
    by_id: dict[str, dict] = {}
    for env, molecules in env_results.items():
        for mol in molecules:
            cpmp_id = mol["CycPeptMPDB_ID"]
            if cpmp_id not in by_id:
                entry: dict = {
                    "nf":         mol["nf"],
                    "adj":        mol["adj"],
                    "bond_types": mol["bond_types"],
                    "envs":       {env: mol["conformers"]},
                    "label":      mol["label"],
                    "CycPeptMPDB_ID": cpmp_id,
                    "rep_frame_idxs": {env: mol["rep_frame_idx"]} if mol.get("rep_frame_idx") is not None else {},
                }
                if "SMILES" in mol:
                    entry["SMILES"] = mol["SMILES"]
                if "Structurally_Unique_ID" in mol:
                    entry["Structurally_Unique_ID"] = mol["Structurally_Unique_ID"]
                by_id[cpmp_id] = entry
            else:
                # Cross-env topology check (inline, includes nf)
                entry = by_id[cpmp_id]
                ref_env = next(iter(entry["envs"]))
                for field, label in [
                    ("adj", "adjacency"),
                    ("bond_types", "bond type"),
                    ("nf", "node feature"),
                ]:
                    if not np.array_equal(mol[field], entry[field]):
                        raise ValueError(
                            f"Cross-env {label} matrix mismatch for {cpmp_id} "
                            f"between {ref_env} and {env}."
                        )
                entry["envs"][env] = mol["conformers"]
                if mol.get("rep_frame_idx") is not None:
                    entry["rep_frame_idxs"][env] = mol["rep_frame_idx"]

    return list(by_id.values())


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

    from src.featurization import featurize_molecules

    base_kwargs = dict(
        data_dir=_REPO_ROOT / paths["data_dir"],
        csv_path=paths["csv_file"],
        target_col=data["target_col"],
        traj_dir=paths["traj_dir"],
    )

    env_str = "+".join(sorted(envs))

    for remove_h, tag in [(True, "noH"), (False, "withH")]:
        env_results: dict[str, list[dict]] = {}
        d_atom = None
        for env in envs:
            molecules, d = featurize_molecules(**base_kwargs, solvent=env, remove_h=remove_h)
            env_results[env] = molecules
            if d_atom is None:
                d_atom = d
            elif d != d_atom:
                raise ValueError(f"d_atom mismatch across envs: {d_atom} vs {d}")

        merged = _merge_env_molecules(env_results)
        output = out_dir / f"cache_traj_{env_str}_{tag}.pt"
        torch.save({"molecules": merged, "d_atom": d_atom, "envs": sorted(envs)}, output)
        logging.info(f"Saved {len(merged)} molecules (d_atom={d_atom}, envs={sorted(envs)}) -> {output}")


if __name__ == "__main__":
    main()
