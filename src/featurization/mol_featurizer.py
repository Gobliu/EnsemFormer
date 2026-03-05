"""Molecule-level trajectory featurizer.

Hierarchy: CSV (all molecules) → single molecule → per env → per traj → per frame.
"""

import logging
import pathlib

import numpy as np
import pandas as pd

from src.featurization.pdb_loader import load_frames_from_traj_pdb


_ENV_MAP = {
    "water":  ("Water",  "H2O_Traj",    "Water_RepFrame"),
    "hexane": ("Hexane", "Hexane_Traj", "Hexane_RepFrame"),
}


def featurize_single_molecule(
    row,
    target_col: str,
    envs: list[str],
    traj_dir: pathlib.Path,
    remove_h: bool = True,
) -> dict | None:
    """Featurize one molecule across all environments.

    For each env, loads the trajectory PDB and extracts all frames.
    Cross-env topology consistency (node_feat, adj, bond_types) is verified inline.

    Parameters
    ----------
    row : pandas namedtuple
        A single CSV row from ``df.itertuples()``.
    target_col : str
        Name of the target column (e.g. ``"PAMPA"``).
    envs : list[str]
        Environments to process (e.g. ``["water", "hexane"]``).
    traj_dir : Path
        Root directory containing per-env trajectory subdirectories.
    remove_h : bool
        If True, strip hydrogen atoms before featurization.

    Returns
    -------
    dict or None
        Merged molecule dict with keys ``node_feat``, ``adj``, ``bond_types``,
        ``envs`` (dict mapping env name to list of (dist, coords) tuples),
        ``rep_frame_idxs``, ``label``, ``CycPeptMPDB_ID``, and optionally
        ``SMILES`` and ``Structurally_Unique_ID``.  Returns None if no
        environment produced valid frames.
    """
    cpmp_id = str(row.CycPeptMPDB_ID)

    ref_node_feat = None
    ref_adj = None
    ref_bond_types = None
    ref_env = None
    env_frames: dict[str, list] = {}
    rep_frame_idxs: dict[str, int] = {}

    for env in envs:
        subdir, suffix, rep_col = _ENV_MAP[env.lower()]
        traj_base = traj_dir / subdir / "Trajectories"
        pdb_path = str(traj_base / f"{row.Source}_{cpmp_id}_{suffix}.pdb")

        try:
            node_feat, adj, bond_types, frames = load_frames_from_traj_pdb(
                pdb_path, remove_h=remove_h,
            )
        except ValueError as e:
            logging.warning(f"Skipping env '{env}' for {cpmp_id}: {e}")
            continue
        if not frames:
            logging.warning(f"Skipping env '{env}' for {cpmp_id}: no valid frames.")
            continue

        # Cross-env topology check
        if ref_node_feat is None:
            ref_node_feat, ref_adj, ref_bond_types, ref_env = node_feat, adj, bond_types, env
        else:
            for arr, ref, name in [
                (node_feat, ref_node_feat, "node feature"),
                (adj, ref_adj, "adjacency"),
                (bond_types, ref_bond_types, "bond type"),
            ]:
                if not np.array_equal(arr, ref):
                    raise ValueError(
                        f"Cross-env {name} matrix mismatch for {cpmp_id} "
                        f"between {ref_env} and {env}."
                    )

        env_frames[env] = frames

        if not hasattr(row, rep_col):
            raise AttributeError(
                f"CSV is missing column '{rep_col}' needed for env '{env}'. "
                f"Expected columns: {[c for _, _, c in _ENV_MAP.values()]}."
            )
        rep_idx = getattr(row, rep_col)
        rep_frame_idxs[env] = int(rep_idx)

    if not env_frames:
        return None

    mol_dict: dict = {
        "node_feat":      ref_node_feat,
        "adj":            ref_adj,
        "bond_types":     ref_bond_types,
        "envs":           env_frames,
        "rep_frame_idxs": rep_frame_idxs,
        "label":          float(getattr(row, target_col)),
        "CycPeptMPDB_ID": cpmp_id,
    }
    if hasattr(row, "SMILES"):
        mol_dict["SMILES"] = str(row.SMILES)
    if hasattr(row, "Structurally_Unique_ID"):
        mol_dict["Structurally_Unique_ID"] = str(row.Structurally_Unique_ID)

    return mol_dict


def featurize_all_molecules(
    csv_path,
    target_col: str,
    traj_dir,
    envs: list[str],
    remove_h: bool = True,
) -> tuple[list[dict], int]:
    """Featurize all molecules from trajectory PDB files across environments.

    Reads the CSV once and processes each molecule across all requested
    environments, producing merged molecule dicts ready for caching.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    target_col : str
        CSV column name for the regression target.
    traj_dir : str or Path
        Root directory of the CycPeptMPDB_4D trajectory database.
    envs : list[str]
        Environments to process (e.g. ``["water", "hexane"]``).
    remove_h : bool
        If True, strip hydrogen atoms before featurization.

    Returns
    -------
    (molecules, d_atom) where molecules is a list of merged dicts.
    """
    csv_path = pathlib.Path(csv_path)
    traj_dir = pathlib.Path(traj_dir)
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. Columns: {list(df.columns)}"
        )

    h_tag = "noH" if remove_h else "withH"
    logging.info(
        f"Featurizing {len(df)} molecules ({h_tag}, envs={envs}) from {csv_path} ..."
    )

    molecules: list[dict] = []

    for row in df.itertuples():
        mol_dict = featurize_single_molecule(
            row, target_col, envs, traj_dir, remove_h,
        )
        if mol_dict is None:
            logging.warning(
                f"Skipping molecule {row.CycPeptMPDB_ID}: no valid frames in any env."
            )
            continue
        molecules.append(mol_dict)

    if not molecules:
        raise RuntimeError("No molecules were successfully featurized.")

    d_atom = molecules[0]["node_feat"].shape[1]
    logging.info(f"Featurized {len(molecules)} molecules. d_atom={d_atom}")
    return molecules, d_atom
