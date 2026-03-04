"""Trajectory featurization pipeline (used by scripts/traj_preprocess.py)."""

import logging
import pathlib

import numpy as np
import pandas as pd

from src.featurization.pdb_loading import load_frames_from_traj_pdb


_SOLVENT_MAP = {
    "water":  ("Water",  "H2O_Traj",    "Water_RepFrame"),
    "hexane": ("Hexane", "Hexane_Traj", "Hexane_RepFrame"),
}


def featurize_molecules(
    data_dir,
    csv_path: str,
    target_col: str,
    traj_dir,
    solvent: str,
    remove_h: bool = True,
) -> tuple[list, int]:
    """Featurize all molecules from trajectory PDB files.

    The CSV must have columns ``Source`` and ``CycPeptMPDB_ID``. PDB files are
    read from ``<traj_dir>/{Water|Hexane}/Trajectories/{Source}_{ID}_{suffix}.pdb``,
    matching the CycPeptMPDB_4D dataset layout.

    Each molecule is returned as a dict with keys ``nf``, ``adj``,
    ``bond_types`` (topology arrays stored once), ``conformers`` (list of
    (dist, coords) 2-tuples), ``label``, ``CycPeptMPDB_ID``, and optionally
    ``SMILES`` and ``Structurally_Unique_ID``.

    Parameters
    ----------
    remove_h : bool
        If True, strip hydrogen atoms before featurization.

    Returns
    -------
    (molecules, d_atom) where molecules is a list of dicts.
    """
    data_dir = pathlib.Path(data_dir)
    csv_file = data_dir / csv_path
    df = pd.read_csv(csv_file)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. Columns: {list(df.columns)}"
        )
    labels = df[target_col].values.astype(np.float32)

    subdir, suffix, rep_col = _SOLVENT_MAP[solvent.lower()]
    traj_base = pathlib.Path(traj_dir) / subdir / "Trajectories"
    h_tag = "noH" if remove_h else "withH"
    logging.info(f"Featurizing {len(df)} molecules ({h_tag}) from {csv_file} ...")
    molecules: list[dict] = []

    has_smiles = "SMILES" in df.columns
    has_su_id = "Structurally_Unique_ID" in df.columns
    has_rep_frame = rep_col in df.columns

    for row, label in zip(df.itertuples(), labels):
        pdb_path = str(traj_base / f"{row.Source}_{row.CycPeptMPDB_ID}_{suffix}.pdb")
        try:
            nf, adj, bt, frames = load_frames_from_traj_pdb(
                pdb_path,
                remove_h=remove_h,
            )
        except ValueError as e:
            logging.warning(f"Skipping molecule {row.CycPeptMPDB_ID} ({pdb_path}): {e}")
            continue
        if not frames:
            logging.warning(f"Skipping molecule {row.CycPeptMPDB_ID}: no valid PDB conformers.")
            continue
        rep_frame_idx = int(getattr(row, rep_col)) if has_rep_frame else None
        mol_dict: dict = {
            "nf":         nf,
            "adj":        adj,
            "bond_types": bt,
            "conformers": frames,
            "label":      float(label),
            "CycPeptMPDB_ID": str(row.CycPeptMPDB_ID),
            "rep_frame_idx":  rep_frame_idx,
        }
        if has_smiles:
            mol_dict["SMILES"] = str(row.SMILES)
        if has_su_id:
            mol_dict["Structurally_Unique_ID"] = str(row.Structurally_Unique_ID)
        molecules.append(mol_dict)

    if not molecules:
        raise RuntimeError("No molecules were successfully featurized.")

    d_atom = molecules[0]["nf"].shape[1]
    logging.info(f"Featurized {len(molecules)} molecules. d_atom={d_atom}")
    return molecules, d_atom
