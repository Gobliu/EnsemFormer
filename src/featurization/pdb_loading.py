"""Load conformer data from PDB files and multi-MODEL trajectory PDB files."""

import logging

import numpy as np
from rdkit import Chem

from src.featurization.atom_features import featurize_mol


def _featurize_pdb_file(
    pdb_path: str,
    remove_h: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and featurize a single PDB file.

    Raises
    ------
    ValueError
        If RDKit fails to parse the PDB file.
    """
    mol = Chem.MolFromPDBFile(pdb_path, removeHs=remove_h, sanitize=True)
    if mol is None:
        raise ValueError(f"RDKit failed to parse PDB file: {pdb_path}")

    return featurize_mol(mol)


def load_frames_from_traj_pdb(
    traj_path: str,
    frame_indices: list[int] | None = None,
    remove_h: bool = True,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Load selected frames from a multi-MODEL trajectory PDB as a conformer ensemble.

    Frame indices are 1-based (matching the renumbered MODEL numbers in the file).
    If frame_indices is None, all frames are loaded.

    Topology arrays (nf, adj, bond_types) are identical across frames and are
    returned once. Per-frame data is returned as a list of (dist, coords) 2-tuples.
    Consistency of all three topology arrays is verified across frames.

    Parameters
    ----------
    traj_path : str
        Path to a trajectory PDB with sequential MODEL 1 ... MODEL N blocks.
    frame_indices : list[int] or None
        1-based frame indices to extract. None means all frames.
    remove_h : bool
        Strip hydrogen atoms before featurization.
    Returns
    -------
    (nf, adj, bond_types, frames) where:
        nf         : (N_atoms, F) node features — None if no frames loaded
        adj        : (N_atoms, N_atoms) adjacency matrix — None if no frames loaded
        bond_types : (N_atoms, N_atoms) bond type matrix — None if no frames loaded
        frames     : list of (dist, coords) 2-tuples — empty list if no frames loaded
    """
    try:
        with open(traj_path, "r") as f:
            raw = f.read()
    except OSError as e:
        logging.error(f"[TrajPDB] Cannot open {traj_path}: {e}")
        return None, None, None, []

    # Split into per-frame blocks using MODEL/ENDMDL boundaries
    frame_blocks: dict[int, str] = {}
    current_idx: int | None = None
    current_lines: list[str] = []

    for line in raw.splitlines(keepends=True):
        if line.startswith("MODEL"):
            current_idx = int(line.split()[1])
            current_lines = [line]
        elif line.startswith("ENDMDL"):
            if current_idx is not None:
                current_lines.append(line)
                frame_blocks[current_idx] = "".join(current_lines)
                current_idx = None
                current_lines = []
        elif current_idx is not None:
            current_lines.append(line)

    if not frame_blocks:
        logging.error(f"[TrajPDB] No MODEL blocks found in {traj_path}")
        return None, None, None, []

    target_indices = frame_indices if frame_indices is not None else sorted(frame_blocks.keys())

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    ref_nf = None
    ref_adj = None
    ref_bt = None

    for idx in target_indices:
        if idx not in frame_blocks:
            logging.warning(f"[TrajPDB] Frame {idx} not found in {traj_path} (available: {min(frame_blocks)}-{max(frame_blocks)})")
            continue
        mol = Chem.MolFromPDBBlock(frame_blocks[idx], removeHs=remove_h, sanitize=True)
        if mol is None:
            raise ValueError(f"RDKit failed to parse frame {idx} of {traj_path}")
        nf, adj, dist, pos, bt = featurize_mol(mol)

        if ref_adj is None:
            ref_nf = nf
            ref_adj = adj
            ref_bt = bt
        else:
            if not np.array_equal(adj, ref_adj):
                raise ValueError(
                    f"Adjacency matrix inconsistency at frame {idx} of {traj_path}: "
                    "topology must be invariant across trajectory frames."
                )
            if not np.array_equal(bt, ref_bt):
                raise ValueError(
                    f"Bond type matrix inconsistency at frame {idx} of {traj_path}: "
                    "topology must be invariant across trajectory frames."
                )
            if not np.array_equal(nf, ref_nf):
                raise ValueError(
                    f"Node feature inconsistency at frame {idx} of {traj_path}: "
                    "topology must be invariant across trajectory frames."
                )

        frames.append((dist, pos))

    if not frames:
        return None, None, None, []

    return ref_nf, ref_adj, ref_bt, frames


def load_ensemble_from_pdb(
    pdb_paths: list[str],
    remove_h: bool = True,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Load multiple PDB files as a conformer ensemble.

    Each PDB file is treated as one conformer. Topology arrays (nf, adj,
    bond_types) are returned once; per-conformer data is a list of (dist,
    coords) 2-tuples. Consistency of topology is verified across files.

    Parameters
    ----------
    pdb_paths : list[str]
        One path per conformer.
    remove_h : bool

    Returns
    -------
    (nf, adj, bond_types, frames) — see load_frames_from_traj_pdb for details.
    """
    frames: list[tuple[np.ndarray, np.ndarray]] = []
    ref_nf = None
    ref_adj = None
    ref_bt = None

    for path in pdb_paths:
        nf, adj, dist, pos, bt = _featurize_pdb_file(path, remove_h)

        if ref_adj is None:
            ref_nf = nf
            ref_adj = adj
            ref_bt = bt
        else:
            if not np.array_equal(adj, ref_adj):
                raise ValueError(
                    f"Adjacency matrix inconsistency in {path}: "
                    "topology must be invariant across conformer PDB files."
                )
            if not np.array_equal(bt, ref_bt):
                raise ValueError(
                    f"Bond type matrix inconsistency in {path}: "
                    "topology must be invariant across conformer PDB files."
                )
            if not np.array_equal(nf, ref_nf):
                raise ValueError(
                    f"Node feature inconsistency in {path}: "
                    "topology must be invariant across conformer PDB files."
                )

        frames.append((dist, pos))

    if not frames:
        return None, None, None, []

    return ref_nf, ref_adj, ref_bt, frames
