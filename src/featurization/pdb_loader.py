"""Load conformer data from multi-MODEL trajectory PDB files."""

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

from src.featurization.graph_builder import mol_to_graph


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

    Topology arrays (node_feat, adj, bond_types) are identical across frames and
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
    (node_feat, adj, bond_types, frames) where:
        node_feat  : (N_atoms, F) node features — None if no frames loaded
        adj        : (N_atoms, N_atoms) adjacency matrix — None if no frames loaded
        bond_types : (N_atoms, N_atoms) bond type matrix — None if no frames loaded
        frames     : list of (dist, coords) 2-tuples — empty list if no frames loaded
    """
    with open(traj_path, "r") as f:
        raw = f.read()

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
        raise ValueError(f"No MODEL/ENDMDL blocks found in {traj_path}")

    target_indices = frame_indices if frame_indices is not None else sorted(frame_blocks.keys())

    frames: list[tuple[np.ndarray, np.ndarray]] = []
    ref_node_feat = None
    ref_adj = None
    ref_bond_types = None

    for idx in target_indices:
        if idx not in frame_blocks:
            logging.warning(f"[TrajPDB] Frame {idx} not found in {traj_path} (available: {min(frame_blocks)}-{max(frame_blocks)})")
            continue
        mol = Chem.MolFromPDBBlock(frame_blocks[idx], removeHs=False, sanitize=True)
        if mol is None:
            raise ValueError(f"RDKit failed to parse frame {idx} of {traj_path}")
        if len(rdmolops.GetMolFrags(mol)) != 1:
            msg = f"Frame {idx} of {traj_path} has disconnected fragments."
            logging.warning(msg)
            raise RuntimeError(msg)
        if remove_h:
            # Record PDB H counts on heavy atoms before stripping
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() != 1:
                    n_hs = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
                    atom.SetUnsignedProp("_pdb_num_hs", n_hs)
            mol = Chem.RemoveHs(mol)
            for atom in mol.GetAtoms():
                atom.SetNumExplicitHs(atom.GetUnsignedProp("_pdb_num_hs"))
                atom.SetNoImplicit(True)
        node_feat, adj, dist, coords, bond_types = mol_to_graph(mol)

        if ref_adj is None:
            ref_node_feat = node_feat
            ref_adj = adj
            ref_bond_types = bond_types
        else:
            for arr, ref, name in [
                (adj, ref_adj, "adjacency"),
                (bond_types, ref_bond_types, "bond type"),
                (node_feat, ref_node_feat, "node feature"),
            ]:
                if not np.array_equal(arr, ref):
                    raise ValueError(
                        f"{name.capitalize()} matrix inconsistency at frame {idx} "
                        f"of {traj_path}: topology must be invariant across "
                        "trajectory frames."
                    )

        frames.append((dist, coords))

    if not frames:
        return None, None, None, []

    return ref_node_feat, ref_adj, ref_bond_types, frames
