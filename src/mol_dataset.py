"""Multi-conformer dataset for EnsemFormer.

Provides:
- ``MolItem``: holds topology arrays once + N (dist, coords) conformer tuples + label
- ``MolList``: PyTorch Dataset wrapping the above
- ``conformer_collate_fn``: pads atoms and conformers and builds batch dicts

DataModule (``MolLoader``) lives in ``src.mol_loader``.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class MolItem:
    """A single molecule represented as an ensemble of conformers.

    Topology arrays (node_feat, adj, bond_types) are stored once at molecule
    level since they are identical across all conformer frames. Per-frame data
    is stored as a list of (dist, coords) 2-tuples.

    Parameters
    ----------
    node_feat : np.ndarray (N_atoms, F)
        Node features — topology-invariant.
    adj : np.ndarray (N_atoms, N_atoms)
        Adjacency matrix — topology-invariant.
    bond_types : np.ndarray (N_atoms, N_atoms)
        Bond type matrix — topology-invariant.
    conformers : list of (dist, coords) tuples
        Per-conformer data: dist (N_atoms, N_atoms), coords (N_atoms, 3).
    label : float
        Regression target.
    CycPeptMPDB_ID : str
        Unique molecule identifier from CycPeptMPDB.
    SMILES : str or None
    Structurally_Unique_ID : str or None
    rep_frame_idxs : dict[str, int] or None
        Per-env representative frame indices, e.g. {"water": 42, "hexane": 17}.
    """

    def __init__(
        self,
        node_feat: np.ndarray,
        adj: np.ndarray,
        bond_types: np.ndarray,
        conformers: list,
        label: float,
        CycPeptMPDB_ID: str,
        SMILES: str | None = None,
        Structurally_Unique_ID: str | None = None,
        rep_frame_idxs: dict | None = None,
    ):
        self.node_feat = node_feat
        self.adj = adj
        self.bond_types = bond_types
        self.conformers = conformers
        self.y = float(label)
        self.CycPeptMPDB_ID = CycPeptMPDB_ID
        self.SMILES = SMILES
        self.Structurally_Unique_ID = Structurally_Unique_ID
        self.rep_frame_idxs = rep_frame_idxs

    @property
    def n_conformers(self) -> int:
        return len(self.conformers)


class MolList(Dataset):
    """PyTorch Dataset of MolItem objects."""

    def __init__(self, data_list: list[MolItem]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MolList(self.data_list[key])
        return self.data_list[key]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def conformer_collate_fn(
    batch: list[MolItem],
) -> dict[str, torch.Tensor]:
    """Collate a list of molecules into a padded batch dict.

    Each molecule has a variable number of conformers; each conformer has a
    variable number of atoms. Both dimensions are padded to the batch maximum.

    Returns
    -------
    dict with keys:
        'node_feat'      : FloatTensor (B, N_conf_max, N_atoms_max, F)
        'adj'            : FloatTensor (B, N_conf_max, N_atoms_max, N_atoms_max)
        'dist'           : FloatTensor (B, N_conf_max, N_atoms_max, N_atoms_max)
        'coords'         : FloatTensor (B, N_conf_max, N_atoms_max, 3)
        'atom_mask'      : BoolTensor  (B, N_atoms_max) — True for real atoms
        'conformer_mask' : BoolTensor  (B, N_conf_max) — True for real conformers
        'target'         : FloatTensor (B, 1)
        'bond_type'      : LongTensor  (B, N_conf_max, N_atoms_max, N_atoms_max)  — if available
    """
    B = len(batch)
    N_conf_max = max(mol.n_conformers for mol in batch)
    N_atoms_max = max(mol.node_feat.shape[0] for mol in batch)
    F = batch[0].node_feat.shape[1]

    # Topology arrays: (B, N_atoms_max, ...) — broadcast to conformer dim after filling
    node_feat_2d = np.zeros((B, N_atoms_max, F), dtype=np.float32)
    adj_2d       = np.zeros((B, N_atoms_max, N_atoms_max), dtype=np.float32)
    bond_type_2d = np.zeros((B, N_atoms_max, N_atoms_max), dtype=np.int64)

    # Per-conformer arrays
    dist          = np.full((B, N_conf_max, N_atoms_max, N_atoms_max), 1e6, dtype=np.float32)
    coords        = np.zeros((B, N_conf_max, N_atoms_max, 3), dtype=np.float32)
    conformer_mask = np.zeros((B, N_conf_max), dtype=bool)
    atom_mask     = np.zeros((B, N_atoms_max), dtype=bool)
    targets       = np.zeros((B, 1), dtype=np.float32)

    for i, mol in enumerate(batch):
        targets[i, 0] = mol.y
        n_a = mol.node_feat.shape[0]
        atom_mask[i, :n_a] = True

        # Topology — stored once per molecule
        node_feat_2d[i, :n_a, :]    = mol.node_feat
        adj_2d[i, :n_a, :n_a]       = mol.adj
        bond_type_2d[i, :n_a, :n_a] = mol.bond_types

        # Per-conformer (dist, coords)
        for j, (d, pos) in enumerate(mol.conformers):
            dist[i, j, :n_a, :n_a] = d
            coords[i, j, :n_a, :]  = pos
            conformer_mask[i, j]   = True

    # Broadcast topology to conformer dimension.
    # .contiguous() is required: expand() returns a stride-0 view that breaks
    # view()/reshape() inside the CPMP encoder.
    node_feat_t = torch.from_numpy(node_feat_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()
    adj_t       = torch.from_numpy(adj_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()
    bond_type_t = torch.from_numpy(bond_type_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()

    return {
        "node_feat":      node_feat_t,
        "adj":            adj_t,
        "dist":           torch.from_numpy(dist),
        "coords":         torch.from_numpy(coords),
        "atom_mask":      torch.from_numpy(atom_mask),
        "conformer_mask": torch.from_numpy(conformer_mask),
        "target":         torch.from_numpy(targets),
        "bond_type":      bond_type_t,
    }

