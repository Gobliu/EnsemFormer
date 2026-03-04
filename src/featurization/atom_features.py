"""Atom-level and bond-level featurization utilities.

Based on:
  Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks"
  Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction"
"""

import numpy as np
from rdkit import Chem
from sklearn.metrics import pairwise_distances


def one_hot_vector(val, lst):
    """Convert a value to a one-hot vector based on options in lst."""
    if val not in lst:
        val = lst[-1]
    return list(map(lambda x: x == val, lst))


def get_atom_features(atom, one_hot_formal_charge: bool = False) -> np.ndarray:
    """Compute atom features as a 1-D float32 array.

    Features: atomic number (11), degree (6), num H (5),
    formal charge (3 one-hot or 1 raw), in-ring (1), aromatic (1).

    Parameters
    ----------
    atom : rdchem.Atom
    one_hot_formal_charge : bool

    Returns
    -------
    np.ndarray  shape (25,) or (27,) depending on one_hot_formal_charge
    """
    attributes = []
    attributes += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    attributes += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    if one_hot_formal_charge:
        attributes += one_hot_vector(atom.GetFormalCharge(), [-1, 0, 1])
    else:
        attributes.append(atom.GetFormalCharge())
    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())
    return np.array(attributes, dtype=np.float32)


# ---------------------------------------------------------------------------
# Bond-type featurization
# ---------------------------------------------------------------------------

_BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


def get_bond_type_matrix(mol) -> np.ndarray:
    """Build an (N, N) integer matrix of bond types for an RDKit Mol.

    Returns
    -------
    np.ndarray  (N_atoms, N_atoms), dtype int8
        0 = no bond, 1 = single, 2 = double, 3 = triple, 4 = aromatic.
    """
    n = mol.GetNumAtoms()
    bt = np.zeros((n, n), dtype=np.int8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        bt[i, j] = bt[j, i] = _BOND_TYPE_MAP.get(bond.GetBondType(), 1)
    return bt


def featurize_mol(
    mol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Featurize a single RDKit Mol object.

    The mol must already have a conformer (3D coordinates embedded).

    Parameters
    ----------
    mol : rdchem.Mol

    Returns
    -------
    node_features    : np.ndarray  (N_atoms, 25)
    adj_matrix       : np.ndarray  (N_atoms, N_atoms), dtype bool
    dist_matrix      : np.ndarray  (N_atoms, N_atoms)
    pos_matrix       : np.ndarray  (N_atoms, 3)
    bond_type_matrix : np.ndarray  (N_atoms, N_atoms), dtype int8
    """
    node_features = np.array(
        [get_atom_features(atom) for atom in mol.GetAtoms()]
    )

    adj_matrix = np.eye(mol.GetNumAtoms(), dtype=np.bool_)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        adj_matrix[i, j] = adj_matrix[j, i] = True

    bond_type_matrix = get_bond_type_matrix(mol)

    conf = mol.GetConformer()
    pos_matrix = np.array(
        [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
         for k in range(mol.GetNumAtoms())],
        dtype=np.float32,
    )
    dist_matrix = pairwise_distances(pos_matrix)

    return node_features, adj_matrix, dist_matrix, pos_matrix, bond_type_matrix


def pad_array(array: np.ndarray, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Pad a 2-D array with zeros to the desired shape."""
    padded = np.zeros(shape, dtype=dtype)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded
