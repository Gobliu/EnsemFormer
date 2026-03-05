"""Graph builder: convert an RDKit Mol into graph arrays (node_feat, adj, dist, coords, bond_types).

Based on:
  Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks"
  Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction"
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from sklearn.metrics import pairwise_distances


def one_hot_vector(val, lst):
    """Convert a value to a one-hot vector based on options in lst."""
    if val not in lst:
        val = lst[-1]
    return list(map(lambda x: x == val, lst))


def get_atom_features(atom) -> np.ndarray:
    """Compute atom features as a 1-D float32 array.

    Features: atomic number (11), degree (6), num H (5),
    formal charge (1 scalar), in-ring (1), aromatic (1).

    Parameters
    ----------
    atom : rdchem.Atom

    Returns
    -------
    np.ndarray  shape (25,)
    """
    attributes = []
    attributes += one_hot_vector(atom.GetAtomicNum(), [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    attributes += one_hot_vector(len(atom.GetNeighbors()), [0, 1, 2, 3, 4, 5])
    attributes += one_hot_vector(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
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
    bond_type = np.zeros((n, n), dtype=np.int8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        bond_type[i, j] = bond_type[j, i] = _BOND_TYPE_MAP[bond.GetBondType()]
    return bond_type


def mol_to_graph(
    mol,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a single RDKit Mol object into graph arrays.

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
    n_frags = len(rdmolops.GetMolFrags(mol))
    if n_frags != 1:
        raise ValueError(
            f"Molecule has {n_frags} disconnected fragments, expected 1"
        )

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
