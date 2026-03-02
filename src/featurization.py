"""Molecular featurization utilities for EnsemFormer.

Provides single-conformer featurization (node features, adjacency, distance
matrix) and multi-conformer ensemble loading from SMILES or PDB files.

Based on:
  Shang et al "Edge Attention-based Multi-Relational Graph Convolutional Networks"
  Coley et al "Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction"
"""

import logging
import os
import subprocess

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.metrics import pairwise_distances
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Atom-level featurization
# ---------------------------------------------------------------------------

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


def featurize_mol(
    mol,
    add_dummy_node: bool = True,
    one_hot_formal_charge: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Featurize a single RDKit Mol object.

    The mol must already have a conformer (3D coordinates embedded).

    Parameters
    ----------
    mol : rdchem.Mol
    add_dummy_node : bool
        If True, prepend a dummy node with a special flag feature.
    one_hot_formal_charge : bool

    Returns
    -------
    node_features : np.ndarray  (N_atoms[+1], F)
    adj_matrix    : np.ndarray  (N_atoms[+1], N_atoms[+1])
    dist_matrix   : np.ndarray  (N_atoms[+1], N_atoms[+1])
    """
    node_features = np.array(
        [get_atom_features(atom, one_hot_formal_charge) for atom in mol.GetAtoms()]
    )

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        adj_matrix[i, j] = adj_matrix[j, i] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array(
        [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
         for k in range(mol.GetNumAtoms())]
    )
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.0
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def pad_array(array: np.ndarray, shape: tuple, dtype=np.float32) -> np.ndarray:
    """Pad a 2-D array with zeros to the desired shape."""
    padded = np.zeros(shape, dtype=dtype)
    padded[: array.shape[0], : array.shape[1]] = array
    return padded


# ---------------------------------------------------------------------------
# Single-conformer loading (legacy / for single-conformer baselines)
# ---------------------------------------------------------------------------

def load_single_conformer_from_smiles(
    x_smiles,
    labels,
    ff: str = "mmff",
    ignore_interfrag: bool = True,
    add_dummy_node: bool = True,
    one_hot_formal_charge: bool = False,
) -> tuple[list, list]:
    """Featurize a list of SMILES into single-conformer graphs.

    Returns
    -------
    x_all : list of [node_features, adj_matrix, dist_matrix]
    y_all : list of [label]
    """
    x_all, y_all = [], []
    for smiles, label in zip(x_smiles, labels):
        try:
            mol = MolFromSmiles(smiles)
            try:
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, maxAttempts=5000)
                if ff == "uff":
                    AllChem.UFFOptimizeMolecule(mol, ignoreInterfragInteractions=ignore_interfrag)
                else:
                    AllChem.MMFFOptimizeMolecule(mol, ignoreInterfragInteractions=ignore_interfrag)
                mol = Chem.RemoveHs(mol)
            except Exception:
                AllChem.Compute2DCoords(mol)

            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            x_all.append([afm, adj, dist])
            y_all.append([label])
        except (ValueError, Exception) as e:
            logging.warning(f"SMILES ({smiles}) could not be featurized. Reason: {e}")
    return x_all, y_all


# ---------------------------------------------------------------------------
# Multi-conformer ensemble loading
# ---------------------------------------------------------------------------

def load_ensemble_from_smiles(
    smiles: str,
    n_conformers: int,
    ff: str = "mmff",
    max_attempts: int = 5000,
    add_dummy_node: bool = True,
    one_hot_formal_charge: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate N conformers from a SMILES string using RDKit.

    All conformers share the same node_features and adj_matrix.
    Only dist_matrix differs per conformer (different 3D geometry).

    Parameters
    ----------
    smiles : str
    n_conformers : int
        Desired number of conformers.
    ff : str
        Force field: 'mmff' or 'uff'.
    max_attempts : int
        Maximum embedding attempts per conformer.
    add_dummy_node : bool
    one_hot_formal_charge : bool

    Returns
    -------
    list of (node_features, adj_matrix, dist_matrix) tuples.
    May be shorter than n_conformers if embedding fails for some.
    """
    try:
        mol = MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Cannot parse SMILES: {smiles}")
            return []

        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.numThreads = 0
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers,
                                               maxAttempts=max_attempts,
                                               params=params)

        if len(conf_ids) == 0:
            logging.warning(f"EmbedMultipleConfs failed for SMILES: {smiles}. Trying 2D fallback.")
            AllChem.Compute2DCoords(mol)
            mol = Chem.RemoveHs(mol)
            afm, adj, dist = featurize_mol(mol, add_dummy_node, one_hot_formal_charge)
            return [(afm, adj, dist)]

        if ff == "uff":
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        else:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

        mol = Chem.RemoveHs(mol)

        # Node features and adjacency are conformation-independent
        node_features = np.array(
            [get_atom_features(atom, one_hot_formal_charge) for atom in mol.GetAtoms()]
        )
        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
            adj_matrix[i, j] = adj_matrix[j, i] = 1

        results = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())]
            )
            dist_matrix = pairwise_distances(pos_matrix)

            nf = node_features.copy()
            adj = adj_matrix.copy()

            if add_dummy_node:
                m = np.zeros((nf.shape[0] + 1, nf.shape[1] + 1))
                m[1:, 1:] = nf
                m[0, 0] = 1.0
                nf = m

                m = np.zeros((adj.shape[0] + 1, adj.shape[1] + 1))
                m[1:, 1:] = adj
                adj = m

                m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
                m[1:, 1:] = dist_matrix
                dist_matrix = m

            results.append((nf, adj, dist_matrix))

        return results

    except Exception as e:
        logging.warning(f"load_ensemble_from_smiles failed for '{smiles}': {e}")
        return []


def _featurize_pdb_file(
    pdb_path: str,
    remove_h: bool = True,
    add_dummy_node: bool = True,
    one_hot_formal_charge: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load and featurize a single PDB file.

    Falls back to Open Babel if RDKit fails to parse the file.

    Returns
    -------
    (node_features, adj_matrix, dist_matrix) or None if parsing failed.
    """
    def _try_obabel(path: str):
        tmp = path.replace(".pdb", "_tmp_obabel.mol")
        try:
            subprocess.run(
                ["obabel", path, "-O", tmp, "--gen3d"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            mol = Chem.MolFromMolFile(tmp, removeHs=remove_h, sanitize=True)
            if mol is None:
                logging.error(f"[RDKit] Failed to parse MOL from Open Babel output: {tmp}")
            else:
                logging.info(f"[Open Babel] Recovered molecule from {path}")
            return mol
        except subprocess.CalledProcessError as e:
            logging.error(f"[Open Babel] Conversion failed for {path}\n{e.stderr}")
            return None
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    mol = Chem.MolFromPDBFile(pdb_path, removeHs=remove_h, sanitize=True)
    if mol is None:
        logging.warning(f"[RDKit] Failed to parse {pdb_path}, trying Open Babel fallback...")
        mol = _try_obabel(pdb_path)
    if mol is None:
        logging.error(f"[ERROR] Skipping {pdb_path} — cannot parse even after Open Babel fallback.")
        return None

    return featurize_mol(mol, add_dummy_node, one_hot_formal_charge)


def load_ensemble_from_pdb(
    pdb_paths: list[str],
    remove_h: bool = True,
    add_dummy_node: bool = True,
    one_hot_formal_charge: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load multiple PDB files as a conformer ensemble.

    Each PDB file is treated as one conformer. Node features and adjacency
    are taken from the first successfully parsed conformer and reused for all.

    Parameters
    ----------
    pdb_paths : list[str]
        One path per conformer.
    remove_h : bool
    add_dummy_node : bool
    one_hot_formal_charge : bool

    Returns
    -------
    list of (node_features, adj_matrix, dist_matrix) tuples.
    """
    results = []
    for path in pdb_paths:
        out = _featurize_pdb_file(path, remove_h, add_dummy_node, one_hot_formal_charge)
        if out is not None:
            results.append(out)
    return results


# ---------------------------------------------------------------------------
# Dataset classes (single-conformer; used by legacy baselines)
# ---------------------------------------------------------------------------

class Molecule:
    """Single-conformer molecule data container."""

    def __init__(self, x, y, index):
        self.node_features = x[0]
        self.adjacency_matrix = x[1]
        self.distance_matrix = x[2]
        self.y = y
        self.index = index


class MolDataset(Dataset):
    """PyTorch Dataset wrapping a list of Molecule objects."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return MolDataset(self.data_list[key])
        return self.data_list[key]
