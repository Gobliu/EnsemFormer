"""Load conformer data from SMILES strings via RDKit embedding."""

import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolFromSmiles
from sklearn.metrics import pairwise_distances

from src.featurization.atom_features import (
    get_atom_features,
    get_bond_type_matrix,
    featurize_mol,
)


def load_single_conformer_from_smiles(
    x_smiles,
    labels,
    ff: str = "mmff",
    ignore_interfrag: bool = True,
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

            afm, adj, dist, pos, bt = featurize_mol(mol, one_hot_formal_charge)
            x_all.append([afm, adj, dist, pos, bt])
            y_all.append([label])
        except (ValueError, Exception) as e:
            logging.warning(f"SMILES ({smiles}) could not be featurized. Reason: {e}")
    return x_all, y_all


def load_ensemble_from_smiles(
    smiles: str,
    n_conformers: int,
    ff: str = "mmff",
    max_attempts: int = 5000,
    one_hot_formal_charge: bool = False,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Generate N conformers from a SMILES string using RDKit.

    All conformers share the same node_features, adj_matrix, and bond_type_matrix.
    Only dist_matrix and pos_matrix differ per conformer (different 3D geometry).

    Parameters
    ----------
    smiles : str
    n_conformers : int
        Desired number of conformers.
    ff : str
        Force field: 'mmff' or 'uff'.
    max_attempts : int
        Maximum embedding attempts per conformer.
    one_hot_formal_charge : bool

    Returns
    -------
    list of (node_features, adj_matrix, dist_matrix, pos_matrix, bond_type_matrix) tuples.
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
            return [featurize_mol(mol, one_hot_formal_charge)]

        if ff == "uff":
            AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=0)
        else:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0)

        mol = Chem.RemoveHs(mol)

        # Node features, adjacency, and bond types are conformation-independent
        node_features = np.array(
            [get_atom_features(atom, one_hot_formal_charge) for atom in mol.GetAtoms()]
        )
        adj_matrix = np.eye(mol.GetNumAtoms())
        for bond in mol.GetBonds():
            i = bond.GetBeginAtom().GetIdx()
            j = bond.GetEndAtom().GetIdx()
            adj_matrix[i, j] = adj_matrix[j, i] = 1
        bond_type_matrix = get_bond_type_matrix(mol)

        results = []
        for conf_id in conf_ids:
            conf = mol.GetConformer(conf_id)
            pos_matrix = np.array(
                [[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                 for k in range(mol.GetNumAtoms())],
                dtype=np.float32,
            )
            dist_matrix = pairwise_distances(pos_matrix)

            results.append((
                node_features.copy(),
                adj_matrix.copy(),
                dist_matrix,
                pos_matrix,
                bond_type_matrix.copy(),
            ))

        return results

    except Exception as e:
        logging.warning(f"load_ensemble_from_smiles failed for '{smiles}': {e}")
        return []
