"""Smoke tests for src/featurization.py."""

import numpy as np
import pytest

from src.featurization import (
    get_atom_features,
    mol_to_graph,
    one_hot_vector,
)


def test_one_hot_vector_known():
    vec = one_hot_vector(6, [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999])
    assert vec[1] is True
    assert sum(vec) == 1


def test_one_hot_vector_unknown():
    vec = one_hot_vector(42, [5, 6, 7, 999])
    assert vec[-1] is True  # unknown → last bucket


def test_get_atom_features_benzene():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("c1ccccc1")
    atom = mol.GetAtomWithIdx(0)
    feats = get_atom_features(atom)
    assert feats.dtype == np.float32
    assert feats.ndim == 1
    assert feats.shape[0] == 25  # 11 + 6 + 5 + 1 + 1 + 1


def test_mol_to_graph_benzene():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=5000)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    nf, adj, dist, pos, bt = mol_to_graph(mol)
    n_atoms = mol.GetNumAtoms()
    assert nf.shape == (n_atoms, 25)
    assert adj.shape == (n_atoms, n_atoms)
    assert dist.shape == (n_atoms, n_atoms)
    assert pos.shape == (n_atoms, 3)
    assert bt.shape == (n_atoms, n_atoms)


