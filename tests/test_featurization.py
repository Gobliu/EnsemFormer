"""Smoke tests for src/featurization.py."""

import numpy as np
import pytest

from src.featurization import (
    get_atom_features,
    featurize_mol,
    load_ensemble_from_smiles,
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
    feats = get_atom_features(atom, one_hot_formal_charge=False)
    assert feats.dtype == np.float32
    assert feats.ndim == 1
    assert feats.shape[0] == 25  # 11 + 6 + 5 + 1 + 1 + 1


def test_get_atom_features_one_hot_charge():
    from rdkit import Chem
    mol = Chem.MolFromSmiles("c1ccccc1")
    atom = mol.GetAtomWithIdx(0)
    feats = get_atom_features(atom, one_hot_formal_charge=True)
    assert feats.shape[0] == 27  # 11 + 6 + 5 + 3 + 1 + 1


def test_featurize_mol_benzene():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles("c1ccccc1")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, maxAttempts=5000)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)

    nf, adj, dist, pos, bt = featurize_mol(mol)
    n_atoms = mol.GetNumAtoms()
    assert nf.shape == (n_atoms, 25)
    assert adj.shape == (n_atoms, n_atoms)
    assert dist.shape == (n_atoms, n_atoms)
    assert pos.shape == (n_atoms, 3)
    assert bt.shape == (n_atoms, n_atoms)


def test_load_ensemble_from_smiles_basic():
    conformers = load_ensemble_from_smiles(
        "c1ccccc1", n_conformers=4, ff="mmff"
    )
    assert len(conformers) > 0
    nf, adj, dist, pos, bt = conformers[0]
    assert nf.ndim == 2
    assert adj.shape[0] == adj.shape[1]
    assert dist.shape == adj.shape


def test_load_ensemble_from_smiles_all_share_topology():
    conformers = load_ensemble_from_smiles(
        "CC(C)CC", n_conformers=3, ff="mmff"
    )
    if len(conformers) < 2:
        pytest.skip("Too few conformers generated")
    nf0, adj0, dist0, pos0, bt0 = conformers[0]
    nf1, adj1, dist1, pos1, bt1 = conformers[1]
    np.testing.assert_array_equal(nf0, nf1)
    np.testing.assert_array_equal(adj0, adj1)
    # Distance matrices should differ (different 3D geometries)
    assert not np.allclose(dist0, dist1)


def test_load_ensemble_from_smiles_invalid():
    conformers = load_ensemble_from_smiles("not_a_smiles", n_conformers=4)
    assert conformers == []
