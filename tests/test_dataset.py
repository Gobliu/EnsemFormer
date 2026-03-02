"""Smoke tests for ConformerEnsembleDataset and conformer_collate_fn."""

import numpy as np
import torch
import pytest

from src.dataset import (
    ConformerEnsembleMolecule,
    ConformerEnsembleDataset,
    conformer_collate_fn,
)


def _make_molecule(n_conformers=3, n_atoms=5, d_atom=25, label=0.5, index=0):
    conformers = [
        (
            np.random.randn(n_atoms, d_atom).astype(np.float32),
            np.eye(n_atoms, dtype=np.float32),
            np.random.rand(n_atoms, n_atoms).astype(np.float32),
        )
        for _ in range(n_conformers)
    ]
    return ConformerEnsembleMolecule(conformers, label, index)


def test_molecule_n_conformers():
    mol = _make_molecule(n_conformers=4)
    assert mol.n_conformers == 4


def test_dataset_len():
    mols = [_make_molecule(index=i) for i in range(5)]
    ds = ConformerEnsembleDataset(mols)
    assert len(ds) == 5


def test_dataset_getitem():
    mols = [_make_molecule(index=i) for i in range(3)]
    ds = ConformerEnsembleDataset(mols)
    mol = ds[1]
    assert mol.index == 1


def test_conformer_collate_fn_shapes():
    mols = [_make_molecule(n_conformers=3, n_atoms=5, d_atom=25) for _ in range(4)]
    batch = conformer_collate_fn(mols)

    B, N_conf, N_atoms = 4, 3, 5
    assert batch["node_feat"].shape == (B, N_conf, N_atoms, 25)
    assert batch["adj"].shape == (B, N_conf, N_atoms, N_atoms)
    assert batch["dist"].shape == (B, N_conf, N_atoms, N_atoms)
    assert batch["conformer_mask"].shape == (B, N_conf)
    assert batch["atom_mask"].shape == (B, N_atoms)
    assert batch["target"].shape == (B, 1)


def test_conformer_collate_fn_variable_atoms():
    """Molecules with different atom counts should be padded to the max."""
    mol_small = _make_molecule(n_conformers=2, n_atoms=4, d_atom=25)
    mol_large = _make_molecule(n_conformers=2, n_atoms=7, d_atom=25)
    batch = conformer_collate_fn([mol_small, mol_large])

    assert batch["node_feat"].shape[2] == 7  # padded to max atoms
    assert batch["atom_mask"][0, 4:].all() == False  # small mol padded out


def test_conformer_collate_fn_variable_conformers():
    """Molecules with different conformer counts should be padded."""
    mol_few = _make_molecule(n_conformers=2, n_atoms=5, d_atom=25)
    mol_many = _make_molecule(n_conformers=5, n_atoms=5, d_atom=25)
    batch = conformer_collate_fn([mol_few, mol_many])

    assert batch["node_feat"].shape[1] == 5  # padded to max conformers
    # mol_few should have False in conformer_mask positions 2, 3, 4
    assert not batch["conformer_mask"][0, 2]
    assert not batch["conformer_mask"][0, 3]
    assert batch["conformer_mask"][1, 4]


def test_conformer_collate_fn_target():
    mols = [_make_molecule(label=float(i)) for i in range(3)]
    batch = conformer_collate_fn(mols)
    assert batch["target"].shape == (3, 1)
    assert float(batch["target"][1]) == pytest.approx(1.0)


def test_conformer_collate_fn_all_real_conformer_mask():
    mols = [_make_molecule(n_conformers=3) for _ in range(2)]
    batch = conformer_collate_fn(mols)
    assert batch["conformer_mask"].all(), "All conformers should be real"
