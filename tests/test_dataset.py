"""Smoke tests for MolList and conformer_collate_fn."""

import numpy as np
import torch
import pytest

from src.mol_dataset import (
    MolItem,
    MolList,
    conformer_collate_fn,
)
from src.mol_loader import MolLoader


def _make_molecule(n_conformers=3, n_atoms=5, d_atom=25, label=0.5, CycPeptMPDB_ID="test_0"):
    nf  = np.random.randn(n_atoms, d_atom).astype(np.float32)
    adj = np.eye(n_atoms, dtype=np.float32)
    bt  = np.zeros((n_atoms, n_atoms), dtype=np.int8)
    conformers = [
        (
            np.random.rand(n_atoms, n_atoms).astype(np.float32),   # dist
            np.random.randn(n_atoms, 3).astype(np.float32),        # coords
        )
        for _ in range(n_conformers)
    ]
    return MolItem(nf, adj, bt, conformers, label, CycPeptMPDB_ID)


def test_molecule_n_conformers():
    mol = _make_molecule(n_conformers=4)
    assert mol.n_conformers == 4


def test_dataset_len():
    mols = [_make_molecule(CycPeptMPDB_ID=str(i)) for i in range(5)]
    ds = MolList(mols)
    assert len(ds) == 5


def test_dataset_getitem():
    mols = [_make_molecule(CycPeptMPDB_ID=str(i)) for i in range(3)]
    ds = MolList(mols)
    mol = ds[1]
    assert mol.CycPeptMPDB_ID == "1"


def test_conformer_collate_fn_shapes():
    mols = [_make_molecule(n_conformers=3, n_atoms=5, d_atom=25) for _ in range(4)]
    batch = conformer_collate_fn(mols)

    B, N_conf, N_atoms = 4, 3, 5
    assert batch["node_feat"].shape == (B, N_conf, N_atoms, 25)
    assert batch["adj"].shape == (B, N_conf, N_atoms, N_atoms)
    assert batch["dist"].shape == (B, N_conf, N_atoms, N_atoms)
    assert batch["bond_type"].shape == (B, N_conf, N_atoms, N_atoms)
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


def test_topology_broadcast():
    """node_feat and adj should be identical across the conformer dimension."""
    mol = _make_molecule(n_conformers=4, n_atoms=5, d_atom=25)
    batch = conformer_collate_fn([mol])

    # node_feat[0, 0] == node_feat[0, 1] == ... (broadcast from mol.node_feat)
    for j in range(1, 4):
        assert torch.equal(batch["node_feat"][0, 0], batch["node_feat"][0, j])
        assert torch.equal(batch["adj"][0, 0], batch["adj"][0, j])
        assert torch.equal(batch["bond_type"][0, 0], batch["bond_type"][0, j])


def _make_selector(rep_frame_only=True, n_conformers=None):
    selector = MolLoader.__new__(MolLoader)
    selector._rep_frame_only = rep_frame_only
    selector._n_conformers = n_conformers
    return selector


def test_select_env_conformers_rep_frame_uses_1_based_index():
    selector = _make_selector(rep_frame_only=True)
    env_dict = {"water": [("d1", "c1"), ("d2", "c2"), ("d3", "c3")]}
    out = selector._select_env_conformers(
        env_dict=env_dict,
        envs_to_use=["water"],
        rep_frame_idxs={"water": 2},
    )
    assert out == [("d2", "c2")]


def test_select_env_conformers_rep_frame_missing_raises():
    selector = _make_selector(rep_frame_only=True)
    env_dict = {"water": [("d1", "c1"), ("d2", "c2")]}
    with pytest.raises(ValueError, match="rep_frame_only=True"):
        selector._select_env_conformers(
            env_dict=env_dict,
            envs_to_use=["water"],
            rep_frame_idxs={},
        )


def test_select_env_conformers_rep_frame_out_of_range_raises():
    selector = _make_selector(rep_frame_only=True)
    env_dict = {"water": [("d1", "c1"), ("d2", "c2")]}
    with pytest.raises(IndexError, match="valid 1..2"):
        selector._select_env_conformers(
            env_dict=env_dict,
            envs_to_use=["water"],
            rep_frame_idxs={"water": 3},
        )
