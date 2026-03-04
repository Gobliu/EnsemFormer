"""Molecular featurization utilities for EnsemFormer.

This package is split into sub-modules. All public symbols are
re-exported here for backward compatibility.
"""

from src.featurization.atom_features import (
    one_hot_vector,
    get_atom_features,
    _BOND_TYPE_MAP,
    get_bond_type_matrix,
    featurize_mol,
    pad_array,
)
from src.featurization.smiles_loading import (
    load_single_conformer_from_smiles,
    load_ensemble_from_smiles,
)
from src.featurization.pdb_loading import (
    _featurize_pdb_file,
    load_frames_from_traj_pdb,
    load_ensemble_from_pdb,
)
from src.featurization.pipeline import (
    _SOLVENT_MAP,
    featurize_molecules,
)
from src.featurization.legacy import Molecule, MolDataset

__all__ = [
    "one_hot_vector", "get_atom_features", "get_bond_type_matrix",
    "featurize_mol", "pad_array",
    "load_single_conformer_from_smiles", "load_ensemble_from_smiles",
    "load_frames_from_traj_pdb", "load_ensemble_from_pdb",
    "featurize_molecules",
    "Molecule", "MolDataset",
]
