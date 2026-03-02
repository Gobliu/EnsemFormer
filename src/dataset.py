"""Multi-conformer dataset for EnsemFormer.

Provides:
- ``ConformerEnsembleMolecule``: holds N conformer feature tuples + label
- ``ConformerEnsembleDataset``: PyTorch Dataset wrapping the above
- ``conformer_collate_fn``: pads atoms and conformers and builds batch dicts
- ``ConformerEnsembleDataModule``: full DataModule with featurization and splitting
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_module import DataModule
from src.featurization import (
    load_ensemble_from_smiles,
    load_ensemble_from_pdb,
    get_atom_features,
)
from src.utils import get_split_sizes

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ConformerEnsembleMolecule:
    """A single molecule represented as an ensemble of conformers.

    Parameters
    ----------
    conformers : list of (node_features, adj_matrix, dist_matrix) tuples
        Each tuple represents one 3-D conformer.
    label : float
        Regression target.
    index : int
        Dataset index.
    """

    def __init__(self, conformers: list, label: float, index: int):
        self.conformers = conformers  # list of (ndarray, ndarray, ndarray)
        self.y = float(label)
        self.index = index

    @property
    def n_conformers(self) -> int:
        return len(self.conformers)


class ConformerEnsembleDataset(Dataset):
    """PyTorch Dataset of ConformerEnsembleMolecule objects."""

    def __init__(self, data_list: list[ConformerEnsembleMolecule]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return ConformerEnsembleDataset(self.data_list[key])
        return self.data_list[key]


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def conformer_collate_fn(
    batch: list[ConformerEnsembleMolecule],
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
        'coords'         : not populated here (EGNN needs 3D pos separately)
        'atom_mask'      : BoolTensor  (B, N_atoms_max) — True for real atoms
        'conformer_mask' : BoolTensor  (B, N_conf_max) — True for real conformers
        'target'         : FloatTensor (B, 1)
    """
    B = len(batch)
    N_conf_max = max(mol.n_conformers for mol in batch)
    N_atoms_max = max(
        conf[0].shape[0]
        for mol in batch
        for conf in mol.conformers
    )
    F = batch[0].conformers[0][0].shape[1]

    node_feat = np.zeros((B, N_conf_max, N_atoms_max, F), dtype=np.float32)
    adj = np.zeros((B, N_conf_max, N_atoms_max, N_atoms_max), dtype=np.float32)
    dist = np.full((B, N_conf_max, N_atoms_max, N_atoms_max), 1e6, dtype=np.float32)
    conformer_mask = np.zeros((B, N_conf_max), dtype=bool)
    atom_mask = np.zeros((B, N_atoms_max), dtype=bool)
    targets = np.zeros((B, 1), dtype=np.float32)

    for i, mol in enumerate(batch):
        targets[i, 0] = mol.y
        n_atoms = mol.conformers[0][0].shape[0]
        atom_mask[i, :n_atoms] = True
        for j, (nf, a, d) in enumerate(mol.conformers):
            n_a = nf.shape[0]
            node_feat[i, j, :n_a, :] = nf
            adj[i, j, :n_a, :n_a] = a
            dist[i, j, :n_a, :n_a] = d
            conformer_mask[i, j] = True

    return {
        "node_feat": torch.from_numpy(node_feat),
        "adj": torch.from_numpy(adj),
        "dist": torch.from_numpy(dist),
        "atom_mask": torch.from_numpy(atom_mask),
        "conformer_mask": torch.from_numpy(conformer_mask),
        "target": torch.from_numpy(targets),
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ConformerEnsembleDataModule(DataModule):
    """DataModule that loads, featurizes, and splits a cyclic-peptide dataset.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Directory that contains the CSV file (and optionally PDB files).
    csv_path : str
        CSV filename relative to data_dir. Expected columns:
        - 'smiles': SMILES string (for conformer_source='smiles')
        - 'pdb_source', 'pdb_id': columns to build PDB file paths
          (for conformer_source='pdb')
        - 'y' or second column: regression target
        - Optionally 'split_0' … 'split_N': pre-computed fold assignments
          ('train', 'val', 'test').
    conformer_source : str
        'smiles' — generate conformers on-the-fly with RDKit.
        'pdb'    — load precomputed conformers from PDB files.
    n_conformers : int
        Number of conformers to generate per molecule (smiles mode).
    pdb_dir : str or None
        Subdirectory inside data_dir containing PDB files (pdb mode).
    split : int or None
        Which split column (0-indexed) to use. If None, use 80/10/10 random.
    ff : str
        Force field for RDKit conformer generation ('mmff' or 'uff').
    add_dummy_node : bool
    one_hot_formal_charge : bool
    batch_size : int
    num_workers : int
    seed : int
    """

    def __init__(
        self,
        data_dir,
        csv_path: str = "pampa.csv",
        conformer_source: str = "smiles",
        n_conformers: int = 8,
        pdb_dir: str | None = None,
        split: int | None = 0,
        ff: str = "mmff",
        add_dummy_node: bool = True,
        one_hot_formal_charge: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self._data_dir = pathlib.Path(data_dir)
        self._csv_path = csv_path
        self._conformer_source = conformer_source
        self._n_conformers = n_conformers
        self._pdb_dir = pdb_dir
        self._split = split
        self._ff = ff
        self._add_dummy_node = add_dummy_node
        self._one_hot_formal_charge = one_hot_formal_charge
        self._seed = seed
        self._d_atom: int | None = None  # set after featurization

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=conformer_collate_fn,
        )

        self._setup()

    def prepare_data(self):
        """No-op: featurization happens in _setup (called after barrier)."""
        pass

    def _setup(self):
        csv_file = self._data_dir / self._csv_path
        df = pd.read_csv(csv_file)

        # Resolve target column
        if "y" in df.columns:
            labels = df["y"].values.astype(np.float32)
        else:
            labels = df.iloc[:, 1].values.astype(np.float32)

        # Featurize
        logging.info(f"Featurizing {len(df)} molecules from {csv_file} ...")
        molecules: list[ConformerEnsembleMolecule] = []

        if self._conformer_source == "smiles":
            smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
            for idx, (smiles, label) in enumerate(zip(df[smiles_col], labels)):
                conformers = load_ensemble_from_smiles(
                    smiles,
                    n_conformers=self._n_conformers,
                    ff=self._ff,
                    add_dummy_node=self._add_dummy_node,
                    one_hot_formal_charge=self._one_hot_formal_charge,
                )
                if len(conformers) == 0:
                    logging.warning(f"Skipping molecule {idx} (SMILES: {smiles}): no conformers.")
                    continue
                molecules.append(ConformerEnsembleMolecule(conformers, label, idx))

        elif self._conformer_source == "pdb":
            assert self._pdb_dir is not None, "pdb_dir must be set when conformer_source='pdb'"
            pdb_base = self._data_dir / self._pdb_dir
            for idx, (row, label) in enumerate(zip(df.itertuples(), labels)):
                pdb_paths = [str(pdb_base / f"{row.pdb_source}_{row.pdb_id}.pdb")]
                conformers = load_ensemble_from_pdb(
                    pdb_paths,
                    add_dummy_node=self._add_dummy_node,
                    one_hot_formal_charge=self._one_hot_formal_charge,
                )
                if len(conformers) == 0:
                    logging.warning(f"Skipping molecule {idx}: no valid PDB conformers.")
                    continue
                molecules.append(ConformerEnsembleMolecule(conformers, label, idx))
        else:
            raise ValueError(f"Unknown conformer_source: {self._conformer_source!r}")

        if len(molecules) == 0:
            raise RuntimeError("No molecules were successfully featurized.")

        # Record atom feature dimension from the first conformer
        self._d_atom = molecules[0].conformers[0][0].shape[1]
        logging.info(f"Featurized {len(molecules)} molecules. d_atom={self._d_atom}")

        # Split
        split_col = f"split_{self._split}" if self._split is not None else None
        if split_col is not None and split_col in df.columns:
            # Use pre-defined splits stored in the CSV
            # Re-index molecules to match the original df rows after skipping
            original_indices = {m.index for m in molecules}
            train_mols, val_mols, test_mols = [], [], []
            for mol in molecules:
                assignment = df.iloc[mol.index][split_col]
                if assignment == "train":
                    train_mols.append(mol)
                elif assignment == "val":
                    val_mols.append(mol)
                elif assignment == "test":
                    test_mols.append(mol)
                else:
                    train_mols.append(mol)  # fallback
        else:
            # Random 80/10/10 split
            rng = np.random.default_rng(self._seed)
            idx = rng.permutation(len(molecules))
            n_train, n_val, n_test = get_split_sizes(molecules)
            train_mols = [molecules[i] for i in idx[:n_train]]
            val_mols = [molecules[i] for i in idx[n_train : n_train + n_val]]
            test_mols = [molecules[i] for i in idx[n_train + n_val :]]

        self.ds_train = ConformerEnsembleDataset(train_mols)
        self.ds_val = ConformerEnsembleDataset(val_mols)
        self.ds_test = ConformerEnsembleDataset(test_mols)

        logging.info(
            f"Split sizes — train: {len(self.ds_train)}, "
            f"val: {len(self.ds_val)}, test: {len(self.ds_test)}"
        )

    @property
    def d_atom(self) -> int:
        """Atom feature dimension, available after __init__ completes."""
        if self._d_atom is None:
            raise RuntimeError("d_atom is not set — featurization has not run yet.")
        return self._d_atom
