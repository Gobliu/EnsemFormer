"""Multi-conformer dataset for EnsemFormer.

Provides:
- ``ConformerEnsembleMolecule``: holds N conformer feature tuples + label
- ``ConformerEnsembleDataset``: PyTorch Dataset wrapping the above
- ``conformer_collate_fn``: pads atoms and conformers and builds batch dicts
- ``featurize_molecules``: standalone featurization function (used by scripts/featurize.py)
- ``ConformerEnsembleDataModule``: full DataModule with featurization and splitting
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_module import DataModule
from src.featurization import load_ensemble_from_pdb
from src.utils import get_split_sizes

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ConformerEnsembleMolecule:
    """A single molecule represented as an ensemble of conformers.

    Parameters
    ----------
    conformers : list of (node_features, adj_matrix, dist_matrix, coords) tuples
    label : float
        Regression target.
    index : int
        Dataset index.
    rep_frame_idx : int or None
        0-based representative frame index.
    """

    def __init__(
        self,
        conformers: list,
        label: float,
        index: int,
        rep_frame_idx: int | None = None,
    ):
        self.conformers = conformers
        self.y = float(label)
        self.index = index
        self.rep_frame_idx = rep_frame_idx

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
        'coords'         : FloatTensor (B, N_conf_max, N_atoms_max, 3)
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
    coords = np.zeros((B, N_conf_max, N_atoms_max, 3), dtype=np.float32)
    conformer_mask = np.zeros((B, N_conf_max), dtype=bool)
    atom_mask = np.zeros((B, N_atoms_max), dtype=bool)
    targets = np.zeros((B, 1), dtype=np.float32)

    for i, mol in enumerate(batch):
        targets[i, 0] = mol.y
        n_atoms = mol.conformers[0][0].shape[0]
        atom_mask[i, :n_atoms] = True
        for j, (nf, a, d, pos) in enumerate(mol.conformers):
            n_a = nf.shape[0]
            node_feat[i, j, :n_a, :] = nf
            adj[i, j, :n_a, :n_a] = a
            dist[i, j, :n_a, :n_a] = d
            coords[i, j, :n_a, :] = pos
            conformer_mask[i, j] = True

    return {
        "node_feat": torch.from_numpy(node_feat),
        "adj": torch.from_numpy(adj),
        "dist": torch.from_numpy(dist),
        "coords": torch.from_numpy(coords),
        "atom_mask": torch.from_numpy(atom_mask),
        "conformer_mask": torch.from_numpy(conformer_mask),
        "target": torch.from_numpy(targets),
    }


# ---------------------------------------------------------------------------
# Standalone featurization (also used by scripts/featurize.py)
# ---------------------------------------------------------------------------

_SOLVENT_MAP = {
    "water":  ("Water",  "H2O_Traj"),
    "hexane": ("Hexane", "Hexane_Traj"),
}


def featurize_molecules(
    data_dir,
    csv_path: str,
    target_col: str,
    traj_dir,
    solvent: str,
    add_dummy_node: bool,
    one_hot_formal_charge: bool,
) -> tuple[list[ConformerEnsembleMolecule], int]:
    """Featurize all molecules from trajectory PDB files.

    The CSV must have columns ``Source`` and ``CycPeptMPDB_ID``. PDB files are
    read from ``<traj_dir>/{Water|Hexane}/Trajectories/{Source}_{ID}_{suffix}.pdb``,
    matching the CycPeptMPDB_4D dataset layout.

    Returns
    -------
    (molecules, d_atom)
    """
    data_dir = pathlib.Path(data_dir)
    csv_file = data_dir / csv_path
    df = pd.read_csv(csv_file)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in CSV. Columns: {list(df.columns)}"
        )
    labels = df[target_col].values.astype(np.float32)

    subdir, suffix = _SOLVENT_MAP[solvent.lower()]
    traj_base = pathlib.Path(traj_dir) / subdir / "Trajectories"
    logging.info(f"Featurizing {len(df)} molecules from {csv_file} ...")
    molecules: list[ConformerEnsembleMolecule] = []

    for idx, (row, label) in enumerate(zip(df.itertuples(), labels)):
        pdb_path = str(traj_base / f"{row.Source}_{row.CycPeptMPDB_ID}_{suffix}.pdb")
        confs = load_ensemble_from_pdb(
            [pdb_path],
            add_dummy_node=add_dummy_node,
            one_hot_formal_charge=one_hot_formal_charge,
        )
        if not confs:
            logging.warning(f"Skipping molecule {idx} ({row.Source}_{row.CycPeptMPDB_ID}): no valid PDB conformers.")
            continue
        molecules.append(ConformerEnsembleMolecule(confs, label, idx))

    if not molecules:
        raise RuntimeError("No molecules were successfully featurized.")

    d_atom = molecules[0].conformers[0][0].shape[1]
    logging.info(f"Featurized {len(molecules)} molecules. d_atom={d_atom}")
    return molecules, d_atom


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ConformerEnsembleDataModule(DataModule):
    """DataModule that loads, featurizes, and splits a cyclic-peptide dataset.

    Parameters
    ----------
    data_dir : pathlib.Path or str
    csv_path : str
        CSV filename relative to data_dir. Must have ``Source`` and
        ``CycPeptMPDB_ID`` columns.
    target_col : str
        Column name to use as regression target.
    traj_dir : pathlib.Path or str
        Root of the CycPeptMPDB_4D dataset. PDB files are read from
        ``<traj_dir>/{Water|Hexane}/Trajectories/``.
    solvent : str
        ``'water'`` or ``'hexane'``.
    n_conformers : int or None
        Maximum number of conformers to use per molecule.
    rep_frame_only : bool
        If True, use only the representative frame per molecule.
    split : int or None
    add_dummy_node : bool
    one_hot_formal_charge : bool
    batch_size : int
    num_workers : int
    seed : int
    cache_file : str or None
        Path to a .pt file produced by ``scripts/featurize.py``. If the file
        exists, featurization is skipped and molecules are loaded from cache.
    """

    def __init__(
        self,
        data_dir,
        csv_path: str = "CycPeptMPDB-4D.csv",
        target_col: str = "PAMPA",
        traj_dir=None,
        solvent: str = "water",
        n_conformers: int | None = None,
        rep_frame_only: bool = False,
        split: int | None = 0,
        add_dummy_node: bool = True,
        one_hot_formal_charge: bool = False,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
        cache_file: str | None = None,
    ):
        self._data_dir = pathlib.Path(data_dir)
        self._csv_path = csv_path
        self._target_col = target_col
        self._traj_dir = pathlib.Path(traj_dir) if traj_dir else self._data_dir
        self._solvent = solvent
        self._n_conformers = n_conformers
        self._rep_frame_only = rep_frame_only
        self._split = split
        self._add_dummy_node = add_dummy_node
        self._one_hot_formal_charge = one_hot_formal_charge
        self._seed = seed
        self._cache_file = pathlib.Path(cache_file) if cache_file else None

        self._d_atom: int | None = None

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=conformer_collate_fn,
        )

        self._setup()

    def prepare_data(self):
        pass

    def _setup(self):
        csv_file = self._data_dir / self._csv_path
        df = pd.read_csv(csv_file)

        if self._cache_file is not None and self._cache_file.exists():
            logging.info(f"Loading featurized molecules from cache: {self._cache_file}")
            data = torch.load(self._cache_file, weights_only=False)
            molecules = data["molecules"]
            self._d_atom = data["d_atom"]
        else:
            molecules, self._d_atom = featurize_molecules(
                data_dir=self._data_dir,
                csv_path=self._csv_path,
                target_col=self._target_col,
                traj_dir=self._traj_dir,
                solvent=self._solvent,
                add_dummy_node=self._add_dummy_node,
                one_hot_formal_charge=self._one_hot_formal_charge,
            )

        molecules = self._subset_conformers(molecules)

        split_col = f"split_{self._split}" if self._split is not None else None
        if split_col is not None and split_col in df.columns:
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
            rng = np.random.default_rng(self._seed)
            idx = rng.permutation(len(molecules))
            n_train, n_val, _ = get_split_sizes(molecules)
            train_mols = [molecules[i] for i in idx[:n_train]]
            val_mols = [molecules[i] for i in idx[n_train:n_train + n_val]]
            test_mols = [molecules[i] for i in idx[n_train + n_val:]]

        self.ds_train = ConformerEnsembleDataset(train_mols)
        self.ds_val = ConformerEnsembleDataset(val_mols)
        self.ds_test = ConformerEnsembleDataset(test_mols)

        logging.info(
            f"Split sizes — train: {len(self.ds_train)}, "
            f"val: {len(self.ds_val)}, test: {len(self.ds_test)}"
        )

    def _subset_conformers(self, molecules: list) -> list:
        """Apply rep_frame_only and n_conformers subsetting."""
        out = []
        for mol in molecules:
            confs = mol.conformers
            rep_idx = mol.rep_frame_idx

            if self._rep_frame_only:
                idx = rep_idx if rep_idx is not None else 0
                idx = min(idx, len(confs) - 1)
                confs = [confs[idx]]
            elif self._n_conformers is not None and len(confs) > self._n_conformers:
                chosen = np.linspace(0, len(confs) - 1, self._n_conformers, dtype=int)
                confs = [confs[i] for i in chosen]

            out.append(ConformerEnsembleMolecule(confs, mol.y, mol.index, rep_idx))
        return out

    @property
    def d_atom(self) -> int:
        if self._d_atom is None:
            raise RuntimeError("d_atom is not set — featurization has not run yet.")
        return self._d_atom
