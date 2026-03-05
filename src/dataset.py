"""Multi-conformer dataset for EnsemFormer.

Provides:
- ``ConformerEnsembleMolecule``: holds topology arrays once + N (dist, coords) conformer tuples + label
- ``ConformerEnsembleDataset``: PyTorch Dataset wrapping the above
- ``conformer_collate_fn``: pads atoms and conformers and builds batch dicts
- ``ConformerEnsembleDataModule``: loads a preprocessed .pt cache and splits data
"""

import logging
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_module import DataModule
from src.utils import get_split_sizes

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class ConformerEnsembleMolecule:
    """A single molecule represented as an ensemble of conformers.

    Topology arrays (node_feat, adj, bond_types) are stored once at molecule
    level since they are identical across all conformer frames. Per-frame data
    is stored as a list of (dist, coords) 2-tuples.

    Parameters
    ----------
    node_feat : np.ndarray (N_atoms, F)
        Node features — topology-invariant.
    adj : np.ndarray (N_atoms, N_atoms)
        Adjacency matrix — topology-invariant.
    bond_types : np.ndarray (N_atoms, N_atoms)
        Bond type matrix — topology-invariant.
    conformers : list of (dist, coords) tuples
        Per-conformer data: dist (N_atoms, N_atoms), coords (N_atoms, 3).
    label : float
        Regression target.
    CycPeptMPDB_ID : str
        Unique molecule identifier from CycPeptMPDB.
    SMILES : str or None
    Structurally_Unique_ID : str or None
    rep_frame_idxs : dict[str, int] or None
        Per-env representative frame indices, e.g. {"water": 42, "hexane": 17}.
    """

    def __init__(
        self,
        node_feat: np.ndarray,
        adj: np.ndarray,
        bond_types: np.ndarray,
        conformers: list,
        label: float,
        CycPeptMPDB_ID: str,
        SMILES: str | None = None,
        Structurally_Unique_ID: str | None = None,
        rep_frame_idxs: dict | None = None,
    ):
        self.node_feat = node_feat
        self.adj = adj
        self.bond_types = bond_types
        self.conformers = conformers
        self.y = float(label)
        self.CycPeptMPDB_ID = CycPeptMPDB_ID
        self.SMILES = SMILES
        self.Structurally_Unique_ID = Structurally_Unique_ID
        self.rep_frame_idxs = rep_frame_idxs

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
        'bond_type'      : LongTensor  (B, N_conf_max, N_atoms_max, N_atoms_max)  — if available
    """
    B = len(batch)
    N_conf_max = max(mol.n_conformers for mol in batch)
    N_atoms_max = max(mol.node_feat.shape[0] for mol in batch)
    F = batch[0].node_feat.shape[1]

    # Topology arrays: (B, N_atoms_max, ...) — broadcast to conformer dim after filling
    node_feat_2d = np.zeros((B, N_atoms_max, F), dtype=np.float32)
    adj_2d       = np.zeros((B, N_atoms_max, N_atoms_max), dtype=np.float32)
    bond_type_2d = np.zeros((B, N_atoms_max, N_atoms_max), dtype=np.int64)

    # Per-conformer arrays
    dist          = np.full((B, N_conf_max, N_atoms_max, N_atoms_max), 1e6, dtype=np.float32)
    coords        = np.zeros((B, N_conf_max, N_atoms_max, 3), dtype=np.float32)
    conformer_mask = np.zeros((B, N_conf_max), dtype=bool)
    atom_mask     = np.zeros((B, N_atoms_max), dtype=bool)
    targets       = np.zeros((B, 1), dtype=np.float32)

    for i, mol in enumerate(batch):
        targets[i, 0] = mol.y
        n_a = mol.node_feat.shape[0]
        atom_mask[i, :n_a] = True

        # Topology — stored once per molecule
        node_feat_2d[i, :n_a, :]    = mol.node_feat
        adj_2d[i, :n_a, :n_a]       = mol.adj
        bond_type_2d[i, :n_a, :n_a] = mol.bond_types

        # Per-conformer (dist, coords)
        for j, (d, pos) in enumerate(mol.conformers):
            dist[i, j, :n_a, :n_a] = d
            coords[i, j, :n_a, :]  = pos
            conformer_mask[i, j]   = True

    # Broadcast topology to conformer dimension.
    # .contiguous() is required: expand() returns a stride-0 view that breaks
    # view()/reshape() inside the CPMP encoder.
    node_feat_t = torch.from_numpy(node_feat_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()
    adj_t       = torch.from_numpy(adj_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()
    bond_type_t = torch.from_numpy(bond_type_2d).unsqueeze(1).expand(-1, N_conf_max, -1, -1).contiguous()

    return {
        "node_feat":      node_feat_t,
        "adj":            adj_t,
        "dist":           torch.from_numpy(dist),
        "coords":         torch.from_numpy(coords),
        "atom_mask":      torch.from_numpy(atom_mask),
        "conformer_mask": torch.from_numpy(conformer_mask),
        "target":         torch.from_numpy(targets),
        "bond_type":      bond_type_t,
    }


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class ConformerEnsembleDataModule(DataModule):
    """DataModule that loads a preprocessed .pt cache and splits data.

    The cache file must be produced by ``scripts/preprocess_trajectories.py`` before
    training. This DataModule does **not** perform featurization.

    Parameters
    ----------
    data_dir : pathlib.Path or str
    csv_path : str
        CSV filename relative to data_dir.
    cache_file : str
        Path to a .pt file produced by ``scripts/preprocess_trajectories.py``.
    env : list[str] or str or None
        Environment(s) to use from the cache (e.g. ``["water"]`` or
        ``["water", "hexane"]``).  If None, all envs in the cache are used.
    n_conformers : int or None
        Maximum number of conformers to use **per env** per molecule.
    rep_frame_only : bool
        If True, use only the representative frame per molecule.
    split : int or None
    batch_size : int
    num_workers : int
    seed : int
    """

    def __init__(
        self,
        data_dir,
        csv_path: str = "CycPeptMPDB-4D.csv",
        cache_file: str = None,
        env: list[str] | str | None = None,
        n_conformers: int | None = None,
        rep_frame_only: bool = False,
        split: int | None = 0,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 42,
    ):
        if cache_file is None:
            raise ValueError(
                "cache_file is required. Run `python scripts/preprocess_trajectories.py` first "
                "to produce a .pt cache, then set paths.cache_file in your config."
            )
        cache_path = pathlib.Path(cache_file)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_path}\n"
                "Run `python scripts/preprocess_trajectories.py` first."
            )

        self._data_dir = pathlib.Path(data_dir)
        self._csv_path = csv_path
        self._cache_file = cache_path
        if env is None:
            self._envs = None  # use all envs in cache
        elif isinstance(env, str):
            self._envs = [env]
        else:
            self._envs = list(env)
        self._n_conformers = n_conformers
        self._rep_frame_only = rep_frame_only
        self._split = split
        self._seed = seed

        self._d_atom: int | None = None

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=conformer_collate_fn,
        )

        self._setup()

    def _setup(self):
        csv_file = self._data_dir / self._csv_path
        df = pd.read_csv(csv_file)

        logging.info(f"Loading preprocessed molecules from cache: {self._cache_file}")
        data = torch.load(self._cache_file, weights_only=False)
        raw_molecules = data["molecules"]
        self._d_atom = data["d_atom"]
        cache_envs = data["envs"]
        selected_envs = self._envs if self._envs is not None else cache_envs

        # Convert raw dicts from cache into ConformerEnsembleMolecule objects
        molecules = []
        for m in raw_molecules:
            if "rep_frame_idxs" not in m:
                raise KeyError(
                    f"Cache entry for '{m.get('CycPeptMPDB_ID', '?')}' is missing "
                    f"'rep_frame_idxs'. Re-run preprocess_trajectories.py to rebuild the cache."
                )
            rep_frame_idxs = m["rep_frame_idxs"] or {}
            confs = self._select_env_conformers(m["envs"], selected_envs, rep_frame_idxs)
            if not confs:
                continue
            molecules.append(
                ConformerEnsembleMolecule(
                    node_feat=m["node_feat"],
                    adj=m["adj"],
                    bond_types=m["bond_types"],
                    conformers=confs,
                    label=m["label"],
                    CycPeptMPDB_ID=m["CycPeptMPDB_ID"],
                    SMILES=m.get("SMILES"),
                    Structurally_Unique_ID=m.get("Structurally_Unique_ID"),
                    rep_frame_idxs=rep_frame_idxs if rep_frame_idxs else None,
                )
            )

        split_col = f"split_{self._split}" if self._split is not None else None
        if split_col is not None and split_col in df.columns:
            split_map = dict(zip(df["CycPeptMPDB_ID"].astype(str), df[split_col]))
            train_mols, val_mols, test_mols = [], [], []
            for mol in molecules:
                if mol.CycPeptMPDB_ID not in split_map:
                    raise KeyError(
                        f"Molecule '{mol.CycPeptMPDB_ID}' not found in split column '{split_col}'."
                    )
                assignment = split_map[mol.CycPeptMPDB_ID]
                if assignment == "train":
                    train_mols.append(mol)
                elif assignment == "val":
                    val_mols.append(mol)
                elif assignment == "test":
                    test_mols.append(mol)
                else:
                    raise ValueError(
                        f"Unknown split assignment '{assignment}' for molecule "
                        f"'{mol.CycPeptMPDB_ID}' in column '{split_col}'. "
                        f"Expected 'train', 'val', or 'test'."
                    )
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

    def _select_env_conformers(
        self,
        env_dict: dict[str, list],
        envs_to_use: list[str],
        rep_frame_idxs: dict[str, int],
    ) -> list:
        """Select and subsample conformers from requested envs.

        ``n_conformers`` is applied **per env** so each environment
        contributes equally to the conformer ensemble.
        """
        confs: list = []
        for env in envs_to_use:
            if env not in env_dict:
                raise KeyError(
                    f"Requested env '{env}' not found in cached molecule. "
                    f"Available envs: {list(env_dict.keys())}. "
                    f"Re-run preprocess_trajectories.py with the correct --env flag."
                )
            env_confs = env_dict[env]
            if not env_confs:
                continue
            if self._rep_frame_only:
                if env not in rep_frame_idxs:
                    raise ValueError(
                        f"rep_frame_only=True requires rep_frame_idxs['{env}'], but it is missing."
                    )
                rep_idx_1b = int(rep_frame_idxs[env])
                if rep_idx_1b < 1 or rep_idx_1b > len(env_confs):
                    raise IndexError(
                        f"Representative frame index out of range for env '{env}': "
                        f"{rep_idx_1b} (valid 1..{len(env_confs)})."
                    )
                env_confs = [env_confs[rep_idx_1b - 1]]
            elif self._n_conformers is not None and len(env_confs) > self._n_conformers:
                chosen = np.linspace(0, len(env_confs) - 1, self._n_conformers, dtype=int)
                env_confs = [env_confs[i] for i in chosen]
            confs.extend(env_confs)
        return confs

    @property
    def d_atom(self) -> int:
        if self._d_atom is None:
            raise RuntimeError("d_atom is not set — featurization has not run yet.")
        return self._d_atom
