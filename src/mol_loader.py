"""MolLoader: DataModule for EnsemFormer.

Provides:
- ``MolLoader``: loads a preprocessed .pt cache and splits data into train/val/test DataLoaders.

Data representation lives in ``src.mol_dataset`` (MolItem, MolList, conformer_collate_fn).
"""

import logging
import pathlib
import sys

# Allow `python src/mol_loader.py` direct execution in addition to `python -m src.mol_loader`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.mol_dataset import MolItem, MolList, conformer_collate_fn
from src.utils import get_local_rank


def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    )
    return DataLoader(
        dataset, shuffle=(shuffle and sampler is None), sampler=sampler, **kwargs
    )


class MolLoader:
    """DataModule that loads a preprocessed .pt cache and splits data.

    The cache file must be produced by ``scripts/preprocess_trajectories.py`` before
    training. This DataModule does **not** perform featurization.

    Parameters
    ----------
    csv_path : str or pathlib.Path
        Full path to the CSV file.
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
        csv_path,
        cache_file: str | pathlib.Path | None = None,
        env: list[str] | str | None = None,
        n_conformers: int | None = None,
        rep_frame_only: bool = False,
        split: int | None = 0,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        self._csv_path = pathlib.Path(csv_path)
        self._cache_file = pathlib.Path(cache_file) if cache_file else None
        if not self._cache_file or not self._cache_file.exists():
            raise FileNotFoundError(
                f"Cache file not found: {cache_file!r}. "
                "Run `python scripts/preprocess_trajectories.py` first."
            )
        if env is None:
            self._envs = None  # use all envs in cache
        elif isinstance(env, str):
            self._envs = [env]
        else:
            self._envs = list(env)
        self._n_conformers = n_conformers
        self._rep_frame_only = rep_frame_only
        self._split = split

        self._d_atom: int | None = None
        self._dataloader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": conformer_collate_fn,
            "pin_memory": True,
            "persistent_workers": num_workers > 0,
        }

        self._setup()

    def _setup(self):
        df = pd.read_csv(self._csv_path)

        logging.info(f"Loading preprocessed molecules from cache: {self._cache_file}")
        assert self._cache_file is not None  # guaranteed by __init__ check
        data = torch.load(self._cache_file, weights_only=False)
        raw_molecules = data["molecules"]
        self._d_atom = data["d_atom"]
        cache_envs = data["envs"]
        selected_envs = self._envs if self._envs is not None else cache_envs

        # Convert raw dicts from cache into MolItem objects
        molecules = []
        for m in raw_molecules:
            rep_frame_idxs = m.get("rep_frame_idxs")
            if not isinstance(rep_frame_idxs, dict):
                raise ValueError(
                    f"Cache entry for '{m.get('CycPeptMPDB_ID', '?')}' has invalid "
                    f"rep_frame_idxs: {rep_frame_idxs!r}. "
                    "Re-run preprocess_trajectories.py to rebuild the cache."
                )
            confs = self._select_env_conformers(m["envs"], selected_envs, rep_frame_idxs)
            if not confs:
                raise ValueError(
                    f"Molecule '{m.get('CycPeptMPDB_ID', '?')}' has no conformers after "
                    f"env/conformer selection (envs={selected_envs}). "
                    "Check your cache or env configuration."
                )
            molecules.append(
                MolItem(
                    node_feat=m["node_feat"],
                    adj=m["adj"],
                    bond_types=m["bond_types"],
                    conformers=confs,
                    label=m["label"],
                    CycPeptMPDB_ID=m["CycPeptMPDB_ID"],
                    SMILES=m.get("SMILES"),
                    Structurally_Unique_ID=m.get("Structurally_Unique_ID"),
                    rep_frame_idxs=rep_frame_idxs,
                )
            )

        if self._split is None:
            raise ValueError(
                "data.split must be set to an integer index. "
                "Run `python scripts/generate_splits.py` to generate split columns, "
                "then set data.split: 0 (or another index) in your config."
            )
        split_col = f"split_{self._split}"
        if split_col not in df.columns:
            raise ValueError(
                f"Split column '{split_col}' not found in CSV '{self._csv_path}'. "
                f"Run `python scripts/generate_splits.py` to generate it."
            )
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

        self.ds_train = MolList(train_mols)
        self.ds_val = MolList(val_mols)
        self.ds_test = MolList(test_mols)

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
                raise ValueError(
                    f"Env '{env}' has no conformers in the cache. "
                    "Re-run preprocess_trajectories.py to rebuild the cache."
                )
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
            elif self._n_conformers is not None:
                if len(env_confs) < self._n_conformers:
                    raise ValueError(
                        f"Env '{env}' has {len(env_confs)} conformers in the cache, "
                        f"but n_conformers={self._n_conformers} was requested. "
                        f"Reduce n_conformers or rebuild the cache with more frames."
                    )
                chosen = np.linspace(0, len(env_confs) - 1, self._n_conformers, dtype=int)
                env_confs = [env_confs[i] for i in chosen]
            confs.extend(env_confs)
        return confs

    def train_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_train, shuffle=True, **self._dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_val, shuffle=False, **self._dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_test, shuffle=False, **self._dataloader_kwargs)

    @property
    def d_atom(self) -> int:
        if self._d_atom is None:
            raise RuntimeError("d_atom is not set — featurization has not run yet.")
        return self._d_atom


if __name__ == "__main__":
    import pathlib

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    REPO = pathlib.Path(__file__).resolve().parent.parent
    CSV   = REPO / "data" / "thumb.csv"
    CACHE = REPO / "data" / "thumbnail_traj_hexane+water_noH.pt"

    loader = MolLoader(
        csv_path=CSV,
        cache_file=CACHE,
        env=["water"],
        n_conformers=5,
        rep_frame_only=False,
        split=0,
        batch_size=4,
        num_workers=0,
    )

    print(f"\nd_atom : {loader.d_atom}")
    print(f"train  : {len(loader.ds_train)} molecules")
    print(f"val    : {len(loader.ds_val)}   molecules")
    print(f"test   : {len(loader.ds_test)}  molecules")

    batch = next(iter(loader.train_dataloader()))
    print("\n--- first train batch ---")
    for k, v in batch.items():
        desc = f"shape={tuple(v.shape)}, dtype={v.dtype}" if isinstance(v, torch.Tensor) else repr(v)
        print(f"  {k:20s}  {desc}")
