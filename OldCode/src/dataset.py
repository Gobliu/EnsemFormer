# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

"""Datasets and collation utilities for QM9 and CycPept3D.

This module provides DataModule implementations and helper functions that wrap
datasets used across the repository:

- SE3-Transformer datasets for QM9 and CycPept3D
- EGNN datasets for QM9 and CycPept3D
- CPMP datasets for QM9 and CycPept3D

Each DataModule standardizes train/val/test splits and returns batches in the
format expected by the corresponding models.
"""

import os
import dgl
import pathlib
import torch
import numpy as np
import pandas as pd

from dgl.data import QM9EdgeDataset
from torch_geometric.data import download_url, extract_zip
from torch.utils.data import random_split, Dataset, Subset
from rdkit import Chem, RDLogger
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data_module import DataModule
from src.utils import get_relative_pos, str2bool, using_tensor_cores, get_split_sizes

from se3_transformer.se3_transformer.data_loading.qm9 import CachedBasesQM9EdgeDataset
from se3_transformer.se3_transformer.data_loading.cycpept3D import CycPept3DDataset
from egnn.qm9.data.dataset import ProcessedDataset
from egnn.qm9.data.prepare import prepare_dataset
from egnn.qm9.data.utils import get_species
from egnn.qm9.data.collate import drop_zeros, batch_stack
from cpmp.featurization.data_utils import (
    construct_dataset,
    pad_array,
    featurize_mol,
    Molecule,
    MolDataset,
)

RDLogger.DisableLog("rdApp.*")


def _get_split_sizes(full_dataset: Dataset) -> tuple[int, int, int]:
    """Return fixed split sizes for QM9-style experiments.

    Parameters
    ----------
    full_dataset : Dataset
        The full dataset instance.

    Returns
    -------
    tuple of int
        A triple ``(len_train, len_val, len_test)`` where train is fixed to
        100k samples, test is 10% of the dataset, and the remainder goes to
        validation.
    """
    len_full = len(full_dataset)
    len_train = 100_000
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test


def to_processed_dataset(ds):
    """Convert a raw iterable of PyG ``Data`` items to an EGNN ``ProcessedDataset``.

    Parameters
    ----------
    ds : Iterable
        Collection of PyTorch Geometric ``Data`` objects containing attributes
        ``z`` (charges), ``pos`` (positions), ``x`` (node features), ``y``
        (labels), and ``name``.

    Returns
    -------
    ProcessedDataset
        Dataset compatible with EGNN utilities.
    """
    raw_props = {
        "charges": [],
        "positions": [],
        "num_atoms": [],
        "one_hot": [],
        "pampa": [],
        "name": [],
    }

    for data in ds:
        raw_props["charges"].append(data.z)
        raw_props["positions"].append(data.pos)
        raw_props["num_atoms"].append(data.z.shape[0])
        raw_props["one_hot"].append(data.x[:, :5] == 1.0)
        raw_props["pampa"].append(data.y.squeeze())
        raw_props["name"].append(data.name)

    raw_props_padded = {}
    raw_props_padded["charges"] = torch.nn.utils.rnn.pad_sequence(
        raw_props["charges"], batch_first=True, padding_value=0
    )
    raw_props_padded["positions"] = torch.nn.utils.rnn.pad_sequence(
        raw_props["positions"], batch_first=True, padding_value=0.0
    )
    raw_props_padded["one_hot"] = torch.nn.utils.rnn.pad_sequence(
        raw_props["one_hot"], batch_first=True, padding_value=False
    )
    raw_props_padded["num_atoms"] = torch.tensor(raw_props["num_atoms"])
    raw_props_padded["pampa"] = torch.tensor(raw_props["pampa"])
    raw_props_padded["name"] = raw_props["name"]

    processed_ds = ProcessedDataset(data=raw_props_padded, subtract_thermo=False)

    return processed_ds


## SE3-Transformer


class SE3TransformerQM9DataModule(DataModule):
    """SE3-Transformer DataModule for the QM9Edge dataset.

    This wraps :class:`dgl.data.QM9EdgeDataset` or its cached-bases variant and
    provides train/val/test splits along with a collate function that yields
    DGL graphs, node/edge features, and normalized targets.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where data is stored or downloaded.
    task : str, default="homo"
        Property to regress. See ``add_argparse_args`` for choices.
    batch_size : int, default=256
        Mini-batch size.
    num_workers : int, default=8
        Number of dataloader workers.
    num_degrees : int, default=4
        Maximum degree used by SE3 fibers (affects feature dimensions).
    amp : bool, default=False
        Enable CUDA AMP autocast during model forward in downstream code.
    precompute_bases : bool, default=False
        If True, use a dataset variant that precomputes SE3 bases.
    **kwargs
        Additional arguments, e.g., ``seed`` for splitting reproducibility.

    Attributes
    ----------
    rescale_factor : float
        Standard deviation of the train targets used to rescale predictions.
    targets_mean : float
        Mean of the train targets used for normalization.
    targets_std : float
        Standard deviation of the train targets used for normalization.
    """

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 4

    def __init__(
        self,
        data_dir: pathlib.Path,
        task: str = "homo",
        batch_size: int = 256,
        num_workers: int = 8,
        num_degrees: int = 4,
        amp: bool = False,
        precompute_bases: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        super().__init__(
            batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate
        )
        self.amp = amp
        self.task = task
        self.batch_size = batch_size
        self.num_degrees = num_degrees

        qm9_kwargs = dict(label_keys=[self.task], verbose=False, raw_dir=str(data_dir))
        if precompute_bases:
            bases_kwargs = dict(
                max_degree=num_degrees - 1,
                use_pad_trick=using_tensor_cores(amp),
                amp=amp,
            )
            full_dataset = CachedBasesQM9EdgeDataset(
                bases_kwargs=bases_kwargs,
                batch_size=batch_size,
                num_workers=num_workers,
                **qm9_kwargs,
            )
        else:
            full_dataset = QM9EdgeDataset(**qm9_kwargs)

        self.ds_train, self.ds_val, self.ds_test = random_split(
            full_dataset,
            _get_split_sizes(full_dataset),
            generator=torch.Generator().manual_seed(kwargs["seed"]),
        )

        train_targets = full_dataset.targets[
            self.ds_train.indices, full_dataset.label_keys[0]
        ]
        self.targets_mean = train_targets.mean()
        self.targets_std = train_targets.std()
        self.rescale_factor = self.targets_std

    def prepare_data(self):
        """Ensure the preprocessed QM9 dataset is available on disk."""
        QM9EdgeDataset(verbose=True, raw_dir=str(self.data_dir))

    def _collate(self, samples):
        """Collate samples into a DGL batch and SE3Transformer inputs.

        Parameters
        ----------
        samples : list
            Each sample is a tuple ``(graph, y[, bases])`` depending on whether
            bases were precomputed.

        Returns
        -------
        tuple
            Either ``(batched_graph, node_feats, edge_feats, targets)`` or
            ``(batched_graph, node_feats, edge_feats, all_bases, targets)`` when
            bases are present.
        """
        graphs, y, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        edge_feats = {
            "0": batched_graph.edata["edge_attr"][:, : self.EDGE_FEATURE_DIM, None]
        }
        batched_graph.edata["rel_pos"] = get_relative_pos(batched_graph)
        # get node features
        node_feats = {
            "0": batched_graph.ndata["attr"][:, : self.NODE_FEATURE_DIM, None]
        }
        targets = (torch.cat(y) - self.targets_mean) / self.targets_std

        if bases:
            # collate bases
            all_bases = {
                key: torch.cat([b[key] for b in bases[0]], dim=0)
                for key in bases[0][0].keys()
            }

            return batched_graph, node_feats, edge_feats, all_bases, targets
        else:
            return batched_graph, node_feats, edge_feats, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SE3-Transformer QM9 dataset")
        parser.add_argument(
            "--precompute_bases",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Precompute bases at the beginning of the script during dataset initialization,"
            " instead of computing them at the beginning of each forward pass.",
        )
        return parent_parser

    def __repr__(self):
        return f"QM9({self.task})"


class SE3TransformerCP3DDataModule(DataModule):
    """SE3-Transformer DataModule for the CycPept3D dataset.

    Parameters
    ----------
    data_dir : pathlib.Path or str
        Root directory containing raw CSV and structures.
    batch_size : int, default=256
        Mini-batch size.
    num_workers : int, default=8
        Number of dataloader workers.
    num_degrees : int, default=4
        Maximum degree used by SE3 fibers.
    split : int or None, optional
        If provided, use predefined split column ``split{split}`` from the CSV.
    remove_hydrogen : bool, default=False
        If True, drop hydrogen atoms during parsing.
    **kwargs
        Additional args passed to the superclass.

    Attributes
    ----------
    rescale_factor : float
        Half the dynamic range of the PAMPA label, used for metric rescaling.
    """

    NODE_FEATURE_DIM = 6
    EDGE_FEATURE_DIM = 4

    def __init__(
        self,
        data_dir: pathlib.Path | str,
        batch_size: int = 256,
        num_workers: int = 8,
        num_degrees: int = 4,
        split: int | None = None,
        remove_hydrogen: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
        self.batch_size = batch_size
        self.num_degrees = num_degrees
        full_dataset = CycPept3DDataset(
            root=self.data_dir, remove_hydrogen=remove_hydrogen
        )
        csv_path = full_dataset.raw_paths[0]
        if split is not None:
            split_col = f"split{split}"
            split_df = pd.read_csv(
                csv_path,
                usecols=["CycPeptMPDB_ID", "Source", split_col],
                dtype={"CycPeptMPDB_ID": str, "Source": str},
            ).assign(
                Source_ID=lambda x: x["Source"].str.cat(
                    x["CycPeptMPDB_ID"].astype(str), sep="_"
                )
            )
            split_dict = split_df.set_index("Source_ID")[split_col].to_dict()

            train_indices = []
            val_indices = []
            test_indices = []
            dispatch = {
                "train": train_indices.append,
                "valid": val_indices.append,
                "test": test_indices.append,
            }

            for i, data in enumerate(full_dataset):
                name = data.name
                try:
                    split_type = split_dict[name]
                except KeyError:
                    raise ValueError(
                        f"Data point '{name}' not found in split dictionary."
                    )

                try:
                    func = dispatch[split_type]
                except KeyError:
                    raise ValueError(
                        f"Unknown split type '{split_type}' for {name}. Expected 'train', 'valid', or 'test'."
                    )
                func(i)

            self.ds_train = Subset(full_dataset, train_indices)
            self.ds_val = Subset(full_dataset, val_indices)
            self.ds_test = Subset(full_dataset, test_indices)
        else:
            # Fallback to random split if no split dictionary is provided
            self.ds_train, self.ds_val, self.ds_test = random_split(
                full_dataset,
                get_split_sizes(full_dataset),
                generator=torch.Generator().manual_seed(kwargs["seed"]),
            )

        pampa_df = pd.read_csv(
            csv_path,
            usecols=["PAMPA"],
        )

        self.rescale_factor = (pampa_df["PAMPA"].max() - pampa_df["PAMPA"].min()) / 2

    def _collate(self, samples):
        """Convert a list of PyG ``Data`` objects to a DGL batch for SE3T.

        Returns
        -------
        tuple
            ``(batched_graph, node_feats, edge_feats, targets)`` suitable for
            :class:`SE3TransformerPooled`.
        """

        def _convert_pyg_to_dgl(data):
            """Convert a single PyG ``Data`` instance to a DGL graph."""
            # Create DGL graph from edge_index
            src, dst = data.edge_index
            graph = dgl.graph((src, dst), num_nodes=data.x.shape[0])

            # Add node features (positions and node attributes)
            graph.ndata["pos"] = data.pos
            graph.ndata["attr"] = data.x

            # Add edge attributes
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                graph.edata["edge_attr"] = data.edge_attr

            return graph

        # Convert all samples to DGL graphs
        graphs = []
        targets = []

        for data in samples:
            graph = _convert_pyg_to_dgl(data)
            graphs.append(graph)
            targets.append(data.y)

        # Batch all graphs together
        batched_graph = dgl.batch(graphs)

        # Add relative positions to edges
        batched_graph.edata["rel_pos"] = get_relative_pos(batched_graph)

        # Format edge features for SE3Transformer (degree 0 features only)
        edge_feats = {
            "0": batched_graph.edata["edge_attr"][:, : self.EDGE_FEATURE_DIM, None]
        }

        # Format node features for SE3Transformer (degree 0 features only)
        # Use only the first NODE_FEATURE_DIM features as node features
        node_feats = {
            "0": batched_graph.ndata["attr"][:, : self.NODE_FEATURE_DIM, None]
        }

        targets = torch.cat(targets, dim=0)

        return batched_graph, node_feats, edge_feats, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("SE3-Transformer CycPept3D dataset")
        parser.add_argument(
            "--remove_hydrogen",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Whether to remove hydrogen atoms when parsing the molecule",
        )
        return parent_parser


## EGNN


class EGNNQM9DataModule(DataModule):
    """EGNN DataModule for QM9.

    Loads preprocessed tensors, computes species information, and exposes a
    collate function matching EGNN expectations.
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        batch_size: int = 256,
        num_workers: int = 8,
        subtract_thermo: bool = True,
        force_download: bool = False,
        charge_power: int = 2,
        task: str = "homo",
        **kwargs,
    ):
        self.data_dir = data_dir
        self.subtract_thermo = subtract_thermo
        self.force_download = force_download
        self.charge_power = charge_power
        self.label = task
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
        datasets = {}
        for split, datafile in self.datafiles.items():
            with np.load(datafile) as f:
                datasets[split] = {key: torch.from_numpy(val) for key, val in f.items()}

        keys = [list(data.keys()) for data in datasets.values()]
        assert all(
            [key == keys[0] for key in keys]
        ), "Datasets must have same set of keys!"

        all_species = get_species(datasets, ignore_check=False)

        datasets = {
            split: ProcessedDataset(
                data,
                included_species=all_species,
                subtract_thermo=self.subtract_thermo,
            )
            for split, data in datasets.items()
        }

        assert (
            len(
                set(tuple(data.included_species.tolist()) for data in datasets.values())
            )
            == 1
        ), "All datasets must have same included_species! {}".format(
            {key: data.included_species for key, data in datasets.items()}
        )

        self.num_species = datasets["train"].num_species
        self.max_charge = datasets["train"].max_charge

        self.ds_train, self.ds_val, self.ds_test = (
            datasets["train"],
            datasets["valid"],
            datasets["test"],
        )

    def prepare_data(self):
        """Prepare or download QM9 tensors and record file paths."""
        self.datafiles = prepare_dataset(
            self.data_dir, "qm9", force_download=self.force_download
        )

    def _collate(self, batch):
        """Collate datapoints into the batch format for EGNN/Cormorant.

        Parameters
        ----------
        batch : list of dict-like
            Each element corresponds to a molecule with tensor properties.

        Returns
        -------
        dict
            A dictionary of tensors with masks and edge information.
        """
        batch = {
            prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()
        }

        to_keep = batch["charges"].sum(0) > 0

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch["charges"] > 0
        batch["atom_mask"] = atom_mask

        # Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        # mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        batch["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        return batch

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("EGNN dataset")
        parser.add_argument(
            "--charge_power",
            type=int,
            default=2,
            help="Charge power to use.",
        )
        return parent_parser


class EGNNCP3DDataModule(DataModule):
    """EGNN DataModule for CycPept3D.

    Builds ``ProcessedDataset`` objects from raw PyG items or predefined splits.
    """

    def __init__(
        self,
        data_dir: pathlib.Path | str,
        split: int,
        batch_size: int = 256,
        num_workers: int = 8,
        remove_hydrogen: bool = False,
        **kwargs,
    ):
        self.data_dir = data_dir
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
        raw_dataset = CycPept3DDataset(
            root=self.data_dir, remove_hydrogen=remove_hydrogen
        )

        csv_path = raw_dataset.raw_paths[0]
        if split is not None:
            split_col = f"split{split}"
            split_df = pd.read_csv(
                csv_path,
                usecols=["CycPeptMPDB_ID", "Source", split_col],
                dtype={"CycPeptMPDB_ID": str, "Source": str},
            ).assign(
                Source_ID=lambda x: x["Source"].str.cat(
                    x["CycPeptMPDB_ID"].astype(str), sep="_"
                )
            )
            split_dict = split_df.set_index("Source_ID")[split_col].to_dict()

            train_indices = []
            val_indices = []
            test_indices = []
            dispatch = {
                "train": train_indices.append,
                "valid": val_indices.append,
                "test": test_indices.append,
            }

            for i, data in enumerate(raw_dataset):
                name = data.name
                try:
                    split_type = split_dict[name]
                except KeyError:
                    raise ValueError(
                        f"Data point '{name}' not found in split dictionary."
                    )

                try:
                    func = dispatch[split_type]
                except KeyError:
                    raise ValueError(
                        f"Unknown split type '{split_type}' for {name}. Expected 'train', 'valid', or 'test'."
                    )
                func(i)

            self.ds_train = to_processed_dataset(Subset(raw_dataset, train_indices))
            self.ds_val = to_processed_dataset(Subset(raw_dataset, val_indices))
            self.ds_test = to_processed_dataset(Subset(raw_dataset, test_indices))
        else:
            # Fallback to random split if no split dictionary is provided
            ds_train, ds_val, ds_test = random_split(
                raw_dataset,
                get_split_sizes(raw_dataset),
                generator=torch.Generator().manual_seed(kwargs["seed"]),
            )
            self.ds_train, self.ds_val, self.ds_test = (
                to_processed_dataset(ds_train),
                to_processed_dataset(ds_val),
                to_processed_dataset(ds_test),
            )

        self.num_species = self.ds_train.num_species
        self.max_charge = self.ds_train.max_charge

    def _collate(self, batch):
        """Collate datapoints into the batch format for EGNN.

        Parameters
        ----------
        batch : list of dict-like
            Each element is a dictionary with molecule-level tensors.

        Returns
        -------
        dict
            A dictionary of tensors with shape and masks expected by EGNN.
        """
        batch = {
            prop: batch_stack([mol[prop] for mol in batch])
            for prop in batch[0].keys()
            if prop != "name"
        }

        to_keep = batch["charges"].sum(0) > 0

        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

        atom_mask = batch["charges"] > 0
        batch["atom_mask"] = atom_mask

        # Obtain edges
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        # mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        batch["edge_mask"] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        return batch


## CPMP


class CPMPQM9DataModule(DataModule):
    """CPMP DataModule for QM9 (SDF + CSV).

    Performs feature extraction from molecules and yields padded adjacency,
    distance, node features, and labels.
    """

    HAR2EV = 27.211386246
    KCALMOL2EV = 0.04336414

    # fmt: off
    conversion = np.array(
        [1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
        1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.], dtype=np.float32
    )
    # fmt: on
    raw_url = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip"
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"

    def __init__(
        self,
        data_dir: pathlib.Path,
        task: str = "homo",
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.target = task
        self.sdf_filename = self.data_dir / "gdb9.sdf"
        self.csv_filename = self.data_dir / "gdb9.sdf.csv"
        self.uncharacterized_filename = self.data_dir / "uncharacterized.txt"
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
        suppl = Chem.SDMolSupplier(self.sdf_filename, removeHs=False, sanitize=False)
        y_df = pd.read_csv(
            self.csv_filename,
            usecols=range(1, 20),
            dtype=np.float32,
        )
        # Move A, B, C columns to the end
        y_df = y_df.iloc[:, list(range(3, y_df.shape[1])) + [0, 1, 2]]
        # Unit conversion
        y_df = y_df.mul(self.conversion, axis=1)

        with open(self.uncharacterized_filename) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        with Parallel(n_jobs=-1) as pool:
            x_all = pool(
                delayed(featurize_mol)(mol, True, False)
                for i, mol in enumerate(tqdm(suppl, total=len(suppl)))
                if i not in skip
            )
        output = [
            Molecule(data[0], data[1], i)
            for i, data in enumerate(zip(x_all, y_df.to_dict(orient="records")))
        ]
        full_dataset = MolDataset(output)
        self.ds_train, self.ds_val, self.ds_test = random_split(
            full_dataset,
            _get_split_sizes(full_dataset),
            generator=torch.Generator().manual_seed(kwargs["seed"]),
        )
        self.d_atom = self.ds_train.dataset.data_list[0].node_features.shape[1]

    def prepare_data(self):
        """Download and unzip QM9 SDF/CSV files if missing."""
        if not (
            self.sdf_filename.is_file()
            and self.csv_filename.is_file()
            and self.uncharacterized_filename.is_file()
        ):
            file_path = download_url(self.raw_url, self.data_dir)
            extract_zip(file_path, self.data_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.data_dir)
            (self.data_dir / "3195404").rename(self.data_dir / "uncharacterized.txt")

    def _collate(self, batch):
        """Pad molecules in a batch to the largest number of atoms.

        Parameters
        ----------
        batch : list[Molecule]
            Input list of molecules with per-graph arrays.

        Returns
        -------
        tuple of torch.Tensor
            ``(adjacency, features, distance, labels)`` arrays with shape
            ``(B, N, N)``, ``(B, N, F)``, ``(B, N, N)``, and ``(B, 1)``.
        """
        adjacency_list, distance_list, features_list = [], [], []
        labels = []

        max_size = 0
        for molecule in batch:
            labels.append(molecule.y[self.target])
            if molecule.adjacency_matrix.shape[0] > max_size:
                max_size = molecule.adjacency_matrix.shape[0]

        for molecule in batch:
            adjacency_list.append(
                pad_array(molecule.adjacency_matrix, (max_size, max_size))
            )
            distance_list.append(
                pad_array(molecule.distance_matrix, (max_size, max_size))
            )
            features_list.append(
                pad_array(
                    molecule.node_features, (max_size, molecule.node_features.shape[1])
                )
            )

        adjacency_array = np.array(adjacency_list)
        features_array = np.array(features_list)
        distance_array = np.array(distance_list)
        labels_array = np.array(labels)[:, None]

        return (
            torch.tensor(adjacency_array, dtype=torch.float32),
            torch.tensor(features_array, dtype=torch.float32),
            torch.tensor(distance_array, dtype=torch.float32),
            torch.tensor(labels_array, dtype=torch.float32),
        )


class CPMPCP3DDataModule(DataModule):
    """CPMP DataModule for CycPept3D.

    Loads pre-pickled features/labels and pads batches for the CPMP model.
    """

    def __init__(
        self,
        data_dir: pathlib.Path,
        pdb: bool,
        ff: str,
        ig: bool,
        wh: bool,
        split: int,
        batch_size: int = 256,
        num_workers: int = 8,
        **kwargs,
    ):
        self.data_dir = data_dir
        self.ff = ff
        self.ig = ig
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self._collate,
        )
        if pdb:
            X_train = pd.read_pickle(
                self.data_dir / f"X_train_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
            X_val = pd.read_pickle(
                self.data_dir / f"X_val_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
            X_test = pd.read_pickle(
                self.data_dir / f"X_test_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
            y_train = pd.read_pickle(
                self.data_dir / f"y_train_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
            y_val = pd.read_pickle(
                self.data_dir / f"y_val_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(
                self.data_dir / f"y_test_pdb_wH_{wh}_{split}.pkl"
            ).values.tolist()
        else:
            X_train = pd.read_pickle(
                self.data_dir / f"X_train_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
            X_val = pd.read_pickle(
                self.data_dir / f"X_val_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
            X_test = pd.read_pickle(
                self.data_dir / f"X_test_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
            y_train = pd.read_pickle(
                self.data_dir / f"y_train_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
            y_val = pd.read_pickle(
                self.data_dir / f"y_val_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
            y_test = pd.read_pickle(
                self.data_dir / f"y_test_{ff}_{ig}_{split}.pkl"
            ).values.tolist()
        self.d_atom = X_train[0][0].shape[1]
        self.ds_train = construct_dataset(X_train, y_train)
        self.ds_val = construct_dataset(X_val, y_val)
        self.ds_test = construct_dataset(X_test, y_test)

    def _collate(self, batch):
        """Create a padded batch of molecule features.

        Parameters
        ----------
        batch : list[Molecule]
            A batch of raw molecules.

        Returns
        -------
        tuple of torch.Tensor
            Padded adjacency matrices, node features, distance matrices, and
            labels as float tensors.
        """
        adjacency_list, distance_list, features_list = [], [], []
        labels = []

        max_size = 0
        for molecule in batch:
            if isinstance(molecule.y[0], np.ndarray):
                labels.append(molecule.y[0])
            else:
                labels.append(molecule.y)
            if molecule.adjacency_matrix.shape[0] > max_size:
                max_size = molecule.adjacency_matrix.shape[0]

        for molecule in batch:
            adjacency_list.append(
                pad_array(molecule.adjacency_matrix, (max_size, max_size))
            )
            distance_list.append(
                pad_array(molecule.distance_matrix, (max_size, max_size))
            )
            features_list.append(
                pad_array(
                    molecule.node_features, (max_size, molecule.node_features.shape[1])
                )
            )

        adjacency_array = np.array(adjacency_list)
        features_array = np.array(features_list)
        distance_array = np.array(distance_list)
        labels_array = np.array(labels)

        return (
            torch.tensor(adjacency_array),
            torch.tensor(features_array),
            torch.tensor(distance_array),
            torch.tensor(labels_array),
        )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("CPMP CycPept3D dataset")
        parser.add_argument(
            "--pdb",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Whether to use the structure read from PDB files",
        )
        parser.add_argument(
            "--ig",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Whether to ",
        )
        parser.add_argument(
            "--wh",
            type=str2bool,
            nargs="?",
            const=True,
            default=False,
            help="Whether to include hydrogen in the structure",
        )
        parser.add_argument(
            "--ff",
            type=str,
            choices=["uff", "mmff"],
            default="mmff",
            help="Type of force field to use",
        )
        return parent_parser
