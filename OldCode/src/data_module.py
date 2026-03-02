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

"""Lightweight DataModule pattern and dataloader helper.

This module defines an abstract :class:`DataModule` with a small interface used
by the training loop and a helper function to construct a classic or
distributed-aware PyTorch ``DataLoader`` depending on the runtime context.
"""

import torch.distributed as dist
from abc import ABC
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from src.utils import get_local_rank


def _get_dataloader(dataset: Dataset, shuffle: bool, **kwargs) -> DataLoader:
    """Create a DataLoader aware of distributed training.

    Parameters
    ----------
    dataset : Dataset
        Dataset instance to load from.
    shuffle : bool
        Whether to shuffle the data. When distributed, shuffling is delegated
        to the :class:`DistributedSampler`.
    **kwargs
        Additional :class:`~torch.utils.data.DataLoader` keyword arguments
        (e.g., ``batch_size``, ``num_workers``, ``collate_fn``).

    Returns
    -------
    DataLoader
        A configured dataloader with the appropriate sampler if distribution is
        initialized.
    """
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle) if dist.is_initialized() else None
    )
    return DataLoader(
        dataset, shuffle=(shuffle and sampler is None), sampler=sampler, **kwargs
    )


class DataModule(ABC):
    """Abstract DataModule.

    Subclasses are expected to set the attributes ``ds_train``, ``ds_val``, and
    ``ds_test`` to objects compatible with :class:`~torch.utils.data.Dataset`.

    Parameters
    ----------
    **dataloader_kwargs
        Keyword arguments passed to internal dataloader construction. Common
        keys include ``batch_size``, ``num_workers``, and ``collate_fn``.
    """

    def __init__(self, **dataloader_kwargs):
        super().__init__()
        if get_local_rank() == 0:
            self.prepare_data()

        # Wait until rank zero has prepared the data (download, preprocessing, ...)
        if dist.is_initialized():
            dist.barrier(device_ids=[get_local_rank()])

        self.dataloader_kwargs = {
            "pin_memory": True,
            "persistent_workers": dataloader_kwargs.get("num_workers", 0) > 0,
            **dataloader_kwargs,
        }
        self.ds_train, self.ds_val, self.ds_test = None, None, None

    def prepare_data(self):
        """Called once per node to download or preprocess data.

        Implement heavy setup here (downloading, precomputations, etc.). The
        default implementation does nothing.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_train, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_val, shuffle=False, **self.dataloader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return _get_dataloader(self.ds_test, shuffle=False, **self.dataloader_kwargs)
