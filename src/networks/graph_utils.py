"""Shared graph/tensor utility functions for the network modules."""

import torch
import torch.nn.functional as F_torch


def bond_type_to_one_hot(bond_type: torch.Tensor) -> torch.Tensor:
    """Convert integer bond type matrix to one-hot edge features.

    Parameters
    ----------
    bond_type : LongTensor  (..., N, N)
        Integer bond types: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic.

    Returns
    -------
    FloatTensor  (..., N, N, 4)
        One-hot encoding for types 1-4.  Type 0 (no bond) maps to all-zeros.
    """
    clamped = bond_type.clamp(0, 4)
    oh = F_torch.one_hot(clamped, num_classes=5).float()
    return oh[..., 1:]  # drop the "no bond" column -> (..., N, N, 4)
