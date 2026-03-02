"""Utility helpers shared across the project.

Contains tensor utilities, distributed helpers, random seeding, and small math
helpers. Functions are designed to be lightweight and framework-agnostic
where possible.
"""

import argparse
import logging
import math
import os
import pathlib
from functools import wraps
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.init import (
    _calculate_fan_in_and_fan_out,
    _no_grad_normal_,
    _no_grad_uniform_,
)
from torch.utils.data import Dataset


def str2bool(v: Union[bool, str]) -> bool:
    """Parse a boolean from a variety of string representations.

    Accepts yes/true/t/y/1 and no/false/f/n/0 (case-insensitive). Boolean
    inputs are returned unchanged.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def to_device(x, device):
    """Recursively move tensors/collections to a device.

    Parameters
    ----------
    x : Any
        Tensor, mapping, sequence, or None.
    device : torch.device
        Target device.

    Returns
    -------
    Any
        Object mirrored on the target device with the same structure as input.
    """
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    elif isinstance(x, list):
        return [to_device(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    elif x is None:
        return x
    else:
        return x.to(device=device, non_blocking=True)


def get_local_rank() -> int:
    """Return the integer LOCAL_RANK from environment (default 0)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed() -> bool:
    """Initialize torch.distributed from environment variables.

    Returns
    -------
    bool
        True if distributed training is initialized, False otherwise.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    if distributed:
        backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if backend == "cpu:gloo,cuda:nccl":
            torch.cuda.set_device(get_local_rank())
        else:
            logging.warning("Running on CPU only!")
        assert torch.distributed.is_initialized()
    return distributed


def rank_zero_only(fn):
    """Decorator that runs a function only on rank 0 in distributed mode."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if not dist.is_initialized() or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def using_tensor_cores(amp: bool) -> bool:
    """Heuristically determine whether to use Tensor Cores.

    Considers the GPU compute capability and whether AMP is enabled.
    """
    major_cc, _ = torch.cuda.get_device_capability()
    return (amp and major_cc >= 7) or major_cc >= 8


def get_split_sizes(full_dataset: Dataset) -> tuple[int, int, int]:
    """Return an 80/10/10 split for a dataset length.

    Parameters
    ----------
    full_dataset : Dataset
        The full dataset instance.

    Returns
    -------
    tuple[int, int, int]
        Train/val/test lengths that sum to ``len(full_dataset)``.
    """
    len_full = len(full_dataset)
    len_train = int(0.8 * len_full)
    len_test = int(0.1 * len_full)
    len_val = len_full - len_train - len_test
    return len_train, len_val, len_test


def get_next_version(log_dir: pathlib.Path) -> int:
    """Return the next integer version based on existing run_N directories."""
    if not log_dir.exists():
        return 0
    existing = []
    for d in log_dir.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                n = int(d.name.split("_", 1)[1])
                existing.append(n)
            except (IndexError, ValueError):
                pass
    return 0 if not existing else max(existing) + 1


def print_parameters_count(model: torch.nn.Module):
    """Log the number of trainable parameters in a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose trainable parameter count is reported.
    """
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params_trainable}")


def xavier_normal_small_init_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier normal init with a smaller variance (fan_out weighted 4×)."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    return _no_grad_normal_(tensor, 0.0, std)


def xavier_uniform_small_init_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier uniform init with a smaller variance (fan_out weighted 4×)."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4 * fan_out))
    a = math.sqrt(3.0) * std
    return _no_grad_uniform_(tensor, -a, a)
