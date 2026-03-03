# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

from ..fiber import Fiber


@torch.jit.script
def clamped_norm(x, clamp: float):
    return x.norm(p=2, dim=-1, keepdim=True).clamp(min=clamp)


@torch.jit.script
def rescale(x, norm, new_norm):
    return x / norm * new_norm


class NormSE3(nn.Module):
    """
    Norm-based SE(3)-equivariant nonlinearity.

                 +-->  feature_norm --> LayerNorm() --> ReLU() --+
    feature_in --+                                               * --> feature_out
                 +-->  feature_phase ----------------------------+
    """

    NORM_CLAMP = 2 ** -24  # Minimum positive subnormal for FP16

    def __init__(self, fiber: Fiber, nonlinearity: nn.Module = nn.ReLU()):
        super().__init__()
        self.fiber = fiber
        self.nonlinearity = nonlinearity

        if len(set(fiber.channels)) == 1:
            self.group_norm = nn.GroupNorm(num_groups=len(fiber.degrees), num_channels=sum(fiber.channels))
        else:
            self.layer_norms = nn.ModuleDict({
                str(degree): nn.LayerNorm(channels)
                for degree, channels in fiber
            })

    def forward(self, features: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        with nvtx_range('NormSE3'):
            output = {}
            if hasattr(self, 'group_norm'):
                norms = [clamped_norm(features[str(d)], self.NORM_CLAMP)
                         for d in self.fiber.degrees]
                fused_norms = torch.cat(norms, dim=-2)

                new_norms = self.nonlinearity(self.group_norm(fused_norms.squeeze(-1))).unsqueeze(-1)
                new_norms = torch.chunk(new_norms, chunks=len(self.fiber.degrees), dim=-2)

                for norm, new_norm, d in zip(norms, new_norms, self.fiber.degrees):
                    output[str(d)] = rescale(features[str(d)], norm, new_norm)
            else:
                for degree, feat in features.items():
                    norm = clamped_norm(feat, self.NORM_CLAMP)
                    new_norm = self.nonlinearity(self.layer_norms[degree](norm.squeeze(-1)).unsqueeze(-1))
                    output[degree] = rescale(new_norm, feat, norm)

            return output
