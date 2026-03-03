# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from typing import Dict, Literal

import torch.nn as nn
from dgl import DGLGraph
from dgl.nn.pytorch import AvgPooling, MaxPooling
from torch import Tensor


class GPooling(nn.Module):
    """
    Graph max/average pooling on a given feature type.
    The average can be taken for any feature type, and equivariance will be maintained.
    The maximum can only be taken for invariant features (type 0).
    """

    def __init__(self, feat_type: int = 0, pool: Literal['max', 'avg'] = 'max'):
        super().__init__()
        assert pool in ['max', 'avg'], f'Unknown pooling: {pool}'
        assert feat_type == 0 or pool == 'avg', 'Max pooling on type > 0 features will break equivariance'
        self.feat_type = feat_type
        self.pool = MaxPooling() if pool == 'max' else AvgPooling()

    def forward(self, features: Dict[str, Tensor], graph: DGLGraph, **kwargs) -> Tensor:
        pooled = self.pool(graph, features[str(self.feat_type)])
        return pooled.squeeze(dim=-1)
