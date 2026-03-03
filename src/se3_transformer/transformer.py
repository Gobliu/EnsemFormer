# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import logging
from typing import Optional, Literal, Dict

import torch
import torch.nn as nn
from dgl import DGLGraph
from torch import Tensor

from .basis import get_basis, update_basis_with_fused
from .layers.attention import AttentionBlockSE3
from .layers.convolution import ConvSE3, ConvSE3FuseLevel
from .layers.norm import NormSE3
from .layers.pooling import GPooling
from .fiber import Fiber


class Sequential(nn.Sequential):
    """Sequential module with arbitrary forward args and kwargs."""

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input


def get_populated_edge_features(relative_pos: Tensor, edge_features: Optional[Dict[str, Tensor]] = None):
    """Add relative positions to existing edge features."""
    edge_features = edge_features.copy() if edge_features else {}
    r = relative_pos.norm(dim=-1, keepdim=True)
    if '0' in edge_features:
        edge_features['0'] = torch.cat([edge_features['0'], r[..., None]], dim=1)
    else:
        edge_features['0'] = r[..., None]
    return edge_features


class SE3Transformer(nn.Module):
    def __init__(self,
                 num_layers: int,
                 fiber_in: Fiber,
                 fiber_hidden: Fiber,
                 fiber_out: Fiber,
                 num_heads: int,
                 channels_div: int,
                 fiber_edge: Fiber = Fiber({}),
                 return_type: Optional[int] = None,
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.return_type = return_type
        self.pooling = pooling
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        self.graph_modules = Sequential(*graph_modules)

        if pooling is not None:
            assert return_type is not None, 'return_type must be specified when pooling'
            self.pooling_module = GPooling(pool=pooling, feat_type=return_type)

    def forward(self, graph: DGLGraph, node_feats: Dict[str, Tensor],
                edge_feats: Optional[Dict[str, Tensor]] = None,
                basis: Optional[Dict[str, Tensor]] = None):
        basis = basis or get_basis(graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                   use_pad_trick=self.tensor_cores and not self.low_memory,
                                   amp=torch.is_autocast_enabled())

        basis = update_basis_with_fused(basis, self.max_degree,
                                        use_pad_trick=self.tensor_cores and not self.low_memory,
                                        fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)

        edge_feats = get_populated_edge_features(graph.edata['rel_pos'], edge_feats)

        node_feats = self.graph_modules(node_feats, edge_feats, graph=graph, basis=basis)

        if self.pooling is not None:
            return self.pooling_module(node_feats, graph=graph)

        if self.return_type is not None:
            return node_feats[str(self.return_type)]

        return node_feats


class SE3TransformerPooled(nn.Module):
    def __init__(self,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber,
                 num_degrees: int,
                 num_channels: int,
                 output_dim: int,
                 **kwargs):
        super().__init__()
        kwargs['pooling'] = kwargs.get('pooling') or 'max'
        self.transformer = SE3Transformer(
            fiber_in=fiber_in,
            fiber_hidden=Fiber.create(num_degrees, num_channels),
            fiber_out=fiber_out,
            fiber_edge=fiber_edge,
            return_type=0,
            **kwargs
        )

        n_out_features = fiber_out.num_features
        self.mlp = nn.Sequential(
            nn.Linear(n_out_features, n_out_features),
            nn.ReLU(),
            nn.Linear(n_out_features, output_dim)
        )

    def forward(self, graph, node_feats, edge_feats, basis=None):
        feats = self.transformer(graph, node_feats, edge_feats, basis).squeeze(-1)
        y = self.mlp(feats).squeeze(-1)
        return y
