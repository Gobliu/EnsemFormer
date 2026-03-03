# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

from enum import Enum
from itertools import product
from typing import Dict

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from dgl import DGLGraph
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

from ..fiber import Fiber
from ..utils import degree_to_dim, unfuse_features


class ConvSE3FuseLevel(Enum):
    FULL = 2
    PARTIAL = 1
    NONE = 0


class RadialProfile(nn.Module):
    """Radial profile function. Outputs weights used to weigh basis matrices."""

    def __init__(self, num_freq: int, channels_in: int, channels_out: int,
                 edge_dim: int = 1, mid_dim: int = 32, use_layer_norm: bool = False):
        super().__init__()
        modules = [
            nn.Linear(edge_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, mid_dim),
            nn.LayerNorm(mid_dim) if use_layer_norm else None,
            nn.ReLU(),
            nn.Linear(mid_dim, num_freq * channels_in * channels_out, bias=False)
        ]
        self.net = torch.jit.script(nn.Sequential(*[m for m in modules if m is not None]))

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


class VersatileConvSE3(nn.Module):
    """Building block for TFN convolutions (fully fused, partially fused, or pairwise)."""

    def __init__(self, freq_sum: int, channels_in: int, channels_out: int,
                 edge_dim: int, use_layer_norm: bool, fuse_level: ConvSE3FuseLevel):
        super().__init__()
        self.freq_sum = freq_sum
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.fuse_level = fuse_level
        self.radial_func = RadialProfile(num_freq=freq_sum, channels_in=channels_in,
                                         channels_out=channels_out, edge_dim=edge_dim,
                                         use_layer_norm=use_layer_norm)

    def forward(self, features: Tensor, invariant_edge_feats: Tensor, basis: Tensor):
        with nvtx_range('VersatileConvSE3'):
            num_edges = features.shape[0]
            in_dim = features.shape[2]
            with nvtx_range('RadialProfile'):
                radial_weights = self.radial_func(invariant_edge_feats) \
                    .view(-1, self.channels_out, self.channels_in * self.freq_sum)

            if basis is not None:
                basis_view = basis.view(num_edges, in_dim, -1)
                tmp = (features @ basis_view).view(num_edges, -1, basis.shape[-1])
                return radial_weights @ tmp
            else:
                return radial_weights @ features


class ConvSE3(nn.Module):
    """SE(3)-equivariant graph convolution (Tensor Field Network convolution)."""

    def __init__(self, fiber_in: Fiber, fiber_out: Fiber, fiber_edge: Fiber,
                 pool: bool = True, use_layer_norm: bool = False, self_interaction: bool = False,
                 max_degree: int = 4, fuse_level: ConvSE3FuseLevel = ConvSE3FuseLevel.FULL,
                 allow_fused_output: bool = False, low_memory: bool = False):
        super().__init__()
        self.pool = pool
        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.self_interaction = self_interaction
        self.max_degree = max_degree
        self.allow_fused_output = allow_fused_output
        self.conv_checkpoint = torch.utils.checkpoint.checkpoint if low_memory else lambda m, *x: m(*x)

        channels_in_set = set([f.channels + fiber_edge[f.degree] * (f.degree > 0) for f in self.fiber_in])
        channels_out_set = set([f.channels for f in self.fiber_out])
        unique_channels_in = (len(channels_in_set) == 1)
        unique_channels_out = (len(channels_out_set) == 1)
        degrees_up_to_max = list(range(max_degree + 1))
        common_args = dict(edge_dim=fiber_edge[0] + 1, use_layer_norm=use_layer_norm)

        if fuse_level.value >= ConvSE3FuseLevel.FULL.value and \
                unique_channels_in and fiber_in.degrees == degrees_up_to_max and \
                unique_channels_out and fiber_out.degrees == degrees_up_to_max:
            self.used_fuse_level = ConvSE3FuseLevel.FULL
            sum_freq = sum([
                degree_to_dim(min(d_in, d_out))
                for d_in, d_out in product(degrees_up_to_max, degrees_up_to_max)
            ])
            self.conv = VersatileConvSE3(sum_freq, list(channels_in_set)[0], list(channels_out_set)[0],
                                         fuse_level=self.used_fuse_level, **common_args)

        elif fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value and \
                unique_channels_in and fiber_in.degrees == degrees_up_to_max:
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_out = nn.ModuleDict()
            for d_out, c_out in fiber_out:
                sum_freq = sum([degree_to_dim(min(d_out, d)) for d in fiber_in.degrees])
                self.conv_out[str(d_out)] = VersatileConvSE3(sum_freq, list(channels_in_set)[0], c_out,
                                                             fuse_level=self.used_fuse_level, **common_args)

        elif fuse_level.value >= ConvSE3FuseLevel.PARTIAL.value and \
                unique_channels_out and fiber_out.degrees == degrees_up_to_max:
            self.used_fuse_level = ConvSE3FuseLevel.PARTIAL
            self.conv_in = nn.ModuleDict()
            for d_in, c_in in fiber_in:
                channels_in_new = c_in + fiber_edge[d_in] * (d_in > 0)
                sum_freq = sum([degree_to_dim(min(d_in, d)) for d in fiber_out.degrees])
                self.conv_in[str(d_in)] = VersatileConvSE3(sum_freq, channels_in_new, list(channels_out_set)[0],
                                                           fuse_level=self.used_fuse_level, **common_args)
        else:
            self.used_fuse_level = ConvSE3FuseLevel.NONE
            self.conv = nn.ModuleDict()
            for (degree_in, channels_in), (degree_out, channels_out) in (self.fiber_in * self.fiber_out):
                dict_key = f'{degree_in},{degree_out}'
                channels_in_new = channels_in + fiber_edge[degree_in] * (degree_in > 0)
                sum_freq = degree_to_dim(min(degree_in, degree_out))
                self.conv[dict_key] = VersatileConvSE3(sum_freq, channels_in_new, channels_out,
                                                       fuse_level=self.used_fuse_level, **common_args)

        if self_interaction:
            self.to_kernel_self = nn.ParameterDict()
            for degree_out, channels_out in fiber_out:
                if fiber_in[degree_out]:
                    self.to_kernel_self[str(degree_out)] = nn.Parameter(
                        torch.randn(channels_out, fiber_in[degree_out]) / np.sqrt(fiber_in[degree_out]))

    def _try_unpad(self, feature, basis):
        if basis is not None:
            out_dim = basis.shape[-1]
            out_dim += out_dim % 2 - 1
            return feature[..., :out_dim]
        else:
            return feature

    def forward(self, node_feats: Dict[str, Tensor], edge_feats: Dict[str, Tensor],
                graph: DGLGraph, basis: Dict[str, Tensor]):
        with nvtx_range('ConvSE3'):
            invariant_edge_feats = edge_feats['0'].squeeze(-1)
            src, dst = graph.edges()
            out = {}
            in_features = []

            for degree_in in self.fiber_in.degrees:
                src_node_features = node_feats[str(degree_in)][src]
                if degree_in > 0 and str(degree_in) in edge_feats:
                    src_node_features = torch.cat([src_node_features, edge_feats[str(degree_in)]], dim=1)
                in_features.append(src_node_features)

            if self.used_fuse_level == ConvSE3FuseLevel.FULL:
                in_features_fused = torch.cat(in_features, dim=-1)
                out = self.conv_checkpoint(
                    self.conv, in_features_fused, invariant_edge_feats, basis['fully_fused']
                )
                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(self, 'conv_out'):
                in_features_fused = torch.cat(in_features, dim=-1)
                for degree_out in self.fiber_out.degrees:
                    basis_used = basis[f'out{degree_out}_fused']
                    out[str(degree_out)] = self._try_unpad(
                        self.conv_checkpoint(
                            self.conv_out[str(degree_out)], in_features_fused, invariant_edge_feats, basis_used
                        ), basis_used)

            elif self.used_fuse_level == ConvSE3FuseLevel.PARTIAL and hasattr(self, 'conv_in'):
                out = 0
                for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                    out = out + self.conv_checkpoint(
                        self.conv_in[str(degree_in)], feature, invariant_edge_feats, basis[f'in{degree_in}_fused']
                    )
                if not self.allow_fused_output or self.self_interaction or self.pool:
                    out = unfuse_features(out, self.fiber_out.degrees)
            else:
                for degree_out in self.fiber_out.degrees:
                    out_feature = 0
                    for degree_in, feature in zip(self.fiber_in.degrees, in_features):
                        dict_key = f'{degree_in},{degree_out}'
                        basis_used = basis.get(dict_key, None)
                        out_feature = out_feature + self._try_unpad(
                            self.conv_checkpoint(
                                self.conv[dict_key], feature, invariant_edge_feats, basis_used
                            ), basis_used)
                    out[str(degree_out)] = out_feature

            for degree_out in self.fiber_out.degrees:
                if self.self_interaction and str(degree_out) in self.to_kernel_self:
                    with nvtx_range('self interaction'):
                        dst_features = node_feats[str(degree_out)][dst]
                        kernel_self = self.to_kernel_self[str(degree_out)]
                        out[str(degree_out)] = out[str(degree_out)] + kernel_self @ dst_features

                if self.pool:
                    with nvtx_range('pooling'):
                        if isinstance(out, dict):
                            out[str(degree_out)] = dgl.ops.copy_e_sum(graph, out[str(degree_out)])
                        else:
                            out = dgl.ops.copy_e_sum(graph, out)
            return out
