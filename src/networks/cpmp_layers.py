"""CPMP internal transformer components.

Carried over verbatim from OldCode/cpmp/model/transformer.py. These form the
building blocks used by CPMPBackbone.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import xavier_normal_small_init_, xavier_uniform_small_init_

INITERS = {
    "uniform": nn.init.xavier_uniform_,
    "normal": nn.init.xavier_normal_,
    "small_normal_init": xavier_normal_small_init_,
    "small_uniform_init": xavier_uniform_small_init_,
}


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ScaleNorm(nn.Module):
    def __init__(self, scale, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(math.sqrt(scale)))
        self.eps = eps

    def forward(self, x):
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, scale_norm):
        super().__init__()
        self.norm = ScaleNorm(size) if scale_norm else LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, scale_norm):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([
            SublayerConnection(size, dropout, scale_norm),
            SublayerConnection(size, dropout, scale_norm),
        ])
        self.size = size

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        x = self.sublayer[0](
            x,
            lambda x: self.self_attn(x, x, x, adj_matrix, distances_matrix, edges_att, mask),
        )
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, N, d_model, dropout, scale_norm, attention_kwargs, feedforward_kwargs):
        super().__init__()
        layers = []
        for _ in range(N):
            attn = MultiHeadedAttention(**attention_kwargs)
            ff = PositionwiseFeedForward(**feedforward_kwargs)
            layers.append(EncoderLayer(d_model, attn, ff, dropout, scale_norm))
        self.layers = nn.ModuleList(layers)
        self.norm = ScaleNorm(d_model) if scale_norm else LayerNorm(d_model)

    def forward(self, x, mask, adj_matrix, distances_matrix, edges_att):
        for layer in self.layers:
            x = layer(x, mask, adj_matrix, distances_matrix, edges_att)
        return self.norm(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, d_atom, dropout):
        super().__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.lut(x))


class EdgeFeaturesLayer(nn.Module):
    def __init__(self, d_model, d_edge, h, dropout):
        super().__init__()
        assert d_model % h == 0
        self.linear = nn.Linear(d_edge, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.fill_(0.25)

    def forward(self, x):
        p_edge = x.permute(0, 2, 3, 1)
        p_edge = self.linear(p_edge).permute(0, 3, 1, 2)
        return torch.relu(p_edge)


def attention(
    query, key, value, adj_matrix, distances_matrix, edges_att,
    mask=None, dropout=None, lambdas=(0.3, 0.3, 0.4),
    trainable_lambda=False, use_edge_features=False,
    eps=1e-6, inf=1e12,
):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        expanded_mask = (
            mask.unsqueeze(1).repeat(1, query.shape[1], query.shape[2], 1).to(device=scores.device)
        )
        if expanded_mask.dtype != torch.bool:
            expanded_mask = (expanded_mask == 0)
        scores = scores.masked_fill(expanded_mask == 0, -inf)
    p_attn = F.softmax(scores, dim=-1)

    if use_edge_features:
        adj_matrix = edges_att.reshape_as(adj_matrix).to(device=query.device, dtype=value.dtype)

    adj_matrix = adj_matrix.to(device=query.device, dtype=value.dtype)
    adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1).unsqueeze(2) + eps)
    adj_matrix = adj_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
    p_adj = adj_matrix
    p_dist = distances_matrix

    if trainable_lambda:
        softmax_attention, softmax_distance, softmax_adjacency = (
            lambdas.to(query.device) if isinstance(lambdas, torch.Tensor) else lambdas
        )
        p_weighted = softmax_attention * p_attn + softmax_distance * p_dist + softmax_adjacency * p_adj
    else:
        lambda_attention, lambda_distance, lambda_adjacency = lambdas
        p_weighted = lambda_attention * p_attn + lambda_distance * p_dist + lambda_adjacency * p_adj

    if dropout is not None:
        p_weighted = dropout(p_weighted)

    atoms_features = torch.matmul(p_weighted, value)
    return atoms_features, p_weighted, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(
        self, h, d_model, dropout=0.1, lambda_attention=0.3, lambda_distance=0.3,
        trainable_lambda=False, distance_matrix_kernel="softmax",
        use_edge_features=False, integrated_distances=False,
        d_edge: int | None = None,
    ):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.trainable_lambda = trainable_lambda
        if trainable_lambda:
            lambda_adjacency = 1.0 - lambda_attention - lambda_distance
            self.lambdas = nn.Parameter(
                torch.tensor([lambda_attention, lambda_distance, lambda_adjacency], requires_grad=True)
            )
        else:
            lambda_adjacency = 1.0 - lambda_attention - lambda_distance
            self.lambdas = (lambda_attention, lambda_distance, lambda_adjacency)

        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        if distance_matrix_kernel == "softmax":
            self.distance_matrix_kernel = lambda x: F.softmax(-x, dim=-1)
        elif distance_matrix_kernel == "exp":
            self.distance_matrix_kernel = lambda x: torch.exp(-x)
        self.integrated_distances = integrated_distances
        self.use_edge_features = use_edge_features
        if use_edge_features:
            if d_edge is None:
                d_edge = 11 if not integrated_distances else 12
            elif integrated_distances:
                d_edge = d_edge + 1
            self.edges_feature_layer = EdgeFeaturesLayer(d_model, d_edge, h, dropout)

    def forward(self, query, key, value, adj_matrix, distances_matrix, edges_att, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        bool_mask = mask.repeat(1, mask.shape[-1], 1).to(device=distances_matrix.device)
        if bool_mask.dtype != torch.bool:
            bool_mask = bool_mask != 0
        distances_matrix = distances_matrix.masked_fill(~bool_mask, torch.inf)
        distances_matrix = self.distance_matrix_kernel(distances_matrix)
        p_dist = distances_matrix.unsqueeze(1).repeat(1, query.shape[1], 1, 1)

        if self.use_edge_features:
            if self.integrated_distances:
                edges_att = torch.cat((edges_att, distances_matrix.unsqueeze(1)), dim=1)
            edges_att = self.edges_feature_layer(edges_att)

        x, self.attn, self.self_attn = attention(
            query, key, value, adj_matrix, p_dist, edges_att,
            mask=mask, dropout=self.dropout, lambdas=self.lambdas,
            trainable_lambda=self.trainable_lambda, use_edge_features=self.use_edge_features,
        )
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, N_dense, dropout=0.1, leaky_relu_slope=0.0, dense_output_nonlinearity="relu"):
        super().__init__()
        self.N_dense = N_dense
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(N_dense)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(N_dense)])
        self.leaky_relu_slope = leaky_relu_slope
        if dense_output_nonlinearity == "relu":
            self.dense_output_nonlinearity = lambda x: F.leaky_relu(x, negative_slope=self.leaky_relu_slope)
        elif dense_output_nonlinearity == "tanh":
            self.tanh = nn.Tanh()
            self.dense_output_nonlinearity = lambda x: self.tanh(x)
        elif dense_output_nonlinearity == "none":
            self.dense_output_nonlinearity = lambda x: x

    def forward(self, x):
        if self.N_dense == 0:
            return x
        for i in range(len(self.linears) - 1):
            x = self.dropout[i](F.leaky_relu(self.linears[i](x), negative_slope=self.leaky_relu_slope))
        return self.dropout[-1](self.dense_output_nonlinearity(self.linears[-1](x)))
