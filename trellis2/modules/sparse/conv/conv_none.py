"""Portable submanifold convolution with CPU coordinate lookup and MPS matmul.

Weights use the original FlexGEMM layout (Co, Kx, Ky, Kz, Ci).
No dense feature volume, CUDA extension or placeholder operators are used.
"""
import itertools
import math
import numpy as np
import torch
from torch import nn


def triple(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,) * 3


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size, stride=1,
                       dilation=1, padding=None, bias=True, indice_key=None):
    if triple(stride) != (1, 1, 1) or padding is not None:
        raise ValueError('Voxel decoder requires submanifold convolution (stride=1, padding=None)')
    self.kernel_size, self.dilation = triple(kernel_size), triple(dilation)
    self.in_channels, self.out_channels = in_channels, out_channels
    self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
    nn.init.uniform_(self.weight, -1 / math.sqrt(in_channels * math.prod(self.kernel_size)),
                     1 / math.sqrt(in_channels * math.prod(self.kernel_size)))
    self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None


def neighbor_pairs(coords, kernel, dilation, device):
    """Vectorized exact lookup, including batch and out-of-bounds checks."""
    c = coords.detach().cpu().numpy().astype(np.int64)
    if not len(c):
        return []
    lo = c.min(0)
    shifted = c - lo
    dims = shifted.max(0) + 1
    strides = np.array([np.prod(dims[1:]), np.prod(dims[2:]), dims[3], 1], dtype=np.int64)
    keys = shifted @ strides
    order = np.argsort(keys)
    sorted_keys = keys[order]
    pairs = []
    for offset in itertools.product(*(range(k) for k in kernel)):
        delta = np.array([0, *((o-k//2)*d for o, k, d in zip(offset, kernel, dilation))])
        query_coords = shifted + delta
        valid = np.all((query_coords >= 0) & (query_coords < dims), axis=1)
        target = np.flatnonzero(valid)
        query = query_coords[target] @ strides
        index = np.searchsorted(sorted_keys, query)
        keep = index < len(sorted_keys)
        target, query, index = target[keep], query[keep], index[keep]
        keep = sorted_keys[index] == query
        src = torch.as_tensor(order[index[keep]].copy(), device=device)
        dst = torch.as_tensor(target[keep].copy(), device=device)
        pairs.append((src, dst))
    return pairs


def sparse_conv3d_forward(self, x):
    key = f'portable_neighbors_{self.kernel_size}_{self.dilation}_{x.device}'
    pairs = x.get_spatial_cache(key)
    if pairs is None:
        pairs = neighbor_pairs(x.coords, self.kernel_size, self.dilation, x.device)
        x.register_spatial_cache(key, pairs)
    weights = self.weight.flatten(1, 3).permute(1, 2, 0)
    out = x.feats.new_zeros((len(x.feats), self.out_channels))
    for weight, (src, dst) in zip(weights, pairs):
        # Each destination is unique per offset. Bound intermediate allocation
        # and Metal dispatch length for decoder-sized voxel sets.
        for start in range(0, src.numel(), 32768):
            source, target = src[start:start+32768], dst[start:start+32768]
            out[target] = out[target] + x.feats[source] @ weight
    if self.bias is not None:
        out = out + self.bias
    return x.replace(out)


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError('Inverse sparse convolution is not used by voxel inference')


def sparse_inverse_conv3d_forward(self, x):
    raise NotImplementedError('Inverse sparse convolution is not used by voxel inference')
