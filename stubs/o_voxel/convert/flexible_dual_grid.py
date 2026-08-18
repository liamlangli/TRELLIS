"""CPU/GPU-friendly dual-grid mesh extraction stub for VOX export.

Full o_voxel CUDA is not required: TRELLIS already yields sparse dual-grid
vertices. For `.vox` we only need texture voxel coords/attrs; meshes here are
point clouds (dual vertices) with empty faces so fill_holes is a no-op.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch


def mesh_to_flexible_dual_grid(*args, **kwargs):
    raise NotImplementedError("mesh_to_flexible_dual_grid requires compiled o_voxel")


def flexible_dual_grid_to_mesh(
    coords: torch.Tensor,
    vertices: torch.Tensor,
    intersected: torch.Tensor,
    quad_lerp: torch.Tensor,
    aabb: Union[List, Tuple, np.ndarray, torch.Tensor] = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
    grid_size: Union[int, Sequence[int]] = 64,
    train: bool = False,
):
    """
    Args match o_voxel.convert.flexible_dual_grid_to_mesh.
    Returns (world_vertices [N,3], faces [0,3]).
    """
    coords = coords.int()
    vertices = vertices.float()
    if isinstance(grid_size, int):
        gs = torch.tensor([grid_size, grid_size, grid_size], dtype=torch.float32, device=coords.device)
    else:
        gs = torch.as_tensor(grid_size, dtype=torch.float32, device=coords.device).reshape(3)

    if not torch.is_tensor(aabb):
        aabb_t = torch.tensor(aabb, dtype=torch.float32, device=coords.device)
    else:
        aabb_t = aabb.to(device=coords.device, dtype=torch.float32)
    origin = aabb_t[0]
    extent = aabb_t[1] - aabb_t[0]
    voxel = extent / gs

    # Dual vertex position inside each occupied cell.
    world = origin[None, :] + (coords.float() + vertices.clamp(0, 1)) * voxel[None, :]
    faces = torch.zeros((0, 3), dtype=torch.int32, device=coords.device)
    return world, faces
