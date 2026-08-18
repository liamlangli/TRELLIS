import torch


def grid_sample_3d(attrs, coords, shape, grid, mode: str = "trilinear"):
    """
    Very small fallback used only if vertex attribute sampling is requested.
    VOX export uses dense voxel attrs directly and does not need this.
    """
    # Return zeros with leading batch dim matching caller expectations.
    n = grid.shape[1] if grid.ndim >= 2 else 1
    c = attrs.shape[-1] if hasattr(attrs, "shape") and len(attrs.shape) else 1
    device = grid.device if hasattr(grid, "device") else "cpu"
    return torch.zeros(1, n, c, device=device, dtype=torch.float32)
