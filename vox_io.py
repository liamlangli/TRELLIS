"""
Reader/writer for the VOX2 sparse-brick format defined in vox_codec.ns.

File layout (little-endian):

    magic       : 4 bytes = b"VOX2"
    version     : u32     = 2
    size_x      : u32
    size_y      : u32
    size_z      : u32
    brick       : u32     = 4
    flags       : u32     bit0 = RLE, bit1 = zstd payload
    brick_count : u32
    payload     : palette then brick records, or a zstd frame of both

Version 2 payload prefix:

    palette_count : u32     = 1..256, index 0 is always air
    palette       : (r:u8, g:u8, b:u8, material:u8) * palette_count
    bricks        : occupied brick records

Each brick cell stores a palette index (not a free-floating world material).
Index 0 is air. Encoder remaps used material bytes into the compact palette.
Decoder expands indices back to material ids for the rest of this project.

Version 1 is still accepted on read (no palette prefix; cells are materials).

Each RLE brick record is:
    cx:u8 cy:u8 cz:u8
    runs:u8
    (count:u8, value:u8) * runs

which expands to exactly 64 cell bytes, X-fastest, then Y, then Z.
Material / palette index 0 is air. Empty bricks are omitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
import struct

import numpy as np

try:
    import zstandard as zstd
except ImportError:
    zstd = None


VOX1_MAGIC = b"VOX1"
VOX2_MAGIC = b"VOX2"
VOX2_VERSION = 2
VOX2_VERSION_MIN = 1
VOX2_HEADER_SIZE = 32
VOX1_HEADER_SIZE = 16
VOX2_FLAG_RLE = 1
VOX2_FLAG_ZSTD = 2
VOX2_BRICK = 4
VOX2_BRICK_CELLS = VOX2_BRICK ** 3
VOX_SIZE_MAX = 1024
VOX_PALETTE_MAX = 256
MATERIAL_SOLID = 1

# Last palette published by encode/decode: shape (count, 4) RGBA bytes where A is material.
LAST_PALETTE = np.zeros((0, 4), dtype=np.uint8)

PathLike = Union[str, Path]


@dataclass
class VoxelGrid:
    """Dense material grid: data shape is (size_z, size_y, size_x)."""

    data: np.ndarray  # uint8, shape (Z, Y, X)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data, dtype=np.uint8)
        if arr.ndim != 3:
            raise ValueError(f"VoxelGrid data must be 3D (Z,Y,X), got shape {arr.shape}")
        self.data = np.ascontiguousarray(arr)

    @property
    def size_x(self) -> int:
        return int(self.data.shape[2])

    @property
    def size_y(self) -> int:
        return int(self.data.shape[1])

    @property
    def size_z(self) -> int:
        return int(self.data.shape[0])

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.size_x, self.size_y, self.size_z

    def count_solid(self) -> int:
        return int(np.count_nonzero(self.data))

    @classmethod
    def empty(cls, size_x: int, size_y: int, size_z: int) -> "VoxelGrid":
        return cls(np.zeros((size_z, size_y, size_x), dtype=np.uint8))

    @classmethod
    def from_coords(
        cls,
        coords: np.ndarray,
        materials: Optional[np.ndarray] = None,
        size: Optional[Sequence[int]] = None,
        origin_min: Optional[Sequence[int]] = None,
        default_material: int = MATERIAL_SOLID,
        pad: int = 0,
    ) -> "VoxelGrid":
        """Build a tight dense grid from sparse (N,3) integer (x,y,z) coords."""
        coords = np.asarray(coords, dtype=np.int64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must be (N,3), got {coords.shape}")
        if coords.shape[0] == 0:
            raise ValueError("coords is empty")

        if origin_min is None:
            mins = coords.min(axis=0) - int(pad)
        else:
            mins = np.asarray(origin_min, dtype=np.int64) - int(pad)
        shifted = coords - mins[None, :]

        if size is None:
            maxs = shifted.max(axis=0)
            size_xyz = (maxs + 1 + int(pad)).astype(np.int64)
        else:
            size_xyz = np.asarray(size, dtype=np.int64)
            if size_xyz.shape != (3,):
                raise ValueError("size must be a sequence of 3 ints (x,y,z)")

        for dim, name in zip(size_xyz, "xyz"):
            if dim <= 0 or dim > VOX_SIZE_MAX:
                raise ValueError(f"invalid size_{name}={dim} (max {VOX_SIZE_MAX})")

        grid = np.zeros((int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0])), dtype=np.uint8)
        xs, ys, zs = shifted[:, 0], shifted[:, 1], shifted[:, 2]
        mask = (
            (xs >= 0) & (ys >= 0) & (zs >= 0)
            & (xs < size_xyz[0]) & (ys < size_xyz[1]) & (zs < size_xyz[2])
        )
        if materials is None:
            vals = np.full(int(mask.sum()), default_material, dtype=np.uint8)
        else:
            materials = np.asarray(materials, dtype=np.uint8).reshape(-1)
            if materials.shape[0] != coords.shape[0]:
                raise ValueError("materials length must match coords")
            vals = materials[mask]
        grid[zs[mask], ys[mask], xs[mask]] = vals
        return cls(grid)

    def crop_to_solid(self, pad: int = 0) -> "VoxelGrid":
        solid = np.argwhere(self.data != 0)
        if solid.size == 0:
            return VoxelGrid.empty(1, 1, 1)
        z0, y0, x0 = solid.min(axis=0)
        z1, y1, x1 = solid.max(axis=0) + 1
        z0 = max(0, int(z0) - pad)
        y0 = max(0, int(y0) - pad)
        x0 = max(0, int(x0) - pad)
        z1 = min(self.size_z, int(z1) + pad)
        y1 = min(self.size_y, int(y1) + pad)
        x1 = min(self.size_x, int(x1) + pad)
        return VoxelGrid(self.data[z0:z1, y0:y1, x0:x1].copy())

    def downsample(
        self,
        target_max: int = 256,
        *,
        device: Optional[str] = None,
    ) -> "VoxelGrid":
        """Fit this grid into a box whose longest side is ≤ ``target_max``.

        Power-of-two integer factors use a CUDA-accelerated block majority vote
        (torch). Non-integer scales fall back to stratified sparse voting.
        No-op when already small enough. Air (0) never wins a block vote.
        """
        return downsample_grid(self, target_max=target_max, device=device)


def _torch_device(requested: Optional[str] = None):
    """Return (torch_module, device_str) or (None, None)."""
    try:
        import torch
    except ImportError:
        return None, None
    if requested is None:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    return torch, requested


def _majority_vote_keys(
    keys: np.ndarray,
    materials: np.ndarray,
    *,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per unique key: material id that appears most often (air never wins).

    CUDA path via torch scatter_add; CPU uses a sorted-lexicographic vote.
    """
    keys = np.asarray(keys, dtype=np.int64)
    mats = np.asarray(materials, dtype=np.uint8).reshape(-1)
    n = keys.shape[0]
    if n == 0:
        return keys, mats

    torch_mod, dev = _torch_device(device)
    # Prefer CUDA only when there's enough data to amortize H2D.
    use_torch = torch_mod is not None and (dev != "cpu" or n >= 50_000)
    if use_torch:
        keys_t = torch_mod.as_tensor(keys, device=dev)
        mats_t = torch_mod.as_tensor(mats.astype(np.int16), device=dev)
        keys_s, order = torch_mod.sort(keys_t)
        mats_s = mats_t[order]
        boundary = torch_mod.ones(n, dtype=torch_mod.bool, device=dev)
        boundary[1:] = keys_s[1:] != keys_s[:-1]
        group_id = torch_mod.cumsum(boundary.to(torch_mod.int32), dim=0) - 1
        n_groups = int(group_id[-1].item()) + 1
        unique_keys = keys_s[boundary]
        solid = mats_s > 0
        idx = group_id.to(torch_mod.int64) * 256 + mats_s.clamp(0, 255).to(torch_mod.int64)
        counts = torch_mod.zeros(n_groups * 256, dtype=torch_mod.int32, device=dev)
        counts.scatter_add_(0, idx, solid.to(torch_mod.int32))
        counts = counts.view(n_groups, 256)
        counts[:, 0] = -1
        winners = counts.argmax(dim=1).to(torch_mod.uint8)
        has = counts[:, 1:].amax(dim=1) > 0
        winners = torch_mod.where(has, winners, torch_mod.zeros_like(winners))
        # sync once
        if dev.startswith("cuda"):
            torch_mod.cuda.synchronize()
        return unique_keys.detach().cpu().numpy(), winners.detach().cpu().numpy()

    # CPU: compound sort by (key, mat) then pick mode per key with counts.
    order = np.lexsort((mats, keys))
    keys_s = keys[order]
    mats_s = mats[order]
    # group edges on either key change or mat change
    same_key = np.r_[False, keys_s[1:] == keys_s[:-1]]
    same_mat = np.r_[False, mats_s[1:] == mats_s[:-1]]
    run_start = ~(same_key & same_mat)
    run_idx = np.flatnonzero(np.r_[run_start, True])
    run_keys = keys_s[run_idx[:-1]]
    run_mats = mats_s[run_idx[:-1]]
    run_counts = np.diff(run_idx)
    # zero-out air votes
    run_counts = run_counts.copy()
    run_counts[run_mats == 0] = 0
    # among runs of the same key, pick max count
    key_breaks = np.r_[True, run_keys[1:] != run_keys[:-1]]
    key_ids = np.cumsum(key_breaks) - 1
    n_keys = int(key_ids[-1]) + 1
    # for each key track best count / mat
    best_count = np.zeros(n_keys, dtype=np.int32)
    best_mat = np.zeros(n_keys, dtype=np.uint8)
    for i in range(run_keys.shape[0]):
        k = key_ids[i]
        c = run_counts[i]
        if c > best_count[k]:
            best_count[k] = c
            best_mat[k] = run_mats[i]
    unique_keys = run_keys[key_breaks]
    keep = best_count > 0
    return unique_keys[keep], best_mat[keep]


def downsample_sparse(
    coords: np.ndarray,
    materials: Optional[np.ndarray],
    *,
    factor: float,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Downsample sparse (N,3) coords by ``factor`` with material majority vote."""
    coords = np.asarray(coords, dtype=np.int64)[:, :3]
    if coords.shape[0] == 0:
        return coords, materials
    f = float(factor)
    if f <= 1.0 + 1e-9:
        return coords, None if materials is None else np.asarray(materials)

    mins = coords.min(axis=0)
    shifted = coords - mins[None, :]
    ds = np.floor(shifted.astype(np.float64) / f).astype(np.int64)
    if materials is None:
        mats = np.full(ds.shape[0], MATERIAL_SOLID, dtype=np.uint8)
    else:
        mats = np.asarray(materials, dtype=np.uint8).reshape(-1)
        if mats.shape[0] != ds.shape[0]:
            raise ValueError("materials length must match coords")

    keys = (ds[:, 0] + (ds[:, 1] << 20) + (ds[:, 2] << 40)).astype(np.int64)
    uk, um = _majority_vote_keys(keys, mats, device=device)
    x = (uk & ((1 << 20) - 1)).astype(np.int64)
    y = ((uk >> 20) & ((1 << 20) - 1)).astype(np.int64)
    z = (uk >> 40).astype(np.int64)
    return np.stack([x, y, z], axis=1), um


def _downsample_dense_integer(
    grid: VoxelGrid,
    factor: int,
    *,
    device: Optional[str] = None,
) -> VoxelGrid:
    """Integer-factor block majority. Fast path for 512→256 (factor=2).

    Uses torch CUDA for a single reshape + scatter_add over all f³ cells when
    CUDA is available; otherwise a vectorized NumPy path on solid voxels only.
    """
    f = int(factor)
    if f <= 1:
        return grid
    data = grid.data  # Z,Y,X
    sz, sy, sx = (int(v) for v in data.shape)
    pz = (f - sz % f) % f
    py = (f - sy % f) % f
    px = (f - sx % f) % f
    if pz or py or px:
        data = np.pad(data, ((0, pz), (0, py), (0, px)), mode="constant", constant_values=0)
    nz, ny, nx = data.shape
    oz, oy, ox = nz // f, ny // f, nx // f

    torch_mod, dev = _torch_device(device)
    if torch_mod is not None and str(dev).startswith("cuda"):
        # Dense GPU majority — ideal when the volume is mostly populated.
        t = torch_mod.as_tensor(np.ascontiguousarray(data), device=dev)
        blocks = (
            t.view(oz, f, oy, f, ox, f)
            .permute(0, 2, 4, 1, 3, 5)
            .contiguous()
            .view(oz * oy * ox, f * f * f)
            .to(torch_mod.int16)
        )
        n_blocks = blocks.shape[0]
        flat = blocks.reshape(-1).to(torch_mod.int64).clamp(0, 255)
        block_ids = torch_mod.arange(n_blocks, device=dev, dtype=torch_mod.int64).repeat_interleave(
            f * f * f
        )
        solid = flat > 0
        idx = block_ids * 256 + flat
        counts = torch_mod.zeros(n_blocks * 256, dtype=torch_mod.int32, device=dev)
        counts.scatter_add_(0, idx, solid.to(torch_mod.int32))
        counts = counts.view(n_blocks, 256)
        counts[:, 0] = -1
        winners = counts.argmax(dim=1).to(torch_mod.uint8)
        has = counts[:, 1:].amax(dim=1) > 0
        winners = torch_mod.where(has, winners, torch_mod.zeros_like(winners))
        torch_mod.cuda.synchronize()
        out = winners.view(oz, oy, ox).detach().cpu().numpy()
        return VoxelGrid(out).crop_to_solid()

    # Sparse CPU (or torch-cpu): only solid voxels — much faster for porous occupancy.
    z, y, x = np.where(data != 0)
    if z.size == 0:
        return VoxelGrid.empty(1, 1, 1)
    bx = (x // f).astype(np.int64)
    by = (y // f).astype(np.int64)
    bz = (z // f).astype(np.int64)
    mats = data[z, y, x].astype(np.uint8)
    keys = bx + (by << 20) + (bz << 40)
    uk, um = _majority_vote_keys(keys, mats, device="cpu" if device is None else device)
    ox_i = (uk & ((1 << 20) - 1)).astype(np.int64)
    oy_i = ((uk >> 20) & ((1 << 20) - 1)).astype(np.int64)
    oz_i = (uk >> 40).astype(np.int64)
    coords = np.stack([ox_i, oy_i, oz_i], axis=1)
    return VoxelGrid.from_coords(coords, materials=um).crop_to_solid()


def downsample_grid(
    grid: VoxelGrid,
    target_max: int = 256,
    *,
    device: Optional[str] = None,
) -> VoxelGrid:
    """Downsample so max(size_x, size_y, size_z) ≤ ``target_max``.

    Integer factors use a dense CUDA block majority when available, else a
    sparse solid-only vote. Non-integer factors use sparse float flooring.
    """
    if target_max <= 0:
        return grid
    m = max(grid.size_x, grid.size_y, grid.size_z)
    if m <= target_max:
        return grid

    factor = m / float(target_max)
    if abs(factor - round(factor)) < 1e-6:
        out = _downsample_dense_integer(grid, max(1, int(round(factor))), device=device)
    else:
        z, y, x = np.where(grid.data != 0)
        if z.size == 0:
            return VoxelGrid.empty(1, 1, 1)
        coords = np.stack([x, y, z], axis=1).astype(np.int64)
        mats = grid.data[z, y, x].astype(np.uint8)
        c2, m2 = downsample_sparse(coords, mats, factor=factor, device=device)
        if c2.shape[0] == 0:
            return VoxelGrid.empty(1, 1, 1)
        out = VoxelGrid.from_coords(c2, materials=m2)

    if max(out.size_x, out.size_y, out.size_z) > target_max:
        return downsample_grid(out, target_max=target_max, device=device)
    return out


def _bricks_on(size: int, brick: int = VOX2_BRICK) -> int:
    if size <= 0:
        return 0
    return (size + brick - 1) // brick


def _rle_encode(src: np.ndarray) -> bytes:
    src = np.asarray(src, dtype=np.uint8).reshape(-1)
    if src.size != VOX2_BRICK_CELLS:
        raise ValueError(f"brick must have {VOX2_BRICK_CELLS} cells, got {src.size}")
    out = bytearray()
    runs_at = len(out)
    out.append(0)
    i = 0
    n = src.size
    runs = 0
    while i < n:
        value = int(src[i])
        count = 1
        while i + count < n and count < 255 and int(src[i + count]) == value:
            count += 1
        out.append(count)
        out.append(value)
        runs += 1
        i += count
    if runs <= 0 or runs > 255:
        raise ValueError(f"invalid RLE run count {runs}")
    out[runs_at] = runs
    return bytes(out)


def _rle_decode(data: bytes, offset: int, length: int = VOX2_BRICK_CELLS) -> Tuple[np.ndarray, int]:
    if offset >= len(data):
        raise ValueError("truncated RLE brick")
    runs = data[offset]
    if runs <= 0:
        raise ValueError("invalid RLE run count")
    cursor = offset + 1
    out = np.empty(length, dtype=np.uint8)
    written = 0
    for _ in range(runs):
        if cursor + 2 > len(data):
            raise ValueError("truncated RLE run")
        count = data[cursor]
        value = data[cursor + 1]
        cursor += 2
        if count <= 0 or written + count > length:
            raise ValueError("RLE run overflows brick")
        out[written : written + count] = value
        written += count
    if written != length:
        raise ValueError(f"RLE decoded {written} cells, expected {length}")
    return out, cursor


def _extract_brick(
    grid: VoxelGrid,
    bx: int,
    by: int,
    bz: int,
    remap: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    x0 = bx * VOX2_BRICK
    y0 = by * VOX2_BRICK
    z0 = bz * VOX2_BRICK
    out = np.zeros(VOX2_BRICK_CELLS, dtype=np.uint8)
    occupied = False
    index = 0
    for lz in range(VOX2_BRICK):
        z = z0 + lz
        for ly in range(VOX2_BRICK):
            y = y0 + ly
            for lx in range(VOX2_BRICK):
                x = x0 + lx
                if 0 <= x < grid.size_x and 0 <= y < grid.size_y and 0 <= z < grid.size_z:
                    material = int(grid.data[z, y, x])
                else:
                    material = 0
                if remap is not None:
                    material = int(remap[material & 255])
                out[index] = material
                if material != 0:
                    occupied = True
                index += 1
    return out, occupied


def _scatter_brick(grid: VoxelGrid, bx: int, by: int, bz: int, src: np.ndarray) -> None:
    x0 = bx * VOX2_BRICK
    y0 = by * VOX2_BRICK
    z0 = bz * VOX2_BRICK
    index = 0
    for lz in range(VOX2_BRICK):
        z = z0 + lz
        for ly in range(VOX2_BRICK):
            y = y0 + ly
            for lx in range(VOX2_BRICK):
                x = x0 + lx
                if 0 <= x < grid.size_x and 0 <= y < grid.size_y and 0 <= z < grid.size_z:
                    grid.data[z, y, x] = src[index]
                index += 1


def encode_bricks(
    grid: VoxelGrid,
    remap: Optional[np.ndarray] = None,
) -> Tuple[bytes, int]:
    if grid.size_x > VOX_SIZE_MAX or grid.size_y > VOX_SIZE_MAX or grid.size_z > VOX_SIZE_MAX:
        raise ValueError("grid exceeds VOX_SIZE_MAX")
    bricks_x = _bricks_on(grid.size_x)
    bricks_y = _bricks_on(grid.size_y)
    bricks_z = _bricks_on(grid.size_z)
    if bricks_x > 256 or bricks_y > 256 or bricks_z > 256:
        raise ValueError("too many bricks along an axis (u8 brick coords)")

    if remap is not None:
        remap = np.asarray(remap, dtype=np.uint8).reshape(-1)
        if remap.shape[0] < VOX_PALETTE_MAX:
            full = np.zeros(VOX_PALETTE_MAX, dtype=np.uint8)
            full[: remap.shape[0]] = remap
            remap = full
        elif remap.shape[0] > VOX_PALETTE_MAX:
            remap = remap[:VOX_PALETTE_MAX].copy()

    chunks: list[bytes] = []
    brick_count = 0
    for bz in range(bricks_z):
        for by in range(bricks_y):
            for bx in range(bricks_x):
                scratch, occupied = _extract_brick(grid, bx, by, bz, remap=remap)
                if not occupied:
                    continue
                packed = _rle_encode(scratch)
                chunks.append(bytes((bx & 255, by & 255, bz & 255)) + packed)
                brick_count += 1
    return b"".join(chunks), brick_count


def _normalize_rgb_palette(palette: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if palette is None:
        return None
    pal = np.asarray(palette)
    if pal.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if pal.ndim == 1:
        if pal.shape[0] % 3 == 0:
            pal = pal.reshape(-1, 3)
        elif pal.shape[0] % 4 == 0:
            pal = pal.reshape(-1, 4)[:, :3]
        else:
            raise ValueError("palette must be (N,3), (N,4), or flat multiples of 3/4")
    if pal.ndim != 2 or pal.shape[1] < 3:
        raise ValueError("palette must be (N,3+) RGB values")
    rgb = pal[:, :3].astype(np.float64)
    if float(np.nanmax(np.abs(rgb))) <= 1.5:
        rgb = np.clip(np.rint(rgb * 255.0), 0, 255)
    else:
        rgb = np.clip(np.rint(rgb), 0, 255)
    return rgb.astype(np.uint8)


def build_palette_table(
    grid: VoxelGrid,
    palette: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build VOX2 palette rows and a 256-entry material->index remap.

    Returns
    -------
    table : (count, 4) uint8
        Rows are (r, g, b, material). Index 0 is always air.
    remap : (256,) uint8
        Maps old material bytes to palette indices for brick packing.
    """
    rgb_pal = _normalize_rgb_palette(palette)
    used = np.unique(grid.data.reshape(-1))
    used = used[used != 0]
    if used.size > (VOX_PALETTE_MAX - 1):
        raise ValueError(
            f"too many distinct materials for VOX palette: {used.size} > {VOX_PALETTE_MAX - 1}"
        )

    remap = np.zeros(VOX_PALETTE_MAX, dtype=np.uint8)
    rows = [np.array((0, 0, 0, 0), dtype=np.uint8)]
    for material in used.tolist():
        material = int(material) & 255
        if material == 0 or remap[material] != 0:
            continue
        idx = len(rows)
        if idx >= VOX_PALETTE_MAX:
            raise ValueError("VOX palette overflow")
        if rgb_pal is not None and 1 <= material <= rgb_pal.shape[0]:
            r, g, b = (int(x) for x in rgb_pal[material - 1])
        elif rgb_pal is not None and 0 <= material < rgb_pal.shape[0]:
            # tolerate 0-based palettes if callers pass them that way
            r, g, b = (int(x) for x in rgb_pal[material])
        else:
            # deterministic grey fallback so missing colors stay readable
            tone = 64 + (material * 17) % 160
            r = g = b = int(tone)
        rows.append(np.array((r, g, b, material), dtype=np.uint8))
        remap[material] = np.uint8(idx)

    table = np.stack(rows, axis=0).astype(np.uint8)
    return table, remap


def encode_palette_prefix(table: np.ndarray) -> bytes:
    table = np.asarray(table, dtype=np.uint8).reshape(-1, 4)
    count = int(table.shape[0])
    if count < 1 or count > VOX_PALETTE_MAX:
        raise ValueError(f"invalid palette_count {count}")
    return struct.pack("<I", count) + table.tobytes(order="C")


def encode(
    grid: VoxelGrid,
    use_zstd: bool = True,
    zstd_level: int = 3,
    palette: Optional[np.ndarray] = None,
    *,
    include_palette: bool = True,
) -> bytes:
    """Encode a VOX2 image.

    Parameters
    ----------
    palette:
        Optional RGB/RGBA table. When material ids are already 1..K into this
        table (as produced by materials_from_colors), those colors are embedded.
    include_palette:
        Version-2 files always embed a palette prefix (default). Set False only
        to emit legacy version-1 payloads for debugging.
    """
    global LAST_PALETTE
    if grid.size_x <= 0 or grid.size_y <= 0 or grid.size_z <= 0:
        raise ValueError("grid sizes must be positive")

    if include_palette:
        table, remap = build_palette_table(grid, palette=palette)
        bricks, brick_count = encode_bricks(grid, remap=remap)
        plain = encode_palette_prefix(table) + bricks
        version = VOX2_VERSION
        LAST_PALETTE = table.copy()
    else:
        plain, brick_count = encode_bricks(grid, remap=None)
        version = VOX2_VERSION_MIN
        LAST_PALETTE = np.zeros((0, 4), dtype=np.uint8)

    flags = VOX2_FLAG_RLE
    payload = plain
    if use_zstd and plain and zstd is not None:
        cctx = zstd.ZstdCompressor(level=zstd_level, write_content_size=True)
        packed = cctx.compress(plain)
        if 0 < len(packed) < len(plain):
            payload = packed
            flags |= VOX2_FLAG_ZSTD
    header = struct.pack(
        "<4sIIIIIII",
        VOX2_MAGIC,
        version,
        grid.size_x,
        grid.size_y,
        grid.size_z,
        VOX2_BRICK,
        flags,
        brick_count,
    )
    return header + payload


def decode(data: bytes) -> VoxelGrid:
    if len(data) < 4:
        raise ValueError("file too short")
    magic = data[:4]
    if magic == VOX2_MAGIC:
        return _decode_v2(data)
    if magic == VOX1_MAGIC:
        return _decode_v1(data)
    raise ValueError(f"unknown vox magic {magic!r}")


def _decode_v1(data: bytes) -> VoxelGrid:
    if len(data) < VOX1_HEADER_SIZE:
        raise ValueError("truncated VOX1 header")
    _, size = struct.unpack_from("<4sI", data, 0)
    if size <= 0 or size > VOX_SIZE_MAX:
        raise ValueError(f"invalid VOX1 size {size}")
    cells = size * size * size
    if len(data) != VOX1_HEADER_SIZE + cells:
        raise ValueError("VOX1 size mismatch")
    dense = np.frombuffer(data, dtype=np.uint8, count=cells, offset=VOX1_HEADER_SIZE).copy()
    return VoxelGrid(dense.reshape((size, size, size)))


def _read_palette_prefix(payload: bytes) -> Tuple[np.ndarray, int]:
    """Return (palette_table (count,4), brick_offset)."""
    if len(payload) < 4:
        raise ValueError("truncated palette prefix")
    (count,) = struct.unpack_from("<I", payload, 0)
    if count < 1 or count > VOX_PALETTE_MAX:
        raise ValueError(f"invalid palette_count {count}")
    need = 4 + count * 4
    if len(payload) < need:
        raise ValueError("truncated palette bytes")
    table = np.frombuffer(payload, dtype=np.uint8, count=count * 4, offset=4).copy()
    return table.reshape(count, 4), need


def _expand_palette_indices(grid: VoxelGrid, table: np.ndarray) -> None:
    """Replace palette indices in-place with their material bytes."""
    table = np.asarray(table, dtype=np.uint8).reshape(-1, 4)
    if table.shape[0] == 0:
        return
    materials = table[:, 3]
    data = grid.data
    idx = data.astype(np.int32, copy=False)
    valid = (idx > 0) & (idx < table.shape[0])
    out = np.zeros_like(data)
    out[valid] = materials[idx[valid]]
    grid.data[:] = out


def decode_ex(data: bytes) -> Tuple[VoxelGrid, np.ndarray]:
    """Decode a VOX image and return (grid, palette_rgba).

    palette_rgba has shape (count, 4) with rows (r,g,b,material).
    Version-1 files return an empty palette table.
    """
    global LAST_PALETTE
    if len(data) < 4:
        raise ValueError("file too short")
    magic = data[:4]
    if magic == VOX1_MAGIC:
        grid = _decode_v1(data)
        LAST_PALETTE = np.zeros((0, 4), dtype=np.uint8)
        return grid, LAST_PALETTE.copy()
    if magic != VOX2_MAGIC:
        raise ValueError(f"unknown vox magic {magic!r}")

    if len(data) < VOX2_HEADER_SIZE:
        raise ValueError("truncated VOX2 header")
    magic, version, size_x, size_y, size_z, brick, flags, brick_count = struct.unpack_from(
        "<4sIIIIIII", data, 0
    )
    if magic != VOX2_MAGIC or version < VOX2_VERSION_MIN or version > VOX2_VERSION:
        raise ValueError("unsupported VOX2 header")
    if brick != VOX2_BRICK:
        raise ValueError(f"unsupported brick size {brick}")
    if (flags & VOX2_FLAG_RLE) == 0:
        raise ValueError("VOX2 payload must be RLE")
    for name, value in (("x", size_x), ("y", size_y), ("z", size_z)):
        if value <= 0 or value > VOX_SIZE_MAX:
            raise ValueError(f"invalid size_{name}={value}")
    if brick_count < 0:
        raise ValueError("negative brick_count")

    frame = data[VOX2_HEADER_SIZE:]
    if flags & VOX2_FLAG_ZSTD:
        if zstd is None:
            raise RuntimeError("zstandard package required to read zstd VOX2 files")
        if not frame:
            raise ValueError("empty zstd frame")
        dctx = zstd.ZstdDecompressor()
        payload = dctx.decompress(frame)
    else:
        payload = frame

    has_palette = version >= VOX2_VERSION
    cursor = 0
    table = np.zeros((0, 4), dtype=np.uint8)
    if has_palette:
        table, cursor = _read_palette_prefix(payload)

    grid = VoxelGrid.empty(size_x, size_y, size_z)
    bricks_x = _bricks_on(size_x)
    bricks_y = _bricks_on(size_y)
    bricks_z = _bricks_on(size_z)
    for _ in range(brick_count):
        if cursor + 4 > len(payload):
            raise ValueError("truncated brick record")
        bx, by, bz = payload[cursor], payload[cursor + 1], payload[cursor + 2]
        if bx >= bricks_x or by >= bricks_y or bz >= bricks_z:
            raise ValueError(f"brick coord out of range {(bx, by, bz)}")
        scratch, cursor = _rle_decode(payload, cursor + 3)
        _scatter_brick(grid, bx, by, bz, scratch)
    if cursor != len(payload):
        raise ValueError(f"payload trailing bytes: {len(payload) - cursor}")

    if has_palette:
        # Viewer keeps indices+palette; this project expands materials back.
        _expand_palette_indices(grid, table)

    LAST_PALETTE = table.copy()
    return grid, table.copy()


def _decode_v2(data: bytes) -> VoxelGrid:
    grid, _palette = decode_ex(data)
    return grid


def swap_yz(grid: VoxelGrid) -> VoxelGrid:
    """Swap Y and Z (Z-up TRELLIS → Y-up VOX). Data layout stays (Z,Y,X).

    Maps voxel coords ``(x, y, z)_zup → (x, z, y)_yup`` so vertical extent
    moves from Z into Y. Self-inverse: calling twice restores the original.
    """
    # data axes: 0=Z, 1=Y, 2=X → (Y, Z, X); sizes become (sx, sz, sy)
    return VoxelGrid(np.ascontiguousarray(np.transpose(grid.data, (1, 0, 2))))


# back-compat alias (older wrong mapping)
def swap_xy(grid: VoxelGrid) -> VoxelGrid:
    return VoxelGrid(np.ascontiguousarray(np.swapaxes(grid.data, 1, 2)))


def write(
    path: PathLike,
    grid: VoxelGrid,
    use_zstd: bool = True,
    zstd_level: int = 3,
    palette: Optional[np.ndarray] = None,
) -> int:
    # VOX is Y-up; TRELLIS occupancy is Z-up → swap Y/Z on export.
    raw = encode(
        swap_yz(grid),
        use_zstd=use_zstd,
        zstd_level=zstd_level,
        palette=palette,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return len(raw)


def read(path: PathLike) -> VoxelGrid:
    # undo write-time Y/Z swap so in-memory orientation matches TRELLIS (Z-up)
    return swap_yz(decode(Path(path).read_bytes()))


def read_ex(path: PathLike) -> Tuple[VoxelGrid, np.ndarray]:
    """Like read but also returns the embedded palette table."""
    grid, palette = decode_ex(Path(path).read_bytes())
    return swap_yz(grid), palette


def _rgb_u8(colors: np.ndarray) -> np.ndarray:
    """Normalize (N,3+) float/uint colors to uint8 RGB."""
    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[1] < 3:
        raise ValueError("colors must be (N,3+)")
    rgb = colors[:, :3].astype(np.float64)
    if float(np.nanmax(np.abs(rgb))) > 16.0:
        return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _nearest_palette_ids(rgb_u8: np.ndarray, palette: np.ndarray, chunk: int = 65536) -> np.ndarray:
    """Assign each RGB row to nearest palette entry (0-based indices)."""
    rgb = np.asarray(rgb_u8, dtype=np.int16).reshape(-1, 3)
    pal = np.asarray(palette, dtype=np.int16).reshape(-1, 3)
    if pal.shape[0] == 0:
        raise ValueError("empty palette")
    out = np.empty(rgb.shape[0], dtype=np.int32)
    # batch to keep memory bounded: |rgb|*|pal|*4 bytes
    for i0 in range(0, rgb.shape[0], chunk):
        i1 = min(i0 + chunk, rgb.shape[0])
        d = rgb[i0:i1, None, :] - pal[None, :, :]
        out[i0:i1] = np.einsum("ijk,ijk->ij", d, d).argmin(axis=1)
    return out


def extract_image_palette(
    image,
    max_colors: int = 255,
    *,
    bg_luma: float = 16.0,
    alpha_threshold: float = 0.1,
    sample_cap: int = 200_000,
) -> np.ndarray:
    """Build an RGB palette from foreground pixels of the source image.

    Uses adaptive (median-cut) quantization so roof/wood/cloth hues stay faithful
    instead of being derived from mis-projected / 4-bit-binned voxel samples.
    """
    from PIL import Image as _Image

    if isinstance(image, _Image.Image):
        has_a = image.mode in ("RGBA", "LA") or ("A" in image.getbands())
        rgba = np.array(image.convert("RGBA") if has_a else image.convert("RGB"))
    else:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError("image must be HxWxC")
        if arr.shape[2] >= 4:
            rgba = arr[..., :4]
            if rgba.dtype != np.uint8:
                if float(np.nanmax(np.abs(rgba))) <= 1.5:
                    rgba = np.clip(np.rint(rgba * 255.0), 0, 255).astype(np.uint8)
                else:
                    rgba = np.clip(np.rint(rgba), 0, 255).astype(np.uint8)
        else:
            rgb = arr[..., :3]
            if rgb.dtype != np.uint8:
                if float(np.nanmax(np.abs(rgb))) <= 1.5:
                    rgb = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
                else:
                    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
            alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            rgba = np.concatenate([rgb, alpha], axis=-1)

    rgb = rgba[..., :3].astype(np.uint8)
    if rgba.shape[-1] >= 4:
        a = rgba[..., 3].astype(np.float64)
        if float(a.max()) <= 1.5:
            a = a * 255.0
        fg = a > (alpha_threshold * 255.0)
    else:
        fg = np.ones(rgb.shape[:2], dtype=bool)
    # Drop near-black backdrop even when alpha is opaque (rembg fill).
    luma = (
        0.2126 * rgb[..., 0].astype(np.float32)
        + 0.7152 * rgb[..., 1].astype(np.float32)
        + 0.0722 * rgb[..., 2].astype(np.float32)
    )
    fg &= luma >= float(bg_luma)
    samples = rgb[fg]
    if samples.size == 0:
        # fall back to any non-pure-black pixel, else grey
        samples = rgb.reshape(-1, 3)
        samples = samples[samples.max(axis=1) > 0]
    if samples.size == 0:
        return np.array([[160, 140, 90]], dtype=np.uint8)

    if samples.shape[0] > sample_cap:
        rng = np.random.default_rng(0)
        samples = samples[rng.choice(samples.shape[0], size=sample_cap, replace=False)]

    k = int(max(1, min(max_colors, 255)))
    # Exact uniques first — often enough for cartoon-ish art.
    keys = (
        samples[:, 0].astype(np.int32) << 16
        | samples[:, 1].astype(np.int32) << 8
        | samples[:, 2].astype(np.int32)
    )
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    if uniq.size <= k:
        pal = np.stack(
            [(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255],
            axis=1,
        ).astype(np.uint8)
        order = np.argsort(counts)[::-1]
        return pal[order]

    # Adaptive median-cut via Pillow (uses the real image FG, not voxel samples).
    # Build a small RGB image so quantize sees correct population.
    # Tile samples into a roughly square bitmap.
    n = samples.shape[0]
    side = int(np.ceil(np.sqrt(n)))
    canvas = np.zeros((side * side, 3), dtype=np.uint8)
    canvas[:n] = samples
    im = _Image.fromarray(canvas.reshape(side, side, 3), mode="RGB")
    q = im.quantize(colors=k, method=_Image.Quantize.MEDIANCUT, dither=_Image.Dither.NONE)
    pal = np.array(q.getpalette()[: k * 3], dtype=np.uint8).reshape(-1, 3)
    # Drop pure-black entries that sometimes survive
    keep = pal.max(axis=1) >= int(bg_luma)
    if np.any(keep):
        pal = pal[keep]
    if pal.shape[0] == 0:
        pal = samples[np.argsort(counts[: samples.shape[0]])[::-1][:k]] if False else samples[:k]
        # frequency-weighted representatives
        top = np.argsort(counts)[::-1][:k]
        pal = np.stack(
            [(uniq[top] >> 16) & 255, (uniq[top] >> 8) & 255, uniq[top] & 255],
            axis=1,
        ).astype(np.uint8)
    return pal[:k].astype(np.uint8)


def materials_from_colors(
    colors: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    alpha_threshold: float = 0.1,
    max_colors: int = 255,
    palette: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize RGB colors to material ids 1..K (0 = air).

    If ``palette`` is provided, assign each solid sample to the nearest entry.
    Otherwise build a palette from the samples (median-cut / unique).
    """
    rgb_u8 = _rgb_u8(colors)

    if alpha is None and np.asarray(colors).ndim == 2 and np.asarray(colors).shape[1] >= 4:
        alpha = np.asarray(colors)[:, 3]
    solid = np.ones(rgb_u8.shape[0], dtype=bool)
    if alpha is not None:
        a = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if float(np.nanmax(np.abs(a))) > 16.0:
            a = a / 255.0
        solid = np.clip(a, 0.0, 1.0) > alpha_threshold

    materials = np.zeros(rgb_u8.shape[0], dtype=np.uint8)
    if not np.any(solid):
        return materials, np.zeros((0, 3), dtype=np.uint8)

    solid_rgb = rgb_u8[solid]
    if palette is None:
        # Quantize sample set itself (no 4-bit crush).
        from PIL import Image as _Image

        k = int(max(1, min(max_colors, 255)))
        keys = (
            solid_rgb[:, 0].astype(np.int32) << 16
            | solid_rgb[:, 1].astype(np.int32) << 8
            | solid_rgb[:, 2].astype(np.int32)
        )
        uniq, inverse = np.unique(keys, return_inverse=True)
        if uniq.size <= k:
            palette = np.stack(
                [(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1
            ).astype(np.uint8)
            # Refine palette to mean of members
            for i in range(palette.shape[0]):
                sel = solid_rgb[inverse == i]
                if sel.size:
                    palette[i] = np.clip(np.rint(sel.mean(axis=0)), 0, 255).astype(np.uint8)
            materials[solid] = (inverse + 1).astype(np.uint8)
            return materials, palette

        n = solid_rgb.shape[0]
        side = int(np.ceil(np.sqrt(n)))
        canvas = np.zeros((side * side, 3), dtype=np.uint8)
        canvas[:n] = solid_rgb
        im = _Image.fromarray(canvas.reshape(side, side, 3), mode="RGB")
        q = im.quantize(colors=k, method=_Image.Quantize.MEDIANCUT, dither=_Image.Dither.NONE)
        palette = np.array(q.getpalette()[: k * 3], dtype=np.uint8).reshape(-1, 3)
        # only used entries
        idx_img = np.array(q, dtype=np.int32).reshape(-1)[:n]
        used = np.unique(idx_img)
        palette = palette[used]
        remap = np.full(k, -1, dtype=np.int32)
        remap[used] = np.arange(used.shape[0], dtype=np.int32)
        materials[solid] = (remap[idx_img] + 1).astype(np.uint8)
        return materials, palette

    palette = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    ids = _nearest_palette_ids(solid_rgb, palette)
    materials[solid] = (ids + 1).astype(np.uint8)
    return materials, palette


def _load_image_rgb_alpha(image, *, alpha_threshold: float = 0.1, bg_luma: float = 16.0):
    """Return (rgb_float01 HxWx3, fg_mask HxW, rgb_u8 HxWx3)."""
    from PIL import Image as _Image

    if isinstance(image, _Image.Image):
        has_a = image.mode in ("RGBA", "LA") or ("A" in image.getbands())
        if has_a:
            rgba = np.array(image.convert("RGBA"), dtype=np.uint8)
            rgb_u8 = rgba[..., :3]
            alpha_u8 = rgba[..., 3].astype(np.float32)
        else:
            rgb_u8 = np.array(image.convert("RGB"), dtype=np.uint8)
            alpha_u8 = np.full(rgb_u8.shape[:2], 255.0, dtype=np.float32)
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("image must be HxWx3[+]")
        if arr.dtype != np.uint8:
            if float(np.nanmax(np.abs(arr[..., :3]))) <= 1.5:
                rgb_u8 = np.clip(np.rint(arr[..., :3] * 255.0), 0, 255).astype(np.uint8)
            else:
                rgb_u8 = np.clip(np.rint(arr[..., :3]), 0, 255).astype(np.uint8)
        else:
            rgb_u8 = arr[..., :3]
        if arr.shape[2] >= 4:
            alpha_u8 = arr[..., 3].astype(np.float32)
            if float(np.nanmax(alpha_u8)) <= 1.5:
                alpha_u8 = alpha_u8 * 255.0
        else:
            alpha_u8 = np.full(rgb_u8.shape[:2], 255.0, dtype=np.float32)

    luma = (
        0.2126 * rgb_u8[..., 0].astype(np.float32)
        + 0.7152 * rgb_u8[..., 1].astype(np.float32)
        + 0.0722 * rgb_u8[..., 2].astype(np.float32)
    )
    fg = (alpha_u8 >= (alpha_threshold * 255.0)) & (luma >= float(bg_luma))
    img = rgb_u8.astype(np.float32) / 255.0
    return img, fg, rgb_u8


def _project_uv(
    u: np.ndarray,
    v: np.ndarray,
    *,
    umin: float,
    umax: float,
    vmin: float,
    vmax: float,
    ix0: int,
    iy0: int,
    ix1: int,
    iy1: int,
    w: int,
    h: int,
    flip_u: bool,
    flip_v: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map voxel UV → integer image samples with aspect-preserving fit."""
    du = max(umax - umin, 1e-6)
    dv = max(vmax - vmin, 1e-6)
    img_w = float(ix1 - ix0)
    img_h = float(iy1 - iy0)
    voxel_aspect = du / dv
    img_aspect = img_w / max(img_h, 1e-6)
    if voxel_aspect > img_aspect:
        used_w = img_w
        used_h = img_w / voxel_aspect
    else:
        used_h = img_h
        used_w = img_h * voxel_aspect
    ox = ix0 + 0.5 * (img_w - used_w)
    oy = iy0 + 0.5 * (img_h - used_h)

    uu = (u - umin) / du
    vv = (v - vmin) / dv
    if flip_u:
        uu = 1.0 - uu
    if flip_v:
        vv = 1.0 - vv
    xs = np.clip(np.rint(ox + uu * used_w).astype(np.int64), 0, w - 1)
    # Image row 0 is top; voxel V grows up → flip within the fitted slot.
    ys = np.clip(np.rint(oy + used_h - vv * used_h).astype(np.int64), 0, h - 1)
    return xs, ys


def project_image_colors(
    coords: np.ndarray,
    image,
    axis: str = "xy",
    percentile: float = 0.5,
    *,
    bg_luma: float = 16.0,
    alpha_threshold: float = 0.1,
    fill_background: bool = True,
    depth_fill: bool = True,
    flip_u: Optional[bool] = None,
    flip_v: Optional[bool] = None,
    front_is_max: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Orthographic-project voxel coords onto an image → colors in [0,1].

    UV is fitted from the voxel plane bbox → image **foreground** bbox (ignores
    black letterbox), aspect-preserved. When ``flip_u`` / ``flip_v`` are None,
    both flips are tried and the pair that lands the most voxels on image
    foreground is kept (fixes left/right and occasional up/down mismatches).

    With ``depth_fill`` (default), each camera ray keeps the front-face sample
    color for the whole column so interior voxels don't pick edge hues.

    Returns ``(colors (N,3) float32, fg_mask (N,) bool)``.
    """
    img, fg_img, _rgb_u8 = _load_image_rgb_alpha(
        image, alpha_threshold=alpha_threshold, bg_luma=bg_luma
    )
    h, w = img.shape[:2]
    c = np.asarray(coords, dtype=np.float64)[:, :3]
    if axis == "xy":
        u, v, depth = c[:, 0].copy(), c[:, 1].copy(), c[:, 2].copy()
        default_front_max = True
    elif axis == "xz":
        u, v, depth = c[:, 0].copy(), c[:, 2].copy(), c[:, 1].copy()
        default_front_max = True
    elif axis in ("zy", "yz"):
        u, v, depth = c[:, 2].copy(), c[:, 1].copy(), c[:, 0].copy()
        default_front_max = True
    else:
        raise ValueError(f"unknown projection axis {axis!r}")
    if front_is_max is None:
        front_is_max = default_front_max

    # Image FG bbox (tight crop of the painted subject).
    if np.any(fg_img):
        fy, fx = np.where(fg_img)
        ix0, ix1 = int(fx.min()), int(fx.max())
        iy0, iy1 = int(fy.min()), int(fy.max())
        pad_x = max(1, int(0.01 * (ix1 - ix0 + 1)))
        pad_y = max(1, int(0.01 * (iy1 - iy0 + 1)))
        ix0 = max(0, ix0 - pad_x)
        iy0 = max(0, iy0 - pad_y)
        ix1 = min(w - 1, ix1 + pad_x)
        iy1 = min(h - 1, iy1 + pad_y)
    else:
        ix0, iy0, ix1, iy1 = 0, 0, w - 1, h - 1

    p = float(percentile)
    if p > 0.0 and u.size >= 16:
        umin, umax = np.percentile(u, [p, 100.0 - p])
        vmin, vmax = np.percentile(v, [p, 100.0 - p])
        umin, umax = float(umin), float(umax)
        vmin, vmax = float(vmin), float(vmax)
    else:
        umin, umax = float(u.min()), float(u.max())
        vmin, vmax = float(v.min()), float(v.max())
    if umax - umin < 1e-3:
        umin, umax = float(u.min()), float(u.max()) + 1e-3
    if vmax - vmin < 1e-3:
        vmin, vmax = float(v.min()), float(v.max()) + 1e-3

    # Choose UV flips that maximize FG hits (or use explicit flags).
    candidates: list[Tuple[bool, bool]] = []
    if flip_u is None and flip_v is None:
        candidates = [(fu, fv) for fu in (False, True) for fv in (False, True)]
    else:
        candidates = [
            (
                bool(flip_u) if flip_u is not None else False,
                bool(flip_v) if flip_v is not None else False,
            )
        ]

    best = None  # (score, xs, ys, fuacruz, fv)
    for fu, fv in candidates:
        xs_t, ys_t = _project_uv(
            u,
            v,
            umin=umin,
            umax=umax,
            vmin=vmin,
            vmax=vmax,
            ix0=ix0,
            iy0=iy0,
            ix1=ix1,
            iy1=iy1,
            w=w,
            h=h,
            flip_u=fu,
            flip_v=fv,
        )
        score = int(fg_img[ys_t, xs_t].sum())
        # Prefer brighter FG mean as a weak tie-break (avoids all-shadow fits).
        if score > 0:
            mean_luma = float(
                (
                    0.2126 * img[ys_t, xs_t, 0]
                    + 0.7152 * img[ys_t, xs_t, 1]
                    + 0.0722 * img[ys_t, xs_t, 2]
                )[fg_img[ys_t, xs_t]].mean()
            ) if np.any(fg_img[ys_t, xs_t]) else 0.0
        else:
            mean_luma = 0.0
        key = (score, mean_luma)
        if best is None or key > best[0]:
            best = (key, xs_t, ys_t, fu, fv)
    assert best is not None
    _, xs, ys, fu_used, fv_used = best

    colors = img[ys, xs].copy()
    fg_mask = fg_img[ys, xs]

    if fill_background and np.any(~fg_mask) and np.any(fg_img):
        try:
            from scipy import ndimage as ndi

            _, (ny, nx) = ndi.distance_transform_edt(~fg_img, return_indices=True)
            fill = ~fg_mask
            colors[fill] = img[ny[ys[fill], xs[fill]], nx[ys[fill], xs[fill]]]
            fg_mask = np.ones_like(fg_mask)
        except Exception:
            fy, fx = np.where(fg_img)
            if fy.size:
                step = max(1, fy.size // 20000)
                fy, fx = fy[::step], fx[::step]
                miss = np.flatnonzero(~fg_mask)
                for i0 in range(0, miss.size, 8192):
                    i1 = min(i0 + 8192, miss.size)
                    mi = miss[i0:i1]
                    dy = ys[mi, None].astype(np.int32) - fy[None, :].astype(np.int32)
                    dx = xs[mi, None].astype(np.int32) - fx[None, :].astype(np.int32)
                    j = (dy * dy + dx * dx).argmin(axis=1)
                    colors[mi] = img[fy[j], fx[j]]
                fg_mask = np.ones_like(fg_mask)

    if depth_fill and coords.shape[0] > 0:
        # Prefer the depth direction whose front face is brighter (exterior).
        key = ys.astype(np.int64) * int(w) + xs.astype(np.int64)

        def _depth_paint(front_max: bool) -> np.ndarray:
            secondary = (-depth) if front_max else depth
            order = np.lexsort((np.arange(key.size), secondary, key))
            key_s = key[order]
            first = np.r_[True, key_s[1:] != key_s[:-1]]
            group_of_sorted = np.cumsum(first) - 1
            front_of_group = order[first]
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            return colors[front_of_group[group_of_sorted[inv]]]

        c_max = _depth_paint(True)
        c_min = _depth_paint(False)
        colors = c_max if float(c_max.mean()) >= float(c_min.mean()) else c_min

    return colors.astype(np.float32), fg_mask


def _image_fg_rgb(color_image, *, alpha_threshold: float = 0.1, bg_luma: float = 16.0) -> np.ndarray:
    """Foreground RGB uint8 samples from a photo (for matching scores)."""
    _img, fg, rgb_u8 = _load_image_rgb_alpha(
        color_image, alpha_threshold=alpha_threshold, bg_luma=bg_luma
    )
    if np.any(fg):
        return rgb_u8[fg]
    return rgb_u8.reshape(-1, 3)


def _nn_dist_to_palette(rgb_u8: np.ndarray, palette: np.ndarray, sample_cap: int = 8000) -> float:
    rgb = np.asarray(rgb_u8, dtype=np.float32).reshape(-1, 3)
    pal = np.asarray(palette, dtype=np.float32).reshape(-1, 3)
    if rgb.size == 0 or pal.size == 0:
        return 1e9
    if rgb.shape[0] > sample_cap:
        rng = np.random.default_rng(0)
        rgb = rgb[rng.choice(rgb.shape[0], size=sample_cap, replace=False)]
    # chunked L2
    acc = 0.0
    n = 0
    chunk = 4096
    for i0 in range(0, rgb.shape[0], chunk):
        part = rgb[i0 : i0 + chunk]
        d = part[:, None, :] - pal[None, :, :]
        nn = np.einsum("ijk,ijk->ij", d, d).min(axis=1)
        acc += float(np.sqrt(nn).sum())
        n += part.shape[0]
    return acc / max(n, 1)


def _color_noise_score(rgb: np.ndarray) -> float:
    """Higher = more rainbow/noisy (local multiple-unique colours + high sat spread)."""
    rgb = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)
    if rgb.shape[0] < 32:
        return 0.0
    # subsample
    if rgb.shape[0] > 20000:
        rng = np.random.default_rng(0)
        rgb = rgb[rng.choice(rgb.shape[0], 20000, replace=False)]
    # unique after 5-bit crush
    q = (rgb / 8.0).astype(np.int32)
    keys = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    uniq = np.unique(keys).size
    uniq_frac = uniq / max(rgb.shape[0], 1)
    mx = rgb.max(axis=1)
    mn = rgb.min(axis=1)
    sat = (mx - mn).mean() / 255.0
    # noisy PBR tends to fill colour space: high unique_frac AND medium/high sat
    return float(uniq_frac * 2.0 + sat)


def _base_color_plausible(base_rgb, color_image=None, *, max_colors: int = 255) -> bool:
    """Reject flat OR rainbow-noise textures that don't track the input photo."""
    base = np.asarray(base_rgb, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] < 3 or base.shape[0] == 0:
        return False
    rgb = base[:, :3]
    # normalize to u8-ish 0..255
    if float(np.nanmax(np.abs(rgb))) <= 1.5:
        rgb_u8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    else:
        rgb_u8 = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    if float(np.std(rgb_u8.astype(np.float64))) < 1.0:
        return False
    noise = _color_noise_score(rgb_u8)
    # Fully random textures explode unique colours; real assets cluster.
    if noise > 1.35:
        return False
    if color_image is None:
        return True
    try:
        img_pal = extract_image_palette(color_image, max_colors=min(int(max_colors), 64))
        dist = _nn_dist_to_palette(rgb_u8, img_pal)
    except Exception:
        return noise <= 1.1
    # house-like assets typically land well under ~35; pure noise is 60+
    return dist < 42.0


def _auto_color_axis(coords: np.ndarray, color_image) -> str:
    """Pick orthographic axis that best matches a single photo to the volume.

    Score = palette match of depth-filled projection, with a strong bias to
    elevation axes (xz / zy). Top-down xy only wins if it is clearly better.
    """
    try:
        img_pal = extract_image_palette(color_image, max_colors=64)
    except Exception:
        img_pal = None
    best_axis = "xz"
    best_score = -1e18
    scores = {}
    for axis, bias in (("xz", 3.0), ("zy", 1.5), ("xy", 0.0)):
        try:
            cols, _fg = project_image_colors(
                coords,
                color_image,
                axis=axis,
                percentile=1.0,
                fill_background=True,
                depth_fill=True,
            )
        except Exception as exc:
            scores[axis] = f"err:{exc}"
            continue
        cols_u8 = np.clip(np.rint(np.asarray(cols, dtype=np.float32) * 255.0), 0, 255).astype(
            np.uint8
        )
        if img_pal is not None and cols_u8.size:
            dist = _nn_dist_to_palette(cols_u8, img_pal)
            # lower NN distance isbetter; add elevation bias in distance units
            score = -(float(dist) - bias)
        else:
            score = float(cols_u8.astype(np.float32).std(axis=0).mean()) + bias
        scores[axis] = score
        if score > best_score:
            best_score = score
            best_axis = axis
    print(f"[vox] axis scores={scores} -> {best_axis}", flush=True)
    return best_axis


def grid_from_mesh_with_voxel(
    mesh,
    material_mode: str = "image",
    alpha_threshold: float = 0.5,
    max_colors: int = 255,
    crop: bool = True,
    solid_material: int = MATERIAL_SOLID,
    color_image=None,
    color_axis: str = "auto",
) -> Tuple[VoxelGrid, Optional[np.ndarray]]:
    """Convert a TRELLIS MeshWithVoxel into a VOX2 VoxelGrid.

    material_mode:
      - "image": orthographic-project ``color_image`` onto occupancy (default)
      - "color": quantize decoded base_color / alpha attrs from TRELLIS
      - "solid": occupancy only
      - "auto": use mesh colour only when it tracks the photo; else image project
    """
    coords = mesh.coords
    if hasattr(coords, "detach"):
        coords = coords.detach().cpu().numpy()
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 3:
        raise ValueError(f"mesh.coords must be (N,3), got {getattr(coords, 'shape', None)}")
    coords = coords[:, :3].astype(np.int64)

    palette = None
    mode = (material_mode or "image").lower()

    def _from_mesh_color():
        attrs = getattr(mesh, "attrs", None)
        if attrs is None:
            raise ValueError("mesh has no attrs")
        if hasattr(attrs, "detach"):
            attrs = attrs.detach().cpu().numpy()
        attrs = np.asarray(attrs)
        if attrs.ndim != 2 or attrs.shape[0] != coords.shape[0] or attrs.shape[1] < 3:
            raise ValueError(f"mesh.attrs shape invalid: {getattr(attrs, 'shape', None)}")
        layout = getattr(mesh, "layout", None) or {
            "base_color": slice(0, 3),
            "alpha": slice(5, 6),
        }
        base = attrs[:, layout["base_color"]]
        alpha = attrs[:, layout["alpha"]] if isinstance(layout, dict) and "alpha" in layout else None
        if not _base_color_plausible(base, color_image, max_colors=max_colors):
            raise ValueError("mesh base_color looks flat/noisy or unmatched to image")
        mats, pal = materials_from_colors(
            base,
            alpha=alpha,
            # keep nearly-opaque surfaces; generative alpha is soft on edges
            alpha_threshold=min(float(alpha_threshold), 0.15) if alpha is not None else 0.0,
            max_colors=max_colors,
        )
        return mats, pal

    def _from_image_projection():
        if color_image is None:
            raise ValueError("material_mode='image' requires color_image=")
        if coords.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.uint8),
                np.zeros((0, 3), dtype=np.uint8),
            )
        axis = (color_axis or "auto").lower()
        if axis in ("", "auto"):
            axis = _auto_color_axis(coords, color_image)
            print(f"[vox] image projection axis={axis}", flush=True)
        else:
            print(f"[vox] image projection axis={axis} (forced)", flush=True)
        # Palette from the real image FG so roof/wood/cloth hues stay faithful.
        pal = extract_image_palette(
            color_image,
            max_colors=max_colors,
            alpha_threshold=min(float(alpha_threshold), 0.1),
        )
        base, _fg = project_image_colors(
            coords,
            color_image,
            axis=axis,
            percentile=1.0,
            alpha_threshold=min(float(alpha_threshold), 0.1),
            fill_background=True,
        )
        mats, pal = materials_from_colors(
            base,
            alpha=None,
            alpha_threshold=0.0,
            max_colors=max_colors,
            palette=pal,
        )
        return mats, pal

    if mode == "solid":
        materials = np.full(coords.shape[0], solid_material, dtype=np.uint8)
        palette = np.array([[180, 180, 180]], dtype=np.uint8)
    elif mode == "color":
        try:
            materials, palette = _from_mesh_color()
        except Exception as exc:
            if color_image is None:
                raise
            print(f"[vox] color mode failed ({exc}); falling back to image projection", flush=True)
            materials, palette = _from_image_projection()
        keep = materials != 0
        if np.any(~keep) and np.any(keep):
            coords = coords[keep]
            materials = materials[keep]
        if coords.shape[0] == 0:
            return VoxelGrid.empty(1, 1, 1), palette if palette is not None else np.zeros((0, 3), dtype=np.uint8)
    elif mode == "image":
        materials, palette = _from_image_projection()
        if coords.shape[0] == 0:
            return VoxelGrid.empty(1, 1, 1), np.zeros((0, 3), dtype=np.uint8)
    elif mode == "auto":
        try:
            materials, palette = _from_mesh_color()
            keep = materials != 0
            if np.any(keep):
                coords = coords[keep]
                materials = materials[keep]
            else:
                raise ValueError("color mode produced no solid voxels")
        except Exception as exc:
            print(f"[vox] auto->color failed ({exc}); using image projection", flush=True)
            materials, palette = _from_image_projection()
            if coords.shape[0] == 0:
                return VoxelGrid.empty(1, 1, 1), np.zeros((0, 3), dtype=np.uint8)
    else:
        raise ValueError(f"unknown material_mode {material_mode!r}")

    grid = VoxelGrid.from_coords(coords, materials=materials)
    if crop:
        grid = grid.crop_to_solid()
    return grid, palette


def write_palette_png(path: PathLike, palette: np.ndarray) -> None:
    """Dump material palette as a 1×K RGB strip."""
    from PIL import Image

    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if pal.size == 0:
        return
    img = Image.fromarray(pal.reshape(1, -1, 3), mode="RGB")
    img = img.resize((max(pal.shape[0] * 8, 8), 32), Image.Resampling.NEAREST)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


__all__ = [
    "VoxelGrid",
    "encode",
    "decode",
    "decode_ex",
    "read",
    "read_ex",
    "write",
    "swap_yz",
    "swap_xy",
    "build_palette_table",
    "encode_palette_prefix",
    "materials_from_colors",
    "extract_image_palette",
    "project_image_colors",
    "grid_from_mesh_with_voxel",
    "downsample_grid",
    "downsample_sparse",
    "write_palette_png",
    "LAST_PALETTE",
    "MATERIAL_SOLID",
    "VOX2_BRICK",
    "VOX2_VERSION",
    "VOX_PALETTE_MAX",
    "VOX_SIZE_MAX",
]
