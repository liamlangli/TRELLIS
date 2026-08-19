#!/usr/bin/env python3
"""CUDA GLB/mesh → VOX2 voxelizer (CuMesh BVH).

Uses the installed CUDA stack:
  trimesh  – load textured GLB
  cumesh   – CuMesh cleanup + cuBVH distance queries  (project "pycu"/CuMesh)
  vox_io   – palette + VOX2 encode

Usage:
    glb_to_vox.py input.glb [output.vox]
    glb2vox.bat a.glb b.vox

Env:
  OUT_RES=256           longest-axis voxels (or fit under this bound)
  VOX_FILL=1            1=solid fill (default), 0=surface shell only
  SURFACE_BAND=0.75     exterior shell thickness in voxel units
  MAX_COLORS=255        palette cap (VOX index 0 is air)
  SIMPLIFY_FACES=0      optional CuMesh face budget before voxelize (0=off)
  CHUNK=1_500_000       query batch size
  PAD_VOXELS=1          empty padding around bbox
  COLOR_MODE=texture    texture|vertex|solid
  DEVICE=cuda
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
    "python.exe" if os.name == "nt" else "python"
)


def _ensure_venv() -> None:
    if os.environ.get("TRELLIS_VENV_OK") == "1":
        return
    if not VENV_PY.is_file():
        return
    try:
        using = Path(sys.executable).resolve()
        target = VENV_PY.resolve()
    except OSError:
        return
    if using == target:
        os.environ["TRELLIS_VENV_OK"] = "1"
        return
    os.environ["TRELLIS_VENV_OK"] = "1"
    print(f"re-exec with project venv: {target}", flush=True)
    os.execv(str(target), [str(target), *sys.argv])


_ensure_venv()
sys.path.insert(0, str(ROOT))


def _usage() -> None:
    print(
        "Usage: glb_to_vox.py input.glb [output.vox]\n"
        "\n"
        "CUDA voxelizer: textured GLB → VOX2 via CuMesh BVH.\n"
        "Env: OUT_RES, VOX_FILL, SURFACE_BAND, MAX_COLORS, SIMPLIFY_FACES,\n"
        "     CHUNK, PAD_VOXELS, COLOR_MODE=texture|vertex|solid, DEVICE\n",
        file=sys.stderr,
    )


@dataclass
class MeshGPU:
    vertices: "torch.Tensor"  # (V,3) cuda float32
    faces: "torch.Tensor"  # (F,3) cuda int64
    uvs: Optional["torch.Tensor"]  # (V,2) cuda float32 or None
    vertex_colors: Optional["torch.Tensor"]  # (V,3) float01 or None
    texture: Optional["torch.Tensor"]  # (H,W,3/4) float01 cuda or None


def _require_cuda_stack():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for glb_to_vox (torch.cuda unavailable)")
    try:
        import cumesh  # noqa: F401
        from cumesh import CuMesh, cuBVH  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "cumesh (CuMesh) is required but failed to import.\n"
            f"  detail: {type(e).__name__}: {e}\n"
            "  build/install the project CUDA extensions first."
        ) from e
    try:
        import trimesh  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"trimesh is required: {e}") from e
    try:
        import vox_io  # noqa: F401
    except Exception as e:
        raise RuntimeError(f"vox_io is required: {e}") from e


def _as_float01_texture(img) -> np.ndarray:
    from PIL import Image

    if img is None:
        return None
    if hasattr(img, "convert"):
        arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
        return arr
    arr = np.asarray(img)
    if arr.dtype != np.float32 and arr.dtype != np.float64:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    else:
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.ones_like(arr)], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.ones(arr.shape[:2] + (1,), dtype=np.float32)
        arr = np.concatenate([arr, alpha], axis=-1)
    return arr


def load_textured_mesh(path: Path, device: str = "cuda") -> MeshGPU:
    """Load GLB/GLTF/OBJ/... into GPU tensors. Applies scene graph transforms."""
    import torch
    import trimesh

    loaded = trimesh.load(str(path), force=None, process=False)
    geoms = []
    if isinstance(loaded, trimesh.Scene):
        # dump with transforms
        for name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue
            # scene graph transform for this geometry
            tf = None
            try:
                # first node that references this geometry
                for node_name in loaded.graph.nodes_geometry:
                    geom_name = loaded.graph[node_name][1]
                    if geom_name == name:
                        tf, _ = loaded.graph.get(node_name)
                        break
            except Exception:
                tf = None
            g = geom.copy()
            if tf is not None:
                g.apply_transform(tf)
            geoms.append(g)
        if not geoms:
            # fallback
            dumped = loaded.dump(concatenate=True)
            if isinstance(dumped, trimesh.Trimesh):
                geoms = [dumped]
    elif isinstance(loaded, trimesh.Trimesh):
        geoms = [loaded]
    else:
        raise ValueError(f"unsupported mesh type: {type(loaded)}")

    if not geoms:
        raise ValueError(f"no triangle meshes found in {path}")

    # Prefer keep textures by processing one primary mesh when possible.
    # Concatenate geometry; textures only preserved if single textured geom.
    if len(geoms) == 1:
        mesh = geoms[0]
    else:
        mesh = trimesh.util.concatenate(geoms)

    if mesh.faces is None or len(mesh.faces) < 8:
        raise ValueError(f"mesh needs at least 8 triangles, got {0 if mesh.faces is None else len(mesh.faces)}")

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    # drop non-finite
    ok = np.isfinite(v).all(axis=1)
    if not ok.all():
        remap = -np.ones(len(v), dtype=np.int64)
        remap[ok] = np.arange(ok.sum())
        f = remap[f]
        good_f = (f >= 0).all(axis=1)
        f = f[good_f]
        v = v[ok]

    uvs = None
    texture = None
    vcol = None
    vis = getattr(mesh, "visual", None)
    if vis is not None:
        # vertex colors
        if hasattr(vis, "vertex_colors") and vis.vertex_colors is not None:
            vc = np.asarray(vis.vertex_colors)
            if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
                # map through ok-filter length: rebuild from remaining indices is hard;
                # sample after filter only if lengths still match
                pass
        if hasattr(vis, "uv") and vis.uv is not None:
            uv = np.asarray(vis.uv, dtype=np.float64)
            if uv.ndim == 2 and uv.shape[0] == np.asarray(mesh.vertices).shape[0]:
                # Align filter
                uv = uv[ok] if ok.shape[0] == uv.shape[0] else uv
                if uv.shape[0] == v.shape[0]:
                    uvs = uv[:, :2].astype(np.float32)
        mat = getattr(vis, "material", None)
        tex_img = None
        if mat is not None:
            tex_img = getattr(mat, "baseColorTexture", None)
            if tex_img is None:
                tex_img = getattr(mat, "image", None)
        if tex_img is not None:
            texture = _as_float01_texture(tex_img)
        # vertex colors as fallback (ColorVisuals or after to_color)
        try:
            if uvs is None or texture is None:
                if hasattr(vis, "vertex_colors") and vis.vertex_colors is not None:
                    vc = np.asarray(vis.vertex_colors)
                    if vc.shape[0] == np.asarray(mesh.vertices).shape[0]:
                        vc = vc[ok] if ok.shape[0] == vc.shape[0] else vc
                        if vc.shape[0] == v.shape[0]:
                            vcol = vc[:, :3].astype(np.float32)
                            if vcol.max() > 1.5:
                                vcol = vcol / 255.0
        except Exception:
            pass

    vertices = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.ascontiguousarray(f, dtype=np.int64)).to(device)
    uv_t = (
        torch.from_numpy(np.ascontiguousarray(uvs, dtype=np.float32)).to(device)
        if uvs is not None
        else None
    )
    vcol_t = (
        torch.from_numpy(np.ascontiguousarray(vcol, dtype=np.float32)).to(device)
        if vcol is not None
        else None
    )
    tex_t = None
    if texture is not None:
        tex_t = torch.from_numpy(np.ascontiguousarray(texture, dtype=np.float32)).to(device)

    return MeshGPU(vertices=vertices, faces=faces, uvs=uv_t, vertex_colors=vcol_t, texture=tex_t)


def maybe_simplify(mesh: MeshGPU, target_faces: int) -> MeshGPU:
    if target_faces <= 0 or mesh.faces.shape[0] <= target_faces:
        return mesh
    import torch
    from cumesh import CuMesh

    print(f"[voxlizer] simplify faces {mesh.faces.shape[0]} → {target_faces}", flush=True)
    cu = CuMesh()
    # CuMesh wants int32 faces typically
    faces_i = mesh.faces.to(dtype=torch.int32).contiguous()
    verts = mesh.vertices.contiguous()
    cu.init(verts, faces_i)
    try:
        cu.remove_duplicate_faces()
        cu.remove_degenerate_faces()
        cu.fill_holes(max_hole_perimeter=0.05)
    except Exception as e:
        print(f"[voxlizer] cleanup warn: {e}", flush=True)
    cu.simplify(int(target_faces), verbose=True)
    nv, nf = cu.read()
    # simplification drops UV correspondence — fall back to nearest old vertex color later via BVH only
    return MeshGPU(
        vertices=nv.contiguous(),
        faces=nf.to(dtype=torch.int64).contiguous(),
        uvs=None,
        vertex_colors=None,
        texture=mesh.texture,
    )


def _fit_grid(
    verts: "torch.Tensor",
    out_res: int,
    pad_voxels: int,
) -> Tuple["torch.Tensor", "torch.Tensor", Tuple[int, int, int]]:
    """Return (origin xyz min), voxel_size scalar, (sx,sy,sz)."""
    import torch

    vmin = verts.min(dim=0).values
    vmax = verts.max(dim=0).values
    extent = (vmax - vmin).clamp_min(1e-6)
    longest = float(extent.max().item())
    # keep physical cube-ish full bbox fitted into out_res on longest axis
    usable = max(1, int(out_res) - 2 * int(pad_voxels))
    voxel = longest / float(usable)
    dims = torch.ceil(extent / voxel).to(dtype=torch.int64)
    dims = torch.clamp(dims, min=1)
    # hard cap VOX_SIZE_MAX later
    size = (dims + 2 * int(pad_voxels)).tolist()
    origin = vmin - float(pad_voxels) * voxel
    return origin, torch.tensor(voxel, device=verts.device, dtype=torch.float32), (
        int(size[0]),
        int(size[1]),
        int(size[2]),
    )


def _sample_texture(tex: "torch.Tensor", uv: "torch.Tensor") -> "torch.Tensor":
    """Bilinear sample tex (H,W,C) at uv (N,2) in [0,1], Origin top-left GLTF style (V up).

    Returns (N,C) float.
    """
    import torch

    h, w = int(tex.shape[0]), int(tex.shape[1])
    # GLTF: u right, v up — image row 0 is top, so flip v
    u = uv[:, 0]
    v = 1.0 - uv[:, 1]
    # wrap
    u = u - torch.floor(u)
    v = v - torch.floor(v)
    x = u * (w - 1)
    y = v * (h - 1)
    x0 = torch.floor(x).long().clamp(0, w - 1)
    y0 = torch.floor(y).long().clamp(0, h - 1)
    x1 = (x0 + 1).clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)
    fx = (x - x0.float()).unsqueeze(-1)
    fy = (y - y0.float()).unsqueeze(-1)
    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]
    return (c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy)


def voxelize_mesh(
    mesh: MeshGPU,
    *,
    out_res: int = 256,
    fill: bool = True,
    surface_band: float = 0.75,
    pad_voxels: int = 1,
    chunk: int = 1_500_000,
    color_mode: str = "texture",
    sdf_mode: str = "raystab",
) -> Tuple["np.ndarray", "np.ndarray", dict]:
    """Return (grid ZYX uint8 materials, palette Nx3 uint8, stats)."""
    import torch
    from cumesh import cuBVH
    import vox_io

    device = mesh.vertices.device
    origin, voxel, (sx, sy, sz) = _fit_grid(mesh.vertices, out_res, pad_voxels)
    # clamp volumes to VOX_SIZE_MAX
    max_dim = max(sx, sy, sz)
    if max_dim > vox_io.VOX_SIZE_MAX:
        scale = vox_io.VOX_SIZE_MAX / float(max_dim)
        sx = max(1, int(sx * scale))
        sy = max(1, int(sy * scale))
        sz = max(1, int(sz * scale))
        # recompute voxel from verts with new sizes
        vmin = mesh.vertices.min(0).values
        vmax = mesh.vertices.max(0).values
        extent = (vmax - vmin).clamp_min(1e-6)
        voxel = (extent / torch.tensor([sx - 2 * pad_voxels, sy - 2 * pad_voxels, sz - 2 * pad_voxels], device=device).clamp_min(1)).max()
        origin = vmin - float(pad_voxels) * voxel

    sx = int(min(sx, vox_io.VOX_SIZE_MAX))
    sy = int(min(sy, vox_io.VOX_SIZE_MAX))
    sz = int(min(sz, vox_io.VOX_SIZE_MAX))
    print(
        f"[voxlizer] grid {sx}x{sy}x{sz}  voxel={float(voxel):.5f}  "
        f"fill={fill} band={surface_band} color={color_mode}",
        flush=True,
    )

    # BVH wants numpy inputs
    bvh = cuBVH(mesh.vertices, mesh.faces)

    # Build centers in world space; index order Z,Y,X dense
    # To save memory, evaluate occupancy/color chunk-wise and write dense grid on CPU.
    grid = np.zeros((sz, sy, sx), dtype=np.uint8)
    # collect solid colors then palette-quantize
    solid_rgb_chunks = []
    solid_index_chunks = []  # flat zyx indices

    band = float(surface_band) * float(voxel)
    total = sx * sy * sz
    print(f"[voxlizer] querying {total} centers via cuBVH …", flush=True)

    # precompute 1d axes
    xs = origin[0] + (torch.arange(sx, device=device, dtype=torch.float32) + 0.5) * voxel
    ys = origin[1] + (torch.arange(sy, device=device, dtype=torch.float32) + 0.5) * voxel
    zs = origin[2] + (torch.arange(sz, device=device, dtype=torch.float32) + 0.5) * voxel

    # iterate z slices to keep peak mem bounded
    # each slice = sy*sx points
    slice_n = sy * sx
    # sub-chunk within slice if needed
    sub = max(1, int(chunk // max(1, sx)))  # rows of y per chunk

    use_tex = (
        color_mode == "texture"
        and mesh.texture is not None
        and mesh.uvs is not None
        and mesh.uvs.shape[0] == mesh.vertices.shape[0]
    )
    use_vcol = color_mode in ("texture", "vertex") and mesh.vertex_colors is not None
    faces = mesh.faces
    uvs = mesh.uvs
    vcols = mesh.vertex_colors
    tex = mesh.texture

    processed = 0
    t_q0 = time.time()
    for z0 in range(sz):
        # centers for this z: (sy,sx,3)
        # process y bands
        for y0 in range(0, sy, sub):
            y1 = min(sy, y0 + sub)
            yy = ys[y0:y1]
            # meshgrid
            # order: y slow? we want x fastest
            # shape (ny, sx, 3)
            ny = y1 - y0
            grid_y, grid_x = torch.meshgrid(yy, xs, indexing="ij")
            grid_z = zs[z0].expand_as(grid_x)
            pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).contiguous()

            if fill:
                dist, face_id, uvw = bvh.signed_distance(pts, return_uvw=True, mode=sdf_mode)
                # solid if inside or near surface
                solid = (dist <= band)
            else:
                dist, face_id, uvw = bvh.unsigned_distance(pts, return_uvw=True)
                solid = dist <= band

            if not bool(solid.any()):
                processed += pts.shape[0]
                continue

            s_idx = torch.nonzero(solid, as_tuple=False).squeeze(1)
            f_id = face_id[s_idx].long()
            w = uvw[s_idx].clamp_min(0)
            # renorm bary weights
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            tri = faces[f_id]  # (S,3)

            colors = None
            if color_mode == "solid":
                colors = torch.full((s_idx.shape[0], 3), 0.7, device=device)
            elif use_tex:
                corner_uv = uvs[tri]  # (S,3,2)
                uv = (corner_uv * w.unsqueeze(-1)).sum(dim=1)
                rgba = _sample_texture(tex, uv)
                colors = rgba[:, :3]
                if rgba.shape[1] >= 4:
                    # Drop only clearly transparent surface samples. If the texture
                    # alpha channel is broken/all-zero (common with some WebP GLBs),
                    # keep occupancy and fall back to RGB/default colors.
                    a = rgba[:, 3]
                    keep = a > 0.05
                    if bool(keep.any()) and (not bool(keep.all())):
                        s_idx = s_idx[keep]
                        colors = colors[keep]
                    elif not bool(keep.any()):
                        # alpha unusable — ignore it
                        pass
            elif use_vcol:
                corner_c = vcols[tri]
                colors = (corner_c * w.unsqueeze(-1)).sum(dim=1)
            else:
                # bake a simple normal-facing gray-brown from face normal
                colors = torch.full((s_idx.shape[0], 3), 0.65, device=device)

            # map local point index → ZYX flat
            # pts laid out as (y-y0)*sx + x
            local = s_idx
            y_local = torch.div(local, sx, rounding_mode="floor")
            x_local = local - y_local * sx
            y_abs = y_local + y0
            flat = (z0 * (sy * sx) + y_abs * sx + x_local).to(dtype=torch.int64)

            solid_index_chunks.append(flat.detach().cpu().numpy())
            solid_rgb_chunks.append(
                (colors.clamp(0, 1).detach().cpu().numpy() * 255.0).astype(np.uint8)
            )
            processed += pts.shape[0]

        if (z0 + 1) % max(1, sz // 8) == 0 or z0 + 1 == sz:
            print(
                f"[voxlizer]  z {z0+1}/{sz}  elapsed={time.time()-t_q0:.1f}s",
                flush=True,
            )

    if not solid_index_chunks:
        raise RuntimeError("voxelization produced zero solid voxels — try larger OUT_RES or SURFACE_BAND")

    idxs = np.concatenate(solid_index_chunks, axis=0)
    rgbs = np.concatenate(solid_rgb_chunks, axis=0)
    # unique idx keep first (or average) — first is fine
    # If duplicates across (shouldn't within shell), np.unique
    order = np.argsort(idxs, kind="mergesort")
    idxs = idxs[order]
    rgbs = rgbs[order]
    uniq_mask = np.ones(len(idxs), dtype=bool)
    uniq_mask[1:] = idxs[1:] != idxs[:-1]
    idxs = idxs[uniq_mask]
    rgbs = rgbs[uniq_mask]

    print(f"[voxlizer] solid voxels={len(idxs)}  quantizing palette…", flush=True)
    mats, palette = vox_io.materials_from_colors(rgbs, max_colors=int(os.environ.get("MAX_COLORS", "255")))
    # materials_from_colors returns 0 for empty — all our samples are solid, fix any 0 → 1
    mats = mats.copy()
    mats[mats == 0] = 1

    # scatter into grid
    zz = idxs // (sy * sx)
    rem = idxs - zz * (sy * sx)
    yy = rem // sx
    xx = rem - yy * sx
    grid[zz, yy, xx] = mats

    stats = {
        "size": (sx, sy, sz),
        "solid": int(np.count_nonzero(grid)),
        "voxel": float(voxel.detach().cpu() if hasattr(voxel, "detach") else voxel),
        "origin": [float(x) for x in origin.detach().cpu().tolist()],
        "palette": int(len(palette)),
        "query_s": time.time() - t_q0,
    }
    return grid, np.asarray(palette, dtype=np.uint8), stats



def convert_glb_file(
    in_path: Path,
    *,
    out_res: int = 256,
    fill: bool = True,
    surface_band: float = 0.75,
    pad_voxels: int = 1,
    chunk: int = 1_500_000,
    color_mode: str = "texture",
    simplify_faces: int = 0,
    device: str = "cuda",
    sdf_mode: str = "raystab",
    crop: bool = True,
    max_colors: int = 255,
) -> Tuple["np.ndarray", "np.ndarray", dict, "object"]:
    """Voxelize a textured GLB/mesh file.

    Returns (grid_zyx_uint8, palette_nx3_uint8, stats, VoxelGrid).
    The returned VoxelGrid is in mesh/world axes (glTF Y-up for GLB). No extra Y/Z swap on encode.
    """
    _require_cuda_stack()
    import vox_io

    old_max = os.environ.get("MAX_COLORS")
    os.environ["MAX_COLORS"] = str(int(max_colors))
    try:
        mesh = load_textured_mesh(Path(in_path), device=device)
        if simplify_faces and int(simplify_faces) > 0:
            mesh = maybe_simplify(mesh, int(simplify_faces))
        grid_np, palette, stats = voxelize_mesh(
            mesh,
            out_res=int(out_res),
            fill=bool(fill),
            surface_band=float(surface_band),
            pad_voxels=int(pad_voxels),
            chunk=int(chunk),
            color_mode=str(color_mode or "texture").lower(),
            sdf_mode=str(sdf_mode or "raystab"),
        )
        grid = vox_io.VoxelGrid(grid_np)
        if crop:
            grid = grid.crop_to_solid(pad=0)
        stats = dict(stats)
        stats["size"] = (grid.size_x, grid.size_y, grid.size_z)
        stats["solid"] = int(grid.count_solid())
        return grid.data, np.asarray(palette, dtype=np.uint8), stats, grid
    finally:
        if old_max is None:
            os.environ.pop("MAX_COLORS", None)
        else:
            os.environ["MAX_COLORS"] = old_max


def encode_vox_bytes(grid, palette=None) -> bytes:
    """Encode mesh/world VoxelGrid to VOX2 bytes.

    GLB/glTF is already Y-up (no TRELLIS Z-up→Y-up swap). Orient with X/Z swap + X flip so
    the default +Z camera matches the source image left-right.
    """
    import vox_io

    return vox_io.encode(vox_io.orient_glb_to_vox(grid), use_zstd=True, palette=palette)


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not args or args[0] in ("-h", "--help", "/?"):
        _usage()
        return 0 if args else 1

    in_path = Path(args[0]).expanduser()
    # resolve relative to caller CWD (argv path as given)
    if not in_path.is_absolute():
        in_path = (Path.cwd() / in_path).resolve()
    else:
        in_path = in_path.resolve()
    out_path = (
        Path(args[1]).expanduser()
        if len(args) >= 2
        else in_path.with_suffix(".vox")
    )
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()

    if not in_path.is_file():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1

    try:
        _require_cuda_stack()
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    import torch
    import vox_io

    out_res = int(os.environ.get("OUT_RES", "256"))
    fill = os.environ.get("VOX_FILL", "1") != "0"
    band = float(os.environ.get("SURFACE_BAND", "0.75"))
    pad = int(os.environ.get("PAD_VOXELS", "1"))
    chunk = int(os.environ.get("CHUNK", "1500000"))
    color_mode = os.environ.get("COLOR_MODE", "texture").lower()
    simplify = int(os.environ.get("SIMPLIFY_FACES", "0"))
    device = os.environ.get("DEVICE", "cuda")
    sdf_mode = os.environ.get("SDF_MODE", "raystab")
    crop = os.environ.get("CROP", "1") != "0"
    max_colors = int(os.environ.get("MAX_COLORS", "255"))

    t0 = time.time()
    print(f"[voxlizer] load {in_path}", flush=True)
    _, palette, stats, grid = convert_glb_file(
        in_path,
        out_res=out_res,
        fill=fill,
        surface_band=band,
        pad_voxels=pad,
        chunk=chunk,
        color_mode=color_mode,
        simplify_faces=simplify,
        device=device,
        sdf_mode=sdf_mode,
        crop=crop,
        max_colors=max_colors,
    )
    print(
        f"[voxlizer] size={grid.size_x}x{grid.size_y}x{grid.size_z} solid={grid.count_solid()} "
        f"palette={stats.get('palette')}",
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # GLB path is Y-up; bypass TRELLIS Z-up swap; orient X/Z + flip X for +Z camera.
    raw = encode_vox_bytes(grid, palette=palette)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    n = len(raw)
    # debug palette/preview
    try:
        from PIL import Image

        if palette is not None and len(palette):
            Image.fromarray(np.asarray(palette, dtype=np.uint8).reshape(-1, 3)).save(
                out_path.with_suffix(".palette.png")
            )
        # top-down preview
        data = grid.data
        solid = data != 0
        if solid.any() and palette is not None and len(palette):
            # project along Y (top-down): for each x,z take highest solid y color
            zs, ys, xs = np.where(solid)
            canvas = np.zeros((grid.size_z, grid.size_x, 3), dtype=np.uint8)
            depth = np.full((grid.size_z, grid.size_x), -1, dtype=np.int32)
            pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
            ids = np.clip(data[solid].astype(np.int32) - 1, 0, len(pal) - 1)
            cols = pal[ids]
            for i in range(len(xs)):
                x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
                if y >= depth[z, x]:
                    depth[z, x] = y
                    canvas[z, x] = cols[i]
            Image.fromarray(canvas[::-1]).save(out_path.with_name(out_path.stem + "_preview_xz.png"))
    except Exception as e:
        print(f"[voxlizer] preview skip: {e}", flush=True)

    # roundtrip against file axes (includes X/Z swap)
    expected = vox_io.orient_glb_to_vox(grid)
    rt = vox_io.decode(out_path.read_bytes())
    ok = (
        rt.count_solid() == expected.count_solid()
        and rt.size_x == expected.size_x
        and rt.size_y == expected.size_y
        and rt.size_z == expected.size_z
    )
    print(
        f"[voxlizer] wrote {out_path}  bytes={n}  size={grid.size_x}x{grid.size_y}x{grid.size_z} "
        f"solid={grid.count_solid()} palette={stats['palette']}  "
        f"roundtrip={'OK' if ok else 'FAIL'}  total={time.time()-t0:.1f}s",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
