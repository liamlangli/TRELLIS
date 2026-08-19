"""Shared TRELLIS.2 image→VOX runtime (load once, convert many).

Default convert path is the full local pipeline:
  image → TRELLIS mesh → textured GLB → CuMesh voxelize → VOX2
Optional mode="direct" keeps the old MeshWithVoxel shortcut.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
STUBS = ROOT / "stubs"
O_VOXEL_SRC = ROOT / "o-voxel"
EXT_CUMESH = ROOT / ".ext_build" / "CuMesh"
EXT_FLEXGEMM = ROOT / ".ext_build" / "FlexGEMM"


def _prepend_sys_path(path: Path) -> None:
    s = str(path)
    if path.is_dir() and s not in sys.path:
        sys.path.insert(0, s)


def _drop_sys_path(path: Path) -> None:
    s = str(path)
    sys.path[:] = [p for p in sys.path if p != s]


def _purge_modules(*prefixes: str) -> None:
    doomed = [k for k in list(sys.modules) if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in doomed:
        sys.modules.pop(k, None)


def bootstrap_env(*, allow_stubs: bool = True) -> None:
    """Windows-friendly TORCH/spconv defaults + path bootstrap (idempotent).

    Prefer installed/built packages. Fall back to lightweight stubs only when
    the real CUDA stack (o_voxel/cumesh/...) cannot be imported — and remove the
    half-broken source trees from sys.path so stubs actually win.
    """
    _prepend_sys_path(ROOT)

    # Prefer already-installed site-packages. Only fall back to local extension
    # source trees if the installed packages are missing.
    try:
        import cumesh  # noqa: F401
        import o_voxel  # noqa: F401
        from o_voxel import postprocess as _pp  # noqa: F401
        _have_full = True
    except Exception:
        _have_full = False
    if not _have_full:
        for root in (EXT_CUMESH, EXT_FLEXGEMM, O_VOXEL_SRC):
            _prepend_sys_path(root)

    # Probe full stack.
    full_ok = False
    try:
        import importlib

        # Drop source shadows so installed wheels win when both exist.
        for shadow in (str(O_VOXEL_SRC), str(EXT_CUMESH), str(EXT_FLEXGEMM), str(STUBS)):
            while shadow in sys.path:
                sys.path.remove(shadow)
        _purge_modules('o_voxel', 'cumesh', 'flex_gemm')
        importlib.invalidate_caches()
        import cumesh  # noqa: F401
        import o_voxel  # noqa: F401
        from o_voxel import postprocess as _pp  # noqa: F401

        modfile = str(getattr(o_voxel, "__file__", "") or "").replace("\\", "/")
        if "stubs" in modfile:
            raise ImportError("o_voxel resolved to stubs")
        if not hasattr(_pp, "to_glb"):
            raise ImportError("o_voxel.postprocess.to_glb missing")
        # ensure to_glb is not the stub raiser
        doc = (getattr(_pp.to_glb, "__doc__", "") or "")
        if "requires full o_voxel" in doc:
            raise ImportError("stub to_glb still active")
        full_ok = True
        print(f"[runtime] full o_voxel stack: {o_voxel.__file__}", flush=True)
    except Exception as e:
        full_ok = False
        if not allow_stubs:
            raise RuntimeError(
                "Full o_voxel/cumesh stack is required but import failed: "
                f"{type(e).__name__}: {e}"
            ) from e
        # Drop broken sources so stubs can own the names.
        for root in (EXT_CUMESH, EXT_FLEXGEMM, O_VOXEL_SRC, STUBS):
            _drop_sys_path(root)
        _purge_modules("o_voxel", "cumesh", "flex_gemm")
        _prepend_sys_path(STUBS)
        print(
            f"[runtime] WARNING: full mesh stack unavailable ({type(e).__name__}: {e}); "
            "using stubs fallback (VOX path only, GLB export disabled)",
            flush=True,
        )

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

    # Configure sparse backends before any heavy trellis2 imports bind CONV/ATTN.
    import trellis2.modules.sparse.config as _sparse_cfg
    import trellis2.modules.sparse.conv.config as _spconv_cfg

    backend = os.environ.get("SPARSE_CONV_BACKEND", "flex_gemm")
    if backend in ("none", "spconv", "torchsparse", "flex_gemm"):
        _sparse_cfg.CONV = backend
    attn = os.environ.get("SPARSE_ATTN_BACKEND") or os.environ.get("ATTN_BACKEND")
    if attn in ("xformers", "flash_attn", "flash_attn_3"):
        _sparse_cfg.ATTN = attn
    _spconv_cfg.SPCONV_ALGO = os.environ.get("SPCONV_ALGO", "native")
    print(f"[runtime] sparse conv backend={_sparse_cfg.CONV} attn={_sparse_cfg.ATTN}", flush=True)


@dataclass
class ConvertResult:
    vox_bytes: bytes
    size_x: int
    size_y: int
    size_z: int
    solid: int
    native_size: Tuple[int, int, int]
    native_solid: int
    seed: int
    pipeline_type: str
    material_mode: str
    out_res: int
    elapsed_s: float
    palette_png: Optional[bytes] = None
    preview_png: Optional[bytes] = None
    mode: str = "glb"
    glb_bytes: Optional[bytes] = None


class TrellisVoxRuntime:
    """Holds a hot pipeline on GPU. Thread-safe: one inference at a time."""

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.device_name: Optional[str] = None
        self.model_id: str = ""
        self._lock = threading.Lock()
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def load(
        self,
        model: Optional[str] = None,
        *,
        dino_repo: Optional[str] = None,
        rembg_name: Optional[str] = None,
        low_vram: bool = False,
        require_full_stack: Optional[bool] = None,
    ) -> None:
        # GLB path needs real o_voxel/cumesh/nvdiffrast. Default on unless CONVERT_MODE=direct.
        if require_full_stack is None:
            mode = (os.environ.get("CONVERT_MODE", "glb") or "glb").strip().lower()
            require_full_stack = mode not in ("direct", "fast", "mesh", "mesh_voxel", "direct_mesh")
        bootstrap_env(allow_stubs=not require_full_stack)
        if require_full_stack:
            # Ensure residual /stubs path cannot shadow installed packages.
            sys.path[:] = [
                p
                for p in sys.path
                if Path(p).resolve().name.lower() != "stubs"
                and "/stubs/" not in Path(p).as_posix().lower()
                and not Path(p).as_posix().lower().endswith("/stubs")
            ]

        import torch
        from transformers import DINOv3ViTModel
        from torchvision import transforms

        import trellis2.modules.image_feature_extractor as ife
        import trellis2.pipelines.rembg as rbg
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.pipelines import rembg as rembg_mod

        model = model or os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
        dino_repo = dino_repo or os.environ.get(
            "DINO_MODEL", "camenduru/dinov3-vitl16-pretrain-lvd1689m"
        )
        rembg_name = rembg_name or os.environ.get("REMBG_MODEL", "ZhengPeng7/BiRefNet")

        print(
            f"[runtime] torch {torch.__version__} cuda={torch.cuda.is_available()} "
            f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}",
            flush=True,
        )
        t0 = time.time()
        print(f"[runtime] loading DINOv3 from {dino_repo} ...", flush=True)
        dino = DinoV3FeatureExtractor.__new__(DinoV3FeatureExtractor)
        dino.model_name = dino_repo
        dino.model = DINOv3ViTModel.from_pretrained(dino_repo)
        dino.model.eval()
        dino.image_size = 512
        dino.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        print(f"[runtime] loading rembg {rembg_name} ...", flush=True)
        rembg = rembg_mod.BiRefNet(model_name=rembg_name)
        rembg.model = rembg.model.float()

        class _Dummy:
            def __init__(self, *a, **k):
                pass

        _orig_dino, _orig_biref = ife.DinoV3FeatureExtractor, rbg.BiRefNet
        ife.DinoV3FeatureExtractor = _Dummy
        rbg.BiRefNet = _Dummy
        try:
            print(f"[runtime] loading TRELLIS pipeline {model} ...", flush=True)
            pipeline = Trellis2ImageTo3DPipeline.from_pretrained(model)
        finally:
            ife.DinoV3FeatureExtractor = _orig_dino
            rbg.BiRefNet = _orig_biref

        pipeline.image_cond_model = dino
        pipeline.rembg_model = rembg
        pipeline.low_vram = bool(low_vram)
        pipeline.cuda()
        self._install_robust_sparse_sampler(pipeline)

        self.pipeline = pipeline
        self.model_id = model
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self._ready = True
        print(f"[runtime] ready in {time.time() - t0:.1f}s", flush=True)

    @staticmethod
    def _install_robust_sparse_sampler(pipeline: Any, min_voxels: int = 64) -> None:
        import torch

        orig = pipeline.sample_sparse_structure

        def sample_sparse_structure(cond, resolution, num_samples=1, sampler_params=None):
            sampler_params = sampler_params or {}
            flow_model = pipeline.models["sparse_structure_flow_model"]
            reso = flow_model.resolution
            in_channels = flow_model.in_channels
            params = {**pipeline.sparse_structure_sampler_params, **sampler_params}
            decoder = pipeline.models["sparse_structure_decoder"]

            best_coords = None
            best_count = -1
            base_seed = int(torch.initial_seed() % (2**31 - 1))

            for attempt in range(8):
                seed = base_seed + attempt
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                noise = torch.randn(
                    num_samples, in_channels, reso, reso, reso, device=pipeline.device
                )
                z_s = pipeline.sparse_structure_sampler.sample(
                    flow_model,
                    noise,
                    **cond,
                    **params,
                    verbose=True,
                    tqdm_desc=f"Sampling sparse structure (try {attempt})",
                ).samples
                logits = decoder(z_s)

                decoded = logits > 0
                count = int(decoded.sum().item())
                if count < min_voxels:
                    thr = torch.quantile(logits.detach().flatten().float(), 0.995)
                    decoded_q = logits > thr
                    count_q = int(decoded_q.sum().item())
                    print(
                        f"  sparse try {attempt}: gt0={count}, quantile thr={float(thr):.2f} occ={count_q}",
                        flush=True,
                    )
                    if count_q > count:
                        decoded = decoded_q
                        count = count_q
                else:
                    print(f"  sparse try {attempt}: occ={count}", flush=True)

                if resolution != decoded.shape[2]:
                    ratio = decoded.shape[2] // resolution
                    decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
                    count = int(decoded.sum().item())

                if count > best_count:
                    best_coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
                    best_count = count
                if count >= min_voxels:
                    break

            if best_coords is None or best_coords.numel() == 0:
                return orig(cond, resolution, num_samples, sampler_params)

            print(f"sparse structure coords={tuple(best_coords.shape)} occ={best_count}", flush=True)
            return best_coords

        pipeline.sample_sparse_structure = sample_sparse_structure  # type: ignore[method-assign]

    @staticmethod
    def _preview_png(grid: Any, palette: Optional[np.ndarray]) -> Optional[bytes]:
        if palette is None:
            return None
        data = grid.data
        solid = data != 0
        if not np.any(solid):
            return None
        pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
        z, y, x = np.where(solid)
        ids = np.clip(data[solid].astype(np.int32) - 1, 0, len(pal) - 1)
        cols = pal[ids].astype(np.float32)
        out = np.zeros((grid.size_y, grid.size_x, 3), dtype=np.uint8)
        order = np.lexsort((z, x, y))
        y_s, x_s = y[order], x[order]
        cols_s = cols[order]
        keys = y_s.astype(np.int64) * (grid.size_x + 1) + x_s.astype(np.int64)
        _, first = np.unique(keys, return_index=True)
        out[y_s[first], x_s[first]] = np.clip(np.rint(cols_s[first]), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()

    def convert(
        self,
        image: Union[Image.Image, bytes, bytearray, memoryview, Path, str],
        *,
        seed: int = 0,
        pipeline_type: str = "512",
        material_mode: str = "image",
        out_res: int = 256,
        alpha_threshold: float = 0.5,
        color_axis: str = "auto",
        downsample_device: Optional[str] = None,
        include_palette: bool = False,
        include_preview: bool = False,
        mode: Optional[str] = None,
        keep_glb: bool = False,
        decimate_target: Optional[int] = None,
        texture_size: Optional[int] = None,
        simplify_target: Optional[int] = None,
        remesh: Optional[bool] = None,
        remesh_band: Optional[float] = None,
        remesh_project: Optional[float] = None,
        vox_fill: Optional[bool] = None,
        surface_band: Optional[float] = None,
        pad_voxels: Optional[int] = None,
        chunk: Optional[int] = None,
        color_mode: Optional[str] = None,
        simplify_faces: Optional[int] = None,
        sdf_mode: Optional[str] = None,
        max_colors: Optional[int] = None,
        crop: Optional[bool] = None,
        device: Optional[str] = None,
    ) -> ConvertResult:
        """image -> VOX.

        Default mode is the full local pipeline:
          preprocess -> TRELLIS mesh -> simplify -> o_voxel.to_glb -> CuMesh voxelize -> VOX2

        mode="direct" keeps the old shortcut that samples TRELLIS MeshWithVoxel
        occupancy without baking a textured GLB.
        """
        if not self._ready or self.pipeline is None:
            raise RuntimeError("runtime not loaded; call load() first")

        import tempfile
        import torch
        import vox_io

        if isinstance(image, (bytes, bytearray, memoryview)):
            pil = Image.open(io.BytesIO(bytes(image)))
        elif isinstance(image, (str, Path)):
            pil = Image.open(image)
        elif isinstance(image, Image.Image):
            pil = image
        else:
            raise TypeError(f"unsupported image type: {type(image)}")

        if pil.mode not in ("RGB", "RGBA"):
            pil = pil.convert("RGBA" if "A" in pil.getbands() else "RGB")

        material_mode = (material_mode or "image").lower()
        pipeline_type = pipeline_type or "512"
        out_res = int(out_res)
        seed = int(seed)
        mode = (mode or os.environ.get("CONVERT_MODE", "glb") or "glb").strip().lower()
        if mode in ("full", "img_glb_vox", "image_glb_vox", "glb_path"):
            mode = "glb"
        if mode in ("fast", "mesh", "mesh_voxel", "direct_mesh"):
            mode = "direct"
        if mode not in ("glb", "direct"):
            raise ValueError(f"unsupported convert mode: {mode!r} (use 'glb' or 'direct')")

        with self._lock:
            t0 = time.time()
            pre_image = self.pipeline.preprocess_image(pil)
            print(
                f"[runtime] convert mode={mode} seed={seed} pipeline={pipeline_type} out_res={out_res}",
                flush=True,
            )
            meshes = self.pipeline.run(
                pre_image,
                seed=seed,
                preprocess_image=False,
                pipeline_type=pipeline_type,
            )
            mesh = meshes[0]

            if mode == "direct":
                grid, palette = vox_io.grid_from_mesh_with_voxel(
                    mesh,
                    material_mode=material_mode,
                    alpha_threshold=float(alpha_threshold),
                    max_colors=int(max_colors or os.environ.get("MAX_COLORS", "255")),
                    crop=True if crop is None else bool(crop),
                    solid_material=1,
                    color_image=pre_image,
                    color_axis=color_axis or "auto",
                )
                native_size = (grid.size_x, grid.size_y, grid.size_z)
                native_solid = grid.count_solid()
                if out_res > 0 and max(grid.size_x, grid.size_y, grid.size_z) > out_res:
                    ds_dev = (
                        downsample_device
                        if downsample_device is not None
                        else os.environ.get("DOWNSAMPLE_DEVICE")
                    )
                    grid = vox_io.downsample_grid(grid, target_max=out_res, device=ds_dev)
                vox_bytes = vox_io.encode(
                    vox_io.swap_yz(grid),
                    use_zstd=True,
                    palette=palette,
                )
                glb_bytes = None
            else:
                # Full path: mesh -> textured GLB -> CuMesh BVH voxelizer.
                import o_voxel
                from glb_to_vox import convert_glb_file

                decimate = int(
                    decimate_target
                    if decimate_target is not None
                    else os.environ.get("DECIMATE_TARGET", "1000000")
                )
                tex_size = int(
                    texture_size if texture_size is not None else os.environ.get("TEXTURE_SIZE", "2048")
                )
                simp_target = int(
                    simplify_target
                    if simplify_target is not None
                    else os.environ.get("SIMPLIFY_TARGET", "16777216")
                )
                do_remesh = (
                    bool(remesh)
                    if remesh is not None
                    else os.environ.get("REMESH", "0") != "0"
                )
                r_band = float(
                    remesh_band if remesh_band is not None else os.environ.get("REMESH_BAND", "1")
                )
                r_proj = float(
                    remesh_project
                    if remesh_project is not None
                    else os.environ.get("REMESH_PROJECT", "0")
                )

                print(
                    f"[runtime] GLB postprocess simplify={simp_target} decimate={decimate} "
                    f"texture={tex_size} remesh={do_remesh}",
                    flush=True,
                )
                try:
                    bc = mesh.attrs[:, mesh.layout["base_color"]].detach().float()
                    print(
                        f"[runtime] base_color mean={bc.mean(0).tolist()} std={bc.std(0).tolist()}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[runtime] base_color stats failed: {e}", flush=True)

                mesh.simplify(simp_target)
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=decimate,
                    texture_size=tex_size,
                    remesh=do_remesh,
                    remesh_band=r_band,
                    remesh_project=r_proj,
                    verbose=True,
                )

                fill = (
                    bool(vox_fill)
                    if vox_fill is not None
                    else os.environ.get("VOX_FILL", "1") != "0"
                )
                s_band = float(
                    surface_band if surface_band is not None else os.environ.get("SURFACE_BAND", "0.75")
                )
                pad = int(pad_voxels if pad_voxels is not None else os.environ.get("PAD_VOXELS", "1"))
                chunk_n = int(chunk if chunk is not None else os.environ.get("CHUNK", "1500000"))
                c_mode = (
                    color_mode if color_mode is not None else os.environ.get("COLOR_MODE", "texture")
                ).lower()
                simp_faces = int(
                    simplify_faces
                    if simplify_faces is not None
                    else os.environ.get("SIMPLIFY_FACES", "0")
                )
                sdf = sdf_mode if sdf_mode is not None else os.environ.get("SDF_MODE", "raystab")
                mcol = int(max_colors if max_colors is not None else os.environ.get("MAX_COLORS", "255"))
                do_crop = True if crop is None else bool(crop)
                dev = device if device is not None else os.environ.get("DEVICE", "cuda")

                tmp_dir = Path(tempfile.mkdtemp(prefix="trellis_img_glb_vox_"))
                glb_path = tmp_dir / "sample.glb"
                try:
                    glb.export(str(glb_path), extension_webp=True)
                    glb_bytes = glb_path.read_bytes() if keep_glb else None
                    keep_dir = os.environ.get("TRELLIS_KEEP_GLB_DIR") or os.environ.get("KEEP_GLB_DIR")
                    if keep_dir or keep_glb:
                        try:
                            dest_dir = Path(keep_dir) if keep_dir else Path(tempfile.gettempdir())
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            dest = dest_dir / f"trellis_{int(time.time())}.glb"
                            dest.write_bytes(glb_path.read_bytes())
                            print(f"[runtime] kept intermediate GLB at {dest}", flush=True)
                            if keep_glb and glb_bytes is None:
                                glb_bytes = dest.read_bytes()
                        except Exception as e:
                            print(f"[runtime] keep GLB failed: {e}", flush=True)
                    print(
                        f"[runtime] GLB baked bytes={glb_path.stat().st_size} -> voxelize out_res={out_res}",
                        flush=True,
                    )
                    bands = [s_band]
                    for extra in (1.5, 2.5, 4.0):
                        if extra not in bands:
                            bands.append(extra)
                    last_err = None
                    palette = None
                    stats = None
                    grid = None
                    for bi, band_try in enumerate(bands):
                        try:
                            print(
                                f"[runtime] voxelize try {bi+1}/{len(bands)} band={band_try} fill={fill}",
                                flush=True,
                            )
                            _, palette, stats, grid = convert_glb_file(
                                glb_path,
                                out_res=out_res,
                                fill=fill,
                                surface_band=band_try,
                                pad_voxels=pad,
                                chunk=chunk_n,
                                color_mode=c_mode,
                                simplify_faces=simp_faces,
                                device=dev,
                                sdf_mode=sdf,
                                crop=do_crop,
                                max_colors=mcol,
                            )
                            last_err = None
                            s_band = float(band_try)
                            break
                        except Exception as e:
                            last_err = e
                            print(f"[runtime] voxelize try failed: {type(e).__name__}: {e}", flush=True)
                    if last_err is not None or grid is None:
                        # Keep GLB for inspection when voxelization fails.
                        keep_path = Path(tempfile.gettempdir()) / f"trellis_fail_{int(time.time())}.glb"
                        try:
                            keep_path.write_bytes(glb_path.read_bytes())
                            print(f"[runtime] preserved failed GLB at {keep_path}", flush=True)
                        except Exception as e:
                            print(f"[runtime] could not preserve GLB: {e}", flush=True)
                        raise RuntimeError(
                            f"GLB voxelization failed after retries: {last_err}"
                        ) from last_err
                    native_size = tuple(stats.get("size", (grid.size_x, grid.size_y, grid.size_z)))
                    native_solid = int(stats.get("solid", grid.count_solid()))
                    # material_mode is not used on the GLB path; report color_mode instead.
                    material_mode = c_mode
                    # GLB/glTF is Y-up already (no TRELLIS Z-up swap).
                    # Orient so default +Z camera matches source image: swap X/Z then flip X.
                    grid = vox_io.orient_glb_to_vox(grid)
                    vox_bytes = vox_io.encode(
                        grid,
                        use_zstd=True,
                        palette=palette,
                    )
                finally:
                    try:
                        if glb_path.is_file():
                            glb_path.unlink()
                        tmp_dir.rmdir()
                    except OSError:
                        pass

            palette_png = None
            if include_palette and palette is not None:
                buf = io.BytesIO()
                Image.fromarray(np.asarray(palette, dtype=np.uint8).reshape(-1, 3)).save(
                    buf, format="PNG"
                )
                palette_png = buf.getvalue()
            preview_png = self._preview_png(grid, palette) if include_preview else None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Guard against collapsed / paper-thin GLB voxelizations.
            # Prefer falling back to MeshWithVoxel direct path over returning a paper sheet.
            if mode == "glb":
                dims = sorted([int(grid.size_x), int(grid.size_y), int(grid.size_z)])
                if dims[0] <= 4 and dims[2] >= 32:
                    print(
                        f"[runtime] GLB voxelization collapsed to {grid.size_x}x{grid.size_y}x{grid.size_z}; "
                        "falling back to direct MeshWithVoxel path",
                        flush=True,
                    )
                    grid, palette = vox_io.grid_from_mesh_with_voxel(
                        mesh,
                        material_mode=(os.environ.get("MATERIAL_MODE", "image") or "image").lower(),
                        alpha_threshold=float(alpha_threshold),
                        max_colors=int(max_colors or os.environ.get("MAX_COLORS", "255")),
                        crop=True if crop is None else bool(crop),
                        solid_material=1,
                        color_image=pre_image,
                        color_axis=color_axis or "auto",
                    )
                    native_size = (grid.size_x, grid.size_y, grid.size_z)
                    native_solid = grid.count_solid()
                    if out_res > 0 and max(grid.size_x, grid.size_y, grid.size_z) > out_res:
                        ds_dev = (
                            downsample_device
                            if downsample_device is not None
                            else os.environ.get("DOWNSAMPLE_DEVICE")
                        )
                        grid = vox_io.downsample_grid(grid, target_max=out_res, device=ds_dev)
                    vox_bytes = vox_io.encode(
                        vox_io.swap_yz(grid),
                        use_zstd=True,
                        palette=palette,
                    )
                    material_mode = (os.environ.get("MATERIAL_MODE", "image") or "image").lower()
                    mode = "glb+direct_fallback"
            elapsed = time.time() - t0
            print(
                f"[runtime] convert done mode={mode} {grid.size_x}x{grid.size_y}x{grid.size_z} "
                f"solid={grid.count_solid()} in {elapsed:.1f}s",
                flush=True,
            )
            return ConvertResult(
                vox_bytes=vox_bytes,
                size_x=grid.size_x,
                size_y=grid.size_y,
                size_z=grid.size_z,
                solid=grid.count_solid(),
                native_size=native_size,
                native_solid=native_solid,
                seed=seed,
                pipeline_type=pipeline_type,
                material_mode=material_mode,
                out_res=out_res,
                elapsed_s=elapsed,
                palette_png=palette_png,
                preview_png=preview_png,
                mode=mode,
                glb_bytes=glb_bytes,
            )
