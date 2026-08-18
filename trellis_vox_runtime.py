"""Shared TRELLIS.2 image→VOX runtime (load once, convert many)."""

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


def bootstrap_env() -> None:
    """Windows-friendly TORCH/spconv defaults + path bootstrap (idempotent)."""
    if str(STUBS) not in sys.path:
        sys.path.insert(0, str(STUBS))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("SPARSE_CONV_BACKEND", "spconv")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

    import trellis2.modules.sparse.conv.config as _spconv_cfg

    _spconv_cfg.SPCONV_ALGO = os.environ.get("SPCONV_ALGO", "native")


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
    ) -> None:
        bootstrap_env()
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
        print(f"[runtime] loading DINOv3 from {dino_repo} …", flush=True)
        dino = DinoV3FeatureExtractor.__new__(DinoV3FeatureExtractor)
        dino.model_name = dino_repo
        dino.model = DINOv3ViTModel.from_pretrained(dino_repo)
        dino.model.eval()
        dino.image_size = 512
        dino.transform = transforms.Compose(
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        print(f"[runtime] loading rembg {rembg_name} …", flush=True)
        rembg = rembg_mod.BiRefNet(model_name=rembg_name)
        rembg.model = rembg.model.float()

        class _Dummy:
            def __init__(self, *a, **k):
                pass

        _orig_dino, _orig_biref = ife.DinoV3FeatureExtractor, rbg.BiRefNet
        ife.DinoV3FeatureExtractor = _Dummy
        rbg.BiRefNet = _Dummy
        try:
            print(f"[runtime] loading TRELLIS pipeline {model} …", flush=True)
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
        color_axis: str = "xy",
        downsample_device: Optional[str] = None,
        include_palette: bool = False,
        include_preview: bool = False,
    ) -> ConvertResult:
        if not self._ready or self.pipeline is None:
            raise RuntimeError("runtime not loaded; call load() first")

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

        with self._lock:
            t0 = time.time()
            pre_image = self.pipeline.preprocess_image(pil)
            meshes = self.pipeline.run(
                pre_image,
                seed=seed,
                preprocess_image=False,
                pipeline_type=pipeline_type,
            )
            mesh = meshes[0]
            grid, palette = vox_io.grid_from_mesh_with_voxel(
                mesh,
                material_mode=material_mode,
                alpha_threshold=float(alpha_threshold),
                max_colors=255,
                crop=True,
                solid_material=1,
                color_image=pre_image if material_mode == "image" else None,
                color_axis=color_axis or "xy",
            )
            native_size = (grid.size_x, grid.size_y, grid.size_z)
            native_solid = grid.count_solid()
            if out_res > 0 and max(grid.size_x, grid.size_y, grid.size_z) > out_res:
                ds_dev = downsample_device if downsample_device is not None else os.environ.get(
                    "DOWNSAMPLE_DEVICE"
                )
                grid = vox_io.downsample_grid(grid, target_max=out_res, device=ds_dev)

            vox_bytes = vox_io.encode(vox_io.swap_yz(grid), use_zstd=True)
            palette_png = None
            if include_palette and palette is not None:
                buf = io.BytesIO()
                Image.fromarray(np.asarray(palette, dtype=np.uint8).reshape(-1, 3)).save(
                    buf, format="PNG"
                )
                palette_png = buf.getvalue()
            preview_png = self._preview_png(grid, palette) if include_preview else None

            # free some activation memory between requests
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(
                f"[runtime] convert done {grid.size_x}x{grid.size_y}x{grid.size_z} "
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
            )
