"""Local image → decoded color voxels → VOX2, using PyTorch MPS."""
from __future__ import annotations

import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def bootstrap_env():
    # Must precede the first torch import.
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    os.environ.setdefault('ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('SPARSE_ATTN_BACKEND', 'sdpa')
    os.environ.setdefault('SPARSE_CONV_BACKEND', 'none')


bootstrap_env()


@dataclass
class ConvertResult:
    vox_bytes: bytes
    size_x: int
    size_y: int
    size_z: int
    solid: int
    native_size: tuple[int, int, int]
    native_solid: int
    elapsed_s: float
    palette_png: bytes | None = None
    preview_png: bytes | None = None
    preprocessed_png: bytes | None = None
    mode: str = 'direct'


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


class TrellisVoxRuntime:
    def __init__(self):
        self.pipeline = None
        self._lock = threading.Lock()

    @property
    def ready(self):
        return self.pipeline is not None

    def load(self, model=None):
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        device = os.environ.get('TRELLIS_DEVICE', 'mps' if torch.backends.mps.is_available() else 'cpu')
        if device not in ('mps', 'cpu'):
            raise ValueError('TRELLIS_DEVICE must be mps or cpu')
        if device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('PyTorch MPS is unavailable; run ./setup.sh with native arm64 Python')
        print(f'[runtime] torch={torch.__version__} device={device}; loading 512 voxel models', flush=True)
        t0 = time.monotonic()
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            model or os.environ.get('TRELLIS_MODEL', 'microsoft/TRELLIS.2-4B'))
        self.pipeline.to(torch.device(device))
        print(f'[runtime] ready in {time.monotonic()-t0:.1f}s', flush=True)

    @staticmethod
    def _preview_png(grid, palette):
        import numpy as np
        from PIL import Image
        if palette is None or not grid.count_solid():
            return None
        # VOX2 is Y-up; display the +Z-facing XY projection with Y upwards.
        solid = grid.data != 0
        depth = grid.size_z - 1 - np.argmax(solid[::-1], axis=0)
        yy, xx = np.indices((grid.size_y, grid.size_x))
        ids = grid.data[depth, yy, xx]
        rgba = np.zeros((*ids.shape, 4), dtype=np.uint8)
        visible = ids > 0
        rgba[visible, :3] = palette[ids[visible].astype(int)-1]
        rgba[visible, 3] = 255
        return png_bytes(Image.fromarray(rgba[::-1]))

    def convert(self, image, *, seed=0, pipeline_type='512', material_mode='color',
                max_height=256, max_colors=220, alpha_threshold=0.5,
                color_axis='auto', downsample_device=None, include_palette=True,
                include_preview=True, mode='direct'):
        import numpy as np
        from PIL import Image, ImageOps
        import vox_io
        if not self.ready:
            raise RuntimeError('Call load() before convert()')
        if mode != 'direct':
            raise ValueError('Only CONVERT_MODE=direct is supported; GLB export has been removed')
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                pil = opened.copy()
        elif isinstance(image, (bytes, bytearray, memoryview)):
            with Image.open(io.BytesIO(bytes(image))) as opened:
                pil = opened.copy()
        else:
            pil = image.copy()
        pil = ImageOps.exif_transpose(pil)
        pil = pil.convert("RGBA" if "A" in pil.getbands() or "transparency" in pil.info else "RGB")
        with self._lock:
            t0 = time.monotonic()
            pre = self.pipeline.preprocess_image(pil)
            steps = int(os.environ.get('STEPS', '12'))
            if not 1 <= steps <= 100:
                raise ValueError('STEPS must be 1..100')
            params = {'steps': steps}
            voxel = self.pipeline.run(pre, seed=seed, preprocess_image=False,
                                      pipeline_type=pipeline_type,
                                      sparse_structure_sampler_params=params,
                                      shape_slat_sampler_params=params,
                                      tex_slat_sampler_params=params)[0]
            grid, palette = vox_io.grid_from_mesh_with_voxel(
                voxel, material_mode=material_mode, alpha_threshold=alpha_threshold,
                max_colors=max_colors, crop=True, color_image=pre,
                color_axis=color_axis, photo_match=os.environ.get('COLOR_TRANSFER', '1') != '0')
            if not grid.count_solid():
                raise RuntimeError('Model produced no occupied voxels; try a different seed or input')
            native_size = (grid.size_x, grid.size_z, grid.size_y)
            native_solid = grid.count_solid()
            if grid.size_z > max_height:
                grid = vox_io.downsample_grid_axis(grid, target_max=max_height, axis='z',
                                                   device=downsample_device or 'cpu')
            # Swap once, then use the same orientation for bytes, dimensions and preview.
            grid = vox_io.swap_yz(grid)
            data = vox_io.encode(grid, use_zstd=True, palette=palette)
            palette_png = (png_bytes(Image.fromarray(np.asarray(palette, dtype=np.uint8).reshape(-1, 1, 3)))
                           if include_palette and palette is not None else None)
            preview = self._preview_png(grid, palette) if include_preview else None
            self.pipeline.empty_cache()
            return ConvertResult(data, grid.size_x, grid.size_y, grid.size_z,
                                 grid.count_solid(), native_size, native_solid,
                                 time.monotonic()-t0, palette_png, preview, png_bytes(pre))
