#!/usr/bin/env python3
"""
CLI: TRELLIS.2 image -> VOX2 (one-shot).

Default path matches serve.bat:
  image -> TRELLIS mesh -> textured GLB -> CuMesh voxelize -> VOX2

Usage:
    img_to_vox.py input.png [output.vox] [max_resolution] [max_colors]

Loads the model in-process. For repeated conversions without reloading
weights, start server_vox.py / serve.bat and use tovox.bat instead.

Env: CONVERT_MODE=glb|direct (default direct), SEED, PIPELINE_TYPE, OUT_RES,
     MATERIAL_MODE (direct only), COLOR_MODE/VOX_FILL/SURFACE_BAND (glb path),
     TRELLIS_MODEL, ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python.exe"
if os.name != "nt":
    VENV_PY = ROOT / ".venv" / "bin" / "python"


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

from trellis_vox_runtime import TrellisVoxRuntime, bootstrap_env  # noqa: E402
os.environ.setdefault("CONVERT_MODE", "direct")

bootstrap_env()

import vox_io  # noqa: E402


def _usage() -> None:
    print(
        "Usage: img_to_vox.py input.png [output.vox] [max_resolution] [max_colors]\n"
        "  max_resolution  maximum voxel-grid dimension (1..1024, default 256)\n"
        "  max_colors      maximum solid color count (1..255, default 255)\n"
        "\n"
        "Optional env overrides: CONVERT_MODE=glb|direct, SEED, PIPELINE_TYPE,\n"
        "OUT_RES, MAX_COLORS, COLOR_MODE, VOX_FILL, SURFACE_BAND, MATERIAL_MODE,\n"
        "TRELLIS_MODEL, ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE",
        file=sys.stderr,
    )


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not args or args[0] in ("-h", "--help", "/?"):
        _usage()
        return 0 if args and args[0] in ("-h", "--help", "/?") else 1

    image_path = Path(args[0])
    out_path = Path(args[1]) if len(args) >= 2 else image_path.with_suffix(".vox")
    if len(args) > 4:
        _usage()
        return 1

    def bounded_int(value: str, name: str, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except ValueError:
            parsed = 0
        if parsed < minimum or parsed > maximum:
            print(
                f"error: {name} must be {minimum}..{maximum}, got {value}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return parsed

    pipeline_type = os.environ.get("PIPELINE_TYPE", "512")
    seed = int(os.environ.get("SEED", "0"))
    model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    material_mode = os.environ.get("MATERIAL_MODE", "color").lower()
    out_res = bounded_int(
        args[2] if len(args) >= 3 else os.environ.get("OUT_RES", "256"),
        "max_resolution",
        1,
        vox_io.VOX_SIZE_MAX,
    )
    max_colors = bounded_int(
        args[3] if len(args) >= 4 else os.environ.get("MAX_COLORS", "255"),
        "max_colors",
        1,
        vox_io.VOX_PALETTE_MAX - 1,
    )

    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        print(
            "error: 'transformers' is not installed for this Python.\n"
            f"  current: {sys.executable}\n"
            f"  use:     {VENV_PY} {Path(__file__).name}",
            file=sys.stderr,
        )
        return 2

    t0 = time.time()
    rt = TrellisVoxRuntime()
    rt.load(model=model)

    result = rt.convert(
        image_path,
        seed=seed,
        pipeline_type=pipeline_type,
        material_mode=material_mode,
        out_res=out_res,
        max_colors=max_colors,
        alpha_threshold=float(os.environ.get("ALPHA_THR", "0.5")),
        color_axis=os.environ.get("COLOR_AXIS", "auto"),
        downsample_device=os.environ.get("DOWNSAMPLE_DEVICE"),
        include_palette=True,
        include_preview=True,
        mode=os.environ.get("CONVERT_MODE", "glb"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result.vox_bytes)
    if result.palette_png:
        out_path.with_suffix(".palette.png").write_bytes(result.palette_png)
    if result.preview_png:
        out_path.with_name(out_path.stem + "_preview_xy.png").write_bytes(result.preview_png)

    # Save preprocessed image for debugging (re-open once; pipeline already ran)
    try:
        from PIL import Image

        pre = rt.pipeline.preprocess_image(Image.open(image_path))
        pre.save(out_path.with_name(out_path.stem + "_pre.png"))
    except Exception:
        pass

    rt_grid = vox_io.read(out_path)
    ok = (
        rt_grid.size_x == result.size_x
        and rt_grid.size_y == result.size_y
        and rt_grid.size_z == result.size_z
        and rt_grid.count_solid() == result.solid
    )
    nx, ny, nz = result.native_size
    print(
        f"native {nx}x{ny}x{nz} solid={result.native_solid} → "
        f"{result.size_x}x{result.size_y}x{result.size_z} solid={result.solid} "
        f"in {result.elapsed_s:.1f}s (convert)",
        flush=True,
    )
    print(
        f"wrote {out_path}  mode={result.mode}  bytes={len(result.vox_bytes)}  "
        f"roundtrip={'OK' if ok else 'FAIL'}  total={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
