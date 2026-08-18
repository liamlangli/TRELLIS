#!/usr/bin/env python3
"""
CLI: TRELLIS.2 image → VOX2 (one-shot).

For repeated conversions without reloading weights, use server_vox.py instead.
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

bootstrap_env()

import vox_io  # noqa: E402


def main() -> int:
    image_path = Path(os.environ.get("HOUSE_IMG", r"C:\Users\lilang02\Downloads\house.jpg"))
    out_path = Path(os.environ.get("HOUSE_VOX", str(ROOT / "house.vox")))
    pipeline_type = os.environ.get("PIPELINE_TYPE", "512")
    seed = int(os.environ.get("SEED", "0"))
    model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    material_mode = os.environ.get("MATERIAL_MODE", "image").lower()
    out_res = int(os.environ.get("OUT_RES", "256"))

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
        alpha_threshold=float(os.environ.get("ALPHA_THR", "0.5")),
        color_axis=os.environ.get("COLOR_AXIS", "xy"),
        downsample_device=os.environ.get("DOWNSAMPLE_DEVICE"),
        include_palette=True,
        include_preview=True,
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
        f"wrote {out_path}  bytes={len(result.vox_bytes)}  "
        f"roundtrip={'OK' if ok else 'FAIL'}  total={time.time() - t0:.1f}s",
        flush=True,
    )
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
