#!/usr/bin/env python3
"""
CLI: TRELLIS.2 image -> textured GLB (one-shot).

Default path matches the web demo:
  image -> TRELLIS 1024 cascade mesh -> textured GLB

Usage:
    img_to_glb.py input.png [output.glb]

Loads the model in-process. For repeated conversions without reloading
weights, keep a server/process loaded instead of invoking this script again.

Env: SEED, RESOLUTION (512|1024|1536), DECIMATION_TARGET, TEXTURE_SIZE,
     SIMPLIFY_TARGET, REMESH (1/0), TRELLIS_MODEL, DINO_MODEL, REMBG_MODEL
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_BIN = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin")
VENV_PY = VENV_BIN / ("python.exe" if os.name == "nt" else "python")


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


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a textured GLB asset from an image using TRELLIS.2."
    )
    parser.add_argument("image", type=Path, help="input image (preferably alpha-masked)")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="output GLB path (default: input path with .glb suffix)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.image.is_file():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1
    output_path = args.output or args.image.with_suffix(".glb")

    resolution = os.environ.get("RESOLUTION", "1024")
    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }.get(resolution)
    if pipeline_type is None:
        print("error: RESOLUTION must be 512, 1024, or 1536", file=sys.stderr)
        return 1

    seed = _int_env("SEED", 0)
    decimation_target = _int_env("DECIMATION_TARGET", 500000)
    texture_size = _int_env("TEXTURE_SIZE", 2048)
    simplify_target = _int_env("SIMPLIFY_TARGET", 16777216)
    remesh = os.environ.get("REMESH", "1") != "0"

    bootstrap_env(allow_stubs=False)
    from PIL import Image  # noqa: E402
    import o_voxel  # noqa: E402
    import torch  # noqa: E402

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

    started = time.time()
    runtime = TrellisVoxRuntime()
    runtime.load(
        model=os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"),
        dino_repo=os.environ.get("DINO_MODEL"),
        rembg_name=os.environ.get("REMBG_MODEL"),
        require_full_stack=True,
    )

    with Image.open(args.image) as image:
        preprocessed = runtime.pipeline.preprocess_image(image)

    meshes = runtime.pipeline.run(
        preprocessed,
        seed=seed,
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.7,
            "rescale_t": 5.0,
        },
        shape_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 7.5,
            "guidance_rescale": 0.5,
            "rescale_t": 3.0,
        },
        tex_slat_sampler_params={
            "steps": 12,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
        pipeline_type=pipeline_type,
    )
    mesh = meshes[0]
    mesh.simplify(simplify_target)

    print(
        f"[glb] postprocess decimate={decimation_target} texture={texture_size} "
        f"remesh={remesh}",
        flush=True,
    )
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(output_path), extension_webp=True)
    torch.cuda.empty_cache()
    print(
        f"wrote {output_path}  bytes={output_path.stat().st_size}  "
        f"total={time.time() - started:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
