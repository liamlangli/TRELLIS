#!/usr/bin/env python3
"""
CLI: TRELLIS.2 image → textured GLB (full official postprocess path).

This is the trusted geometry/texture pipeline:
  preprocess → sparse structure → shape/tex SLat → MeshWithVoxel
  → mesh.simplify → o_voxel.postprocess.to_glb → .glb

Use this to verify the foundation model + CUDA mesh stack before any VOX path.

Usage:
    img_to_glb.py input.png [output.glb]
    toglb.bat a.png b.glb
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


def _usage() -> None:
    print(
        "Usage: img_to_glb.py input.png [output.glb]\n"
        "\n"
        "Env overrides:\n"
        "  SEED, PIPELINE_TYPE, TRELLIS_MODEL, LOW_VRAM,\n"
        "  DECIMATE_TARGET (default 1000000),\n"
        "  TEXTURE_SIZE    (default 2048),\n"
        "  SIMPLIFY_TARGET (default 16777216 nvdiffrast limit),\n"
        "  REMESH=0|1 (default 1), REMESH_BAND, REMESH_PROJECT\n",
        file=sys.stderr,
    )


def _require_full_stack() -> None:
    import importlib

    failures = []
    for name in ("cumesh", "flex_gemm", "nvdiffrast", "trimesh", "o_voxel"):
        try:
            mod = importlib.import_module(name)
            print(f"[glb] ok {name}: {getattr(mod, '__file__', name)}", flush=True)
        except Exception as e:
            failures.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[glb] FAIL {name}: {type(e).__name__}: {e}", flush=True)
    try:
        from o_voxel import postprocess

        if not hasattr(postprocess, "to_glb"):
            failures.append("o_voxel.postprocess.to_glb missing")
        else:
            # Reject stub implementation.
            src = getattr(postprocess.to_glb, "__code__", None)
            modname = getattr(postprocess, "__file__", "") or ""
            if "stubs" in str(modname).replace("\\", "/"):
                failures.append(f"o_voxel is stub: {modname}")
    except Exception as e:
        failures.append(f"o_voxel.postprocess: {e}")

    if failures:
        raise SystemExit(
            "Full GLB stack is not available:\n  - "
            + "\n  - ".join(failures)
            + "\n\nBuild CUDA extensions first, e.g.:\n"
            "  .ext_build\\build_ext.bat\n"
            "  .venv\\Scripts\\python.exe -m pip install nvdiffrast --no-build-isolation\n"
            "Then re-run img_to_glb.py / toglb.bat"
        )


def main() -> int:
    args = [a for a in sys.argv[1:] if a]
    if not args or args[0] in ("-h", "--help", "/?"):
        _usage()
        return 0 if args and args[0] in ("-h", "--help", "/?") else 1

    image_path = Path(args[0]).expanduser().resolve()
    out_path = (
        Path(args[1]).expanduser().resolve()
        if len(args) >= 2
        else image_path.with_suffix(".glb")
    )
    if not image_path.is_file():
        print(f"error: image not found: {image_path}", file=sys.stderr)
        return 1

    # Full stack only — no stubs for GLB.
    bootstrap_env(allow_stubs=False)
    # Make sure stubs are not sitting ahead if residual.
    sys.path[:] = [p for p in sys.path if not p.replace("\\", "/").endswith("/stubs")]
    _require_full_stack()

    import torch
    from PIL import Image
    import o_voxel

    seed = int(os.environ.get("SEED", "0"))
    pipeline_type = os.environ.get("PIPELINE_TYPE", "512")
    model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    low_vram = os.environ.get("LOW_VRAM", "0") == "1"
    decimate = int(os.environ.get("DECIMATE_TARGET", "1000000"))
    tex_size = int(os.environ.get("TEXTURE_SIZE", "2048"))
    simplify_target = int(os.environ.get("SIMPLIFY_TARGET", "16777216"))
    remesh = os.environ.get("REMESH", "1") != "0"
    remesh_band = float(os.environ.get("REMESH_BAND", "1"))
    remesh_project = float(os.environ.get("REMESH_PROJECT", "0"))

    t0 = time.time()
    rt = TrellisVoxRuntime()
    # load uses bootstrap with stubs by default; force no-stubs before / after
    bootstrap_env(allow_stubs=False)
    rt.load(model=model, low_vram=low_vram)

    pil = Image.open(image_path)
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGBA" if "A" in pil.getbands() else "RGB")

    print(
        f"[glb] running pipeline type={pipeline_type} seed={seed} image={image_path}",
        flush=True,
    )
    with rt._lock:
        mesh = rt.pipeline.run(
            pil,
            seed=seed,
            preprocess_image=True,
            pipeline_type=pipeline_type,
        )[0]

    print(
        f"[glb] mesh verts={tuple(mesh.vertices.shape)} faces={tuple(mesh.faces.shape)} "
        f"voxels={tuple(mesh.coords.shape)} attrs={tuple(mesh.attrs.shape)}",
        flush=True,
    )
    # Quick texture sanity: if base_color is rainbow noise, still export GLB so we can judge.
    try:
        bc = mesh.attrs[:, mesh.layout["base_color"]].detach().float()
        print(
            f"[glb] base_color mean={bc.mean(0).tolist()} std={bc.std(0).tolist()}",
            flush=True,
        )
    except Exception as e:
        print(f"[glb] base_color stats failed: {e}", flush=True)

    print(f"[glb] simplify target={simplify_target} ...", flush=True)
    mesh.simplify(simplify_target)

    print(
        f"[glb] to_glb decimate={decimate} texture={tex_size} remesh={remesh} ...",
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
        decimation_target=decimate,
        texture_size=tex_size,
        remesh=remesh,
        remesh_band=remesh_band,
        remesh_project=remesh_project,
        verbose=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glb.export(str(out_path), extension_webp=True)
    print(
        f"[glb] wrote {out_path}  bytes={out_path.stat().st_size}  total={time.time()-t0:.1f}s",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
