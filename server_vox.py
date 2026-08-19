#!/usr/bin/env python3
"""
Persistent HTTP server: load TRELLIS.2 once, convert images to VOX2 on demand.

Default local pipeline is the full path:
    image -> TRELLIS mesh -> textured GLB -> CuMesh voxelize -> VOX2

Start (project venv, GPU):
    .venv\Scripts\python.exe server_vox.py
    # or
    set HOST=0.0.0.0& set PORT=8080& .venv\Scripts\python.exe server_vox.py
    serve.bat

Endpoints:
    GET  /health              -> JSON status
    POST /convert             -> raw image body OR multipart field "image"
                              <- application/octet-stream VOX2 bytes
                              headers carry size / solid / timing metadata
    POST /convert.json        -> same input; JSON with base64 vox (+ optional palette/preview)

Query / form fields (all optional):
    mode=glb|direct           default glb (full image->GLB->VOX); direct = old shortcut
    seed, pipeline_type, out_res,
    # glb path:
    decimate_target, texture_size, simplify_target, remesh, remesh_band, remesh_project,
    vox_fill, surface_band, pad_voxels, chunk, color_mode, simplify_faces, sdf_mode,
    max_colors, crop, keep_glb,
    # direct path only:
    material_mode, alpha_threshold, color_axis, downsample_device,
    include_palette, include_preview
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import tempfile
import threading
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

# Load runtime (also bootstraps stubs/env)
sys.path.insert(0, str(ROOT))
from trellis_vox_runtime import TrellisVoxRuntime, bootstrap_env  # noqa: E402

# Default server path is full image->GLB->VOX, so refuse stubs early.
_mode = (os.environ.get("CONVERT_MODE", "glb") or "glb").strip().lower()
bootstrap_env(allow_stubs=_mode in ("direct", "fast", "mesh", "mesh_voxel", "direct_mesh"))

try:
    from flask import Flask, Request, jsonify, request, Response
except ImportError as e:
    raise SystemExit(
        "flask is required for server_vox.py.\n"
        f"  install: {VENV_PY} -m pip install flask\n"
        f"  detail: {e}"
    ) from e


RUNTIME = TrellisVoxRuntime()
APP = Flask(__name__)
APP.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "32")) * 1024 * 1024
STARTED_AT = time.time()


def _as_bool(v, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _pick(req: Request, *names: str, default=None):
    for n in names:
        if n in req.args:
            return req.args.get(n)
        if n in req.form:
            return req.form.get(n)
    return default


def _parse_convert_options(req: Request) -> dict:
    out_res = _pick(req, "out_res", "outRes", default=os.environ.get("OUT_RES", "256"))
    seed = _pick(req, "seed", default=os.environ.get("SEED", "0"))
    mode = str(
        _pick(
            req,
            "mode",
            "convert_mode",
            "convertMode",
            default=os.environ.get("CONVERT_MODE", "glb"),
        )
    ).lower()
    remesh_raw = _pick(req, "remesh", default=None)
    vox_fill_raw = _pick(req, "vox_fill", "voxFill", "fill", default=None)
    crop_raw = _pick(req, "crop", default=None)
    opts = {
        "seed": int(seed),
        "pipeline_type": str(
            _pick(req, "pipeline_type", "pipelineType", default=os.environ.get("PIPELINE_TYPE", "512"))
        ),
        "material_mode": str(
            _pick(req, "material_mode", "materialMode", default=os.environ.get("MATERIAL_MODE", "image"))
        ).lower(),
        "out_res": int(out_res),
        "alpha_threshold": float(
            _pick(req, "alpha_threshold", "alphaThreshold", default=os.environ.get("ALPHA_THR", "0.5"))
        ),
        "color_axis": str(
            _pick(req, "color_axis", "colorAxis", default=os.environ.get("COLOR_AXIS", "auto"))
        ),
        "downsample_device": _pick(req, "downsample_device", "downsampleDevice", default=None),
        "include_palette": _as_bool(_pick(req, "include_palette", "includePalette"), False),
        "include_preview": _as_bool(_pick(req, "include_preview", "includePreview"), False),
        "mode": mode,
        "keep_glb": _as_bool(_pick(req, "keep_glb", "keepGlb", "include_glb", "includeGlb"), False),
        "decimate_target": _pick(req, "decimate_target", "decimateTarget", default=None),
        "texture_size": _pick(req, "texture_size", "textureSize", default=None),
        "simplify_target": _pick(req, "simplify_target", "simplifyTarget", default=None),
        "remesh_band": _pick(req, "remesh_band", "remeshBand", default=None),
        "remesh_project": _pick(req, "remesh_project", "remeshProject", default=None),
        "surface_band": _pick(req, "surface_band", "surfaceBand", default=None),
        "pad_voxels": _pick(req, "pad_voxels", "padVoxels", default=None),
        "chunk": _pick(req, "chunk", default=None),
        "color_mode": _pick(req, "color_mode", "colorMode", default=None),
        "simplify_faces": _pick(req, "simplify_faces", "simplifyFaces", default=None),
        "sdf_mode": _pick(req, "sdf_mode", "sdfMode", default=None),
        "max_colors": _pick(req, "max_colors", "maxColors", default=None),
        "device": _pick(req, "device", default=None),
    }
    if remesh_raw is not None:
        opts["remesh"] = _as_bool(remesh_raw, False)
    if vox_fill_raw is not None:
        opts["vox_fill"] = _as_bool(vox_fill_raw, True)
    if crop_raw is not None:
        opts["crop"] = _as_bool(crop_raw, True)

    for key, caster in (
        ("decimate_target", int),
        ("texture_size", int),
        ("simplify_target", int),
        ("remesh_band", float),
        ("remesh_project", float),
        ("surface_band", float),
        ("pad_voxels", int),
        ("chunk", int),
        ("simplify_faces", int),
        ("max_colors", int),
    ):
        if opts.get(key) is not None:
            opts[key] = caster(opts[key])
    if opts.get("color_mode") is not None:
        opts["color_mode"] = str(opts["color_mode"]).lower()
    if opts.get("sdf_mode") is not None:
        opts["sdf_mode"] = str(opts["sdf_mode"])
    return opts


def _read_image_bytes(req: Request) -> bytes:
    # 1) multipart field
    if req.files:
        for key in ("image", "file", "img", "upload"):
            if key in req.files:
                data = req.files[key].read()
                if data:
                    return data
        # first file on the form
        f = next(iter(req.files.values()))
        data = f.read()
        if data:
            return data

    # 2) raw body
    raw = req.get_data(cache=False)
    if raw:
        ctype = (req.content_type or "").lower()
        # reject accidental JSON without image
        if "application/json" in ctype:
            raise ValueError(
                "JSON body not supported for image; POST raw bytes or multipart form field 'image'"
            )
        return raw

    raise ValueError("no image provided (raw body or multipart field 'image')")


def _result_headers(result) -> dict:
    nx, ny, nz = result.native_size
    return {
        "X-Size-X": str(result.size_x),
        "X-Size-Y": str(result.size_y),
        "X-Size-Z": str(result.size_z),
        "X-Solid": str(result.solid),
        "X-Native-Size": f"{nx}x{ny}x{nz}",
        "X-Native-Solid": str(result.native_solid),
        "X-Elapsed-Sec": f"{result.elapsed_s:.3f}",
        "X-Seed": str(result.seed),
        "X-Pipeline-Type": result.pipeline_type,
        "X-Material-Mode": result.material_mode,
        "X-Out-Res": str(result.out_res),
        "X-Convert-Mode": getattr(result, "mode", "glb"),
        "X-Model": RUNTIME.model_id,
        "Cache-Control": "no-store",
    }


@APP.get("/health")
@APP.get("/")
def health():
    return jsonify(
        {
            "ok": True,
            "ready": RUNTIME.ready,
            "model": RUNTIME.model_id,
            "device": RUNTIME.device_name,
            "uptime_s": round(time.time() - STARTED_AT, 1),
            "convert_mode_default": os.environ.get("CONVERT_MODE", "glb"),
            "pipeline": "image -> GLB -> VOX (mode=glb) | image -> MeshWithVoxel -> VOX (mode=direct)",
            "endpoints": {
                "GET /health": "status",
                "POST /convert": "image bytes -> VOX2 bytes (default full img->glb->vox)",
                "POST /convert.json": "image → JSON{vox_base64,...}",
            },
        }
    )


@APP.post("/convert")
def convert_binary():
    if not RUNTIME.ready:
        return jsonify({"error": "model not loaded yet"}), 503
    try:
        img = _read_image_bytes(request)
        opts = _parse_convert_options(request)
        result = RUNTIME.convert(img, **opts)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return Response(
        result.vox_bytes,
        status=200,
        mimetype="application/octet-stream",
        headers={
            **_result_headers(result),
            "Content-Disposition": 'attachment; filename="output.vox"',
            "Content-Length": str(len(result.vox_bytes)),
        },
    )


@APP.post("/convert.json")
def convert_json():
    if not RUNTIME.ready:
        return jsonify({"error": "model not loaded yet"}), 503
    try:
        img = _read_image_bytes(request)
        opts = _parse_convert_options(request)
        # JSON endpoint always returns palette/preview if asked via query; default off
        result = RUNTIME.convert(img, **opts)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    body = {
        "ok": True,
        "size": [result.size_x, result.size_y, result.size_z],
        "solid": result.solid,
        "native_size": list(result.native_size),
        "native_solid": result.native_solid,
        "elapsed_s": round(result.elapsed_s, 3),
        "seed": result.seed,
        "pipeline_type": result.pipeline_type,
        "material_mode": result.material_mode,
        "out_res": result.out_res,
        "model": RUNTIME.model_id,
        "vox_base64": base64.b64encode(result.vox_bytes).decode("ascii"),
        "vox_bytes": len(result.vox_bytes),
    }
    if result.palette_png is not None:
        body["palette_png_base64"] = base64.b64encode(result.palette_png).decode("ascii")
    if result.preview_png is not None:
        body["preview_png_base64"] = base64.b64encode(result.preview_png).decode("ascii")
    if getattr(result, "glb_bytes", None):
        body["glb_base64"] = base64.b64encode(result.glb_bytes).decode("ascii")
        body["glb_bytes"] = len(result.glb_bytes)
    resp = jsonify(body)
    for k, v in _result_headers(result).items():
        resp.headers[k] = v
    return resp


def main() -> int:
    parser = argparse.ArgumentParser(description="TRELLIS.2 image->GLB->VOX HTTP server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--model", default=os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B"))
    parser.add_argument("--low-vram", action="store_true", default=os.environ.get("LOW_VRAM", "0") == "1")
    parser.add_argument(
        "--preload",
        action="store_true",
        default=os.environ.get("PRELOAD", "1") != "0",
        help="Load weights before accepting traffic (default on). Set PRELOAD=0 to defer.",
    )
    args = parser.parse_args()
    os.environ.setdefault("CONVERT_MODE", "glb")

    if args.preload:
        print(f"[server] preloading model={args.model} …", flush=True)
        RUNTIME.load(model=args.model, low_vram=args.low_vram)
    else:
        def _bg_load():
            try:
                RUNTIME.load(model=args.model, low_vram=args.low_vram)
            except Exception as e:
                print(f"[server] background load FAILED: {e}", flush=True)

        threading.Thread(target=_bg_load, name="model-load", daemon=True).start()
        print("[server] loading model in background; /health will report ready=false until done", flush=True)

    print(f"[server] listening on http://{args.host}:{args.port}", flush=True)
    print(
        f"[server] POST /convert  image -> VOX2  default mode={os.environ.get('CONVERT_MODE', 'glb')} "
        "(full image->GLB->VOX; pass mode=direct for old shortcut)",
        flush=True,
    )
    # threaded=False: GPU conversion already serialized by runtime lock;
    # keep Flask simple (one waiters queue). Use waitress/gunicorn for prod if needed.
    APP.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
