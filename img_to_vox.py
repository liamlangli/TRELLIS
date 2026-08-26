#!/usr/bin/env python3
"""
CLI: TRELLIS.2 image -> VOX2 (one-shot).

Default path matches serve.bat:
  image -> TRELLIS mesh -> textured GLB -> CuMesh voxelize -> VOX2

Usage:
    img_to_vox.py -i input.png [-o output.vox] [-h max_height] [--max_colors n]
    img_to_vox.py --input_folder folder [-h max_height] [--max_colors n] [--skip]

When --input_folder is used, each image may have a sibling ``<stem>.conf``
file with ``max_height = N`` / ``max_colors = N`` lines that override the
CLI defaults for that image only.

With ``--skip``, an existing target ``.vox`` is left as-is and that image is
not converted again.

Loads the model in-process. For repeated conversions without reloading
weights, start server_vox.py / serve.bat and use tovox.bat instead.

Env: CONVERT_MODE=glb|direct (default direct), SEED, PIPELINE_TYPE, MAX_HEIGHT,
     MATERIAL_MODE (direct only), COLOR_MODE/VOX_FILL/SURFACE_BAND (glb path),
     TRELLIS_MODEL, ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE
"""

from __future__ import annotations

import argparse
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
        "Usage:\n"
        "  img_to_vox.py -i input.png [-o output.vox] [-h max_height] [--max_colors n]\n"
        "  img_to_vox.py --input_folder folder [-h max_height] [--max_colors n] [--skip]\n"
        "\n"
        "  --input_folder writes every PNG/JPG/JPEG to folder/vox/*.vox\n"
        "                 per-image <stem>.conf may override max_height / max_colors\n"
        "                 (key = value lines, e.g. `max_height = 128`)\n"
        "  -h, --max_height  maximum VOX Y-axis resolution (1..1024, default 256)\n"
        "  --max_colors      maximum solid color count (1..255, default 220)\n"
        "  --skip            skip conversion when the target .vox already exists\n"
        "\n"
        "Optional env overrides: CONVERT_MODE=glb|direct, SEED, PIPELINE_TYPE,\n"
        "MAX_HEIGHT, MAX_COLORS, COLOR_MODE, VOX_FILL, SURFACE_BAND, MATERIAL_MODE,\n"
        "TRELLIS_MODEL, ALPHA_THR, COLOR_AXIS, DOWNSAMPLE_DEVICE",
        file=sys.stderr,
    )


def _bounded_int(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed < minimum or parsed > maximum:
        print(
            f"error: {name} must be {minimum}..{maximum}, got {value}",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return parsed


def _load_image_conf(image_path: Path) -> dict:
    """Load per-image config (same stem, .conf extension).

    Format: simple ``key = value`` lines. ``#`` starts a comment.
    Recognized keys: ``max_height``, ``max_colors``. Unknown keys are ignored.
    Returns an empty dict if the file is absent or unreadable.
    """
    conf_path = image_path.with_suffix(".conf")
    if not conf_path.is_file():
        return {}
    result: dict = {}
    try:
        text = conf_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"warning: cannot read {conf_path}: {exc}", file=sys.stderr)
        return {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line:
            key, _, value = line.partition("=")
        elif ":" in line:
            key, _, value = line.partition(":")
        else:
            continue
        key = key.strip().lower()
        value = value.strip()
        if key in {"max_height", "max_colors"} and value:
            result[key] = value
        else:
            # unknown / empty - skip silently but keep a debug hint
            if key not in {"max_height", "max_colors"}:
                continue
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert an image or image folder to VOX using TRELLIS.2.",
        add_help=False,
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("-i", "--input", type=Path, help="input PNG/JPG/JPEG image")
    inputs.add_argument(
        "--input_folder",
        type=Path,
        help="convert all PNG/JPG/JPEG images in this folder",
    )
    parser.add_argument("-o", "--output", type=Path, help="output VOX path (single input only)")
    parser.add_argument(
        "-h",
        "--max_height",
        default=os.environ.get("MAX_HEIGHT", "256"),
        help="maximum VOX Y-axis resolution (1..1024, default: 256)",
    )
    parser.add_argument(
        "--max_colors",
        default=os.environ.get("MAX_COLORS", "220"),
        help="maximum solid color count (1..255, default: 220)",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="skip conversion when the target .vox already exists",
    )
    parser.add_argument("--help", action="store_true", help="show this help and exit")
    args = parser.parse_args()
    if args.help:
        parser.print_help()
        raise SystemExit(0)
    if args.input is None and args.input_folder is None:
        parser.error("one of the arguments -i/--input --input_folder is required")
    if args.input_folder is not None and args.output is not None:
        parser.error("--output cannot be combined with --input_folder")
    return args


def _convert_one(
    rt: TrellisVoxRuntime,
    image_path: Path,
    out_path: Path,
    *,
    pipeline_type: str,
    seed: int,
    material_mode: str,
    max_height: int,
    max_colors: int,
) -> bool:
    t0 = time.time()
    result = rt.convert(
        image_path,
        seed=seed,
        pipeline_type=pipeline_type,
        material_mode=material_mode,
        max_height=max_height,
        max_colors=max_colors,
        alpha_threshold=float(os.environ.get("ALPHA_THR", "0.5")),
        color_axis=os.environ.get("COLOR_AXIS", "auto"),
        downsample_device=os.environ.get("DOWNSAMPLE_DEVICE"),
        include_palette=True,
        include_preview=True,
        mode=os.environ.get("CONVERT_MODE", "direct"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(result.vox_bytes)
    if result.palette_png:
        out_path.with_suffix(".palette.png").write_bytes(result.palette_png)
    if result.preview_png:
        out_path.with_name(out_path.stem + "_preview_xy.png").write_bytes(result.preview_png)

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
    return ok


def main() -> int:
    args = _parse_args()
    pipeline_type = os.environ.get("PIPELINE_TYPE", "512")
    seed = int(os.environ.get("SEED", "0"))
    model = os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS.2-4B")
    material_mode = os.environ.get("MATERIAL_MODE", "color").lower()
    max_height = _bounded_int(
        args.max_height,
        "max_height",
        1,
        vox_io.VOX_SIZE_MAX,
    )
    max_colors = _bounded_int(
        args.max_colors,
        "max_colors",
        1,
        vox_io.VOX_PALETTE_MAX - 1,
    )

    if args.input is not None:
        if not args.input.is_file():
            print(f"error: image not found: {args.input}", file=sys.stderr)
            return 1
        image_paths = [args.input]
        output_paths = [args.output or args.input.with_suffix(".vox")]
    else:
        if not args.input_folder.is_dir():
            print(f"error: input folder not found: {args.input_folder}", file=sys.stderr)
            return 1
        image_paths = sorted(
            path
            for path in args.input_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if not image_paths:
            print(f"error: no PNG/JPG/JPEG images found in {args.input_folder}", file=sys.stderr)
            return 1
        vox_dir = args.input_folder / "vox"
        output_paths = [vox_dir / f"{path.stem}.vox" for path in image_paths]

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

    all_ok = True
    skipped = 0
    converted = 0
    for index, (image_path, out_path) in enumerate(zip(image_paths, output_paths), start=1):
        if len(image_paths) > 1:
            print(f"=== [{index}/{len(image_paths)}] {image_path.name} ===", flush=True)

        if args.skip and out_path.is_file():
            skipped += 1
            print(f"skip: exists {out_path}", flush=True)
            continue

        image_max_height = max_height
        image_max_colors = max_colors
        if args.input_folder is not None:
            conf = _load_image_conf(image_path)
            if "max_height" in conf:
                image_max_height = _bounded_int(
                    conf["max_height"], f"{image_path.stem}.conf:max_height",
                    1, vox_io.VOX_SIZE_MAX,
                )
            if "max_colors" in conf:
                image_max_colors = _bounded_int(
                    conf["max_colors"], f"{image_path.stem}.conf:max_colors",
                    1, vox_io.VOX_PALETTE_MAX - 1,
                )
            if conf:
                print(
                    f"  conf: max_height={image_max_height} max_colors={image_max_colors}",
                    flush=True,
                )

        try:
            all_ok &= _convert_one(
                rt,
                image_path,
                out_path,
                pipeline_type=pipeline_type,
                seed=seed,
                material_mode=material_mode,
                max_height=image_max_height,
                max_colors=image_max_colors,
            )
            converted += 1
        except Exception as exc:
            all_ok = False
            print(f"error: failed to convert {image_path}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if args.skip:
        print(
            f"batch total={time.time() - t0:.1f}s  converted={converted} skipped={skipped}",
            flush=True,
        )
    else:
        print(f"batch total={time.time() - t0:.1f}s", flush=True)
    return 0 if all_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
