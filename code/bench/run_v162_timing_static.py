# -*- coding: utf-8 -*-
"""Retimed four methods on the comparison set + fits on the stationary sequence.

Tables 6–8. Does not overwrite frozen measure scores under results/fits/.
Package-relative paths via pkg_paths. Use --limit N for a smoke run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    DATA_STATIONARY,
    RESULTS,
    WEIGHTS_CKPT,
    comparison_raw_path,
    setup_import_path,
)

setup_import_path()

import run_deployed_measure as base  # noqa: E402
from NewModelDetection import NewModelDetector  # noqa: E402
from circle_fit_helpers import _fit, _to_canvas  # noqa: E402
from run_classical_centerline import classical_valley  # noqa: E402
from run_hough_frft_look import frft_rings, hough_rings  # noqa: E402

SESSION = DATA_COMPARISON
STATIC = DATA_STATIONARY
NAMES = ("classical", "v1.3", "Hough", "FRFT")


def _one(detector, raw, canvas, *, skip_manet: bool = False) -> dict:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    packed: dict[str, tuple] = {}
    if detector is not None and not skip_manet:
        started = time.perf_counter()
        prob, _contours = detector.get_refined_skeleton(raw)
        prepared = detector.inference.last_preprocess
        mapping = prepared.mapping
        center_m, radii_m = _fit(prob, prepared.model_u8, nocap=False, physics=None, blend=0.0)
        center_v, radii_v = _to_canvas(center_m, radii_m, mapping, canvas.shape[0], mapping.crop_side)
        packed["v1.3"] = (center_v, radii_v, time.perf_counter() - started)
    started = time.perf_counter()
    valley = classical_valley(gray)
    center_c, radii_c = _fit(valley, gray, nocap=False, physics=None, blend=0.0)
    packed["classical"] = (center_c, radii_c, time.perf_counter() - started)
    started = time.perf_counter()
    center_h, radii_h = hough_rings(canvas)
    packed["Hough"] = (center_h, radii_h, time.perf_counter() - started)
    started = time.perf_counter()
    center_f, radii_f = frft_rings(canvas)
    packed["FRFT"] = (center_f, radii_f, time.perf_counter() - started)
    out = {}
    for name, (center, radii, elapsed) in packed.items():
        radii = [float(r) for r in radii]
        out[name] = {
            "seconds": round(float(elapsed), 3),
            "n": len(radii),
            "unformed": len(radii) < 3 or center is None,
            "center_xy": None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)],
            "radii_px": [round(float(r), 2) for r in radii],
        }
    return out


def _timing(detector, *, limit: int, skip_manet: bool) -> list[dict]:
    rows = []
    canvases = sorted(SESSION.glob("*_canvas.png"))
    if limit > 0:
        canvases = canvases[:limit]
    for index, canvas_path in enumerate(canvases, 1):
        stem = canvas_path.name.replace("_canvas.png", "")
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(comparison_raw_path(stem)), cv2.IMREAD_COLOR)
        record = json.loads(canvas_path.with_name(stem + ".json").read_text(encoding="utf-8"))
        subset = str(record.get("subset") or stem.split("_", 1)[-1])
        if subset == "Broken":
            subset = "Illegible"
        fitted = _one(detector, raw, canvas, skip_manet=skip_manet)
        rows.append({"stem": stem, "subset": subset, "fits": fitted})
        brief = "  ".join(f"{name} {fitted[name]['seconds']:.2f}s" for name in fitted)
        print(f"[time {index}/{len(canvases)}] {stem}  {brief}", flush=True)
    return rows


def _static(detector, *, limit: int, skip_manet: bool) -> list[dict]:
    rows = []
    canvases = sorted(STATIC.glob("*_canvas.png"))
    if limit > 0:
        canvases = canvases[:limit]
    for index, canvas_path in enumerate(canvases, 1):
        stem = canvas_path.name.replace("_canvas.png", "")
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        raw_path = canvas_path.with_name(stem + "_raw.png")
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        fitted = _one(detector, raw, canvas, skip_manet=skip_manet)
        rows.append({"stem": stem, "fits": fitted})
        brief = "  ".join(
            f"{name} n={fitted[name]['n']} {fitted[name]['seconds']:.2f}s" for name in fitted
        )
        print(f"[static {index}/{len(canvases)}] {stem}  {brief}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="At most N frames per stage (0 = all).")
    parser.add_argument("--skip-manet", action="store_true", help="Skip MA-Net timing/fits.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: results/, or results/smoke_v162 when --limit).",
    )
    args = parser.parse_args(argv)

    skip_manet = bool(args.skip_manet) or not WEIGHTS_CKPT.is_file()
    if skip_manet and not args.skip_manet:
        print(f"WARNING: {WEIGHTS_CKPT} missing — skipping MA-Net.", flush=True)

    out = args.out
    if out is None:
        out = RESULTS / "smoke_v162" if (args.limit and args.limit > 0) else RESULTS
    out.mkdir(parents=True, exist_ok=True)

    detector = None
    if not skip_manet:
        print(f"Loading student model {WEIGHTS_CKPT.name}", flush=True)
        detector = NewModelDetector(checkpoint_path=WEIGHTS_CKPT, strict=False, repo_root=base._REPO)

    timing = _timing(detector, limit=args.limit, skip_manet=skip_manet)
    static = _static(detector, limit=args.limit, skip_manet=skip_manet)
    (out / "timing.json").write_text(json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "static.json").write_text(json.dumps(static, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
