# -*- coding: utf-8 -*-
"""Comparison set: classical / MA-Net / Hough / FRFT circle lists (S1.1).

Source for Tables 1–5 frozen fits. Package-relative paths via pkg_paths.
Use --limit N for a smoke run; --skip-manet skips the MA-Net column when
weights are missing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    FITS_ROOT,
    RESULTS,
    WEIGHTS_CKPT,
    comparison_raw_path,
    setup_import_path,
)

setup_import_path()

import coherency  # noqa: E402
import run_deployed_measure as base  # noqa: E402
from NewModelDetection import NewModelDetector  # noqa: E402
from band_score import load_band  # noqa: E402
from circle_fit_helpers import _fit, _to_canvas  # noqa: E402
from run_classical_centerline import classical_valley  # noqa: E402
from run_hough_frft_look import frft_rings, hough_rings  # noqa: E402
from score_s11 import judge_s11  # noqa: E402

SESSION = DATA_COMPARISON
NAMES = ("classical", "v1.3", "Hough", "FRFT")
SUBSETS = ("Easy", "Open-arc", "Illegible")


def _origin(stem: str) -> str:
    head = stem.split("_", 1)[0]
    return "october" if len(head) == 4 else "july"


def _raw_path(stem: str) -> Path:
    return comparison_raw_path(stem)


def _mark(row: dict) -> str:
    if row["success"]:
        return "pass"
    if row.get("unformed") or row.get("refused"):
        return "unformed"
    return "fail"


def _tally(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["subset"]].append(row)
        groups["All"].append(row)
        groups[f"{row['origin']}/{row['subset']}"].append(row)
        groups[row["origin"]].append(row)
    out = {}
    for key, group in groups.items():
        out[key] = {
            "n": len(group),
            "ok": sum(1 for row in group if row["success"]),
            "unformed": sum(1 for row in group if row.get("unformed") or row.get("refused")),
        }
    return out


def _write(name: str, rows: list[dict], note: str, out: Path) -> None:
    folder = out / name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tally = _tally(rows)
    lines = [note, "Fewer than 3 rings = unformed (stays in denominator).", ""]
    for key in ("Easy", "Open-arc", "Illegible", "All", "october", "july"):
        item = tally.get(key) or {"ok": 0, "n": 0, "unformed": 0}
        lines.append(f"{key}: {item['ok']}/{item['n']}  unformed {item['unformed']}")
    lines.append("")
    for row in rows:
        lines.append(
            f"{row['stem']}: {_mark(row)}  pred={row['n_pred']} kept={row['n_kept']} "
            f"match={row['matches']}  {row['seconds']}s"
        )
    text = "\n".join(lines) + "\n"
    (folder / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def _coherency(stored: dict[str, list[dict]], out: Path) -> None:
    fits = {name: {row["stem"]: row for row in rows} for name, rows in stored.items() if name != "classical"}
    stems = [row["stem"] for row in stored.get("v1.3") or stored.get("Hough") or []]
    if not stems:
        return
    lines = ["In-band coherency on this comparison run.", ""]
    saved: dict[str, list[dict]] = {name: [] for name in fits}
    for subset in ("Easy", "Open-arc", "Illegible"):
        lines.append(f"## {subset}")
        for name in fits:
            chosen = []
            for stem in stems:
                visible = coherency._visible(stem)
                if visible["subset"] != subset and not stem.endswith("_" + subset):
                    continue
                if stem not in fits[name]:
                    continue
                frame = coherency._frame(visible, fits[name][stem])
                frame["stem"] = stem
                frame["subset"] = subset
                saved[name].append(frame)
                chosen.append(frame)
            lines.append(f"{name}: {coherency._aggregate(chosen)}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (out / "coherency_summary.txt").write_text(text, encoding="utf-8")
    (out / "coherency.json").write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text, flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="Process at most N canvases (0 = all).")
    parser.add_argument(
        "--skip-manet",
        action="store_true",
        help="Skip MA-Net column (no weight required).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: results/fits, or results/smoke_comparison when --limit).",
    )
    args = parser.parse_args(argv)

    canvases = sorted(SESSION.glob("*_canvas.png"))
    if args.limit and args.limit > 0:
        canvases = canvases[: args.limit]
    missing = [stem for path in canvases if not _raw_path(stem := path.name.replace("_canvas.png", "")).is_file()]
    if missing:
        raise SystemExit("Missing raw images: " + ", ".join(missing))

    skip_manet = bool(args.skip_manet) or not WEIGHTS_CKPT.is_file()
    if skip_manet and not args.skip_manet:
        print(f"WARNING: {WEIGHTS_CKPT} missing — skipping MA-Net column.", flush=True)

    out = args.out
    if out is None:
        out = RESULTS / "smoke_comparison" if (args.limit and args.limit > 0) else FITS_ROOT
    out.mkdir(parents=True, exist_ok=True)
    detector = None
    if not skip_manet:
        print(f"Loading student model {WEIGHTS_CKPT.name}  n={len(canvases)}", flush=True)
        detector = NewModelDetector(checkpoint_path=WEIGHTS_CKPT, strict=False, repo_root=base._REPO)

    active = [n for n in NAMES if not (skip_manet and n == "v1.3")]
    stored: dict[str, list[dict]] = {name: [] for name in active}
    for index, canvas_path in enumerate(canvases, 1):
        stem = canvas_path.name.replace("_canvas.png", "")
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        raw = cv2.imread(str(_raw_path(stem)), cv2.IMREAD_COLOR)
        record = json.loads(canvas_path.with_name(stem + ".json").read_text(encoding="utf-8"))
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        subset = str(record.get("subset") or stem.split("_", 1)[-1])
        if subset == "Broken":
            subset = "Illegible"
        origin = _origin(stem)
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

        fits: dict[str, tuple] = {}
        if detector is not None:
            started = time.perf_counter()
            prob, _contours = detector.get_refined_skeleton(raw)
            prepared = detector.inference.last_preprocess
            mapping = prepared.mapping
            center_m, radii_m = _fit(prob, prepared.model_u8, nocap=False, physics=None, blend=0.0)
            center_v, radii_v = _to_canvas(center_m, radii_m, mapping, canvas.shape[0], mapping.crop_side)
            fits["v1.3"] = (center_v, radii_v, time.perf_counter() - started)

        started = time.perf_counter()
        valley = classical_valley(gray)
        center_c, radii_c = _fit(valley, gray, nocap=False, physics=None, blend=0.0)
        fits["classical"] = (center_c, radii_c, time.perf_counter() - started)

        started = time.perf_counter()
        center_h, radii_h = hough_rings(canvas)
        fits["Hough"] = (center_h, radii_h, time.perf_counter() - started)

        started = time.perf_counter()
        center_f, radii_f = frft_rings(canvas)
        fits["FRFT"] = (center_f, radii_f, time.perf_counter() - started)

        marks = []
        for name, (center, radii, elapsed) in fits.items():
            judged = judge_s11(human, list(radii), center)
            row = {
                "stem": stem,
                "subset": subset,
                "origin": origin,
                "seconds": round(float(elapsed), 3),
                "n_human": human["n"],
                "n_pred": len(list(radii)),
                "center_xy": None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)],
                "radii_px": [round(float(radius), 2) for radius in radii],
                **judged,
            }
            stored[name].append(row)
            marks.append(f"{name} {_mark(row)}")
        print(f"[{index}/{len(canvases)}] {stem}  " + "  ".join(marks), flush=True)

    notes = {
        "classical": "Classical valley on analysis canvas → frozen v1.3 assembly.",
        "v1.3": "MA-Net on raw frame → same assembly; physics blend 0.",
        "Hough": "Circle Hough on canvas gray (Canny 35/110).",
        "FRFT": "Dark-minima spectral search on canvas gray.",
    }
    for name, rows in stored.items():
        _write(name, rows, notes[name], out)
    _coherency(stored, out)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
