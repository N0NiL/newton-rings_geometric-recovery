# -*- coding: utf-8 -*-
"""评分常数敏感性：对 measure_comparison 已存圆做 S1.2 重判（不重拟合）。

先用默认常数复现表 1（69 = 24 Easy + 45 Open-arc；排除 Illegible）：
classical 57/58/58/58，v1.3 47/57/59/60，Hough 0/1/1/1，FRFT 30/39/41/50
（门 0.5/1/2 Λ/不测圆心）。随后逐项扰动 10 个设置。
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    FITS_ROOT,
    GATES,
    NAMES,
    READABLE,
    RESULTS,
    TABLE1_EXPECTED,
    setup_import_path,
)

setup_import_path()

from band_score import load_band  # noqa: E402
import score_s11  # noqa: E402
import score_s12  # noqa: E402
from score_s12 import judge_s12  # noqa: E402

SESSION = DATA_COMPARISON
SRC = FITS_ROOT
OUT = RESULTS / "score_sensitivity"
EXPECTED = TABLE1_EXPECTED

SETTINGS = [
    ("default", {}),
    ("unmatched_0.24", {"s11.UNMATCHED_LIMIT": 0.24}),
    ("unmatched_0.36", {"s11.UNMATCHED_LIMIT": 0.36}),
    ("dense_0.60", {"s11.DENSE_SCALE": 0.60}),
    ("dense_0.90", {"s11.DENSE_SCALE": 0.90}),
    ("matchwin_0.32", {"s11.MATCH_WINDOW_SCALE": 0.32}),
    ("matchwin_0.48", {"s11.MATCH_WINDOW_SCALE": 0.48}),
    ("dispersion_0.32", {"s12.DISPERSION_SCALE": 0.32}),
    ("dispersion_0.48", {"s12.DISPERSION_SCALE": 0.48}),
    ("bandpad_1", {"s11.BAND_PAD_PX": 1.0, "s12.BAND_PAD_PX": 1.0}),
    ("bandpad_3", {"s11.BAND_PAD_PX": 3.0, "s12.BAND_PAD_PX": 3.0}),
]

_MODULES = {"s11": score_s11, "s12": score_s12}


@contextmanager
def patched(overrides: dict[str, float]):
    saved = []
    for key, value in overrides.items():
        mod_name, attr = key.split(".", 1)
        mod = _MODULES[mod_name]
        saved.append((mod, attr, getattr(mod, attr)))
        setattr(mod, attr, value)
    try:
        yield
    finally:
        for mod, attr, old in saved:
            setattr(mod, attr, old)


def load_frames() -> dict[str, list[dict]]:
    """每条路线 69 个可读帧：human 标注 + 已存圆。"""
    frames: dict[str, list[dict]] = {}
    for name in NAMES:
        rows = json.loads((SRC / name / "results.json").read_text(encoding="utf-8"))
        entries = []
        for row in rows:
            if row["subset"] not in READABLE:
                continue
            stem = row["stem"]
            human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
            entries.append(
                {
                    "stem": stem,
                    "human": human,
                    "radii": list(row.get("radii_px") or []),
                    "center": None if row.get("center_xy") is None else tuple(row["center_xy"]),
                }
            )
        frames[name] = entries
    return frames


def passes_at(frames: dict[str, list[dict]], gate: float | None) -> dict[str, int]:
    out = {}
    for name in NAMES:
        ok = 0
        for entry in frames[name]:
            score = judge_s12(entry["human"], entry["radii"], entry["center"])
            if not score["success"]:
                continue
            if gate is None:
                ok += 1
                continue
            spacing = float(entry["human"]["spacing"])
            err = score.get("center_error_px")
            if err is not None and spacing > 1e-6 and float(err) <= gate * spacing + 1e-9:
                ok += 1
        out[name] = ok
    return out


def order_ok(p05: dict[str, int], p1: dict[str, int]) -> tuple[bool, bool]:
    bs, mn, fr, ho = p05["classical"], p05["v1.3"], p05["FRFT"], p05["Hough"]
    o5 = bs >= mn > fr > ho
    b1, m1, f1, h1 = p1["classical"], p1["v1.3"], p1["FRFT"], p1["Hough"]
    o1 = b1 > f1 > h1 and m1 > f1 > h1
    return o5, o1


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    counts = {n: len(frames[n]) for n in NAMES}
    print("readable frames per route:", counts, flush=True)
    assert all(v == 69 for v in counts.values()), counts

    # 1. 默认常数复现表 1
    baseline = {}
    for gate in GATES:
        baseline[gate] = passes_at(frames, gate)
    ok = True
    for name in NAMES:
        got = [baseline[g][name] for g in GATES]
        if got != EXPECTED[name]:
            ok = False
            print(f"MISMATCH {name}: got {got} expected {EXPECTED[name]}", flush=True)
    if not ok:
        raise SystemExit("默认常数未复现表 1，停止。")
    print("baseline reproduces Table 1:", {g: baseline[g] for g in GATES}, flush=True)

    # 2. 逐项扰动
    results = {"baseline": {"gates": {str(g): baseline[g] for g in GATES}}}
    order_exceptions = []
    for label, overrides in SETTINGS[1:]:
        with patched(overrides):
            p05 = passes_at(frames, 0.5)
            p1 = passes_at(frames, 1.0)
        o5, o1 = order_ok(p05, p1)
        results[label] = {"overrides": overrides, "passes_0p5": p05, "passes_1p0": p1,
                          "order_0p5_ok": o5, "order_1p0_ok": o1}
        if not (o5 and o1):
            order_exceptions.append((label, o5, o1))
        print(f"{label}: 0.5L {p05} | 1L {p1} | order05={o5} order1={o1}", flush=True)

    # 3. 每路线在 10 个扰动下的 min–max（不含 default）
    perturbed = [label for label, _ in SETTINGS[1:]]
    ranges = {}
    for name in NAMES:
        v05 = [results[l]["passes_0p5"][name] for l in perturbed]
        v1 = [results[l]["passes_1p0"][name] for l in perturbed]
        ranges[name] = {
            "0.5L": {"min": min(v05), "max": max(v05)},
            "1.0L": {"min": min(v1), "max": max(v1)},
        }
    results["ranges_across_perturbations"] = ranges
    results["order_exceptions"] = order_exceptions
    results["expected_table1"] = EXPECTED
    (OUT / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 评分常数敏感性（69 可读帧，S1.2 重判已存圆）",
        "",
        "基线复现表 1：0.5/1/2/不测 Λ —— "
        + "; ".join(f"{n} {EXPECTED[n]}" for n in NAMES),
        "",
        "## 各设置通过数（0.5 Λ / 1 Λ）",
        "",
        "| 设置 | BS 0.5 | BS 1 | MA-Net 0.5 | MA-Net 1 | Hough 0.5 | Hough 1 | FRFT 0.5 | FRFT 1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gate_names = {"classical": "BS", "v1.3": "MA-Net", "Hough": "Hough", "FRFT": "FRFT"}
    for label, _ov in SETTINGS:
        if label == "default":
            p05 = {n: baseline[0.5][n] for n in NAMES}
            p1 = {n: baseline[1.0][n] for n in NAMES}
        else:
            p05 = results[label]["passes_0p5"]
            p1 = results[label]["passes_1p0"]
        lines.append(
            f"| {label} | {p05['classical']} | {p1['classical']} | {p05['v1.3']} | {p1['v1.3']} | "
            f"{p05['Hough']} | {p1['Hough']} | {p05['FRFT']} | {p1['FRFT']} |"
        )
    lines += [
        "",
        "## 每路线在 10 个扰动设置下的 min–max",
        "",
        "| 路线 | 0.5 Λ min | 0.5 Λ max | 1 Λ min | 1 Λ max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in NAMES:
        r = ranges[name]
        lines.append(f"| {gate_names[name]} | {r['0.5L']['min']} | {r['0.5L']['max']} | {r['1.0L']['min']} | {r['1.0L']['max']} |")
    lines += ["", "## 排序保持", ""]
    if order_exceptions:
        lines.append("排序被破坏的设置：")
        for label, o5, o1 in order_exceptions:
            lines.append(f"- {label}: 0.5Λ 序 {'OK' if o5 else 'FAIL'}；1Λ 序 {'OK' if o1 else 'FAIL'}")
    else:
        lines.append("全部 10 个扰动设置下排序均保持：0.5 Λ 时 BS ≥ MA-Net > FRFT > Hough；1 Λ 时 {BS, MA-Net} > FRFT > Hough。")
    lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
