# -*- coding: utf-8 -*-
"""复现正文表 1 通过数与第 3.6 节多透镜 R（median Δ(r²)×s²/λ）。"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "bench"))

from pkg_paths import (  # noqa: E402
    DATA_MULTI_LENS,
    GATES,
    LAMBDA_M,
    NAMES,
    SCALE_M_PER_PX,
    TABLE1_EXPECTED,
    setup_import_path,
)

setup_import_path()

from table1 import load_readable_frames, table1_matrix  # noqa: E402


def _lens_r(radii: list[float], meters_per_px: float | None = None) -> float | None:
    """相邻半径视为连续暗级。R = median(Δ(r²)) × s² / λ。"""
    scale = float(meters_per_px) if meters_per_px else SCALE_M_PER_PX
    vals = sorted(float(r) for r in radii if float(r) > 0)
    if len(vals) < 2:
        return None
    gaps = [b * b - a * a for a, b in zip(vals, vals[1:]) if b * b > a * a]
    if not gaps:
        return None
    return statistics.median(gaps) * (scale**2) / LAMBDA_M


def _multi_lens() -> None:
    path = DATA_MULTI_LENS / "confidence_db_multilens.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    print("[Sec 3.6 multi-lens] R = median(Δ(r²)) × s²/λ  from fit.rings")
    print(f"{'idx':>3}  {'role':<28}  {'n':>3}  {'R_m':>7}  paper_group")
    groups: dict[str, list[float]] = {}
    for s in samples:
        radii = [float(r["radius_px"]) for r in (s.get("fit") or {}).get("rings") or []]
        R = _lens_r(radii, s.get("meters_per_px"))
        role = str(s.get("role") or s.get("role_zh") or "")
        # 正文：样品1两帧中位 1.091；样品2两帧中位 1.075；原透镜复装 1.916
        if s["index"] in (1, 2):
            group = "specimen1"
        elif s["index"] in (3, 4):
            group = "specimen2"
        else:
            group = "original_reinstall"
        if R is not None:
            groups.setdefault(group, []).append(R)
        print(
            f"{s['index']:3d}  {role:<28}  {len(radii):3d}  "
            f"{(R if R is not None else float('nan')):7.3f}  {group}"
        )
    paper = {
        "specimen1": 1.091,
        "specimen2": 1.075,
        "original_reinstall": 1.916,
    }
    print("— paper column-4 (group medians) —")
    for key, target in paper.items():
        vals = groups.get(key) or []
        med = statistics.median(vals) if vals else None
        mark = "OK" if med is not None and abs(med - target) < 0.002 else "CHECK"
        print(f"  {key}: median={med:.3f} m  paper={target:.3f} m  [{mark}]")


def _table1() -> int:
    print("[Table 1] S1.2 structure + center gates on results/fits/*")
    frames = load_readable_frames()
    n = {name: len(frames[name]) for name in NAMES}
    print(f"  readable frames: {n}")
    if any(v != 69 for v in n.values()):
        print("  WARN: expected 69 readable frames per method")
    matrix = table1_matrix(frames)
    headers = ["0.5Λ", "1.0Λ", "2.0Λ", "none"]
    print(f"  {'method':<12} " + " ".join(f"{h:>6}" for h in headers) + "   paper")
    ok_all = True
    for name in NAMES:
        got = matrix[name]
        exp = TABLE1_EXPECTED[name]
        mark = "OK" if got == exp else "MISMATCH"
        if got != exp:
            ok_all = False
        print(
            f"  {name:<12} "
            + " ".join(f"{v:6d}" for v in got)
            + f"   {exp} [{mark}]"
        )
    print(f"  gate order: {GATES}")
    print("Table 1 match." if ok_all else "Table 1 MISMATCH — see counts above.")
    return 0 if ok_all else 1


def _extras() -> None:
    db = ROOT / "results" / "dual_blind" / "annotator_AB_pole_delta.json"
    if db.is_file():
        s = json.loads(db.read_text(encoding="utf-8"))["summary"]
        print(
            f"[Sec 3.2 second-annotator] n={s.get('n_scored')} "
            f"median={s.get('median_err_s')} Λ "
            f"within_1s={s.get('within_1s')}/{s.get('n_scored')}"
        )
    p2 = ROOT / "results" / "c1_c4" / "part2_results.json"
    if p2.is_file():
        rg = json.loads(p2.read_text(encoding="utf-8")).get("radial_scaling_generalization") or {}
        print("[Sec 3.6 gamma IQR]")
        for k, v in rg.items():
            if isinstance(v, dict):
                print(f"  {k}: median={v.get('R_median_m')} IQR={v.get('R_iqr_m')}")
    for name in ("timing.json", "static.json"):
        p = ROOT / "results" / name
        print(f"[{name}] {'ok' if p.is_file() else 'MISSING'}")


def main() -> int:
    print(f"ROOT = {ROOT}")
    _multi_lens()
    print()
    rc = _table1()
    print()
    _extras()
    return rc


if __name__ == "__main__":
    sys.exit(main())
