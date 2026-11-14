"""比较第一/第二标注员（second-annotator）外推极心偏差，折合为条纹间距 Λ。

A：data/comparison/{stem}_human.json
B：data/dual_blind_b15/{stem}_human_B.json
Λ：该帧 A 标注牛顿半径相邻差的中位数（与正文圆心门同口径；勿与物方 pixel scale 混淆）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))
from pkg_paths import DATA_COMPARISON, DATA_DUAL_BLIND, RESULTS, setup_import_path  # noqa: E402

setup_import_path()
_A_ROOT = DATA_COMPARISON
_B_ROOT = DATA_DUAL_BLIND
_OUT = RESULTS / "dual_blind"


def _spacing(radii: list[float]) -> float:
    if len(radii) < 2:
        return float("nan")
    gaps = np.diff(sorted(float(r) for r in radii))
    gaps = gaps[gaps > 0.5]
    if len(gaps) == 0:
        return float("nan")
    return float(np.median(gaps))


def _newton_center(payload: dict) -> tuple[list[float] | None, list[float]]:
    rings = [ring for ring in (payload.get("rings") or []) if ring.get("role", "newton") == "newton"]
    radii = [float(ring["radius_px"]) for ring in rings if ring.get("radius_px") is not None]
    center = payload.get("center_xy")
    if center is None and rings:
        c0 = rings[0].get("center_xy")
        if c0:
            center = c0
    if center is None:
        return None, radii
    return [float(center[0]), float(center[1])], radii


def main() -> int:
    manifest_path = _B_ROOT / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"缺少 {_B_ROOT}/manifest.json，请先 prepare_dual_blind_set.py")
    stems = list(json.loads(manifest_path.read_text(encoding="utf-8"))["stems"])
    rows: list[dict] = []
    missing_b: list[str] = []

    for stem in stems:
        path_a = _A_ROOT / f"{stem}_human.json"
        path_b = _B_ROOT / f"{stem}_human_B.json"
        if not path_a.is_file():
            raise SystemExit(f"缺少第一标注员：{path_a}")
        if not path_b.is_file():
            missing_b.append(stem)
            continue
        a = json.loads(path_a.read_text(encoding="utf-8"))
        b = json.loads(path_b.read_text(encoding="utf-8"))
        ca, ra = _newton_center(a)
        cb, rb = _newton_center(b)
        spacing = _spacing(ra)
        if ca is None or cb is None or not np.isfinite(spacing) or spacing <= 1e-6:
            rows.append(
                {
                    "stem": stem,
                    "ok": False,
                    "reason": "缺圆心或无法估计条纹间距",
                    "center_A": ca,
                    "center_B": cb,
                    "spacing_px": None if not np.isfinite(spacing) else round(spacing, 3),
                    "n_newton_A": len(ra),
                    "n_newton_B": len(rb),
                }
            )
            continue
        err_px = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
        err_s = err_px / spacing
        rows.append(
            {
                "stem": stem,
                "ok": True,
                "center_A": [round(ca[0], 2), round(ca[1], 2)],
                "center_B": [round(cb[0], 2), round(cb[1], 2)],
                "err_px": round(err_px, 3),
                "spacing_px": round(spacing, 3),
                "err_in_s": round(err_s, 4),
                "within_0p5s": bool(err_s <= 0.5 + 1e-9),
                "within_1s": bool(err_s <= 1.0 + 1e-9),
                "within_2s": bool(err_s <= 2.0 + 1e-9),
                "n_newton_A": len(ra),
                "n_newton_B": len(rb),
            }
        )

    ok_rows = [row for row in rows if row.get("ok")]
    errs = [float(row["err_in_s"]) for row in ok_rows]
    summary = {
        "n_manifest": len(stems),
        "n_scored": len(ok_rows),
        "n_missing_B": len(missing_b),
        "missing_B": missing_b,
        "mean_err_s": None if not errs else round(float(np.mean(errs)), 4),
        "median_err_s": None if not errs else round(float(np.median(errs)), 4),
        "max_err_s": None if not errs else round(float(np.max(errs)), 4),
        "within_0p5s": sum(1 for row in ok_rows if row["within_0p5s"]),
        "within_1s": sum(1 for row in ok_rows if row["within_1s"]),
        "within_2s": sum(1 for row in ok_rows if row["within_2s"]),
    }
    _OUT.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "rows": rows}
    out_json = _OUT / "annotator_AB_pole_delta.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 双盲标注员极心一致性",
        "",
        f"- 清单帧数：{summary['n_manifest']}",
        f"- 已评分：{summary['n_scored']}（缺 B：{summary['n_missing_B']}）",
        f"- 偏差均值 / 中位 / 最大（单位 s）："
        f"{summary['mean_err_s']} / {summary['median_err_s']} / {summary['max_err_s']}",
        f"- ≤0.5 s：{summary['within_0p5s']}/{summary['n_scored']}",
        f"- ≤1.0 s：{summary['within_1s']}/{summary['n_scored']}",
        f"- ≤2.0 s：{summary['within_2s']}/{summary['n_scored']}",
        "",
        "| stem | err_px | s_px | err/s | ≤0.5s | ≤1s | ≤2s |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        if not row.get("ok"):
            lines.append(f"| {row['stem']} | — | — | — | {row.get('reason','')} | | |")
            continue
        lines.append(
            f"| {row['stem']} | {row['err_px']:.3f} | {row['spacing_px']:.3f} | "
            f"{row['err_in_s']:.4f} | {row['within_0p5s']} | {row['within_1s']} | {row['within_2s']} |"
        )
    out_md = _OUT / "annotator_AB_pole_delta.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"写入 {out_json}")
    print(f"写入 {out_md}")
    if missing_b:
        print(f"尚缺第二标注员：{', '.join(missing_b)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
