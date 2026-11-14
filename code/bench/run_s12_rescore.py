# -*- coding: utf-8 -*-
"""用 S1.2 重判对照集上已经存好的圆。不重新拟合，不改 S1.1 的 results.json。"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import DATA_COMPARISON, FITS_ROOT, NAMES, setup_import_path  # noqa: E402

setup_import_path()

from band_score import load_band  # noqa: E402
from score_s12 import judge_s12  # noqa: E402

SESSION = DATA_COMPARISON
SRC = FITS_ROOT
OUT = FITS_ROOT / "s12"
SUBSETS = ("Easy", "Open-arc", "Illegible", "All")


def _mark(row: dict) -> str:
    if row["success"]:
        return "成功"
    if row.get("unformed"):
        return "未成形"
    return "未成功"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "S1.2 重判。圆来自 measure_comparison，没有重拟合。",
        "圆心坐标不进及格线。圆环离散度是已配圆到标记弧的距离中位数，上限 0.4 s。",
        "",
    ]
    for name in NAMES:
        rows_in = json.loads((SRC / name / "results.json").read_text(encoding="utf-8"))
        judged = []
        for row in rows_in:
            stem = row["stem"]
            human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
            center = None if row.get("center_xy") is None else tuple(row["center_xy"])
            score = judge_s12(human, list(row.get("radii_px") or []), center)
            judged.append({
                "stem": stem,
                "subset": row["subset"],
                "origin": row.get("origin"),
                "success_s11": bool(row["success"]),
                "success": score["success"],
                "unformed": score["unformed"],
                "center_ok": score["center_ok"],
                "center_error_px": score["center_error_px"],
                "dispersion_px": score["dispersion_px"],
                "dispersion_step": score["dispersion_step"],
                "dispersion_ok": score["dispersion_ok"],
                "matches": score["matches"],
                "misses": score["misses"],
                "miss_limit": score["miss_limit"],
                "false_rings": score["false_rings"],
                "extra_limit": score["extra_limit"],
                "order_ok": score["order_ok"],
                "n_kept": score["n_kept"],
            })
        (OUT / f"{name}.json").write_text(json.dumps(judged, ensure_ascii=False, indent=2), encoding="utf-8")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in judged:
            groups[row["subset"]].append(row)
            groups["All"].append(row)
        lines.append(f"## {name}")
        for subset in SUBSETS:
            group = groups[subset]
            ok = sum(1 for row in group if row["success"])
            old = sum(1 for row in group if row["success_s11"])
            unformed = sum(1 for row in group if row["unformed"])
            lines.append(f"{subset}: S1.2 {ok}/{len(group)}  未成形 {unformed}    S1.1 {old}/{len(group)}")
        gained = [row for row in judged if row["success"] and not row["success_s11"]]
        lost = [row for row in judged if row["success_s11"] and not row["success"]]
        lines.append(f"仅因去掉圆心而新过 {len(gained)}    环离开标记而新失败 {len(lost)}")
        for row in gained:
            lines.append(
                f"  新过 {row['stem']}  离散度 {row['dispersion_px']} px / {row['dispersion_step']} s  圆心偏差 {row['center_error_px']}"
            )
        for row in lost:
            lines.append(
                f"  新失败 {row['stem']}  离散度 {row['dispersion_px']} px / {row['dispersion_step']} s  圆心偏差 {row['center_error_px']}"
            )
        lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
