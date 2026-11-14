"""用 S1.1 重判已经保存的 v1.3、Hough、FRFT 圆。不重新拟合，不覆盖旧目录。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import run_deployed_measure as base  # noqa: E402
from band_score import load_band  # noqa: E402
from score_s11 import judge_s11  # noqa: E402

SESSION = HERE / "sessions" / "comparison"
OUT = HERE / "measure_s11_dense"
SOURCES = {
    "v1.3": HERE / "measure_s11" / "v1.3" / "results.json",
    "Hough": HERE / "measure_s11" / "Hough" / "results.json",
    "FRFT": HERE / "measure_s11" / "FRFT" / "results.json",
}


def _mark(row: dict) -> str:
    if row["success"]:
        return "成功"
    if row["refused"]:
        return "未成形"
    return "未成功"


def _draw(canvas, human, radii, center, lines, dense_indices):
    view = canvas.copy()
    hx, hy = human["center"]
    origin = (int(round(hx)), int(round(hy)))
    dense = set(dense_indices)
    for index, band in enumerate(human["bands"]):
        radius = float(band["radius"])
        if radius > 1:
            # 细密纹用暗青色，结构暗带用绿色。
            color = (160, 120, 40) if index in dense else (0, 160, 0)
            cv2.circle(view, origin, int(round(radius)), color, 1, cv2.LINE_AA)
    if center is not None:
        pred_origin = (int(round(center[0])), int(round(center[1])))
        for radius in radii:
            cv2.circle(view, pred_origin, max(1, int(round(radius))), (0, 200, 255), 1, cv2.LINE_AA)
    return base._caption(view, lines)


def _score_one(name: str, source: Path) -> list[dict]:
    saved = {row["stem"]: row for row in json.loads(source.read_text(encoding="utf-8"))}
    out_dir = OUT / name
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []
    for canvas_path in sorted(SESSION.glob("*_canvas.png")):
        stem = canvas_path.name.replace("_canvas.png", "")
        previous = saved.get(stem)
        if previous is None:
            continue
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        center = None if previous["center_xy"] is None else tuple(previous["center_xy"])
        radii = [float(value) for value in previous["radii_px"]]
        judged = judge_s11(human, radii, center)
        row = {
            "stem": stem,
            "subset": previous["subset"],
            "n_human": human["n"],
            "n_pred": len(radii),
            "center_xy": previous["center_xy"],
            "radii_px": previous["radii_px"],
            **judged,
        }
        rows.append(row)
        mark = _mark(row)
        view = _draw(canvas, human, radii, center, [
            f"{stem}  {name}  S1.1  {mark}",
            f"暗带 {human['n']} / 结构 {judged['n_struct']} / 细密 {judged['n_dense']} / 配上 {judged['matches']}",
            f"结构漏 {judged['misses']}/{judged['miss_limit']}  细密未配 {judged['dense_misses']}  多余 {judged['false_rings']}/{judged['extra_limit']}  圆心 {judged['center_ok']}",
        ], judged["dense_indices"])
        cv2.imwrite(str(frame_dir / f"{stem}.png"), view)
        tiles.append((stem, view))
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["subset"]].append(row)
    lines = [
        f"{name}。判据 S1.1，带两侧加 2 像素，细密纹不占漏检三成。圆来自 measure_s11，没有重拟合。",
        "局部间距 < 0.75 × 内半段间距中位数的记为细密纹。结构暗带少于 3 条时退回按全部暗带计数。",
        "这不是论文表。",
        "",
    ]
    for subset in ("Easy", "Open-arc", "Illegible"):
        group = groups.get(subset, [])
        ok = sum(1 for row in group if row["success"])
        refused = sum(1 for row in group if row["refused"])
        lines.append(f"{subset}: {ok}/{len(group)}  拒测 {refused}")
    ok = sum(1 for row in rows if row["success"])
    refused = sum(1 for row in rows if row["refused"])
    lines += ["", f"合计: {ok}/{len(rows)}  拒测 {refused}", ""]
    for row in rows:
        lines.append(
            f"{row['stem']}: {_mark(row)}  暗带{row['n_human']} 输出{row['n_kept']} 配上{row['matches']} "
            f"结构漏{row['misses']}/{row['miss_limit']} 细密{row['n_dense']}未配{row['dense_misses']} "
            f"多余{row['false_rings']}/{row['extra_limit']} "
            f"孔{row['n_hole']} 顺序{row['order_ok']} 圆心{row['center_ok']}  "
            f"残差步{row['dr2_residual_step']}"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if tiles:
        cv2.imwrite(str(out_dir / "review_all.png"), base._sheet(tiles))
    print(text, flush=True)
    return rows


def _compare(tables: dict[str, list[dict]]) -> None:
    names = list(tables)
    stems = [row["stem"] for row in tables[names[0]]]
    subsets = {row["stem"]: row["subset"] for row in tables[names[0]]}
    by_name = {name: {row["stem"]: row for row in rows} for name, rows in tables.items()}
    lines = [
        "S1.1。细密纹不占漏检三成。圆来自 measure_s11 的重拟合，没有再跑模型。",
        "成功数 / 该子集张数。拒测另计，拒测也不是成功。这不是论文表。",
        "",
    ]
    for subset in ("Easy", "Open-arc", "Illegible"):
        chosen = [stem for stem in stems if subsets[stem] == subset]
        bits = []
        for name in names:
            ok = sum(1 for stem in chosen if by_name[name].get(stem, {}).get("success"))
            bits.append(f"{name} {ok}/{len(chosen)}")
        lines.append(f"{subset}: " + " | ".join(bits))
    lines.append("")
    lines.append(f"{'帧':<18}{'子集':<10}" + "".join(f"{name:<8}" for name in names))
    for stem in stems:
        marks = []
        for name in names:
            row = by_name[name].get(stem)
            marks.append("-" if row is None else _mark(row))
        lines.append(f"{stem:<18}{subsets[stem]:<10}" + "".join(f"{mark:<8}" for mark in marks))
    compare = OUT / "compare"
    compare.mkdir(parents=True, exist_ok=True)
    (compare / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tables = {name: _score_one(name, path) for name, path in SOURCES.items()}
    _compare(tables)


if __name__ == "__main__":
    main()
