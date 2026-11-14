# -*- coding: utf-8 -*-
"""古典暗谷前端 + 冻结的 v1.3 几何。不加载分割模型，不写论文。

参数在看这 43 帧的分数之前定死，不按成功数回改：
- 高斯去噪 σ=1.5
- 慢变底亮 σ=40，暗谷 = max(底亮 − 去噪图, 0)
- 视场沿用谱搜索的灰度 > 14，再腐蚀 15 像素，去掉圆边
- 按 99.5 分位归一化到 1，后面的 0.42 / 0.50 门槛不动
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import DATA_COMPARISON, RESULTS, setup_import_path  # noqa: E402

setup_import_path()

SESSION = DATA_COMPARISON
OUT = RESULTS / "measure_classical"

DENOISE_SIGMA = 1.5
BACKGROUND_SIGMA = 40.0
FIELD_GRAY = 14
FIELD_ERODE = 15
SCALE_PERCENTILE = 99.5


def classical_valley(gray_u8: np.ndarray) -> np.ndarray:
    gray = gray_u8.astype(np.float32)
    denoised = cv2.GaussianBlur(gray, (0, 0), DENOISE_SIGMA)
    background = cv2.GaussianBlur(denoised, (0, 0), BACKGROUND_SIGMA)
    valley = np.maximum(background - denoised, 0.0)
    field = gray > FIELD_GRAY
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FIELD_ERODE * 2 + 1, FIELD_ERODE * 2 + 1))
    field = cv2.erode(field.astype(np.uint8), kernel) > 0
    valley = valley * field.astype(np.float32)
    scale = float(np.percentile(valley, SCALE_PERCENTILE))
    if scale < 1e-3:
        return np.zeros_like(valley, dtype=np.float32)
    return np.clip(valley / scale, 0.0, 1.0).astype(np.float32)


def main() -> None:
    from band_score import load_band
    from circle_fit_helpers import _fit
    from score_s11 import judge_s11

    def _tally(rows: list[dict]) -> dict:
        from collections import defaultdict

        groups: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            groups[row["subset"]].append(row)
        out = {}
        for subset in ("Easy", "Open-arc", "Illegible"):
            group = groups.get(subset, [])
            out[subset] = {
                "n": len(group),
                "ok": sum(1 for row in group if row["success"]),
                "refused": sum(1 for row in group if row["refused"]),
            }
        out["All"] = {
            "n": len(rows),
            "ok": sum(1 for row in rows if row["success"]),
            "refused": sum(1 for row in rows if row["refused"]),
        }
        return out

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for canvas_path in sorted(SESSION.glob("*_canvas.png")):
        stem = canvas_path.name.replace("_canvas.png", "")
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        record = json.loads(canvas_path.with_name(stem + ".json").read_text(encoding="utf-8"))
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        subset = str(record.get("subset") or stem.split("_", 1)[-1])
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        signal = classical_valley(gray)
        started = time.perf_counter()
        center, radii = _fit(signal, gray, nocap=False, physics=None, blend=0.0)
        elapsed = time.perf_counter() - started
        judged = judge_s11(human, radii, center)
        row = {
            "stem": stem,
            "subset": subset,
            "seconds": round(elapsed, 3),
            "valley_frac_042": round(float((signal >= 0.42).mean()), 4),
            "center_xy": None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)],
            "radii_px": [round(float(radius), 2) for radius in radii],
            **judged,
        }
        rows.append(row)
        mark = "成功" if judged["success"] else ("拒测" if judged["refused"] else "未成功")
        print(
            f"{stem} {mark} n={len(radii)} match={judged['matches']} "
            f"miss={judged['misses']} extra={judged['false_rings']} "
            f"center={judged['center_ok']} {elapsed:.1f}s",
            flush=True,
        )
    tally = _tally(rows)
    note = (
        "古典暗谷前端，不加载分割模型。去噪 σ=1.5，底亮 σ=40，"
        "暗谷按 99.5 分位归一化后送入冻结的 v1.3 两轮装配。"
        "视场为灰度>14 再腐蚀 15 像素。几何门槛未改。不计入论文。"
    )
    lines = [note, ""]
    for subset in ("Easy", "Open-arc", "Illegible", "All"):
        item = tally[subset]
        lines.append(f"{subset}: {item['ok']}/{item['n']}  拒测 {item['refused']}")
    lines.append("")
    for row in rows:
        mark = "成功" if row["success"] else ("拒测" if row["refused"] else "未成功")
        lines.append(
            f"{row['stem']}: {mark}  输出{len(row['radii_px'])} 配上{row['matches']} "
            f"漏{row['misses']} 多{row['false_rings']} 结构暗带{row['n_struct']} "
            f"圆心{row['center_ok']} 偏差{row['center_error_px']}  {row['seconds']}s"
        )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text, flush=True)


def draw_reviews() -> None:
    """用已保存的圆画单帧和总览，不重新拟合。"""
    import run_deployed_measure as base
    import run_s11_rescore as dense
    from band_score import load_band
    from score_s11 import judge_s11

    saved = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    frame_dir = OUT / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for row in saved:
        stem = row["stem"]
        canvas = cv2.imread(str(SESSION / f"{stem}_canvas.png"), cv2.IMREAD_COLOR)
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        center = None if row.get("center_xy") is None else tuple(row["center_xy"])
        radii = [float(value) for value in row.get("radii_px") or []]
        judged = judge_s11(human, radii, center)
        mark = "成功" if judged["success"] else ("拒测" if judged["refused"] else "未成功")
        draw_center = center
        if center is not None and (abs(center[0]) > 4000 or abs(center[1]) > 4000):
            draw_center = None
        view = dense._draw(canvas, human, radii, draw_center, [
            f"{stem}  古典暗谷  {mark}",
            f"暗带 {human['n']} / 结构 {judged['n_struct']} / 细密 {judged['n_dense']} / 配上 {judged['matches']}",
            f"结构漏 {judged['misses']}/{judged['miss_limit']}  细密未配 {judged['dense_misses']}  多余 {judged['false_rings']}/{judged['extra_limit']}  圆心 {judged['center_ok']}",
        ], judged["dense_indices"])
        cv2.imwrite(str(frame_dir / f"{stem}.png"), view)
        tiles.append((stem, view))
    sheet = base._sheet(tiles)
    cv2.imwrite(str(OUT / "review_all.png"), sheet)
    print(f"frames {len(tiles)}  {frame_dir}  {OUT / 'review_all.png'}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "draw":
        draw_reviews()
    else:
        main()
