# -*- coding: utf-8 -*-
"""OpenCV 圆霍夫参数网格 + HOUGH_GRADIENT_ALT 的测试集 oracle。

在 69 帧可读对照集（Easy+Open-arc）上扫描，报告使 0.5 s 通过率最高的参数。
口径：S1.2 结构成功 + 圆心门（与 build_data_prep_docx._passes 一致）。
这是测试集上的事后最优（oracle），不是验证集调参。
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for path in (HERE, REPO, REPO / "VisionAnalysis", REPO / "ExtraUtilities"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from band_score import load_band  # noqa: E402
from run_hough_frft_look import hough_rings  # noqa: E402
from score_s12 import judge_s12  # noqa: E402

SESSION = HERE / "sessions" / "comparison"
OUT = HERE / "measure_supplement" / "hough_oracle"
READABLE = ("Easy", "Open-arc")
# 文件名仍是 Open-arc，但子集已改记为难辨（与 jitter_july_labels / 论文 69 帧口径一致）
ILLEGIBLE_NAMED_OPEN = {
    "000042_Open-arc",
    "000043_Open-arc",
    "000044_Open-arc",
    "000045_Open-arc",
    "000046_Open-arc",
    "000047_Open-arc",
    "000048_Open-arc",
    "000050_Open-arc",
    "000060_Open-arc",
    "000061_Open-arc",
    "000064_Open-arc",
    "000065_Open-arc",
    "000068_Open-arc",
}


def _stems() -> list[str]:
    out = []
    for path in sorted(SESSION.glob("*_canvas.png")):
        stem = path.name.replace("_canvas.png", "")
        if stem in ILLEGIBLE_NAMED_OPEN:
            continue
        record = json.loads(path.with_name(stem + ".json").read_text(encoding="utf-8"))
        subset = str(record.get("subset") or stem.split("_", 1)[-1])
        if subset == "Broken":
            subset = "Illegible"
        if subset in READABLE:
            out.append(stem)
    return out


def _inside(gray: np.ndarray) -> np.ndarray:
    mask = gray > 14
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def _merge_radii(radii: list[float], min_gap: float = 6.0) -> list[float]:
    merged: list[float] = []
    for radius in sorted(float(r) for r in radii):
        if merged and radius - merged[-1] < min_gap:
            merged[-1] = 0.5 * (merged[-1] + radius)
        else:
            merged.append(radius)
    return merged


def opencv_hough(
    canvas: np.ndarray,
    *,
    method: int,
    dp: float,
    min_dist: float,
    param1: float,
    param2: float,
    min_radius: int,
    max_radius: int,
) -> tuple[tuple[float, float] | None, list[float]]:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    field = _inside(gray)
    masked = blur.copy()
    masked[~field] = 0
    try:
        circles = cv2.HoughCircles(
            masked,
            method=method,
            dp=float(dp),
            minDist=float(min_dist),
            param1=float(param1),
            param2=float(param2),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
    except cv2.error:
        return None, []
    if circles is None or len(circles) == 0:
        return None, []
    arr = np.asarray(circles[0], dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None, []
    cx = float(np.median(arr[:, 0]))
    cy = float(np.median(arr[:, 1]))
    keep_r = [
        float(r)
        for x, y, r in arr
        if float(np.hypot(x - cx, y - cy)) <= max(25.0, 0.08 * float(r))
    ]
    radii = _merge_radii(keep_r)
    if len(radii) < 3:
        hist, xedges, yedges = np.histogram2d(arr[:, 0], arr[:, 1], bins=48)
        peak = np.unravel_index(int(np.argmax(hist)), hist.shape)
        cx = float(0.5 * (xedges[peak[0]] + xedges[peak[0] + 1]))
        cy = float(0.5 * (yedges[peak[1]] + yedges[peak[1] + 1]))
        keep_r = [float(r) for x, y, r in arr if np.hypot(x - cx, y - cy) <= 40.0]
        radii = _merge_radii(keep_r)
    if len(radii) < 3:
        return (cx, cy), []
    return (cx, cy), radii


def _tally_gates(frames: list[dict], get_fit) -> dict:
    ok05 = ok10 = ok20 = ok_none = 0
    for frame in frames:
        center, radii = get_fit(frame)
        judged = judge_s12(frame["human"], list(radii), center)
        if not judged.get("success"):
            continue
        ok_none += 1
        spacing = float(frame["human"]["spacing"])
        err = judged.get("center_error_px")
        if err is None or spacing <= 1e-6:
            continue
        err = float(err)
        if err <= 2.0 * spacing + 1e-9:
            ok20 += 1
        if err <= 1.0 * spacing + 1e-9:
            ok10 += 1
        if err <= 0.5 * spacing + 1e-9:
            ok05 += 1
    return {
        "ok_0p5": ok05,
        "ok_1p0": ok10,
        "ok_2p0": ok20,
        "ok_none": ok_none,
        "n": len(frames),
    }


def evaluate_combo(frames: list[dict], combo: dict) -> dict:
    tallied = _tally_gates(frames, lambda frame: opencv_hough(frame["canvas"], **combo))
    return {
        **tallied,
        **combo,
        "method_name": "HOUGH_GRADIENT_ALT" if combo["method"] == cv2.HOUGH_GRADIENT_ALT else "HOUGH_GRADIENT",
    }


def paper_fixed_baseline(frames: list[dict]) -> dict:
    tallied = _tally_gates(frames, lambda frame: hough_rings(frame["canvas"]))
    return {"name": "paper_custom_hough_canny_35_110", **tallied}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = _stems()
    frames = []
    for stem in stems:
        canvas = cv2.imread(str(SESSION / f"{stem}_canvas.png"), cv2.IMREAD_COLOR)
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        frames.append({"stem": stem, "subset": stem.split("_", 1)[-1], "canvas": canvas, "human": human})
    print(f"可读帧 {len(frames)}（Easy+Open-arc）", flush=True)

    fixed = paper_fixed_baseline(frames)
    print(
        f"论文固定自定义 Hough：0.5s {fixed['ok_0p5']}/{fixed['n']}  "
        f"1s {fixed['ok_1p0']}  2s {fixed['ok_2p0']}  none {fixed['ok_none']}",
        flush=True,
    )

    # HOUGH_GRADIENT：param1≈Canny 高阈，param2=累加器阈值
    # HOUGH_GRADIENT_ALT：param2 必须 < 1（圆完美度），与 GRADIENT 不共用同一网格
    combos = []
    for dp, min_dist, p1, p2 in itertools.product(
        [1.2, 1.5], [20.0, 40.0], [70.0, 110.0], [40.0, 60.0]
    ):
        combos.append(
            {
                "method": cv2.HOUGH_GRADIENT,
                "dp": dp,
                "min_dist": min_dist,
                "param1": p1,
                "param2": p2,
                "min_radius": 25,
                "max_radius": 650,
            }
        )
    for dp, min_dist, p1, p2 in itertools.product(
        [1.5], [20.0, 40.0], [100.0, 300.0], [0.80, 0.90]
    ):
        combos.append(
            {
                "method": cv2.HOUGH_GRADIENT_ALT,
                "dp": dp,
                "min_dist": min_dist,
                "param1": p1,
                "param2": p2,
                "min_radius": 25,
                "max_radius": 650,
            }
        )
    print(f"网格组合数 {len(combos)}", flush=True)
    started = time.perf_counter()
    results = []
    best = None
    for index, combo in enumerate(combos, 1):
        print(f"开始组合 {index}/{len(combos)} …", flush=True)
        row = evaluate_combo(frames, combo)
        results.append(row)
        key = (row["ok_0p5"], row["ok_1p0"], row["ok_none"])
        if best is None or key > (best["ok_0p5"], best["ok_1p0"], best["ok_none"]):
            best = row
        if index % 20 == 0 or index == len(combos):
            print(
                f"[{index}/{len(combos)}] best@0.5s={best['ok_0p5']}/{best['n']} "
                f"({best['method_name']} dp={best['dp']} minDist={best['min_dist']} "
                f"p1={best['param1']} p2={best['param2']})",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    results.sort(key=lambda r: (r["ok_0p5"], r["ok_1p0"], r["ok_2p0"], r["ok_none"]), reverse=True)
    slim = []
    for row in results:
        item = dict(row)
        item.pop("method", None)
        slim.append(item)
    payload = {
        "n_readable": len(frames),
        "n_combos": len(combos),
        "seconds": round(elapsed, 1),
        "paper_fixed_custom_hough": fixed,
        "oracle_best": {k: v for k, v in best.items() if k != "method"} if best else None,
        "top10": slim[:10],
        "grid": {
            "HOUGH_GRADIENT": {
                "dp": [1.2, 1.5],
                "minDist": [20.0, 40.0],
                "param1_canny_high": [70.0, 110.0],
                "param2_accumulator": [40.0, 60.0],
                "minRadius": 25,
                "maxRadius": 650,
            },
            "HOUGH_GRADIENT_ALT": {
                "dp": [1.5],
                "minDist": [20.0, 40.0],
                "param1": [100.0, 300.0],
                "param2_perfectness": [0.80, 0.90],
                "minRadius": 25,
                "maxRadius": 650,
                "note": "OpenCV 要求 ALT 的 param2 < 1.0",
            },
        },
    }
    (OUT / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Hough Oracle 摘要",
        "",
        f"可读帧：{len(frames)}；组合：{len(combos)}；耗时 {elapsed:.1f}s",
        "",
        f"论文固定自定义 Hough（Canny 35/110）：0.5s **{fixed['ok_0p5']}/{fixed['n']}**",
        "",
        "## 测试集 oracle 最优",
        "",
    ]
    if best:
        lines += [
            f"- method: {best['method_name']}",
            f"- dp={best['dp']}, minDist={best['min_dist']}, param1={best['param1']}, param2={best['param2']}, "
            f"minRadius={best['min_radius']}, maxRadius={best['max_radius']}",
            f"- 0.5 s: **{best['ok_0p5']}/{best['n']}**",
            f"- 1.0 s: {best['ok_1p0']}/{best['n']}",
            f"- 2.0 s: {best['ok_2p0']}/{best['n']}",
            f"- 不测圆心: {best['ok_none']}/{best['n']}",
            "",
            "说明：参数在同一测试集上挑最优，是乐观上界（oracle），不是独立验证集调参。",
        ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
