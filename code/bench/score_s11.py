"""S1.1 判据。定义见 S_STRUCT_S1.1.md。"""

from __future__ import annotations

from typing import Any

import numpy as np

# 标注半宽两侧再各放宽 2 像素。不改标注文件。
BAND_PAD_PX = 2.0
# 局部间距低于内半段间距的这个比例，记为细密纹。
DENSE_SCALE = 0.75
# 未匹配预测环 / 未匹配结构暗带的允许比例上限（floor 计数，预测与结构同时用）。
UNMATCHED_LIMIT = 0.3
# 半径配对窗口尺度：max(MATCH_WINDOW_SCALE * spacing, half + BAND_PAD_PX)。
MATCH_WINDOW_SCALE = 0.4


def _local_spacing(radii: list[float]) -> np.ndarray:
    """到左右邻环的较小间距。两端只用一侧。"""
    values = np.asarray(radii, np.float64)
    local = np.zeros(len(values), np.float64)
    if len(values) < 2:
        return local
    gaps = np.diff(values)
    local[0] = gaps[0]
    local[-1] = gaps[-1]
    if len(values) > 2:
        local[1:-1] = np.minimum(gaps[:-1], gaps[1:])
    return local


def _dense_mask(bands: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    """内半段间距中位数是这张图还能分开的纹的尺度。"""
    radii = [float(band["radius"]) for band in bands]
    local = _local_spacing(radii)
    cut = len(bands) // 2
    if cut < 1:
        return np.zeros(len(bands), dtype=bool), 0.0
    s_clear = float(np.median(local[:cut]))
    if s_clear <= 1e-6:
        return np.zeros(len(bands), dtype=bool), s_clear
    return local < DENSE_SCALE * s_clear, s_clear


def _median_crossing(
    center: tuple[float, float],
    pred_xy: tuple[float, float],
    radius: float,
    angles: np.ndarray,
    target: float,
) -> float | None:
    if len(angles) < 1:
        return None
    dx = pred_xy[0] - center[0]
    dy = pred_xy[1] - center[1]
    along = np.cos(angles) * dx + np.sin(angles) * dy
    disc = along * along - (dx * dx + dy * dy) + radius * radius
    picked: list[float] = []
    for index in range(len(angles)):
        if disc[index] < -1e-6:
            continue
        root = float(np.sqrt(max(float(disc[index]), 0.0)))
        positive = [value for value in (float(along[index]) - root, float(along[index]) + root) if value > 0.0]
        if not positive:
            continue
        picked.append(min(positive, key=lambda value: abs(value - target)))
    if len(picked) * 2 < len(angles):
        return None
    return float(np.median(picked))


def judge_s11(
    human: dict[str, Any],
    pred_rs: list[float],
    pred_xy: tuple[float, float] | None,
) -> dict[str, Any]:
    bands = human["bands"]
    n_h = len(bands)
    spacing = float(human["spacing"])
    limit = MATCH_WINDOW_SCALE * spacing
    center_limit = 0.5 * spacing
    center_error = None
    center_ok = False
    if pred_xy is not None and human["center"] is not None and spacing > 1e-6:
        center_error = float(np.hypot(pred_xy[0] - human["center"][0], pred_xy[1] - human["center"][1]))
        center_ok = center_error <= center_limit

    kept: list[float] = []
    n_hole = 0
    r_hole = float(bands[0]["r_in"]) - BAND_PAD_PX if bands else 0.0
    if pred_xy is not None and human["center"] is not None and bands:
        offset = float(np.hypot(pred_xy[0] - human["center"][0], pred_xy[1] - human["center"][1]))
        for radius in pred_rs:
            if offset + float(radius) < r_hole:
                n_hole += 1
            else:
                kept.append(float(radius))
    else:
        kept = [float(radius) for radius in pred_rs]
    kept.sort()

    pairs: list[tuple[float, int, int, float]] = []
    if pred_xy is not None and human["center"] is not None and spacing > 1e-6:
        for band_index, band in enumerate(bands):
            for pred_index, radius in enumerate(kept):
                crossing = _median_crossing(human["center"], pred_xy, radius, band["angles"], float(band["radius"]))
                if crossing is None:
                    continue
                gap = abs(crossing - float(band["radius"]))
                half = 0.5 * (float(band["r_out"]) - float(band["r_in"]))
                band_limit = max(limit, half + BAND_PAD_PX)
                if gap < band_limit:
                    pairs.append((gap, band_index, pred_index, crossing))
    pairs.sort(key=lambda item: item[0])
    used_band: set[int] = set()
    used_pred: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for gap, band_index, pred_index, crossing in pairs:
        if band_index in used_band or pred_index in used_pred:
            continue
        used_band.add(band_index)
        used_pred.add(pred_index)
        matched.append((band_index, pred_index, crossing))
    matched.sort()
    crossings = [crossing for _band_index, _pred_index, crossing in matched]
    order_ok = crossings == sorted(crossings) and len(crossings) == len(set(np.round(crossings, 6)))
    dense, s_clear = _dense_mask(bands)
    n_dense = int(np.count_nonzero(dense))
    n_struct = n_h - n_dense
    struct_miss = sum(1 for index in range(n_h) if index not in used_band and not bool(dense[index]))
    dense_miss = sum(1 for index in range(n_h) if index not in used_band and bool(dense[index]))
    # 结构暗带太少时，细密纹不能把漏检义务掏空，退回按全部暗带计数。
    if n_struct >= 3:
        misses = struct_miss
        miss_limit = int(np.floor(UNMATCHED_LIMIT * n_struct))
        dense_gate = True
    else:
        misses = n_h - len(matched)
        miss_limit = int(np.floor(UNMATCHED_LIMIT * n_h))
        dense_gate = False
    extras = len(kept) - len(matched)
    extra_limit = int(np.floor(UNMATCHED_LIMIT * len(kept)))
    step = 0.0
    if n_h >= 2:
        human_r2 = np.square([float(band["radius"]) for band in bands])
        step = float(np.median(np.diff(human_r2)))
    residuals = []
    if step > 1.0:
        for band_index, _pred_index, crossing in matched:
            residuals.append(abs(crossing ** 2 - float(bands[band_index]["radius"]) ** 2) / step)
    residual_med = float(np.median(residuals)) if residuals else None
    enough = len(kept) >= 3
    success = bool(
        n_h >= 3
        and enough
        and len(matched) >= 3
        and misses <= miss_limit
        and extras <= extra_limit
        and order_ok
        and center_ok
    )
    return {
        "spacing_px": None if spacing <= 1e-6 else round(spacing, 2),
        "center_error_px": None if center_error is None else round(center_error, 2),
        "matches": len(matched),
        "misses": misses,
        "false_rings": extras,
        "miss_limit": miss_limit,
        "n_dense": n_dense,
        "n_struct": n_struct,
        "dense_misses": dense_miss,
        "s_clear_px": None if s_clear <= 1e-6 else round(s_clear, 2),
        "dense_gate": dense_gate,
        "dense_indices": [index for index, flag in enumerate(dense) if bool(flag)],
        "extra_limit": extra_limit,
        "n_hole": n_hole,
        "n_kept": len(kept),
        "order_ok": order_ok,
        "center_ok": center_ok,
        "dr2_residual_step": None if residual_med is None else round(residual_med, 3),
        # 字段名沿用 refused。论文和汇总里叫未成形（unformed）：不足三环，留在分母里。
        "refused": len(kept) < 3,
        "unformed": len(kept) < 3,
        "success": success,
    }
