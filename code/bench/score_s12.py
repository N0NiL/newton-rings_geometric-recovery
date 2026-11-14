# -*- coding: utf-8 -*-
"""S1.2 判据。定义见 S_STRUCT_S1.2.md。

圆心坐标不再进及格线。几何是否正确，看配上的圆离标记弧有多远。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from score_s11 import BAND_PAD_PX, judge_s11

# 与 S1.1「同一条纹」同一个尺度。不另找一个新数。
DISPERSION_SCALE = 0.4


def _point_gap(
    human_center: tuple[float, float],
    pred_xy: tuple[float, float],
    pred_radius: float,
    band_radius: float,
    angles: np.ndarray,
) -> float | None:
    """标记弧上的点到这条预测圆的距离，取中位数。"""
    if len(angles) < 1:
        return None
    px = human_center[0] + band_radius * np.cos(angles)
    py = human_center[1] + band_radius * np.sin(angles)
    distance = np.hypot(px - pred_xy[0], py - pred_xy[1])
    return float(np.median(np.abs(distance - pred_radius)))


def _matched_gaps(
    human: dict[str, Any],
    kept: list[float],
    pred_xy: tuple[float, float] | None,
) -> list[float]:
    """和 S1.1 同一套配对，再量每条配上的圆到标记点的距离。"""
    from score_s11 import _median_crossing

    bands = human["bands"]
    spacing = float(human["spacing"])
    if pred_xy is None or human["center"] is None or spacing <= 1e-6:
        return []
    limit = DISPERSION_SCALE * spacing
    pairs: list[tuple[float, int, int]] = []
    for band_index, band in enumerate(bands):
        for pred_index, radius in enumerate(kept):
            crossing = _median_crossing(human["center"], pred_xy, radius, band["angles"], float(band["radius"]))
            if crossing is None:
                continue
            gap = abs(crossing - float(band["radius"]))
            half = 0.5 * (float(band["r_out"]) - float(band["r_in"]))
            if gap < max(limit, half + BAND_PAD_PX):
                pairs.append((gap, band_index, pred_index))
    pairs.sort(key=lambda item: item[0])
    used_band: set[int] = set()
    used_pred: set[int] = set()
    gaps: list[float] = []
    for _gap, band_index, pred_index in pairs:
        if band_index in used_band or pred_index in used_pred:
            continue
        used_band.add(band_index)
        used_pred.add(pred_index)
        point_gap = _point_gap(
            human["center"],
            pred_xy,
            kept[pred_index],
            float(bands[band_index]["radius"]),
            bands[band_index]["angles"],
        )
        if point_gap is not None:
            gaps.append(point_gap)
    return gaps


def _kept(human: dict[str, Any], pred_rs: list[float], pred_xy: tuple[float, float] | None) -> list[float]:
    bands = human["bands"]
    kept: list[float] = []
    r_hole = float(bands[0]["r_in"]) - BAND_PAD_PX if bands else 0.0
    if pred_xy is not None and human["center"] is not None and bands:
        offset = float(np.hypot(pred_xy[0] - human["center"][0], pred_xy[1] - human["center"][1]))
        for radius in pred_rs:
            if offset + float(radius) >= r_hole:
                kept.append(float(radius))
    else:
        kept = [float(radius) for radius in pred_rs]
    kept.sort()
    return kept


def judge_s12(
    human: dict[str, Any],
    pred_rs: list[float],
    pred_xy: tuple[float, float] | None,
) -> dict[str, Any]:
    base = judge_s11(human, pred_rs, pred_xy)
    spacing = float(human["spacing"])
    pred = None if pred_xy is None else (float(pred_xy[0]), float(pred_xy[1]))
    gaps = _matched_gaps(human, _kept(human, pred_rs, pred), pred)
    dispersion_px = float(np.median(gaps)) if gaps else None
    dispersion_step = None
    if dispersion_px is not None and spacing > 1e-6:
        dispersion_step = dispersion_px / spacing
    dispersion_ok = bool(dispersion_step is not None and dispersion_step <= DISPERSION_SCALE + 1e-9)
    rings_ok = bool(
        human["n"] >= 3
        and not base["unformed"]
        and base["matches"] >= 3
        and base["misses"] <= base["miss_limit"]
        and base["false_rings"] <= base["extra_limit"]
        and base["order_ok"]
        and dispersion_ok
    )
    row = dict(base)
    row["dispersion_px"] = None if dispersion_px is None else round(dispersion_px, 2)
    row["dispersion_step"] = None if dispersion_step is None else round(dispersion_step, 3)
    row["dispersion_ok"] = dispersion_ok
    row["center_ok"] = base["center_ok"]
    row["success_s11"] = base["success"]
    row["success"] = rings_ok
    return row
