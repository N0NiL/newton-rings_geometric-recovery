"""第一轮之后的物理环带二次拟合。

预测的 r² 只用来框住缺级的位置。环带内的进入概率低于第一轮的二值门槛，
取的是实测峰，权重是该峰的概率，不向预测半径靠。
退出仍要有和第一轮相同的射线票数，并且峰落在已经写死的 0.55 倍步长以内。
最内一条再向内会出现负 r² 时，不向内搜。没有峰的缺口保持缺级。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates

from RadialRingAccumulator import AccumulatorConfig
from RingAssemblyV2 import REG_ZERO

# 第一轮二值门槛的一半。环带外不用这个值。写出时尚未看这 43 张的第二轮结果。
_ENTRY_SCALE = 0.5


def second_round(
    prob_hw: np.ndarray,
    center_xy: tuple[float, float] | None,
    radii: list[float],
) -> dict[str, Any]:
    prob = np.clip(np.asarray(prob_hw, dtype=np.float32), 0.0, 1.0)
    if prob.ndim == 3:
        prob = prob[..., 0]
    kept = sorted(float(r) for r in radii if np.isfinite(r) and r > 0)
    empty = {"rs": kept, "added": [], "targets": 0}
    if center_xy is None or len(kept) < 3:
        return empty
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if not np.isfinite(cx) or not np.isfinite(cy):
        return empty

    cfg = AccumulatorConfig()
    rs = np.asarray(kept, dtype=np.float64)
    r2 = rs ** 2
    diffs = np.diff(r2)
    step = float(np.median(diffs))
    if not np.isfinite(step) or step <= 1.0:
        return empty
    entry = float(cfg.prob_thr) * _ENTRY_SCALE
    vote_min = max(3, int(0.015 * cfg.n_angles))
    half = float(REG_ZERO) * step
    targets = _targets(r2, step)
    added: list[float] = []
    occupied = list(kept)
    for target in targets:
        radius = _refit_band(prob, cx, cy, target, half, entry, vote_min, occupied, cfg)
        if radius is None:
            continue
        added.append(radius)
        occupied.append(radius)
        occupied.sort()
    return {"rs": sorted(occupied), "added": added, "targets": len(targets)}


def _targets(r2: np.ndarray, step: float) -> list[float]:
    found: list[float] = []
    for left, right in zip(r2[:-1], r2[1:]):
        gap = float(right - left)
        orders = int(round(gap / step))
        if orders < 2:
            continue
        for k in range(1, orders):
            found.append(float(left + k * step))
    inward = float(r2[0] - step)
    if inward > 0.0:
        found.append(inward)
    found.append(float(r2[-1] + step))
    return found


def _refit_band(
    prob: np.ndarray,
    cx: float,
    cy: float,
    target_r2: float,
    half_r2: float,
    entry: float,
    vote_min: int,
    occupied: list[float],
    cfg: AccumulatorConfig,
) -> float | None:
    low = target_r2 - half_r2
    high = target_r2 + half_r2
    if high <= 1.0:
        return None
    r_lo = float(np.sqrt(max(low, 0.0)))
    r_hi = float(np.sqrt(high))
    if r_hi <= r_lo + 2.0:
        return None
    height, width = prob.shape[:2]
    r_corner = float(np.hypot(max(cx, (width - 1) - cx), max(cy, (height - 1) - cy)))
    if r_lo >= r_corner:
        return None
    n_radial = max(8, int((r_hi - r_lo) / 2.0))
    radii = np.linspace(r_lo, r_hi, n_radial, dtype=np.float64)
    peaks: list[float] = []
    weights: list[float] = []
    guard = float(cfg.min_peak_distance_px)
    for index in range(cfg.n_angles):
        theta = 2.0 * np.pi * (index / float(cfg.n_angles))
        xs = cx + radii * np.cos(theta)
        ys = cy + radii * np.sin(theta)
        signal = map_coordinates(prob, np.vstack([ys, xs]), order=1, mode="constant", cval=0.0)
        signal = np.clip(signal, 0.0, None)
        for known in occupied:
            signal[np.abs(radii - known) < guard] = 0.0
        if signal.size == 0 or float(signal.max()) < entry:
            continue
        at = int(np.argmax(signal))
        peaks.append(float(radii[at]))
        weights.append(float(signal[at]))
    if len(peaks) < vote_min:
        return None
    radius = _weighted_median(np.asarray(peaks), np.asarray(weights))
    if any(abs(radius - known) < guard for known in occupied):
        return None
    if abs(radius ** 2 - target_r2) > half_r2:
        return None
    return radius


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    at = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    at = min(at, len(ordered) - 1)
    return float(ordered[at])
