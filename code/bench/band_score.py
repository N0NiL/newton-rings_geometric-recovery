"""暗带判据。匹配定义见 S_STRUCT_BAND.md，不改中线文件里的 0.4 / 0.5 / 0.3。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _center_ok(
    human: dict[str, Any],
    pred_xy: tuple[float, float] | None,
    width: int,
    height: int,
    spacing: float,
) -> bool:
    """圆心门：画布内用 0.5 Λ；画布外只要求极心落在对应侧外。"""
    if pred_xy is None or human["center"] is None:
        return False
    if human.get("in_frame"):
        hx, hy = human["center"]
        return float(np.hypot(pred_xy[0] - hx, pred_xy[1] - hy)) <= 0.5 * spacing
    side = human.get("side")
    px, py = pred_xy
    if side == "left":
        return px < 0
    if side == "right":
        return px >= width
    if side == "top":
        return py < 0
    if side == "bottom":
        return py >= height
    return False


def load_band(path: Path, human_path: Path) -> dict[str, Any]:
    band = json.loads(path.read_text(encoding="utf-8"))
    human = json.loads(human_path.read_text(encoding="utf-8"))
    center = (float(band["center_xy"][0]), float(band["center_xy"][1]))
    rings = [ring for ring in band.get("rings") or [] if str(ring.get("role") or "newton") == "newton"]
    rings.sort(key=lambda ring: float(ring["radius_px"]))
    human_rings = [ring for ring in human.get("rings") or [] if str(ring.get("role") or "newton") == "newton"]
    used: set[int] = set()
    bands: list[dict[str, Any]] = []
    for ring in rings:
        radius = float(ring["radius_px"])
        half = float(ring["half_width_px"])
        source = float(ring.get("source_radius_px", radius))
        best: int | None = None
        best_gap = 1e9
        for index, item in enumerate(human_rings):
            if index in used:
                continue
            gap = abs(float(item["radius_px"]) - source)
            if gap < best_gap:
                best, best_gap = index, gap
        angles = np.zeros(0, np.float64)
        if str(ring.get("origin") or "") == "added":
            best = None
            best_gap = 1e9
        if best is not None and best_gap <= 2.0:
            used.add(best)
            points = np.asarray(human_rings[best].get("points_xy") or [], np.float64)
            if len(points) >= 1:
                angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        if len(angles) < 1 and str(ring.get("origin") or "") == "added":
            nearest = None
            nearest_gap = 1e9
            for item in human_rings:
                points = np.asarray(item.get("points_xy") or [], np.float64)
                gap = abs(float(item["radius_px"]) - radius)
                if len(points) >= 1 and gap < nearest_gap:
                    nearest = points
                    nearest_gap = gap
            if nearest is not None:
                angles = np.arctan2(nearest[:, 1] - center[1], nearest[:, 0] - center[0])
        bands.append({
            "radius": radius,
            "r_in": radius - half,
            "r_out": radius + half,
            "angles": angles,
        })
    radii = [item["radius"] for item in bands]
    gaps = np.diff(radii) if len(radii) >= 2 else np.array([])
    spacing = float(np.median(gaps)) if gaps.size else 0.0
    return {
        "n": len(bands),
        "bands": bands,
        "radii": radii,
        "center": center,
        "in_frame": bool(band.get("center_in_frame")),
        "side": band.get("off_side"),
        "spacing": spacing,
    }


def _crossings(center: tuple[float, float], pred_xy: tuple[float, float], radius: float, angles: np.ndarray) -> np.ndarray | None:
    """从人标圆心沿可见方向看，算法圆的正向交点半径。射线打不中就配不上。"""
    dx = pred_xy[0] - center[0]
    dy = pred_xy[1] - center[1]
    ux = np.cos(angles)
    uy = np.sin(angles)
    along = ux * dx + uy * dy
    disc = along * along - (dx * dx + dy * dy) + radius * radius
    if np.any(disc < -1e-6):
        return None
    root = np.sqrt(np.maximum(disc, 0.0))
    return np.stack([along - root, along + root], axis=0)


def _on_band(cross: np.ndarray, r_in: float, r_out: float) -> tuple[bool, float]:
    mid = 0.5 * (r_in + r_out)
    picked = np.empty(cross.shape[1], np.float64)
    for index in range(cross.shape[1]):
        positive = [float(value) for value in cross[:, index] if value > 0.0]
        if not positive:
            return False, 0.0
        picked[index] = min(positive, key=lambda value: abs(value - mid))
    inside = bool(np.all((picked >= r_in - 1e-3) & (picked <= r_out + 1e-3)))
    return inside, float(np.median(np.abs(picked - mid)))


def judge_band(
    human: dict[str, Any],
    pred_rs: list[float],
    pred_xy: tuple[float, float] | None,
    width: int,
    height: int,
) -> dict[str, Any]:
    bands = human["bands"]
    n_h = len(bands)
    spacing = float(human["spacing"])
    center_ok = _center_ok(human, pred_xy, width, height, spacing) if spacing > 1e-6 else False
    kept: list[float] = []
    n_hole = 0
    r_hole = float(bands[0]["r_in"]) if bands else 0.0
    if pred_xy is not None and human["center"] is not None and bands:
        hx, hy = human["center"]
        offset = float(np.hypot(pred_xy[0] - hx, pred_xy[1] - hy))
        for radius in pred_rs:
            if offset + float(radius) < r_hole:
                n_hole += 1
            else:
                kept.append(float(radius))
    else:
        kept = [float(radius) for radius in pred_rs]
    kept.sort()
    matched: list[tuple[int, int]] = []
    used: set[int] = set()
    if pred_xy is not None and human["center"] is not None:
        for band_index, band in enumerate(bands):
            angles = band["angles"]
            if len(angles) < 1:
                continue
            candidates: list[tuple[float, int]] = []
            for pred_index, radius in enumerate(kept):
                if pred_index in used:
                    continue
                cross = _crossings(human["center"], pred_xy, radius, angles)
                if cross is None:
                    continue
                inside, gap = _on_band(cross, float(band["r_in"]), float(band["r_out"]))
                if inside:
                    candidates.append((gap, pred_index))
            if not candidates:
                continue
            candidates.sort()
            chosen = candidates[0][1]
            used.add(chosen)
            matched.append((band_index, chosen))
    pred_order = [kept[pred_index] for _band_index, pred_index in matched]
    order_ok = pred_order == sorted(pred_order) and len(pred_order) == len(set(np.round(pred_order, 6)))
    misses = n_h - len(matched)
    false_rings = len(kept) - len(matched)
    miss_limit = int(np.floor(0.3 * n_h))
    enough = len(kept) >= 3
    success = bool(
        n_h >= 3
        and enough
        and len(matched) >= 3
        and misses <= miss_limit
        and false_rings == 0
        and order_ok
        and center_ok
    )
    return {
        "spacing_px": None if spacing <= 1e-6 else round(spacing, 2),
        "matches": len(matched),
        "misses": misses,
        "false_rings": false_rings,
        "miss_limit": miss_limit,
        "n_hole": n_hole,
        "order_ok": order_ok,
        "center_ok": center_ok,
        "refused": len(kept) < 3,
        "success": success,
    }
