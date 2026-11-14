"""标记环带内的同心率。不改 struct，不设及格线。

不量标记点到拟合圆的距离。
每条标记环带是中心线半径加减半宽，角度取标记弧盖住的那一段。
沿这段弧每 2° 取一条从标记圆心发出的射线，看拟合圆交在带内还是带外。
同心率是落在半宽内的射线所占比例。一对一，游离更小的先占。孔内圆先拿掉。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import DATA_COMPARISON, FITS_ROOT, RESULTS, setup_import_path  # noqa: E402

setup_import_path()

from score_s11 import BAND_PAD_PX, _local_spacing  # noqa: E402

SESSION = DATA_COMPARISON
FITS = FITS_ROOT
OUT = RESULTS / "coherency"
NAMES = ("v1.3", "Hough", "FRFT")


def _sector(points: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """标记点盖住的那一段弧，按 2° 采样。最大的空档视为弧的外侧。"""
    ang = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ang = np.sort(np.unique(np.round(ang, 6)))
    if len(ang) == 1:
        return ang
    gaps = np.diff(ang)
    wrap = float(ang[0] + 2 * np.pi - ang[-1])
    all_gaps = np.append(gaps, wrap)
    cut = int(np.argmax(all_gaps))
    if cut == len(gaps):
        start, end = float(ang[0]), float(ang[-1])
    else:
        start, end = float(ang[cut + 1]), float(ang[cut] + 2 * np.pi)
    span = max(end - start, np.deg2rad(2.0))
    count = max(5, int(np.ceil(np.rad2deg(span) / 2.0)) + 1)
    return np.linspace(start, start + span, count)


def _rho(
    human_center: tuple[float, float],
    pred_xy: tuple[float, float],
    radius: float,
    angles: np.ndarray,
    target: float,
) -> np.ndarray:
    dx = pred_xy[0] - human_center[0]
    dy = pred_xy[1] - human_center[1]
    along = np.cos(angles) * dx + np.sin(angles) * dy
    disc = along * along - (dx * dx + dy * dy) + radius * radius
    root = np.sqrt(np.maximum(disc, 0.0))
    near = np.where((disc >= -1e-6) & (along - root > 0.0), along - root, np.nan)
    far = np.where((disc >= -1e-6) & (along + root > 0.0), along + root, np.nan)
    choose_near = np.isnan(far) | (np.abs(near - target) <= np.abs(far - target))
    return np.where(choose_near, near, far)


def _visible(stem: str) -> dict:
    human = json.loads((SESSION / f"{stem}_human.json").read_text(encoding="utf-8"))
    band = json.loads((SESSION / f"{stem}_band.json").read_text(encoding="utf-8"))
    center = (float(human["center_xy"][0]), float(human["center_xy"][1]))
    band_rings = [ring for ring in band.get("rings") or [] if str(ring.get("role") or "newton") == "newton"]
    rings = []
    for ring in human.get("rings") or []:
        if str(ring.get("role") or "newton") != "newton":
            continue
        points = np.asarray(ring.get("points_xy") or [], np.float64)
        if len(points) < 1:
            continue
        radius = float(ring["radius_px"])
        half = 4.0
        if band_rings:
            nearest = min(band_rings, key=lambda item: abs(float(item["radius_px"]) - radius))
            if abs(float(nearest["radius_px"]) - radius) <= 2.0:
                half = float(nearest.get("half_width_px") or 4.0)
        rings.append(
            {
                "radius": radius,
                "half": max(half, 0.5),
                "angles": _sector(points, center),
            }
        )
    rings.sort(key=lambda item: item["radius"])
    local = _local_spacing([item["radius"] for item in rings])
    for index, item in enumerate(rings):
        item["local"] = float(local[index])
    if band_rings:
        band_rings.sort(key=lambda ring: float(ring["radius_px"]))
        first = band_rings[0]
        hole = float(first["radius_px"]) - float(first["half_width_px"]) - BAND_PAD_PX
    else:
        hole = 0.0
    return {"center": center, "rings": rings, "hole": hole, "subset": str(human.get("subset") or "")}


def _kept(row: dict, human_center: tuple[float, float], hole: float) -> list[float]:
    pred = row.get("center_xy")
    radii = [float(value) for value in row.get("radii_px") or []]
    if pred is None:
        return []
    offset = float(np.hypot(float(pred[0]) - human_center[0], float(pred[1]) - human_center[1]))
    return [radius for radius in radii if offset + radius >= hole]


def _frame(visible: dict, row: dict) -> dict:
    pred = row.get("center_xy")
    rings = visible["rings"]
    if pred is None or not rings:
        return {"n": len(rings), "covered": 0, "assigned": 0, "median_px": None, "median_step": None}
    center = (float(pred[0]), float(pred[1]))
    radii = _kept(row, visible["center"], visible["hole"])
    pairs: list[tuple[float, int, int, float]] = []
    for ring_index, ring in enumerate(rings):
        for pred_index, radius in enumerate(radii):
            rho = _rho(visible["center"], center, radius, ring["angles"], ring["radius"])
            finite = np.isfinite(rho)
            if not np.any(finite):
                continue
            wander = np.abs(rho - ring["radius"])
            filled = np.where(finite, wander, ring["half"] * 4.0)
            inside_count = int(np.sum(finite & (wander <= ring["half"])))
            med = float(np.median(filled))
            pairs.append((med, ring_index, pred_index, inside_count))
    pairs.sort(key=lambda item: item[0])
    used_ring: set[int] = set()
    used_pred: set[int] = set()
    inside_of: dict[int, int] = {}
    wanders: list[float] = []
    for med, ring_index, pred_index, inside_count in pairs:
        if ring_index in used_ring or pred_index in used_pred:
            continue
        used_ring.add(ring_index)
        used_pred.add(pred_index)
        inside_of[ring_index] = inside_count
        wanders.append(med)
    covered = 0.0
    for index, ring in enumerate(rings):
        covered += inside_of.get(index, 0) / len(ring["angles"])
    steps = [med / rings[index]["half"] for index, med in zip(inside_of, wanders) if rings[index]["half"] > 1e-6]
    return {
        "n": len(rings),
        "covered": covered,
        "assigned": len(inside_of),
        "median_px": None if not wanders else round(float(np.median(wanders)), 2),
        "median_step": None if not steps else round(float(np.median(steps)), 3),
    }


def _aggregate(rows: list[dict]) -> str:
    if not rows:
        return "无"
    cover = [row["covered"] / row["n"] if row["n"] else 0.0 for row in rows]
    pixels = [row["median_px"] for row in rows if row["median_px"] is not None]
    steps = [row["median_step"] for row in rows if row["median_step"] is not None]
    px = "—" if not pixels else f"{float(np.median(pixels)):.2f}"
    step = "—" if not steps else f"{float(np.median(steps)):.3f}"
    return (
        f"同心率中位 {float(np.median(cover)):.2f}  平均 {float(np.mean(cover)):.2f}  "
        f"已配环游离中位 {px} px  {step} 倍半宽  "
        f"同心率≥0.5 的帧 {sum(1 for row in rows if row['n'] and row['covered'] / row['n'] >= 0.5)}/{len(rows)}"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fits = {
        name: {row["stem"]: row for row in json.loads((FITS / name / "results.json").read_text(encoding="utf-8"))}
        for name in NAMES
    }
    stems = [row["stem"] for row in json.loads((FITS / "v1.3" / "results.json").read_text(encoding="utf-8"))]
    stored: dict[str, list[dict]] = {name: [] for name in NAMES}
    lines = [
        "标记环带内的同心率。圆来自 measure_s11，没有重拟合。struct 未改。",
        "沿标记弧每 2° 看拟合圆是否落在半宽内。一对一。孔内圆已去掉。",
        "同心率 = 带内射线 / 该弧上的射线。这是描述，不是及格线，也不是论文表。",
        "",
    ]
    for subset in ("Open-arc", "Illegible"):
        lines.append(f"## {subset}")
        chosen = []
        for stem in stems:
            visible = _visible(stem)
            if visible["subset"] != subset:
                continue
            chosen.append(stem)
            bits = [f"{stem}  可见弧{len(visible['rings'])}"]
            for name in NAMES:
                item = _frame(visible, fits[name][stem])
                item["stem"] = stem
                item["subset"] = subset
                stored[name].append(item)
                rate = 0.0 if item["n"] == 0 else item["covered"] / item["n"]
                px = "—" if item["median_px"] is None else f"{item['median_px']}"
                bits.append(f"{name} {item['covered']}/{item['n']} ({rate:.2f}) 游离{px}px")
            lines.append("  ".join(bits))
        lines.append("")
        for name in NAMES:
            group = [row for row in stored[name] if row["subset"] == subset]
            lines.append(f"{name}: {_aggregate(group)}")
        lines.append("")
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text, encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(text)


if __name__ == "__main__":
    main()
