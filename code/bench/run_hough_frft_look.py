"""同一正式 43 张上的 Hough 草图与 FRFT 型全局估计。只供查看，不写入论文。

Hough：Canny 边缘沿梯度向圆内、圆外投票圆心，圆心允许出画；再取边缘到该圆心的距离峰。
这是实验台里的 hough_sketch，不是已经锁定的 CV-A 协议。

FRFT：在降采样图上搜索使径向信号在 r^2 上最像单一频率的圆心（纯二次相位经 r^2 重采样后，
分数阶傅里叶会把它收成一个峰；这里直接用该频谱峰的高度）。再在全分辨率径向剖面上取暗纹极小。
Lu / Wu 估的是一个圆心和一份曲率，不是这份圆名单。D2「必须逐式复现他们」已于 2026-10-30 否定。
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import DATA_COMPARISON, RESULTS, setup_import_path  # noqa: E402

setup_import_path()

SESSION = DATA_COMPARISON
DEPLOYED = RESULTS / "measure_deployed" / "results.json"


def _inside(gray: np.ndarray) -> np.ndarray:
    mask = gray > 14
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def hough_rings(
    canvas: np.ndarray, signal: np.ndarray | None = None
) -> tuple[tuple[float, float] | None, list[float]]:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # signal 为空时仍用照片灰度。传入时只替换边缘图，孔径仍从照片来。
    src = gray if signal is None else signal
    blur = cv2.GaussianBlur(src, (5, 5), 0)
    field = _inside(gray)
    edges = cv2.Canny(blur, 35, 110)
    edges = edges & field
    ys, xs = np.nonzero(edges)
    if len(xs) < 80:
        return None, []
    if len(xs) > 2500:
        pick = np.linspace(0, len(xs) - 1, 2500).astype(np.int32)
        xs = xs[pick]
        ys = ys[pick]
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    vx = gx[ys, xs]
    vy = gy[ys, xs]
    norm = np.hypot(vx, vy)
    keep = norm > 8
    if int(keep.sum()) < 80:
        return None, []
    xs = xs[keep].astype(np.float64)
    ys = ys[keep].astype(np.float64)
    nx = vx[keep] / norm[keep]
    ny = vy[keep] / norm[keep]
    radii = np.arange(18.0, 1200.0, 4.0)
    cx = np.concatenate([
        xs[:, None] - radii[None, :] * nx[:, None],
        xs[:, None] + radii[None, :] * nx[:, None],
    ]).ravel()
    cy = np.concatenate([
        ys[:, None] - radii[None, :] * ny[:, None],
        ys[:, None] + radii[None, :] * ny[:, None],
    ]).ravel()
    hist, x_edges, y_edges = np.histogram2d(cx, cy, bins=[576, 576], range=[[-768, 1536], [-768, 1536]])
    peak = np.unravel_index(int(np.argmax(hist)), hist.shape)
    if float(hist[peak]) < 25:
        return None, []
    center_x = float(0.5 * (x_edges[peak[0]] + x_edges[peak[0] + 1]))
    center_y = float(0.5 * (y_edges[peak[1]] + y_edges[peak[1] + 1]))
    dist = np.hypot(xs - center_x, ys - center_y)
    bins = np.arange(12.0, float(dist.max()) + 2.0, 1.0)
    counts, edges = np.histogram(dist, bins=bins)
    smooth = gaussian_filter1d(counts.astype(np.float64), 1.6)
    peaks, _ = find_peaks(smooth, height=max(6.0, 0.18 * float(smooth.max())), distance=7)
    found = [float(0.5 * (edges[i] + edges[i + 1])) for i in peaks]
    merged: list[float] = []
    for radius in found:
        if merged and radius - merged[-1] < 6.0:
            merged[-1] = 0.5 * (merged[-1] + radius)
        else:
            merged.append(radius)
    if len(merged) < 3:
        return (center_x, center_y), []
    return (center_x, center_y), merged


def _r2_score(gray: np.ndarray, xx: np.ndarray, yy: np.ndarray, field: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    rr = np.hypot(xx - cx, yy - cy)
    sel = field & (rr > 8) & (rr < 130)
    if int(sel.sum()) < 250:
        return -1.0, 0.0
    r2 = rr[sel] ** 2
    values = gray[sel].astype(np.float64)
    lo, hi = float(r2.min()), float(r2.max())
    if hi - lo < 400:
        return -1.0, 0.0
    bins = np.linspace(lo, hi, 65)
    index = np.clip(np.digitize(r2, bins) - 1, 0, 63)
    weight = np.bincount(index, minlength=64).astype(np.float64)
    signal = np.bincount(index, weights=values, minlength=64) / np.maximum(weight, 1.0)
    signal = (signal - signal.mean()) * np.hanning(64)
    spectrum = np.abs(np.fft.rfft(signal))
    band = spectrum[2:22]
    k = int(np.argmax(band)) + 2
    return float(band.max()), (hi - lo) / k


def frft_rings(
    canvas: np.ndarray, signal: np.ndarray | None = None
) -> tuple[tuple[float, float] | None, list[float]]:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    src = gray if signal is None else signal
    small = cv2.resize(src, (256, 256), interpolation=cv2.INTER_AREA)
    small_gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    scale = gray.shape[1] / 256.0
    field = small_gray > 14
    yy, xx = np.mgrid[0:256, 0:256]
    best_score = -1.0
    best_xy = (128.0, 128.0)
    best_dr2 = 0.0
    for cy in np.arange(-256.0, 520.0, 8.0):
        for cx in np.arange(-256.0, 520.0, 8.0):
            score, dr2 = _r2_score(small, xx, yy, field, float(cx), float(cy))
            if score > best_score:
                best_score, best_xy, best_dr2 = score, (float(cx), float(cy)), dr2
    cx0, cy0 = best_xy
    for cy in np.arange(cy0 - 10.0, cy0 + 11.0, 2.0):
        for cx in np.arange(cx0 - 10.0, cx0 + 11.0, 2.0):
            score, dr2 = _r2_score(small, xx, yy, field, float(cx), float(cy))
            if score > best_score:
                best_score, best_xy, best_dr2 = score, (float(cx), float(cy)), dr2
    if best_score < 0:
        return None, []
    center_x = best_xy[0] * scale
    center_y = best_xy[1] * scale
    full = src.astype(np.float64)
    field_full = _inside(gray)
    ys, xs = np.nonzero(field_full)
    dist = np.hypot(xs - center_x, ys - center_y)
    if len(dist) < 500:
        return (center_x, center_y), []
    r_max = min(900.0, float(np.percentile(dist, 99)))
    bins = np.arange(10.0, r_max, 1.0)
    index = np.clip(np.digitize(dist, bins) - 1, 0, len(bins) - 2)
    weight = np.bincount(index, minlength=len(bins) - 1).astype(np.float64)
    profile = np.bincount(index, weights=full[ys, xs], minlength=len(bins) - 1) / np.maximum(weight, 1.0)
    usable = weight > 25
    profile = gaussian_filter1d(profile, 1.8)
    # 暗纹是径向剖面上的极小。间距用 r^2 频谱给出的周期换到半径上，避免把噪声凹坑都算成环。
    dr2_full = best_dr2 * scale * scale
    usable_r = bins[:-1][usable]
    mid = float(np.median(usable_r)) if len(usable_r) else 120.0
    min_gap = 6.0
    if dr2_full > 1.0:
        min_gap = max(6.0, 0.45 * dr2_full / (2.0 * max(mid, 1.0)))
    peaks, _ = find_peaks(-profile, prominence=3.5, distance=max(1, int(min_gap)), height=None)
    found = []
    for i in peaks:
        if not usable[i]:
            continue
        if profile[i] > np.nanmedian(profile[usable]) + 6:
            continue
        found.append(float(bins[i] + 0.5))
    if len(found) < 3:
        return (center_x, center_y), []
    return (center_x, center_y), found


def _run(name: str, fn, out_dir: Path) -> list[dict]:
    from run_deployed_measure import _draw_fit, _human, _judge, _sheet  # noqa: WPS433

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    rows = []
    tiles = []
    for human_path in sorted(SESSION.glob("*_human.json")):
        stem = human_path.name[: -len("_human.json")]
        canvas = cv2.imread(str(SESSION / f"{stem}_canvas.png"))
        if canvas is None:
            continue
        height, width = canvas.shape[:2]
        payload = json.loads(human_path.read_text(encoding="utf-8"))
        t0 = time.perf_counter()
        center, radii = fn(canvas)
        elapsed = time.perf_counter() - t0
        human = _human(payload, width, height)
        verdict = _judge(human, radii, center, width, height)
        subset = str(payload.get("subset") or stem.split("_", 1)[-1])
        row = {
            "stem": stem,
            "subset": subset,
            "seconds": round(elapsed, 3),
            "n_human": human["n"],
            "n_pred": len(radii),
            "center_xy": None if center is None else [round(center[0], 2), round(center[1], 2)],
            "radii_px": [round(v, 2) for v in radii],
            **verdict,
        }
        rows.append(row)
        mark = "成功" if verdict["success"] else ("拒测" if verdict["refused"] else "未成功")
        view = _draw_fit(canvas, human, radii, center, [
            f"{stem}  {name}  {mark}",
            f"人标 {human['n']} 环 / 输出 {len(radii)} 环 / 配上 {verdict['matches']}",
            f"漏 {verdict['misses']}  多 {verdict['false_rings']}  顺序 {verdict['order_ok']}  圆心 {verdict['center_ok']}",
        ])
        cv2.imwrite(str(out_dir / "frames" / f"{stem}.png"), view)
        tiles.append((stem, view))
        print(f"{name} {stem} {mark} pred={len(radii)} match={verdict['matches']} {elapsed:.2f}s", flush=True)
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["subset"]].append(row)
    lines = [
        f"{name}。同一 43 张，结构判据未改：0.4 间距、0.5 间距、漏环不超过三成、多出的环必须为 0。",
        "绿圈是人工同心圆，黄圈是本方法。这不是论文表格。",
        "",
    ]
    for subset in ("Easy", "Open-arc", "Illegible"):
        group = groups.get(subset, [])
        if not group:
            continue
        ok = sum(1 for row in group if row["success"])
        refused = sum(1 for row in group if row["refused"])
        lines.append(f"{subset}: {ok}/{len(group)} = {100.0 * ok / len(group):.1f}%  拒测 {refused}")
    ok = sum(1 for row in rows if row["success"])
    refused = sum(1 for row in rows if row["refused"])
    lines += ["", f"合计: {ok}/{len(rows)} = {100.0 * ok / max(1, len(rows)):.1f}%  拒测 {refused}", ""]
    for row in rows:
        mark = "成功" if row["success"] else ("拒测" if row["refused"] else "未成功")
        lines.append(
            f"{row['stem']}: {mark}  人标{row['n_human']} 输出{row['n_pred']} 配上{row['matches']} "
            f"漏{row['misses']} 多{row['false_rings']} 顺序{row['order_ok']} 圆心{row['center_ok']}  {row['seconds']}s"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    if tiles:
        cv2.imwrite(str(out_dir / "review_all.png"), _sheet(tiles))
    return rows


def _compare(hough_rows: list[dict], frft_rows: list[dict]) -> None:
    deployed = json.loads(DEPLOYED.read_text(encoding="utf-8")) if DEPLOYED.is_file() else []
    by_name = {
        "本文部署": {row["stem"]: row for row in deployed},
        "Hough草图": {row["stem"]: row for row in hough_rows},
        "FRFT型": {row["stem"]: row for row in frft_rows},
    }
    lines = [
        "三列都用同一结构判据。本文列是已经跑过的部署流程。Hough 与 FRFT 是本次查看用初版。",
        "成功数 / 该子集张数。拒测另计，拒测本身也不是成功。",
        "",
    ]
    stems = [row["stem"] for row in hough_rows]
    subsets = {row["stem"]: row["subset"] for row in hough_rows}
    for subset in ("Easy", "Open-arc", "Illegible"):
        chosen = [stem for stem in stems if subsets[stem] == subset]
        bits = []
        for name, table in by_name.items():
            ok = sum(1 for stem in chosen if table.get(stem, {}).get("success"))
            bits.append(f"{name} {ok}/{len(chosen)}")
        lines.append(f"{subset}: " + " | ".join(bits))
    lines.append("")
    header = f"{'帧':<16}{'子集':<10}{'本文':<8}{'Hough':<8}{'FRFT':<8}"
    lines.append(header)
    for stem in stems:
        marks = []
        for name in ("本文部署", "Hough草图", "FRFT型"):
            row = by_name[name].get(stem)
            if row is None:
                marks.append("-")
            elif row["success"]:
                marks.append("成功")
            elif row["refused"]:
                marks.append("拒测")
            else:
                marks.append("未成")
        lines.append(f"{stem:<16}{subsets[stem]:<10}{marks[0]:<8}{marks[1]:<8}{marks[2]:<8}")
    out = RESULTS / "measure_compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hough_rows = _run("Hough草图", hough_rings, RESULTS / "measure_hough")
    frft_rows = _run("FRFT型", frft_rings, RESULTS / "measure_frft")
    _compare(hough_rows, frft_rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
