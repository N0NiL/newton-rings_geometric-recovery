"""
从极心沿角度射线在概率×灰度响应上找峰，得到同心环稀疏点集（论文式径向采样思路）。
输出为 OpenCV 轮廓列表，供 augment_with_radial_peak_auxiliary 等辅助展示使用。
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


def _normalize_gray(gray: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    g = gray.astype(np.float32)
    if g.shape[:2] != (h, w):
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    g = np.clip(g / 255.0, 0.0, 1.0)
    return g


class RadialPeakSampler:
    def __init__(
        self,
        n_angles: int = 144,
        n_radial: int = 512,
        min_peak_distance_px: float = 5.0,
        prominence_frac: float = 0.04,
        max_rings_per_ray: int = 24,
        gray_blend: float = 0.72,
    ):
        self.n_angles = int(max(24, n_angles))
        self.n_radial = int(max(64, n_radial))
        self.min_peak_distance_px = float(max(1.0, min_peak_distance_px))
        self.prominence_frac = float(max(1e-6, prominence_frac))
        self.max_rings_per_ray = int(max(2, max_rings_per_ray))
        self.gray_blend = float(np.clip(gray_blend, 0.0, 1.0))

    @staticmethod
    def default_center_hint(
        prob_hw: np.ndarray,
        prev_center: tuple[float, float] | None = None,
        warm_alpha: float = 0.55,
        prob_thr: float = 0.42,
    ) -> tuple[float, float]:
        p = np.clip(prob_hw.astype(np.float32), 0.0, 1.0)
        mask = p >= prob_thr
        if not np.any(mask):
            t = float(np.quantile(p, 0.82))
            mask = p >= t
        if not np.any(mask):
            if prev_center is not None:
                return float(prev_center[0]), float(prev_center[1])
            h, w = prob_hw.shape[:2]
            return float(w) * 0.5, float(h) * 0.5
        ys, xs = np.nonzero(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
        if prev_center is not None:
            a = float(np.clip(warm_alpha, 0.0, 1.0))
            cx = (1.0 - a) * cx + a * prev_center[0]
            cy = (1.0 - a) * cy + a * prev_center[1]
        return cx, cy

    def _r_bounds(self, h: int, w: int, cx: float, cy: float) -> tuple[float, float]:
        r_max = float(
            min(cx, cy, w - 1 - cx, h - 1 - cy, 0.48 * min(h, w))
        )
        r_min = max(6.0, 0.02 * min(h, w))
        if r_max <= r_min + 5.0:
            r_max = r_min + float(min(h, w)) * 0.45
        return r_min, r_max

    def build_ring_point_contours(
        self,
        prob_hw: np.ndarray,
        gray_hw: np.ndarray,
        cx: float,
        cy: float,
    ) -> list[np.ndarray]:
        prob = np.clip(prob_hw.astype(np.float32), 0.0, 1.0)
        h, w = prob.shape[:2]
        gray = _normalize_gray(gray_hw, (h, w))
        r_min, r_max = self._r_bounds(h, w, cx, cy)
        radii = np.linspace(r_min, r_max, self.n_radial, dtype=np.float64)
        dr = float(radii[1] - radii[0]) if len(radii) > 1 else 1.0
        dist_idx = max(2, int(self.min_peak_distance_px / (dr + 1e-9)))

        peak_count_max = 0
        per_ray_peaks: list[list[float]] = []

        for i in range(self.n_angles):
            theta = 2.0 * np.pi * (i / float(self.n_angles))
            c, s = np.cos(theta), np.sin(theta)
            xs = cx + radii * c
            ys = cy + radii * s
            coords = np.array([ys, xs], dtype=np.float64)
            pv = map_coordinates(prob, coords, order=1, mode="constant", cval=0.0)
            gv = map_coordinates(gray, coords, order=1, mode="constant", cval=0.0)
            sig = pv * ((1.0 - self.gray_blend) + self.gray_blend * gv)
            sig = np.clip(sig, 0.0, None)
            prom = max(float(np.max(sig)) * self.prominence_frac, 1e-9)
            peaks, _ = find_peaks(sig, distance=dist_idx, prominence=prom)
            rpk = [float(radii[k]) for k in peaks[: self.max_rings_per_ray]]
            per_ray_peaks.append(rpk)
            peak_count_max = max(peak_count_max, len(rpk))

        if peak_count_max < 1:
            return []

        ring_xy: list[list[tuple[float, float]]] = [[] for _ in range(peak_count_max)]
        for i in range(self.n_angles):
            theta = 2.0 * np.pi * (i / float(self.n_angles))
            c, s = np.cos(theta), np.sin(theta)
            for j, rpk in enumerate(per_ray_peaks[i]):
                if j >= peak_count_max:
                    break
                ring_xy[j].append((float(cx + rpk * c), float(cy + rpk * s)))

        contours: list[np.ndarray] = []
        for pts in ring_xy:
            if len(pts) < 16:
                continue
            contours.append(np.array(pts, dtype=np.float32).reshape(-1, 1, 2))
        return contours
