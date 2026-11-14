"""算法装配 2.0。

网络权重不变，输入仍是学生模型的中心线概率图。不二值化，不做骨架，不用 80.6/198.8 黑名单，也不设射线峰数上限。

四项都归一化到 0–1，环的分数是乘积。乘积够三条就用乘积结果。不够三条时不整帧拒测，改发 v1 射线聚出来的环。

- 连通：这条半径上，高于阈值的概率是否集中在少数几段里。径向间距只用来决定多近才算同一条纹，不给整帧贴密纹/疏纹标签。
- 跨度：最长一段沿圆周的连续高概率，18° 饱和。横竖投影不算，避免把目镜十字当成环。
- 规律：检测到的半径是否落在同一条 r²–k 直线上。残差小到一定程度就饱和为 1，不再加分，也不按公式补没看见的环。残差过大则这一项为 0。
- 重合：几何圆在视场内采样到的概率，相对本帧概率脊的高度。

这些常数在看到正式 43 张的分数之前写定。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.stats import siegelslopes

from RadialRingAccumulator import RadialRingAccumulator

# 乘积低于此值的环不发出。0.12 约等于四项都在 0.6 附近。
PRODUCT_MIN = 0.12
# 沿圆周的跨度达到该角度后，跨度项饱和。
SPAN_SATURATE_DEG = 18.0
# 规律项：相对步长的 r² 残差低于此值则饱和为 1，高于 REG_ZERO 则为 0。
REG_SATURATE = 0.25
REG_ZERO = 0.55
# 连通项：最长段占高概率样本的比例达到此值则饱和。
CONN_SATURATE = 0.45


class RingAssemblyV2:
    def fit(
        self,
        prob_hw: np.ndarray,
        field_mask: np.ndarray | None = None,
        gray_hw: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        prob = np.clip(np.asarray(prob_hw, dtype=np.float32), 0.0, 1.0)
        if prob.ndim == 3:
            prob = prob[..., 0]
        height, width = prob.shape[:2]
        field = self._field(prob, field_mask)
        reference = self._ridge_reference(prob, field)
        if reference < 0.02:
            return None

        guide = gray_hw if gray_hw is not None else prob
        seeds = self._spectral_seeds(guide, field)
        refined_seeds = []
        for seed_x, seed_y in seeds:
            refined_seeds.append(self._local_product_peak(prob, field, reference, seed_x, seed_y))
        best: dict[str, Any] | None = None
        best_key = -1.0
        for seed_x, seed_y in refined_seeds:
            refined_x, refined_y = self._refine_center(prob, field, seed_x, seed_y, 8.0, reference)
            fitted = self._score_center(prob, field, refined_x, refined_y, 8.0, reference)
            if len(fitted["rs"]) < 3:
                ray_fit = self._score_center(
                    prob, field, refined_x, refined_y, 8.0, reference, use_v1_rays=True, fallback=True
                )
                if len(ray_fit["rs"]) >= 3:
                    fitted = ray_fit
            count = len(fitted["rs"])
            if count >= 3 and float(fitted["frame_score"]) > 0:
                key = float(fitted["frame_score"]) * min(count, 12) / 12.0
            elif count >= 3:
                key = 0.01 * min(count, 12) / 12.0
            else:
                key = 0.0
            if key > best_key:
                best_key = key
                best = fitted
        return best

    def _spectral_seeds(self, guide: np.ndarray, field: np.ndarray) -> list[tuple[float, float]]:
        """灰度在 r² 上的频谱峰。只做圆心初值，环仍由概率上的四项乘积决定。"""
        image = np.asarray(guide, dtype=np.float32)
        if image.ndim == 3:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        small = cv2.resize(image, (192, 192), interpolation=cv2.INTER_AREA)
        small_field = cv2.resize(field.astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST) > 0
        scale = image.shape[1] / 192.0
        yy, xx = np.mgrid[0:192, 0:192]
        xs = np.arange(-192.0, 384.0, 12.0)
        ys = np.arange(-192.0, 384.0, 12.0)
        grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                grid[iy, ix] = self._r2_concentration(small, small_field, xx, yy, float(cx), float(cy))
        peak = float(grid.max())
        if peak <= 0:
            return [(image.shape[1] * 0.5, image.shape[0] * 0.5)]
        ranked: list[tuple[float, float, float]] = []
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                value = float(grid[iy, ix])
                if value < 0.65 * peak:
                    continue
                patch = grid[max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2]
                if value + 1e-9 < float(patch.max()):
                    continue
                ranked.append((value, float(xs[ix]), float(ys[iy])))
        ranked.sort(key=lambda item: item[0], reverse=True)
        seeds: list[tuple[float, float]] = []
        for _value, cx, cy in ranked:
            best_score, best_xy = -1.0, (cx, cy)
            for cy2 in np.arange(cy - 14.0, cy + 15.0, 2.0):
                for cx2 in np.arange(cx - 14.0, cx + 15.0, 2.0):
                    score = self._r2_concentration(small, small_field, xx, yy, float(cx2), float(cy2))
                    if score > best_score:
                        best_score, best_xy = score, (float(cx2), float(cy2))
            full_xy = (best_xy[0] * scale, best_xy[1] * scale)
            if seeds and min(np.hypot(full_xy[0] - sx, full_xy[1] - sy) for sx, sy in seeds) <= 36.0:
                continue
            seeds.append(full_xy)
            if len(seeds) >= 3:
                break
        return seeds or [(image.shape[1] * 0.5, image.shape[0] * 0.5)]

    @staticmethod
    def _r2_concentration(
        small: np.ndarray,
        field: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
    ) -> float:
        rr = np.hypot(xx - cx, yy - cy)
        selected = field & (rr > 6) & (rr < 100)
        if int(selected.sum()) < 200:
            return 0.0
        r2 = rr[selected] ** 2
        values = small[selected].astype(np.float64)
        lo, hi = float(r2.min()), float(r2.max())
        if hi - lo < 400:
            return 0.0
        bins = np.linspace(lo, hi, 65)
        index = np.clip(np.digitize(r2, bins) - 1, 0, 63)
        weight = np.bincount(index, minlength=64).astype(np.float64)
        signal = np.bincount(index, weights=values, minlength=64) / np.maximum(weight, 1.0)
        signal = (signal - float(signal.mean())) * np.hanning(64)
        band = np.abs(np.fft.rfft(signal))[2:22]
        return float(band.max() / (float(np.median(band)) + 1e-9))

    def _local_product_peak(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        reference: float,
        cx: float,
        cy: float,
    ) -> tuple[float, float]:
        best = (0.0, cx, cy)
        for cy2 in np.arange(cy - 72.0, cy + 73.0, 12.0):
            for cx2 in np.arange(cx - 72.0, cx + 73.0, 12.0):
                fitted = self._score_center(
                    prob, field, float(cx2), float(cy2), 8.0, reference, n_angles=72, n_radial=420
                )
                count = len(fitted["rs"])
                score = float(fitted["frame_score"]) * min(count, 12) / 12.0 if count >= 3 else 0.0
                if score > best[0]:
                    best = (score, float(cx2), float(cy2))
        return best[1], best[2]

    def _search_centers(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        reference: float,
        width: int,
        height: int,
    ) -> list[tuple[float, float]]:
        """粗网格上用四项乘积搜圆心，画外也搜。快搜会高估环数，所以留下几个种子再精评。"""

        def rank(cx: float, cy: float) -> float:
            groups = self._ray_groups(prob, field, cx, cy, 8.0, reference, n_angles=72, n_radial=420)
            usable = [group for group in groups if len(group["angles"]) >= 4]
            if len(usable) < 3:
                return 0.0
            radii = np.array([group["radius"] for group in usable], dtype=np.float64)
            slope, intercept = siegelslopes(radii ** 2, np.arange(radii.size, dtype=np.float64))
            step = abs(float(slope)) + 1e-6
            relative = float(np.median(np.abs(radii ** 2 - (intercept + slope * np.arange(radii.size))) / step))
            regularity = 1.0 if relative <= 0.45 else max(0.0, 1.0 - relative)
            return float(len(usable)) * regularity

        found: list[tuple[float, float, float]] = []
        for cy in np.arange(-240.0, 1320.0, 120.0):
            for cx in np.arange(-480.0, 1200.0, 120.0):
                value = rank(float(cx), float(cy))
                if value > 0:
                    found.append((value, float(cx), float(cy)))
        found.sort(key=lambda item: item[0], reverse=True)
        seeds: list[tuple[float, float, float]] = []
        for value, cx, cy in found:
            if seeds and min(np.hypot(cx - sx, cy - sy) for _sv, sx, sy in seeds) <= 80.0:
                continue
            local_best = (value, cx, cy)
            for cy2 in np.arange(cy - 90.0, cy + 91.0, 30.0):
                for cx2 in np.arange(cx - 90.0, cx + 91.0, 30.0):
                    local = rank(float(cx2), float(cy2))
                    if local > local_best[0]:
                        local_best = (local, float(cx2), float(cy2))
            seeds.append(local_best)
            if len(seeds) >= 4:
                break
        if not seeds:
            return [(width * 0.5, height * 0.5)]
        return [(cx, cy) for _value, cx, cy in seeds]

    def _field(self, prob: np.ndarray, field_mask: np.ndarray | None) -> np.ndarray:
        height, width = prob.shape[:2]
        if field_mask is None:
            mask = prob > 0.02
        else:
            mask = np.asarray(field_mask)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = mask > 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        return cv2.erode(mask.astype(np.uint8), kernel) > 0

    @staticmethod
    def _ridge_reference(prob: np.ndarray, field: np.ndarray) -> float:
        values = prob[field] if int(field.sum()) > 50 else prob.ravel()
        if values.size == 0:
            return 0.0
        return float(np.percentile(values, 99.5))

    def _coarse_seeds(
        self, prob: np.ndarray, field: np.ndarray
    ) -> list[tuple[float, float, float]]:
        small = cv2.resize(prob, (192, 192), interpolation=cv2.INTER_AREA)
        small_field = cv2.resize(field.astype(np.uint8), (192, 192), interpolation=cv2.INTER_NEAREST) > 0
        scale = prob.shape[1] / 192.0
        yy, xx = np.mgrid[0:192, 0:192]
        xs = np.arange(-192.0, 384.0, 12.0)
        ys = np.arange(-192.0, 384.0, 12.0)
        grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
        spacing_grid = np.zeros_like(grid)
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                score, spacing = self._cheap_score(small, small_field, xx, yy, float(cx), float(cy))
                grid[iy, ix] = score
                spacing_grid[iy, ix] = spacing
        peak = float(grid.max())
        if peak <= 0:
            return []
        ranked: list[tuple[float, float, float, float]] = []
        for iy in range(len(ys)):
            for ix in range(len(xs)):
                value = float(grid[iy, ix])
                if value < 0.72 * peak:
                    continue
                patch = grid[max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2]
                if value + 1e-9 < float(patch.max()):
                    continue
                spacing = float(spacing_grid[iy, ix]) * scale
                ranked.append((value, float(xs[ix]) * scale, float(ys[iy]) * scale, spacing if spacing > 1.0 else 8.0))
        ranked.sort(key=lambda item: item[0], reverse=True)
        polished: list[tuple[float, float, float, float]] = []
        for value, center_x, center_y, _spacing in ranked[:3]:
            px, py, spacing, score = self._polish_peak(small, small_field, xx, yy, center_x / scale, center_y / scale)
            polished.append((score, px * scale, py * scale, spacing * scale if spacing > 0.4 else 8.0))
        polished.sort(key=lambda item: item[0], reverse=True)
        ranked = polished
        seeds: list[tuple[float, float, float]] = []
        for _value, center_x, center_y, spacing in ranked:
            if seeds and min(np.hypot(center_x - sx, center_y - sy) for sx, sy, _sp in seeds) <= 18.0:
                continue
            seeds.append((center_x, center_y, spacing))
            if len(seeds) >= 4:
                break
        if not seeds:
            iy, ix = np.unravel_index(int(np.argmax(grid)), grid.shape)
            spacing = float(spacing_grid[iy, ix]) * scale
            seeds.append((float(xs[ix]) * scale, float(ys[iy]) * scale, spacing if spacing > 1.0 else 8.0))
        return seeds

    def _polish_peak(
        self,
        small: np.ndarray,
        field: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
    ) -> tuple[float, float, float, float]:
        best_score = -1.0
        best = (cx, cy, 0.0, 0.0)
        for cy2 in np.arange(cy - 16.0, cy + 17.0, 2.0):
            for cx2 in np.arange(cx - 16.0, cx + 17.0, 2.0):
                score, spacing = self._cheap_score(small, field, xx, yy, float(cx2), float(cy2))
                if score > best_score:
                    best_score = score
                    best = (float(cx2), float(cy2), spacing, score)
        return best

    @staticmethod
    def _cheap_score(
        small: np.ndarray,
        field: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
    ) -> tuple[float, float]:
        del xx, yy
        height, width = small.shape[:2]
        radii = np.linspace(5.0, 96.0, 64)
        accumulated = np.zeros(radii.shape, dtype=np.float64)
        for angle in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
            px = cx + radii * np.cos(angle)
            py = cy + radii * np.sin(angle)
            inside = (px >= 1) & (px < width - 2) & (py >= 1) & (py < height - 2)
            if int(inside.sum()) < 6:
                continue
            signal = np.zeros(radii.shape, dtype=np.float64)
            sampled = map_coordinates(small, np.vstack([py[inside], px[inside]]), order=1, mode="constant", cval=0.0)
            signal[inside] = sampled
            fx = np.clip(np.rint(px[inside]).astype(np.int32), 0, width - 1)
            fy = np.clip(np.rint(py[inside]).astype(np.int32), 0, height - 1)
            signal[inside] = np.where(field[fy, fx], signal[inside], 0.0)
            peaks, _ = find_peaks(signal, distance=3, prominence=0.015)
            accumulated[peaks] += signal[peaks]
        if float(accumulated.max()) <= 0:
            return 0.0, 0.0
        strong, _ = find_peaks(accumulated, distance=2, height=0.25 * float(accumulated.max()))
        if strong.size >= 2:
            spacing = float(np.median(np.diff(radii[strong])))
        else:
            spacing = 0.0
        # 同一半径上有越多射线同时踩到概率脊，圆心越站得住。
        return float(accumulated.max()), spacing

    def _refine_center(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        cx: float,
        cy: float,
        spacing_hint: float,
        reference: float,
    ) -> tuple[float, float]:
        groups = self._ray_groups(prob, field, cx, cy, spacing_hint, reference)
        usable = [group for group in groups if len(group["angles"]) >= 8]
        if len(usable) < 2:
            return cx, cy
        points = []
        indices = []
        weights = []
        guesses = []
        for index, group in enumerate(usable):
            angles = group["angles"]
            radius = float(group["radius"])
            px = cx + radius * np.cos(angles)
            py = cy + radius * np.sin(angles)
            points.append(np.column_stack([px, py]))
            indices.append(np.full(len(angles), index))
            weights.append(np.maximum(group["heights"], 1e-3))
            guesses.append(radius)
        all_pts = np.vstack(points)
        all_idx = np.concatenate(indices)
        all_w = np.sqrt(np.concatenate(weights))

        def residuals(params: np.ndarray) -> np.ndarray:
            center_x, center_y = params[0], params[1]
            radii = params[2:]
            raw = np.hypot(all_pts[:, 0] - center_x, all_pts[:, 1] - center_y) - radii[all_idx]
            return all_w * raw

        init = np.array([cx, cy, *guesses], dtype=np.float64)
        lower = np.array([-np.inf, -np.inf, *([0.0] * len(guesses))])
        upper = np.full_like(init, np.inf)
        try:
            solved = least_squares(residuals, init, bounds=(lower, upper), method="trf", loss="soft_l1", f_scale=2.5)
        except Exception:
            return cx, cy
        new_x, new_y = float(solved.x[0]), float(solved.x[1])
        if not np.isfinite(new_x) or not np.isfinite(new_y):
            return cx, cy
        if np.hypot(new_x - cx, new_y - cy) > 90.0:
            return cx, cy
        return new_x, new_y

    def _score_center(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        cx: float,
        cy: float,
        spacing_hint: float,
        reference: float,
        n_angles: int = 180,
        n_radial: int = 720,
        use_v1_rays: bool = False,
        fallback: bool = False,
    ) -> dict[str, Any]:
        if use_v1_rays:
            groups = self._v1_ray_groups(prob, cx, cy)
        else:
            groups = self._ray_groups(
                prob, field, cx, cy, spacing_hint, reference, n_angles=n_angles, n_radial=n_radial
            )
        if len(groups) < 3:
            kept = [
                {"radius": float(group["radius"]), "product": 0.0, "source": "v1_ray"}
                for group in groups
            ] if fallback else []
            return {
                "x0": cx,
                "y0": cy,
                "rs": [float(ring["radius"]) for ring in kept],
                "frame_score": 0.0,
                "refused": len(kept) < 3,
                "rings": kept,
            }
        measured = self._measure_groups(prob, field, cx, cy, groups, reference)
        strict = [ring for ring in measured if ring["product"] >= PRODUCT_MIN]
        strict.sort(key=lambda ring: ring["radius"])
        strict = self._merge_close(strict)
        # 乘积够 3 条就用乘积。不够时不整帧拒测，改发 v1 射线聚出的环。
        if len(strict) >= 3 or not fallback:
            kept = strict
            score = float(np.median([ring["product"] for ring in kept])) if kept else 0.0
        else:
            kept = [
                {
                    "radius": float(group["radius"]),
                    "overlap": 0.0,
                    "span": 0.0,
                    "connectivity": 0.0,
                    "regularity": 0.0,
                    "product": 0.0,
                    "source": "v1_ray",
                }
                for group in groups
                if len(group["angles"]) >= 4
            ]
            score = 0.0
        return {
            "x0": cx,
            "y0": cy,
            "rs": [float(ring["radius"]) for ring in kept],
            "frame_score": score,
            "refused": len(kept) < 3,
            "rings": kept,
        }

    def _v1_ray_groups(self, prob: np.ndarray, cx: float, cy: float) -> list[dict[str, Any]]:
        """v1 的圆心射线：扫到视场远角，不设峰数上限，不走半径黑名单。"""
        height, width = prob.shape[:2]
        scanner = RadialRingAccumulator()
        ray_peaks, _counts = scanner._scan_rays(prob, float(cx), float(cy), height, width)
        found: list[tuple[float, float, float]] = []
        for index, peaks in enumerate(ray_peaks):
            theta = 2.0 * np.pi * (index / float(max(len(ray_peaks), 1)))
            for radius in peaks:
                found.append((float(radius), theta, 1.0))
        if len(found) < 4:
            return []
        found.sort(key=lambda item: item[0])
        clusters: list[list[tuple[float, float, float]]] = [[found[0]]]
        for item in found[1:]:
            if item[0] - clusters[-1][-1][0] <= 6.0:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        groups = []
        for cluster in clusters:
            if len(cluster) < 4:
                continue
            groups.append({
                "radius": float(np.median([item[0] for item in cluster])),
                "angles": np.array([item[1] for item in cluster], dtype=np.float64),
                "heights": np.ones(len(cluster), dtype=np.float64),
            })
        groups.sort(key=lambda group: group["radius"])
        return groups

    def _ray_groups(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        cx: float,
        cy: float,
        spacing_hint: float,
        reference: float,
        n_angles: int = 180,
        n_radial: int = 720,
    ) -> list[dict[str, Any]]:
        height, width = prob.shape[:2]
        ys, xs = np.nonzero(field)
        if len(xs) < 50:
            return []
        farthest = float(np.percentile(np.hypot(xs - cx, ys - cy), 99))
        r_max = float(np.clip(farthest, 40.0, 1100.0))
        radii = np.linspace(8.0, r_max, n_radial)
        step = float(radii[1] - radii[0])
        # 找峰间距固定在纹宽量级。过宽的间距提示会把密纹并成一条。
        distance_px = 4.0
        distance_idx = max(2, int(distance_px / (step + 1e-9)))
        found: list[tuple[float, float, float]] = []
        for angle in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
            px = cx + radii * np.cos(angle)
            py = cy + radii * np.sin(angle)
            inside = (px >= 0) & (px < width - 1) & (py >= 0) & (py < height - 1)
            signal = np.zeros(radii.shape, dtype=np.float64)
            if int(inside.sum()) < 8:
                continue
            sampled = map_coordinates(prob, np.vstack([py[inside], px[inside]]), order=1, mode="constant", cval=0.0)
            signal[inside] = sampled
            field_hit = np.zeros(radii.shape, dtype=bool)
            fx = np.clip(np.rint(px[inside]).astype(np.int32), 0, width - 1)
            fy = np.clip(np.rint(py[inside]).astype(np.int32), 0, height - 1)
            field_hit[inside] = field[fy, fx]
            signal = np.where(field_hit, signal, 0.0)
            peak_max = float(signal.max()) if signal.size else 0.0
            if peak_max < 0.05 * reference:
                continue
            prominence = max(0.05 * reference, 0.10 * peak_max)
            peaks, _ = find_peaks(signal, distance=distance_idx, prominence=prominence)
            for peak in peaks:
                if signal[peak] < 0.12 * reference:
                    continue
                found.append((float(radii[peak]), float(angle), float(signal[peak])))
        if len(found) < 6:
            return []
        found.sort(key=lambda item: item[0])
        gap = 4.5
        clusters: list[list[tuple[float, float, float]]] = [[found[0]]]
        for item in found[1:]:
            if item[0] - clusters[-1][-1][0] <= gap:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        groups = []
        for cluster in clusters:
            if len(cluster) < 5:
                continue
            radii_c = np.array([item[0] for item in cluster])
            groups.append({
                "radius": float(np.median(radii_c)),
                "angles": np.array([item[1] for item in cluster], dtype=np.float64),
                "heights": np.array([item[2] for item in cluster], dtype=np.float64),
            })
        groups.sort(key=lambda group: group["radius"])
        return groups

    def _measure_groups(
        self,
        prob: np.ndarray,
        field: np.ndarray,
        cx: float,
        cy: float,
        groups: list[dict[str, Any]],
        reference: float,
    ) -> list[dict[str, Any]]:
        radii = np.array([group["radius"] for group in groups], dtype=np.float64)
        r2 = radii ** 2
        order = np.arange(r2.size, dtype=np.float64)
        slope, intercept = siegelslopes(r2, order) if r2.size >= 3 else (0.0, float(r2[0]) if r2.size else 0.0)
        step = abs(float(slope)) + 1e-6
        predicted = intercept + slope * order
        relative = np.abs(r2 - predicted) / step
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        cosine = np.cos(angles)
        sine = np.sin(angles)
        height, width = prob.shape[:2]
        rings = []
        for index, group in enumerate(groups):
            radius = float(group["radius"])
            px = cx + radius * cosine
            py = cy + radius * sine
            valid = (px >= 0) & (px < width - 1) & (py >= 0) & (py < height - 1)
            samples = np.zeros(angles.shape, dtype=np.float64)
            if int(valid.sum()) >= 12:
                sampled = map_coordinates(
                    prob, np.vstack([py[valid], px[valid]]), order=1, mode="constant", cval=0.0
                )
                samples[valid] = sampled
                fx = np.clip(np.rint(px[valid]).astype(np.int32), 0, width - 1)
                fy = np.clip(np.rint(py[valid]).astype(np.int32), 0, height - 1)
                valid = valid.copy()
                valid_idx = np.flatnonzero(valid)
                # 圆穿出视场的角度不参与重合和跨度。
                drop = ~field[fy, fx]
                samples[valid_idx[drop]] = 0.0
                valid[valid_idx[drop]] = False
            present = samples >= 0.28 * reference
            present &= valid
            longest = self._longest_run(present)
            span = float(np.clip((longest * 2.0) / SPAN_SATURATE_DEG, 0.0, 1.0))
            on_count = int(present.sum())
            connectivity = 0.0 if on_count == 0 else float(np.clip((longest / on_count) / CONN_SATURATE, 0.0, 1.0))
            if int(valid.sum()) >= 12:
                overlap = float(np.percentile(samples[valid], 75) / reference)
            else:
                overlap = 0.0
            overlap = float(np.clip(overlap, 0.0, 1.0))
            residual = float(relative[index]) if r2.size >= 3 else 0.0
            if residual <= REG_SATURATE:
                regularity = 1.0
            elif residual >= REG_ZERO:
                regularity = 0.0
            else:
                regularity = float((REG_ZERO - residual) / (REG_ZERO - REG_SATURATE))
            product = overlap * span * connectivity * regularity
            rings.append({
                "radius": radius,
                "overlap": round(overlap, 4),
                "span": round(span, 4),
                "connectivity": round(connectivity, 4),
                "regularity": round(regularity, 4),
                "product": round(product, 4),
            })
        return rings

    @staticmethod
    def _longest_run(flags: np.ndarray) -> int:
        if flags.size == 0 or not bool(flags.any()):
            return 0
        doubled = np.concatenate([flags, flags])
        best = current = 0
        for value in doubled:
            if value:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return int(min(best, flags.size))

    @staticmethod
    def _merge_close(rings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(rings) < 2:
            return rings
        gaps = np.diff([ring["radius"] for ring in rings])
        spacing = float(np.median(gaps)) if gaps.size else 8.0
        limit = max(4.0, 0.40 * spacing)
        merged: list[dict[str, Any]] = []
        for ring in rings:
            if merged and ring["radius"] - merged[-1]["radius"] < limit:
                previous = merged[-1]
                if ring["product"] > previous["product"]:
                    merged[-1] = ring
            else:
                merged.append(ring)
        return merged
