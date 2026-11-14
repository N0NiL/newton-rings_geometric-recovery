"""
径向扫描定环（与 AlgorithmFitter 完全隔离）。

思路：
1. 以模型圆心为极心，向 360° 射线，与模型概率/二值图重叠处记峰点；
2. 筛掉点数过多（杂斑）/过少（断裂或缺环）的射线；
3. 保留峰数落在「期望环数 ± tol」内的射线；
4. 用这些峰半径聚成候选环组，按与模型图的重叠率选最优组；
5. 采样与重叠选组完成、得到全部圆环之后，再按 5px 环带「连通成片」剔除零散假环（不做闭环度门控）；
6. 跨帧惯性累积：圆心/环组不轻易重构，摄像头微动时继续补采样完善。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


@dataclass
class RingSlot:
    """单个稳定环槽：半径 EMA + 角度覆盖 + 命中次数。"""

    r_ema: float
    hit_count: int = 0
    angle_hits: np.ndarray = field(default_factory=lambda: np.zeros(360, dtype=np.float32))
    last_r: float = 0.0
    continuity_ema: float = 0.0  # 仅显示：角向跨度，不作闭环门控
    coherent_mass_ema: float = 0.0  # 定环后剔除用：连通质量

    def coverage_frac(self) -> float:
        return float(np.mean(self.angle_hits > 0.0))


@dataclass
class AccumulatorConfig:
    n_angles: int = 360
    n_radial: int = 640
    prob_thr: float = 0.42
    min_peak_distance_px: float = 6.0
    prominence_frac: float = 0.05
    # 相对期望环数的射线峰数容差
    ray_count_tol: int = 4
    # 峰数绝对上下限（过少=断裂，过多=杂斑）
    ray_count_min_abs: int = 2
    ray_count_max_abs: int = 28
    # 杂斑：峰数相对期望的上限倍数（再与 abs max 取严）
    ray_count_max_over_expect: int = 6
    # 圆心惯性：越大越跟模型；越小越稳
    center_follow: float = 0.18
    # 半径匹配容差（像素）与 EMA
    radius_match_px: float = 8.0
    radius_ema_alpha: float = 0.12
    # 结构变更门控：新环组重叠率需显著更好，且连续确认若干帧
    promote_overlap_margin: float = 0.06
    promote_hold_frames: int = 8
    # 新环槽准入：需在 hold 内多次被看到
    new_slot_min_hits: int = 12
    # 绘制/重叠评分用的环带半宽
    band_half_px: float = 2.0
    # 检测环线粗度（像素）：与模型概率重叠的环带总宽度
    band_width_px: float = 5.0
    # 角向跨度仅作显示，不参与硬剔除（摄像头常只看到开弧，禁止用闭环度门控）
    continuity_min: float = 0.0
    # 环带重叠像素中落入「非零散连通域」的质量占比下限（放宽，避免只剩 1 环）
    coherent_mass_min: float = 0.22
    # 非零散连通域最小像素数（放宽）
    min_cc_area_px: int = 8
    # 零散连通域占比上限：小碎块过多才判假环
    scatter_cc_frac_max: float = 0.85
    # 定环后剔除时至少保留的环数比例（相对剔除前）
    cull_keep_frac_min: float = 0.55
    # 定环后剔除时至少保留的环数绝对下限
    cull_keep_abs_min: int = 3
    # 双环叠加：两半径间距小于此（px）则随机删掉其中一个
    close_radius_dedupe_px: float = 6.0
    # 评分中连续率权重（仅显示用，定环评分不用）
    continuity_score_weight: float = 0.0
    # 期望环数相对历史锁定的软约束
    expect_from_locked: bool = True


class RadialRingAccumulator:
    """跨帧惯性累积的径向定环器。"""

    def __init__(self, cfg: AccumulatorConfig | None = None) -> None:
        self.cfg = cfg or AccumulatorConfig()
        self.center: tuple[float, float] | None = None
        self.slots: list[RingSlot] = []
        self.locked_n: int | None = None
        self.frame_i: int = 0
        self._promote_candidate: list[float] | None = None
        self._promote_hits: int = 0
        self._last_status: str = "未初始化"
        self._last_debug: dict[str, Any] = {}
        self._last_scan_meta: dict[str, float] = {}

    def reset(self) -> None:
        self.center = None
        self.slots = []
        self.locked_n = None
        self.frame_i = 0
        self._promote_candidate = None
        self._promote_hits = 0
        self._last_status = "已复位"
        self._last_debug = {}
        self._last_scan_meta = {}

    def fit_static_once(
        self,
        prob_hw: np.ndarray,
        center_xy: tuple[float, float] | None = None,
        model_n_rings: int | None = None,
    ) -> dict[str, Any] | None:
        """
        静止单帧精拟合：复位后摄入一帧并立即定环（不做跨帧惯性等待）。
        """
        self.reset()
        # 单帧时圆心直接采纳外部/质心，不拖惯性
        old_follow = float(self.cfg.center_follow)
        try:
            self.cfg.center_follow = 1.0
            res = self.update(prob_hw, center_xy=center_xy, model_n_rings=model_n_rings)
        finally:
            self.cfg.center_follow = old_follow
        if res is not None:
            res = dict(res)
            res["fit_mode"] = "radial_static_once"
        self._last_status = (
            "静止单帧定环 | " + self._last_status.replace("径向累积 | ", "", 1)
        )
        return res

    @property
    def status_zh(self) -> str:
        return self._last_status

    @property
    def last_debug(self) -> dict[str, Any]:
        return dict(self._last_debug)

    def radii(self) -> list[float]:
        return [float(s.r_ema) for s in sorted(self.slots, key=lambda s: s.r_ema)]

    def result_dict(self) -> dict[str, Any] | None:
        if self.center is None or len(self.slots) < 1:
            return None
        ordered = sorted(self.slots, key=lambda s: s.r_ema)
        rs = [float(s.r_ema) for s in ordered]
        cov = [float(s.coverage_frac()) for s in ordered]
        cont = [float(s.continuity_ema) for s in ordered]
        coh = [float(s.coherent_mass_ema) for s in ordered]
        return {
            "x0": float(self.center[0]),
            "y0": float(self.center[1]),
            "rs": rs,
            "rs_display": rs,
            "coverages": cov,
            "continuities": cont,
            "coherent_masses": coh,
            "n_rings": len(rs),
            "locked_n": self.locked_n,
            "frame_i": self.frame_i,
            "fit_mode": "radial_accum",
        }

    def update(
        self,
        prob_hw: np.ndarray,
        center_xy: tuple[float, float] | None = None,
        model_n_rings: int | None = None,
    ) -> dict[str, Any] | None:
        """
        摄入一帧模型概率图，更新累积环组。

        :param prob_hw: [0,1] 概率图（模型边长坐标）
        :param center_xy: 模型物理头圆心；None 则用概率质心
        :param model_n_rings: 外部给出的环数提示（可选）
        """
        self.frame_i += 1
        cfg = self.cfg
        prob = np.clip(np.asarray(prob_hw, dtype=np.float32), 0.0, 1.0)
        h, w = prob.shape[:2]
        mask = (prob >= cfg.prob_thr).astype(np.float32)
        if float(mask.sum()) < 50:
            # 阈值过严时放宽到分位数
            thr = float(np.quantile(prob, 0.82))
            mask = (prob >= thr).astype(np.float32)

        cx_m, cy_m = self._resolve_center(prob, center_xy)
        center_src = "fitter" if center_xy is not None else "prob_fallback"
        if self.center is None:
            self.center = (cx_m, cy_m)
        else:
            a = float(np.clip(cfg.center_follow, 0.0, 1.0))
            # 有 AlgorithmFitter 圆心时跟得更快；无圆心时几乎不动，避免被坏质心拖飞
            if center_xy is None:
                a = min(a, 0.05)
            self.center = (
                (1.0 - a) * self.center[0] + a * cx_m,
                (1.0 - a) * self.center[1] + a * cy_m,
            )
        cx, cy = float(self.center[0]), float(self.center[1])

        # —— 360° 射线找峰（采样；本段不做连续率约束）——
        ray_peaks, ray_counts = self._scan_rays(mask, cx, cy, h, w)
        n_expect = self._estimate_n_expect(ray_counts, model_n_rings)
        good_idx = self._filter_rays(ray_counts, n_expect)

        # 本帧候选半径（来自好射线）
        frame_radii: list[float] = []
        for i in good_idx:
            frame_radii.extend(ray_peaks[i])

        # 聚类 + 重叠率选最优环组（仍不涉及连续率）
        cand_groups = self._cluster_radius_groups(frame_radii, n_expect)
        best_group, best_ov = self._pick_best_overlap_group(
            cand_groups, mask, cx, cy, n_expect=n_expect
        )
        n_before_cull = len(best_group)

        # —— 定环之后：5px 环带连续率剔除零散假环 ——
        best_group = self._cull_rings_by_continuity(best_group, mask, cx, cy)
        # 双环叠加：半径仅差几像素 → 随机删一个
        best_group = self._random_drop_close_radii(best_group)
        best_cont = self._mean_continuity(best_group, mask, cx, cy)

        # 与已锁定环组比较：惯性累积，不轻易换结构
        locked_rs = self.radii()
        locked_ov = (
            self._overlap_score(locked_rs, mask, cx, cy, n_expect=n_expect)
            if locked_rs
            else 0.0
        )
        locked_cont = self._mean_continuity(locked_rs, mask, cx, cy) if locked_rs else 0.0

        structural_change = self._maybe_promote_structure(
            best_group, best_ov, locked_rs, locked_ov, n_expect
        )

        # 用本帧好射线峰去更新现有槽（采样累积，与连续率无关）
        self._accumulate_peaks(ray_peaks, good_idx, h, w)

        # 对已定环槽刷新连续率；连续率过低的槽再剔除
        self._refresh_slot_continuity(mask, cx, cy)
        self._prune_incoherent_slots()
        self._prune_weak_slots()
        self._dedupe_close_slots()

        rs = self.radii()
        self.locked_n = len(rs) if rs else self.locked_n
        cov_mean = (
            float(np.mean([s.coverage_frac() for s in self.slots])) if self.slots else 0.0
        )
        cont_mean = (
            float(np.mean([s.continuity_ema for s in self.slots])) if self.slots else 0.0
        )
        self._last_debug = {
            "n_expect": int(n_expect),
            "n_good_rays": int(len(good_idx)),
            "n_rays": int(cfg.n_angles),
            "best_group_ov": float(best_ov),
            "locked_ov": float(locked_ov),
            "n_rings_before_cull": int(n_before_cull),
            "n_rings_after_cull": int(len(best_group)),
            "best_continuity": float(best_cont),
            "locked_continuity": float(locked_cont),
            "structural_change": bool(structural_change),
            "cov_mean": float(cov_mean),
            "cont_mean": float(cont_mean),
            "n_slots": int(len(self.slots)),
            "center_model": (float(cx_m), float(cy_m)),
            "center": (cx, cy),
            "center_src": center_src,
            "band_width_px": float(cfg.band_width_px),
            **{f"scan_{k}": v for k, v in self._last_scan_meta.items()},
        }
        sm = self._last_scan_meta
        self._last_status = (
            f"径向累积 | 帧={self.frame_i} | 圆心=({cx:.1f},{cy:.1f})[{center_src}] | "
            f"环={len(rs)} | 期望n={n_expect} | 好射线={len(good_idx)}/{cfg.n_angles} | "
            f"扫描 r=[{sm.get('r_min', 0):.0f},{sm.get('r_max', 0):.0f}] "
            f"有峰射线={int(sm.get('n_rays_with_peak', 0))} "
            f"峰均={sm.get('mean_peaks', 0):.2f} | "
            f"重叠 锁定={locked_ov:.3f} 候选={best_ov:.3f} | "
            f"定环后连续率剔除 {n_before_cull}→{len(best_group)} "
            f"(候选连续率={best_cont:.2%} 槽均={cont_mean:.2%} 环带{cfg.band_width_px:.0f}px) | "
            f"覆盖均值={cov_mean:.2%} | "
            f"{'结构更新' if structural_change else '保持结构·补采样'}"
        )
        return self.result_dict()

    # ------------------------------------------------------------------
    def _resolve_center(
        self, prob: np.ndarray, center_xy: tuple[float, float] | None
    ) -> tuple[float, float]:
        if center_xy is not None:
            cx, cy = float(center_xy[0]), float(center_xy[1])
            if np.isfinite(cx) and np.isfinite(cy):
                return cx, cy
        thr = float(np.quantile(prob, 0.85))
        ys, xs = np.nonzero(prob >= thr)
        if len(xs) == 0:
            h, w = prob.shape[:2]
            return float(w) * 0.5, float(h) * 0.5
        return float(xs.mean()), float(ys.mean())

    def _scan_rays(
        self, mask: np.ndarray, cx: float, cy: float, h: int, w: int
    ) -> tuple[list[list[float]], np.ndarray]:
        """
        360° 径向采样找峰。
        注意：r_max 取到角点距离，不因圆心靠边而把扫描半径压成 0
        （否则会出现「好射线=0」的假死）。
        """
        cfg = self.cfg
        # 到四角/边界的最远可达距离；越界采样由 map_coordinates cval=0 处理
        r_corner = float(
            np.hypot(max(cx, (w - 1) - cx), max(cy, (h - 1) - cy))
        )
        r_max = float(max(r_corner * 0.98, 0.35 * min(h, w)))
        r_min = max(4.0, 0.015 * min(h, w))
        if r_max <= r_min + 8.0:
            r_max = r_min + 0.45 * float(min(h, w))
        radii = np.linspace(r_min, r_max, cfg.n_radial, dtype=np.float64)
        dr = float(radii[1] - radii[0]) if len(radii) > 1 else 1.0
        dist_idx = max(2, int(cfg.min_peak_distance_px / (dr + 1e-9)))

        ray_peaks: list[list[float]] = []
        counts = np.zeros(cfg.n_angles, dtype=np.int32)
        for i in range(cfg.n_angles):
            theta = 2.0 * np.pi * (i / float(cfg.n_angles))
            c, s = np.cos(theta), np.sin(theta)
            xs = cx + radii * c
            ys = cy + radii * s
            coords = np.array([ys, xs], dtype=np.float64)
            sig = map_coordinates(mask, coords, order=1, mode="constant", cval=0.0)
            sig = np.clip(sig, 0.0, None)
            peak_max = float(np.max(sig)) if sig.size else 0.0
            if peak_max < 1e-6:
                ray_peaks.append([])
                counts[i] = 0
                continue
            prom = max(peak_max * cfg.prominence_frac, 1e-6)
            peaks, _ = find_peaks(sig, distance=dist_idx, prominence=prom)
            # 二值平台上 find_peaks 可能漏检：对连通高响应段取中点补峰
            if len(peaks) == 0 and peak_max >= 0.5:
                on = sig >= max(0.5 * peak_max, cfg.prob_thr * 0.5)
                peaks = self._plateau_midpoints(on)
            rpk = [float(radii[int(k)]) for k in peaks]
            # 按最小峰距再并一次，避免平台中点过密
            rpk = self._merge_close_radii(rpk, cfg.min_peak_distance_px)
            ray_peaks.append(rpk)
            counts[i] = len(rpk)
        self._last_scan_meta = {
            "r_min": float(r_min),
            "r_max": float(r_max),
            "n_rays_with_peak": int(np.sum(counts > 0)),
            "mean_peaks": float(np.mean(counts)),
            "max_peaks": int(np.max(counts)) if counts.size else 0,
        }
        return ray_peaks, counts

    @staticmethod
    def _plateau_midpoints(on: np.ndarray) -> np.ndarray:
        """二值平台的中点索引，供 find_peaks 漏检时补峰。"""
        on = np.asarray(on, dtype=bool)
        if not np.any(on):
            return np.zeros(0, dtype=np.int32)
        idx = np.where(on)[0]
        cuts = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[cuts + 1]]
        ends = np.r_[idx[cuts], idx[-1]]
        mids = ((starts + ends) * 0.5).astype(np.int32)
        return mids

    @staticmethod
    def _merge_close_radii(rs: list[float], min_dist: float) -> list[float]:
        if not rs:
            return []
        out: list[float] = [float(rs[0])]
        for r in rs[1:]:
            r = float(r)
            if r - out[-1] < min_dist:
                out[-1] = 0.5 * (out[-1] + r)
            else:
                out.append(r)
        return out

    def _estimate_n_expect(
        self, ray_counts: np.ndarray, model_n_rings: int | None
    ) -> int:
        if self.cfg.expect_from_locked and self.locked_n is not None and self.locked_n >= 2:
            return int(self.locked_n)
        if model_n_rings is not None and int(model_n_rings) >= 2:
            return int(model_n_rings)
        # 去掉 0 与极端高值后取中位数
        pos = ray_counts[ray_counts > 0]
        if len(pos) == 0:
            return 6
        hi = int(np.percentile(pos, 90))
        core = pos[pos <= max(hi, 3)]
        if len(core) == 0:
            core = pos
        return int(max(2, round(float(np.median(core)))))

    def _filter_rays(self, ray_counts: np.ndarray, n_expect: int) -> list[int]:
        cfg = self.cfg
        lo = max(cfg.ray_count_min_abs, n_expect - cfg.ray_count_tol)
        hi = min(
            cfg.ray_count_max_abs,
            n_expect + cfg.ray_count_tol,
            n_expect + cfg.ray_count_max_over_expect,
        )
        if hi < lo:
            hi = lo
        good = [i for i, c in enumerate(ray_counts.tolist()) if lo <= int(c) <= hi]
        # 若过严导致几乎无射线，放宽到「非零且不超过绝对上限」
        if len(good) < max(12, self.cfg.n_angles // 20):
            good = [
                i
                for i, c in enumerate(ray_counts.tolist())
                if cfg.ray_count_min_abs <= int(c) <= cfg.ray_count_max_abs
            ]
        # 再兜底：凡有峰的射线都保留（避免好射线被滤成 0）
        if len(good) == 0:
            good = [i for i, c in enumerate(ray_counts.tolist()) if int(c) > 0]
        return good

    def _cluster_radii_1d(self, frame_radii: list[float]) -> list[float]:
        """按半径间隙合并为环中心（比直方图峰更稳）。"""
        if len(frame_radii) < 2:
            return [float(frame_radii[0])] if frame_radii else []
        rs = np.sort(np.asarray(frame_radii, dtype=np.float64))
        gap = float(max(self.cfg.min_peak_distance_px * 0.85, 4.0))
        groups: list[list[float]] = [[float(rs[0])]]
        for r in rs[1:]:
            if float(r) - groups[-1][-1] <= gap:
                groups[-1].append(float(r))
            else:
                groups.append([float(r)])
        # 要求每簇有足够票数，抑制孤峰
        min_votes = max(3, int(0.015 * self.cfg.n_angles))
        centers = [float(np.median(g)) for g in groups if len(g) >= min_votes]
        if not centers and groups:
            # 放宽：取票数最多的若干簇
            groups.sort(key=len, reverse=True)
            centers = [float(np.median(g)) for g in groups[:8] if len(g) >= 2]
        return sorted(centers)

    def _cluster_radius_groups(
        self, frame_radii: list[float], n_expect: int
    ) -> list[list[float]]:
        """由本帧峰半径聚出若干候选环组。"""
        if len(frame_radii) < 4:
            return []
        centers = self._cluster_radii_1d(frame_radii)
        if not centers:
            return []
        rs = np.asarray(frame_radii, dtype=np.float64)
        groups: list[list[float]] = [list(centers)]

        # 围绕期望环数取子集：按簇票数（邻近峰数）排序
        votes = []
        for c in centers:
            votes.append(int(np.sum(np.abs(rs - c) <= self.cfg.radius_match_px)))
        order = np.argsort(-np.asarray(votes))
        lo = max(2, n_expect - self.cfg.ray_count_tol)
        hi = min(len(centers), n_expect + self.cfg.ray_count_tol)
        for k in range(lo, hi + 1):
            pick = sorted(float(centers[int(order[i])]) for i in range(min(k, len(order))))
            if pick and pick not in groups:
                groups.append(pick)

        locked = self.radii()
        if locked:
            refined = []
            for lr in locked:
                near = rs[np.abs(rs - lr) <= self.cfg.radius_match_px]
                refined.append(float(np.median(near)) if len(near) else lr)
            groups.insert(0, refined)
        return groups

    @staticmethod
    def _longest_angular_span(angles: np.ndarray) -> float:
        """角度数组（弧度）的最长连续跨度，处理环绕；返回 [0, 2π]。"""
        if angles.size == 0:
            return 0.0
        a = np.sort(np.mod(angles.astype(np.float64), 2.0 * np.pi))
        if a.size == 1:
            return 0.0
        gaps = np.diff(a)
        wrap = float(a[0] + 2.0 * np.pi - a[-1])
        gaps = np.append(gaps, wrap)
        max_gap = float(np.max(gaps))
        # 最长连续弧 = 整圆减去最大空隙
        return float(max(0.0, 2.0 * np.pi - max_gap))

    def _ring_band_continuity(
        self, mask: np.ndarray, cx: float, cy: float, r: float
    ) -> dict[str, float]:
        """
        5px 环带内模型像素「是否成片连通」（抗零散假环）。

        明确不做闭环度门控：摄像头常只看到开弧，角向跨度只作显示。
        ok 仅看：连通质量 coherent_mass + 碎块是否过多。
        """
        cfg = self.cfg
        h, w = mask.shape[:2]
        band_w = float(max(1.0, cfg.band_width_px))
        band = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(
            band,
            (int(round(cx)), int(round(cy))),
            int(round(r)),
            1,
            thickness=max(1, int(round(band_w))),
            lineType=cv2.LINE_8,
        )
        m = (mask > 0.5).astype(np.uint8)
        overlap = ((band > 0) & (m > 0)).astype(np.uint8)
        n_ov = int(overlap.sum())
        empty = {
            "continuity": 0.0,
            "coherent_mass": 0.0,
            "n_cc": 0.0,
            "n_large_cc": 0.0,
            "n_overlap": 0.0,
            "scatter_frac": 1.0,
            "ok": 0.0,
        }
        if n_ov < 3:
            return empty

        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(overlap, connectivity=8)
        n_cc = max(0, n_lab - 1)
        # 小环半径时 min_area 不要随圆周涨太狠
        min_area = int(max(cfg.min_cc_area_px, 0.12 * r * (np.pi / 36.0)))
        ys, xs = np.nonzero(overlap)
        lab_flat = labels[ys, xs]
        ang_all = np.arctan2(ys.astype(np.float64) - cy, xs.astype(np.float64) - cx)

        large_pix = 0
        small_pix = 0
        best_span = 0.0
        n_large = 0
        for lab in range(1, n_lab):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                small_pix += area
                continue
            n_large += 1
            large_pix += area
            sel = lab_flat == lab
            if not np.any(sel):
                continue
            span = self._longest_angular_span(ang_all[sel])
            if span > best_span:
                best_span = span

        continuity = float(np.clip(best_span / (2.0 * np.pi), 0.0, 1.0))  # 仅显示
        coherent_mass = float(large_pix) / float(n_ov + 1e-6)
        scatter_frac = float(small_pix) / float(n_ov + 1e-6)
        # 硬门控：不要角向闭环；只要「成片」且碎点不太爆
        ok = float(
            (coherent_mass >= cfg.coherent_mass_min and n_large >= 1)
            or (coherent_mass >= cfg.coherent_mass_min * 0.85 and scatter_frac <= cfg.scatter_cc_frac_max)
        )
        return {
            "continuity": continuity,
            "coherent_mass": coherent_mass,
            "n_cc": float(n_cc),
            "n_large_cc": float(n_large),
            "n_overlap": float(n_ov),
            "scatter_frac": scatter_frac,
            "ok": ok,
        }

    def _random_drop_close_radii(self, radii: list[float]) -> list[float]:
        """
        双环叠加：排序后相邻半径差 < close_radius_dedupe_px 时，随机删掉其中一个。
        """
        if len(radii) < 2:
            return list(radii)
        thr = float(max(1.0, self.cfg.close_radius_dedupe_px))
        rs = sorted(float(r) for r in radii)
        out: list[float] = [rs[0]]
        for r in rs[1:]:
            if r - out[-1] < thr:
                # 随机保留旧的或新的
                if random.random() < 0.5:
                    out[-1] = r
            else:
                out.append(r)
        return out

    def _dedupe_close_slots(self) -> None:
        """对已锁定槽做近邻双环随机去重。"""
        if len(self.slots) < 2:
            return
        thr = float(max(1.0, self.cfg.close_radius_dedupe_px))
        ordered = sorted(self.slots, key=lambda s: s.r_ema)
        kept: list[RingSlot] = [ordered[0]]
        for s in ordered[1:]:
            if float(s.r_ema) - float(kept[-1].r_ema) < thr:
                if random.random() < 0.5:
                    kept[-1] = s
            else:
                kept.append(s)
        self.slots = kept

    def _cull_rings_by_continuity(
        self, radii: list[float], mask: np.ndarray, cx: float, cy: float
    ) -> list[float]:
        """
        定环之后：剔除环带内零散像素假环。
        放宽：若硬筛后环数过少，按连通质量保留前 K 个，避免只剩 1 环。
        """
        if not radii:
            return []
        scored: list[tuple[float, float, float]] = []  # (r, coherent_mass, ok)
        for r in radii:
            m = self._ring_band_continuity(mask, cx, cy, float(r))
            scored.append((float(r), float(m["coherent_mass"]), float(m["ok"])))

        kept = [r for r, _cm, ok in scored if ok >= 0.5]
        want = max(
            self.cfg.cull_keep_abs_min,
            int(np.ceil(len(radii) * self.cfg.cull_keep_frac_min)),
        )
        want = min(want, len(radii))
        if len(kept) >= want:
            return sorted(kept)

        # 过少：按连通质量补足到 want（仍优先真环，不强制闭环）
        scored_sorted = sorted(scored, key=lambda t: t[1], reverse=True)
        out = [r for r, _cm, _ok in scored_sorted[:want]]
        return sorted(out)

    def _mean_continuity(
        self, radii: list[float], mask: np.ndarray, cx: float, cy: float
    ) -> float:
        if not radii:
            return 0.0
        vals = [
            float(self._ring_band_continuity(mask, cx, cy, float(r))["continuity"])
            for r in radii
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _overlap_score(
        self,
        radii: list[float],
        mask: np.ndarray,
        cx: float,
        cy: float,
        n_expect: int | None = None,
    ) -> float:
        """纯重叠评分；不含连续率（连续率只在定环后剔除）。"""
        if not radii:
            return 0.0
        h, w = mask.shape[:2]
        band = np.zeros((h, w), dtype=np.uint8)
        half = max(1, int(round(self.cfg.band_half_px)))
        for r in radii:
            cv2.circle(
                band,
                (int(round(cx)), int(round(cy))),
                int(round(r)),
                1,
                thickness=max(1, 2 * half),
                lineType=cv2.LINE_AA,
            )
        band_f = band.astype(np.float32)
        m = mask.astype(np.float32)
        inter = float(np.sum(band_f * m))
        band_sum = float(np.sum(band_f)) + 1e-6
        precision = inter / band_sum
        recall = inter / (float(np.sum(m)) + 1e-6)
        base = float(0.65 * precision + 0.35 * min(1.0, recall * 6.0))
        if n_expect is not None and n_expect >= 2:
            dn = abs(len(radii) - int(n_expect))
            base *= float(np.exp(-0.35 * dn))
        return base

    def _pick_best_overlap_group(
        self,
        groups: list[list[float]],
        mask: np.ndarray,
        cx: float,
        cy: float,
        n_expect: int | None = None,
    ) -> tuple[list[float], float]:
        """按重叠率选组；不在此做连续率剔除。"""
        best: list[float] = []
        best_ov = -1.0
        for g in groups:
            ov = self._overlap_score(g, mask, cx, cy, n_expect=n_expect)
            if ov > best_ov:
                best_ov = ov
                best = list(g)
        return best, float(max(0.0, best_ov))

    def _refresh_slot_continuity(
        self, mask: np.ndarray, cx: float, cy: float
    ) -> None:
        a = 0.25
        for s in self.slots:
            met = self._ring_band_continuity(mask, cx, cy, float(s.r_ema))
            c = float(met["continuity"])
            cm = float(met["coherent_mass"])
            if s.continuity_ema <= 1e-9:
                s.continuity_ema = c
            else:
                s.continuity_ema = (1.0 - a) * s.continuity_ema + a * c
            if s.coherent_mass_ema <= 1e-9:
                s.coherent_mass_ema = cm
            else:
                s.coherent_mass_ema = (1.0 - a) * s.coherent_mass_ema + a * cm

    def _prune_incoherent_slots(self) -> None:
        """已定环之后：连通质量长期极差的槽才剔除（不看闭环度）。"""
        if len(self.slots) < 1:
            return
        if self.frame_i < 10:
            return
        thr = self.cfg.coherent_mass_min * 0.55
        kept = [s for s in self.slots if s.coherent_mass_ema >= thr]
        # 至少保留 cull_keep_abs_min 个（按连通质量排序）
        want = min(len(self.slots), max(self.cfg.cull_keep_abs_min, 2))
        if len(kept) >= want:
            self.slots = kept
            return
        ranked = sorted(self.slots, key=lambda s: s.coherent_mass_ema, reverse=True)
        self.slots = ranked[:want]

    def _maybe_promote_structure(
        self,
        best_group: list[float],
        best_ov: float,
        locked_rs: list[float],
        locked_ov: float,
        n_expect: int,
    ) -> bool:
        """仅当候选显著更好并连续确认时，才替换环组结构。"""
        if not best_group:
            return False
        if not locked_rs:
            self.slots = [RingSlot(r_ema=float(r), last_r=float(r)) for r in best_group]
            self.locked_n = len(self.slots)
            self._promote_candidate = None
            self._promote_hits = 0
            return True

        # 同构：环数接近且半径可一一匹配 → 不换结构
        if abs(len(best_group) - len(locked_rs)) <= 1:
            matched = 0
            for r in best_group:
                if any(abs(r - lr) <= self.cfg.radius_match_px * 1.5 for lr in locked_rs):
                    matched += 1
            if matched >= max(2, len(locked_rs) - 1):
                self._promote_candidate = None
                self._promote_hits = 0
                return False

        better = best_ov >= locked_ov + self.cfg.promote_overlap_margin
        if not better:
            self._promote_candidate = None
            self._promote_hits = 0
            return False

        key = [round(r, 1) for r in best_group]
        if self._promote_candidate == key:
            self._promote_hits += 1
        else:
            self._promote_candidate = key
            self._promote_hits = 1

        if self._promote_hits < self.cfg.promote_hold_frames:
            return False

        # 晋升：尽量继承可匹配旧槽的 EMA/覆盖
        new_slots: list[RingSlot] = []
        for r in best_group:
            inherited = None
            for s in self.slots:
                if abs(s.r_ema - r) <= self.cfg.radius_match_px * 1.5:
                    inherited = s
                    break
            if inherited is not None:
                inherited.r_ema = 0.7 * inherited.r_ema + 0.3 * float(r)
                inherited.last_r = float(r)
                new_slots.append(inherited)
            else:
                new_slots.append(RingSlot(r_ema=float(r), last_r=float(r)))
        self.slots = new_slots
        self.locked_n = len(self.slots)
        self._promote_candidate = None
        self._promote_hits = 0
        return True

    def _accumulate_peaks(
        self,
        ray_peaks: list[list[float]],
        good_idx: list[int],
        h: int,
        w: int,
    ) -> None:
        if not self.slots:
            return
        cfg = self.cfg
        a = float(np.clip(cfg.radius_ema_alpha, 0.01, 1.0))
        n_ang = cfg.n_angles
        for i in good_idx:
            ang_bin = int(round(360.0 * i / float(n_ang))) % 360
            for rpk in ray_peaks[i]:
                # 匹配最近槽
                dists = [abs(rpk - s.r_ema) for s in self.slots]
                j = int(np.argmin(dists))
                if dists[j] > cfg.radius_match_px:
                    # 未匹配：仅当锁定尚少时轻量开新槽候选（立刻开槽需多次命中，这里记 pending 简化为跳过）
                    continue
                s = self.slots[j]
                s.r_ema = (1.0 - a) * s.r_ema + a * float(rpk)
                s.last_r = float(rpk)
                s.hit_count += 1
                # 角度覆盖：衰减 + 本帧命中
                s.angle_hits *= 0.997
                s.angle_hits[ang_bin] = min(1.0, s.angle_hits[ang_bin] + 0.35)

    def _prune_weak_slots(self) -> None:
        if len(self.slots) <= 2:
            return
        # 前几帧先攒样本，不急着剪
        if self.frame_i < 8:
            return
        kept: list[RingSlot] = []
        for s in self.slots:
            if s.hit_count >= 3 or s.coverage_frac() >= 0.08:
                kept.append(s)
            elif self.locked_n is not None and len(self.slots) <= int(self.locked_n):
                kept.append(s)
        if len(kept) >= 2:
            self.slots = kept

    def draw_debug_bgr(
        self,
        base_bgr: np.ndarray | None,
        prob_hw: np.ndarray | None,
        *,
        show_rays: bool = True,
    ) -> np.ndarray:
        """可视化：概率底 + 锁定环 + 可选稀疏射线。"""
        if prob_hw is not None:
            p = np.clip(prob_hw, 0.0, 1.0)
            u8 = (p * 255.0).astype(np.uint8)
            img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        elif base_bgr is not None:
            img = base_bgr.copy()
        else:
            img = np.zeros((512, 512, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        res = self.result_dict()
        if res is None:
            return img
        cx, cy = int(round(res["x0"])), int(round(res["y0"]))
        cv2.drawMarker(img, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)

        if show_rays and self._last_debug.get("n_good_rays"):
            # 稀疏画 24 条示意
            for k in range(24):
                ang = 2.0 * np.pi * k / 24.0
                x2 = int(round(cx + 0.45 * min(h, w) * np.cos(ang)))
                y2 = int(round(cy + 0.45 * min(h, w) * np.sin(ang)))
                cv2.line(img, (cx, cy), (x2, y2), (40, 40, 40), 1, cv2.LINE_AA)

        for r in res["rs"]:
            cv2.circle(img, (cx, cy), int(round(r)), (80, 220, 80), 1, cv2.LINE_AA)
        return img
