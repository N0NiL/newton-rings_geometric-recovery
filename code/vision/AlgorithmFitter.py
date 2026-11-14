from __future__ import annotations

from typing import Any

import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import siegelslopes

from RadialPeakSampler import RadialPeakSampler
from geom_head_coarse import GeomHeadCoarseHint

# 物面尺度：米/像素（1920×1080 下标定：103 px = 0.5 mm → 0.5e-3/103）
DEFAULT_PIXEL_SIZE_M_PER_PX = 0.5e-3 / 103.0


class NewtonFitter:
    def __init__(
        self,
        lambda_val=589.3e-9,
        pixel_size=DEFAULT_PIXEL_SIZE_M_PER_PX,
        prob_weight_power: float = 2.0,
        prob_grad_beta: float = 0.75,
        gray_grad_gamma: float = 0.5,
        k_match_tol_px: float = 20.0,
        robust_ls_f_scale: float = 2.5,
        center_warm_alpha: float = 0.0,
        radial_min_contour: int = 24,
        radial_peak_sampler: RadialPeakSampler | None = None,
        geom_head_ckpt: str | Path | None = None,
        geom_head_blend: float = 0.2,
        # 50distill / NEWMODEL 物理头与经典 Siegel R、圆心初值融合权重（0=纯经典，1=纯物理头）
        physics_head_blend: float = 0.5,
        radius_merge_close_px: float = 5.0,
        skeleton_premerge_radius_px: float = 0.0,
        # 定 R 时丢掉两端环的比例（每侧）；内外环易漏检/眩光
        r_fit_edge_trim_frac: float = 0.2,
        # Δ(r²) 缺口：判为 Δk≥2 的倍数阈值（相对 λR）
        gap_dk_round_tol: float = 0.35,
        # 匈牙利匹配成功率低于此则视为不可信（不做裸 1..n 重标）
        k_match_min_frac: float = 0.45,
        # 环数相对跳变超过此比例 → k 不可信
        ring_count_jump_max: float = 0.35,
    ):
        self.lambda_val = lambda_val
        self.pixel_size = pixel_size
        self.prob_weight_power = float(max(prob_weight_power, 0.0))
        self.prob_grad_beta = float(max(prob_grad_beta, 0.0))
        self.gray_grad_gamma = float(max(gray_grad_gamma, 0.0))
        self.k_match_tol_px = float(max(k_match_tol_px, 1e-3))
        self.robust_ls_f_scale = float(max(robust_ls_f_scale, 1e-6))
        self.center_warm_alpha = float(np.clip(center_warm_alpha, 0.0, 1.0))
        self.radial_min_contour = int(max(8, radial_min_contour))
        self._radial_sampler = radial_peak_sampler or RadialPeakSampler()
        self.geom_head_blend = float(np.clip(geom_head_blend, 0.0, 1.0))
        self.physics_head_blend = float(np.clip(physics_head_blend, 0.0, 1.0))
        self.radius_merge_close_px = float(max(0.0, radius_merge_close_px))
        self.skeleton_premerge_radius_px = float(max(0.0, skeleton_premerge_radius_px))
        self.r_fit_edge_trim_frac = float(np.clip(r_fit_edge_trim_frac, 0.0, 0.45))
        self.gap_dk_round_tol = float(max(0.05, gap_dk_round_tol))
        self.k_match_min_frac = float(np.clip(k_match_min_frac, 0.0, 1.0))
        self.ring_count_jump_max = float(max(0.05, ring_count_jump_max))

        self._prev_merged_rs: np.ndarray | None = None
        self._prev_k_orders: np.ndarray | None = None
        self._prev_center: tuple[float, float] | None = None
        self._anchor_k: int | None = None
        self._anchor_r_ema: float | None = None
        self._k_hit_counts: dict[int, int] = {}
        # NEWMODEL physics_refined / physics_initial（模型边长坐标系）
        self._physics_hint: dict[str, Any] | None = None

        self._geom_hint: GeomHeadCoarseHint | None = None
        if geom_head_ckpt is not None:
            gp = Path(str(geom_head_ckpt)).expanduser().resolve()
            if gp.is_file():
                try:
                    self._geom_hint = GeomHeadCoarseHint(gp)
                except Exception as e:
                    print(f"幾何頭粗輔助加載失敗（將跳過）: {e}")
                    self._geom_hint = None
            else:
                print(f"幾何頭權重文件不存在，跳過粗輔助: {gp}")

    def set_physics_hint(self, physics: dict[str, Any] | None) -> None:
        """
        注入 NEWMODEL 物理头输出（通常为 inference 的 physics_refined）。
        期望含 center_xy（模型像素）与 q_r2（模型像素²，对应 Δk=1 的 r² 周期）。
        """
        self._physics_hint = physics if isinstance(physics, dict) else None

    def clear_physics_hint(self) -> None:
        self._physics_hint = None

    @staticmethod
    def _as_xy(raw: Any) -> tuple[float, float] | None:
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return None
        cx, cy = float(arr[0]), float(arr[1])
        if not (np.isfinite(cx) and np.isfinite(cy)):
            return None
        return cx, cy

    def _physics_center_xy(self) -> tuple[float, float] | None:
        ph = self._physics_hint
        if not ph or self.physics_head_blend <= 1e-9:
            return None
        return self._as_xy(ph.get("center_xy"))

    def _physics_R_m(self) -> float | None:
        """
        由物理头 q_r2（模型像素²）换算曲率半径 R（米）：
        Δ(r_px²)=q_r2 ↔ Δ(r_m²)=λR ⇒ R = q_r2 · pixel_size² / λ
        """
        ph = self._physics_hint
        if not ph or self.physics_head_blend <= 1e-9:
            return None
        raw = ph.get("q_r2")
        if raw is None:
            return None
        q = float(np.asarray(raw, dtype=np.float64).reshape(-1)[0])
        if not np.isfinite(q) or q <= 0:
            return None
        R = q * (self.pixel_size ** 2) / float(self.lambda_val)
        if not np.isfinite(R) or R <= 0:
            return None
        return float(R)

    def _blend_R_with_physics(self, R_classical: float) -> tuple[float, float | None]:
        R_p = self._physics_R_m()
        if R_p is None:
            return float(R_classical), None
        b = self.physics_head_blend
        return float((1.0 - b) * float(R_classical) + b * R_p), R_p

    @staticmethod
    def _align_hw(src: np.ndarray, ref_hw: tuple[int, int]) -> np.ndarray:
        h, w = ref_hw
        if src.shape[:2] == (h, w):
            return src.astype(np.float32)
        return cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    @staticmethod
    def _grad_mag_norm(img_f: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        m = float(g.max()) + 1e-9
        return np.clip(g / m, 0.0, 1.0)

    @staticmethod
    def _bilinear_sample(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x = np.clip(x, 0.0, w - 1 - 1e-6)
        y = np.clip(y, 0.0, h - 1 - 1e-6)
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        xa = (x - x0).astype(np.float64)
        ya = (y - y0).astype(np.float64)
        Ia = img[y0, x0]
        Ib = img[y0, x1]
        Ic = img[y1, x0]
        Id = img[y1, x1]
        return (1.0 - xa) * (1.0 - ya) * Ia + xa * (1.0 - ya) * Ib + (
            1.0 - xa
        ) * ya * Ic + xa * ya * Id

    def _sqrt_weights_at_points(
        self,
        all_pts: np.ndarray,
        prob_hw: np.ndarray | None,
        gray_hw: np.ndarray | None,
    ) -> np.ndarray:
        n = len(all_pts)
        x = all_pts[:, 0].astype(np.float64)
        y = all_pts[:, 1].astype(np.float64)
        if prob_hw is None:
            return np.ones(n, dtype=np.float64)

        h, w = prob_hw.shape[:2]
        prob_f = np.clip(prob_hw.astype(np.float32), 0.0, 1.0)
        p = self._bilinear_sample(prob_f, x, y)
        p = np.clip(p, 1e-6, 1.0)
        if self.prob_weight_power <= 0.0:
            wgt = np.ones(n, dtype=np.float64)
        else:
            wgt = p**self.prob_weight_power

        if self.prob_grad_beta > 0.0:
            gm_p = self._grad_mag_norm(prob_f)
            gp = self._bilinear_sample(gm_p.astype(np.float64), x, y)
            wgt *= 1.0 + self.prob_grad_beta * gp

        if gray_hw is not None and self.gray_grad_gamma > 0.0:
            g = self._align_hw(gray_hw, (h, w))
            gm_g = self._grad_mag_norm(g)
            gg = self._bilinear_sample(gm_g.astype(np.float64), x, y)
            wgt *= 1.0 + self.gray_grad_gamma * gg

        return np.sqrt(np.clip(wgt, 1e-12, None))

    def _interp_k_from_prev_skeleton(self, r_px: float) -> int:
        """匹配差时：用上一帧 (r→k) 骨架插值/外推级次，禁止裸 1..n 重标。"""
        assert self._prev_merged_rs is not None and self._prev_k_orders is not None
        prev_r = self._prev_merged_rs.astype(np.float64)
        prev_k = self._prev_k_orders.astype(np.float64)
        order = np.argsort(prev_r)
        prev_r = prev_r[order]
        prev_k = prev_k[order]
        if len(prev_r) == 1:
            return max(1, int(round(float(prev_k[0]))))
        if r_px <= float(prev_r[0]):
            # 外推内侧
            dr = float(prev_r[1] - prev_r[0]) + 1e-9
            dk = float(prev_k[1] - prev_k[0])
            k = float(prev_k[0]) + dk * (r_px - float(prev_r[0])) / dr
            return max(1, int(round(k)))
        if r_px >= float(prev_r[-1]):
            dr = float(prev_r[-1] - prev_r[-2]) + 1e-9
            dk = float(prev_k[-1] - prev_k[-2])
            k = float(prev_k[-1]) + dk * (r_px - float(prev_r[-1])) / dr
            return max(1, int(round(k)))
        i = int(np.searchsorted(prev_r, r_px, side="right") - 1)
        i = int(np.clip(i, 0, len(prev_r) - 2))
        dr = float(prev_r[i + 1] - prev_r[i]) + 1e-9
        t = (r_px - float(prev_r[i])) / dr
        k = (1.0 - t) * float(prev_k[i]) + t * float(prev_k[i + 1])
        return max(1, int(round(k)))

    def _align_k_orders(
        self, rs_sorted: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        半径 → 环序。返回 (rs, ks, meta)。
        meta: match_frac, mean_match_cost_px, ring_count_jump, used_skeleton_fallback
        """
        n = len(rs_sorted)
        rs = rs_sorted.astype(np.float64)
        meta: dict[str, Any] = {
            "match_frac": 0.0,
            "mean_match_cost_px": float("inf"),
            "ring_count_jump": 1.0,
            "used_skeleton_fallback": False,
            "n_matched": 0,
        }

        # 首帧：暂标 1..n，后续靠 Δ(r²) 修间距；不强绑死物理第一环
        if (
            self._prev_merged_rs is None
            or self._prev_k_orders is None
            or n < 2
            or len(self._prev_merged_rs) < 1
        ):
            meta["match_frac"] = 1.0
            meta["mean_match_cost_px"] = 0.0
            meta["ring_count_jump"] = 0.0
            return rs, np.arange(1, n + 1, dtype=np.float64), meta

        prev_r = self._prev_merged_rs.astype(np.float64)
        prev_k = self._prev_k_orders.astype(np.float64)
        m = len(prev_r)
        meta["ring_count_jump"] = abs(n - m) / float(max(m, 1))

        cost = np.abs(rs.reshape(-1, 1) - prev_r.reshape(1, -1))
        row_ind, col_ind = linear_sum_assignment(cost)
        pair_cost = cost[row_ind, col_ind]
        good = pair_cost <= self.k_match_tol_px
        n_good = int(np.sum(good))
        meta["n_matched"] = n_good
        meta["match_frac"] = float(n_good) / float(max(n, 1))
        meta["mean_match_cost_px"] = (
            float(np.mean(pair_cost[good])) if n_good > 0 else float(np.mean(pair_cost))
        )

        k_out = np.zeros(n, dtype=np.float64)
        matched_rows: set[int] = set()

        if n_good > 0 and meta["match_frac"] >= self.k_match_min_frac * 0.5:
            for i, j, ok, c in zip(row_ind, col_ind, good, pair_cost):
                if not ok:
                    continue
                k_out[int(i)] = float(prev_k[int(j)])
                matched_rows.add(int(i))
            # 未匹配：用上一帧骨架插值级次（插空/外推），不整表重标
            for i in range(n):
                if i not in matched_rows:
                    k_out[i] = float(self._interp_k_from_prev_skeleton(float(rs[i])))
                    meta["used_skeleton_fallback"] = True
        else:
            # 匹配整体很差：全部走骨架插值，禁止 1..n
            meta["used_skeleton_fallback"] = True
            for i in range(n):
                k_out[i] = float(self._interp_k_from_prev_skeleton(float(rs[i])))

        # 同 k 冲突时：保留更接近上一帧同 k 半径的那个，另一个重新插值避开
        for _ in range(3):
            uniq, counts = np.unique(k_out.astype(int), return_counts=True)
            dups = uniq[counts > 1]
            if len(dups) == 0:
                break
            for dk in dups:
                idxs = np.where(k_out.astype(int) == int(dk))[0]
                # 相对上一帧该 k 的半径
                prev_same = np.where(prev_k.astype(int) == int(dk))[0]
                if len(prev_same):
                    target_r = float(prev_r[prev_same[0]])
                    best = idxs[int(np.argmin(np.abs(rs[idxs] - target_r)))]
                else:
                    best = idxs[0]
                for i in idxs:
                    if i == best:
                        continue
                    k_out[i] = float(self._interp_k_from_prev_skeleton(float(rs[i])))
                    if int(k_out[i]) == int(dk):
                        k_out[i] = float(int(dk) + (1 if rs[i] > rs[best] else -1))
                        k_out[i] = max(1.0, k_out[i])

        order = np.argsort(rs)
        return rs[order], k_out[order], meta

    def _correct_k_gaps_by_dr2(
        self,
        rs: np.ndarray,
        ks: np.ndarray,
        R_hint: float,
    ) -> tuple[np.ndarray, bool, int]:
        """
        按 Δ(r²)≈Δk·λ·R 修正级次间距：漏中间环时扩 Δk，而不是假装 Δk=1。
        返回 (ks_fixed, has_gap, n_gap_units)。
        """
        n = len(rs)
        if n < 2 or not np.isfinite(R_hint) or R_hint <= 1e-6:
            return ks.astype(np.float64), False, 0

        order = np.argsort(rs)
        rs_o = rs[order].astype(np.float64)
        ks_o = ks[order].astype(np.float64)
        r2 = (rs_o * self.pixel_size) ** 2
        step = float(self.lambda_val * R_hint)
        if step <= 1e-18:
            return ks.astype(np.float64), False, 0

        # 以匹配链的「锚点」为起点：优先用历史命中最多的中间 k
        anchor_idx = 0
        if self._anchor_k is not None:
            d = np.abs(ks_o - float(self._anchor_k))
            anchor_idx = int(np.argmin(d))
        elif self._anchor_r_ema is not None:
            d = np.abs(rs_o - float(self._anchor_r_ema))
            anchor_idx = int(np.argmin(d))
        else:
            anchor_idx = n // 2

        ks_new = ks_o.copy()
        ks_new[anchor_idx] = float(max(1, int(round(float(ks_o[anchor_idx])))))

        has_gap = False
        n_gap_units = 0

        # 向外侧（半径增大）
        for i in range(anchor_idx + 1, n):
            dr2 = float(r2[i] - r2[i - 1])
            dk_f = dr2 / step
            dk = int(max(1, int(round(dk_f))))
            if abs(dk_f - dk) > self.gap_dk_round_tol and dk_f > 1.2:
                dk = int(max(1, int(round(dk_f))))
            if dk >= 2:
                has_gap = True
                n_gap_units += dk - 1
            ks_new[i] = ks_new[i - 1] + float(dk)

        # 向内侧（半径减小）
        for i in range(anchor_idx - 1, -1, -1):
            dr2 = float(r2[i + 1] - r2[i])
            dk_f = dr2 / step
            dk = int(max(1, int(round(dk_f))))
            if dk >= 2:
                has_gap = True
                n_gap_units += dk - 1
            ks_new[i] = ks_new[i + 1] - float(dk)

        # 平移使最小 k≥1，并尽量贴近锚点历史
        min_k = float(np.min(ks_new))
        if min_k < 1.0:
            ks_new = ks_new + (1.0 - min_k)

        # 还原到原顺序
        inv = np.empty(n, dtype=np.int32)
        inv[order] = np.arange(n)
        return ks_new[inv], has_gap, int(n_gap_units)

    def _middle_band_mask(self, n: int) -> np.ndarray:
        """定 R 用中段环；内外环仍保留在 rs/k 供显示。"""
        if n <= 0:
            return np.zeros(0, dtype=bool)
        if n < 4:
            return np.ones(n, dtype=bool)
        frac = self.r_fit_edge_trim_frac
        trim = int(np.floor(n * frac))
        trim = int(np.clip(trim, 1 if n >= 5 else 0, max(0, (n - 2) // 2)))
        mask = np.ones(n, dtype=bool)
        if trim > 0:
            mask[:trim] = False
            mask[-trim:] = False
        if int(np.sum(mask)) < 2:
            return np.ones(n, dtype=bool)
        return mask

    def _update_anchor(self, rs: np.ndarray, ks: np.ndarray) -> None:
        """用出现率最高的中段 k 做绝对级次锚点。"""
        if len(ks) < 2:
            return
        for k in ks.astype(int):
            self._k_hit_counts[int(k)] = self._k_hit_counts.get(int(k), 0) + 1
        # 中段候选
        order = np.argsort(rs)
        mid = order[len(order) // 2]
        mid_k = int(ks[mid])
        # 全局命中最多且半径靠近中位数者优先
        if self._k_hit_counts:
            ranked = sorted(
                self._k_hit_counts.items(), key=lambda t: (-t[1], abs(t[0] - mid_k))
            )
            self._anchor_k = int(ranked[0][0])
        else:
            self._anchor_k = mid_k
        if self._anchor_k in set(ks.astype(int)):
            idx = int(np.where(ks.astype(int) == self._anchor_k)[0][0])
            r = float(rs[idx])
            if self._anchor_r_ema is None:
                self._anchor_r_ema = r
            else:
                self._anchor_r_ema = 0.8 * float(self._anchor_r_ema) + 0.2 * r

    def _siegel_R_middle(
        self, rs: np.ndarray, ks: np.ndarray
    ) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """
        中段环 Siegel 得 R。返回 R, R2, stderr, rms, resids_all, used_mask
        """
        n = len(rs)
        mask = self._middle_band_mask(n)
        r2_all = (rs * self.pixel_size) ** 2
        if int(np.sum(mask)) < 2:
            mask = np.ones(n, dtype=bool)
        ss = siegelslopes(r2_all[mask], x=ks[mask])
        slope = float(ss[0])
        intercept = float(ss[1])
        R_val = slope / self.lambda_val
        pred = slope * ks + intercept
        resids = r2_all - pred
        ss_res = float(np.sum(resids[mask] ** 2))
        ss_tot = float(np.sum((r2_all[mask] - np.mean(r2_all[mask])) ** 2))
        r_sq = 1.0 - ss_res / (ss_tot + 1e-18)
        mad = float(np.median(np.abs(resids[mask] - np.median(resids[mask]))) + 1e-18)
        R_stderr = float(1.4826 * mad / max(np.sqrt(float(np.sum(mask))), 1.0))
        residual_rms = float(np.sqrt(np.mean(resids[mask] ** 2)))
        return R_val, r_sq, R_stderr, residual_rms, resids, mask

    @staticmethod
    def _merge_radii_sorted(sorted_rs: np.ndarray, dist_threshold: float) -> np.ndarray:
        if len(sorted_rs) == 0:
            return sorted_rs
        merged: list[float] = []
        grp = [float(sorted_rs[0])]
        for i in range(1, len(sorted_rs)):
            if sorted_rs[i] - sorted_rs[i - 1] < dist_threshold:
                grp.append(float(sorted_rs[i]))
            else:
                merged.append(float(np.mean(grp)))
                grp = [float(sorted_rs[i])]
        merged.append(float(np.mean(grp)))
        return np.array(merged, dtype=np.float64)

    @staticmethod
    def _concat_contours_to_one(cnt_list: list) -> np.ndarray:
        parts = [c.reshape(-1, 2).astype(np.float32) for c in cnt_list]
        arr = np.vstack(parts)
        return arr.reshape(-1, 1, 2)

    def _merge_skeleton_contours_by_median_radius(
        self,
        valid_cnts: list,
        cx0: float,
        cy0: float,
        bin_width_px: float,
    ) -> list:
        if bin_width_px <= 1e-6 or len(valid_cnts) <= 1:
            return valid_cnts
        scored: list[tuple[float, np.ndarray]] = []
        for cnt in valid_cnts:
            pts = cnt.reshape(-1, 2).astype(np.float64)
            r_med = float(np.median(np.hypot(pts[:, 0] - cx0, pts[:, 1] - cy0)))
            scored.append((r_med, cnt))
        scored.sort(key=lambda t: t[0])
        out: list[np.ndarray] = []
        cur_group: list[np.ndarray] = [scored[0][1]]
        r_max_in_group = scored[0][0]
        for r_med, cnt in scored[1:]:
            if r_med - r_max_in_group <= bin_width_px:
                cur_group.append(cnt)
                r_max_in_group = max(r_max_in_group, r_med)
            else:
                out.append(self._concat_contours_to_one(cur_group))
                cur_group = [cnt]
                r_max_in_group = r_med
        out.append(self._concat_contours_to_one(cur_group))
        return out

    def _fit_multi_arc_contours(
        self,
        valid_cnts: list,
        prob_u: np.ndarray | None,
        gray_u: np.ndarray | None,
        fit_mode: str,
        geom_center_hint: tuple[float, float] | None = None,
    ) -> dict | None:
        """多轮廓点 → 加权 soft_l1 圆拟合 → Siegel r²–k。"""
        points = []
        indices = []
        for i, cnt in enumerate(valid_cnts):
            pts = cnt.reshape(-1, 2).astype(np.float64)
            points.append(pts)
            indices.append(np.full(len(pts), i))

        all_pts = np.vstack(points)
        all_idx = np.hstack(indices)

        cx_mean = float(all_pts[:, 0].mean())
        cy_mean = float(all_pts[:, 1].mean())
        if self._prev_center is not None:
            a = self.center_warm_alpha
            cx_guess = (1.0 - a) * cx_mean + a * self._prev_center[0]
            cy_guess = (1.0 - a) * cy_mean + a * self._prev_center[1]
        else:
            cx_guess, cy_guess = cx_mean, cy_mean

        if geom_center_hint is not None and self.geom_head_blend > 1e-9:
            gcx, gcy = float(geom_center_hint[0]), float(geom_center_hint[1])
            b = self.geom_head_blend
            cx_guess = (1.0 - b) * cx_guess + b * gcx
            cy_guess = (1.0 - b) * cy_guess + b * gcy

        # 物理头圆心：与经典初值五五开（默认 physics_head_blend=0.5）
        phys_c = self._physics_center_xy()
        if phys_c is not None:
            b = self.physics_head_blend
            cx_guess = (1.0 - b) * cx_guess + b * float(phys_c[0])
            cy_guess = (1.0 - b) * cy_guess + b * float(phys_c[1])

        r_guesses: list[float] = []
        for cnt in valid_cnts:
            pts = cnt.reshape(-1, 2).astype(np.float64)
            r_guesses.append(
                float(np.mean(np.hypot(pts[:, 0] - cx_guess, pts[:, 1] - cy_guess)))
            )

        x_init = [cx_guess, cy_guess] + r_guesses
        sqrt_w = self._sqrt_weights_at_points(all_pts, prob_u, gray_u)

        def residuals(params, x, y, idx, sw):
            x0, y0 = params[0], params[1]
            rs = params[2:]
            raw = np.hypot(x - x0, y - y0) - rs[idx]
            return sw * raw

        n_r = len(valid_cnts)
        lb = [-np.inf, -np.inf] + [0.0] * n_r
        ub = [np.inf, np.inf] + [np.inf] * n_r

        try:
            res = least_squares(
                residuals,
                x_init,
                args=(all_pts[:, 0], all_pts[:, 1], all_idx, sqrt_w),
                bounds=(lb, ub),
                method="trf",
                loss="soft_l1",
                f_scale=self.robust_ls_f_scale,
            )

            x0_final = res.x[0]
            y0_final = res.x[1]
            rs_raw = res.x[2:]

            sorted_rs = np.sort(rs_raw)
            if self.radius_merge_close_px > 1e-6:
                merged_rs = self._merge_radii_sorted(
                    sorted_rs, self.radius_merge_close_px
                )
            else:
                merged_rs = sorted_rs.copy()

            if len(merged_rs) < 2:
                return None

            rs_sorted = np.sort(merged_rs)
            rs_for_line, k_orders, k_meta = self._align_k_orders(rs_sorted)

            # 粗 R（全环）仅作缺口步长 hint
            r2_tmp = (rs_for_line * self.pixel_size) ** 2
            try:
                R_hint = float(siegelslopes(r2_tmp, x=k_orders)[0]) / self.lambda_val
                if not np.isfinite(R_hint) or R_hint <= 0:
                    R_hint = 1.9
            except Exception:
                R_hint = 1.9

            k_orders, has_gap, n_gap_units = self._correct_k_gaps_by_dr2(
                rs_for_line, k_orders, R_hint
            )

            R_val, r_sq, R_stderr, residual_rms_m2, resids, mid_mask = (
                self._siegel_R_middle(rs_for_line, k_orders)
            )
            R_classical = float(R_val)
            R_val, R_physics = self._blend_R_with_physics(R_classical)
            # 全环残差供显示；定 R 已用中段
            mad = float(np.median(np.abs(resids - np.median(resids))) + 1e-18)
            inlier = np.abs(resids) < 2.5 * mad
            line_inlier_frac = float(np.mean(inlier))

            match_ok = (
                float(k_meta["match_frac"]) >= self.k_match_min_frac
                and float(k_meta["ring_count_jump"]) <= self.ring_count_jump_max
            )
            # 首帧无历史时允许信任；有缺口仍可定 R，但池化需更谨慎
            first_frame = self._prev_merged_rs is None
            k_trustworthy = bool(first_frame or match_ok) and (
                int(np.sum(mid_mask)) >= 2
            )

            self._update_anchor(rs_for_line, k_orders)
            self._prev_merged_rs = rs_for_line.copy()
            self._prev_k_orders = k_orders.copy()
            self._prev_center = (float(x0_final), float(y0_final))

            out = {
                "x0": x0_final,
                "y0": y0_final,
                "rs": [float(r) for r in rs_for_line],
                "rs_k": [float(r) for r in rs_for_line],
                "k_orders": [int(k) for k in np.round(k_orders)],
                "ring_resid_m2": [float(x) for x in resids],
                "R": R_val,
                "R_frame": R_val,
                "R_classical": R_classical,
                "R2": r_sq,
                "R_stderr": R_stderr,
                "residual_rms_m2": residual_rms_m2,
                "line_inlier_frac": line_inlier_frac,
                "fit_mode": fit_mode,
                "has_gap": bool(has_gap),
                "n_gap_units": int(n_gap_units),
                "k_match_frac": float(k_meta["match_frac"]),
                "k_mean_match_cost_px": float(k_meta["mean_match_cost_px"]),
                "ring_count_jump": float(k_meta["ring_count_jump"]),
                "k_used_skeleton_fallback": bool(k_meta["used_skeleton_fallback"]),
                "k_match_ok": bool(match_ok or first_frame),
                "k_trustworthy": bool(k_trustworthy),
                "n_rings_for_R": int(np.sum(mid_mask)),
                "anchor_k": self._anchor_k,
                "physics_head_blend": float(self.physics_head_blend),
            }
            if R_physics is not None:
                out["R_physics"] = float(R_physics)
            return out
        except Exception as e:
            print(f"拟合计算失败: {e}")
            return None

    def fit_circles_with_topology(self, skeleton_img, prob_hw=None, gray_hw=None):
        """
        经典路径：骨架二值图 findContours → 多圆弧加权最小二乘 → Siegel r²–k。
        """
        h, w = skeleton_img.shape[:2]
        prob_u: np.ndarray | None = None
        if prob_hw is not None:
            prob_u = np.clip(self._align_hw(prob_hw, (h, w)), 0.0, 1.0)
        gray_u = None
        if gray_hw is not None:
            gray_u = self._align_hw(gray_hw, (h, w))

        geom_center: tuple[float, float] | None = None
        if self._geom_hint is not None:
            geom_center = self._geom_hint.predict_center_hw(skeleton_img)

        contours, _ = cv2.findContours(
            skeleton_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        valid_cnts = [c for c in contours if len(c) > 80]

        if len(valid_cnts) < 2:
            return None

        if self.skeleton_premerge_radius_px > 1e-6:
            allp = np.vstack([c.reshape(-1, 2) for c in valid_cnts]).astype(np.float64)
            cx_t = float(allp[:, 0].mean())
            cy_t = float(allp[:, 1].mean())
            valid_cnts = self._merge_skeleton_contours_by_median_radius(
                valid_cnts, cx_t, cy_t, self.skeleton_premerge_radius_px
            )
            if len(valid_cnts) < 2:
                return None

        return self._fit_multi_arc_contours(
            valid_cnts,
            prob_u,
            gray_u,
            fit_mode="skeleton",
            geom_center_hint=geom_center,
        )

    def augment_with_radial_peak_auxiliary(
        self,
        res: dict | None,
        prob_hw: np.ndarray | None,
        gray_hw: np.ndarray | None,
    ) -> dict | None:
        """
        每帧必跑的径向峰辅助统计（不参与圆心/半径/R 的数值解）。
        失败时仍写入 radial_peak_aux（ok=False），避免状态栏时隐时现。
        """
        if res is None:
            return None
        out = dict(res)
        aux_empty = {
            "ok": False,
            "n_rings": 0,
            "ring_mean_peak": 0.0,
            "best_ring_idx": -1,
            "best_ring_mean_peak": 0.0,
            "intra_max_cv": 0.0,
            "reason": "no_input",
        }
        if prob_hw is None or gray_hw is None:
            out["radial_peak_aux"] = aux_empty
            return out

        h, w = prob_hw.shape[:2]
        prob_u = np.clip(self._align_hw(prob_hw, (h, w)), 0.0, 1.0)
        gray_u = self._align_hw(gray_hw, (h, w))

        cx, cy = float(out["x0"]), float(out["y0"])
        raw_cnts = self._radial_sampler.build_ring_point_contours(prob_u, gray_u, cx, cy)
        if not raw_cnts:
            aux_empty["reason"] = "no_peaks"
            out["radial_peak_aux"] = aux_empty
            return out

        ring_means: list[float] = []
        ring_cvs: list[float] = []
        for c in raw_cnts:
            pts = c.reshape(-1, 2).astype(np.float64)
            xv = np.clip(pts[:, 0], 0.0, float(w - 1) - 1e-6)
            yv = np.clip(pts[:, 1], 0.0, float(h - 1) - 1e-6)
            vals = self._bilinear_sample(prob_u.astype(np.float64), xv, yv)
            ring_means.append(float(np.mean(vals)))
            m = float(np.mean(vals)) + 1e-9
            s = float(np.std(vals))
            ring_cvs.append(s / m)

        best_idx = int(np.argmax(ring_means))
        best_mean = float(ring_means[best_idx])
        cvmax = float(np.max(ring_cvs)) if ring_cvs else 0.0

        out["radial_peak_aux"] = {
            "ok": True,
            "n_rings": int(len(raw_cnts)),
            "ring_mean_peak": best_mean,
            "best_ring_idx": best_idx,
            "best_ring_mean_peak": best_mean,
            "intra_max_cv": cvmax,
            "reason": "ok",
        }
        return out
