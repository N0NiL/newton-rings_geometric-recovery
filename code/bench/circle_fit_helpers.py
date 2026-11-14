# -*- coding: utf-8 -*-
"""共享两轮装配：供平移实验等脚本在包内调用（原 run_fill_ablations._fit）。"""
from __future__ import annotations

import random

import numpy as np

import run_deployed_measure as base
from AfterProcess import NewtonSecondaryPipeline
from AlgorithmFitter import NewtonFitter
from RadialRingAccumulator import AccumulatorConfig, RadialRingAccumulator
from Round2PhysicsFit import second_round


class _NoR0(RadialRingAccumulator):
    """只抬掉射线峰数绝对上限。"""


def _fit(prob: np.ndarray, gray_u8: np.ndarray, *, nocap: bool, physics: dict | None, blend: float):
    skeleton = NewtonSecondaryPipeline().run(prob).skeleton
    fitter = NewtonFitter(physics_head_blend=blend, geom_head_blend=0.0, center_warm_alpha=0.0)
    if physics and blend > 0:
        fitter.set_physics_hint(physics)
    else:
        fitter.clear_physics_hint()
    try:
        fit = fitter.fit_circles_with_topology(skeleton, prob_hw=prob, gray_hw=gray_u8)
    except Exception:
        fit = None
    center = None
    model_n = None
    if isinstance(fit, dict) and np.isfinite(fit.get("x0", np.nan)):
        center = (float(fit["x0"]), float(fit["y0"]))
        rs_fit = fit.get("rs") or []
        if len(rs_fit) >= 2:
            model_n = int(len(rs_fit))
    random.seed(0)
    np.random.seed(0)
    if nocap:
        cfg = AccumulatorConfig(center_follow=1.0, ray_count_max_abs=10**6)
        radial = _NoR0(cfg).fit_static_once(prob, center_xy=center, model_n_rings=model_n)
    else:
        radial = RadialRingAccumulator(AccumulatorConfig(center_follow=1.0)).fit_static_once(
            prob, center_xy=center, model_n_rings=model_n
        )
    radii: list[float] = []
    used = center
    if isinstance(radial, dict) and radial.get("rs"):
        radii = [float(r) for r in radial["rs"]]
        used = (float(radial["x0"]), float(radial["y0"]))
    extra = second_round(prob, used, radii)
    return used, [float(r) for r in extra["rs"]]


def _to_canvas(center, radii, mapping, canvas_side, crop_side):
    out_c = None
    if center is not None:
        out_c = base._to_canvas(center[0], center[1], mapping, canvas_side, crop_side)
    out_r = sorted(base._length_to_canvas(radius, mapping, canvas_side, crop_side) for radius in radii)
    return out_c, out_r
