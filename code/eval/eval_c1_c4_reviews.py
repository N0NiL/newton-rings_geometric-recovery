# -*- coding: utf-8 -*-
"""
C1–C3 quantitative reviews (Monte Carlo pole uncertainty, session split,
McNemar/Wilcoxon, stationary clusters). No DOCX builders.

Outputs JSON under results/c1_c4/ (or results/smoke_c1_c4 with --limit).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parents[1]
BENCH = PKG_ROOT / "code" / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    FITS_ROOT,
    NAMES,
    RESULTS,
    setup_import_path,
)

setup_import_path()

import coherency  # noqa: E402
from band_score import load_band  # noqa: E402
from paper_annotate import fit_concentric  # noqa: E402
from score_s12 import judge_s12  # noqa: E402

SESSION = DATA_COMPARISON
STATIC_JSON = RESULTS / "static.json"
TIMING_JSON = RESULTS / "timing.json"


def _included(stem: str) -> bool:
    head = str(stem).split("_", 1)[0]
    return len(head) in (4, 6)


def _passes(row: dict, spacing: float, limit: float | None) -> bool:
    if not row["success"]:
        return False
    if limit is None:
        return True
    err = row.get("center_error_px")
    if err is None or spacing <= 1e-6:
        return False
    return float(err) <= limit * spacing + 1e-9


def _gather(
    *,
    limit_stems: int = 0,
) -> tuple[dict[str, list[dict]], dict[str, float], dict[str, int], dict[str, list[dict]]]:
    """Rescore frozen fits with current human/band labels (S1.2 + coherency)."""
    raw: dict[str, dict[str, dict]] = {}
    for name in NAMES:
        rows = json.loads((FITS_ROOT / name / "results.json").read_text(encoding="utf-8"))
        raw[name] = {row["stem"]: row for row in rows if _included(row["stem"])}
    stems = sorted(raw["classical"])
    if limit_stems and limit_stems > 0:
        # Prefer readable frames for smoke
        readable = [
            s for s in stems if raw["classical"][s].get("subset") in ("Easy", "Open-arc")
        ]
        stems = (readable or stems)[:limit_stems]
    stored = {name: [] for name in NAMES}
    coh = {name: [] for name in NAMES}
    spacing: dict[str, float] = {}
    human_n: dict[str, int] = {}
    for stem in stems:
        source = raw["classical"][stem]
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        visible = coherency._visible(stem)
        spacing[stem] = float(human["spacing"])
        human_n[stem] = int(human["n"])
        for name in NAMES:
            row = raw[name][stem]
            center = None if row.get("center_xy") is None else tuple(row["center_xy"])
            score = judge_s12(human, list(row.get("radii_px") or []), center)
            score["stem"] = stem
            score["subset"] = source["subset"]
            stored[name].append(score)
            frame = coherency._frame(visible, row)
            frame["stem"] = stem
            frame["subset"] = source["subset"]
            coh[name].append(frame)
    return stored, spacing, human_n, coh


def eval_c1_gt_uncertainty_and_geometry(*, limit_frames: int = 0):
    """
    C1-A: Monte Carlo perturbation on human points (sigma = 1.0, 1.5, 2.0 px).
    C1-B: Angular span / off-frame distance on Open-arc (and Easy).
    """
    stored, spacing, _human_n, _coh = _gather(limit_stems=0)
    rng = np.random.default_rng(42)

    mc_results = {sigma: {"Easy": [], "Open-arc": []} for sigma in (1.0, 1.5, 2.0)}
    geom_stats = {"Easy": [], "Open-arc": []}

    rows = [r for r in stored["classical"] if r["subset"] in ("Easy", "Open-arc")]
    if limit_frames and limit_frames > 0:
        rows = rows[:limit_frames]

    for row in rows:
        stem = row["stem"]
        subset = row["subset"]
        s = spacing[stem]
        h_path = SESSION / f"{stem}_human.json"
        human_raw = json.loads(h_path.read_text(encoding="utf-8"))

        groups = []
        for ring in human_raw.get("rings") or []:
            if str(ring.get("role") or "newton") != "newton":
                continue
            pts = np.asarray(ring.get("points_xy") or [], dtype=np.float64)
            if len(pts) >= 3:
                groups.append(pts)

        base_fit = fit_concentric(groups)
        if base_fit is None:
            continue
        bx, by = base_fit[0], base_fit[1]

        arc_spans_deg = []
        all_angles = []
        total_pts = sum(len(g) for g in groups)
        for g in groups:
            ang = np.arctan2(g[:, 1] - by, g[:, 0] - bx)
            ang_u = np.unwrap(np.sort(ang))
            span_deg = float(np.degrees(ang_u[-1] - ang_u[0])) if len(ang_u) >= 2 else 0.0
            arc_spans_deg.append(span_deg)
            all_angles.append(ang)
        all_ang_concat = np.concatenate(all_angles)
        all_ang_u = np.unwrap(np.sort(all_ang_concat))
        frame_span_deg = float(np.degrees(all_ang_u[-1] - all_ang_u[0]))

        dx_border = 0.0 if 0.0 <= bx <= 767.0 else (abs(bx) if bx < 0 else bx - 767.0)
        dy_border = 0.0 if 0.0 <= by <= 767.0 else (abs(by) if by < 0 else by - 767.0)
        off_border_px = math.hypot(dx_border, dy_border)
        off_center_px = math.hypot(bx - 383.5, by - 383.5)

        geom_stats[subset].append({
            "stem": stem,
            "spacing_px": s,
            "n_arcs": len(groups),
            "total_pts": total_pts,
            "median_arc_span_deg": float(np.median(arc_spans_deg)),
            "max_arc_span_deg": float(np.max(arc_spans_deg)),
            "frame_span_deg": frame_span_deg,
            "off_border_px": off_border_px,
            "off_border_Lambda": off_border_px / s if s > 0 else 0.0,
            "off_center_px": off_center_px,
            "off_center_Lambda": off_center_px / s if s > 0 else 0.0,
        })

        n_trials = 20 if limit_frames else 100
        for sigma in (1.0, 1.5, 2.0):
            errs_px = []
            for _ in range(n_trials):
                pert_groups = [g + rng.normal(0.0, sigma, size=g.shape) for g in groups]
                p_fit = fit_concentric(pert_groups)
                if p_fit is not None:
                    err = math.hypot(p_fit[0] - bx, p_fit[1] - by)
                    errs_px.append(err)
            if errs_px:
                rms_px = float(np.sqrt(np.mean(np.square(errs_px))))
                med_px = float(np.median(errs_px))
                p95_px = float(np.percentile(errs_px, 95))
                mc_results[sigma][subset].append({
                    "stem": stem,
                    "rms_px": rms_px,
                    "rms_Lambda": rms_px / s,
                    "med_px": med_px,
                    "med_Lambda": med_px / s,
                    "p95_Lambda": p95_px / s,
                })

    summary = {"mc_pole_uncertainty": {}, "geometry": {}}
    for sigma in (1.0, 1.5, 2.0):
        summary["mc_pole_uncertainty"][f"sigma_{sigma}px"] = {}
        for subset in ("Easy", "Open-arc"):
            rows_s = mc_results[sigma][subset]
            if not rows_s:
                summary["mc_pole_uncertainty"][f"sigma_{sigma}px"][subset] = {"n_frames": 0}
                continue
            rms_L = [r["rms_Lambda"] for r in rows_s]
            rms_px = [r["rms_px"] for r in rows_s]
            summary["mc_pole_uncertainty"][f"sigma_{sigma}px"][subset] = {
                "n_frames": len(rows_s),
                "median_pole_sd_px": round(float(np.median(rms_px)), 2),
                "iqr_pole_sd_px": round(float(np.percentile(rms_px, 75) - np.percentile(rms_px, 25)), 2),
                "median_pole_sd_in_Lambda": round(float(np.median(rms_L)), 4),
                "p90_pole_sd_in_Lambda": round(float(np.percentile(rms_L, 90)), 4),
                "max_pole_sd_in_Lambda": round(float(np.max(rms_L)), 4),
                "frames_under_0.15Lambda": sum(1 for v in rms_L if v < 0.15),
                "frames_under_0.25Lambda": sum(1 for v in rms_L if v < 0.25),
            }

    for subset in ("Easy", "Open-arc"):
        rows_g = geom_stats[subset]
        if not rows_g:
            summary["geometry"][subset] = {"n_frames": 0}
            continue
        summary["geometry"][subset] = {
            "n_frames": len(rows_g),
            "median_frame_span_deg": round(float(np.median([r["frame_span_deg"] for r in rows_g])), 1),
            "iqr_frame_span_deg": round(
                float(np.percentile([r["frame_span_deg"] for r in rows_g], 75)
                      - np.percentile([r["frame_span_deg"] for r in rows_g], 25)),
                1,
            ),
            "min_frame_span_deg": round(float(np.min([r["frame_span_deg"] for r in rows_g])), 1),
            "max_frame_span_deg": round(float(np.max([r["frame_span_deg"] for r in rows_g])), 1),
            "median_off_border_px": round(float(np.median([r["off_border_px"] for r in rows_g])), 1),
            "max_off_border_px": round(float(np.max([r["off_border_px"] for r in rows_g])), 1),
            "median_off_border_Lambda": round(float(np.median([r["off_border_Lambda"] for r in rows_g])), 2),
            "max_off_border_Lambda": round(float(np.max([r["off_border_Lambda"] for r in rows_g])), 2),
        }
    return summary


def eval_c2_session_split_and_distortion(*, limit_stems: int = 0):
    stored, spacing, _human_n, _coh = _gather(limit_stems=limit_stems)
    sess_summary = {}
    for sess_label, check_fn in (
        ("Session1_July_HeldOut (6-digit)", lambda st: len(st.split("_", 1)[0]) == 6),
        ("Session2_Oct_PostFreeze (4-digit)", lambda st: len(st.split("_", 1)[0]) == 4),
    ):
        sess_summary[sess_label] = {}
        for name in NAMES:
            rows_all = [r for r in stored[name] if check_fn(r["stem"])]
            rows_main = [r for r in rows_all if r["subset"] in ("Easy", "Open-arc")]
            rows_easy = [r for r in rows_all if r["subset"] == "Easy"]
            rows_open = [r for r in rows_all if r["subset"] == "Open-arc"]
            ok_05 = sum(1 for r in rows_main if _passes(r, spacing[r["stem"]], 0.5))
            ok_10 = sum(1 for r in rows_main if _passes(r, spacing[r["stem"]], 1.0))
            ok_none = sum(1 for r in rows_main if _passes(r, spacing[r["stem"]], None))
            sess_summary[sess_label][name] = {
                "n_total": len(rows_all),
                "n_readable": len(rows_main),
                "n_easy": len(rows_easy),
                "n_open": len(rows_open),
                "pass_0.5Lambda": f"{ok_05}/{len(rows_main)} ({ok_05/max(1,len(rows_main)):.1%})",
                "pass_1.0Lambda": f"{ok_10}/{len(rows_main)} ({ok_10/max(1,len(rows_main)):.1%})",
                "pass_untested": f"{ok_none}/{len(rows_main)} ({ok_none/max(1,len(rows_main)):.1%})",
            }

    dr2_rel_slopes = []
    max_boundary_drifts = []
    for row in stored["classical"]:
        stem = row["stem"]
        if row["subset"] not in ("Easy", "Open-arc"):
            continue
        human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        rs = sorted(human["radii"])
        if len(rs) >= 5:
            r2 = np.square(rs)
            dr2 = np.diff(r2)
            med_dr2 = float(np.median(dr2))
            if med_dr2 > 100:
                rel_dev = np.abs(dr2 - med_dr2) / med_dr2
                max_boundary_drifts.append(float(np.percentile(rel_dev, 90)))
                ks = np.arange(len(dr2), dtype=np.float64)
                slope = float(np.polyfit(ks, dr2 / med_dr2, 1)[0])
                dr2_rel_slopes.append(slope)

    distortion_stats = {
        "n_frames_with_5plus_rings": len(dr2_rel_slopes),
        "median_per_order_dr2_drift_frac": round(float(np.median(dr2_rel_slopes)), 4) if dr2_rel_slopes else None,
        "iqr_per_order_dr2_drift_frac": (
            round(float(np.percentile(dr2_rel_slopes, 75) - np.percentile(dr2_rel_slopes, 25)), 4)
            if dr2_rel_slopes
            else None
        ),
        "median_p90_step_deviation_frac": (
            round(float(np.median(max_boundary_drifts)), 3) if max_boundary_drifts else None
        ),
        "window_half_width_frac": 0.55,
    }
    return {"session_split": sess_summary, "empirical_dr2_regularity": distortion_stats}


def eval_c3_stats_and_stationary_clusters(*, limit_stems: int = 0):
    stored, spacing, human_n, coh = _gather(limit_stems=limit_stems)

    stems_main = [r["stem"] for r in stored["classical"] if r["subset"] in ("Easy", "Open-arc")]

    pass_vec = {name: {} for name in NAMES}
    ib_vec = {name: [] for name in NAMES}
    mr_vec = {name: [] for name in NAMES}

    for name in NAMES:
        row_map = {r["stem"]: r for r in stored[name]}
        coh_map = {r["stem"]: r for r in coh[name]}
        for gate in (0.5, 1.0, None):
            pass_vec[name][gate] = [
                1 if _passes(row_map[st], spacing[st], gate) else 0 for st in stems_main
            ]
        for st in stems_main:
            c_row = coh_map[st]
            ib_vec[name].append(float(c_row["covered"]) / max(1, int(c_row["n"])))
            s_row = row_map[st]
            mr_vec[name].append(0.0 if s_row["unformed"] else float(s_row["matches"]) / max(1, human_n[st]))

    def mcnemar_exact(a: list[int], b: list[int]) -> dict:
        b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
        b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
        n_disc = b10 + b01
        if n_disc == 0:
            p_val = 1.0
        else:
            p_val = float(stats.binomtest(min(b10, b01), n_disc, 0.5).pvalue)
        return {"a_only": b10, "b_only": b01, "p_exact": p_val}

    def wilcoxon_paired(a: list[float], b: list[float]) -> dict:
        diff = np.asarray(a) - np.asarray(b)
        if len(a) < 1 or np.all(np.abs(diff) < 1e-12):
            return {"median_diff": 0.0, "p_value": 1.0}
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return {
            "median_a": round(float(np.median(a)), 4),
            "median_b": round(float(np.median(b)), 4),
            "median_diff": round(float(np.median(diff)), 4),
            "p_value": float(res.pvalue),
        }

    pairs = [
        ("classical", "FRFT"),
        ("classical", "Hough"),
        ("v1.3", "FRFT"),
        ("v1.3", "Hough"),
        ("classical", "v1.3"),
    ]
    stat_tests = {
        "mcnemar_0.5Lambda": {},
        "mcnemar_1.0Lambda": {},
        "wilcoxon_in_band": {},
        "wilcoxon_matched_ring": {},
    }
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        stat_tests["mcnemar_0.5Lambda"][key] = mcnemar_exact(pass_vec[a][0.5], pass_vec[b][0.5])
        stat_tests["mcnemar_1.0Lambda"][key] = mcnemar_exact(pass_vec[a][1.0], pass_vec[b][1.0])
        stat_tests["wilcoxon_in_band"][key] = wilcoxon_paired(ib_vec[a], ib_vec[b])
        stat_tests["wilcoxon_matched_ring"][key] = wilcoxon_paired(mr_vec[a], mr_vec[b])

    static_cluster_stats = {}
    if STATIC_JSON.is_file():
        static_rows = json.loads(STATIC_JSON.read_text(encoding="utf-8"))
        for name in NAMES:
            all_rs = [
                row["fits"][name]["radii_px"]
                for row in static_rows
                if name in row.get("fits", {}) and not row["fits"][name]["unformed"]
            ]
            flat = sorted(float(r) for rs in all_rs for r in rs)
            clusters: list[list[float]] = []
            for r in flat:
                if not clusters or r - np.median(clusters[-1]) > 12.0:
                    clusters.append([r])
                else:
                    clusters[-1].append(r)
            valid_clusters = [c for c in clusters if len(c) >= 20]
            cluster_sds = [float(statistics.stdev(c)) for c in valid_clusters if len(c) >= 2]
            static_cluster_stats[name] = {
                "n_supported_orders": len(valid_clusters),
                "median_cluster_sd_px": round(float(statistics.median(cluster_sds)), 2) if cluster_sds else None,
                "max_cluster_sd_px": round(float(max(cluster_sds)), 2) if cluster_sds else None,
                "all_cluster_sds_px": [round(s, 2) for s in cluster_sds],
            }

    slow_info = {}
    if TIMING_JSON.is_file():
        timing = json.loads(TIMING_JSON.read_text(encoding="utf-8"))
        main_timing = [r for r in timing if r.get("subset") in ("Easy", "Open-arc")]
        if main_timing and "v1.3" in (main_timing[0].get("fits") or {}):
            slow_v13 = sorted(main_timing, key=lambda r: r["fits"]["v1.3"]["seconds"], reverse=True)[:5]
            slow_cls = sorted(main_timing, key=lambda r: r["fits"]["classical"]["seconds"], reverse=True)[:5]
            slow_info = {
                "top5_slowest_v1.3": [(r["stem"], r["subset"], r["fits"]["v1.3"]["seconds"]) for r in slow_v13],
                "top5_slowest_classical": [
                    (r["stem"], r["subset"], r["fits"]["classical"]["seconds"]) for r in slow_cls
                ],
            }

    return {
        "statistical_tests": stat_tests,
        "stationary_physical_clusters": static_cluster_stats,
        "slowest_frames": slow_info,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Smoke: only N readable frames for C1 MC / gather-dependent sections.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output directory.")
    args = parser.parse_args(argv)

    out_dir = args.out
    if out_dir is None:
        out_dir = RESULTS / "smoke_c1_c4" if args.limit else RESULTS / "c1_c4"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {
        "C1": eval_c1_gt_uncertainty_and_geometry(limit_frames=args.limit),
        "C2_part1": eval_c2_session_split_and_distortion(limit_stems=args.limit),
        "C3_part1": eval_c3_stats_and_stationary_clusters(limit_stems=args.limit),
    }
    out_file = out_dir / ("part1_smoke.json" if args.limit else "part1_results.json")
    out_file.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_file}", flush=True)
    # Compact console summary
    c1 = res["C1"]["mc_pole_uncertainty"].get("sigma_1.5px", {})
    print("C1 MC sigma=1.5px:", {k: v.get("n_frames") for k, v in c1.items()}, flush=True)
    w = res["C3_part1"]["statistical_tests"]["wilcoxon_in_band"].get("classical_vs_v1.3")
    print("Wilcoxon in-band classical vs v1.3:", w, flush=True)


if __name__ == "__main__":
    main()
