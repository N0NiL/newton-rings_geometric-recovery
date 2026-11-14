# -*- coding: utf-8 -*-
"""Wilson CI, exact McNemar, Wilcoxon, Holm — recompute paper statistics from fits."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parents[1]
BENCH = PKG_ROOT / "code" / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from pkg_paths import NAMES, setup_import_path  # noqa: E402

setup_import_path()

import coherency  # noqa: E402
from table1 import load_readable_frames, pass_vector, score_entry  # noqa: E402


def wilson_interval(k: int, n: int, *, alpha: float = 0.05) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    k = int(k)
    n = int(n)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return float(phat), float(max(0.0, centre - half)), float(min(1.0, centre + half))


@dataclass
class McNemarResult:
    b10: int
    b01: int
    n_discordant: int
    p_exact: float


def mcnemar_exact(a: Sequence[int] | np.ndarray, b: Sequence[int] | np.ndarray) -> McNemarResult:
    aa = np.asarray(a, dtype=int).ravel()
    bb = np.asarray(b, dtype=int).ravel()
    if aa.shape != bb.shape:
        raise ValueError("paired vectors must share shape")
    b10 = int(np.sum((aa == 1) & (bb == 0)))
    b01 = int(np.sum((aa == 0) & (bb == 1)))
    n_disc = b10 + b01
    if n_disc == 0:
        p = 1.0
    else:
        p = float(stats.binomtest(min(b10, b01), n_disc, 0.5).pvalue)
    return McNemarResult(b10=b10, b01=b01, n_discordant=n_disc, p_exact=p)


def holm_correct(p_values: Iterable[float]) -> list[float]:
    ps = list(p_values)
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        cand = factor * float(ps[i])
        running = max(running, cand)
        adj[i] = min(1.0, running)
    return adj


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    diff = aa - bb
    if len(aa) < 1 or np.all(np.abs(diff) < 1e-12):
        return {"median_diff": 0.0, "p_value": 1.0}
    res = stats.wilcoxon(aa, bb, zero_method="wilcox", alternative="two-sided")
    return {
        "median_a": round(float(np.median(aa)), 4),
        "median_b": round(float(np.median(bb)), 4),
        "median_diff": round(float(np.median(diff)), 4),
        "p_value": float(res.pvalue),
    }


def _fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _per_frame_metrics(frames: dict[str, list[dict]]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """In-band fraction and matched-ring fraction per readable frame."""
    ib: dict[str, list[float]] = {name: [] for name in NAMES}
    mr: dict[str, list[float]] = {name: [] for name in NAMES}
    # stems aligned on classical order
    for i, entry in enumerate(frames["classical"]):
        stem = entry["stem"]
        visible = coherency._visible(stem)
        human_n = max(1, int(entry["human"]["n"]))
        for name in NAMES:
            e = frames[name][i]
            assert e["stem"] == stem
            row = {
                "center_xy": None if e["center"] is None else list(e["center"]),
                "radii_px": e["radii"],
            }
            c_row = coherency._frame(visible, row)
            ib[name].append(float(c_row["covered"]) / max(1, int(c_row["n"])))
            score = score_entry(e)
            mr[name].append(0.0 if score.get("unformed") else float(score["matches"]) / human_n)
    return ib, mr


def main() -> None:
    frames = load_readable_frames()
    n = len(frames["classical"])
    assert n == 69, n

    vec05 = {name: pass_vector(frames, name, 0.5) for name in NAMES}
    vec10 = {name: pass_vector(frames, name, 1.0) for name in NAMES}
    ib, mr = _per_frame_metrics(frames)

    print(f"=== Wilson 95% CI @ 0.5 Λ (n={n}) ===")
    paper_wilson = {
        "classical": (57, (0.720, 0.898)),
        "v1.3": (47, (0.564, 0.779)),
        "FRFT": (30, (0.324, 0.552)),
        "Hough": (0, (0.0, 0.053)),
    }
    for name in NAMES:
        k = sum(vec05[name])
        r, lo, hi = wilson_interval(k, n)
        target_k, (plo, phi) = paper_wilson[name]
        print(
            f"  {name:<12} {k}/{n} = {100*r:.1f}% [{100*lo:.1f}, {100*hi:.1f}]"
            f"   paper {target_k}/69 ~[{100*plo:.1f}, {100*phi:.1f}]"
        )

    mcnemar_specs = [
        ("classical_vs_FRFT@0.5Λ", vec05["classical"], vec05["FRFT"]),
        ("v1.3_vs_FRFT@0.5Λ", vec05["v1.3"], vec05["FRFT"]),
        ("classical_vs_Hough@0.5Λ", vec05["classical"], vec05["Hough"]),
        ("v1.3_vs_Hough@0.5Λ", vec05["v1.3"], vec05["Hough"]),
        ("classical_vs_v1.3@0.5Λ", vec05["classical"], vec05["v1.3"]),
        ("classical_vs_v1.3@1.0Λ", vec10["classical"], vec10["v1.3"]),
    ]

    print("\n=== Exact McNemar (readable 69) ===")
    raw_ps: list[float] = []
    labels: list[str] = []
    for label, a, b in mcnemar_specs:
        m = mcnemar_exact(a, b)
        raw_ps.append(m.p_exact)
        labels.append(label)
        print(f"  {label}: disc={m.b10}:{m.b01}  p={_fmt_p(m.p_exact)}")

    open_idx = [i for i, e in enumerate(frames["classical"]) if e["subset"] == "Open-arc"]
    m_open = mcnemar_exact(
        [vec05["classical"][i] for i in open_idx],
        [vec05["v1.3"][i] for i in open_idx],
    )
    print(
        f"  open-arc classical_vs_v1.3@0.5Λ: disc={m_open.b10}:{m_open.b01} "
        f"p={_fmt_p(m_open.p_exact)}"
    )

    wilcoxon_specs = [
        ("wilcoxon_inband_v13_vs_FRFT", ib["v1.3"], ib["FRFT"]),
        ("wilcoxon_match_v13_vs_FRFT", mr["v1.3"], mr["FRFT"]),
        ("wilcoxon_inband_v13_vs_classical", ib["v1.3"], ib["classical"]),
        ("wilcoxon_match_v13_vs_classical", mr["v1.3"], mr["classical"]),
    ]
    print("\n=== Wilcoxon signed-rank (per-frame fractions) ===")
    wil_ps: list[float] = []
    wil_labels: list[str] = []
    for label, a, b in wilcoxon_specs:
        w = wilcoxon_paired(a, b)
        wil_ps.append(w["p_value"])
        wil_labels.append(label)
        print(
            f"  {label}: med_a={w['median_a']} med_b={w['median_b']} "
            f"diff={w['median_diff']} p={_fmt_p(w['p_value'])}"
        )

    ten_labels = labels + wil_labels
    ten_ps = raw_ps + wil_ps
    adj = holm_correct(ten_ps)
    print("\n=== Holm correction (10 paired tests: 6 McNemar + 4 Wilcoxon) ===")
    for lab, p, a in zip(ten_labels, ten_ps, adj):
        print(f"  {lab}: raw={_fmt_p(p)}  Holm={_fmt_p(a)}")
    print(
        "  note: paper states every comparison remains <0.05 after Holm "
        "except the 1.0 Λ frontend comparison (p=1.00)."
    )


if __name__ == "__main__":
    main()
