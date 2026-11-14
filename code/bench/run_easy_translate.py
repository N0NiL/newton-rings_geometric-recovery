# -*- coding: utf-8 -*-
"""人工平移截断实验：24 帧 Easy 画布整数像素平移，极心落出画布外 d 像素。

与 run_easy_truncate.py 的裁窗+重采样不同，本实验只做平移（无缩放），
保持条纹空间尺度不变。四个方向 pole_left/right/top/bottom，
每 (帧, d) 选保留弧点最多的方向；d=0 为不平移恒等对照。
平移约定：warpAffine(M=[[1,0,tx],[0,1,ty]]) 与点变换 p' = p + (tx, ty) 一致
（warpAffine 不传 WARP_INVERSE_MAP 时特征点位移为 +t）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    RESULTS,
    WEIGHTS_CKPT,
    setup_import_path,
)

PKG_ROOT = setup_import_path()

import run_deployed_measure as base  # noqa: E402
from NewModelDetection import NewModelDetector  # noqa: E402
from band_score import load_band  # noqa: E402
from run_classical_centerline import classical_valley  # noqa: E402
from circle_fit_helpers import _fit, _to_canvas  # noqa: E402
from run_hough_frft_look import frft_rings, hough_rings  # noqa: E402
from score_s12 import judge_s12  # noqa: E402

# 覆盖实验室硬编码：权重与数据落在包内
base._REPO = PKG_ROOT
base._CHECKPOINT = WEIGHTS_CKPT
base._SESSION = DATA_COMPARISON

SESSION = DATA_COMPARISON
OUT = RESULTS / "easy_translate"
DISTANCES = (0, 40, 100, 150, 200)
DIRECTIONS = ("pole_left", "pole_right", "pole_top", "pole_bottom")
CANVAS = 768
NAMES = ("classical", "v1.3", "Hough", "FRFT")
GATES = (0.5, 1.0, 2.0, None)
MIN_VISIBLE_POINTS = 20
PREVIEW_DISTANCES = (0, 100, 200)
PREVIEW_N = 4


def _origin(stem: str) -> str:
    head = stem.split("_", 1)[0]
    return "october" if len(head) == 4 else "july"


def _translation(cx: float, cy: float, d: int, direction: str) -> tuple[int, int]:
    """返回整数平移 (tx, ty)：图像与标注点统一 p' = p + (tx, ty)。

    使标注极心沿 direction 恰好落在画布外 d 像素（取整后真实距离在 d±0.5 px 内）。
    d = 0 时恒等 (0, 0)。
    """
    if d == 0:
        return 0, 0
    if direction == "pole_left":      # 目标 x' = -d
        return int(round(-d - cx)), 0
    if direction == "pole_right":     # 目标 x' = 768 + d
        return int(round(CANVAS + d - cx)), 0
    if direction == "pole_top":       # 目标 y' = -d
        return 0, int(round(-d - cy))
    if direction == "pole_bottom":    # 目标 y' = 768 + d
        return 0, int(round(CANVAS + d - cy))
    raise ValueError(direction)


def _shifted_pole(cx: float, cy: float, tx: int, ty: int) -> tuple[float, float]:
    return cx + tx, cy + ty


def _outside_px(px: float, py: float) -> float:
    """极心超出画布的距离（在画布内时为负值=到最近边的距离）。"""
    return float(max(-px, px - CANVAS, -py, py - CANVAS))


def transform_translate(
    band_human: dict, human_json: dict, tx: int, ty: int, direction: str, d: int
) -> tuple[dict, int, int]:
    """把标注平移到新画布坐标系（scale=1），保留落在 [0,768)^2 内的点。"""
    cx0, cy0 = float(human_json["center_xy"][0]), float(human_json["center_xy"][1])
    cx, cy = _shifted_pole(cx0, cy0, tx, ty)
    band_half = []
    for ring in band_human["bands"]:
        band_half.append((float(ring["radius"]), 0.5 * (float(ring["r_out"]) - float(ring["r_in"]))))
    bands = []
    total_points = 0
    rings_hit = 0
    for item in human_json.get("rings") or []:
        if str(item.get("role") or "newton") != "newton":
            continue
        points = np.asarray(item.get("points_xy") or [], np.float64)
        if len(points) < 1:
            continue
        pts = points + np.array([tx, ty], dtype=np.float64)
        keep = (pts[:, 0] >= 0) & (pts[:, 0] < CANVAS) & (pts[:, 1] >= 0) & (pts[:, 1] < CANVAS)
        kept = pts[keep]
        if len(kept) < 1:
            continue
        src_radius = float(item["radius_px"])
        half = 5.0
        if band_half:
            nearest = min(band_half, key=lambda pair: abs(pair[0] - src_radius))
            half = float(nearest[1])
        angles = np.arctan2(kept[:, 1] - cy, kept[:, 0] - cx)
        bands.append(
            {
                "radius": src_radius,
                "r_in": src_radius - half,
                "r_out": src_radius + half,
                "angles": angles,
                "points_xy": kept.tolist(),
            }
        )
        rings_hit += 1
        total_points += len(kept)
    bands.sort(key=lambda item: item["radius"])
    radii = [item["radius"] for item in bands]
    gaps = np.diff(radii) if len(radii) >= 2 else np.array([])
    spacing = float(np.median(gaps)) if gaps.size else 0.0
    human = {
        "n": len(bands),
        "bands": bands,
        "radii": radii,
        "center": (cx, cy),
        "in_frame": d == 0,
        "side": direction,
        "spacing": spacing,
        "ref_pole_source": "easy_closed_ring_shared_center_translated",
        "scale": 1.0,
    }
    return human, rings_hit, total_points


def _passes(row: dict, spacing: float, limit: float | None) -> bool:
    if not row.get("success"):
        return False
    if limit is None:
        return True
    err = row.get("center_error_px")
    if err is None or spacing <= 1e-6:
        return False
    return float(err) <= limit * spacing + 1e-9


def _gate_table(rows_by_name: dict[str, list[dict]]) -> dict:
    out = {}
    for name, rows in rows_by_name.items():
        out[name] = {}
        for gate in GATES:
            key = "none" if gate is None else f"{gate:g}s"
            ok = sum(1 for row in rows if _passes(row, float(row["spacing"]), gate))
            out[name][key] = {"ok": ok, "n": len(rows)}
    return out


def invent_raw_from_canvas(canvas_bgr: np.ndarray, mapping_crop_x0: int = 420) -> np.ndarray:
    """把 768 画布回填到 1920×1080 的标准裁切位，供 MA-Net 预处理吃到同一内容。"""
    raw = np.zeros((1080, 1920, 3), dtype=np.uint8)
    patch = cv2.resize(canvas_bgr, (1080, 1080), interpolation=cv2.INTER_AREA)
    raw[:, mapping_crop_x0 : mapping_crop_x0 + 1080] = patch
    return raw


def _median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else None


def _preview(
    stems_d: list[tuple[str, int]],
    transforms: dict,
    rows_by_name: dict[tuple[str, int], dict[str, dict]],
    d: int,
) -> None:
    """4 帧拼图：绿色=保留标注点，黄色=classical 拟合圆，箭头指向极心方位。"""
    panels = []
    for stem, dd in stems_d:
        info = transforms[(stem, dd)]
        canvas = cv2.imread(str(OUT / f"{stem}_d{dd}.png"), cv2.IMREAD_COLOR)
        if canvas is None:
            continue
        human = info["human"]
        cx, cy = human["center"]
        for band in human["bands"]:
            for x, y in band.get("points_xy") or []:
                cv2.circle(canvas, (int(round(x)), int(round(y))), 1, (0, 255, 0), -1)
        row = rows_by_name.get((stem, dd), {}).get("classical")
        if row and row.get("center_xy") and row.get("radii_px"):
            ccx, ccy = row["center_xy"]
            for r in row["radii_px"]:
                cv2.circle(canvas, (int(round(ccx)), int(round(ccy))), int(round(r)), (0, 255, 255), 1)
        direction = info["direction"]
        if direction != "identity":
            mx, my = CANVAS / 2, CANVAS / 2
            ex, ey = {
                "pole_left": (0, my),
                "pole_right": (CANVAS - 1, my),
                "pole_top": (mx, 0),
                "pole_bottom": (mx, CANVAS - 1),
            }[direction]
            cv2.arrowedLine(canvas, (int(mx), int(my)), (int(ex), int(ey)), (255, 0, 0), 2, tipLength=0.08)
        cv2.putText(
            canvas,
            f"{stem} d={dd} {direction} out={info['pole_outside_px']:.1f}px",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(cv2.resize(canvas, (384, 384), interpolation=cv2.INTER_AREA))
    if not panels:
        return
    while len(panels) < PREVIEW_N:
        panels.append(np.zeros_like(panels[0]))
    top = np.hstack(panels[:2])
    bot = np.hstack(panels[2:])
    montage = np.vstack([top, bot])
    cv2.imwrite(str(OUT / f"preview_d{d}.png"), montage)
    print(f"Saved preview_d{d}.png", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS_CKPT.is_file():
        raise SystemExit(
            f"缺少权重 {WEIGHTS_CKPT}。请将 50distill_last.pt 放到 weights/ 后重跑；"
            f"冻结结果见 {OUT}。"
        )
    stems = sorted(
        p.name.replace("_human.json", "")
        for p in SESSION.glob("*Easy_human.json")
    )
    if len(stems) != 24:
        print(f"警告：Easy 帧数为 {len(stems)}，期望 24", flush=True)

    print(f"加载学生模型 {base._CHECKPOINT.name}", flush=True)
    detector = NewModelDetector(checkpoint_path=base._CHECKPOINT, strict=False, repo_root=base._REPO)
    stored: dict[int, dict[str, list[dict]]] = {d: {name: [] for name in NAMES} for d in DISTANCES}
    transforms: dict[tuple[str, int], dict] = {}
    frame_log: list[dict] = []
    ineligible: dict[int, list[str]] = {d: [] for d in DISTANCES}
    t0 = time.perf_counter()

    for index, stem in enumerate(stems, 1):
        canvas_full = cv2.imread(str(SESSION / f"{stem}_canvas.png"), cv2.IMREAD_COLOR)
        human_json = json.loads((SESSION / f"{stem}_human.json").read_text(encoding="utf-8"))
        band_human = load_band(SESSION / f"{stem}_band.json", SESSION / f"{stem}_human.json")
        cx0, cy0 = float(human_json["center_xy"][0]), float(human_json["center_xy"][1])

        for d in DISTANCES:
            directions = ("identity",) if d == 0 else DIRECTIONS
            # 4 方向择优：rings_hit -> visible_points
            best = None
            for direction in directions:
                tx, ty = _translation(cx0, cy0, d, direction)
                human, rings_hit, points = transform_translate(band_human, human_json, tx, ty, direction, d)
                if rings_hit < 3 or points < MIN_VISIBLE_POINTS:
                    continue
                score = (rings_hit, points)
                if best is None or score > best[0]:
                    best = (score, direction, tx, ty, human, rings_hit, points)
            if best is None:
                ineligible[d].append(stem)
                frame_log.append({"stem": stem, "d": d, "eligible": False})
                print(f"[{index}/{len(stems)}] {stem} d={d} INELIGIBLE", flush=True)
                continue
            _, direction, tx, ty, human, rings_hit, points = best
            if human["n"] < 3 or human["spacing"] <= 1e-6:
                ineligible[d].append(stem)
                frame_log.append({"stem": stem, "d": d, "eligible": False})
                print(f"[{index}/{len(stems)}] {stem} d={d} INELIGIBLE(spacing)", flush=True)
                continue

            M = np.float32([[1, 0, tx], [0, 1, ty]])
            canvas = cv2.warpAffine(canvas_full, M, (CANVAS, CANVAS), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            fake_raw = invent_raw_from_canvas(canvas)
            cxp, cyp = human["center"]
            pole_out = _outside_px(cxp, cyp)
            spacing = float(human["spacing"])
            black_frac = float((gray == 0).mean())

            # 硬性校验：保留标注点必须落在暗条纹上（亮度低于非黑像素均值）
            kept_pts = [pt for b in human["bands"] for pt in (b.get("points_xy") or [])]
            if kept_pts:
                xs = np.clip([int(round(p[0])) for p in kept_pts], 0, CANVAS - 1)
                ys = np.clip([int(round(p[1])) for p in kept_pts], 0, CANVAS - 1)
                pt_mean = float(gray[ys, xs].mean())
                nb_mean = float(gray[gray > 0].mean()) if (gray > 0).any() else 0.0
                assert pt_mean < nb_mean, (
                    f"{stem} d={d} {direction}: 标注点均值 {pt_mean:.1f} >= 非黑像素均值 {nb_mean:.1f}"
                )

            transforms[(stem, d)] = {
                "human": human,
                "direction": direction,
                "tx": tx,
                "ty": ty,
                "pole_outside_px": pole_out,
            }

            started = time.perf_counter()
            prob, _ = detector.get_refined_skeleton(fake_raw)
            prepared = detector.inference.last_preprocess
            mapping = prepared.mapping
            center_m, radii_m = _fit(prob, prepared.model_u8, nocap=False, physics=None, blend=0.0)
            center_v, radii_v = _to_canvas(center_m, radii_m, mapping, CANVAS, mapping.crop_side)
            model_s = time.perf_counter() - started

            started = time.perf_counter()
            valley = classical_valley(gray)
            center_c, radii_c = _fit(valley, gray, nocap=False, physics=None, blend=0.0)
            classical_s = time.perf_counter() - started

            started = time.perf_counter()
            center_h, radii_h = hough_rings(canvas)
            hough_s = time.perf_counter() - started

            started = time.perf_counter()
            center_f, radii_f = frft_rings(canvas)
            frft_s = time.perf_counter() - started

            fits = {
                "classical": (center_c, radii_c, classical_s),
                "v1.3": (center_v, radii_v, model_s),
                "Hough": (center_h, radii_h, hough_s),
                "FRFT": (center_f, radii_f, frft_s),
            }
            marks = []
            for name, (center, radii, elapsed) in fits.items():
                judged = judge_s12(human, list(radii), center)
                err = judged.get("center_error_px")
                row = {
                    "stem": stem,
                    "subset": "Easy-translate",
                    "origin": _origin(stem),
                    "d": d,
                    "direction": direction,
                    "seconds": round(float(elapsed), 3),
                    "n_human": human["n"],
                    "n_pred": len(list(radii)),
                    "center_xy": None if center is None else [round(float(center[0]), 2), round(float(center[1]), 2)],
                    "ref_center_xy": [round(float(human["center"][0]), 2), round(float(human["center"][1]), 2)],
                    "spacing": round(spacing, 3),
                    "pole_outside_px": round(float(pole_out), 2),
                    "pole_outside_in_spacing": round(float(pole_out) / spacing, 3) if spacing > 1e-6 else None,
                    "visible_points": points,
                    "black_frac": round(black_frac, 4),
                    "radii_px": [round(float(r), 2) for r in radii],
                    "center_ok_0p5": bool(err is not None and err <= 0.5 * spacing + 1e-9),
                    "center_ok_1p0": bool(err is not None and err <= 1.0 * spacing + 1e-9),
                    "center_ok_2p0": bool(err is not None and err <= 2.0 * spacing + 1e-9),
                    **judged,
                }
                stored[d][name].append(row)
                marks.append(f"{name}{'✓' if row['success'] else '×'}")
            frame_log.append(
                {
                    "stem": stem,
                    "d": d,
                    "eligible": True,
                    "direction": direction,
                    "pole_outside_px": round(float(pole_out), 2),
                    "spacing": round(spacing, 3),
                    "rings_hit": rings_hit,
                    "visible_points": points,
                    "black_frac": round(black_frac, 4),
                }
            )
            print(
                f"[{index}/{len(stems)}] {stem} d={d} {direction} out={pole_out:.1f}px "
                f"rings={rings_hit} pts={points} black={black_frac:.2f} " + " ".join(marks),
                flush=True,
            )
            cv2.imwrite(str(OUT / f"{stem}_d{d}.png"), canvas)

    # d=0 恒等对照硬性校验：classical 圆心在 0.5 Λ 内 ≥20/24
    d0_ok = sum(1 for r in stored[0]["classical"] if r.get("center_ok_0p5"))
    assert d0_ok >= 20, f"d=0 恒等对照失败：classical 0.5Λ 通过 {d0_ok}/24 < 20"
    print(f"d=0 control: classical center_ok_0p5 = {d0_ok}/24", flush=True)

    # 预览拼图：每个预览距离取前 PREVIEW_N 个合格帧
    rows_by_name: dict[tuple[str, int], dict[str, dict]] = {}
    for d in DISTANCES:
        for name in NAMES:
            for r in stored[d][name]:
                rows_by_name.setdefault((r["stem"], d), {})[name] = r
    for d in PREVIEW_DISTANCES:
        eligible = sorted({r["stem"] for r in stored[d]["classical"]})
        picks = [(s, d) for s in eligible[:PREVIEW_N]]
        _preview(picks, transforms, rows_by_name, d)

    summary = {
        "n_frames": len(stems),
        "distances_px": list(DISTANCES),
        "directions": list(DIRECTIONS),
        "canvas": CANVAS,
        "min_visible_points": MIN_VISIBLE_POINTS,
        "ref_pole": "Easy 完整闭合环共享圆心，整数平移到画布坐标系；非开弧外推；无缩放；d=0 恒等对照",
        "runtime_s": round(time.perf_counter() - t0, 1),
        "d0_control_classical_0p5": d0_ok,
        "ineligible": {str(d): v for d, v in ineligible.items()},
        "per_d": {},
        "frames": frame_log,
    }
    for d in DISTANCES:
        n_elig = len(stored[d]["classical"])
        per_route = {}
        for name in NAMES:
            rows = stored[d][name]
            errs = [r.get("center_error_px") for r in rows]
            over = [r.get("pole_outside_in_spacing") for r in rows]
            per_route[name] = {
                "median_center_error_px": _median(errs),
                "median_outside_in_spacing": _median(over),
            }
        summary["per_d"][str(d)] = {
            "n_eligible": n_elig,
            "gates": _gate_table(stored[d]),
            "median_center_error_px": {n: per_route[n]["median_center_error_px"] for n in NAMES},
            "median_d_over_spacing": _median([r.get("pole_outside_in_spacing") for r in stored[d]["classical"]]),
            "median_black_frac": _median([r.get("black_frac") for r in stored[d]["classical"]]),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for d in DISTANCES:
        for name in NAMES:
            (OUT / f"{name}_d{d}.json").write_text(
                json.dumps(stored[d][name], ensure_ascii=False, indent=2), encoding="utf-8"
            )

    lines = [
        "# Easy 平移截断实验摘要",
        "",
        f"帧数：{len(stems)}；距离 {list(DISTANCES)} px（768 画布，整数平移无缩放）；方向 {list(DIRECTIONS)}，每 (帧,d) 取保留点最多者；d=0 恒等对照。",
        "参考极心：完整闭合环共享圆心平移，非开弧外推。",
        f"d=0 对照：classical 圆心 0.5 Λ 通过 {d0_ok}/24。",
        f"总耗时：{summary['runtime_s']} s。",
        "",
    ]
    for d in DISTANCES:
        info = summary["per_d"][str(d)]
        bf = info["median_black_frac"]
        bf_s = "—" if bf is None else f"{bf:.3f}"
        lines += [
            f"## d = {d} px（合格 {info['n_eligible']}/24；中位 d/Λ = {info['median_d_over_spacing']}；中位黑像素占比 = {bf_s}）",
            "",
            "| 路线 | 0.5 Λ | 1 Λ | 2 Λ | 不测圆心 | 中位圆心误差 px |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        g = info["gates"]
        for name in NAMES:
            med = info["median_center_error_px"][name]
            med_s = "—" if med is None else f"{med:.1f}"
            lines.append(
                f"| {name} | {g[name]['0.5s']['ok']}/{g[name]['0.5s']['n']} | {g[name]['1s']['ok']}/{g[name]['1s']['n']} | "
                f"{g[name]['2s']['ok']}/{g[name]['2s']['n']} | {g[name]['none']['ok']}/{g[name]['none']['n']} | {med_s} |"
            )
        if ineligible[d]:
            lines.append("")
            lines.append(f"不合格帧：{', '.join(ineligible[d])}")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("done", OUT, flush=True)


if __name__ == "__main__":
    main()
