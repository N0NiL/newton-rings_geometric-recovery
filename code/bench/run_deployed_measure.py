"""按 /v2 确认拟合跑正式场，对照人工弧，写出文本和审阅图。

学生网络，物理头不掺入圆心。每张图单独定环，不把上一张的圆心带过来。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from pkg_paths import (  # noqa: E402
    DATA_COMPARISON,
    PKG_ROOT,
    RESULTS,
    WEIGHTS_CKPT,
    setup_import_path,
)

_REPO = setup_import_path()

from AfterProcess import NewtonSecondaryPipeline  # noqa: E402
from AlgorithmFitter import NewtonFitter  # noqa: E402
from NewModelDetection import NewModelDetector  # noqa: E402
from RadialRingAccumulator import AccumulatorConfig, RadialRingAccumulator  # noqa: E402
from paper_annotate import fit_concentric, off_side  # noqa: E402

try:
    from vision_v2_web import _physical_from_radii_px  # noqa: E402
except ImportError:  # 发布包不带平台 web 模块；仅影响物理读数旁注
    def _physical_from_radii_px(pred_rs):  # type: ignore[misc]
        return None

_SESSION = DATA_COMPARISON
_OUT = RESULTS / "measure_deployed"
_CHECKPOINT = WEIGHTS_CKPT
_FONT = Path(r"C:\Windows\Fonts\msyh.ttc")


def _font(size: int) -> ImageFont.ImageFont:
    if _FONT.is_file():
        return ImageFont.truetype(str(_FONT), size=size)
    return ImageFont.load_default()


def _human(payload: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    groups = []
    for ring in payload.get("rings") or []:
        if str(ring.get("role") or "newton") != "newton":
            continue
        points = np.asarray(ring.get("points_xy") or [], np.float64)
        if len(points) >= 3:
            groups.append(points)
    fitted = fit_concentric(groups) if groups else None
    if fitted is None:
        return {"n": 0, "radii": [], "center": None, "in_frame": None, "side": None}
    center = (float(fitted[0]), float(fitted[1]))
    radii = [float(r) for r in fitted[2]]
    order = np.argsort(radii)
    radii = [radii[i] for i in order]
    side = off_side(center[0], center[1], width, height)
    return {
        "n": len(radii),
        "radii": radii,
        "center": center,
        "in_frame": side is None,
        "side": side,
    }


def _to_canvas(x: float, y: float, mapping: Any, canvas_side: int, crop_side: int) -> tuple[float, float]:
    source_x, source_y = mapping.model_to_source(float(x), float(y))
    scale = canvas_side / float(crop_side)
    return (source_x - mapping.crop_x0) * scale, (source_y - mapping.crop_y0) * scale


def _length_to_canvas(value: float, mapping: Any, canvas_side: int, crop_side: int) -> float:
    return float(mapping.model_length_to_source(float(value))) * canvas_side / float(crop_side)


def _match(human_rs: list[float], pred_rs: list[float], spacing: float) -> dict[str, Any]:
    limit = 0.4 * spacing
    pairs: list[tuple[float, int, int]] = []
    for i, human_r in enumerate(human_rs):
        for j, pred_r in enumerate(pred_rs):
            gap = abs(human_r - pred_r)
            if gap < limit:
                pairs.append((gap, i, j))
    pairs.sort()
    used_h: set[int] = set()
    used_p: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _gap, i, j in pairs:
        if i in used_h or j in used_p:
            continue
        used_h.add(i)
        used_p.add(j)
        matched.append((i, j))
    matched.sort()
    pred_order = [j for _i, j in matched]
    return {
        "matches": len(matched),
        "misses": len(human_rs) - len(matched),
        "false_rings": len(pred_rs) - len(matched),
        "order_ok": pred_order == sorted(pred_order),
    }


def _center_ok(
    human: dict[str, Any], pred_xy: tuple[float, float] | None, width: int, height: int, spacing: float
) -> bool:
    if pred_xy is None or human["center"] is None:
        return False
    if human["in_frame"]:
        hx, hy = human["center"]
        return float(np.hypot(pred_xy[0] - hx, pred_xy[1] - hy)) <= 0.5 * spacing
    side = human["side"]
    px, py = pred_xy
    if side == "left":
        return px < 0
    if side == "right":
        return px >= width
    if side == "top":
        return py < 0
    if side == "bottom":
        return py >= height
    return False


def _judge(human: dict[str, Any], pred_rs: list[float], pred_xy: tuple[float, float] | None, width: int, height: int) -> dict[str, Any]:
    radii = human["radii"]
    n_h = len(radii)
    gaps = np.diff(radii) if n_h >= 2 else np.array([])
    spacing = float(np.median(gaps)) if gaps.size else 0.0
    compared = _match(radii, pred_rs, spacing) if spacing > 1e-6 else {
        "matches": 0, "misses": n_h, "false_rings": len(pred_rs), "order_ok": False
    }
    enough = len(pred_rs) >= 3
    miss_limit = int(np.floor(0.3 * n_h))
    center_ok = _center_ok(human, pred_xy, width, height, spacing) if spacing > 1e-6 else False
    success = bool(
        n_h >= 3
        and enough
        and compared["matches"] >= 3
        and compared["misses"] <= miss_limit
        and compared["false_rings"] == 0
        and compared["order_ok"]
        and center_ok
    )
    refused = len(pred_rs) < 3
    return {
        "spacing_px": None if spacing <= 1e-6 else round(spacing, 2),
        "matches": compared["matches"],
        "misses": compared["misses"],
        "false_rings": compared["false_rings"],
        "miss_limit": miss_limit,
        "order_ok": compared["order_ok"],
        "center_ok": center_ok,
        "refused": refused,
        "success": success,
    }


def _caption(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    bar_h = 28 + 22 * len(lines)
    sheet = Image.new("RGB", (canvas.width, canvas.height + bar_h), (18, 18, 18))
    sheet.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for i, line in enumerate(lines):
        draw.text((12, canvas.height + 8 + 22 * i), line, fill=(235, 235, 235), font=font)
    return cv2.cvtColor(np.asarray(sheet), cv2.COLOR_RGB2BGR)


def _draw_fit(
    canvas: np.ndarray,
    human: dict[str, Any],
    pred_rs: list[float],
    pred_xy: tuple[float, float] | None,
    lines: list[str],
) -> np.ndarray:
    view = canvas.copy()
    if human["center"] is not None:
        hx, hy = human["center"]
        for radius in human["radii"]:
            cv2.circle(view, (int(round(hx)), int(round(hy))), max(1, int(round(radius))), (0, 180, 0), 1, cv2.LINE_AA)
    if pred_xy is not None:
        for radius in pred_rs:
            cv2.circle(view, (int(round(pred_xy[0])), int(round(pred_xy[1]))), max(1, int(round(radius))), (0, 200, 255), 1, cv2.LINE_AA)
    return _caption(view, lines)


def _sheet(tiles: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not tiles:
        return np.zeros((64, 64, 3), np.uint8)
    cols = 6
    thumb_w = 360
    rows = int(np.ceil(len(tiles) / cols))
    sample = tiles[0][1]
    scale = thumb_w / float(sample.shape[1])
    thumb_h = int(round(sample.shape[0] * scale))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 12, np.uint8)
    for index, (_name, image) in enumerate(tiles):
        thumb = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        r, c = divmod(index, cols)
        sheet[r * thumb_h : (r + 1) * thumb_h, c * thumb_w : (c + 1) * thumb_w] = thumb
    return sheet


def _measure_one(detector: NewModelDetector, raw: np.ndarray, canvas: np.ndarray, record: dict[str, Any]) -> dict[str, Any]:
    prob, _contours = detector.get_refined_skeleton(raw)
    prepared = detector.inference.last_preprocess
    mapping = prepared.mapping
    skeleton = NewtonSecondaryPipeline().run(prob).skeleton
    fitter = NewtonFitter(physics_head_blend=0.0, geom_head_blend=0.0, center_warm_alpha=0.0)
    fitter.clear_physics_hint()
    try:
        fit = fitter.fit_circles_with_topology(skeleton, prob_hw=prob, gray_hw=prepared.model_u8)
    except Exception:
        fit = None
    center = None
    model_n = None
    if isinstance(fit, dict) and np.isfinite(fit.get("x0", np.nan)):
        center = (float(fit["x0"]), float(fit["y0"]))
        rs_fit = fit.get("rs") or []
        if len(rs_fit) >= 2:
            model_n = int(len(rs_fit))
    accum = RadialRingAccumulator(AccumulatorConfig(center_follow=1.0))
    radial = accum.fit_static_once(prob, center_xy=center, model_n_rings=model_n)
    info = record.get("mapping") or {}
    canvas_side = int(info.get("model_side") or canvas.shape[0])
    crop_side = int(info.get("crop_side") or 1080)
    pred_rs: list[float] = []
    pred_xy = None
    if isinstance(radial, dict) and radial.get("rs"):
        pred_rs = sorted(_length_to_canvas(r, mapping, canvas_side, crop_side) for r in radial["rs"])
        pred_xy = _to_canvas(float(radial["x0"]), float(radial["y0"]), mapping, canvas_side, crop_side)
    elif center is not None:
        pred_xy = _to_canvas(center[0], center[1], mapping, canvas_side, crop_side)
    physical = _physical_from_radii_px(pred_rs)
    return {"radii": pred_rs, "center": pred_xy, "R_m": physical.get("R_m"), "fit_n": model_n}


def main() -> int:
    frames = sorted(_SESSION.glob("*_canvas.png"))
    if not frames:
        print("正式场没有分析画布。")
        return 1
    _OUT.mkdir(parents=True, exist_ok=True)
    frame_dir = _OUT / "frames"
    frame_dir.mkdir(exist_ok=True)
    print(f"加载学生模型 {_CHECKPOINT.name}")
    detector = NewModelDetector(checkpoint_path=_CHECKPOINT, strict=False, repo_root=_REPO)
    rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, np.ndarray]] = []
    for index, canvas_path in enumerate(frames, 1):
        stem = canvas_path.name.replace("_canvas.png", "")
        raw = cv2.imread(str(canvas_path.with_name(stem + "_raw.png")), cv2.IMREAD_COLOR)
        canvas = cv2.imread(str(canvas_path), cv2.IMREAD_COLOR)
        record = json.loads(canvas_path.with_name(stem + ".json").read_text(encoding="utf-8"))
        human_payload = json.loads(canvas_path.with_name(stem + "_human.json").read_text(encoding="utf-8"))
        height, width = canvas.shape[:2]
        started = time.perf_counter()
        try:
            measured = _measure_one(detector, raw, canvas, record)
            error = None
        except Exception as exc:
            measured = {"radii": [], "center": None, "R_m": None, "fit_n": None}
            error = repr(exc)
        elapsed = time.perf_counter() - started
        human = _human(human_payload, width, height)
        judged = _judge(human, measured["radii"], measured["center"], width, height)
        subset = str(record.get("subset") or stem.split("_", 1)[-1])
        row = {
            "stem": stem,
            "subset": subset,
            "seconds": round(elapsed, 2),
            "n_human": human["n"],
            "n_pred": len(measured["radii"]),
            "human_center": None if human["center"] is None else [round(human["center"][0], 1), round(human["center"][1], 1)],
            "pred_center": None if measured["center"] is None else [round(measured["center"][0], 1), round(measured["center"][1], 1)],
            "human_side": human["side"],
            "R_m": None if measured["R_m"] is None else round(float(measured["R_m"]), 4),
            "error": error,
            **judged,
        }
        rows.append(row)
        mark = "成功" if row["success"] else ("拒测" if row["refused"] else "未成功")
        lines = [
            f"{stem}  {subset}  {mark}  {row['seconds']:.1f} 秒",
            f"人工 {row['n_human']} 条，算法 {row['n_pred']} 条；配对 {row['matches']}，漏 {row['misses']}，假环 {row['false_rings']}",
            "绿圈是人工弧，黄圈是部署拟合。曲率半径只在结构成功后另看文本。",
        ]
        image = _draw_fit(canvas, human, measured["radii"], measured["center"], lines)
        cv2.imwrite(str(frame_dir / f"{stem}.png"), image)
        tiles.append((stem, image))
        print(f"[{index}/{len(frames)}] {stem}  {mark}  人工{row['n_human']} 算法{row['n_pred']}", flush=True)

    (_OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    text = _render_text(rows)
    (_OUT / "summary.txt").write_text(text, encoding="utf-8")
    cv2.imwrite(str(_OUT / "review_all.png"), _sheet(tiles))
    print(text)
    print(f"输出目录：{_OUT}")
    return 0


def _render_text(rows: list[dict[str, Any]]) -> str:
    lines = [
        "部署测量：学生网络 50distill_last.pt，768 分析画布。",
        "圆心来自骨架共心拟合，物理头混合为 0。环半径来自单帧径向定环，与 /v2 点「确认拟合」相同。",
        "对照是人工弧。成功条件沿用 0.4 倍间距、0.5 倍间距圆心、漏弧不超过 0.3。",
        "曲率半径使用软件里的 7.189 微米/px，只作附录，不作为这张表的主结果。",
        "可评：这 43 张都有至少 3 条人工弧。标注文件里的可评字段除 0001 外仍是空的，这里按「已标上环即可评」计入分母。",
        "",
    ]
    by_subset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_subset.setdefault(row["subset"], []).append(row)
    lines.append("子集汇总")
    for subset in ("Easy", "Open-arc", "Illegible"):
        group = by_subset.get(subset, [])
        n = len(group)
        ok = sum(1 for row in group if row["success"])
        refused = sum(1 for row in group if row["refused"])
        rate = 0.0 if n == 0 else 100.0 * ok / n
        lines.append(f"{subset}  {ok}/{n}  成功率 {rate:.1f}%  拒测 {refused}")
    n = len(rows)
    ok = sum(1 for row in rows if row["success"])
    refused = sum(1 for row in rows if row["refused"])
    lines.append(f"合计  {ok}/{n}  成功率 {0.0 if n == 0 else 100.0 * ok / n:.1f}%  拒测 {refused}")
    lines.append("")
    lines.append("逐张")
    lines.append("文件\t子集\t结果\t人工\t算法\t配对\t漏\t假环\t顺序\t圆心\t秒\tR_m")
    for row in rows:
        if row["success"]:
            mark = "成功"
        elif row["refused"]:
            mark = "拒测"
        else:
            mark = "未成功"
        lines.append(
            "\t".join(
                [
                    row["stem"],
                    row["subset"],
                    mark,
                    str(row["n_human"]),
                    str(row["n_pred"]),
                    str(row["matches"]),
                    str(row["misses"]),
                    str(row["false_rings"]),
                    "是" if row["order_ok"] else "否",
                    "是" if row["center_ok"] else "否",
                    f"{row['seconds']:.1f}",
                    "" if row["R_m"] is None else str(row["R_m"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
