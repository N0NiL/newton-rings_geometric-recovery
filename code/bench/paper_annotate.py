"""
离线人工对照。不打开摄像头，也不读取算法输出。

沿可见暗弧点至少 3 个点。所有牛顿弧共用一个拟合圆心，每条弧保留自己的半径。
圆心可以落在画面外。眩光弧单独拟合，不跟牛顿弧锁在一起。
"""

from __future__ import annotations

import copy
import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SESSION_ROOT = Path(__file__).resolve().parent / "sessions"
_BAR_H = 136
_PAD = 480
_SCHEMA = "paper_human_arc_v1"


def _cjk_font(size: int):
    from PIL import ImageFont

    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _paint_lines(bgr: np.ndarray, lines: list[str], origin: tuple[int, int]) -> np.ndarray:
    from PIL import Image, ImageDraw

    font = _cjk_font(18)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        y += 24
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _work_area() -> tuple[int, int]:
    class _RECT(ctypes.Structure):
        _fields_ = (
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        )

    rect = _RECT()
    ok = ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
    width = int(rect.right - rect.left)
    height = int(rect.bottom - rect.top)
    if not ok or width < 640 or height < 480:
        return 1600, 900
    return width, height


def off_side(cx: float, cy: float, width: int, height: int) -> str | None:
    """圆心相对画面的主方向。在画面内则返回 None。"""
    inside_x = 0.0 <= cx <= width - 1
    inside_y = 0.0 <= cy <= height - 1
    if inside_x and inside_y:
        return None
    dx = 0.0 if inside_x else (cx if cx < 0 else cx - (width - 1))
    dy = 0.0 if inside_y else (cy if cy < 0 else cy - (height - 1))
    if abs(dx) >= abs(dy):
        return "left" if cx < 0 else "right"
    return "top" if cy < 0 else "bottom"


def _wheel_steps(flags: int) -> int:
    """旧版 OpenCV 没有 getMouseWheelDelta，滚轮量在 flags 高 16 位。"""
    getter = getattr(cv2, "getMouseWheelDelta", None)
    if getter is not None:
        delta = int(getter(flags))
    else:
        delta = (int(flags) >> 16) & 0xFFFF
        if delta >= 0x8000:
            delta -= 0x10000
    if delta == 0:
        return 0
    return 1 if delta > 0 else -1


def fit_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """代数圆拟合。返回圆心和半径；点太少、太直或半径失控时返回 None。"""
    pts = np.asarray(points, np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None
    x = pts[:, 0]
    y = pts[:, 1]
    matrix = np.column_stack([x, y, np.ones(len(pts))])
    target = -(x * x + y * y)
    solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
    if int(rank) < 3:
        return None
    coeff_d, coeff_e, coeff_f = (float(value) for value in solution)
    cx = -0.5 * coeff_d
    cy = -0.5 * coeff_e
    radius2 = cx * cx + cy * cy - coeff_f
    if not np.isfinite(radius2) or radius2 < 1.0:
        return None
    radius = float(np.sqrt(radius2))
    if radius > 20000.0 or not np.isfinite(cx) or not np.isfinite(cy):
        return None
    return cx, cy, radius


def fit_concentric(groups: list[np.ndarray]) -> tuple[float, float, list[float]] | None:
    """多条弧共用一个圆心。先做代数初值，再按到圆心的距离做几何修正。"""
    usable = [np.asarray(group, np.float64) for group in groups if len(group) >= 1]
    total = sum(len(group) for group in usable)
    if len(usable) == 0 or total < 3:
        return None
    rows: list[np.ndarray] = []
    targets: list[float] = []
    for arc_index, points in enumerate(usable):
        for x_coord, y_coord in points:
            row = np.zeros(2 + len(usable), np.float64)
            row[0] = 2.0 * float(x_coord)
            row[1] = 2.0 * float(y_coord)
            row[2 + arc_index] = 1.0
            rows.append(row)
            targets.append(float(x_coord) * float(x_coord) + float(y_coord) * float(y_coord))
    solution, _, rank, _ = np.linalg.lstsq(np.vstack(rows), np.asarray(targets, np.float64), rcond=None)
    if int(rank) < 3:
        return None
    center = np.array([float(solution[0]), float(solution[1])], np.float64)
    if not np.all(np.isfinite(center)):
        return None
    for _ in range(8):
        jacobian: list[np.ndarray] = []
        residual: list[np.ndarray] = []
        for points in usable:
            delta = points - center
            distance = np.hypot(delta[:, 0], delta[:, 1])
            usable_row = distance > 1e-3
            if not np.any(usable_row):
                continue
            radius = float(np.median(distance[usable_row]))
            unit = delta[usable_row] / distance[usable_row, None]
            jacobian.append(-unit)
            residual.append(distance[usable_row] - radius)
        if not jacobian:
            break
        step, _, _, _ = np.linalg.lstsq(np.vstack(jacobian), -np.concatenate(residual), rcond=None)
        center = center + step
        if float(np.hypot(step[0], step[1])) < 0.02:
            break
    if not np.all(np.isfinite(center)) or abs(float(center[0])) > 30000 or abs(float(center[1])) > 30000:
        return None
    radii: list[float] = []
    for points in usable:
        distance = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1])
        radius = float(np.median(distance))
        if not np.isfinite(radius) or radius < 1.0 or radius > 20000.0:
            return None
        radii.append(radius)
    return float(center[0]), float(center[1]), radii


def consensus_center(rings: list[dict[str, Any]]) -> list[float] | None:
    """牛顿弧的共同拟合圆心。只作汇总，不要求人去点。"""
    xs: list[float] = []
    ys: list[float] = []
    for ring in rings:
        if ring.get("role") != "newton":
            continue
        center = ring.get("center_xy")
        if not center:
            continue
        xs.append(float(center[0]))
        ys.append(float(center[1]))
    if not xs:
        return None
    return [round(float(np.median(xs)), 2), round(float(np.median(ys)), 2)]


def _stroke_circle(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """只画落在图内的圆弧，避免大半径采样点溢出。"""
    height, width = image.shape[:2]
    steps = int(min(4000, max(180, radius * 2.0)))
    run: list[list[int]] = []

    def flush() -> None:
        if len(run) >= 2:
            cv2.polylines(image, [np.asarray(run, np.int32)], False, color, thickness, cv2.LINE_AA)
        run.clear()

    for index in range(steps + 1):
        theta = 2.0 * np.pi * index / steps
        x = cx + radius * np.cos(theta)
        y = cy + radius * np.sin(theta)
        if -2.0 <= x <= width + 1 and -2.0 <= y <= height + 1:
            run.append([int(round(x)), int(round(y))])
        else:
            flush()
    flush()


def _latest_session() -> Path | None:
    if not _SESSION_ROOT.is_dir():
        return None
    dirs = [path for path in _SESSION_ROOT.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


def _canvas_list(folder: Path) -> list[Path]:
    return sorted(folder.glob("*_canvas.png"))


class ArcAnnotator:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.paths = _canvas_list(folder)
        if not self.paths:
            raise SystemExit(f"这个目录里没有 *_canvas.png：{folder}")
        self.index = 0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._need_fit = True
        self.rings: list[dict[str, Any]] = []
        self.active = -1
        self.draft: list[list[float]] = []
        self.draft_role = "newton"
        self.evaluable: bool | None = None
        self.subset = "未知"
        self.review: str | None = None
        self.frame_index: int | None = None
        self.sample_id: str | None = None
        self.dirty = False
        self.discard_armed = False
        self._undo: list[dict[str, Any]] = []
        self._drag: tuple[int, int] | None = None
        self._middle: tuple[int, int] | None = None
        self._point_drag: tuple | None = None
        self._pending_add: tuple[int, int] | None = None
        self._right_arm: tuple | None = None
        self._selected: tuple | None = None
        self.view_h = 780
        self.status = "沿暗纹左键点至少 3 下，Enter 结束这条。不用点圆心。"
        self._gray: np.ndarray | None = None
        self._padded: np.ndarray | None = None
        work_w, work_h = _work_area()
        self._client = (max(960, work_w - 80), max(640, work_h - 80))
        self._load_image()

    def _capture_subset(self, canvas_path: Path) -> str:
        record = canvas_path.with_name(canvas_path.name.replace("_canvas.png", ".json"))
        if not record.is_file():
            return "未知"
        try:
            payload = json.loads(record.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "未知"
        return str(payload.get("subset") or "未知")

    def _human_path(self, canvas_path: Path) -> Path:
        return canvas_path.with_name(canvas_path.name.replace("_canvas.png", "_human.json"))

    def _push_undo(self) -> None:
        self._undo.append(
            {
                "rings": copy.deepcopy(self.rings),
                "active": self.active,
                "draft": copy.deepcopy(self.draft),
                "draft_role": self.draft_role,
                "evaluable": self.evaluable,
                "subset": self.subset,
            }
        )
        if len(self._undo) > 100:
            del self._undo[0]

    def _undo_mark(self) -> None:
        if not self._undo:
            self.status = "没有可撤回的标注"
            return
        snap = self._undo.pop()
        self.rings = snap["rings"]
        self.active = snap["active"]
        self.draft = snap["draft"]
        self.draft_role = snap["draft_role"]
        self.evaluable = snap["evaluable"]
        self.subset = snap.get("subset", self.subset)
        self.dirty = True
        self._point_drag = None
        self._middle = None
        self._drag = None
        self._pending_add = None
        self._right_arm = None
        self._selected = None
        self.status = "已撤回上一步"

    def _load_image(self) -> None:
        path = self.paths[self.index]
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"读不到分析画布：{path}")
        self._gray = gray
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        padded = np.full((gray.shape[0] + 2 * _PAD, gray.shape[1] + 2 * _PAD, 3), 32, np.uint8)
        padded[_PAD : _PAD + gray.shape[0], _PAD : _PAD + gray.shape[1]] = bgr
        cv2.rectangle(
            padded,
            (_PAD, _PAD),
            (_PAD + gray.shape[1] - 1, _PAD + gray.shape[0] - 1),
            (180, 180, 180),
            1,
        )
        self._padded = padded
        self._need_fit = True
        self._load_human()

    def _fit_view(self, view_w: int, view_h: int) -> None:
        assert self._gray is not None
        span_w = self._gray.shape[1] + 80
        span_h = self._gray.shape[0] + 80
        self.zoom = max(0.2, min(view_w / float(span_w), view_h / float(span_h)))
        src_x = _PAD + self._gray.shape[1] / 2.0
        src_y = _PAD + self._gray.shape[0] / 2.0
        self.pan_x = src_x * self.zoom - view_w / 2.0
        self.pan_y = src_y * self.zoom - view_h / 2.0
        self._need_fit = False

    def _ring_from_payload(self, item: dict[str, Any]) -> dict[str, Any] | None:
        points = [[float(pt[0]), float(pt[1])] for pt in item.get("points_xy") or []]
        fitted = fit_circle(np.asarray(points, np.float64)) if len(points) >= 3 else None
        if fitted is None and item.get("center_xy") and item.get("radius_px"):
            center = [float(item["center_xy"][0]), float(item["center_xy"][1])]
            radius = float(item["radius_px"])
        elif fitted is not None:
            center = [round(fitted[0], 2), round(fitted[1], 2)]
            radius = round(fitted[2], 2)
        else:
            return None
        ring = {
            "role": str(item.get("role") or "newton"),
            "points_xy": points,
            "center_xy": center,
            "radius_px": radius,
        }
        if item.get("half_width_px") is not None:
            ring["half_width_px"] = float(item["half_width_px"])
        if item.get("source_radius_px") is not None:
            ring["source_radius_px"] = float(item["source_radius_px"])
        return ring

    def _load_human(self) -> None:
        path = self._human_path(self.paths[self.index])
        self.rings = []
        self.active = -1
        self.draft = []
        self.draft_role = "newton"
        self.evaluable = None
        self.subset = self._capture_subset(self.paths[self.index])
        self.review = None
        self.frame_index = None
        self.sample_id = None
        self._undo = []
        self.dirty = False
        self.discard_armed = False
        self._point_drag = None
        self._middle = None
        self._drag = None
        self._pending_add = None
        self._right_arm = None
        self._selected = None
        if not path.is_file():
            self.status = "尚未标注。沿暗纹左键点至少 3 下，Enter 结束这条。不用点圆心。"
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("rings") or []:
            ring = self._ring_from_payload(item)
            if ring is not None:
                self.rings.append(ring)
        self.active = len(self.rings) - 1
        self.evaluable = payload.get("evaluable")
        self.review = payload.get("review")
        if payload.get("subset"):
            self.subset = str(payload["subset"])
        frame = payload.get("frame_index")
        self.frame_index = None if frame is None else int(frame)
        self.sample_id = None if payload.get("sample_id") is None else str(payload["sample_id"])
        self.status = f"已载入 {path.name}"

    def _source_of(self, x: int, y: int) -> tuple[float, float]:
        return (x + self.pan_x) / self.zoom, (y + self.pan_y) / self.zoom

    def _canvas_of(self, x: int, y: int) -> tuple[float, float]:
        src_x, src_y = self._source_of(x, y)
        return src_x - _PAD, src_y - _PAD

    def _zoom_at(self, x: int, y: int, steps: int) -> None:
        if steps == 0:
            return
        src_x, src_y = self._source_of(x, y)
        factor = 1.25 if steps > 0 else 0.8
        self.zoom = float(np.clip(self.zoom * factor, 0.15, 16.0))
        self.pan_x = src_x * self.zoom - x
        self.pan_y = src_y * self.zoom - y

    def _point_refs(self) -> list[tuple[tuple, list[float]]]:
        found: list[tuple[tuple, list[float]]] = []
        for index in range(len(self.draft) - 1, -1, -1):
            found.append((("draft", index), self.draft[index]))
        order: list[int] = []
        if 0 <= self.active < len(self.rings):
            order.append(self.active)
        order.extend(index for index in range(len(self.rings) - 1, -1, -1) if index != self.active)
        for ring_index in order:
            points = self.rings[ring_index]["points_xy"]
            for point_index in range(len(points) - 1, -1, -1):
                found.append((("ring", ring_index, point_index), points[point_index]))
        return found

    def _hit_point(self, x: int, y: int) -> tuple | None:
        best: tuple | None = None
        best_dist = 12.0
        for ref, point in self._point_refs():
            sx, sy = self._screen_of(float(point[0]), float(point[1]))
            dist = float(np.hypot(sx - x, sy - y))
            if dist <= best_dist:
                best = ref
                best_dist = dist
        return best

    def _point_at(self, ref: tuple) -> list[float]:
        if ref[0] == "draft":
            return self.draft[ref[1]]
        return self.rings[ref[1]]["points_xy"][ref[2]]

    def _refit_ring(self, ring: dict[str, Any]) -> bool:
        fitted = fit_circle(np.asarray(ring["points_xy"], np.float64))
        if fitted is None:
            ring["center_xy"] = None
            ring["radius_px"] = None
            return False
        ring["center_xy"] = [round(fitted[0], 2), round(fitted[1], 2)]
        ring["radius_px"] = round(fitted[2], 2)
        return True

    def _move_point(self, ref: tuple, cx: float, cy: float) -> None:
        point = self._point_at(ref)
        point[0] = round(cx, 2)
        point[1] = round(cy, 2)
        if ref[0] == "ring":
            self._refit_ring(self.rings[ref[1]])
            self.active = ref[1]
        self.dirty = True
        self.discard_armed = False

    def _delete_point(self, ref: tuple) -> None:
        self._push_undo()
        if ref[0] == "draft":
            del self.draft[ref[1]]
            self.status = f"已撤掉当前弧上的一个点，还剩 {len(self.draft)} 个。"
        else:
            ring = self.rings[ref[1]]
            del ring["points_xy"][ref[2]]
            if not ring["points_xy"]:
                del self.rings[ref[1]]
                self.active = len(self.rings) - 1
                self.status = "这条弧上的点已全部撤掉。"
            elif self._refit_ring(ring):
                self.active = ref[1]
                self.status = f"已撤掉一个点。当前半径 {float(ring['radius_px']):.2f}。"
            else:
                self.active = ref[1]
                self.status = "已撤掉一个点。剩下的点还不够拟合圆。"
        self._selected = None
        self.dirty = True
        self.discard_armed = False

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: Any) -> None:
        try:
            if event == cv2.EVENT_MBUTTONUP:
                self._middle = None
                return
            if event == cv2.EVENT_MOUSEMOVE and self._middle is not None:
                last_x, last_y = self._middle
                self.pan_x -= x - last_x
                self.pan_y -= y - last_y
                self._middle = (x, y)
                return
            if event == cv2.EVENT_LBUTTONUP:
                if self._point_drag is not None:
                    ref = self._point_drag
                    self._point_drag = None
                    self._selected = ref
                    if ref[0] == "draft":
                        fitted = fit_circle(np.asarray(self.draft, np.float64))
                        self.status = "点已拖到新位置。" if fitted is None else f"点已拖动。当前半径 {fitted[2]:.2f}。"
                    else:
                        radius = self.rings[ref[1]].get("radius_px")
                        self.status = "点已拖动。剩下的点还不够拟合圆。" if radius is None else f"点已拖动。当前半径 {float(radius):.2f}。"
                    return
                if self._pending_add is not None:
                    origin_x, origin_y = self._pending_add
                    self._pending_add = None
                    if y < self.view_h and float(np.hypot(x - origin_x, y - origin_y)) <= 5.0:
                        self._add_point(*self._canvas_of(origin_x, origin_y))
                return
            if event == cv2.EVENT_RBUTTONUP:
                if self._right_arm is not None:
                    _origin_x, _origin_y, hit = self._right_arm
                    self._right_arm = None
                    self._delete_point(hit)
                    return
                self._drag = None
                return
            if event == cv2.EVENT_MBUTTONDOWN and y < self.view_h:
                self._middle = (x, y)
                self._drag = None
                self._right_arm = None
                return
            if y >= self.view_h:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                hit = self._hit_point(x, y)
                if hit is not None:
                    self._push_undo()
                    self._point_drag = hit
                    self._selected = hit
                    self._pending_add = None
                    self.status = "正在拖动这个点。"
                else:
                    self._pending_add = (x, y)
                return
            if event == cv2.EVENT_MOUSEMOVE and self._point_drag is not None:
                self._move_point(self._point_drag, *self._canvas_of(x, y))
                return
            if event == cv2.EVENT_RBUTTONDOWN:
                hit = self._hit_point(x, y)
                if hit is not None:
                    self._right_arm = (x, y, hit)
                    self._drag = None
                else:
                    self._right_arm = None
                    self._drag = (x, y)
                return
            if event == cv2.EVENT_MOUSEMOVE and self._right_arm is not None:
                origin_x, origin_y, _hit = self._right_arm
                if float(np.hypot(x - origin_x, y - origin_y)) > 5.0:
                    self._right_arm = None
                    self._drag = (x, y)
                return
            if event == cv2.EVENT_MOUSEMOVE and self._drag is not None:
                last_x, last_y = self._drag
                self.pan_x -= x - last_x
                self.pan_y -= y - last_y
                self._drag = (x, y)
                return
            if event == cv2.EVENT_MOUSEWHEEL:
                self._zoom_at(x, y, _wheel_steps(flags))
        except Exception as exc:
            self.status = f"鼠标事件失败：{exc}"

    def _add_point(self, cx: float, cy: float) -> None:
        assert self._gray is not None
        height, width = self._gray.shape[:2]
        if cx < -20 or cy < -20 or cx > width + 20 or cy > height + 20:
            self.status = "请点在可见暗纹上"
            return
        if self.draft:
            last_x, last_y = self.draft[-1]
            if float(np.hypot(cx - last_x, cy - last_y)) < 1.0:
                self.status = "这个点和上一个太近"
                return
        self._push_undo()
        self.draft.append([round(cx, 2), round(cy, 2)])
        self._selected = ("draft", len(self.draft) - 1)
        self.dirty = True
        self.discard_armed = False
        solved = self._shared_newton().get(("draft", -1))
        if solved is not None:
            self.status = f"已点 {len(self.draft)} 个，当前半径 {solved[2]:.2f}。这条和已有牛顿弧共用圆心。"
            return
        if len(self.draft) < 3:
            self.status = f"已点 {len(self.draft)} 个。沿暗纹继续点，两端尽量拉开。"
            return
        self.status = f"已点 {len(self.draft)} 个。点够后按 Enter。再加几条弧后，圆心会合成一个。"

    def _commit_draft(self) -> None:
        if len(self.draft) < 3:
            self.status = "这条还不到 3 个点"
            return
        fitted = fit_circle(np.asarray(self.draft, np.float64))
        if fitted is None:
            self.status = "这些点太直或太挤，圆不稳定。请沿暗纹把两端再点开一些。"
            return
        self._push_undo()
        cx, cy, radius = fitted
        added = {
            "role": self.draft_role,
            "points_xy": copy.deepcopy(self.draft),
            "center_xy": [round(cx, 2), round(cy, 2)],
            "radius_px": round(radius, 2),
        }
        if self.review in ("pending", "edited") and self.draft_role == "newton":
            added["half_width_px"] = 4.0
            added["source_radius_px"] = round(radius, 2)
        self.rings.append(added)
        self.active = len(self.rings) - 1
        label = "眩光弧" if self.draft_role == "distractor" else "牛顿弧"
        self.draft = []
        self._selected = None
        self.dirty = True
        self.discard_armed = False
        solved = self._shared_newton().get(("ring", self.active))
        if solved is not None and self.draft_role == "newton":
            radius = solved[2]
        self.status = f"已记下一条{label}，半径 {radius:.2f}。牛顿弧共用一个圆心。"

    def _nudge_last_point(self, dx: float, dy: float) -> None:
        ref = self._selected
        if ref is None and self.draft:
            ref = ("draft", len(self.draft) - 1)
        if ref is None:
            self.status = "先点中一个点，再用方向键微调。"
            return
        try:
            point = self._point_at(ref)
        except IndexError:
            self._selected = None
            self.status = "选中的点已经不在了。"
            return
        self._push_undo()
        self._selected = ref
        self._move_point(ref, point[0] + dx, point[1] + dy)
        moved = self._point_at(ref)
        self.status = f"点已挪到 {moved[0]:.2f}, {moved[1]:.2f}"

    def _shared_newton(self) -> dict[tuple, tuple[float, float, float]]:
        """牛顿弧（含正在点的这一条）共用一个圆心。键是 ("ring", 序号) 或 ("draft", -1)。"""
        groups: list[np.ndarray] = []
        keys: list[tuple] = []
        for index, ring in enumerate(self.rings):
            if ring.get("role") != "newton" or len(ring.get("points_xy") or []) < 1:
                continue
            groups.append(np.asarray(ring["points_xy"], np.float64))
            keys.append(("ring", index))
        if self.draft_role == "newton" and self.draft:
            groups.append(np.asarray(self.draft, np.float64))
            keys.append(("draft", -1))
        fitted = fit_concentric(groups)
        if fitted is None:
            return {}
        cx, cy, radii = fitted
        return {key: (cx, cy, radii[index]) for index, key in enumerate(keys)}

    def _screen_of(self, cx: float, cy: float) -> tuple[float, float]:
        return (_PAD + cx) * self.zoom - self.pan_x, (_PAD + cy) * self.zoom - self.pan_y

    def _draw_fit(self, view: np.ndarray, cx: float, cy: float, radius: float, color: tuple[int, int, int], thick: int) -> None:
        height, width = view.shape[:2]
        screen_r = max(radius * self.zoom, 1.0)
        steps = int(min(2000, max(96, screen_r * 1.5)))
        run: list[list[int]] = []

        def flush() -> None:
            if len(run) >= 2:
                cv2.polylines(view, [np.asarray(run, np.int32)], False, color, thick, cv2.LINE_AA)
            run.clear()

        for index in range(steps + 1):
            theta = 2.0 * np.pi * index / steps
            sx, sy = self._screen_of(cx + radius * np.cos(theta), cy + radius * np.sin(theta))
            if -80.0 <= sx <= width + 80 and -80.0 <= sy <= height + 80:
                run.append([int(round(sx)), int(round(sy))])
                if len(run) >= 2:
                    jump = abs(run[-1][0] - run[-2][0]) + abs(run[-1][1] - run[-2][1])
                    if jump > 80:
                        tail = run.pop()
                        flush()
                        run.append(tail)
            else:
                flush()
        flush()

    def _draw_points(self, view: np.ndarray, points: list[list[float]], color: tuple[int, int, int]) -> None:
        for px, py in points:
            sx, sy = self._screen_of(px, py)
            if 0 <= sx < view.shape[1] and 0 <= sy < view.shape[0]:
                cv2.circle(view, (int(round(sx)), int(round(sy))), 3, color, -1, cv2.LINE_AA)

    def _render(self, win_w: int, win_h: int) -> np.ndarray:
        assert self._padded is not None and self._gray is not None
        win_w = max(640, int(win_w))
        win_h = max(480, int(win_h))
        view_h = max(200, win_h - _BAR_H)
        self.view_h = view_h
        if self._need_fit:
            self._fit_view(win_w, view_h)
        matrix = np.array([[self.zoom, 0.0, -self.pan_x], [0.0, self.zoom, -self.pan_y]], np.float32)
        interp = cv2.INTER_NEAREST if self.zoom >= 1.0 else cv2.INTER_LINEAR
        view = cv2.warpAffine(self._padded, matrix, (win_w, view_h), flags=interp)
        shared = self._shared_newton()
        for i, ring in enumerate(self.rings):
            active = i == self.active and not self.draft
            if ring["role"] == "distractor":
                color = (0, 0, 255)
                center = ring.get("center_xy")
                radius = ring.get("radius_px")
            else:
                color = (0, 255, 255) if active else (0, 220, 0)
                solved = shared.get(("ring", i))
                center = None if solved is None else [solved[0], solved[1]]
                radius = None if solved is None else solved[2]
            if center is not None and radius is not None:
                self._draw_fit(view, float(center[0]), float(center[1]), float(radius), color, 2 if active else 1)
                half = ring.get("half_width_px")
                if half:
                    edge = (0, 140, 255) if active else (40, 90, 140)
                    self._draw_fit(view, float(center[0]), float(center[1]), max(1.0, float(radius) - float(half)), edge, 1)
                    self._draw_fit(view, float(center[0]), float(center[1]), float(radius) + float(half), edge, 1)
            self._draw_points(view, ring["points_xy"], color)
        draft_solved = shared.get(("draft", -1))
        if draft_solved is not None:
            self._draw_fit(view, draft_solved[0], draft_solved[1], draft_solved[2], (0, 255, 255), 2)
        elif len(self.draft) >= 3:
            fitted = fit_circle(np.asarray(self.draft, np.float64))
            if fitted is not None:
                self._draw_fit(view, fitted[0], fitted[1], fitted[2], (0, 255, 255), 2)
        if self.draft:
            self._draw_points(view, self.draft, (255, 255, 255))
        if self._selected is not None:
            try:
                chosen = self._point_at(self._selected)
            except IndexError:
                chosen = None
            if chosen is not None:
                sx, sy = self._screen_of(float(chosen[0]), float(chosen[1]))
                if 0 <= sx < view.shape[1] and 0 <= sy < view.shape[0]:
                    cv2.circle(view, (int(round(sx)), int(round(sy))), 7, (255, 255, 255), 1, cv2.LINE_AA)
        bar = np.full((_BAR_H, win_w, 3), 24, np.uint8)
        frame = np.vstack([view, bar])
        path = self.paths[self.index]
        n_newton = sum(1 for ring in self.rings if ring["role"] == "newton")
        evaluable = {None: "未标", True: "可评", False: "不可评"}[self.evaluable]
        radius = "—"
        if draft_solved is not None:
            radius = f"{draft_solved[2]:.2f}"
        elif ("ring", self.active) in shared:
            radius = f"{shared[('ring', self.active)][2]:.2f}"
        shared_center = next(iter(shared.values()), None)
        side = "—"
        if shared_center is not None:
            side = off_side(shared_center[0], shared_center[1], self._gray.shape[1], self._gray.shape[0]) or "画内"
        role_name = "眩光" if self.draft_role == "distractor" else "牛顿"
        half = "—"
        if 0 <= self.active < len(self.rings) and self.rings[self.active].get("half_width_px") is not None:
            half = f"{float(self.rings[self.active]['half_width_px']):.1f}"
        help_edit = "左键在空白处加点，左键拖动已有的点。右键单击某个点可撤掉它。Enter 结束这条。"
        help_keys = "中键拖动画面。滚轮或 + - 缩放，0 整图入窗。Del 或 X 删当前弧。Z 撤回。E 可评，S 保存，N/B 换张，Q 退出。"
        if self.review in ("pending", "edited"):
            help_keys = "1 Easy，2 Open-arc，3 难辨。Tab 换弧，Del 或 X 删当前弧。[ ] 改半宽。S 保存。"
        lines = [
            f"{self.index + 1}/{len(self.paths)}  {path.name}  类别={self.subset}  半宽={half}  可评={evaluable}  未保存={'是' if self.dirty else '否'}",
            f"正在点={role_name}  本条点数={len(self.draft)}  牛顿弧={n_newton}  眩光弧={len(self.rings) - n_newton}  半径={radius}  拟合圆心={side}  缩放={self.zoom:.2f}x",
            help_edit,
            help_keys,
            self.status,
        ]
        return _paint_lines(frame, lines, (8, view_h + 6))

    def _save(self) -> None:
        assert self._gray is not None
        if self.draft:
            self.status = "当前这条还没按 Enter 结束"
            return
        shared = self._shared_newton()
        newton_indexes = [index for index, ring in enumerate(self.rings) if ring.get("role") == "newton"]
        if newton_indexes and any(("ring", index) not in shared for index in newton_indexes):
            self.status = "牛顿弧还不能合成同一个圆心。每条至少留 3 个点，并沿弧拉开。"
            return
        if any(ring.get("role") == "distractor" and ring.get("radius_px") is None for ring in self.rings):
            self.status = "有一条眩光弧不足 3 个点，补点或撤掉后再保存。"
            return
        path = self.paths[self.index]
        height, width = self._gray.shape[:2]
        for index, ring in enumerate(self.rings):
            solved = shared.get(("ring", index))
            if solved is None:
                continue
            ring["center_xy"] = [round(solved[0], 2), round(solved[1], 2)]
            ring["radius_px"] = round(solved[2], 2)
        newton = [ring for ring in self.rings if ring["role"] == "newton"]
        center = None if not shared else [round(next(iter(shared.values()))[0], 2), round(next(iter(shared.values()))[1], 2)]
        side = None if center is None else off_side(center[0], center[1], width, height)
        rings = sorted(self.rings, key=lambda item: (item["role"] != "newton", float(item["radius_px"])))
        stem = self._target_stem()
        canvas_name = f"{stem}_canvas.png"
        if self.review in ("pending", "edited"):
            self.review = "edited"
        payload = {
            "schema": _SCHEMA,
            "canvas_png": canvas_name,
            "subset": self.subset,
            "coordinate": "analysis_canvas",
            "method": "shared_center_arc_fit",
            "method_note": "牛顿弧上的点共同拟合一个圆心，每条弧的半径是这些点到该圆心距离的中位数。不手工点圆心。眩光弧单独拟合。",
            "center_xy": center,
            "center_in_frame": None if center is None else side is None,
            "off_side": side,
            "center_note": "center_xy 是全部牛顿弧共用的拟合圆心。",
            "evaluable": self.evaluable,
            "rings": [
                {
                    "radius_px": round(float(ring["radius_px"]), 2),
                    "role": ring["role"],
                    "center_xy": [round(float(ring["center_xy"][0]), 2), round(float(ring["center_xy"][1]), 2)],
                    "points_xy": [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in ring["points_xy"]],
                    **(
                        {
                            "half_width_px": round(float(ring["half_width_px"]), 2),
                            "source_radius_px": round(float(ring.get("source_radius_px", ring["radius_px"])), 2),
                        }
                        if ring.get("half_width_px") is not None
                        else {}
                    ),
                }
                for ring in rings
            ],
            "n_newton": len(newton),
        }
        if self.review == "edited":
            payload["review"] = "edited"
            payload["source"] = "classical_v13_draft"
            if self.frame_index is not None:
                payload["frame_index"] = self.frame_index
            if self.sample_id:
                payload["sample_id"] = self.sample_id
        out = path.with_name(f"{stem}_human.json")
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.review == "edited":
            self._write_band(stem, center, side, rings)
            path.with_name(f"{stem}.json").write_text(
                json.dumps(
                    {"subset": self.subset, "review": "edited", "frame_index": self.frame_index},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        preview = cv2.cvtColor(self._gray, cv2.COLOR_GRAY2BGR)
        for ring in payload["rings"]:
            color = (0, 220, 0) if ring["role"] == "newton" else (0, 0, 255)
            _stroke_circle(preview, float(ring["center_xy"][0]), float(ring["center_xy"][1]), float(ring["radius_px"]), color, 1)
            for pt in ring["points_xy"]:
                cv2.circle(preview, (int(round(pt[0])), int(round(pt[1]))), 2, color, -1, cv2.LINE_AA)
        cv2.imwrite(str(path.with_name(f"{stem}_human_preview.png")), preview)
        self._retire_old_names(stem)
        self.paths[self.index] = path.with_name(canvas_name)
        self.dirty = False
        self.discard_armed = False
        self.rings = rings
        self.active = len(self.rings) - 1
        self.status = f"已保存 {out.name}"

    def _target_stem(self) -> str:
        stem = self.paths[self.index].name.replace("_canvas.png", "")
        if self.review not in ("pending", "edited"):
            return stem
        label = {"Easy": "Easy", "Open-arc": "Open-arc", "Illegible": "Illegible", "Broken": "Illegible"}.get(self.subset, "Unset")
        return f"{stem.rsplit('_', 1)[0]}_{label}"

    def _write_band(self, stem: str, center: list[float] | None, side: str | None, rings: list[dict[str, Any]]) -> None:
        assert self._gray is not None
        if center is None:
            return
        payload = {
            "schema": "paper_human_band_v1",
            "stem": stem,
            "review": "edited",
            "half_init_px": 4.0,
            "center_xy": center,
            "center_in_frame": side is None,
            "off_side": side,
            "note": "半宽按分析画布上暗纹的半高宽起稿，供人工再改。不替代 *_human.json。",
            "rings": [
                {
                    "radius_px": round(float(ring["radius_px"]), 2),
                    "half_width_px": round(float(ring.get("half_width_px") or 4.0), 2),
                    "source_radius_px": round(float(ring.get("source_radius_px", ring["radius_px"])), 2),
                    "role": "newton",
                }
                for ring in rings
                if ring.get("role") == "newton"
            ],
        }
        self.paths[self.index].with_name(f"{stem}_band.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _retire_old_names(self, stem: str) -> None:
        old = self.paths[self.index].name.replace("_canvas.png", "")
        if old == stem:
            return
        folder = self.paths[self.index].parent
        canvas_src = folder / f"{old}_canvas.png"
        canvas_dst = folder / f"{stem}_canvas.png"
        if canvas_src.is_file() and canvas_src != canvas_dst:
            canvas_src.replace(canvas_dst)
        for suffix in ("_human.json", "_band.json", "_human_preview.png", ".json"):
            src = folder / f"{old}{suffix}"
            if src.is_file():
                src.unlink()

    def _set_subset(self, name: str) -> None:
        if self.review not in ("pending", "edited"):
            self.status = "这张不是待审草稿，类别键不起作用。"
            return
        self._push_undo()
        self.subset = name
        self.dirty = True
        self.status = f"类别改为 {name}。按 S 保存。"

    def _nudge_half(self, delta: float) -> None:
        if self.review not in ("pending", "edited"):
            return
        if not (0 <= self.active < len(self.rings)):
            self.status = "先用 Tab 选中一条弧"
            return
        ring = self.rings[self.active]
        if ring.get("role") != "newton":
            self.status = "眩光弧没有半宽"
            return
        self._push_undo()
        current = float(ring.get("half_width_px") or 4.0)
        ring["half_width_px"] = round(float(np.clip(current + delta, 1.0, 40.0)), 2)
        self.dirty = True
        self.status = f"当前弧半宽 {ring['half_width_px']:.1f}"

    def _delete_active_ring(self) -> None:
        if self.draft:
            self.status = "先按 Enter 结束正在点的这条，再删已有的弧。"
            return
        if not (0 <= self.active < len(self.rings)):
            self.status = "没有选中的弧"
            return
        self._push_undo()
        del self.rings[self.active]
        self.active = len(self.rings) - 1
        self.dirty = True
        self.status = "已删掉当前弧。Z 可撤回。"

    def _switch(self, step: int) -> None:
        if self.dirty:
            if not self.discard_armed:
                self.discard_armed = True
                self.status = "这张还没保存。按 S 保存，或再按一次 N/B 放弃修改。"
                return
        nxt = self.index + step
        if nxt < 0 or nxt >= len(self.paths):
            self.status = "已经是第一张或最后一张"
            self.discard_armed = False
            return
        self.index = nxt
        self._load_image()

    def loop(self) -> None:
        window = "PaperAnnotate"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, self._client[0], self._client[1])
        cv2.setMouseCallback(window, self._on_mouse)
        print(f"标注目录：{self.folder}")
        print(f"共 {len(self.paths)} 张分析画布。不连接摄像头。")
        print("沿暗纹点点，Enter 结束一条。窗口可拉大小，滚轮或 +- 缩放。")
        arrows = {
            2424832: (-1.0, 0.0),
            2555904: (1.0, 0.0),
            2490368: (0.0, -1.0),
            2621440: (0.0, 1.0),
        }
        try:
            while True:
                frame = self._render(*self._client)
                cv2.imshow(window, frame)
                key = cv2.waitKeyEx(20)
                _x, _y, win_w, win_h = cv2.getWindowImageRect(window)
                if win_w >= 640 and win_h >= 480:
                    self._client = (int(win_w), int(win_h))
                if key == -1:
                    continue
                low = key & 0xFF
                if low in (ord("q"), ord("Q")) or key == 27:
                    if self.dirty and not self.discard_armed:
                        self.discard_armed = True
                        self.status = "这张还没保存。再按一次 Q 退出并放弃，或先按 S。"
                        continue
                    return
                if key in arrows:
                    dx, dy = arrows[key]
                    self._nudge_last_point(dx, dy)
                elif low in (ord("i"), ord("I")):
                    self._nudge_last_point(0.0, -0.1)
                elif low in (ord("k"), ord("K")):
                    self._nudge_last_point(0.0, 0.1)
                elif low in (ord("j"), ord("J")):
                    self._nudge_last_point(-0.1, 0.0)
                elif low in (ord("l"), ord("L")):
                    self._nudge_last_point(0.1, 0.0)
                elif low in (ord("s"), ord("S")):
                    self._save()
                elif low in (ord("r"), ord("R")):
                    if self.draft and self.draft_role != "newton":
                        self.status = "先按 Enter 结束当前眩光弧，再按 R。"
                    else:
                        self.draft_role = "newton"
                        self.status = "接下来的点记为牛顿暗弧。沿暗纹点，Enter 结束。"
                elif low in (ord("g"), ord("G")):
                    if self.draft and self.draft_role != "distractor":
                        self.status = "先按 Enter 结束当前牛顿弧，再按 G。"
                    else:
                        self.draft_role = "distractor"
                        self.status = "接下来的点记为眩光弧，不计入牛顿环。Enter 结束。"
                elif low in (13, 10):
                    self._commit_draft()
                elif low in (ord("+"), ord("=")):
                    self._zoom_at(self._client[0] // 2, max(1, self.view_h // 2), 1)
                elif low in (ord("-"), ord("_")):
                    self._zoom_at(self._client[0] // 2, max(1, self.view_h // 2), -1)
                elif low == ord("0"):
                    self._need_fit = True
                elif low == 9:
                    if self.rings:
                        self.active = (self.active + 1) % len(self.rings)
                        ring = self.rings[self.active]
                        shown = "—" if ring.get("radius_px") is None else f"{float(ring['radius_px']):.2f}"
                        self.status = f"当前弧 {self.active + 1}/{len(self.rings)}，半径 {shown}"
                elif key in (3014656, 127):
                    self._delete_active_ring()
                elif low in (8, 26) or low in (ord("z"), ord("Z")):
                    self._undo_mark()
                elif low in (ord("e"), ord("E")):
                    self._push_undo()
                    order = (None, True, False)
                    self.evaluable = order[(order.index(self.evaluable) + 1) % 3]
                    self.dirty = True
                    self.status = f"可评 = {self.evaluable}"
                elif low == ord("1"):
                    self._set_subset("Easy")
                elif low == ord("2"):
                    self._set_subset("Open-arc")
                elif low == ord("3"):
                    self._set_subset("Illegible")
                elif low == ord("["):
                    self._nudge_half(-0.5)
                elif low == ord("]"):
                    self._nudge_half(0.5)
                elif low in (ord("x"), ord("X")):
                    self._delete_active_ring()
                elif low in (ord("n"), ord("N")):
                    self._switch(1)
                elif low in (ord("b"), ord("B")):
                    self._switch(-1)
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    folder = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else _latest_session()
    if folder is None or not folder.is_dir():
        print("还没有截图目录。先运行 run_paper_bench.bat 保存若干帧。")
        return 1
    try:
        ArcAnnotator(folder).loop()
        return 0
    except Exception:
        import traceback

        text = traceback.format_exc()
        log = Path(__file__).resolve().parent / "last_crash.txt"
        log.write_text(text, encoding="utf-8")
        print(text)
        print(f"错误已写入 {log}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
