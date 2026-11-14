"""训练、验证、离线推理和在线推理共用的唯一预处理实现。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
try:
    import torch
except ImportError:  # 采集软件便携版会排除数 GB 的 PyTorch；训练环境仍正常导入。
    torch = None  # type: ignore[assignment]

from .config import PreprocessConfig


@dataclass(frozen=True)
class SpatialMapping:
    crop_x0: int
    crop_y0: int
    crop_side: int
    model_side: int
    spatial_mode: str
    pad_left: int = 0
    pad_top: int = 0

    def model_to_source(self, x: float, y: float) -> tuple[float, float]:
        if self.spatial_mode == "pad":
            cx = x - self.pad_left
            cy = y - self.pad_top
        else:
            scale = self.crop_side / float(self.model_side)
            cx, cy = x * scale, y * scale
        return cx + self.crop_x0, cy + self.crop_y0

    def source_to_model(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = x - self.crop_x0, y - self.crop_y0
        if self.spatial_mode == "pad":
            return cx + self.pad_left, cy + self.pad_top
        scale = self.model_side / float(self.crop_side)
        return cx * scale, cy * scale

    def model_length_to_source(self, value: float) -> float:
        if self.spatial_mode == "pad":
            return float(value)
        return float(value) * self.crop_side / float(self.model_side)


@dataclass
class PreprocessResult:
    source_bgr: np.ndarray
    crop_bgr: np.ndarray
    gray_u8: np.ndarray
    roi_mask_u8: np.ndarray
    model_u8: np.ndarray
    model_mask_u8: np.ndarray
    tensor: torch.Tensor
    mapping: SpatialMapping


def roi_file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def load_alpha_roi(path: str | Path, cfg: PreprocessConfig) -> np.ndarray:
    """读取全幅 ROI；RGBA 默认采用 alpha 低值为有效区。"""
    p = Path(path).expanduser().resolve()
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"无法读取 ROI: {p}")
    if raw.shape[:2] != (cfg.source_height, cfg.source_width):
        raise ValueError(
            f"ROI 必须与原始帧一致，期望 {cfg.source_width}×{cfg.source_height}，"
            f"实际 {raw.shape[1]}×{raw.shape[0]}"
        )
    if raw.ndim == 3 and raw.shape[2] >= 4:
        alpha = raw[:, :, 3]
        valid = alpha <= cfg.alpha_threshold if cfg.alpha_low_is_valid else alpha > cfg.alpha_threshold
    else:
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) if raw.ndim == 3 else raw
        valid = gray > cfg.alpha_threshold
    mask = valid.astype(np.uint8) * 255
    coverage = float(np.mean(valid))
    if not 0.005 < coverage < 0.995:
        raise ValueError(f"ROI 有效覆盖率异常: {coverage * 100:.3f}%")
    return mask


class FringePreprocessor:
    def __init__(
        self,
        cfg: PreprocessConfig,
        *,
        repo_root: str | Path | None = None,
        undistort: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
        roi_path = Path(cfg.roi_path)
        if not roi_path.is_absolute():
            roi_path = self.repo_root / roi_path
        self.roi_path = roi_path.resolve()
        self.full_mask = load_alpha_roi(self.roi_path, cfg)
        self.undistort = undistort
        self.clahe = cv2.createCLAHE(
            clipLimit=float(cfg.clahe_clip_limit),
            tileGridSize=(int(cfg.clahe_grid), int(cfg.clahe_grid)),
        )

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        c = self.cfg
        return arr[c.crop_y0 : c.crop_y1, c.crop_x0 : c.crop_x1]

    def _to_model_space(
        self, gray: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, SpatialMapping]:
        c = self.cfg
        crop_side = c.crop_x1 - c.crop_x0
        if c.spatial_mode == "resize":
            model = cv2.resize(gray, (c.model_side, c.model_side), interpolation=cv2.INTER_AREA)
            model_mask = cv2.resize(
                mask, (c.model_side, c.model_side), interpolation=cv2.INTER_NEAREST
            )
            mapping = SpatialMapping(
                c.crop_x0, c.crop_y0, crop_side, c.model_side, c.spatial_mode
            )
        else:
            delta = c.model_side - crop_side
            left = delta // 2
            right = delta - left
            top = delta // 2
            bottom = delta - top
            model = cv2.copyMakeBorder(
                gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
            )
            model_mask = cv2.copyMakeBorder(
                mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
            )
            mapping = SpatialMapping(
                c.crop_x0,
                c.crop_y0,
                crop_side,
                c.model_side,
                c.spatial_mode,
                pad_left=left,
                pad_top=top,
            )
        model_mask = (model_mask > 128).astype(np.uint8) * 255
        model = cv2.bitwise_and(model, model, mask=model_mask)
        return model, model_mask, mapping

    def __call__(self, frame_bgr: np.ndarray) -> PreprocessResult:
        c = self.cfg
        if frame_bgr is None or frame_bgr.shape[:2] != (
            c.source_height,
            c.source_width,
        ):
            shape = None if frame_bgr is None else frame_bgr.shape
            raise ValueError(
                f"原始帧尺寸必须为 {c.source_width}×{c.source_height}，收到 {shape}"
            )
        source = self.undistort(frame_bgr) if self.undistort else frame_bgr
        crop_bgr = self._crop(source)
        crop_mask = self._crop(self.full_mask)
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_and(gray, gray, mask=crop_mask)
        gray = self.clahe.apply(gray)
        gray = cv2.bitwise_and(gray, gray, mask=crop_mask)
        model_u8, model_mask, mapping = self._to_model_space(gray, crop_mask)
        x = model_u8.astype(np.float32) / 255.0
        x = (x - float(c.normalize_mean)) / float(c.normalize_std)
        tensor = (
            torch.from_numpy(x).unsqueeze(0).contiguous()
            if torch is not None
            else np.ascontiguousarray(x[None])
        )
        return PreprocessResult(
            source_bgr=source,
            crop_bgr=crop_bgr,
            gray_u8=gray,
            roi_mask_u8=crop_mask,
            model_u8=model_u8,
            model_mask_u8=model_mask,
            tensor=tensor,
            mapping=mapping,
        )
