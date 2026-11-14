"""在线 logits 时序稳定器与轻量三帧特征融合。"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import threading
import time
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TemporalStats:
    frame_count: int
    motion_count: int
    last_motion_score: float
    last_frame_age_ms: float
    mean_frame_age_ms: float
    last_processing_ms: float
    mean_processing_ms: float


class LogitsTemporalFusion:
    """线程安全的 EMA + 窗口中位数在线融合器。

    检测到运动时清空历史并直接采用当前 logits，避免旧位置拖影。静止时，
    EMA 与窗口中位数按 ``median_weight`` 混合。内部历史均会 detach，
    但返回值仍保留当前 logits 的梯度路径。
    """

    def __init__(
        self,
        ema_decay: float = 0.8,
        window_size: int = 5,
        median_weight: float = 0.5,
        motion_threshold: float = 0.03,
    ) -> None:
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay 必须位于 [0,1)")
        if window_size < 1:
            raise ValueError("window_size 至少为 1")
        if not 0.0 <= median_weight <= 1.0:
            raise ValueError("median_weight 必须位于 [0,1]")
        if motion_threshold < 0:
            raise ValueError("motion_threshold 不能为负")
        self.ema_decay = float(ema_decay)
        self.window_size = int(window_size)
        self.median_weight = float(median_weight)
        self.motion_threshold = float(motion_threshold)
        self._lock = threading.RLock()
        self._window: deque[Tensor] = deque(maxlen=window_size)
        self._ema: Tensor | None = None
        self._previous_frame: Tensor | None = None
        self._frame_count = 0
        self._motion_count = 0
        self._last_motion_score = 0.0
        self._last_frame_age_ms = 0.0
        self._frame_age_sum_ms = 0.0
        self._last_processing_ms = 0.0
        self._processing_sum_ms = 0.0

    @staticmethod
    def _motion_frame(frame: Tensor) -> Tensor:
        if frame.ndim == 4:
            if frame.shape[0] != 1:
                raise ValueError("在线运动检测的 frame batch 必须为 1")
            frame = frame[0]
        if frame.ndim == 3:
            frame = frame.float().mean(dim=0)
        elif frame.ndim == 2:
            frame = frame.float()
        else:
            raise ValueError("frame 必须为 [H,W]、[C,H,W] 或 [1,C,H,W]")
        return frame.detach().cpu()

    def _detect_motion_locked(self, frame: Tensor | None) -> tuple[bool, float]:
        if frame is None:
            return False, 0.0
        current = self._motion_frame(frame)
        if self._previous_frame is None or self._previous_frame.shape != current.shape:
            score = 0.0
        else:
            score = float(torch.mean(torch.abs(current - self._previous_frame)).item())
        self._previous_frame = current
        return score > self.motion_threshold, score

    def update(
        self,
        logits: Tensor,
        *,
        frame: Tensor | None = None,
        frame_timestamp: float | None = None,
        processing_time_ms: float | None = None,
        now: float | None = None,
    ) -> Tensor:
        """加入一帧并返回融合结果；时间戳使用 ``time.monotonic()`` 同一时基。"""
        if not logits.is_floating_point():
            raise ValueError("logits 必须是浮点 Tensor")
        with self._lock:
            motion, score = self._detect_motion_locked(frame)
            incompatible = (
                self._ema is not None
                and (
                    self._ema.shape != logits.shape
                    or self._ema.device != logits.device
                    or self._ema.dtype != logits.dtype
                )
            )
            if motion or incompatible:
                self._window.clear()
                self._ema = None
            if motion:
                self._motion_count += 1

            if self._ema is None:
                ema = logits
            else:
                ema = self.ema_decay * self._ema + (1.0 - self.ema_decay) * logits
            self._window.append(logits.detach())
            median = torch.stack([*self._window][:-1] + [logits], dim=0).median(dim=0).values
            fused = (
                logits
                if motion
                else (1.0 - self.median_weight) * ema + self.median_weight * median
            )
            self._ema = ema.detach()

            current_time = time.monotonic() if now is None else float(now)
            age_ms = (
                max(0.0, (current_time - float(frame_timestamp)) * 1000.0)
                if frame_timestamp is not None
                else 0.0
            )
            elapsed_ms = max(0.0, float(processing_time_ms or 0.0))
            self._frame_count += 1
            self._last_motion_score = score
            self._last_frame_age_ms = age_ms
            self._frame_age_sum_ms += age_ms
            self._last_processing_ms = elapsed_ms
            self._processing_sum_ms += elapsed_ms
            return fused

    def reset(self) -> None:
        with self._lock:
            self._window.clear()
            self._ema = None
            self._previous_frame = None
            self._frame_count = 0
            self._motion_count = 0
            self._last_motion_score = 0.0
            self._last_frame_age_ms = 0.0
            self._frame_age_sum_ms = 0.0
            self._last_processing_ms = 0.0
            self._processing_sum_ms = 0.0

    def stats(self) -> TemporalStats:
        with self._lock:
            count = self._frame_count
            return TemporalStats(
                frame_count=count,
                motion_count=self._motion_count,
                last_motion_score=self._last_motion_score,
                last_frame_age_ms=self._last_frame_age_ms,
                mean_frame_age_ms=self._frame_age_sum_ms / count if count else 0.0,
                last_processing_ms=self._last_processing_ms,
                mean_processing_ms=self._processing_sum_ms / count if count else 0.0,
            )

    def stats_dict(self) -> dict[str, int | float]:
        return asdict(self.stats())


class MultiHeadTemporalFusion:
    """为多任务 logits 字典维护相互独立的融合状态，共享运动判定输入。"""

    def __init__(self, **fusion_options: float | int) -> None:
        self._options = fusion_options
        self._fusers: dict[str, LogitsTemporalFusion] = {}
        self._lock = threading.RLock()

    def update(
        self,
        logits: dict[str, Tensor],
        *,
        frame: Tensor | None = None,
        frame_timestamp: float | None = None,
        processing_time_ms: float | None = None,
        now: float | None = None,
    ) -> dict[str, Tensor]:
        if not logits:
            raise ValueError("logits 字典不能为空")
        with self._lock:
            result = {}
            for name, value in logits.items():
                fuser = self._fusers.setdefault(name, LogitsTemporalFusion(**self._options))
                result[name] = fuser.update(
                    value,
                    frame=frame,
                    frame_timestamp=frame_timestamp,
                    processing_time_ms=processing_time_ms,
                    now=now,
                )
            return result

    def reset(self) -> None:
        with self._lock:
            for fuser in self._fusers.values():
                fuser.reset()


class ThreeFrameFeatureFusion(nn.Module):
    """逐通道三帧加权后做轻量空间混合，输出中心帧空间特征。"""

    def __init__(self, channels: int, out_channels: int | None = None) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels 必须大于 0")
        output = channels if out_channels is None else out_channels
        self.channels = channels
        self.temporal_logits = nn.Parameter(torch.zeros(3, channels))
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Conv2d(channels, output, 1),
        )
        self.residual = nn.Identity() if output == channels else nn.Conv2d(channels, output, 1)

    def _stack(self, features: Tensor | Sequence[Tensor]) -> Tensor:
        if isinstance(features, Tensor):
            if features.ndim == 5:
                stacked = features
            elif features.ndim == 4 and features.shape[1] == 3 * self.channels:
                batch, _, height, width = features.shape
                stacked = features.reshape(batch, 3, self.channels, height, width)
            else:
                raise ValueError("Tensor 输入必须为 [B,3,C,H,W] 或 [B,3C,H,W]")
        else:
            if len(features) != 3:
                raise ValueError("必须恰好提供三帧特征")
            stacked = torch.stack(tuple(features), dim=1)
        if stacked.ndim != 5 or stacked.shape[1:3] != (3, self.channels):
            raise ValueError(f"期望 [B,3,{self.channels},H,W]，收到 {tuple(stacked.shape)}")
        return stacked

    def forward(self, features: Tensor | Sequence[Tensor]) -> Tensor:
        stacked = self._stack(features)
        weights = torch.softmax(self.temporal_logits, dim=0)
        fused = (stacked * weights[None, :, :, None, None]).sum(dim=1)
        return self.spatial(fused) + self.residual(stacked[:, 1])


# 简短别名，方便在线推理侧使用。
TemporalLogitsFusion = LogitsTemporalFusion

__all__ = [
    "LogitsTemporalFusion",
    "MultiHeadTemporalFusion",
    "TemporalLogitsFusion",
    "TemporalStats",
    "ThreeFrameFeatureFusion",
]
