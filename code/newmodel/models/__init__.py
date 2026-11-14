"""MA-Net 学生网络模型工厂（仅推理）。"""

from __future__ import annotations

from typing import Any, Mapping

from ..config import ModelConfig
from .baseline import BaselineSegmenter


def build_model(cfg: ModelConfig | Mapping[str, Any]):
    if not isinstance(cfg, ModelConfig):
        cfg = ModelConfig(**dict(cfg))
    return BaselineSegmenter(cfg)


__all__ = ["BaselineSegmenter", "build_model"]
