"""与旧 NeuroDetection 并行存在的 NEWMODEL 适配器。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# 发布包：code/vision + code/newmodel；兼容旧布局下的 NEWMODEL 包名。
_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

try:
    from newmodel.inference import NewModelInference  # noqa: E402
except ImportError:  # pragma: no cover
    from NEWMODEL.inference import NewModelInference  # noqa: E402


class NewModelDetector:
    """将丰富的 NEWMODEL 字典输出适配为旧检测器二元组接口。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        output_dir: str = "test_results",
        *,
        checkpoint_path: str | Path | None = None,
        return_debug: bool = False,
        inference: NewModelInference | Any | None = None,
        **inference_kwargs: Any,
    ) -> None:
        # output_dir 仅为兼容旧构造参数；NEWMODEL 推理不落盘。
        self.output_dir = output_dir
        self.return_debug = bool(return_debug)
        selected_path = checkpoint_path or model_path
        if inference is not None:
            self.inference = inference
        elif selected_path is None:
            raise ValueError("必须提供 model_path/checkpoint_path 或注入 inference")
        else:
            self.inference = NewModelInference(selected_path, **inference_kwargs)
        self.last_debug: dict[str, Any] | None = None
        self.last_physics: dict[str, Any] | None = None

    def get_refined_skeleton(
        self, frame: np.ndarray, return_debug: bool | None = None
    ):
        """默认与旧方法同样返回 ``(prob, contours)``；按需追加 debug 字典。"""
        result = self.inference.infer(frame)
        prob = np.asarray(result["prob"], dtype=np.float32)
        contours = result["contours"]
        physics = result.get("physics_refined") or result.get("physics_initial")
        self.last_physics = physics if isinstance(physics, dict) else None
        debug = {
            "heads": result.get("heads", {}),
            "mapping": result.get("mapping"),
            "timing_ms": result.get("timing_ms", {}),
            "physics": self.last_physics,
        }
        self.last_debug = debug
        include_debug = self.return_debug if return_debug is None else bool(return_debug)
        return (prob, contours, debug) if include_debug else (prob, contours)


# 允许现有调用方只替换导入模块，不替换类名。
NewtonDetector = NewModelDetector

__all__ = ["NewModelDetector", "NewtonDetector"]
