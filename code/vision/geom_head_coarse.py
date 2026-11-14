"""
幾何頭（SmallGeomRegressor）粗圓心推理：與訓練一致輸入 1×360×360 骨架灰度 [0,1]，
輸出圓心映射回當前骨架圖的 H×W 像素座標，供 NewtonFitter 與輪廓統計初值低權重融合。

權重文件格式與 ExtraUtilities/train_geometry_synth 保存的 checkpoint 一致（含 model / k_max / pool_hw）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_EXTRA = _REPO / "ExtraUtilities"
if str(_EXTRA) not in sys.path:
    sys.path.insert(0, str(_EXTRA))


class GeomHeadCoarseHint:
    """延遲加載 torch 與 SmallGeomRegressor，避免未使用幾何頭時強依賴。"""

    def __init__(self, ckpt_path: str | Path, *, device: str | None = None) -> None:
        self.ckpt_path = str(Path(ckpt_path).expanduser().resolve())
        self._device_pref = device
        self._model: Any = None
        self._torch_device: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as e:
            raise RuntimeError("幾何頭需要 PyTorch（torch），請先安裝") from e

        from train_geometry_synth import (  # noqa: E402
            K_MAX_DEFAULT,
            SmallGeomRegressor,
            infer_pool_hw_from_state_dict,
        )

        dev = self._device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch_device = torch.device(dev)
        ck = torch.load(self.ckpt_path, map_location=self._torch_device)
        k_max = int(ck.get("k_max", K_MAX_DEFAULT)) if isinstance(ck, dict) else K_MAX_DEFAULT
        pool_hw = 7
        if isinstance(ck, dict):
            if ck.get("pool_hw") is not None:
                pool_hw = int(ck["pool_hw"])
            elif isinstance(ck.get("model"), dict):
                pool_hw = infer_pool_hw_from_state_dict(ck["model"])

        m = SmallGeomRegressor(k_max=k_max, pool_hw=pool_hw).to(self._torch_device)
        if isinstance(ck, dict) and "model" in ck:
            m.load_state_dict(ck["model"])
        elif isinstance(ck, dict) and "state_dict" in ck:
            m.load_state_dict(ck["state_dict"])
        else:
            m.load_state_dict(ck)
        m.eval()
        self._model = m

    @staticmethod
    def _to_float01_u8(img_u8: np.ndarray) -> np.ndarray:
        return np.clip(img_u8.astype(np.float32) / 255.0, 0.0, 1.0)

    def predict_center_hw(self, skeleton_u8: np.ndarray) -> tuple[float, float] | None:
        """
        :param skeleton_u8: 單通道 uint8，0/255 骨架（與擬合輸入同尺寸）。
        :return: 圓心 (cx, cy) 像素座標，與 skeleton_u8 同座標系；失敗則 None。
        """
        if skeleton_u8 is None or skeleton_u8.size == 0:
            return None
        if skeleton_u8.ndim != 2:
            return None
        h0, w0 = int(skeleton_u8.shape[0]), int(skeleton_u8.shape[1])
        if h0 < 4 or w0 < 4:
            return None

        self._ensure_loaded()
        import torch

        img = self._to_float01_u8(skeleton_u8)
        if (h0, w0) != (360, 360):
            img = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            img = np.clip(img, 0.0, 1.0)

        with torch.no_grad():
            inp = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self._torch_device)
            pred = self._model(inp).squeeze(0)
            cx_360 = float(pred[0].item())
            cy_360 = float(pred[1].item())

        cx = cx_360 * (w0 / 360.0)
        cy = cy_360 * (h0 / 360.0)
        return cx, cy
