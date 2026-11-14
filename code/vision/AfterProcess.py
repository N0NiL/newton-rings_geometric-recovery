"""
NeuroDetection 与 AlgorithmFitter 之间的二级处理流水线。
二值化后先做形态学闭运算（bridging 小断裂），再骨架化，减轻对单像素细骨架的依赖。
同时保留与图同尺寸的 float32 概率图 prob_hw，供拟合阶段加权利用。
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SecondaryPipelineResult:
    """二级流水线输出：几何掩膜用于拟合/显示，prob_hw 为 [0,1] float32。"""

    skeleton: np.ndarray
    prob_hw: np.ndarray


def _morphological_skeleton(binary_u8: np.ndarray) -> np.ndarray:
    """
    对 0/255 二值图做形态学骨架提取（迭代 erode / subtract / or）。
    输出仍为 0/255 单通道 uint8，便于 findContours 与 NewtonFitter 使用。
    """
    if binary_u8 is None or binary_u8.size == 0:
        return binary_u8
    work = (binary_u8 > 0).astype(np.uint8)
    if work.max() == 0:
        return binary_u8

    skel = np.zeros_like(work)
    elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.erode(work, elem)
        opened = cv2.dilate(eroded, elem)
        temp = cv2.subtract(work, opened)
        skel = cv2.bitwise_or(skel, temp)
        work = eroded
        if cv2.countNonZero(work) == 0:
            break

    return (skel * 255).astype(np.uint8)


class NewtonSecondaryPipeline:
    """
    接收 NeuroDetection 输出的概率图，经二级处理后供 AlgorithmFitter 做半径拟合。
    """

    def __init__(
        self,
        binarize_thresh: int = 127,
        closing_kernel_size: int = 3,
        closing_iterations: int = 1,
    ):
        self.binarize_thresh = binarize_thresh
        self.closing_kernel_size = int(max(3, closing_kernel_size | 1))
        self.closing_iterations = int(max(0, closing_iterations))
        k = self.closing_kernel_size
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _prob_hw(self, ai_mask: np.ndarray) -> np.ndarray:
        if ai_mask.dtype in (np.float32, np.float64):
            return np.clip(ai_mask.astype(np.float32), 0.0, 1.0)
        return np.clip(ai_mask.astype(np.float32) / 255.0, 0.0, 1.0)

    def _binarize(self, ai_mask: np.ndarray) -> np.ndarray:
        if ai_mask.dtype in (np.float32, np.float64):
            t = float(self.binarize_thresh) / 255.0
            p = np.clip(ai_mask.astype(np.float32), 0.0, 1.0)
            return ((p >= t).astype(np.uint8) * 255)
        return ((ai_mask > self.binarize_thresh).astype(np.uint8) * 255)

    def _close_binary(self, binary_u8: np.ndarray) -> np.ndarray:
        if self.closing_iterations <= 0:
            return binary_u8
        b = (binary_u8 > 0).astype(np.uint8) * 255
        return cv2.morphologyEx(
            b, cv2.MORPH_CLOSE, self._close_kernel, iterations=self.closing_iterations
        )

    def extra_steps(self, skeleton_u8: np.ndarray) -> np.ndarray:
        return skeleton_u8

    def run(self, ai_mask: np.ndarray) -> SecondaryPipelineResult:
        prob_hw = self._prob_hw(ai_mask)
        binary = self._binarize(ai_mask)
        binary_closed = self._close_binary(binary)
        skeleton = _morphological_skeleton(binary_closed)
        skel_out = self.extra_steps(skeleton)
        return SecondaryPipelineResult(skeleton=skel_out, prob_hw=prob_hw)
