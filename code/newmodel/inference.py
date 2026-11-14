"""NEWMODEL 的契约校验、统一预处理与推理入口。"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig, ModelConfig, PhysicsConfig, PreprocessConfig
from .preprocess import FringePreprocessor, PreprocessResult, roi_file_hash
from .temporal import LogitsTemporalFusion

HEAD_ALIASES = {
    "centerline": ("centerline", "centerline_logits", "segmentation", "mask", "logits", "out"),
    "boundary": ("boundary", "boundary_logits", "edge", "edges"),
    "center": ("center", "center_logits", "center_heatmap", "heatmap"),
    "direction": ("direction", "direction_logits", "orientation", "vector"),
}


def _config_from_dict(data: Mapping[str, Any]) -> ExperimentConfig:
    """仅提取 ExperimentConfig 字段，允许 checkpoint 带额外训练元数据。"""
    cfg = ExperimentConfig(
        name=str(data.get("name", "checkpoint")),
        preprocess=PreprocessConfig(**dict(data.get("preprocess") or {})),
        model=ModelConfig(**dict(data.get("model") or {})),
        physics=PhysicsConfig(**dict(data.get("physics") or {})),
        data_root=str(data.get("data_root", "NEWMODEL/data/dataset")),
        output_dir=str(data.get("output_dir", "NEWMODEL/runs")),
    )
    cfg.validate()
    return cfg


def normalize_model_output(output: Any) -> dict[str, torch.Tensor]:
    """把不同模型的 Tensor/tuple/dict 输出归一为多头字典。"""
    if isinstance(output, torch.Tensor):
        return {"centerline": output}
    if isinstance(output, (tuple, list)):
        names = ("centerline", "boundary", "center", "direction")
        return {name: value for name, value in zip(names, output) if isinstance(value, torch.Tensor)}
    if not isinstance(output, Mapping):
        raise TypeError(f"模型输出必须是 Tensor、序列或 dict，收到 {type(output).__name__}")

    result: dict[str, torch.Tensor] = {}
    for canonical, aliases in HEAD_ALIASES.items():
        for key in aliases:
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                result[canonical] = value
                break
    for key, value in output.items():
        if isinstance(value, torch.Tensor) and (
            str(key).startswith("physics_") or str(key) == "ring_q_log"
        ):
            result[str(key)] = value
    if "centerline" not in result:
        tensor_keys = [k for k, v in output.items() if isinstance(v, torch.Tensor)]
        raise ValueError(f"模型输出缺少中心线 head；现有 Tensor 键: {tensor_keys}")
    return result


def _import_model_class(model_name: str):
    """延迟导入，避免未使用的可选模型依赖阻塞模块导入。"""
    key = model_name.lower()
    if key in {"baseline", "baseline_manet", "manet"}:
        module_name, candidates = ".models.baseline", ("BaselineModel", "BaselineMANet", "build_model")
    else:
        raise ValueError(f"未知 NEWMODEL 模型: {model_name}")
    module = importlib.import_module(module_name, package=__package__)
    for name in candidates:
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    raise ImportError(f"{module.__name__} 未提供受支持的模型工厂: {candidates}")


def build_model(cfg: ModelConfig) -> torch.nn.Module:
    factory = _import_model_class(cfg.name)
    # Baseline/MAnet：同样禁止构造时下载 ImageNet；权重由 checkpoint 覆盖。
    from dataclasses import replace

    cfg = replace(cfg, pretrained=False)
    kwargs = asdict(cfg)
    try:
        return factory(cfg)
    except TypeError:
        try:
            return factory(**kwargs)
        except TypeError:
            return factory()


def load_checkpoint_payload(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("NEWMODEL checkpoint 必须是包含 config/contract/state_dict 的字典")
    return payload


class ContractError(ValueError):
    """checkpoint 与运行时图像/模型契约不一致。"""


class NewModelInference:
    """带严格契约检查的单帧/时序 NEWMODEL 推理器。"""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        model: torch.nn.Module | None = None,
        config: ExperimentConfig | Mapping[str, Any] | None = None,
        preprocessor: Any | None = None,
        device: str | torch.device | None = None,
        threshold: float = 0.5,
        temporal_alpha: float | None = None,
        temporal_window: int = 5,
        motion_threshold: float = 0.03,
        strict: bool = True,
        repo_root: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = float(threshold)
        self.temporal_alpha = temporal_alpha
        self._temporal_fuser = (
            LogitsTemporalFusion(
                ema_decay=1.0 - float(np.clip(temporal_alpha, 0.0, 1.0)),
                window_size=max(1, int(temporal_window)),
                median_weight=0.5,
                motion_threshold=float(motion_threshold),
            )
            if temporal_alpha is not None
            else None
        )
        payload: dict[str, Any] = {}
        if checkpoint_path is not None:
            payload = load_checkpoint_payload(checkpoint_path, self.device)

        raw_cfg = config or payload.get("config") or payload.get("resolved_config")
        if raw_cfg is None:
            raise ContractError("缺少 config：推理必须由 checkpoint 或调用方提供完整配置")
        self.config = raw_cfg if isinstance(raw_cfg, ExperimentConfig) else _config_from_dict(raw_cfg)
        self._validate_contract(payload, strict=strict)

        self.preprocessor = preprocessor or FringePreprocessor(
            self.config.preprocess, repo_root=repo_root
        )
        if strict and payload.get("roi_hash") and hasattr(self.preprocessor, "roi_path"):
            actual_roi_hash = roi_file_hash(self.preprocessor.roi_path)
            if payload["roi_hash"] != actual_roi_hash:
                raise ContractError(
                    f"ROI 契约不匹配: checkpoint={payload['roi_hash']}，当前={actual_roi_hash}"
                )

        self.model = (model or build_model(self.config.model)).to(self.device)
        state = None
        for key in ("student_state_dict", "state_dict", "model_state_dict", "model"):
            if isinstance(payload.get(key), Mapping):
                state = payload[key]
                break
        if checkpoint_path is not None:
            if not isinstance(state, Mapping):
                raise ValueError("checkpoint 缺少 state_dict/model_state_dict/model")
            self.model.load_state_dict(state, strict=strict)
        self.model.eval()

    def _validate_contract(self, payload: Mapping[str, Any], *, strict: bool) -> None:
        expected_hash = self.config.contract_hash()
        stored_hash = payload.get("contract_hash")
        contract = payload.get("contract")
        if isinstance(contract, Mapping):
            stored_hash = contract.get("hash", contract.get("contract_hash", stored_hash))
            contract_body = contract.get("definition", contract.get("dict"))
            if strict and contract_body is not None and contract_body != self.config.contract_dict():
                raise ContractError("checkpoint contract 定义与 config 不一致")
        if strict and payload and not stored_hash:
            raise ContractError("checkpoint 缺少 contract_hash，拒绝不受控推理")
        if stored_hash and str(stored_hash) != expected_hash:
            raise ContractError(
                f"checkpoint 契约不匹配: checkpoint={stored_hash}，运行时={expected_hash}"
            )

    @staticmethod
    def _to_probability(value: torch.Tensor, key: str) -> torch.Tensor:
        if key == "direction":
            return F.normalize(value.float(), dim=1, eps=1e-6)
        return torch.sigmoid(value.float())

    def _prepare_temporal_input(self, current: torch.Tensor) -> torch.Tensor:
        # 预处理输出为 C×H×W。模型声明多帧时，启动阶段重复首帧。
        frames = int(self.config.model.temporal_frames)
        history = getattr(self, "_tensor_history", None)
        if history is None:
            history = self._tensor_history = deque(maxlen=frames)
        history.append(current)
        seq = list(history)
        while len(seq) < frames:
            seq.insert(0, seq[0])
        if frames == 1:
            return current.unsqueeze(0)
        # 统一首选 B×T×C×H×W；不支持该形状的模型可声明 accepts_flat_temporal。
        stacked = torch.stack(seq, dim=0).unsqueeze(0)
        if getattr(self.model, "accepts_flat_temporal", False):
            b, t, c, h, w = stacked.shape
            return stacked.reshape(b, t * c, h, w)
        return stacked

    def _forward(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prepare_temporal_input(tensor).to(self.device, non_blocking=True)
        with torch.inference_mode():
            try:
                raw = self.model(x)
            except (RuntimeError, ValueError) as first_error:
                if x.ndim != 5:
                    raise
                b, t, c, h, w = x.shape
                try:
                    raw = self.model(x.reshape(b, t * c, h, w))
                except (RuntimeError, ValueError):
                    raise first_error
        return normalize_model_output(raw)

    def infer(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        started = time.perf_counter()
        prep: PreprocessResult = self.preprocessor(frame_bgr)
        self.last_preprocess = prep
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        infer_started = time.perf_counter()
        heads = self._forward(prep.tensor)
        infer_ms = (time.perf_counter() - infer_started) * 1000.0
        if self._temporal_fuser is not None:
            heads["centerline"] = self._temporal_fuser.update(
                heads["centerline"],
                frame=prep.tensor,
                processing_time_ms=infer_ms,
            )

        probabilities: dict[str, np.ndarray] = {}
        side = int(self.config.preprocess.model_side)
        for key in ("centerline", "boundary", "center", "direction"):
            tensor = heads.get(key)
            if tensor is None:
                continue
            value = self._to_probability(tensor, key)
            if value.ndim == 3:
                value = value.unsqueeze(1)
            if value.shape[-2:] != (side, side):
                value = F.interpolate(value, (side, side), mode="bilinear", align_corners=False)
            array = value[0].detach().cpu().numpy().astype(np.float32)
            probabilities[key] = array[0] if array.shape[0] == 1 else array

        prob = probabilities["centerline"]
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
        probabilities["centerline"] = prob

        physics_initial = physics_refined = uncertainty = solver_stats = None
        required_physics = {
            "physics_center_norm",
            "physics_phase_sin_cos",
            "physics_ellipticity_e1_e2",
        }
        if required_physics <= heads.keys() and (
            "physics_log_q_norm" in heads or "ring_q_log" in heads
        ):
            from .models.physics import (
                decode_physical_parameters,
                physical_head_to_parameters,
                refine_physical_parameters,
            )

            initial_parameters = physical_head_to_parameters(heads)
            refined_parameters = initial_parameters
            physics_initial = {
                key: value[0].detach().cpu().numpy().astype(np.float32)
                for key, value in decode_physical_parameters(
                    initial_parameters, (side, side)
                ).items()
            }
            if self.config.physics.solver_enabled:
                downsample = max(1, int(self.config.physics.solver_downsample))
                solver_side = max(16, side // downsample)
                observation = F.interpolate(
                    heads["centerline"].float(),
                    (solver_side, solver_side),
                    mode="bilinear",
                    align_corners=False,
                )
                roi = torch.from_numpy(
                    (prep.model_mask_u8 > 127).astype(np.float32)
                ).to(self.device)[None, None]
                roi = F.interpolate(roi, (solver_side, solver_side), mode="nearest")
                solved = refine_physical_parameters(
                    initial_parameters.float(),
                    observation,
                    steps=self.config.physics.solver_steps,
                    damping=self.config.physics.damping,
                    robust_delta=self.config.physics.robust_delta,
                    roi_mask=roi,
                )
                refined_parameters = solved["refined"]
                solver_stats = {
                    "success": bool(solved["success"][0].item()),
                    "residual_initial": float(solved["residual_initial"][0].item()),
                    "residual_refined": float(solved["residual_refined"][0].item()),
                }
            physics_refined = {
                key: value[0].detach().cpu().numpy().astype(np.float32)
                for key, value in decode_physical_parameters(
                    refined_parameters, (side, side)
                ).items()
            }
            log_variance = heads.get("physics_log_variance")
            if log_variance is not None:
                uncertainty = (
                    torch.exp(0.5 * log_variance.float())[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        binary = (prob >= self.threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        total_ms = (time.perf_counter() - started) * 1000.0
        return {
            "prob": prob,
            "contours": contours,
            "heads": probabilities,
            "mapping": prep.mapping,
            "physics_initial": physics_initial,
            "physics_refined": physics_refined,
            "uncertainty": uncertainty,
            "solver_stats": solver_stats,
            "temporal_stats": (
                self._temporal_fuser.stats_dict()
                if self._temporal_fuser is not None
                else None
            ),
            "timing_ms": {
                "preprocess": preprocess_ms,
                "inference": infer_ms,
                "postprocess": max(0.0, total_ms - preprocess_ms - infer_ms),
                "total": total_ms,
            },
        }

    def __call__(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        return self.infer(frame_bgr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEWMODEL 单图推理")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {args.image}")
    result = NewModelInference(args.checkpoint, device=args.device).infer(image)
    cv2.imwrite(args.output, np.clip(result["prob"] * 255.0, 0, 255).astype(np.uint8))
    print(json.dumps(result["timing_ms"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
