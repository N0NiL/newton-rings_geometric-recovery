"""无额外依赖的实验配置与契约哈希。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessConfig:
    source_width: int = 1920
    source_height: int = 1080
    crop_x0: int = 420
    crop_x1: int = 1500
    crop_y0: int = 0
    crop_y1: int = 1080
    roi_path: str = "VisionAnalysis/roi.png"
    alpha_low_is_valid: bool = True
    alpha_threshold: int = 128
    clahe_clip_limit: float = 2.5
    clahe_grid: int = 8
    model_side: int = 768
    spatial_mode: str = "resize"  # resize | pad
    normalize_mean: float = 0.5
    normalize_std: float = 0.5

    def validate(self) -> None:
        if self.crop_x1 <= self.crop_x0 or self.crop_y1 <= self.crop_y0:
            raise ValueError("裁切范围无效")
        crop_w = self.crop_x1 - self.crop_x0
        crop_h = self.crop_y1 - self.crop_y0
        if crop_w != crop_h:
            raise ValueError(f"模型内容区必须为正方形，当前 {crop_w}×{crop_h}")
        if self.spatial_mode not in {"resize", "pad"}:
            raise ValueError("spatial_mode 必须是 resize 或 pad")
        if self.spatial_mode == "pad" and self.model_side < crop_w:
            raise ValueError("pad 模式的 model_side 不能小于内容边长")
        if self.model_side % 32 != 0:
            raise ValueError("model_side 必须能被 32 整除")
        if self.normalize_std <= 0:
            raise ValueError("normalize_std 必须大于 0")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "baseline_manet"
    encoder: str = "resnet34"
    in_channels: int = 1
    temporal_frames: int = 1
    pretrained: bool = True
    center_head: bool = False
    direction_head: bool = False
    boundary_head: bool = True
    physics_head: bool = False


@dataclass(frozen=True)
class PhysicsConfig:
    """神经初值、可微渲染与展开式 LM 的统一配置。"""

    enabled: bool = False
    solver_enabled: bool = False
    solver_steps: int = 4
    solver_downsample: int = 8
    damping: float = 0.01
    robust_delta: float = 0.15
    joint_warmup_epochs: int = 3
    center_weight: float = 1.0
    q_weight: float = 1.0
    phase_weight: float = 0.5
    ellipticity_weight: float = 0.5
    reconstruction_weight: float = 0.5
    refined_weight: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 80
    batch_size: int = 4
    grad_accum_steps: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    patience: int = 15
    num_workers: int = 0
    amp_dtype: str = "bf16"
    seed: int = 42
    real_fraction: float = 0.5
    ema_decay: float = 0.999


@dataclass(frozen=True)
class LossConfig:
    focal: float = 0.25
    dice: float = 0.25
    cldice: float = 0.25
    sdf: float = 0.15
    boundary: float = 0.10
    physics: float = 0.10
    focal_gamma: float = 2.0
    soft_skeleton_iterations: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "baseline_768"
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data_root: str = "NEWMODEL/data/dataset"
    output_dir: str = "NEWMODEL/runs"

    def validate(self) -> None:
        self.preprocess.validate()
        if self.model.temporal_frames < 1:
            raise ValueError("temporal_frames 至少为 1")
        if self.physics.solver_steps < 0 or self.physics.solver_downsample < 1:
            raise ValueError("solver_steps 不能为负，solver_downsample 必须大于 0")
        if self.physics.damping <= 0 or self.physics.robust_delta <= 0:
            raise ValueError("LM damping 与 robust_delta 必须大于 0")
        if self.physics.enabled and not self.model.physics_head:
            raise ValueError("启用 physics 时 model.physics_head 必须为 true")
        if self.train.batch_size < 1 or self.train.grad_accum_steps < 1:
            raise ValueError("batch_size 与 grad_accum_steps 必须大于 0")
        if not 0.0 <= self.train.real_fraction <= 1.0:
            raise ValueError("real_fraction 必须位于 [0,1]")
        weights = asdict(self.loss)
        weighted = [v for k, v in weights.items() if k not in {"focal_gamma", "soft_skeleton_iterations"}]
        if sum(float(x) for x in weighted) <= 0:
            raise ValueError("至少启用一个损失项")

    def contract_dict(self) -> dict[str, Any]:
        model = asdict(self.model)
        # 未启用物理头时维持历史 baseline 契约哈希，保证既有 checkpoint 可用。
        if not self.model.physics_head:
            model.pop("physics_head", None)
        contract: dict[str, Any] = {
            "schema_version": 2 if self.model.physics_head else 1,
            "preprocess": asdict(self.preprocess),
            "model": model,
        }
        if self.model.physics_head:
            contract["physics"] = asdict(self.physics)
        return contract

    def contract_hash(self) -> str:
        raw = json.dumps(self.contract_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig(
        name=str(data.get("name", "experiment")),
        preprocess=PreprocessConfig(**dict(data.get("preprocess") or {})),
        model=ModelConfig(**dict(data.get("model") or {})),
        physics=PhysicsConfig(**dict(data.get("physics") or {})),
        train=TrainConfig(**dict(data.get("train") or {})),
        loss=LossConfig(**dict(data.get("loss") or {})),
        data_root=str(data.get("data_root", "NEWMODEL/data/dataset")),
        output_dir=str(data.get("output_dir", "NEWMODEL/runs")),
    )
    cfg.validate()
    return cfg


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    p = Path(path).expanduser().resolve()
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置根节点必须是对象: {p}")
    return _construct(data)


def save_resolved_config(cfg: ExperimentConfig, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["contract_hash"] = cfg.contract_hash()
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
