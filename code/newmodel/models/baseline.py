"""基于 segmentation-models-pytorch 的多任务分割基线。"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import torch
from torch import nn

from ..config import ModelConfig
from .physics import PhysicalParameterHead


_ARCHITECTURES = {
    "manet": "MAnet",
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "fpn": "FPN",
    "deeplabv3plus": "DeepLabV3Plus",
}


class BaselineSegmenter(nn.Module):
    """共享 SMP 分割骨干，并将输出通道拆为统一命名的任务字典。"""

    def __init__(self, cfg: ModelConfig | Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(cfg, ModelConfig):
            cfg = ModelConfig(**dict(cfg))
        self.cfg = cfg

        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise RuntimeError(
                "NEWMODEL 基线需要 segmentation-models-pytorch；"
                "请先执行 `pip install segmentation-models-pytorch`。"
            ) from exc

        requested = cfg.name.lower().replace("-", "").replace("_", "")
        if requested.startswith("baseline"):
            requested = requested.removeprefix("baseline") or "manet"
        class_name = _ARCHITECTURES.get(requested)
        if class_name is None:
            supported = ", ".join(sorted(_ARCHITECTURES))
            raise ValueError(f"不支持的基线架构 {cfg.name!r}；可选：{supported}")

        tasks = ["centerline", "boundary"]
        if cfg.center_head:
            tasks.append("center")
        if cfg.direction_head:
            tasks.extend(("direction_x", "direction_y"))
        self.tasks = tuple(tasks)

        model_class = getattr(smp, class_name, None)
        if model_class is None:
            raise RuntimeError(f"当前 segmentation-models-pytorch 版本不含 {class_name}")
        encoder_weights = "imagenet" if cfg.pretrained else None
        input_channels = int(cfg.in_channels) * int(cfg.temporal_frames)
        self.network = model_class(
            encoder_name=cfg.encoder,
            encoder_weights=encoder_weights,
            in_channels=input_channels,
            classes=len(self.tasks),
            activation=None,
        )
        self.physics_head: PhysicalParameterHead | None = None
        if cfg.physics_head:
            channels = int(self.network.encoder.out_channels[-1])
            self.physics_head = PhysicalParameterHead(channels)

    def _decode_features(self, features: list[torch.Tensor]) -> torch.Tensor:
        decoder = self.network.decoder
        try:
            return decoder(features)
        except TypeError:
            return decoder(*features)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = None
        if self.physics_head is None:
            logits = self.network(images)
        else:
            features = self.network.encoder(images)
            decoder_output = self._decode_features(features)
            logits = self.network.segmentation_head(decoder_output)
        channels = logits.split(1, dim=1)
        output = {f"{name}_logits": value for name, value in zip(self.tasks, channels)}
        # boundary 是稳定接口的一部分；关闭该监督时仍返回该预测，损失端可忽略。
        output["boundary"] = output["boundary_logits"]
        output["centerline"] = output["centerline_logits"]
        if "center_logits" in output:
            output["center"] = output["center_logits"]
            output["center_heatmap"] = output["center_logits"]
        if "direction_x_logits" in output:
            output["direction"] = torch.cat(
                (output["direction_x_logits"], output["direction_y_logits"]), dim=1
            )
        if self.physics_head is not None and features is not None:
            physics = self.physics_head(features[-1])
            output.update(physics)
            output["ring_q_log"] = physics["physics_log_q_norm"]
        return output

    def extra_repr(self) -> str:
        return str(asdict(self.cfg))


def build_model(cfg: ModelConfig | Mapping[str, Any]) -> BaselineSegmenter:
    """按配置创建基线模型。"""

    return BaselineSegmenter(cfg)
