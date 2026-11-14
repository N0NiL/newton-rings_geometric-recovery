"""可微中心定位、极坐标展开、物理参数头及同心环约束。"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PhysicalParameterHead(nn.Module):
    """从融合特征回归全局条纹物理参数及其异方差。

    圆心采用图像宽高归一化坐标，允许落在画外；其余参数使用无量纲表示。
    Teacher / Student 共用此模块。
    """

    field_names = ("center", "q", "phase", "ellipse")

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden = hidden_channels or channels
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.center = nn.Linear(hidden, 2)
        self.log_q = nn.Linear(hidden, 1)
        self.phase = nn.Linear(hidden, 2)
        self.ellipse = nn.Linear(hidden, 2)
        self.log_variance = nn.Linear(hidden, len(self.field_names))

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        latent = self.shared(self.pool(features))
        # 0.5 + tanh 严格给出约 (-0.5, 1.5)，可表达常见画外圆心。
        center = 0.5 + torch.tanh(self.center(latent))
        phase = F.normalize(self.phase(latent), dim=-1, eps=1e-6)
        ellipse_raw = torch.tanh(self.ellipse(latent))
        ellipse_norm = ellipse_raw.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        ellipse = ellipse_raw * torch.clamp(0.85 / ellipse_norm, max=1.0)
        return {
            "physics_center_norm": center,
            "physics_log_q_norm": self.log_q(latent).clamp(-12.0, 2.0),
            "physics_phase_sin_cos": phase,
            "physics_ellipticity_e1_e2": ellipse,
            "physics_log_variance": self.log_variance(latent).clamp(-8.0, 6.0),
        }


def spatial_soft_argmax_2d(
    logits: Tensor, temperature: float = 1.0, *, normalized: bool = False
) -> Tensor:
    """返回 ``(x, y)`` 中心。

    默认坐标为像素中心坐标：左上像素中心是 ``(0, 0)``，右下是
    ``(W-1, H-1)``。若 ``normalized=True``，则返回 ``[-1, 1]`` 坐标。
    """
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("logits 必须为 [B,1,H,W] 或 [B,H,W]")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    batch, _, height, width = logits.shape
    probability = F.softmax(logits.reshape(batch, -1) / temperature, dim=-1)
    dtype, device = logits.dtype, logits.device
    if normalized:
        xs = torch.linspace(-1.0, 1.0, width, dtype=dtype, device=device)
        ys = torch.linspace(-1.0, 1.0, height, dtype=dtype, device=device)
    else:
        xs = torch.arange(width, dtype=dtype, device=device)
        ys = torch.arange(height, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    x = torch.sum(probability * grid_x.reshape(1, -1), dim=-1)
    y = torch.sum(probability * grid_y.reshape(1, -1), dim=-1)
    return torch.stack((x, y), dim=-1)


class CoarseCenterSoftArgmax(nn.Module):
    """先在低分辨率热图定位，再映射回原图像素中心坐标。"""

    def __init__(self, downsample_factor: int = 4, temperature: float = 0.1) -> None:
        super().__init__()
        if downsample_factor < 1:
            raise ValueError("downsample_factor 至少为 1")
        self.downsample_factor = int(downsample_factor)
        self.temperature = float(temperature)

    def forward(self, logits: Tensor) -> Tensor:
        if logits.ndim == 3:
            logits = logits.unsqueeze(1)
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("中心热图必须为 [B,1,H,W]")
        original_h, original_w = logits.shape[-2:]
        coarse = (
            F.avg_pool2d(
                logits,
                kernel_size=self.downsample_factor,
                stride=self.downsample_factor,
                ceil_mode=True,
            )
            if self.downsample_factor > 1
            else logits
        )
        center = spatial_soft_argmax_2d(coarse, self.temperature)
        coarse_h, coarse_w = coarse.shape[-2:]
        # 使用像素中心仿射映射，兼容 ceil_mode 产生的非整除尺寸。
        center_x = (center[:, 0] + 0.5) * original_w / coarse_w - 0.5
        center_y = (center[:, 1] + 0.5) * original_h / coarse_h - 0.5
        return torch.stack((center_x, center_y), dim=-1)


def pixel_to_grid_coordinates(
    xy: Tensor, height: int, width: int, *, align_corners: bool = False
) -> Tensor:
    """像素中心 ``(x,y)`` 转换为 ``grid_sample`` 的归一化坐标。"""
    if align_corners:
        x = 2.0 * xy[..., 0] / max(width - 1, 1) - 1.0
        y = 2.0 * xy[..., 1] / max(height - 1, 1) - 1.0
    else:
        x = 2.0 * (xy[..., 0] + 0.5) / width - 1.0
        y = 2.0 * (xy[..., 1] + 0.5) / height - 1.0
    return torch.stack((x, y), dim=-1)


def build_polar_grid(
    center_xy: Tensor,
    height: int,
    width: int,
    radial_bins: int,
    angular_bins: int,
    *,
    max_radius: float | Tensor | None = None,
    align_corners: bool = False,
) -> Tensor:
    """构造形状 ``[B,R,A,2]`` 的 grid_sample 网格。

    径向轴从中心 ``r=0`` 到 ``max_radius``（含端点），角度轴为
    ``[0, 2π)``，不重复首尾角度。
    """
    if center_xy.ndim != 2 or center_xy.shape[1] != 2:
        raise ValueError("center_xy 必须为 [B,2]，坐标顺序是 (x,y)")
    if radial_bins < 2 or angular_bins < 1:
        raise ValueError("radial_bins 至少为 2，angular_bins 至少为 1")
    dtype, device = center_xy.dtype, center_xy.device
    if max_radius is None:
        radius: Tensor = torch.as_tensor(
            math.hypot(height - 1, width - 1) / 2.0, dtype=dtype, device=device
        )
    else:
        radius = torch.as_tensor(max_radius, dtype=dtype, device=device)
    if radius.ndim == 0:
        radius = radius.expand(center_xy.shape[0])
    if radius.shape != (center_xy.shape[0],):
        raise ValueError("max_radius 必须是标量或 [B]")
    radial_fraction = torch.linspace(0.0, 1.0, radial_bins, dtype=dtype, device=device)
    angles = (
        torch.arange(angular_bins, dtype=dtype, device=device)
        * (2.0 * math.pi / angular_bins)
    )
    radii = radius[:, None, None] * radial_fraction[None, :, None]
    x = center_xy[:, 0, None, None] + radii * torch.cos(angles)[None, None, :]
    y = center_xy[:, 1, None, None] + radii * torch.sin(angles)[None, None, :]
    xy = torch.stack((x.expand(-1, -1, angular_bins), y.expand(-1, -1, angular_bins)), -1)
    return pixel_to_grid_coordinates(xy, height, width, align_corners=align_corners)


def polar_grid_sample(
    image: Tensor,
    center_xy: Tensor,
    radial_bins: int,
    angular_bins: int,
    *,
    max_radius: float | Tensor | None = None,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = False,
) -> Tensor:
    """将 ``[B,C,H,W]`` 展开为 ``[B,C,R,A]``，全过程可微。"""
    if image.ndim != 4:
        raise ValueError("image 必须为 [B,C,H,W]")
    if image.shape[0] != center_xy.shape[0]:
        raise ValueError("image 与 center_xy 的 batch 大小不一致")
    grid = build_polar_grid(
        center_xy,
        image.shape[-2],
        image.shape[-1],
        radial_bins,
        angular_bins,
        max_radius=max_radius,
        align_corners=align_corners,
    )
    return F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def radial_profile(polar: Tensor, *, angular_weights: Tensor | None = None) -> Tensor:
    """沿角度聚合，返回 ``[B,C,R]`` 径向 profile。"""
    if polar.ndim != 4:
        raise ValueError("polar 必须为 [B,C,R,A]")
    if angular_weights is None:
        return polar.mean(dim=-1)
    weights = torch.broadcast_to(angular_weights, polar.shape)
    denominator = weights.sum(dim=-1).clamp_min(torch.finfo(polar.dtype).eps)
    return (polar * weights).sum(dim=-1) / denominator


def radial_peak_softargmax(
    profile: Tensor, temperature: float = 0.1, *, max_radius: float | Tensor | None = None
) -> Tensor:
    """在径向 profile 上软选峰；默认返回径向 bin 坐标。"""
    if profile.ndim not in (2, 3):
        raise ValueError("profile 必须为 [B,R] 或 [B,C,R]")
    if temperature <= 0:
        raise ValueError("temperature 必须大于 0")
    bins = profile.shape[-1]
    coordinate = torch.arange(bins, dtype=profile.dtype, device=profile.device)
    peak = (F.softmax(profile / temperature, dim=-1) * coordinate).sum(dim=-1)
    if max_radius is not None:
        radius = torch.as_tensor(max_radius, dtype=profile.dtype, device=profile.device)
        while radius.ndim < peak.ndim:
            radius = radius.unsqueeze(-1)
        peak = peak * radius / max(bins - 1, 1)
    return peak


def r2_ring_linearity_loss(
    radii: Tensor,
    ring_orders: Tensor,
    valid_mask: Tensor | None = None,
    *,
    weights: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """约束 ``r² = a·n + b``，仅使用每个样本中的有效环标注。

    ``radii`` 可由网络预测并保留梯度；``ring_orders`` 是对应整数/浮点环序。
    每个样本少于两个有效环时跳过。所有样本均无效时返回与 ``radii``
    计算图相连的零值。
    """
    if radii.shape != ring_orders.shape or radii.ndim != 2:
        raise ValueError("radii 与 ring_orders 必须同为 [B,N]")
    valid = torch.isfinite(radii) & torch.isfinite(ring_orders) & (ring_orders >= 0)
    if valid_mask is not None:
        if valid_mask.shape != radii.shape:
            raise ValueError("valid_mask 形状必须与 radii 相同")
        valid = valid & valid_mask.bool()
    if weights is None:
        all_weights = torch.ones_like(radii)
    else:
        if weights.shape != radii.shape:
            raise ValueError("weights 形状必须与 radii 相同")
        all_weights = weights.clamp_min(0)

    losses: list[Tensor] = []
    for batch_index in range(radii.shape[0]):
        mask = valid[batch_index]
        if int(mask.sum()) < 2:
            continue
        x = ring_orders[batch_index, mask].to(radii.dtype)
        y = radii[batch_index, mask].square()
        w = all_weights[batch_index, mask]
        weight_sum = w.sum().clamp_min(eps)
        x_mean = (w * x).sum() / weight_sum
        y_mean = (w * y).sum() / weight_sum
        slope = (w * (x - x_mean) * (y - y_mean)).sum() / (
            (w * (x - x_mean).square()).sum() + eps
        )
        prediction = slope * (x - x_mean) + y_mean
        scale = (w * (y - y_mean).square()).sum() / weight_sum
        losses.append((w * (y - prediction).square()).sum() / weight_sum / (scale + eps))
    return torch.stack(losses).mean() if losses else radii.sum() * 0.0


class PolarPhysicsModule(nn.Module):
    """中心热图到极坐标径向特征的组合模块。"""

    def __init__(
        self,
        radial_bins: int = 128,
        angular_bins: int = 180,
        center_downsample: int = 4,
        center_temperature: float = 0.1,
        max_radius: float | None = None,
    ) -> None:
        super().__init__()
        self.center = CoarseCenterSoftArgmax(center_downsample, center_temperature)
        self.radial_bins = radial_bins
        self.angular_bins = angular_bins
        self.max_radius = max_radius

    def forward(self, image: Tensor, center_logits: Tensor) -> dict[str, Tensor]:
        center_xy = self.center(center_logits)
        polar = polar_grid_sample(
            image,
            center_xy,
            self.radial_bins,
            self.angular_bins,
            max_radius=self.max_radius,
        )
        profile = radial_profile(polar)
        peak = radial_peak_softargmax(profile, max_radius=self.max_radius)
        return {"center_xy": center_xy, "polar": polar, "radial_profile": profile, "radial_peak": peak}


def coarse_center_soft_argmax(
    logits: Tensor, downsample_factor: int = 4, temperature: float = 0.1
) -> Tensor:
    """``CoarseCenterSoftArgmax`` 的函数式接口，返回原图像素坐标。"""
    return CoarseCenterSoftArgmax(downsample_factor, temperature)(logits)


# 语义清晰的兼容别名。
polar_unwrap = polar_grid_sample
ring_linearity_loss = r2_ring_linearity_loss


def encode_physical_parameters(
    center_xy: Tensor,
    q_r2: Tensor,
    phase: Tensor,
    ellipticity_e1_e2: Tensor,
    image_size: tuple[int, int],
    *,
    eps: float = 1e-8,
) -> Tensor:
    """将像素物理量编码为 ``[cx,cy,log_q,phase,e1,e2]`` 无量纲参数。"""
    height, width = image_size
    if center_xy.shape[-1] != 2 or ellipticity_e1_e2.shape[-1] != 2:
        raise ValueError("center_xy 与 ellipticity_e1_e2 的末维必须为 2")
    scale2 = float(max(height, width) ** 2)
    center_norm = center_xy / center_xy.new_tensor((max(width, 1), max(height, 1)))
    log_q_norm = (q_r2.reshape(-1, 1) / scale2).clamp_min(eps).log()
    return torch.cat(
        (center_norm, log_q_norm, phase.reshape(-1, 1), ellipticity_e1_e2), dim=-1
    )


def decode_physical_parameters(
    parameters: Tensor, image_size: tuple[int, int]
) -> dict[str, Tensor]:
    """解码六维无量纲参数；不截断圆心，因其可合法位于画外。"""
    if parameters.ndim != 2 or parameters.shape[-1] != 6:
        raise ValueError("parameters 必须为 [B,6]")
    height, width = image_size
    scale2 = float(max(height, width) ** 2)
    center_xy = parameters[:, :2] * parameters.new_tensor(
        (max(width, 1), max(height, 1))
    )
    return {
        "center_norm": parameters[:, :2],
        "center_xy": center_xy,
        "log_q_norm": parameters[:, 2:3],
        "q_r2": parameters[:, 2:3].exp() * scale2,
        "phase": parameters[:, 3:4],
        "ellipticity_e1_e2": parameters[:, 4:6],
    }


def physical_head_to_parameters(outputs: Mapping[str, Tensor]) -> Tensor:
    """把 ``PhysicalParameterHead`` 的扁平输出转换为六维优化参数。"""
    center = outputs["physics_center_norm"]
    log_q = outputs.get("physics_log_q_norm", outputs.get("ring_q_log"))
    phase_sc = outputs["physics_phase_sin_cos"]
    ellipse = outputs["physics_ellipticity_e1_e2"]
    if log_q is None:
        raise KeyError("缺少 physics_log_q_norm/ring_q_log")
    # 约定 phase_sin_cos 的顺序是 (sin, cos)。
    phase = torch.atan2(phase_sc[:, :1], phase_sc[:, 1:2])
    return torch.cat((center, log_q.reshape(-1, 1), phase, ellipse), dim=-1)


def normalized_elliptical_radius_squared(
    parameters: Tensor, height: int, width: int
) -> Tensor:
    """返回每个像素的无量纲椭圆半径平方 ``[B,H,W]``。"""
    if parameters.ndim != 2 or parameters.shape[-1] != 6:
        raise ValueError("parameters 必须为 [B,6]")
    dtype, device = parameters.dtype, parameters.device
    scale = float(max(height, width))
    xs = torch.arange(width, dtype=dtype, device=device) / scale
    ys = torch.arange(height, dtype=dtype, device=device) / scale
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cx = parameters[:, 0, None, None] * max(width, 1) / scale
    cy = parameters[:, 1, None, None] * max(height, 1) / scale
    dx, dy = gx[None] - cx, gy[None] - cy
    e1 = parameters[:, 4, None, None]
    e2 = parameters[:, 5, None, None]
    # M=[[1+e1,e2],[e2,1-e1]]；|e|<1 时严格正定。
    return (1.0 + e1) * dx.square() + (1.0 - e1) * dy.square() + 2.0 * e2 * dx * dy


def periodic_ring_phase(parameters: Tensor, height: int, width: int) -> Tensor:
    """计算包裹到 ``[-π,π]`` 的周期相位图。"""
    radius2 = normalized_elliptical_radius_squared(parameters, height, width)
    q_norm = parameters[:, 2, None, None].clamp(-12.0, 2.0).exp()
    phase = 2.0 * math.pi * radius2 / q_norm + parameters[:, 3, None, None]
    return torch.atan2(torch.sin(phase), torch.cos(phase))


def render_centerline_probability(
    parameters: Tensor,
    height: int,
    width: int,
    *,
    line_width: float = 0.16,
    roi_mask: Tensor | None = None,
) -> Tensor:
    """可微渲染 ROI 内环中心线概率，返回 ``[B,1,H,W]``。

    ``line_width`` 是相位域标准差（弧度）；使用圆周距离避免相位边界断梯度。
    """
    if line_width <= 0:
        raise ValueError("line_width 必须大于 0")
    wrapped = periodic_ring_phase(parameters, height, width)
    probability = torch.exp(-0.5 * (wrapped / line_width).square()).unsqueeze(1)
    if roi_mask is not None:
        mask = roi_mask
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            raise ValueError("roi_mask 必须为 [B,H,W] 或 [B,1,H,W]")
        if mask.shape[-2:] != (height, width):
            mask = F.interpolate(
                mask.float(), size=(height, width), mode="bilinear", align_corners=False
            )
        # LM 中心差分会把参数扩成 [B*6,...]，ROI 需按样本重复对齐。
        batch = int(probability.shape[0])
        mask_batch = int(mask.shape[0])
        if mask_batch == 1 and batch > 1:
            mask = mask.expand(batch, -1, -1, -1)
        elif mask_batch != batch:
            if batch % mask_batch != 0:
                raise ValueError(
                    f"roi_mask batch={mask_batch} 无法对齐渲染 batch={batch}"
                )
            mask = mask.repeat_interleave(batch // mask_batch, dim=0)
        probability = probability * mask.to(
            device=probability.device, dtype=probability.dtype
        )
    return probability


def _bounded_parameters(parameters: Tensor) -> Tensor:
    defaults = parameters.new_tensor((0.5, 0.5, -4.0, 0.0, 0.0, 0.0))
    parameters = torch.where(torch.isfinite(parameters), parameters, defaults)
    center = parameters[:, :2].clamp(-0.5, 1.5)
    log_q = parameters[:, 2:3].clamp(-12.0, 2.0)
    phase = torch.atan2(torch.sin(parameters[:, 3:4]), torch.cos(parameters[:, 3:4]))
    ellipse = parameters[:, 4:6]
    norm = ellipse.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    ellipse = ellipse * torch.clamp(0.9 / norm, max=1.0)
    return torch.cat((center, log_q, phase, ellipse), dim=-1)


def refine_physical_parameters(
    initial: Tensor,
    observation: Tensor,
    *,
    steps: int = 4,
    damping: float = 1e-2,
    robust_delta: float = 0.15,
    line_width: float = 0.16,
    roi_mask: Tensor | None = None,
    observation_is_logits: bool = True,
) -> dict[str, Tensor]:
    """固定步数、可训练回传的阻尼 Gauss-Newton/LM 精修。

    雅可比使用六维中心差分并批量渲染，避免逐像素 autograd Jacobian 的高开销。
    非有限、无纹理或病态样本会逐样本回退至输入参数。
    整个求解强制 float32，避免 BF16 autocast 下 linalg.solve 类型不一致。
    """
    if initial.ndim != 2 or initial.shape[-1] != 6:
        raise ValueError("initial 必须为 [B,6]")
    if observation.ndim == 3:
        observation = observation.unsqueeze(1)
    if observation.ndim != 4 or observation.shape[1] != 1:
        raise ValueError("observation 必须为 [B,1,H,W] 或 [B,H,W]")
    if observation.shape[0] != initial.shape[0]:
        raise ValueError("initial 与 observation 的 batch 不一致")

    device = initial.device
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", enabled=False)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        initial_f = initial.float()
        observation_f = observation.float()
        obs = torch.sigmoid(observation_f) if observation_is_logits else observation_f
        obs = obs.clamp(0.0, 1.0)
        height, width = obs.shape[-2:]
        mask = None
        if roi_mask is not None:
            mask = roi_mask.to(device=device, dtype=torch.float32)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[-2:] != (height, width):
                mask = F.interpolate(
                    mask, (height, width), mode="bilinear", align_corners=False
                )
        parameters = _bounded_parameters(initial_f)

        def render(value: Tensor) -> Tensor:
            return render_centerline_probability(
                value, height, width, line_width=line_width, roi_mask=mask
            ).flatten(1)

        target = (obs if mask is None else obs * mask).flatten(1)
        initial_render = render(parameters)
        initial_residual = (initial_render - target).square().mean(dim=-1)
        valid = (
            torch.isfinite(initial_f).all(dim=-1)
            & torch.isfinite(target).all(dim=-1)
            & (target.var(dim=-1) > 1e-8)
        )
        eps = initial_f.new_tensor((2e-3, 2e-3, 2e-3, 3e-3, 2e-3, 2e-3))
        eye = torch.eye(6, dtype=torch.float32, device=device).unsqueeze(0)
        for _ in range(max(int(steps), 0)):
            base = render(parameters)
            perturbation = eye * eps.view(1, 1, 6)
            plus = _bounded_parameters(
                (parameters[:, None, :] + perturbation).reshape(-1, 6)
            )
            minus = _bounded_parameters(
                (parameters[:, None, :] - perturbation).reshape(-1, 6)
            )
            plus_render = render(plus).reshape(initial_f.shape[0], 6, -1)
            minus_render = render(minus).reshape(initial_f.shape[0], 6, -1)
            jacobian = (
                (plus_render - minus_render) / (2.0 * eps.view(1, 6, 1))
            ).transpose(1, 2)
            residual = base - target
            robust = torch.rsqrt(1.0 + (residual / robust_delta).square()).detach()
            weighted_j = jacobian * robust.unsqueeze(-1)
            normal = jacobian.transpose(1, 2) @ weighted_j
            gradient = jacobian.transpose(1, 2) @ (robust * residual).unsqueeze(-1)
            diagonal = normal.diagonal(dim1=-2, dim2=-1).clamp_min(1e-6)
            system = normal + damping * torch.diag_embed(diagonal) + 1e-6 * eye
            finite = (
                torch.isfinite(system).all(dim=(-2, -1))
                & torch.isfinite(gradient).all(dim=(-2, -1))
            )
            finite_system = torch.where(finite[:, None, None], system, eye)
            eigenvalues = torch.linalg.eigvalsh(finite_system)
            conditioned = (
                eigenvalues[:, 0] > eigenvalues[:, -1].clamp_min(1e-12) * 1e-8
            )
            safe_system = torch.where(
                (finite & conditioned)[:, None, None], system, eye
            ).float()
            rhs = (-gradient).float()
            step = torch.linalg.solve(safe_system, rhs).squeeze(-1)
            step = torch.clamp(step, -0.25, 0.25)
            candidate = _bounded_parameters(parameters + step)
            candidate_residual = (render(candidate) - target).square().mean(dim=-1)
            accepted = valid & finite & conditioned & torch.isfinite(candidate_residual)
            accepted = accepted & (
                candidate_residual <= (base - target).square().mean(dim=-1)
            )
            parameters = torch.where(accepted[:, None], candidate, parameters)
            valid = valid & finite & conditioned

        refined_residual = (render(parameters) - target).square().mean(dim=-1)
        success = (
            valid
            & torch.isfinite(refined_residual)
            & (refined_residual < initial_residual)
        )
        refined = torch.where(
            success[:, None], parameters, _bounded_parameters(initial_f)
        )
        refined_residual = torch.where(success, refined_residual, initial_residual)
        return {
            "refined": refined,
            "residual_initial": initial_residual,
            "residual_refined": refined_residual,
            "success": success,
        }


class DampedGaussNewtonRefiner(nn.Module):
    """``refine_physical_parameters`` 的可配置模块包装。"""

    def __init__(
        self, steps: int = 4, damping: float = 1e-2, robust_delta: float = 0.15
    ) -> None:
        super().__init__()
        self.steps = int(steps)
        self.damping = float(damping)
        self.robust_delta = float(robust_delta)

    def forward(
        self,
        initial: Tensor,
        observation: Tensor,
        roi_mask: Tensor | None = None,
        *,
        observation_is_logits: bool = True,
    ) -> dict[str, Tensor]:
        return refine_physical_parameters(
            initial,
            observation,
            steps=self.steps,
            damping=self.damping,
            robust_delta=self.robust_delta,
            roi_mask=roi_mask,
            observation_is_logits=observation_is_logits,
        )


def _valid_field(
    geometry: Mapping[str, Any], name: str, fallback: Tensor, device: torch.device
) -> Tensor:
    value = geometry.get(f"{name}_valid", geometry.get(f"valid_{name}"))
    return (
        value.to(device).float().reshape(-1)
        if isinstance(value, Tensor)
        else fallback
    )


def _heteroscedastic_nll(error: Tensor, log_variance: Tensor, valid: Tensor) -> Tensor:
    per_sample = 0.5 * (
        torch.exp(-log_variance) * error + log_variance
    )
    per_sample = per_sample.reshape(per_sample.shape[0], -1).mean(dim=-1)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)


def supervised_newton_physics_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
) -> dict[str, Tensor]:
    """监督全局物理参数；热图圆心仅对画内样本生效。"""
    center_logits = outputs.get("center_heatmap", outputs.get("center"))
    center_norm = outputs.get("physics_center_norm")
    q_log = outputs.get("physics_log_q_norm", outputs.get("ring_q_log"))
    phase_sc = outputs.get("physics_phase_sin_cos")
    ellipse = outputs.get("physics_ellipticity_e1_e2")
    log_variance = outputs.get("physics_log_variance")
    direction = outputs.get("direction")
    anchor = next(iter(outputs.values()))
    zero = anchor.sum() * 0.0
    result = {
        "center": zero,
        "center_global": zero,
        "center_heatmap": zero,
        "ring_q": zero,
        "phase": zero,
        "ellipse": zero,
        "direction": zero,
    }
    geometry = batch.get("geometry")
    if not isinstance(geometry, Mapping):
        return result
    valid_value = geometry.get("valid")
    valid = (
        valid_value.to(anchor.device).float().reshape(-1)
        if isinstance(valid_value, Tensor)
        else anchor.new_ones(anchor.shape[0], dtype=torch.float32)
    )
    batch_size = valid.shape[0]
    if isinstance(log_variance, Tensor) and log_variance.shape == (batch_size, 4):
        variances = log_variance.float()
    else:
        variances = anchor.new_zeros((batch_size, 4), dtype=torch.float32)

    image_h, image_w = (
        center_logits.shape[-2:]
        if isinstance(center_logits, Tensor)
        else batch.get("image", anchor).shape[-2:]
    )
    cx, cy = geometry.get("center_x"), geometry.get("center_y")
    cx_norm, cy_norm = geometry.get("center_x_norm"), geometry.get("center_y_norm")
    target_xy: Tensor | None = None
    target_norm: Tensor | None = None
    if isinstance(cx_norm, Tensor) and isinstance(cy_norm, Tensor):
        target_norm = torch.stack(
            (cx_norm.to(anchor.device).float(), cy_norm.to(anchor.device).float()), dim=-1
        )
        # schema v2 的归一化定义使用 side，而非 side-1。
        target_xy = target_norm * target_norm.new_tensor((image_w, image_h))
        center_valid = (
            _valid_field(geometry, "center_x_norm", valid, anchor.device)
            * _valid_field(geometry, "center_y_norm", valid, anchor.device)
        )
    elif isinstance(cx, Tensor) and isinstance(cy, Tensor):
        target_xy = torch.stack(
            (cx.to(anchor.device).float(), cy.to(anchor.device).float()), dim=-1
        )
        target_norm = target_xy / target_xy.new_tensor((image_w, image_h))
        center_valid = (
            _valid_field(geometry, "center_x", valid, anchor.device)
            * _valid_field(geometry, "center_y", valid, anchor.device)
        )
    else:
        center_valid = valid * 0.0
    if center_norm is not None and target_norm is not None:
        result["center_global"] = _heteroscedastic_nll(
            (center_norm.float() - target_norm).square().sum(dim=-1, keepdim=True),
            variances[:, 0:1],
            center_valid,
        )

    if center_logits is not None and target_xy is not None:
        explicit_in_frame = geometry.get("center_in_frame")
        if isinstance(explicit_in_frame, Tensor):
            in_frame = (
                explicit_in_frame.to(anchor.device).float().reshape(-1)
                * _valid_field(geometry, "center_in_frame", valid, anchor.device)
            )
        else:
            in_frame = (
                (target_xy[:, 0] >= 0)
                & (target_xy[:, 0] <= image_w - 1)
                & (target_xy[:, 1] >= 0)
                & (target_xy[:, 1] <= image_h - 1)
            ).float()
        heatmap_valid = center_valid * in_frame
        predicted_xy = spatial_soft_argmax_2d(center_logits.float(), temperature=0.1)
        scale = float(max(center_logits.shape[-2:]))
        per_sample = ((predicted_xy - target_xy) / scale).square().sum(dim=-1)
        result["center_heatmap"] = (
            (per_sample * heatmap_valid).sum() / heatmap_valid.sum().clamp_min(1.0)
        )
    result["center"] = result["center_global"] + result["center_heatmap"]

    q_target_norm = geometry.get("q_norm")
    q_target = geometry.get("q_r2")
    if isinstance(q_target_norm, Tensor):
        target_log = q_target_norm.to(anchor.device).float().clamp_min(1e-8).log().reshape(-1, 1)
        q_valid = _valid_field(geometry, "q_norm", valid, anchor.device)
    elif isinstance(q_target, Tensor):
        q_valid = _valid_field(geometry, "q_r2", valid, anchor.device)
        target_log = (
            q_target.to(anchor.device).float().clamp_min(1e-8)
            / float(max(image_h, image_w) ** 2)
        ).log().reshape(-1, 1)
    else:
        target_log, q_valid = None, valid * 0.0
    if q_log is not None and target_log is not None:
        result["ring_q"] = _heteroscedastic_nll(
            (q_log.float().reshape(-1, 1) - target_log).square(),
            variances[:, 1:2],
            q_valid,
        )

    phase_sin, phase_cos = geometry.get("phase_sin"), geometry.get("phase_cos")
    phase_target = geometry.get("phase")
    if phase_sc is not None and isinstance(phase_sin, Tensor) and isinstance(phase_cos, Tensor):
        target_sc = torch.stack(
            (phase_sin.to(anchor.device).float(), phase_cos.to(anchor.device).float()), dim=-1
        )
        phase_valid = (
            _valid_field(geometry, "phase_sin", valid, anchor.device)
            * _valid_field(geometry, "phase_cos", valid, anchor.device)
        )
        result["phase"] = _heteroscedastic_nll(
            (phase_sc.float() - target_sc).square().sum(dim=-1, keepdim=True),
            variances[:, 2:3],
            phase_valid,
        )
    elif phase_sc is not None and isinstance(phase_target, Tensor):
        phase_valid = _valid_field(geometry, "phase", valid, anchor.device)
        phase_value = phase_target.to(anchor.device).float().reshape(-1)
        target_sc = torch.stack((torch.sin(phase_value), torch.cos(phase_value)), dim=-1)
        result["phase"] = _heteroscedastic_nll(
            (phase_sc.float() - target_sc).square().sum(dim=-1, keepdim=True),
            variances[:, 2:3],
            phase_valid,
        )

    e1 = geometry.get("ellipticity_e1", geometry.get("e1"))
    e2 = geometry.get("ellipticity_e2", geometry.get("e2"))
    if ellipse is not None and isinstance(e1, Tensor) and isinstance(e2, Tensor):
        ellipse_valid = (
            _valid_field(geometry, "ellipticity_e1", valid, anchor.device)
            * _valid_field(geometry, "ellipticity_e2", valid, anchor.device)
        )
        target_ellipse = torch.stack(
            (e1.to(anchor.device).float(), e2.to(anchor.device).float()), dim=-1
        )
        result["ellipse"] = _heteroscedastic_nll(
            (ellipse.float() - target_ellipse).square().sum(dim=-1, keepdim=True),
            variances[:, 3:4],
            ellipse_valid,
        )

    if (
        direction is not None
        and target_xy is not None
        and direction.ndim == 4
        and direction.shape[1] == 2
    ):
        b, _, h, w = direction.shape
        xs = torch.arange(w, device=direction.device, dtype=torch.float32)
        ys = torch.arange(h, device=direction.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        dx = gx[None] - target_xy[:, 0].to(direction.device).reshape(b, 1, 1)
        dy = gy[None] - target_xy[:, 1].to(direction.device).reshape(b, 1, 1)
        expected = F.normalize(torch.stack((dx, dy), dim=1), dim=1, eps=1e-6)
        predicted = F.normalize(direction.float(), dim=1, eps=1e-6)
        roi = batch.get("roi_mask")
        if isinstance(roi, Tensor):
            mask = roi.to(direction.device).float()
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = mask[:, :1] * center_valid[:, None, None, None]
        else:
            mask = center_valid[:, None, None, None].expand(b, 1, h, w)
        cosine = 1.0 - (predicted * expected).sum(dim=1, keepdim=True)
        result["direction"] = (cosine * mask).sum() / mask.sum().clamp_min(1.0)

    # 若未来数据提供显式 radii/orders，则同时启用严格 r² 线性约束。
    radii = geometry.get("radii")
    orders = geometry.get("ring_orders")
    if isinstance(radii, Tensor) and isinstance(orders, Tensor):
        result["r2_linearity"] = r2_ring_linearity_loss(
            radii.to(anchor.device), orders.to(anchor.device)
        )
    return result


def _probability_bce(prediction: Tensor, target: Tensor) -> Tensor:
    """对概率图计算 BCE；在 autocast/BF16 下强制 float32，避免不安全算子。"""
    device_type = prediction.device.type
    pred = prediction.float().clamp(1e-4, 1.0 - 1e-4)
    tgt = target.float().clamp(0.0, 1.0)
    if device_type == "cuda":
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return F.binary_cross_entropy(pred, tgt, reduction="none")
    return F.binary_cross_entropy(pred, tgt, reduction="none")


def _geometry_valid_any(geometry: Mapping[str, Any], device: torch.device, batch: int) -> Tensor:
    """至少一个关键物理字段有效时才计算重建损失。"""
    keys = (
        "center_x_norm_valid",
        "center_y_norm_valid",
        "q_norm_valid",
        "phase_sin_valid",
        "ellipticity_e1_valid",
        "valid",
    )
    mask = None
    for key in keys:
        value = geometry.get(key)
        if isinstance(value, Tensor):
            sample = value.to(device).float().reshape(-1)
            mask = sample if mask is None else torch.maximum(mask, sample)
    if mask is None:
        return torch.ones(batch, device=device, dtype=torch.float32)
    return mask


def joint_newton_physics_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    physics_cfg: Any | None = None,
    *,
    enable_solver: bool = False,
    enable_reconstruction: bool | None = None,
) -> dict[str, Tensor]:
    """参数监督 + 可微渲染 +（可选）展开式 LM 精修的联合物理损失。

    返回值仅含可加总叶子项，避免 ``center`` 与其子项重复计权。
    ``enable_reconstruction`` 默认跟随 ``enable_solver``：warmup 阶段只做轻量
    参数回归，避免每 step 都做可微渲染拖慢预训练。
    """
    from ..config import PhysicsConfig

    cfg = physics_cfg if physics_cfg is not None else PhysicsConfig()
    if enable_reconstruction is None:
        enable_reconstruction = bool(enable_solver)
    supervised = supervised_newton_physics_loss(outputs, batch)
    anchor = next(iter(outputs.values()))
    zero = anchor.sum() * 0.0
    parts: dict[str, Tensor] = {
        "center_global": supervised["center_global"] * float(cfg.center_weight),
        "center_heatmap": supervised["center_heatmap"] * float(cfg.center_weight),
        "ring_q": supervised["ring_q"] * float(cfg.q_weight),
        "phase": supervised["phase"] * float(cfg.phase_weight),
        "ellipse": supervised["ellipse"] * float(cfg.ellipticity_weight),
        "direction": supervised["direction"],
        "reconstruction": zero,
        "refined_reconstruction": zero,
        "refined_parameters": zero,
    }
    if "r2_linearity" in supervised:
        parts["r2_linearity"] = supervised["r2_linearity"]

    required = {
        "physics_center_norm",
        "physics_phase_sin_cos",
        "physics_ellipticity_e1_e2",
    }
    has_q = "physics_log_q_norm" in outputs or "ring_q_log" in outputs
    centerline = batch.get("centerline")
    if (
        not enable_reconstruction
        or float(cfg.reconstruction_weight) <= 0
        or not required <= set(outputs)
        or not has_q
        or not isinstance(centerline, Tensor)
    ):
        return parts

    geometry = batch.get("geometry")
    if not isinstance(geometry, Mapping):
        return parts
    valid = _geometry_valid_any(geometry, anchor.device, centerline.shape[0])
    if float(valid.sum()) <= 0:
        return parts

    initial = physical_head_to_parameters(outputs).float()
    height, width = centerline.shape[-2:]
    downsample = max(1, int(cfg.solver_downsample))
    solver_h = max(16, height // downsample)
    solver_w = max(16, width // downsample)
    target = F.interpolate(
        centerline.float(), (solver_h, solver_w), mode="bilinear", align_corners=False
    ).clamp(0.0, 1.0)
    roi = batch.get("roi_mask")
    roi_small = None
    if isinstance(roi, Tensor):
        roi_small = F.interpolate(
            roi.float() if roi.ndim == 4 else roi.float().unsqueeze(1),
            (solver_h, solver_w),
            mode="nearest",
        )
        target = target * roi_small

    rendered = render_centerline_probability(
        initial, solver_h, solver_w, roi_mask=roi_small
    )
    per_sample = _probability_bce(rendered, target).mean(dim=(1, 2, 3))
    parts["reconstruction"] = (
        (per_sample * valid).sum() / valid.sum().clamp_min(1.0)
    ) * float(cfg.reconstruction_weight)

    if not (
        enable_solver
        and bool(cfg.solver_enabled)
        and float(cfg.refined_weight) > 0
    ):
        return parts

    observation = outputs.get("centerline_logits", outputs.get("centerline"))
    if not isinstance(observation, Tensor):
        return parts
    observation = F.interpolate(
        observation.float(), (solver_h, solver_w), mode="bilinear", align_corners=False
    )
    # LM 与重建 BCE 均强制 float32，避免 BF16 autocast 不稳定。
    solved = refine_physical_parameters(
        initial,
        observation,
        steps=int(cfg.solver_steps),
        damping=float(cfg.damping),
        robust_delta=float(cfg.robust_delta),
        roi_mask=roi_small,
        observation_is_logits=True,
    )
    refined = solved["refined"]
    refined_render = render_centerline_probability(
        refined, solver_h, solver_w, roi_mask=roi_small
    )
    refined_bce = _probability_bce(refined_render, target).mean(dim=(1, 2, 3))
    success = solved["success"].float() * valid
    parts["refined_reconstruction"] = (
        (refined_bce * success).sum() / success.sum().clamp_min(1.0)
    ) * float(cfg.refined_weight)
    parts["refined_parameters"] = (
        (solved["residual_refined"] * success).sum() / success.sum().clamp_min(1.0)
    ) * float(cfg.refined_weight)
    return parts


__all__ = [
    "CoarseCenterSoftArgmax",
    "DampedGaussNewtonRefiner",
    "PhysicalParameterHead",
    "PolarPhysicsModule",
    "build_polar_grid",
    "coarse_center_soft_argmax",
    "decode_physical_parameters",
    "encode_physical_parameters",
    "joint_newton_physics_loss",
    "normalized_elliptical_radius_squared",
    "periodic_ring_phase",
    "physical_head_to_parameters",
    "pixel_to_grid_coordinates",
    "polar_grid_sample",
    "polar_unwrap",
    "refine_physical_parameters",
    "render_centerline_probability",
    "r2_ring_linearity_loss",
    "radial_peak_softargmax",
    "radial_profile",
    "ring_linearity_loss",
    "spatial_soft_argmax_2d",
    "supervised_newton_physics_loss",
]
