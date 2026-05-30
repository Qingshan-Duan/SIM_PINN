"""参数化 PDE 残差 / IC / BC 三类 loss。与 pinn.losses 的唯一区别：源项随采样到的流量 q̂ 缩放。

关键：源强 ∝ 当前点的流量比率 r = q/well_rate（由 q̂ 反映射得到），
即 s_hat(x̂; r) = r · q_nd · g_σ(x̂ − x̂_w)，其中 cfg.q_nd 是基准流量(r=1)的无量纲源强。
q̂ 本身**不被求导**（不在 PDE 微分项里，只乘源项），但反传仍会穿过 q̂ 的输入权重学出 q 依赖。

IC/BC 与 q 无关（初值=p_ref、两端定压恒定，目标都是 p̂=0），但网络仍要在采样到的 q̂ 上取值。
低层工具（高斯常数、求导、SI↔无量纲换算）直接复用 pinn.losses，避免口径漂移。
"""
from __future__ import annotations

from typing import Tuple

import torch

from pinn.losses import _SQRT_2PI, _grad, _p_hat_of
from pinn_param.config import PinnParamConfig
from pinn_param.net import ParamPinnMLP


def source_hat(cfg: PinnParamConfig, x_hat: torch.Tensor,
               ratio: torch.Tensor) -> torch.Tensor:
    """井的无量纲源项 s_hat = ratio · q_nd · g_σ(x̂ − x̂_w)，形状 (N, 1)。

    ratio 形状 (N,1)，与 x_hat 逐点对应：每个 collocation 点带自己的流量比率。
    cfg.q_nd 是基准流量(r=1)的源强，乘 ratio 得到该点真实流量的源强。
    """
    sig = cfg.well_sigma_hat
    g = torch.exp(-((x_hat - cfg.well_x_hat) ** 2) / (2.0 * sig * sig)) / (sig * _SQRT_2PI)
    return ratio * cfg.q_nd * g


def pde_residual(net: ParamPinnMLP, cfg: PinnParamConfig,
                 x_hat: torch.Tensor, t_hat: torch.Tensor,
                 q_hat: torch.Tensor) -> torch.Tensor:
    """残差 r = ∂p̂/∂t̂ − alpha_nd·∂²p̂/∂x̂² − s_hat(x̂; q̂)，形状 (N, 1)。

    只对 x̂、t̂ 求导；q̂ 作为常数输入（不 requires_grad），仅用于反映射出 ratio 喂源项。
    """
    x_hat = x_hat.requires_grad_(True)
    t_hat = t_hat.requires_grad_(True)
    p_hat = net(x_hat, t_hat, q_hat)
    p_t = _grad(p_hat, t_hat)
    p_x = _grad(p_hat, x_hat)
    p_xx = _grad(p_x, x_hat)
    ratio = cfg.qhat_to_ratio(q_hat)
    return p_t - cfg.alpha_nd * p_xx - source_hat(cfg, x_hat, ratio)


def ic_residual(net: ParamPinnMLP, cfg: PinnParamConfig,
                x_ic: torch.Tensor, q_ic: torch.Tensor) -> torch.Tensor:
    """初始条件：p̂(x, t=0; q̂) − p̂(P0)。对所有 q̂ 目标相同（=0）。"""
    t0 = torch.zeros_like(x_ic)
    return net(x_ic, t0, q_ic) - _p_hat_of(cfg.P0, cfg)


def bc_left_residual(net: ParamPinnMLP, cfg: PinnParamConfig,
                     t_bc: torch.Tensor, q_bc: torch.Tensor) -> torch.Tensor:
    """左边界 x̂=0 的 Dirichlet：p̂ − p̂(P_left)。"""
    x0 = torch.zeros_like(t_bc)
    return net(x0, t_bc, q_bc) - _p_hat_of(cfg.P_left, cfg)


def bc_right_residual(net: ParamPinnMLP, cfg: PinnParamConfig,
                      t_bc: torch.Tensor, q_bc: torch.Tensor) -> torch.Tensor:
    """右边界 x̂=1 的 Dirichlet：p̂ − p̂(P_right)。"""
    x1 = torch.ones_like(t_bc)
    return net(x1, t_bc, q_bc) - _p_hat_of(cfg.P_right, cfg)


def loss_components(net: ParamPinnMLP, cfg: PinnParamConfig,
                    x_int: torch.Tensor, t_int: torch.Tensor, q_int: torch.Tensor,
                    x_ic: torch.Tensor, q_ic: torch.Tensor,
                    t_bc: torch.Tensor, q_bc: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回**未加权**的三个分量损失 (L_pde, L_ic, L_bc)，保留计算图（供自适应权重求分量梯度）。"""
    r_pde = pde_residual(net, cfg, x_int, t_int, q_int)
    r_bcl = bc_left_residual(net, cfg, t_bc, q_bc)
    r_bcr = bc_right_residual(net, cfg, t_bc, q_bc)

    L_pde = (r_pde ** 2).mean()
    L_bc = (r_bcl ** 2).mean() + (r_bcr ** 2).mean()
    if cfg.hard_ic:
        L_ic = torch.zeros((), device=L_pde.device)
    else:
        L_ic = (ic_residual(net, cfg, x_ic, q_ic) ** 2).mean()
    return L_pde, L_ic, L_bc
