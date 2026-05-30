"""时变井控 PDE 残差 / IC / BC。源项随【当前时间所在调度段】的流量缩放。

源项 s_hat(x̂, t̂; 调度) = r(t̂)·q_nd·g_σ(x̂−x̂_w)，其中 r(t̂) = 调度第 ⌊t̂·K⌋ 段的流量比率。
分段常数源：在时间上是阶梯（段边界处跳变），解仍 C0 连续，∂p/∂t 在边界有拐点，tanh 网络平滑近似。
源项对网络参数 θ 无依赖（纯粹是 x̂,t̂,调度 的已知函数），故不影响残差对 θ 的反传。
低层工具复用 pinn.losses（高斯常数、求导、SI↔无量纲换算），口径一致。
"""
from __future__ import annotations

from typing import Tuple

import torch

from pinn.losses import _SQRT_2PI, _grad, _p_hat_of
from pinn_tv.config import PinnTvConfig
from pinn_tv.net import TvPinnMLP


def _active_ratio(cfg: PinnTvConfig, t_hat: torch.Tensor, sched: torch.Tensor) -> torch.Tensor:
    """取每个点当前时间所在段的流量比率 r，形状 (N,1)。

    段号 k = clamp(⌊t̂·K⌋, 0, K−1)（整数索引，detach，不参与对 t 的求导——源项不需要 ∂/∂t）。
    """
    K = cfg.n_segments
    k = (t_hat.detach() * K).long().clamp(0, K - 1)         # (N,1) int64
    qhat_active = torch.gather(sched, 1, k)                  # (N,1)
    return cfg.qhat_to_ratio(qhat_active)


def source_hat(cfg: PinnTvConfig, x_hat: torch.Tensor,
               t_hat: torch.Tensor, sched: torch.Tensor) -> torch.Tensor:
    """时变无量纲源项 s_hat = r(t̂)·q_nd·g_σ(x̂−x̂_w)，形状 (N,1)。"""
    sig = cfg.well_sigma_hat
    g = torch.exp(-((x_hat - cfg.well_x_hat) ** 2) / (2.0 * sig * sig)) / (sig * _SQRT_2PI)
    r = _active_ratio(cfg, t_hat, sched)
    return r * cfg.q_nd * g


def pde_residual(net: TvPinnMLP, cfg: PinnTvConfig,
                 x_hat: torch.Tensor, t_hat: torch.Tensor,
                 sched: torch.Tensor) -> torch.Tensor:
    """残差 r = ∂p̂/∂t̂ − alpha_nd·∂²p̂/∂x̂² − s_hat(x̂,t̂;调度)，形状 (N,1)。"""
    x_hat = x_hat.requires_grad_(True)
    t_hat = t_hat.requires_grad_(True)
    p_hat = net(x_hat, t_hat, sched)
    p_t = _grad(p_hat, t_hat)
    p_x = _grad(p_hat, x_hat)
    p_xx = _grad(p_x, x_hat)
    return p_t - cfg.alpha_nd * p_xx - source_hat(cfg, x_hat, t_hat, sched)


def ic_residual(net: TvPinnMLP, cfg: PinnTvConfig,
                x_ic: torch.Tensor, sched_ic: torch.Tensor) -> torch.Tensor:
    """初值：p̂(x,0;调度) − p̂(P0)。对任意调度目标都是 0。"""
    t0 = torch.zeros_like(x_ic)
    return net(x_ic, t0, sched_ic) - _p_hat_of(cfg.P0, cfg)


def bc_left_residual(net: TvPinnMLP, cfg: PinnTvConfig,
                     t_bc: torch.Tensor, sched_bc: torch.Tensor) -> torch.Tensor:
    x0 = torch.zeros_like(t_bc)
    return net(x0, t_bc, sched_bc) - _p_hat_of(cfg.P_left, cfg)


def bc_right_residual(net: TvPinnMLP, cfg: PinnTvConfig,
                      t_bc: torch.Tensor, sched_bc: torch.Tensor) -> torch.Tensor:
    x1 = torch.ones_like(t_bc)
    return net(x1, t_bc, sched_bc) - _p_hat_of(cfg.P_right, cfg)


def loss_components(net: TvPinnMLP, cfg: PinnTvConfig,
                    x_int: torch.Tensor, t_int: torch.Tensor, s_int: torch.Tensor,
                    x_ic: torch.Tensor, s_ic: torch.Tensor,
                    t_bc: torch.Tensor, s_bc: torch.Tensor,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """未加权三分量损失 (L_pde, L_ic, L_bc)，保留计算图。"""
    r_pde = pde_residual(net, cfg, x_int, t_int, s_int)
    r_bcl = bc_left_residual(net, cfg, t_bc, s_bc)
    r_bcr = bc_right_residual(net, cfg, t_bc, s_bc)

    L_pde = (r_pde ** 2).mean()
    L_bc = (r_bcl ** 2).mean() + (r_bcr ** 2).mean()
    if cfg.hard_ic:
        L_ic = torch.zeros((), device=L_pde.device)
    else:
        L_ic = (ic_residual(net, cfg, x_ic, s_ic) ** 2).mean()
    return L_pde, L_ic, L_bc
