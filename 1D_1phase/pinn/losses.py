"""PDE 残差 / 初始条件 / 边界条件三类 loss。全部在无量纲坐标下计算。"""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from pinn.config import PinnConfig
from pinn.net import PinnMLP


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """对标量场 y 求 ∂y/∂x（x 需 requires_grad）。create_graph=True 以便再求高阶导。"""
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
    )[0]


def pde_residual(net: PinnMLP, cfg: PinnConfig,
                 x_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
    """残差 r = ∂p_hat/∂t_hat − alpha_nd · ∂²p_hat/∂x_hat²，形状 (N, 1)。"""
    x_hat = x_hat.requires_grad_(True)
    t_hat = t_hat.requires_grad_(True)
    p_hat = net(x_hat, t_hat)
    p_t = _grad(p_hat, t_hat)
    p_x = _grad(p_hat, x_hat)
    p_xx = _grad(p_x, x_hat)
    return p_t - cfg.alpha_nd * p_xx


def _p_hat_of(P: float, cfg: PinnConfig) -> float:
    """把 SI 压力 P 换算成无量纲 p_hat。"""
    return (P - cfg.p_ref) / cfg.dp_scale


def ic_residual(net: PinnMLP, cfg: PinnConfig, x_hat_ic: torch.Tensor) -> torch.Tensor:
    """初始条件：p_hat(x, t=0) − p_hat(P0)。"""
    t0 = torch.zeros_like(x_hat_ic)
    return net(x_hat_ic, t0) - _p_hat_of(cfg.P0, cfg)


def bc_left_residual(net: PinnMLP, cfg: PinnConfig, t_hat_bc: torch.Tensor) -> torch.Tensor:
    """左边界 x_hat=0 的 Dirichlet：p_hat − p_hat(P_left)。"""
    x0 = torch.zeros_like(t_hat_bc)
    return net(x0, t_hat_bc) - _p_hat_of(cfg.P_left, cfg)


def bc_right_residual(net: PinnMLP, cfg: PinnConfig, t_hat_bc: torch.Tensor) -> torch.Tensor:
    """右边界 x_hat=1 的 Dirichlet：p_hat − p_hat(P_right)。"""
    x1 = torch.ones_like(t_hat_bc)
    return net(x1, t_hat_bc) - _p_hat_of(cfg.P_right, cfg)


def total_loss(net: PinnMLP, cfg: PinnConfig,
               x_int: torch.Tensor, t_int: torch.Tensor,
               x_ic: torch.Tensor, t_bc: torch.Tensor,
               ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """加权总 loss + 各分量（分量已 detach，仅供日志）。"""
    r_pde = pde_residual(net, cfg, x_int, t_int)
    r_ic = ic_residual(net, cfg, x_ic)
    r_bcl = bc_left_residual(net, cfg, t_bc)
    r_bcr = bc_right_residual(net, cfg, t_bc)

    L_pde = (r_pde ** 2).mean()
    L_ic = (r_ic ** 2).mean()
    L_bc = (r_bcl ** 2).mean() + (r_bcr ** 2).mean()

    total = cfg.w_pde * L_pde + cfg.w_ic * L_ic + cfg.w_bc * L_bc
    components = {"pde": L_pde.detach(), "ic": L_ic.detach(), "bc": L_bc.detach()}
    return total, components
