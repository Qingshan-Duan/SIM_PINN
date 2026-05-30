"""把 (x̂, t̂, q̂₁..q̂_K) 映到 p̂ 的全连接网络（时变井控参数化 PINN）。

输入维度 = 2 + n_segments（坐标 2 维 + 整条调度 K 维）。tanh 同理（PDE 要二阶导）。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class TvPinnMLP(nn.Module):
    """tanh 全连接网络：输入 (x̂, t̂, 调度 q̂[K])，输出 p̂。

    hard_ic=True 时 p̂ = t̂·N（t=0 处恒 0，初值对任意调度精确）。默认软约束。
    """

    def __init__(self, n_segments: int, hidden_layers: int, hidden_units: int,
                 hard_ic: bool = False) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        in_dim = 2 + n_segments
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_hat: torch.Tensor, t_hat: torch.Tensor,
                sched: torch.Tensor) -> torch.Tensor:
        """x_hat,t_hat 形状 (N,1)，sched 形状 (N,K)，返回 p_hat (N,1)。"""
        raw = self.net(torch.cat([x_hat, t_hat, sched], dim=1))
        return t_hat * raw if self.hard_ic else raw
