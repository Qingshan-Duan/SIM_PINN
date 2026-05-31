"""把 (x̂, t̂, q̂) 映到 p̂ 的全连接网络（参数化 PINN）。

与 pinn.net.PinnMLP 唯一的区别是输入维度：这里 3 维（多一根流量轴 q̂）。
PinnMLP 把输入维度写死成 2，故另写一个而不是改它，保持 pinn/ 不被动。
选 tanh 同理：PDE 残差要二阶导 ∂²p̂/∂x̂²，ReLU 二阶导恒 0。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ParamPinnMLP(nn.Module):
    """tanh 全连接网络：输入 (x̂, t̂, q̂)，输出 p̂。

    hard_ic=True 时用硬约束 ansatz p̂ = t̂ · N_θ(x̂, t̂, q̂)，使 t=0 处输出恒 0（初值精确）。
    本场景初值与 q 无关（恒 p_ref），故硬约束对所有 q 同时成立。默认软约束（见 config 注释）。
    """

    def __init__(self, hidden_layers: int, hidden_units: int,
                 hard_ic: bool = False) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        layers: List[nn.Module] = [nn.Linear(3, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_hat: torch.Tensor, t_hat: torch.Tensor,
                q_hat: torch.Tensor) -> torch.Tensor:
        """x_hat, t_hat, q_hat 形状均为 (N, 1)，返回 p_hat 形状 (N, 1)。"""
        raw = self.net(torch.cat([x_hat, t_hat, q_hat], dim=1))
        return t_hat * raw if self.hard_ic else raw
