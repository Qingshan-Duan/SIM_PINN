"""把 (x_hat, t_hat) 映到 p_hat 的全连接网络。"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class PinnMLP(nn.Module):
    """tanh 全连接网络：输入 (x_hat, t_hat)，输出 p_hat。

    选 tanh 而不是 ReLU，是因为 PDE 残差要用到二阶导 ∂²p/∂x²；ReLU 的二阶导处处为 0，
    autograd 求出来会是 0，网络学不到曲率。tanh 光滑、任意阶可导，且输出 [-1,1] 数值稳定。
    """

    def __init__(self, hidden_layers: int, hidden_units: int) -> None:
        super().__init__()
        layers: List[nn.Module] = [nn.Linear(2, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_hat: torch.Tensor, t_hat: torch.Tensor) -> torch.Tensor:
        """x_hat, t_hat 形状均为 (N, 1)，返回 p_hat 形状 (N, 1)。"""
        return self.net(torch.cat([x_hat, t_hat], dim=1))
