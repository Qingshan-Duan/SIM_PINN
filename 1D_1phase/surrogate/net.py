"""自回归代理网络：(场ⁿ, q̂ⁿ) → 场ⁿ⁺¹。

输入维度 = nx + 1（整张无量纲场 + 该步无量纲井控），输出 = nx（下一张场）。
离散物理损失在网络外用有限差分算（见 physics.py），不靠 autograd 二阶导，所以激活不必是
tanh；但用 tanh 保持平滑、与 PINN 线一致。残差式输出 p̂ⁿ⁺¹ = 场ⁿ + Δ，让网络只学“增量”
（场变化小，残差连接显著好训）。
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class AutoRegNet(nn.Module):
    def __init__(self, nx: int, hidden_layers: int, hidden_units: int) -> None:
        super().__init__()
        self.nx = nx
        layers: List[nn.Module] = [nn.Linear(nx + 1, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, nx))
        self.net = nn.Sequential(*layers)

    def forward(self, field: torch.Tensor, qhat: torch.Tensor) -> torch.Tensor:
        """field (B, nx), qhat (B, 1) → field_next (B, nx)。残差连接。"""
        delta = self.net(torch.cat([field, qhat], dim=1))
        return field + delta
