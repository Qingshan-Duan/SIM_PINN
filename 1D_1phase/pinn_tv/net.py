"""把 (x̂, t̂, q̂₁..q̂_K) 映到 p̂ 的全连接网络（时变井控参数化 PINN）。

输入维度 = 2 + n_segments（坐标 2 维 + 整条调度 K 维）。tanh 同理（PDE 要二阶导）。
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


def _fourier_encode(c: torch.Tensor, m: int) -> torch.Tensor:
    """坐标 c (N,1) 的 Fourier 特征：[c, sin(kπc), cos(kπc)]_{k=1..m}，返回 (N, 1+2m)。

    保留原始 c（m=0 即原样返回）。给 MLP 一组基去表示尖锐结构（段边界跳变、早期瞬态），
    缓解谱偏置。纯属网络输入编码，不改物理（源项的段号查表用的是原始 t，见 losses）。
    """
    if m <= 0:
        return c
    k = torch.arange(1, m + 1, device=c.device, dtype=c.dtype) * math.pi   # (m,)
    cf = c * k                                                              # (N, m)
    return torch.cat([c, torch.sin(cf), torch.cos(cf)], dim=1)


class TvPinnMLP(nn.Module):
    """tanh 全连接网络：输入 (x̂, t̂, 调度 q̂[K])，输出 p̂。

    fourier_x / fourier_t > 0 时，对 x̂ / t̂ 加 Fourier 特征（改进②）。t 的频率取到 ~K 段数最匹配
    阶梯结构。hard_ic=True 时 p̂ = t̂·N（用原始 t̂，与编码无关）。默认软约束。
    """

    def __init__(self, n_segments: int, hidden_layers: int, hidden_units: int,
                 hard_ic: bool = False, fourier_x: int = 0, fourier_t: int = 0) -> None:
        super().__init__()
        self.hard_ic = hard_ic
        self.fourier_x = fourier_x
        self.fourier_t = fourier_t
        in_dim = (1 + 2 * fourier_x) + (1 + 2 * fourier_t) + n_segments
        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_units), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_units, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_hat: torch.Tensor, t_hat: torch.Tensor,
                sched: torch.Tensor) -> torch.Tensor:
        """x_hat,t_hat 形状 (N,1)，sched 形状 (N,K)，返回 p_hat (N,1)。"""
        x_e = _fourier_encode(x_hat, self.fourier_x)
        t_e = _fourier_encode(t_hat, self.fourier_t)
        raw = self.net(torch.cat([x_e, t_e, sched], dim=1))
        return t_hat * raw if self.hard_ic else raw
