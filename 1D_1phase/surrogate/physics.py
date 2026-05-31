"""离散后向欧拉残差（torch，可对网络输出求导），无量纲化到 O(1)。

模拟器一步 BE（物理量，每个内部格）：
    α(Pⁿ⁺¹ − Pⁿ) − T(Pⁿ⁺¹ᵢ₋₁ − 2Pⁿ⁺¹ᵢ + Pⁿ⁺¹ᵢ₊₁) − qᵢ = 0
两端 Dirichlet（面距格心 dx/2，面传导率 2T）：边界格主对角 α+3T、右端 +2T·P_bc。
定流量井 diag_term=0 ⇒ 矩阵 A 只含几何、为常数，只有右端 b 随 (Pⁿ, q) 变。

无量纲化：P = P0 + dp·p̂，整条方程除以 α 再除以 dp（推导见 NOTES.md）。因 P_bc = P0 ⇒ p̂_bc=0，
边界项与 P0 项整齐抵消，得无量纲残差（再除以对角系数 1+2D 让量级 ~O(1)）：
    Â p̂ⁿ⁺¹ = p̂ⁿ⁺¹ 的离散算子；  r̂ = (Â p̂ⁿ⁺¹ − p̂ⁿ − ŝ) / (1+2D)
    Â 行：内部 (1+2D)p̂ᵢ − D p̂ᵢ₋₁ − D p̂ᵢ₊₁；边界 (1+3D)p̂₀ − D p̂₁（另一端对称）
    D = T/α，  ŝ[well] = q·s_scale = q/(α·dp)，其余 0
模拟器自己的输出令 r̂ = 0（到机器精度），所以“数据 + 物理”指向同一函数、不打架。
"""
from __future__ import annotations

import torch

from surrogate.config import SurrogateConfig


def be_operator(p_new: torch.Tensor, D: float) -> torch.Tensor:
    """Â p̂ⁿ⁺¹：离散 BE 空间算子。p_new (B, nx) → (B, nx)。两端 Dirichlet。"""
    out = torch.empty_like(p_new)
    out[:, 1:-1] = (1.0 + 2.0 * D) * p_new[:, 1:-1] - D * p_new[:, :-2] - D * p_new[:, 2:]
    out[:, 0] = (1.0 + 3.0 * D) * p_new[:, 0] - D * p_new[:, 1]
    out[:, -1] = (1.0 + 3.0 * D) * p_new[:, -1] - D * p_new[:, -2]
    return out


def be_residual(p_new: torch.Tensor, p_old: torch.Tensor, q_phys: torch.Tensor,
                cfg: SurrogateConfig) -> torch.Tensor:
    """无量纲离散 BE 残差 r̂，形状 (B, nx)。

    p_new / p_old：无量纲场 (B, nx)；q_phys：该步物理流量 (B, 1) 或 (B,)，m^3/s。
    """
    D = cfg.D
    s = torch.zeros_like(p_new)
    s[:, cfg.well_cell] = q_phys.reshape(-1) * cfg.s_scale
    return (be_operator(p_new, D) - p_old - s) / (1.0 + 2.0 * D)
