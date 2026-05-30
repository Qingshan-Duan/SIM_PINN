"""时变（分段常数）井控的解析真解：正弦级数 + 逐段递推（Duhamel）。

控制方程（无量纲）：∂p̂/∂t̂ = α·∂²p̂/∂x̂² + r(t̂)·ŝ₀(x̂)，
其中 ŝ₀(x̂)=q_nd·g_σ(x̂−x̂_w) 是基准流量(r=1)的源形状，r(t̂) 分段常数（第 k 段 = r_k）。

正弦展开 p̂=Σ a_n sin(nπx̂)，模幅 ODE：ȧ_n = −λ_n a_n + r(t̂)·s_n⁰，λ_n=α(nπ)²，
s_n⁰ = 2∫ŝ₀ sin(nπx̂)dx̂。分段常数 r → 逐段闭式递推（段长 Δ̂）：
    a_n(t_{j+1}) = a_n(t_j)·e^{−λ_n Δ̂} + (r_j·s_n⁰/λ_n)·(1 − e^{−λ_n Δ̂})
段边界与 eval 时间点对齐（n_segments==n_steps_eval），故 a_n 在每个 eval 点上恰好是递推值。
线性区精确；两相非线性时本法失效。
"""
from __future__ import annotations

import numpy as np

from pinn_tv.config import PinnTvConfig


def analytical_schedule_on_eval_grid(cfg: PinnTvConfig, ratios: np.ndarray,
                                     n_modes: int = 200) -> np.ndarray:
    """给定调度（K 个段流量比率 ratios，形状 (K,)），返回 eval 网格压力 (n_steps+1, nx)，Pa。"""
    nx, nt = cfg.nx_eval, cfg.n_steps_eval
    K = cfg.n_segments
    x = (np.arange(nx) + 0.5) / nx                       # x̂ 格心
    n = np.arange(1, n_modes + 1)                        # 模号

    # s_n⁰：基准流量源形状的正弦系数（细网格数值积分）
    xg = np.linspace(0.0, 1.0, 20001)
    g = np.exp(-((xg - cfg.well_x_hat) ** 2) / (2.0 * cfg.well_sigma_hat ** 2)) / (
        cfg.well_sigma_hat * np.sqrt(2.0 * np.pi)
    )
    s0 = cfg.q_nd * g
    s_n0 = 2.0 * np.trapezoid(s0[None, :] * np.sin(np.outer(n * np.pi, xg)), xg, axis=1)  # (N,)

    lam = cfg.alpha_nd * (n * np.pi) ** 2                # (N,)
    dhat = cfg.dt_eval / cfg.T_end                       # 段长（无量纲），= 1/K
    decay = np.exp(-lam * dhat)                          # (N,)
    gain = (s_n0 / lam) * (1.0 - decay)                  # (N,) 每段满流量(r=1)的增量

    a = np.zeros(n_modes)                                # a_n(t_0)=0
    A = [a.copy()]
    for j in range(nt):                                  # 逐段递推（nt==K）
        r_j = float(ratios[j])
        a = a * decay + r_j * gain
        A.append(a.copy())
    A = np.array(A)                                      # (nt+1, N)

    p_hat = A @ np.sin(np.outer(n * np.pi, x))           # (nt+1, nx)
    return cfg.p_ref + cfg.dp_scale * p_hat
