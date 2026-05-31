"""1D 单相压力方程的解析解（正弦级数 / 本征模展开），作为「真解」校验 PINN。

控制方程（无量纲，见 config.py）：
    ∂p̂/∂t̂ = α·∂²p̂/∂x̂² + ŝ(x̂)
边界 p̂(0,·)=p̂(1,·)=0（两端同压=p_ref），初值 p̂(·,0)=0，源 ŝ(x̂)=q_nd·g_σ(x̂−x̂_w)。

方程线性 + 齐次 Dirichlet → 用正弦本征函数展开 p̂(x̂,t̂)=Σ_{n≥1} a_n(t̂)·sin(nπx̂)，
模幅满足一阶 ODE
    ȧ_n = −λ_n·a_n + s_n,   λ_n = α(nπ)²,   s_n = 2∫₀¹ ŝ(x̂)·sin(nπx̂) dx̂
**定流量**（源不随时间变 + 零初值）闭式解为
    a_n(t̂) = (s_n/λ_n)·(1 − e^{−λ_n t̂})

时变井控的扩展（留给后续 surrogate / schedule）：把 q 写成分段常数 q_k，逐段递推
    a_n(t) = a_n(t_k)·e^{−λ_n(t−t_k)} + (s_n·q_k/λ_n)·(1 − e^{−λ_n(t−t_k)})
仍是闭式（指数和）。2D 单相同理（二维正弦积），时变照样 Duhamel。
**两相流非线性，本法失效**——那时回退到收敛验证过的细网格模拟器（见 eval._reference_on_eval_grid）。
"""
from __future__ import annotations

import numpy as np

from pinn.config import PinnConfig


def analytical_on_eval_grid(cfg: PinnConfig, n_modes: int = 200) -> np.ndarray:
    """在 eval 网格 (格心 x, eval 时间点 t) 上返回解析压力，形状 (n_steps_eval+1, nx_eval)，单位 Pa。

    n_modes：正弦级数截断模数。源越尖（σ̂ 越小）需要的模数越多；σ̂=0.03 时 ~100 已收敛，默认 200 留余量。
    """
    nx, nt = cfg.nx_eval, cfg.n_steps_eval
    x = (np.arange(nx) + 0.5) / nx                       # x̂ 格心，(nx,)
    t = (np.arange(nt + 1) * cfg.dt_eval) / cfg.T_end    # t̂，(nt+1,)
    n = np.arange(1, n_modes + 1)                        # 模号，(N,)

    # s_n = 2∫₀¹ ŝ·sin(nπx̂) dx̂：细网格数值积分（对任意 σ̂/井位都准）
    xg = np.linspace(0.0, 1.0, 20001)
    g = np.exp(-((xg - cfg.well_x_hat) ** 2) / (2.0 * cfg.well_sigma_hat ** 2)) / (
        cfg.well_sigma_hat * np.sqrt(2.0 * np.pi)
    )
    s_hat = cfg.q_nd * g
    s_n = 2.0 * np.trapezoid(s_hat[None, :] * np.sin(np.outer(n * np.pi, xg)), xg, axis=1)  # (N,)

    lam = cfg.alpha_nd * (n * np.pi) ** 2                # (N,)
    a = (s_n / lam)[None, :] * (1.0 - np.exp(-np.outer(t, lam)))  # (nt+1, N)
    p_hat = a @ np.sin(np.outer(n * np.pi, x))           # (nt+1, N)·(N, nx) → (nt+1, nx)
    return cfg.p_ref + cfg.dp_scale * p_hat
