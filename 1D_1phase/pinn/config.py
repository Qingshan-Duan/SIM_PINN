"""PINN 的全部超参 + 无量纲化常数。

无量纲化约定（带 hat 的量都是无量纲，网络只看无量纲）：
    x_hat = x / L            ∈ [0, 1]
    t_hat = t / T_end        ∈ [0, 1]
    p_hat = (P - p_ref) / dp_scale

代入控制方程 φ·ct·∂P/∂t = (k/μ)·∂²P/∂x² 后，无量纲形式为
    ∂p_hat/∂t_hat = alpha_nd · ∂²p_hat/∂x_hat²
其中 alpha_nd = k·T_end / (μ·φ·ct·L²)，是个 O(1) 量级的纯数（默认场景 ≈ 7.7）。
这一步是 PINN 能训得动的关键：原始 SI 量级（t~1e6、P~1e7）会让梯度爆炸/消失。

物理参数默认值与 simulator.Config 的默认场景（无井 + 两端定压）保持一致；
这里显式重申而不是 import simulator.Config，是为了让 PINN 这条线的超参自洽、可单独调。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PinnConfig:
    # ---------- 物理 / 几何（与 simulator 默认场景一致） ----------
    L: float = 150.0                 # m
    A: float = 100.0                 # m^2
    k: float = 1e-13                 # m^2   (≈100 mD)
    mu: float = 5e-3                 # Pa·s  (≈5 cp)
    phi: float = 0.2                 # -
    ct: float = 1e-9                 # Pa^-1
    P0: float = 2e7                  # Pa    初始压力 20 MPa
    P_left: float = 2.0e7            # Pa    左 Dirichlet 20 MPa
    P_right: float = 1.9e7           # Pa    右 Dirichlet 19 MPa
    T_end: float = 4 * 86400.0       # s     4 天，聚焦早期瞬态（稳态约 2 天就到，再往后无信息）

    # ---------- 无量纲化常数 ----------
    p_ref: float = 2e7               # Pa    取作 P0
    dp_scale: float = 1e6            # Pa    压差量级（P_left - P_right = 1 MPa）

    # ---------- 网络架构 ----------
    hidden_layers: int = 4           # 隐层数
    hidden_units: int = 50           # 每层神经元

    # ---------- 训练采样（每个量都是每轮重新采的点数） ----------
    n_int: int = 5000                # 内部 PDE 残差采点
    n_ic: int = 200                  # 初始条件采点（在 x 上）
    n_bc: int = 200                  # 每条边界采点（在 t 上）

    # ---------- 优化 ----------
    adam_iters: int = 20000
    adam_lr: float = 1e-3
    lbfgs_iters: int = 1500

    # ---------- loss 权重（手调起步：约束项给更大权重） ----------
    w_pde: float = 1.0
    w_ic: float = 10.0
    w_bc: float = 10.0

    # ---------- 评估网格（与 simulator 默认一致，方便逐点对照） ----------
    nx_eval: int = 15
    n_steps_eval: int = 4
    dt_eval: float = 86400.0         # s  (4 步 × 1 天 = 4 天 = T_end)
    # 参考解用细 dt（每个 eval 步切成 ref_substeps 个子步）跑 simulator，再在 eval 时间点取样。
    # 否则 dt=1天 的后向欧拉在早期瞬态有 ~0.1 MPa 时间离散误差，会被误当成 PINN 的误差。
    ref_substeps: int = 100

    # ---------- 随机种子 ----------
    seed: int = 0

    # ---------- 派生量 ----------
    alpha_nd: float = field(init=False)

    def __post_init__(self) -> None:
        eta = self.k / (self.mu * self.phi * self.ct)       # 扩散系数 m^2/s
        alpha = eta * self.T_end / (self.L ** 2)            # 无量纲扩散系数
        object.__setattr__(self, "alpha_nd", alpha)
