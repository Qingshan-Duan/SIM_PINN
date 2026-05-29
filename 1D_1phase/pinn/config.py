"""PINN 的全部超参 + 无量纲化常数。

无量纲化约定（带 hat 的量都是无量纲，网络只看无量纲）：
    x_hat = x / L            ∈ [0, 1]
    t_hat = t / T_end        ∈ [0, 1]
    p_hat = (P - p_ref) / dp_scale

代入控制方程 φ·ct·∂P/∂t = (k/μ)·∂²P/∂x² + (q/A)·δ(x−x_well) 后，无量纲形式为
    ∂p_hat/∂t_hat = alpha_nd · ∂²p_hat/∂x_hat² + s_hat(x_hat)
其中
    alpha_nd = k·T_end / (μ·φ·ct·L²)                  无量纲扩散系数，O(1)（本场景 ≈5.76）
    s_hat    = q_nd · g_σ(x_hat − well_x_hat)          无量纲源项（井）
    q_nd     = q·T_end / (A·L·φ·ct·dp_scale)           无量纲源强（producer<0）
g_σ 是单位积分高斯（半宽 well_sigma_hat），把点源正则化掉 δ 函数，便于 PINN 自动微分。
无量纲化是 PINN 能训得动的关键：原始 SI 量级（t~1e6、P~1e7）会让梯度爆炸/消失。

场景：两端同压 20 MPa（井是唯一驱动力），中心一口定流量生产井。物理参数与 simulator
默认场景一致；这里显式重申而不是 import simulator.Config，是为了让 PINN 这条线自洽可单独调。
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
    P_right: float = 2.0e7           # Pa    右 Dirichlet 20 MPa（两端同压：井是唯一驱动力）
    T_end: float = 15 * 86400.0      # s     15 天，覆盖瞬态(达稳态约 8 天=3τ)+稳态平台

    # ---------- 井（中心定流量生产井） ----------
    well_x_hat: float = 0.5          # -     井位（无量纲），0.5=域中心=cell 7
    well_rate: float = -1.0e-4       # m^3/s 流量，负=采出(producer)；稳态井底压降≈1.875 MPa
    well_sigma_hat: float = 0.03     # -     高斯点源无量纲半宽（≈井所在 cell 尺度）

    # ---------- 无量纲化常数 ----------
    p_ref: float = 2e7               # Pa    取作 P0
    dp_scale: float = 1e6            # Pa    压降量级（井稳态压降 ≈1.875 MPa，取 1 MPa 量级）

    # ---------- 网络架构 ----------
    hidden_layers: int = 4           # 隐层数
    hidden_units: int = 50           # 每层神经元
    hard_ic: bool = True             # True: 硬约束 p_hat=t_hat·N（IC 精确满足，不用 IC 罚项）

    # ---------- 训练采样（每个量都是每轮重新采的点数） ----------
    n_int: int = 5000                # 内部 PDE 残差采点（全域均匀）
    n_int_well: int = 1500           # 井附近额外加密采点（源项局部、梯度陡，必须采够）
    well_band_hat: float = 0.12      # 井附近加密的半带宽（在 [x_well±band] 内采）
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
    n_steps_eval: int = 15
    dt_eval: float = 86400.0         # s  (15 步 × 1 天 = 15 天 = T_end)
    # 参考解用细 dt（每个 eval 步切成 ref_substeps 个子步）跑 simulator，再在 eval 时间点取样。
    # 否则 dt=1天 的后向欧拉在早期瞬态有 ~0.1 MPa 时间离散误差，会被误当成 PINN 的误差。
    ref_substeps: int = 100
    # 参考解空间也细化 ref_space_refine 倍（井附近梯度陡，nx=15 截断误差大）。
    # 必须取奇数：井保持在格心、且细网格格心是粗网格格心的超集，算完能整齐降采样回 nx_eval。
    # 并且参考解的井按 PINN 同一个高斯（well_sigma_hat）摊到各细格上，让两边解「同一个井」。
    ref_space_refine: int = 5

    # ---------- 随机种子 ----------
    seed: int = 0

    # ---------- 派生量 ----------
    alpha_nd: float = field(init=False)   # 无量纲扩散系数
    q_nd: float = field(init=False)       # 无量纲源强（井）

    def __post_init__(self) -> None:
        if self.ref_space_refine < 1 or self.ref_space_refine % 2 == 0:
            raise ValueError(
                f"ref_space_refine 必须是正奇数（保证井在格心、降采样对齐），收到 {self.ref_space_refine}"
            )
        eta = self.k / (self.mu * self.phi * self.ct)       # 扩散系数 m^2/s
        alpha = eta * self.T_end / (self.L ** 2)            # 无量纲扩散系数
        object.__setattr__(self, "alpha_nd", alpha)
        # 无量纲源强：q·T_end / (A·L·φ·ct·dp_scale)，由 (q/A)·δ 无量纲化而来
        q_nd = self.well_rate * self.T_end / (
            self.A * self.L * self.phi * self.ct * self.dp_scale
        )
        object.__setattr__(self, "q_nd", q_nd)

    def well_cell_index(self, nx: int) -> int:
        """井所在 cell 序号（格心最接近 well_x_hat·L 的那格）。"""
        return int(self.well_x_hat * nx)
