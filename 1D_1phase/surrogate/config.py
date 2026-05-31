"""代理模型超参 + 无量纲化常数（与 simulator 粗步长场景逐字对齐）。

物理场景与 pinn_tv 一致：两端定压 20 MPa（井是唯一驱动），中心 cell 7 一口定流量井，
基准流量 −1e-4，每个时间步独立扰动 ±20%，15 步 × 1 天。**但井是原生单格**（不是高斯），
因为物理损失用的是模拟器的离散后向欧拉方程，点井在离散格点上就是单格源项，无 δ 奇点。

无量纲化（网络只看无量纲量）：
    p̂ = (P − P0) / dp_scale          场值（边界 = 0，井稳态压降 ≈ −1.875）
    q̂ = (r − r_mid) / r_half         井控比率 r∈[0.8,1.2] 线性映到 [−1,1]
离散残差也无量纲化到 O(1)（见 physics.py 的推导），残差里要用的两个常数：
    D       = T/α       每步无量纲扩散数（≈86.4）
    s_scale = 1/(α·dp_scale)   把物理流量 q 换成无量纲源 ŝ = q·s_scale
其中 α = φ·ct·V/dt（后向欧拉的时间项系数），T = k·A/(μ·dx)（面传导率）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SurrogateConfig:
    # ---------- 物理 / 几何（与 simulator 默认物性一致） ----------
    nx: int = 15
    L: float = 150.0                 # m
    A: float = 100.0                 # m^2
    k: float = 1e-13                 # m^2   (≈100 mD)
    mu: float = 5e-3                 # Pa·s  (≈5 cp)
    phi: float = 0.2                 # -
    ct: float = 1e-9                 # Pa^-1
    P0: float = 2e7                  # Pa    初始压力 20 MPa
    P_left: float = 2.0e7            # Pa    左 Dirichlet 20 MPa
    P_right: float = 2.0e7           # Pa    右 Dirichlet 20 MPa（两端同压）
    dt: float = 86400.0              # s     1 天（粗步长，数据与物理损失共用）
    n_steps: int = 15                # 时间步数（= 调度段数）

    # ---------- 井（中心定流量生产井，原生单格） ----------
    well_cell: int = 7               # 井所在 cell（域中心）
    well_rate_base: float = -1.0e-4  # m^3/s 基准流量，负=采出

    # ---------- 调度参数化（每步独立扰动 ±20%） ----------
    q_ratio_min: float = 0.8
    q_ratio_max: float = 1.2

    # ---------- 无量纲化常数 ----------
    p_ref: float = 2e7               # Pa  取作 P0
    dp_scale: float = 1e6            # Pa  压降量级（井稳态压降 ≈1.875 MPa）

    # ---------- 网络（场→场，离散物理无需 autograd 二阶导） ----------
    hidden_layers: int = 4
    hidden_units: int = 128

    # ---------- 训练（全批梯度：数据量小，不分 mini-batch，一次 iter = 一步全批梯度）----------
    n_iters: int = 4000              # Adam 迭代步数（= 全批梯度步；对照 pinn 家族 20k–40k，这里很克制）
    lr: float = 1e-3
    lr_decay_every: int = 1500       # 每多少步学习率 ×0.5
    seed: int = 0

    # ---------- 物理损失 ----------
    use_data: bool = True            # False = 物理-only（不要数据 MSE，只在场状态上施加 BE 残差）
    lambda_phys: float = 0.0         # 0 = 纯数据；>0 = 数据 + 物理正则（物理-only 时为残差总权重）
    # 物理 collocation 状态来源：
    #   "data"      只在训练转移对的 field^n 上施加（最弱，点与数据重合）
    #   "augmented" 额外在“扰动/插值出来的场 + 自由采样 q”上施加（填满 (场,q) 空间，最强）
    phys_states: str = "augmented"
    n_phys: int = 4096               # augmented 模式下每轮采的物理 collocation 状态数
    phys_perturb: float = 0.15       # 对已知场做高斯扰动的相对幅度（augmented）

    # ---------- 数据划分（由 data_gen 落盘，这里记录约定） ----------
    n_train_schedules: int = 16      # 主实验用的训练调度条数（由 sweep 定）
    n_val_schedules: int = 32        # 验证（held-out，调超参/早停参考）
    n_test_schedules: int = 256      # 测试（held-out，最终泛化指标）

    # ---------- 派生量 ----------
    dx: float = field(init=False)
    V: float = field(init=False)
    T: float = field(init=False)
    alpha: float = field(init=False)     # φ·ct·V/dt
    D: float = field(init=False)         # T/α，每步无量纲扩散数
    s_scale: float = field(init=False)   # 1/(α·dp_scale)，q → ŝ
    _r_mid: float = field(init=False)
    _r_half: float = field(init=False)

    def __post_init__(self) -> None:
        dx = self.L / self.nx
        V = self.A * dx
        T = self.k * self.A / (self.mu * dx)
        alpha = self.phi * self.ct * V / self.dt
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "V", V)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "D", T / alpha)
        object.__setattr__(self, "s_scale", 1.0 / (alpha * self.dp_scale))
        object.__setattr__(self, "_r_mid", 0.5 * (self.q_ratio_min + self.q_ratio_max))
        object.__setattr__(self, "_r_half", 0.5 * (self.q_ratio_max - self.q_ratio_min))

    # --- 比率 ↔ q̂ 线性映射 ---
    def ratio_to_qhat(self, r):
        return (r - self._r_mid) / self._r_half

    def qhat_to_ratio(self, q_hat):
        return self._r_mid + q_hat * self._r_half
