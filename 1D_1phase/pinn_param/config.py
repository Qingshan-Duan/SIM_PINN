"""参数化 PINN 的配置：在 PinnConfig 基础上加「井流量范围」这一根参数轴。

这是 pinn/ 的升级版（Family 1，纯物理无数据）：网络从 (x̂,t̂)→p̂ 变成 (x̂,t̂,q̂)→p̂，
一个网络覆盖基准流量 well_rate 上下扰动的一整段【定常】流量。物理/无量纲/自适应权重/
eval 网格等字段全部从 PinnConfig 继承，这里只补 q 轴相关的东西。

流量参数化：比率 r = q / well_rate ∈ [q_ratio_min, q_ratio_max]（基准 r=1）。
网络输入用居中归一 q̂ = (r − r_mid)/r_half ∈ [−1, 1]（条件数好）；物理里源强用真实 r·q_nd。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pinn.config import PinnConfig


@dataclass(frozen=True)
class PinnParamConfig(PinnConfig):
    # ---------- 井流量参数化范围（相对基准 well_rate 的比率）----------
    q_ratio_min: float = 0.8         # 基准流量的 80%（采得少 20%）
    q_ratio_max: float = 1.2         # 基准流量的 120%（采得多 20%）

    # ---------- 评估用的一组比率（含未被"强调"的插值点，验证 q 方向真泛化而非记采样点）----------
    eval_q_ratios: tuple = (0.80, 0.85, 0.90, 1.00, 1.10, 1.15, 1.20)

    # ---------- 画图时随机抽一个比率的种子（落盘记录，保证可复现）----------
    plot_q_seed: int = 12345

    # ---------- 派生：比率 ↔ q̂ 的线性映射常数 ----------
    _r_mid: float = field(init=False)
    _r_half: float = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()                       # 先算 alpha_nd / q_nd（基准）
        if not (self.q_ratio_min < self.q_ratio_max):
            raise ValueError(
                f"需要 q_ratio_min < q_ratio_max，收到 [{self.q_ratio_min}, {self.q_ratio_max}]"
            )
        object.__setattr__(self, "_r_mid", 0.5 * (self.q_ratio_min + self.q_ratio_max))
        object.__setattr__(self, "_r_half", 0.5 * (self.q_ratio_max - self.q_ratio_min))

    def ratio_to_qhat(self, r):
        """比率 r → 网络输入 q̂ ∈ [−1,1]。r 可为标量、numpy 或 torch 张量。"""
        return (r - self._r_mid) / self._r_half

    def qhat_to_ratio(self, q_hat):
        """网络输入 q̂ → 比率 r。q_hat 可为标量、numpy 或 torch 张量。"""
        return self._r_mid + q_hat * self._r_half
