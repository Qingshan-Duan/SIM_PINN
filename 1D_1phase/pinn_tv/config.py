"""时变井控参数化 PINN 的配置（实验性，可能丢弃）。

把【整条调度曲线】当输入：网络 (x̂, t̂, q̂₁..q̂_K) → p̂，K 段分段常数流量，每段独立扰动 ±20%。
高维输入（2 + K）是必须的——扩散有记忆（Duhamel），瞬时流量不足以定解（病态），见 CLAUDE.md
"历史依赖" 那条。这是 Family 1 的高维尝试；纯物理无数据，看 15 维调度空间能不能靠物理自监督训出来。

段数 K 取 = n_steps_eval，使调度段边界与 eval 时间点对齐（解析解逐段递推时刚好落在格点上）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pinn.config import PinnConfig


@dataclass(frozen=True)
class PinnTvConfig(PinnConfig):
    # ---------- 网络（17 维输入，比单标量难，给宽一点）----------
    hidden_layers: int = 5
    hidden_units: int = 64

    # ---------- 时变调度参数化 ----------
    q_ratio_min: float = 0.8         # 每段流量相对基准的下限（采得少 20%）
    q_ratio_max: float = 1.2         # 每段流量相对基准的上限（采得多 20%）
    n_segments: int = 15             # 调度段数（=每个时间步一个独立扰动），需 == n_steps_eval

    # ---------- 评估 ----------
    n_test_schedules: int = 16       # 随机测试调度数（逐条对解析真解算指标，再汇总）
    schedule_eval_seed: int = 2024   # 测试调度采样种子
    plot_schedule_seed: int = 777    # 画图随机抽一条调度的种子

    # ---------- 派生：比率 ↔ q̂ 线性映射 ----------
    _r_mid: float = field(init=False)
    _r_half: float = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_segments != self.n_steps_eval:
            raise ValueError(
                f"n_segments({self.n_segments}) 需 == n_steps_eval({self.n_steps_eval})，"
                "以保证调度段边界与 eval 时间点对齐（解析解逐段递推落在格点上）"
            )
        object.__setattr__(self, "_r_mid", 0.5 * (self.q_ratio_min + self.q_ratio_max))
        object.__setattr__(self, "_r_half", 0.5 * (self.q_ratio_max - self.q_ratio_min))

    def ratio_to_qhat(self, r):
        """比率 r → 网络输入 q̂ ∈ [−1,1]。支持标量 / numpy / torch。"""
        return (r - self._r_mid) / self._r_half

    def qhat_to_ratio(self, q_hat):
        """网络输入 q̂ → 比率 r。支持标量 / numpy / torch。"""
        return self._r_mid + q_hat * self._r_half
