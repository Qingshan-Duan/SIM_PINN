"""井模型：Protocol 接口 + 定流量井 + 定流压井 + Config 用的 WellSpec。"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Union, runtime_checkable

# 控制量可以是常数（每一步都一样），或长度为 n_steps 的序列（每步给一个值）
Schedule = Union[float, Sequence[float]]


def _value_at(schedule: Schedule, step: int) -> float:
    """从标量/序列里取第 step 步的值。"""
    if hasattr(schedule, "__len__") and not isinstance(schedule, (str, bytes)):
        return float(schedule[step])  # type: ignore[index]
    return float(schedule)  # type: ignore[arg-type]


@runtime_checkable
class Well(Protocol):
    """求解器侧的井接口。求解器只通过 cell_index、rhs_term、diag_term 与井交互。

    扩展约定：
      - rhs_term(p_old, dt)：返回加到右端项 b[cell_index] 的部分
      - diag_term(dt)     ：返回加到矩阵主对角线 main[cell_index] 的部分
    定流量井：rhs_term = rate,        diag_term = 0
    定流压井：rhs_term = PI · P_wf,   diag_term = PI
    """
    cell_index: int

    def rhs_term(self, p_block_old: float, dt: float) -> float: ...
    def diag_term(self, dt: float) -> float: ...


@dataclass(frozen=True)
class ConstantRateWell:
    """定流量井。rate 单位 m^3/s，正值=注入，负值=采出。"""
    cell_index: int
    rate: float

    def rhs_term(self, p_block_old: float, dt: float) -> float:
        return self.rate

    def diag_term(self, dt: float) -> float:
        return 0.0


@dataclass(frozen=True)
class BHPWell:
    """定流压井。q = PI · (P_wf − P_cell)。PI 单位 m^3/(s·Pa)。"""
    cell_index: int
    p_wf: float
    pi: float

    def rhs_term(self, p_block_old: float, dt: float) -> float:
        return self.pi * self.p_wf

    def diag_term(self, dt: float) -> float:
        return self.pi


# --- Config 用的轻量井规格（不含派生量）---

@dataclass(frozen=True)
class RateWellSpec:
    """Config 中的定流量井规格。

    rate 可以是标量（每步同值），也可以是长度等于 n_steps 的序列（每步一个值）。
    正=注入，负=采出，单位 m^3/s。
    """
    cell_index: int
    rate: Schedule


@dataclass(frozen=True)
class BHPWellSpec:
    """Config 中的定流压井规格。PI 由 rw 通过 1D 类 Peaceman 公式算出。

    p_wf 可以是标量或长度等于 n_steps 的序列（与 RateWellSpec.rate 同样的语义）。
    kind 必须显式声明 producer / injector；运行期发现 sign(q) 与 kind 不一致会终止模拟。
    """
    cell_index: int
    p_wf: Schedule                                 # Pa
    kind: Literal["producer", "injector"]
    rw: float = 0.1                                # m，井筒半径

    def __post_init__(self) -> None:
        if self.kind not in ("producer", "injector"):
            raise ValueError(
                f"BHPWellSpec.kind 必须是 'producer' 或 'injector'，收到 {self.kind!r}"
            )


WellSpec = Union[RateWellSpec, BHPWellSpec]


def peaceman_pi_1d(k: float, A: float, dx: float, mu: float, rw: float) -> float:
    """1D 类 Peaceman 产能指数。

    把网格视作 dx × √A × √A 的 3D 盒子，井沿 √A 方向贯穿。
    标准 2D Peaceman:
        WI = 2π · k · h / ln(re/rw),   re = 0.14·√(dx² + A)
        PI = WI / μ
    """
    h = math.sqrt(A)
    re = 0.14 * math.sqrt(dx * dx + A)
    return 2.0 * math.pi * k * h / (mu * math.log(re / rw))


def make_wells(cfg, step: int = 0) -> list[Well]:
    """根据 cfg.wells 实例化第 `step` 步的运行时井对象（取出该步的调度值）。"""
    out: list[Well] = []
    for spec in cfg.wells:
        if isinstance(spec, RateWellSpec):
            out.append(ConstantRateWell(
                cell_index=spec.cell_index,
                rate=_value_at(spec.rate, step),
            ))
        elif isinstance(spec, BHPWellSpec):
            pi = peaceman_pi_1d(cfg.k, cfg.A, cfg.dx, cfg.mu, spec.rw)
            out.append(BHPWell(
                cell_index=spec.cell_index,
                p_wf=_value_at(spec.p_wf, step),
                pi=pi,
            ))
        else:
            raise TypeError(f"未知的 WellSpec 类型: {type(spec).__name__}")
    return out
