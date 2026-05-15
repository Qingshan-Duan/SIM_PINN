from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Well(Protocol):
    """井模型接口。求解器只通过 cell_index、rhs_term、diag_term 与井交互。

    扩展约定（未来 Peaceman 等）：
      - rhs_term(p_old, dt)：返回加到右端项 b[cell_index] 的部分
      - diag_term(dt)     ：返回加到矩阵主对角线 main[cell_index] 的部分
    定流量井：rhs_term = rate, diag_term = 0
    定 BHP 井（Peaceman）：rhs_term = WI/μ · Pwf, diag_term = WI/μ
    """
    cell_index: int

    def rhs_term(self, p_block_old: float, dt: float) -> float: ...
    def diag_term(self, dt: float) -> float: ...


@dataclass(frozen=True)
class ConstantRateWell:
    """定流量井。rate 单位 m^3/s，负值表示采出。"""
    cell_index: int
    rate: float

    def rhs_term(self, p_block_old: float, dt: float) -> float:
        return self.rate

    def diag_term(self, dt: float) -> float:
        return 0.0
