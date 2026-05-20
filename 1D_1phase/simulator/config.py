"""1D 单相模拟器全部配置（SI 单位）。多井 + 可独立设置左右边界 + 每步井控调度。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal

from simulator.well import BHPWellSpec, RateWellSpec, WellSpec


# 默认 20 步 BHP 调度：17 MPa 基准 ± 1 MPa，正弦两个周期。
# 始终远低于初始/边界压力（≥19 MPa），保证 sign(q)<0（采出），不会触发 kind 校验。
DEFAULT_BHP_SCHEDULE = (
    1.70e7, 1.76e7, 1.80e7, 1.80e7, 1.76e7,
    1.70e7, 1.64e7, 1.60e7, 1.60e7, 1.64e7,
    1.70e7, 1.76e7, 1.80e7, 1.80e7, 1.76e7,
    1.70e7, 1.64e7, 1.60e7, 1.60e7, 1.64e7,
)


@dataclass(frozen=True)
class BoundarySpec:
    """边界条件：定压(dirichlet) 或 封闭(noflow)。"""
    kind: Literal["dirichlet", "noflow"] = "dirichlet"
    pressure: float = 0.0   # 仅在 kind == "dirichlet" 时使用，单位 Pa

    def __post_init__(self) -> None:
        if self.kind not in ("dirichlet", "noflow"):
            raise ValueError(f"边界类型必须是 'dirichlet' 或 'noflow'，收到 {self.kind!r}")


def _default_wells() -> List[WellSpec]:
    return [BHPWellSpec(
        cell_index=7,
        p_wf=DEFAULT_BHP_SCHEDULE,
        kind="producer",
        rw=0.1,
    )]


def _schedule_len(sched) -> int | None:
    """如果 sched 是序列返回长度，标量返回 None。"""
    if hasattr(sched, "__len__") and not isinstance(sched, (str, bytes)):
        return len(sched)
    return None


@dataclass(frozen=True)
class Config:
    # 离散
    nx: int = 15
    L: float = 150.0          # m
    A: float = 100.0          # m^2

    # 物性（均匀常数）
    k: float = 1e-13          # m^2  (≈100 mD)
    mu: float = 5e-3          # Pa·s (≈5 cp)
    phi: float = 0.2          # -
    ct: float = 1e-9          # Pa^-1

    # 初始压力
    P0: float = 2e7           # Pa

    # 边界（默认两侧都定压；左 20 MPa，右 19 MPa，有 1 MPa 整体压降）
    left_bc:  BoundarySpec = field(default_factory=lambda: BoundarySpec("dirichlet", 2.0e7))
    right_bc: BoundarySpec = field(default_factory=lambda: BoundarySpec("dirichlet", 1.9e7))

    # 时间
    dt: float = 86400.0       # s   (1 day)
    n_steps: int = 20

    # 井（可后于 n_steps 出现，但 __post_init__ 里要按 n_steps 校验调度长度）
    wells: List[WellSpec] = field(default_factory=_default_wells)

    # 派生量
    dx: float = field(init=False)
    V: float = field(init=False)
    T: float = field(init=False)

    def __post_init__(self) -> None:
        for w in self.wells:
            if not 0 <= w.cell_index < self.nx:
                raise ValueError(
                    f"井 cell_index={w.cell_index} 超出 [0, {self.nx})"
                )
            sched = w.rate if isinstance(w, RateWellSpec) else w.p_wf
            n = _schedule_len(sched)
            if n is not None and n != self.n_steps:
                raise ValueError(
                    f"井 cell_index={w.cell_index} 的调度长度 {n} != n_steps {self.n_steps}"
                )
        object.__setattr__(self, "dx", self.L / self.nx)
        object.__setattr__(self, "V", self.A * self.dx)
        object.__setattr__(self, "T", self.k * self.A / (self.mu * self.dx))
