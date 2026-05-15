from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    """1D 单相模拟器全部参数。SI 单位。"""

    # 离散
    nx: int = 15
    L: float = 150.0          # m
    A: float = 100.0          # m^2

    # 物性（均匀常数）
    k: float = 1e-13          # m^2  (≈100 mD)
    mu: float = 5e-3          # Pa·s (≈5 cp)
    phi: float = 0.2          # -
    ct: float = 1e-9          # Pa^-1

    # 初始/边界
    P0: float = 2e7           # Pa  (初始压力)
    P_right: float = 2e7      # Pa  (右边界定压)

    # 井
    well_index: int = 7
    well_rate: float = -1e-4  # m^3/s, 负=采出

    # 时间
    dt: float = 86400.0       # s   (1 day)
    n_steps: int = 100

    # 派生量
    dx: float = field(init=False)
    V: float = field(init=False)
    T: float = field(init=False)
    x_well: float = field(init=False)

    def __post_init__(self):
        assert 0 <= self.well_index < self.nx, (
            f"well_index={self.well_index} 超出 [0, {self.nx})"
        )
        object.__setattr__(self, "dx", self.L / self.nx)
        object.__setattr__(self, "V", self.A * self.dx)
        object.__setattr__(self, "T", self.k * self.A / (self.mu * self.dx))
        object.__setattr__(self, "x_well", (self.well_index + 0.5) * self.dx)
