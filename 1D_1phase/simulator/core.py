from dataclasses import dataclass
from typing import Sequence
import numpy as np

from simulator.config import Config
from simulator.well import Well
from simulator.solver import assemble_and_solve


@dataclass(frozen=True)
class PressureHistory:
    """模拟结果：p[n,i] 是第 n 个时间点（含初始）cell i 的压力。"""
    p: np.ndarray      # shape (n_steps+1, nx), dtype float64, 单位 Pa
    times: np.ndarray  # shape (n_steps+1,),    dtype float64, 单位 s


def run(cfg: Config, wells: Sequence[Well]) -> PressureHistory:
    """执行 cfg.n_steps 个时间步，返回全场压力历史。"""
    p = np.full((cfg.n_steps + 1, cfg.nx), cfg.P0, dtype=np.float64)
    times = np.arange(cfg.n_steps + 1, dtype=np.float64) * cfg.dt
    for n in range(cfg.n_steps):
        p[n + 1] = assemble_and_solve(p[n], cfg, wells)
    return PressureHistory(p=p, times=times)
