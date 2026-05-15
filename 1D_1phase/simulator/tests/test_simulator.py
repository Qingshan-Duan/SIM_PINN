"""集成测试：spec 第 7 节定义的三条物理验证。"""
import numpy as np
import pytest

from simulator.config import Config
from simulator.well import ConstantRateWell
from simulator.core import run


def test_steady_state_matches_analytical():
    """100 天 >> 扩散时间尺度，应当达到稳态。

    解析解：
      - 左封闭，所以稳态时 cell 0..well_index 全部同压（含井所在格）
      - 井右侧线性梯度，每相邻 cell 压力差 = -rate · dx · μ / (k · A)
      - 总压降 = -rate · (L - x_well) · μ / (k · A)
    """
    cfg = Config()
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]
    history = run(cfg, wells)
    p_final = history.p[-1]

    # 总压降（正值，单位 Pa）
    total_drop = -cfg.well_rate * (cfg.L - cfg.x_well) * cfg.mu / (cfg.k * cfg.A)
    P_left = cfg.P_right - total_drop

    # 左侧（含井所在格）均匀
    iw = cfg.well_index
    assert np.allclose(p_final[: iw + 1], P_left, rtol=1e-3), (
        f"左半区不均匀: {p_final[: iw + 1]}, 期望 {P_left}"
    )

    # 右侧线性
    drop_per_cell = -cfg.well_rate * cfg.dx * cfg.mu / (cfg.k * cfg.A)
    expected_right = P_left + drop_per_cell * np.arange(1, cfg.nx - iw)
    assert np.allclose(p_final[iw + 1 :], expected_right, rtol=1e-3), (
        f"右半区不线性: {p_final[iw + 1:]}, 期望 {expected_right}"
    )


def test_equilibrium_stays_equilibrium():
    """无井 + 初始 = 右边界 → 任何时刻 P 都应等于 P_0。"""
    cfg = Config()
    history = run(cfg, wells=[])
    assert np.allclose(history.p, cfg.P0, atol=1e-6)


def test_mass_balance_every_step():
    """每个时间步都必须严格质量守恒（离散格式保证，rtol 可极紧）。"""
    cfg = Config()
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]
    history = run(cfg, wells)

    for n in range(cfg.n_steps):
        p_old = history.p[n]
        p_new = history.p[n + 1]
        accumulation = cfg.phi * cfg.ct * cfg.V * np.sum(p_new - p_old)
        boundary_inflow = 2.0 * cfg.T * (cfg.P_right - p_new[-1]) * cfg.dt
        well_total = sum(w.rate for w in wells) * cfg.dt
        rhs = boundary_inflow + well_total
        assert np.isclose(accumulation, rhs, rtol=1e-10), (
            f"步 {n} 质量不守恒: accum={accumulation:.6e}, rhs={rhs:.6e}"
        )
