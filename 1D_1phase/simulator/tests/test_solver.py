import numpy as np
import pytest
from simulator.config import Config
from simulator.well import ConstantRateWell
from simulator.solver import assemble_and_solve


def test_no_well_uniform_initial_at_boundary_stays_uniform():
    """无井 + 初始压力 = 右边界压力 → 一步之后仍均匀。"""
    cfg = Config()
    p_old = np.full(cfg.nx, cfg.P_right)
    p_new = assemble_and_solve(p_old, cfg, wells=[])
    assert np.allclose(p_new, cfg.P_right, atol=1e-6)


def test_step_returns_correct_shape_and_dtype():
    cfg = Config()
    p_old = np.full(cfg.nx, cfg.P0)
    p_new = assemble_and_solve(p_old, cfg, wells=[])
    assert p_new.shape == (cfg.nx,)
    assert p_new.dtype == np.float64


def test_well_decreases_pressure_at_well_cell():
    """开井一步，井所在格压力应当下降。"""
    cfg = Config()
    p_old = np.full(cfg.nx, cfg.P0)
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]
    p_new = assemble_and_solve(p_old, cfg, wells=wells)
    assert p_new[cfg.well_index] < cfg.P0
    # 远端（cell 0）由于扩散很快也会有点变化，但应小于井所在格
    assert p_new[0] >= p_new[cfg.well_index]


def test_single_step_mass_balance():
    """单步质量守恒：φ·ct·V·Σ(p_new-p_old) = dt·(右边界净流入 + Σ井流量)。"""
    cfg = Config()
    p_old = np.full(cfg.nx, cfg.P0)
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]
    p_new = assemble_and_solve(p_old, cfg, wells=wells)

    accumulation = cfg.phi * cfg.ct * cfg.V * np.sum(p_new - p_old)
    boundary_inflow = 2 * cfg.T * (cfg.P_right - p_new[-1]) * cfg.dt
    well_total = sum(w.rate for w in wells) * cfg.dt
    rhs = boundary_inflow + well_total

    assert np.isclose(accumulation, rhs, rtol=1e-10)
