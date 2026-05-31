"""模拟器全流程集成测试：守恒性 + 解析参考 + 调度。"""
import numpy as np
import pytest

from simulator.config import BoundarySpec, Config
from simulator.core import run
from simulator.well import (
    BHPWellSpec,
    RateWellSpec,
    make_wells,
    peaceman_pi_1d,
)


def test_equilibrium_stays_equilibrium():
    """无井 + 两侧定压 = P0 + 初始 = P0 → 任意时刻都是 P0。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 2.0e7),
        wells=[],
    )
    history = run(cfg)
    assert np.allclose(history.p, cfg.P0, atol=1e-6)


def test_no_well_steady_state_is_linear_between_boundaries():
    """无井 + 两侧不同定压 → 长时间后应当接近端到端线性插值。"""
    P_L, P_R = 2.0e7, 1.0e7
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", P_L),
        right_bc=BoundarySpec("dirichlet", P_R),
        wells=[],
        n_steps=300,  # 给足时间到稳态
    )
    history = run(cfg)
    p_final = history.p[-1]

    x = (np.arange(cfg.nx) + 0.5) * cfg.dx
    expected = P_L + (P_R - P_L) * x / cfg.L
    assert np.allclose(p_final, expected, rtol=1e-3)


def test_bhp_well_steady_state_q_matches_pi_drawdown():
    """单口 BHP 井稳态：井项 + 左右边界净流入 ≈ 0（质量守恒）。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 2.0e7),
        wells=[BHPWellSpec(cell_index=7, p_wf=1.0e7, kind="producer", rw=0.1)],
        n_steps=300,
    )
    history = run(cfg)
    p_final = history.p[-1]

    pi = peaceman_pi_1d(cfg.k, cfg.A, cfg.dx, cfg.mu, 0.1)
    q_well  = pi * (cfg.wells[0].p_wf - p_final[7])
    q_left  = 2.0 * cfg.T * (2.0e7 - p_final[0])
    q_right = 2.0 * cfg.T * (2.0e7 - p_final[-1])
    assert np.isclose(q_well + q_left + q_right, 0.0, atol=1e-7 * abs(q_well))


def test_mass_balance_every_step_default():
    """每步质量守恒（默认场景，含井控调度）— 离散格式保证，rtol 可极紧。"""
    cfg = Config()
    history = run(cfg)
    for n in range(cfg.n_steps):
        p_old, p_new = history.p[n], history.p[n + 1]
        wells_n = make_wells(cfg, step=n)
        accumulation = cfg.phi * cfg.ct * cfg.V * np.sum(p_new - p_old)
        left  = 2.0 * cfg.T * (cfg.left_bc.pressure  - p_new[0])  * cfg.dt
        right = 2.0 * cfg.T * (cfg.right_bc.pressure - p_new[-1]) * cfg.dt
        well_total = sum(
            cfg.dt * (w.rhs_term(p_old[w.cell_index], cfg.dt)
                      - w.diag_term(cfg.dt) * p_new[w.cell_index])
            for w in wells_n
        )
        assert np.isclose(accumulation, left + right + well_total, rtol=1e-10), (
            f"step {n}: accum={accumulation:.6e}, src={left+right+well_total:.6e}"
        )


def test_injector_balances_producer_keeps_total_mass():
    """等量注采 + 两侧定压 = P0 → 整体压力不漂离 P0。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 2.0e7),
        wells=[
            RateWellSpec(cell_index=3,  rate=+5e-5),
            RateWellSpec(cell_index=11, rate=-5e-5),
        ],
    )
    history = run(cfg)
    p_mean = history.p[-1].mean()
    assert abs(p_mean - cfg.P0) < 5e4   # < 0.05 MPa


def test_bhp_kind_mismatch_raises():
    """声明 producer 但 P_wf > P_cell（变成实际注入）→ run 应当抛 RuntimeError。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 2.0e7),
        wells=[BHPWellSpec(cell_index=7, p_wf=2.5e7, kind="producer", rw=0.1)],
    )
    with pytest.raises(RuntimeError, match="producer"):
        run(cfg)


def test_rate_schedule_takes_per_step_value():
    """rate 列表：step 0 rate=0 → 第 1 步井格压力几乎不动；step 1 起强采 → 之后下降。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 2.0e7),
        wells=[RateWellSpec(cell_index=7, rate=[0.0] + [-1e-4] * 19)],
        n_steps=20,
    )
    history = run(cfg)
    # step 0 用 rate=0：第 1 步井格压力还应接近 P0
    assert abs(history.p[1, 7] - cfg.P0) < 1e2  # 100 Pa 以内
    # step 1 起用 rate=-1e-4：第 2 步起明显下降
    assert history.p[2, 7] < history.p[1, 7] - 1e4
