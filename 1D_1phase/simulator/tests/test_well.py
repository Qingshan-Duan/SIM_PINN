"""井模块单元测试：BHP 项、Peaceman PI、Config→Well 工厂。"""
import math

import pytest

from simulator.config import Config
from simulator.well import (
    BHPWell,
    BHPWellSpec,
    ConstantRateWell,
    RateWellSpec,
    make_wells,
    peaceman_pi_1d,
)


def test_constant_rate_well_contributions():
    w = ConstantRateWell(cell_index=7, rate=-1e-4)
    assert w.rhs_term(p_block_old=2e7, dt=86400.0) == -1e-4
    assert w.diag_term(dt=86400.0) == 0.0


def test_bhp_well_contributions():
    """BHP 井：rhs = PI·P_wf，diag = PI，与 p_old、dt 无关。"""
    w = BHPWell(cell_index=3, p_wf=1.5e7, pi=2.0e-10)
    assert w.rhs_term(p_block_old=2e7, dt=86400.0) == pytest.approx(2.0e-10 * 1.5e7)
    assert w.diag_term(dt=86400.0) == 2.0e-10


def test_peaceman_pi_1d_uses_textbook_formula():
    """PI = 2π·k·√A / (μ·ln(re/rw))，re = 0.14·√(dx² + A)。"""
    k, A, dx, mu, rw = 1e-13, 100.0, 10.0, 5e-3, 0.1
    h = math.sqrt(A)
    re = 0.14 * math.sqrt(dx * dx + A)
    expected = 2.0 * math.pi * k * h / (mu * math.log(re / rw))
    assert peaceman_pi_1d(k, A, dx, mu, rw) == pytest.approx(expected)


def test_make_wells_dispatches_specs():
    """RateWellSpec→ConstantRateWell, BHPWellSpec→BHPWell（且 PI 取自 peaceman_pi_1d）。"""
    cfg = Config(wells=[
        RateWellSpec(cell_index=2, rate=3e-5),
        BHPWellSpec(cell_index=10, p_wf=1.5e7, kind="producer", rw=0.1),
    ])
    wells = make_wells(cfg)
    assert isinstance(wells[0], ConstantRateWell)
    assert wells[0].rate == 3e-5
    assert isinstance(wells[1], BHPWell)
    assert wells[1].p_wf == 1.5e7
    assert wells[1].pi == pytest.approx(peaceman_pi_1d(cfg.k, cfg.A, cfg.dx, cfg.mu, 0.1))
