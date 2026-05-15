import pytest
from simulator.well import Well, ConstantRateWell


def test_constant_rate_well_construction():
    w = ConstantRateWell(cell_index=7, rate=-1e-4)
    assert w.cell_index == 7
    assert w.rate == -1e-4


def test_constant_rate_well_rhs_term_returns_rate():
    w = ConstantRateWell(cell_index=7, rate=-1e-4)
    assert w.rhs_term(p_block_old=2e7, dt=86400.0) == -1e-4


def test_constant_rate_well_diag_term_is_zero():
    w = ConstantRateWell(cell_index=7, rate=-1e-4)
    assert w.diag_term(dt=86400.0) == 0.0


def test_constant_rate_well_independent_of_p_and_dt():
    w = ConstantRateWell(cell_index=5, rate=-2.0)
    assert w.rhs_term(p_block_old=1e7, dt=1.0) == -2.0
    assert w.rhs_term(p_block_old=5e7, dt=86400.0) == -2.0


def test_well_protocol_compliance():
    """ConstantRateWell 满足 Well 协议（runtime_checkable，isinstance 真起作用）。"""
    w = ConstantRateWell(cell_index=0, rate=0.0)
    assert isinstance(w, Well)


def test_constant_rate_well_is_frozen():
    from dataclasses import FrozenInstanceError
    w = ConstantRateWell(cell_index=0, rate=0.0)
    with pytest.raises(FrozenInstanceError):
        w.rate = 1.0
