import math
import pytest
from simulator.config import Config


def test_default_config_values():
    cfg = Config()
    assert cfg.nx == 15
    assert cfg.L == 150.0
    assert cfg.A == 100.0
    assert cfg.k == 1e-13
    assert cfg.mu == 5e-3
    assert cfg.phi == 0.2
    assert cfg.ct == 1e-9
    assert cfg.P0 == 2e7
    assert cfg.P_right == 2e7
    assert cfg.well_index == 7
    assert cfg.well_rate == -1e-4
    assert cfg.dt == 86400.0
    assert cfg.n_steps == 100


def test_derived_quantities():
    cfg = Config()
    assert cfg.dx == pytest.approx(10.0)
    assert cfg.V == pytest.approx(1000.0)
    expected_T = cfg.k * cfg.A / (cfg.mu * cfg.dx)
    assert cfg.T == pytest.approx(expected_T)
    assert cfg.x_well == pytest.approx(75.0)


def test_well_index_validation():
    with pytest.raises(AssertionError):
        Config(well_index=20)
    with pytest.raises(AssertionError):
        Config(well_index=-1)


def test_config_is_frozen():
    cfg = Config()
    with pytest.raises(Exception):  # FrozenInstanceError
        cfg.nx = 30


def test_config_with_overrides():
    cfg = Config(nx=10, L=100.0, well_index=5)
    assert cfg.nx == 10
    assert cfg.dx == pytest.approx(10.0)
    assert cfg.x_well == pytest.approx(55.0)
