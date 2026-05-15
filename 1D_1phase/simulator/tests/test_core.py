import numpy as np
import pytest
from simulator.config import Config
from simulator.well import ConstantRateWell
from simulator.core import run, PressureHistory


def test_history_shape():
    cfg = Config()
    history = run(cfg, wells=[])
    assert history.p.shape == (cfg.n_steps + 1, cfg.nx)
    assert history.times.shape == (cfg.n_steps + 1,)


def test_initial_pressure_at_step_zero():
    cfg = Config()
    history = run(cfg, wells=[])
    assert np.allclose(history.p[0], cfg.P0)


def test_times_start_at_zero_and_step_by_dt():
    cfg = Config()
    history = run(cfg, wells=[])
    assert history.times[0] == 0.0
    assert history.times[1] == pytest.approx(cfg.dt)
    assert history.times[-1] == pytest.approx(cfg.dt * cfg.n_steps)


def test_run_with_well_changes_pressure():
    cfg = Config()
    wells = [ConstantRateWell(cell_index=cfg.well_index, rate=cfg.well_rate)]
    history = run(cfg, wells=wells)
    # 100 天后井所在格压力应当显著下降
    assert history.p[-1, cfg.well_index] < cfg.P0 - 1e6  # 至少下降 1 MPa
