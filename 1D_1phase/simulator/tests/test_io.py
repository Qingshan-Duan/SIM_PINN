import json
import numpy as np
import pytest
from pathlib import Path

from simulator.config import Config
from simulator.core import PressureHistory
from simulator.io import save_results, load_pressure


def _make_dummy_history(cfg):
    p = np.linspace(cfg.P0, cfg.P0 * 0.9, (cfg.n_steps + 1) * cfg.nx).reshape(
        cfg.n_steps + 1, cfg.nx
    )
    times = np.arange(cfg.n_steps + 1) * cfg.dt
    return PressureHistory(p=p.astype(np.float64), times=times.astype(np.float64))


def test_save_creates_files(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    save_results(history, cfg, tmp_path)
    assert (tmp_path / "pressure.npy").exists()
    assert (tmp_path / "pressure.csv").exists()
    assert (tmp_path / "config.json").exists()


def test_save_then_load_npy_roundtrip(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    save_results(history, cfg, tmp_path)
    loaded = load_pressure(tmp_path)
    assert np.array_equal(loaded, history.p)


def test_config_json_has_all_fields(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    save_results(history, cfg, tmp_path)
    with open(tmp_path / "config.json") as f:
        data = json.load(f)
    for key in ["nx", "L", "k", "mu", "phi", "ct", "P0", "P_right",
                "well_index", "well_rate", "dt", "n_steps"]:
        assert key in data


def test_csv_first_column_is_time_in_days(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    save_results(history, cfg, tmp_path)
    arr = np.loadtxt(tmp_path / "pressure.csv", delimiter=",", skiprows=1)
    assert arr.shape == (cfg.n_steps + 1, cfg.nx + 1)
    assert np.isclose(arr[0, 0], 0.0)
    assert np.isclose(arr[-1, 0], cfg.n_steps)  # 100 天
