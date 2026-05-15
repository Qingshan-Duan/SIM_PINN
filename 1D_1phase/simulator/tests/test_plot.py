import matplotlib
matplotlib.use("Agg")  # 测试环境无显示

import numpy as np
import pytest
from pathlib import Path

from simulator.config import Config
from simulator.core import PressureHistory
from simulator.plot import plot_profiles, plot_heatmap


def _make_dummy_history(cfg):
    p = np.full((cfg.n_steps + 1, cfg.nx), cfg.P0, dtype=np.float64)
    times = np.arange(cfg.n_steps + 1, dtype=np.float64) * cfg.dt
    return PressureHistory(p=p, times=times)


def test_plot_profiles_creates_file(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    out = tmp_path / "profiles.png"
    plot_profiles(history, cfg, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_heatmap_creates_file(tmp_path):
    cfg = Config()
    history = _make_dummy_history(cfg)
    out = tmp_path / "heatmap.png"
    plot_heatmap(history, cfg, out)
    assert out.exists()
    assert out.stat().st_size > 0
