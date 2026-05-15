"""压力剖面图与热图。SI Pa 输出时转换为 MPa 显示。"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from simulator.config import Config
from simulator.core import PressureHistory


PathLike = Union[str, Path]

# 想要叠加显示的天数（不在 history 范围内的会被忽略）
PROFILE_DAYS = [0, 1, 5, 20, 50, 100]


def plot_profiles(history: PressureHistory, cfg: Config, save_path: PathLike) -> None:
    """画几条时刻的压力剖面叠加图（横轴 x 位置，纵轴 P/MPa）。"""
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for day in PROFILE_DAYS:
        if 0 <= day <= cfg.n_steps:
            ax.plot(x, history.p[day] / 1e6, marker="o", label=f"t = {day} day")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("1D pressure profiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(history: PressureHistory, cfg: Config, save_path: PathLike) -> None:
    """画时空热图（横轴 x、纵轴时间、颜色为压力/MPa）。"""
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx
    times_day = history.times / 86400.0
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    im = ax.pcolormesh(x, times_day, history.p / 1e6, shading="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="Pressure (MPa)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("time (day)")
    ax.set_title("Pressure (x, t)")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
