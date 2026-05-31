"""压力剖面图、热图、井产量曲线。SI Pa 输出时转换为 MPa 显示。"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from simulator.config import Config
from simulator.core import PressureHistory


PathLike = Union[str, Path]

# 取这几个时间比例（0、10%、25%、50%、75%、100%）作剖面切片，自动适配 n_steps
PROFILE_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def plot_profiles(history: PressureHistory, cfg: Config, save_path: PathLike) -> None:
    """画几条时刻的压力剖面叠加图（横轴 x 位置，纵轴 P/MPa）。"""
    x = (np.arange(cfg.nx) + 0.5) * cfg.dx
    days = sorted({int(round(f * cfg.n_steps)) for f in PROFILE_FRACTIONS})
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for day in days:
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


def plot_well_rates(history: PressureHistory, cfg: Config, save_path: PathLike) -> None:
    """画每口井的实际流量曲线（横轴 day，纵轴 q m^3/s）。

    q 的符号约定：正=注入、负=采出。0 处画一条灰色参考线。
    """
    if history.q.size == 0:
        return
    times_day = history.times[1:] / 86400.0  # q 从第 1 步起，对齐 (n_steps,) 行
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, w_spec in enumerate(cfg.wells):
        label = f"well {i} @ cell {w_spec.cell_index}"
        ax.plot(times_day, history.q[:, i], marker="o", label=label)
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time (day)")
    ax.set_ylabel(r"rate q (m$^3$/s)   (+ inject  /  − produce)")
    ax.set_title("Well rates")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
