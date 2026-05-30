"""时变井控 PINN 画图：调度曲线（额外）+ 复用 pinn.plot 的剖面/热图/误差。"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

PathLike = Union[str, Path]


def plot_schedule(ratios: np.ndarray, base_rate: float, save_path: PathLike) -> None:
    """画这条被评估的调度：每段流量（m³/s）随时间步的阶梯曲线。"""
    K = len(ratios)
    rates = ratios * base_rate
    steps = np.arange(K)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.step(np.append(steps, K), np.append(rates, rates[-1]), where="post",
            color="tab:red", linewidth=1.6)
    ax.axhline(base_rate, color="grey", linestyle=":", linewidth=1.0, label="基准流量")
    ax.set_xlabel("time step (day)")
    ax.set_ylabel("well rate (m³/s)")
    ax.set_title("被评估的时变井控调度（每步独立扰动 ±20%）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
