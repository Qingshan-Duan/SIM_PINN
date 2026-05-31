"""PINN 可视化：loss 曲线、PINN/simulator 剖面对比、三张热图（PINN/参考/误差）。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无显示环境下也能出图
# 中文字体：标题/标签里有中文（如"尺子自检"），不配的话 DejaVu Sans 缺字形会告警 + 显示成方块。
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False  # 用 ASCII 连字符画负号，免得中文字体里负号缺字形
import matplotlib.pyplot as plt

from pinn.eval import EvalResult


PathLike = Union[str, Path]

# 取这几个时间比例作剖面切片，自动适配 n_steps
PROFILE_FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def plot_loss_curve(history: List[dict], save_path: PathLike) -> None:
    """total / pde / ic / bc 四条 loss 随迭代的对数曲线。"""
    iters = [h["iter"] for h in history]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for key, width in (("total", 1.6), ("pde", 1.0), ("ic", 1.0), ("bc", 1.0)):
        ax.semilogy(iters, [h[key] for h in history], label=key, linewidth=width,
                    alpha=1.0 if key == "total" else 0.7)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("PINN training loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_profiles(result: EvalResult, save_path: PathLike,
                  well_x: Optional[float] = None) -> None:
    """几个时刻的压力剖面：simulator 实线、PINN 虚线+点，同色配对。well_x 给定则画井位竖线。"""
    n_steps = len(result.grid_t) - 1
    days = sorted({int(round(f * n_steps)) for f in PROFILE_FRACTIONS})
    x = result.grid_x
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(days)))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    if well_x is not None:
        ax.axvline(well_x, color="grey", linestyle=":", linewidth=1.2, label="well")
    for c, d in zip(colors, days):
        ax.plot(x, result.p_exact[d] / 1e6, color=c, linestyle="-", label=f"exact t={d}d")
        ax.plot(x, result.p_pinn[d] / 1e6, color=c, linestyle="--", marker="o",
                markersize=4, label=f"pinn  t={d}d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("Pressure (MPa)")
    ax.set_title("Profiles: analytical (solid) vs PINN (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(field: np.ndarray, x: np.ndarray, t: np.ndarray, title: str,
                 save_path: PathLike, *, vmin: Optional[float] = None,
                 vmax: Optional[float] = None, cmap: str = "viridis",
                 cbar_label: str = "Pressure (MPa)") -> None:
    """通用 (x, t) 热图。field 形状 (len(t), len(x))。"""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    im = ax.pcolormesh(x, t / 86400.0, field, shading="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("time (day)")
    ax.set_title(title)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_all_eval(result: EvalResult, plots_dir: Path,
                  well_x: Optional[float] = None) -> None:
    """批量出评估图：剖面对比 + PINN/参考热图（共用色标）+ 误差热图。"""
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_profiles(result, plots_dir / "profiles.png", well_x=well_x)

    vmin = min(result.p_pinn.min(), result.p_exact.min()) / 1e6
    vmax = max(result.p_pinn.max(), result.p_exact.max()) / 1e6
    plot_heatmap(result.p_pinn / 1e6, result.grid_x, result.grid_t,
                 "PINN  P(x, t)", plots_dir / "heatmap_pinn.png", vmin=vmin, vmax=vmax)
    plot_heatmap(result.p_exact / 1e6, result.grid_x, result.grid_t,
                 "analytical  P(x, t)", plots_dir / "heatmap_exact.png", vmin=vmin, vmax=vmax)
    plot_heatmap(np.abs(result.err) / 1e6, result.grid_x, result.grid_t,
                 "|PINN - analytical|", plots_dir / "error_heatmap.png",
                 cmap="magma", cbar_label="|error| (MPa)")
    # 模拟器自检：尺子相对真解的偏差（保留模拟器验证，确认它能复现真解）
    plot_heatmap(np.abs(result.p_ref - result.p_exact) / 1e6, result.grid_x, result.grid_t,
                 "|simulator - analytical|  (尺子自检)", plots_dir / "sim_vs_exact_heatmap.png",
                 cmap="magma", cbar_label="|diff| (MPa)")
