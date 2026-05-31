"""代理实验画图（中文字体已配）。三档对比 + 仿射 LSQ 参照。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

# 各档统一配色/样式
STYLE = {
    "纯数据": dict(color="C3", marker="s", ls="--"),
    "数据+物理": dict(color="C0", marker="o", ls="-"),
    "物理-only": dict(color="C2", marker="^", ls="-."),
    "仿射LSQ": dict(color="gray", marker="x", ls=":"),
}


def plot_grid_2panel(Ns: List[int], series: Dict[str, dict],
                     title: str, save_path: Path) -> None:
    """N-grid 双面板：左 rollout R²，右 max|err|。series[label] = {'r2':[..],'max':[..]}。"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for label, s in series.items():
        st = STYLE.get(label, {})
        ax[0].plot(Ns, s["r2"], label=label, **st)
        ax[1].plot(Ns, s["max"], label=label, **st)
    for a, yl, ti in ((ax[0], "rollout R²", "R²（越高越好）"),
                      (ax[1], "rollout max|err| (MPa)", "max|err|（越低越好）")):
        a.set_xscale("log"); a.set_xlabel("训练调度数 N"); a.set_ylabel(yl)
        a.set_title(ti); a.legend(); a.grid(True, alpha=0.3)
    ax[1].set_yscale("log")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_lambda_sweep(lams: List[float], rows: List[dict],
                      pure: dict, physonly: dict, save_path: Path) -> None:
    """λ-sweep 双面板：含噪下物理权重 vs 误差/R²，标出纯数据与物理-only两条参照。"""
    r2 = [r["r2"]["mean"] for r in rows]; mx = [r["max_abs_mpa"]["mean"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(lams, mx, "o-", color="C0", label="数据+物理")
    ax[0].axhline(pure["max_abs_mpa"]["mean"], color="C3", ls="--", label="纯数据(λ=0)")
    ax[0].axhline(physonly["max_abs_mpa"]["mean"], color="C2", ls="-.", label="物理-only")
    ax[0].set_ylabel("test rollout max|err| (MPa)"); ax[0].set_yscale("log")
    ax[1].plot(lams, r2, "o-", color="C0", label="数据+物理")
    ax[1].axhline(pure["r2"]["mean"], color="C3", ls="--", label="纯数据(λ=0)")
    ax[1].axhline(physonly["r2"]["mean"], color="C2", ls="-.", label="物理-only")
    ax[1].set_ylabel("test rollout R²")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("λ_phys"); a.legend(); a.grid(True, alpha=0.3)
    fig.suptitle("含噪 σ=0.1 MPa：物理正则强度 λ 的去噪效果（N=4）", fontsize=13)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_field_example(true: np.ndarray, preds: Dict[str, np.ndarray],
                       step: int, save_path: Path, title: str = "") -> None:
    """某测试调度第 step 步剖面：真值 vs 各档。preds[label] = 轨迹 (n_steps+1, nx)。"""
    x = np.arange(true.shape[-1])
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x, true[step], "k-", lw=2.5, label="模拟器真值")
    for label, tr in preds.items():
        st = STYLE.get(label, {})
        ax.plot(x, tr[step], label=label, **st)
    ax.set_xlabel("cell"); ax.set_ylabel("Δp = P−P0 (MPa)")
    ax.set_title(title or f"rollout 第 {step} 步剖面"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_extrap_bars(res: dict, save_path: Path) -> None:
    """外推柱状：各档 带内 vs 带外 max|err|。"""
    arms = [a for a in ("纯数据", "数据+物理", "物理-only") if a in res]
    inside = [res[a]["inside_max_mpa"] for a in arms]
    outside = [res[a]["outside_max_mpa"] for a in arms]
    x = np.arange(len(arms)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, inside, w, label="带内 [0.9,1.1]", color="C0")
    ax.bar(x + w / 2, outside, w, label="带外（外推）", color="C1")
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylabel("rollout max|err| (MPa)")
    ax.set_title(f"外推：窄带 {res['train_band']} 训练，测带内/带外")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)


def plot_pure_sweep(rows: List[dict], save_path: Path) -> None:
    """纯数据数据量 sweep（阶段 0 用）。"""
    N = [r["N"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(N, [r["r2_mean"] for r in rows], "o-")
    ax[0].set_xscale("log"); ax[0].set_xlabel("训练调度数 N"); ax[0].set_ylabel("rollout R²")
    ax[0].set_title("纯数据：数据量 vs 精度（干净）"); ax[0].grid(True, alpha=0.3)
    ax[1].plot(N, [r["max_mean"] for r in rows], "s-", color="C1")
    ax[1].set_xscale("log"); ax[1].set_xlabel("训练调度数 N"); ax[1].set_ylabel("rollout max|err| (MPa)")
    ax[1].set_title("纯数据：max|err| 地板 ≈0.026 MPa"); ax[1].grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=120); plt.close(fig)
