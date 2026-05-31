"""训练后评估：以**解析解**为真解算 PINN 误差，同时保留**模拟器**参考做交叉验证。

两条真解并存（用户要求）：
  - 解析解（analytical.py）：本算例的金标准，PINN 误差对它算（精确、免费）。
  - 模拟器参考（_reference_on_eval_grid）：保留下来，① 与解析解交叉校验（确认尺子可信），
    ② 通往两相/非均质等无解析解的情形时，它是唯一可用的参考（手艺不能丢）。
这里直接 `from simulator...` 在内存里调用数值模拟器（两者共享包根 1D_1phase/）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from simulator.config import BoundarySpec, Config as SimConfig
from simulator.core import run as sim_run
from simulator.well import RateWellSpec

from pinn.analytical import analytical_on_eval_grid
from pinn.config import PinnConfig
from pinn.net import PinnMLP


@dataclass(frozen=True)
class EvalResult:
    grid_x: np.ndarray   # (nx,)            m
    grid_t: np.ndarray   # (n_steps+1,)     s
    p_pinn: np.ndarray   # (n_steps+1, nx)  Pa
    p_exact: np.ndarray  # (n_steps+1, nx)  Pa  解析真解（误差对它算）
    p_ref: np.ndarray    # (n_steps+1, nx)  Pa  模拟器参考（保留，做交叉验证）
    err: np.ndarray      # p_pinn - p_exact, Pa
    l2_rel: float        # ||err|| / ||p_exact||      全局，分母含 ~20 MPa 基线，偏小
    max_abs: float       # max|err|,        Pa
    mape: float          # mean(|err|/|p_exact|)      逐点平均相对误差（分数）
    r2: float            # 1 - SS_res/SS_tot          相对真解方差，对「信号」诚实
    sim_vs_exact: float  # max|p_ref - p_exact|, Pa   模拟器自检：尺子离真解多远


def metrics_dict(result: EvalResult) -> dict:
    """所有结果统一的指标字典（main 与临时脚本共用，保证口径一致）。"""
    return {
        "truth": "analytical",                              # 误差基准是解析解
        "l2_relative": result.l2_rel,
        "mape": result.mape,
        "r2": result.r2,
        "max_abs_Pa": result.max_abs,
        "max_abs_MPa": result.max_abs / 1e6,
        "sim_vs_exact_max_MPa": result.sim_vs_exact / 1e6,  # 模拟器参考相对真解的最大偏差
    }


def _distributed_well(cfg: PinnConfig, nx_fine: int) -> list:
    """把总流量 q 按 PINN 同一个高斯（well_sigma_hat）摊到细网格各格上，返回一串定流量井。

    每格权重 w_j ∝ exp(−(x̂_j−well_x_hat)²/2σ̂²)，归一化到 Σw_j=1，则该格流量 = q·w_j，
    总流量守恒（=q）。这样参考解的井和 PINN 的高斯源是「同一个井」，近井对照才公平。
    """
    x_hat = (np.arange(nx_fine) + 0.5) / nx_fine                       # 细格格心（无量纲）
    w = np.exp(-((x_hat - cfg.well_x_hat) ** 2) / (2.0 * cfg.well_sigma_hat ** 2))
    w /= w.sum()
    return [RateWellSpec(cell_index=int(j), rate=cfg.well_rate * float(w[j]))
            for j in range(nx_fine) if w[j] > 1e-6]                    # 丢掉可忽略的远端格


def _reference_on_eval_grid(cfg: PinnConfig) -> np.ndarray:
    """跑细 dt + 细 nx 的 simulator 当「接近连续真解」的参考，返回 eval 网格上的压力。

    时间：每个 eval 步切成 ref_substeps 个子步，跑完每隔 ref_substeps 行取一次，落在 eval 时间点上。
    空间：细化 ref_space_refine(=m, 奇数) 倍 → nx_fine=nx_eval·m，井按高斯摊在各细格上；
          算完每隔 m 格、从第 (m−1)/2 格起取一次，正好落在 nx_eval 个粗格心上。
    返回形状 (n_steps_eval+1, nx_eval)。
    """
    sub = cfg.ref_substeps
    m = cfg.ref_space_refine
    nx_fine = cfg.nx_eval * m
    fine = SimConfig(
        nx=nx_fine, L=cfg.L, A=cfg.A,
        k=cfg.k, mu=cfg.mu, phi=cfg.phi, ct=cfg.ct,
        P0=cfg.P0,
        left_bc=BoundarySpec("dirichlet", cfg.P_left),
        right_bc=BoundarySpec("dirichlet", cfg.P_right),
        dt=cfg.dt_eval / sub, n_steps=cfg.n_steps_eval * sub,
        wells=_distributed_well(cfg, nx_fine),
    )
    return sim_run(fine).p[::sub, (m - 1) // 2 :: m]


def _eval_pinn_on_grid(net: PinnMLP, cfg: PinnConfig
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 simulator 的 (x_i, t_j) 网格上算 PINN 预测，返回 (x_si, t_si, p_si)。

    网格与 simulator 一致：x 取格心 (i+0.5)·dx，t 取 0..n_steps 的整步。
    p_si 形状 (n_steps+1, nx)，已反归一化回 SI Pa。
    """
    dx = cfg.L / cfg.nx_eval
    x_si = (np.arange(cfg.nx_eval) + 0.5) * dx           # (nx,)
    t_si = np.arange(cfg.n_steps_eval + 1) * cfg.dt_eval  # (n_steps+1,)

    XX, TT = np.meshgrid(x_si, t_si, indexing="xy")       # 均为 (n_steps+1, nx)
    x_hat = torch.tensor(XX.reshape(-1, 1) / cfg.L, dtype=torch.float32)
    t_hat = torch.tensor(TT.reshape(-1, 1) / cfg.T_end, dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        p_hat = net(x_hat, t_hat).cpu().numpy().reshape(TT.shape)  # (n_steps+1, nx)
    p_si = cfg.p_ref + p_hat * cfg.dp_scale
    return x_si, t_si, p_si


def evaluate(net: PinnMLP, cfg: PinnConfig) -> EvalResult:
    """以解析解为真解算 PINN 误差；同时跑模拟器参考做交叉验证（sim_vs_exact）。"""
    p_exact = analytical_on_eval_grid(cfg)                 # 解析真解（金标准）
    p_ref = _reference_on_eval_grid(cfg)                   # 模拟器参考（保留，交叉校验）

    x_si, t_si, p_pinn = _eval_pinn_on_grid(net, cfg)
    err = p_pinn - p_exact                                 # 误差对「真解」算
    l2_rel = float(np.linalg.norm(err) / np.linalg.norm(p_exact))
    max_abs = float(np.abs(err).max())
    mape = float(np.mean(np.abs(err) / np.abs(p_exact)))   # p_exact ~2e7 Pa，无近零值，安全
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((p_exact - p_exact.mean()) ** 2))  # 真解绕其均值的方差
    r2 = 1.0 - ss_res / ss_tot
    sim_vs_exact = float(np.abs(p_ref - p_exact).max())    # 模拟器尺子自检

    return EvalResult(
        grid_x=x_si, grid_t=t_si,
        p_pinn=p_pinn, p_exact=p_exact, p_ref=p_ref, err=err,
        l2_rel=l2_rel, max_abs=max_abs, mape=mape, r2=r2, sim_vs_exact=sim_vs_exact,
    )
