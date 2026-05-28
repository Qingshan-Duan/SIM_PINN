"""训练后评估：跑 simulator 拿参考解，在同一网格上比 PINN 预测，算误差。

这里直接 `from simulator...` 在内存里调用数值模拟器（两者共享包根 1D_1phase/）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from simulator.config import BoundarySpec, Config as SimConfig
from simulator.core import run as sim_run

from pinn.config import PinnConfig
from pinn.net import PinnMLP


@dataclass(frozen=True)
class EvalResult:
    grid_x: np.ndarray   # (nx,)            m
    grid_t: np.ndarray   # (n_steps+1,)     s
    p_pinn: np.ndarray   # (n_steps+1, nx)  Pa
    p_ref: np.ndarray    # (n_steps+1, nx)  Pa  (simulator)
    err: np.ndarray      # p_pinn - p_ref,  Pa
    l2_rel: float        # ||err|| / ||p_ref||        全局，分母含 ~20 MPa 基线，偏小
    max_abs: float       # max|err|,        Pa
    mape: float          # mean(|err|/|p_ref|)        逐点平均相对误差（分数）
    r2: float            # 1 - SS_res/SS_tot          相对参考方差，对「信号」诚实


def metrics_dict(result: EvalResult) -> dict:
    """所有结果统一的指标字典（main 与临时脚本共用，保证口径一致）。"""
    return {
        "l2_relative": result.l2_rel,
        "mape": result.mape,
        "r2": result.r2,
        "max_abs_Pa": result.max_abs,
        "max_abs_MPa": result.max_abs / 1e6,
    }


def _reference_on_eval_grid(cfg: PinnConfig) -> np.ndarray:
    """跑细 dt 的 simulator 当「接近连续真解」的参考，返回 eval 时间点上的压力。

    每个 eval 步切成 ref_substeps 个子步：fine_dt = dt_eval/ref_substeps，
    fine 步数 = n_steps_eval·ref_substeps。跑完每隔 ref_substeps 行取一次，
    正好落在 0, dt_eval, 2·dt_eval, ... 这些 eval 时间点上。
    返回形状 (n_steps_eval+1, nx)。
    """
    sub = cfg.ref_substeps
    fine = SimConfig(
        nx=cfg.nx_eval, L=cfg.L, A=cfg.A,
        k=cfg.k, mu=cfg.mu, phi=cfg.phi, ct=cfg.ct,
        P0=cfg.P0,
        left_bc=BoundarySpec("dirichlet", cfg.P_left),
        right_bc=BoundarySpec("dirichlet", cfg.P_right),
        dt=cfg.dt_eval / sub, n_steps=cfg.n_steps_eval * sub,
        wells=[],
    )
    return sim_run(fine).p[::sub]


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
    """跑细 dt simulator 拿参考、在同网格上跑 PINN，算 L2 相对误差与最大绝对误差。"""
    p_ref = _reference_on_eval_grid(cfg)                   # (n_steps+1, nx)

    x_si, t_si, p_pinn = _eval_pinn_on_grid(net, cfg)
    err = p_pinn - p_ref
    l2_rel = float(np.linalg.norm(err) / np.linalg.norm(p_ref))
    max_abs = float(np.abs(err).max())
    mape = float(np.mean(np.abs(err) / np.abs(p_ref)))     # p_ref ~2e7 Pa，无近零值，安全
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((p_ref - p_ref.mean()) ** 2))    # 参考绕其均值的方差
    r2 = 1.0 - ss_res / ss_tot

    return EvalResult(
        grid_x=x_si, grid_t=t_si,
        p_pinn=p_pinn, p_ref=p_ref, err=err,
        l2_rel=l2_rel, max_abs=max_abs, mape=mape, r2=r2,
    )
