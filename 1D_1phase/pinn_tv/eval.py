"""时变井控 PINN 评估：对一批随机测试调度逐条算指标（对解析真解），并为画图那条做模拟器交叉校验。

真解由分段常数 Duhamel 解析解给（线性区精确）。模拟器参考：时变定流量井（每段流量按高斯摊到细格，
每个细格一条长度 = n_steps_fine 的流量序列），细 dt + 细 nx，复用 pinn 的网格对齐/降采样口径。
EvalResult 复用 pinn.eval。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from simulator.config import BoundarySpec, Config as SimConfig
from simulator.core import run as sim_run
from simulator.well import RateWellSpec

from pinn.eval import EvalResult
from pinn_tv.analytical import analytical_schedule_on_eval_grid
from pinn_tv.config import PinnTvConfig
from pinn_tv.net import TvPinnMLP


def _metrics(err: np.ndarray, p_exact: np.ndarray) -> dict:
    l2_rel = float(np.linalg.norm(err) / np.linalg.norm(p_exact))
    max_abs = float(np.abs(err).max())
    mape = float(np.mean(np.abs(err) / np.abs(p_exact)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((p_exact - p_exact.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"l2_relative": l2_rel, "mape": mape, "r2": r2,
            "max_abs_Pa": max_abs, "max_abs_MPa": max_abs / 1e6}


def _eval_pinn_on_grid(net: TvPinnMLP, cfg: PinnTvConfig, sched_qhat: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """给定一条调度（K 个 q̂，形状 (K,)），在 simulator 网格上算 PINN 预测，返回 (x_si,t_si,p_si[Pa])。"""
    dx = cfg.L / cfg.nx_eval
    x_si = (np.arange(cfg.nx_eval) + 0.5) * dx
    t_si = np.arange(cfg.n_steps_eval + 1) * cfg.dt_eval

    XX, TT = np.meshgrid(x_si, t_si, indexing="xy")
    npts = XX.size
    x_hat = torch.tensor(XX.reshape(-1, 1) / cfg.L, dtype=torch.float32)
    t_hat = torch.tensor(TT.reshape(-1, 1) / cfg.T_end, dtype=torch.float32)
    sched = torch.tensor(np.tile(sched_qhat.reshape(1, -1), (npts, 1)), dtype=torch.float32)

    net.eval()
    with torch.no_grad():
        p_hat = net(x_hat, t_hat, sched).cpu().numpy().reshape(TT.shape)
    p_si = cfg.p_ref + p_hat * cfg.dp_scale
    return x_si, t_si, p_si


def _reference_on_eval_grid(cfg: PinnTvConfig, ratios: np.ndarray) -> np.ndarray:
    """时变井控的细网格模拟器参考。每段流量按高斯摊到细格，每个细格给一条流量序列。"""
    sub, m = cfg.ref_substeps, cfg.ref_space_refine
    nx_fine = cfg.nx_eval * m
    n_steps_fine = cfg.n_steps_eval * sub
    K = cfg.n_segments

    # 每个细时间步落在哪一调度段
    seg = np.minimum(np.arange(n_steps_fine) // sub, K - 1)         # (n_steps_fine,)
    rate_base_per_step = ratios[seg] * cfg.well_rate                # (n_steps_fine,) 该段总流量

    # 高斯权重摊到细格（Σw=1），每格一条流量序列
    x_hat = (np.arange(nx_fine) + 0.5) / nx_fine
    w = np.exp(-((x_hat - cfg.well_x_hat) ** 2) / (2.0 * cfg.well_sigma_hat ** 2))
    w /= w.sum()
    wells = [RateWellSpec(cell_index=int(j), rate=(rate_base_per_step * float(w[j])).tolist())
             for j in range(nx_fine) if w[j] > 1e-6]

    fine = SimConfig(
        nx=nx_fine, L=cfg.L, A=cfg.A, k=cfg.k, mu=cfg.mu, phi=cfg.phi, ct=cfg.ct,
        P0=cfg.P0,
        left_bc=BoundarySpec("dirichlet", cfg.P_left),
        right_bc=BoundarySpec("dirichlet", cfg.P_right),
        dt=cfg.dt_eval / sub, n_steps=n_steps_fine, wells=wells,
    )
    return sim_run(fine).p[::sub, (m - 1) // 2 :: m]


def _random_schedules(cfg: PinnTvConfig, n: int, seed: int) -> np.ndarray:
    """采 n 条随机调度（比率），形状 (n, K)，每段独立 ~U[ratio_min,ratio_max]。"""
    rng = np.random.default_rng(seed)
    return rng.uniform(cfg.q_ratio_min, cfg.q_ratio_max, size=(n, cfg.n_segments))


def metrics_over_schedules(net: TvPinnMLP, cfg: PinnTvConfig
                           ) -> Tuple[List[dict], dict, np.ndarray]:
    """对 n_test_schedules 条随机调度逐条算指标（只对解析解，便宜），返回 (per_sched, 汇总, 调度数组)。"""
    ratios_all = _random_schedules(cfg, cfg.n_test_schedules, cfg.schedule_eval_seed)
    per = []
    for i, ratios in enumerate(ratios_all):
        p_exact = analytical_schedule_on_eval_grid(cfg, ratios)
        _, _, p_pinn = _eval_pinn_on_grid(net, cfg, cfg.ratio_to_qhat(ratios))
        m = _metrics(p_pinn - p_exact, p_exact)
        per.append({"idx": i, **m})
    agg = {
        "n_schedules": len(per),
        "r2_mean": float(np.mean([d["r2"] for d in per])),
        "r2_min": float(np.min([d["r2"] for d in per])),
        "mape_mean": float(np.mean([d["mape"] for d in per])),
        "l2_relative_mean": float(np.mean([d["l2_relative"] for d in per])),
        "max_abs_MPa_worst": float(np.max([d["max_abs_MPa"] for d in per])),
    }
    return per, agg, ratios_all


def evaluate_at_schedule(net: TvPinnMLP, cfg: PinnTvConfig, ratios: np.ndarray) -> EvalResult:
    """单条调度完整评估（含时变模拟器参考），用于画图 + 交叉校验。"""
    p_exact = analytical_schedule_on_eval_grid(cfg, ratios)
    p_ref = _reference_on_eval_grid(cfg, ratios)
    x_si, t_si, p_pinn = _eval_pinn_on_grid(net, cfg, cfg.ratio_to_qhat(ratios))

    err = p_pinn - p_exact
    m = _metrics(err, p_exact)
    sim_vs_exact = float(np.abs(p_ref - p_exact).max())
    return EvalResult(
        grid_x=x_si, grid_t=t_si, p_pinn=p_pinn, p_exact=p_exact, p_ref=p_ref, err=err,
        l2_rel=m["l2_relative"], max_abs=m["max_abs_Pa"], mape=m["mape"],
        r2=m["r2"], sim_vs_exact=sim_vs_exact,
    )
