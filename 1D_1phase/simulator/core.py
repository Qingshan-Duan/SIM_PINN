"""时间循环。每步从 cfg.wells 重新拼出 runtime well，记录实际 q，校验 BHP 井类型。"""
from dataclasses import dataclass

import numpy as np

from simulator.config import Config
from simulator.solver import assemble_and_solve
from simulator.well import BHPWellSpec, _value_at, make_wells


@dataclass(frozen=True)
class PressureHistory:
    """模拟结果：
        p[n, i]   是第 n 个时间点（含初始）cell i 的压力，Pa。
        times[n]  对应时刻，s。
        q[n, w]   是第 n 步（n=0..n_steps-1，不含初始）第 w 口井的实际流量，m^3/s，
                   符号约定：正=注入、负=采出。对定流量井就是 spec.rate；对 BHP 井是
                   PI·(P_wf − P_cell^{n+1})。
    """
    p: np.ndarray      # (n_steps+1, nx),  float64, Pa
    times: np.ndarray  # (n_steps+1,),     float64, s
    q: np.ndarray      # (n_steps, n_wells), float64, m^3/s


def run(cfg: Config) -> PressureHistory:
    """执行 cfg.n_steps 个时间步，返回压力 + 实际井流量历史。

    每步：
      1. make_wells(cfg, step=n) 取出该步的井控数值
      2. assemble_and_solve 求解 p^{n+1}
      3. 计算每口井实际 q = rhs_term − diag_term·p^{n+1}
      4. BHP 井若 sign(q) 与 spec.kind 矛盾 → 抛 RuntimeError 终止模拟
    """
    nw = len(cfg.wells)
    p = np.full((cfg.n_steps + 1, cfg.nx), cfg.P0, dtype=np.float64)
    times = np.arange(cfg.n_steps + 1, dtype=np.float64) * cfg.dt
    q = np.zeros((cfg.n_steps, nw), dtype=np.float64)

    for n in range(cfg.n_steps):
        wells_n = make_wells(cfg, step=n)
        p[n + 1] = assemble_and_solve(p[n], cfg, wells_n)

        for i, (w, spec) in enumerate(zip(wells_n, cfg.wells)):
            cell = w.cell_index
            q_in = w.rhs_term(p[n, cell], cfg.dt) - w.diag_term(cfg.dt) * p[n + 1, cell]
            q[n, i] = q_in
            if isinstance(spec, BHPWellSpec):
                _check_bhp_kind(spec, q_in, n, p_cell_new=p[n + 1, cell])

    return PressureHistory(p=p, times=times, q=q)


def _check_bhp_kind(spec: BHPWellSpec, q: float, step: int, p_cell_new: float) -> None:
    """BHP 井 sign(q) 与声明的 kind 矛盾时终止模拟。"""
    p_wf_now = _value_at(spec.p_wf, step)
    if spec.kind == "producer" and q > 0.0:
        raise RuntimeError(
            f"步 {step}: BHP 井 cell={spec.cell_index} 声明为 'producer'，"
            f"但 P_wf={p_wf_now:.3e} ≥ P_cell^(n+1)={p_cell_new:.3e}，"
            f"实际 q={q:+.3e} m^3/s > 0（变成注入）。"
            f"请调低 P_wf 或把 kind 改为 'injector'。"
        )
    if spec.kind == "injector" and q < 0.0:
        raise RuntimeError(
            f"步 {step}: BHP 井 cell={spec.cell_index} 声明为 'injector'，"
            f"但 P_wf={p_wf_now:.3e} ≤ P_cell^(n+1)={p_cell_new:.3e}，"
            f"实际 q={q:+.3e} m^3/s < 0（变成采出）。"
            f"请调高 P_wf 或把 kind 改为 'producer'。"
        )
