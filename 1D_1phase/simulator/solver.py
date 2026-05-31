"""单步隐式求解：组装三对角并求解 A · p^{n+1} = b。"""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.linalg import solve_banded

from simulator.config import BoundarySpec, Config
from simulator.well import Well


def _boundary_contrib(bc: BoundarySpec, T: float) -> Tuple[float, float]:
    """返回边界格在主对角与右端项上的额外贡献 (diag_add, rhs_add)。

    - noflow:   面流量=0，无贡献                 -> (0,    0)
    - dirichlet 面距格心 dx/2，面传导率 2T:      -> (2T,   2T·P_bc)
    """
    if bc.kind == "noflow":
        return 0.0, 0.0
    return 2.0 * T, 2.0 * T * bc.pressure


def assemble_and_solve(
    p_old: np.ndarray,
    cfg: Config,
    wells: Sequence[Well],
) -> np.ndarray:
    """组装三对角线性系统并求解一步，返回 p^{n+1}，shape (nx,)。

    内部格子方程：
        α·p_i^{n+1} + 2T·p_i^{n+1} − T·p_{i−1}^{n+1} − T·p_{i+1}^{n+1} = α·p_i^n + q_i
        α := φ·ct·V/Δt

    左/右边界格的 main 与 b 的额外项由 _boundary_contrib 给出：
        noflow:   两项都为 0（无外通量）
        dirichlet: 主对角 += 2T,  b += 2T·P_bc

    井贡献来自 well.rhs_term / well.diag_term。
    """
    N = cfg.nx
    T = cfg.T
    alpha = cfg.phi * cfg.ct * cfg.V / cfg.dt

    # 主对角：内部格 α + 2T，边界格先按 α + T（只算 1 个内部面）起步，
    # 然后由 _boundary_contrib 加上边界面贡献。
    main = np.full(N, alpha + 2.0 * T)
    main[0] = alpha + T
    main[-1] = alpha + T

    # 上下副对角
    sub = np.full(N, -T)
    sup = np.full(N, -T)

    # 右端项
    b = alpha * p_old.astype(np.float64, copy=True)

    # 边界贡献
    ld, lr = _boundary_contrib(cfg.left_bc, T)
    rd, rr = _boundary_contrib(cfg.right_bc, T)
    main[0]  += ld;  b[0]  += lr
    main[-1] += rd;  b[-1] += rr

    # 井贡献
    for w in wells:
        b[w.cell_index] += w.rhs_term(p_old[w.cell_index], cfg.dt)
        main[w.cell_index] += w.diag_term(cfg.dt)

    # solve_banded 期望的 (3, N) 格式：
    #   ab[0, 1:]  = 上对角  (ab[0,0]  未用)
    #   ab[1, :]   = 主对角
    #   ab[2, :-1] = 下对角  (ab[2,-1] 未用)
    ab = np.zeros((3, N))
    ab[0, 1:]  = sup[:-1]
    ab[1, :]   = main
    ab[2, :-1] = sub[1:]

    return solve_banded((1, 1), ab, b)
