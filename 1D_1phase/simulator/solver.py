from typing import Sequence
import numpy as np
from scipy.linalg import solve_banded

from simulator.config import Config
from simulator.well import Well


def assemble_and_solve(
    p_old: np.ndarray,
    cfg: Config,
    wells: Sequence[Well],
) -> np.ndarray:
    """组装三对角线性系统并求解一步：A · p^{n+1} = b。

    返回 p^{n+1}，shape (nx,)。

    矩阵结构（左封闭、右 Dirichlet P_R，面传导率 2T）：
        main[0]    = α + T
        main[i]    = α + 2T          (i in 1..N-2)
        main[N-1]  = α + 3T          (= T + 2T)
        sub/sup    = -T
        b[i]       = α·p_old[i] + q_i
        b[N-1]    += 2T·P_R

    α := φ·ct·V/Δt。井项通过 well.rhs_term / well.diag_term 加入。
    """
    N = cfg.nx
    T = cfg.T
    alpha = cfg.phi * cfg.ct * cfg.V / cfg.dt

    # 主对角与上下副对角
    main = np.full(N, alpha + 2.0 * T)
    main[0] = alpha + T
    main[-1] = alpha + 3.0 * T
    sub = np.full(N, -T)   # sub[i] 对应行 i 左侧的元素，i=0 未用
    sup = np.full(N, -T)   # sup[i] 对应行 i 右侧的元素，i=N-1 未用

    # 右端项
    b = alpha * p_old.astype(np.float64, copy=True)
    b[-1] += 2.0 * T * cfg.P_right

    # 井贡献
    for w in wells:
        b[w.cell_index] += w.rhs_term(p_old[w.cell_index], cfg.dt)
        main[w.cell_index] += w.diag_term(cfg.dt)

    # solve_banded 期望的 (3, N) 格式：
    #   ab[0, 1:] = 上对角  (ab[0,0] 未用)
    #   ab[1, :]  = 主对角
    #   ab[2,:-1] = 下对角  (ab[2,-1] 未用)
    ab = np.zeros((3, N))
    ab[0, 1:] = sup[:-1]
    ab[1, :] = main
    ab[2, :-1] = sub[1:]

    return solve_banded((1, 1), ab, b)
