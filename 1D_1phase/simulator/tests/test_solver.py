"""求解器单步质量守恒：对各种边界 × 井组合都应严格成立 (rtol=1e-10)。"""
import numpy as np
import pytest

from simulator.config import BoundarySpec, Config
from simulator.solver import assemble_and_solve
from simulator.well import (
    BHPWellSpec,
    RateWellSpec,
    make_wells,
)


def _mass_balance(p_old, p_new, cfg, wells):
    """返回 (accumulation, source_total) — 离散守恒下两者必须相等。"""
    accumulation = cfg.phi * cfg.ct * cfg.V * np.sum(p_new - p_old)

    def bflux(bc, p_face_cell):
        if bc.kind == "noflow":
            return 0.0
        return 2.0 * cfg.T * (bc.pressure - p_face_cell) * cfg.dt

    left = bflux(cfg.left_bc, p_new[0])
    right = bflux(cfg.right_bc, p_new[-1])

    well_total = 0.0
    for w in wells:
        rhs = w.rhs_term(p_old[w.cell_index], cfg.dt)
        diag = w.diag_term(cfg.dt)
        well_total += cfg.dt * (rhs - diag * p_new[w.cell_index])

    return accumulation, left + right + well_total


# 五个有代表性的场景：边界类型 × 井类型 的组合
SCENARIOS = [
    pytest.param(
        Config(),  # 默认：左右都定压 20/19 MPa，1 口定流量井
        id="default-bothDirichlet-rate",
    ),
    pytest.param(
        Config(
            left_bc=BoundarySpec("noflow"),
            right_bc=BoundarySpec("noflow"),
            wells=[RateWellSpec(cell_index=7, rate=-5e-5)],
        ),
        id="bothNoFlow-rate",
    ),
    pytest.param(
        Config(
            left_bc=BoundarySpec("noflow"),
            right_bc=BoundarySpec("dirichlet", 2.0e7),
            wells=[RateWellSpec(cell_index=7, rate=-1e-4)],
        ),
        id="mixedBC-rate",
    ),
    pytest.param(
        Config(
            left_bc=BoundarySpec("dirichlet", 2.0e7),
            right_bc=BoundarySpec("dirichlet", 2.0e7),
            wells=[BHPWellSpec(cell_index=7, p_wf=1.0e7, kind="producer", rw=0.1)],
        ),
        id="bothDirichlet-bhp",
    ),
    pytest.param(
        Config(
            left_bc=BoundarySpec("dirichlet", 2.0e7),
            right_bc=BoundarySpec("dirichlet", 2.0e7),
            wells=[
                RateWellSpec(cell_index=3, rate=+5e-5),   # 注入
                RateWellSpec(cell_index=11, rate=-5e-5),  # 采出
            ],
        ),
        id="bothDirichlet-injector+producer",
    ),
]


@pytest.mark.parametrize("cfg", SCENARIOS)
def test_single_step_mass_balance(cfg):
    wells = make_wells(cfg)
    p_old = np.full(cfg.nx, cfg.P0)
    p_new = assemble_and_solve(p_old, cfg, wells)
    accum, src = _mass_balance(p_old, p_new, cfg, wells)
    assert np.isclose(accum, src, rtol=1e-10), (
        f"accumulation={accum:.6e}, source={src:.6e}"
    )


def test_no_well_uniform_initial_at_both_dirichlet_drifts_to_linear():
    """两侧定压不同压力 + 无井，一步后内部应介于两端之间。"""
    cfg = Config(
        left_bc=BoundarySpec("dirichlet", 2.0e7),
        right_bc=BoundarySpec("dirichlet", 1.0e7),
        wells=[],
    )
    p_old = np.full(cfg.nx, 1.5e7)
    p_new = assemble_and_solve(p_old, cfg, wells=[])
    assert np.all(p_new <= 2.0e7 + 1e-6)
    assert np.all(p_new >= 1.0e7 - 1e-6)


def test_no_flow_both_sides_no_well_preserves_pressure():
    """两侧封闭 + 无井 + 初始均匀 → 任何一步都不变。"""
    cfg = Config(
        left_bc=BoundarySpec("noflow"),
        right_bc=BoundarySpec("noflow"),
        wells=[],
    )
    p_old = np.full(cfg.nx, cfg.P0)
    p_new = assemble_and_solve(p_old, cfg, wells=[])
    assert np.allclose(p_new, cfg.P0, atol=1e-6)
