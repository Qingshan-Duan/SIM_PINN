"""生成 + 处理 + 落盘训练数据：扰动每个时间步的井控，跑粗步长模拟器。

数据与离散物理损失**同一个粗 dt**（关键，见 NOTES.md）。落盘三个 disjoint 的调度池
（train_pool / val / test，不同随机种子保证不重叠）。train_pool 取够大，sweep 时按前 N 条
嵌套取子集（更多数据 = 超集，干净）。

一条样本 = 一条 15 段调度跑出来的整段场 p̂ (16, 15) + 比率 (15,)。训练用的“转移对”
(场ⁿ, qⁿ)→场ⁿ⁺¹ 在 make_transitions 里现切。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simulator.config import BoundarySpec, Config
from simulator.core import run
from simulator.well import RateWellSpec
from surrogate.config import SurrogateConfig

DATA_DIR = Path(__file__).resolve().parents[1] / "output" / "surrogate" / "data"


def _sim_config(cfg: SurrogateConfig, rates) -> Config:
    return Config(
        nx=cfg.nx, L=cfg.L, A=cfg.A, k=cfg.k, mu=cfg.mu, phi=cfg.phi, ct=cfg.ct,
        P0=cfg.P0,
        left_bc=BoundarySpec("dirichlet", cfg.P_left),
        right_bc=BoundarySpec("dirichlet", cfg.P_right),
        dt=cfg.dt, n_steps=cfg.n_steps,
        wells=[RateWellSpec(cell_index=cfg.well_cell, rate=list(rates))],
    )


def simulate_schedules(cfg: SurrogateConfig, ratios: np.ndarray) -> np.ndarray:
    """对每条调度 ratios (N, n_steps) 跑模拟器，返回无量纲场 p̂ (N, n_steps+1, nx)。"""
    out = np.empty((ratios.shape[0], cfg.n_steps + 1, cfg.nx), dtype=np.float64)
    for i, r in enumerate(ratios):
        hist = run(_sim_config(cfg, r * cfg.well_rate_base))
        out[i] = (hist.p - cfg.P0) / cfg.dp_scale
    return out


def _draw_ratios(cfg: SurrogateConfig, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(cfg.q_ratio_min, cfg.q_ratio_max, size=(n, cfg.n_steps))


def generate_and_save(cfg: SurrogateConfig, n_train_pool: int = 512,
                      out_dir: Path = DATA_DIR) -> Path:
    """生成 train_pool / val / test 三池并落盘 dataset.npz（+ meta.json）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = {
        "train": (_draw_ratios(cfg, n_train_pool, seed=1001)),
        "val": (_draw_ratios(cfg, cfg.n_val_schedules, seed=2002)),
        "test": (_draw_ratios(cfg, cfg.n_test_schedules, seed=3003)),
    }
    arrays = {}
    for name, ratios in pools.items():
        arrays[f"{name}_ratios"] = ratios
        arrays[f"{name}_phat"] = simulate_schedules(cfg, ratios)
    np.savez_compressed(out_dir / "dataset.npz", **arrays)
    meta = {
        "n_train_pool": n_train_pool,
        "n_val": cfg.n_val_schedules,
        "n_test": cfg.n_test_schedules,
        "n_steps": cfg.n_steps, "nx": cfg.nx,
        "well_cell": cfg.well_cell, "well_rate_base": cfg.well_rate_base,
        "q_ratio_min": cfg.q_ratio_min, "q_ratio_max": cfg.q_ratio_max,
        "dt": cfg.dt, "dp_scale": cfg.dp_scale, "P0": cfg.P0,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    return out_dir / "dataset.npz"


def load_dataset(out_dir: Path = DATA_DIR) -> dict:
    """读回 dataset.npz，返回 {train/val/test}_{ratios,phat} 的 dict。"""
    with np.load(out_dir / "dataset.npz") as d:
        return {k: d[k] for k in d.files}


def make_transitions(phat: np.ndarray, ratios: np.ndarray, cfg: SurrogateConfig):
    """把整段场切成转移对。

    phat (N, n_steps+1, nx), ratios (N, n_steps) →
        field_in  (N*n_steps, nx)   场ⁿ
        qhat_in   (N*n_steps, 1)    无量纲井控（网络输入）
        field_out (N*n_steps, nx)   场ⁿ⁺¹（真值标签）
        q_phys    (N*n_steps, 1)    物理流量（算离散残差用）
    """
    field_in = phat[:, :-1, :].reshape(-1, cfg.nx)
    field_out = phat[:, 1:, :].reshape(-1, cfg.nx)
    qhat = cfg.ratio_to_qhat(ratios).reshape(-1, 1)
    q_phys = (ratios * cfg.well_rate_base).reshape(-1, 1)
    return field_in, qhat, field_out, q_phys


if __name__ == "__main__":
    cfg = SurrogateConfig()
    path = generate_and_save(cfg)
    d = load_dataset()
    print("saved:", path)
    for k, v in d.items():
        print(f"  {k}: {v.shape}")
