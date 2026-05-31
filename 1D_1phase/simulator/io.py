"""结果存盘 / 读取。后续 PINN 训练会用 load_pressure。"""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Union

import numpy as np

from simulator.config import Config
from simulator.core import PressureHistory


PathLike = Union[str, Path]


def save_results(history: PressureHistory, cfg: Config, out_dir: PathLike) -> None:
    """把压力场、井实际流量、配置快照写入 out_dir/。目录不存在则创建。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 压力全张量
    np.save(out / "pressure.npy", history.p)

    # 压力 CSV：列 = time_day, cell_0, ..., cell_{nx-1}
    times_day = history.times / 86400.0
    table = np.column_stack([times_day, history.p])
    header = ",".join(["time_day"] + [f"cell_{i}" for i in range(cfg.nx)])
    np.savetxt(out / "pressure.csv", table, delimiter=",", header=header, comments="")

    # 井实际流量 CSV：列 = time_day, well_0_cell{X}, well_1_cell{Y}, ...
    # time_day 从第 1 步起（n_steps 行），与 history.q 对齐
    if history.q.size:
        q_times_day = times_day[1:]
        q_table = np.column_stack([q_times_day, history.q])
        q_header = ",".join(
            ["time_day"] + [f"well_{i}_cell{w.cell_index}" for i, w in enumerate(cfg.wells)]
        )
        np.savetxt(out / "rates.csv", q_table, delimiter=",", header=q_header, comments="")

    # 配置快照（含派生量）
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def load_pressure(out_dir: PathLike) -> np.ndarray:
    """读回 pressure.npy。"""
    return np.load(Path(out_dir) / "pressure.npy")
