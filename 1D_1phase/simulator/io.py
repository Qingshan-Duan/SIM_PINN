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
    """把压力场、CSV、配置快照写入 out_dir/。目录不存在则创建。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 全张量
    np.save(out / "pressure.npy", history.p)

    # CSV：列 = time_day, cell_0, cell_1, ..., cell_{nx-1}
    times_day = history.times / 86400.0
    table = np.column_stack([times_day, history.p])
    header = ",".join(["time_day"] + [f"cell_{i}" for i in range(cfg.nx)])
    np.savetxt(out / "pressure.csv", table, delimiter=",", header=header, comments="")

    # 配置快照（含派生量）
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)


def load_pressure(out_dir: PathLike) -> np.ndarray:
    """读回 pressure.npy。"""
    return np.load(Path(out_dir) / "pressure.npy")
