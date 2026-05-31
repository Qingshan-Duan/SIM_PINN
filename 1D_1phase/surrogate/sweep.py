"""数据量 sweep：纯数据驱动需要多少调度才"刚好拟合"。

对一组 N（训练调度条数，嵌套子集）训练纯数据模型（lambda_phys=0），每个 N 跑多个网络初始化
种子取均值（小 N 噪声大）。记录 test 集 teacher-forced + rollout 指标。结果落盘供 main/实验引用。
找"刚好拟合"的拐点：test rollout 刚从差变好、再加数据收益骤减的那个 N*。
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from surrogate.config import SurrogateConfig
from surrogate.data_gen import load_dataset, make_transitions
from surrogate.eval import evaluate
from surrogate.train import train

OUT = Path(__file__).resolve().parents[1] / "output" / "surrogate"


def run_sweep(n_list=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
              seeds=(0, 1, 2), n_iters=3000):
    d = load_dataset()
    base = SurrogateConfig()
    rows = []
    for N in n_list:
        phat = d["train_phat"][:N]
        ratios = d["train_ratios"][:N]
        fi, qh, fo, qp = make_transitions(phat, ratios, base)
        train_data = dict(field_in=fi, qhat=qh, field_out=fo, q_phys=qp)
        accs = {"r2": [], "max": [], "worst": [], "tf_r2": [], "tf_max": []}
        for s in seeds:
            cfg = dataclasses.replace(base, n_iters=n_iters, lambda_phys=0.0, seed=s)
            net, hist, _ = train(cfg, train_data)
            m = evaluate(net, d, cfg, "test")
            accs["r2"].append(m["rollout"]["r2"])
            accs["max"].append(m["rollout"]["max_abs_mpa"])
            accs["worst"].append(m["rollout"]["worst_schedule_max_mpa"])
            accs["tf_r2"].append(m["teacher_forced"]["r2"])
            accs["tf_max"].append(m["teacher_forced"]["max_abs_mpa"])
        row = {"N": N, "n_transitions": N * base.n_steps}
        for k, v in accs.items():
            row[f"{k}_mean"] = float(np.mean(v))
            row[f"{k}_std"] = float(np.std(v))
        rows.append(row)
        print(f"N={N:3d} ({row['n_transitions']:4d} trans) | "
              f"roll R2={row['r2_mean']:.5f}±{row['r2_std']:.5f} "
              f"max={row['max_mean']:.4f} worst={row['worst_mean']:.4f} MPa | "
              f"TF R2={row['tf_r2_mean']:.5f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_pure_data.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    return rows


if __name__ == "__main__":
    run_sweep()
