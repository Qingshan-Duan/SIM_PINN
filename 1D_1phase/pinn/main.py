"""PINN 入口：训练 → 评估（内嵌 simulator）→ 结构化保存。

从 1D_1phase/ 运行：
    python -m pinn.main                    # 产物落到 output/pinn/<时间戳>/
    python -m pinn.main --name baseline    # 产物落到 output/pinn/<时间戳>_baseline/

每次运行写独立子文件夹（时间戳前缀，按时间排序），不覆盖历史结果，保留开发历程。
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pinn.config import PinnConfig
from pinn.eval import EvalResult, evaluate, metrics_dict
from pinn.plot import plot_all_eval, plot_loss_curve
from pinn.train import train


def _save_loss_history(history: list, path: Path) -> None:
    keys = ["iter", "phase", "total", "pde", "ic", "bc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in history:
            w.writerow({k: h.get(k, "") for k in keys})


def _save_pred_csv(result: EvalResult, path: Path) -> None:
    """与 simulator pressure.csv 同格式：time_day, cell_0, ..., cell_{nx-1}。"""
    nx = result.p_pinn.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_day"] + [f"cell_{i}" for i in range(nx)])
        for j, t in enumerate(result.grid_t):
            w.writerow([t / 86400.0, *result.p_pinn[j].tolist()])


def _run_dir(name: Optional[str]) -> Path:
    """本次运行的输出子文件夹：output/pinn/<时间戳>[_<name>]/。

    时间戳前缀保证按时间排序、不覆盖历史；name 是可选的人类可读标签。
    """
    base = Path(__file__).resolve().parent.parent / "output" / "pinn"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{stamp}_{name}" if name else stamp
    return base / folder


def main(run_name: Optional[str] = None, no_lbfgs: bool = False,
         cfg: Optional[PinnConfig] = None) -> None:
    if cfg is None:                    # 允许外部传入定制 config（消融/对照），否则用默认
        cfg = PinnConfig()
    if no_lbfgs:                       # 消融：只跑 Adam，跳过 L-BFGS 精修
        cfg = replace(cfg, lbfgs_iters=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}  alpha_nd={cfg.alpha_nd:.4f}  q_nd={cfg.q_nd:.3f}  "
          f"lbfgs_iters={cfg.lbfgs_iters}")

    # 1. 训练
    net, history = train(cfg, device=device)

    # 2. 评估（移到 CPU，评估网格很小）
    net = net.to("cpu")
    result = evaluate(net, cfg)
    print(f"[eval] (vs 解析真解) L2 = {result.l2_rel:.3e}   MAPE = {result.mape:.3e}   "
          f"R2 = {result.r2:.6f}   max|err| = {result.max_abs/1e6:.4f} MPa")
    print(f"[validate] 模拟器 vs 解析 max = {result.sim_vs_exact/1e6:.4f} MPa  (尺子自检)")

    # 3. 结构化保存到本次运行的独立子文件夹
    out_dir = _run_dir(run_name)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(net.state_dict(), out_dir / "checkpoint.pt")
    _save_loss_history(history, out_dir / "loss_history.csv")
    np.save(out_dir / "pred.npy", result.p_pinn)
    np.save(out_dir / "exact.npy", result.p_exact)
    np.save(out_dir / "reference.npy", result.p_ref)
    _save_pred_csv(result, out_dir / "pred.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict(result), f, indent=2)

    plot_loss_curve(history, plots_dir / "loss_curve.png")
    plot_all_eval(result, plots_dir, well_x=cfg.well_x_hat * cfg.L)

    print(f"[main] done. Output -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 1D 单相 PINN 并评估")
    parser.add_argument("--name", default=None,
                        help="本次运行的标签，作为输出子文件夹后缀（便于回顾开发历程）")
    parser.add_argument("--no-lbfgs", action="store_true",
                        help="消融：跳过 L-BFGS 精修，只用 Adam 训练")
    args = parser.parse_args()
    main(run_name=args.name, no_lbfgs=args.no_lbfgs)
