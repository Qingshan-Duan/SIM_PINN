"""时变井控 PINN 入口（实验性）：训练 → 多调度评估 → 随机抽一条调度画图/交叉校验 → 保存。

从 1D_1phase/ 运行：
    python -m pinn_tv.main                 # → output/pinn_tv/<时间戳>/
    python -m pinn_tv.main --name try1     # → output/pinn_tv/<时间戳>_try1/

整条调度作输入（17 维），纯物理无数据。产物只进 output/pinn_tv/，与 pinn/、pinn_param/ 分开。
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from pinn.eval import EvalResult
from pinn.plot import plot_all_eval, plot_loss_curve
from pinn_tv.config import PinnTvConfig
from pinn_tv.eval import evaluate_at_schedule, metrics_over_schedules
from pinn_tv.plot import plot_schedule
from pinn_tv.train import train


def _save_loss_history(history: list, path: Path) -> None:
    keys = ["iter", "phase", "total", "pde", "ic", "bc", "w_ic", "w_bc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in history:
            w.writerow({k: h.get(k, "") for k in keys})


def _run_dir(name: Optional[str]) -> Path:
    base = Path(__file__).resolve().parent.parent / "output" / "pinn_tv"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / (f"{stamp}_{name}" if name else stamp)


def _write_readme(path: Path, cfg: PinnTvConfig, agg: dict, per: List[dict],
                  plot_ratios: np.ndarray, result: EvalResult, run_name: Optional[str]) -> None:
    lines = f"""# pinn_tv run: {run_name or '(unnamed)'}  （实验性：时变井控）

整条 {cfg.n_segments} 段调度作输入：网络 `(x̂,t̂,q̂₁..q̂_{cfg.n_segments})→p̂`，每段独立扰动
±{(cfg.q_ratio_max-1)*100:.0f}%（比率 [{cfg.q_ratio_min},{cfg.q_ratio_max}]），纯物理无数据。输入维度 {2+cfg.n_segments}。

- 网络：{cfg.hidden_layers} 隐层 × {cfg.hidden_units}，tanh；hard_ic={cfg.hard_ic}，自适应权重={cfg.adaptive_weights}
- 训练：Adam {cfg.adam_iters} + L-BFGS {cfg.lbfgs_iters}
- 误差对**分段常数 Duhamel 解析真解**算（线性区精确）

## 跨调度汇总（{agg['n_schedules']} 条随机测试调度，vs 解析真解）

- R²_mean = {agg['r2_mean']:.6f}，R²_min = {agg['r2_min']:.6f}
- MAPE_mean = {agg['mape_mean']:.3e}，L2(rel)_mean = {agg['l2_relative_mean']:.3e}
- max|err|_worst = {agg['max_abs_MPa_worst']:.4f} MPa

逐条 R²：{', '.join(f"{d['r2']:.4f}" for d in per)}

## 画图的调度（随机抽，seed={cfg.plot_schedule_seed}）

R²={result.r2:.6f}，MAPE={result.mape:.3e}，max|err|={result.max_abs/1e6:.4f} MPa，
模拟器交叉校验 sim_vs_exact={result.sim_vs_exact/1e6:.4f} MPa。
图见 `plots/`：调度曲线、剖面、PINN/解析/误差热图、模拟器自检、loss 曲线。

> 结论备注：见下方人工补充（这是高维纯物理尝试，效果好坏都留作记录）。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(lines)


def main(run_name: Optional[str] = None, no_lbfgs: bool = False,
         cfg: Optional[PinnTvConfig] = None) -> None:
    if cfg is None:
        cfg = PinnTvConfig()
    if no_lbfgs:
        cfg = replace(cfg, lbfgs_iters=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}  alpha_nd={cfg.alpha_nd:.4f}  q_nd(base)={cfg.q_nd:.3f}  "
          f"K={cfg.n_segments}  in_dim={2+cfg.n_segments}  net={cfg.hidden_layers}x{cfg.hidden_units}  "
          f"adam={cfg.adam_iters} lbfgs={cfg.lbfgs_iters}")

    net, history = train(cfg, device=device)
    net = net.to("cpu")

    per, agg, _ = metrics_over_schedules(net, cfg)
    print(f"[eval] {agg['n_schedules']} 条随机调度 (vs 解析真解): "
          f"R2_mean={agg['r2_mean']:.6f}  R2_min={agg['r2_min']:.6f}  "
          f"MAPE_mean={agg['mape_mean']:.3e}  max|err|_worst={agg['max_abs_MPa_worst']:.4f} MPa")

    rng = np.random.default_rng(cfg.plot_schedule_seed)
    plot_ratios = rng.uniform(cfg.q_ratio_min, cfg.q_ratio_max, size=cfg.n_segments)
    result = evaluate_at_schedule(net, cfg, plot_ratios)
    print(f"[plot-sched] R2={result.r2:.6f}  max|err|={result.max_abs/1e6:.4f} MPa  "
          f"sim_vs_exact={result.sim_vs_exact/1e6:.4f} MPa")

    out_dir = _run_dir(run_name)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(net.state_dict(), out_dir / "checkpoint.pt")
    _save_loss_history(history, out_dir / "loss_history.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "truth": "analytical (piecewise-constant Duhamel)",
            "aggregate": agg,
            "per_schedule": per,
            "plot_schedule": {
                "ratios": plot_ratios.tolist(),
                "r2": result.r2, "mape": result.mape, "l2_relative": result.l2_rel,
                "max_abs_MPa": result.max_abs / 1e6,
                "sim_vs_exact_max_MPa": result.sim_vs_exact / 1e6,
            },
        }, f, indent=2)
    np.save(out_dir / "pred.npy", result.p_pinn)
    np.save(out_dir / "exact.npy", result.p_exact)
    np.save(out_dir / "reference.npy", result.p_ref)
    np.save(out_dir / "plot_schedule_ratios.npy", plot_ratios)

    plot_loss_curve(history, plots_dir / "loss_curve.png")
    plot_schedule(plot_ratios, cfg.well_rate, plots_dir / "schedule.png")
    plot_all_eval(result, plots_dir, well_x=cfg.well_x_hat * cfg.L)
    _write_readme(out_dir / "README.md", cfg, agg, per, plot_ratios, result, run_name)

    print(f"[main] done. Output -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="时变井控参数化 PINN（实验性）")
    parser.add_argument("--name", default=None, help="本次运行标签（输出子文件夹后缀）")
    parser.add_argument("--no-lbfgs", action="store_true", help="消融：跳过 L-BFGS")
    args = parser.parse_args()
    main(run_name=args.name, no_lbfgs=args.no_lbfgs)
