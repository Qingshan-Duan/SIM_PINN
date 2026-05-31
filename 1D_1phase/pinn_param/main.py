"""参数化 PINN 入口：训练 → 多 q 评估 → 随机抽一个 q 画图/交叉校验 → 结构化保存。

从 1D_1phase/ 运行：
    python -m pinn_param.main                 # 产物落到 output/pinn_param/<时间戳>/
    python -m pinn_param.main --name v1       # 产物落到 output/pinn_param/<时间戳>_v1/

每次运行写独立子文件夹（时间戳前缀），不覆盖历史，保留开发历程。落盘只进 output/pinn_param/，
与 pinn/ 的产物完全分开。画图复用 pinn.plot（图含义一致：解析实线 vs PINN 虚线）。
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
from pinn_param.config import PinnParamConfig
from pinn_param.eval import evaluate_at_ratio, metrics_over_ratios
from pinn_param.train import train


def _save_loss_history(history: list, path: Path) -> None:
    keys = ["iter", "phase", "total", "pde", "ic", "bc", "w_ic", "w_bc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in history:
            w.writerow({k: h.get(k, "") for k in keys})


def _save_pred_csv(result: EvalResult, path: Path) -> None:
    """画图那个 q 的预测，格式同 simulator pressure.csv：time_day, cell_0, ..."""
    nx = result.p_pinn.shape[1]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_day"] + [f"cell_{i}" for i in range(nx)])
        for j, t in enumerate(result.grid_t):
            w.writerow([t / 86400.0, *result.p_pinn[j].tolist()])


def _run_dir(name: Optional[str]) -> Path:
    """本次运行输出子文件夹：output/pinn_param/<时间戳>[_<name>]/。"""
    base = Path(__file__).resolve().parent.parent / "output" / "pinn_param"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = f"{stamp}_{name}" if name else stamp
    return base / folder


def _write_readme(path: Path, cfg: PinnParamConfig, per_q: List[dict],
                  aggregate: dict, plot_ratio: float, result: EvalResult,
                  run_name: Optional[str]) -> None:
    """自动给本次 run 写一份 README（保证"每个 run 都有 README"的约定不被忘记）。"""
    rows = "\n".join(
        f"| {d['ratio']:.2f} | {d['rate']:.3e} | {d['r2']:.6f} | {d['mape']:.3e} | "
        f"{d['l2_relative']:.3e} | {d['max_abs_MPa']:.4f} |"
        for d in per_q
    )
    lines = f"""# pinn_param run: {run_name or '(unnamed)'}

参数化 PINN（Family 1，纯物理无数据）：一个网络 `(x̂,t̂,q̂)→p̂` 覆盖基准流量
`{cfg.well_rate:.3e}` 上下扰动 ±{(cfg.q_ratio_max-1)*100:.0f}%（比率 [{cfg.q_ratio_min}, {cfg.q_ratio_max}]）的所有定常流量。

- 网络：{cfg.hidden_layers} 隐层 × {cfg.hidden_units}，tanh，3 输入；hard_ic={cfg.hard_ic}，自适应权重={cfg.adaptive_weights}
- 训练：Adam {cfg.adam_iters} + L-BFGS {cfg.lbfgs_iters}
- 误差对**解析真解**算（线性区精确，逐 q 由 analytical 缩放给出）

## 逐 q 指标（vs 解析真解）

| ratio | rate (m³/s) | R² | MAPE | L2(rel) | max\\|err\\| (MPa) |
|---|---|---|---|---|---|
{rows}

**跨 q 汇总**：R²_mean={aggregate['r2_mean']:.6f}，R²_min={aggregate['r2_min']:.6f}，
MAPE_mean={aggregate['mape_mean']:.3e}，max\\|err\\|_worst={aggregate['max_abs_MPa_worst']:.4f} MPa。

## 画图的 q（随机抽，seed={cfg.plot_q_seed}）

ratio={plot_ratio:.4f}（rate={plot_ratio*cfg.well_rate:.3e} m³/s）。
该 q：R²={result.r2:.6f}，MAPE={result.mape:.3e}，max|err|={result.max_abs/1e6:.4f} MPa。
模拟器交叉校验 sim_vs_exact={result.sim_vs_exact/1e6:.4f} MPa（尺子自检）。

图见 `plots/`：剖面（解析实线 vs PINN 虚线）、PINN/解析/误差热图、模拟器自检热图、loss 曲线。
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(lines)


def main(run_name: Optional[str] = None, no_lbfgs: bool = False,
         cfg: Optional[PinnParamConfig] = None) -> None:
    if cfg is None:
        cfg = PinnParamConfig()
    if no_lbfgs:
        cfg = replace(cfg, lbfgs_iters=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[main] device={device}  alpha_nd={cfg.alpha_nd:.4f}  q_nd(base)={cfg.q_nd:.3f}  "
          f"ratio∈[{cfg.q_ratio_min},{cfg.q_ratio_max}]  lbfgs_iters={cfg.lbfgs_iters}")

    # 1. 训练
    net, history = train(cfg, device=device)
    net = net.to("cpu")

    # 2. 多 q 评估（只用解析解，便宜）
    per_q, aggregate = metrics_over_ratios(net, cfg)
    print("[eval] 逐 q (vs 解析真解):")
    for d in per_q:
        print(f"   ratio={d['ratio']:.2f}  R2={d['r2']:.6f}  MAPE={d['mape']:.3e}  "
              f"L2={d['l2_relative']:.3e}  max|err|={d['max_abs_MPa']:.4f} MPa")
    print(f"[eval] 汇总: R2_mean={aggregate['r2_mean']:.6f}  R2_min={aggregate['r2_min']:.6f}  "
          f"MAPE_mean={aggregate['mape_mean']:.3e}  max|err|_worst={aggregate['max_abs_MPa_worst']:.4f} MPa")

    # 3. 随机抽一个 q 做完整评估（含模拟器参考）+ 画图
    rng = np.random.default_rng(cfg.plot_q_seed)
    plot_ratio = float(rng.uniform(cfg.q_ratio_min, cfg.q_ratio_max))
    result = evaluate_at_ratio(net, cfg, plot_ratio)
    print(f"[plot-q] ratio={plot_ratio:.4f}  R2={result.r2:.6f}  max|err|={result.max_abs/1e6:.4f} MPa  "
          f"sim_vs_exact={result.sim_vs_exact/1e6:.4f} MPa")

    # 4. 结构化保存（只进 output/pinn_param/）
    out_dir = _run_dir(run_name)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    torch.save(net.state_dict(), out_dir / "checkpoint.pt")
    _save_loss_history(history, out_dir / "loss_history.csv")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "truth": "analytical",
            "plot_q": {"ratio": plot_ratio, "rate": plot_ratio * cfg.well_rate,
                       "r2": result.r2, "mape": result.mape, "l2_relative": result.l2_rel,
                       "max_abs_MPa": result.max_abs / 1e6,
                       "sim_vs_exact_max_MPa": result.sim_vs_exact / 1e6},
            "per_q": per_q,
            "aggregate": aggregate,
        }, f, indent=2)

    # 画图那个 q 的场（其余 q 不逐个落盘，避免文件爆炸；要复现任意 q 用 checkpoint+ratio 即可）
    np.save(out_dir / "pred.npy", result.p_pinn)
    np.save(out_dir / "exact.npy", result.p_exact)
    np.save(out_dir / "reference.npy", result.p_ref)
    _save_pred_csv(result, out_dir / "pred.csv")

    plot_loss_curve(history, plots_dir / "loss_curve.png")
    plot_all_eval(result, plots_dir, well_x=cfg.well_x_hat * cfg.L)
    _write_readme(out_dir / "README.md", cfg, per_q, aggregate, plot_ratio, result, run_name)

    print(f"[main] done. Output -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 1D 单相参数化 PINN（流量为输入轴）并评估")
    parser.add_argument("--name", default=None, help="本次运行标签，作为输出子文件夹后缀")
    parser.add_argument("--no-lbfgs", action="store_true", help="消融：跳过 L-BFGS")
    args = parser.parse_args()
    main(run_name=args.name, no_lbfgs=args.no_lbfgs)
