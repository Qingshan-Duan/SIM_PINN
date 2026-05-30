"""参数化 PINN 评估：对一组流量比率逐 q 算指标（对解析真解），并为画图那个 q 做模拟器交叉校验。

两件事分开，避免浪费：
  - metrics_over_ratios：对 eval_q_ratios 每个 q **只**用解析解（便宜）算 R²/MAPE/L2/max，看 q 方向泛化。
  - evaluate_at_ratio：对**单个**（随机抽来画图的）q 跑完整 EvalResult，含细网格模拟器参考（贵），
    保留两层验证（解析真解 + 模拟器交叉校验 sim_vs_exact）。

真解逐 q 由解析解给：本场景线性 ⇒ 换 q 只要 replace(cfg, well_rate=ratio·基准) 重算 q_nd。
EvalResult / 参考模拟器 / 解析解全部复用 pinn 那套，口径与 pinn/ 完全一致。
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Tuple

import numpy as np
import torch

from pinn.analytical import analytical_on_eval_grid
from pinn.eval import EvalResult, _reference_on_eval_grid
from pinn_param.config import PinnParamConfig
from pinn_param.net import ParamPinnMLP


def _cfg_at_ratio(cfg: PinnParamConfig, ratio: float) -> PinnParamConfig:
    """把基准流量按比率缩放后的 config（well_rate=ratio·基准；__post_init__ 会重算 q_nd）。

    注意 cfg.well_rate 本身就是基准（r=1），所以该 q 的流量 = ratio · cfg.well_rate。
    """
    return replace(cfg, well_rate=ratio * cfg.well_rate)


def _eval_pinn_on_grid(net: ParamPinnMLP, cfg: PinnParamConfig, ratio: float
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 simulator 网格 (格心 x, 整步 t) 上算给定 ratio 的 PINN 预测，返回 (x_si, t_si, p_si[Pa])。

    网格尺寸与 ratio 无关（用基准 cfg 即可）；ratio 只决定喂进网络的常数 q̂。
    """
    dx = cfg.L / cfg.nx_eval
    x_si = (np.arange(cfg.nx_eval) + 0.5) * dx
    t_si = np.arange(cfg.n_steps_eval + 1) * cfg.dt_eval

    XX, TT = np.meshgrid(x_si, t_si, indexing="xy")
    x_hat = torch.tensor(XX.reshape(-1, 1) / cfg.L, dtype=torch.float32)
    t_hat = torch.tensor(TT.reshape(-1, 1) / cfg.T_end, dtype=torch.float32)
    q_hat = torch.full_like(x_hat, float(cfg.ratio_to_qhat(ratio)))

    net.eval()
    with torch.no_grad():
        p_hat = net(x_hat, t_hat, q_hat).cpu().numpy().reshape(TT.shape)
    p_si = cfg.p_ref + p_hat * cfg.dp_scale
    return x_si, t_si, p_si


def _metrics(err: np.ndarray, p_exact: np.ndarray) -> dict:
    """统一指标口径（与 pinn.eval 一致）：相对 L2、MAPE、R²、max|err|。"""
    l2_rel = float(np.linalg.norm(err) / np.linalg.norm(p_exact))
    max_abs = float(np.abs(err).max())
    mape = float(np.mean(np.abs(err) / np.abs(p_exact)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((p_exact - p_exact.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"l2_relative": l2_rel, "mape": mape, "r2": r2,
            "max_abs_Pa": max_abs, "max_abs_MPa": max_abs / 1e6}


def metrics_at_ratio(net: ParamPinnMLP, cfg: PinnParamConfig, ratio: float) -> dict:
    """单个比率的指标（只对解析真解，不跑模拟器，便宜）。"""
    cfg_q = _cfg_at_ratio(cfg, ratio)
    p_exact = analytical_on_eval_grid(cfg_q)
    _, _, p_pinn = _eval_pinn_on_grid(net, cfg, ratio)
    m = _metrics(p_pinn - p_exact, p_exact)
    return {"ratio": ratio, "rate": cfg_q.well_rate, **m}


def metrics_over_ratios(net: ParamPinnMLP, cfg: PinnParamConfig
                        ) -> Tuple[List[dict], dict]:
    """对 cfg.eval_q_ratios 逐 q 算指标，返回 (per_q 列表, 跨 q 汇总)。"""
    per_q = [metrics_at_ratio(net, cfg, float(r)) for r in cfg.eval_q_ratios]
    aggregate = {
        "n_q": len(per_q),
        "r2_mean": float(np.mean([d["r2"] for d in per_q])),
        "r2_min": float(np.min([d["r2"] for d in per_q])),
        "mape_mean": float(np.mean([d["mape"] for d in per_q])),
        "l2_relative_mean": float(np.mean([d["l2_relative"] for d in per_q])),
        "max_abs_MPa_worst": float(np.max([d["max_abs_MPa"] for d in per_q])),
    }
    return per_q, aggregate


def evaluate_at_ratio(net: ParamPinnMLP, cfg: PinnParamConfig, ratio: float) -> EvalResult:
    """单个 ratio 的完整评估（含细网格模拟器参考），用于画图 + 模拟器交叉校验。"""
    cfg_q = _cfg_at_ratio(cfg, ratio)
    p_exact = analytical_on_eval_grid(cfg_q)        # 解析真解（金标准）
    p_ref = _reference_on_eval_grid(cfg_q)          # 模拟器参考（细 dt+细 nx+匹配高斯井）
    x_si, t_si, p_pinn = _eval_pinn_on_grid(net, cfg, ratio)

    err = p_pinn - p_exact
    m = _metrics(err, p_exact)
    sim_vs_exact = float(np.abs(p_ref - p_exact).max())

    return EvalResult(
        grid_x=x_si, grid_t=t_si,
        p_pinn=p_pinn, p_exact=p_exact, p_ref=p_ref, err=err,
        l2_rel=m["l2_relative"], max_abs=m["max_abs_Pa"], mape=m["mape"],
        r2=m["r2"], sim_vs_exact=sim_vs_exact,
    )
