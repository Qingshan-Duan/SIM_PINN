"""训练循环：先 Adam 探索，再 L-BFGS 精修。支持梯度范数平衡的自适应损失权重。"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from pinn.config import PinnConfig
from pinn.losses import loss_components
from pinn.net import PinnMLP


def _sample(cfg: PinnConfig, device: torch.device):
    """在无量纲域 [0,1]×[0,1] 上随机采点：内部、IC（t=0 的 x 切片）、BC（x=0/1 的 t 切片）。

    内部点 = 全域均匀 n_int + 井附近加密 n_int_well（源项局部、梯度陡，均匀采样会欠采井区）。
    """
    x_uni = torch.rand(cfg.n_int, 1, device=device)
    # 井附近 [well_x_hat ± well_band_hat] 内额外采点，clamp 回 [0,1]
    band = cfg.well_band_hat
    x_well = cfg.well_x_hat + (torch.rand(cfg.n_int_well, 1, device=device) - 0.5) * 2.0 * band
    x_well = x_well.clamp(0.0, 1.0)
    x_int = torch.cat([x_uni, x_well], dim=0)
    t_int = torch.rand(x_int.shape[0], 1, device=device)
    x_ic = torch.rand(cfg.n_ic, 1, device=device)
    t_bc = torch.rand(cfg.n_bc, 1, device=device)
    return x_int, t_int, x_ic, t_bc


def _flat_grad_abs(loss: torch.Tensor, params: List[torch.Tensor]) -> torch.Tensor:
    """∂loss/∂θ 拉平取绝对值（retain_graph 以便后续还能 backward 总 loss）。"""
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([g.reshape(-1).abs() for g in grads if g is not None])


def _update_weights(net: PinnMLP, weights: Dict[str, float],
                    L_pde: torch.Tensor, L_ic: torch.Tensor, L_bc: torch.Tensor,
                    cfg: PinnConfig) -> None:
    """梯度范数平衡（Wang 2021 的对称变体）：ŵ_i = ‖∇L_pde‖ / ‖∇L_i‖，再 EMA 更新（就地）。

    PDE 权重固定作基准，只调 w_ic / w_bc。哪一项梯度被 PDE 压得越小，权重抬得越高。
    用 L2 范数比（而非原版 max/mean）：尖源会让 PDE 梯度某分量畸大，max/mean 会把比值拉到病态高。
    末了夹到 [1, adaptive_w_max] 防跑飞。
    """
    params = [p for p in net.parameters() if p.requires_grad]
    g_pde = _flat_grad_abs(L_pde, params).norm()
    a = cfg.adaptive_ema

    def _ema(w_old: float, g_i: torch.Tensor) -> float:
        w_hat = float(g_pde / (g_i.norm() + 1e-12))
        w_new = a * w_old + (1.0 - a) * w_hat
        return float(min(max(w_new, 1.0), cfg.adaptive_w_max))

    if not cfg.hard_ic:                       # 硬约束无 IC 项，跳过
        weights["ic"] = _ema(weights["ic"], _flat_grad_abs(L_ic, params))
    weights["bc"] = _ema(weights["bc"], _flat_grad_abs(L_bc, params))


def _log(history: List[dict], phase: str, it: int, loss: float,
         comp: Dict[str, float], weights: Dict[str, float]) -> dict:
    entry = {"iter": it, "phase": phase, "total": loss,
             **{k: float(v) for k, v in comp.items()},
             "w_ic": weights["ic"], "w_bc": weights["bc"]}
    history.append(entry)
    print(f"[{phase:5s} {it:6d}] total={entry['total']:.3e}  "
          f"pde={entry['pde']:.3e} ic={entry['ic']:.3e} bc={entry['bc']:.3e}  "
          f"w_ic={weights['ic']:.1f} w_bc={weights['bc']:.1f}")
    return entry


def _weighted_total(weights: Dict[str, float],
                    L_pde: torch.Tensor, L_ic: torch.Tensor, L_bc: torch.Tensor) -> torch.Tensor:
    return weights["pde"] * L_pde + weights["ic"] * L_ic + weights["bc"] * L_bc


def train(cfg: PinnConfig, device: str = "cpu") -> Tuple[PinnMLP, List[dict]]:
    """返回训练好的网络和 loss 历史（list of dict）。"""
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    net = PinnMLP(cfg.hidden_layers, cfg.hidden_units, hard_ic=cfg.hard_ic).to(dev)
    history: List[dict] = []
    # 动态权重，从 cfg 初值起步；开自适应时在 Adam 阶段更新，L-BFGS 阶段冻结沿用
    weights = {"pde": cfg.w_pde, "ic": cfg.w_ic, "bc": cfg.w_bc}

    # ---------------- 阶段一：Adam（每步重采 collocation 点） ----------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    log_every = max(1, cfg.adam_iters // 50)
    for it in range(cfg.adam_iters):
        x_int, t_int, x_ic, t_bc = _sample(cfg, dev)
        L_pde, L_ic, L_bc = loss_components(net, cfg, x_int, t_int, x_ic, t_bc)
        if cfg.adaptive_weights and it % cfg.adaptive_update_every == 0:
            _update_weights(net, weights, L_pde, L_ic, L_bc, cfg)
        loss = _weighted_total(weights, L_pde, L_ic, L_bc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == cfg.adam_iters - 1:
            comp = {"pde": L_pde.detach(), "ic": L_ic.detach(), "bc": L_bc.detach()}
            _log(history, "adam", it, float(loss.detach()), comp, weights)

    # ---------------- 阶段二：L-BFGS（固定一批点，二阶精修；权重冻结） ----------------
    # L-BFGS 是全批量优化器，需要固定数据集；每步重采会破坏它的曲率估计。
    if cfg.lbfgs_iters > 0:
        x_int, t_int, x_ic, t_bc = _sample(cfg, dev)
        opt_lbfgs = torch.optim.LBFGS(
            net.parameters(),
            max_iter=cfg.lbfgs_iters,
            history_size=50,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )
        counter = {"it": 0}

        def closure():
            opt_lbfgs.zero_grad()
            L_pde, L_ic, L_bc = loss_components(net, cfg, x_int, t_int, x_ic, t_bc)
            loss = _weighted_total(weights, L_pde, L_ic, L_bc)
            loss.backward()
            if counter["it"] % 50 == 0:
                comp = {"pde": L_pde.detach(), "ic": L_ic.detach(), "bc": L_bc.detach()}
                _log(history, "lbfgs", cfg.adam_iters + counter["it"],
                     float(loss.detach()), comp, weights)
            counter["it"] += 1
            return loss

        opt_lbfgs.step(closure)

    return net, history
