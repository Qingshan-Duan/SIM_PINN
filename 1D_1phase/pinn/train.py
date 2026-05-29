"""训练循环：先 Adam 探索，再 L-BFGS 精修。"""
from __future__ import annotations

from typing import List, Tuple

import torch

from pinn.config import PinnConfig
from pinn.losses import total_loss
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


def _log(history: List[dict], phase: str, it: int, loss: float, comp: dict) -> dict:
    entry = {"iter": it, "phase": phase, "total": loss,
             **{k: float(v) for k, v in comp.items()}}
    history.append(entry)
    print(f"[{phase:5s} {it:6d}] total={entry['total']:.3e}  "
          f"pde={entry['pde']:.3e} ic={entry['ic']:.3e} bc={entry['bc']:.3e}")
    return entry


def train(cfg: PinnConfig, device: str = "cpu") -> Tuple[PinnMLP, List[dict]]:
    """返回训练好的网络和 loss 历史（list of dict）。"""
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    net = PinnMLP(cfg.hidden_layers, cfg.hidden_units, hard_ic=cfg.hard_ic).to(dev)
    history: List[dict] = []

    # ---------------- 阶段一：Adam（每步重采 collocation 点） ----------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    log_every = max(1, cfg.adam_iters // 50)
    for it in range(cfg.adam_iters):
        x_int, t_int, x_ic, t_bc = _sample(cfg, dev)
        loss, comp = total_loss(net, cfg, x_int, t_int, x_ic, t_bc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == cfg.adam_iters - 1:
            _log(history, "adam", it, float(loss.detach()), comp)

    # ---------------- 阶段二：L-BFGS（固定一批点，二阶精修） ----------------
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
            loss, comp = total_loss(net, cfg, x_int, t_int, x_ic, t_bc)
            loss.backward()
            if counter["it"] % 50 == 0:
                _log(history, "lbfgs", cfg.adam_iters + counter["it"],
                     float(loss.detach()), comp)
            counter["it"] += 1
            return loss

        opt_lbfgs.step(closure)

    return net, history
