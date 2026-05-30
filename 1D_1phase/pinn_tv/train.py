"""时变井控 PINN 训练：Adam（每步重采，含每点独立的整条调度）→ L-BFGS。

每个 collocation 点采自己随机的一整条 K 段调度——这样在 (x,t,调度) 联合高维空间做蒙特卡洛覆盖，
物理对每条调度逐点自洽。自适应权重三件套复用 pinn.train。
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from pinn.train import _log, _update_weights, _weighted_total
from pinn_tv.config import PinnTvConfig
from pinn_tv.losses import loss_components
from pinn_tv.net import TvPinnMLP


def _sample(cfg: PinnTvConfig, device: torch.device):
    """采 (x,t) + 每点独立的整条调度 q̂[K]（每段独立 ~U[ratio_min,ratio_max] 再归一）。"""
    rmin, rmax = cfg.q_ratio_min, cfg.q_ratio_max
    K = cfg.n_segments

    def _rand_sched(n: int) -> torch.Tensor:
        r = rmin + (rmax - rmin) * torch.rand(n, K, device=device)
        return cfg.ratio_to_qhat(r)

    x_uni = torch.rand(cfg.n_int, 1, device=device)
    band = cfg.well_band_hat
    x_well = cfg.well_x_hat + (torch.rand(cfg.n_int_well, 1, device=device) - 0.5) * 2.0 * band
    x_well = x_well.clamp(0.0, 1.0)
    x_int = torch.cat([x_uni, x_well], dim=0)
    t_int = torch.rand(x_int.shape[0], 1, device=device)
    s_int = _rand_sched(x_int.shape[0])

    x_ic = torch.rand(cfg.n_ic, 1, device=device)
    s_ic = _rand_sched(cfg.n_ic)
    t_bc = torch.rand(cfg.n_bc, 1, device=device)
    s_bc = _rand_sched(cfg.n_bc)
    return x_int, t_int, s_int, x_ic, s_ic, t_bc, s_bc


def train(cfg: PinnTvConfig, device: str = "cpu") -> Tuple[TvPinnMLP, List[dict]]:
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    net = TvPinnMLP(cfg.n_segments, cfg.hidden_layers, cfg.hidden_units,
                    hard_ic=cfg.hard_ic).to(dev)
    history: List[dict] = []
    weights = {"pde": cfg.w_pde, "ic": cfg.w_ic, "bc": cfg.w_bc}

    # ---------------- 阶段一：Adam ----------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    log_every = max(1, cfg.adam_iters // 50)
    for it in range(cfg.adam_iters):
        batch = _sample(cfg, dev)
        L_pde, L_ic, L_bc = loss_components(net, cfg, *batch)
        if cfg.adaptive_weights and it % cfg.adaptive_update_every == 0:
            _update_weights(net, weights, L_pde, L_ic, L_bc, cfg)
        loss = _weighted_total(weights, L_pde, L_ic, L_bc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == cfg.adam_iters - 1:
            comp = {"pde": L_pde.detach(), "ic": L_ic.detach(), "bc": L_bc.detach()}
            _log(history, "adam", it, float(loss.detach()), comp, weights)

    # ---------------- 阶段二：L-BFGS（固定一批，权重冻结） ----------------
    if cfg.lbfgs_iters > 0:
        batch = _sample(cfg, dev)
        opt_lbfgs = torch.optim.LBFGS(
            net.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe",
        )
        counter = {"it": 0}

        def closure():
            opt_lbfgs.zero_grad()
            L_pde, L_ic, L_bc = loss_components(net, cfg, *batch)
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
