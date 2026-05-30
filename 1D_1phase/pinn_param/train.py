"""参数化 PINN 训练循环：Adam（每步重采 collocation，含 q 轴）→ L-BFGS 精修。

与 pinn.train 结构一致，只是采样多一根流量轴 q̂、loss 用参数化版。
自适应梯度范数权重的三个辅助函数（_update_weights / _log / _weighted_total）与采样、loss
形式无关，直接从 pinn.train 复用，避免重复实现。
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from pinn.train import _log, _update_weights, _weighted_total
from pinn_param.config import PinnParamConfig
from pinn_param.losses import loss_components
from pinn_param.net import ParamPinnMLP


def _sample(cfg: PinnParamConfig, device: torch.device):
    """在 [0,1]×[0,1]×[q_ratio_min,q_ratio_max] 上随机采点。

    内部点 = 全域均匀 n_int + 井附近加密 n_int_well（源项局部、梯度陡）。
    每个点都带自己随机的流量比率 r，再映射成网络输入 q̂——这样 q 轴被稠密覆盖，物理对每个 q 自洽。
    """
    def _rand_qhat(n: int) -> torch.Tensor:
        r = cfg.q_ratio_min + (cfg.q_ratio_max - cfg.q_ratio_min) * torch.rand(n, 1, device=device)
        return cfg.ratio_to_qhat(r)

    x_uni = torch.rand(cfg.n_int, 1, device=device)
    band = cfg.well_band_hat
    x_well = cfg.well_x_hat + (torch.rand(cfg.n_int_well, 1, device=device) - 0.5) * 2.0 * band
    x_well = x_well.clamp(0.0, 1.0)
    x_int = torch.cat([x_uni, x_well], dim=0)
    t_int = torch.rand(x_int.shape[0], 1, device=device)
    q_int = _rand_qhat(x_int.shape[0])

    x_ic = torch.rand(cfg.n_ic, 1, device=device)
    q_ic = _rand_qhat(cfg.n_ic)
    t_bc = torch.rand(cfg.n_bc, 1, device=device)
    q_bc = _rand_qhat(cfg.n_bc)
    return x_int, t_int, q_int, x_ic, q_ic, t_bc, q_bc


def train(cfg: PinnParamConfig, device: str = "cpu") -> Tuple[ParamPinnMLP, List[dict]]:
    """返回训练好的参数化网络和 loss 历史。"""
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    net = ParamPinnMLP(cfg.hidden_layers, cfg.hidden_units, hard_ic=cfg.hard_ic).to(dev)
    history: List[dict] = []
    weights = {"pde": cfg.w_pde, "ic": cfg.w_ic, "bc": cfg.w_bc}

    # ---------------- 阶段一：Adam（每步重采） ----------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg.adam_lr)
    log_every = max(1, cfg.adam_iters // 50)
    for it in range(cfg.adam_iters):
        x_int, t_int, q_int, x_ic, q_ic, t_bc, q_bc = _sample(cfg, dev)
        L_pde, L_ic, L_bc = loss_components(net, cfg, x_int, t_int, q_int, x_ic, q_ic, t_bc, q_bc)
        if cfg.adaptive_weights and it % cfg.adaptive_update_every == 0:
            _update_weights(net, weights, L_pde, L_ic, L_bc, cfg)
        loss = _weighted_total(weights, L_pde, L_ic, L_bc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == cfg.adam_iters - 1:
            comp = {"pde": L_pde.detach(), "ic": L_ic.detach(), "bc": L_bc.detach()}
            _log(history, "adam", it, float(loss.detach()), comp, weights)

    # ---------------- 阶段二：L-BFGS（固定一批点，权重冻结） ----------------
    if cfg.lbfgs_iters > 0:
        x_int, t_int, q_int, x_ic, q_ic, t_bc, q_bc = _sample(cfg, dev)
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
            L_pde, L_ic, L_bc = loss_components(net, cfg, x_int, t_int, q_int,
                                                x_ic, q_ic, t_bc, q_bc)
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
