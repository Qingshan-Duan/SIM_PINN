"""训练自回归代理：Adam 全批（数据量小更稳），可选离散物理正则。

lambda_phys=0 → 纯数据；>0 → 数据 + 物理。物理 collocation 状态由 cfg.phys_states 决定
（data：训练 field^n 原地；augmented：流形附近采 n_phys 个似真场 + 自由 q）。
"""
from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from surrogate.config import SurrogateConfig
from surrogate.losses import data_loss, phys_loss, sample_phys_states
from surrogate.net import AutoRegNet


def _to_t(a: np.ndarray, dev) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.float32, device=dev)


def train(cfg: SurrogateConfig, train_data: Dict[str, np.ndarray],
          device: str = "cpu") -> Tuple[AutoRegNet, List[dict], float]:
    """返回 (net, history, wall_time_sec)。train_data 含 field_in/qhat/field_out/q_phys。"""
    torch.manual_seed(cfg.seed)
    dev = torch.device(device)
    net = AutoRegNet(cfg.nx, cfg.hidden_layers, cfg.hidden_units).to(dev)

    fi = _to_t(train_data["field_in"], dev)
    qh = _to_t(train_data["qhat"], dev)
    fo = _to_t(train_data["field_out"], dev)
    qp_data = _to_t(train_data["q_phys"], dev)

    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 1)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_decay_every, gamma=0.5)
    history: List[dict] = []
    log_every = max(1, cfg.n_iters // 50)

    t0 = time.perf_counter()
    for it in range(cfg.n_iters):
        opt.zero_grad()
        Ld = data_loss(net, fi, qh, fo) if cfg.use_data else torch.zeros((), device=dev)
        if cfg.lambda_phys > 0.0:
            if cfg.phys_states == "data":
                Lp = phys_loss(net, fi, qh, qp_data, cfg)
            else:
                # 物理 collocation 的场状态池 = 训练 field^n（物理-only 时只用场位置、不用标签）
                f, q, qp = sample_phys_states(fi, cfg, cfg.n_phys, gen)
                Lp = phys_loss(net, f, q, qp, cfg)
        else:
            Lp = torch.zeros((), device=dev)
        loss = Ld + cfg.lambda_phys * Lp
        loss.backward()
        opt.step()
        sched.step()
        if it % log_every == 0 or it == cfg.n_iters - 1:
            history.append({"iter": it, "data": float(Ld.detach()),
                            "phys": float(Lp.detach()), "total": float(loss.detach())})
    wall = time.perf_counter() - t0
    return net, history, wall
