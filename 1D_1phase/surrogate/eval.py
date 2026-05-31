"""评估：两个层次的指标（对模拟器真值，held-out 测试调度）。

1. teacher-forced 单步：喂真 field^n 推一步 → 隔离“每步学得准不准”（干净，无累积）。
2. rollout 整条：从 IC 出发喂自己上一步输出滚 15 步 → 实战，看“漂不漂”。
物理正则的收益主要体现在 rollout（约束了 rollout 漂出训练流形的状态）。

p̂ 的单位本身就是 MPa（dp_scale=1e6 Pa），所以 max|err| 直接是 MPa。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from surrogate.config import SurrogateConfig
from surrogate.data_gen import make_transitions
from surrogate.net import AutoRegNet


def _metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "r2": 1.0 - ss_res / ss_tot,
        "rel_l2": float(np.linalg.norm(err) / np.linalg.norm(true)),
        "max_abs_mpa": float(np.max(np.abs(err))),
        "mean_abs_mpa": float(np.mean(np.abs(err))),
    }


@torch.no_grad()
def teacher_forced_metrics(net: AutoRegNet, phat: np.ndarray, ratios: np.ndarray,
                           cfg: SurrogateConfig, device: str = "cpu") -> Dict[str, float]:
    dev = torch.device(device)
    fi, qh, fo, _ = make_transitions(phat, ratios, cfg)
    pred = net(torch.tensor(fi, dtype=torch.float32, device=dev),
               torch.tensor(qh, dtype=torch.float32, device=dev)).cpu().numpy()
    return _metrics(pred, fo)


@torch.no_grad()
def rollout(net: AutoRegNet, phat: np.ndarray, ratios: np.ndarray,
            cfg: SurrogateConfig, device: str = "cpu") -> np.ndarray:
    """从 IC 自回归滚 n_steps，返回预测轨迹 (N, n_steps+1, nx)。"""
    dev = torch.device(device)
    qhat_all = cfg.ratio_to_qhat(ratios)                       # (N, n_steps)
    field = torch.tensor(phat[:, 0, :], dtype=torch.float32, device=dev)  # IC
    traj = [field]
    for n in range(cfg.n_steps):
        q = torch.tensor(qhat_all[:, n:n + 1], dtype=torch.float32, device=dev)
        field = net(field, q)
        traj.append(field)
    return torch.stack(traj, dim=1).cpu().numpy()


def rollout_metrics(net: AutoRegNet, phat: np.ndarray, ratios: np.ndarray,
                    cfg: SurrogateConfig, device: str = "cpu"
                    ) -> Tuple[Dict[str, float], np.ndarray]:
    """rollout 后 field^1..field^n 对真值的指标 + 预测轨迹。"""
    pred_traj = rollout(net, phat, ratios, cfg, device)
    pred = pred_traj[:, 1:, :]
    true = phat[:, 1:, :]
    m = _metrics(pred, true)
    # 逐调度最坏 max|err|（看最坏情况鲁棒性）
    per_sched_max = np.max(np.abs(pred - true), axis=(1, 2))
    m["worst_schedule_max_mpa"] = float(np.max(per_sched_max))
    return m, pred_traj


def evaluate(net: AutoRegNet, data: dict, cfg: SurrogateConfig,
             split: str = "test", device: str = "cpu") -> Dict[str, dict]:
    """teacher-forced + rollout 两套指标，对指定 split。"""
    phat = data[f"{split}_phat"]
    ratios = data[f"{split}_ratios"]
    tf = teacher_forced_metrics(net, phat, ratios, cfg, device)
    ro, _ = rollout_metrics(net, phat, ratios, cfg, device)
    return {"teacher_forced": tf, "rollout": ro}
