"""损失：数据 MSE（主力） + 离散 BE 残差（正则）。

数据项：teacher-forced，喂真场 field^n 与 q̂，让输出贴 field^{n+1} 真值。
物理项：要求“网络输出的 field^{n+1} 对任意 (field^n, q) 满足离散 BE”，**不需要真值标签**——
这正是正则发力处（在数据点之外约束这张一步映射的形状）。collocation 状态两种来源见 cfg.phys_states。
"""
from __future__ import annotations

import torch

from surrogate.config import SurrogateConfig
from surrogate.net import AutoRegNet
from surrogate.physics import be_residual


def data_loss(net: AutoRegNet, field_in: torch.Tensor, qhat: torch.Tensor,
              field_out: torch.Tensor) -> torch.Tensor:
    pred = net(field_in, qhat)
    return torch.mean((pred - field_out) ** 2)


def phys_loss(net: AutoRegNet, field_in: torch.Tensor, qhat: torch.Tensor,
              q_phys: torch.Tensor, cfg: SurrogateConfig) -> torch.Tensor:
    pred = net(field_in, qhat)
    r = be_residual(pred, field_in, q_phys, cfg)
    return torch.mean(r ** 2)


def sample_phys_states(field_pool: torch.Tensor, cfg: SurrogateConfig,
                       n: int, generator: torch.Generator):
    """augmented 物理 collocation：在已知场附近填满 (场, q) 空间，q 自由采样。

    field_pool (P, nx) = 训练转移对里的所有 field^n。做法：随机取两条场做凸插值 + 高斯扰动，
    得到流形附近的“似真场”；q̂ ~ U[−1,1] 自由采。BE 对任意场都成立，故这些点都是合法物理监督，
    且专治 rollout 漂出流形的状态。返回 (field, qhat, q_phys)，均在 field_pool.device 上。
    """
    dev = field_pool.device
    P = field_pool.shape[0]
    i = torch.randint(P, (n,), generator=generator, device=dev)
    j = torch.randint(P, (n,), generator=generator, device=dev)
    w = torch.rand(n, 1, generator=generator, device=dev)
    field = w * field_pool[i] + (1.0 - w) * field_pool[j]
    field = field + cfg.phys_perturb * field.std() * torch.randn(
        field.shape, generator=generator, device=dev)
    qhat = 2.0 * torch.rand(n, 1, generator=generator, device=dev) - 1.0
    q_phys = cfg.qhat_to_ratio(qhat) * cfg.well_rate_base
    return field, qhat, q_phys
