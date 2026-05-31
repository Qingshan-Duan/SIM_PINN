"""仿射最小二乘基线：数据下限参照。

1D 单相线性 ⇒ 一步算子精确仿射 `场ⁿ⁺¹ = W·[场ⁿ, q̂, 1]`。直接最小二乘拟合 W（连神经网络
都不用），在测试集 rollout。它解释了"为什么一条调度就够"：算子是仿射的，少量秩足够的样本
就能辨识 W，之后对任意 (场,q) 都成立。作为对比锚：纯数据 MLP/物理 都不应比它更差太多。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from surrogate.config import SurrogateConfig
from surrogate.data_gen import make_transitions
from surrogate.eval import _metrics


def affine_fit(phat: np.ndarray, ratios: np.ndarray, cfg: SurrogateConfig) -> np.ndarray:
    """最小二乘拟合仿射映射，返回 W (nx+2, nx)。"""
    fi, qh, fo, _ = make_transitions(phat, ratios, cfg)
    K = fi.shape[0]
    X = np.concatenate([fi, qh, np.ones((K, 1))], axis=1)   # (K, nx+2)
    W, *_ = np.linalg.lstsq(X, fo, rcond=None)
    return W


def affine_rollout_metrics(train_phat: np.ndarray, train_ratios: np.ndarray,
                           test_phat: np.ndarray, test_ratios: np.ndarray,
                           cfg: SurrogateConfig) -> Dict[str, float]:
    """用训练集拟合仿射映射，在测试集从 IC rollout，返回 rollout 指标。"""
    W = affine_fit(train_phat, train_ratios, cfg)
    qhat_all = cfg.ratio_to_qhat(test_ratios)               # (M, n_steps)
    M = test_phat.shape[0]
    field = test_phat[:, 0, :].copy()                        # IC
    preds = [field.copy()]
    for n in range(cfg.n_steps):
        Xn = np.concatenate([field, qhat_all[:, n:n + 1], np.ones((M, 1))], axis=1)
        field = Xn @ W
        preds.append(field.copy())
    pred = np.stack(preds, 1)[:, 1:, :]
    m = _metrics(pred, test_phat[:, 1:, :])
    per = np.max(np.abs(pred - test_phat[:, 1:, :]), axis=(1, 2))
    m["worst_schedule_max_mpa"] = float(np.max(per))
    return m
