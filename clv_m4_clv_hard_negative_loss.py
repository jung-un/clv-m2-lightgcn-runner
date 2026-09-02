"""CLV-conditioned allocation across uniformly sampled BPR negatives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clv_conditioned_negative_weights(
    negative_scores: torch.Tensor,
    q_clv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unit-mass weights and the highest-scored negative per row."""

    if negative_scores.ndim != 2:
        raise ValueError("negative_scores는 [batch, K] shape이어야 합니다")
    batch, k = negative_scores.shape
    if k <= 0:
        raise ValueError("K는 1 이상이어야 합니다")
    if q_clv.shape != (batch,):
        raise ValueError("q_clv shape은 batch 길이와 같아야 합니다")
    if not torch.isfinite(negative_scores).all() or not torch.isfinite(q_clv).all():
        raise ValueError("점수와 q_clv는 모두 유한해야 합니다")
    if torch.any((q_clv < 0.0) | (q_clv > 1.0)):
        raise ValueError("q_clv 범위는 [0,1]이어야 합니다")

    hardest = negative_scores.detach().argmax(dim=1)
    weights = (1.0 - q_clv[:, None]).expand(-1, k) / float(k)
    weights = weights.clone()
    weights.scatter_add_(1, hardest[:, None], q_clv[:, None])
    return weights, hardest


def multi_negative_bpr(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    q_clv: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Average BPR at q=0 and highest-scored-negative BPR at q=1."""

    if positive_scores.ndim != 1:
        raise ValueError("positive_scores는 [batch] shape이어야 합니다")
    if negative_scores.shape[0] != positive_scores.shape[0]:
        raise ValueError("positive/negative batch shape이 다릅니다")
    weights, hardest = clv_conditioned_negative_weights(
        negative_scores, q_clv
    )
    row_losses = F.softplus(negative_scores - positive_scores[:, None])
    per_row = (weights * row_losses).sum(dim=1)
    hardest_weight = weights.gather(1, hardest[:, None]).squeeze(1)
    diagnostics = {
        "row_weight_sum_error": (weights.sum(1) - 1.0).abs().max(),
        "hardest_weight_mean": hardest_weight.mean(),
        "effective_gradient_mass": (
            weights * torch.sigmoid(negative_scores - positive_scores[:, None])
        ).sum(dim=1).mean(),
        "p_correct": (positive_scores[:, None] > negative_scores).float().mean(),
        "positive_hardest_gap": (
            positive_scores - negative_scores.gather(1, hardest[:, None]).squeeze(1)
        ).mean(),
    }
    return per_row.mean(), diagnostics


def sampled_l2_multineg(
    user_layer0: torch.Tensor,
    positive_layer0: torch.Tensor,
    negative_layer0: torch.Tensor,
    *,
    coefficient: float,
) -> torch.Tensor:
    """Match single-negative sampled-L2 scale by averaging K negatives."""

    if coefficient < 0:
        raise ValueError("coefficient는 0 이상이어야 합니다")
    if user_layer0.ndim != 2 or positive_layer0.shape != user_layer0.shape:
        raise ValueError("user/positive layer0 shape이 다릅니다")
    if (
        negative_layer0.ndim != 3
        or negative_layer0.shape[0] != user_layer0.shape[0]
        or negative_layer0.shape[2] != user_layer0.shape[1]
        or negative_layer0.shape[1] <= 0
    ):
        raise ValueError("negative_layer0는 [batch, K, dim]이어야 합니다")
    batch = user_layer0.shape[0]
    negative_mean = negative_layer0.pow(2).sum(dim=2).mean(dim=1).sum()
    return float(coefficient) * (
        user_layer0.pow(2).sum()
        + positive_layer0.pow(2).sum()
        + negative_mean
    ) / batch
