"""Purchase-history user representation with an N/V-conditioned low-rank map.

This M2 variant has no free user-ID embedding.  A user starts from the
normalised aggregate of jointly learned item-ID embeddings on the same binary
graph as M1.  Centred, fixed train-history N/V percentiles condition two
bias-free low-rank linear maps of that history representation.  The resulting
user and item representations are trained together by one ordinary BPR loss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LowRankTransform(nn.Module):
    """Bias-free rank-r map, initialised as the zero map."""

    def __init__(self, dimension: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


def _centred_condition(
    percentile: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    values = np.asarray(percentile, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    centred = np.zeros_like(values, dtype=np.float32)
    if mask.any():
        centred[mask] = values[mask] - values[mask].mean(dtype=np.float64)
    return centred


class CLVHistoryConditionedLowRankLightGCN(nn.Module):
    """One LightGCN whose user layer-0 is a conditioned purchase history."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        user_activity_valid: np.ndarray,
        user_value_valid: np.ndarray,
        adj: torch.Tensor,
        embedding_dim: int = 64,
        transform_rank: int = 4,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or embedding_dim <= 0:
            raise ValueError("사용자·아이템·임베딩 차원은 양수여야 합니다")
        if not 0 < transform_rank <= embedding_dim:
            raise ValueError("변환 rank는 1 이상이고 임베딩 차원 이하여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        activity_valid = np.asarray(user_activity_valid, dtype=bool)
        value_valid = np.asarray(user_value_valid, dtype=bool)
        expected = (n_users,)
        if any(
            values.shape != expected
            for values in (q_n, q_v, activity_valid, value_valid)
        ):
            raise ValueError("N/V 수준·유효성 shape이 n_users와 다릅니다")
        if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
            raise ValueError("N/V 수준은 모두 유한해야 합니다")
        if ((q_n < 0) | (q_n > 1)).any() or ((q_v < 0) | (q_v > 1)).any():
            raise ValueError("N/V 백분위는 [0, 1] 범위여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.transform_rank = int(transform_rank)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # There is deliberately no free user-ID table.  Item IDs and both
        # conditional maps receive gradients from the same recommendation loss.
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.activity_transform = _LowRankTransform(embedding_dim, transform_rank)
        self.value_transform = _LowRankTransform(embedding_dim, transform_rank)

        self.register_buffer(
            "c_n", torch.from_numpy(_centred_condition(q_n, activity_valid))
        )
        self.register_buffer(
            "c_v", torch.from_numpy(_centred_condition(q_v, value_valid))
        )
        self.register_buffer(
            "user_activity_valid",
            torch.from_numpy(activity_valid.astype(np.float32)),
        )
        self.register_buffer(
            "user_value_valid", torch.from_numpy(value_valid.astype(np.float32))
        )
        self.register_buffer("adj", adj.coalesce())

    def purchase_history_expression(self) -> torch.Tensor:
        """Normalised one-hop aggregate of jointly learned item-ID vectors."""

        empty_users = self.E_i.weight.new_zeros(
            (self.n_users, self.embedding_dim)
        )
        item_seed = torch.cat([empty_users, self.E_i.weight], dim=0)
        aggregate = torch.sparse.mm(self.adj, item_seed)[: self.n_users]
        return F.normalize(aggregate, p=2, dim=1, eps=1e-12)

    def axis_corrections(self) -> tuple[torch.Tensor, torch.Tensor]:
        history = self.purchase_history_expression()
        activity = self.c_n[:, None] * self.activity_transform(history)
        value = self.c_v[:, None] * self.value_transform(history)
        return activity, value

    def _conditioned_user_layer0(self) -> torch.Tensor:
        history = self.purchase_history_expression()
        if self.rho == 0.0:
            return history
        activity = self.c_n[:, None] * self.activity_transform(history)
        value = self.c_v[:, None] * self.value_transform(history)
        raw = history + self.rho * (activity + value)
        history_norm = history.norm(dim=1, keepdim=True)
        raw_norm = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return raw * (history_norm / raw_norm)

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._conditioned_user_layer0(), self.E_i.weight

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        # need_value is kept only for the shared evaluation interface.
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_i.weight.new_zeros(())
        sampled_item_l2 = (
            self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled_item_l2

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        # gate is kept only for the shared trainer interface.
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if float(lam) != 0.0:
            raise ValueError("외부 점수 가산은 이 M2의 forward graph가 아닙니다")
        user, item = self.propagate()
        selected_user = user[users]
        positive_score = (selected_user * item[positives]).sum(1)
        negative_score = (selected_user * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "objective": "plain_bpr",
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict:
        def gradient_norm(parameter: torch.Tensor) -> float:
            if parameter.grad is None:
                return 0.0
            return float(parameter.grad.norm())

        return {
            "item_id_gradient_norm": gradient_norm(self.E_i.weight),
            "activity_down_gradient_norm": gradient_norm(
                self.activity_transform.down.weight
            ),
            "activity_up_gradient_norm": gradient_norm(
                self.activity_transform.up.weight
            ),
            "value_down_gradient_norm": gradient_norm(
                self.value_transform.down.weight
            ),
            "value_up_gradient_norm": gradient_norm(
                self.value_transform.up.weight
            ),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict:
        diagnostics = self.representation_diagnostics()
        keys = (
            "activity_correction_mean_norm",
            "value_correction_mean_norm",
            "activity_effective_ratio_to_history",
            "value_effective_ratio_to_history",
            "activity_transform_norm",
            "value_transform_norm",
            "mean_user_representation_change",
        )
        return {key: diagnostics[key] for key in keys}

    @torch.no_grad()
    def representation_diagnostics(self) -> dict:
        history = self.purchase_history_expression()
        user0, item0 = self.layer0_embeddings()
        activity, value = self.axis_corrections()
        history_mean_norm = history.norm(dim=1).mean().clamp_min(1e-12)
        return {
            "rho": self.rho,
            "transform_rank": self.transform_rank,
            "total_dim": self.embedding_dim,
            "free_user_id_embedding": False,
            "explicit_item_features": False,
            "separate_axis_scores": False,
            "user_base": "normalized_purchase_history",
            "activity_condition_mean": float(self.c_n.mean()),
            "activity_condition_std": float(self.c_n.std(unbiased=False)),
            "value_condition_mean": float(self.c_v.mean()),
            "value_condition_std": float(self.c_v.std(unbiased=False)),
            "activity_valid_share": float(self.user_activity_valid.mean()),
            "value_valid_share": float(self.user_value_valid.mean()),
            "history_mean_norm": float(history.norm(dim=1).mean()),
            "activity_correction_mean_norm": float(activity.norm(dim=1).mean()),
            "value_correction_mean_norm": float(value.norm(dim=1).mean()),
            "activity_effective_ratio_to_history": float(
                self.rho * activity.norm(dim=1).mean() / history_mean_norm
            ),
            "value_effective_ratio_to_history": float(
                self.rho * value.norm(dim=1).mean() / history_mean_norm
            ),
            "activity_transform_norm": float(
                self.activity_transform.up.weight.norm()
                * self.activity_transform.down.weight.norm()
            ),
            "value_transform_norm": float(
                self.value_transform.up.weight.norm()
                * self.value_transform.down.weight.norm()
            ),
            "mean_user_norm": float(user0.norm(dim=1).mean()),
            "mean_item_norm": float(item0.norm(dim=1).mean()),
            "max_user_norm_change": float(
                (user0.norm(dim=1) - history.norm(dim=1)).abs().max()
            ),
            "mean_user_representation_change": float(
                (user0 - history).norm(dim=1).mean()
            ),
        }
