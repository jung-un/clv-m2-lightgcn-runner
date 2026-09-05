"""Joint LightGCN economic representation and positive-row weighted BPR."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def positive_row_weights(
    q_c: torch.Tensor,
    item_amount_percentile: torch.Tensor,
    *,
    train_mean_raw_weight: float,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """Return fixed CLV-by-economic positive weights with global mean mass one."""

    if q_c.ndim != 1 or item_amount_percentile.shape != q_c.shape:
        raise ValueError("q_c와 amount percentile shape이 같아야 합니다")
    if not torch.isfinite(q_c).all() or not torch.isfinite(item_amount_percentile).all():
        raise ValueError("q_c와 amount percentile은 유한해야 합니다")
    if torch.any((q_c < 0.0) | (q_c > 1.0)):
        raise ValueError("q_c 범위는 [0,1]이어야 합니다")
    if torch.any((item_amount_percentile < 0.0) | (item_amount_percentile > 1.0)):
        raise ValueError("amount percentile 범위는 [0,1]이어야 합니다")
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda는 [0,1]이어야 합니다")
    if not math.isfinite(float(train_mean_raw_weight)) or train_mean_raw_weight <= 0:
        raise ValueError("normalizer는 양의 유한값이어야 합니다")
    raw = 1.0 + float(lambda_) * q_c * (2.0 * item_amount_percentile - 1.0)
    if torch.any(raw <= 0.0):
        raise ValueError("양성 row weight는 양수여야 합니다")
    return raw / float(train_mean_raw_weight)


def weighted_multi_negative_bpr(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    row_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Average per-negative BPR first, then apply one fixed weight per positive."""

    if positive_scores.ndim != 1:
        raise ValueError("positive_scores는 [batch] shape이어야 합니다")
    if negative_scores.ndim != 2 or negative_scores.shape[0] != len(positive_scores):
        raise ValueError("negative_scores는 [batch,K] shape이어야 합니다")
    if negative_scores.shape[1] <= 0:
        raise ValueError("negative K는 1 이상이어야 합니다")
    if row_weights.shape != positive_scores.shape:
        raise ValueError("row_weights shape이 batch와 다릅니다")
    if not all(torch.isfinite(x).all() for x in (positive_scores, negative_scores, row_weights)):
        raise ValueError("점수와 row weight는 유한해야 합니다")
    if torch.any(row_weights <= 0.0):
        raise ValueError("row weight는 양수여야 합니다")
    per_negative = F.softplus(negative_scores - positive_scores[:, None])
    per_row = per_negative.mean(dim=1)
    loss = (row_weights * per_row).mean()
    with torch.no_grad():
        weight_mean = row_weights.mean()
        diagnostics: dict[str, torch.Tensor | int] = {
            "negative_count": int(negative_scores.shape[1]),
            "row_weight_mean": weight_mean,
            "row_weight_std": row_weights.std(unbiased=False),
            "row_weight_cv": row_weights.std(unbiased=False)
            / weight_mean.clamp_min(1e-12),
            "row_weight_min": row_weights.min(),
            "row_weight_max": row_weights.max(),
            "p_correct": (
                positive_scores[:, None] > negative_scores
            ).float().mean(),
            "effective_gradient_mass": (
                row_weights[:, None]
                * torch.sigmoid(negative_scores - positive_scores[:, None])
            ).mean(),
        }
    return loss, diagnostics


class M5EconomicLightGCN(nn.Module):
    """Binary LightGCN with one bounded four-dimensional economic block."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_economic_input: np.ndarray,
        user_economic_valid: np.ndarray,
        item_economic_input: np.ndarray,
        item_economic_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        economic_dim: int = 4,
        rho: float = 0.15,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, economic_dim) <= 0:
            raise ValueError("사용자·상품·임베딩 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0 or n_layers < 0 or pref_reg < 0:
            raise ValueError("rho, n_layers 또는 pref_reg 설정이 잘못됐습니다")
        user_input = np.asarray(user_economic_input, dtype=np.float32)
        item_input = np.asarray(item_economic_input, dtype=np.float32)
        user_valid = np.asarray(user_economic_valid, dtype=bool)
        item_valid = np.asarray(item_economic_valid, dtype=bool)
        if user_input.ndim != 2 or user_input.shape[0] != n_users:
            raise ValueError("사용자 경제입력 shape이 잘못됐습니다")
        if item_input.ndim != 2 or item_input.shape[0] != n_items:
            raise ValueError("상품 경제입력 shape이 잘못됐습니다")
        if user_valid.shape != (n_users,) or item_valid.shape != (n_items,):
            raise ValueError("경제입력 valid mask shape이 잘못됐습니다")
        if not np.isfinite(user_input).all() or not np.isfinite(item_input).all():
            raise ValueError("경제입력은 모두 유한해야 합니다")
        if np.any(user_input[~user_valid] != 0.0):
            raise ValueError("invalid 사용자의 경제입력은 0이어야 합니다")
        if np.any(item_input[~item_valid] != 0.0):
            raise ValueError("invalid 상품의 경제입력은 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = int(economic_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.user_economic_projection = nn.Linear(
            user_input.shape[1], economic_dim, bias=False
        )
        self.item_economic_projection = nn.Linear(
            item_input.shape[1], economic_dim, bias=False
        )
        nn.init.xavier_uniform_(self.user_economic_projection.weight)
        nn.init.xavier_uniform_(self.item_economic_projection.weight)

        self.register_buffer(
            "user_economic_input", torch.from_numpy(user_input.copy()), persistent=False
        )
        self.register_buffer(
            "item_economic_input", torch.from_numpy(item_input.copy()), persistent=False
        )
        self.register_buffer(
            "user_economic_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_valid",
            torch.from_numpy(item_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.economic_dim

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        user = 0.5 * torch.tanh(
            self.user_economic_projection(self.user_economic_input)
        )
        item = 0.5 * torch.tanh(
            self.item_economic_projection(self.item_economic_input)
        )
        return (
            user * self.user_economic_valid[:, None],
            item * self.item_economic_valid[:, None],
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_economic, item_economic = self.economic_coordinates()
        scale = math.sqrt(self.rho)
        return (
            torch.cat([self.E_u.weight, scale * user_economic], dim=1),
            torch.cat([self.E_i.weight, scale * item_economic], dim=1),
        )

    def _propagate(
        self, user: torch.Tensor, item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(self.E_u.weight, self.E_i.weight)

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(*self.layer0_embeddings())

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        user, item = self.propagated_embeddings()
        selected_user = user[users]
        selected_item = item[items]
        id_score = (
            selected_user[:, : self.id_dim] * selected_item[:, : self.id_dim]
        ).sum(dim=1)
        economic_score = (
            selected_user[:, self.id_dim :] * selected_item[:, self.id_dim :]
        ).sum(dim=1)
        return {
            "id": id_score,
            "economic": economic_score,
            "full": id_score + economic_score,
        }

    def sampled_l2(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        if negatives.ndim != 2:
            raise ValueError("negatives는 [batch,K] shape이어야 합니다")
        batch = len(users)
        negative_mean = self.E_i.weight[negatives].pow(2).sum(dim=2).mean(dim=1).sum()
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + negative_mean
        ) / batch

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "user_economic_projection_gradient_norm": norm(
                self.user_economic_projection.weight
            ),
            "item_economic_projection_gradient_norm": norm(
                self.item_economic_projection.weight
            ),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user, item = self.economic_coordinates()
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "bounded_projection": True,
            "per_row_unit_normalization": False,
            "learned_global_scale": False,
            "joint_graph_propagation": True,
            "layer0_intervention": True,
            "one_dot_score": True,
            "external_reranking": False,
            "user_economic_mean_norm": float(user.norm(dim=1).mean()),
            "item_economic_mean_norm": float(item.norm(dim=1).mean()),
            "user_projection_norm": float(
                self.user_economic_projection.weight.norm()
            ),
            "item_projection_norm": float(
                self.item_economic_projection.weight.norm()
            ),
        }
