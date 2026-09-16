"""Candidate-specific historical-CLV composition block for LightGCN.

The two fixed coordinates compare a user's historical purchase-frequency and
transaction-value percentiles with item-side contexts built from training
data.  Only the relative N/V axis weights are learned.  ID and the two
coordinates are propagated together in one LightGCN and optimized by the same
recommendation loss.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def candidate_nv_fit_numpy(
    user_q_n: np.ndarray,
    user_q_v: np.ndarray,
    item_buyer_q_n: np.ndarray,
    item_amount_percentile: np.ndarray,
) -> np.ndarray:
    """Return F(u,i) for aligned one-dimensional user/item arrays."""

    q_n = np.asarray(user_q_n, dtype=np.float64)
    q_v = np.asarray(user_q_v, dtype=np.float64)
    b_n = np.asarray(item_buyer_q_n, dtype=np.float64)
    p_v = np.asarray(item_amount_percentile, dtype=np.float64)
    if not (q_n.shape == q_v.shape == b_n.shape == p_v.shape):
        raise ValueError("q_N·q_V·b_N·p_V shape이 같아야 합니다")
    if not all(np.isfinite(value).all() for value in (q_n, q_v, b_n, p_v)):
        raise ValueError("N/V 적합도 입력은 유한해야 합니다")
    if any(
        np.any((value < 0.0) | (value > 1.0))
        for value in (q_n, q_v, b_n, p_v)
    ):
        raise ValueError("N/V 적합도 입력은 [0,1] 범위여야 합니다")
    return (1.0 - 0.5 * (np.abs(q_n - b_n) + np.abs(q_v - p_v))).astype(
        np.float32
    )


def candidate_nv_fit_torch(
    user_q_n: torch.Tensor,
    user_q_v: torch.Tensor,
    item_buyer_q_n: torch.Tensor,
    item_amount_percentile: torch.Tensor,
) -> torch.Tensor:
    """Torch equivalent of :func:`candidate_nv_fit_numpy`."""

    if not (
        user_q_n.shape
        == user_q_v.shape
        == item_buyer_q_n.shape
        == item_amount_percentile.shape
    ):
        raise ValueError("q_N·q_V·b_N·p_V shape이 같아야 합니다")
    return 1.0 - 0.5 * (
        (user_q_n - item_buyer_q_n).abs()
        + (user_q_v - item_amount_percentile).abs()
    )


class CandidateNVFitLightGCN(nn.Module):
    """LightGCN with a fixed two-axis candidate-specific N/V representation."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_q_n: np.ndarray,
        user_q_v: np.ndarray,
        user_q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        item_buyer_q_n: np.ndarray,
        item_amount_percentile: np.ndarray,
        item_context_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.15,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if not 0.0 <= float(rho) <= 1.0 or n_layers < 0 or pref_reg < 0.0:
            raise ValueError("rho, n_layers 또는 pref_reg 설정이 잘못됐습니다")
        if adj.layout != torch.sparse_coo:
            raise ValueError("adj는 sparse COO tensor여야 합니다")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape이 사용자·상품 수와 다릅니다")

        q_n = np.asarray(user_q_n, dtype=np.float32)
        q_v = np.asarray(user_q_v, dtype=np.float32)
        q_c = np.asarray(user_q_c, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        buyer_q_n = np.asarray(item_buyer_q_n, dtype=np.float32)
        amount = np.asarray(item_amount_percentile, dtype=np.float32)
        item_valid = np.asarray(item_context_valid, dtype=bool)
        expected_user = (n_users,)
        expected_item = (n_items,)
        if any(value.shape != expected_user for value in (q_n, q_v, q_c, user_valid)):
            raise ValueError("사용자 CLV 입력 shape이 잘못됐습니다")
        if any(value.shape != expected_item for value in (buyer_q_n, amount, item_valid)):
            raise ValueError("상품 맥락 입력 shape이 잘못됐습니다")
        numeric = (q_n, q_v, q_c, buyer_q_n, amount)
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("N/V 입력은 모두 유한해야 합니다")
        if any(np.any((value < 0.0) | (value > 1.0)) for value in numeric):
            raise ValueError("N/V 입력은 [0,1] 범위여야 합니다")
        if any(np.any(value[~user_valid] != 0.0) for value in (q_n, q_v, q_c)):
            raise ValueError("CLV 무효 사용자의 q_N·q_V·q_C는 0이어야 합니다")

        user_coordinates = np.column_stack(
            [q_c * (2.0 * q_n - 1.0), q_c * (2.0 * q_v - 1.0)]
        ).astype(np.float32)
        user_coordinates[~user_valid] = 0.0
        item_coordinates = np.column_stack(
            [2.0 * buyer_q_n - 1.0, 2.0 * amount - 1.0]
        ).astype(np.float32)
        item_coordinates[~item_valid] = 0.0

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = 2
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        # 2 * softmax([0,0]) = [1,1], and the two weights always sum to 2.
        self.axis_logits = nn.Parameter(torch.zeros(2))
        self.register_buffer(
            "user_coordinates", torch.from_numpy(user_coordinates), persistent=False
        )
        self.register_buffer(
            "item_coordinates", torch.from_numpy(item_coordinates), persistent=False
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.economic_dim

    def axis_weights(self) -> torch.Tensor:
        return 2.0 * torch.softmax(self.axis_logits, dim=0)

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_weight = self.axis_weights().sqrt()
        return (
            self.user_coordinates * root_weight[None, :],
            self.item_coordinates * root_weight[None, :],
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
        selected_user, selected_item = user[users], item[items]
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
        if self.pref_reg <= 0.0:
            return self.E_u.weight.new_zeros(())
        if negatives.ndim != 2:
            raise ValueError("negatives는 [batch,K] shape이어야 합니다")
        batch = len(users)
        negative_mean = (
            self.E_i.weight[negatives].pow(2).sum(dim=2).mean(dim=1).sum()
        )
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
            "axis_weight_gradient_norm": norm(self.axis_logits),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        alpha = self.axis_weights()
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "alpha_n": float(alpha[0]),
            "alpha_v": float(alpha[1]),
            "alpha_sum": float(alpha.sum()),
            "joint_graph_propagation": True,
            "layer0_intervention": True,
            "one_dot_score": True,
            "external_reranking": False,
        }
