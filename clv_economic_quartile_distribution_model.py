"""LightGCN with one CLV-conditioned economic-quartile distribution block."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLVEconomicQuartileDistributionLightGCN(nn.Module):
    """Jointly propagate ID preference and a four-bin economic response block."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        user_economic_profile: np.ndarray,
        user_profile_valid: np.ndarray,
        item_economic_basis: np.ndarray,
        item_economic_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_c = np.asarray(q_c, dtype=np.float32)
        clv_valid = np.asarray(user_clv_valid, dtype=bool)
        user_profile = np.asarray(user_economic_profile, dtype=np.float32)
        profile_valid = np.asarray(user_profile_valid, dtype=bool)
        item_basis = np.asarray(item_economic_basis, dtype=np.float32)
        item_valid = np.asarray(item_economic_valid, dtype=bool)
        if q_c.shape != (n_users,) or clv_valid.shape != (n_users,):
            raise ValueError("사용자 CLV 입력 shape이 n_users와 다릅니다")
        if user_profile.ndim != 2 or user_profile.shape[0] != n_users:
            raise ValueError("사용자 경제 프로파일 shape이 잘못됐습니다")
        if item_basis.shape != (n_items, user_profile.shape[1]):
            raise ValueError("상품 경제 basis shape이 사용자 프로파일과 다릅니다")
        if profile_valid.shape != (n_users,) or item_valid.shape != (n_items,):
            raise ValueError("경제 입력 valid mask shape이 잘못됐습니다")
        if not all(
            np.isfinite(array).all()
            for array in (q_c, user_profile, item_basis)
        ):
            raise ValueError("CLV·경제 입력은 모두 유한해야 합니다")
        if ((q_c < 0.0) | (q_c > 1.0)).any():
            raise ValueError("q_c는 [0,1] 범위여야 합니다")
        effective_valid = clv_valid & profile_valid
        if np.any(q_c[~clv_valid] != 0.0):
            raise ValueError("CLV 계산 불가 사용자의 q_c는 0이어야 합니다")
        if np.any(user_profile[~profile_valid] != 0.0):
            raise ValueError("프로파일 계산 불가 사용자는 0 벡터여야 합니다")
        if np.any(item_basis[~item_valid] != 0.0):
            raise ValueError("경제 구간 계산 불가 상품은 0 벡터여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = int(user_profile.shape[1])
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # M1 parameters are initialised first, preserving exact rho=0 parity.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        # Only relative importance across bins is learned. Softmax fixes total
        # mass at one, so this parameter cannot inflate the global rho budget.
        self.economic_bin_logits = nn.Parameter(torch.zeros(self.economic_dim))

        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer(
            "user_economic_profile",
            torch.from_numpy(user_profile.copy()),
            persistent=False,
        )
        self.register_buffer(
            "effective_user_valid",
            torch.from_numpy(effective_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_basis",
            torch.from_numpy(item_basis.copy()),
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

    def economic_bin_weights(self) -> torch.Tensor:
        return torch.softmax(self.economic_bin_logits, dim=0)

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_weight = self.economic_bin_weights().sqrt()
        user = (
            self.q_c[:, None]
            * self.effective_user_valid[:, None]
            * self.user_economic_profile
            * root_weight[None, :]
        )
        item = (
            self.item_economic_valid[:, None]
            * self.item_economic_basis
            * root_weight[None, :]
        )
        return user, item

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
        ).sum(1)
        economic_score = (
            selected_user[:, self.id_dim :] * selected_item[:, self.id_dim :]
        ).sum(1)
        return {
            "id": id_score,
            "economic": economic_score,
            "full": id_score + economic_score,
        }

    def layer0_economic_scores(self) -> torch.Tensor:
        user, item = self.economic_coordinates()
        return self.rho * (user @ item.T)

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if float(lam) != 0.0:
            raise ValueError("완성된 점수에 외부 보정을 더할 수 없습니다")
        user, item = self.propagated_embeddings()
        selected_user = user[users]
        positive_score = (selected_user * item[positives]).sum(1)
        negative_score = (selected_user * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "objective": "plain_bpr",
                "p_correct": float((positive_score > negative_score).float().mean()),
            }
        return loss, diagnostics

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "economic_bin_weight_gradient_norm": norm(self.economic_bin_logits),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        weights = self.economic_bin_weights()
        return {
            "economic_bin_weight_min": float(weights.min()),
            "economic_bin_weight_max": float(weights.max()),
            "economic_bin_weight_entropy": float(
                -(weights * weights.clamp_min(1e-12).log()).sum()
            ),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user, item = self.economic_coordinates()
        weights = self.economic_bin_weights()
        diagnostics: dict[str, float | int | bool] = {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "user_economic_distribution_input": True,
            "economic_bin_count": self.economic_dim,
            "fixed_global_intervention_budget": True,
            "learned_global_scale": False,
            "learned_relative_bin_weights": True,
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "external_reranking": False,
            "effective_user_valid_share": float(self.effective_user_valid.mean()),
            "item_economic_valid_share": float(self.item_economic_valid.mean()),
            "user_economic_mean_norm": float(user.norm(dim=1).mean()),
            "item_economic_mean_norm": float(item.norm(dim=1).mean()),
            "economic_bin_weight_sum_error": float(abs(weights.sum() - 1.0)),
            "economic_bin_weight_min": float(weights.min()),
            "economic_bin_weight_max": float(weights.max()),
            "economic_bin_weight_entropy": float(
                -(weights * weights.clamp_min(1e-12).log()).sum()
            ),
            "layer0_economic_score_abs_upper_bound": float(
                self.rho * user.norm(dim=1).max() * item.norm(dim=1).max()
            ),
            "rho_zero_auxiliary_max_abs": (
                0.0 if self.rho == 0.0 else float("nan")
            ),
        }
        for index, weight in enumerate(weights, start=1):
            diagnostics[f"economic_bin_{index}_weight"] = float(weight)
        return diagnostics
