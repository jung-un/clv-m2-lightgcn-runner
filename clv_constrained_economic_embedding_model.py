"""One LightGCN with explicit CLV composition and price coordinates."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedCLVEconomicLightGCN(nn.Module):
    """Match explicit user CLV coordinates to relation and price item coordinates."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        item_economic_features: np.ndarray,
        item_economic_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        clv_dim: int = 3,
        rho: float = 0.05,
        item_price_budget: float = 0.25,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, clv_dim) <= 0:
            raise ValueError("사용자·상품·표현 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if clv_dim != 3:
            raise ValueError("CLV 보조표현은 관계 2차원과 가격 1차원이어야 합니다")
        if not 0.0 <= item_price_budget <= 1.0:
            raise ValueError("상품 가격 예산은 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_features = np.asarray(item_economic_features, dtype=np.float32)
        item_valid = np.asarray(item_economic_valid, dtype=bool)
        expected_user = (n_users,)
        if any(array.shape != expected_user for array in (q_n, q_v, q_c, user_valid)):
            raise ValueError("사용자 CLV 입력 shape이 n_users와 다릅니다")
        if item_features.shape != (n_items, 2):
            raise ValueError("상품 경제 입력은 [n_items, 2]여야 합니다")
        if item_valid.shape != (n_items,):
            raise ValueError("상품 경제 입력 유효성 shape이 n_items와 다릅니다")
        if not all(
            np.isfinite(array).all()
            for array in (q_n, q_v, q_c, item_features)
        ):
            raise ValueError("CLV·상품 경제 입력은 모두 유한해야 합니다")
        for name, values in (("q_n", q_n), ("q_v", q_v), ("q_c", q_c)):
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name}은 [0,1] 범위여야 합니다")
        if float(np.max(np.abs(item_features), initial=0.0)) > 1.0 + 1e-6:
            raise ValueError("상품 경제 입력은 [-1,1] 범위여야 합니다")
        if any(np.any(values[~user_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.clv_dim = int(clv_dim)
        self.rho = float(rho)
        self.item_price_budget = float(item_price_budget)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # ID parameters are initialised first, preserving the matched M1 start.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.item_collaborative_projection = nn.Linear(id_dim, 2, bias=False)
        nn.init.normal_(self.item_collaborative_projection.weight, std=0.02)
        # A positive convex combination preserves the direction of both centred
        # price inputs while still letting the recommendation loss choose their
        # relative importance. The price channel therefore cannot be zeroed by
        # an unconstrained projection.
        self.item_price_logits = nn.Parameter(torch.zeros(2))

        self.register_buffer("q_n", torch.from_numpy(q_n.copy()), persistent=False)
        self.register_buffer("q_v", torch.from_numpy(q_v.copy()), persistent=False)
        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_features",
            torch.from_numpy(item_features.copy()),
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
        return self.id_dim + self.clv_dim

    def clv_user_embeddings(self) -> torch.Tensor:
        # The relation direction contains CLV level and N/V composition without
        # a learned user map that can silently discard either input.
        composition = self.q_n - self.q_v
        relation = self.q_c[:, None] * F.normalize(
            torch.stack([self.q_c, composition], dim=1),
            p=2,
            dim=1,
            eps=1e-12,
        )
        # This explicit coordinate is negative for low-V users and positive for
        # high-V users, so it has a fixed interpretation against centred price.
        price_preference = self.q_c * (2.0 * self.q_v - 1.0)
        return self.user_clv_valid[:, None] * torch.cat(
            [
                math.sqrt(1.0 - self.item_price_budget) * relation,
                math.sqrt(self.item_price_budget) * price_preference[:, None],
            ],
            dim=1,
        )

    def clv_item_embeddings(self) -> torch.Tensor:
        collaborative = F.normalize(
            self.item_collaborative_projection(self.E_i.weight),
            p=2,
            dim=1,
            eps=1e-12,
        )
        price_weights = torch.softmax(self.item_price_logits, dim=0)
        price = (
            (self.item_economic_features * price_weights[None, :]).sum(
                dim=1, keepdim=True
            )
            * self.item_economic_valid[:, None]
        )
        return torch.cat(
            [
                math.sqrt(1.0 - self.item_price_budget) * collaborative,
                math.sqrt(self.item_price_budget) * price,
            ],
            dim=1,
        )

    def component_embeddings(
        self, component: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if component not in {"relation", "price"}:
            raise ValueError("component는 relation 또는 price여야 합니다")
        scale = math.sqrt(self.rho)
        user_clv = self.clv_user_embeddings().clone()
        item_clv = self.clv_item_embeddings().clone()
        if component == "relation":
            user_clv[:, 2:] = 0.0
            item_clv[:, 2:] = 0.0
        else:
            user_clv[:, :2] = 0.0
            item_clv[:, :2] = 0.0
        return self._propagate(
            torch.cat([self.E_u.weight, scale * user_clv], dim=1),
            torch.cat([self.E_i.weight, scale * item_clv], dim=1),
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(self.rho)
        return (
            torch.cat(
                [self.E_u.weight, scale * self.clv_user_embeddings()], dim=1
            ),
            torch.cat(
                [self.E_i.weight, scale * self.clv_item_embeddings()], dim=1
            ),
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
        # need_value is retained only for the shared evaluator interface.
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        user, item = self.propagated_embeddings()
        id_score = (
            user[users, : self.id_dim] * item[items, : self.id_dim]
        ).sum(1)
        clv_score = (
            user[users, self.id_dim :] * item[items, self.id_dim :]
        ).sum(1)
        return {"id": id_score, "clv": clv_score, "full": id_score + clv_score}

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
            "item_collaborative_projection_gradient_norm": norm(
                self.item_collaborative_projection.weight
            ),
            "item_price_mixer_gradient_norm": norm(self.item_price_logits),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        return {
            key: diagnostics[key]
            for key in (
                "clv_user_mean_norm",
                "clv_item_mean_norm",
                "item_collaborative_projection_norm",
                "user_clv_norm_qc_mae",
                "item_price_weight_overall",
                "item_price_weight_within_category",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user_clv = self.clv_user_embeddings()
        item_clv = self.clv_item_embeddings()
        collaborative_item = item_clv[:, :2]
        price_item = item_clv[:, 2:]
        value_coordinate = self.q_c * (2.0 * self.q_v - 1.0)
        expected_user_norm = (
            self.q_c
            * torch.sqrt(
                (1.0 - self.item_price_budget)
                + self.item_price_budget * (2.0 * self.q_v - 1.0).pow(2)
            )
            * self.user_clv_valid
        )
        price_weights = torch.softmax(self.item_price_logits, dim=0)
        rho0_max = 0.0 if self.rho == 0.0 else float("nan")
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "clv_dim": self.clv_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "constrained_clv_economic_block": True,
            "explicit_clv_price_coordinates": True,
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "user_tanh": False,
            "free_item_response_embedding": False,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "explicit_item_price": True,
            "external_reranking": False,
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "item_economic_valid_share": float(self.item_economic_valid.mean()),
            "clv_user_mean_norm": float(user_clv.norm(dim=1).mean()),
            "clv_item_mean_norm": float(item_clv.norm(dim=1).mean()),
            "collaborative_item_mean_norm": float(
                collaborative_item.norm(dim=1).mean()
            ),
            "price_item_mean_norm": float(price_item.norm(dim=1).mean()),
            "item_price_budget": self.item_price_budget,
            "user_relation_level_mean_abs": float(self.q_c.abs().mean()),
            "user_relation_composition_mean_abs": float(
                (self.q_n - self.q_v).abs().mean()
            ),
            "user_price_coordinate_mean_abs": float(value_coordinate.abs().mean()),
            "item_collaborative_projection_norm": float(
                self.item_collaborative_projection.weight.norm()
            ),
            "item_price_weight_overall": float(price_weights[0]),
            "item_price_weight_within_category": float(price_weights[1]),
            "user_clv_norm_qc_mae": float(
                (user_clv.norm(dim=1) - expected_user_norm).abs().mean()
            ),
            "rho_zero_auxiliary_max_abs": rho0_max,
        }
