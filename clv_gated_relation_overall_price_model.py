"""LightGCN with a small CLV-conditioned relation and overall-price block."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedRelationOverallPriceLightGCN(nn.Module):
    """Jointly propagate ID, CLV-conditioned relation, and price coordinates."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        user_overall_price: np.ndarray,
        user_price_valid: np.ndarray,
        item_overall_price: np.ndarray,
        item_price_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        auxiliary_dim: int = 3,
        rho: float = 0.05,
        price_budget: float = 0.25,
        price_scale_delta: float = 0.5,
        relation_gate_initial: float = 0.9,
        relation_level_slope_initial: float = 0.1,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, auxiliary_dim) <= 0:
            raise ValueError("사용자·상품·표현 차원은 양수여야 합니다")
        if auxiliary_dim != 3:
            raise ValueError("보조표현은 관계 2차원과 가격 1차원이어야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if not 0.0 <= price_budget <= 1.0:
            raise ValueError("가격 예산은 0 이상 1 이하여야 합니다")
        if not 0.0 < price_scale_delta < 1.0:
            raise ValueError("가격 scale 범위는 부호가 바뀌지 않도록 (0,1)이어야 합니다")
        if not 0.0 < relation_gate_initial < 1.0:
            raise ValueError("관계 gate 초기값은 (0,1)이어야 합니다")
        if relation_level_slope_initial <= 0.0:
            raise ValueError("관계 level slope 초기값은 양수여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        user_clv_valid = np.asarray(user_clv_valid, dtype=bool)
        user_overall_price = np.asarray(user_overall_price, dtype=np.float32)
        user_price_valid = np.asarray(user_price_valid, dtype=bool)
        item_overall_price = np.asarray(item_overall_price, dtype=np.float32)
        item_price_valid = np.asarray(item_price_valid, dtype=bool)
        expected_user = (n_users,)
        expected_item = (n_items,)
        if any(
            values.shape != expected_user
            for values in (
                q_n,
                q_v,
                q_c,
                user_clv_valid,
                user_overall_price,
                user_price_valid,
            )
        ):
            raise ValueError("사용자 입력 shape이 n_users와 다릅니다")
        if any(
            values.shape != expected_item
            for values in (item_overall_price, item_price_valid)
        ):
            raise ValueError("상품 입력 shape이 n_items와 다릅니다")
        numeric = (q_n, q_v, q_c, user_overall_price, item_overall_price)
        if not all(np.isfinite(values).all() for values in numeric):
            raise ValueError("CLV·가격 입력은 모두 유한해야 합니다")
        for name, values in (
            ("q_n", q_n),
            ("q_v", q_v),
            ("q_c", q_c),
            ("user_overall_price", user_overall_price),
            ("item_overall_price", item_overall_price),
        ):
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name}은 [0,1] 범위여야 합니다")
        if any(np.any(values[~user_clv_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")
        if np.any(user_overall_price[~user_price_valid] != 0.5):
            raise ValueError("가격 계산 불가 사용자의 가격 위치는 0.5여야 합니다")
        if np.any(item_overall_price[~item_price_valid] != 0.5):
            raise ValueError("가격 계산 불가 상품의 가격 위치는 0.5여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.rho = float(rho)
        self.price_budget = float(price_budget)
        self.price_scale_delta = float(price_scale_delta)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # Keep M1's ID initialization order unchanged for the rho=0 control.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.item_relation_projection = nn.Linear(id_dim, 2, bias=False)
        nn.init.normal_(self.item_relation_projection.weight, std=0.02)

        gate_logit = math.log(relation_gate_initial / (1.0 - relation_gate_initial))
        level_raw = math.log(math.expm1(relation_level_slope_initial))
        self.relation_gate_bias = nn.Parameter(torch.tensor(gate_logit))
        self.relation_gate_level_raw = nn.Parameter(torch.tensor(level_raw))
        self.relation_gate_composition = nn.Parameter(torch.tensor(0.0))
        # tanh(0)=0, hence the initial positive price scale is exactly one.
        self.item_price_scale_raw = nn.Parameter(torch.tensor(0.0))

        def buffer(name: str, values: np.ndarray):
            self.register_buffer(name, torch.from_numpy(values.copy()), persistent=False)

        buffer("q_n", q_n)
        buffer("q_v", q_v)
        buffer("q_c", q_c)
        buffer("user_clv_valid", user_clv_valid.astype(np.float32))
        buffer("user_overall_price", user_overall_price)
        buffer("user_price_valid", user_price_valid.astype(np.float32))
        buffer("item_overall_price", item_overall_price)
        buffer("item_price_valid", item_price_valid.astype(np.float32))
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.auxiliary_dim

    def relation_gate(self) -> torch.Tensor:
        composition = self.q_n - self.q_v
        level_slope = F.softplus(self.relation_gate_level_raw)
        logits = (
            self.relation_gate_bias
            + level_slope * (self.q_c - 0.5)
            + self.relation_gate_composition * composition
        )
        return torch.sigmoid(logits) * self.user_clv_valid

    def item_price_scale(self) -> torch.Tensor:
        return 1.0 + self.price_scale_delta * torch.tanh(self.item_price_scale_raw)

    def auxiliary_user_embeddings(self) -> torch.Tensor:
        composition = self.q_n - self.q_v
        relation_direction = F.normalize(
            torch.stack([self.q_c, composition], dim=1),
            p=2,
            dim=1,
            eps=1e-12,
        )
        relation = (
            self.q_c[:, None]
            * self.relation_gate()[:, None]
            * relation_direction
            * self.user_clv_valid[:, None]
        )
        price = (
            self.q_c
            * (2.0 * self.user_overall_price - 1.0)
            * self.user_clv_valid
            * self.user_price_valid
        )
        return torch.cat(
            [
                math.sqrt(1.0 - self.price_budget) * relation,
                math.sqrt(self.price_budget) * price[:, None],
            ],
            dim=1,
        )

    def auxiliary_item_embeddings(self) -> torch.Tensor:
        relation = F.normalize(
            self.item_relation_projection(self.E_i.weight),
            p=2,
            dim=1,
            eps=1e-12,
        )
        price = (
            self.item_price_scale()
            * (2.0 * self.item_overall_price - 1.0)
            * self.item_price_valid
        )
        return torch.cat(
            [
                math.sqrt(1.0 - self.price_budget) * relation,
                math.sqrt(self.price_budget) * price[:, None],
            ],
            dim=1,
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(self.rho)
        return (
            torch.cat([self.E_u.weight, scale * self.auxiliary_user_embeddings()], dim=1),
            torch.cat([self.E_i.weight, scale * self.auxiliary_item_embeddings()], dim=1),
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

    def component_embeddings(
        self, component: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if component not in {"relation", "price"}:
            raise ValueError("component는 relation 또는 price여야 합니다")
        user_aux = self.auxiliary_user_embeddings().clone()
        item_aux = self.auxiliary_item_embeddings().clone()
        if component == "relation":
            user_aux[:, 2:] = 0.0
            item_aux[:, 2:] = 0.0
        else:
            user_aux[:, :2] = 0.0
            item_aux[:, :2] = 0.0
        scale = math.sqrt(self.rho)
        return self._propagate(
            torch.cat([self.E_u.weight, scale * user_aux], dim=1),
            torch.cat([self.E_i.weight, scale * item_aux], dim=1),
        )

    def embeddings(self, need_value: bool = True):
        # need_value is retained only for the shared evaluator interface.
        user, item = self.propagated_embeddings()
        return (
            user,
            item,
            user.new_zeros((self.n_users, 1)),
            item.new_zeros((self.n_items, 1)),
        )

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        user, item = self.propagated_embeddings()
        id_score = (user[users, : self.id_dim] * item[items, : self.id_dim]).sum(1)
        auxiliary_score = (
            user[users, self.id_dim :] * item[items, self.id_dim :]
        ).sum(1)
        return {
            "id": id_score,
            "clv": auxiliary_score,
            "full": id_score + auxiliary_score,
        }

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
            "item_relation_projection_gradient_norm": norm(
                self.item_relation_projection.weight
            ),
            "relation_gate_gradient_norm": sum(
                norm(parameter)
                for parameter in (
                    self.relation_gate_bias,
                    self.relation_gate_level_raw,
                    self.relation_gate_composition,
                )
            ),
            "item_price_scale_gradient_norm": norm(self.item_price_scale_raw),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        keys = (
            "auxiliary_user_mean_norm",
            "auxiliary_item_mean_norm",
            "relation_gate_mean",
            "relation_gate_std",
            "relation_level_slope",
            "relation_composition_weight",
            "item_price_scale",
        )
        return {key: diagnostics[key] for key in keys}

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user_aux = self.auxiliary_user_embeddings()
        item_aux = self.auxiliary_item_embeddings()
        gate = self.relation_gate()
        valid_gate = gate[self.user_clv_valid > 0]
        price_user = self.q_c * (2.0 * self.user_overall_price - 1.0)
        rho0_max = 0.0 if self.rho == 0.0 else float("nan")
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "auxiliary_dim": self.auxiliary_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "continuous_relation_gate": True,
            "overall_price_fit_only": True,
            "within_category_price_input": False,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "external_reranking": False,
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "user_price_valid_share": float(self.user_price_valid.mean()),
            "item_price_valid_share": float(self.item_price_valid.mean()),
            "auxiliary_user_mean_norm": float(user_aux.norm(dim=1).mean()),
            "auxiliary_item_mean_norm": float(item_aux.norm(dim=1).mean()),
            "relation_item_mean_norm": float(item_aux[:, :2].norm(dim=1).mean()),
            "price_item_mean_abs": float(item_aux[:, 2].abs().mean()),
            "price_budget": self.price_budget,
            "relation_gate_mean": float(valid_gate.mean()) if len(valid_gate) else 0.0,
            "relation_gate_std": float(valid_gate.std(unbiased=False)) if len(valid_gate) else 0.0,
            "relation_gate_min": float(valid_gate.min()) if len(valid_gate) else 0.0,
            "relation_gate_max": float(valid_gate.max()) if len(valid_gate) else 0.0,
            "relation_gate_bias": float(self.relation_gate_bias),
            "relation_level_slope": float(F.softplus(self.relation_gate_level_raw)),
            "relation_composition_weight": float(self.relation_gate_composition),
            "item_relation_projection_norm": float(
                self.item_relation_projection.weight.norm()
            ),
            "item_price_scale": float(self.item_price_scale()),
            "user_overall_price_coordinate_mean_abs": float(
                (price_user * self.user_price_valid).abs().mean()
            ),
            "rho_zero_auxiliary_max_abs": rho0_max,
        }
