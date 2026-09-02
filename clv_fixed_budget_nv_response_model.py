"""LightGCN with one fixed-budget N axis and one fixed-budget V-price axis."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedBudgetNVResponseLightGCN(nn.Module):
    """Jointly propagate ID preference and two explicit historical-CLV axes."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        item_overall_price: np.ndarray,
        item_price_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        price_scale_initial: float = 0.9,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")
        if not 0.0 < price_scale_initial < 1.0:
            raise ValueError("price_scale_initial은 0과 1 사이여야 합니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_price = np.asarray(item_overall_price, dtype=np.float32)
        item_valid = np.asarray(item_price_valid, dtype=bool)
        if any(array.shape != (n_users,) for array in (q_n, q_v, q_c, user_valid)):
            raise ValueError("사용자 CLV 입력 shape이 n_users와 다릅니다")
        if item_price.shape != (n_items,) or item_valid.shape != (n_items,):
            raise ValueError("상품 가격 입력 shape이 n_items와 다릅니다")
        if not all(
            np.isfinite(array).all() for array in (q_n, q_v, q_c, item_price)
        ):
            raise ValueError("CLV·상품 가격 입력은 모두 유한해야 합니다")
        for name, values in (
            ("q_n", q_n),
            ("q_v", q_v),
            ("q_c", q_c),
            ("item_overall_price", item_price),
        ):
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name}은 [0,1] 범위여야 합니다")
        if any(np.any(values[~user_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.clv_dim = 2
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # Initialise M1 parameters first so rho=0 is a matched LightGCN control.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        # N asks how each candidate responds to the repeat-transaction component.
        # It reuses the item ID representation instead of adding a free item table.
        self.item_n_projection = nn.Linear(id_dim, 1, bias=False)
        nn.init.normal_(self.item_n_projection.weight, std=0.02)

        # V keeps the price direction positive and bounded while allowing one
        # shared magnitude to be learned by the same recommendation objective.
        initial_logit = math.log(price_scale_initial / (1.0 - price_scale_initial))
        self.price_scale_raw = nn.Parameter(torch.tensor(initial_logit))

        self.register_buffer("q_n", torch.from_numpy(q_n.copy()), persistent=False)
        self.register_buffer("q_v", torch.from_numpy(q_v.copy()), persistent=False)
        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_overall_price",
            torch.from_numpy(item_price.copy()),
            persistent=False,
        )
        self.register_buffer(
            "item_price_valid",
            torch.from_numpy(item_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.clv_dim

    def clv_user_budget(self) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = self.q_n + self.q_v
        pi_n = torch.where(
            denominator > 0,
            self.q_n / denominator.clamp_min(1e-12),
            torch.full_like(denominator, 0.5),
        )
        b_n = self.user_clv_valid * self.q_c * pi_n
        b_v = self.user_clv_valid * self.q_c * (1.0 - pi_n)
        return b_n, b_v

    def item_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_response = torch.tanh(self.item_n_projection(self.E_i.weight)).squeeze(1)
        centred_price = 2.0 * self.item_overall_price - 1.0
        v_response = (
            torch.sigmoid(self.price_scale_raw)
            * centred_price
            * self.item_price_valid
        )
        return n_response, v_response

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        b_n, b_v = self.clv_user_budget()
        r_n, r_v = self.item_responses()
        scale = math.sqrt(self.rho)
        return (
            torch.cat(
                [self.E_u.weight, scale * torch.stack([b_n, b_v], dim=1)],
                dim=1,
            ),
            torch.cat(
                [self.E_i.weight, scale * torch.stack([r_n, r_v], dim=1)],
                dim=1,
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

    def component_embeddings(
        self, component: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if component not in {"n", "v"}:
            raise ValueError("component는 n 또는 v여야 합니다")
        user, item = self.layer0_embeddings()
        other = self.id_dim + (1 if component == "n" else 0)
        user = user.clone()
        item = item.clone()
        user[:, other] = 0.0
        item[:, other] = 0.0
        return self._propagate(user, item)

    def embeddings(self, need_value: bool = True):
        # need_value is retained for the shared evaluator interface.
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
        n_score = selected_user[:, self.id_dim] * selected_item[:, self.id_dim]
        v_score = (
            selected_user[:, self.id_dim + 1]
            * selected_item[:, self.id_dim + 1]
        )
        return {
            "id": id_score,
            "n": n_score,
            "v": v_score,
            "clv": n_score + v_score,
            "full": id_score + n_score + v_score,
        }

    def layer0_auxiliary_scores(self) -> torch.Tensor:
        b_n, b_v = self.clv_user_budget()
        r_n, r_v = self.item_responses()
        return self.rho * (
            b_n[:, None] * r_n[None, :] + b_v[:, None] * r_v[None, :]
        )

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        # need_value is retained for the shared trainer interface.
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        # gate is retained for the shared trainer interface.
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
            "item_n_projection_gradient_norm": norm(
                self.item_n_projection.weight
            ),
            "price_scale_gradient_norm": norm(self.price_scale_raw),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        return {
            key: diagnostics[key]
            for key in (
                "budget_sum_max_error",
                "b_n_mean",
                "b_v_mean",
                "item_n_response_std",
                "item_v_response_std",
                "price_scale",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        b_n, b_v = self.clv_user_budget()
        r_n, r_v = self.item_responses()
        expected_budget = self.q_c * self.user_clv_valid
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "clv_dim": self.clv_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "fixed_clv_budget": True,
            "clv_budget_identity": "b_N+b_V=q_C",
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "learned_user_projection": False,
            "free_item_response_embedding": False,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "explicit_item_price": True,
            "external_reranking": False,
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "item_price_valid_share": float(self.item_price_valid.mean()),
            "budget_sum_max_error": float(
                ((b_n + b_v) - expected_budget).abs().max()
            ),
            "b_n_mean": float(b_n.mean()),
            "b_n_std": float(b_n.std(unbiased=False)),
            "b_v_mean": float(b_v.mean()),
            "b_v_std": float(b_v.std(unbiased=False)),
            "item_n_response_mean_abs": float(r_n.abs().mean()),
            "item_n_response_std": float(r_n.std(unbiased=False)),
            "item_v_response_mean_abs": float(r_v.abs().mean()),
            "item_v_response_std": float(r_v.std(unbiased=False)),
            "item_n_projection_norm": float(self.item_n_projection.weight.norm()),
            "price_scale": float(torch.sigmoid(self.price_scale_raw)),
            "layer0_auxiliary_score_abs_bound": self.rho,
            "rho_zero_auxiliary_max_abs": 0.0 if self.rho == 0.0 else float("nan"),
        }
