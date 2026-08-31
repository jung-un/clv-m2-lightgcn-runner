"""One jointly trained LightGCN with bounded CLV/economic side coordinates.

The ordinary ID block is propagated exactly as in M1.  A small auxiliary block
reads detached final ID representations so its gradient cannot directly rewrite
the ID tables.  Both blocks still enter one embedding, one dot product, and one
plain BPR objective from the first epoch.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientIsolatedCLVEconomicInteractionLightGCN(nn.Module):
    """ID(64), CLV-conditioned relation(3), and explicit price relation(1)."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        q_c: np.ndarray,
        user_clv_valid: np.ndarray,
        item_price_percentile: np.ndarray,
        item_price_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        relation_dim: int = 3,
        rho: float = 0.05,
        beta: float = 0.25,
        delta: float = 0.25,
        eta: float = 0.5,
        price_epsilon: float = 0.5,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, relation_dim) <= 0:
            raise ValueError("사용자·상품·표현 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta는 0 이상 1 이하여야 합니다")
        if not 0.0 <= delta < 1.0:
            raise ValueError("delta는 0 이상 1 미만이어야 합니다")
        if not 0.0 <= eta <= 1.0:
            raise ValueError("eta는 0 이상 1 이하여야 합니다")
        if not 0.0 < price_epsilon < 1.0:
            raise ValueError("price_epsilon은 0과 1 사이여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        q_c = np.asarray(q_c, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_price = np.asarray(item_price_percentile, dtype=np.float32)
        item_valid = np.asarray(item_price_valid, dtype=bool)
        if any(values.shape != (n_users,) for values in (q_n, q_v, q_c, user_valid)):
            raise ValueError("사용자 CLV 입력 shape이 n_users와 다릅니다")
        if any(values.shape != (n_items,) for values in (item_price, item_valid)):
            raise ValueError("상품 가격 입력 shape이 n_items와 다릅니다")
        if not all(np.isfinite(values).all() for values in (q_n, q_v, q_c, item_price)):
            raise ValueError("CLV·가격 입력은 모두 유한해야 합니다")
        for name, values in (("q_n", q_n), ("q_v", q_v), ("q_c", q_c)):
            if ((values < 0.0) | (values > 1.0)).any():
                raise ValueError(f"{name}은 [0,1] 범위여야 합니다")
        if ((item_price < 0.0) | (item_price > 1.0)).any():
            raise ValueError("상품 가격 백분위는 [0,1] 범위여야 합니다")
        if any(np.any(values[~user_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.relation_dim = int(relation_dim)
        self.rho = float(rho)
        self.beta = float(beta)
        self.delta = float(delta)
        self.eta = float(eta)
        self.price_epsilon = float(price_epsilon)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # ID initialisation happens first so rho=0 and active arms share it exactly.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.user_projection = nn.Linear(id_dim, relation_dim, bias=False)
        self.item_projection = nn.Linear(id_dim, relation_dim, bias=False)
        self.condition_mixer = nn.Linear(3, relation_dim, bias=False)
        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.xavier_uniform_(self.item_projection.weight)
        nn.init.normal_(self.condition_mixer.weight, std=0.05)
        # kappa starts near its structural maximum but remains trainable and positive.
        self.raw_price_calibration = nn.Parameter(torch.tensor(3.0))

        self.register_buffer("q_n", torch.from_numpy(q_n.copy()), persistent=False)
        self.register_buffer("q_v", torch.from_numpy(q_v.copy()), persistent=False)
        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_price_percentile",
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
        return self.id_dim + self.relation_dim + 1

    def price_calibration(self) -> torch.Tensor:
        return (
            1.0 + self.price_epsilon * torch.tanh(self.raw_price_calibration)
        ) / (1.0 + self.price_epsilon)

    def id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def auxiliary_embeddings(
        self,
        id_user: torch.Tensor | None = None,
        id_item: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if id_user is None or id_item is None:
            id_user, id_item = self.id_embeddings()
        base_user = F.normalize(
            self.user_projection(id_user.detach()), p=2, dim=1, eps=1e-12
        )
        base_item = F.normalize(
            self.item_projection(id_item.detach()), p=2, dim=1, eps=1e-12
        )
        conditions = torch.stack([self.q_n, self.q_v, self.q_c], dim=1)
        modulation = 1.0 + self.delta * torch.tanh(self.condition_mixer(conditions))
        relation_user = self.user_clv_valid[:, None] * F.normalize(
            base_user * modulation, p=2, dim=1, eps=1e-12
        )
        relation_item = base_item
        clv_strength = self.user_clv_valid * (
            1.0 + self.eta * (self.q_c - 0.5)
        )
        price_user = (
            clv_strength
            * self.price_calibration()
            * (2.0 * self.q_v - 1.0)
        )
        price_item = (
            self.item_price_valid * (2.0 * self.item_price_percentile - 1.0)
        )
        return relation_user, relation_item, price_user, price_item

    def component_embeddings(
        self, *, include_relation: bool = True, include_price: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        id_user, id_item = self.id_embeddings()
        if self.rho == 0.0 or (not include_relation and not include_price):
            return id_user, id_item
        relation_user, relation_item, price_user, price_item = (
            self.auxiliary_embeddings(id_user, id_item)
        )
        clv_strength = self.user_clv_valid * (
            1.0 + self.eta * (self.q_c - 0.5)
        )
        user_blocks = [id_user]
        item_blocks = [id_item]
        if include_relation:
            scale = math.sqrt(self.rho * (1.0 - self.beta))
            user_blocks.append(scale * clv_strength[:, None] * relation_user)
            item_blocks.append(scale * relation_item)
        if include_price:
            scale = math.sqrt(self.rho * self.beta)
            user_blocks.append(scale * price_user[:, None])
            item_blocks.append(scale * price_item[:, None])
        return torch.cat(user_blocks, dim=1), torch.cat(item_blocks, dim=1)

    def embeddings(self, need_value: bool = True):
        # need_value is retained only for the shared evaluator interface.
        user, item = self.component_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        id_user, id_item = self.id_embeddings()
        relation_user, relation_item, price_user, price_item = (
            self.auxiliary_embeddings(id_user, id_item)
        )
        clv_strength = self.user_clv_valid * (
            1.0 + self.eta * (self.q_c - 0.5)
        )
        id_score = (id_user[users] * id_item[items]).sum(1)
        relation = (
            clv_strength[users]
            * (relation_user[users] * relation_item[items]).sum(1)
        )
        price = price_user[users] * price_item[items]
        weighted_relation = self.rho * (1.0 - self.beta) * relation
        weighted_price = self.rho * self.beta * price
        return {
            "id": id_score,
            "relation": relation,
            "price": price,
            "weighted_relation": weighted_relation,
            "weighted_price": weighted_price,
            "auxiliary": weighted_relation + weighted_price,
            "full": id_score + weighted_relation + weighted_price,
        }

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled_id_l2 = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled_id_l2

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if float(lam) != 0.0:
            raise ValueError("완성된 점수에 외부 보정을 더할 수 없습니다")
        user, item, _, _ = self.embeddings()
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
            "user_projection_gradient_norm": norm(self.user_projection.weight),
            "item_projection_gradient_norm": norm(self.item_projection.weight),
            "condition_mixer_gradient_norm": norm(self.condition_mixer.weight),
            "price_calibration_gradient_norm": norm(self.raw_price_calibration),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        return {
            key: diagnostics[key]
            for key in (
                "relation_user_mean_norm",
                "relation_item_mean_norm",
                "price_calibration",
                "condition_mixer_weight_norm",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        id_user, id_item = self.id_embeddings()
        relation_user, relation_item, price_user, price_item = (
            self.auxiliary_embeddings(id_user, id_item)
        )
        valid_users = self.user_clv_valid.bool()
        valid_items = self.item_price_valid.bool()
        rho0_max = 0.0
        if self.rho != 0.0:
            rho0_max = float("nan")
        return {
            "rho": self.rho,
            "beta": self.beta,
            "delta": self.delta,
            "eta": self.eta,
            "price_epsilon": self.price_epsilon,
            "id_dim": self.id_dim,
            "relation_dim": self.relation_dim,
            "price_dim": 1,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "explicit_item_price": True,
            "gradient_isolated_auxiliary": True,
            "external_reranking": False,
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "item_price_valid_share": float(self.item_price_valid.mean()),
            "q_n_std": float(self.q_n[valid_users].std(unbiased=False)),
            "q_v_std": float(self.q_v[valid_users].std(unbiased=False)),
            "q_c_std": float(self.q_c[valid_users].std(unbiased=False)),
            "relation_user_mean_norm": float(relation_user.norm(dim=1).mean()),
            "relation_item_mean_norm": float(relation_item.norm(dim=1).mean()),
            "price_user_mean_abs": float(price_user.abs().mean()),
            "price_item_mean_abs": float(price_item.abs().mean()),
            "price_item_min": float(price_item[valid_items].min()),
            "price_item_max": float(price_item[valid_items].max()),
            "price_calibration": float(self.price_calibration()),
            "condition_mixer_weight_norm": float(self.condition_mixer.weight.norm()),
            "mean_id_user_norm": float(id_user.norm(dim=1).mean()),
            "mean_id_item_norm": float(id_item.norm(dim=1).mean()),
            "rho_zero_auxiliary_max_abs": rho0_max,
        }
