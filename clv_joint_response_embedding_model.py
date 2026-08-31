"""One LightGCN with an ID block and one joint CLV-response block.

The CLV block enters at layer 0 and is propagated together with the ordinary
ID block.  Its user side keeps overall historical-CLV level and N/V
composition separate; its item side combines a small learned response with
overall and within-category price positions.  No RepeatShare, item degree,
gate, learned global axis weight, external reranking, or extra loss is used.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _center_and_scale(
    values: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    """Centre train-user values and bound valid entries to [-1, 1]."""

    output = np.zeros_like(values, dtype=np.float32)
    if not valid.any():
        return output
    selected = values[valid].astype(np.float64, copy=False)
    centered = selected - selected.mean(axis=0, keepdims=True)
    scale = np.maximum(np.abs(centered).max(axis=0, keepdims=True), 1e-12)
    output[valid] = (centered / scale).astype(np.float32)
    return output


class JointCLVResponseLightGCN(nn.Module):
    """ID(64) plus one jointly propagated CLV-response block(4)."""

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
        clv_dim: int = 4,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, clv_dim) <= 0:
            raise ValueError("사용자·상품·표현 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
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
        if np.abs(item_features).max(initial=0.0) > 1.0 + 1e-6:
            raise ValueError("상품 경제 입력은 [-1,1] 범위여야 합니다")
        if any(np.any(values[~user_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")

        context_raw = np.column_stack([q_c, q_n - q_v]).astype(np.float32)
        context = _center_and_scale(context_raw, user_valid)

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.clv_dim = int(clv_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # Keep M1's ID initialisation first so matched rho=0 starts identically.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.user_clv_projection = nn.Linear(2, clv_dim, bias=False)
        self.item_response = nn.Embedding(n_items, clv_dim)
        self.item_economic_projection = nn.Linear(2, clv_dim, bias=False)
        nn.init.normal_(self.user_clv_projection.weight, std=0.05)
        nn.init.normal_(self.item_response.weight, std=0.02)
        nn.init.normal_(self.item_economic_projection.weight, std=0.02)

        self.register_buffer(
            "clv_context", torch.from_numpy(context), persistent=False
        )
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
        return self.user_clv_valid[:, None] * torch.tanh(
            self.user_clv_projection(self.clv_context)
        )

    def clv_item_embeddings(self) -> torch.Tensor:
        economic = self.item_economic_projection(
            self.item_economic_features * self.item_economic_valid[:, None]
        )
        return F.normalize(
            self.item_response.weight + economic,
            p=2,
            dim=1,
            eps=1e-12,
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        scale = math.sqrt(self.rho)
        user_clv = scale * self.clv_user_embeddings()
        item_clv = scale * self.clv_item_embeddings()
        return (
            torch.cat([self.E_u.weight, user_clv], dim=1),
            torch.cat([self.E_i.weight, item_clv], dim=1),
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
        self, *, include_clv: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if include_clv:
            return self.propagated_embeddings()
        return self.id_embeddings()

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
        return {
            "id": id_score,
            "clv": clv_score,
            "full": id_score + clv_score,
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
            "user_clv_projection_gradient_norm": norm(
                self.user_clv_projection.weight
            ),
            "item_response_gradient_norm": norm(self.item_response.weight),
            "item_economic_projection_gradient_norm": norm(
                self.item_economic_projection.weight
            ),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        return {
            key: diagnostics[key]
            for key in (
                "clv_user_mean_norm",
                "clv_item_mean_norm",
                "user_clv_projection_norm",
                "item_response_mean_norm",
                "item_economic_projection_norm",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user_clv = self.clv_user_embeddings()
        item_clv = self.clv_item_embeddings()
        economic = self.item_economic_projection(
            self.item_economic_features * self.item_economic_valid[:, None]
        )
        valid_users = self.user_clv_valid.bool()
        rho0_max = 0.0
        if self.rho != 0.0:
            rho0_max = float("nan")
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "clv_dim": self.clv_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "joint_clv_response_block": True,
            "layer0_intervention": True,
            "joint_graph_propagation": True,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "explicit_item_price": True,
            "external_reranking": False,
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "item_economic_valid_share": float(self.item_economic_valid.mean()),
            "clv_level_context_mean_abs": float(
                self.clv_context[valid_users, 0].mean().abs()
            ),
            "clv_composition_context_mean_abs": float(
                self.clv_context[valid_users, 1].mean().abs()
            ),
            "clv_level_context_std": float(
                self.clv_context[valid_users, 0].std(unbiased=False)
            ),
            "clv_composition_context_std": float(
                self.clv_context[valid_users, 1].std(unbiased=False)
            ),
            "clv_user_mean_norm": float(user_clv.norm(dim=1).mean()),
            "clv_item_mean_norm": float(item_clv.norm(dim=1).mean()),
            "user_clv_projection_norm": float(self.user_clv_projection.weight.norm()),
            "item_response_mean_norm": float(
                self.item_response.weight.norm(dim=1).mean()
            ),
            "item_economic_projection_norm": float(
                self.item_economic_projection.weight.norm()
            ),
            "item_economic_contribution_mean_norm": float(
                economic.norm(dim=1).mean()
            ),
            "rho_zero_auxiliary_max_abs": rho0_max,
        }
