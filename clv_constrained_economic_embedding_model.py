"""One LightGCN with ID and one constrained CLV-economic layer-0 block."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstrainedCLVEconomicLightGCN(nn.Module):
    """Use CLV level as magnitude, N/V as direction, and price-only items."""

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
        if float(np.max(np.abs(item_features), initial=0.0)) > 1.0 + 1e-6:
            raise ValueError("상품 경제 입력은 [-1,1] 범위여야 합니다")
        if any(np.any(values[~user_valid] != 0.0) for values in (q_n, q_v, q_c)):
            raise ValueError("CLV 계산 불가 사용자의 q_n/q_v/q_c는 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.clv_dim = int(clv_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # ID parameters are initialised first, preserving the matched M1 start.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.user_clv_projection = nn.Linear(2, clv_dim, bias=False)
        self.item_economic_projection = nn.Linear(2, clv_dim, bias=False)
        nn.init.normal_(self.user_clv_projection.weight, std=0.05)
        nn.init.normal_(self.item_economic_projection.weight, std=0.02)

        self.register_buffer(
            "user_nv_composition",
            torch.from_numpy(np.column_stack([q_n, q_v]).astype(np.float32)),
            persistent=False,
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
        direction = F.normalize(
            self.user_clv_projection(self.user_nv_composition),
            p=2,
            dim=1,
            eps=1e-12,
        )
        return (
            self.user_clv_valid[:, None]
            * self.q_c[:, None]
            * direction
        )

    def clv_item_embeddings(self) -> torch.Tensor:
        direction = self.item_economic_projection(
            self.item_economic_features * self.item_economic_valid[:, None]
        )
        return F.normalize(direction, p=2, dim=1, eps=1e-12)

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
            "user_clv_projection_gradient_norm": norm(
                self.user_clv_projection.weight
            ),
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
                "item_economic_projection_norm",
                "user_clv_norm_qc_mae",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        user_clv = self.clv_user_embeddings()
        item_clv = self.clv_item_embeddings()
        expected_user_norm = self.q_c * self.user_clv_valid
        rho0_max = 0.0 if self.rho == 0.0 else float("nan")
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "clv_dim": self.clv_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "constrained_clv_economic_block": True,
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
            "user_clv_projection_norm": float(self.user_clv_projection.weight.norm()),
            "item_economic_projection_norm": float(
                self.item_economic_projection.weight.norm()
            ),
            "user_clv_norm_qc_mae": float(
                (user_clv.norm(dim=1) - expected_user_norm).abs().mean()
            ),
            "rho_zero_auxiliary_max_abs": rho0_max,
        }
