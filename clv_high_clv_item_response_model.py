"""LightGCN with a hard historical-CLV routed item-response subspace."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighCLVItemResponseLightGCN(nn.Module):
    """Jointly learn M1 and one small item-originated high-CLV subspace.

    The auxiliary user layer-0 representation is exactly zero.  A user's
    auxiliary representation can therefore only be formed by propagating the
    trainable item responses through that user's purchase history.  The fixed
    historical-CLV gate is applied to user nodes after every auxiliary hop, so
    low/mid-CLV users receive exactly no auxiliary score.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        high_clv_gate: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        response_dim: int = 8,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim, response_dim) <= 0:
            raise ValueError("사용자·상품·임베딩 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 1 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        gate = np.asarray(high_clv_gate, dtype=np.float32)
        if gate.shape != (n_users,):
            raise ValueError("high_clv_gate shape이 n_users와 다릅니다")
        if not np.isfinite(gate).all() or not np.isin(gate, (0.0, 1.0)).all():
            raise ValueError("high_clv_gate는 유한한 0/1 값이어야 합니다")
        if not 0 < int(gate.sum()) < n_users:
            raise ValueError("고CLV와 비고CLV 사용자가 모두 있어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.response_dim = int(response_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # M1 parameters are initialised first so every arm starts from exactly
        # the same ID embeddings.  There is deliberately no auxiliary user ID.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.item_response = nn.Embedding(n_items, response_dim)
        nn.init.normal_(self.item_response.weight, std=0.1)

        self.register_buffer(
            "high_clv_gate", torch.from_numpy(gate.copy()), persistent=False
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.response_dim

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

    def response_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Row normalisation removes response-table norm as an uncontrolled
        # strength parameter.  The score magnitude is then bounded by rho.
        item = F.normalize(self.item_response.weight, dim=1)
        user = item.new_zeros((self.n_users, self.response_dim))
        current = torch.cat([user, item], dim=0)
        total = current
        gate = self.high_clv_gate[:, None]
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            current_user = current[: self.n_users] * gate
            current_item = current[self.n_users :]
            current = torch.cat([current_user, current_item], dim=0)
            total = total + current
        total = total / (self.n_layers + 1)
        user = F.normalize(total[: self.n_users], dim=1) * gate
        item = F.normalize(total[self.n_users :], dim=1)
        return user, item

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        id_user, id_item = self.id_embeddings()
        response_user, response_item = self.response_embeddings()
        scale = math.sqrt(self.rho)
        return (
            torch.cat([id_user, scale * response_user], dim=1),
            torch.cat([id_item, scale * response_item], dim=1),
        )

    def embeddings(self, need_value: bool = True):
        # need_value is retained for the shared evaluator interface.
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        id_user, id_item = self.id_embeddings()
        response_user, response_item = self.response_embeddings()
        id_score = (id_user[users] * id_item[items]).sum(1)
        response_score = self.rho * (
            response_user[users] * response_item[items]
        ).sum(1)
        return {
            "id": id_score,
            "response": response_score,
            "full": id_score + response_score,
        }

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        # Keep the same sampled ID regularisation as M1.  The response rows are
        # unit-normalised in the forward graph and do not carry a free norm.
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        # gate is retained only for the common trainer signature.
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
            "item_response_gradient_norm": norm(self.item_response.weight),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict[str, float]:
        diagnostics = self.representation_diagnostics()
        return {
            key: diagnostics[key]
            for key in (
                "high_clv_gate_share",
                "high_clv_response_user_mean_norm",
                "response_item_mean_norm",
                "non_high_response_max_abs",
            )
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        response_user, response_item = self.response_embeddings()
        high = self.high_clv_gate.bool()
        non_high = ~high
        user_norm = response_user.norm(dim=1)
        item_norm = response_item.norm(dim=1)
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "response_dim": self.response_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "historical_clv_role": "fixed high-segment routing",
            "free_auxiliary_user_embedding": False,
            "free_item_response_embedding": True,
            "joint_graph_propagation": True,
            "user_gate_applied_each_auxiliary_hop": True,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "explicit_item_price": False,
            "external_reranking": False,
            "high_clv_user_count": int(high.sum()),
            "high_clv_gate_share": float(self.high_clv_gate.mean()),
            "high_clv_response_user_mean_norm": float(user_norm[high].mean()),
            "response_item_mean_norm": float(item_norm.mean()),
            "item_response_raw_mean_norm": float(
                self.item_response.weight.norm(dim=1).mean()
            ),
            "non_high_response_max_abs": float(
                response_user[non_high].abs().max()
            ),
            "auxiliary_score_abs_bound": self.rho,
            "rho_zero_auxiliary_max_abs": 0.0 if self.rho == 0.0 else float("nan"),
        }
