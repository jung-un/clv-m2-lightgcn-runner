"""CLV-conditioned low-rank transformation of the ordinary user ID embedding.

This M2 changes *how an existing collaborative user preference is represented*.
It does not append an independent N/V item-scoring space.  The item layer-0
embedding, graph, negative sampler, and BPR objective remain those of M1.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _LowRankTransform(nn.Module):
    """Bias-free rank-r linear transform, initialised as the zero map."""

    def __init__(self, dimension: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


class CLVConditionalIDTransformLightGCN(nn.Module):
    """One 64-d LightGCN whose user ID representation is N/V-conditioned.

    The train-history conditions are fixed observations.  The two low-rank
    transforms and the ordinary ID embeddings are jointly learned by one plain
    BPR loss and one optimiser.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_n: np.ndarray,
        q_v: np.ndarray,
        user_activity_valid: np.ndarray,
        user_value_valid: np.ndarray,
        adj: torch.Tensor,
        embedding_dim: int = 64,
        transform_rank: int = 4,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or embedding_dim <= 0:
            raise ValueError("사용자·아이템·임베딩 차원은 양수여야 합니다")
        if not 0 < transform_rank <= embedding_dim:
            raise ValueError("변환 rank는 1 이상이고 임베딩 차원 이하여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        activity_valid = np.asarray(user_activity_valid, dtype=bool)
        value_valid = np.asarray(user_value_valid, dtype=bool)
        expected = (n_users,)
        if any(
            values.shape != expected
            for values in (q_n, q_v, activity_valid, value_valid)
        ):
            raise ValueError("N/V 수준·유효성 shape이 n_users와 다릅니다")
        if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
            raise ValueError("N/V 수준은 모두 유한해야 합니다")
        if ((q_n < 0) | (q_n > 1)).any() or ((q_v < 0) | (q_v > 1)).any():
            raise ValueError("N/V 백분위는 [0, 1] 범위여야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.transform_rank = int(transform_rank)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # Keep M1's parameter initialisation order so rho=0 has the same ID
        # starting point under the same random seed.
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.activity_transform = _LowRankTransform(embedding_dim, transform_rank)
        self.value_transform = _LowRankTransform(embedding_dim, transform_rank)

        # Invalid users receive no axis intervention.  Mid-rank percentiles
        # make the valid-user condition approximately mean zero, preventing a
        # global shift shared by all users.
        condition_n = (2.0 * q_n - 1.0) * activity_valid.astype(np.float32)
        condition_v = (2.0 * q_v - 1.0) * value_valid.astype(np.float32)
        self.register_buffer("condition_n", torch.from_numpy(condition_n))
        self.register_buffer("condition_v", torch.from_numpy(condition_v))
        self.register_buffer(
            "user_activity_valid",
            torch.from_numpy(activity_valid.astype(np.float32)),
        )
        self.register_buffer(
            "user_value_valid", torch.from_numpy(value_valid.astype(np.float32))
        )
        self.register_buffer("adj", adj.coalesce())

    def _conditioned_user_layer0(self) -> torch.Tensor:
        base = self.E_u.weight
        correction = (
            self.condition_n[:, None] * self.activity_transform(base)
            + self.condition_v[:, None] * self.value_transform(base)
        )
        raw = base + self.rho * correction
        base_norm = base.norm(dim=1, keepdim=True)
        raw_norm = raw.norm(dim=1, keepdim=True).clamp_min(1e-12)
        # Preserve each user's collaborative-preference magnitude.  CLV may
        # rotate/rebalance the representation, but cannot win by norm inflation.
        return raw * (base_norm / raw_norm)

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._conditioned_user_layer0(), self.E_i.weight

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 표현 실험에 M4 표본 가중치를 넣을 수 없습니다")
        if float(lam) != 0.0:
            raise ValueError("외부 점수 가산은 이 M2의 forward graph가 아닙니다")
        user, item = self.propagate()
        selected_user = user[users]
        positive_score = (selected_user * item[positives]).sum(1)
        negative_score = (selected_user * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "objective": "plain_bpr",
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def representation_diagnostics(self) -> dict:
        user0, item0 = self.layer0_embeddings()
        base = self.E_u.weight
        activity = self.condition_n[:, None] * self.activity_transform(base)
        value = self.condition_v[:, None] * self.value_transform(base)

        def condition_summary(condition: torch.Tensor, valid: torch.Tensor) -> dict:
            selected = condition[valid.bool()]
            return {
                "mean": float(selected.mean()) if selected.numel() else 0.0,
                "std": float(selected.std(unbiased=False)) if selected.numel() else 0.0,
                "negative_share": (
                    float((selected < 0).float().mean()) if selected.numel() else 0.0
                ),
                "positive_share": (
                    float((selected > 0).float().mean()) if selected.numel() else 0.0
                ),
            }

        return {
            "rho": self.rho,
            "transform_rank": self.transform_rank,
            "total_dim": self.embedding_dim,
            "explicit_item_features": False,
            "item_transformation": False,
            "mean_user_norm": float(user0.norm(dim=1).mean()),
            "mean_item_norm": float(item0.norm(dim=1).mean()),
            "max_user_norm_change": float(
                (user0.norm(dim=1) - base.norm(dim=1)).abs().max()
            ),
            "mean_user_representation_change": float((user0 - base).norm(dim=1).mean()),
            "activity_correction_mean_norm": float(activity.norm(dim=1).mean()),
            "value_correction_mean_norm": float(value.norm(dim=1).mean()),
            "activity_transform_norm": float(
                self.activity_transform.up.weight.norm()
                * self.activity_transform.down.weight.norm()
            ),
            "value_transform_norm": float(
                self.value_transform.up.weight.norm()
                * self.value_transform.down.weight.norm()
            ),
            **{
                f"activity_condition_{key}": value
                for key, value in condition_summary(
                    self.condition_n, self.user_activity_valid
                ).items()
            },
            **{
                f"value_condition_{key}": value
                for key, value in condition_summary(
                    self.condition_v, self.user_value_valid
                ).items()
            },
        }
