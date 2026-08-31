"""One LightGCN with a bounded CLV-conditioned user-item interaction.

The ordinary 64-dimensional user/item ID representation is propagated for two
LightGCN layers and retained without modification.  Fixed train-history overall
CLV level and N/V composition condition one small, jointly learned interaction
space.  The interaction coordinates are concatenated before the single dot
product used by the same plain BPR objective; they are not an external reranker.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLVConditionedUserItemInteractionLightGCN(nn.Module):
    """ID(64) plus one bounded context-conditioned interaction block."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        q_c: np.ndarray,
        d_nv: np.ndarray,
        user_clv_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        context_dim: int = 4,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if n_users <= 0 or n_items <= 0 or id_dim <= 0 or context_dim <= 0:
            raise ValueError("사용자·아이템·표현 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 0 이상 1 이하여야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers와 pref_reg 설정이 잘못됐습니다")

        q_c = np.asarray(q_c, dtype=np.float32)
        d_nv = np.asarray(d_nv, dtype=np.float32)
        valid = np.asarray(user_clv_valid, dtype=bool)
        expected = (n_users,)
        if any(values.shape != expected for values in (q_c, d_nv, valid)):
            raise ValueError("CLV 수준·구성·유효성 shape이 n_users와 다릅니다")
        if not np.isfinite(q_c).all() or not np.isfinite(d_nv).all():
            raise ValueError("CLV 수준과 N/V 구성은 모두 유한해야 합니다")
        if ((q_c < 0.0) | (q_c > 1.0)).any():
            raise ValueError("전체 CLV 백분위는 [0, 1] 범위여야 합니다")
        if ((d_nv < -1.0) | (d_nv > 1.0)).any():
            raise ValueError("N/V 구성 차이는 [-1, 1] 범위여야 합니다")
        if np.any(q_c[~valid] != 0.0) or np.any(d_nv[~valid] != 0.0):
            raise ValueError("CLV 계산 불가 사용자의 q_c와 d_nv는 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.context_dim = int(context_dim)
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        # Initialise the ID tables before auxiliary parameters so the matched
        # rho=0 arm starts from the same ID state as the active arm.
        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.user_projection = nn.Linear(id_dim, context_dim, bias=False)
        self.item_projection = nn.Linear(id_dim, context_dim, bias=False)
        nn.init.xavier_uniform_(self.user_projection.weight)
        nn.init.xavier_uniform_(self.item_projection.weight)
        self.overall_clv_context = nn.Parameter(torch.empty(context_dim))
        self.nv_composition_context = nn.Parameter(torch.empty(context_dim))
        nn.init.normal_(self.overall_clv_context, std=1.0 / math.sqrt(context_dim))
        nn.init.normal_(
            self.nv_composition_context, std=1.0 / math.sqrt(context_dim)
        )

        # All buffers can be reconstructed from the train split.  Excluding
        # them from state_dict keeps checkpoints small, especially for H&M.
        self.register_buffer("q_c", torch.from_numpy(q_c.copy()), persistent=False)
        self.register_buffer("d_nv", torch.from_numpy(d_nv.copy()), persistent=False)
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.context_dim

    def id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordinary LightGCN 0/1/2-layer mean."""

        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def interaction_embeddings(
        self,
        id_user: torch.Tensor | None = None,
        id_item: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return user interaction, item interaction, and CLV context vectors."""

        if id_user is None or id_item is None:
            id_user, id_item = self.id_embeddings()
        user_direction = F.normalize(
            self.user_projection(id_user), p=2, dim=1, eps=1e-12
        )
        item_direction = F.normalize(
            self.item_projection(id_item), p=2, dim=1, eps=1e-12
        )
        raw_context = (
            self.overall_clv_context[None, :]
            + self.d_nv[:, None] * self.nv_composition_context[None, :]
        )
        context = F.normalize(raw_context, p=2, dim=1, eps=1e-12)
        user_interaction = (
            self.user_clv_valid[:, None]
            * self.q_c[:, None]
            * user_direction
            * context
        )
        return user_interaction, item_direction, context

    def embeddings(self, need_value: bool = True):
        # need_value is retained only for the shared evaluator interface.
        id_user, id_item = self.id_embeddings()
        if self.rho == 0.0:
            # Returning the original width avoids even the reduction-order
            # round-off caused by appending four exact-zero coordinates.
            zero_user = id_user.new_zeros((self.n_users, 1))
            zero_item = id_item.new_zeros((self.n_items, 1))
            return id_user, id_item, zero_user, zero_item
        user_interaction, item_interaction, _ = self.interaction_embeddings(
            id_user, id_item
        )
        scale = math.sqrt(self.rho)
        user = torch.cat([id_user, scale * user_interaction], dim=1)
        item = torch.cat([id_item, scale * item_interaction], dim=1)
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ID score, raw interaction R, and rho-weighted R for pairs."""

        id_user, id_item = self.id_embeddings()
        user_interaction, item_interaction, _ = self.interaction_embeddings(
            id_user, id_item
        )
        id_score = (id_user[users] * id_item[items]).sum(1)
        interaction = (
            user_interaction[users] * item_interaction[items]
        ).sum(1)
        return id_score, interaction, self.rho * interaction

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
        # gate/lam are retained only for the common trainer signature.
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
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict:
        def gradient_norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "user_projection_gradient_norm": gradient_norm(
                self.user_projection.weight
            ),
            "item_projection_gradient_norm": gradient_norm(
                self.item_projection.weight
            ),
            "overall_context_gradient_norm": gradient_norm(
                self.overall_clv_context
            ),
            "composition_context_gradient_norm": gradient_norm(
                self.nv_composition_context
            ),
        }

    @torch.no_grad()
    def epoch_training_diagnostics(self) -> dict:
        diagnostics = self.representation_diagnostics()
        keys = (
            "interaction_user_mean_norm",
            "interaction_item_mean_norm",
            "population_mean_interaction_ratio",
            "overall_composition_context_cosine",
        )
        return {key: diagnostics[key] for key in keys}

    @torch.no_grad()
    def representation_diagnostics(self) -> dict:
        id_user, id_item = self.id_embeddings()
        user_interaction, item_interaction, context = self.interaction_embeddings(
            id_user, id_item
        )
        user_norm = user_interaction.norm(dim=1)
        valid = self.user_clv_valid.bool()
        selected_q = self.q_c[valid]
        selected_d = self.d_nv[valid]
        mean_user_norm = user_norm.mean().clamp_min(1e-12)
        context_cosine = F.cosine_similarity(
            self.overall_clv_context[None, :],
            self.nv_composition_context[None, :],
        )
        weighted_auxiliary_max = float(
            math.sqrt(self.rho) * user_interaction.abs().max()
        )
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "context_dim": self.context_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "historical_clv_input": True,
            "separate_n_v_scores": False,
            "explicit_item_features": False,
            "external_reranking": False,
            "q_c_mean": float(selected_q.mean()) if selected_q.numel() else 0.0,
            "q_c_std": (
                float(selected_q.std(unbiased=False)) if selected_q.numel() else 0.0
            ),
            "d_nv_mean": float(selected_d.mean()) if selected_d.numel() else 0.0,
            "d_nv_std": (
                float(selected_d.std(unbiased=False)) if selected_d.numel() else 0.0
            ),
            "clv_valid_share": float(self.user_clv_valid.mean()),
            "interaction_user_mean_norm": float(user_norm.mean()),
            "interaction_user_max_norm": float(user_norm.max()),
            "interaction_item_mean_norm": float(item_interaction.norm(dim=1).mean()),
            "context_mean_norm": float(context.norm(dim=1).mean()),
            "population_mean_interaction_ratio": float(
                user_interaction.mean(dim=0).norm() / mean_user_norm
            ),
            "overall_composition_context_cosine": float(context_cosine),
            "mean_id_user_norm": float(id_user.norm(dim=1).mean()),
            "mean_id_item_norm": float(id_item.norm(dim=1).mean()),
            "rho_zero_auxiliary_max_abs": (
                weighted_auxiliary_max if self.rho == 0.0 else float("nan")
            ),
        }
