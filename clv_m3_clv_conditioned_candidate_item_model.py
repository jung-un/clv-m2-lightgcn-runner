"""M1 LightGCN with a CLV-conditioned candidate-item message path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_m3_clv_conditioned_category_transition_model import (
    build_binary_directional_blocks,
)


class CLVCandidateItemLightGCN(nn.Module):
    """Keep the binary M1 path and add a direct user-to-candidate-item message."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        user_candidate_item: torch.Tensor,
        gamma: float = 0.075,
        dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(n_users, n_items, dim) <= 0:
            raise ValueError("graph sizes and embedding dimension must be positive")
        if n_layers != 2:
            raise ValueError("candidate-item M3 requires exactly two M1 layers")
        if gamma <= 0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and positive")
        if pref_reg < 0:
            raise ValueError("pref_reg must be non-negative")
        expected = {
            "base_user_from_item": ((n_users, n_items), base_user_from_item),
            "base_item_from_user": ((n_items, n_users), base_item_from_user),
            "user_candidate_item": ((n_users, n_items), user_candidate_item),
        }
        for name, (shape, matrix) in expected.items():
            if tuple(matrix.shape) != shape:
                raise ValueError(f"{name} has the wrong shape")
            if matrix.layout != torch.sparse_coo:
                raise ValueError(f"{name} must be a sparse COO tensor")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.gamma = float(gamma)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, dim)
        self.E_i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        for name, (_, matrix) in expected.items():
            self.register_buffer(name, matrix.coalesce(), persistent=False)

    def m1_layer_embeddings(self) -> dict[str, torch.Tensor]:
        user0 = self.E_u.weight
        item0 = self.E_i.weight
        user1 = torch.sparse.mm(self.base_user_from_item, item0)
        item1 = torch.sparse.mm(self.base_item_from_user, user0)
        user2 = torch.sparse.mm(self.base_user_from_item, item1)
        item2 = torch.sparse.mm(self.base_item_from_user, user1)
        return {
            "user0": user0,
            "item0": item0,
            "user1": user1,
            "item1": item1,
            "user2": user2,
            "item2": item2,
        }

    def representation_parts(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layers = self.m1_layer_embeddings()
        base_user = (
            layers["user0"] + layers["user1"] + layers["user2"]
        ) / 3.0
        base_item = (
            layers["item0"] + layers["item1"] + layers["item2"]
        ) / 3.0
        candidate_message = torch.sparse.mm(
            self.user_candidate_item,
            base_item,
        )
        return base_user, base_item, candidate_message

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_user, base_item, candidate_message = self.representation_parts()
        return base_user + self.gamma * candidate_message, base_item

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        return (
            user,
            item,
            user.new_zeros((self.n_users, 1)),
            item.new_zeros((self.n_items, 1)),
        )

    def batch_l2(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        need_value: bool = False,
    ) -> torch.Tensor:
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        sampled = (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)
        return self.pref_reg * sampled

    def bpr_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        gate=None,
        lam: float = 0.0,
        weights=None,
    ):
        if weights is not None:
            raise ValueError("M3 graph experiment cannot use M4 sample weights")
        if float(lam) != 0.0:
            raise ValueError("M3 graph experiment cannot add an external score")
        user, item = self.propagated_embeddings()
        selected = user[users]
        positive_score = (selected * item[positives]).sum(1)
        negative_score = (selected * item[negatives]).sum(1)
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
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        base_user, _, message = self.representation_parts()
        base_norm = base_user.norm(dim=1)
        message_norm = (self.gamma * message).norm(dim=1)
        active = message_norm > 0
        valid = active & (base_norm > 0)
        ratio = torch.zeros_like(message_norm)
        ratio[valid] = message_norm[valid] / base_norm[valid]
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "gamma": self.gamma,
            "binary_m1_path_preserved": True,
            "direct_candidate_item_relation_inside_forward": True,
            "external_reranking": False,
            "active_candidate_user_share": float(active.float().mean()),
            "mean_aux_to_m1_user_norm_ratio": float(
                ratio[valid].mean() if valid.any() else 0.0
            ),
            "median_aux_to_m1_user_norm_ratio": float(
                ratio[valid].median() if valid.any() else 0.0
            ),
        }


__all__ = ["CLVCandidateItemLightGCN", "build_binary_directional_blocks"]
