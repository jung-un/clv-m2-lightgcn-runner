"""Joint LightGCN with a CLV-conditioned taste-neighbor propagation path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLVTasteNeighborLightGCN(nn.Module):
    """Replace only the second user-layer message by a fixed neighbor mixture."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        user_neighbor_operator: torch.Tensor,
        gamma: float = 0.075,
        dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(n_users, n_items, dim) <= 0:
            raise ValueError("n_users, n_items and dim must be positive")
        if n_layers != 2:
            raise ValueError("CLV taste-neighbor M3 requires exactly two layers")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be finite and in [0, 1]")
        if pref_reg < 0:
            raise ValueError("pref_reg must be non-negative")
        expected = {
            "base_user_from_item": ((n_users, n_items), base_user_from_item),
            "base_item_from_user": ((n_items, n_users), base_item_from_user),
            "user_neighbor_operator": ((n_users, n_users), user_neighbor_operator),
        }
        for name, (shape, operator) in expected.items():
            if tuple(operator.shape) != shape:
                raise ValueError(f"{name} has the wrong shape")
            if operator.layout != torch.sparse_coo:
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
        for name, (_, operator) in expected.items():
            self.register_buffer(name, operator.coalesce(), persistent=False)
        neighbor_mass = torch.sparse.sum(
            self.user_neighbor_operator, dim=1
        ).to_dense()
        self.register_buffer(
            "eligible_neighbor", neighbor_mass > 0, persistent=False
        )

    def m1_layers(self) -> dict[str, torch.Tensor]:
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

    def m1_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self.m1_layers()
        user = (layers["user0"] + layers["user1"] + layers["user2"]) / 3.0
        item = (layers["item0"] + layers["item1"] + layers["item2"]) / 3.0
        return user, item

    def representation_parts(self) -> dict[str, torch.Tensor]:
        layers = self.m1_layers()
        m1_user = (layers["user0"] + layers["user1"] + layers["user2"]) / 3.0
        m1_item = (layers["item0"] + layers["item1"] + layers["item2"]) / 3.0
        neighbor_message = torch.sparse.mm(
            self.user_neighbor_operator, layers["user1"]
        )
        mixed = (
            (1.0 - self.gamma) * layers["user2"]
            + self.gamma * neighbor_message
        )
        if self.gamma == 0.0:
            arm_user2 = layers["user2"]
        else:
            arm_user2 = torch.where(
                self.eligible_neighbor[:, None], mixed, layers["user2"]
            )
        return {
            **layers,
            "m1_user2": layers["user2"],
            "m1_user": m1_user,
            "m1_item": m1_item,
            "neighbor_message": neighbor_message,
            "arm_user2": arm_user2,
        }

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.representation_parts()
        if self.gamma == 0.0:
            return parts["m1_user"], parts["m1_item"]
        user = (parts["user0"] + parts["user1"] + parts["arm_user2"]) / 3.0
        return user, parts["m1_item"]

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
        positive_score = (selected * item[positives]).sum(dim=1)
        negative_score = (selected * item[negatives]).sum(dim=1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        return loss, {
            "bpr": float(bpr.detach()),
            "objective": "plain_bpr",
            "p_correct": float(
                (positive_score > negative_score).float().mean().detach()
            ),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        parts = self.representation_parts()
        change = parts["arm_user2"] - parts["m1_user2"]
        base_norm = parts["m1_user2"].norm(dim=1)
        change_norm = change.norm(dim=1)
        valid = self.eligible_neighbor & (base_norm > 0)
        ratio = torch.zeros_like(change_norm)
        ratio[valid] = change_norm[valid] / base_norm[valid]
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "gamma": self.gamma,
            "eligible_user_count": int(self.eligible_neighbor.sum()),
            "eligible_user_share": float(self.eligible_neighbor.float().mean()),
            "mean_user2_change_to_m1_user2_norm_ratio": (
                float(ratio[valid].mean()) if bool(valid.any()) else 0.0
            ),
            "median_user2_change_to_m1_user2_norm_ratio": (
                float(ratio[valid].median()) if bool(valid.any()) else 0.0
            ),
            "binary_m1_path_preserved": True,
            "item_representation_unchanged": True,
            "external_reranking": False,
        }

    def epoch_training_diagnostics(self) -> dict[str, float | int | bool]:
        return self.representation_diagnostics()

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(value: torch.Tensor) -> float:
            return 0.0 if value.grad is None else float(value.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
        }


__all__ = ["CLVTasteNeighborLightGCN"]
