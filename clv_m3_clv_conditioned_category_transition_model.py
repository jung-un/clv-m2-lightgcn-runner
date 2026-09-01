"""M1 LightGCN with one CLV-conditioned category-transition message path."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_binary_directional_blocks(
    edge_users: np.ndarray,
    edge_items: np.ndarray,
    n_users: int,
    n_items: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_users = np.asarray(edge_users, dtype=np.int64)
    edge_items = np.asarray(edge_items, dtype=np.int64)
    if edge_users.shape != edge_items.shape or not len(edge_users):
        raise ValueError("aligned non-empty binary edges are required")
    user_degree = np.bincount(edge_users, minlength=n_users).astype(np.float64)
    item_degree = np.bincount(edge_items, minlength=n_items).astype(np.float64)
    if np.any(user_degree[edge_users] <= 0) or np.any(item_degree[edge_items] <= 0):
        raise RuntimeError("binary graph degree must be positive on every edge")
    values = 1.0 / np.sqrt(
        user_degree[edge_users] * item_degree[edge_items]
    )
    indices = torch.from_numpy(np.stack([edge_users, edge_items])).long()
    coefficients = torch.from_numpy(values.astype(np.float32))
    with torch.sparse.check_sparse_tensor_invariants():
        user_from_item = torch.sparse_coo_tensor(
            indices,
            coefficients,
            size=(n_users, n_items),
        ).coalesce()
    return user_from_item.to(device), user_from_item.transpose(0, 1).coalesce().to(device)


class CLVCategoryTransitionLightGCN(nn.Module):
    """Preserve M1 and add a fixed-strength graph message inside its forward."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_cat: int,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        user_target_category: torch.Tensor,
        category_item: torch.Tensor,
        gamma: float = 0.075,
        dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ) -> None:
        super().__init__()
        if min(n_users, n_items, n_cat, dim) <= 0:
            raise ValueError("graph sizes and embedding dimension must be positive")
        if n_layers != 2:
            raise ValueError("category-transition M3 requires exactly two M1 layers")
        if gamma <= 0 or not np.isfinite(gamma):
            raise ValueError("gamma must be finite and positive")
        if pref_reg < 0:
            raise ValueError("pref_reg must be non-negative")
        expected = {
            "base_user_from_item": ((n_users, n_items), base_user_from_item),
            "base_item_from_user": ((n_items, n_users), base_item_from_user),
            "user_target_category": ((n_users, n_cat), user_target_category),
            "category_item": ((n_cat, n_items), category_item),
        }
        for name, (shape, matrix) in expected.items():
            if tuple(matrix.shape) != shape:
                raise ValueError(f"{name} has the wrong shape")
            if matrix.layout != torch.sparse_coo:
                raise ValueError(f"{name} must be a sparse COO tensor")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_cat = int(n_cat)
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
        category = torch.sparse.mm(self.category_item, base_item)
        transition_message = torch.sparse.mm(
            self.user_target_category, category
        )
        return base_user, base_item, transition_message

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_user, base_item, transition_message = self.representation_parts()
        return base_user + self.gamma * transition_message, base_item

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

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
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        base_user, _, message = self.representation_parts()
        base_norm = base_user.norm(dim=1)
        message_norm = (self.gamma * message).norm(dim=1)
        active = message_norm > 0
        ratio = torch.zeros_like(message_norm)
        valid = active & (base_norm > 0)
        ratio[valid] = message_norm[valid] / base_norm[valid]
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "gamma": self.gamma,
            "binary_m1_path_preserved": True,
            "graph_relation_inside_forward": True,
            "external_reranking": False,
            "active_transition_user_share": float(active.float().mean()),
            "mean_aux_to_m1_user_norm_ratio": float(
                ratio[valid].mean() if valid.any() else 0.0
            ),
            "median_aux_to_m1_user_norm_ratio": float(
                ratio[valid].median() if valid.any() else 0.0
            ),
        }
