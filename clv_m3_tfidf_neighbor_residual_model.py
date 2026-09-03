"""Joint LightGCN with a historical-CLV-gated user-neighbor residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TFIDFNeighborResidualLightGCN(nn.Module):
    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        user_neighbor_operator: torch.Tensor,
        gate: torch.Tensor,
        rho: float = 0.075,
        dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        eps_z: float = 1e-12,
        eps_m: float = 1e-12,
        eps_h: float = 1e-12,
    ):
        super().__init__()
        if min(n_users, n_items, dim) <= 0:
            raise ValueError("n_users, n_items and dim must be positive")
        if n_layers != 2:
            raise ValueError("TF-IDF neighbor residual M3 requires two layers")
        if rho < 0 or pref_reg < 0:
            raise ValueError("rho and pref_reg must be non-negative")
        if tuple(base_user_from_item.shape) != (n_users, n_items):
            raise ValueError("base_user_from_item has the wrong shape")
        if tuple(base_item_from_user.shape) != (n_items, n_users):
            raise ValueError("base_item_from_user has the wrong shape")
        if tuple(user_neighbor_operator.shape) != (n_users, n_users):
            raise ValueError("user_neighbor_operator has the wrong shape")
        if tuple(gate.shape) != (n_users,):
            raise ValueError("gate has the wrong shape")
        operators = (
            base_user_from_item,
            base_item_from_user,
            user_neighbor_operator,
        )
        if not all(value.layout == torch.sparse_coo for value in operators):
            raise ValueError("all propagation operators must be sparse COO tensors")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.rho = float(rho)
        self.pref_reg = float(pref_reg)
        self.eps_z = float(eps_z)
        self.eps_m = float(eps_m)
        self.eps_h = float(eps_h)

        self.E_u = nn.Embedding(n_users, dim)
        self.E_i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.register_buffer(
            "base_user_from_item", base_user_from_item.coalesce(), persistent=False
        )
        self.register_buffer(
            "base_item_from_user", base_item_from_user.coalesce(), persistent=False
        )
        neighbor = user_neighbor_operator.coalesce()
        self.register_buffer("user_neighbor_operator", neighbor, persistent=False)
        self.register_buffer("gate", gate.float(), persistent=False)
        neighbor_mass = torch.sparse.sum(neighbor, dim=1).to_dense()
        self.register_buffer(
            "eligible_neighbor", neighbor_mass > 0, persistent=False
        )

    def m1_layers(self) -> dict[str, torch.Tensor]:
        user0, item0 = self.E_u.weight, self.E_i.weight
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
        layer = self.m1_layers()
        user = (layer["user0"] + layer["user1"] + layer["user2"]) / 3.0
        item = (layer["item0"] + layer["item1"] + layer["item2"]) / 3.0
        return user, item

    def representation_parts(self) -> dict[str, torch.Tensor]:
        layer = self.m1_layers()
        base_user = (layer["user0"] + layer["user1"] + layer["user2"]) / 3.0
        base_item = (layer["item0"] + layer["item1"] + layer["item2"]) / 3.0
        message = torch.sparse.mm(self.user_neighbor_operator, layer["user1"])
        base_squared = base_user.pow(2).sum(dim=1)
        base_norm = base_squared.sqrt()
        message_norm = message.norm(dim=1)
        valid = (
            self.eligible_neighbor
            & (base_squared > self.eps_z**2)
            & (message_norm > self.eps_m)
        )
        safe_squared = torch.where(valid, base_squared, torch.ones_like(base_squared))
        projection = (message * base_user).sum(dim=1) / safe_squared
        residual = message - projection[:, None] * base_user
        residual = torch.where(valid[:, None], residual, torch.zeros_like(residual))
        safe_message_norm = torch.where(
            valid, message_norm, torch.ones_like(message_norm)
        )
        eta_raw = residual.norm(dim=1) / safe_message_norm
        eta = torch.where(valid, eta_raw.clamp(0.0, 1.0), torch.zeros_like(eta_raw))
        scaled_residual = base_norm[:, None] * residual / safe_message_norm[:, None]
        scaled_residual = torch.where(
            valid[:, None], scaled_residual, torch.zeros_like(scaled_residual)
        )
        return {
            "m1_user": base_user,
            "m1_item": base_item,
            "neighbor_message": message,
            "residual": residual,
            "eta": eta,
            "scaled_residual": scaled_residual,
            "valid": valid,
        }

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.representation_parts()
        base_user, base_item = parts["m1_user"], parts["m1_item"]
        if self.rho == 0.0 or not bool(torch.any(self.gate != 0)):
            return base_user, base_item
        h = base_user + self.rho * self.gate[:, None] * parts["scaled_residual"]
        h_norm = h.norm(dim=1)
        base_norm = base_user.norm(dim=1)
        valid = parts["valid"] & (h_norm > self.eps_h)
        safe_h_norm = torch.where(valid, h_norm, torch.ones_like(h_norm))
        normalized = base_norm[:, None] * h / safe_h_norm[:, None]
        user = torch.where(valid[:, None], normalized, base_user)
        return user, base_item

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
    def intervention_diagnostics(self) -> dict[str, float | int | bool]:
        parts = self.representation_parts()
        valid = parts["valid"]
        eta = parts["eta"]
        budget = self.gate * eta
        dot = (parts["residual"] * parts["m1_user"]).sum(dim=1).abs()
        user, _ = self.propagated_embeddings()
        norm_error = (user.norm(dim=1) - parts["m1_user"].norm(dim=1)).abs()
        return {
            "rho": self.rho,
            "eligible_user_count": int(valid.sum()),
            "eligible_user_share": float(valid.float().mean()),
            "eta_mean_all": float(eta.mean()),
            "eta_mean_eligible": float(eta[valid].mean()) if bool(valid.any()) else 0.0,
            "eta_std_eligible": float(eta[valid].std(unbiased=False))
            if bool(valid.any())
            else 0.0,
            "effective_budget_all": float(budget.mean()),
            "effective_budget_eligible": float(budget[valid].mean())
            if bool(valid.any())
            else 0.0,
            "max_absolute_orthogonality_error": float(dot.max()),
            "max_user_norm_absolute_error": float(norm_error.max()),
            "item_representation_unchanged": True,
            "external_reranking": False,
        }

    def epoch_training_diagnostics(self) -> dict[str, float | int | bool]:
        return self.intervention_diagnostics()

    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        return self.intervention_diagnostics()

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(value: torch.Tensor) -> float:
            return 0.0 if value.grad is None else float(value.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
        }

