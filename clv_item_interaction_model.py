"""Minimal jointly-trained CLV-conditioned item preference for LightGCN."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLVItemInteractionLightGCN(nn.Module):
    """Add one item-specific coefficient to the ordinary LightGCN score.

    The score is ``z_u @ z_i + c_u * a_i``.  ``c_u`` is a fixed centered
    historical-CLV percentile and ``a_i`` is learned jointly by the same BPR
    loss.  Zero initialization makes the first forward pass exactly M1.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        clv_coordinate: np.ndarray,
        adj: torch.Tensor,
        embedding_dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        coordinate = np.asarray(clv_coordinate, dtype=np.float32)
        if coordinate.shape != (n_users,) or not np.isfinite(coordinate).all():
            raise ValueError(f"clv_coordinate must be finite shape ({n_users},)")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape does not match user/item counts")
        if n_layers < 0 or embedding_dim <= 0 or pref_reg < 0:
            raise ValueError("invalid model configuration")

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.pref_reg = float(pref_reg)
        self.E_u = nn.Embedding(n_users, embedding_dim)
        self.E_i = nn.Embedding(n_items, embedding_dim)
        self.item_clv = nn.Embedding(n_items, 1)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        nn.init.zeros_(self.item_clv.weight)
        self.register_buffer("clv_coordinate", torch.from_numpy(coordinate))
        self.register_buffer("adj", adj.coalesce())

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        user, item = self.propagate()
        user = torch.cat([user, self.clv_coordinate[:, None]], dim=1)
        item = torch.cat([item, self.item_clv.weight], dim=1)
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
            + self.item_clv.weight[positives].pow(2).sum()
            + self.item_clv.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, w=None):
        if w is not None:
            raise ValueError("M4 sample weights are not part of this M2 model")
        if float(lam) != 0.0:
            raise ValueError("external score lambda is not supported")
        user, item, _, _ = self.embeddings()
        positive_score = (user[users] * item[positives]).sum(1)
        negative_score = (user[users] * item[negatives]).sum(1)
        bpr = -F.logsigmoid(positive_score - negative_score).mean()
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "p_correct": float(
                    (positive_score > negative_score).float().mean()
                ),
            }
        return loss, diagnostics

    @torch.no_grad()
    def interaction_diagnostics(self) -> dict[str, float]:
        coefficient = self.item_clv.weight[:, 0]
        return {
            "item_clv_coefficient_mean": float(coefficient.mean()),
            "item_clv_coefficient_std": float(coefficient.std(unbiased=False)),
            "item_clv_coefficient_abs_mean": float(coefficient.abs().mean()),
            "item_clv_coefficient_max_abs": float(coefficient.abs().max()),
            "clv_coordinate_mean": float(self.clv_coordinate.mean()),
            "clv_coordinate_std": float(
                self.clv_coordinate.std(unbiased=False)
            ),
        }
