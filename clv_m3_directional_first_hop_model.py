"""LightGCN whose only graph intervention is the final user layer-1 term."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalFirstHopLightGCN(nn.Module):
    """Keep M1's layer-0, item path, and two-hop path exactly unchanged.

    The selected graph supplies only ``U1 = A_selected @ I0`` in the final
    user representation.  ``I1``, ``U2`` and ``I2`` are always computed with
    M1's binary symmetric-normalized directed blocks.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        active_user_from_item: torch.Tensor,
        dim: int = 64,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
    ):
        super().__init__()
        if min(n_users, n_items, dim) <= 0:
            raise ValueError("n_users, n_items and dim must be positive")
        if n_layers != 2:
            raise ValueError("directional first-hop M3 requires exactly two layers")
        if pref_reg < 0:
            raise ValueError("pref_reg must be non-negative")
        expected_user_item = (n_users, n_items)
        expected_item_user = (n_items, n_users)
        if tuple(base_user_from_item.shape) != expected_user_item:
            raise ValueError("base_user_from_item has the wrong shape")
        if tuple(active_user_from_item.shape) != expected_user_item:
            raise ValueError("active_user_from_item has the wrong shape")
        if tuple(base_item_from_user.shape) != expected_item_user:
            raise ValueError("base_item_from_user has the wrong shape")
        if not all(
            matrix.layout == torch.sparse_coo
            for matrix in (
                base_user_from_item,
                base_item_from_user,
                active_user_from_item,
            )
        ):
            raise ValueError("all propagation operators must be sparse COO tensors")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.dim = int(dim)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)

        self.E_u = nn.Embedding(n_users, dim)
        self.E_i = nn.Embedding(n_items, dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.register_buffer(
            "base_user_from_item",
            base_user_from_item.coalesce(),
            persistent=False,
        )
        self.register_buffer(
            "base_item_from_user",
            base_item_from_user.coalesce(),
            persistent=False,
        )
        self.register_buffer(
            "active_user_from_item",
            active_user_from_item.coalesce(),
            persistent=False,
        )

    def layer_embeddings(self) -> dict[str, torch.Tensor]:
        user0 = self.E_u.weight
        item0 = self.E_i.weight
        user1_m1 = torch.sparse.mm(self.base_user_from_item, item0)
        item1_m1 = torch.sparse.mm(self.base_item_from_user, user0)
        user2_m1 = torch.sparse.mm(self.base_user_from_item, item1_m1)
        item2_m1 = torch.sparse.mm(self.base_item_from_user, user1_m1)
        user1_final = torch.sparse.mm(self.active_user_from_item, item0)
        return {
            "user0": user0,
            "item0": item0,
            "user1_m1": user1_m1,
            "item1_m1": item1_m1,
            "user2_m1": user2_m1,
            "item2_m1": item2_m1,
            "user1_final": user1_final,
            "item1_final": item1_m1,
            "user2_final": user2_m1,
            "item2_final": item2_m1,
        }

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self.layer_embeddings()
        user = (
            layers["user0"]
            + layers["user1_final"]
            + layers["user2_m1"]
        ) / 3.0
        item = (
            layers["item0"]
            + layers["item1_m1"]
            + layers["item2_m1"]
        ) / 3.0
        return user, item

    def embeddings(self, need_value: bool = True):
        # The final two zero columns keep the shared evaluator interface intact.
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
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        base = self.base_user_from_item.coalesce()
        active = self.active_user_from_item.coalesce()
        if not torch.equal(base.indices(), active.indices()):
            raise RuntimeError("M1 and active M3 operators have different edge sets")
        ratio = active.values() / base.values()
        return {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "changed_user_first_hop_only": True,
            "binary_item_path_preserved": True,
            "binary_two_hop_path_preserved": True,
            "external_reranking": False,
            "first_hop_log_ratio_std": float(torch.log(ratio).std(unbiased=False)),
            "first_hop_ratio_min": float(ratio.min()),
            "first_hop_ratio_max": float(ratio.max()),
        }

