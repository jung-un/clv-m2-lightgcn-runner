"""Semantic N/V M5 with a graph intervention isolated to user layer 1."""

from __future__ import annotations

import torch

from clv_m5_semantic_nv_model import M5SemanticNVEconomicLightGCN


class M5SemanticNVIsolatedFirstHopLightGCN(M5SemanticNVEconomicLightGCN):
    """Keep M2 and M4 intact while replacing only the user layer-1 message.

    The item path and both two-hop paths always use the binary M1 operator.
    The active operator is used exactly once: ``U1 = A_active_UI @ I0``.
    """

    def __init__(
        self,
        *,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        active_user_from_item: torch.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.n_layers != 2:
            raise ValueError("isolated first-hop M3 requires exactly two layers")
        expected_ui = (self.n_users, self.n_items)
        expected_iu = (self.n_items, self.n_users)
        if tuple(base_user_from_item.shape) != expected_ui:
            raise ValueError("base_user_from_item has the wrong shape")
        if tuple(active_user_from_item.shape) != expected_ui:
            raise ValueError("active_user_from_item has the wrong shape")
        if tuple(base_item_from_user.shape) != expected_iu:
            raise ValueError("base_item_from_user has the wrong shape")
        operators = (
            base_user_from_item,
            base_item_from_user,
            active_user_from_item,
        )
        if not all(operator.layout == torch.sparse_coo for operator in operators):
            raise ValueError("all propagation operators must be sparse COO tensors")
        self.register_buffer(
            "base_user_from_item", base_user_from_item.coalesce(), persistent=False
        )
        self.register_buffer(
            "base_item_from_user", base_item_from_user.coalesce(), persistent=False
        )
        self.register_buffer(
            "active_user_from_item", active_user_from_item.coalesce(), persistent=False
        )

    def layer_embeddings(self) -> dict[str, torch.Tensor]:
        user0 = self.E_u.weight
        item0 = self.E_i.weight
        user1_base = torch.sparse.mm(self.base_user_from_item, item0)
        item1_base = torch.sparse.mm(self.base_item_from_user, user0)
        user2_base = torch.sparse.mm(self.base_user_from_item, item1_base)
        item2_base = torch.sparse.mm(self.base_item_from_user, user1_base)
        user1_active = torch.sparse.mm(self.active_user_from_item, item0)
        return {
            "user0": user0,
            "item0": item0,
            "user1_base": user1_base,
            "item1_base": item1_base,
            "user2_base": user2_base,
            "item2_base": item2_base,
            "user1_active": user1_active,
        }

    def _propagate_id(self) -> tuple[torch.Tensor, torch.Tensor]:
        layers = self.layer_embeddings()
        user = (
            layers["user0"]
            + layers["user1_active"]
            + layers["user2_base"]
        ) / 3.0
        item = (
            layers["item0"]
            + layers["item1_base"]
            + layers["item2_base"]
        ) / 3.0
        return user, item

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        diagnostics = super().representation_diagnostics()
        base = self.base_user_from_item.coalesce()
        active = self.active_user_from_item.coalesce()
        if not torch.equal(base.indices(), active.indices()):
            raise RuntimeError("base and active first-hop operators have different edges")
        ratio = active.values() / base.values()
        diagnostics.update(
            {
                "m3_user_first_hop_only": True,
                "m3_item_path_binary": True,
                "m3_two_hop_paths_binary": True,
                "m3_first_hop_log_ratio_std": float(
                    torch.log(ratio).std(unbiased=False)
                ),
                "m3_first_hop_ratio_min": float(ratio.min()),
                "m3_first_hop_ratio_max": float(ratio.max()),
                "m3_changed_edge_share": float(
                    (~torch.isclose(ratio, torch.ones_like(ratio))).float().mean()
                ),
            }
        )
        return diagnostics
