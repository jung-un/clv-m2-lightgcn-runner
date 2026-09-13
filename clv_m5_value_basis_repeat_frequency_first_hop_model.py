"""q_V value basis plus an isolated q_N-conditioned user first hop."""

from __future__ import annotations

import torch

from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)


class M5ValueBasisRepeatFrequencyFirstHopLightGCN(
    M5NConditionedValueBasisLightGCN
):
    """Use q_V in M2 and reserve q_N for M3 first-hop reallocation."""

    def __init__(
        self,
        *,
        base_user_from_item: torch.Tensor,
        base_item_from_user: torch.Tensor,
        active_user_from_item: torch.Tensor,
        **kwargs,
    ):
        kwargs["constant_gate"] = 1.0
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

    def _propagate(
        self, user: torch.Tensor, item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        user1_base = torch.sparse.mm(self.base_user_from_item, item)
        item1_base = torch.sparse.mm(self.base_item_from_user, user)
        user2_base = torch.sparse.mm(self.base_user_from_item, item1_base)
        item2_base = torch.sparse.mm(self.base_item_from_user, user1_base)
        user1_active = torch.sparse.mm(self.active_user_from_item, item)
        return (
            (user + user1_active + user2_base) / 3.0,
            (item + item1_base + item2_base) / 3.0,
        )

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
                "explicit_q_n_in_m2": False,
                "n_role": "M3 user first-hop repeat-frequency reallocation only",
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
