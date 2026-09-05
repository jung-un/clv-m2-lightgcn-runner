"""Role-separated N/V economic representation for the M2+M4 M5 arm."""

from __future__ import annotations

import numpy as np
import torch

from clv_m5_economic_positive_weight_model import M5EconomicLightGCN


class M5NVEconomicLightGCN(M5EconomicLightGCN):
    """LightGCN whose M2 block uses N only as a post-projection strength gate.

    The user projection receives V level and the shrunken economic-position
    profile.  The separate N gate is applied afterwards, so purchase frequency
    cannot learn an item-direction of its own.  The complete layer-0 embedding
    is propagated and trained with the same ranking loss and optimizer.
    """

    def __init__(
        self,
        *,
        user_activity_gate: np.ndarray,
        **kwargs,
    ):
        gate = np.asarray(user_activity_gate, dtype=np.float32)
        n_users = int(kwargs.get("n_users", -1))
        if gate.shape != (n_users,):
            raise ValueError("q_N gate shape이 n_users와 다릅니다")
        if not np.isfinite(gate).all() or np.any((gate < 0.0) | (gate > 1.0)):
            raise ValueError("q_N gate는 [0,1]의 유한값이어야 합니다")
        super().__init__(**kwargs)
        self.register_buffer(
            "user_activity_gate", torch.from_numpy(gate.copy()), persistent=False
        )

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        user = 0.5 * torch.tanh(
            self.user_economic_projection(self.user_economic_input)
        )
        item = 0.5 * torch.tanh(
            self.item_economic_projection(self.item_economic_input)
        )
        user = (
            user
            * self.user_activity_gate[:, None]
            * self.user_economic_valid[:, None]
        )
        item = item * self.item_economic_valid[:, None]
        return user, item

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool]:
        diagnostics = super().representation_diagnostics()
        valid = self.user_economic_valid.bool()
        active_gate = self.user_activity_gate[valid]
        diagnostics.update(
            {
                "explicit_q_n_in_m2": True,
                "explicit_q_v_in_m2": True,
                "q_c_in_m2": False,
                "q_n_role": "post_projection_strength_only",
                "v_role": "economic_direction",
                "q_n_gate_mean": (
                    float(active_gate.mean()) if active_gate.numel() else 0.0
                ),
                "q_n_gate_std": (
                    float(active_gate.std(unbiased=False))
                    if active_gate.numel()
                    else 0.0
                ),
            }
        )
        return diagnostics
