"""High-CLV routed candidate-N representation after LightGCN ID propagation."""

from __future__ import annotations

import numpy as np

from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)


class HighCLVNFitLightGCN(M5NConditionedValueBasisLightGCN):
    """Append a fixed N-position basis only for prespecified high-CLV users."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_q_n: np.ndarray,
        high_clv_gate: np.ndarray,
        clv_valid: np.ndarray,
        item_buyer_q_n: np.ndarray,
        item_context_valid: np.ndarray,
        adj,
        id_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        basis_bandwidth: float = 0.25,
    ):
        gate = np.asarray(high_clv_gate, dtype=np.float32)
        valid = np.asarray(clv_valid, dtype=bool)
        super().__init__(
            n_users=n_users,
            n_items=n_items,
            user_q_n=np.zeros(n_users, dtype=np.float32),
            user_q_v=np.asarray(user_q_n, dtype=np.float32),
            user_q_c=gate,
            user_clv_valid=valid,
            item_price_percentile=np.asarray(item_buyer_q_n, dtype=np.float32),
            item_price_valid=np.asarray(item_context_valid, dtype=bool),
            adj=adj,
            id_dim=id_dim,
            rho=rho,
            n_layers=n_layers,
            pref_reg=pref_reg,
            basis_bandwidth=basis_bandwidth,
            constant_gate=1.0,
            economic_propagation=False,
        )

    def representation_diagnostics(self) -> dict:
        values = super().representation_diagnostics()
        high = self.user_clv_level > 0
        values.update(
            {
                "research_axis": "M2 representation",
                "historical_clv_role": "fixed upper-20-percent direct gate",
                "n_role": "user q_N versus shrunk item-purchaser q_N position",
                "v_input_in_m2": False,
                "candidate_n_graph_propagation": False,
                "high_clv_direct_user_count": int(high.sum()),
            }
        )
        return values

