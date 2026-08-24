"""Train-only CLV-profile-to-item relation for a LightGCN M3 pilot.

The original binary user-item operator remains unchanged.  A separate sparse
relation links each historical-CLV profile to items whose train buyers
over-represent that profile, measured by positive pointwise mutual information
(PPMI) on unique user-item pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

from clv_m3_profile_graph import build_clv_profile_graph
import lightgcn_clv_v3 as v3


@dataclass(frozen=True)
class CLVProfileItemGraph:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    profile_bin: np.ndarray
    profile_size: np.ndarray
    profile_item_operator: torch.Tensor
    diagnostics: dict


def build_clv_profile_item_graph(
    train: pd.DataFrame,
    n_users: int,
    n_items: int,
    *,
    n_profile_bins: int = 10,
) -> CLVProfileItemGraph:
    """Build a row-normalized PPMI relation using train-only unique pairs."""
    if "i_idx" not in train:
        raise ValueError("CLV profile-item graph requires i_idx")
    profile = build_clv_profile_graph(
        train, n_users, n_profile_bins=n_profile_bins
    )
    if np.any(profile.profile_bin < 0):
        raise ValueError("every indexed training user must have a CLV profile")

    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    users = pairs["u_idx"].to_numpy(np.int64)
    items = pairs["i_idx"].to_numpy(np.int64)
    if items.min() < 0 or items.max() >= n_items:
        raise ValueError("train item index is outside n_items")
    bins = profile.profile_bin[users]

    counts = np.zeros((n_profile_bins, n_items), dtype=np.float64)
    np.add.at(counts, (bins, items), 1.0)
    item_degree = counts.sum(axis=0)
    profile_prior = profile.profile_size.astype(np.float64) / n_users
    observed_share = np.divide(
        counts,
        item_degree[None, :],
        out=np.zeros_like(counts),
        where=item_degree[None, :] > 0,
    )
    ratio = np.divide(
        observed_share,
        profile_prior[:, None],
        out=np.zeros_like(observed_share),
        where=profile_prior[:, None] > 0,
    )
    ppmi = np.zeros_like(ratio)
    positive = ratio > 1.0
    ppmi[positive] = np.log(ratio[positive])

    row_mass = ppmi.sum(axis=1)
    if np.any(row_mass <= 0):
        raise RuntimeError("a CLV profile has no positive-PMI item relation")
    normalized = ppmi / row_mass[:, None]
    rows, cols = np.nonzero(normalized)
    values = normalized[rows, cols].astype(np.float32)
    operator = torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([rows, cols])).long(),
        torch.from_numpy(values),
        size=(n_profile_bins, n_items),
        check_invariants=True,
    ).coalesce()

    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "profile_membership": (
                "one edge per active user to a mid-rank historical-CLV decile"
            ),
            "profile_item_weight": (
                "positive pointwise mutual information from train-only unique user-item pairs"
            ),
            "purchase_graph": "unchanged binary M1 graph",
            "item_price_used": False,
        },
        "n_active_users": int(n_users),
        "n_profile_bins": int(n_profile_bins),
        "profile_size": profile.profile_size.tolist(),
        "profile_item_relation": {
            "n_positive_edges": int(len(values)),
            "share_positive_of_possible": float(
                len(values) / (n_profile_bins * n_items)
            ),
            "row_mass_after_normalization": (
                np.bincount(rows, weights=values, minlength=n_profile_bins).tolist()
            ),
            "n_items_with_positive_relation": int(np.unique(cols).size),
            "n_nonselective_items": int(np.sum(~positive.any(axis=0))),
        },
        "n_hat": profile.diagnostics["n_hat"],
        "v_hat": profile.diagnostics["v_hat"],
        "clv_proxy": profile.diagnostics["clv_proxy"],
    }
    return CLVProfileItemGraph(
        n_hat=profile.n_hat,
        v_hat=profile.v_hat,
        clv_proxy=profile.clv_proxy,
        profile_bin=profile.profile_bin,
        profile_size=profile.profile_size,
        profile_item_operator=operator,
        diagnostics=diagnostics,
    )


class CLVProfileItemLightGCN(v3.DualSpaceLightGCN):
    """M1 LightGCN plus a jointly trained profile-to-item message path."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_cat: int,
        x_val_u: np.ndarray,
        x_item: np.ndarray,
        item_cat: np.ndarray,
        cfg: dict,
        adj: torch.Tensor,
        profile_item: CLVProfileItemGraph,
        *,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
            n_users, n_items, n_cat, x_val_u, x_item, item_cat, cfg, adj
        )
        if not 0 < alpha_init < 1:
            raise ValueError("alpha_init must be strictly between 0 and 1")
        if np.any(profile_item.profile_bin < 0):
            raise ValueError("every indexed training user must have a CLV profile")
        self.register_buffer(
            "profile_bin", torch.from_numpy(profile_item.profile_bin).long()
        )
        self.register_buffer(
            "profile_item_operator", profile_item.profile_item_operator
        )
        self.profile_alpha_logit = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init)))
        )

    @property
    def profile_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.profile_alpha_logit)

    def pref_params(self):
        return super().pref_params() + [self.profile_alpha_logit]

    def _profile_item_message(self, item_embedding: torch.Tensor) -> torch.Tensor:
        profile_message = torch.sparse.mm(
            self.profile_item_operator, item_embedding
        )
        return profile_message[self.profile_bin]

    def propagate_pref(self):
        base_user, base_item = super().propagate_pref()
        message = self._profile_item_message(base_item)
        return base_user + self.profile_alpha * message, base_item

    def profile_item_diagnostics(self) -> dict:
        return {
            "learned_profile_alpha": float(self.profile_alpha.detach().cpu()),
            "profile_item_edges": int(self.profile_item_operator._nnz()),
            "purchase_operator_changed": False,
            "item_embedding_receives_recommendation_gradient": True,
        }
