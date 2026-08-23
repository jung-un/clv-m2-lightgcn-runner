"""CLV-profile relation for a LightGCN historical pilot.

The binary user-item graph is left untouched.  A second, train-only relation
assigns every active user to exactly one historical-CLV decile.  Users sharing
the same profile node can exchange preference representations without changing
the coefficients of the original purchase graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch import nn

import lightgcn_clv_v3 as v3


@dataclass(frozen=True)
class CLVProfileGraph:
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    profile_bin: np.ndarray
    profile_size: np.ndarray
    diagnostics: dict


def _basket_keys(train: pd.DataFrame) -> list[str]:
    return ["u_idx", "b_raw"] if "b_raw" in train else ["u_idx", "t"]


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def build_clv_profile_graph(
    train: pd.DataFrame,
    n_users: int,
    *,
    n_profile_bins: int = 10,
) -> CLVProfileGraph:
    """Build one train-only CLV-profile membership edge per active user."""
    required = {"u_idx", "t", "v"}
    missing = required - set(train.columns)
    if missing:
        raise ValueError(f"CLV profile graph requires {sorted(missing)}")
    if train.empty:
        raise ValueError("cannot build a CLV profile graph from empty training data")
    if n_profile_bins < 2:
        raise ValueError("n_profile_bins must be at least 2")

    basket = (
        train.groupby(_basket_keys(train), sort=False)["v"]
        .sum()
        .rename("basket_value")
    )
    grouped = basket.groupby(level="u_idx", sort=False)
    user = pd.DataFrame({"n_hat": grouped.size(), "v_hat": grouped.mean()})
    user["clv_proxy"] = user["n_hat"] * user["v_hat"]
    active_users = user.index.to_numpy(np.int64)
    if active_users.min() < 0 or active_users.max() >= n_users:
        raise ValueError("train user index is outside n_users")

    n_hat = np.full(n_users, np.nan, dtype=np.float64)
    v_hat = np.full(n_users, np.nan, dtype=np.float64)
    clv_proxy = np.full(n_users, np.nan, dtype=np.float64)
    profile_bin = np.full(n_users, -1, dtype=np.int64)
    n_hat[active_users] = user["n_hat"].to_numpy(np.float64)
    v_hat[active_users] = user["v_hat"].to_numpy(np.float64)
    clv_proxy[active_users] = user["clv_proxy"].to_numpy(np.float64)

    # Mid-rank percentiles make the intervention invariant to the monetary scale
    # while treating tied CLV values identically.
    rank = user["clv_proxy"].rank(method="average").to_numpy(np.float64)
    percentile = (rank - 0.5) / len(user)
    bins = np.minimum(
        (percentile * n_profile_bins).astype(np.int64), n_profile_bins - 1
    )
    profile_bin[active_users] = bins
    profile_size = np.bincount(bins, minlength=n_profile_bins).astype(np.int64)

    if np.any(profile_size == 0):
        raise RuntimeError("CLV percentile construction produced an empty profile node")
    if not np.all(np.isfinite(clv_proxy[active_users])):
        raise RuntimeError("active users must have finite historical CLV proxies")

    diagnostics = {
        "definition": {
            "historical_clv_proxy": "N_hat * V_hat",
            "n_hat": "number of distinct train baskets",
            "v_hat": "mean train basket value",
            "profile_membership": (
                "one edge per active user to a mid-rank historical-CLV decile"
            ),
            "purchase_graph": "unchanged binary M1 graph",
            "item_price_used_for_profile": False,
        },
        "n_active_users": int(len(active_users)),
        "n_profile_bins": int(n_profile_bins),
        "profile_size": profile_size.tolist(),
        "singleton_profile_count": int((profile_size == 1).sum()),
        "n_hat": _summary(n_hat[active_users]),
        "v_hat": _summary(v_hat[active_users]),
        "clv_proxy": _summary(clv_proxy[active_users]),
    }
    return CLVProfileGraph(
        n_hat=n_hat,
        v_hat=v_hat,
        clv_proxy=clv_proxy,
        profile_bin=profile_bin,
        profile_size=profile_size,
        diagnostics=diagnostics,
    )


class CLVProfileLightGCN(v3.DualSpaceLightGCN):
    """M1 LightGCN plus a jointly trained user-profile-user message path."""

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
        profile: CLVProfileGraph,
        *,
        alpha_init: float = 0.1,
    ) -> None:
        super().__init__(
            n_users, n_items, n_cat, x_val_u, x_item, item_cat, cfg, adj
        )
        if not 0 < alpha_init < 1:
            raise ValueError("alpha_init must be strictly between 0 and 1")
        active = profile.profile_bin >= 0
        if not np.all(active):
            raise ValueError("every indexed training user must have a CLV profile")
        self.register_buffer(
            "profile_bin", torch.from_numpy(profile.profile_bin).long()
        )
        self.register_buffer(
            "profile_size", torch.from_numpy(profile.profile_size).float()
        )
        self.profile_alpha_logit = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init)))
        )

    @property
    def profile_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.profile_alpha_logit)

    def pref_params(self):
        return super().pref_params() + [self.profile_alpha_logit]

    def _peer_profile_message(self, user_embedding: torch.Tensor) -> torch.Tensor:
        n_bins = int(self.profile_size.numel())
        group_sum = user_embedding.new_zeros((n_bins, user_embedding.shape[1]))
        group_sum.index_add_(0, self.profile_bin, user_embedding)
        peer_sum = group_sum[self.profile_bin] - user_embedding
        peer_count = self.profile_size[self.profile_bin] - 1.0
        return torch.where(
            (peer_count > 0).unsqueeze(1),
            peer_sum / peer_count.clamp_min(1.0).unsqueeze(1),
            torch.zeros_like(peer_sum),
        )

    def propagate_pref(self):
        base_user, base_item = super().propagate_pref()
        peer = self._peer_profile_message(base_user)
        return base_user + self.profile_alpha * peer, base_item

    def profile_diagnostics(self) -> dict:
        return {
            "learned_profile_alpha": float(self.profile_alpha.detach().cpu()),
            "profile_size": self.profile_size.detach().cpu().to(torch.int64).tolist(),
            "self_message_excluded": True,
            "purchase_operator_changed": False,
        }
