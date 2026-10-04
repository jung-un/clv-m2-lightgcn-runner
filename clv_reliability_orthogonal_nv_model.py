"""Jointly trained 64-D N/V matching representation for M2 and M5.

Historical q_N/q_V stay fixed observations. Shared projections are trained by
the same BPR objective as the ID embeddings. Item inputs are shrunk profiles
of their training buyers; positive rows exclude the current buyer. The N/V
residual is orthogonal to the propagated ID vector and has a support-dependent
norm, so free ID scale cannot silently erase it and sparse profiles back off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis


AXES = ("user_n", "user_v", "item_n", "item_v", "item_price")


def _midrank_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, float))
    if not len(reference):
        raise ValueError("가격 백분위 기준이 비었습니다")
    values = np.asarray(values, float)
    return (
        np.searchsorted(reference, values, side="left")
        + np.searchsorted(reference, values, side="right")
    ) / (2.0 * len(reference))


def _basis(values: np.ndarray, valid: np.ndarray, bandwidth: float) -> np.ndarray:
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("축 값과 유효 마스크 shape이 다릅니다")
    clean = np.where(valid, values, 0.5)
    return fixed_value_basis(clean, valid, bandwidth=bandwidth).astype(np.float32)


def _positive_median(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    positive = values[np.isfinite(values) & (values > 0)]
    return float(np.median(positive)) if len(positive) else 1.0


def _reliability(support: np.ndarray, tau: float, floor: float) -> np.ndarray:
    support = np.asarray(support, float)
    result = floor + (1.0 - floor) * support / (support + tau)
    return np.where(support > 0, result, floor).astype(np.float32)


def build_features(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    q_n,
    q_v,
    valid,
    bandwidth: float = 0.25,
    reliability_floor: float = 0.05,
) -> dict:
    """Build train-only user/item N/V inputs and positive-row LOO inputs."""

    required = {"u_idx", "i_idx", "up", "b_raw"}
    missing = required.difference(train.columns)
    if missing or train.empty:
        raise ValueError(f"비어 있지 않은 TRAIN 입력 열이 필요합니다: {sorted(missing)}")
    if min(n_users, n_items) <= 0 or not 0 < bandwidth <= 1:
        raise ValueError("카탈로그 크기와 bandwidth가 올바르지 않습니다")
    if not 0 < reliability_floor < 1:
        raise ValueError("reliability_floor는 (0,1)이어야 합니다")

    q_n, q_v, valid = np.asarray(q_n, float), np.asarray(q_v, float), np.asarray(valid, bool)
    if any(x.shape != (n_users,) for x in (q_n, q_v, valid)):
        raise ValueError("사용자 N/V 입력 크기가 다릅니다")
    for values in (q_n, q_v):
        if not np.isfinite(values[valid]).all() or np.any((values[valid] < 0) | (values[valid] > 1)):
            raise ValueError("유효 사용자 N/V는 유한한 [0,1] 값이어야 합니다")

    frame = train[["u_idx", "i_idx", "up", "b_raw"]].copy()
    for column, size in (("u_idx", n_users), ("i_idx", n_items)):
        ids = frame[column].to_numpy()
        if not np.isfinite(ids).all() or np.any((ids < 0) | (ids >= size) | (ids != np.floor(ids))):
            raise ValueError(f"{column}이 카탈로그 범위를 벗어났습니다")
    frame["up"] = frame.up.where(np.isfinite(frame.up) & (frame.up > 0))
    pairs = (
        frame.groupby(["u_idx", "i_idx"], sort=True)
        .agg(up=("up", "mean"), baskets=("b_raw", "nunique"))
        .reset_index()
    )
    u = pairs.u_idx.to_numpy(np.int64)
    i = pairs.i_idx.to_numpy(np.int64)
    price = pairs.up.to_numpy(float)
    price_valid = np.isfinite(price)
    keys = u * n_items + i
    if len(keys) == 0 or np.any(np.diff(keys) <= 0):
        raise ValueError("정렬된 고유 사용자-상품 쌍이 필요합니다")

    purchase_users = frame.drop_duplicates(["u_idx", "b_raw"]).u_idx.to_numpy(np.int64)
    user_purchase_support = np.bincount(purchase_users, minlength=n_users).astype(float)
    user_n_support = user_purchase_support
    user_v_support = user_purchase_support
    item_support = np.bincount(i, minlength=n_items).astype(float)
    tau_user_n = _positive_median(user_n_support)
    tau_user_v = _positive_median(user_v_support)
    tau_item = _positive_median(item_support)
    user_n_reliability = _reliability(user_n_support, tau_user_n, reliability_floor)
    user_v_reliability = _reliability(user_v_support, tau_user_v, reliability_floor)
    item_reliability = _reliability(item_support, tau_item, reliability_floor)

    valid_n = valid[u].astype(float)
    valid_v = valid[u].astype(float)
    n_count = np.bincount(i, weights=valid_n, minlength=n_items)
    v_count = np.bincount(i, weights=valid_v, minlength=n_items)
    n_mass = np.where(valid[u], q_n[u], 0.0)
    v_mass = np.where(valid[u], q_v[u], 0.0)
    n_sum = np.bincount(i, weights=n_mass, minlength=n_items)
    v_sum = np.bincount(i, weights=v_mass, minlength=n_items)
    prior_n = float(q_n[valid].mean()) if valid.any() else 0.5
    prior_v = float(q_v[valid].mean()) if valid.any() else 0.5
    item_n = (n_sum + tau_item * prior_n) / (n_count + tau_item)
    item_v = (v_sum + tau_item * prior_v) / (v_count + tau_item)

    price_count = np.bincount(i, weights=price_valid.astype(float), minlength=n_items)
    price_sum = np.bincount(i, weights=np.where(price_valid, price, 0.0), minlength=n_items)
    prior_price = float(price[price_valid].mean()) if price_valid.any() else np.nan
    if not np.isfinite(prior_price) or prior_price <= 0:
        raise ValueError("양의 TRAIN 단가가 없습니다")
    item_price = (price_sum + tau_item * prior_price) / (price_count + tau_item)
    reference = item_price[price_count > 0]
    item_price_pct = _midrank_percentile(reference, item_price)

    loo_n_count = n_count[i] - valid_n
    loo_v_count = v_count[i] - valid_v
    loo_item_n = (n_sum[i] - n_mass + tau_item * prior_n) / (loo_n_count + tau_item)
    loo_item_v = (v_sum[i] - v_mass + tau_item * prior_v) / (loo_v_count + tau_item)
    loo_price_count = price_count[i] - price_valid.astype(float)
    loo_item_price = (
        price_sum[i] - np.where(price_valid, price, 0.0) + tau_item * prior_price
    ) / (loo_price_count + tau_item)
    loo_item_price_pct = _midrank_percentile(reference, loo_item_price)
    loo_item_reliability = _reliability(
        np.maximum(item_support[i] - 1.0, 0.0), tau_item, reliability_floor
    )

    features = {
        "user_n": _basis(q_n, valid, bandwidth),
        "user_v": _basis(q_v, valid, bandwidth),
        "item_n": _basis(item_n, n_count > 0, bandwidth),
        "item_v": _basis(item_v, v_count > 0, bandwidth),
        "item_price": _basis(item_price_pct, price_count > 0, bandwidth),
        "loo_item_n": _basis(loo_item_n, loo_n_count > 0, bandwidth),
        "loo_item_v": _basis(loo_item_v, loo_v_count > 0, bandwidth),
        "loo_item_price": _basis(loo_item_price_pct, loo_price_count > 0, bandwidth),
        "user_n_reliability": user_n_reliability,
        "user_v_reliability": user_v_reliability,
        "item_reliability": item_reliability,
        "loo_item_reliability": loo_item_reliability,
        "keys": keys.astype(np.int64),
    }
    features["diagnostics"] = {
        "pairs": int(len(keys)), "valid_user_share": float(valid.mean()),
        "singleton_item_share": float((item_support == 1).mean()),
        "tau_user_n": tau_user_n, "tau_user_v": tau_user_v, "tau_item": tau_item,
        "reliability_floor": reliability_floor,
        "user_n_reliability_mean": float(user_n_reliability.mean()),
        "user_v_reliability_mean": float(user_v_reliability.mean()),
        "item_reliability_mean": float(item_reliability.mean()),
        "positive_loo_profile_share": float((loo_n_count > 0).mean()),
        "item_n_std": float(item_n[n_count > 0].std()) if np.any(n_count > 0) else 0.0,
        "item_v_std": float(item_v[v_count > 0].std()) if np.any(v_count > 0) else 0.0,
        "source": "training rows only", "positive_profile": "leave current buyer out",
        "missing_policy": "prior profile with minimum positive reliability; ID path retained",
    }
    return features


class ReliabilityOrthogonalNVLightGCN(nn.Module):
    """64-D ID representation plus a support-gated orthogonal N/V residual."""

    supports_clv_score_split = False

    def __init__(self, *, n_users, n_items, features, adj, id_dim=64, n_layers=2,
                 pref_reg=1e-3, shared_l2=1e-6, rho=0.10):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0 or n_layers < 0:
            raise ValueError("모형 차원이 올바르지 않습니다")
        if not 0 < rho < 0.5 or not np.isfinite(shared_l2) or shared_l2 < 0:
            raise ValueError("rho/shared_l2가 올바르지 않습니다")
        if adj.layout != torch.sparse_coo or adj.shape != (n_users + n_items,) * 2:
            raise ValueError("sparse COO 인접행렬 shape이 다릅니다")
        self.n_users, self.n_items = n_users, n_items
        self.id_dim, self.n_layers = id_dim, n_layers
        self.pref_reg, self.shared_l2, self.rho = pref_reg, shared_l2, rho
        self.E_u, self.E_i = nn.Embedding(n_users, id_dim), nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.encoders = nn.ModuleDict({name: nn.Linear(3, id_dim, bias=False) for name in AXES})
        for layer in self.encoders.values():
            nn.init.normal_(layer.weight, std=0.02)
        self.register_buffer("adj", adj.coalesce(), persistent=False)
        keys = np.asarray(features["keys"], np.int64)
        if len(keys) == 0 or np.any(np.diff(keys) <= 0):
            raise ValueError("정렬된 고유 positive key가 필요합니다")
        self.register_buffer("pair_keys", torch.from_numpy(keys), persistent=False)
        rows = {"user_n": n_users, "user_v": n_users, "item_n": n_items,
                "item_v": n_items, "item_price": n_items, "loo_item_n": len(keys),
                "loo_item_v": len(keys), "loo_item_price": len(keys)}
        for name, expected_rows in rows.items():
            values = np.asarray(features[name], np.float32)
            if values.shape != (expected_rows, 3) or not np.isfinite(values).all():
                raise ValueError(f"특징 행렬이 올바르지 않습니다: {name}")
            self.register_buffer(name + "_input", torch.from_numpy(values), persistent=False)
        reliability_rows = {"user_n_reliability": n_users, "user_v_reliability": n_users,
                            "item_reliability": n_items, "loo_item_reliability": len(keys)}
        for name, expected_rows in reliability_rows.items():
            values = np.asarray(features[name], np.float32)
            if values.shape != (expected_rows,) or not np.isfinite(values).all():
                raise ValueError(f"신뢰도 벡터가 올바르지 않습니다: {name}")
            self.register_buffer(name, torch.from_numpy(values), persistent=False)

    def _id_embeddings(self):
        current = torch.cat([self.E_u.weight, self.E_i.weight])
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[:self.n_users], total[self.n_users:]

    def _user_feature(self, rows=None):
        rows = slice(None) if rows is None else rows
        n = self.user_n_reliability[rows, None] * self.encoders["user_n"](self.user_n_input[rows])
        v = self.user_v_reliability[rows, None] * self.encoders["user_v"](self.user_v_input[rows])
        reliability = torch.sqrt((self.user_n_reliability[rows].square()
                                  + self.user_v_reliability[rows].square()) / 2)
        return n + v, reliability

    def _item_feature(self, rows=None, *, loo=False):
        rows = slice(None) if rows is None else rows
        prefix = "loo_item" if loo else "item"
        n = self.encoders["item_n"](getattr(self, prefix + "_n_input")[rows])
        v = self.encoders["item_v"](getattr(self, prefix + "_v_input")[rows])
        price = self.encoders["item_price"](getattr(self, prefix + "_price_input")[rows])
        reliability = (self.loo_item_reliability if loo else self.item_reliability)[rows]
        return reliability[..., None] * (n + v + price) / np.sqrt(3.0), reliability

    def _mix(self, identity, feature, reliability):
        identity_norm = identity.norm(dim=1, keepdim=True).clamp_min(1e-12)
        direction = identity / identity_norm
        orthogonal = feature - (feature * direction).sum(dim=1, keepdim=True) * direction
        orthogonal_norm = orthogonal.norm(dim=1, keepdim=True)
        unit = orthogonal / orthogonal_norm.clamp_min(1e-12)
        strength = self.rho * reliability[:, None]
        active = (orthogonal_norm > 1e-10).to(identity.dtype)
        return identity + active * strength * identity_norm * unit

    def id_only_embeddings(self):
        return self._id_embeddings()

    def embeddings(self, need_value=True):
        user_id, item_id = self._id_embeddings()
        user_feature, user_reliability = self._user_feature()
        item_feature, item_reliability = self._item_feature()
        user = self._mix(user_id, user_feature, user_reliability)
        item = self._mix(item_id, item_feature, item_reliability)
        return user, item, user.new_zeros((self.n_users, 1)), item.new_zeros((self.n_items, 1))

    def _pair_scores(self, users, positives, negatives):
        user_id, item_id = self._id_embeddings()
        user_feature, user_reliability = self._user_feature(users)
        user = self._mix(user_id[users], user_feature, user_reliability)
        keys = users * self.n_items + positives
        positions = torch.searchsorted(self.pair_keys, keys).clamp(max=len(self.pair_keys) - 1)
        if not torch.equal(self.pair_keys[positions], keys):
            raise ValueError("positive가 TRAIN pair에 없습니다")
        positive_feature, positive_reliability = self._item_feature(positions, loo=True)
        negative_feature, negative_reliability = self._item_feature(negatives)
        positive = self._mix(item_id[positives], positive_feature, positive_reliability)
        negative = self._mix(item_id[negatives], negative_feature, negative_reliability)
        return (user * positive).sum(1), (user * negative).sum(1)

    def batch_l2(self, users, positives, negatives):
        sampled = sum(table.square().sum() for table in
                      (self.E_u(users), self.E_i(positives), self.E_i(negatives))) / len(users)
        shared = sum(parameter.square().sum() for parameter in self.encoders.parameters())
        return self.pref_reg * sampled + self.shared_l2 * shared

    def _loss(self, users, positives, negatives, weights=None):
        positive, negative = self._pair_scores(users, positives, negatives)
        rows = F.softplus(negative - positive)
        if weights is not None:
            if weights.shape != rows.shape or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("M5 행가중치는 유한한 양수여야 합니다")
            rows = weights * rows
        bpr = rows.mean()
        return bpr + self.batch_l2(users, positives, negatives), {
            "bpr": float(bpr.detach()), "p_correct": float((positive > negative).float().mean().detach())}

    def bpr_loss(self, users, positives, negatives, weights=None):
        if weights is not None:
            raise ValueError("M2는 plain BPR만 사용합니다")
        return self._loss(users, positives, negatives)

    def weighted_bpr_loss(self, users, positives, negatives, weights):
        return self._loss(users, positives, negatives, weights)

    def training_gradient_diagnostics(self):
        return {name + "_last_batch_gradient_norm":
                (float(layer.weight.grad.norm()) if layer.weight.grad is not None else None)
                for name, layer in self.encoders.items()}

    @torch.no_grad()
    def representation_diagnostics(self):
        user_feature, user_reliability = self._user_feature()
        item_feature, item_reliability = self._item_feature()
        return {"rho_max": self.rho,
                "shared_parameters": sum(p.numel() for p in self.encoders.parameters()),
                "user_reliability_mean": float(user_reliability.mean()),
                "item_reliability_mean": float(item_reliability.mean()),
                "user_feature_mean_norm": float(user_feature.norm(dim=1).mean()),
                "item_feature_mean_norm": float(item_feature.norm(dim=1).mean()),
                **{name + "_parameter_norm": float(layer.weight.norm())
                   for name, layer in self.encoders.items()}}


def self_test() -> None:
    frame = pd.DataFrame({"u_idx": [0, 0, 1, 1], "i_idx": [0, 1, 1, 2],
                          "up": [1.0, 2.0, 3.0, 4.0], "b_raw": [10, 11, 20, 21]})
    features = build_features(frame, n_users=2, n_items=5,
        q_n=np.array([0.2, 0.8]), q_v=np.array([0.3, 0.7]), valid=np.ones(2, bool))
    assert features["keys"].tolist() == [0, 1, 6, 7]
    assert np.all(features["loo_item_reliability"] > 0)
    adj = torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
        torch.empty(0), (7, 7)).coalesce()
    torch.manual_seed(43)
    model = ReliabilityOrthogonalNVLightGCN(n_users=2, n_items=5,
        features=features, adj=adj, n_layers=0)
    user, item, *_ = model.embeddings()
    assert user.shape == (2, 64) and item.shape == (5, 64)
    users, positives, negatives = map(torch.tensor, ([0, 1], [1, 2], [3, 4]))
    loss, diagnostics = model.bpr_loss(users, positives, negatives)
    loss.backward()
    assert np.isfinite(diagnostics["bpr"])
    assert all(layer.weight.grad is not None and layer.weight.grad.norm() > 0
               for layer in model.encoders.values())


if __name__ == "__main__":
    self_test()
    print("reliability orthogonal N/V model self-test ok")
