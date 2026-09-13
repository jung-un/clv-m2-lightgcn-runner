"""Category-allocated historical N and value-position basis in one LightGCN.

The historical purchase-frequency component q_N is not redefined.  It is
allocated across categories according to the user's train-only transaction
composition.  The population category composition is subtracted so that the
new block represents user-specific frequency allocation rather than category
popularity.  Exact previously purchased items are still handled only by the
new-item evaluation mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis


@dataclass(frozen=True)
class CategoryFrequencyFeatures:
    """Fixed train-only inputs for the category allocation of q_N."""

    user_category_n_residual: np.ndarray
    item_category: np.ndarray
    user_valid: np.ndarray
    diagnostics: dict[str, float | int | str]


def build_category_frequency_features(
    train: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    n_categories: int,
    q_n: np.ndarray,
    clv_valid: np.ndarray,
    shrinkage_strength: float = 10.0,
) -> CategoryFrequencyFeatures:
    """Allocate normalized purchase frequency over train-only categories.

    A transaction contributes total mass one.  When a basket contains several
    categories, that mass is divided equally among its distinct categories.
    Thus basket size cannot inflate N and, before centering, a user's category
    components sum back to the user's total q_N.
    """

    required = {"u_idx", "i_idx", "cat_idx", "b_raw"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"카테고리별 N 입력 열 누락: {sorted(missing)}")
    if min(n_users, n_items, n_categories) <= 0:
        raise ValueError("사용자·상품·카테고리 수는 양수여야 합니다")
    if not np.isfinite(shrinkage_strength) or shrinkage_strength < 0.0:
        raise ValueError("축소강도는 0 이상의 유한값이어야 합니다")

    q_n = np.asarray(q_n, dtype=np.float64)
    valid = np.asarray(clv_valid, dtype=bool)
    if q_n.shape != (n_users,) or valid.shape != (n_users,):
        raise ValueError("q_N 또는 유효성 마스크 shape이 잘못됐습니다")
    if not np.isfinite(q_n).all() or np.any((q_n < 0.0) | (q_n > 1.0)):
        raise ValueError("q_N은 유한한 [0,1] 값이어야 합니다")

    frame = train.loc[:, ["u_idx", "i_idx", "cat_idx", "b_raw"]].copy()
    for column in ("u_idx", "i_idx", "cat_idx"):
        frame[column] = frame[column].astype(np.int64)
    if not frame["u_idx"].between(0, n_users - 1).all():
        raise ValueError("u_idx가 사용자 범위를 벗어났습니다")
    if not frame["i_idx"].between(0, n_items - 1).all():
        raise ValueError("i_idx가 상품 범위를 벗어났습니다")
    if not frame["cat_idx"].between(0, n_categories - 1).all():
        raise ValueError("cat_idx가 카테고리 범위를 벗어났습니다")

    # Every catalog item receives one train-only category label.
    item_mode = frame.groupby("i_idx", sort=True)["cat_idx"].agg(
        lambda values: int(values.mode(dropna=True).iat[0])
    )
    if len(item_mode) != n_items:
        raise RuntimeError("MIN_ITEM_INTER=1인데 카테고리가 없는 상품이 있습니다")
    item_category = np.empty(n_items, dtype=np.int64)
    item_category[item_mode.index.to_numpy(np.int64)] = item_mode.to_numpy(
        np.int64
    )

    # One basket contributes one unit in total, regardless of item-line count.
    basket_category = frame.drop_duplicates(
        ["u_idx", "b_raw", "cat_idx"]
    ).loc[:, ["u_idx", "b_raw", "cat_idx"]]
    category_count = basket_category.groupby(
        ["u_idx", "b_raw"], sort=False
    )["cat_idx"].transform("size")
    basket_category["mass"] = 1.0 / category_count.to_numpy(np.float64)

    basket_count_s = basket_category.drop_duplicates(
        ["u_idx", "b_raw"]
    ).groupby("u_idx", sort=False).size()
    basket_count = np.zeros(n_users, dtype=np.float64)
    basket_count[basket_count_s.index.to_numpy(np.int64)] = (
        basket_count_s.to_numpy(np.float64)
    )
    category_mass = np.zeros((n_users, n_categories), dtype=np.float64)
    grouped = basket_category.groupby(["u_idx", "cat_idx"], sort=False)[
        "mass"
    ].sum()
    grouped_users = grouped.index.get_level_values(0).to_numpy(np.int64)
    grouped_categories = grouped.index.get_level_values(1).to_numpy(np.int64)
    category_mass[grouped_users, grouped_categories] = grouped.to_numpy(
        np.float64
    )

    n_transactions = float(basket_count.sum())
    if n_transactions <= 0.0:
        raise RuntimeError("카테고리별 N을 계산할 거래가 없습니다")
    population_share = category_mass.sum(axis=0) / n_transactions
    if not np.isclose(population_share.sum(), 1.0, atol=1e-8):
        raise RuntimeError("모집단 카테고리 거래비중의 합이 1이 아닙니다")

    denominator = basket_count[:, None] + float(shrinkage_strength)
    shrunken_share = np.divide(
        category_mass
        + float(shrinkage_strength) * population_share[None, :],
        denominator,
        out=np.zeros_like(category_mass),
        where=denominator > 0.0,
    )
    observed = basket_count > 0.0
    if observed.any() and not np.allclose(
        shrunken_share[observed].sum(axis=1), 1.0, atol=1e-7
    ):
        raise RuntimeError("축소한 사용자 카테고리 비중의 합이 1이 아닙니다")

    centered_share = shrunken_share - population_share[None, :]
    category_n = q_n[:, None] * centered_share
    category_n[~valid] = 0.0
    raw_share = np.divide(
        category_mass,
        basket_count[:, None],
        out=np.zeros_like(category_mass),
        where=basket_count[:, None] > 0.0,
    )
    uncentered_qn = q_n[:, None] * shrunken_share

    diagnostics: dict[str, float | int | str] = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_categories": int(n_categories),
        "n_transactions": int(round(n_transactions)),
        "valid_user_count": int(valid.sum()),
        "shrinkage_strength": float(shrinkage_strength),
        "transaction_mass_rule": (
            "one per basket, equally divided over distinct categories"
        ),
        "category_n_definition": (
            "q_N * (shrunken user basket-category share - population share)"
        ),
        "raw_category_share_row_sum_max_error": float(
            np.abs(raw_share[observed].sum(axis=1) - 1.0).max()
            if observed.any()
            else 0.0
        ),
        "uncentered_category_n_sum_max_error": float(
            np.abs(uncentered_qn[valid].sum(axis=1) - q_n[valid]).max()
            if valid.any()
            else 0.0
        ),
        "centered_category_n_row_sum_max_abs": float(
            np.abs(category_n[valid].sum(axis=1)).max() if valid.any() else 0.0
        ),
        "category_n_nonzero_share": float(np.mean(category_n != 0.0)),
        "category_n_mean_abs": float(np.abs(category_n[valid]).mean())
        if valid.any()
        else 0.0,
        "category_n_max_abs": float(np.abs(category_n[valid]).max())
        if valid.any()
        else 0.0,
        "population_category_share_max": float(population_share.max()),
    }
    return CategoryFrequencyFeatures(
        user_category_n_residual=category_n.astype(np.float32),
        item_category=item_category,
        user_valid=valid,
        diagnostics=diagnostics,
    )


class M5CategoryFrequencyValueLightGCN(nn.Module):
    """Joint q_V basis and category-allocated q_N representation plus ID."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_categories: int,
        user_q_v: np.ndarray,
        user_clv_valid: np.ndarray,
        item_price_percentile: np.ndarray,
        item_price_valid: np.ndarray,
        category_features: CategoryFrequencyFeatures,
        adj: torch.Tensor,
        id_dim: int = 64,
        category_dim: int = 4,
        rho_value: float = 0.05,
        rho_category_n: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        basis_bandwidth: float = 0.25,
    ):
        super().__init__()
        if min(n_users, n_items, n_categories, id_dim, category_dim) <= 0:
            raise ValueError("사용자·상품·카테고리·임베딩 차원은 양수여야 합니다")
        if not 0.0 <= rho_value <= 1.0 or not 0.0 <= rho_category_n <= 1.0:
            raise ValueError("rho_value와 rho_category_n은 [0,1]이어야 합니다")
        if n_layers < 0 or pref_reg < 0:
            raise ValueError("n_layers 또는 pref_reg가 잘못됐습니다")
        if adj.layout != torch.sparse_coo:
            raise ValueError("adj는 sparse COO tensor여야 합니다")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape이 사용자·상품 수와 다릅니다")

        user_q_v = np.asarray(user_q_v, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_price = np.asarray(item_price_percentile, dtype=np.float32)
        item_valid = np.asarray(item_price_valid, dtype=bool)
        category_n = np.asarray(
            category_features.user_category_n_residual, dtype=np.float32
        )
        item_category = np.asarray(category_features.item_category, dtype=np.int64)
        expected = {
            "user_q_v": (user_q_v, (n_users,)),
            "user_clv_valid": (user_valid, (n_users,)),
            "item_price_percentile": (item_price, (n_items,)),
            "item_price_valid": (item_valid, (n_items,)),
            "user_category_n_residual": (category_n, (n_users, n_categories)),
            "item_category": (item_category, (n_items,)),
        }
        for name, (values, shape) in expected.items():
            if values.shape != shape:
                raise ValueError(f"{name} shape이 잘못됐습니다")
        if not all(
            np.isfinite(values).all()
            for values in (user_q_v, item_price, category_n)
        ):
            raise ValueError("표현 입력은 모두 유한해야 합니다")
        if item_category.min(initial=0) < 0 or item_category.max(initial=-1) >= n_categories:
            raise ValueError("상품 카테고리가 범위를 벗어났습니다")

        user_value = fixed_value_basis(
            user_q_v, user_valid, bandwidth=basis_bandwidth
        )
        item_value = fixed_value_basis(
            item_price, item_valid, bandwidth=basis_bandwidth
        )

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_categories = int(n_categories)
        self.id_dim = int(id_dim)
        self.value_dim = 3
        self.category_dim = int(category_dim)
        self.rho_value = float(rho_value)
        self.rho_category_n = float(rho_category_n)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.basis_bandwidth = float(basis_bandwidth)
        self.feature_diagnostics = dict(category_features.diagnostics)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.category_basis = nn.Embedding(n_categories, category_dim)
        nn.init.normal_(self.category_basis.weight, std=0.1)

        self.register_buffer(
            "user_value_basis", torch.from_numpy(user_value), persistent=False
        )
        self.register_buffer(
            "item_value_basis", torch.from_numpy(item_value), persistent=False
        )
        self.register_buffer(
            "user_category_n_residual", torch.from_numpy(category_n), persistent=False
        )
        self.register_buffer(
            "item_category", torch.from_numpy(item_category), persistent=False
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.value_dim + self.category_dim

    def category_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        basis = F.normalize(self.category_basis.weight, p=2, dim=1, eps=1e-12)
        user = self.user_category_n_residual @ basis
        item = basis[self.item_category]
        return user, item

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_category, item_category = self.category_coordinates()
        value_scale = math.sqrt(self.rho_value)
        category_scale = math.sqrt(self.rho_category_n)
        return (
            torch.cat(
                [
                    self.E_u.weight,
                    value_scale * self.user_value_basis,
                    category_scale * user_category,
                ],
                dim=1,
            ),
            torch.cat(
                [
                    self.E_i.weight,
                    value_scale * self.item_value_basis,
                    category_scale * item_category,
                ],
                dim=1,
            ),
        )

    def _propagate(
        self, user: torch.Tensor, item: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(self.E_u.weight, self.E_i.weight)

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate(*self.layer0_embeddings())

    def embeddings(self, need_value: bool = True):
        user, item = self.propagated_embeddings()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def candidate_score_components(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        user, item = self.propagated_embeddings()
        selected_user, selected_item = user[users], item[items]
        value_start = self.id_dim
        category_start = value_start + self.value_dim
        id_score = (
            selected_user[:, :value_start] * selected_item[:, :value_start]
        ).sum(dim=1)
        value_score = (
            selected_user[:, value_start:category_start]
            * selected_item[:, value_start:category_start]
        ).sum(dim=1)
        category_n_score = (
            selected_user[:, category_start:] * selected_item[:, category_start:]
        ).sum(dim=1)
        economic_score = value_score + category_n_score
        return {
            "id": id_score,
            "value": value_score,
            "category_n": category_n_score,
            "economic": economic_score,
            "full": id_score + economic_score,
        }

    def sampled_l2(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        if negatives.ndim != 2:
            raise ValueError("negatives는 [batch,K] shape이어야 합니다")
        batch = len(users)
        negative_mean = (
            self.E_i.weight[negatives].pow(2).sum(dim=2).mean(dim=1).sum()
        )
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + negative_mean
        ) / batch

    @torch.no_grad()
    def training_gradient_diagnostics(self) -> dict[str, float]:
        def norm(parameter: torch.Tensor) -> float:
            return 0.0 if parameter.grad is None else float(parameter.grad.norm())

        return {
            "id_user_gradient_norm": norm(self.E_u.weight),
            "id_item_gradient_norm": norm(self.E_i.weight),
            "category_basis_gradient_norm": norm(self.category_basis.weight),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        user_category, item_category = self.category_coordinates()
        return {
            "rho": self.rho_value,
            "rho_value": self.rho_value,
            "rho_category_n": self.rho_category_n,
            "id_dim": self.id_dim,
            "value_dim": self.value_dim,
            "category_dim": self.category_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "explicit_q_n_in_m2": self.rho_category_n > 0.0,
            "explicit_q_v_in_m2": True,
            "q_c_in_m2": False,
            "item_n_or_item_clv_input": False,
            "item_side_category_basis": True,
            "n_role": (
                "q_N allocated over user basket-category shares, population-centered"
                if self.rho_category_n > 0.0
                else "category-specific N representation off"
            ),
            "v_role": "user transaction-value position versus item price position",
            "value_basis": "fixed normalized Gaussian RBF at 0.0, 0.5, 1.0",
            "basis_bandwidth": self.basis_bandwidth,
            "category_basis_row_norm_mean": float(
                item_category.new_ones(self.n_categories).mean()
            ),
            "user_category_coordinate_std": float(
                user_category.std(unbiased=False)
            ),
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "external_reranking": False,
            **self.feature_diagnostics,
        }
