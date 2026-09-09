"""Meaning-preserving N/V economic representation for a compact M5."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class M5SemanticNVEconomicLightGCN(nn.Module):
    """LightGCN ID embeddings plus one unpropagated semantic CLV block.

    The first auxiliary coordinate is the signed q_V--item-price relation.
    The remaining coordinates compare the user's shrunken spending profile
    with the candidate item's centered price-bin basis. q_N scales only that
    detailed profile block. Two positive bounded scalars are the only learned
    parameters in the auxiliary representation, so its axes cannot rotate.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_q_n: np.ndarray,
        user_q_v_centered: np.ndarray,
        user_centered_profile: np.ndarray,
        user_economic_valid: np.ndarray,
        item_price_centered: np.ndarray,
        item_centered_bin: np.ndarray,
        item_economic_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.15,
        beta: float = 0.25,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        scale_delta: float = 0.25,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 임베딩 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0 or not 0.0 < beta < 1.0:
            raise ValueError("rho는 [0,1], beta는 (0,1)이어야 합니다")
        if n_layers < 0 or pref_reg < 0 or not 0.0 <= scale_delta < 1.0:
            raise ValueError("n_layers, pref_reg 또는 scale_delta 설정이 잘못됐습니다")

        q_n = np.asarray(user_q_n, dtype=np.float32)
        q_v = np.asarray(user_q_v_centered, dtype=np.float32)
        user_profile = np.asarray(user_centered_profile, dtype=np.float32)
        item_price = np.asarray(item_price_centered, dtype=np.float32)
        item_bin = np.asarray(item_centered_bin, dtype=np.float32)
        user_valid = np.asarray(user_economic_valid, dtype=bool)
        item_valid = np.asarray(item_economic_valid, dtype=bool)
        if q_n.shape != (n_users,) or q_v.shape != (n_users,):
            raise ValueError("사용자 q_N·q_V shape이 잘못됐습니다")
        if user_profile.ndim != 2 or user_profile.shape[0] != n_users:
            raise ValueError("사용자 가격구간 profile shape이 잘못됐습니다")
        if item_price.shape != (n_items,):
            raise ValueError("상품 가격축 shape이 잘못됐습니다")
        if item_bin.shape != (n_items, user_profile.shape[1]):
            raise ValueError("상품 가격구간 기저 shape이 잘못됐습니다")
        if user_valid.shape != (n_users,) or item_valid.shape != (n_items,):
            raise ValueError("경제입력 valid mask shape이 잘못됐습니다")
        arrays = (q_n, q_v, user_profile, item_price, item_bin)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("N/V·경제입력은 모두 유한해야 합니다")
        if np.any((q_n < 0.0) | (q_n > 1.0)):
            raise ValueError("q_N 범위는 [0,1]이어야 합니다")
        if np.any((q_v < -1.0) | (q_v > 1.0)):
            raise ValueError("중심화 q_V 범위는 [-1,1]이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = 1 + int(user_profile.shape[1])
        self.rho = float(rho)
        self.beta = float(beta)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.scale_delta = float(scale_delta)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.value_scale_parameter = nn.Parameter(torch.zeros(()))
        self.profile_scale_parameter = nn.Parameter(torch.zeros(()))

        self.register_buffer("user_q_n", torch.from_numpy(q_n.copy()), persistent=False)
        self.register_buffer(
            "user_q_v_centered", torch.from_numpy(q_v.copy()), persistent=False
        )
        self.register_buffer(
            "user_centered_profile",
            torch.from_numpy(user_profile.copy()),
            persistent=False,
        )
        self.register_buffer(
            "item_price_centered", torch.from_numpy(item_price.copy()), persistent=False
        )
        self.register_buffer(
            "item_centered_bin", torch.from_numpy(item_bin.copy()), persistent=False
        )
        self.register_buffer(
            "user_economic_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_economic_valid",
            torch.from_numpy(item_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.economic_dim

    def value_scale(self) -> torch.Tensor:
        return 1.0 + self.scale_delta * torch.tanh(self.value_scale_parameter)

    def profile_scale(self) -> torch.Tensor:
        return 1.0 + self.scale_delta * torch.tanh(self.profile_scale_parameter)

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        value_budget = math.sqrt(self.beta)
        profile_budget = math.sqrt(1.0 - self.beta)
        value_calibration = torch.sqrt(self.value_scale())
        profile_calibration = torch.sqrt(self.profile_scale())
        user_value = (
            value_budget * value_calibration * self.user_q_v_centered[:, None]
        )
        item_value = (
            value_budget * value_calibration * self.item_price_centered[:, None]
        )
        user_profile = (
            profile_budget
            * profile_calibration
            * self.user_q_n[:, None]
            * self.user_centered_profile
        )
        item_profile = (
            profile_budget * profile_calibration * self.item_centered_bin
        )
        user = torch.cat([user_value, user_profile], dim=1)
        item = torch.cat([item_value, item_profile], dim=1)
        return (
            user * self.user_economic_valid[:, None],
            item * self.item_economic_valid[:, None],
        )

    def _propagate_id(self) -> tuple[torch.Tensor, torch.Tensor]:
        current = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def id_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._propagate_id()

    def propagated_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_id, item_id = self._propagate_id()
        user_economic, item_economic = self.economic_coordinates()
        scale = math.sqrt(self.rho)
        return (
            torch.cat([user_id, scale * user_economic], dim=1),
            torch.cat([item_id, scale * item_economic], dim=1),
        )

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
        id_score = (
            selected_user[:, : self.id_dim] * selected_item[:, : self.id_dim]
        ).sum(dim=1)
        economic_score = (
            selected_user[:, self.id_dim :] * selected_item[:, self.id_dim :]
        ).sum(dim=1)
        return {"id": id_score, "economic": economic_score, "full": id_score + economic_score}

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
            "value_scale_gradient_norm": norm(self.value_scale_parameter),
            "profile_scale_gradient_norm": norm(self.profile_scale_parameter),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        user, item = self.economic_coordinates()
        return {
            "rho": self.rho,
            "beta": self.beta,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "explicit_q_n_in_m2": True,
            "explicit_q_v_in_m2": True,
            "q_c_in_m2": False,
            "semantic_axis_rotation": False,
            "economic_graph_propagation": False,
            "joint_end_to_end_training": True,
            "external_reranking": False,
            "q_n_role": "personalized_price_bin_profile_strength",
            "q_v_role": "signed_overall_price_direction",
            "value_scale": float(self.value_scale()),
            "profile_scale": float(self.profile_scale()),
            "user_economic_mean_norm": float(user.norm(dim=1).mean()),
            "item_economic_mean_norm": float(item.norm(dim=1).mean()),
        }
