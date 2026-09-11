"""Two-axis N/V representation propagated inside one binary LightGCN."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class M5MinimalNVEconomicLightGCN(nn.Module):
    """Add one frequency coordinate and one value coordinate to ID embeddings."""

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_n_centered: np.ndarray,
        user_v_centered: np.ndarray,
        user_clv_valid: np.ndarray,
        item_repeat_centered: np.ndarray,
        item_price_centered: np.ndarray,
        item_semantic_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        scale_delta: float = 0.25,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 [0,1]이어야 합니다")
        if n_layers < 0 or pref_reg < 0 or not 0.0 <= scale_delta < 1.0:
            raise ValueError("n_layers, pref_reg 또는 scale_delta가 잘못됐습니다")
        if adj.layout != torch.sparse_coo:
            raise ValueError("adj는 sparse COO tensor여야 합니다")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape이 사용자·상품 수와 다릅니다")

        user_n = np.asarray(user_n_centered, dtype=np.float32)
        user_v = np.asarray(user_v_centered, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_repeat = np.asarray(item_repeat_centered, dtype=np.float32)
        item_price = np.asarray(item_price_centered, dtype=np.float32)
        item_valid = np.asarray(item_semantic_valid, dtype=bool)
        for name, values, expected in (
            ("user_n_centered", user_n, (n_users,)),
            ("user_v_centered", user_v, (n_users,)),
            ("user_clv_valid", user_valid, (n_users,)),
            ("item_repeat_centered", item_repeat, (n_items,)),
            ("item_price_centered", item_price, (n_items,)),
            ("item_semantic_valid", item_valid, (n_items,)),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape이 잘못됐습니다")
        numeric = (user_n, user_v, item_repeat, item_price)
        if not all(np.isfinite(values).all() for values in numeric):
            raise ValueError("N/V 의미축 입력은 모두 유한해야 합니다")
        if any(np.any((values < -1.0) | (values > 1.0)) for values in numeric):
            raise ValueError("중심화된 N/V 의미축 입력은 [-1,1]이어야 합니다")
        if np.any(user_n[~user_valid] != 0.0) or np.any(
            user_v[~user_valid] != 0.0
        ):
            raise ValueError("CLV invalid 사용자의 N/V 입력은 0이어야 합니다")
        if np.any(item_repeat[~item_valid] != 0.0) or np.any(
            item_price[~item_valid] != 0.0
        ):
            raise ValueError("invalid 상품의 의미축 입력은 0이어야 합니다")

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = 2
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.scale_delta = float(scale_delta)

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        self.n_scale_parameter = nn.Parameter(torch.zeros(()))
        self.v_scale_parameter = nn.Parameter(torch.zeros(()))

        self.register_buffer(
            "user_n_centered", torch.from_numpy(user_n.copy()), persistent=False
        )
        self.register_buffer(
            "user_v_centered", torch.from_numpy(user_v.copy()), persistent=False
        )
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "item_repeat_centered",
            torch.from_numpy(item_repeat.copy()),
            persistent=False,
        )
        self.register_buffer(
            "item_price_centered",
            torch.from_numpy(item_price.copy()),
            persistent=False,
        )
        self.register_buffer(
            "item_semantic_valid",
            torch.from_numpy(item_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.economic_dim

    def n_scale(self) -> torch.Tensor:
        return 1.0 + self.scale_delta * torch.tanh(self.n_scale_parameter)

    def v_scale(self) -> torch.Tensor:
        return 1.0 + self.scale_delta * torch.tanh(self.v_scale_parameter)

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        n_calibration = torch.sqrt(self.n_scale())
        v_calibration = torch.sqrt(self.v_scale())
        user = torch.stack(
            [
                n_calibration * self.user_n_centered,
                v_calibration * self.user_v_centered,
            ],
            dim=1,
        )
        item = torch.stack(
            [
                n_calibration * self.item_repeat_centered,
                v_calibration * self.item_price_centered,
            ],
            dim=1,
        )
        return (
            user * self.user_clv_valid[:, None],
            item * self.item_semantic_valid[:, None],
        )

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_semantic, item_semantic = self.economic_coordinates()
        scale = math.sqrt(self.rho)
        return (
            torch.cat([self.E_u.weight, scale * user_semantic], dim=1),
            torch.cat([self.E_i.weight, scale * item_semantic], dim=1),
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
        id_score = (
            selected_user[:, : self.id_dim] * selected_item[:, : self.id_dim]
        ).sum(dim=1)
        economic_score = (
            selected_user[:, self.id_dim :] * selected_item[:, self.id_dim :]
        ).sum(dim=1)
        return {
            "id": id_score,
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
            "n_scale_gradient_norm": norm(self.n_scale_parameter),
            "v_scale_gradient_norm": norm(self.v_scale_parameter),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        user, item = self.economic_coordinates()
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "explicit_q_n_in_m2": True,
            "explicit_q_v_in_m2": True,
            "q_c_in_m2": False,
            "n_role": "user frequency versus item repeat-purchase propensity",
            "v_role": "user transaction value versus item amount percentile",
            "semantic_axis_rotation": False,
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "external_reranking": False,
            "n_scale": float(self.n_scale()),
            "v_scale": float(self.v_scale()),
            "user_semantic_mean_norm": float(user.norm(dim=1).mean()),
            "item_semantic_mean_norm": float(item.norm(dim=1).mean()),
        }
