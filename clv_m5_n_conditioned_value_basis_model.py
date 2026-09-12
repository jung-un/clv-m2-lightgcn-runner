"""N-conditioned value-position basis inside one binary LightGCN."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def fixed_value_basis(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    bandwidth: float = 0.25,
) -> np.ndarray:
    """Map a [0,1] value percentile to a fixed low/mid/high RBF basis."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("value percentile과 valid mask shape이 다릅니다")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("value percentile은 유한한 [0,1] 값이어야 합니다")
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise ValueError("basis bandwidth는 양수여야 합니다")

    centers = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    basis = np.exp(
        -0.5 * ((values[:, None] - centers[None, :]) / bandwidth) ** 2
    )
    norm = np.linalg.norm(basis, axis=1, keepdims=True)
    basis = np.divide(basis, norm, out=np.zeros_like(basis), where=norm > 0.0)
    basis[~valid] = 0.0
    return basis.astype(np.float32)


class M5NConditionedValueBasisLightGCN(nn.Module):
    """Use q_N as a bounded gate over q_V--item-price value matching.

    q_N remains the historical purchase-frequency component. It does not
    receive an item-side N coordinate and therefore cannot directly promote
    repeat-prone products. q_V and item price percentiles share a fixed
    three-dimensional value-position basis. ID and value coordinates are
    propagated together by the same binary LightGCN.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_q_n: np.ndarray,
        user_q_v: np.ndarray,
        user_clv_valid: np.ndarray,
        item_price_percentile: np.ndarray,
        item_price_valid: np.ndarray,
        adj: torch.Tensor,
        id_dim: int = 64,
        rho: float = 0.05,
        n_layers: int = 2,
        pref_reg: float = 1e-3,
        gate_delta: float = 0.25,
        basis_bandwidth: float = 0.25,
        constant_gate: float | None = None,
    ):
        super().__init__()
        if min(n_users, n_items, id_dim) <= 0:
            raise ValueError("사용자·상품·ID 차원은 양수여야 합니다")
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho는 [0,1]이어야 합니다")
        if n_layers < 0 or pref_reg < 0 or not 0.0 <= gate_delta < 1.0:
            raise ValueError("n_layers, pref_reg 또는 gate_delta가 잘못됐습니다")
        if constant_gate is not None and not (
            1.0 - gate_delta <= float(constant_gate) <= 1.0 + gate_delta
        ):
            raise ValueError("constant_gate는 기존 gate 허용범위 안이어야 합니다")
        if adj.layout != torch.sparse_coo:
            raise ValueError("adj는 sparse COO tensor여야 합니다")
        if tuple(adj.shape) != (n_users + n_items, n_users + n_items):
            raise ValueError("adj shape이 사용자·상품 수와 다릅니다")

        q_n = np.asarray(user_q_n, dtype=np.float32)
        q_v = np.asarray(user_q_v, dtype=np.float32)
        user_valid = np.asarray(user_clv_valid, dtype=bool)
        item_price = np.asarray(item_price_percentile, dtype=np.float32)
        item_valid = np.asarray(item_price_valid, dtype=bool)
        for name, values, expected in (
            ("user_q_n", q_n, (n_users,)),
            ("user_q_v", q_v, (n_users,)),
            ("user_clv_valid", user_valid, (n_users,)),
            ("item_price_percentile", item_price, (n_items,)),
            ("item_price_valid", item_valid, (n_items,)),
        ):
            if values.shape != expected:
                raise ValueError(f"{name} shape이 잘못됐습니다")
        if not np.isfinite(q_n).all() or not np.isfinite(q_v).all():
            raise ValueError("q_N·q_V는 모두 유한해야 합니다")
        if np.any((q_n < 0.0) | (q_n > 1.0)) or np.any(
            (q_v < 0.0) | (q_v > 1.0)
        ):
            raise ValueError("q_N·q_V 범위는 [0,1]이어야 합니다")

        user_basis = fixed_value_basis(
            q_v, user_valid, bandwidth=basis_bandwidth
        )
        item_basis = fixed_value_basis(
            item_price, item_valid, bandwidth=basis_bandwidth
        )

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.id_dim = int(id_dim)
        self.economic_dim = 3
        self.rho = float(rho)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        self.gate_delta = float(gate_delta)
        self.basis_bandwidth = float(basis_bandwidth)
        self.constant_gate = (
            None if constant_gate is None else float(constant_gate)
        )

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)
        if self.constant_gate is None:
            self.gate_offset_parameter = nn.Parameter(torch.zeros(()))
            self.gate_slope_parameter = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer(
                "gate_offset_parameter", torch.zeros(()), persistent=False
            )
            self.register_buffer(
                "gate_slope_parameter", torch.zeros(()), persistent=False
            )

        self.register_buffer(
            "user_q_n_centered", torch.from_numpy(2.0 * q_n - 1.0), persistent=False
        )
        self.register_buffer(
            "user_clv_valid",
            torch.from_numpy(user_valid.astype(np.float32)),
            persistent=False,
        )
        self.register_buffer(
            "user_value_basis", torch.from_numpy(user_basis), persistent=False
        )
        self.register_buffer(
            "item_value_basis", torch.from_numpy(item_basis), persistent=False
        )
        self.register_buffer("adj", adj.coalesce(), persistent=False)

    @property
    def total_dim(self) -> int:
        return self.id_dim + self.economic_dim

    def n_gate(self) -> torch.Tensor:
        if self.constant_gate is not None:
            return self.constant_gate * self.user_clv_valid
        raw = self.gate_offset_parameter + (
            self.gate_slope_parameter * self.user_q_n_centered
        )
        gate = 1.0 + self.gate_delta * torch.tanh(raw)
        return gate * self.user_clv_valid

    def economic_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.n_gate()[:, None] * self.user_value_basis, self.item_value_basis

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_value, item_value = self.economic_coordinates()
        scale = math.sqrt(self.rho)
        return (
            torch.cat([self.E_u.weight, scale * user_value], dim=1),
            torch.cat([self.E_i.weight, scale * item_value], dim=1),
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
            "n_gate_offset_gradient_norm": norm(self.gate_offset_parameter),
            "n_gate_slope_gradient_norm": norm(self.gate_slope_parameter),
        }

    @torch.no_grad()
    def representation_diagnostics(self) -> dict[str, float | int | bool | str]:
        gate = self.n_gate()
        valid = self.user_clv_valid > 0.0
        valid_gate = gate[valid]
        return {
            "rho": self.rho,
            "id_dim": self.id_dim,
            "economic_dim": self.economic_dim,
            "total_dim": self.total_dim,
            "n_layers": self.n_layers,
            "explicit_q_n_in_m2": self.constant_gate is None,
            "explicit_q_v_in_m2": True,
            "q_c_in_m2": False,
            "item_n_or_item_clv_input": False,
            "n_role": (
                "bounded strength of the user value-position basis"
                if self.constant_gate is None
                else "constant-gate q_V-only ablation"
            ),
            "v_role": "user transaction-value position versus item price position",
            "basis": "fixed normalized Gaussian RBF at 0.0, 0.5, 1.0",
            "basis_bandwidth": self.basis_bandwidth,
            "semantic_axis_rotation": False,
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "external_reranking": False,
            "n_gate_mode": (
                "learned_q_n_conditioned"
                if self.constant_gate is None
                else "fixed_constant"
            ),
            "constant_gate": self.constant_gate,
            "n_gate_offset": float(self.gate_offset_parameter),
            "n_gate_slope": float(self.gate_slope_parameter),
            "n_gate_mean": float(valid_gate.mean()) if len(valid_gate) else 0.0,
            "n_gate_std": float(valid_gate.std(unbiased=False)) if len(valid_gate) else 0.0,
            "n_gate_min": float(valid_gate.min()) if len(valid_gate) else 0.0,
            "n_gate_max": float(valid_gate.max()) if len(valid_gate) else 0.0,
        }
