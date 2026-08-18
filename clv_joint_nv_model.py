"""Jointly trained CLV N/V subspaces inside a single LightGCN.

The model changes only M2 (the layer-0 representation and gradient routing
between its ID and N/V blocks).  The binary graph, uniform negative sampling
and pairwise BPR ranking objective remain unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from clv_dual_axis_model import DualItemProfile, apply_gate_shape


VARIANTS = frozenset({"joint_nv", "joint_shuffled_user", "joint_constant_user"})


class _AxisEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(values), dim=1, eps=1e-8)


class JointNVLightGCN(nn.Module):
    """ID/N/V layer-0 blocks propagated together by one LightGCN.

    ``sqrt_gamma_n`` and ``sqrt_gamma_v`` are learned directly.  The same
    parameter scales the user and item side of each axis, so the reported
    non-negative score-level strength is ``gamma = sqrt_gamma ** 2``.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        user_activity: np.ndarray,
        user_value: np.ndarray,
        user_activity_valid: np.ndarray | None = None,
        user_value_valid: np.ndarray | None = None,
        item_profile: DualItemProfile,
        q_n: np.ndarray,
        q_v: np.ndarray,
        adj: torch.Tensor,
        id_dim: int,
        axis_dim: int,
        hidden_dim: int,
        n_layers: int,
        variant: str = "joint_nv",
        gate_shape: str = "equal",
        shuffle_seed: int = 42,
        pref_reg: float = 1e-4,
        gamma_init: float = 0.01,
        anchor_weight: float = 0.0,
        preference_preserving: bool = False,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"지원하지 않는 variant: {variant}")
        if n_users <= 0 or n_items <= 0 or axis_dim <= 0 or id_dim <= 0:
            raise ValueError("사용자·아이템·embedding 크기는 양수여야 합니다")
        if n_layers < 0:
            raise ValueError("n_layers는 0 이상이어야 합니다")
        if not 0.0 < gamma_init < 1.0:
            raise ValueError("gamma_init은 0과 1 사이여야 합니다")

        user_activity = np.asarray(user_activity, dtype=np.float32)
        user_value = np.asarray(user_value, dtype=np.float32)
        user_activity_valid = np.asarray(
            np.ones(n_users, bool)
            if user_activity_valid is None
            else user_activity_valid,
            dtype=bool,
        )
        user_value_valid = np.asarray(
            np.ones(n_users, bool) if user_value_valid is None else user_value_valid,
            dtype=bool,
        )
        q_n = np.asarray(q_n, dtype=np.float32)
        q_v = np.asarray(q_v, dtype=np.float32)
        if user_activity.shape[0] != n_users or user_value.shape[0] != n_users:
            raise ValueError("사용자 특징의 행 수가 n_users와 다릅니다")
        if q_n.shape != (n_users,) or q_v.shape != (n_users,):
            raise ValueError("q_n/q_v shape이 n_users와 다릅니다")
        if user_activity_valid.shape != (n_users,) or user_value_valid.shape != (n_users,):
            raise ValueError("사용자 N/V 유효성 마스크 shape이 n_users와 다릅니다")
        if item_profile.activity.shape[0] != n_items or item_profile.value.shape[0] != n_items:
            raise ValueError("아이템 특징의 행 수가 n_items와 다릅니다")
        arrays = (user_activity, user_value, item_profile.activity, item_profile.value, q_n, q_v)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("Joint N/V 입력은 모두 유한해야 합니다")

        if variant == "joint_shuffled_user":
            permutation = np.random.default_rng(shuffle_seed).permutation(n_users)
            user_activity = user_activity[permutation]
            user_value = user_value[permutation]
            user_activity_valid = user_activity_valid[permutation]
            user_value_valid = user_value_valid[permutation]
            q_n = q_n[permutation]
            q_v = q_v[permutation]
        elif variant == "joint_constant_user":
            user_activity = np.broadcast_to(user_activity.mean(0, keepdims=True), user_activity.shape).copy()
            user_value = np.broadcast_to(user_value.mean(0, keepdims=True), user_value.shape).copy()
            q_n = np.full(n_users, 0.5, np.float32)
            q_v = np.full(n_users, 0.5, np.float32)
            user_activity_valid = np.ones(n_users, bool)
            user_value_valid = np.ones(n_users, bool)

        gate_n = apply_gate_shape(q_n, gate_shape, user_activity_valid)
        gate_v = apply_gate_shape(q_v, gate_shape, user_value_valid)
        if variant == "joint_constant_user":
            gate_n = np.ones(n_users, np.float32)
            gate_v = np.ones(n_users, np.float32)

        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.n_layers = int(n_layers)
        self.pref_reg = float(pref_reg)
        if not 0.0 <= anchor_weight <= 1.0:
            raise ValueError("anchor_weight는 0과 1 사이여야 합니다")
        if preference_preserving and anchor_weight > 0.0:
            raise ValueError("preference-preserving과 anchored 목적함수를 동시에 쓸 수 없습니다")
        self.anchor_weight = float(anchor_weight)
        self.preference_preserving = bool(preference_preserving)
        self.id_dim = int(id_dim)
        self.axis_dim = int(axis_dim)
        self.variant = variant
        self.gate_shape = gate_shape

        self.E_u = nn.Embedding(n_users, id_dim)
        self.E_i = nn.Embedding(n_items, id_dim)
        nn.init.normal_(self.E_u.weight, std=0.1)
        nn.init.normal_(self.E_i.weight, std=0.1)

        self.activity_user = _AxisEncoder(user_activity.shape[1], hidden_dim, axis_dim)
        self.activity_item = _AxisEncoder(item_profile.activity.shape[1], hidden_dim, axis_dim)
        self.value_user = _AxisEncoder(user_value.shape[1], hidden_dim, axis_dim)
        self.value_item = _AxisEncoder(item_profile.value.shape[1], hidden_dim, axis_dim)
        sqrt_gamma_init = float(np.sqrt(gamma_init))
        self.sqrt_gamma_n = nn.Parameter(torch.tensor(sqrt_gamma_init))
        self.sqrt_gamma_v = nn.Parameter(torch.tensor(sqrt_gamma_init))

        self.register_buffer("user_activity", torch.from_numpy(user_activity.copy()))
        self.register_buffer("user_value", torch.from_numpy(user_value.copy()))
        self.register_buffer(
            "user_activity_valid",
            torch.from_numpy(user_activity_valid.astype(np.float32)),
        )
        self.register_buffer(
            "user_value_valid", torch.from_numpy(user_value_valid.astype(np.float32))
        )
        self.register_buffer("item_activity", torch.from_numpy(item_profile.activity.copy()))
        self.register_buffer("item_value", torch.from_numpy(item_profile.value.copy()))
        self.register_buffer("valid_item", torch.from_numpy(item_profile.valid_item.astype(np.float32)))
        self.register_buffer("gate_n", torch.from_numpy(gate_n.copy()))
        self.register_buffer("gate_v", torch.from_numpy(gate_v.copy()))
        self.register_buffer("adj", adj.coalesce())

    @property
    def gamma_n(self) -> torch.Tensor:
        return self.sqrt_gamma_n.square()

    @property
    def gamma_v(self) -> torch.Tensor:
        return self.sqrt_gamma_v.square()

    def layer0_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        user_n = (
            self.activity_user(self.user_activity) * self.user_activity_valid[:, None]
        )
        item_n = self.activity_item(self.item_activity) * self.valid_item[:, None]
        user_v = self.value_user(self.user_value) * self.user_value_valid[:, None]
        item_v = self.value_item(self.item_value) * self.valid_item[:, None]
        scale_n = self.sqrt_gamma_n
        scale_v = self.sqrt_gamma_v
        user = torch.cat(
            [
                self.E_u.weight,
                scale_n * self.gate_n[:, None] * user_n,
                scale_v * self.gate_v[:, None] * user_v,
            ],
            dim=1,
        )
        item = torch.cat(
            [self.E_i.weight, scale_n * item_n, scale_v * item_v], dim=1
        )
        return user, item

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        user, item = self.layer0_embeddings()
        current = torch.cat([user, item], dim=0)
        total = current
        for _ in range(self.n_layers):
            current = torch.sparse.mm(self.adj, current)
            total = total + current
        total = total / (self.n_layers + 1)
        return total[: self.n_users], total[self.n_users :]

    def embeddings(self, need_value: bool = True):
        """Compatibility with the project's common evaluator.

        N/V are already inside the propagated embedding, so the external value
        slot is an exact zero and evaluation must use ``lambda=0``.
        """
        user, item = self.propagate()
        zero_user = user.new_zeros((self.n_users, 1))
        zero_item = item.new_zeros((self.n_items, 1))
        return user, item, zero_user, zero_item

    def batch_l2(self, users, positives, negatives, need_value: bool = False):
        if self.pref_reg <= 0:
            return self.E_u.weight.new_zeros(())
        return self.pref_reg * (
            self.E_u.weight[users].pow(2).sum()
            + self.E_i.weight[positives].pow(2).sum()
            + self.E_i.weight[negatives].pow(2).sum()
        ) / len(users)

    def bpr_loss(self, users, positives, negatives, gate=None, lam=0.0, weights=None):
        if weights is not None:
            raise ValueError("M2 joint model에 M4 표본 가중치를 넣을 수 없습니다")
        user, item = self.propagate()
        positive_score = (user[users] * item[positives]).sum(1)
        negative_score = (user[users] * item[negatives]).sum(1)
        bpr_full = -F.logsigmoid(positive_score - negative_score).mean()
        positive_id = (
            user[users, : self.id_dim] * item[positives, : self.id_dim]
        ).sum(1)
        negative_id = (
            user[users, : self.id_dim] * item[negatives, : self.id_dim]
        ).sum(1)
        id_margin = positive_id - negative_id
        bpr_id = -F.logsigmoid(id_margin).mean()
        bpr_residual = None
        if self.preference_preserving:
            positive_nv = (
                user[users, self.id_dim :] * item[positives, self.id_dim :]
            ).sum(1)
            negative_nv = (
                user[users, self.id_dim :] * item[negatives, self.id_dim :]
            ).sum(1)
            nv_margin = positive_nv - negative_nv
            bpr_residual = -F.logsigmoid(id_margin.detach() + nv_margin).mean()
            # ID receives exactly the ordinary ID-BPR gradient.  The second
            # term trains only N/V while conditioning on the current ID score.
            bpr = bpr_id + bpr_residual
            objective = "preference_preserving_joint"
        else:
            bpr = (
                (1.0 - self.anchor_weight) * bpr_full
                + self.anchor_weight * bpr_id
            )
            objective = "anchored" if self.anchor_weight > 0 else "plain"
        loss = bpr + self.batch_l2(users, positives, negatives)
        with torch.no_grad():
            diagnostics = {
                "bpr": float(bpr),
                "bpr_full": float(bpr_full),
                "bpr_id": float(bpr_id),
                "bpr_residual": (
                    float(bpr_residual) if bpr_residual is not None else None
                ),
                "objective": objective,
                "p_correct": float((positive_score > negative_score).float().mean()),
                "gamma_n": float(self.gamma_n),
                "gamma_v": float(self.gamma_v),
            }
        return loss, diagnostics

    @torch.no_grad()
    def score_diagnostics(self, n_sample: int = 512, seed: int = 0) -> dict:
        user, item = self.propagate()
        rng = np.random.default_rng(seed)
        sample = rng.choice(self.n_users, min(n_sample, self.n_users), replace=False)
        sample_t = torch.as_tensor(sample, dtype=torch.long, device=user.device)
        score = user[sample_t] @ item.T
        return {
            "gamma_n": float(self.gamma_n),
            "gamma_v": float(self.gamma_v),
            "score_std": float(score.std()),
            "mean_user_norm": float(user.norm(dim=1).mean()),
            "mean_item_norm": float(item.norm(dim=1).mean()),
            "gate_n_mean": float(self.gate_n.mean()),
            "gate_n_std": float(self.gate_n.std()),
            "gate_n_negative_share": float((self.gate_n < 0).float().mean()),
            "gate_n_positive_share": float((self.gate_n > 0).float().mean()),
            "gate_v_mean": float(self.gate_v.mean()),
            "gate_v_std": float(self.gate_v.std()),
            "gate_v_negative_share": float((self.gate_v < 0).float().mean()),
            "gate_v_positive_share": float((self.gate_v > 0).float().mean()),
        }
