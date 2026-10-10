"""Dataset-adaptive N/V representation for the M2 intervention.

The LightGCN graph and plain BPR objective stay identical to M1.  Historical
q_N/q_V and item economic attributes enter the same forward pass through the
existing reliability-gated orthogonal representation.  Only their relative
mixture and the total residual strength are dataset-level trainable parameters.
They are initialized from training-only support and dispersion, then updated
together with the ID embeddings and feature encoders by one optimizer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from clv_reliability_orthogonal_nv_model import ReliabilityOrthogonalNVLightGCN


AXIS_NAMES = ("n", "v", "price")
AXIS_FLOOR = 0.15
RHO_MIN = 0.03
RHO_MAX = 0.15


def _dispersion(values) -> float:
    values = np.asarray(values, np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("RBF3 특징 행렬이 필요합니다")
    return float(np.linalg.norm(values.std(axis=0)))


def dataset_parameter_initialization(features: dict) -> dict:
    """Derive a dataset-specific, train-only initialization.

    N and V each combine their user-side and item-buyer-profile dispersion.
    Price is paired with the user V dispersion because it is the candidate-side
    economic attribute.  Reliability discounts axes supported by fewer observed
    transactions.  All three axes remain strictly active.
    """

    required = {
        "user_n", "user_v", "item_n", "item_v", "item_price",
        "user_n_reliability", "user_v_reliability", "item_reliability",
    }
    missing = required.difference(features)
    if missing:
        raise ValueError(f"데이터별 초기화 입력이 없습니다: {sorted(missing)}")

    rel_n = float(np.asarray(features["user_n_reliability"], float).mean())
    rel_v = float(np.asarray(features["user_v_reliability"], float).mean())
    rel_i = float(np.asarray(features["item_reliability"], float).mean())
    if not all(np.isfinite(value) and 0 < value <= 1 for value in (rel_n, rel_v, rel_i)):
        raise ValueError("관측 지지도는 (0,1]의 유한한 값이어야 합니다")

    epsilon = 1e-8
    signals = np.asarray(
        [
            np.sqrt(max(_dispersion(features["user_n"]), epsilon)
                    * max(_dispersion(features["item_n"]), epsilon)
                    * rel_n * rel_i),
            np.sqrt(max(_dispersion(features["user_v"]), epsilon)
                    * max(_dispersion(features["item_v"]), epsilon)
                    * rel_v * rel_i),
            np.sqrt(max(_dispersion(features["user_v"]), epsilon)
                    * max(_dispersion(features["item_price"]), epsilon)
                    * rel_v * rel_i),
        ],
        dtype=np.float64,
    )
    probabilities = signals / signals.sum()
    weights = AXIS_FLOOR + (3.0 - 3.0 * AXIS_FLOOR) * probabilities
    support = float(np.sqrt(((rel_n + rel_v) / 2.0) * rel_i))
    rho = float(RHO_MIN + (RHO_MAX - RHO_MIN) * support)
    return {
        "axis_weights": dict(zip(AXIS_NAMES, map(float, weights), strict=True)),
        "rho": rho,
        "support_factor": support,
        "raw_axis_signal": dict(zip(AXIS_NAMES, map(float, signals), strict=True)),
        "source": "training-only feature dispersion and observed support",
    }


class DatasetAdaptiveOrthogonalNVLightGCN(ReliabilityOrthogonalNVLightGCN):
    """Joint M2 with trainable dataset-level N/V/price composition and strength."""

    def __init__(self, *, features, **kwargs):
        initialization = dataset_parameter_initialization(features)
        super().__init__(features=features, rho=initialization["rho"], **kwargs)
        self.dataset_initialization = initialization

        initial_weights = np.asarray(
            [initialization["axis_weights"][name] for name in AXIS_NAMES],
            dtype=np.float32,
        )
        probabilities = (initial_weights - AXIS_FLOOR) / (3.0 - 3.0 * AXIS_FLOOR)
        self.axis_logits = nn.Parameter(torch.from_numpy(np.log(probabilities)))
        rho_probability = (initialization["rho"] - RHO_MIN) / (RHO_MAX - RHO_MIN)
        rho_probability = float(np.clip(rho_probability, 1e-6, 1 - 1e-6))
        self.rho_logit = nn.Parameter(torch.tensor(np.log(rho_probability / (1 - rho_probability)),
                                                    dtype=torch.float32))

    def axis_weights(self):
        return AXIS_FLOOR + (3.0 - 3.0 * AXIS_FLOOR) * torch.softmax(self.axis_logits, dim=0)

    def adaptive_rho(self):
        return RHO_MIN + (RHO_MAX - RHO_MIN) * torch.sigmoid(self.rho_logit)

    def _user_feature(self, rows=None):
        rows = slice(None) if rows is None else rows
        weights = self.axis_weights()
        n = self.user_n_reliability[rows, None] * self.encoders["user_n"](
            self.user_n_input[rows]
        )
        v = self.user_v_reliability[rows, None] * self.encoders["user_v"](
            self.user_v_input[rows]
        )
        scale = torch.sqrt(weights[0].square() + weights[1].square())
        reliability = torch.sqrt(
            (self.user_n_reliability[rows].square()
             + self.user_v_reliability[rows].square()) / 2
        )
        return (weights[0] * n + weights[1] * v) / scale, reliability

    def _item_feature(self, rows=None, *, loo=False):
        rows = slice(None) if rows is None else rows
        prefix = "loo_item" if loo else "item"
        weights = self.axis_weights()
        n = self.encoders["item_n"](getattr(self, prefix + "_n_input")[rows])
        v = self.encoders["item_v"](getattr(self, prefix + "_v_input")[rows])
        price = self.encoders["item_price"](getattr(self, prefix + "_price_input")[rows])
        reliability = (self.loo_item_reliability if loo else self.item_reliability)[rows]
        scale = weights.square().sum().sqrt()
        feature = (weights[0] * n + weights[1] * v + weights[2] * price) / scale
        return reliability[..., None] * feature, reliability

    def _mix(self, identity, feature, reliability):
        identity_norm = identity.norm(dim=1, keepdim=True).clamp_min(1e-12)
        direction = identity / identity_norm
        orthogonal = feature - (feature * direction).sum(dim=1, keepdim=True) * direction
        orthogonal_norm = orthogonal.norm(dim=1, keepdim=True)
        unit = orthogonal / orthogonal_norm.clamp_min(1e-12)
        strength = self.adaptive_rho() * reliability[:, None]
        active = (orthogonal_norm > 1e-10).to(identity.dtype)
        return identity + active * strength * identity_norm * unit

    def batch_l2(self, users, positives, negatives):
        base = super().batch_l2(users, positives, negatives)
        adaptive = self.axis_logits.square().sum() + self.rho_logit.square()
        return base + self.shared_l2 * adaptive

    def training_gradient_diagnostics(self):
        diagnostics = super().training_gradient_diagnostics()
        weights = self.axis_weights().detach()
        diagnostics.update(
            axis_logits_last_batch_gradient_norm=(
                float(self.axis_logits.grad.norm()) if self.axis_logits.grad is not None else None
            ),
            rho_logit_last_batch_gradient_abs=(
                float(self.rho_logit.grad.abs()) if self.rho_logit.grad is not None else None
            ),
            axis_weight_n=float(weights[0]),
            axis_weight_v=float(weights[1]),
            axis_weight_price=float(weights[2]),
            adaptive_rho=float(self.adaptive_rho().detach()),
        )
        return diagnostics

    @torch.no_grad()
    def representation_diagnostics(self):
        diagnostics = super().representation_diagnostics()
        diagnostics.pop("rho_max", None)
        weights = self.axis_weights()
        diagnostics.update(
            adaptive_rho=float(self.adaptive_rho()),
            axis_weight_n=float(weights[0]),
            axis_weight_v=float(weights[1]),
            axis_weight_price=float(weights[2]),
            initial_rho=self.dataset_initialization["rho"],
            initial_axis_weights=self.dataset_initialization["axis_weights"],
            initialization_source=self.dataset_initialization["source"],
            adaptive_parameters=sum(
                parameter.numel() for parameter in (self.axis_logits, self.rho_logit)
            ),
        )
        return diagnostics
