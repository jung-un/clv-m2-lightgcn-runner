"""Literature-grounded two-axis CLV user representation.

The representation follows the non-contractual CLV decomposition used by
Pareto/NBD or BG/NBD plus a monetary-value model: expected future transaction
count times expected value per transaction.  It deliberately excludes general
recommendation-profile variables such as category entropy and premium share.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

import lightgcn_clv_residual as residual
from clv_moe_features import UserProfileArtifact


REPURCHASE_FEATURES = (
    "recency_days",
    "basket_count",
    "observed_days",
    "gap_mean",
)
MONETARY_FEATURES = ("avg_basket_value",)


@dataclass
class CLVCoreArtifact:
    model: nn.Module
    transform: residual.FeatureTransform
    best_epoch: int
    diagnostics: dict
    h_all: np.ndarray
    ev_all: np.ndarray
    n_hat_all: np.ndarray
    v_hat_all: np.ndarray


class CLVCoreEncoder(nn.Module):
    """Keep repeat-purchase and transaction-value representations separate."""

    def __init__(self, repurchase_input_dim: int, monetary_input_dim: int):
        super().__init__()
        self.repurchase = nn.Sequential(
            nn.Linear(repurchase_input_dim, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
        )
        self.monetary = nn.Sequential(
            nn.Linear(monetary_input_dim, 8),
            nn.GELU(),
            nn.Linear(8, 8),
            nn.GELU(),
        )
        self.transaction_head = nn.Linear(8, 1)
        self.value_head = nn.Linear(8, 1)

    def forward(self, repurchase_x, monetary_x):
        h_n = self.repurchase(repurchase_x)
        h_v = self.monetary(monetary_x)
        log_n = F.softplus(self.transaction_head(h_n).squeeze(-1))
        log_v = F.softplus(self.value_head(h_v).squeeze(-1))
        return h_n, h_v, log_n, log_v


def _axis_indices(names: Sequence[str]) -> list[int]:
    width = len(residual.NUMERIC_FEATURES)
    numeric = [residual.NUMERIC_FEATURES.index(name) for name in names]
    return numeric + [width + index for index in numeric]


def _axis_inputs(
    anchor: residual.AnchorExamples, transform: residual.FeatureTransform
) -> tuple[np.ndarray, np.ndarray]:
    transformed = residual.transform_features(anchor, transform)
    return (
        transformed[:, _axis_indices(REPURCHASE_FEATURES)],
        transformed[:, _axis_indices(MONETARY_FEATURES)],
    )


def _targets(anchor: residual.AnchorExamples) -> tuple[np.ndarray, np.ndarray]:
    if (
        anchor.transaction_target is None
        or anchor.mean_transaction_value_target is None
    ):
        raise ValueError("CLV-core anchor에 거래횟수·거래당 금액 target이 필요합니다")
    return anchor.transaction_target, anchor.mean_transaction_value_target


def _stack(
    anchors: Sequence[residual.AnchorExamples], transform: residual.FeatureTransform
):
    repurchase, monetary, count, value = [], [], [], []
    for anchor in anchors:
        x_n, x_v = _axis_inputs(anchor, transform)
        y_n, y_v = _targets(anchor)
        repurchase.append(x_n)
        monetary.append(x_v)
        count.append(y_n)
        value.append(y_v)
    return tuple(np.concatenate(parts) for parts in (repurchase, monetary, count, value))


def _loss(log_n, log_v, count, value):
    count_loss = F.huber_loss(log_n, torch.log1p(count))
    buyers = count > 0
    value_loss = (
        F.huber_loss(log_v[buyers], torch.log1p(value[buyers]))
        if buyers.any()
        else log_v.sum() * 0.0
    )
    return count_loss + value_loss


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_epochs(model, arrays, epochs, batch_size, lr, seed, device):
    x_n, x_v, y_n, y_v = arrays
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(y_n))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            batch = [
                torch.as_tensor(values[idx], dtype=torch.float32, device=device)
                for values in (x_n, x_v, y_n, y_v)
            ]
            _, _, log_n, log_v = model(batch[0], batch[1])
            loss = _loss(log_n, log_v, batch[2], batch[3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _evaluate_loss(model, arrays, device) -> float:
    tensors = [
        torch.as_tensor(values, dtype=torch.float32, device=device)
        for values in arrays
    ]
    model.eval()
    with torch.no_grad():
        _, _, log_n, log_v = model(tensors[0], tensors[1])
        return float(_loss(log_n, log_v, tensors[2], tensors[3]))


def _safe_spearman(x, y) -> float:
    value = spearmanr(x, y, nan_policy="omit").statistic
    return float(value) if np.isfinite(value) else float("nan")


def _diagnostics(model, anchor, transform, device) -> dict:
    x_n, x_v = _axis_inputs(anchor, transform)
    count, value = _targets(anchor)
    model.eval()
    with torch.no_grad():
        _, _, log_n, log_v = model(
            torch.as_tensor(x_n, dtype=torch.float32, device=device),
            torch.as_tensor(x_v, dtype=torch.float32, device=device),
        )
    n_hat = np.expm1(log_n.cpu().numpy())
    v_hat = np.expm1(log_v.cpu().numpy())
    amount = count * value
    buyers = count > 0
    return {
        "future_transaction_spearman": _safe_spearman(n_hat, count),
        "future_transaction_log_mae": float(
            np.abs(np.log1p(n_hat) - np.log1p(count)).mean()
        ),
        "transaction_value_spearman": (
            _safe_spearman(v_hat[buyers], value[buyers])
            if buyers.any()
            else float("nan")
        ),
        "transaction_value_log_mae": (
            float(np.abs(np.log1p(v_hat[buyers]) - np.log1p(value[buyers])).mean())
            if buyers.any()
            else float("nan")
        ),
        "clv_proxy_amount_spearman": _safe_spearman(n_hat * v_hat, amount),
    }


def train_clv_core_encoder(
    dataset: residual.AnchorDataset,
    final_snapshot: residual.AnchorExamples,
    *,
    encoder_epochs: int,
    encoder_patience: int,
    encoder_batch_size: int,
    encoder_lr: float,
    seed: int,
    device: torch.device,
) -> CLVCoreArtifact:
    if len(dataset.anchors) != 3:
        raise ValueError("CLV-core encoder는 학습 anchor 2개와 내부 validation 1개를 요구합니다")
    transform = residual.fit_feature_transform(dataset.anchors[:2])
    train_arrays = _stack(dataset.anchors[:2], transform)
    val_arrays = _stack(dataset.anchors[2:], transform)
    _seed(seed)
    model = CLVCoreEncoder(train_arrays[0].shape[1], train_arrays[1].shape[1]).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=encoder_lr)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_state, bad = float("inf"), 0, None, 0
    for epoch in range(1, encoder_epochs + 1):
        model.train()
        order = rng.permutation(len(train_arrays[2]))
        for start in range(0, len(order), encoder_batch_size):
            idx = order[start : start + encoder_batch_size]
            batch = [
                torch.as_tensor(values[idx], dtype=torch.float32, device=device)
                for values in train_arrays
            ]
            _, _, log_n, log_v = model(batch[0], batch[1])
            loss = _loss(log_n, log_v, batch[2], batch[3])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_loss = _evaluate_loss(model, val_arrays, device)
        if val_loss < best_loss - 1e-10:
            best_loss, best_epoch, bad = val_loss, epoch, 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            bad += 1
        if bad >= encoder_patience:
            break
    if best_state is None:
        raise RuntimeError("CLV-core encoder의 최적 epoch를 선택하지 못했습니다")
    model.load_state_dict(best_state)
    diagnostics = _diagnostics(model, dataset.anchors[-1], transform, device)
    diagnostics.update(best_epoch=best_epoch, best_val_loss=best_loss)

    final_transform = residual.fit_feature_transform(dataset.anchors)
    all_arrays = _stack(dataset.anchors, final_transform)
    _seed(seed)
    final_model = CLVCoreEncoder(all_arrays[0].shape[1], all_arrays[1].shape[1]).to(
        device
    )
    _fit_epochs(
        final_model,
        all_arrays,
        best_epoch,
        encoder_batch_size,
        encoder_lr,
        seed,
        device,
    )
    snapshot_n, snapshot_v = _axis_inputs(final_snapshot, final_transform)
    final_model.eval()
    with torch.no_grad():
        h_n, h_v, log_n, log_v = final_model(
            torch.as_tensor(snapshot_n, dtype=torch.float32, device=device),
            torch.as_tensor(snapshot_v, dtype=torch.float32, device=device),
        )
    n_hat = np.expm1(log_n.cpu().numpy()).astype(np.float32)
    v_hat = np.expm1(log_v.cpu().numpy()).astype(np.float32)
    n_users = dataset.n_users
    h_all = np.zeros((n_users, 16), np.float32)
    n_hat_all = np.zeros(n_users, np.float32)
    v_hat_all = np.zeros(n_users, np.float32)
    ids = final_snapshot.user_ids
    h_all[ids] = torch.cat([h_n, h_v], dim=1).cpu().numpy()
    n_hat_all[ids] = n_hat
    v_hat_all[ids] = v_hat
    for parameter in final_model.parameters():
        parameter.requires_grad_(False)
    return CLVCoreArtifact(
        final_model,
        final_transform,
        best_epoch,
        diagnostics,
        h_all,
        n_hat_all * v_hat_all,
        n_hat_all,
        v_hat_all,
    )


def compose_clv_core_profiles(
    artifact: CLVCoreArtifact,
    snapshot: residual.AnchorExamples,
    device: torch.device,
) -> UserProfileArtifact:
    x_n, x_v = _axis_inputs(snapshot, artifact.transform)
    artifact.model.eval()
    with torch.no_grad():
        h_n, h_v, log_n, log_v = artifact.model(
            torch.as_tensor(x_n, dtype=torch.float32, device=device),
            torch.as_tensor(x_v, dtype=torch.float32, device=device),
        )
        log_clv = torch.log1p(torch.expm1(log_n) * torch.expm1(log_v))
    local = np.concatenate(
        [
            x_n,
            x_v,
            h_n.cpu().numpy(),
            h_v.cpu().numpy(),
            log_n.cpu().numpy()[:, None],
            log_v.cpu().numpy()[:, None],
            log_clv.cpu().numpy()[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    n_users = artifact.h_all.shape[0]
    values = np.zeros((n_users, local.shape[1]), np.float32)
    valid_user = np.zeros(n_users, dtype=bool)
    values[snapshot.user_ids] = local
    valid_user[snapshot.user_ids] = True
    names = tuple(f"repurchase_{name}" for name in REPURCHASE_FEATURES)
    names += tuple(f"repurchase_valid_{name}" for name in REPURCHASE_FEATURES)
    names += tuple(f"monetary_{name}" for name in MONETARY_FEATURES)
    names += tuple(f"monetary_valid_{name}" for name in MONETARY_FEATURES)
    names += tuple(f"repurchase_embedding_{index}" for index in range(8))
    names += tuple(f"monetary_embedding_{index}" for index in range(8))
    names += (
        "pred_log_future_transactions",
        "pred_log_transaction_value",
        "pred_log_clv_proxy",
    )
    if values.shape[1] != len(names):
        raise RuntimeError("CLV-core profile schema와 행렬 차원이 다릅니다")
    if not np.isfinite(values).all():
        raise ValueError("CLV-core profile에 유한하지 않은 값이 있습니다")
    return UserProfileArtifact(values, valid_user, names)
