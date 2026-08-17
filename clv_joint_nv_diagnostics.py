"""Checkpoint-only diagnostics for the joint N/V LightGCN model."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr


BLOCKS = ("id", "n", "v")


def find_joint_checkpoint(
    out_dir, *, dataset: str, seed: int, model_id: str = "joint_nv"
) -> Path:
    candidates = list(Path(out_dir).glob(f"{model_id}_{dataset}_s{seed}_*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"{model_id} checkpoint not found: {out_dir} / {dataset} / seed {seed}"
        )
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def load_joint_checkpoint(model, path, *, dataset: str, seed: int, input_hash: str):
    path = Path(path)
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    config = payload.get("config", {})
    if config.get("dataset") != dataset or int(config.get("seed", -1)) != int(seed):
        raise RuntimeError("checkpoint dataset/seed identity mismatch")
    if payload.get("input_hash") != input_hash:
        raise RuntimeError("checkpoint input hash does not match current data")
    model.load_state_dict(payload["state"])
    model.eval()
    return payload


def _block_slices(model) -> dict[str, slice]:
    id_dim = int(model.E_u.embedding_dim)
    axis_dim = int(model.activity_user.net[-1].out_features)
    return {
        "id": slice(0, id_dim),
        "n": slice(id_dim, id_dim + axis_dim),
        "v": slice(id_dim + axis_dim, id_dim + 2 * axis_dim),
    }


def _mask_embeddings(model, blocks: Iterable[str]):
    selected = frozenset(blocks)
    unknown = selected.difference(BLOCKS)
    if not selected or unknown:
        raise ValueError(f"blocks must be a non-empty subset of {BLOCKS}: {sorted(unknown)}")
    user, item = model.propagate()
    mask = user.new_zeros(user.shape[1])
    for block in selected:
        mask[_block_slices(model)[block]] = 1.0
    return user * mask, item * mask


class JointNVBlockView(nn.Module):
    """Read-only evaluator view that keeps selected propagated score blocks."""

    def __init__(self, model, blocks: Iterable[str]):
        super().__init__()
        self.model = model
        self.blocks = tuple(blocks)

    def embeddings(self, need_value: bool = True):
        user, item = _mask_embeddings(self.model, self.blocks)
        zero_user = user.new_zeros((self.model.n_users, 1))
        zero_item = item.new_zeros((self.model.n_items, 1))
        return user, item, zero_user, zero_item


class JointNVStrengthView(nn.Module):
    """Read-only evaluator view that rescales the learned N/V score contribution."""

    def __init__(self, model, multiplier: float):
        super().__init__()
        if not np.isfinite(multiplier) or multiplier < 0:
            raise ValueError("multiplier must be finite and non-negative")
        self.model = model
        self.multiplier = float(multiplier)

    def embeddings(self, need_value: bool = True):
        user, item = self.model.propagate()
        coordinate_scale = user.new_ones(user.shape[1])
        axis_scale = self.multiplier**0.5
        slices = _block_slices(self.model)
        coordinate_scale[slices["n"]] = axis_scale
        coordinate_scale[slices["v"]] = axis_scale
        zero_user = user.new_zeros((self.model.n_users, 1))
        zero_item = item.new_zeros((self.model.n_items, 1))
        return (
            user * coordinate_scale,
            item * coordinate_scale,
            zero_user,
            zero_item,
        )


def evaluate_block_views(model, evaluator) -> dict[str, dict]:
    """Evaluate the full model and exact propagated block ablations."""
    views = {
        "id_only": ("id",),
        "n_only": ("n",),
        "v_only": ("v",),
        "id_n": ("id", "n"),
        "id_v": ("id", "v"),
        "full": ("id", "n", "v"),
    }
    return {
        name: evaluator(JointNVBlockView(model, blocks).eval())
        for name, blocks in views.items()
    }


def evaluate_strength_curve(model, evaluator, multipliers=(0.0, 1.0, 2.0, 4.0, 8.0)):
    """Evaluate whether the learned N/V direction helps when only its score scale changes."""
    return {
        float(multiplier): evaluator(JointNVStrengthView(model, multiplier).eval())
        for multiplier in multipliers
    }


@torch.no_grad()
def block_score_diagnostics(model, users: torch.Tensor, items: torch.Tensor) -> dict:
    """Return exact paired ID/N/V scores and their reconstruction error."""
    user, item = model.propagate()
    slices = _block_slices(model)
    scores = {}
    for name, block in slices.items():
        scores[f"{name}_scores"] = (
            user[users, block] * item[items, block]
        ).sum(dim=1)
    full = (user[users] * item[items]).sum(dim=1)
    reconstructed = scores["id_scores"] + scores["n_scores"] + scores["v_scores"]
    return {
        "full_scores": full,
        **scores,
        "reconstruction_max_abs_error": float((full - reconstructed).abs().max()),
    }


@torch.no_grad()
def sampled_block_score_summary(
    model, *, n_users: int = 512, n_items: int = 2048, seed: int = 42
) -> dict:
    """Measure effective block strength on a bounded user-item score grid."""
    user, item = model.propagate()
    rng = np.random.default_rng(seed)
    user_ids = rng.choice(len(user), min(n_users, len(user)), replace=False)
    item_ids = rng.choice(len(item), min(n_items, len(item)), replace=False)
    user_ids = torch.as_tensor(user_ids, dtype=torch.long, device=user.device)
    item_ids = torch.as_tensor(item_ids, dtype=torch.long, device=item.device)
    slices = _block_slices(model)
    block_scores = {
        name: user[user_ids, block] @ item[item_ids, block].T
        for name, block in slices.items()
    }
    std = {name: float(score.std()) for name, score in block_scores.items()}
    id_std = max(std["id"], 1e-12)
    flat = {name: score.flatten().float().cpu().numpy() for name, score in block_scores.items()}
    return {
        "gamma_n": float(model.gamma_n),
        "gamma_v": float(model.gamma_v),
        "id_score_std": std["id"],
        "n_score_std": std["n"],
        "v_score_std": std["v"],
        "n_to_id_std_ratio": std["n"] / id_std,
        "v_to_id_std_ratio": std["v"] / id_std,
        "n_v_score_spearman": float(spearmanr(flat["n"], flat["v"]).statistic),
        "sample_users": int(len(user_ids)),
        "sample_items": int(len(item_ids)),
    }


def _midrank(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def _max_tie_share(values: np.ndarray) -> float:
    if not len(values):
        return 0.0
    return float(np.unique(values, return_counts=True)[1].max() / len(values))


def axis_distribution_diagnostics(
    n_values: np.ndarray, v_values: np.ndarray, valid_user: np.ndarray
) -> dict:
    """Describe gate-source ties using the same midrank convention as training."""
    n_values = np.asarray(n_values, dtype=np.float64)
    v_values = np.asarray(v_values, dtype=np.float64)
    valid = np.asarray(valid_user, dtype=bool)
    if n_values.shape != v_values.shape or n_values.shape != valid.shape:
        raise ValueError("N/V/valid_user shapes must match")
    n = n_values[valid]
    v = v_values[valid]
    q_n = _midrank(n) if len(n) else np.array([], dtype=np.float64)
    q_v = _midrank(v) if len(v) else np.array([], dtype=np.float64)
    correlation = 0.0
    if len(n) >= 3 and np.unique(n).size > 1 and np.unique(v).size > 1:
        correlation = float(spearmanr(n, v).statistic)
    return {
        "n_valid_users": int(valid.sum()),
        "q_n_unique": int(np.unique(q_n).size),
        "q_v_unique": int(np.unique(q_v).size),
        "n_max_tie_share": _max_tie_share(n),
        "v_max_tie_share": _max_tie_share(v),
        "n_zero_share": float(np.mean(n == 0)) if len(n) else 0.0,
        "v_zero_share": float(np.mean(v == 0)) if len(v) else 0.0,
        "q_n_min": float(q_n.min()) if len(q_n) else 0.0,
        "q_n_max": float(q_n.max()) if len(q_n) else 0.0,
        "q_v_min": float(q_v.min()) if len(q_v) else 0.0,
        "q_v_max": float(q_v.max()) if len(q_v) else 0.0,
        "n_v_spearman": correlation,
    }
