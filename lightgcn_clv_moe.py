"""Runner for CLV-conditioned mixture-of-embedding LightGCN M2."""

from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn.functional as F

import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-moe-v1.0"


@dataclass
class MoEConfig:
    dataset: str = "dunnhumby"
    seed_list: tuple[int, ...] = (42,)
    input_days: int = 365
    target_days: int = 90
    anchor_offsets: tuple[int, ...] = (270, 180, 90)
    encoder_epochs: int = 100
    encoder_patience: int = 10
    encoder_batch_size: int = 1024
    encoder_lr: float = 1e-3
    expert_count: int = 3
    expert_hidden_dim: int = 32
    expert_dim: int = 16
    category_dim: int = 8
    frozen_epochs: int = 5
    max_epochs: int = 100
    patience: int = 20
    adapter_lr: float = 5e-4
    base_lr: float = 5e-5
    lambda_train: float = 1.0
    lambda_eval: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    accuracy_tolerance: float = 0.01
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None
    m1_checkpoint_dir: str | None = None


def configure_moe_run(dataset: str, **overrides) -> MoEConfig:
    dataset = dataset.lower()
    if dataset not in {"hm", "dunnhumby"}:
        raise ValueError("dataset은 'hm' 또는 'dunnhumby'여야 합니다")
    valid = {field.name for field in fields(MoEConfig)}
    unknown = set(overrides).difference(valid)
    if unknown:
        raise TypeError(f"알 수 없는 MoE 설정: {sorted(unknown)}")
    cfg = MoEConfig(dataset=dataset, **overrides)
    if cfg.expert_count != 3:
        raise ValueError("승인된 screening의 expert_count는 3입니다")
    if cfg.frozen_epochs < 0 or cfg.max_epochs <= cfg.frozen_epochs:
        raise ValueError("max_epochs는 frozen_epochs보다 커야 합니다")
    if cfg.eval_test and cfg.eval_holdout:
        raise ValueError("test와 holdout은 한 실행에서 동시에 열지 않습니다")
    return cfg


def state_hash(module: torch.nn.Module) -> str:
    return residual.state_hash(module)


def _base_parameters(base_model: torch.nn.Module) -> list[torch.nn.Parameter]:
    if hasattr(base_model, "pref_params"):
        return list(base_model.pref_params())
    return list(base_model.parameters())


def _set_base_trainable(base_model: torch.nn.Module, enabled: bool) -> None:
    for parameter in _base_parameters(base_model):
        parameter.requires_grad_(enabled)
    if enabled and hasattr(base_model, "_pref_cache"):
        base_model._pref_cache = None


def _plain_bpr_batches(data: dict, base_cfg: dict, rng, device):
    order = rng.permutation(len(data["tr_u"]))
    batch_size = int(base_cfg["BATCH_SIZE"])
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        users = data["tr_u"][idx]
        positives = data["tr_i"][idx]
        negatives = v3.sample_negatives(
            users,
            positives,
            data["n_items"],
            data["pos_key"],
            rng,
            base_cfg["NEG_MODE"],
            data["item_cat"],
            data["cat_items"],
        )
        yield tuple(
            torch.as_tensor(values, dtype=torch.long, device=device)
            for values in (users, positives, negatives)
        )


def _base_regularization(base_model, users, positives, negatives):
    if hasattr(base_model, "batch_l2"):
        return base_model.batch_l2(users, positives, negatives, need_value=False)
    return users.new_zeros((), dtype=torch.float32)


def _clone_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def train_moe(
    model,
    data: dict,
    base_cfg: dict,
    cfg: MoEConfig,
    seed: int,
    eval_recall,
    freeze_base: bool = False,
) -> dict:
    """Train adapters with plain BPR, then optionally fine-tune M1 embeddings."""
    residual._seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    _set_base_trainable(model.base_model, False)
    optimizer = torch.optim.Adam(
        [
            {"params": model.adapter_parameters(), "lr": cfg.adapter_lr},
            {"params": _base_parameters(model.base_model), "lr": cfg.base_lr},
        ]
    )
    best = -float("inf")
    best_epoch = 0
    best_state = None
    bad = 0
    updates = 0
    samples = 0
    base_updates_by_epoch = []
    started = time.time()
    for epoch in range(1, cfg.max_epochs + 1):
        base_active = (not freeze_base) and epoch > cfg.frozen_epochs
        _set_base_trainable(model.base_model, base_active)
        model.train()
        base_updates = 0
        for users, positives, negatives in _plain_bpr_batches(
            data, base_cfg, rng, device
        ):
            loss = model.bpr_loss(users, positives, negatives, cfg.lambda_train)
            if base_active:
                loss = loss + _base_regularization(
                    model.base_model, users, positives, negatives
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            samples += len(users)
            base_updates += int(base_active)
        base_updates_by_epoch.append(base_updates)
        model.eval()
        score = float(eval_recall(model))
        if score > best + 1e-12:
            best = score
            best_epoch = epoch
            best_state = _clone_state(model)
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("MoE 학습에서 유효한 validation checkpoint를 만들지 못했습니다")
    model.load_state_dict(best_state)
    return {
        "loss": "plain_bpr",
        "best_epoch": int(best_epoch),
        "epochs_run": int(epoch),
        "best_val_recall@10": float(best),
        "updates": int(updates),
        "samples": int(samples),
        "base_updates": int(sum(base_updates_by_epoch)),
        "base_updates_by_epoch": base_updates_by_epoch,
        "wall_clock_sec": float(time.time() - started),
        "freeze_base": bool(freeze_base),
    }


def _base_bpr_loss(base_model, users, positives, negatives):
    user_embedding, item_embedding, *_ = base_model.embeddings(need_value=False)
    positive_score = (user_embedding[users] * item_embedding[positives]).sum(1)
    negative_score = (user_embedding[users] * item_embedding[negatives]).sum(1)
    return -F.logsigmoid(positive_score - negative_score).mean()


def train_pref_continue(
    base_model,
    data: dict,
    base_cfg: dict,
    cfg: MoEConfig,
    seed: int,
    target_base_updates: int,
) -> dict:
    """Continue pure M1 for the exact number of base updates used by joint-warm."""
    if target_base_updates < 0:
        raise ValueError("target_base_updates는 음수일 수 없습니다")
    residual._seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = next(base_model.parameters()).device
    _set_base_trainable(base_model, True)
    optimizer = torch.optim.Adam(_base_parameters(base_model), lr=cfg.base_lr)
    updates = 0
    samples = 0
    started = time.time()
    while updates < target_base_updates:
        for users, positives, negatives in _plain_bpr_batches(
            data, base_cfg, rng, device
        ):
            if updates >= target_base_updates:
                break
            loss = _base_bpr_loss(base_model, users, positives, negatives)
            loss = loss + _base_regularization(
                base_model, users, positives, negatives
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates += 1
            samples += len(users)
    return {
        "loss": "plain_bpr",
        "base_updates": int(updates),
        "samples": int(samples),
        "wall_clock_sec": float(time.time() - started),
    }
