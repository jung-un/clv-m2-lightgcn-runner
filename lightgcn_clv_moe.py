"""Runner for CLV-conditioned mixture-of-embedding LightGCN M2."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3
from clv_moe_features import build_item_profiles, compose_user_profiles
from clv_moe_model import CLVMixtureEmbeddingModel, moe_diagnostics


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
    run_controls_after_success: bool = True
    confirmation_ready: bool = False
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
    if cfg.eval_holdout and not cfg.eval_test:
        raise ValueError("holdout 확증은 test 확증 설정 뒤에만 활성화합니다")
    if (cfg.eval_test or cfg.eval_holdout) and not cfg.confirmation_ready:
        raise ValueError("확증 split은 confirmation_ready=True로 명시 승인해야 합니다")
    return cfg


def state_hash(module: torch.nn.Module) -> str:
    return residual.state_hash(module)


def select_lambda(rows, baseline: dict, tolerance: float = 0.01):
    """Apply six accuracy guardrails, then maximize weighted hit at positive λ."""
    table = pd.DataFrame(rows).copy()
    guards = [f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)]
    eligible = np.ones(len(table), dtype=bool)
    for key in guards:
        if key not in table.columns or key not in baseline:
            raise KeyError(f"lambda 선택에 필요한 {key}가 없습니다")
        eligible &= table[key].to_numpy(dtype=float) >= float(baseline[key]) * (
            1.0 - tolerance
        )
    table["eligible"] = eligible
    candidates = table[table["eligible"] & table["lambda"].gt(0)]
    if candidates.empty:
        selected = 0.0
    else:
        best_revenue = float(candidates["revenue@10"].max())
        tied = candidates[
            np.isclose(candidates["revenue@10"].to_numpy(dtype=float), best_revenue)
        ]
        selected = float(tied["lambda"].min())
    table.attrs["success"] = selected > 0.0
    return selected, table


def validate_result_metrics(flat: dict, ks=(10, 20, 50)) -> None:
    residual.validate_result_metrics(flat, ks)


def preflight_summary(cfg: MoEConfig) -> dict:
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed_list": list(cfg.seed_list),
        "window": "full official train (~2 years)",
        "encoder_windows": {
            "input_days": cfg.input_days,
            "target_days": cfg.target_days,
            "anchor_offsets": list(cfg.anchor_offsets),
        },
        "expert_count": cfg.expert_count,
        "expert_hidden_dim": cfg.expert_hidden_dim,
        "expert_dim": cfg.expert_dim,
        "frozen_epochs": cfg.frozen_epochs,
        "adapter_lr": cfg.adapter_lr,
        "base_lr": cfg.base_lr,
        "lambda_train": cfg.lambda_train,
        "lambda_eval": list(cfg.lambda_eval),
        "selection": "six Recall/NDCG@10/20/50 >= 99% of external M1, then max weighted-hit@10",
        "graph_mode": "binary",
        "loss_mode": "plain",
        "negative_sampling": "uniform",
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "run_controls_after_success": cfg.run_controls_after_success,
        "models": [
            "m1",
            "clv_moe",
            "pref_continue",
            "frozen_moe",
            "constant_gate",
            "shuffled_clv",
            "single_adapter",
        ],
        "config": asdict(cfg),
    }


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
    best_updates = 0
    best_base_updates = 0
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
            best_updates = updates
            best_base_updates = sum(base_updates_by_epoch)
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
        "updates_at_best": int(best_updates),
        "base_updates_at_best": int(best_base_updates),
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


def _encoder_config(cfg: MoEConfig) -> residual.ResidualConfig:
    return residual.ResidualConfig(
        dataset=cfg.dataset,
        seed_list=cfg.seed_list,
        input_days=cfg.input_days,
        target_days=cfg.target_days,
        anchor_offsets=cfg.anchor_offsets,
        encoder_epochs=cfg.encoder_epochs,
        encoder_patience=cfg.encoder_patience,
        encoder_batch_size=cfg.encoder_batch_size,
        encoder_lr=cfg.encoder_lr,
        eval_test=False,
        eval_holdout=False,
    )


def _result_fingerprint(cfg: MoEConfig, base_cfg: dict) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "moe": asdict(cfg),
        "base": {
            key: base_cfg[key]
            for key in (
                "DIM",
                "N_LAYERS",
                "BATCH_SIZE",
                "EPOCHS",
                "WINDOW_DAYS",
                "VAL_DAYS",
                "TEST_DAYS",
                "HOLDOUT_DAYS",
                "MIN_USER_INTER",
                "MIN_ITEM_INTER",
                "NEG_MODE",
                "GRAPH_MODE",
                "LOSS_MODE",
            )
        },
        "user_features": residual.NUMERIC_FEATURES,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _pure_m1_config(cfg: MoEConfig, m1_dir: str) -> dict:
    configured = v3.configure_run(
        dataset=cfg.dataset,
        out_dir=m1_dir,
        ARCH="pref_only",
        SEED_LIST=list(cfg.seed_list),
        WINDOW_DAYS=None,
        GRAPH_MODE="binary",
        LOSS_MODE="plain",
        GATE_MODE="clv",
        NEG_MODE="uniform",
        EVAL_TEST=cfg.eval_test,
        EVAL_HOLDOUT=cfg.eval_holdout,
    )
    base_cfg = dict(configured)
    required = {
        "ARCH": "pref_only",
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
        "NEG_MODE": "uniform",
    }
    for key, expected in required.items():
        if base_cfg[key] != expected:
            raise RuntimeError(f"M2 기준설정 오염: {key}={base_cfg[key]!r}")
    return base_cfg


def _flat_evaluation(
    model,
    lam: float,
    cache,
    meta,
    data,
    base_cfg,
    *,
    per_user: bool,
):
    ones = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    result = v3.evaluate(
        model,
        lam,
        ones,
        cache,
        meta,
        base_cfg["K_LIST"],
        data["csr_ptr"],
        data["csr_items"],
        base_cfg,
        per_user=per_user,
    )
    per_user_result = result.pop("per_user", None)
    flat = residual.normalize_flat_metrics(v3.flatten(result))
    validate_result_metrics(flat, base_cfg["K_LIST"])
    return flat, per_user_result


def _fresh_external_m1(context: dict, seed: int, data: dict, base_cfg: dict):
    model, stats = v3.get_or_train(
        "pref_only",
        seed,
        data,
        context["ones_gate"],
        data["x_val_u"],
        context["x_item"],
        context["item_cat"],
        context["meta"],
        context["caches"]["val"],
        base_cfg,
    )
    model.eval()
    return model, stats


def _build_model(base_model, context: dict, cfg: MoEConfig, control: str, seed: int):
    return CLVMixtureEmbeddingModel(
        base_model,
        context["user_profile"],
        context["item_profile"],
        control=control,
        seed=seed,
        expert_count=cfg.expert_count,
        expert_hidden_dim=cfg.expert_hidden_dim,
        expert_dim=cfg.expert_dim,
        category_dim=cfg.category_dim,
    ).to(v3.DEVICE)


def _diagnostic_columns(diagnostics: dict) -> dict:
    out = {
        "gate_entropy_mean": diagnostics["gate_entropy_mean"],
        "residual_to_base_score_std": diagnostics[
            "residual_to_base_score_std"
        ],
        "parameter_match_ratio": diagnostics["parameter_match_ratio"],
    }
    for index, value in enumerate(diagnostics["expert_usage_mean"]):
        out[f"expert_usage_{index}"] = value
    return out


def checkpoint_paths_for_json(paths: dict[tuple[str, int], str]) -> dict[str, str]:
    return {
        f"{model_id}_s{seed}": str(path)
        for (model_id, seed), path in sorted(paths.items())
    }


def _save_model_checkpoint(path: Path, model, context, stats, diagnostics):
    torch.save(
        {
            "state": model.state_dict(),
            "user_profile": context["user_profile"].values,
            "user_valid": context["user_profile"].valid_user,
            "user_feature_names": context["user_profile"].feature_names,
            "item_numeric": context["item_profile"].numeric,
            "item_category_ids": context["item_profile"].category_ids,
            "item_valid": context["item_profile"].valid_item,
            "item_numeric_names": context["item_profile"].numeric_names,
            "item_n_categories": context["item_profile"].n_categories,
            "ev_all": context["artifact"].ev_all,
            "training": stats,
            "diagnostics": diagnostics,
        },
        path,
    )


def run_experiment(cfg: MoEConfig | None = None) -> pd.DataFrame:
    """Run validation screening, optional controls, and protected confirmation."""
    cfg = cfg or configure_moe_run("dunnhumby")
    out_dir = Path(
        cfg.out_dir or f"{v3.default_out_dir(cfg.dataset)}_clv_moe"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    m1_dir = cfg.m1_checkpoint_dir or v3.default_out_dir(cfg.dataset)
    base_cfg = _pure_m1_config(cfg, str(m1_dir))
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    data = v3.prepare_data(base_cfg, v3.DCFG)
    encoder_cfg = _encoder_config(cfg)
    anchors = residual.build_anchor_examples(
        data["train"],
        data["n_users"],
        v3.DCFG["is_date"],
        cfg.input_days,
        cfg.target_days,
        cfg.anchor_offsets,
    )
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    item_profile = build_item_profiles(data["train"], data["n_items"])
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    ones_gate = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    fingerprint = _result_fingerprint(cfg, base_cfg)

    rows: list[dict] = []
    delta_records: list[dict] = []
    contexts: dict[int, dict] = {}
    baseline_rows: dict[int, dict] = {}
    baseline_per_user: dict[tuple[str, int], dict] = {}
    model_per_user: dict[tuple[str, str, int, float], dict] = {}
    train_records: dict[str, dict] = {}
    diagnostic_records: dict[str, dict] = {}
    checkpoint_paths: dict[tuple[str, int], str] = {}
    encoder_records: dict[str, dict] = {}

    for seed in cfg.seed_list:
        artifact = residual.train_future_value_encoder(
            anchors, snapshot, encoder_cfg, seed, v3.DEVICE
        )
        encoder_records[str(seed)] = artifact.diagnostics
        encoder_path = out_dir / f"encoder_{cfg.dataset}_s{seed}_{fingerprint}.pt"
        torch.save(
            {
                "state": artifact.model.state_dict(),
                "transform_mean": artifact.transform.mean,
                "transform_std": artifact.transform.std,
                "feature_names": artifact.transform.feature_names,
                "h_all": artifact.h_all,
                "ev_all": artifact.ev_all,
                "best_epoch": artifact.best_epoch,
                "diagnostics": artifact.diagnostics,
            },
            encoder_path,
        )
        user_profile = compose_user_profiles(artifact, snapshot, v3.DEVICE)
        segment_thresholds = v3.segment_thresholds(
            artifact.ev_all, base_cfg["SEG_EDGES"]
        )
        caches = {
            name: v3.EvalCache(
                gt,
                revenue,
                artifact.ev_all,
                segment_thresholds,
                data["n_items"],
            )
            for name, (gt, revenue) in data["splits"].items()
        }
        context = {
            "artifact": artifact,
            "user_profile": user_profile,
            "item_profile": item_profile,
            "x_item": x_item,
            "item_cat": item_cat,
            "meta": meta,
            "ones_gate": ones_gate,
            "caches": caches,
            "encoder_path": str(encoder_path),
        }
        contexts[seed] = context
        external_m1, _ = _fresh_external_m1(context, seed, data, base_cfg)
        baseline_flat, baseline_pu = _flat_evaluation(
            external_m1,
            0.0,
            caches["val"],
            meta,
            data,
            base_cfg,
            per_user=True,
        )
        baseline_rows[seed] = baseline_flat
        baseline_per_user[("val", seed)] = baseline_pu
        rows.append(
            {
                "seed": seed,
                "model_id": "m1",
                "split": "val",
                "lambda": 0.0,
                "role": "baseline",
                **baseline_flat,
            }
        )

        model = _build_model(external_m1, context, cfg, "clv", seed)
        encoder_hash = state_hash(artifact.model)

        def validation_recall(candidate):
            flat, _ = _flat_evaluation(
                candidate,
                cfg.lambda_train,
                caches["val"],
                meta,
                data,
                base_cfg | {"K_LIST": [10]},
                per_user=False,
            )
            return flat["recall@10"]

        stats = train_moe(
            model, data, base_cfg, cfg, seed, validation_recall
        )
        if state_hash(artifact.model) != encoder_hash:
            raise RuntimeError("MoE 학습 중 동결된 future-value encoder가 변경됐습니다")
        diagnostics = moe_diagnostics(model, seed=seed)
        train_records[f"clv_moe_s{seed}"] = stats
        diagnostic_records[f"clv_moe_s{seed}"] = diagnostics
        checkpoint = out_dir / f"clv_moe_{cfg.dataset}_s{seed}_{fingerprint}.pt"
        _save_model_checkpoint(checkpoint, model, context, stats, diagnostics)
        checkpoint_paths[("clv_moe", seed)] = str(checkpoint)
        for lam in cfg.lambda_eval:
            flat, per_user = _flat_evaluation(
                model,
                lam,
                caches["val"],
                meta,
                data,
                base_cfg,
                per_user=True,
            )
            model_per_user[("val", "clv_moe", seed, lam)] = per_user
            rows.append(
                {
                    "seed": seed,
                    "model_id": "clv_moe",
                    "split": "val",
                    "lambda": lam,
                    "role": "model",
                    **_diagnostic_columns(diagnostics),
                    **flat,
                }
            )
        del model, external_m1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    baseline_mean = {
        key: float(np.mean([row[key] for row in baseline_rows.values()]))
        for key in next(iter(baseline_rows.values()))
    }
    main_mean_rows = (
        frame[(frame.model_id == "clv_moe") & (frame.split == "val")]
        .groupby("lambda", as_index=False)
        .mean(numeric_only=True)
        .to_dict("records")
    )
    main_lambda, main_table = select_lambda(
        main_mean_rows, baseline_mean, cfg.accuracy_tolerance
    )
    selected = {"clv_moe": main_lambda}
    selection_success = {"clv_moe": bool(main_table.attrs["success"])}
    selection_tables = {"clv_moe": main_table.to_dict("records")}

    if selection_success["clv_moe"] and cfg.run_controls_after_success:
        seed = 42
        if seed not in contexts:
            raise RuntimeError("대조군 screening에는 seed 42가 필요합니다")
        context = contexts[seed]
        control_specs = {
            "frozen_moe": ("clv", True),
            "constant_gate": ("constant_gate", False),
            "shuffled_clv": ("shuffled_clv", False),
            "single_adapter": ("single_adapter", False),
        }
        for model_id, (control, freeze_base) in control_specs.items():
            external_m1, _ = _fresh_external_m1(context, seed, data, base_cfg)
            model = _build_model(external_m1, context, cfg, control, seed)

            def control_recall(candidate):
                flat, _ = _flat_evaluation(
                    candidate,
                    cfg.lambda_train,
                    context["caches"]["val"],
                    meta,
                    data,
                    base_cfg | {"K_LIST": [10]},
                    per_user=False,
                )
                return flat["recall@10"]

            stats = train_moe(
                model,
                data,
                base_cfg,
                cfg,
                seed,
                control_recall,
                freeze_base=freeze_base,
            )
            diagnostics = moe_diagnostics(model, seed=seed)
            train_records[f"{model_id}_s{seed}"] = stats
            diagnostic_records[f"{model_id}_s{seed}"] = diagnostics
            checkpoint = out_dir / f"{model_id}_{cfg.dataset}_s{seed}_{fingerprint}.pt"
            _save_model_checkpoint(checkpoint, model, context, stats, diagnostics)
            checkpoint_paths[(model_id, seed)] = str(checkpoint)
            control_rows = []
            for lam in cfg.lambda_eval:
                flat, per_user = _flat_evaluation(
                    model,
                    lam,
                    context["caches"]["val"],
                    meta,
                    data,
                    base_cfg,
                    per_user=True,
                )
                model_per_user[("val", model_id, seed, lam)] = per_user
                row = {
                    "seed": seed,
                    "model_id": model_id,
                    "split": "val",
                    "lambda": lam,
                    "role": "control",
                    **_diagnostic_columns(diagnostics),
                    **flat,
                }
                rows.append(row)
                control_rows.append(row)
            lam, table = select_lambda(
                control_rows, baseline_rows[seed], cfg.accuracy_tolerance
            )
            selected[model_id] = lam
            selection_success[model_id] = bool(table.attrs["success"])
            selection_tables[model_id] = table.to_dict("records")
            del model, external_m1

        external_m1, _ = _fresh_external_m1(context, seed, data, base_cfg)
        target_updates = train_records[f"clv_moe_s{seed}"]["base_updates_at_best"]
        stats = train_pref_continue(
            external_m1,
            data,
            base_cfg,
            cfg,
            seed,
            target_updates,
        )
        flat, per_user = _flat_evaluation(
            external_m1,
            0.0,
            context["caches"]["val"],
            meta,
            data,
            base_cfg,
            per_user=True,
        )
        train_records[f"pref_continue_s{seed}"] = stats
        model_per_user[("val", "pref_continue", seed, 0.0)] = per_user
        rows.append(
            {
                "seed": seed,
                "model_id": "pref_continue",
                "split": "val",
                "lambda": 0.0,
                "role": "control",
                **flat,
            }
        )
        selected["pref_continue"] = 0.0
        selection_success["pref_continue"] = True

    for model_id, lam in selected.items():
        relevant_seeds = [
            seed
            for seed in cfg.seed_list
            if ("val", model_id, seed, lam) in model_per_user
        ]
        if model_id != "clv_moe":
            relevant_seeds = [
                seed
                for seed in relevant_seeds
                if seed == 42
            ]
        for metric in ("recall", "ndcg", "revenue", "arp"):
            diffs = [
                model_per_user[("val", model_id, seed, lam)][metric]
                - baseline_per_user[("val", seed)][metric]
                for seed in relevant_seeds
            ]
            if diffs:
                delta_records.append(
                    {
                        "model_id": model_id,
                        "split": "val",
                        "lambda": lam,
                        "metric": metric,
                        **v3.paired_bootstrap(diffs, base_cfg["N_BOOT"]),
                    }
                )

    if cfg.eval_test or cfg.eval_holdout:
        if not selection_success["clv_moe"]:
            raise RuntimeError("validation에서 실패한 M2는 확증 split으로 진행할 수 없습니다")
        for split in ("test", "holdout"):
            if split not in data["splits"]:
                continue
            for seed in cfg.seed_list:
                context = contexts[seed]
                external_m1, _ = _fresh_external_m1(context, seed, data, base_cfg)
                base_flat, base_pu = _flat_evaluation(
                    external_m1,
                    0.0,
                    context["caches"][split],
                    meta,
                    data,
                    base_cfg,
                    per_user=True,
                )
                baseline_per_user[(split, seed)] = base_pu
                rows.append(
                    {
                        "seed": seed,
                        "model_id": "m1",
                        "split": split,
                        "lambda": 0.0,
                        "role": "baseline",
                        **base_flat,
                    }
                )
                model = _build_model(external_m1, context, cfg, "clv", seed)
                checkpoint = torch.load(
                    checkpoint_paths[("clv_moe", seed)],
                    map_location=v3.DEVICE,
                    weights_only=False,
                )
                model.load_state_dict(checkpoint["state"])
                lam = selected["clv_moe"]
                flat, per_user = _flat_evaluation(
                    model,
                    lam,
                    context["caches"][split],
                    meta,
                    data,
                    base_cfg,
                    per_user=True,
                )
                rows.append(
                    {
                        "seed": seed,
                        "model_id": "clv_moe",
                        "split": split,
                        "lambda": lam,
                        "role": "primary",
                        **flat,
                    }
                )
                for metric in ("recall", "ndcg", "revenue", "arp"):
                    diff = per_user[metric] - base_pu[metric]
                    delta_records.append(
                        {
                            "model_id": "clv_moe",
                            "split": split,
                            "lambda": lam,
                            "metric": metric,
                            **v3.paired_bootstrap([diff], base_cfg["N_BOOT"]),
                        }
                    )

    frame = pd.DataFrame(rows)
    stem = f"clv_moe_{cfg.dataset}_{fingerprint}"
    result_csv = out_dir / f"{stem}.csv"
    delta_csv = out_dir / f"{stem}_delta.csv"
    result_json = out_dir / f"{stem}.json"
    frame.to_csv(result_csv, index=False, float_format="%.8f")
    pd.DataFrame(delta_records).to_csv(delta_csv, index=False)
    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "code_version": CODE_VERSION,
                "config": asdict(cfg),
                "base_config": {
                    key: value for key, value in base_cfg.items() if key != "OUT_DIR"
                },
                "data_stats": data["data_stats"],
                "selected_lambda": selected,
                "selection_success": selection_success,
                "selection_tables": selection_tables,
                "encoder_diagnostics": encoder_records,
                "training": train_records,
                "moe_diagnostics": diagnostic_records,
                "checkpoint_paths": checkpoint_paths_for_json(checkpoint_paths),
                "delta": delta_records,
                "interpretation": {
                    "clv": "train-only CLV-related behavior representation; not realized lifetime CLV",
                    "revenue": "price/purchase-amount weighted hit; not incremental revenue",
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"저장: {result_csv}")
    print(f"validation 선택 λ: {selected}")
    print(f"선택 성공: {selection_success}")
    return frame


if __name__ == "__main__":
    run_experiment()
