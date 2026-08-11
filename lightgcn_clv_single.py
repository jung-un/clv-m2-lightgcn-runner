"""Validation-only runner for identifying single-adapter CLV information effects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
from clv_moe_model import CLVMixtureEmbeddingModel


PRIMARY_MODEL_ID = "single_full"
REQUIRED_CONTROLS = (
    "single_zero_user",
    "single_shuffled_user",
    "single_base_only",
)
MECHANISM_CONTROLS = ("single_zero_item",)
ALL_SINGLE_MODELS = (PRIMARY_MODEL_ID, *REQUIRED_CONTROLS, *MECHANISM_CONTROLS)
CODE_VERSION = "clv-single-identification-v1.0"
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
ACCURACY_TOLERANCE = 0.01
REUSE_CONFIG_KEYS = (
    "dataset",
    "seed_list",
    "input_days",
    "target_days",
    "anchor_offsets",
    "encoder_epochs",
    "encoder_patience",
    "encoder_batch_size",
    "encoder_lr",
    "expert_count",
    "expert_hidden_dim",
    "expert_dim",
    "category_dim",
    "frozen_epochs",
    "max_epochs",
    "patience",
    "adapter_lr",
    "base_lr",
    "lambda_train",
    "lambda_eval",
    "accuracy_tolerance",
)
REUSE_BASE_CONFIG_KEYS = (
    "DIM",
    "N_LAYERS",
    "BATCH_SIZE",
    "EPOCHS",
    "EARLY_STOP",
    "LR",
    "REG_MODE",
    "PREF_REG",
    "WD",
    "NEG_MODE",
    "WINDOW_DAYS",
    "VAL_DAYS",
    "TEST_DAYS",
    "HOLDOUT_DAYS",
    "MIN_USER_INTER",
    "MIN_ITEM_INTER",
    "K_LIST",
    "SEG_EDGES",
    "EVAL_BATCH",
    "GRAPH_MODE",
    "LOSS_MODE",
    "N_BOOT",
)


@dataclass(frozen=True)
class ReusableSingleFull:
    model: CLVMixtureEmbeddingModel
    rows: tuple[dict, ...]
    training: dict
    diagnostics: dict
    result_json_sha256: str
    legacy_source_revision: str
    legacy_checkpoint: str
    legacy_checkpoint_sha256: str


@dataclass
class PreparedSingleContext:
    out_dir: Path
    baseline_row: dict
    baseline_metrics: dict
    baseline_per_user: dict
    input_manifest: dict
    baseline_state_hash: str
    base_cfg: dict
    data: dict
    context: dict
    source_revision: str
    encoder_checkpoint: str
    m1_checkpoint: str
    run_fingerprint: str


@dataclass(frozen=True)
class VariantRun:
    model_id: str
    rows: tuple[dict, ...]
    per_user: dict[float, dict]
    training: dict
    diagnostics: dict
    checkpoint: str
    reuse_provenance: dict | None


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    payload = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _variant_audit(
    model: CLVMixtureEmbeddingModel, starting_base_state_hash: str
) -> dict:
    def tensor_hash(value: torch.Tensor) -> str:
        return array_sha256(value.detach().cpu().numpy())

    adapter_count = sum(parameter.numel() for parameter in model.adapter_parameters())
    base_count = sum(parameter.numel() for parameter in model.base_parameters())
    return {
        "starting_base_state_hash": starting_base_state_hash,
        "original_profile_sha256": tensor_hash(model.original_profile),
        "routed_profile_sha256": tensor_hash(model.routed_profile),
        "item_numeric_sha256": tensor_hash(model.item_numeric),
        "item_category_ids_sha256": tensor_hash(model.item_category_ids),
        "has_profile_sha256": tensor_hash(model.has_profile),
        "valid_item_sha256": tensor_hash(model.valid_item),
        "adapter_parameter_count": int(adapter_count),
        "base_parameter_count": int(base_count),
        "joint_trainable_parameter_count": int(adapter_count + base_count),
    }


def _diagnostic_columns_for_lambda(diagnostics: dict, lam: float) -> dict:
    columns = moe._diagnostic_columns(diagnostics)
    columns["effective_residual_to_base_score_std"] = abs(float(lam)) * float(
        diagnostics["residual_to_base_score_std"]
    )
    return columns


def configure_single_run(dataset: str, **overrides) -> moe.MoEConfig:
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "lambda_eval": LAMBDA_GRID,
        "run_controls_after_success": True,
        "out_dir": f"{v3.default_out_dir(dataset)}_clv_single",
    }
    return validate_single_config(
        moe.configure_moe_run(dataset, **(defaults | overrides))
    )


def validate_single_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError(
            "single-adapter screening-only runner cannot open test/holdout"
        )
    cfg = moe.validate_moe_config(cfg)
    if tuple(cfg.seed_list) != (42,):
        raise ValueError("single-adapter screening-only runner requires seed 42")
    if tuple(cfg.lambda_eval) != LAMBDA_GRID:
        raise ValueError("single-adapter lambda grid is frozen by the approved design")
    if not np.isclose(
        float(cfg.accuracy_tolerance), ACCURACY_TOLERANCE, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "single-adapter accuracy tolerance is frozen by the approved design"
        )
    return cfg


def preflight_summary(cfg: moe.MoEConfig) -> dict:
    cfg = validate_single_config(cfg)
    summary = moe.preflight_summary(cfg)
    summary.update(
        {
            "code_version": CODE_VERSION,
            "primary_model_id": PRIMARY_MODEL_ID,
            "required_controls": list(REQUIRED_CONTROLS),
            "mechanism_controls": list(MECHANISM_CONTROLS),
            "models": ["m1", *ALL_SINGLE_MODELS, "pref_continue"],
            "config": asdict(cfg),
        }
    )
    return summary


def _selected_revenue(
    table: pd.DataFrame, selected: dict[str, float], model_id: str
) -> float | None:
    if model_id not in selected:
        return None
    subset = table[
        table["model_id"].eq(model_id)
        & table["split"].eq("val")
        & table["seed"].eq(42)
        & np.isclose(
            table["lambda"].to_numpy(dtype=float), float(selected[model_id])
        )
    ]
    if subset.empty:
        return None
    return float(subset["revenue@10"].iloc[0])


def single_screening_decision(
    rows: list[dict],
    selected: dict[str, float],
    selection_success: dict[str, bool],
) -> dict:
    table = pd.DataFrame(rows)
    main = _selected_revenue(table, selected, PRIMARY_MODEL_ID)
    required = {
        control: _selected_revenue(table, selected, control)
        for control in REQUIRED_CONTROLS
    }
    mechanism = {
        control: _selected_revenue(table, selected, control)
        for control in MECHANISM_CONTROLS
    }
    failed = [
        control
        for control, revenue in required.items()
        if main is None or revenue is None or not main > revenue
    ]
    main_passed = bool(selection_success.get(PRIMARY_MODEL_ID, False))
    success = main_passed and not failed
    return {
        "success": success,
        "reason": (
            "single_full economically outperformed M1 and required controls"
            if success
            else "single_full improvement is absent or explained by a required control"
        ),
        "main_revenue@10": main,
        "required_control_revenue@10": required,
        "mechanism_comparison": mechanism,
        "failed_controls": failed,
    }


def _same_json_value(left, right) -> bool:
    return json.dumps(left, sort_keys=True, default=str) == json.dumps(
        right, sort_keys=True, default=str
    )


def _require_reuse_payload(payload: dict) -> None:
    required = {
        "source_revision",
        "input_manifest",
        "config",
        "base_config",
        "baseline_state_hashes",
        "feature_schema",
        "checkpoint_paths",
        "absolute_rows",
        "training",
        "moe_diagnostics",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"saved result JSON is missing fields: {sorted(missing)}")


def load_reusable_single_full(
    result_json: str | Path,
    *,
    current_manifest: dict,
    baseline_state_hash: str,
    cfg: moe.MoEConfig,
    base_cfg: dict,
    context: dict,
    data: dict,
) -> ReusableSingleFull:
    """Reuse a legacy single adapter only after exact provenance round-trip checks."""
    cfg = validate_single_config(cfg)
    result_path = Path(result_json)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"saved result JSON cannot be read: {result_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("saved result JSON must contain an object")
    _require_reuse_payload(payload)

    if payload["input_manifest"] != current_manifest:
        raise RuntimeError("input manifest mismatch; refusing single-adapter reuse")
    if payload["baseline_state_hashes"].get("42") != baseline_state_hash:
        raise RuntimeError("M1 state mismatch; refusing single-adapter reuse")

    current_config = asdict(cfg)
    saved_config = payload["config"]
    for key in REUSE_CONFIG_KEYS:
        if key not in saved_config or not _same_json_value(
            saved_config[key], current_config[key]
        ):
            raise RuntimeError(f"saved config mismatch for {key}")
    saved_base_config = payload["base_config"]
    for key in REUSE_BASE_CONFIG_KEYS:
        if key not in saved_base_config or key not in base_cfg or not _same_json_value(
            saved_base_config[key], base_cfg[key]
        ):
            raise RuntimeError(f"saved base config mismatch for {key}")

    expected_schema = {
        "user": list(context["user_profile"].feature_names),
        "item_numeric": list(context["item_profile"].numeric_names),
    }
    if payload["feature_schema"] != expected_schema:
        raise RuntimeError("feature schema mismatch; refusing single-adapter reuse")

    checkpoint_value = payload["checkpoint_paths"].get("single_adapter_s42")
    if not checkpoint_value:
        raise RuntimeError("single_adapter seed-42 checkpoint is missing")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise RuntimeError(f"single_adapter checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint_payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError) as error:
        raise RuntimeError("single_adapter checkpoint cannot be read") from error
    expected_arrays = {
        "user_profile": context["user_profile"].values,
        "user_valid": context["user_profile"].valid_user,
        "item_numeric": context["item_profile"].numeric,
        "item_category_ids": context["item_profile"].category_ids,
        "item_valid": context["item_profile"].valid_item,
    }
    for key, expected in expected_arrays.items():
        if key not in checkpoint_payload or array_sha256(
            checkpoint_payload[key]
        ) != array_sha256(expected):
            raise RuntimeError(
                f"checkpoint feature values mismatch for {key}; refusing reuse"
            )
    checkpoint_schema = {
        "user": list(checkpoint_payload.get("user_feature_names", ())),
        "item_numeric": list(checkpoint_payload.get("item_numeric_names", ())),
    }
    if checkpoint_schema != expected_schema or int(
        checkpoint_payload.get("item_n_categories", -1)
    ) != int(context["item_profile"].n_categories):
        raise RuntimeError("checkpoint feature schema mismatch; refusing reuse")
    if "ev_all" not in checkpoint_payload or array_sha256(
        checkpoint_payload["ev_all"]
    ) != array_sha256(context["artifact"].ev_all):
        raise RuntimeError("encoder ev_all hash mismatch")

    base_model = context.get("base_model", context.get("external_m1"))
    model = moe.load_moe_checkpoint(
        checkpoint_path,
        base_model,
        cfg,
        control="single_adapter",
        device=v3.DEVICE,
    )

    saved_rows = [
        row
        for row in payload["absolute_rows"]
        if row.get("seed") == 42
        and row.get("split") == "val"
        and row.get("model_id") == "single_adapter"
    ]
    by_lambda = {float(row["lambda"]): row for row in saved_rows}
    if len(saved_rows) != len(cfg.lambda_eval) or set(by_lambda) != set(
        map(float, cfg.lambda_eval)
    ):
        raise RuntimeError("saved single_adapter lambda curve is incomplete")

    relabeled_rows = []
    for lam in cfg.lambda_eval:
        flat, _ = moe._flat_evaluation(
            model,
            float(lam),
            context["caches"]["val"],
            context.get("meta"),
            data,
            base_cfg,
            per_user=False,
        )
        saved = by_lambda[float(lam)]
        for key, value in flat.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                if key not in saved or not np.isclose(
                    float(saved[key]), float(value), rtol=0.0, atol=5e-8
                ):
                    raise RuntimeError(
                        f"metric round-trip mismatch for lambda={lam}, metric={key}"
                    )
        relabeled_rows.append(saved | {"model_id": PRIMARY_MODEL_ID, "role": "model"})

    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    key = "single_adapter_s42"
    return ReusableSingleFull(
        model=model,
        rows=tuple(relabeled_rows),
        training=dict(payload["training"].get(key, {})),
        diagnostics=dict(payload["moe_diagnostics"].get(key, {})),
        result_json_sha256=result_sha,
        legacy_source_revision=str(payload["source_revision"]),
        legacy_checkpoint=str(checkpoint_path),
        legacy_checkpoint_sha256=moe.file_sha256(checkpoint_path),
    )


VARIANT_DEFINITIONS = {
    "single_full": {"user_profile": "observed", "item_features": "observed"},
    "single_zero_user": {"user_profile": "zero", "item_features": "observed"},
    "single_shuffled_user": {
        "user_profile": "seeded_valid-user permutation",
        "item_features": "observed",
    },
    "single_zero_item": {"user_profile": "observed", "item_features": "zero"},
    "single_base_only": {"user_profile": "zero", "item_features": "zero"},
}


def _single_result_fingerprint(
    cfg: moe.MoEConfig,
    base_cfg: dict,
    input_manifest: dict,
    baseline_state_hash: str,
    revision: str,
) -> str:
    base_keys = (
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
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": revision,
        "input_manifest_hash": moe.manifest_hash(input_manifest),
        "baseline_state_hash": baseline_state_hash,
        "config": asdict(cfg),
        "base": {key: base_cfg[key] for key in base_keys},
        "variants": VARIANT_DEFINITIONS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _prepare_validation_context(cfg: moe.MoEConfig) -> PreparedSingleContext:
    cfg = validate_single_config(cfg)
    out_dir = Path(cfg.out_dir or f"{v3.default_out_dir(cfg.dataset)}_clv_single")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_id = moe.manifest_hash(input_manifest)
    m1_root = Path(cfg.m1_checkpoint_dir or v3.default_out_dir(cfg.dataset))
    m1_dir = m1_root / f"data_{input_id[:12]}"
    base_cfg = moe._pure_m1_config(cfg, str(m1_dir))
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    revision = moe.source_revision()
    data = v3.prepare_data(base_cfg, v3.DCFG)

    encoder_cfg = moe._encoder_config(cfg)
    anchors = moe.residual.build_anchor_examples(
        data["train"],
        data["n_users"],
        v3.DCFG["is_date"],
        cfg.input_days,
        cfg.target_days,
        cfg.anchor_offsets,
    )
    snapshot = moe.residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    seed = 42
    artifact = moe.residual.train_future_value_encoder(
        anchors, snapshot, encoder_cfg, seed, v3.DEVICE
    )
    encoder_seed = hashlib.sha256(
        json.dumps(asdict(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    encoder_path = out_dir / f"encoder_{cfg.dataset}_s{seed}_{encoder_seed}.pt"
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
    user_profile = moe.compose_user_profiles(artifact, snapshot, v3.DEVICE)
    item_profile = moe.build_item_profiles(data["train"], data["n_items"])
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    ones_gate = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)
    thresholds = v3.segment_thresholds(artifact.ev_all, base_cfg["SEG_EDGES"])
    caches = {
        name: v3.EvalCache(
            gt,
            revenue,
            artifact.ev_all,
            thresholds,
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

    m1_checkpoint = Path(base_cfg["OUT_DIR"]) / (
        f"ckpt_pref_only_{cfg.dataset}_s{seed}_"
        f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', seed)}.pt"
    )
    existed_before = m1_checkpoint.exists()
    external_m1, _ = moe._fresh_external_m1(context, seed, data, base_cfg)
    baseline_hash = moe.state_hash(external_m1)
    if not m1_checkpoint.exists():
        raise RuntimeError(f"M1 checkpoint was not saved: {m1_checkpoint}")
    moe.validate_or_write_m1_manifest(
        m1_checkpoint,
        input_manifest,
        config_hash=v3.cfg_hash(base_cfg, v3.DCFG, "pref_only", seed),
        state_hash_value=baseline_hash,
        existed_before=existed_before,
    )
    context["base_model"] = external_m1
    baseline_flat, baseline_per_user = moe._flat_evaluation(
        external_m1,
        0.0,
        caches["val"],
        meta,
        data,
        base_cfg,
        per_user=True,
    )
    baseline_row = {
        "seed": seed,
        "model_id": "m1",
        "split": "val",
        "lambda": 0.0,
        "role": "baseline",
        **baseline_flat,
    }
    run_fingerprint = _single_result_fingerprint(
        cfg, base_cfg, input_manifest, baseline_hash, revision
    )
    return PreparedSingleContext(
        out_dir=out_dir,
        baseline_row=baseline_row,
        baseline_metrics=baseline_flat,
        baseline_per_user=baseline_per_user,
        input_manifest=input_manifest,
        baseline_state_hash=baseline_hash,
        base_cfg=base_cfg,
        data=data,
        context=context,
        source_revision=revision,
        encoder_checkpoint=str(encoder_path),
        m1_checkpoint=str(m1_checkpoint),
        run_fingerprint=run_fingerprint,
    )


def _train_evaluate_variant(
    prepared: PreparedSingleContext, cfg: moe.MoEConfig, model_id: str
) -> VariantRun:
    seed = 42
    external_m1, _ = moe._fresh_external_m1(
        prepared.context, seed, prepared.data, prepared.base_cfg
    )
    if moe.state_hash(external_m1) != prepared.baseline_state_hash:
        raise RuntimeError(f"{model_id} did not start from the shared M1 state")
    model = moe._build_model(external_m1, prepared.context, cfg, model_id, seed)
    encoder_hash = moe.state_hash(prepared.context["artifact"].model)

    def validation_recall(candidate):
        flat, _ = moe._flat_evaluation(
            candidate,
            cfg.lambda_train,
            prepared.context["caches"]["val"],
            prepared.context["meta"],
            prepared.data,
            prepared.base_cfg | {"K_LIST": [10]},
            per_user=False,
        )
        return flat["recall@10"]

    stats = moe.train_moe(
        model,
        prepared.data,
        prepared.base_cfg,
        cfg,
        seed,
        validation_recall,
        freeze_base=False,
    )
    if moe.state_hash(prepared.context["artifact"].model) != encoder_hash:
        raise RuntimeError("single-adapter training changed the frozen encoder")
    diagnostics = {
        **moe.moe_diagnostics(model, seed=seed),
        **_variant_audit(model, prepared.baseline_state_hash),
    }
    checkpoint = prepared.out_dir / (
        f"{model_id}_{cfg.dataset}_s{seed}_{prepared.run_fingerprint}.pt"
    )
    moe._save_model_checkpoint(
        checkpoint, model, prepared.context, stats, diagnostics
    )
    rows = []
    per_user = {}
    for lam in cfg.lambda_eval:
        flat, user_metrics = moe._flat_evaluation(
            model,
            float(lam),
            prepared.context["caches"]["val"],
            prepared.context["meta"],
            prepared.data,
            prepared.base_cfg,
            per_user=True,
        )
        per_user[float(lam)] = user_metrics
        rows.append(
            {
                "seed": seed,
                "model_id": model_id,
                "split": "val",
                "lambda": float(lam),
                "role": "model" if model_id == PRIMARY_MODEL_ID else "control",
                **_diagnostic_columns_for_lambda(diagnostics, float(lam)),
                **flat,
            }
        )
    return VariantRun(
        model_id=model_id,
        rows=tuple(rows),
        per_user=per_user,
        training=stats,
        diagnostics=diagnostics,
        checkpoint=str(checkpoint),
        reuse_provenance=None,
    )


def _reuse_or_train_full(
    prepared: PreparedSingleContext,
    cfg: moe.MoEConfig,
    reuse_full_result_json: str | Path | None,
) -> VariantRun:
    if reuse_full_result_json is None:
        return _train_evaluate_variant(prepared, cfg, PRIMARY_MODEL_ID)
    reused = load_reusable_single_full(
        reuse_full_result_json,
        current_manifest=prepared.input_manifest,
        baseline_state_hash=prepared.baseline_state_hash,
        cfg=cfg,
        base_cfg=prepared.base_cfg,
        context=prepared.context,
        data=prepared.data,
    )
    per_user = {}
    for lam in cfg.lambda_eval:
        _, user_metrics = moe._flat_evaluation(
            reused.model,
            float(lam),
            prepared.context["caches"]["val"],
            prepared.context["meta"],
            prepared.data,
            prepared.base_cfg,
            per_user=True,
        )
        per_user[float(lam)] = user_metrics
    provenance = {
        "legacy_result_json": str(Path(reuse_full_result_json)),
        "legacy_result_json_sha256": reused.result_json_sha256,
        "legacy_source_revision": reused.legacy_source_revision,
        "legacy_checkpoint": reused.legacy_checkpoint,
        "legacy_checkpoint_sha256": reused.legacy_checkpoint_sha256,
        "validation_metric_round_trip": True,
    }
    diagnostics = {
        **reused.diagnostics,
        **_variant_audit(reused.model, prepared.baseline_state_hash),
    }
    rows = tuple(
        row
        | _diagnostic_columns_for_lambda(diagnostics, float(row["lambda"]))
        for row in reused.rows
    )
    return VariantRun(
        model_id=PRIMARY_MODEL_ID,
        rows=rows,
        per_user=per_user,
        training=reused.training,
        diagnostics=diagnostics,
        checkpoint=reused.legacy_checkpoint,
        reuse_provenance=provenance,
    )


def _select_models(
    rows: list[dict], baseline: dict, model_ids: tuple[str, ...]
) -> tuple[dict, dict, dict]:
    table = pd.DataFrame(rows)
    selected = {}
    success = {}
    selection_tables = {}
    for model_id in model_ids:
        model_rows = table[
            table["model_id"].eq(model_id) & table["split"].eq("val")
        ]
        if model_rows.empty:
            selected[model_id] = 0.0
            success[model_id] = False
            selection_tables[model_id] = []
            continue
        mean_rows = (
            model_rows.groupby("lambda", as_index=False)
            .mean(numeric_only=True)
            .to_dict("records")
        )
        selected_lambda, selection = moe.select_lambda(
            mean_rows, baseline, tolerance=ACCURACY_TOLERANCE
        )
        selected[model_id] = float(selected_lambda)
        success[model_id] = bool(selection.attrs["success"])
        selection_tables[model_id] = selection.to_dict("records")
    return selected, success, selection_tables


def _run_pref_continue(
    prepared: PreparedSingleContext, cfg: moe.MoEConfig, full_training: dict
) -> dict | None:
    target_updates = full_training.get("base_updates_at_best")
    if target_updates is None:
        raise RuntimeError("single_full training record lacks base_updates_at_best")
    seed = 42
    external_m1, _ = moe._fresh_external_m1(
        prepared.context, seed, prepared.data, prepared.base_cfg
    )
    if moe.state_hash(external_m1) != prepared.baseline_state_hash:
        raise RuntimeError("pref_continue did not start from the shared M1 state")
    stats = moe.train_pref_continue(
        external_m1,
        prepared.data,
        prepared.base_cfg,
        cfg,
        seed,
        int(target_updates),
    )
    flat, per_user = moe._flat_evaluation(
        external_m1,
        0.0,
        prepared.context["caches"]["val"],
        prepared.context["meta"],
        prepared.data,
        prepared.base_cfg,
        per_user=True,
    )
    checkpoint = prepared.out_dir / (
        f"pref_continue_{cfg.dataset}_s{seed}_{prepared.run_fingerprint}.pt"
    )
    torch.save(
        {
            "state": external_m1.state_dict(),
            "training": stats,
            "source_revision": prepared.source_revision,
            "starting_m1_state_hash": prepared.baseline_state_hash,
        },
        checkpoint,
    )
    prepared.pref_continue_per_user = per_user
    prepared.pref_continue_training = stats
    prepared.pref_continue_checkpoint = str(checkpoint)
    return {
        "seed": seed,
        "model_id": "pref_continue",
        "split": "val",
        "lambda": 0.0,
        "role": "control",
        **flat,
    }


def _result_fingerprint_from_prepared(
    prepared: PreparedSingleContext, cfg: moe.MoEConfig
) -> str:
    existing = getattr(prepared, "run_fingerprint", None)
    if existing:
        return str(existing)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared.source_revision,
        "input_manifest": prepared.input_manifest,
        "baseline_state_hash": prepared.baseline_state_hash,
        "config": asdict(cfg),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _persist_result(
    prepared: PreparedSingleContext,
    cfg: moe.MoEConfig,
    rows: list[dict],
    selected: dict,
    selection_success: dict,
    selection_tables: dict,
    decision: dict,
    full: VariantRun,
    controls: dict[str, VariantRun],
) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    delta_records = []
    runs = {PRIMARY_MODEL_ID: full, **controls}
    delta_inputs = [
        (model_id, float(lam), per_user)
        for model_id, run in runs.items()
        for lam, per_user in sorted(run.per_user.items())
    ]
    pref_per_user = getattr(prepared, "pref_continue_per_user", None)
    if pref_per_user is not None:
        delta_inputs.append(("pref_continue", 0.0, pref_per_user))
    for model_id, lam, per_user in delta_inputs:
        for metric in ("recall", "ndcg", "revenue", "arp"):
            diff = per_user[metric] - prepared.baseline_per_user[metric]
            delta_records.append(
                {
                    "model_id": model_id,
                    "split": "val",
                    "lambda": lam,
                    "metric": metric,
                    **v3.paired_bootstrap([diff], prepared.base_cfg["N_BOOT"]),
                }
            )

    fingerprint = _result_fingerprint_from_prepared(prepared, cfg)
    stem = f"clv_single_{cfg.dataset}_{fingerprint}"
    prepared.out_dir.mkdir(parents=True, exist_ok=True)
    result_csv = prepared.out_dir / f"{stem}.csv"
    delta_csv = prepared.out_dir / f"{stem}_delta.csv"
    result_json = prepared.out_dir / f"{stem}.json"
    frame.attrs["screening_decision"] = decision
    frame.attrs["selected_lambda"] = selected
    frame.attrs["lambda_selection_success"] = selection_success
    frame.attrs["result_paths"] = {
        "csv": str(result_csv),
        "delta_csv": str(delta_csv),
        "json": str(result_json),
    }
    frame.to_csv(result_csv, index=False, float_format="%.8f")
    pd.DataFrame(delta_records).to_csv(delta_csv, index=False)

    training = {
        f"{model_id}_s42": run.training for model_id, run in runs.items()
    }
    diagnostics = {
        f"{model_id}_s42": run.diagnostics for model_id, run in runs.items()
    }
    checkpoints = {
        f"{model_id}_s42": run.checkpoint for model_id, run in runs.items()
    }
    encoder_checkpoint = getattr(prepared, "encoder_checkpoint", None)
    m1_checkpoint = getattr(prepared, "m1_checkpoint", None)
    if encoder_checkpoint:
        checkpoints["encoder_s42"] = encoder_checkpoint
    if m1_checkpoint:
        checkpoints["m1_s42"] = m1_checkpoint
    pref_checkpoint = getattr(prepared, "pref_continue_checkpoint", None)
    if pref_checkpoint:
        checkpoints["pref_continue_s42"] = pref_checkpoint
        training["pref_continue_s42"] = prepared.pref_continue_training
    checkpoint_sha256 = {
        key: moe.file_sha256(path)
        for key, path in checkpoints.items()
        if Path(path).is_file()
    }
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared.source_revision,
        "result_fingerprint": fingerprint,
        "input_manifest": prepared.input_manifest,
        "config": asdict(cfg),
        "base_config": {
            key: value for key, value in prepared.base_cfg.items() if key != "OUT_DIR"
        },
        "data_stats": prepared.data.get("data_stats", {}),
        "feature_schema": {
            "user": list(prepared.context["user_profile"].feature_names),
            "item_numeric": list(prepared.context["item_profile"].numeric_names),
        },
        "variant_definitions": VARIANT_DEFINITIONS,
        "baseline_state_hashes": {"42": prepared.baseline_state_hash},
        "selected_lambda": selected,
        "lambda_selection_success": selection_success,
        "screening_decision": decision,
        "selection_tables": selection_tables,
        "encoder_diagnostics": {
            "42": getattr(prepared.context["artifact"], "diagnostics", {})
        },
        "training": training,
        "diagnostics": diagnostics,
        "checkpoint_paths": checkpoints,
        "checkpoint_sha256": checkpoint_sha256,
        "reuse_provenance": full.reuse_provenance,
        "absolute_rows": frame.to_dict("records"),
        "delta": delta_records,
        "interpretation": {
            "clv": "train-only CLV-related behavior representation; not realized lifetime CLV",
            "revenue": "price/purchase-amount weighted hit; not incremental revenue",
        },
    }
    with result_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
    print(f"저장: {result_csv}")
    print(f"validation 선택 λ: {selected}")
    print(f"λ 선택 단계 통과: {selection_success}")
    print(
        "최종 screening 판정: "
        f"success={decision['success']} | reason={decision['reason']} | "
        f"failed_controls={decision['failed_controls']}"
    )
    return frame


def run_experiment(
    cfg: moe.MoEConfig | None = None,
    *,
    reuse_full_result_json: str | Path | None = None,
) -> pd.DataFrame:
    cfg = validate_single_config(cfg or configure_single_run("dunnhumby"))
    prepared = _prepare_validation_context(cfg)
    rows = [prepared.baseline_row]
    full = _reuse_or_train_full(prepared, cfg, reuse_full_result_json)
    rows.extend(full.rows)
    selected, selection_success, selection_tables = _select_models(
        rows, prepared.baseline_metrics, (PRIMARY_MODEL_ID,)
    )
    controls = {}
    if selection_success[PRIMARY_MODEL_ID] and cfg.run_controls_after_success:
        for model_id in (*REQUIRED_CONTROLS, *MECHANISM_CONTROLS):
            controls[model_id] = _train_evaluate_variant(prepared, cfg, model_id)
            rows.extend(controls[model_id].rows)
        selected, selection_success, selection_tables = _select_models(
            rows, prepared.baseline_metrics, ALL_SINGLE_MODELS
        )
        pref_row = _run_pref_continue(prepared, cfg, full.training)
        if pref_row is not None:
            rows.append(pref_row)
        selected["pref_continue"] = 0.0
        selection_success["pref_continue"] = True
    decision = single_screening_decision(rows, selected, selection_success)
    return _persist_result(
        prepared,
        cfg,
        rows,
        selected,
        selection_success,
        selection_tables,
        decision,
        full,
        controls,
    )


def main_cli() -> None:
    print(
        json.dumps(
            preflight_summary(configure_single_run("dunnhumby")),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("고비용 학습은 검토된 Colab 승인 셀에서만 실행하세요.")


if __name__ == "__main__":
    main_cli()
