"""Seed-42 validation runner for fixed-gate dual-axis CLV embeddings."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_core_features as core
import lightgcn_clv_moe as moe
import lightgcn_clv_single as single
import lightgcn_clv_v3 as v3
from clv_dual_axis_model import (
    CLVDualAxisEmbeddingModel,
    build_dual_item_profiles,
    fixed_percentile_gates,
)


CODE_VERSION = "clv-dual-axis-fixed-v1.0"
PRIMARY_MODEL = "dual_clv_fixed"
CONTROLS = ("dual_shuffled_gate", "dual_base_only")
MODELS = ("m1", PRIMARY_MODEL, *CONTROLS)
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
ACCURACY_TOLERANCE = 0.01
HM_60DAY = {
    "window_days": 60,
    "input_days": 14,
    "target_days": 7,
    "anchor_offsets": (21, 14, 7),
}


def configure_dual_run(
    dataset: str, *, short_hm: bool = False, **overrides
) -> moe.MoEConfig:
    dataset = dataset.lower()
    if short_hm and dataset != "hm":
        raise ValueError("short_hm은 H&M에서만 사용할 수 있습니다")
    suffix = "hm_w60" if short_hm else dataset
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "lambda_eval": LAMBDA_GRID,
        "accuracy_tolerance": ACCURACY_TOLERANCE,
        "run_controls_after_success": True,
        "out_dir": f"{v3.default_out_dir(dataset)}_clv_dual_{suffix}",
        **(HM_60DAY if short_hm else {}),
    }
    return validate_dual_config(
        moe.configure_moe_run(dataset, **(defaults | overrides))
    )


def validate_dual_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("dual-axis runner는 seed 42 validation-only입니다")
    if tuple(cfg.seed_list) != (42,):
        raise ValueError("dual-axis screening은 seed 42 하나만 허용합니다")
    if tuple(cfg.lambda_eval) != LAMBDA_GRID:
        raise ValueError("dual-axis lambda grid는 승인 설계로 고정됩니다")
    if not np.isclose(cfg.accuracy_tolerance, ACCURACY_TOLERANCE):
        raise ValueError("dual-axis accuracy tolerance는 1%로 고정됩니다")
    return moe.validate_moe_config(cfg)


def preflight_summary(cfg: moe.MoEConfig) -> dict:
    cfg = validate_dual_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed_list": list(cfg.seed_list),
        "window_days": cfg.window_days,
        "input_days": cfg.input_days,
        "target_days": cfg.target_days,
        "anchor_offsets": list(cfg.anchor_offsets),
        "models": list(MODELS),
        "lambda_eval": list(cfg.lambda_eval),
        "base_frozen": True,
        "graph_mode": "binary",
        "loss_mode": "plain_bpr",
        "negative_sampling": "uniform",
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
        "m1_checkpoint_dir": cfg.m1_checkpoint_dir,
    }


def _selected_revenue(rows, selected, model_id):
    if model_id not in selected:
        return None
    table = pd.DataFrame(rows)
    match = table[
        table.model_id.eq(model_id)
        & np.isclose(table["lambda"].to_numpy(float), selected[model_id])
    ]
    return None if match.empty else float(match["revenue@10"].iloc[0])


def screening_decision(rows, selected, selection_success):
    if not selection_success.get(PRIMARY_MODEL, False):
        return {
            "success": False,
            "reason": "dual_clv_fixed did not improve M1 under accuracy guardrails",
            "main_revenue@10": _selected_revenue(rows, selected, PRIMARY_MODEL),
            "control_revenue@10": {control: None for control in CONTROLS},
            "failed_controls": list(CONTROLS),
        }
    main = _selected_revenue(rows, selected, PRIMARY_MODEL)
    control_revenue = {
        control: _selected_revenue(rows, selected, control) for control in CONTROLS
    }
    failed = [
        control
        for control, revenue in control_revenue.items()
        if revenue is None or main is None or not main > revenue
    ]
    return {
        "success": not failed,
        "reason": (
            "dual_clv_fixed improved M1 and outperformed both required controls"
            if not failed
            else "dual_clv_fixed improvement is absent or explained by a required control"
        ),
        "main_revenue@10": main,
        "control_revenue@10": control_revenue,
        "failed_controls": failed,
    }


def _select(rows, baseline, model_ids):
    frame = pd.DataFrame(rows)
    selected, success, tables = {}, {}, {}
    for model_id in model_ids:
        candidates = frame[frame.model_id.eq(model_id)]
        if candidates.empty:
            selected[model_id], success[model_id], tables[model_id] = 0.0, False, []
            continue
        lam, table = moe.select_lambda(
            candidates.to_dict("records"), baseline, ACCURACY_TOLERANCE
        )
        selected[model_id] = float(lam)
        success[model_id] = bool(table.attrs["success"])
        tables[model_id] = table.to_dict("records")
    return selected, success, tables


def _fingerprint(cfg, manifest, baseline_hash, revision):
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_manifest_hash": moe.manifest_hash(manifest),
        "baseline_state_hash": baseline_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]


def _diagnostics(model, artifact):
    return {
        "gate_n_mean": float(model.g_n[model.has_profile].mean()),
        "gate_v_mean": float(model.g_v[model.has_profile].mean()),
        "gate_n_std": float(model.g_n[model.has_profile].std()),
        "gate_v_std": float(model.g_v[model.has_profile].std()),
        "n_hat_sha256": single.array_sha256(artifact.n_hat_all),
        "v_hat_sha256": single.array_sha256(artifact.v_hat_all),
        "clv_proxy_sha256": single.array_sha256(artifact.ev_all),
        "gate_n_sha256": single.array_sha256(model.g_n.cpu().numpy()),
        "gate_v_sha256": single.array_sha256(model.g_v.cpu().numpy()),
        "adapter_parameter_count": int(
            sum(parameter.numel() for parameter in model.adapter_parameters())
        ),
    }


def _train_variant(model_id, base_model, prepared, cfg):
    model = CLVDualAxisEmbeddingModel(
        base_model,
        prepared["user_profile"],
        prepared["item_profile"],
        prepared["g_n"],
        prepared["g_v"],
        control=model_id,
        seed=42,
        hidden_dim=cfg.expert_hidden_dim,
        expert_dim=cfg.expert_dim,
    ).to(v3.DEVICE)

    def validation_recall(candidate):
        flat, _ = moe._flat_evaluation(
            candidate,
            cfg.lambda_train,
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"] | {"K_LIST": [10]},
            per_user=False,
        )
        return flat["recall@10"]

    training = moe.train_moe(
        model,
        prepared["data"],
        prepared["base_cfg"],
        cfg,
        42,
        validation_recall,
        freeze_base=True,
    )
    diagnostics = _diagnostics(model, prepared["artifact"])
    checkpoint = prepared["out_dir"] / (
        f"{model_id}_{cfg.dataset}_s42_{prepared['fingerprint']}.pt"
    )
    torch.save(
        {
            "state": model.state_dict(),
            "training": training,
            "diagnostics": diagnostics,
            "user_feature_names": prepared["user_profile"].feature_names,
            "item_activity_names": prepared["item_profile"].activity_names,
            "item_value_names": prepared["item_profile"].value_names,
            "source_revision": prepared["revision"],
            "baseline_state_hash": prepared["baseline_hash"],
        },
        checkpoint,
    )
    rows, per_user = [], {}
    for lam in cfg.lambda_eval:
        flat, user_metrics = moe._flat_evaluation(
            model,
            float(lam),
            prepared["cache"],
            prepared["meta"],
            prepared["data"],
            prepared["base_cfg"],
            per_user=True,
        )
        rows.append(
            {
                "seed": 42,
                "model_id": model_id,
                "split": "val",
                "lambda": float(lam),
                "role": "model" if model_id == PRIMARY_MODEL else "control",
                **flat,
            }
        )
        per_user[float(lam)] = user_metrics
    return {
        "model": model,
        "rows": rows,
        "per_user": per_user,
        "training": training,
        "diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
    }


def _prepare(cfg):
    out_dir = Path(cfg.out_dir or f"{v3.default_out_dir(cfg.dataset)}_clv_dual")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_id = moe.manifest_hash(manifest)
    m1_root = Path(cfg.m1_checkpoint_dir or v3.default_out_dir(cfg.dataset))
    base_cfg = moe._pure_m1_config(cfg, str(m1_root / f"data_{input_id[:12]}"))
    revision = moe.source_revision()
    data = v3.prepare_data(base_cfg, v3.DCFG)
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
    artifact = core.train_clv_core_encoder(
        anchors,
        snapshot,
        encoder_epochs=cfg.encoder_epochs,
        encoder_patience=cfg.encoder_patience,
        encoder_batch_size=cfg.encoder_batch_size,
        encoder_lr=cfg.encoder_lr,
        seed=42,
        device=v3.DEVICE,
    )
    user_profile = core.compose_clv_core_profiles(artifact, snapshot, v3.DEVICE)
    item_profile = build_dual_item_profiles(
        data["train"], data["n_items"], v3.DCFG["is_date"]
    )
    g_n, g_v = fixed_percentile_gates(
        artifact.n_hat_all, artifact.v_hat_all, user_profile.valid_user
    )
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(artifact.ev_all, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["val"], artifact.ev_all, thresholds, data["n_items"]
    )
    base_context = {
        "ones_gate": torch.ones(data["n_users"], device=v3.DEVICE),
        "x_item": x_item,
        "item_cat": item_cat,
        "meta": meta,
        "caches": {"val": cache},
    }
    m1_checkpoint = Path(base_cfg["OUT_DIR"]) / (
        f"ckpt_pref_only_{cfg.dataset}_s42_"
        f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', 42)}.pt"
    )
    existed = m1_checkpoint.exists()
    baseline_model, _ = moe._fresh_external_m1(base_context, 42, data, base_cfg)
    baseline_hash = moe.state_hash(baseline_model)
    moe.validate_or_write_m1_manifest(
        m1_checkpoint,
        manifest,
        config_hash=v3.cfg_hash(base_cfg, v3.DCFG, "pref_only", 42),
        state_hash_value=baseline_hash,
        existed_before=existed,
    )
    baseline_flat, baseline_per_user = moe._flat_evaluation(
        baseline_model, 0.0, cache, meta, data, base_cfg, per_user=True
    )
    fingerprint = _fingerprint(cfg, manifest, baseline_hash, revision)
    encoder_checkpoint = out_dir / f"clv_core_encoder_{cfg.dataset}_s42_{fingerprint}.pt"
    torch.save(
        {
            "state": artifact.model.state_dict(),
            "transform_mean": artifact.transform.mean,
            "transform_std": artifact.transform.std,
            "feature_names": artifact.transform.feature_names,
            "n_hat_all": artifact.n_hat_all,
            "v_hat_all": artifact.v_hat_all,
            "clv_proxy_all": artifact.ev_all,
            "diagnostics": artifact.diagnostics,
            "source_revision": revision,
        },
        encoder_checkpoint,
    )
    return {
        "out_dir": out_dir,
        "manifest": manifest,
        "base_cfg": base_cfg,
        "data": data,
        "artifact": artifact,
        "user_profile": user_profile,
        "item_profile": item_profile,
        "g_n": g_n,
        "g_v": g_v,
        "meta": meta,
        "cache": cache,
        "base_context": base_context,
        "baseline_model": baseline_model,
        "baseline_hash": baseline_hash,
        "baseline_flat": baseline_flat,
        "baseline_per_user": baseline_per_user,
        "m1_checkpoint": str(m1_checkpoint),
        "encoder_checkpoint": str(encoder_checkpoint),
        "revision": revision,
        "fingerprint": fingerprint,
    }


def _fresh_base(prepared):
    model, _ = moe._fresh_external_m1(
        prepared["base_context"], 42, prepared["data"], prepared["base_cfg"]
    )
    if moe.state_hash(model) != prepared["baseline_hash"]:
        raise RuntimeError("dual variant가 공통 M1 state에서 시작하지 않았습니다")
    return model


def _persist(cfg, prepared, rows, runs, selected, success, tables, decision):
    frame = pd.DataFrame(rows)
    delta_records = []
    for model_id, run in runs.items():
        for lam, per_user in run["per_user"].items():
            for metric in ("recall", "ndcg", "revenue", "arp"):
                diff = per_user[metric] - prepared["baseline_per_user"][metric]
                delta_records.append(
                    {
                        "model_id": model_id,
                        "split": "val",
                        "lambda": float(lam),
                        "metric": metric,
                        **v3.paired_bootstrap([diff], prepared["base_cfg"]["N_BOOT"]),
                    }
                )
    stem = f"clv_dual_{cfg.dataset}_{prepared['fingerprint']}"
    csv_path = prepared["out_dir"] / f"{stem}.csv"
    delta_path = prepared["out_dir"] / f"{stem}_delta.csv"
    json_path = prepared["out_dir"] / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_records).to_csv(delta_path, index=False)
    checkpoints = {
        "m1_s42": prepared["m1_checkpoint"],
        "encoder_s42": prepared["encoder_checkpoint"],
        **{f"{name}_s42": run["checkpoint"] for name, run in runs.items()},
    }
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "result_fingerprint": prepared["fingerprint"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "base_config": {
            key: value
            for key, value in prepared["base_cfg"].items()
            if key != "OUT_DIR"
        },
        "data_stats": prepared["data"].get("data_stats", {}),
        "models": list(MODELS),
        "feature_schema": {
            "user": list(prepared["user_profile"].feature_names),
            "item_activity": list(prepared["item_profile"].activity_names),
            "item_value": list(prepared["item_profile"].value_names),
        },
        "encoder_diagnostics": prepared["artifact"].diagnostics,
        "selected_lambda": selected,
        "lambda_selection_success": success,
        "selection_tables": tables,
        "screening_decision": decision,
        "training": {name: run["training"] for name, run in runs.items()},
        "diagnostics": {name: run["diagnostics"] for name, run in runs.items()},
        "checkpoint_paths": checkpoints,
        "checkpoint_sha256": {
            name: moe.file_sha256(path) for name, path in checkpoints.items()
        },
        "absolute_rows": frame.to_dict("records"),
        "delta": delta_records,
        "interpretation": {
            "clv": "fixed-horizon revenue-based CLV proxy, not realized lifetime CLV",
            "item": "activity/economic item attributes, not item CLV",
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["screening_decision"] = decision
    frame.attrs["selected_lambda"] = selected
    frame.attrs["lambda_selection_success"] = success
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "json": str(json_path),
    }
    print(f"저장: {json_path}")
    print(f"선택 lambda: {selected}")
    print(f"최종 screening 판정: {decision}")
    return frame


def run_experiment(cfg: moe.MoEConfig | None = None) -> pd.DataFrame:
    cfg = validate_dual_config(cfg or configure_dual_run("dunnhumby"))
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "lambda": 0.0,
            "role": "baseline",
            **prepared["baseline_flat"],
        }
    ]
    runs = {
        PRIMARY_MODEL: _train_variant(
            PRIMARY_MODEL, _fresh_base(prepared), prepared, cfg
        )
    }
    rows.extend(runs[PRIMARY_MODEL]["rows"])
    for control in CONTROLS:
        runs[control] = _train_variant(
            control, _fresh_base(prepared), prepared, cfg
        )
        rows.extend(runs[control]["rows"])
    selected, success, tables = _select(
        rows, prepared["baseline_flat"], (PRIMARY_MODEL, *CONTROLS)
    )
    decision = screening_decision(rows, selected, success)
    return _persist(
        cfg, prepared, rows, runs, selected, success, tables, decision
    )


def main_cli():
    print(
        json.dumps(
            preflight_summary(configure_dual_run("dunnhumby")),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("학습은 Colab에서만 시작하세요.")


if __name__ == "__main__":
    main_cli()
