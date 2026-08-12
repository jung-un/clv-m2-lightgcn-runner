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
    GATE_SHAPES,
    build_dual_item_profiles,
    fixed_percentile_ranks,
)


CODE_VERSION = "clv-dual-axis-fixed-v1.1"
PRIMARY_MODEL = "dual_clv_fixed"
CONTROLS = ("dual_shuffled_user", "dual_adapter_only")
MODELS = ("m1", PRIMARY_MODEL, *CONTROLS)
LAMBDA_GRID = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
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
        "gate_shapes": list(GATE_SHAPES),
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


def _accuracy_eligible(table: pd.DataFrame, baseline: dict) -> np.ndarray:
    eligible = np.ones(len(table), dtype=bool)
    for metric in ("recall", "ndcg"):
        for k in (10, 20, 50):
            key = f"{metric}@{k}"
            if key not in table or key not in baseline:
                raise KeyError(f"선택에 필요한 정확도 지표 누락: {key}")
            eligible &= table[key].to_numpy(float) >= float(baseline[key]) * (
                1.0 - ACCURACY_TOLERANCE
            )
    return eligible


def select_primary_operating_point(rows, baseline):
    """Select only the proposed model; controls never choose their own maxima."""
    table = pd.DataFrame(rows)
    table = table[table.model_id.eq(PRIMARY_MODEL)].copy()
    table["eligible"] = _accuracy_eligible(table, baseline)
    candidates = table[
        table.eligible
        & table["lambda"].gt(0)
        & table["revenue@10"].gt(float(baseline["revenue@10"]))
    ]
    if candidates.empty:
        return {
            "gate_shape": "equal",
            "lambda": 0.0,
            "revenue@10": float(baseline["revenue@10"]),
            "effective_strength": 0.0,
        }, False, table
    gate_order = {name: index for index, name in enumerate(GATE_SHAPES)}
    candidates = candidates.assign(
        _gate_order=candidates.gate_shape.map(gate_order)
    ).sort_values(
        ["revenue@10", "lambda", "_gate_order"],
        ascending=[False, True, True],
    )
    best = candidates.iloc[0]
    return {
        "gate_shape": str(best.gate_shape),
        "lambda": float(best["lambda"]),
        "revenue@10": float(best["revenue@10"]),
        "effective_strength": float(best["effective_strength"]),
    }, True, table


def _matched_row(table, model_id, shape, lam):
    match = table[
        table.model_id.eq(model_id)
        & table.gate_shape.eq(shape)
        & np.isclose(table["lambda"].to_numpy(float), float(lam))
    ]
    return None if match.empty else match.iloc[0]


def screening_decision(rows, selected, selection_success, selection_table):
    if not selection_success:
        return {
            "success": False,
            "reason": "dual_clv_fixed did not improve M1 under accuracy guardrails",
            "selected_operating_point": selected,
            "failed_controls": list(CONTROLS),
            "comparisons": {},
        }
    table = pd.DataFrame(rows)
    shape = selected["gate_shape"]
    selected_lam = selected["lambda"]
    main_points = selection_table[
        selection_table.eligible
        & selection_table.gate_shape.eq(shape)
        & selection_table["lambda"].gt(0)
    ]
    comparisons, failed = {}, []
    selected_main = _matched_row(table, PRIMARY_MODEL, shape, selected_lam)
    for control in CONTROLS:
        selected_control = _matched_row(table, control, shape, selected_lam)
        same_lambda_all, matched_strength_all = True, True
        control_curve = table[
            table.model_id.eq(control)
            & table.gate_shape.eq(shape)
            & table["lambda"].gt(0)
        ]
        for _, main_row in main_points.iterrows():
            same = _matched_row(table, control, shape, main_row["lambda"])
            same_lambda_all &= bool(
                same is not None and main_row["revenue@10"] > same["revenue@10"]
            )
            if control_curve.empty:
                matched_strength_all = False
            else:
                nearest = control_curve.iloc[
                    np.abs(
                        control_curve.effective_strength.to_numpy(float)
                        - float(main_row.effective_strength)
                    ).argmin()
                ]
                matched_strength_all &= bool(
                    main_row["revenue@10"] > nearest["revenue@10"]
                )
        passed = bool(
            selected_control is not None
            and selected_main["revenue@10"] > selected_control["revenue@10"]
            and same_lambda_all
            and matched_strength_all
        )
        if not passed:
            failed.append(control)
        comparisons[control] = {
            "same_lambda_revenue": (
                None
                if selected_control is None
                else float(selected_control["revenue@10"])
            ),
            "same_lambda_all_eligible_points": same_lambda_all,
            "matched_strength_all_eligible_points": matched_strength_all,
            "passed": passed,
        }
    return {
        "success": not failed,
        "reason": (
            "dual_clv_fixed improved M1 and dominated both controls on matched curves"
            if not failed
            else "dual_clv_fixed improvement is absent or explained by a required control"
        ),
        "selected_operating_point": selected,
        "main_revenue@10": float(selected_main["revenue@10"]),
        "comparisons": comparisons,
        "failed_controls": failed,
    }


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


def _safe_corr(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def axis_preflight_diagnostics(
    *,
    transaction_targets,
    n_hat,
    v_hat,
    q_n,
    q_v,
    valid,
    user_repeat_gap_valid,
    item_repeat_gap_valid,
):
    targets = np.concatenate([np.asarray(values, float) for values in transaction_targets])
    valid = np.asarray(valid, bool)
    n_hat, v_hat = np.asarray(n_hat, float)[valid], np.asarray(v_hat, float)[valid]
    q_n, q_v = np.asarray(q_n, float)[valid], np.asarray(q_v, float)[valid]
    rounded_q = np.round(q_n, 6)
    _, counts = np.unique(rounded_q, return_counts=True)
    ge2_share = float(np.mean(targets >= 2))
    max_tie = float(counts.max() / len(rounded_q)) if len(rounded_q) else 1.0
    return {
        "future_transactions_zero_share": float(np.mean(targets == 0)),
        "future_transactions_one_share": float(np.mean(targets == 1)),
        "future_transactions_ge2_share": ge2_share,
        "future_transactions_std": float(np.std(targets)),
        "n_hat_std": float(np.std(n_hat)) if len(n_hat) else float("nan"),
        "q_n_unique_count": int(len(np.unique(rounded_q))),
        "q_n_max_tie_share": max_tie,
        "q_n_q_v_corr": _safe_corr(q_n, q_v),
        "n_hat_v_hat_corr": _safe_corr(n_hat, v_hat),
        "user_repeat_gap_valid_share": float(np.mean(user_repeat_gap_valid)),
        "item_repeat_gap_valid_share": float(np.mean(item_repeat_gap_valid)),
        "n_axis_warning": bool(ge2_share < 0.01 or max_tie > 0.9),
    }


def _diagnostics(model, artifact):
    gate_diagnostics = {
        shape: model.axis_diagnostics(shape) for shape in GATE_SHAPES
    }
    return {
        "q_n_mean": float(model.q_n[model.has_profile].mean()),
        "q_v_mean": float(model.q_v[model.has_profile].mean()),
        "q_n_std": float(model.q_n[model.has_profile].std()),
        "q_v_std": float(model.q_v[model.has_profile].std()),
        "n_hat_sha256": single.array_sha256(artifact.n_hat_all),
        "v_hat_sha256": single.array_sha256(artifact.v_hat_all),
        "clv_proxy_sha256": single.array_sha256(artifact.ev_all),
        "q_n_sha256": single.array_sha256(model.q_n.cpu().numpy()),
        "q_v_sha256": single.array_sha256(model.q_v.cpu().numpy()),
        "gate_shape_diagnostics": gate_diagnostics,
        "adapter_parameter_count": int(
            sum(parameter.numel() for parameter in model.adapter_parameters())
        ),
    }


def _train_variant(
    model_id,
    base_model,
    prepared,
    cfg,
    *,
    seed=42,
    gate_shapes=GATE_SHAPES,
    lambda_eval=None,
):
    model = CLVDualAxisEmbeddingModel(
        base_model,
        prepared["user_profile"],
        prepared["item_profile"],
        prepared["q_n"],
        prepared["q_v"],
        control=model_id,
        seed=seed,
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
        seed,
        validation_recall,
        freeze_base=True,
    )
    diagnostics = _diagnostics(model, prepared["artifact"])
    checkpoint = prepared["out_dir"] / (
        f"{model_id}_{cfg.dataset}_s{seed}_{prepared['fingerprint']}.pt"
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
    lambdas = tuple(cfg.lambda_eval if lambda_eval is None else lambda_eval)
    for gate_shape in tuple(gate_shapes):
        model.set_gate_shape(gate_shape)
        axis_diag = diagnostics["gate_shape_diagnostics"][gate_shape]
        for lam in lambdas:
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
                    "seed": seed,
                    "model_id": model_id,
                    "split": "val",
                    "gate_shape": gate_shape,
                    "lambda": float(lam),
                    "role": "model" if model_id == PRIMARY_MODEL else "control",
                    **axis_diag,
                    "effective_strength": float(lam)
                    * axis_diag["effective_total_ratio"],
                    **flat,
                }
            )
            per_user[(gate_shape, float(lam))] = user_metrics
    return {
        "model": model,
        "rows": rows,
        "per_user": per_user,
        "training": training,
        "diagnostics": diagnostics,
        "checkpoint": str(checkpoint),
    }


def _prepare(cfg, seed=42):
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
        seed=seed,
        device=v3.DEVICE,
    )
    user_profile = core.compose_clv_core_profiles(artifact, snapshot, v3.DEVICE)
    item_profile = build_dual_item_profiles(
        data["train"], data["n_items"], v3.DCFG["is_date"]
    )
    q_n, q_v = fixed_percentile_ranks(
        artifact.n_hat_all, artifact.v_hat_all, user_profile.valid_user
    )
    gap_index = moe.residual.NUMERIC_FEATURES.index("gap_mean")
    item_gap_index = item_profile.activity_names.index("repeat_gap_valid")
    axis_preflight = axis_preflight_diagnostics(
        transaction_targets=[anchor.transaction_target for anchor in anchors.anchors],
        n_hat=artifact.n_hat_all,
        v_hat=artifact.v_hat_all,
        q_n=q_n,
        q_v=q_v,
        valid=user_profile.valid_user,
        user_repeat_gap_valid=snapshot.valid[:, gap_index],
        item_repeat_gap_valid=item_profile.activity[
            item_profile.valid_item, item_gap_index
        ],
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
        f"ckpt_pref_only_{cfg.dataset}_s{seed}_"
        f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', seed)}.pt"
    )
    existed = m1_checkpoint.exists()
    baseline_model, _ = moe._fresh_external_m1(
        base_context, seed, data, base_cfg
    )
    baseline_hash = moe.state_hash(baseline_model)
    moe.validate_or_write_m1_manifest(
        m1_checkpoint,
        manifest,
        config_hash=v3.cfg_hash(base_cfg, v3.DCFG, "pref_only", seed),
        state_hash_value=baseline_hash,
        existed_before=existed,
    )
    baseline_flat, baseline_per_user = moe._flat_evaluation(
        baseline_model, 0.0, cache, meta, data, base_cfg, per_user=True
    )
    fingerprint = _fingerprint(cfg, manifest, baseline_hash, revision)
    encoder_checkpoint = out_dir / (
        f"clv_core_encoder_{cfg.dataset}_s{seed}_{fingerprint}.pt"
    )
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
        "q_n": q_n,
        "q_v": q_v,
        "axis_preflight": axis_preflight,
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


def _fresh_base(prepared, seed=42):
    model, _ = moe._fresh_external_m1(
        prepared["base_context"], seed, prepared["data"], prepared["base_cfg"]
    )
    if moe.state_hash(model) != prepared["baseline_hash"]:
        raise RuntimeError("dual variant가 공통 M1 state에서 시작하지 않았습니다")
    return model


def _persist(cfg, prepared, rows, runs, selected, success, table, decision):
    frame = pd.DataFrame(rows)
    delta_records = []
    for model_id, run in runs.items():
        for (gate_shape, lam), per_user in run["per_user"].items():
            for metric in ("recall", "ndcg", "revenue", "arp"):
                diff = per_user[metric] - prepared["baseline_per_user"][metric]
                delta_records.append(
                    {
                        "model_id": model_id,
                        "split": "val",
                        "gate_shape": gate_shape,
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
        "axis_preflight": prepared["axis_preflight"],
        "selected_operating_point": selected,
        "lambda_selection_success": success,
        "selection_table": table.to_dict("records"),
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
    frame.attrs["selected_lambda"] = {PRIMARY_MODEL: selected["lambda"]}
    frame.attrs["selected_operating_point"] = selected
    frame.attrs["lambda_selection_success"] = success
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "json": str(json_path),
    }
    print(f"저장: {json_path}")
    print(f"선택 운영점: {selected}")
    print(f"최종 screening 판정: {decision}")
    return frame


def run_experiment(cfg: moe.MoEConfig | None = None) -> pd.DataFrame:
    cfg = validate_dual_config(cfg or configure_dual_run("dunnhumby"))
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("학습 전 이중축 식별 진단:")
    print(json.dumps(prepared["axis_preflight"], ensure_ascii=False, indent=2))
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "lambda": 0.0,
            "role": "baseline",
            "effective_strength": 0.0,
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
    selected, success, table = select_primary_operating_point(
        rows, prepared["baseline_flat"]
    )
    decision = screening_decision(rows, selected, success, table)
    return _persist(
        cfg, prepared, rows, runs, selected, success, table, decision
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
