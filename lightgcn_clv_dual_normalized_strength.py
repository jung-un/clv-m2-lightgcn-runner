"""Re-evaluate trained dual-axis models on a common effective-strength scale."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_dual as dual
import lightgcn_clv_dual_multiseed_controls as controls
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
from clv_dual_axis_model import CLVDualAxisEmbeddingModel


CODE_VERSION = "clv-dual-normalized-strength-v1.0"
RHO_GRID = (0.2, 0.4, 0.6, 0.8, 1.0)
MODEL_IDS = (dual.PRIMARY_MODEL, *dual.CONTROLS)
ACCURACY_METRICS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)
)


@dataclass(frozen=True)
class NormalizedStrengthConfig:
    dataset: str
    seed42_result_json: str
    multiseed_result_json: str
    control_result_json: str
    window_days: int | None
    gate_shape: str
    seeds: tuple[int, ...] = (42, 43, 44)
    rho_grid: tuple[float, ...] = RHO_GRID
    model_ids: tuple[str, ...] = MODEL_IDS
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None


def configure_normalized_strength(
    dataset,
    seed42_result_json,
    multiseed_result_json,
    control_result_json,
    *,
    short_hm=False,
    out_dir=None,
):
    dataset = dataset.lower()
    if dataset == "dunnhumby":
        if short_hm:
            raise ValueError("short_hm은 H&M에서만 사용합니다")
        window_days, gate_shape = None, "equal"
    elif dataset == "hm" and short_hm:
        window_days, gate_shape = 60, "high"
    else:
        raise ValueError("이 runner는 Dunnhumby 전체 또는 H&M 60일만 허용합니다")
    default_out = Path(multiseed_result_json).parent / "normalized_strength"
    return validate_normalized_config(
        NormalizedStrengthConfig(
            dataset=dataset,
            seed42_result_json=str(seed42_result_json),
            multiseed_result_json=str(multiseed_result_json),
            control_result_json=str(control_result_json),
            window_days=window_days,
            gate_shape=gate_shape,
            out_dir=str(out_dir or default_out),
        )
    )


def validate_normalized_config(cfg):
    if tuple(cfg.seeds) != (42, 43, 44):
        raise ValueError("정규화 진단 seed는 정확히 42, 43, 44입니다")
    if tuple(cfg.rho_grid) != RHO_GRID:
        raise ValueError("rho grid는 사전 설정으로 고정됩니다")
    if tuple(cfg.model_ids) != MODEL_IDS:
        raise ValueError("주모형과 두 필수 대조군만 허용합니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("정규화 진단은 validation-only입니다")
    expected = (
        (None, "equal")
        if cfg.dataset == "dunnhumby"
        else (60, "high")
        if cfg.dataset == "hm"
        else None
    )
    if expected is None or (cfg.window_days, cfg.gate_shape) != expected:
        raise ValueError("동결 데이터셋별 gate 설정과 다릅니다")
    return cfg


def equivalent_lambda(rho, raw_effective_ratio):
    ratio = float(raw_effective_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("실효강도 비율은 유한한 양수여야 합니다")
    return float(rho) / ratio


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_path(payload, name, *, legacy=False):
    paths_key = "checkpoint_paths" if legacy else "checkpoints"
    path = Path(payload[paths_key][name])
    expected = payload["checkpoint_sha256"][name]
    if not path.is_file() or moe.file_sha256(path) != expected:
        raise RuntimeError(f"checkpoint 검증 실패: {name}")
    return path


def _model_checkpoint(seed42, multiseed, control_payload, seed, model_id):
    name = f"{model_id}_s{seed}"
    if seed == 42:
        return _verified_path(seed42, name, legacy=True)
    source = multiseed if model_id == dual.PRIMARY_MODEL else control_payload
    return _verified_path(source, name)


def _run_cfg(cfg, seed42_payload, seed):
    proxy = controls.ControlValidationConfig(
        dataset=cfg.dataset,
        seed42_result_json=cfg.seed42_result_json,
        multiseed_result_json=cfg.multiseed_result_json,
        window_days=cfg.window_days,
        gate_shape=cfg.gate_shape,
        fixed_lambda=2.0 if cfg.dataset == "dunnhumby" else 1.0,
        out_dir=cfg.out_dir,
    )
    return controls._run_cfg(proxy, seed42_payload, seed)


def _load_model(prepared, cfg, seed, model_id, checkpoint):
    model = CLVDualAxisEmbeddingModel(
        dual._fresh_base(prepared, seed=seed),
        prepared["user_profile"],
        prepared["item_profile"],
        prepared["q_n"],
        prepared["q_v"],
        control=model_id,
        seed=seed,
        hidden_dim=cfg.expert_hidden_dim,
        expert_dim=cfg.expert_dim,
    ).to(v3.DEVICE)
    blob = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    if blob.get("baseline_state_hash") != prepared["baseline_hash"]:
        raise RuntimeError("adapter와 M1 기준상태가 다릅니다")
    model.load_state_dict(blob["state"])
    model.eval()
    model.set_gate_shape(cfg.gate_shape)
    model.set_eval_axes("n_plus_v")
    return model


def _evaluate_seed(cfg, seed42, multiseed, control_payload, seed):
    run_cfg = _run_cfg(cfg, seed42, seed)
    encoder = (
        _verified_path(seed42, "encoder_s42", legacy=True)
        if seed == 42
        else _verified_path(multiseed, f"encoder_s{seed}")
    )
    prepared = dual._prepare(run_cfg, seed=seed, encoder_checkpoint=encoder)
    rows = [
        {
            "seed": seed,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "rho": 0.0,
            "lambda_equivalent": 0.0,
            "raw_effective_ratio": 0.0,
            "effective_strength": 0.0,
            **prepared["baseline_flat"],
        }
    ]
    per_user = {"m1": {0.0: prepared["baseline_per_user"]}}
    for model_id in cfg.model_ids:
        checkpoint = _model_checkpoint(
            seed42, multiseed, control_payload, seed, model_id
        )
        model = _load_model(prepared, run_cfg, seed, model_id, checkpoint)
        diagnostics = model.axis_diagnostics(cfg.gate_shape)
        ratio = float(diagnostics["effective_total_ratio"])
        per_user[model_id] = {}
        for rho in cfg.rho_grid:
            lam = equivalent_lambda(rho, ratio)
            flat, user_metrics = moe._flat_evaluation(
                model,
                lam,
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
                    "gate_shape": cfg.gate_shape,
                    "rho": float(rho),
                    "lambda_equivalent": lam,
                    "raw_effective_ratio": ratio,
                    "effective_strength": float(rho),
                    **diagnostics,
                    **flat,
                }
            )
            per_user[model_id][float(rho)] = user_metrics
    return {"rows": rows, "per_user": per_user, "prepared": prepared}


def normalized_strength_decision(rows):
    table = pd.DataFrame(rows)
    baselines = table[table.model_id.eq("m1")].set_index("seed")
    if set(baselines.index) != {42, 43, 44}:
        raise ValueError("M1 seed 42·43·44가 모두 필요합니다")
    candidates = []
    for rho in sorted(table.loc[table.model_id.eq(dual.PRIMARY_MODEL), "rho"].unique()):
        main = table[
            table.model_id.eq(dual.PRIMARY_MODEL) & table.rho.eq(rho)
        ].set_index("seed")
        if set(main.index) != {42, 43, 44}:
            raise ValueError("각 rho에 주모형 seed 42·43·44가 필요합니다")
        accuracy = all(
            float(main[metric].mean() / baselines[metric].mean()) >= 0.99
            for metric in ACCURACY_METRICS
        )
        m1_delta = main["revenue@10"] - baselines["revenue@10"]
        controls_pass = True
        control_summary = {}
        for control_id in dual.CONTROLS:
            control = table[
                table.model_id.eq(control_id) & table.rho.eq(rho)
            ].set_index("seed")
            delta = main["revenue@10"] - control["revenue@10"]
            passed = float(delta.mean()) > 0 and int((delta > 0).sum()) >= 2
            controls_pass &= passed
            control_summary[control_id] = {
                "mean_delta": float(delta.mean()),
                "positive_seed_count": int((delta > 0).sum()),
                "passed": passed,
            }
        conditions = {
            "accuracy_guard": accuracy,
            "mean_revenue_above_m1": float(m1_delta.mean()) > 0,
            "revenue_positive_in_two_seeds": int((m1_delta > 0).sum()) >= 2,
            "beats_required_controls": controls_pass,
        }
        candidates.append(
            {
                "rho": float(rho),
                "main_mean_revenue@10": float(main["revenue@10"].mean()),
                "conditions": conditions,
                "controls": control_summary,
                "eligible": all(conditions.values()),
            }
        )
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return {
            "success": False,
            "selected_rho": 0.0,
            "failed_conditions": ["no jointly eligible rho"],
            "rho_table": candidates,
        }
    selected = max(
        eligible, key=lambda row: (row["main_mean_revenue@10"], -row["rho"])
    )
    return {
        "success": True,
        "selected_rho": selected["rho"],
        "failed_conditions": [],
        "selected": selected,
        "rho_table": candidates,
    }


def _delta_rows(results):
    records = []
    for model_id in MODEL_IDS:
        for rho in RHO_GRID:
            for metric in ("recall", "ndcg", "revenue", "arp"):
                diffs = []
                for result in results:
                    per_user = result["per_user"]
                    diffs.append(
                        np.asarray(per_user[model_id][rho][metric], float)
                        - np.asarray(per_user["m1"][0.0][metric], float)
                    )
                records.append(
                    {
                        "model_id": model_id,
                        "rho": rho,
                        "metric": metric,
                        **v3.paired_bootstrap(diffs, 2000),
                    }
                )
    return records


def _persist(cfg, seed42, multiseed, control_payload, results, decision):
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row for result in results for row in result["rows"]])
    delta = _delta_rows(results)
    stem = f"clv_dual_normalized_strength_{cfg.dataset}"
    csv_path, delta_path = output / f"{stem}.csv", output / f"{stem}_delta.csv"
    decision_path, json_path = output / f"{stem}_decision.csv", output / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta).to_csv(delta_path, index=False)
    pd.DataFrame(decision["rho_table"]).to_csv(decision_path, index=False)
    report = {
        "code_version": CODE_VERSION,
        "source_revision": moe.source_revision(),
        "config": asdict(cfg),
        "source_result_fingerprints": {
            "seed42": seed42.get("result_fingerprint"),
            "multiseed": multiseed.get("original_seed42_result_fingerprint"),
        },
        "absolute_rows": frame.to_dict("records"),
        "delta_rows": delta,
        "normalized_strength_decision": decision,
        "interpretation": {
            "evaluation_only": True,
            "training_executed": False,
            "validation_only": True,
            "test_executed": False,
            "holdout_executed": False,
            "hm_two_year_executed": False,
            "normalizer": "bounded model-score diagnostic; no validation labels",
        },
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["normalized_strength_decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "decision_csv": str(decision_path),
        "json": str(json_path),
    }
    return frame


def run_normalized_strength(cfg):
    cfg = validate_normalized_config(cfg)
    seed42 = _read_json(cfg.seed42_result_json)
    multiseed = _read_json(cfg.multiseed_result_json)
    control_payload = _read_json(cfg.control_result_json)
    results = [
        _evaluate_seed(cfg, seed42, multiseed, control_payload, seed)
        for seed in cfg.seeds
    ]
    rows = [row for result in results for row in result["rows"]]
    decision = normalized_strength_decision(rows)
    return _persist(cfg, seed42, multiseed, control_payload, results, decision)


__all__ = [
    "NormalizedStrengthConfig",
    "configure_normalized_strength",
    "equivalent_lambda",
    "normalized_strength_decision",
    "run_normalized_strength",
    "validate_normalized_config",
]
