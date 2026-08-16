"""Evaluation-only diagnosis of the completed H&M full-period M2 suite."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_dual as dual
import lightgcn_clv_dual_hm2y_suite as suite
import lightgcn_clv_dual_normalized_strength as normalized
import lightgcn_clv_moe as moe
from clv_run_state import file_sha256


GATE_SHAPES = ("high", "equal", "low")
RHO_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
AXIS_MODES = ("n_only", "v_only", "n_plus_v")
MODEL_IDS = ("dual_clv_fixed", "dual_shuffled_user", "dual_adapter_only")
ACCURACY_METRICS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)
)
FROZEN_GATE_SHAPE = "high"
FROZEN_RHO = 0.20


@dataclass(frozen=True)
class Hm2YDiagnosticConfig:
    suite_result_json: str
    gate_shapes: tuple[str, ...] = GATE_SHAPES
    rho_grid: tuple[float, ...] = RHO_GRID
    axis_modes: tuple[str, ...] = AXIS_MODES
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None


def configure_hm2y_diagnostic(
    suite_result_json: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> Hm2YDiagnosticConfig:
    source = Path(suite_result_json)
    return validate_hm2y_diagnostic_config(
        Hm2YDiagnosticConfig(
            suite_result_json=str(source),
            out_dir=str(out_dir or source.parent / "checkpoint_diagnostics"),
        )
    )


def validate_hm2y_diagnostic_config(
    cfg: Hm2YDiagnosticConfig,
) -> Hm2YDiagnosticConfig:
    if tuple(cfg.gate_shapes) != GATE_SHAPES:
        raise ValueError("H&M 2년 진단 gate grid가 승인 설정과 다릅니다")
    if tuple(cfg.rho_grid) != RHO_GRID:
        raise ValueError("H&M 2년 진단 rho grid가 승인 설정과 다릅니다")
    if tuple(cfg.axis_modes) != AXIS_MODES:
        raise ValueError("H&M 2년 진단 axis mode가 승인 설정과 다릅니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("H&M 2년 checkpoint 진단은 validation-only입니다")
    return cfg


def load_verified_suite_payload(cfg: Hm2YDiagnosticConfig):
    path = Path(cfg.suite_result_json)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_cfg = payload.get("config", {})
    if (
        payload.get("code_version") != suite.CODE_VERSION
        or payload.get("models") != list(suite.MODELS)
        or source_cfg.get("dataset") != "hm"
        or source_cfg.get("window_days") is not None
        or tuple(source_cfg.get("seed_list", ())) != (42,)
        or bool(source_cfg.get("eval_test"))
        or bool(source_cfg.get("eval_holdout"))
    ):
        raise RuntimeError("H&M 2년 validation suite 결과가 아닙니다")
    expected_names = {"m1", "encoder", *MODEL_IDS}
    checkpoint_paths = payload.get("checkpoint_paths", {})
    checkpoint_hashes = payload.get("checkpoint_sha256", {})
    if set(checkpoint_paths) != expected_names or set(checkpoint_hashes) != expected_names:
        raise RuntimeError("suite checkpoint manifest가 완전하지 않습니다")
    verified = {}
    for name in sorted(expected_names):
        checkpoint = Path(checkpoint_paths[name])
        if not checkpoint.is_file() or file_sha256(checkpoint) != checkpoint_hashes[name]:
            raise RuntimeError(f"checkpoint hash 검증 실패: {name}")
        verified[name] = checkpoint
    return payload, verified


def diagnostic_decision(rows: pd.DataFrame | list[dict]):
    """Require a two-point plateau before selecting the lowest passing rho."""
    frame = pd.DataFrame(rows).copy()
    baseline_rows = frame[frame.model_id.eq("m1")]
    if len(baseline_rows) != 1:
        raise ValueError("H&M 2년 진단에는 M1 행이 정확히 하나 필요합니다")
    baseline = baseline_rows.iloc[0]
    records = []
    for gate_shape in GATE_SHAPES:
        gate_rows = frame[frame.gate_shape.eq(gate_shape)]
        if gate_rows.empty:
            continue
        for rho in RHO_GRID:
            point = gate_rows[gate_rows.rho.eq(rho)].set_index("model_id")
            if not set(MODEL_IDS).issubset(point.index):
                continue
            main = point.loc["dual_clv_fixed"]
            accuracy_pass = all(
                float(main[metric]) >= 0.99 * float(baseline[metric])
                for metric in ACCURACY_METRICS
            )
            revenue_above_m1 = (
                float(main["revenue@10"]) > float(baseline["revenue@10"])
            )
            shuffled_margin = float(main["revenue@10"]) - float(
                point.loc["dual_shuffled_user", "revenue@10"]
            )
            adapter_margin = float(main["revenue@10"]) - float(
                point.loc["dual_adapter_only", "revenue@10"]
            )
            beats_controls = shuffled_margin > 0 and adapter_margin > 0
            records.append(
                {
                    "gate_shape": gate_shape,
                    "rho": float(rho),
                    "accuracy_6_guard_pass": bool(accuracy_pass),
                    "revenue_above_m1": bool(revenue_above_m1),
                    "beats_shuffled_user": bool(shuffled_margin > 0),
                    "beats_adapter_only": bool(adapter_margin > 0),
                    "shuffled_revenue_margin": shuffled_margin,
                    "adapter_revenue_margin": adapter_margin,
                    "joint_pass": bool(
                        accuracy_pass and revenue_above_m1 and beats_controls
                    ),
                }
            )
    table = pd.DataFrame(records)
    plateau_candidates = []
    for gate_shape in GATE_SHAPES:
        gate = table[table.gate_shape.eq(gate_shape)].sort_values("rho")
        passing = gate["joint_pass"].tolist()
        rhos = gate["rho"].tolist()
        for index in range(len(gate) - 1):
            if passing[index] and passing[index + 1]:
                plateau_candidates.append(
                    {
                        "gate_shape": gate_shape,
                        "rho": float(rhos[index]),
                        "plateau_rhos": [
                            float(rhos[index]),
                            float(rhos[index + 1]),
                        ],
                    }
                )
    if plateau_candidates:
        gate_priority = {name: index for index, name in enumerate(GATE_SHAPES)}
        selected = min(
            plateau_candidates,
            key=lambda row: (row["rho"], gate_priority[row["gate_shape"]]),
        )
        decision = {
            "success": True,
            "selected_gate_shape": selected["gate_shape"],
            "selected_rho": selected["rho"],
            "plateau_rhos": selected["plateau_rhos"],
            "axis_diagnostic_gate_shape": selected["gate_shape"],
            "axis_diagnostic_rho": selected["rho"],
            "reason": "two adjacent rho points pass all guards and controls",
        }
    else:
        decision = {
            "success": False,
            "selected_gate_shape": None,
            "selected_rho": None,
            "plateau_rhos": [],
            "axis_diagnostic_gate_shape": FROZEN_GATE_SHAPE,
            "axis_diagnostic_rho": FROZEN_RHO,
            "reason": "no two adjacent rho points pass all guards and controls",
        }
    return decision, table


def evaluate_model_grid(
    model,
    model_id: str,
    prepared: dict,
    *,
    gate_shapes: tuple[str, ...],
    rho_grid: tuple[float, ...],
    axis_mode: str,
    evaluator=None,
    progress: bool = False,
):
    if model_id not in MODEL_IDS:
        raise ValueError(f"지원하지 않는 진단 model_id: {model_id}")
    if axis_mode not in AXIS_MODES:
        raise ValueError(f"지원하지 않는 axis mode: {axis_mode}")
    evaluator = evaluator or moe._flat_evaluation
    ratio_key = {
        "n_only": "effective_n_ratio",
        "v_only": "effective_v_ratio",
        "n_plus_v": "effective_total_ratio",
    }[axis_mode]
    rows, per_user = [], {}
    model.set_eval_axes(axis_mode)
    for gate_shape in gate_shapes:
        model.set_gate_shape(gate_shape)
        diagnostics = model.axis_diagnostics(gate_shape)
        ratio = float(diagnostics[ratio_key])
        if not np.isfinite(ratio) or ratio <= 0:
            raise RuntimeError(
                f"{model_id}/{gate_shape}/{axis_mode} 실효강도 비율이 유효하지 않습니다"
            )
        for rho in rho_grid:
            rho = float(rho)
            lam = rho / ratio
            if progress:
                print(
                    f"[checkpoint diagnostic] {model_id} | gate={gate_shape} "
                    f"| axis={axis_mode} | rho={rho:.2f} | lambda={lam:.6f}"
                )
            flat, user_metrics = evaluator(
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
                    "model_id": model_id,
                    "gate_shape": gate_shape,
                    "axis_mode": axis_mode,
                    "rho": rho,
                    "raw_effective_ratio": ratio,
                    "lambda_equivalent": lam,
                    "effective_strength": rho,
                    **diagnostics,
                    **flat,
                }
            )
            per_user[(gate_shape, rho)] = user_metrics
    return rows, per_user


def _prepare_from_suite(payload: dict, paths: dict[str, Path]):
    source_config = dict(payload["config"])
    run_cfg = suite.validate_suite_config(moe.MoEConfig(**source_config))
    prepared = dual._prepare(
        run_cfg,
        seed=42,
        encoder_checkpoint=paths["encoder"],
    )
    if file_sha256(prepared["m1_checkpoint"]) != file_sha256(paths["m1"]):
        raise RuntimeError("현재 준비된 M1과 suite의 M1 checkpoint가 다릅니다")
    return run_cfg, prepared


def _load_checkpoint_model(
    model_id: str,
    prepared: dict,
    run_cfg: moe.MoEConfig,
    checkpoint: Path,
):
    return normalized._load_model(
        prepared,
        run_cfg,
        FROZEN_GATE_SHAPE,
        seed=42,
        model_id=model_id,
        checkpoint=checkpoint,
    )


def _baseline_row(prepared: dict) -> dict:
    return {
        "seed": 42,
        "model_id": "m1",
        "split": "val",
        "gate_shape": "none",
        "axis_mode": "base",
        "rho": 0.0,
        "raw_effective_ratio": 0.0,
        "lambda_equivalent": 0.0,
        "effective_strength": 0.0,
        **prepared["baseline_flat"],
    }


def _add_axis_deltas(axis: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    axis = axis.copy()
    for metric in (
        "recall@10",
        "ndcg@10",
        "revenue@10",
        "arp@10",
        "coverage@10",
        "value_alignment",
    ):
        if metric not in axis or metric not in baseline:
            continue
        delta = axis[metric].astype(float) - float(baseline[metric])
        axis[f"delta_{metric}"] = delta
        denominator = float(baseline[metric])
        axis[f"change_pct_{metric}"] = (
            np.nan if denominator == 0 else 100.0 * delta / denominator
        )
    return axis


def run_hm2y_diagnostic(
    cfg: Hm2YDiagnosticConfig,
) -> pd.DataFrame:
    cfg = validate_hm2y_diagnostic_config(cfg)
    payload, paths = load_verified_suite_payload(cfg)
    run_cfg, prepared = _prepare_from_suite(payload, paths)
    curve_rows = [_baseline_row(prepared)]
    for model_id in MODEL_IDS:
        model = _load_checkpoint_model(
            model_id, prepared, run_cfg, paths[model_id]
        )
        rows, _ = evaluate_model_grid(
            model,
            model_id,
            prepared,
            gate_shapes=cfg.gate_shapes,
            rho_grid=cfg.rho_grid,
            axis_mode="n_plus_v",
            progress=True,
        )
        curve_rows.extend(
            {"seed": 42, "split": "val", **row} for row in rows
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    curve = pd.DataFrame(curve_rows)
    decision, decision_table = diagnostic_decision(curve)
    gate_shape = decision["axis_diagnostic_gate_shape"]
    rho = float(decision["axis_diagnostic_rho"])
    axis_rows = [{**_baseline_row(prepared), "diagnostic_point": True}]
    for model_id in MODEL_IDS:
        model = _load_checkpoint_model(
            model_id, prepared, run_cfg, paths[model_id]
        )
        selected_n_plus_v = curve[
            curve.model_id.eq(model_id)
            & curve.gate_shape.eq(gate_shape)
            & curve.rho.eq(rho)
        ]
        if len(selected_n_plus_v) != 1:
            raise RuntimeError("선택 진단점의 N+V 결과를 하나로 찾을 수 없습니다")
        for axis_mode in ("n_only", "v_only"):
            rows, _ = evaluate_model_grid(
                model,
                model_id,
                prepared,
                gate_shapes=(gate_shape,),
                rho_grid=(rho,),
                axis_mode=axis_mode,
                progress=True,
            )
            axis_rows.extend(
                {"seed": 42, "split": "val", "diagnostic_point": True, **row}
                for row in rows
            )
        axis_rows.append(
            {**selected_n_plus_v.iloc[0].to_dict(), "diagnostic_point": True}
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    axis = _add_axis_deltas(pd.DataFrame(axis_rows), prepared["baseline_flat"])
    result = persist_diagnostic(
        cfg,
        payload,
        curve,
        decision_table,
        axis,
        decision,
    )
    print("H&M 2년 checkpoint 진단 판정:", decision)
    print("결과 파일:", result.attrs["result_paths"])
    return result


def persist_diagnostic(
    cfg: Hm2YDiagnosticConfig,
    suite_payload: dict,
    curve: pd.DataFrame,
    decision_table: pd.DataFrame,
    axis: pd.DataFrame,
    decision: dict,
) -> pd.DataFrame:
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "source_suite_result_fingerprint": suite_payload.get("result_fingerprint"),
        "config": asdict(cfg),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    stem = f"clv_dual_hm2y_checkpoint_diagnostic_{fingerprint}"
    paths = {
        "curve_csv": output / f"{stem}_curve.csv",
        "decision_csv": output / f"{stem}_decision.csv",
        "axis_csv": output / f"{stem}_axis.csv",
        "json": output / f"{stem}.json",
    }
    curve.to_csv(paths["curve_csv"], index=False, float_format="%.8f")
    decision_table.to_csv(
        paths["decision_csv"], index=False, float_format="%.8f"
    )
    axis.to_csv(paths["axis_csv"], index=False, float_format="%.8f")
    report = {
        "code_version": "clv-dual-hm2y-checkpoint-diagnostic-v1.0",
        "source_revision": moe.source_revision(),
        "result_fingerprint": fingerprint,
        "source_suite_result_fingerprint": suite_payload.get("result_fingerprint"),
        "config": asdict(cfg),
        "source_checkpoint_paths": suite_payload.get("checkpoint_paths", {}),
        "source_checkpoint_sha256": suite_payload.get("checkpoint_sha256", {}),
        "decision": decision,
        "decision_rows": decision_table.to_dict("records"),
        "curve_rows": curve.to_dict("records"),
        "axis_rows": axis.to_dict("records"),
        "interpretation": {
            "evaluation_only": True,
            "training_executed": False,
            "validation_only": True,
            "test_executed": False,
            "holdout_executed": False,
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
        },
    }
    suite._atomic_json(paths["json"], report)
    result = curve.copy()
    result.attrs["decision"] = decision
    result.attrs["axis_rows"] = axis.to_dict("records")
    result.attrs["result_paths"] = {
        name: str(path) for name, path in paths.items()
    }
    return result


__all__ = [
    "Hm2YDiagnosticConfig",
    "configure_hm2y_diagnostic",
    "diagnostic_decision",
    "evaluate_model_grid",
    "load_verified_suite_payload",
    "persist_diagnostic",
    "run_hm2y_diagnostic",
    "validate_hm2y_diagnostic_config",
]
