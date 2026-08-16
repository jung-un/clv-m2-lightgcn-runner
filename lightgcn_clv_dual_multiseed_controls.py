"""Validation-only identification controls for the frozen dual-axis M2."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_dual as dual
import lightgcn_clv_moe as moe


CODE_VERSION = "clv-dual-multiseed-controls-v1.0"
CONTROL_IDS = ("dual_shuffled_user", "dual_adapter_only")


@dataclass(frozen=True)
class ControlValidationConfig:
    dataset: str
    seed42_result_json: str
    multiseed_result_json: str
    window_days: int | None
    gate_shape: str
    fixed_lambda: float
    new_seeds: tuple[int, ...] = (43, 44)
    control_ids: tuple[str, ...] = CONTROL_IDS
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str | None = None


def configure_multiseed_controls(
    dataset,
    seed42_result_json,
    multiseed_result_json,
    *,
    short_hm=False,
    out_dir=None,
):
    dataset = dataset.lower()
    if dataset == "dunnhumby":
        if short_hm:
            raise ValueError("short_hm은 H&M에서만 사용합니다")
        window_days, gate_shape, fixed_lambda = None, "equal", 2.0
    elif dataset == "hm" and short_hm:
        window_days, gate_shape, fixed_lambda = 60, "high", 1.0
    else:
        raise ValueError("이 runner는 Dunnhumby 전체 또는 H&M 60일만 허용합니다")
    default_out = Path(multiseed_result_json).parent / "control_validation"
    return validate_control_config(
        ControlValidationConfig(
            dataset=dataset,
            seed42_result_json=str(seed42_result_json),
            multiseed_result_json=str(multiseed_result_json),
            window_days=window_days,
            gate_shape=gate_shape,
            fixed_lambda=fixed_lambda,
            out_dir=str(out_dir or default_out),
        )
    )


def validate_control_config(cfg):
    if tuple(cfg.new_seeds) != (43, 44):
        raise ValueError("추가 validation seed는 정확히 43, 44입니다")
    if tuple(cfg.control_ids) != CONTROL_IDS:
        raise ValueError("두 식별 대조군만 허용합니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("대조군 runner는 validation-only입니다")
    expected = (
        (None, "equal", 2.0)
        if cfg.dataset == "dunnhumby"
        else (60, "high", 1.0)
        if cfg.dataset == "hm"
        else None
    )
    if expected is None or (
        cfg.window_days != expected[0]
        or cfg.gate_shape != expected[1]
        or not np.isclose(cfg.fixed_lambda, expected[2])
    ):
        raise ValueError("동결 데이터셋별 운영점과 다릅니다")
    return cfg


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _verified_checkpoint(payload, name):
    path = Path(payload["checkpoints"][name])
    expected = payload["checkpoint_sha256"][name]
    if not path.is_file() or moe.file_sha256(path) != expected:
        raise RuntimeError(f"재사용 checkpoint 검증 실패: {name}")
    return path


def _run_cfg(cfg, seed42_payload, seed):
    source = seed42_payload["config"]
    reusable = {
        field.name: source[field.name]
        for field in fields(moe.MoEConfig)
        if field.name in source
        and field.name
        not in {
            "dataset", "seed_list", "window_days", "eval_test", "eval_holdout",
            "lambda_eval", "out_dir",
        }
    }
    base = dual.configure_dual_run(
        cfg.dataset,
        short_hm=cfg.dataset == "hm",
        out_dir=str(Path(cfg.out_dir) / f"seed_{seed}"),
        **reusable,
    )
    return replace(
        base,
        seed_list=(seed,),
        lambda_eval=(cfg.fixed_lambda,),
        eval_test=False,
        eval_holdout=False,
    )


def _load_reusable_prepared(cfg, seed42_payload, multiseed_payload, seed):
    run_cfg = _run_cfg(cfg, seed42_payload, seed)
    encoder = _verified_checkpoint(multiseed_payload, f"encoder_s{seed}")
    prepared = dual._prepare(run_cfg, seed=seed, encoder_checkpoint=encoder)
    expected_m1 = _verified_checkpoint(multiseed_payload, f"m1_s{seed}")
    if Path(prepared["m1_checkpoint"]).resolve() != expected_m1.resolve():
        raise RuntimeError("재사용 M1 checkpoint 경로가 기존 multiseed 결과와 다릅니다")
    return prepared


def _run_control_seed(cfg, seed42_payload, multiseed_payload, seed):
    prepared = _load_reusable_prepared(cfg, seed42_payload, multiseed_payload, seed)
    run_cfg = _run_cfg(cfg, seed42_payload, seed)
    rows, runs = [], {}
    for model_id in cfg.control_ids:
        run = dual._train_variant(
            model_id,
            dual._fresh_base(prepared, seed=seed),
            prepared,
            run_cfg,
            seed=seed,
            gate_shapes=(cfg.gate_shape,),
            lambda_eval=(cfg.fixed_lambda,),
        )
        rows.append(run["rows"][0])
        runs[model_id] = run
    return {
        "rows": rows,
        "runs": runs,
        "checkpoints": {name: run["checkpoint"] for name, run in runs.items()},
    }


def _existing_rows(cfg, seed42_payload, multiseed_payload):
    wanted = {dual.PRIMARY_MODEL, *cfg.control_ids}
    rows = [
        row for row in seed42_payload["absolute_rows"]
        if int(row["seed"]) == 42
        and row["model_id"] in wanted
        and row["gate_shape"] == cfg.gate_shape
        and np.isclose(float(row["lambda"]), cfg.fixed_lambda)
    ]
    rows += [
        row for row in multiseed_payload["absolute_rows"]
        if int(row["seed"]) in cfg.new_seeds
        and row["model_id"] == dual.PRIMARY_MODEL
    ]
    if len(rows) != 5:
        raise RuntimeError("기존 seed42 대조군 또는 seed43·44 주모형 행을 찾지 못했습니다")
    return rows


def control_reproducibility_decision(rows):
    table = pd.DataFrame(rows)
    main = table[table.model_id.eq(dual.PRIMARY_MODEL)].set_index("seed")
    comparisons, failed = {}, []
    if set(main.index) != {42, 43, 44}:
        raise ValueError("주모형 seed 42·43·44가 모두 필요합니다")
    for control_id in CONTROL_IDS:
        control = table[table.model_id.eq(control_id)].set_index("seed")
        if set(control.index) != {42, 43, 44}:
            raise ValueError(f"{control_id} seed 42·43·44가 모두 필요합니다")
        delta = main["revenue@10"] - control["revenue@10"]
        mean_delta = float(delta.mean())
        positive_count = int((delta > 0).sum())
        passed = mean_delta > 0 and positive_count >= 2
        comparisons[control_id] = {
            "passed": passed,
            "mean_delta": mean_delta,
            "positive_seed_count": positive_count,
            "seed_delta": {str(seed): float(value) for seed, value in delta.items()},
        }
        if not passed:
            failed.append(control_id)
    return {
        "success": not failed,
        "comparisons": comparisons,
        "failed_controls": failed,
    }


def _persist(cfg, seed42_payload, multiseed_payload, rows, runs, decision):
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["seed", "model_id"])
    stem = f"clv_dual_multiseed_controls_{cfg.dataset}"
    csv_path, json_path = output / f"{stem}.csv", output / f"{stem}.json"
    decision_path = output / f"{stem}_decision.csv"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(
        [
            {"control": name, **values}
            for name, values in decision["comparisons"].items()
        ]
    ).to_csv(decision_path, index=False)
    checkpoints = {
        f"{model_id}_s{seed}": run["checkpoint"]
        for seed, seed_runs in runs.items()
        for model_id, run in seed_runs.items()
    }
    report = {
        "code_version": CODE_VERSION,
        "source_revision": moe.source_revision(),
        "config": asdict(cfg),
        "seed42_result_json": cfg.seed42_result_json,
        "multiseed_result_json": cfg.multiseed_result_json,
        "absolute_rows": frame.to_dict("records"),
        "control_reproducibility_decision": decision,
        "checkpoints": checkpoints,
        "checkpoint_sha256": {name: moe.file_sha256(path) for name, path in checkpoints.items()},
        "interpretation": {
            "validation_only": True,
            "test_executed": False,
            "holdout_executed": False,
            "hm_two_year_executed": False,
            "lambda_reselected": False,
        },
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    frame.attrs["control_reproducibility_decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path), "decision_csv": str(decision_path), "json": str(json_path)
    }
    return frame


def run_multiseed_controls(cfg):
    cfg = validate_control_config(cfg)
    seed42_payload = _read_json(cfg.seed42_result_json)
    multiseed_payload = _read_json(cfg.multiseed_result_json)
    rows = _existing_rows(cfg, seed42_payload, multiseed_payload)
    runs = {}
    for seed in cfg.new_seeds:
        result = _run_control_seed(cfg, seed42_payload, multiseed_payload, seed)
        rows.extend(result["rows"])
        runs[seed] = result["runs"]
    decision = control_reproducibility_decision(rows)
    return _persist(cfg, seed42_payload, multiseed_payload, rows, runs, decision)


__all__ = [
    "ControlValidationConfig", "configure_multiseed_controls",
    "control_reproducibility_decision", "run_multiseed_controls",
    "validate_control_config",
]
