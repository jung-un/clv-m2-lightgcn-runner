"""Fast Dunnhumby screen for user-specific centered signed CLV embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-centered-signed-clv-v1"
MODEL_ID = "m2_centered_signed"
SHUFFLED_ID = "m2_centered_signed_shuffled_user"
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


def configure_centered_signed_dunnhumby_run(
    **overrides,
) -> joint.JointNVConfig:
    """Dunnhumby seed-42 validation-only M1/centered/shuffled preset."""
    defaults = {
        "gate_shape": "centered",
        "gamma_init": 0.1,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_centered_signed_clv_v1"
        ),
    }
    cfg = joint.configure_joint_nv_run(
        "dunnhumby", short_hm=False, **(defaults | overrides)
    )
    return validate_centered_signed_config(cfg)


def validate_centered_signed_config(
    cfg: joint.JointNVConfig,
) -> joint.JointNVConfig:
    joint.validate_joint_nv_config(cfg)
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "window_days": None,
        "input_days": 365,
        "gate_shape": "centered",
        "id_dim": 64,
        "axis_dim": 16,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"centered signed screening requires {key}={expected!r}"
            )
    return cfg


def preflight_summary(cfg: joint.JointNVConfig) -> dict:
    cfg = validate_centered_signed_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": ["m1", MODEL_ID, SHUFFLED_ID],
        "architecture": (
            "separate ID|N|V layer-0 blocks -> one binary LightGCN -> "
            "one final dot score"
        ),
        "user_condition": {
            "N": "2 * train-history activity percentile - 1",
            "V": "2 * train-history value percentile - 1",
            "meaning": "below/above median users intervene in opposite directions",
        },
        "preference_preservation": (
            "one model/optimizer/batch loop: BPR(S_ID) + "
            "BPR(stopgrad(S_ID)+S_N+S_V)"
        ),
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "m4_sample_weighting": False,
        "post_score_external_model": False,
        "eval_test": cfg.eval_test,
        "eval_holdout": cfg.eval_holdout,
        "out_dir": cfg.out_dir,
    }


def _public(metrics: dict) -> dict:
    row = dict(metrics)
    for k in (10, 20, 50):
        source, target = f"entropy@{k}", f"exposure_entropy@{k}"
        if source in row and target not in row:
            row[target] = row[source]
    return row


def screening_decision(rows: list[dict]) -> dict:
    table = pd.DataFrame(rows).set_index("model_id")
    baseline = table.loc["m1"]
    model = table.loc[MODEL_ID]
    shuffled = table.loc[SHUFFLED_ID]
    ratios = {
        metric: float(model[metric] / max(float(baseline[metric]), 1e-12))
        for metric in ACCURACY_METRICS
    }
    economic_delta = float(model["revenue@10"] - baseline["revenue@10"])
    assignment_delta = float(
        model["revenue@10"] - shuffled["revenue@10"]
    )
    return {
        "success": bool(
            min(ratios.values()) >= 0.99
            and economic_delta > 0.0
            and assignment_delta > 0.0
        ),
        "economic_improved_vs_m1": bool(economic_delta > 0.0),
        "correct_assignment_beats_shuffled": bool(assignment_delta > 0.0),
        "revenue@10_delta_vs_m1": economic_delta,
        "revenue@10_delta_vs_shuffled": assignment_delta,
        "accuracy_ratios_vs_m1": ratios,
        "note": (
            "This is a post-hoc reading rule. It does not force metric "
            "improvement during training."
        ),
    }


def _paired_rows(baseline_per_user, runs) -> list[dict]:
    comparisons = [
        (MODEL_ID, "m1", runs[MODEL_ID]["per_user"], baseline_per_user),
        (
            SHUFFLED_ID,
            "m1",
            runs[SHUFFLED_ID]["per_user"],
            baseline_per_user,
        ),
        (
            MODEL_ID,
            SHUFFLED_ID,
            runs[MODEL_ID]["per_user"],
            runs[SHUFFLED_ID]["per_user"],
        ),
    ]
    rows = []
    for model_id, reference, current, base in comparisons:
        for metric in ("recall", "ndcg", "revenue", "arp"):
            difference = current[metric] - base[metric]
            rows.append(
                {
                    "model_id": model_id,
                    "reference": reference,
                    "split": "val",
                    "metric": metric,
                    **v3.paired_bootstrap([difference], v3.CFG["N_BOOT"]),
                }
            )
    return rows


def _persist(prepared, cfg, rows, baseline_per_user, runs, decision):
    frame = pd.DataFrame(rows)
    paired = _paired_rows(baseline_per_user, runs)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"m2_centered_signed_dunnhumby_{prepared['config_hash']}"
    csv_path = out / f"{stem}.csv"
    paired_path = out / f"{stem}_paired.csv"
    json_path = out / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(paired).to_csv(paired_path, index=False)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "decision": decision,
        "diagnostics": {name: run["diagnostics"] for name, run in runs.items()},
        "training": {name: run["training"] for name, run in runs.items()},
        "checkpoints": {name: run["checkpoint"] for name, run in runs.items()},
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": paired,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["screening_decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "paired_csv": str(paired_path),
        "json": str(json_path),
    }
    return frame


def run_experiment(
    cfg: joint.JointNVConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_centered_signed_config(
        cfg or configure_centered_signed_dunnhumby_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = joint._prepare(cfg)
    _, m1_training, baseline, baseline_per_user = joint._train_m1(prepared, cfg)
    runs = {
        MODEL_ID: joint._train_variant(
            prepared, cfg, MODEL_ID, variant="joint_nv"
        ),
        SHUFFLED_ID: joint._train_variant(
            prepared,
            cfg,
            SHUFFLED_ID,
            variant="joint_shuffled_user",
        ),
    }
    for run in runs.values():
        run["diagnostics"].update(
            run["model"].score_diagnostics(seed=cfg.seed)
        )
    rows = [
        joint.result_row("m1", "baseline", "none", cfg.seed, _public(baseline)),
        joint.result_row(
            MODEL_ID,
            "model",
            "centered",
            cfg.seed,
            _public(runs[MODEL_ID]["metrics"]),
            runs[MODEL_ID]["diagnostics"],
        ),
        joint.result_row(
            SHUFFLED_ID,
            "control",
            "centered_shuffled",
            cfg.seed,
            _public(runs[SHUFFLED_ID]["metrics"]),
            runs[SHUFFLED_ID]["diagnostics"],
        ),
    ]
    decision = screening_decision(rows)
    frame = _persist(
        prepared, cfg, rows, baseline_per_user, runs, decision
    )
    frame.attrs["m1_training"] = m1_training
    print("centered signed M2 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_centered_signed_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
