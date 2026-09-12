"""Four-arm M2 screen that changes only the historical N input."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from clv_m2_n_proxy_candidates import build_n_proxy_candidates
from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_n_conditioned_value_basis_screen as value_basis
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-n-proxy-value-basis-development-screen-v1"
M1_MODEL_ID = value_basis.M1_MODEL_ID
QV_ONLY_MODEL_ID = "m2_qv_only_value_basis_gate1_k5"
REPEAT_RATE_MODEL_ID = "m2_repeat_rate_n_value_basis_k5"
RECENCY_ACTIVITY_MODEL_ID = "m2_recency_adjusted_activity_n_value_basis_k5"
BGNBD_MODEL_ID = "m2_bgnbd_expected_count_n_value_basis_k5"
TRAINED_MODEL_IDS = (
    QV_ONLY_MODEL_ID,
    REPEAT_RATE_MODEL_ID,
    RECENCY_ACTIVITY_MODEL_ID,
    BGNBD_MODEL_ID,
)
MODEL_IDS = (M1_MODEL_ID,) + TRAINED_MODEL_IDS
ECONOMIC_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)
TOP10_ACCURACY_METRICS = ("recall@10", "ndcg@10")
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M2NProxyValueBasisConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    economic_dim: int = 3
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho: float = 0.05
    gate_delta: float = 0.25
    basis_bandwidth: float = 0.25
    positive_weight_lambda: float = 0.5
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    bgnbd_horizon_days: float = 7.0
    out_dir: str = ""
    baseline_result_dir: str = ""
    m1_reference_json: str = ""


def configure_m2_n_proxy_screen(**overrides) -> M2NProxyValueBasisConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{data_root}_m2_n_proxy_value_basis_development_screen_v1",
        "baseline_result_dir": (
            f"{data_root}_m2_repeatshare_historical_backtest_v1"
        ),
        "m1_reference_json": (
            f"{data_root}_m5_m2_m4_joint_historical_screen_v1/"
            "m5_m2_m4_joint_7cd818302fb1.json"
        ),
    }
    return validate_config(M2NProxyValueBasisConfig(**(defaults | overrides)))


def validate_config(cfg: M2NProxyValueBasisConfig) -> M2NProxyValueBasisConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "economic_dim": 3,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho": 0.05,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
        "positive_weight_lambda": 0.5,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
        "bgnbd_horizon_days": 7.0,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"M2 N-proxy screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not all((cfg.out_dir, cfg.baseline_result_dir, cfg.m1_reference_json)):
        raise ValueError("out_dir, baseline_result_dir, M1 reference JSON이 필요합니다")
    return cfg


def preflight_summary(cfg: M2NProxyValueBasisConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(TRAINED_MODEL_IDS),
        "reused_models": [M1_MODEL_ID],
        "research_question": (
            "Which train-history definition of the CLV transaction-frequency "
            "component N best complements the same q_V--item-price basis in M2?"
        ),
        "m1_to_m5_scope": (
            "This is an M2 representation screen only. It does not replace the "
            "independent M3 graph and M4 loss research axes."
        ),
        "common_m2": {
            "user_v": "q_V historical mean-transaction-value percentile",
            "item_input": "train-only item purchase-amount percentile",
            "value_basis": (
                "fixed normalized Gaussian RBF at [0.0, 0.5, 1.0]"
            ),
            "rho": cfg.rho,
            "gate": (
                "1 + 0.25*tanh(a1*(2*q_N-1)); one learned slope and no offset"
            ),
            "qv_only_gate": 1.0,
            "economic_graph_propagation": True,
            "joint_end_to_end_training": True,
            "external_reranking": False,
        },
        "arms": {
            "A": "q_V-only, fixed gate=1",
            "B": "current repeat-transaction-rate q_N plus q_V",
            "C": "recency-adjusted activity q_N plus q_V",
            "D": "BG/NBD next-7-day expected repeat-count q_N plus q_V",
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "negative_sampling": "uniform",
            "negative_count": cfg.negative_count,
            "sample_weighting": False,
            "new_loss_term": False,
            "m3_edge_weight": False,
            "m4_loss_weight": False,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
        },
        "reading_rule": {
            "total_economic_signal": (
                "candidate > reused compatible M1 on both top-10 economic metrics"
            ),
            "n_increment_signal": (
                "B/C/D > A q_V-only on both top-10 economic metrics"
            ),
            "development_candidate": (
                "total_economic_signal and n_increment_signal; top-10 accuracy, "
                "wider cutoffs, exposure, and CLV segments are all reported"
            ),
            "single_seed_selection_for_final_test_permitted": False,
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final CLV-effect claim"
            ),
        },
        "reference_source": cfg.m1_reference_json,
        "different_run_disclosure": True,
        "out_dir": cfg.out_dir,
    }


def _config_hash(cfg: M2NProxyValueBasisConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M2NProxyValueBasisConfig) -> dict:
    prepared = value_basis._prepare(cfg)
    candidates, diagnostics = build_n_proxy_candidates(
        prepared["axes"],
        prepared["clv_valid"],
        horizon=cfg.bgnbd_horizon_days,
    )
    if not np.allclose(
        candidates["repeat_rate"], prepared["q_n"], atol=1e-7, rtol=0.0
    ):
        raise RuntimeError("B arm의 현재 반복거래율 q_N이 기존 q_N과 다릅니다")
    prepared["n_proxy_candidates"] = candidates
    prepared["n_proxy_diagnostics"] = diagnostics
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict, cfg: M2NProxyValueBasisConfig) -> list[dict]:
    common = {
        "q_v": prepared["q_v"],
        "clv_valid": prepared["clv_valid"],
    }
    candidates = prepared["n_proxy_candidates"]
    specifications = (
        (
            QV_ONLY_MODEL_ID,
            "qv_only_gate1_control",
            "qv_only",
            np.zeros_like(prepared["q_n"]),
            1.0,
        ),
        (
            REPEAT_RATE_MODEL_ID,
            "current_repeat_rate_n_plus_qv",
            "repeat_rate",
            candidates["repeat_rate"],
            None,
        ),
        (
            RECENCY_ACTIVITY_MODEL_ID,
            "recency_adjusted_activity_n_plus_qv",
            "recency_adjusted_activity",
            candidates["recency_adjusted_activity"],
            None,
        ),
        (
            BGNBD_MODEL_ID,
            "bgnbd_expected_count_n_plus_qv",
            "bgnbd_expected_count",
            candidates["bgnbd_expected_count"],
            None,
        ),
    )
    return [
        {
            "model_id": model_id,
            "role": role,
            "architecture": "fixed_qv_value_basis",
            "rho": cfg.rho,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "unweighted_plain_bpr",
            "n_proxy": proxy_name,
            "m2_assignment": common | {"q_n": q_n},
            "constant_gate": constant_gate,
        }
        for model_id, role, proxy_name, q_n, constant_gate in specifications
    ]


def _build_model(prepared: dict, cfg: M2NProxyValueBasisConfig, spec: dict):
    data = prepared["data"]
    assignment = spec["m2_assignment"]
    v3.set_seed(cfg.seed)
    return M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v=assignment["q_v"],
        user_clv_valid=assignment["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=spec["rho"],
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        gate_delta=cfg.gate_delta,
        basis_bandwidth=cfg.basis_bandwidth,
        constant_gate=spec["constant_gate"],
        learn_gate_offset=False,
    ).to(v3.DEVICE)


def candidate_reading(metric_rows: dict[str, dict]) -> dict:
    baseline = metric_rows[M1_MODEL_ID]
    qv_only = metric_rows[QV_ONLY_MODEL_ID]

    def beats(model: dict, reference: dict, metrics: tuple[str, ...]) -> bool:
        return all(model[metric] > reference[metric] for metric in metrics)

    def deltas(model: dict, reference: dict, metrics: tuple[str, ...]) -> dict:
        return {
            metric: float(model[metric] - reference[metric]) for metric in metrics
        }

    by_model = {}
    for model_id in TRAINED_MODEL_IDS:
        metrics = metric_rows[model_id]
        total_economic = beats(metrics, baseline, ECONOMIC_METRICS)
        n_increment = (
            None
            if model_id == QV_ONLY_MODEL_ID
            else beats(metrics, qv_only, ECONOMIC_METRICS)
        )
        by_model[model_id] = {
            "total_economic_signal_vs_reused_m1": total_economic,
            "top10_accuracy_beats_reused_m1": beats(
                metrics, baseline, TOP10_ACCURACY_METRICS
            ),
            "n_increment_signal_vs_qv_only": n_increment,
            "development_candidate": bool(total_economic and n_increment)
            if n_increment is not None
            else False,
            "top10_deltas_vs_reused_m1": deltas(
                metrics,
                baseline,
                TOP10_ACCURACY_METRICS + ECONOMIC_METRICS,
            ),
            "top10_deltas_vs_qv_only": None
            if model_id == QV_ONLY_MODEL_ID
            else deltas(
                metrics,
                qv_only,
                TOP10_ACCURACY_METRICS + ECONOMIC_METRICS,
            ),
            "six_accuracy_geomean_ratio_vs_reused_m1": float(
                math.exp(
                    np.mean(
                        [
                            math.log(metrics[name] / baseline[name])
                            for name in ACCURACY_METRICS
                        ]
                    )
                )
            ),
        }
    candidates = [
        model_id
        for model_id in TRAINED_MODEL_IDS
        if by_model[model_id]["development_candidate"]
    ]
    return {
        "by_model": by_model,
        "candidates_for_next_stage": candidates,
        "winner_selected_from_single_seed": False,
        "clv_assignment_tested": False,
        "comparison_scope": (
            "single exposed development seed; M1 is a compatible different-run reference"
        ),
        "next_if_candidates_exist": (
            "inspect complete absolute/comparison/score/gate results before freezing "
            "one candidate for a same-run attribution control"
        ),
        "next_if_no_candidate": (
            "stop these N proxies without tuning their formulas on the same interval"
        ),
        "statistical_note": (
            "no significance, stability, generalization, or final CLV-effect claim"
        ),
    }


def run_m2_n_proxy_screen(
    cfg: M2NProxyValueBasisConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m2_n_proxy_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    reference, provenance = value_basis._load_reused_reference(
        cfg.m1_reference_json,
        expected_model_id=M1_MODEL_ID,
    )
    print(f"\n[재사용 M1] {provenance['path']}")

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        with patch.object(legacy, "_build_model", _build_model):
            arm, model = legacy._run_arm(prepared, cfg, spec)
        arm["n_proxy"] = spec["n_proxy"]
        arm["m2_architecture"] = spec["architecture"]
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    rows = [reference["row"]]
    metric_rows = {M1_MODEL_ID: reference["metrics"]}
    for model_id in TRAINED_MODEL_IDS:
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho": arm["rho"],
                "n_proxy": arm["n_proxy"],
                "m2_architecture": arm["m2_architecture"],
                "execution_source": "current_four_arm_run",
                "source_code_version": CODE_VERSION,
                "source_revision": prepared["revision"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    full_comparison = report_helpers._metric_comparison(
        metric_rows,
        references=(M1_MODEL_ID, QV_ONLY_MODEL_ID),
    )
    comparison = full_comparison.loc[
        (full_comparison["reference"] == M1_MODEL_ID)
        | (
            (full_comparison["reference"] == QV_ONLY_MODEL_ID)
            & full_comparison["model_id"].isin(TRAINED_MODEL_IDS[1:])
        )
    ].reset_index(drop=True)

    score_rows = []
    for model_id in TRAINED_MODEL_IDS:
        users, top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        score_rows.append(
            legacy._score_diagnostics(
                models[model_id], users, top50, model_id=model_id
            )
            | {"n_proxy": arms[model_id]["n_proxy"]}
        )
    score_frame = pd.DataFrame(score_rows)
    gate_frame = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "n_proxy": arms[model_id]["n_proxy"],
                **models[model_id].representation_diagnostics(),
            }
            for model_id in TRAINED_MODEL_IDS
        ]
    )
    n_frame = pd.DataFrame(
        prepared["n_proxy_diagnostics"]["candidate_rows"]
    )
    reading = candidate_reading(metric_rows)

    out = Path(cfg.out_dir)
    stem = f"m2_n_proxy_value_basis_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "gate_diagnostics_csv": out / f"{stem}_gate_diagnostics.csv",
        "n_proxy_diagnostics_csv": out / f"{stem}_n_proxy_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    legacy.test10._atomic_csv(paths["gate_diagnostics_csv"], gate_frame)
    legacy.test10._atomic_csv(paths["n_proxy_diagnostics_csv"], n_frame)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "gate_diagnostic_rows": gate_frame.to_dict("records"),
            "n_proxy_diagnostics": prepared["n_proxy_diagnostics"],
            "screening_reading": reading,
            "reused_reference": {
                "provenance": provenance,
                "arm": reference["arm"],
            },
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["gate_diagnostics"] = gate_frame
    frame.attrs["n_proxy_diagnostics"] = n_frame
    frame.attrs["bgnbd_diagnostics"] = prepared["n_proxy_diagnostics"]["bgnbd"]
    frame.attrs["decision"] = reading
    frame.attrs["reference_provenance"] = provenance
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    print("\n1) 재사용 M1과 A~D M2 절대지표")
    print(frame.to_string(index=False))
    print("\n2) M1 및 q_V-only 대비 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) ID 점수 대비 경제점수 영향력")
    print(score_frame.to_string(index=False))
    print("\n4) N 입력과 BG/NBD 적합 진단")
    print(n_frame.to_string(index=False))
    print(json.dumps(frame.attrs["bgnbd_diagnostics"], ensure_ascii=False, indent=2))
    print("\n5) 게이트 작동 진단")
    print(gate_frame.to_string(index=False))
    print("\n6) 사전 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n7) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_m2_n_proxy_screen()."
    )
