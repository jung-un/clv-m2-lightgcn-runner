"""Seed-42 screen for a hard high-CLV routed item-response subspace."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from clv_high_clv_item_response_model import HighCLVItemResponseLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_constrained_economic_embedding as prior
import lightgcn_clv_gated_relation_overall_price as shared_inputs
import lightgcn_clv_gradient_isolated_economic_interaction as helpers
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-high-clv-routed-item-response-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_high_clv_routed_item_response"
SHUFFLED_MODEL_ID = "m2_degree_matched_high_gate_shuffle"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"


@dataclass(frozen=True)
class HighCLVItemResponseConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    response_dim: int = 8
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    include_degree_matched_shuffle: bool = True
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_high_clv_item_response_run(
    **overrides,
) -> HighCLVItemResponseConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_high_clv_routed_item_response_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_high_clv_item_response_config(
        HighCLVItemResponseConfig(**(defaults | overrides))
    )


def validate_high_clv_item_response_config(
    cfg: HighCLVItemResponseConfig,
) -> HighCLVItemResponseConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "response_dim": 8,
        "rho": 0.05,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "include_degree_matched_shuffle": True,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: HighCLVItemResponseConfig) -> dict:
    cfg = validate_high_clv_item_response_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MATCHED_MODEL_ID, MODEL_ID, SHUFFLED_MODEL_ID],
        "reused_comparator": "m1_64 (display only)",
        "research_axis": "M2 representation intervention",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|high-CLV-routed item response(8)",
            "total_dim": cfg.id_dim + cfg.response_dim,
            "historical_clv": "training-window N_hat*V_hat proxy",
            "clv_use": "fixed high-segment hard routing",
            "high_segment_threshold": "training-population upper CLV boundary",
            "auxiliary_user_layer0": "exact zero; no free user response table",
            "auxiliary_item_layer0": "trainable unit-row item response table",
            "auxiliary_user_representation": "purchase-history propagation only",
            "gate_location": "high-CLV mask after every auxiliary user hop",
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "score_formula": "S_ID + rho*g_high(u)*<Z_u^H,Z_i^H>",
            "rho": cfg.rho,
            "auxiliary_score_abs_bound": cfg.rho,
            "repeatshare_input": False,
            "item_popularity_input": False,
            "explicit_item_price": False,
            "external_reranking": False,
            "degree_matched_gate_shuffle": True,
        },
        "fixed": {
            "graph": "binary",
            "negative_sampling": "uniform",
            "sample_weighting": False,
            "loss": "one plain pairwise BPR plus existing sampled ID L2",
            "new_loss_term": False,
            "one_training_loop_and_optimizer": True,
            "min_user_interactions": 1,
            "min_item_interactions": 1,
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
        },
        "reading_rule": {
            "baseline_accuracy": (
                "all Recall/NDCG@10/20/50 >= 99% of matched rho=0, "
                "with Recall/NDCG@10 both improved"
            ),
            "high_clv": (
                "full improves high-CLV Recall/NDCG@10 versus matched and "
                "jointly-trained ID-only"
            ),
            "semantic_attribution": (
                "observed high-CLV routing beats the degree-matched shuffled "
                "gate on high-CLV Recall/NDCG@10 and six-metric geometric mean"
            ),
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _config_hash(
    cfg: HighCLVItemResponseConfig, input_hash: str, revision: str
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]


def _gate_hash(gate: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(gate, dtype=np.uint8).tobytes()).hexdigest()


def _derive_high_gates(
    proxy: np.ndarray, high_threshold: float, source_user: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    proxy = np.asarray(proxy, dtype=np.float64)
    source = np.asarray(source_user, dtype=np.int64)
    if proxy.ndim != 1 or source.shape != proxy.shape:
        raise ValueError("CLV proxy와 순열 source shape이 다릅니다")
    if not np.array_equal(np.sort(source), np.arange(len(source))):
        raise ValueError("source_user는 전체 사용자의 순열이어야 합니다")
    observed = np.isfinite(proxy) & (proxy >= float(high_threshold))
    if not 0 < int(observed.sum()) < len(observed):
        raise RuntimeError("고CLV hard gate가 유효한 두 집단을 만들지 못했습니다")
    shuffled = observed[source]
    changed_share = float(np.mean(observed != shuffled))
    if observed.sum() != shuffled.sum() or changed_share <= 0.0:
        raise RuntimeError("degree-matched gate 순열이 gate 배정을 바꾸지 못했습니다")
    return observed.astype(np.float32), shuffled.astype(np.float32), changed_share


def _prepare(cfg: HighCLVItemResponseConfig) -> dict:
    prepared = shared_inputs._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    proxy = np.asarray(prepared["axes"]["clv_proxy"], dtype=np.float64)
    high_threshold = float(prepared["thresholds"][1])
    shuffled = prior._degree_matched_clv_shuffle(prepared, cfg)
    source = np.asarray(shuffled["source_user"], dtype=np.int64)
    observed_gate, shuffled_gate, changed_share = _derive_high_gates(
        proxy, high_threshold, source
    )
    prepared["observed_high_gate"] = observed_gate
    prepared["shuffled_high_gate"] = shuffled_gate
    prepared["gate_shuffle"] = {
        **shuffled,
        "changed_high_gate_user_share": changed_share,
        "observed_high_user_count": int(observed_gate.sum()),
        "shuffled_high_user_count": int(shuffled_gate.sum()),
    }
    return prepared


def _build_model(
    prepared: dict,
    cfg: HighCLVItemResponseConfig,
    rho: float,
    *,
    high_gate: np.ndarray,
) -> tuple[HighCLVItemResponseLightGCN, list[nn.Parameter]]:
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = HighCLVItemResponseLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        high_clv_gate=high_gate,
        adj=data["adj"],
        id_dim=cfg.id_dim,
        response_dim=cfg.response_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


class _IDOnlyView(nn.Module):
    def __init__(self, model: HighCLVItemResponseLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_embeddings()
        zero_user = user.new_zeros((len(user), 1))
        zero_item = item.new_zeros((len(item), 1))
        return user, item, zero_user, zero_item


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(
    prepared: dict, model_id: str, rho: float, assignment_name: str, gate: np.ndarray
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": model_id,
        "rho": rho,
        "gate_assignment": assignment_name,
        "gate_hash": _gate_hash(gate),
        "seed": 42,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[
        :12
    ]


def _run_arm(
    prepared: dict,
    cfg: HighCLVItemResponseConfig,
    *,
    model_id: str,
    rho: float,
    high_gate: np.ndarray,
    assignment_name: str,
) -> tuple[dict, HighCLVItemResponseLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(
        prepared, cfg, rho, high_gate=np.asarray(high_gate, dtype=np.float32)
    )
    expected_gate_hash = _gate_hash(high_gate)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = helpers._load_state(paths["checkpoint"])
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        if checkpoint.get("high_gate_hash") != expected_gate_hash:
            raise RuntimeError("cached checkpoint와 현재 high-CLV gate가 다릅니다")
        print(f"  [cached] {model_id} 완료 결과 재사용")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(
                prepared, model_id, rho, assignment_name, high_gate
            ),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = test10._fixed_epoch_train(
        model, params, prepared, cfg, model_id, cfg.seed, store
    )
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": model_id,
            "rho": rho,
            "gate_assignment": assignment_name,
            "high_gate_hash": expected_gate_hash,
            "config": asdict(cfg),
            "training": training,
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": model_id,
        "role": (
            "matched_control"
            if rho == 0.0
            else "attribution_control"
            if assignment_name == "degree_matched_shuffle"
            else "model"
        ),
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
        "gate_assignment": assignment_name,
        "high_gate_hash": expected_gate_hash,
        "metrics": test10._public_metrics(metrics),
        "diagnostics": model.representation_diagnostics(),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
    }
    test10._atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="historical_development_days_684_690",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload, model


@torch.no_grad()
def _score_diagnostics(
    model: HighCLVItemResponseLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
) -> pd.DataFrame:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    collected = {key: [] for key in ("id", "response", "full")}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for key in collected:
            collected[key].append(components[key].cpu().numpy())
    values = {
        key: np.concatenate(chunks).astype(np.float64)
        for key, chunks in collected.items()
    }
    high_pair = np.repeat(model.high_clv_gate[users].cpu().numpy().astype(bool), width)
    id_std = float(values["id"].std())
    response_std = float(values["response"].std())
    return pd.DataFrame(
        [
            {
                "candidate_pair_count": len(pair_users),
                "high_gate_candidate_pair_count": int(high_pair.sum()),
                "id_score_std": id_std,
                "response_score_std": response_std,
                "response_score_std_ratio_to_id": (
                    response_std / id_std if id_std > 0 else np.nan
                ),
                "high_gate_response_score_std": float(
                    values["response"][high_pair].std()
                ),
                "non_high_response_max_abs": float(
                    np.abs(values["response"][~high_pair]).max(initial=0.0)
                ),
                "max_full_decomposition_error": float(
                    np.max(
                        np.abs(
                            values["full"] - values["id"] - values["response"]
                        )
                    )
                ),
            }
        ]
    )


def _screening_reading(
    matched: dict,
    active: dict,
    shuffled: dict,
    id_only: dict,
    overlap: pd.DataFrame,
    rho0_diagnostics: dict,
) -> dict:
    accuracy = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    )
    ratios = {metric: float(active[metric] / matched[metric]) for metric in accuracy}
    shuffle_ratios = {
        metric: float(active[metric] / shuffled[metric]) for metric in accuracy
    }
    high_metrics = ("고CLV_recall@10", "고CLV_ndcg@10")
    high_vs_matched = {
        metric: float(active[metric] - matched[metric]) for metric in high_metrics
    }
    high_vs_id = {
        metric: float(active[metric] - id_only[metric]) for metric in high_metrics
    }
    high_vs_shuffle = {
        metric: float(active[metric] - shuffled[metric]) for metric in high_metrics
    }
    top10_improved = bool(
        active["recall@10"] > matched["recall@10"]
        and active["ndcg@10"] > matched["ndcg@10"]
    )
    baseline_guard = min(ratios.values()) >= 0.99
    direct_high = all(value > 0.0 for value in high_vs_matched.values()) and all(
        value > 0.0 for value in high_vs_id.values()
    )
    shuffle_geomean = float(
        np.exp(np.mean(np.log(np.asarray(list(shuffle_ratios.values())))))
    )
    attribution = bool(
        shuffle_geomean > 1.0
        and all(value > 0.0 for value in high_vs_shuffle.values())
    )
    high_changed = float(
        overlap.set_index("group").at["고CLV", "top10_set_changed_user_share"]
    )
    rho0_exact = float(rho0_diagnostics["rho_zero_auxiliary_max_abs"]) == 0.0
    return {
        "positive_screen": bool(
            baseline_guard
            and top10_improved
            and direct_high
            and attribution
            and high_changed > 0.0
            and rho0_exact
        ),
        "baseline_accuracy_guard": baseline_guard,
        "overall_top10_improved": top10_improved,
        "accuracy_ratios_vs_matched": ratios,
        "high_clv_deltas_vs_matched": high_vs_matched,
        "high_clv_deltas_vs_joint_id_only": high_vs_id,
        "high_clv_deltas_vs_degree_matched_shuffle": high_vs_shuffle,
        "accuracy_ratios_vs_degree_matched_shuffle": shuffle_ratios,
        "accuracy_geomean_ratio_vs_degree_matched_shuffle": shuffle_geomean,
        "high_clv_top10_changed_user_share": high_changed,
        "price_purchase_amount_weighted_hit@10_delta_vs_matched": float(
            active["price_purchase_amount_weighted_hit@10"]
            - matched["price_purchase_amount_weighted_hit@10"]
        ),
        "rho0_exact_nonintervention": rho0_exact,
        "next_if_positive": (
            "repeat development seeds, then split the high-CLV expert by "
            "N/V composition before H&M"
        ),
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_high_clv_item_response_screen(
    cfg: HighCLVItemResponseConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_high_clv_item_response_config(
        cfg or configure_high_clv_item_response_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared,
        cfg,
        model_id=MATCHED_MODEL_ID,
        rho=0.0,
        high_gate=prepared["observed_high_gate"],
        assignment_name="observed",
    )
    print("\n===== high-CLV routed item response | seed 42 =====")
    active, active_model = _run_arm(
        prepared,
        cfg,
        model_id=MODEL_ID,
        rho=cfg.rho,
        high_gate=prepared["observed_high_gate"],
        assignment_name="observed",
    )
    print("\n===== degree-matched high-gate shuffle | seed 42 =====")
    shuffled, shuffled_model = _run_arm(
        prepared,
        cfg,
        model_id=SHUFFLED_MODEL_ID,
        rho=cfg.rho,
        high_gate=prepared["shuffled_high_gate"],
        assignment_name="degree_matched_shuffle",
    )

    id_view = _IDOnlyView(active_model).to(v3.DEVICE)
    id_raw, _ = moe._flat_evaluation(
        id_view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    id_metrics = test10._public_metrics(id_raw)

    users, matched_top50 = helpers._masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = helpers._masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    shuffled_users, shuffled_top50 = helpers._masked_topk(
        shuffled_model, prepared, max_k=cfg.diagnostic_max_k
    )
    id_users, id_top50 = helpers._masked_topk(
        id_view, prepared, max_k=cfg.diagnostic_max_k
    )
    if not (
        np.array_equal(users, active_users)
        and np.array_equal(users, shuffled_users)
        and np.array_equal(users, id_users)
    ):
        raise RuntimeError("대조군과 M2 평가 사용자 순서가 다릅니다")

    overlap = helpers.topk_overlap_summary(
        matched_top50, active_top50, prepared["cache"].seg, k=10
    )
    overlap.insert(0, "reference", MATCHED_MODEL_ID)
    overlap.insert(1, "model_id", MODEL_ID)
    direct_overlap = helpers.topk_overlap_summary(
        id_top50, active_top50, prepared["cache"].seg, k=10
    )
    direct_overlap.insert(0, "reference", ID_ONLY_MODEL_ID)
    direct_overlap.insert(1, "model_id", MODEL_ID)
    attribution_overlap = helpers.topk_overlap_summary(
        shuffled_top50, active_top50, prepared["cache"].seg, k=10
    )
    attribution_overlap.insert(0, "reference", SHUFFLED_MODEL_ID)
    attribution_overlap.insert(1, "model_id", MODEL_ID)
    score_diagnostics = _score_diagnostics(active_model, users, active_top50)

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    for arm in (matched, active, shuffled):
        rows.append(
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                **arm["diagnostics"],
                **arm["metrics"],
            }
        )
    rows.append(
        {
            "model_id": ID_ONLY_MODEL_ID,
            "role": "joint_training_ablation",
            "seed": cfg.seed,
            "split": "historical_development_days_684_690",
            "final_epoch": cfg.epochs,
            **id_metrics,
        }
    )
    frame = pd.DataFrame(rows)
    metric_rows = {
        "m1_64": {
            key: value
            for key, value in baseline.items()
            if "@" in key and isinstance(value, (int, float, np.number))
        },
        MATCHED_MODEL_ID: matched["metrics"],
        MODEL_ID: active["metrics"],
        SHUFFLED_MODEL_ID: shuffled["metrics"],
        ID_ONLY_MODEL_ID: id_metrics,
    }
    comparison = helpers._metric_comparison(
        metric_rows,
        references=(MATCHED_MODEL_ID, "m1_64", SHUFFLED_MODEL_ID, ID_ONLY_MODEL_ID),
    )
    reading = _screening_reading(
        matched["metrics"],
        active["metrics"],
        shuffled["metrics"],
        id_metrics,
        overlap,
        matched["diagnostics"],
    )
    reading["degree_matched_shuffle_changed_high_gate_user_share"] = prepared[
        "gate_shuffle"
    ]["changed_high_gate_user_share"]

    stem = f"m2_high_clv_item_response_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "direct_overlap_csv": prepared["out_dir"] / f"{stem}_direct_overlap.csv",
        "attribution_overlap_csv": (
            prepared["out_dir"] / f"{stem}_attribution_overlap.csv"
        ),
        "score_diagnostics_csv": (
            prepared["out_dir"] / f"{stem}_score_diagnostics.csv"
        ),
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_csv(paths["direct_overlap_csv"], direct_overlap)
    test10._atomic_csv(paths["attribution_overlap_csv"], attribution_overlap)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_diagnostics)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "gate_shuffle": {
            key: value
            for key, value in prepared["gate_shuffle"].items()
            if key
            in {
                "changed_valid_user_share",
                "changed_high_gate_user_share",
                "observed_high_user_count",
                "shuffled_high_user_count",
            }
        },
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "top10_overlap_rows": overlap.to_dict("records"),
        "direct_overlap_rows": direct_overlap.to_dict("records"),
        "attribution_overlap_rows": attribution_overlap.to_dict("records"),
        "score_diagnostic_rows": score_diagnostics.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["top10_overlap"] = overlap.to_dict("records")
    frame.attrs["direct_overlap"] = direct_overlap.to_dict("records")
    frame.attrs["attribution_overlap"] = attribution_overlap.to_dict("records")
    frame.attrs["score_diagnostics"] = score_diagnostics.to_dict("records")
    frame.attrs["screening_reading"] = reading
    frame.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}

    key_metrics = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
        "price_purchase_amount_weighted_hit@10",
        "고CLV_recall@10",
        "고CLV_ndcg@10",
    )
    key_table = comparison[
        (comparison.reference == MATCHED_MODEL_ID)
        & (comparison.model_id == MODEL_ID)
        & comparison.metric.isin(key_metrics)
    ]
    print("\n1) 절대지표: M1, rho=0, 실제 고CLV gate, degree-matched gate shuffle, ID-only")
    print(frame.to_string(index=False))
    print("\n2) 동일 초기화 rho=0 대비 핵심 변화")
    print(key_table.to_string(index=False))
    print("\n3) rho=0 대비 Top-10 변경")
    print(overlap.to_string(index=False))
    print("\n4) 공동학습 ID-only 대비 Top-10 변경")
    print(direct_overlap.to_string(index=False))
    print("\n5) 실제 gate 대 degree-matched shuffle Top-10 변경")
    print(attribution_overlap.to_string(index=False))
    print("\n6) 실제 점수 영향력")
    print(score_diagnostics.to_string(index=False))
    print("\n7) 사전 판정 규칙 결과")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n8) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_high_clv_item_response_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
