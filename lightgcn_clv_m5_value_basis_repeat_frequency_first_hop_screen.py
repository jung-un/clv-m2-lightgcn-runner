"""Two-arm M5 screen: q_V representation, q_N propagation, full-CLV loss."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from clv_m3_repeat_frequency_first_hop import (
    build_repeat_frequency_first_hop,
)
from clv_m5_n_conditioned_value_basis_model import (
    M5NConditionedValueBasisLightGCN,
)
from clv_m5_value_basis_repeat_frequency_first_hop_model import (
    M5ValueBasisRepeatFrequencyFirstHopLightGCN,
)
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_nv_economic_positive_weight as nv
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-value-basis-repeat-frequency-first-hop-two-arm-development-screen-v1"
M3_OFF_MODEL_ID = "m5_value_basis_m3off_personalized_positive_weight_k5"
M3_N_MODEL_ID = "m5_value_basis_n_repeat_reallocated_first_hop_positive_weight_k5"
MODEL_IDS = (M3_OFF_MODEL_ID, M3_N_MODEL_ID)
ECONOMIC_METRICS = (
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)
ACCURACY_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)


@dataclass(frozen=True)
class M5ValueBasisRepeatFrequencyConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    value_dim: int = 3
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    rho_value: float = 0.05
    basis_bandwidth: float = 0.25
    positive_weight_lambda: float = 0.5
    m3_target_strength: float = 0.075
    m3_beta_cap: float = 20.0
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_value_basis_repeat_frequency_screen(
    **overrides,
) -> M5ValueBasisRepeatFrequencyConfig:
    root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": (
            f"{root}_m5_value_basis_repeat_frequency_first_hop_"
            "two_arm_development_screen_v1"
        ),
        "baseline_result_dir": f"{root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(
        M5ValueBasisRepeatFrequencyConfig(**(defaults | overrides))
    )


def validate_config(
    cfg: M5ValueBasisRepeatFrequencyConfig,
) -> M5ValueBasisRepeatFrequencyConfig:
    fixed = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "value_dim": 3,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "rho_value": 0.05,
        "basis_bandwidth": 0.25,
        "positive_weight_lambda": 0.5,
        "m3_target_strength": 0.075,
        "m3_beta_cap": 20.0,
        "n_layers": 2,
        "negative_count": 5,
        "input_days": 365,
        "diagnostic_max_k": 50,
        "shuffle_degree_bins": 10,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"V-M2·N-M3·CLV-M4 최소 screen은 {key}={expected!r}이어야 합니다"
            )
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: M5ValueBasisRepeatFrequencyConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "prior_result_file_required": False,
        "research_question": (
            "Within M5, does using historical purchase-frequency q_N to move "
            "user first-hop mass away from repeatedly purchased train items add "
            "to a q_V value-position M2 plus the unchanged full-CLV M4?"
        ),
        "c3_change_basis": (
            "Earlier M3-N designs strengthened repeated train relationships. This "
            "new hypothesis reverses that rejected direction: q_N controls a "
            "mass-preserving reduction of highly repeated relationships because "
            "all train user-item pairs are excluded from new-item evaluation."
        ),
        "arms": {
            M3_OFF_MODEL_ID: (
                "q_V value-position M2 + unchanged M4; same isolated path with M3 off"
            ),
            M3_N_MODEL_ID: (
                "same M2 and M4 + q_N-conditioned repeat-frequency first-hop reallocation"
            ),
        },
        "m2": {
            "input": "observed q_V only; q_N is not used in the M2 representation",
            "item_side": "train-only item purchase-amount percentile",
            "basis": "fixed normalized low/mid/high Gaussian RBF",
            "rho": cfg.rho_value,
            "basis_bandwidth": cfg.basis_bandwidth,
            "joint_graph_propagation": True,
        },
        "m3": {
            "n_definition": (
                "unchanged q_N percentile of repeat transactions per customer age"
            ),
            "edge_relation": (
                "within-user centered rank of negative log distinct-basket repeat count"
            ),
            "direction": (
                "high q_N moves first-hop mass from more-repeated to less-repeated "
                "observed items"
            ),
            "changed_path": "user receives item messages at layer 1 only",
            "binary_item_and_two_hop_paths_preserved": True,
            "per_user_first_hop_mass_preserved": True,
            "target_log_coefficient_ratio_std": cfg.m3_target_strength,
            "beta_cap": cfg.m3_beta_cap,
        },
        "m4": {
            "q_c": "percentile(n_u*v_u), not q_N*q_V",
            "formula": (
                "1 + 0.5*q_C*item_amount_percentile*clipped_user_bin_fit"
            ),
            "normalization": "mean raw weight over every positive train row",
            "uniform_negative_count": cfg.negative_count,
            "same_observed_assignment_in_both_arms": True,
        },
        "fixed": {
            "new_item_task": True,
            "train_pairs_excluded_from_evaluation": True,
            "min_item_interactions": 1,
            "binary_base_graph": True,
            "negative_sampling": "uniform",
            "epochs": cfg.epochs,
            "validation_or_epoch_selection": False,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
            "pretraining_or_freezing": False,
            "external_reranking": False,
        },
        "reading_rule": {
            "directional_pass": (
                "N-M3 M5 > M3-off M5 on both top-10 economic metrics"
            ),
            "accuracy": "all Recall/NDCG metrics are reported but are not gates",
            "attribution": (
                "not tested in this minimum run; a positive result only licenses "
                "relation-only and degree-matched q_N assignment controls"
            ),
            "statistical_note": (
                "one exposed historical development seed; no significance, "
                "stability, generalization, or final CLV-effect claim"
            ),
        },
        "out_dir": cfg.out_dir,
    }


def _config_hash(
    cfg: M5ValueBasisRepeatFrequencyConfig,
    input_hash: str,
    revision: str,
) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _extract_base_first_hop(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_users, n_items = data["n_users"], data["n_items"]
    adjacency = data["adj"].coalesce()
    indices = adjacency.indices().detach().cpu().numpy()
    values = adjacency.values().detach().cpu().numpy()
    mask = (indices[0] < n_users) & (indices[1] >= n_users)
    users = indices[0, mask].astype(np.int64)
    items = (indices[1, mask] - n_users).astype(np.int64)
    coefficients = values[mask].astype(np.float64)
    order = np.argsort(users * n_items + items, kind="stable")
    users, items, coefficients = users[order], items[order], coefficients[order]
    expected = np.asarray(data["pos_key"], dtype=np.int64)
    if not np.array_equal(users * n_items + items, expected):
        raise RuntimeError("actual LightGCN adjacency and train edge set differ")
    return users, items, coefficients


def _sparse_ui(
    users: np.ndarray,
    items: np.ndarray,
    values: np.ndarray,
    n_users: int,
    n_items: int,
) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([users, items])),
        torch.from_numpy(np.asarray(values, dtype=np.float32)),
        size=(n_users, n_items),
        check_invariants=False,
    ).coalesce().to(v3.DEVICE)


def _prepare(cfg: M5ValueBasisRepeatFrequencyConfig) -> dict:
    with patch.object(
        legacy.common.gatefree, "_load_compatible_baseline", return_value=None
    ):
        prepared = legacy.common._prepare(legacy._common_config(cfg))
    economic = nv.build_nv_economic_inputs(
        prepared["data"]["train"],
        n_users=prepared["data"]["n_users"],
        n_items=prepared["data"]["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        clv_valid=prepared["clv_valid"],
        n_bins=cfg.economic_bins,
        shrinkage_strength=cfg.shrinkage_strength,
        degree_bins=cfg.shuffle_degree_bins,
    )
    prepared.update(economic)
    users, items, base = _extract_base_first_hop(prepared["data"])
    graph = build_repeat_frequency_first_hop(
        prepared["data"]["train"],
        users,
        items,
        base,
        prepared["q_n"],
        prepared["clv_valid"],
        n_users=prepared["data"]["n_users"],
        target_strength=cfg.m3_target_strength,
        beta_cap=cfg.m3_beta_cap,
    )
    base_ui = _sparse_ui(
        users,
        items,
        base,
        prepared["data"]["n_users"],
        prepared["data"]["n_items"],
    )
    active_ui = _sparse_ui(
        users,
        items,
        graph.adjusted_coefficients,
        prepared["data"]["n_users"],
        prepared["data"]["n_items"],
    )
    prepared["m3_repeat_graph"] = graph
    prepared["m3_base_ui"] = base_ui
    prepared["m3_base_iu"] = base_ui.transpose(0, 1).coalesce()
    prepared["m3_operators"] = {"off": base_ui, "n_repeat": active_ui}
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(
    prepared: dict,
    cfg: M5ValueBasisRepeatFrequencyConfig,
) -> list[dict]:
    common = {
        "rho": cfg.rho_value,
        "weighted": True,
        "assignment": prepared,
        "assignment_name": "observed_q_v_and_q_c",
    }
    return [
        {
            "model_id": M3_OFF_MODEL_ID,
            "role": "matched_value_basis_m4_m3_off",
            "m3_arm": "off",
            **common,
        },
        {
            "model_id": M3_N_MODEL_ID,
            "role": "q_n_repeat_frequency_first_hop_plus_value_basis_m4",
            "m3_arm": "n_repeat",
            **common,
        },
    ]


def _model_kwargs(prepared: dict, cfg: M5ValueBasisRepeatFrequencyConfig) -> dict:
    data = prepared["data"]
    return {
        "n_users": data["n_users"],
        "n_items": data["n_items"],
        "user_q_n": prepared["q_n"],
        "user_q_v": prepared["q_v"],
        "user_clv_valid": prepared["clv_valid"],
        "item_price_percentile": prepared["item_amount_percentile"],
        "item_price_valid": prepared["item_economic_valid"],
        "adj": data["adj"],
        "id_dim": cfg.id_dim,
        "rho": cfg.rho_value,
        "n_layers": cfg.n_layers,
        "pref_reg": cfg.pref_reg,
        "basis_bandwidth": cfg.basis_bandwidth,
    }


def _build_model(
    prepared: dict,
    cfg: M5ValueBasisRepeatFrequencyConfig,
    spec: dict,
) -> M5ValueBasisRepeatFrequencyFirstHopLightGCN:
    v3.set_seed(cfg.seed)
    return M5ValueBasisRepeatFrequencyFirstHopLightGCN(
        **_model_kwargs(prepared, cfg),
        base_user_from_item=prepared["m3_base_ui"],
        base_item_from_user=prepared["m3_base_iu"],
        active_user_from_item=prepared["m3_operators"][spec["m3_arm"]],
    ).to(v3.DEVICE)


def _identity_preflight(
    prepared: dict,
    cfg: M5ValueBasisRepeatFrequencyConfig,
) -> dict:
    v3.set_seed(cfg.seed)
    reference = M5NConditionedValueBasisLightGCN(
        **_model_kwargs(prepared, cfg), constant_gate=1.0
    ).to(v3.DEVICE)
    isolated = _build_model(prepared, cfg, arm_specifications(prepared, cfg)[0])
    isolated.load_state_dict(reference.state_dict(), strict=True)
    tolerance = 1e-5 if v3.DEVICE.type == "cuda" else 1e-6
    with torch.no_grad():
        ref_user, ref_item = reference.propagated_embeddings()
        iso_user, iso_item = isolated.propagated_embeddings()
    errors = {
        "user_embedding_max_abs_error": float((ref_user - iso_user).abs().max()),
        "item_embedding_max_abs_error": float((ref_item - iso_item).abs().max()),
    }
    return {
        "device": str(v3.DEVICE),
        "tolerance": tolerance,
        **errors,
        "passed": bool(all(value <= tolerance for value in errors.values())),
    }


def operational_preflight(
    prepared: dict,
    cfg: M5ValueBasisRepeatFrequencyConfig,
) -> dict:
    diagnostics = prepared["m3_repeat_graph"].diagnostics
    identity = _identity_preflight(prepared, cfg)
    result = {
        "target_reached": diagnostics["target_reached"],
        "first_hop_strength": diagnostics["first_hop_strength"],
        "max_user_mass_abs_error": diagnostics["max_user_mass_abs_error"],
        "repeat_direction_spearman": diagnostics[
            "repeat_count_vs_coefficient_ratio_spearman"
        ],
        "has_repeated_edges": diagnostics["repeated_edge_share"] > 0.0,
        "has_users_with_varying_repeat_counts": (
            diagnostics["users_with_varying_edge_repeat_count"] > 0
        ),
        "identity": identity,
    }
    result["passed"] = bool(
        result["target_reached"]
        and result["max_user_mass_abs_error"] < 1e-8
        and result["repeat_direction_spearman"] < 0.0
        and result["has_repeated_edges"]
        and result["has_users_with_varying_repeat_counts"]
        and identity["passed"]
    )
    return result


@torch.no_grad()
def _score_diagnostics(
    model: M5ValueBasisRepeatFrequencyFirstHopLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    *,
    model_id: str,
) -> dict[str, float | int | str]:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    collected = {name: [] for name in ("id", "economic", "full")}
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        for name in collected:
            collected[name].append(components[name].cpu().numpy())
    values = {
        name: np.concatenate(parts).astype(np.float64)
        for name, parts in collected.items()
    }
    id_std = float(values["id"].std())
    economic_std = float(values["economic"].std())
    return {
        "model_id": model_id,
        "candidate_pair_count": int(len(pair_users)),
        "id_score_std": id_std,
        "economic_score_std": economic_std,
        "economic_score_std_ratio_to_id": (
            economic_std / id_std if id_std > 0.0 else np.nan
        ),
        "economic_score_mean_abs": float(np.abs(values["economic"]).mean()),
        "max_full_decomposition_error": float(
            np.max(np.abs(values["full"] - values["id"] - values["economic"]))
        ),
    }


def screening_reading(metric_rows: dict[str, dict], operational: bool = True) -> dict:
    reference = metric_rows[M3_OFF_MODEL_ID]
    candidate = metric_rows[M3_N_MODEL_ID]
    economic_deltas = {
        metric: float(candidate[metric] - reference[metric])
        for metric in ECONOMIC_METRICS
    }
    accuracy_deltas = {
        metric: float(candidate[metric] - reference[metric])
        for metric in ACCURACY_METRICS
    }
    return {
        "operational_preflight_passed": bool(operational),
        "n_m3_directional_pass": bool(
            operational and all(delta > 0.0 for delta in economic_deltas.values())
        ),
        "top10_economic_deltas_n_m3_minus_m3_off": economic_deltas,
        "accuracy_deltas_reported_not_gated": accuracy_deltas,
        "clv_assignment_tested": False,
        "final_candidate_decision_permitted": False,
        "interpretation_scope": (
            "whether q_N-conditioned repeat-frequency first-hop reallocation "
            "deserves attribution controls"
        ),
        "next_if_positive": (
            "add a q_N-free relation-only control and a degree-matched q_N "
            "assignment control before any CLV attribution claim"
        ),
        "next_if_nonpositive": (
            "stop this M3-N mechanism; do not retune strength on the same "
            "exposed development interval"
        ),
        "statistical_note": (
            "one exposed historical development seed; no significance, stability, "
            "generalization, or final CLV-effect claim"
        ),
    }


def run_value_basis_repeat_frequency_screen(
    cfg: M5ValueBasisRepeatFrequencyConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(
        cfg or configure_value_basis_repeat_frequency_screen()
    )
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    operational = operational_preflight(prepared, cfg)
    print("\n[학습 전 M3 작동·동일성 점검]")
    print(json.dumps(operational, ensure_ascii=False, indent=2))
    print("\n[M3 반복횟수 관계 진단]")
    print(
        json.dumps(
            prepared["m3_repeat_graph"].diagnostics,
            ensure_ascii=False,
            indent=2,
        )
    )
    if not operational["passed"]:
        raise RuntimeError("M3 강도·방향·총량·동일성 사전점검 실패로 학습을 중단합니다")

    arms: dict[str, dict] = {}
    models: dict[str, M5ValueBasisRepeatFrequencyFirstHopLightGCN] = {}
    with patch.object(legacy, "_build_model", _build_model):
        for spec in arm_specifications(prepared, cfg):
            print(
                f"\n===== {spec['model_id']} | seed {cfg.seed} | "
                f"fixed {cfg.epochs} epochs ====="
            )
            arms[spec["model_id"]], models[spec["model_id"]] = legacy._run_arm(
                prepared, cfg, spec
            )

    rows = []
    metric_rows = {}
    for model_id in MODEL_IDS:
        arm = arms[model_id]
        metric_rows[model_id] = arm["metrics"]
        rows.append(
            {
                "model_id": model_id,
                "role": arm["role"],
                "seed": arm["seed"],
                "split": arm["split"],
                "final_epoch": arm["final_epoch"],
                "rho_value": arm["rho"],
                "positive_weight_lambda": arm["positive_weight_lambda"],
                "m4_assignment": arm["clv_assignment"],
                **arm["diagnostics"],
                **arm["training"].get("final_diagnostics", {}),
                **arm["metrics"],
            }
        )
    frame = pd.DataFrame(rows)
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M3_OFF_MODEL_ID,)
    )
    comparison = comparison.loc[
        comparison["model_id"] == M3_N_MODEL_ID
    ].reset_index(drop=True)

    topk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    score_rows = []
    for model_id in MODEL_IDS:
        users, top50 = report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        topk[model_id] = (users, top50)
        score_rows.append(
            _score_diagnostics(
                models[model_id], users, top50, model_id=model_id
            )
        )
    score_frame = pd.DataFrame(score_rows)

    reference_users, reference_top50 = topk[M3_OFF_MODEL_ID]
    candidate_users, candidate_top50 = topk[M3_N_MODEL_ID]
    if not np.array_equal(reference_users, candidate_users):
        raise RuntimeError("두 arm의 평가 사용자 순서가 다릅니다")
    reference_top10 = reference_top50[:, :10]
    candidate_top10 = candidate_top50[:, :10]
    identical = np.all(reference_top10 == candidate_top10, axis=1)
    overlap = np.array(
        [
            len(set(left).intersection(right)) / 10.0
            for left, right in zip(reference_top10, candidate_top10, strict=True)
        ],
        dtype=np.float64,
    )
    ranking_change = {
        "evaluation_user_count": int(len(reference_users)),
        "identical_ordered_top10_user_share": float(identical.mean()),
        "changed_ordered_top10_user_share": float(1.0 - identical.mean()),
        "mean_top10_set_overlap": float(overlap.mean()),
    }
    reading = screening_reading(metric_rows, operational=operational["passed"])
    reading["ranking_change"] = ranking_change

    out = Path(cfg.out_dir)
    stem = f"m5_value_basis_n_first_hop_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "score_diagnostics_csv": out / f"{stem}_score_diagnostics.csv",
        "json": out / f"{stem}.json",
    }
    legacy.test10._atomic_csv(paths["absolute_csv"], frame)
    legacy.test10._atomic_csv(paths["comparison_csv"], comparison)
    legacy.test10._atomic_csv(paths["score_diagnostics_csv"], score_frame)
    legacy.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "operational_preflight": operational,
            "input_manifest": prepared["manifest"],
            "m3_repeat_frequency_diagnostics": (
                prepared["m3_repeat_graph"].diagnostics
            ),
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "score_diagnostic_rows": score_frame.to_dict("records"),
            "ranking_change": ranking_change,
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs["comparison"] = comparison
    frame.attrs["score_diagnostics"] = score_frame
    frame.attrs["m3_repeat_frequency_diagnostics"] = (
        prepared["m3_repeat_graph"].diagnostics
    )
    frame.attrs["operational_preflight"] = operational
    frame.attrs["ranking_change"] = ranking_change
    frame.attrs["decision"] = reading
    frame.attrs["preflight"] = summary
    frame.attrs["result_paths"] = {
        key: str(value) for key, value in paths.items()
    }

    print("\n1) M3-off 기준과 N 반복편향 보정 M5 절대지표")
    print(frame.to_string(index=False))
    print("\n2) N-M3 증분 전체 지표 비교")
    print(comparison.to_string(index=False))
    print("\n3) 점수 성분 진단")
    print(score_frame.to_string(index=False))
    print("\n4) Top-10 변경 진단")
    print(json.dumps(ranking_change, ensure_ascii=False, indent=2))
    print("\n5) 사전 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n6) 저장 파일")
    print(json.dumps(frame.attrs["result_paths"], ensure_ascii=False, indent=2))
    return frame


if __name__ == "__main__":
    print(
        "Import this module from the dedicated Colab notebook and call "
        "run_value_basis_repeat_frequency_screen()."
    )
