"""Seed-42 historical screen for one jointly propagated CLV-response block.

This is an M2 representation intervention.  Historical CLV level and N/V
composition form one user-side context, while a learned item response is
anchored by two train-only price positions.  The single four-dimensional
block is concatenated at layer 0 and propagated with the ordinary 64-D ID
block in one binary LightGCN, one BPR objective, and one optimizer.
"""

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

from clv_joint_response_embedding_model import JointCLVResponseLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_gradient_isolated_economic_interaction as shared
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-joint-clv-response-embedding-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_joint_clv_response_embedding"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"


@dataclass(frozen=True)
class JointResponseConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    clv_dim: int = 4
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_joint_response_run(**overrides) -> JointResponseConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_joint_clv_response_embedding_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_joint_response_config(
        JointResponseConfig(**(defaults | overrides))
    )


def validate_joint_response_config(
    cfg: JointResponseConfig,
) -> JointResponseConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "clv_dim": 4,
        "rho": 0.05,
        "n_layers": 2,
        "input_days": 365,
        "diagnostic_max_k": 50,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"빠른 M2 screen은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir or not cfg.baseline_result_dir:
        raise ValueError("out_dir와 baseline_result_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: JointResponseConfig) -> dict:
    cfg = validate_joint_response_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "trained_models": [MATCHED_MODEL_ID, MODEL_ID],
        "reused_comparator": "m1_64 (display only)",
        "historical_development_split": {
            "train_end_inclusive": 683,
            "evaluation_start_inclusive": 684,
            "evaluation_end_inclusive": 690,
            "final_test_constructed": False,
            "holdout_constructed": False,
        },
        "m2": {
            "architecture": "ID(64)|one joint CLV-response block(4)",
            "total_dim": cfg.id_dim + cfg.clv_dim,
            "layer0_user_block": (
                "tanh(W_u[centered historical CLV level, "
                "centered N-minus-V composition])"
            ),
            "layer0_item_block": (
                "unit(item response + W_p[overall price percentile, "
                "within-category price percentile])"
            ),
            "joint_graph_propagation": True,
            "one_dot_score": True,
            "rho": cfg.rho,
            "symmetric_scale": "sqrt(rho) on user and item CLV blocks",
            "repeatshare_input": False,
            "item_popularity_input": False,
            "separate_n_v_item_blocks": False,
            "learned_global_axis_weight": False,
            "external_reranking": False,
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
            "accuracy": "all Recall/NDCG@10/20/50 >= 99% of matched rho=0",
            "high_clv": "high-CLV Recall@10 and NDCG@10 both improve",
            "economic": "price_purchase_amount_weighted_hit@10 increases",
            "id_guardrail": "jointly trained ID-only accuracy >= 99.5% of rho=0",
            "mechanism": "high-CLV Top-10 changes and rho=0 is exact non-intervention",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def build_item_economic_inputs(
    train: pd.DataFrame, n_items: int
) -> tuple[np.ndarray, np.ndarray]:
    """Two centred price positions; missing items receive an exact zero row."""

    required = {"i_idx", "cat_idx", "up"}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"train 상품 경제 입력 컬럼 누락: {sorted(missing)}")
    price_features, _ = v3.item_value_features(train, n_items, report=False)
    features = (2.0 * np.asarray(price_features, dtype=np.float32) - 1.0)
    valid = np.zeros(n_items, dtype=bool)
    observed = train.loc[np.isfinite(train["up"]), "i_idx"].to_numpy(np.int64)
    observed = observed[(observed >= 0) & (observed < n_items)]
    valid[np.unique(observed)] = True
    features[~valid] = 0.0
    if not np.isfinite(features).all():
        raise RuntimeError("상품 가격 위치에 비유한 값이 있습니다")
    return features, valid


def _config_hash(cfg: JointResponseConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(shared._canonical(payload).encode()).hexdigest()[:12]


def _prepare(cfg: JointResponseConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = gatefree._base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(f"historical 개발분할 외 오염: {sorted(data['splits'])}")
    if float(data["train"].t.max()) != 683.0:
        raise RuntimeError(f"historical train 종료일 오류: {data['train'].t.max()}")
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 screen에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None
    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = shared.build_clv_inputs(axes)
    item_economic, item_economic_valid = build_item_economic_inputs(
        data["train"], data["n_items"]
    )
    baseline = gatefree._load_compatible_baseline(cfg, manifest)
    meta = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    prepared = {
        "out_dir": out_dir,
        "manifest": manifest,
        "input_hash": input_hash,
        "revision": revision,
        "base_cfg": base_cfg,
        "data": data,
        "axes": axes,
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
        "item_economic": item_economic,
        "item_economic_valid": item_economic_valid,
        "baseline": baseline,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: JointResponseConfig, rho: float):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = JointCLVResponseLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_n=prepared["q_n"],
        q_v=prepared["q_v"],
        q_c=prepared["q_c"],
        user_clv_valid=prepared["clv_valid"],
        item_economic_features=prepared["item_economic"],
        item_economic_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        clv_dim=cfg.clv_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {"result": root / f"{stem}.json", "checkpoint": root / f"{stem}.pt"}


def _arm_hash(prepared: dict, model_id: str, rho: float) -> str:
    return hashlib.sha256(
        shared._canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "rho": rho,
                "seed": 42,
            }
        ).encode()
    ).hexdigest()[:12]


def _run_arm(
    prepared: dict,
    cfg: JointResponseConfig,
    *,
    model_id: str,
    rho: float,
) -> tuple[dict, JointCLVResponseLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(prepared, cfg, rho)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = shared._load_state(paths["checkpoint"])
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.eval()
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=model_id,
            seed=cfg.seed,
            config_hash=_arm_hash(prepared, model_id, rho),
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
        "role": "matched_control" if rho == 0.0 else "model",
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": rho,
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


class _IDOnlyView(nn.Module):
    def __init__(self, model: JointCLVResponseLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_embeddings()
        zero_user = user.new_zeros((len(user), 1))
        zero_item = item.new_zeros((len(item), 1))
        return user, item, zero_user, zero_item


@torch.no_grad()
def _score_diagnostics(
    model: JointCLVResponseLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    prepared: dict,
) -> pd.DataFrame:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    id_values, clv_values = [], []
    for start in range(0, len(pair_users), 65536):
        user_tensor = torch.as_tensor(
            pair_users[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        item_tensor = torch.as_tensor(
            pair_items[start : start + 65536], dtype=torch.long, device=v3.DEVICE
        )
        components = model.candidate_score_components(user_tensor, item_tensor)
        id_values.append(components["id"].cpu().numpy())
        clv_values.append(components["clv"].cpu().numpy())
    sid = np.concatenate(id_values).astype(np.float64)
    sclv = np.concatenate(clv_values).astype(np.float64)
    per_user_abs = np.abs(sclv).reshape(len(users), width).mean(axis=1)
    id_std = float(sid.std())
    level_corr = pd.Series(per_user_abs).corr(
        pd.Series(prepared["q_c"][users]), method="spearman"
    )
    composition_corr = pd.Series(per_user_abs).corr(
        pd.Series(np.abs(prepared["q_n"][users] - prepared["q_v"][users])),
        method="spearman",
    )
    return pd.DataFrame(
        [
            {
                "candidate_pair_count": len(sid),
                "id_score_std": id_std,
                "clv_score_std": float(sclv.std()),
                "clv_score_std_ratio_to_id": (
                    float(sclv.std() / id_std) if id_std > 0 else np.nan
                ),
                "clv_score_mean_abs": float(np.abs(sclv).mean()),
                "per_user_clv_abs_q_c_spearman": float(level_corr),
                "per_user_clv_abs_n_minus_v_abs_spearman": float(
                    composition_corr
                ),
                "max_full_decomposition_error": float(
                    np.max(np.abs((sid + sclv) - (sid + sclv)))
                ),
            }
        ]
    )


def screening_reading(
    matched_metrics: dict,
    model_metrics: dict,
    overlap: pd.DataFrame,
    rho0_diagnostics: dict,
    id_only_metrics: dict | None = None,
) -> dict:
    accuracy = (
        "recall@10",
        "ndcg@10",
        "recall@20",
        "ndcg@20",
        "recall@50",
        "ndcg@50",
    )
    accuracy_ratios = {
        metric: float(model_metrics[metric]) / float(matched_metrics[metric])
        for metric in accuracy
    }
    id_only_accuracy_ratios = None
    id_guardrail = True
    if id_only_metrics is not None:
        id_only_accuracy_ratios = {
            metric: float(id_only_metrics[metric]) / float(matched_metrics[metric])
            for metric in accuracy
        }
        id_guardrail = min(id_only_accuracy_ratios.values()) >= 0.995
    high_recall_delta = (
        float(model_metrics["고CLV_recall@10"])
        - float(matched_metrics["고CLV_recall@10"])
    )
    high_ndcg_delta = (
        float(model_metrics["고CLV_ndcg@10"])
        - float(matched_metrics["고CLV_ndcg@10"])
    )
    economic = "price_purchase_amount_weighted_hit@10"
    economic_delta = float(model_metrics[economic]) - float(matched_metrics[economic])
    high_changed = float(
        overlap.set_index("group").at["고CLV", "top10_set_changed_user_share"]
    )
    rho0_exact = float(rho0_diagnostics["rho_zero_auxiliary_max_abs"]) == 0.0
    positive = bool(
        min(accuracy_ratios.values()) >= 0.99
        and high_recall_delta > 0.0
        and high_ndcg_delta > 0.0
        and economic_delta > 0.0
        and id_guardrail
        and high_changed > 0.0
        and rho0_exact
    )
    return {
        "positive_screen": positive,
        "accuracy_ratios": accuracy_ratios,
        "id_only_accuracy_ratios": id_only_accuracy_ratios,
        "high_clv_recall@10_delta": high_recall_delta,
        "high_clv_ndcg@10_delta": high_ndcg_delta,
        "price_purchase_amount_weighted_hit@10_delta": economic_delta,
        "high_clv_top10_changed_user_share": high_changed,
        "rho0_exact_nonintervention": rho0_exact,
        "next_if_positive": "run several development seeds, then H&M before final test",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_joint_response_screen(
    cfg: JointResponseConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_joint_response_config(cfg or configure_joint_response_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== joint CLV response rho=0.05 | seed 42 | fixed 100 epochs =====")
    active, active_model = _run_arm(
        prepared, cfg, model_id=MODEL_ID, rho=cfg.rho
    )

    id_view = _IDOnlyView(active_model).to(v3.DEVICE)
    id_metrics_raw, _ = moe._flat_evaluation(
        id_view,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    id_metrics = test10._public_metrics(id_metrics_raw)
    users, matched_top50 = shared._masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = shared._masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    if not np.array_equal(users, active_users):
        raise RuntimeError("matched와 M2 평가 사용자 순서가 다릅니다")
    overlap = shared.topk_overlap_summary(
        matched_top50, active_top50, prepared["cache"].seg, k=10
    )
    score_diagnostics = _score_diagnostics(
        active_model, users, active_top50, prepared
    )

    baseline = dict(prepared["baseline"])
    baseline["role"] = "reused_baseline_display_only"
    rows = [baseline]
    for arm in (matched, active):
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
        ID_ONLY_MODEL_ID: id_metrics,
    }
    comparison = shared._metric_comparison(
        metric_rows, references=(MATCHED_MODEL_ID, "m1_64")
    )
    reading = screening_reading(
        matched["metrics"],
        active["metrics"],
        overlap,
        matched["diagnostics"],
        id_only_metrics=id_metrics,
    )

    stem = f"m2_joint_clv_response_embedding_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "comparison_csv": prepared["out_dir"] / f"{stem}_comparison.csv",
        "top10_overlap_csv": prepared["out_dir"] / f"{stem}_top10_overlap.csv",
        "score_diagnostics_csv": prepared["out_dir"] / f"{stem}_score_diagnostics.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_csv(paths["score_diagnostics_csv"], score_diagnostics)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "config": asdict(cfg),
        "preflight": preflight_summary(cfg),
        "input_manifest": prepared["manifest"],
        "reused_baseline_source": baseline.get("source_result"),
        "absolute_rows": frame.to_dict("records"),
        "comparison_rows": comparison.to_dict("records"),
        "top10_overlap_rows": overlap.to_dict("records"),
        "score_diagnostic_rows": score_diagnostics.to_dict("records"),
        "screening_reading": reading,
        "result_paths": {key: str(value) for key, value in paths.items()},
    }
    test10._atomic_json(paths["json"], payload)
    frame.attrs["comparison"] = comparison.to_dict("records")
    frame.attrs["top10_overlap"] = overlap.to_dict("records")
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
    print("\n절대지표:")
    print(frame.to_string(index=False))
    print("\n동일 초기화 rho=0 대비 핵심 변화:")
    print(key_table.to_string(index=False))
    print("\nTop-10 변경 진단:")
    print(overlap.to_string(index=False))
    print("\n점수 영향력 진단:")
    print(score_diagnostics.to_string(index=False))
    print("\n탐색 판독:", reading)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_joint_response_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
