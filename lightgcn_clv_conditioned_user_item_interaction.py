"""Seed-42 historical screen for one CLV-conditioned interaction M2.

The runner trains two matched arms in the same historical development split:

* ``m1_matched_rho0``: the exact non-intervention control;
* ``m2_clv_conditioned_user_item_interaction``: the same model with rho=0.05.

Both arms use one binary LightGCN, uniform negatives, one plain BPR objective,
and one optimizer.  The fixed train-history CLV proxy enters a small jointly
learned user-item interaction block before the single dot-product score.
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

from clv_conditioned_user_item_interaction_model import (
    CLVConditionedUserItemInteractionLightGCN,
)
from clv_dual_axis_model import fixed_percentile_ranks
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gatefree_lowdim as gatefree
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-conditioned-user-item-interaction-historical-screen-v1"
MATCHED_MODEL_ID = "m1_matched_rho0"
MODEL_ID = "m2_clv_conditioned_user_item_interaction"
ID_ONLY_MODEL_ID = "m2_jointly_trained_id_only"


@dataclass(frozen=True)
class ConditionedInteractionConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    id_dim: int = 64
    context_dim: int = 4
    rho: float = 0.05
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    out_dir: str = ""
    baseline_result_dir: str = ""


def configure_conditioned_interaction_run(**overrides) -> ConditionedInteractionConfig:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_clv_conditioned_user_item_interaction_historical_screen_v1"
        ),
        "baseline_result_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_repeatshare_historical_backtest_v1"
        ),
    }
    return validate_config(
        ConditionedInteractionConfig(**(defaults | overrides))
    )


def validate_config(cfg: ConditionedInteractionConfig) -> ConditionedInteractionConfig:
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "time_cutoff": 690,
        "evaluation_days": 7,
        "epochs": 100,
        "id_dim": 64,
        "context_dim": 4,
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


def preflight_summary(cfg: ConditionedInteractionConfig) -> dict:
    cfg = validate_config(cfg)
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
            "historical_clv_proxy": "C_hat=N_hat*V_hat from train history",
            "overall_level": "q_C=train-user midrank percentile of C_hat",
            "composition": "d_NV=q_N-q_V",
            "id_representation": "ordinary 64-d LightGCN layer 0/1/2 mean",
            "interaction_dim": cfg.context_dim,
            "score": "<z_u,z_i> + rho*<q_C*h_u*h_C(d_NV),h_i>",
            "rho": cfg.rho,
            "separate_n_v_scores": False,
            "explicit_item_features": False,
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
            "economic": "price_purchase_amount_weighted_hit@10 does not decrease",
            "mechanism": "high-CLV Top-10 changes and rho=0 is exact non-intervention",
            "statistical_note": "seed 42 exploratory screen; no significance claim",
        },
        "out_dir": cfg.out_dir,
        "baseline_result_dir": cfg.baseline_result_dir,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _config_hash(cfg: ConditionedInteractionConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:12]


def _base_config(cfg: ConditionedInteractionConfig) -> dict:
    return gatefree._base_config(cfg)


def build_clv_conditions(axes: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build overall CLV level and N/V composition from train-only inputs."""

    valid = (
        np.asarray(axes["valid_user"], dtype=bool)
        & np.asarray(axes["activity_valid"], dtype=bool)
        & np.asarray(axes["value_valid"], dtype=bool)
    )
    clv_proxy = np.asarray(axes["clv_proxy"], dtype=np.float64)
    q_c, _ = fixed_percentile_ranks(clv_proxy, clv_proxy, valid)
    q_n = np.asarray(axes["q_n"], dtype=np.float32)
    q_v = np.asarray(axes["q_v"], dtype=np.float32)
    d_nv = np.where(valid, q_n - q_v, 0.0).astype(np.float32)
    q_c[~valid] = 0.0
    if not np.isfinite(q_c).all() or not np.isfinite(d_nv).all():
        raise RuntimeError("CLV 조건에 비유한 값이 있습니다")
    return q_c.astype(np.float32), d_nv, valid


def _prepare(cfg: ConditionedInteractionConfig) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
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
    q_c, d_nv, valid = build_clv_conditions(axes)
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
        "q_c": q_c,
        "d_nv": d_nv,
        "clv_valid": valid,
        "baseline": baseline,
        "meta": meta,
        "thresholds": thresholds,
        "cache": cache,
    }
    prepared["config_hash"] = _config_hash(cfg, input_hash, revision)
    return prepared


def _build_model(prepared: dict, cfg: ConditionedInteractionConfig, rho: float):
    data = prepared["data"]
    v3.set_seed(cfg.seed)
    model = CLVConditionedUserItemInteractionLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        q_c=prepared["q_c"],
        d_nv=prepared["d_nv"],
        user_clv_valid=prepared["clv_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        context_dim=cfg.context_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
    ).to(v3.DEVICE)
    return model, list(model.parameters())


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s42"
    return {
        "result": root / f"{stem}.json",
        "checkpoint": root / f"{stem}.pt",
    }


def _arm_hash(prepared: dict, model_id: str, rho: float) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "rho": rho,
                "seed": 42,
            }
        ).encode()
    ).hexdigest()[:12]


def _load_state(path: Path) -> dict:
    try:
        return torch.load(path, map_location=v3.DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=v3.DEVICE)


def _run_arm(
    prepared: dict,
    cfg: ConditionedInteractionConfig,
    *,
    model_id: str,
    rho: float,
) -> tuple[dict, CLVConditionedUserItemInteractionLightGCN]:
    paths = _arm_paths(prepared, model_id)
    model, params = _build_model(prepared, cfg, rho)
    if paths["result"].exists() and paths["checkpoint"].exists():
        print(f"  [cached] {model_id} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = _load_state(paths["checkpoint"])
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
    """Evaluation-only view of the jointly trained ordinary ID block."""

    def __init__(self, model: CLVConditionedUserItemInteractionLightGCN):
        super().__init__()
        self.model = model

    def embeddings(self, need_value: bool = True):
        user, item = self.model.id_embeddings()
        zu = user.new_zeros((len(user), 1))
        zi = item.new_zeros((len(item), 1))
        return user, item, zu, zi


@torch.no_grad()
def _masked_topk(model, prepared: dict, *, max_k: int) -> tuple[np.ndarray, np.ndarray]:
    user, item, *_ = model.embeddings(need_value=False)
    users = prepared["cache"].users.astype(np.int64)
    topk = np.empty((len(users), max_k), np.int32)
    csr_ptr = prepared["data"]["csr_ptr"]
    csr_items = prepared["data"]["csr_items"]
    batch_size = min(int(prepared["base_cfg"]["EVAL_BATCH"]), 64)
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        tensor_users = torch.as_tensor(
            batch_users, dtype=torch.long, device=user.device
        )
        scores = user.index_select(0, tensor_users) @ item.T
        for row, user_id in enumerate(batch_users):
            left, right = csr_ptr[user_id], csr_ptr[user_id + 1]
            if right > left:
                scores[row, csr_items[left:right]] = -torch.inf
        topk[start : start + len(batch_users)] = (
            scores.topk(max_k, dim=1).indices.cpu().numpy()
        )
    return users, topk


def topk_overlap_summary(
    reference_topk: np.ndarray,
    model_topk: np.ndarray,
    segments: np.ndarray,
    *,
    k: int = 10,
) -> pd.DataFrame:
    reference = np.asarray(reference_topk)[:, :k]
    model = np.asarray(model_topk)[:, :k]
    segments = np.asarray(segments, dtype=object)
    if reference.shape != model.shape or len(reference) != len(segments):
        raise ValueError("Top-K와 CLV 구간 배열 크기가 다릅니다")
    changed = []
    order_changed = []
    jaccard = []
    for left, right in zip(reference, model, strict=True):
        left_set, right_set = set(map(int, left)), set(map(int, right))
        union = left_set | right_set
        changed.append(float(left_set != right_set))
        order_changed.append(float(not np.array_equal(left, right)))
        jaccard.append(len(left_set & right_set) / max(len(union), 1))
    changed = np.asarray(changed)
    order_changed = np.asarray(order_changed)
    jaccard = np.asarray(jaccard)
    rows = []
    for group in ("전체", "저CLV", "중CLV", "고CLV"):
        mask = np.ones(len(segments), dtype=bool) if group == "전체" else segments == group
        rows.append(
            {
                "group": group,
                "n_users": int(mask.sum()),
                "top10_set_changed_user_share": float(changed[mask].mean()),
                "top10_order_changed_user_share": float(order_changed[mask].mean()),
                "top10_mean_jaccard": float(jaccard[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


@torch.no_grad()
def _score_diagnostics(
    model: CLVConditionedUserItemInteractionLightGCN,
    users: np.ndarray,
    top50: np.ndarray,
    prepared: dict,
) -> pd.DataFrame:
    width = top50.shape[1]
    pair_users = np.repeat(users.astype(np.int64), width)
    pair_items = top50.reshape(-1).astype(np.int64)
    id_values = []
    interaction_values = []
    weighted_values = []
    batch_size = 65536
    for start in range(0, len(pair_users), batch_size):
        ut = torch.as_tensor(
            pair_users[start : start + batch_size],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        it = torch.as_tensor(
            pair_items[start : start + batch_size],
            dtype=torch.long,
            device=v3.DEVICE,
        )
        sid, raw, weighted = model.candidate_score_components(ut, it)
        id_values.append(sid.cpu().numpy())
        interaction_values.append(raw.cpu().numpy())
        weighted_values.append(weighted.cpu().numpy())
    sid = np.concatenate(id_values).astype(np.float64)
    raw = np.concatenate(interaction_values).astype(np.float64)
    weighted = np.concatenate(weighted_values).astype(np.float64)
    full = sid + weighted
    per_user_abs = np.abs(raw).reshape(len(users), width).mean(axis=1)
    q_c = prepared["q_c"][users]
    d_nv = prepared["d_nv"][users]
    q_corr = pd.Series(per_user_abs).corr(pd.Series(q_c), method="spearman")
    d_corr = pd.Series(per_user_abs).corr(pd.Series(d_nv), method="spearman")
    id_std = float(sid.std())
    rows = [
        {
            "group": "전체",
            "candidate_pair_count": len(sid),
            "id_score_std": id_std,
            "raw_interaction_std": float(raw.std()),
            "weighted_interaction_std": float(weighted.std()),
            "full_score_std": float(full.std()),
            "weighted_to_id_std_ratio": (
                float(weighted.std() / id_std) if id_std > 0 else np.nan
            ),
            "mean_abs_raw_interaction": float(np.abs(raw).mean()),
            "mean_abs_weighted_interaction": float(np.abs(weighted).mean()),
            "per_user_interaction_abs_q_c_spearman": float(q_corr),
            "per_user_interaction_abs_d_nv_spearman": float(d_corr),
        }
    ]
    segments = prepared["cache"].seg
    for group in ("저CLV", "중CLV", "고CLV"):
        mask = segments == group
        rows.append(
            {
                "group": group,
                "candidate_pair_count": int(mask.sum() * width),
                "mean_abs_raw_interaction": float(per_user_abs[mask].mean()),
                "mean_abs_weighted_interaction": float(
                    (model.rho * per_user_abs[mask]).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _metric_comparison(metric_rows: dict[str, dict], references: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for reference in references:
        for model_id, metrics in metric_rows.items():
            if model_id == reference:
                continue
            for metric in sorted(set(metric_rows[reference]) & set(metrics)):
                left = metric_rows[reference][metric]
                right = metrics[metric]
                if not isinstance(left, (int, float, np.number)) or not isinstance(
                    right, (int, float, np.number)
                ):
                    continue
                rows.append(
                    {
                        "reference": reference,
                        "model_id": model_id,
                        "metric": metric,
                        "reference_value": left,
                        "model_value": right,
                        "absolute_delta": right - left,
                        "relative_change_pct": (
                            100.0 * (right - left) / left if left != 0 else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def screening_reading(
    matched_metrics: dict,
    model_metrics: dict,
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
    accuracy_ratios = {
        metric: float(model_metrics[metric]) / float(matched_metrics[metric])
        for metric in accuracy
    }
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
        and economic_delta >= 0.0
        and high_changed > 0.0
        and rho0_exact
    )
    return {
        "positive_screen": positive,
        "accuracy_ratios": accuracy_ratios,
        "high_clv_recall@10_delta": high_recall_delta,
        "high_clv_ndcg@10_delta": high_ndcg_delta,
        "price_purchase_amount_weighted_hit@10_delta": economic_delta,
        "high_clv_top10_changed_user_share": high_changed,
        "rho0_exact_nonintervention": rho0_exact,
        "next_if_positive": "run several development seeds, then H&M before final test",
        "statistical_note": "seed 42 exploratory screen; no significance claim",
    }


def run_conditioned_interaction_screen(
    cfg: ConditionedInteractionConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_conditioned_interaction_run())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n===== matched rho=0 | seed 42 | fixed 100 epochs =====")
    matched, matched_model = _run_arm(
        prepared, cfg, model_id=MATCHED_MODEL_ID, rho=0.0
    )
    print("\n===== CLV interaction rho=0.05 | seed 42 | fixed 100 epochs =====")
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

    users, matched_top50 = _masked_topk(
        matched_model, prepared, max_k=cfg.diagnostic_max_k
    )
    active_users, active_top50 = _masked_topk(
        active_model, prepared, max_k=cfg.diagnostic_max_k
    )
    if not np.array_equal(users, active_users):
        raise RuntimeError("matched와 M2 평가 사용자 순서가 다릅니다")
    overlap = topk_overlap_summary(
        matched_top50,
        active_top50,
        prepared["cache"].seg,
        k=10,
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
    comparison = _metric_comparison(
        metric_rows, references=(MATCHED_MODEL_ID, "m1_64")
    )
    reading = screening_reading(
        matched["metrics"], active["metrics"], overlap, matched["diagnostics"]
    )

    stem = f"m2_clv_conditioned_user_item_interaction_{prepared['config_hash']}"
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
    frame.attrs["comparison"] = comparison
    frame.attrs["top10_overlap"] = overlap
    frame.attrs["score_diagnostics"] = score_diagnostics
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
            preflight_summary(configure_conditioned_interaction_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
