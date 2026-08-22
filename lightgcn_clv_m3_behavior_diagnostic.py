"""Post-hoc behavior diagnostic for the fixed M3 seed-42 test result.

The module never trains or selects a model.  It reloads the already evaluated
M1 and CLV-influence checkpoints, reconstructs their masked Top-50 lists, and
describes which new-item truths moved in rank for train-history user cohorts.
The protected test interval is used only for descriptive error analysis; a
changed model must be evaluated on a new predeclared split or independent data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

from clv_run_state import file_sha256
import lightgcn_clv_axis_specific_behavior_diagnostic as common
import lightgcn_clv_m3_mass_preserving as m3
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-clv-influence-test-behavior-diagnostic-v1"
M1_ID = m3.MODEL_IDS["m1"]
M3_ID = m3.MODEL_IDS["clv"]
MODELS = (M1_ID, M3_ID)
QUADRANT_LABELS = {
    "low_n_low_v": "낮은 거래횟수·낮은 평균 거래금액",
    "high_n_low_v": "높은 거래횟수·낮은 평균 거래금액",
    "low_n_high_v": "낮은 거래횟수·높은 평균 거래금액",
    "high_n_high_v": "높은 거래횟수·높은 평균 거래금액",
    "invalid": "N/V 산출 불가",
}
ITEM_NUMERIC_TRAITS = (
    "price_percentile",
    "category_price_percentile",
    "mean_unit_price",
    "train_user_count",
    "train_row_count",
    "repeat_purchase_share",
    "median_repeat_gap",
    "mean_transaction_value_share",
)
METRICS = (
    "recall@10",
    "ndcg@10",
    "price_purchase_amount_weighted_hit@10",
    "mean_recommended_price_percentile@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
    "price_purchase_amount_weighted_hit@50",
)


@dataclass(frozen=True)
class M3BehaviorDiagnosticConfig:
    out_dir: str = ""
    run_json: str = ""
    seeds: tuple[int, ...] = m3.PILOT_SEEDS
    ks: tuple[int, ...] = (10, 20, 50)
    score_batch_size: int = 64
    top_item_examples: int = 50
    representative_per_quadrant: int = 2


def configure_m3_behavior_diagnostic(**overrides) -> M3BehaviorDiagnosticConfig:
    defaults = {"out_dir": m3.configure_m3_clv_influence_test_run().out_dir}
    return validate_config(M3BehaviorDiagnosticConfig(**(defaults | overrides)))


def validate_config(cfg: M3BehaviorDiagnosticConfig) -> M3BehaviorDiagnosticConfig:
    if not cfg.out_dir or "m3_clv_influence_test_dunnhumby" not in cfg.out_dir:
        raise ValueError("M3 Dunnhumby test 결과 폴더가 필요합니다")
    if not cfg.seeds or not set(cfg.seeds).issubset(m3.FULL_SEEDS):
        raise ValueError(f"seeds must be a non-empty subset of {m3.FULL_SEEDS}")
    if tuple(cfg.ks) != (10, 20, 50):
        raise ValueError("진단 K는 확정 평가와 같은 10·20·50이어야 합니다")
    if min(cfg.score_batch_size, cfg.top_item_examples, cfg.representative_per_quadrant) <= 0:
        raise ValueError("배치 크기와 예시 개수는 양수여야 합니다")
    return cfg


def preflight_summary(cfg: M3BehaviorDiagnosticConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc error analysis",
        "source": "already evaluated fixed M3 test checkpoints",
        "models": list(MODELS),
        "seeds": list(cfg.seeds),
        "training": False,
        "checkpoint_selection": False,
        "test_interval": "Dunnhumby DAY 698--704",
        "truth": "new user-item pairs absent from merged training through DAY 697",
        "segments": {
            "nv_quadrants": "train-history N/V percentiles split at 0.5",
            "clv_quintiles": "train-history N*V percentile quintiles",
        },
        "outputs": [
            "every user's truth and M1/M3 Top-50 recommendation rows",
            "segment metric and item-profile comparisons",
            "truth-rank transitions and deterministic representative cases",
            "item promotion versus train-neighbor CLV mechanism checks",
            "metric, masking, duplicate, and source-integrity QA",
        ],
        "prohibition": (
            "do not tune a changed model on this test diagnostic; use a new "
            "predeclared time split or independent dataset"
        ),
        "out_dir": cfg.out_dir,
    }


def _percentile(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    out = np.full(len(values), np.nan, dtype=float)
    if valid.any():
        out[valid] = (rankdata(values[valid], method="average") - 0.5) / valid.sum()
    return out, valid


def _user_cohorts(prepared: dict) -> pd.DataFrame:
    graph = prepared["graph"]
    q_n, valid_n = _percentile(graph.n_hat)
    q_v, valid_v = _percentile(graph.v_hat)
    q_clv, valid_clv = _percentile(graph.clv_proxy)
    valid = valid_n & valid_v & valid_clv
    quadrant = np.full(len(q_n), "invalid", dtype=object)
    quadrant[valid & (q_n < 0.5) & (q_v < 0.5)] = "low_n_low_v"
    quadrant[valid & (q_n >= 0.5) & (q_v < 0.5)] = "high_n_low_v"
    quadrant[valid & (q_n < 0.5) & (q_v >= 0.5)] = "low_n_high_v"
    quadrant[valid & (q_n >= 0.5) & (q_v >= 0.5)] = "high_n_high_v"
    quintile = np.full(len(q_n), "invalid", dtype=object)
    quintile[valid] = [f"Q{min(int(value * 5), 4) + 1}" for value in q_clv[valid]]
    train = prepared["data"]["train"]
    raw_user = train.groupby("u_idx", sort=False).u_raw.first()
    frame = pd.DataFrame(
        {
            "user_idx": np.arange(len(q_n), dtype=int),
            "user_id": raw_user.reindex(np.arange(len(q_n))).astype(object).to_numpy(),
            "n_hat": graph.n_hat,
            "v_hat": graph.v_hat,
            "clv_proxy": graph.clv_proxy,
            "q_n": q_n,
            "q_v": q_v,
            "q_clv": q_clv,
            "clv_factor": graph.user_factors["clv"],
            "nv_quadrant": quadrant.astype(str),
            "nv_quadrant_label": [QUADRANT_LABELS[value] for value in quadrant],
            "clv_quintile": quintile.astype(str),
        }
    )
    return frame


def _find_run_json(cfg: M3BehaviorDiagnosticConfig) -> tuple[Path, dict, str]:
    root = Path(cfg.out_dir)
    candidates = (
        [Path(cfg.run_json)]
        if cfg.run_json
        else sorted(
            root.glob("m3_clv_influence_test_????????????.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    if not candidates or not candidates[0].exists():
        raise FileNotFoundError("M3 test 종합 JSON을 찾지 못했습니다")
    path = candidates[0]
    match = re.fullmatch(r"m3_clv_influence_test_([0-9a-f]{12})", path.stem)
    if match is None:
        raise ValueError(f"예상하지 못한 결과 파일명입니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("code_version") != m3.CODE_VERSION:
        raise RuntimeError(
            "결과 runner 버전과 현재 M3 runner가 다릅니다: "
            f"{payload.get('code_version')} != {m3.CODE_VERSION}"
        )
    return path, payload, match.group(1)


def _prepare_source(cfg: M3BehaviorDiagnosticConfig, payload: dict) -> tuple[dict, m3.M3TestConfig]:
    stored = dict(payload["config"])
    stored["out_dir"] = cfg.out_dir
    stored["seeds"] = tuple(stored["seeds"])
    run_cfg = m3.validate_test_config(m3.M3TestConfig(**stored))
    if not set(cfg.seeds).issubset(run_cfg.seeds):
        raise RuntimeError("요청 seed가 저장된 실행에 포함되지 않습니다")
    prepared = m3._prepare(run_cfg)
    stored_manifest_hash = m3.moe.manifest_hash(payload["input_manifest"])
    if stored_manifest_hash != prepared["input_hash"]:
        raise RuntimeError("저장 결과와 현재 원천 데이터 manifest가 다릅니다")
    # Diagnostic source files make the working tree dirty.  Reconstruct the
    # original arm directory with the revision recorded by the fixed run.
    prepared["method_hash"] = m3._method_hash(
        run_cfg, prepared["input_hash"], payload["source_revision"]
    )
    return prepared, run_cfg


def _load_model(prepared: dict, run_cfg: m3.M3TestConfig, model_id: str, seed: int):
    paths = m3._arm_paths(prepared, model_id, seed)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"완료 arm 산출물이 없습니다: {path}")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    if result.get("test_evaluation_count") != 1:
        raise RuntimeError(f"test 평가 횟수가 1이 아닙니다: {paths['result']}")
    if result.get("checkpoint_sha256") != file_sha256(paths["checkpoint"]):
        raise RuntimeError(f"checkpoint hash 불일치: {paths['checkpoint']}")
    model, _ = m3._build_model(prepared, run_cfg, model_id, seed)
    try:
        blob = torch.load(paths["checkpoint"], map_location=v3.DEVICE, weights_only=False)
    except TypeError:
        blob = torch.load(paths["checkpoint"], map_location=v3.DEVICE)
    if blob.get("input_hash") != prepared["input_hash"]:
        raise RuntimeError(f"checkpoint input hash 불일치: {paths['checkpoint']}")
    if blob.get("model_id") != model_id or int(blob.get("seed")) != seed:
        raise RuntimeError(f"checkpoint model/seed 불일치: {paths['checkpoint']}")
    model.load_state_dict(blob["state"], strict=True)
    model.eval()
    return model, result, paths


def _per_user_metrics(
    users: np.ndarray,
    topk: np.ndarray,
    prepared: dict,
    cohorts: pd.DataFrame,
    model_id: str,
    seed: int,
) -> pd.DataFrame:
    cache, meta, data = prepared["cache"], prepared["meta"], prepared["data"]
    item_novelty = -np.log2(meta["pop_prob"] + 1e-12)
    scored = v3.score_topk(
        topk,
        users,
        [10, 20, 50],
        cache.pos_key,
        cache.pos_rev,
        data["n_items"],
        cache.P_arr,
        meta["price_pct"],
        item_novelty,
        meta["cat"],
        cache.ideal,
    )
    frame = cohorts.set_index("user_idx").loc[users].reset_index()
    frame.insert(0, "seed", seed)
    frame.insert(1, "model_id", model_id)
    frame["truth_item_count"] = cache.P_arr[users]
    for k, values in scored.items():
        names = {
            "revenue": "price_purchase_amount_weighted_hit",
            "arp": "mean_recommended_price_percentile",
        }
        for key, array in values.items():
            frame[f"{names.get(key, key)}@{k}"] = array
    return frame


def _recommendation_rows(
    seed: int,
    model_id: str,
    users: np.ndarray,
    topk: np.ndarray,
    prepared: dict,
    cohorts: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> pd.DataFrame:
    cache = prepared["cache"]
    cohort = cohorts.set_index("user_idx")
    records = []
    for row, user in enumerate(users):
        truth_amount = {
            int(item): float(value)
            for item, value in zip(cache.gt[int(user)], cache.rev[int(user)])
        }
        user_info = cohort.loc[int(user)]
        for rank, item in enumerate(topk[row], start=1):
            records.append(
                {
                    "seed": seed,
                    "model_id": model_id,
                    "user_idx": int(user),
                    "user_id": user_info.user_id,
                    "nv_quadrant": user_info.nv_quadrant,
                    "clv_quintile": user_info.clv_quintile,
                    "rank": rank,
                    "item_idx": int(item),
                    "is_test_truth": int(item) in truth_amount,
                    "test_purchase_amount": truth_amount.get(int(item), 0.0),
                }
            )
    return pd.DataFrame(records).merge(item_traits, on="item_idx", how="left")


def _truth_rows(
    seed: int,
    users: np.ndarray,
    topk_by_model: dict[str, np.ndarray],
    prepared: dict,
    cohorts: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> pd.DataFrame:
    cache = prepared["cache"]
    cohort = cohorts.set_index("user_idx")
    rows = []
    for row, user in enumerate(users):
        ranks = {
            model_id: {int(item): rank + 1 for rank, item in enumerate(topk[row])}
            for model_id, topk in topk_by_model.items()
        }
        info = cohort.loc[int(user)]
        for item, amount in zip(cache.gt[int(user)], cache.rev[int(user)]):
            item = int(item)
            m1_rank = ranks[M1_ID].get(item, 51)
            m3_rank = ranks[M3_ID].get(item, 51)
            rows.append(
                {
                    "seed": seed,
                    "user_idx": int(user),
                    "user_id": info.user_id,
                    "nv_quadrant": info.nv_quadrant,
                    "clv_quintile": info.clv_quintile,
                    "item_idx": item,
                    "test_purchase_amount": float(amount),
                    "m1_rank_capped_51": m1_rank,
                    "m3_rank_capped_51": m3_rank,
                    "rank_improvement": m1_rank - m3_rank,
                    "entered_top10": m1_rank > 10 and m3_rank <= 10,
                    "left_top10": m1_rank <= 10 and m3_rank > 10,
                    "entered_top20": m1_rank > 20 and m3_rank <= 20,
                    "left_top20": m1_rank <= 20 and m3_rank > 20,
                    "entered_top50": m1_rank > 50 and m3_rank <= 50,
                    "left_top50": m1_rank <= 50 and m3_rank > 50,
                }
            )
    return pd.DataFrame(rows).merge(item_traits, on="item_idx", how="left")


def _segment_metric_summary(per_user: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for segment_type, label_column in (
        ("nv_quadrant", "nv_quadrant_label"),
        ("clv_quintile", None),
    ):
        for (seed, segment_id), group in per_user.groupby(["seed", segment_type], sort=False):
            m1_frame = group[group.model_id.eq(M1_ID)].set_index("user_idx")
            m3_frame = group[group.model_id.eq(M3_ID)].set_index("user_idx")
            common_users = m1_frame.index.intersection(m3_frame.index)
            for metric in METRICS:
                baseline = m1_frame.loc[common_users, metric].to_numpy(float)
                changed = m3_frame.loc[common_users, metric].to_numpy(float)
                delta = changed - baseline
                rows.append(
                    {
                        "seed": int(seed),
                        "segment_type": segment_type,
                        "segment_id": segment_id,
                        "segment_label": (
                            str(m1_frame.loc[common_users[0], label_column])
                            if label_column and len(common_users)
                            else segment_id
                        ),
                        "n_users": len(common_users),
                        "metric": metric,
                        "m1_mean": float(baseline.mean()),
                        "m3_mean": float(changed.mean()),
                        "mean_delta": float(delta.mean()),
                        "relative_change_pct": (
                            float(delta.mean() / baseline.mean() * 100)
                            if abs(baseline.mean()) > 1e-12 else np.nan
                        ),
                        "improved_user_share": float((delta > 0).mean()),
                        "degraded_user_share": float((delta < 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def _segment_item_profiles(
    recs: pd.DataFrame,
    truth: pd.DataFrame,
    cohorts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_role = truth.assign(role="test_truth")
    rec_role = recs[recs["rank"] <= 10].assign(
        role=lambda frame: np.where(frame.model_id.eq(M1_ID), "m1_top10", "m3_top10")
    )
    columns = ["seed", "user_idx", "item_idx", "category", *ITEM_NUMERIC_TRAITS, "role"]
    occurrences = pd.concat([truth_role[columns], rec_role[columns]], ignore_index=True)
    occurrences = occurrences.merge(
        cohorts[["user_idx", "nv_quadrant", "clv_quintile"]],
        on="user_idx",
        how="left",
        validate="many_to_one",
    )
    summaries, categories = [], []
    for segment_type in ("nv_quadrant", "clv_quintile"):
        keys = ["seed", segment_type, "role"]
        for key, group in occurrences.groupby(keys, sort=False):
            seed, segment_id, role = key
            row = {
                "seed": seed,
                "segment_type": segment_type,
                "segment_id": segment_id,
                "role": role,
                "n_occurrences": len(group),
                "n_unique_items": group.item_idx.nunique(),
            }
            for trait in ITEM_NUMERIC_TRAITS:
                row[f"mean_{trait}"] = float(group[trait].mean())
            counts = group.category.astype(str).value_counts(dropna=False)
            row["top_category"] = counts.index[0] if len(counts) else ""
            row["top_category_share"] = float(counts.iloc[0] / len(group)) if len(group) else np.nan
            summaries.append(row)
            for category, count in counts.items():
                categories.append(
                    {
                        "seed": seed,
                        "segment_type": segment_type,
                        "segment_id": segment_id,
                        "role": role,
                        "category": category,
                        "occurrence_count": int(count),
                        "occurrence_share": float(count / len(group)),
                    }
                )
    return pd.DataFrame(summaries), pd.DataFrame(categories)


def _item_mechanism(
    seed: int,
    users: np.ndarray,
    topk_by_model: dict[str, np.ndarray],
    prepared: dict,
    cohorts: pd.DataFrame,
    item_traits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = prepared["graph"]
    n_items = prepared["data"]["n_items"]
    cohort = cohorts.set_index("user_idx")
    buyer_q = cohort.q_clv.to_numpy(float)[graph.edge_users]
    buyer_factor = graph.user_factors["clv"][graph.edge_users]
    high_buyer = (buyer_q >= 0.8).astype(float)
    degree = np.bincount(graph.edge_items, minlength=n_items)
    q_sum = np.bincount(graph.edge_items, weights=buyer_q, minlength=n_items)
    factor_sum = np.bincount(graph.edge_items, weights=buyer_factor, minlength=n_items)
    high_sum = np.bincount(graph.edge_items, weights=high_buyer, minlength=n_items)
    coefficient_shift = np.bincount(
        graph.edge_items,
        weights=np.abs(graph.item_user_coefficients["clv"] - graph.base_coefficients),
        minlength=n_items,
    )
    promotions = np.zeros(n_items, dtype=int)
    demotions = np.zeros(n_items, dtype=int)
    promoted_hits = np.zeros(n_items, dtype=int)
    for row, user in enumerate(users):
        m1_top10 = set(map(int, topk_by_model[M1_ID][row, :10]))
        m3_top10 = set(map(int, topk_by_model[M3_ID][row, :10]))
        truth = set(map(int, prepared["cache"].gt[int(user)]))
        for item in m3_top10 - m1_top10:
            promotions[item] += 1
            promoted_hits[item] += int(item in truth)
        for item in m1_top10 - m3_top10:
            demotions[item] += 1
    frame = pd.DataFrame(
        {
            "seed": seed,
            "item_idx": np.arange(n_items, dtype=int),
            "train_neighbor_count": degree,
            "mean_train_neighbor_q_clv": np.divide(q_sum, degree, out=np.full(n_items, np.nan), where=degree > 0),
            "mean_train_neighbor_clv_factor": np.divide(factor_sum, degree, out=np.full(n_items, np.nan), where=degree > 0),
            "high_clv_neighbor_share": np.divide(high_sum, degree, out=np.full(n_items, np.nan), where=degree > 0),
            "operator_l1_shift": coefficient_shift,
            "top10_promotion_count": promotions,
            "top10_demotion_count": demotions,
            "promoted_hit_count": promoted_hits,
            "promoted_hit_rate": np.divide(promoted_hits, promotions, out=np.full(n_items, np.nan), where=promotions > 0),
        }
    ).merge(item_traits, on="item_idx", how="left")
    correlations = []
    candidates = (
        "mean_train_neighbor_q_clv",
        "high_clv_neighbor_share",
        "operator_l1_shift",
        "train_neighbor_count",
        "price_percentile",
        "repeat_purchase_share",
    )
    for target in ("top10_promotion_count", "top10_demotion_count", "promoted_hit_count"):
        for feature in candidates:
            valid = frame[[target, feature]].dropna()
            correlation = spearmanr(valid[target], valid[feature]).correlation if len(valid) >= 3 else np.nan
            correlations.append(
                {
                    "seed": seed,
                    "target": target,
                    "feature": feature,
                    "n_items": len(valid),
                    "spearman": float(correlation) if np.isfinite(correlation) else np.nan,
                }
            )
    return frame, pd.DataFrame(correlations)


def _representative_cases(
    per_user: pd.DataFrame,
    truth: pd.DataFrame,
    recs: pd.DataFrame,
    cfg: M3BehaviorDiagnosticConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    m1_users = per_user[per_user.model_id.eq(M1_ID)].set_index(["seed", "user_idx"])
    m3_users = per_user[per_user.model_id.eq(M3_ID)].set_index(["seed", "user_idx"])
    common_index = m1_users.index.intersection(m3_users.index)
    delta = m3_users.loc[common_index, ["recall@10", "ndcg@10", "recall@50"]].copy()
    delta.columns = [f"m3_{column}" for column in delta.columns]
    for metric in ("recall@10", "ndcg@10", "recall@50"):
        delta[f"m1_{metric}"] = m1_users.loc[common_index, metric]
        delta[f"delta_{metric}"] = delta[f"m3_{metric}"] - delta[f"m1_{metric}"]
    delta = delta.join(m1_users.loc[common_index, ["user_id", "nv_quadrant", "nv_quadrant_label", "clv_quintile", "n_hat", "v_hat", "clv_proxy"]])
    delta = delta.reset_index()
    selected = []

    def take(frame: pd.DataFrame, rule: str, ascending: bool, n: int = 1):
        ordered = frame.sort_values(["delta_ndcg@10", "user_idx"], ascending=[ascending, True])
        for _, row in ordered.head(n).iterrows():
            selected.append({"selection_rule": rule, **row.to_dict()})

    take(delta, "largest_ndcg10_gain", False)
    take(delta, "largest_ndcg10_loss", True)
    take(delta[delta["delta_recall@10"] > 0], "m3_only_or_more_top10_truth", False)
    take(delta[delta["delta_recall@10"] < 0], "m1_only_or_more_top10_truth", True)
    for quadrant, group in delta.groupby("nv_quadrant", sort=False):
        take(group, f"{quadrant}_largest_gain", False, cfg.representative_per_quadrant)
        take(group, f"{quadrant}_largest_loss", True, cfg.representative_per_quadrant)
    summary = pd.DataFrame(selected).drop_duplicates(["selection_rule", "seed", "user_idx"])
    if summary.empty:
        return summary, pd.DataFrame()
    keys = summary[["seed", "user_idx", "selection_rule"]]
    truth_detail = keys.merge(truth, on=["seed", "user_idx"], how="left").assign(detail_role="test_truth")
    rec_detail = keys.merge(recs[recs["rank"] <= 10], on=["seed", "user_idx"], how="left").assign(
        detail_role=lambda frame: np.where(frame.model_id.eq(M1_ID), "m1_top10", "m3_top10")
    )
    detail_columns = sorted(set(truth_detail.columns).union(rec_detail.columns))
    detail = pd.concat(
        [truth_detail.reindex(columns=detail_columns), rec_detail.reindex(columns=detail_columns)],
        ignore_index=True,
    )
    return summary, detail


def _quality_checks(
    seed: int,
    users: np.ndarray,
    topk_by_model: dict[str, np.ndarray],
    per_user: pd.DataFrame,
    truth: pd.DataFrame,
    prepared: dict,
    results: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    n_items = prepared["data"]["n_items"]
    train_keys = prepared["data"]["pos_key"]
    for model_id, topk in topk_by_model.items():
        top_keys = users[:, None].astype(np.int64) * n_items + topk.astype(np.int64)
        positions = np.clip(np.searchsorted(train_keys, top_keys), 0, len(train_keys) - 1)
        overlap = int((train_keys[positions] == top_keys).sum())
        duplicate_rows = int(sum(len(np.unique(row)) != len(row) for row in topk))
        rows.extend(
            [
                {"seed": seed, "model_id": model_id, "check": "top50_excludes_train_pairs", "value": overlap, "passed": overlap == 0},
                {"seed": seed, "model_id": model_id, "check": "top50_has_no_duplicates", "value": duplicate_rows, "passed": duplicate_rows == 0},
            ]
        )
        model_users = per_user[(per_user.seed == seed) & per_user.model_id.eq(model_id)]
        for metric in METRICS:
            observed = float(model_users[metric].mean())
            expected = float(results[model_id]["metrics"][metric])
            error = abs(observed - expected)
            rows.append(
                {"seed": seed, "model_id": model_id, "check": f"recomputed_{metric}", "value": error, "passed": error < 1e-7}
            )
    truth_keys = truth.user_idx.to_numpy(np.int64) * n_items + truth.item_idx.to_numpy(np.int64)
    positions = np.clip(np.searchsorted(train_keys, truth_keys), 0, len(train_keys) - 1)
    truth_overlap = int((train_keys[positions] == truth_keys).sum())
    rows.append({"seed": seed, "model_id": "all", "check": "test_truth_excludes_train_pairs", "value": truth_overlap, "passed": truth_overlap == 0})
    rows.append({"seed": seed, "model_id": "all", "check": "evaluated_user_count", "value": len(users), "passed": len(users) == len(np.unique(users))})
    return pd.DataFrame(rows)


def _persist(
    cfg: M3BehaviorDiagnosticConfig,
    source_json: Path,
    source_hash: str,
    source_payload: dict,
    frames: dict[str, pd.DataFrame],
    source_paths: list[dict],
) -> dict[str, str]:
    diagnostic_hash = hashlib.sha256(
        json.dumps({"version": CODE_VERSION, "source": source_hash, "config": asdict(cfg)}, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]
    root = Path(cfg.out_dir) / "behavior_diagnostic" / diagnostic_hash
    paths = {name: root / f"{name}.csv" for name in frames}
    paths["json"] = root / "diagnostic.json"
    for name, frame in frames.items():
        common._atomic_csv(paths[name], frame)
    quality = frames["quality_checks"]
    payload = {
        "code_version": CODE_VERSION,
        "analysis_type": "descriptive post-hoc analysis; not confirmatory model selection",
        "source_run_json": str(source_json),
        "source_run_hash": source_hash,
        "source_revision": source_payload.get("source_revision"),
        "input_manifest": source_payload.get("input_manifest"),
        "config": asdict(cfg),
        "source_checkpoints": source_paths,
        "quality_passed": bool(len(quality) and quality.passed.all()),
        "quality_failed": quality.loc[~quality.passed].to_dict("records"),
        "result_paths": {name: str(path) for name, path in paths.items()},
        "metric_contract": {
            "population": "users with at least one fixed-test new-item truth",
            "baseline": M1_ID,
            "changed_model": M3_ID,
            "ranking_cutoffs": [10, 20, 50],
            "segments": "train-history N/V quadrants and CLV quintiles",
            "price_weighted_hit": "not actual incremental revenue",
        },
        "interpretation_limits": [
            "One-seed results have no dispersion or significance estimate.",
            "User segments use train-history variables; test outcomes do not define cohorts.",
            "This test diagnostic may generate hypotheses but cannot tune or confirm a changed model on the same interval.",
        ],
    }
    common._atomic_json(paths["json"], payload)
    return {name: str(path) for name, path in paths.items()}


def run_m3_behavior_diagnostic(
    cfg: M3BehaviorDiagnosticConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_m3_behavior_diagnostic())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    source_json, source_payload, source_hash = _find_run_json(cfg)
    prepared, run_cfg = _prepare_source(cfg, source_payload)
    cohorts = _user_cohorts(prepared)
    item_traits = common._raw_item_traits(prepared["data"]["train"], prepared["data"]["n_items"])
    all_per_user, all_recs, all_truth = [], [], []
    all_mechanism, all_correlations, all_quality = [], [], []
    source_paths = []

    for seed in cfg.seeds:
        loaded, topk_by_model = {}, {}
        for model_id in MODELS:
            print(f"\n===== 사후 진단 seed {seed} | {model_id} =====")
            model, result, paths = _load_model(prepared, run_cfg, model_id, seed)
            users, topk, _, _ = common._masked_topk(model, prepared, max(cfg.ks), cfg.score_batch_size)
            frame = _per_user_metrics(users, topk, prepared, cohorts, model_id, seed)
            recs = _recommendation_rows(seed, model_id, users, topk, prepared, cohorts, item_traits)
            all_per_user.append(frame)
            all_recs.append(recs)
            topk_by_model[model_id] = topk
            loaded[model_id] = result
            source_paths.append(
                {"seed": seed, "model_id": model_id, "result": str(paths["result"]), "checkpoint": str(paths["checkpoint"]), "checkpoint_sha256": file_sha256(paths["checkpoint"])}
            )
        if not np.array_equal(users, prepared["cache"].users):
            raise RuntimeError(f"seed {seed}: 평가 사용자 순서가 cache와 다릅니다")
        truth = _truth_rows(seed, users, topk_by_model, prepared, cohorts, item_traits)
        all_truth.append(truth)
        mechanism, correlations = _item_mechanism(seed, users, topk_by_model, prepared, cohorts, item_traits)
        all_mechanism.append(mechanism)
        all_correlations.append(correlations)
        seed_per_user = pd.concat([frame for frame in all_per_user if int(frame.seed.iloc[0]) == seed], ignore_index=True)
        all_quality.append(_quality_checks(seed, users, topk_by_model, seed_per_user, truth, prepared, loaded))

    per_user = pd.concat(all_per_user, ignore_index=True)
    recs = pd.concat(all_recs, ignore_index=True)
    truth = pd.concat(all_truth, ignore_index=True)
    segment_metrics = _segment_metric_summary(per_user)
    item_profiles, category_profiles = _segment_item_profiles(recs, truth, cohorts)
    mechanism = pd.concat(all_mechanism, ignore_index=True)
    correlations = pd.concat(all_correlations, ignore_index=True)
    cases, case_details = _representative_cases(per_user, truth, recs, cfg)
    quality = pd.concat(all_quality, ignore_index=True)
    promoted_examples = (
        mechanism.sort_values(["seed", "top10_promotion_count", "promoted_hit_count"], ascending=[True, False, False])
        .groupby("seed", sort=False)
        .head(cfg.top_item_examples)
        .reset_index(drop=True)
    )
    frames = {
        "user_metrics": per_user,
        "recommendation_top50": recs,
        "test_truth_rank_transition": truth,
        "segment_metric_summary": segment_metrics,
        "segment_item_profile": item_profiles,
        "segment_category_profile": category_profiles,
        "item_clv_mechanism": mechanism,
        "item_mechanism_correlations": correlations,
        "promoted_item_examples": promoted_examples,
        "representative_user_cases": cases,
        "representative_case_details": case_details,
        "quality_checks": quality,
    }
    paths = _persist(cfg, source_json, source_hash, source_payload, frames, source_paths)
    segment_metrics.attrs["result_paths"] = paths
    segment_metrics.attrs["quality_passed"] = bool(quality.passed.all())
    print("\n===== 사용자 성향별 M3-M1 차이 =====")
    print(segment_metrics[segment_metrics.metric.isin(("recall@10", "ndcg@10", "recall@50"))].to_string(index=False))
    print("\n===== QA =====")
    print(quality.to_string(index=False))
    print("\n결과 파일:", paths)
    if not quality.passed.all():
        raise RuntimeError("QA 불일치가 있어 결과 해석을 중단합니다. quality_checks.csv를 확인하세요")
    return segment_metrics


if __name__ == "__main__":
    run_m3_behavior_diagnostic()
