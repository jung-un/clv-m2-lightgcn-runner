"""How much purchase value sits below the top ten, and can a CLV gate reach it?

Every trained model left the top-10 price- and purchase-amount weighted hit
value flat or slightly lower than M1, while the same value rose at rank 50.
Ten slots is the binding constraint: putting an expensive hit into the top ten
costs a cheap hit its place.

This diagnostic re-scores the finished development checkpoints, so it trains
nothing.  For every evaluation customer it measures

* ``captured_value_10``   - value of the customer's purchases already inside the top ten,
* ``band_value_11_50``    - value of purchases the model ranks 11th to 50th (what a
  re-ranking stage could still reach),
* ``beyond_value_50``     - value the model never surfaces at all,
* ``oracle_value_10``     - the ceiling: the ten highest-value purchases within the
  top fifty, which knows the answers and is therefore an upper bound, not a method.

It then applies one feasible, train-only rule to the same candidates: the top
fifty are re-ordered by ``(1 - w) * position weight + w * expected item amount``
with ``w = alpha * q_C(u)``, so only high-CLV customers are pushed toward value
and the strength is the customer's own CLV percentile.  Alpha is swept to draw
the accuracy-value trade-off; nothing is selected on these numbers.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-value-rerank-diagnostic-dev5-v1"
SEGMENTS = ("전체", "고CLV", "중CLV", "저CLV")


@dataclass(frozen=True)
class ValueRerankConfig:
    models: tuple[str, ...] = (
        recheck.M1_MODEL_ID,
        recheck.M2_MODEL_ID,
        recheck.M4_MODEL_ID,
        recheck.M5_MODEL_ID,
    )
    seeds: tuple[int, ...] = recheck.SEEDS
    candidate_depth: int = 50
    top_k: int = 10
    alphas: tuple[float, ...] = (0.0, 0.1, 0.2, 0.4)
    eval_batch: int = 256
    out_dir: str = ""


def configure_rerank_diagnostic(**overrides) -> ValueRerankConfig:
    defaults = {
        "out_dir": f"{v3.default_out_dir('dunnhumby')}_clv_value_rerank_diagnostic_v1"
    }
    return validate_config(ValueRerankConfig(**(defaults | overrides)))


def validate_config(cfg: ValueRerankConfig) -> ValueRerankConfig:
    if cfg.candidate_depth != 50 or cfg.top_k != 10:
        raise ValueError("진단은 상위 50 후보에서 상위 10을 고르는 설정이어야 합니다")
    if cfg.alphas[0] != 0.0:
        raise ValueError("alpha=0(재정렬 없음)이 대조 기준으로 포함돼야 합니다")
    if not set(cfg.models).issubset(set(recheck.MODEL_IDS)):
        raise ValueError("구성요소 재확인에서 학습한 모형만 다시 채점할 수 있습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: ValueRerankConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "trains_anything": False,
        "source_checkpoints": (
            f"{recheck.CODE_VERSION} development arms, "
            f"{list(cfg.models)} x seeds {list(cfg.seeds)}"
        ),
        "split": "historical_development_days_684_690",
        "value_definition": (
            "per customer-item purchase amount in the evaluation window, the same "
            "quantity the price and purchase amount weighted hit value sums"
        ),
        "headroom": (
            "captured_value_10 / band_value_11_50 / beyond_value_50 and the "
            "oracle top-10 value within the top 50 (upper bound, uses answers)"
        ),
        "rerank_rule": (
            "(1 - w) * position weight + w * expected item amount (train-only "
            "item mean, normalized within the candidate set), w = alpha * q_C(u)"
        ),
        "alphas": list(cfg.alphas),
        "limits": (
            "five development seeds on a repeatedly exposed split; the sweep is a "
            "screen, no alpha is selected and no significance is claimed"
        ),
    }


def _model_top_candidates(model, prepared: dict, cfg: ValueRerankConfig):
    """Top-``candidate_depth`` items per evaluation user, with their purchase value."""

    cache, data = prepared["cache"], prepared["data"]
    csr_ptr, csr_items = data["csr_ptr"], data["csr_items"]
    n_items = data["n_items"]
    user_embeddings, item_embeddings, *_ = model.embeddings()
    candidates, values = [], []
    for start in range(0, len(cache.users), cfg.eval_batch):
        batch_users = cache.users[start : start + cfg.eval_batch]
        rows = torch.as_tensor(batch_users, dtype=torch.long, device=v3.DEVICE)
        scores = user_embeddings[rows] @ item_embeddings.T
        for offset, user in enumerate(batch_users):
            lo, hi = csr_ptr[user], csr_ptr[user + 1]
            if hi > lo:
                scores[offset, csr_items[lo:hi]] = -1e9
        top = scores.topk(cfg.candidate_depth, dim=1).indices.cpu().numpy()
        keys = batch_users[:, None].astype(np.int64) * n_items + top.astype(np.int64)
        index = np.clip(
            np.searchsorted(cache.pos_key, keys), 0, len(cache.pos_key) - 1
        )
        hit = cache.pos_key[index] == keys
        candidates.append(top)
        values.append(np.where(hit, cache.pos_rev[index], 0.0))
    return np.concatenate(candidates), np.concatenate(values)


def headroom_table(values: np.ndarray, cache, cfg: ValueRerankConfig) -> pd.DataFrame:
    """Per-user value already captured, still reachable, and out of reach."""

    top = values[:, : cfg.top_k]
    captured = top.sum(axis=1)
    band = values[:, cfg.top_k :].sum(axis=1)
    total = np.array([cache.rev[user].sum() for user in cache.users], dtype=np.float64)
    ordered = -np.sort(-values, axis=1)[:, : cfg.top_k]
    return pd.DataFrame(
        {
            "user": cache.users,
            "segment": cache.seg,
            "captured_value_10": captured,
            "band_value_11_50": band,
            "beyond_value_50": total - captured - band,
            "total_value": total,
            "oracle_value_10": ordered.sum(axis=1),
            "hits_10": (top > 0).sum(axis=1),
            "hits_50": (values > 0).sum(axis=1),
            "positives": cache.P_arr[cache.users],
        }
    )


def rerank_table(
    candidates: np.ndarray,
    values: np.ndarray,
    prepared: dict,
    cfg: ValueRerankConfig,
) -> pd.DataFrame:
    """Sweep the CLV-gated value push over the same candidate sets."""

    cache = prepared["cache"]
    amount = np.asarray(prepared["item_expected_amount"], dtype=np.float64)[candidates]
    amount = amount / np.maximum(amount.max(axis=1, keepdims=True), 1e-12)
    relevance = 1.0 / np.log2(np.arange(2, cfg.candidate_depth + 2))
    relevance = np.broadcast_to(relevance / relevance[0], candidates.shape)
    q_c = np.asarray(prepared["q_c"], dtype=np.float64)[cache.users]

    rows = []
    for alpha in cfg.alphas:
        weight = (alpha * q_c)[:, None]
        utility = (1.0 - weight) * relevance + weight * amount
        chosen = np.argsort(-utility, axis=1, kind="stable")[:, : cfg.top_k]
        picked = np.take_along_axis(values, chosen, axis=1)
        changed = (
            np.take_along_axis(candidates, chosen, axis=1) != candidates[:, : cfg.top_k]
        ).any(axis=1)
        rows.append(
            pd.DataFrame(
                {
                    "alpha": alpha,
                    "user": cache.users,
                    "segment": cache.seg,
                    "value_10": picked.sum(axis=1),
                    "hits_10": (picked > 0).sum(axis=1),
                    "positives": cache.P_arr[cache.users],
                    "list_changed": changed.astype(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _segment_means(frame: pd.DataFrame, columns: list[str], keys: list[str]) -> pd.DataFrame:
    rows = []
    for segment in SEGMENTS:
        part = frame if segment == "전체" else frame[frame.segment.eq(segment)]
        if part.empty:
            continue
        for _, group in part.groupby(keys, sort=False):
            row = {key: group.iloc[0][key] for key in keys}
            row["segment"] = segment
            row["n_users"] = len(group)
            row.update({column: float(group[column].mean()) for column in columns})
            if {"hits_10", "positives"}.issubset(group.columns):
                row["recall_10"] = float(
                    (group.hits_10 / group.positives.clip(lower=1)).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _load_model(prepared: dict, recheck_cfg, spec: dict, seed: int):
    paths = recheck._arm_paths(prepared, spec["model_id"], seed)
    if not paths["checkpoint"].exists():
        raise FileNotFoundError(f"체크포인트가 없습니다: {paths['checkpoint']}")
    payload = torch.load(paths["checkpoint"], map_location=v3.DEVICE, weights_only=False)
    model = recheck._build_model(prepared, recheck_cfg, spec, seed)
    model.load_state_dict(payload["state"])
    model.eval()
    return model


def run_value_rerank_diagnostic(cfg: ValueRerankConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_rerank_diagnostic())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    recheck_cfg = recheck.configure_component_recheck()
    prepared = recheck._prepare(recheck_cfg)
    train = prepared["data"]["train"]
    expected = train.groupby("i_idx")["v"].mean()
    item_expected_amount = np.zeros(prepared["data"]["n_items"], dtype=np.float64)
    item_expected_amount[expected.index.values] = expected.to_numpy()
    prepared["item_expected_amount"] = item_expected_amount

    specs = {spec["model_id"]: spec for spec in recheck.arm_specifications()}
    headroom, rerank = [], []
    with torch.no_grad():
        for model_id in cfg.models:
            for seed in cfg.seeds:
                model = _load_model(prepared, recheck_cfg, specs[model_id], seed)
                candidates, values = _model_top_candidates(model, prepared, cfg)
                head = headroom_table(values, prepared["cache"], cfg)
                head.insert(0, "seed", seed)
                head.insert(0, "model_id", model_id)
                headroom.append(head)
                sweep = rerank_table(candidates, values, prepared, cfg)
                sweep.insert(0, "seed", seed)
                sweep.insert(0, "model_id", model_id)
                rerank.append(sweep)
                print(f"  [{model_id} s{seed}] 후보 {cfg.candidate_depth}개 재채점 완료")
    headroom = pd.concat(headroom, ignore_index=True)
    rerank = pd.concat(rerank, ignore_index=True)

    head_columns = [
        "captured_value_10",
        "band_value_11_50",
        "beyond_value_50",
        "total_value",
        "oracle_value_10",
        "hits_10",
        "hits_50",
    ]
    head_per_seed = _segment_means(headroom, head_columns, ["model_id", "seed"])
    rerank_per_seed = _segment_means(
        rerank, ["value_10", "hits_10", "list_changed"], ["model_id", "seed", "alpha"]
    )
    head_summary = _across_seeds(head_per_seed, ["model_id", "segment"], head_columns + ["recall_10"])
    rerank_summary = _across_seeds(
        rerank_per_seed, ["model_id", "alpha", "segment"], ["value_10", "recall_10", "list_changed"]
    )
    reading = _reading(head_summary, rerank_summary, cfg)

    out = Path(cfg.out_dir)
    stem = f"clv_value_rerank_diagnostic_{recheck._config_hash(recheck_cfg, prepared['input_hash'])}"
    paths = {
        "headroom_seed_csv": out / f"{stem}_headroom_seed.csv",
        "headroom_csv": out / f"{stem}_headroom.csv",
        "rerank_seed_csv": out / f"{stem}_rerank_seed.csv",
        "rerank_csv": out / f"{stem}_rerank.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["headroom_seed_csv"], head_per_seed)
    test10._atomic_csv(paths["headroom_csv"], head_summary)
    test10._atomic_csv(paths["rerank_seed_csv"], rerank_per_seed)
    test10._atomic_csv(paths["rerank_csv"], rerank_summary)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "source_revision": prepared["revision"],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "headroom": head_summary.to_dict("records"),
            "rerank": rerank_summary.to_dict("records"),
            "reading": reading,
            "result_paths": {name: str(path) for name, path in paths.items()},
        },
    )
    print("\n상위 10 밖에 남은 구매금액 (5시드 평균):")
    print(head_summary.to_string(index=False))
    print("\nCLV 게이트 가치 재정렬 스윕:")
    print(rerank_summary.to_string(index=False))
    print("\n판독:", json.dumps(reading, ensure_ascii=False, indent=2))
    head_summary.attrs.update(
        rerank=rerank_summary, reading=reading,
        result_paths={name: str(path) for name, path in paths.items()},
    )
    return head_summary


def _across_seeds(frame: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
        for column in columns:
            row.update(
                {f"{column}_{name}": value
                 for name, value in test10._mean_ci(group[column].to_numpy()).items()
                 if name in {"mean", "lo", "hi"}}
            )
        row["n_seeds"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def _reading(head: pd.DataFrame, rerank: pd.DataFrame, cfg: ValueRerankConfig) -> dict:
    def head_value(model_id: str, segment: str, column: str) -> float:
        row = head[head.model_id.eq(model_id) & head.segment.eq(segment)]
        return float(row.iloc[0][f"{column}_mean"])

    def swept(model_id: str, segment: str, alpha: float, column: str) -> float:
        row = rerank[
            rerank.model_id.eq(model_id)
            & rerank.segment.eq(segment)
            & rerank.alpha.eq(alpha)
        ]
        return float(row.iloc[0][f"{column}_mean"])

    model_id = cfg.models[-1]
    best_alpha = max(
        cfg.alphas, key=lambda alpha: swept(model_id, "고CLV", alpha, "value_10")
    )
    captured = head_value(model_id, "고CLV", "captured_value_10")
    return {
        "reference_model": model_id,
        "high_clv_reachable_share": head_value(model_id, "고CLV", "band_value_11_50")
        / max(head_value(model_id, "고CLV", "total_value"), 1e-12),
        "high_clv_ceiling_gain_share": (
            head_value(model_id, "고CLV", "oracle_value_10") / max(captured, 1e-12) - 1.0
        ),
        "best_swept_alpha_high_clv": best_alpha,
        "best_swept_value_gain_share": (
            swept(model_id, "고CLV", best_alpha, "value_10") / max(captured, 1e-12) - 1.0
        ),
        "recall_cost_at_best_alpha": (
            swept(model_id, "고CLV", best_alpha, "recall_10")
            / max(swept(model_id, "고CLV", 0.0, "recall_10"), 1e-12) - 1.0
        ),
        "alpha_selected": False,
        "significance_claimed": False,
    }


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_rerank_diagnostic()), ensure_ascii=False, indent=2))
