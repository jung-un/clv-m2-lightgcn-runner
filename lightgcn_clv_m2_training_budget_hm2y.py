"""H&M seed-43 learning-budget diagnostic for the current personal-history M2."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_history_item_fit_model import HistoryItemFitLightGCN, build_personal_history_weights
from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m2-training-budget-hm2y-seed43-v1"
M1_MODEL_ID = "m1_bpr_k1_hm2y_training_budget_s43"
M2_MODEL_ID = "m2_nv_history_fit_bpr_k1_hm2y_training_budget_s43"


@dataclass(frozen=True)
class HMM2TrainingBudgetConfig:
    dataset: str = "hm"
    seed: int = 43
    window_days: None = None
    input_days: int = 365
    epochs: int = 300
    evaluation_epochs: tuple[int, ...] = (100, 200, 300)
    id_dim: int = 64
    history_axis_dim: int = 4
    history_rho: float = 0.05
    n_layers: int = 2
    batch_size: int = common.DEFAULT_BATCH_SIZE
    lr: float = 5e-4
    pref_reg: float = 1e-3
    negative_count: int = 1
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""


def configure_hm2y_training_budget(**overrides) -> HMM2TrainingBudgetConfig:
    defaults = {
        "out_dir": f"{v3.default_out_dir('hm')}_clv_m2_training_budget_seed43_v1"
    }
    return validate_config(HMM2TrainingBudgetConfig(**(defaults | overrides)))


def validate_config(cfg: HMM2TrainingBudgetConfig) -> HMM2TrainingBudgetConfig:
    required = {
        "dataset": "hm", "seed": 43, "window_days": None, "input_days": 365,
        "epochs": 300, "evaluation_epochs": (100, 200, 300), "id_dim": 64,
        "history_axis_dim": 4, "history_rho": 0.05, "n_layers": 2,
        "pref_reg": 1e-3, "negative_count": 1, "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"H&M M2 학습예산 진단은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size not in common.BATCH_CANDIDATES:
        raise ValueError("batch_size는 131072/65536/32768 중 하나여야 합니다")
    if cfg.lr <= 0 or not cfg.out_dir:
        raise ValueError("H&M M2 학습 설정이 잘못됐습니다")
    return cfg


def preflight_summary(cfg: HMM2TrainingBudgetConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "research_question": (
            "Does the current personal-history q_N/q_V M2 need more than 100 "
            "epochs on H&M, or does its paired deficit persist?"
        ),
        "dataset": "hm", "seed": cfg.seed,
        "split": "hm2y_validation_2020-09-02_08",
        "models": [M1_MODEL_ID, M2_MODEL_ID],
        "epochs": cfg.epochs,
        "evaluated_at_epochs": list(cfg.evaluation_epochs),
        "m2": {
            "scope": "CLV-component representation M2; q_C is not used",
            "inputs": "q_N and q_V scale leave-one-out personal-history N/V blocks",
            "joint_training": True, "rho": cfg.history_rho,
            "axis_dim": cfg.history_axis_dim, "posthoc_reranking": False,
        },
        "fixed": {
            "new_item_task": True, "train_pairs_excluded": True,
            "min_item_interactions": 1, "graph": "binary",
            "negative_sampling": "one uniform unseen item",
            "loss": "plain BPR; no row weighting and no additional term",
            "id_dim": cfg.id_dim, "layers": cfg.n_layers,
            "pref_reg": cfg.pref_reg, "test_constructed": False,
            "holdout_constructed": False,
        },
        "reading_rule": (
            "Compare paired M2-M1 gaps at fixed epochs 100, 200 and 300. "
            "Peak-to-peak values are exploratory only and cannot select an epoch."
        ),
        "limits": (
            "one new development seed; no significance, stability, generalization, "
            "CLV attribution or epoch selection claim"
        ),
        "checkpointing": "every completed epoch; optimizer and RNG included",
        "out_dir": cfg.out_dir,
    }


def _prepare(cfg: HMM2TrainingBudgetConfig) -> dict:
    prepared = common.prepare_hm2y(cfg, code_version=CODE_VERSION)
    data = prepared["data"]
    prepared["history"] = build_personal_history_weights(
        data["train"], n_users=data["n_users"], n_items=data["n_items"]
    )
    return prepared


def arm_specifications() -> list[dict]:
    return [
        {"model_id": M1_MODEL_ID, "role": "matched_m1", "kind": "m1"},
        {"model_id": M2_MODEL_ID, "role": "personal_history_m2", "kind": "m2"},
    ]


def _item_price_percentile(prepared: dict) -> np.ndarray:
    """Restore the first centred H&M economic feature to its [0,1] percentile."""

    centred = np.asarray(prepared["item_economic"], dtype=np.float32)[:, 0]
    valid = np.asarray(prepared["item_economic_valid"], dtype=bool)
    if centred.shape != valid.shape:
        raise ValueError("H&M 상품 가격 위치와 valid mask shape이 다릅니다")
    percentile = np.clip((centred + 1.0) / 2.0, 0.0, 1.0)
    percentile[~valid] = 0.0
    if not np.isfinite(percentile).all():
        raise ValueError("H&M 상품 가격 백분위에 비유한 값이 있습니다")
    return percentile.astype(np.float32)


def _build_model(prepared: dict, cfg: HMM2TrainingBudgetConfig, spec: dict):
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], dtype=bool)
    v3.set_seed(cfg.seed)
    if spec["kind"] == "m2":
        model = HistoryItemFitLightGCN(
            n_users=data["n_users"], n_items=data["n_items"],
            history=prepared["history"],
            q_n=np.where(valid, prepared["q_n"], 0.0).astype(np.float32),
            q_v=np.where(valid, prepared["q_v"], 0.0).astype(np.float32),
            activity_valid=valid, value_valid=valid, adj=data["adj"],
            id_dim=cfg.id_dim, axis_dim=cfg.history_axis_dim,
            n_layers=cfg.n_layers, rho=cfg.history_rho, pref_reg=cfg.pref_reg,
        )
    else:
        model = M5NConditionedValueBasisLightGCN(
            n_users=data["n_users"], n_items=data["n_items"],
            user_q_n=prepared["q_n"], user_q_v=prepared["q_v"],
            user_q_c=prepared["q_c"], user_clv_valid=valid,
            item_price_percentile=_item_price_percentile(prepared),
            item_price_valid=prepared["item_economic_valid"], adj=data["adj"],
            id_dim=cfg.id_dim, rho=0.0, n_layers=cfg.n_layers,
            pref_reg=cfg.pref_reg, economic_propagation=False,
        )
    return model.to(v3.DEVICE)


def _arm_hash(prepared: dict, cfg: HMM2TrainingBudgetConfig, spec: dict) -> str:
    payload = {"run": prepared["config_hash"], "model_id": spec["model_id"], "seed": cfg.seed}
    return hashlib.sha256(common.canonical(payload).encode()).hexdigest()[:12]


def _arm_path(prepared: dict, spec: dict) -> Path:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    return root / f"{spec['model_id']}.json"


@torch.no_grad()
def _score_share(model, prepared: dict) -> dict:
    if not isinstance(model, HistoryItemFitLightGCN):
        return {"id_score_mean_abs": float("nan"), "clv_score_mean_abs": 0.0,
                "clv_score_share": 0.0}
    cache, data = prepared["cache"], prepared["data"]
    rng = np.random.default_rng(0)
    sample = cache.users[rng.choice(len(cache.users), min(256, len(cache.users)), replace=False)]
    users, items, *_ = model.embeddings()
    rows = torch.as_tensor(sample, dtype=torch.long, device=v3.DEVICE)
    scores = users[rows] @ items.T
    for offset, user in enumerate(sample):
        lo, hi = data["csr_ptr"][user], data["csr_ptr"][user + 1]
        if hi > lo:
            scores[offset, data["csr_items"][lo:hi]] = -1e9
    top = scores.topk(10, dim=1).indices
    picked_users = rows[:, None].expand_as(top).reshape(-1)
    picked_items = top.reshape(-1)
    cut = model.id_dim
    id_part = (users[picked_users, :cut] * items[picked_items, :cut]).sum(1)
    clv_part = (users[picked_users, cut:] * items[picked_items, cut:]).sum(1)
    denominator = id_part.abs().mean() + clv_part.abs().mean() + 1e-12
    return {"id_score_mean_abs": float(id_part.abs().mean()),
            "clv_score_mean_abs": float(clv_part.abs().mean()),
            "clv_score_share": float(clv_part.abs().mean() / denominator)}


def _train_curve(model, prepared, cfg, spec, store) -> list[dict]:
    data = prepared["data"]
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    restored = common.restore_compact(store, model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["epoch"]) + 1
    history = [] if restored is None else list(restored.get("history", []))
    previous_wall = 0.0 if restored is None else float(restored.get("wall_clock_sec", 0.0))
    if restored is not None:
        print(f"  [{spec['model_id']}] epoch {start_epoch - 1}에서 자동 재개")
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    started = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        permutation = rng.permutation(len(tr_u))
        loss_sum = correct_sum = 0.0
        epoch_started = time.time()
        for batch in range(n_batches):
            index = permutation[batch * cfg.batch_size:(batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = v3.sample_negatives(
                users_np, positives_np, data["n_items"], positive_keys, rng,
                "uniform", data["item_cat"], data["cat_items"],
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=v3.DEVICE)
            negatives = torch.as_tensor(negatives_np[:, None], dtype=torch.long, device=v3.DEVICE)
            loss, _, correct = recheck._batch_loss(model, users, positives, negatives, None)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()); correct_sum += correct
            store.heartbeat(epoch=epoch, max_epoch=cfg.epochs, batch=batch + 1,
                            batches=n_batches, loss=loss_sum / (batch + 1), selection="none")
        record = {"epoch": epoch, "loss": loss_sum / n_batches,
                  "p_correct": correct_sum / n_batches,
                  "epoch_sec": time.time() - epoch_started}
        if epoch in cfg.evaluation_epochs:
            model.eval()
            record["metrics"] = common.evaluate(model, prepared)
            record["score_split"] = _score_share(model, prepared)
            record["gradient_diagnostics"] = model.training_gradient_diagnostics()
            print(f"  [{spec['model_id']}] ep {epoch:3d} | loss {record['loss']:.4f} | "
                  f"recall@10 {record['metrics']['recall@10']:.6f} | "
                  f"ndcg@10 {record['metrics']['ndcg@10']:.6f}")
        else:
            print(f"  [{spec['model_id']}] ep {epoch:3d}/{cfg.epochs} | "
                  f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
                  f"{record['epoch_sec']:.0f}s")
        history.append(record)
        common.save_compact(store, model, optimizer, rng, epoch=epoch,
                            max_epoch=cfg.epochs, history=history,
                            wall_clock_sec=previous_wall + time.time() - started,
                            selection="none")
    return history


def _run_arm(prepared, cfg, spec) -> dict:
    path = _arm_path(prepared, spec)
    if path.exists():
        print(f"  [cached] {spec['model_id']} 완료 결과 재사용")
        return json.loads(path.read_text(encoding="utf-8"))
    model = _build_model(prepared, cfg, spec)
    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(stage="hm2y_m2_training_budget_dev", model_id=spec["model_id"],
                    seed=cfg.seed, config_hash=_arm_hash(prepared, cfg, spec),
                    source_revision=prepared["revision"], input_hash=prepared["input_hash"]),
    )
    history = _train_curve(model, prepared, cfg, spec, store)
    payload = {**spec, "seed": cfg.seed, "split": "hm2y_validation_2020-09-02_08",
               "code_version": CODE_VERSION, "source_revision": prepared["revision"],
               "evaluated_at": datetime.now(timezone.utc).isoformat(), "curve": history}
    common.atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs,
                        selection="none", result_path=str(path))
    return payload


def curve_table(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" in record:
                rows.append({"model_id": arm["model_id"], "role": arm["role"],
                             "seed": arm["seed"], "epoch": record["epoch"],
                             "loss": record["loss"], "p_correct": record["p_correct"],
                             **record.get("score_split", {}),
                             **record.get("gradient_diagnostics", {}), **record["metrics"]})
    return pd.DataFrame(rows)


def gap_table(curve: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in curve.columns if "@" in column]
    baseline = curve[curve.model_id.eq(M1_MODEL_ID)].set_index("epoch")
    rows = []
    for _, m2 in curve[curve.model_id.eq(M2_MODEL_ID)].iterrows():
        m1 = baseline.loc[int(m2["epoch"])]
        row = {"seed": int(m2["seed"]), "epoch": int(m2["epoch"])}
        for metric in metrics:
            base, value = float(m1[metric]), float(m2[metric])
            row[metric] = value - base
            row[f"{metric}_ratio"] = value / base if base else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def exploratory_peak_comparison(curve: pd.DataFrame) -> dict:
    peaks = {}
    for model_id in (M1_MODEL_ID, M2_MODEL_ID):
        subset = curve[curve.model_id.eq(model_id)]
        row = subset.loc[subset["recall@10"].idxmax()]
        peaks[model_id] = {"epoch": int(row["epoch"]), "recall@10": float(row["recall@10"])}
    return {"metric": "recall@10", "exploratory_only": True,
            "epoch_selection_allowed": False, "peaks": peaks,
            "m2_minus_m1_peak": peaks[M2_MODEL_ID]["recall@10"] - peaks[M1_MODEL_ID]["recall@10"]}


def run_hm2y_training_budget(cfg: HMM2TrainingBudgetConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_hm2y_training_budget())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    arms = []
    for spec in arm_specifications():
        print(f"\n===== {spec['model_id']} | seed {cfg.seed} | {cfg.epochs} epochs =====")
        arms.append(_run_arm(prepared, cfg, spec))
    curve = curve_table(arms); gap = gap_table(curve)
    peaks = exploratory_peak_comparison(curve)
    stem = f"clv_m2_training_budget_hm2y_{prepared['config_hash']}"
    paths = {"curve_csv": prepared["out_dir"] / f"{stem}_curve.csv",
             "gap_csv": prepared["out_dir"] / f"{stem}_gap.csv",
             "json": prepared["out_dir"] / f"{stem}.json"}
    test10._atomic_csv(paths["curve_csv"], curve); test10._atomic_csv(paths["gap_csv"], gap)
    test10._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg),
        "preflight": preflight_summary(cfg), "source_revision": prepared["revision"],
        "history_diagnostics": prepared["history"].diagnostics,
        "curve": curve.to_dict("records"), "gap": gap.to_dict("records"),
        "exploratory_peak_comparison": peaks,
        "result_paths": {key: str(value) for key, value in paths.items()},
    })
    print("\n1) H&M seed 43 M1·M2 학습곡선"); print(curve.to_string(index=False))
    print("\n2) 동일 epoch M2-M1 비교"); print(gap.to_string(index=False))
    print("\n3) 탐색적 최고점 비교 (epoch 선택 근거 아님)")
    print(json.dumps(peaks, ensure_ascii=False, indent=2))
    curve.attrs.update(gap=gap, exploratory_peak_comparison=peaks,
                       result_paths={key: str(value) for key, value in paths.items()})
    return curve


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_hm2y_training_budget()), ensure_ascii=False, indent=2))
