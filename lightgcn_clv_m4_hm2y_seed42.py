"""H&M two-year seed-42 validation of the selected M4 loss intervention."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m4_clv_hard_negative_loss import multi_negative_bpr, sampled_l2_multineg
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_m2_hm2y_seed42 as hm2
import lightgcn_clv_m4_clv_hard_negative as selected
import lightgcn_clv_m4_clv_hard_negative_multiseed as attribution
import lightgcn_clv_v3 as v3


CODE_VERSION = "m4-clv-conditioned-hard-negative-hm2y-seed42-v1.1"
K1_MODEL_ID = selected.K1_MODEL_ID
MEAN_K5_MODEL_ID = selected.MEAN_K5_MODEL_ID
M4_MODEL_ID = selected.M4_MODEL_ID
SHUFFLED_M4_MODEL_ID = attribution.SHUFFLED_M4_MODEL_ID
MODELS = (K1_MODEL_ID, MEAN_K5_MODEL_ID, M4_MODEL_ID, SHUFFLED_M4_MODEL_ID)
ACCURACY_METRICS = common.ACCURACY_METRICS
PRIMARY_METRICS = (
    "고CLV_recall@10",
    "고CLV_ndcg@10",
    "price_purchase_amount_weighted_hit@10",
)


@dataclass(frozen=True)
class HMM4Seed42Config:
    dataset: str = "hm"
    seed: int = 42
    window_days: None = None
    input_days: int = 365
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    negative_count: int = 5
    batch_size: int = common.DEFAULT_BATCH_SIZE
    lr: float = 5e-4
    pref_reg: float = 1e-3
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 1042
    eval_test: bool = False
    eval_holdout: bool = False
    out_dir: str = ""
    m2_result_dir: str = ""


def configure_hm2y_seed42_run(**overrides) -> HMM4Seed42Config:
    defaults = {
        "out_dir": (
            f"{v3.default_out_dir('hm')}_m4_clv_hard_negative_hm2y_seed42_v1"
        ),
        "m2_result_dir": (
            f"{v3.default_out_dir('hm')}"
            "_m2_level_composition_price_hm2y_seed42_v1"
        ),
    }
    return validate_hm2y_seed42_config(
        HMM4Seed42Config(**(defaults | overrides))
    )


def validate_hm2y_seed42_config(cfg: HMM4Seed42Config) -> HMM4Seed42Config:
    required = {
        "dataset": "hm",
        "seed": 42,
        "window_days": None,
        "input_days": 365,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "negative_count": 5,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 1042,
        "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"H&M M4 seed-42 실행은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size not in common.BATCH_CANDIDATES:
        raise ValueError("batch_size는 131072/65536/32768 중 하나여야 합니다")
    if cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir or not cfg.m2_result_dir:
        raise ValueError("H&M M4 학습·M1 기준 경로 설정이 잘못됐습니다")
    return cfg


def preflight_summary(cfg: HMM4Seed42Config) -> dict:
    cfg = validate_hm2y_seed42_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full_history_about_2_years",
        "seed": 42,
        "split": "hm2y_validation",
        "trained_models": list(MODELS),
        "reported_models": list(MODELS),
        "baseline_source": (
            "reuse matching M2 rho=0 arm when available; otherwise train K=1 M1"
        ),
        "m4": {
            "uniform_negative_count": 5,
            "control": "mean BPR over the same five uniform negatives",
            "intervention": "(1-q_C)*mean_BPR + q_C*hardest-negative BPR",
            "assignment_control": "degree-matched q_C shuffle",
            "changed_from_dunnhumby_ten_seed_model": False,
        },
        "fixed": {
            "task": "new-item recommendation",
            "graph": "binary",
            "id_dim": 64,
            "layers": 2,
            "negative_sampling": "uniform",
            "min_item_interactions": 1,
            "epochs": 100,
            "epoch_selection": False,
            "test_constructed": False,
            "holdout_constructed": False,
        },
        "decision": (
            "seed-42 exploratory attribution screen: actual M4 must preserve "
            "all six accuracy metrics within 99% of K=1 M1 and beat both K=5 "
            "mean and degree-matched CLV shuffle on all three primary metrics"
        ),
        "statistical_note": (
            "H&M 2-year seed 42 validation only; no significance or generalization claim"
        ),
        "automatic_epoch_resume": True,
        "compact_parameter_only_checkpoint": True,
        "out_dir": cfg.out_dir,
        "m2_result_dir": cfg.m2_result_dir,
    }


def _arm_hash(prepared: dict, model_id: str, assignment: str) -> str:
    negative_count = 1 if model_id == K1_MODEL_ID else 5
    return hashlib.sha256(
        common.canonical(
            {
                "run": prepared["config_hash"],
                "model_id": model_id,
                "seed": 42,
                "negative_count": negative_count,
                "assignment": assignment,
            }
        ).encode()
    ).hexdigest()[:12]


def _arm_paths(prepared: dict, model_id: str) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    return {
        "result": root / f"{model_id}_s42.json",
        "checkpoint": root / f"{model_id}_s42.pt",
    }


def _build_model(prepared: dict, cfg: HMM4Seed42Config):
    v3.set_seed(42)
    model_cfg = {**prepared["base_cfg"], "ARCH": "pref_only", "DIM": 64}
    return v3.build_model(
        prepared["data"],
        prepared["data"]["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        model_cfg,
    ).to(v3.DEVICE)


def _train_arm(model, prepared, cfg, model_id, q_clv, store) -> dict:
    optimizer = torch.optim.Adam(model.pref_params(), lr=cfg.lr, weight_decay=0.0)
    rng = np.random.default_rng(42)
    restored = common.restore_compact(store, model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["epoch"]) + 1
    history = [] if restored is None else list(restored.get("history", []))
    updates = 0 if restored is None else int(restored.get("updates", 0))
    samples = 0 if restored is None else int(restored.get("samples", 0))
    if restored is not None:
        print(f"  [{model_id}] epoch {start_epoch - 1}에서 자동 재개")
    data = prepared["data"]
    negative_count = 1 if model_id == K1_MODEL_ID else cfg.negative_count
    tr_u, tr_i, pos_key = data["tr_u"], data["tr_i"], data["pos_key"]
    n_batches = math.ceil(len(tr_u) / cfg.batch_size)
    q_all = torch.as_tensor(q_clv, dtype=torch.float32, device=v3.DEVICE)
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        started = time.time()
        permutation = rng.permutation(len(tr_u))
        loss_sum = bpr_sum = correct_sum = hard_weight_sum = gap_sum = 0.0
        weight_error = 0.0
        for batch in range(n_batches):
            index = permutation[
                batch * cfg.batch_size : (batch + 1) * cfg.batch_size
            ]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = selected.sample_uniform_negative_matrix(
                users_np,
                positives_np,
                data["n_items"],
                pos_key,
                rng,
                k=negative_count,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(
                positives_np, dtype=torch.long, device=v3.DEVICE
            )
            negatives = torch.as_tensor(
                negatives_np, dtype=torch.long, device=v3.DEVICE
            )
            user_z, item_z, _, _ = model.embeddings(need_value=False)
            positive_scores = (user_z[users] * item_z[positives]).sum(1)
            negative_scores = (user_z[users, None, :] * item_z[negatives]).sum(2)
            bpr, diagnostics = multi_negative_bpr(
                positive_scores, negative_scores, q_all[users]
            )
            reg = sampled_l2_multineg(
                model.E_u.weight[users],
                model.E_i.weight[positives],
                model.E_i.weight[negatives],
                coefficient=cfg.pref_reg,
            )
            loss = bpr + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            bpr_sum += float(bpr.detach())
            correct_sum += float(diagnostics["p_correct"])
            hard_weight_sum += float(diagnostics["hardest_weight_mean"])
            gap_sum += float(diagnostics["positive_hardest_gap"])
            weight_error = max(
                weight_error, float(diagnostics["row_weight_sum_error"])
            )
            updates += 1
            samples += len(index)
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=loss_sum / (batch + 1),
            )
        record = {
            "epoch": epoch,
            "loss": loss_sum / n_batches,
            "bpr": bpr_sum / n_batches,
            "p_correct": correct_sum / n_batches,
            "hardest_negative_weight_mean": hard_weight_sum / n_batches,
            "positive_hardest_gap": gap_sum / n_batches,
            "row_weight_sum_max_error": weight_error,
            "epoch_sec": time.time() - started,
        }
        history.append(record)
        common.save_compact(
            store,
            model,
            optimizer,
            rng,
            epoch=epoch,
            max_epoch=cfg.epochs,
            history=history,
            updates=updates,
            samples=samples,
        )
        print(
            f"  [{model_id}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"hard-w {record['hardest_negative_weight_mean']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
    return {
        "epochs_run": 100,
        "selection": "none",
        "early_stopping": False,
        "resumed_from_epoch": start_epoch - 1,
        "updates": updates,
        "samples": samples,
        "history": history,
        "final_diagnostics": history[-1] if history else {},
    }


def _run_arm(prepared, cfg, *, model_id, q_clv, assignment_name):
    paths = _arm_paths(prepared, model_id)
    model = _build_model(prepared, cfg)
    if paths["result"].exists() and paths["checkpoint"].exists():
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        checkpoint = torch.load(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached M4 checkpoint와 H&M 입력 hash가 다릅니다")
        common.load_parameter_state(model, checkpoint["parameter_state"])
        model.eval()
        print(f"  [cached] {model_id}")
        return payload
    store = common.progress_store(
        prepared,
        cfg,
        model_id,
        _arm_hash(prepared, model_id, assignment_name),
    )
    training = _train_arm(model, prepared, cfg, model_id, q_clv, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    common.final_parameter_checkpoint(
        paths["checkpoint"], model, prepared, cfg, model_id, training
    )
    payload = {
        "model_id": model_id,
        "role": {
            K1_MODEL_ID: "baseline",
            MEAN_K5_MODEL_ID: "multineg_control",
            M4_MODEL_ID: "model",
            SHUFFLED_M4_MODEL_ID: "assignment_control",
        }[model_id],
        "seed": 42,
        "split": "hm2y_validation",
        "final_epoch": 100,
        "negative_count": 1 if model_id == K1_MODEL_ID else 5,
        "clv_assignment": assignment_name,
        "metrics": common.evaluate(model, prepared),
        "training": training,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": common.checkpoint_sha256(paths["checkpoint"]),
        "input_hash": prepared["input_hash"],
    }
    common.atomic_json(paths["result"], payload)
    store.mark_complete(
        epoch=100,
        max_epoch=100,
        selection="none",
        split="hm2y_validation",
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload


def _load_m1_baseline(cfg: HMM4Seed42Config, prepared: dict) -> dict | None:
    candidates = sorted(
        Path(cfg.m2_result_dir).glob("m2_level_composition_price_hm2y_seed42_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if common.canonical(payload.get("input_manifest")) != common.canonical(
            prepared["manifest"]
        ):
            continue
        arm = payload.get("arms", {}).get(hm2.MATCHED_MODEL_ID)
        if arm is None:
            continue
        return {
            "model_id": K1_MODEL_ID,
            "role": "reused_matching_m2_rho0_baseline",
            "seed": 42,
            "split": "hm2y_validation",
            "final_epoch": 100,
            "negative_count": 1,
            "metrics": arm["metrics"],
            "training": {
                "additional_training": False,
                "source_result": str(path),
            },
        }
    return None


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": arm["model_id"],
                "role": arm["role"],
                "seed": 42,
                "split": "hm2y_validation",
                "final_epoch": 100,
                "negative_count": arm["negative_count"],
                **arm.get("training", {}).get("final_diagnostics", {}),
                **arm["metrics"],
            }
            for arm in arms
        ]
    )


def seed42_decision(absolute: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    indexed = absolute.set_index("model_id")
    missing = set(MODELS) - set(indexed.index)
    if missing or len(absolute) != len(MODELS):
        raise ValueError(f"M4 seed-42 비교 arm 누락: {sorted(missing)}")
    ratios = {
        metric: float(indexed.loc[M4_MODEL_ID, metric])
        / max(float(indexed.loc[K1_MODEL_ID, metric]), 1e-12)
        for metric in ACCURACY_METRICS
    }
    paired_rows = []
    for reference in (MEAN_K5_MODEL_ID, SHUFFLED_M4_MODEL_ID):
        for metric in PRIMARY_METRICS:
            delta = float(
                indexed.loc[M4_MODEL_ID, metric] - indexed.loc[reference, metric]
            )
            paired_rows.append(
                {
                    "reference": reference,
                    "metric": metric,
                    "delta": delta,
                    "passes": delta > 0.0,
                }
            )
    paired = pd.DataFrame(paired_rows)
    guard = all(value >= 0.99 for value in ratios.values())
    economic_vs_m1 = float(
        indexed.loc[M4_MODEL_ID, "price_purchase_amount_weighted_hit@10"]
        - indexed.loc[K1_MODEL_ID, "price_purchase_amount_weighted_hit@10"]
    )
    return {
        "positive_screen": bool(
            guard and economic_vs_m1 > 0.0 and paired["passes"].all()
        ),
        "m1_accuracy_guard_pass": bool(guard),
        "m1_economic_guard_pass": bool(economic_vs_m1 > 0.0),
        "all_control_comparisons_pass": bool(paired["passes"].all()),
        "accuracy_ratios_vs_m1": ratios,
        "weighted_hit_at_10_delta_vs_m1": economic_vs_m1,
        "statistical_note": (
            "H&M 2-year seed 42 validation only; no significance or generalization claim"
        ),
    }, paired


def run_hm2y_seed42(cfg: HMM4Seed42Config | None = None) -> pd.DataFrame:
    cfg = validate_hm2y_seed42_config(cfg or configure_hm2y_seed42_run())
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = common.prepare_hm2y(cfg, code_version=CODE_VERSION)
    baseline = _load_m1_baseline(cfg, prepared)
    zeros = np.zeros_like(prepared["q_c"], dtype=np.float32)
    if baseline is None:
        print("M2 rho=0 결과가 없어 동일한 K=1 M1을 이 실행에서 학습합니다.")
        baseline = _run_arm(
            prepared,
            cfg,
            model_id=K1_MODEL_ID,
            q_clv=zeros,
            assignment_name="nonconditioned_k1",
        )
    else:
        print("동일 H&M 입력의 M2 rho=0 결과를 K=1 M1 기준으로 재사용합니다.")
    shuffle_meta = common.degree_matched_sources(
        prepared["clv_valid"],
        prepared["binary_user_degree"],
        n_bins=cfg.shuffle_degree_bins,
        seed=cfg.shuffle_seed,
    )
    shuffled_q = prepared["q_c"][shuffle_meta["source_user"]]
    arms = [
        baseline,
        _run_arm(
            prepared,
            cfg,
            model_id=MEAN_K5_MODEL_ID,
            q_clv=zeros,
            assignment_name="nonconditioned_mean",
        ),
        _run_arm(
            prepared,
            cfg,
            model_id=M4_MODEL_ID,
            q_clv=prepared["q_c"],
            assignment_name="observed",
        ),
        _run_arm(
            prepared,
            cfg,
            model_id=SHUFFLED_M4_MODEL_ID,
            q_clv=shuffled_q,
            assignment_name="degree_matched_shuffle",
        ),
    ]
    absolute = _absolute_rows(arms)
    decision, paired = seed42_decision(absolute)
    out = prepared["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    stem = f"m4_clv_hard_negative_hm2y_seed42_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "paired_csv": out / f"{stem}_paired.csv",
        "json": out / f"{stem}.json",
    }
    absolute.to_csv(paths["absolute_csv"], index=False)
    paired.to_csv(paths["paired_csv"], index=False)
    common.atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight,
            "input_manifest": prepared["manifest"],
            "absolute_rows": absolute.to_dict("records"),
            "paired_control_rows": paired.to_dict("records"),
            "decision": decision,
            "shuffle_diagnostics": {
                key: value
                for key, value in shuffle_meta.items()
                if key not in {"source_user", "stratum"}
            },
            "arms": {arm["model_id"]: arm for arm in arms},
        },
    )
    absolute.attrs["decision"] = decision
    absolute.attrs["paired_control"] = paired.to_dict("records")
    absolute.attrs["result_paths"] = {key: str(value) for key, value in paths.items()}
    print("H&M 2년 M4 seed-42 validation 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", absolute.attrs["result_paths"])
    return absolute


def read_progress(out_dir: str | Path) -> dict:
    return common.read_progress(out_dir)


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_hm2y_seed42_run()), ensure_ascii=False, indent=2))
