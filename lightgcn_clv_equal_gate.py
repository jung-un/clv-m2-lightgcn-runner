"""Fast Dunnhumby screen for equal-gate CLV N/V embeddings.

This runner changes one mechanism from the centered-signed experiment:
``g_N(u)=g_V(u)=1``.  User specificity remains in the learned N/V user
representations.  M1 and M2 are trained with the same seed, batches, negative
sampler, budget, validation split, and checkpoint-selection rules.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import clone_state
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-equal-gate-symmetric-selection-v1"
MODEL_ID = "m2_equal_gate"
GUARD_METRICS = (
    "recall@10",
    "ndcg@10",
    "recall@20",
    "ndcg@20",
    "recall@50",
    "ndcg@50",
)
SELECTION_PRIMARY = "recall_primary"
SELECTION_ECONOMIC = "economic_guarded"


def configure_equal_gate_dunnhumby_run(**overrides) -> joint.JointNVConfig:
    """Dunnhumby seed-42 validation-only M1 versus equal-gate M2 preset."""
    defaults = {
        "gate_shape": "equal",
        "gamma_init": 0.1,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "out_dir": (
            f"{v3.default_out_dir('dunnhumby')}"
            "_m2_equal_gate_symmetric_v1"
        ),
    }
    cfg = joint.configure_joint_nv_run(
        "dunnhumby", short_hm=False, **(defaults | overrides)
    )
    return validate_equal_gate_config(cfg)


def validate_equal_gate_config(
    cfg: joint.JointNVConfig,
) -> joint.JointNVConfig:
    joint.validate_joint_nv_config(cfg)
    required = {
        "dataset": "dunnhumby",
        "seed": 42,
        "window_days": None,
        "input_days": 365,
        "gate_shape": "equal",
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
                f"equal-gate screening requires {key}={expected!r}"
            )
    return cfg


def preflight_summary(cfg: joint.JointNVConfig) -> dict:
    cfg = validate_equal_gate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "models": ["m1", MODEL_ID],
        "architecture": (
            "separate ID|N|V layer-0 blocks -> one binary LightGCN -> "
            "one final dot score"
        ),
        "user_specificity": (
            "equal gate g_N=g_V=1; user-specific effects are learned in "
            "h_u^N and h_u^V"
        ),
        "preference_preservation": (
            "BPR(S_ID) + BPR(stopgrad(S_ID)+S_N+S_V)"
        ),
        "gamma": (
            "learned sqrt-gamma on both user/item sides; initial gamma=0.1"
        ),
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "m4_sample_weighting": False,
        "selection_rules": {
            SELECTION_PRIMARY: "maximum validation recall@10",
            SELECTION_ECONOMIC: (
                "maximum price/purchase-amount weighted hit@10 among epochs "
                "passing the same six 99% M1 accuracy guardrails"
            ),
        },
        "selection_symmetry": (
            "the identical rules and guard thresholds are applied to M1 and M2"
        ),
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


def _canonical_m1_reference(prepared, cfg):
    """Load the established M1 only to fix guard thresholds before training."""
    data = prepared["data"]
    gate = torch.ones(data["n_users"], device=v3.DEVICE)
    model, training = v3.get_or_train(
        "pref_only",
        cfg.seed,
        data,
        gate,
        data["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        prepared["meta"],
        prepared["cache"],
        prepared["base_cfg"],
    )
    model.eval()
    metrics, _ = joint._evaluate(model, prepared, per_user=False)
    return metrics, training


def _fresh_m1(prepared, cfg):
    v3.set_seed(cfg.seed)
    data = prepared["data"]
    return v3.build_model(
        data,
        data["x_val_u"],
        prepared["x_item"],
        prepared["item_cat"],
        prepared["base_cfg"],
    )


def _passes_guard(metrics: dict, thresholds: dict[str, float]) -> bool:
    return all(float(metrics[key]) >= thresholds[key] for key in GUARD_METRICS)


def _state_metrics(model, prepared):
    model.eval()
    metrics, _ = joint._evaluate(model, prepared, per_user=False)
    return metrics


def _train_with_symmetric_selection(
    model,
    params,
    prepared,
    cfg,
    model_id: str,
    guard_thresholds: dict[str, float],
) -> dict:
    """Train one model and retain recall-primary and guarded-economic states."""
    base_cfg, data = prepared["base_cfg"], prepared["data"]
    optimizer = torch.optim.Adam(params, lr=base_cfg["LR"], weight_decay=0.0)
    rng = np.random.default_rng(cfg.seed)
    tr_u, tr_i, pos_key = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / base_cfg["BATCH_SIZE"])
    ones = torch.ones(data["n_users"], device=v3.DEVICE)
    store = joint._progress_store(
        prepared["out_dir"],
        model_id,
        cfg,
        prepared["config_hash"],
        prepared["input_hash"],
        prepared["revision"],
    )

    primary_score = -float("inf")
    primary_epoch = 0
    primary_state = None
    economic_score = -float("inf")
    economic_epoch = 0
    economic_state = None
    history = []
    bad = updates = samples = 0
    start_epoch, resumed_from = 1, 0
    previous_wall = 0.0
    restored = store.restore_epoch(model, optimizer, rng)
    if restored is not None:
        start_epoch = int(restored["next_epoch"])
        resumed_from = start_epoch - 1
        primary_score = float(restored["primary_score"])
        primary_epoch = int(restored["primary_epoch"])
        primary_state = restored["primary_state"]
        economic_score = float(restored["economic_score"])
        economic_epoch = int(restored["economic_epoch"])
        economic_state = restored["economic_state"]
        history = list(restored.get("history", []))
        bad = int(restored["bad"])
        updates = int(restored["updates"])
        samples = int(restored["samples"])
        previous_wall = float(restored.get("wall_clock_sec", 0.0))
        print(f"  [{model_id}] epoch {resumed_from}에서 자동 재개")
    store.mark_stage("running", epoch=resumed_from, max_epoch=cfg.max_epochs)

    started = time.time()
    last_epoch = resumed_from
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        total_loss = total_bpr = total_correct = 0.0
        for batch in range(n_batches):
            idx = permutation[
                batch * base_cfg["BATCH_SIZE"] : (batch + 1) * base_cfg["BATCH_SIZE"]
            ]
            users, positives = tr_u[idx], tr_i[idx]
            negatives = v3.sample_negatives(
                users,
                positives,
                data["n_items"],
                pos_key,
                rng,
                "uniform",
                data["item_cat"],
                data["cat_items"],
            )
            loss, diagnostics = model.bpr_loss(
                torch.as_tensor(users, dtype=torch.long, device=v3.DEVICE),
                torch.as_tensor(positives, dtype=torch.long, device=v3.DEVICE),
                torch.as_tensor(negatives, dtype=torch.long, device=v3.DEVICE),
                ones,
                0.0,
                None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            total_bpr += float(diagnostics["bpr"])
            total_correct += float(diagnostics["p_correct"])
            updates += 1
            samples += len(idx)
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.max_epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=total_loss / (batch + 1),
            )

        metrics = _state_metrics(model, prepared)
        recall_score = float(metrics["recall@10"])
        guard_pass = _passes_guard(metrics, guard_thresholds)
        economic_value = float(metrics["revenue@10"])
        primary_mark = economic_mark = ""
        if recall_score > primary_score + 1e-12:
            primary_score = recall_score
            primary_epoch = epoch
            primary_state = clone_state(model)
            bad = 0
            primary_mark = " R★"
        else:
            bad += 1
        if guard_pass and economic_value > economic_score + 1e-12:
            economic_score = economic_value
            economic_epoch = epoch
            economic_state = clone_state(model)
            economic_mark = " E★"

        gamma_n = (
            float(model.gamma_n.detach().cpu())
            if hasattr(model, "gamma_n")
            else float("nan")
        )
        gamma_v = (
            float(model.gamma_v.detach().cpu())
            if hasattr(model, "gamma_v")
            else float("nan")
        )
        epoch_row = {
            "epoch": int(epoch),
            "loss": total_loss / n_batches,
            "bpr": total_bpr / n_batches,
            "p_correct": total_correct / n_batches,
            "guard_pass": bool(guard_pass),
            "gamma_n": gamma_n,
            "gamma_v": gamma_v,
            "epoch_sec": time.time() - epoch_started,
            **{key: float(value) for key, value in metrics.items()},
        }
        history.append(epoch_row)
        print(
            f"  [{model_id}] ep {epoch:3d} | "
            f"R@10 {recall_score:.6f} | N@10 {metrics['ndcg@10']:.6f} | "
            f"weighted-hit@10 {economic_value:.6f} | "
            f"distinct {int(metrics['n_distinct@10'])} | "
            f"eff.cat {metrics['eff_catalog@10']:.2f} | "
            f"guard {'Y' if guard_pass else 'N'}{primary_mark}{economic_mark}"
        )
        store.save_epoch(
            model,
            optimizer,
            rng,
            epoch=epoch,
            primary_score=primary_score,
            primary_epoch=primary_epoch,
            primary_state=primary_state,
            economic_score=economic_score,
            economic_epoch=economic_epoch,
            economic_state=economic_state,
            best_epoch=primary_epoch,
            best_metric=primary_score,
            bad=bad,
            updates=updates,
            samples=samples,
            history=history,
            wall_clock_sec=previous_wall + time.time() - started,
        )
        if bad >= cfg.early_stop:
            print(f"  [{model_id}] early stop")
            break

    if primary_state is None:
        raise RuntimeError(f"{model_id}: recall-primary checkpoint가 없습니다")
    selections = {
        SELECTION_PRIMARY: {
            "epoch": primary_epoch,
            "score": primary_score,
            "state": primary_state,
        }
    }
    if economic_state is not None:
        selections[SELECTION_ECONOMIC] = {
            "epoch": economic_epoch,
            "score": economic_score,
            "state": economic_state,
        }

    evaluated = {}
    for rule, selection in selections.items():
        model.load_state_dict(selection["state"])
        model.eval()
        metrics, per_user = joint._evaluate(model, prepared, per_user=True)
        diagnostics = (
            model.score_diagnostics(seed=cfg.seed)
            if hasattr(model, "gamma_n")
            else {}
        )
        checkpoint = prepared["out_dir"] / (
            f"{model_id}_{rule}_{cfg.dataset}_s{cfg.seed}_"
            f"{prepared['config_hash']}.pt"
        )
        torch.save(
            {
                "state": selection["state"],
                "selection_rule": rule,
                "selected_epoch": selection["epoch"],
                "metrics": metrics,
                "diagnostics": diagnostics,
                "config": asdict(cfg),
                "source_revision": prepared["revision"],
                "input_hash": prepared["input_hash"],
            },
            checkpoint,
        )
        evaluated[rule] = {
            "epoch": selection["epoch"],
            "score": selection["score"],
            "metrics": metrics,
            "per_user": per_user,
            "diagnostics": diagnostics,
            "checkpoint": str(checkpoint),
        }
    store.mark_complete(
        best_epoch=primary_epoch,
        best_metric=primary_score,
        checkpoint_path=evaluated[SELECTION_PRIMARY]["checkpoint"],
    )
    return {
        "evaluated": evaluated,
        "history": history,
        "epochs_run": last_epoch,
        "resumed_from_epoch": resumed_from,
        "updates": updates,
        "samples": samples,
    }


def _result_rows(
    cfg,
    runs,
    *,
    model_id: str = MODEL_ID,
    gate_label: str = "equal",
) -> list[dict]:
    rows = []
    for current_model_id, role in (("m1", "baseline"), (model_id, "model")):
        for rule, result in runs[current_model_id]["evaluated"].items():
            row = joint.result_row(
                current_model_id,
                role,
                "none" if current_model_id == "m1" else gate_label,
                cfg.seed,
                _public(result["metrics"]),
                result["diagnostics"],
            )
            row.update(
                selection_rule=rule,
                selected_epoch=int(result["epoch"]),
                selection_score=float(result["score"]),
            )
            rows.append(row)
    return rows


def screening_decision(
    rows: list[dict], *, model_id: str = MODEL_ID
) -> dict:
    table = pd.DataFrame(rows).set_index(["selection_rule", "model_id"])
    views = {}
    for rule in (SELECTION_PRIMARY, SELECTION_ECONOMIC):
        if (rule, "m1") not in table.index or (rule, model_id) not in table.index:
            views[rule] = {"available": False, "success": False}
            continue
        baseline = table.loc[(rule, "m1")]
        model = table.loc[(rule, model_id)]
        ratios = {
            metric: float(model[metric] / max(float(baseline[metric]), 1e-12))
            for metric in GUARD_METRICS
        }
        delta = float(model["revenue@10"] - baseline["revenue@10"])
        views[rule] = {
            "available": True,
            "success": bool(delta > 0.0 and min(ratios.values()) >= 0.99),
            "weighted_hit@10_delta": delta,
            "accuracy_ratios": ratios,
        }
    primary_success = views[SELECTION_PRIMARY]["success"]
    economic_success = views[SELECTION_ECONOMIC]["success"]
    return {
        "success": bool(primary_success and economic_success),
        "classification": (
            "strong_success"
            if primary_success and economic_success
            else "conditional_economic_success"
            if economic_success
            else "failed"
        ),
        "views": views,
        "note": (
            "All decisions are post-hoc readings. revenue@10 is a "
            "price/purchase-amount weighted hit, not actual incremental revenue."
        ),
    }


def _paired_rows(runs, *, model_id: str = MODEL_ID) -> list[dict]:
    rows = []
    for rule in (SELECTION_PRIMARY, SELECTION_ECONOMIC):
        if rule not in runs["m1"]["evaluated"] or rule not in runs[model_id]["evaluated"]:
            continue
        baseline = runs["m1"]["evaluated"][rule]["per_user"]
        model = runs[model_id]["evaluated"][rule]["per_user"]
        for metric in ("recall", "ndcg", "revenue", "arp"):
            rows.append(
                {
                    "model_id": model_id,
                    "reference": "m1",
                    "selection_rule": rule,
                    "split": "val",
                    "metric": metric,
                    **v3.paired_bootstrap(
                        [model[metric] - baseline[metric]],
                        prepared_bootstrap_count(),
                    ),
                }
            )
    return rows


def prepared_bootstrap_count() -> int:
    return int(v3.CFG.get("N_BOOT", 1000))


def _persist(
    prepared,
    cfg,
    rows,
    paired,
    runs,
    guard_reference,
    decision,
    *,
    stem_prefix: str = "m2_equal_gate_dunnhumby",
    code_version: str = CODE_VERSION,
    preflight: dict | None = None,
):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_prefix}_{prepared['config_hash']}"
    csv_path = out / f"{stem}.csv"
    paired_path = out / f"{stem}_paired.csv"
    history_path = out / f"{stem}_epoch_history.csv"
    json_path = out / f"{stem}.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(paired).to_csv(paired_path, index=False)
    history_rows = []
    for model_id, run in runs.items():
        history_rows.extend(
            {"model_id": model_id, **row} for row in run["history"]
        )
    pd.DataFrame(history_rows).to_csv(history_path, index=False)
    payload = {
        "code_version": code_version,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight if preflight is not None else preflight_summary(cfg),
        "guard_reference": guard_reference,
        "decision": decision,
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": paired,
        "training": {
            model_id: {
                key: value
                for key, value in run.items()
                if key != "evaluated"
            }
            for model_id, run in runs.items()
        },
        "checkpoints": {
            model_id: {
                rule: result["checkpoint"]
                for rule, result in run["evaluated"].items()
            }
            for model_id, run in runs.items()
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["screening_decision"] = decision
    frame.attrs["guard_reference"] = guard_reference
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "paired_csv": str(paired_path),
        "epoch_history_csv": str(history_path),
        "json": str(json_path),
    }
    return frame


def run_experiment(
    cfg: joint.JointNVConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_equal_gate_config(
        cfg or configure_equal_gate_dunnhumby_run()
    )
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = joint._prepare(cfg)
    canonical_metrics, canonical_training = _canonical_m1_reference(prepared, cfg)
    guard_thresholds = {
        metric: 0.99 * float(canonical_metrics[metric])
        for metric in GUARD_METRICS
    }
    guard_reference = {
        "source": "established same-seed M1 validation checkpoint",
        "metrics": {metric: float(canonical_metrics[metric]) for metric in GUARD_METRICS},
        "thresholds_99pct": guard_thresholds,
        "training": canonical_training,
    }
    print("대칭 경제 체크포인트의 정확도 하한:")
    print(json.dumps(guard_thresholds, ensure_ascii=False, indent=2))

    fresh_m1 = _fresh_m1(prepared, cfg)
    runs = {
        "m1": _train_with_symmetric_selection(
            fresh_m1,
            list(fresh_m1.pref_params()),
            prepared,
            cfg,
            "m1_symmetric",
            guard_thresholds,
        )
    }
    v3.set_seed(cfg.seed)
    m2 = joint._build_model(prepared, cfg, "joint_nv")
    runs[MODEL_ID] = _train_with_symmetric_selection(
        m2,
        list(m2.parameters()),
        prepared,
        cfg,
        MODEL_ID,
        guard_thresholds,
    )
    rows = _result_rows(cfg, runs)
    decision = screening_decision(rows)
    paired = _paired_rows(runs)
    frame = _persist(
        prepared,
        cfg,
        rows,
        paired,
        runs,
        guard_reference,
        decision,
    )
    print("equal-gate M2 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_equal_gate_dunnhumby_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
