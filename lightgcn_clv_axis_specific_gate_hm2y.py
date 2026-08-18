"""H&M full-period validation run for the selected axis-specific M2.

The model is the same ID|N|V LightGCN that passed the Dunnhumby seed-42
screen.  This runner changes only the dataset scale and the operational
checkpoint format: static graph/feature buffers are rebuilt from the input
data and are not duplicated in every epoch checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_run_state import ProgressStore
import lightgcn_clv_axis_specific_gate as axis_gate
import lightgcn_clv_equal_gate as equal
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-axis-specific-gate-hm2y-v1"
MODEL_ID = axis_gate.MODEL_ID
DEFAULT_BATCH_SIZE = 131_072


def configure_axis_specific_gate_hm2y_run(**overrides) -> joint.JointNVConfig:
    """H&M full-period, seed-42, validation-only generalization preset."""
    defaults = {
        "gate_shape": "axis_positive",
        "gamma_init": 0.1,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "batch_size": DEFAULT_BATCH_SIZE,
        "out_dir": (
            f"{v3.default_out_dir('hm')}"
            "_m2_axis_specific_nonnegative_gate_hm2y_v1"
        ),
        "m1_checkpoint_dir": v3.default_out_dir("hm"),
    }
    cfg = joint.configure_joint_nv_run(
        "hm", short_hm=False, **(defaults | overrides)
    )
    return validate_hm2y_config(cfg)


def validate_hm2y_config(cfg: joint.JointNVConfig) -> joint.JointNVConfig:
    joint.validate_joint_nv_config(cfg)
    required = {
        "dataset": "hm",
        "seed": 42,
        "window_days": None,
        # Keep the successful model's one-year N/V history while the
        # collaborative graph uses the full H&M period.
        "input_days": 365,
        "gate_shape": "axis_positive",
        "id_dim": 64,
        "axis_dim": 16,
        "hidden_dim": 32,
        "n_layers": 2,
        "gamma_init": 0.1,
        "anchor_weight": 0.0,
        "preference_preserving": True,
        "compute_variable_validity": False,
        "eval_test": False,
        "eval_holdout": False,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(
                f"H&M 2년 M2 실행은 {key}={expected!r}만 허용합니다"
            )
    if cfg.batch_size not in {131_072, 65_536, 32_768}:
        raise ValueError("batch_size는 131072/65536/32768 중 하나여야 합니다")
    return cfg


def preflight_summary(cfg: joint.JointNVConfig) -> dict:
    cfg = validate_hm2y_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full_history_about_2_years",
        "axis_input_history_days": cfg.input_days,
        "seed": cfg.seed,
        "models": ["existing_or_new_m1", MODEL_ID],
        "architecture": (
            "ID|N|V layer-0 blocks -> one binary LightGCN -> one dot score"
        ),
        "user_allocation": {
            "N": "positive mean-one gate from each user's N percentile",
            "V": "positive mean-one gate from each user's V percentile",
        },
        "loss": "BPR(S_ID) + BPR(stopgrad(S_ID)+S_N+S_V)",
        "graph_mode": "binary",
        "negative_sampling": "uniform",
        "sample_weighting": False,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "early_stop": cfg.early_stop,
        "official_selection": "maximum validation recall@10 for both M1 and M2",
        "exploratory_checkpoint": (
            "M2 weighted-hit@10 maximum among epochs passing the frozen "
            "six-metric 99% M1 guard"
        ),
        "checkpointing": (
            "every epoch to Drive; optimizer/RNG restored automatically; "
            "static graph and feature buffers are rebuilt rather than duplicated"
        ),
        "eval_test": False,
        "eval_holdout": False,
        "out_dir": cfg.out_dir,
    }


def _parameter_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Checkpoint trainable state only; H&M adjacency is reconstructed."""
    names = set(dict(model.named_parameters()))
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in names
    }


def _load_parameter_state(
    model: torch.nn.Module, state: dict[str, torch.Tensor]
) -> None:
    parameter_names = set(dict(model.named_parameters()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_parameters = parameter_names.intersection(missing)
    if missing_parameters or unexpected:
        raise RuntimeError(
            "경량 checkpoint 파라미터 불일치: "
            f"missing={sorted(missing_parameters)}, unexpected={unexpected}"
        )


def _atomic_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_best(path: Path, model: torch.nn.Module, epoch: int, score: float) -> None:
    _atomic_torch(
        path,
        {
            "parameter_state": _parameter_state(model),
            "epoch": int(epoch),
            "score": float(score),
        },
    )


def _restore_latest(
    store: ProgressStore,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
) -> dict | None:
    if not store.latest_checkpoint.exists():
        return None
    payload = torch.load(
        store.latest_checkpoint, map_location="cpu", weights_only=False
    )
    store._validate_identity(payload.get("identity", {}))
    _load_parameter_state(model, payload["parameter_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    rng.bit_generator.state = payload["numpy_rng_state"]
    torch.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    return payload


def _save_latest(
    store: ProgressStore,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: np.random.Generator,
    state: dict,
) -> None:
    payload = {
        "identity": asdict(store.identity),
        "parameter_state": _parameter_state(model),
        "optimizer_state": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        **state,
    }
    _atomic_torch(store.latest_checkpoint, payload)
    store.mark_stage(
        "running",
        epoch=int(state["epoch"]),
        max_epoch=int(state["max_epoch"]),
        best_epoch=int(state["primary_epoch"]),
        best_metric=float(state["primary_score"]),
        checkpoint_path=str(store.latest_checkpoint),
    )


def _train_m2(
    model: torch.nn.Module,
    prepared: dict,
    cfg: joint.JointNVConfig,
    guard_thresholds: dict[str, float],
) -> dict:
    """Train current M2 with compact epoch resume and two saved views."""
    base_cfg, data = prepared["base_cfg"], prepared["data"]
    optimizer = torch.optim.Adam(
        list(model.parameters()), lr=base_cfg["LR"], weight_decay=0.0
    )
    rng = np.random.default_rng(cfg.seed)
    tr_u, tr_i, pos_key = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / base_cfg["BATCH_SIZE"])
    ones = torch.ones(data["n_users"], device=v3.DEVICE)
    store = joint._progress_store(
        prepared["out_dir"],
        "m2_axis_specific_gate_hm2y",
        cfg,
        prepared["config_hash"],
        prepared["input_hash"],
        prepared["revision"],
    )
    best_dir = store.root / "best"
    primary_path = best_dir / "recall_primary.pt"
    economic_path = best_dir / "economic_guarded.pt"

    start_epoch = 1
    primary_score = economic_score = -float("inf")
    primary_epoch = economic_epoch = 0
    bad = updates = samples = 0
    history: list[dict] = []
    restored = _restore_latest(store, model, optimizer, rng)
    if restored is not None:
        start_epoch = int(restored["epoch"]) + 1
        primary_score = float(restored["primary_score"])
        economic_score = float(restored["economic_score"])
        primary_epoch = int(restored["primary_epoch"])
        economic_epoch = int(restored["economic_epoch"])
        bad = int(restored["bad"])
        updates = int(restored["updates"])
        samples = int(restored["samples"])
        history = list(restored.get("history", []))
        print(f"  [{MODEL_ID}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage(
        "running",
        epoch=start_epoch - 1,
        max_epoch=cfg.max_epochs,
        batches=n_batches,
    )

    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.max_epochs + 1):
        last_epoch = epoch
        epoch_started = time.time()
        model.train()
        permutation = rng.permutation(n_train)
        total_loss = total_bpr = total_correct = 0.0
        for batch in range(n_batches):
            idx = permutation[
                batch * base_cfg["BATCH_SIZE"] :
                (batch + 1) * base_cfg["BATCH_SIZE"]
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
            tensors = [
                torch.as_tensor(values, dtype=torch.long, device=v3.DEVICE)
                for values in (users, positives, negatives)
            ]
            loss, diagnostics = model.bpr_loss(
                *tensors, ones, 0.0, None
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

        model.eval()
        metrics, _ = joint._evaluate(model, prepared, per_user=False)
        recall_score = float(metrics["recall@10"])
        economic_value = float(metrics["revenue@10"])
        guard_pass = equal._passes_guard(metrics, guard_thresholds)
        marks = ""
        if recall_score > primary_score + 1e-12:
            primary_score = recall_score
            primary_epoch = epoch
            _save_best(primary_path, model, epoch, recall_score)
            bad = 0
            marks += " R★"
        else:
            bad += 1
        if guard_pass and economic_value > economic_score + 1e-12:
            economic_score = economic_value
            economic_epoch = epoch
            _save_best(economic_path, model, epoch, economic_value)
            marks += " E★"

        row = {
            "epoch": int(epoch),
            "loss": total_loss / n_batches,
            "bpr": total_bpr / n_batches,
            "p_correct": total_correct / n_batches,
            "guard_pass": bool(guard_pass),
            "gamma_n": float(model.gamma_n.detach().cpu()),
            "gamma_v": float(model.gamma_v.detach().cpu()),
            "epoch_sec": time.time() - epoch_started,
            **{key: float(value) for key, value in metrics.items()},
        }
        history.append(row)
        print(
            f"  [{MODEL_ID}] ep {epoch:3d} | "
            f"R@10 {recall_score:.6f} | N@10 {metrics['ndcg@10']:.6f} | "
            f"weighted-hit@10 {economic_value:.6f} | "
            f"distinct {int(metrics['n_distinct@10'])} | "
            f"guard {'Y' if guard_pass else 'N'}{marks}"
        )
        _save_latest(
            store,
            model,
            optimizer,
            rng,
            {
                "epoch": epoch,
                "max_epoch": cfg.max_epochs,
                "primary_score": primary_score,
                "primary_epoch": primary_epoch,
                "economic_score": economic_score,
                "economic_epoch": economic_epoch,
                "bad": bad,
                "updates": updates,
                "samples": samples,
                "history": history,
            },
        )
        if bad >= cfg.early_stop:
            print(f"  [{MODEL_ID}] early stop")
            break

    if not primary_path.exists():
        raise RuntimeError("recall-primary checkpoint가 없습니다")
    selected_paths = {equal.SELECTION_PRIMARY: primary_path}
    if economic_path.exists():
        selected_paths[equal.SELECTION_ECONOMIC] = economic_path
    evaluated = {}
    for rule, path in selected_paths.items():
        blob = torch.load(path, map_location="cpu", weights_only=False)
        _load_parameter_state(model, blob["parameter_state"])
        model.eval()
        metrics, per_user = joint._evaluate(model, prepared, per_user=True)
        diagnostics = model.score_diagnostics(seed=cfg.seed)
        evaluated[rule] = {
            "epoch": int(blob["epoch"]),
            "score": float(blob["score"]),
            "metrics": metrics,
            "per_user": per_user,
            "diagnostics": diagnostics,
            "checkpoint": str(path),
        }
    store.mark_complete(
        best_epoch=primary_epoch,
        best_metric=primary_score,
        checkpoint_path=str(primary_path),
    )
    return {
        "evaluated": evaluated,
        "history": history,
        "epochs_run": last_epoch,
        "resumed_from_epoch": start_epoch - 1,
        "updates": updates,
        "samples": samples,
    }


def _m1_reference(prepared: dict, cfg: joint.JointNVConfig) -> dict:
    data = prepared["data"]
    gate = torch.ones(data["n_users"], device=v3.DEVICE)
    store = joint._progress_store(
        prepared["out_dir"],
        "m1_reference",
        cfg,
        prepared["config_hash"],
        prepared["input_hash"],
        prepared["revision"],
    )
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
        progress_store=store,
    )
    model.eval()
    metrics, per_user = joint._evaluate(model, prepared, per_user=True)
    store.mark_complete(best_metric=float(metrics["recall@10"]))
    phases = training if isinstance(training, list) else [training]
    selected_epoch = next(
        (
            int(phase.get("best_epoch", 0))
            for phase in reversed(phases)
            if isinstance(phase, dict)
        ),
        0,
    )
    return {
        "training": training,
        "metrics": metrics,
        "per_user": per_user,
        "epoch": selected_epoch,
    }


def _decision(m1: dict, m2: dict) -> dict:
    ratios = {
        metric: float(m2["metrics"][metric])
        / max(float(m1["metrics"][metric]), 1e-12)
        for metric in equal.GUARD_METRICS
    }
    delta = float(m2["metrics"]["revenue@10"] - m1["metrics"]["revenue@10"])
    return {
        "success": bool(delta > 0.0 and min(ratios.values()) >= 0.99),
        "classification": (
            "generalizes_on_hm2y" if delta > 0.0 and min(ratios.values()) >= 0.99
            else "does_not_generalize_on_hm2y"
        ),
        "official_selection_rule": equal.SELECTION_PRIMARY,
        "weighted_hit@10_delta": delta,
        "accuracy_ratios": ratios,
        "statistical_note": (
            "seed 42 validation screen; statistical significance is not claimed"
        ),
    }


def _row(model_id: str, role: str, rule: str, epoch: int, metrics: dict, diagnostics=None):
    row = joint.result_row(
        model_id,
        role,
        "none" if role == "baseline" else "axis_positive",
        42,
        equal._public(metrics),
        diagnostics,
    )
    row.update(selection_rule=rule, selected_epoch=int(epoch))
    return row


def read_progress(out_dir: str | Path) -> dict:
    root = Path(out_dir) / "progress"
    candidates = sorted(
        root.glob("*/progress.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return {"status": "not_started", "root": str(root)}
    payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    payload["progress_path"] = str(candidates[0])
    return payload


def load_completed_result(out_dir: str | Path) -> pd.DataFrame:
    """Reload final artifacts after a Colab runtime reconnect."""
    root = Path(out_dir)
    candidates = sorted(
        root.glob("m2_axis_specific_gate_hm2y_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"완료 결과 JSON을 찾을 수 없습니다: {root}")
    json_path = candidates[0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    csv_path = json_path.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"완료 결과 CSV를 찾을 수 없습니다: {csv_path}")
    frame = pd.read_csv(csv_path)
    frame.attrs["decision"] = payload["decision"]
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "paired_csv": str(csv_path.with_name(f"{csv_path.stem}_paired.csv")),
        "epoch_history_csv": str(
            csv_path.with_name(f"{csv_path.stem}_epoch_history.csv")
        ),
        "json": str(json_path),
        "progress": read_progress(root).get("progress_path"),
    }
    return frame


def run_experiment(cfg: joint.JointNVConfig | None = None) -> pd.DataFrame:
    cfg = validate_hm2y_config(cfg or configure_axis_specific_gate_hm2y_run())
    preflight = preflight_summary(cfg)
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    prepared = joint._prepare(cfg)
    m1 = _m1_reference(prepared, cfg)
    guard_thresholds = {
        metric: 0.99 * float(m1["metrics"][metric])
        for metric in equal.GUARD_METRICS
    }
    v3.set_seed(cfg.seed)
    model = joint._build_model(prepared, cfg, "joint_nv")
    m2_run = _train_m2(model, prepared, cfg, guard_thresholds)
    primary = m2_run["evaluated"][equal.SELECTION_PRIMARY]
    decision = _decision(m1, primary)

    rows = [
        _row("m1", "baseline", equal.SELECTION_PRIMARY, m1["epoch"], m1["metrics"]),
        _row(
            MODEL_ID,
            "model",
            equal.SELECTION_PRIMARY,
            primary["epoch"],
            primary["metrics"],
            primary["diagnostics"],
        ),
    ]
    if equal.SELECTION_ECONOMIC in m2_run["evaluated"]:
        exploratory = m2_run["evaluated"][equal.SELECTION_ECONOMIC]
        rows.append(
            _row(
                MODEL_ID,
                "model_exploratory",
                equal.SELECTION_ECONOMIC,
                exploratory["epoch"],
                exploratory["metrics"],
                exploratory["diagnostics"],
            )
        )

    paired = []
    for rule, result in m2_run["evaluated"].items():
        for metric in ("recall", "ndcg", "revenue", "arp"):
            paired.append(
                {
                    "model_id": MODEL_ID,
                    "reference": "m1_recall_primary",
                    "selection_rule": rule,
                    "split": "val",
                    "metric": metric,
                    **v3.paired_bootstrap(
                        [result["per_user"][metric] - m1["per_user"][metric]],
                        prepared["base_cfg"]["N_BOOT"],
                    ),
                }
            )

    out = Path(cfg.out_dir)
    stem = f"m2_axis_specific_gate_hm2y_{prepared['config_hash']}"
    csv_path = out / f"{stem}.csv"
    paired_path = out / f"{stem}_paired.csv"
    history_path = out / f"{stem}_epoch_history.csv"
    json_path = out / f"{stem}.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(paired).to_csv(paired_path, index=False)
    pd.DataFrame(m2_run["history"]).to_csv(history_path, index=False)
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "preflight": preflight,
        "guard_thresholds": guard_thresholds,
        "decision": decision,
        "m1_training": m1["training"],
        "m2_training": {
            key: value for key, value in m2_run.items() if key != "evaluated"
        },
        "checkpoints": {
            rule: result["checkpoint"]
            for rule, result in m2_run["evaluated"].items()
        },
        "absolute_rows": frame.to_dict("records"),
        "paired_delta": paired,
        "interpretation": {
            "clv": "historical N and V components condition the internal representation",
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
            "selection": (
                "recall-primary is official; the economic-guarded M2 row is exploratory "
                "because no separately economic-selected M1 is trained in this fast run"
            ),
        },
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    frame.attrs["decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "paired_csv": str(paired_path),
        "epoch_history_csv": str(history_path),
        "json": str(json_path),
        "progress": read_progress(cfg.out_dir).get("progress_path"),
    }
    print("H&M 2년 M2 validation 판정:")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_axis_specific_gate_hm2y_run()),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("학습은 Colab 실행 셀에서 시작하세요.")
