"""Single-negative screen for two M4 positive-weight improvements.

Every arm uses the original LightGCN BPR with one uniform unseen negative per
positive row, so no arm needs more than one negative. The CLV-conditioned
hard-negative M4 is therefore out of scope here: picking the highest scoring
negative requires at least two of them.

The 2026-09-16 runs showed that the current M4 and the value-basis M2 push the
same lever. M5 ended below M2 on the economic metrics and below M4 on accuracy,
the interaction was negative on every metric, and with M4 switched on the CLV
assignment stopped mattering (+0.09% versus its shuffle, against +2.01% for M2
alone). Two changes are screened here, each in its own pair of arms so their
effects stay separable:

* ``first_purchase`` - weight only the row where the user first buys that item.
  In Dunnhumby 53% of the high-CLV training positives are repeat purchases that
  evaluation removes; in H&M only 6%.
* ``complementary``  - weight what the M2 value basis cannot explain:
  ``1 + lambda * q_C * (1 - <RBF(q_V), RBF(price percentile)>)``. The basis dot
  product is precomputed from train-only inputs, so no gradient flows into the
  weight and the model cannot avoid the emphasis by shrinking its own fit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m5_n_conditioned_value_basis_model import fixed_value_basis
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-k1-m4-improvement-screen-v1"
M1_MODEL_ID = "m1_bpr_k1"
M2_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1"
M4_FIRST_MODEL_ID = "m4_first_purchase_weight_bpr_k1"
M5_FIRST_MODEL_ID = "m5_value_basis_first_purchase_weight_bpr_k1"
M4_COMPLEMENT_MODEL_ID = "m4_complementary_weight_bpr_k1"
M5_COMPLEMENT_MODEL_ID = "m5_value_basis_complementary_weight_bpr_k1"
MODEL_IDS = (
    M1_MODEL_ID,
    M2_MODEL_ID,
    M4_FIRST_MODEL_ID,
    M5_FIRST_MODEL_ID,
    M4_COMPLEMENT_MODEL_ID,
    M5_COMPLEMENT_MODEL_ID,
)
IMPROVEMENTS = {
    "first_purchase": (M4_FIRST_MODEL_ID, M5_FIRST_MODEL_ID),
    "complementary": (M4_COMPLEMENT_MODEL_ID, M5_COMPLEMENT_MODEL_ID),
}
ECONOMIC_METRICS = screen.ECONOMIC_METRICS
TOP10_ACCURACY_METRICS = screen.TOP10_ACCURACY_METRICS
ACCURACY_METRICS = screen.ACCURACY_METRICS
ACCURACY_GUARD = 0.99


@dataclass(frozen=True)
class M5K1ImprovementConfig(screen.M5ValueBasisK1Config):
    pass


def configure_improvement_screen(**overrides) -> M5K1ImprovementConfig:
    data_root = v3.default_out_dir("dunnhumby")
    defaults = {
        "out_dir": f"{data_root}_m5_k1_m4_improvement_screen_v1",
        "baseline_result_dir": f"{data_root}_m2_repeatshare_historical_backtest_v1",
    }
    return validate_config(M5K1ImprovementConfig(**(defaults | overrides)))


def validate_config(cfg: M5K1ImprovementConfig) -> M5K1ImprovementConfig:
    screen.validate_config(cfg)
    return cfg


def preflight_summary(cfg: M5K1ImprovementConfig) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "trained_models": list(MODEL_IDS),
        "reused_models": [],
        "research_question": (
            "Do the two M4 weight changes stop M4 and the value-basis M2 from "
            "cancelling each other, so that M5 beats both parts?"
        ),
        "loss": {
            "bpr": "mean[row weight * softplus(s(u,j) - s(u,i+))] + batch layer-0 L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "uniform over unseen items",
            "hard_negative": False,
            "hard_negative_note": (
                "excluded by design: selecting the highest scoring negative needs "
                "at least two negatives, and this study fixes the original "
                "single-negative BPR"
            ),
        },
        "weights": {
            M1_MODEL_ID: "1",
            M2_MODEL_ID: "1",
            M4_FIRST_MODEL_ID: (
                "1 + lambda*q_C*item_amount_percentile*clipped_user_bin_fit on "
                "first-purchase rows only, 1 on repeat rows"
            ),
            M4_COMPLEMENT_MODEL_ID: (
                "1 + lambda*q_C*(1 - <RBF(q_V), RBF(item amount percentile)>)"
            ),
            "normalization": "divided by the mean raw weight over all train rows",
            "lambda": cfg.positive_weight_lambda,
        },
        "m2": {
            "user_block": "sqrt(rho) * q_C * g_N(q_N) * RBF(q_V)",
            "rho": cfg.rho,
            "economic_graph_propagation": False,
        },
        "reading_rule": {
            "evaluable": "each M5 arm changes at least one user's Top-10 set versus its M4 arm",
            "beats_both_parts": (
                "M5 > max(M2, its M4 arm) on both economic metrics"
            ),
            "accuracy_geomean": "six-metric geometric mean of M5 >= its M4 arm",
            "accuracy_guard": f"each of the six accuracy metrics >= {ACCURACY_GUARD} x its M4 arm",
            "baseline": "M5 > M1 on both economic metrics",
            "improvement_pass": (
                "evaluable and beats_both_parts and accuracy_geomean and "
                "accuracy_guard and baseline"
            ),
            "clv_attribution": (
                "not tested here; the M2 assignment result of 58f4ae4e2d29 does "
                "not transfer automatically to a different loss, so a shuffle "
                "control follows only for an arm that passes"
            ),
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "epochs": cfg.epochs,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
        },
        "statistical_note": "one historical development seed; no significance or generalization claim",
        "out_dir": cfg.out_dir,
    }


def first_purchase_row_flags(train: pd.DataFrame) -> np.ndarray:
    """True on the row where a user first buys that item, in train row order."""

    order = ["t"] + (["b_raw"] if "b_raw" in train.columns else [])
    frame = train[["u_idx", "i_idx", *order]].copy()
    frame["_row"] = np.arange(len(frame), dtype=np.int64)
    ordered = frame.sort_values([*order, "_row"], kind="stable")
    first = ~ordered.duplicated(subset=["u_idx", "i_idx"], keep="first")
    flags = np.zeros(len(frame), dtype=bool)
    flags[ordered["_row"].to_numpy()[first.to_numpy()]] = True
    return flags


def value_basis_fit(prepared: dict, cfg: M5K1ImprovementConfig) -> np.ndarray:
    """Train-only <RBF(q_V(u)), RBF(price percentile(i))> for every train row."""

    user_basis = fixed_value_basis(
        np.asarray(prepared["q_v"], dtype=np.float32),
        np.asarray(prepared["clv_valid"], dtype=bool),
        bandwidth=cfg.basis_bandwidth,
    )
    item_basis = fixed_value_basis(
        np.asarray(prepared["item_amount_percentile"], dtype=np.float32),
        np.asarray(prepared["item_economic_valid"], dtype=bool),
        bandwidth=cfg.basis_bandwidth,
    )
    users = prepared["data"]["tr_u"].astype(np.int64)
    items = prepared["data"]["tr_i"].astype(np.int64)
    return np.einsum("ij,ij->i", user_basis[users], item_basis[items]).astype(
        np.float64
    )


def row_weights(
    prepared: dict, cfg: M5K1ImprovementConfig, improvement: str | None
) -> tuple[np.ndarray, dict]:
    """Per-train-row positive weights, normalized to mean 1."""

    rows = len(prepared["data"]["tr_u"])
    if improvement is None:
        return np.ones(rows, dtype=np.float64), {"weight_mode": "unweighted"}

    users = prepared["data"]["tr_u"].astype(np.int64)
    items = prepared["data"]["tr_i"].astype(np.int64)
    q_c = np.asarray(prepared["q_c"], dtype=np.float64)[users]
    if improvement == "first_purchase":
        amount = np.asarray(prepared["item_amount_percentile"], dtype=np.float64)[items]
        fit = np.clip(
            np.asarray(prepared["user_bin_fit"], dtype=np.float64)[
                users, np.asarray(prepared["item_bin"], dtype=np.int64)[items]
            ],
            0.0,
            None,
        )
        flags = first_purchase_row_flags(prepared["data"]["train"])
        raw = 1.0 + cfg.positive_weight_lambda * q_c * amount * fit * flags
        diagnostics = {
            "weight_mode": "first_purchase",
            "first_purchase_row_share": float(flags.mean()),
        }
    elif improvement == "complementary":
        fit = np.clip(value_basis_fit(prepared, cfg), 0.0, 1.0)
        raw = 1.0 + cfg.positive_weight_lambda * q_c * (1.0 - fit)
        diagnostics = {
            "weight_mode": "complementary",
            "mean_value_basis_fit": float(fit.mean()),
        }
    else:
        raise ValueError(f"알 수 없는 개선안: {improvement}")

    mean_raw = float(raw.mean())
    if not np.isfinite(mean_raw) or mean_raw <= 0:
        raise RuntimeError("학습행 평균 가중치가 유효하지 않습니다")
    weights = raw / mean_raw
    diagnostics.update(
        train_mean_raw_weight=mean_raw,
        row_weight_mean=float(weights.mean()),
        row_weight_std=float(weights.std()),
        row_weight_cv=float(weights.std() / weights.mean()),
        row_weight_min=float(weights.min()),
        row_weight_max=float(weights.max()),
    )
    return weights, diagnostics


def arm_specifications(cfg: M5K1ImprovementConfig) -> list[dict]:
    return [
        {"model_id": M1_MODEL_ID, "role": "k1_m1", "rho": 0.0, "improvement": None},
        {"model_id": M2_MODEL_ID, "role": "k1_m2_value_basis", "rho": cfg.rho, "improvement": None},
        {
            "model_id": M4_FIRST_MODEL_ID,
            "role": "k1_m4_first_purchase",
            "rho": 0.0,
            "improvement": "first_purchase",
        },
        {
            "model_id": M5_FIRST_MODEL_ID,
            "role": "k1_m5_first_purchase",
            "rho": cfg.rho,
            "improvement": "first_purchase",
        },
        {
            "model_id": M4_COMPLEMENT_MODEL_ID,
            "role": "k1_m4_complementary",
            "rho": 0.0,
            "improvement": "complementary",
        },
        {
            "model_id": M5_COMPLEMENT_MODEL_ID,
            "role": "k1_m5_complementary",
            "rho": cfg.rho,
            "improvement": "complementary",
        },
    ]


def _config_hash(cfg: M5K1ImprovementConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5K1ImprovementConfig) -> dict:
    prepared = screen._prepare(cfg)
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def _arm_paths(prepared: dict, spec: dict, cfg: M5K1ImprovementConfig) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{spec['model_id']}_s{cfg.seed}"
    return {"checkpoint": root / f"{stem}.pt", "result": root / f"{stem}.json"}


def _train_arm(
    model,
    prepared: dict,
    cfg: M5K1ImprovementConfig,
    spec: dict,
    weights: np.ndarray,
    store: ProgressStore,
) -> dict:
    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=v3.DEVICE)

    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["next_epoch"])
    history = list(restored.get("history", [])) if restored else []
    store.mark_stage("running", epoch=start_epoch - 1, max_epoch=cfg.epochs)

    started = time.time()
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.epochs + 1):
        last_epoch = epoch
        model.train()
        epoch_started = time.time()
        permutation = rng.permutation(n_train)
        totals = {"loss": 0.0, "bpr": 0.0, "p_correct": 0.0}
        for batch in range(n_batches):
            index = permutation[batch * cfg.batch_size : (batch + 1) * cfg.batch_size]
            users_np, positives_np = tr_u[index], tr_i[index]
            negatives_np = m4_helpers.sample_uniform_negative_matrix(
                users_np,
                positives_np,
                data["n_items"],
                positive_keys,
                rng,
                k=cfg.negative_count,
            )
            users = torch.as_tensor(users_np, dtype=torch.long, device=v3.DEVICE)
            positives = torch.as_tensor(positives_np, dtype=torch.long, device=v3.DEVICE)
            negatives = torch.as_tensor(negatives_np, dtype=torch.long, device=v3.DEVICE)
            batch_weights = weight_tensor[
                torch.as_tensor(index, dtype=torch.long, device=v3.DEVICE)
            ]
            user_z, item_z = model.propagated_embeddings()
            positive_scores = (user_z[users] * item_z[positives]).sum(dim=1)
            negative_scores = (user_z[users, None, :] * item_z[negatives]).sum(dim=2)
            per_row = torch.nn.functional.softplus(
                negative_scores - positive_scores[:, None]
            ).mean(dim=1)
            bpr = (batch_weights * per_row).mean()
            loss = bpr + model.sampled_l2(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["bpr"] += float(bpr.detach())
            totals["p_correct"] += float(
                (negative_scores < positive_scores[:, None]).float().mean()
            )
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=totals["loss"] / (batch + 1),
            )
        record = {
            "epoch": int(epoch),
            **{key: float(value / n_batches) for key, value in totals.items()},
            "epoch_sec": float(time.time() - epoch_started),
        }
        history.append(record)
        print(
            f"  [{spec['model_id']}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
        store.save_epoch(model, optimizer, rng, {"epoch": epoch, "history": history})
    return {
        "epochs_run": int(last_epoch),
        "wall_clock_sec": round(time.time() - started, 1),
        "history": history,
        "final_diagnostics": history[-1] if history else {},
    }


def _run_arm(prepared: dict, cfg: M5K1ImprovementConfig, spec: dict) -> tuple[dict, object]:
    paths = _arm_paths(prepared, spec, cfg)
    model = screen._build_model(prepared, cfg, spec | {"m2_assignment": prepared["m2_actual"]})
    weights, weight_diagnostics = row_weights(prepared, cfg, spec["improvement"])
    checkpoint = (
        legacy.load_checkpoint_or_discard(paths["checkpoint"])
        if paths["result"].exists() and paths["checkpoint"].exists()
        else None
    )
    if checkpoint is not None:
        print(f"  [cached] {spec['model_id']} 완료 결과 재사용")
        payload = json.loads(paths["result"].read_text(encoding="utf-8"))
        if checkpoint.get("input_hash") != prepared["input_hash"]:
            raise RuntimeError("cached checkpoint와 현재 입력 hash가 다릅니다")
        model.load_state_dict(checkpoint["state"], strict=True)
        model.to(v3.DEVICE)
        model.eval()
        return payload, model

    store = ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="historical_development_train",
            model_id=spec["model_id"],
            seed=cfg.seed,
            config_hash=prepared["config_hash"],
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )
    training = _train_arm(model, prepared, cfg, spec, weights, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": spec["model_id"],
            "config": asdict(cfg),
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])
    metrics_raw, _ = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=False,
    )
    payload = {
        "model_id": spec["model_id"],
        "role": spec["role"],
        "seed": cfg.seed,
        "split": "historical_development_days_684_690",
        "final_epoch": cfg.epochs,
        "rho": spec["rho"],
        "improvement": spec["improvement"],
        "weight_diagnostics": weight_diagnostics,
        "diagnostics": model.representation_diagnostics(),
        "metrics": test10._public_metrics(metrics_raw),
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


def improvement_reading(
    metric_rows: dict[str, dict], *, top10_change_shares: dict[str, float]
) -> dict:
    m1, m2 = metric_rows[M1_MODEL_ID], metric_rows[M2_MODEL_ID]

    def geomean_ratio(model: dict, reference: dict) -> float:
        ratios = [model[metric] / reference[metric] for metric in ACCURACY_METRICS]
        return float(math.exp(np.log(ratios).mean()))

    readings = {}
    for improvement, (m4_id, m5_id) in IMPROVEMENTS.items():
        m4, m5 = metric_rows[m4_id], metric_rows[m5_id]
        evaluable = top10_change_shares[m5_id] > 0.0
        beats_both = all(
            m5[metric] > max(m2[metric], m4[metric]) for metric in ECONOMIC_METRICS
        )
        accuracy_geomean = geomean_ratio(m5, m4) >= 1.0
        accuracy_guard = all(
            m5[metric] >= ACCURACY_GUARD * m4[metric] for metric in ACCURACY_METRICS
        )
        baseline = all(m5[metric] > m1[metric] for metric in ECONOMIC_METRICS)
        passed = bool(
            evaluable and beats_both and accuracy_geomean and accuracy_guard and baseline
        )
        reported = TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
        readings[improvement] = {
            "improvement_pass": passed,
            "evaluable": bool(evaluable),
            "m5_beats_both_parts": bool(beats_both),
            "m5_accuracy_geomean_at_least_m4": bool(accuracy_geomean),
            "m5_accuracy_guard_vs_m4": bool(accuracy_guard),
            "m5_beats_m1_economics": bool(baseline),
            "top10_set_changed_user_share_vs_m4": float(top10_change_shares[m5_id]),
            "accuracy_geomean_ratio_m5_vs_m4": geomean_ratio(m5, m4),
            "deltas_m4_minus_m1": {
                metric: float(m4[metric] - m1[metric]) for metric in reported
            },
            "deltas_m5_minus_m4": {
                metric: float(m5[metric] - m4[metric]) for metric in reported
            },
            "deltas_m5_minus_m2": {
                metric: float(m5[metric] - m2[metric]) for metric in reported
            },
            "interaction_m5_minus_m4_minus_m2_minus_m1": {
                metric: float((m5[metric] - m4[metric]) - (m2[metric] - m1[metric]))
                for metric in reported
            },
        }
    passing = [name for name, reading in readings.items() if reading["improvement_pass"]]
    return {
        "classification": (
            "no_improvement_passes"
            if not passing
            else ("both_improvements_pass" if len(passing) == 2 else f"{passing[0]}_passes")
        ),
        "passing_improvements": passing,
        "improvements": readings,
        "deltas_m2_minus_m1": {
            metric: float(m2[metric] - m1[metric])
            for metric in TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
        },
        "clv_attribution_tested": False,
        "next_if_pass": (
            "run the degree-matched CLV assignment shuffle for the passing arm; "
            "the earlier M2 attribution does not transfer to a different loss"
        ),
        "next_if_nonpass": (
            "report that the value basis and the loss weight do not combine, and "
            "propose M2 alone as the CLV-attributed component"
        ),
        "statistical_note": "one historical development seed; no significance or generalization claim",
    }


def run_improvement_screen(cfg: M5K1ImprovementConfig | None = None) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_improvement_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | K={cfg.negative_count} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        arm, model = _run_arm(prepared, cfg, spec)
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model

    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in MODEL_IDS}
    frame = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "role": arms[model_id]["role"],
                "rho": arms[model_id]["rho"],
                "improvement": arms[model_id]["improvement"],
                **arms[model_id]["weight_diagnostics"],
                **arms[model_id]["diagnostics"],
                **arms[model_id]["metrics"],
            }
            for model_id in MODEL_IDS
        ]
    )
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M1_MODEL_ID, M4_FIRST_MODEL_ID, M4_COMPLEMENT_MODEL_ID)
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in MODEL_IDS
    }
    segments = prepared["cache"].seg
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[reference][1], topk[model_id][1], segments
            ).assign(reference=reference, model_id=model_id)
            for reference, model_id in (
                (M1_MODEL_ID, M2_MODEL_ID),
                (M4_FIRST_MODEL_ID, M5_FIRST_MODEL_ID),
                (M4_COMPLEMENT_MODEL_ID, M5_COMPLEMENT_MODEL_ID),
            )
        ],
        ignore_index=True,
    )
    overall = overlap[overlap.group.eq("전체")].set_index("model_id")
    change_shares = {
        model_id: float(overall.at[model_id, "top10_set_changed_user_share"])
        for model_id in (M2_MODEL_ID, M5_FIRST_MODEL_ID, M5_COMPLEMENT_MODEL_ID)
    }
    reading = improvement_reading(metric_rows, top10_change_shares=change_shares)

    out = Path(cfg.out_dir)
    stem = f"m5_k1_m4_improvement_{prepared['config_hash']}"
    paths = {
        "absolute_csv": out / f"{stem}.csv",
        "comparison_csv": out / f"{stem}_comparison.csv",
        "top10_overlap_csv": out / f"{stem}_top10_overlap.csv",
        "json": out / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], frame)
    test10._atomic_csv(paths["comparison_csv"], comparison)
    test10._atomic_csv(paths["top10_overlap_csv"], overlap)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "config": asdict(cfg),
            "preflight": summary,
            "input_manifest": prepared["manifest"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "screening_reading": reading,
            "arms": arms,
            "result_paths": {key: str(value) for key, value in paths.items()},
        },
    )
    frame.attrs.update(
        comparison=comparison,
        top10_overlap=overlap,
        decision=reading,
        result_paths={key: str(value) for key, value in paths.items()},
    )
    print("\n1) 판독")
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    print("\n결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_improvement_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
