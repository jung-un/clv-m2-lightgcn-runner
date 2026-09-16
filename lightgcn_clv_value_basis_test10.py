"""Final Dunnhumby test-only 10-seed runner for the CLV-scaled value-basis M2.

The professor's protocol for the final report: no validation split and no
selection of any kind, train every arm for exactly 100 epochs, evaluate the
protected test split once per arm, and report the mean over 10 seeds.

Three arms run here.  M4 is deliberately absent: the 2026-09-16 development
screen showed the current positive weight and the value basis cancel each
other, the weight is about to be redesigned, and a test number for a weight we
are replacing would be discarded work.

* ``m1_bpr_k1``                 - the original LightGCN with one uniform unseen
  negative per positive row.
* ``m2_clv_scaled_value_basis_bpr_k1``  - the same model with rho=0.25 of a
  fixed q_V/price RBF basis whose user block is scaled by q_C.
* ``..._degree_matched_shuffle`` - the same M2 with q_N/q_V/q_C/valid jointly
  permuted inside binary user-degree deciles.  Without this arm an improvement
  cannot be attributed to the customer's own CLV.

Cache key, deliberately different from ``lightgcn_clv_axis_specific_test10``:
the run hash covers the training-relevant configuration, the input manifest and
the sha256 of the model definition module -- not the arm list and not the git
revision.  A later run that appends a redesigned M4 arm therefore reuses the
completed M1 and M2 seeds instead of retraining them, while any edit to the
model definition still invalidates the cache.  ``source_revision`` stays in
every result payload for provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity, clone_state, file_sha256
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as evaluation
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m4_clv_hard_negative as m4_helpers
import lightgcn_clv_m5_economic_positive_weight as legacy
import lightgcn_clv_m5_n_conditioned_value_basis_controls as controls
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-value-basis-test10-v1"
MODEL_DEFINITION_FILE = "clv_m5_n_conditioned_value_basis_model.py"
SEEDS = tuple(range(42, 52))

M1_MODEL_ID = "m1_bpr_k1"
M2_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1"
M2_SHUFFLE_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1_degree_matched_shuffle"
MODEL_IDS = (M1_MODEL_ID, M2_MODEL_ID, M2_SHUFFLE_MODEL_ID)

ECONOMIC_METRIC = "price_purchase_amount_weighted_hit@10"
ACCURACY_METRIC = "recall@10"
SECONDARY_ECONOMIC_METRIC = "vndcg@10"


@dataclass(frozen=True)
class ValueBasisTest10Config:
    dataset: str = "dunnhumby"
    seeds: tuple[int, ...] = SEEDS
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    rho: float = 0.25
    gate_delta: float = 0.25
    basis_bandwidth: float = 0.25
    negative_count: int = 1
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    economic_bins: int = 4
    shrinkage_strength: float = 10.0
    shuffle_seed: int = 42
    shuffle_degree_bins: int = 10
    out_dir: str = ""


#: Fields that change what M1 and M2 learn.  The run hash covers exactly these,
#: so appending a later arm keeps the completed seeds of these two models.
PROTOCOL_FIELDS = (
    "dataset",
    "seeds",
    "epochs",
    "id_dim",
    "n_layers",
    "rho",
    "gate_delta",
    "basis_bandwidth",
    "negative_count",
    "batch_size",
    "lr",
    "pref_reg",
    "input_days",
    "economic_bins",
    "shrinkage_strength",
    "shuffle_seed",
    "shuffle_degree_bins",
)


def configure_value_basis_test10(**overrides) -> ValueBasisTest10Config:
    defaults = {
        "out_dir": f"{v3.default_out_dir('dunnhumby')}_m2_value_basis_test10_v1"
    }
    return validate_config(ValueBasisTest10Config(**(defaults | overrides)))


def validate_config(cfg: ValueBasisTest10Config) -> ValueBasisTest10Config:
    """Fail closed if the final protocol or the accepted M2 design is altered."""

    required = {
        "dataset": "dunnhumby",
        "seeds": SEEDS,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "rho": 0.25,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
        "negative_count": 1,
        "input_days": 365,
        "economic_bins": 4,
        "shrinkage_strength": 10.0,
        "shuffle_seed": 42,
        "shuffle_degree_bins": 10,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"test-only 확증 설정은 {key}={expected!r}이어야 합니다")
    if cfg.batch_size <= 0 or cfg.lr <= 0 or cfg.pref_reg < 0:
        raise ValueError("학습 설정이 잘못됐습니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def arm_specifications() -> list[dict]:
    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "baseline",
            "value_basis": False,
            "assignment": "inactive",
        },
        {
            "model_id": M2_MODEL_ID,
            "role": "model",
            "value_basis": True,
            "assignment": "observed_clv",
        },
        {
            "model_id": M2_SHUFFLE_MODEL_ID,
            "role": "assignment_control",
            "value_basis": True,
            "assignment": "degree_matched_clv_shuffle",
        },
    ]


def preflight_summary(cfg: ValueBasisTest10Config) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seeds": list(cfg.seeds),
        "split": "protected_test_last_7_days",
        "validation_selection": False,
        "test_evaluations_per_arm": 1,
        "trained_models": list(MODEL_IDS),
        "m4_present": False,
        "m4_note": (
            "the positive weight is being redesigned; testing the current one "
            "would be discarded work"
        ),
        "research_question": (
            "On the protected test split and averaged over 10 seeds, does the "
            "q_C-scaled q_V price basis raise the price/purchase-amount "
            "weighted hit@10 over M1 without losing accuracy, and does that "
            "gain need the customer's own CLV assignment?"
        ),
        "loss": {
            "bpr": "mean softplus(s(u,j) - s(u,i+)) + batch layer-0 L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "uniform over unseen items",
            "hard_negative": False,
            "row_weighting": False,
        },
        "m2": {
            "user_block": "sqrt(rho) * q_C * g_N(q_N) * RBF(q_V)",
            "item_block": "sqrt(rho) * RBF(item amount percentile)",
            "rho": cfg.rho,
            "basis": "fixed normalized Gaussian RBF at 0.0, 0.5, 1.0",
            "basis_bandwidth": cfg.basis_bandwidth,
            "gate_delta": cfg.gate_delta,
            "learned_parameters_in_value_block": 2,
            "economic_propagation": False,
        },
        "decision_rule": {
            "economic_gain": (
                f"95% CI of the paired M2-M1 {ECONOMIC_METRIC} difference lies "
                "entirely above zero"
            ),
            "accuracy_guard": (
                f"95% CI of the paired M2-M1 {ACCURACY_METRIC} difference does "
                "not lie entirely below zero"
            ),
            "clv_attribution": (
                f"95% CI of the paired M2-shuffle {ECONOMIC_METRIC} difference "
                "lies entirely above zero"
            ),
            "reported_not_gating": [SECONDARY_ECONOMIC_METRIC, "all other metrics"],
        },
        "next_if_nonpass": (
            "report that the value basis did not reproduce on the protected "
            "test split and do not tune it on these numbers"
        ),
    }


def _base_config(cfg: ValueBasisTest10Config) -> dict:
    """Reuse the accepted final-protocol base configuration."""

    return test10._base_config(cfg)


def _config_hash(cfg: ValueBasisTest10Config, input_hash: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "protocol": {field: getattr(cfg, field) for field in PROTOCOL_FIELDS},
        "model_definition_sha256": file_sha256(
            Path(__file__).with_name(MODEL_DEFINITION_FILE)
        ),
        "input_hash": input_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: ValueBasisTest10Config) -> dict:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash = moe.manifest_hash(manifest)
    revision = moe.source_revision()
    base_cfg = _base_config(cfg)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    if set(data["splits"]) != {"test"}:
        raise RuntimeError(
            f"test-only runner에 보호 split 오염: {sorted(data['splits'])}"
        )
    if data.get("loss_w") is not None:
        raise RuntimeError("M2 test에 M4 표본 가중치가 섞였습니다")
    data["loss_w"] = None

    snapshot = residual.build_final_snapshot(
        data["train"], data["n_users"], v3.DCFG["is_date"], cfg.input_days
    )
    axes = joint.build_user_axis_inputs(snapshot, data["n_users"])
    q_n, q_v, q_c, clv_valid = evaluation.build_clv_inputs(axes)
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
    }
    prepared.update(
        legacy.build_economic_inputs(
            data["train"],
            n_users=data["n_users"],
            n_items=data["n_items"],
            q_v=q_v,
            q_c=q_c,
            clv_valid=clv_valid,
            n_bins=cfg.economic_bins,
            shrinkage_strength=cfg.shrinkage_strength,
            degree_bins=cfg.shuffle_degree_bins,
        )
    )
    prepared["q_n"] = q_n
    prepared["m2_actual"] = {
        "q_n": np.asarray(prepared["q_n"]).copy(),
        "q_v": np.asarray(prepared["q_v"]).copy(),
        "q_c": np.asarray(prepared["q_c"]).copy(),
        "clv_valid": np.asarray(prepared["clv_valid"]).copy(),
    }
    prepared["m2_shuffle"] = controls.degree_matched_nv_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    prepared["control_diagnostics"] = _shuffle_diagnostics(prepared)
    prepared["meta"] = v3.item_meta(data["train"], data["n_items"])
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base_cfg["SEG_EDGES"])
    prepared["cache"] = v3.EvalCache(
        *data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]
    )
    prepared["config_hash"] = _config_hash(cfg, input_hash)
    return prepared


def _shuffle_diagnostics(prepared: dict) -> dict:
    """Fail closed unless the permutation only moved users inside a stratum."""

    source = np.asarray(prepared["m2_shuffle"]["source_user"])
    diagnostics = {
        "n_users": int(len(source)),
        "valid_users": int(np.asarray(prepared["clv_valid"]).sum()),
        "moved_user_share": float(np.mean(source != np.arange(len(source)))),
        "same_degree_bin": bool(
            np.all(
                np.asarray(prepared["degree_bin"])[source]
                == np.asarray(prepared["degree_bin"])
            )
        ),
    }
    for key in ("q_n", "q_v", "q_c", "clv_valid"):
        diagnostics[f"{key}_multiset_preserved"] = bool(
            np.array_equal(
                np.sort(np.asarray(prepared["m2_shuffle"][key])),
                np.sort(np.asarray(prepared["m2_actual"][key])),
            )
        )
    broken = [
        key
        for key, value in diagnostics.items()
        if key.endswith(("preserved", "bin")) and not value
    ]
    if broken:
        raise RuntimeError(f"CLV 순열 불변식이 성립하지 않습니다: {broken}")
    return diagnostics


def _build_model(
    prepared: dict, cfg: ValueBasisTest10Config, spec: dict, seed: int
) -> M5NConditionedValueBasisLightGCN:
    data = prepared["data"]
    assignment = (
        prepared["m2_shuffle"]
        if spec["assignment"] == "degree_matched_clv_shuffle"
        else prepared["m2_actual"]
    )
    rho = cfg.rho if spec["value_basis"] else 0.0
    v3.set_seed(seed)
    return M5NConditionedValueBasisLightGCN(
        n_users=data["n_users"],
        n_items=data["n_items"],
        user_q_n=assignment["q_n"],
        user_q_v=assignment["q_v"],
        user_q_c=assignment["q_c"],
        user_clv_valid=assignment["clv_valid"],
        item_price_percentile=prepared["item_amount_percentile"],
        item_price_valid=prepared["item_economic_valid"],
        adj=data["adj"],
        id_dim=cfg.id_dim,
        rho=rho,
        n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg,
        gate_delta=cfg.gate_delta,
        basis_bandwidth=cfg.basis_bandwidth,
        economic_propagation=False,
    ).to(v3.DEVICE)


def _arm_hash(
    prepared: dict, cfg: ValueBasisTest10Config, spec: dict, seed: int
) -> str:
    payload = {
        "run": prepared["config_hash"],
        "model_id": spec["model_id"],
        "assignment": spec["assignment"],
        "value_basis": spec["value_basis"],
        "seed": seed,
        "epochs": cfg.epochs,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _progress_store(
    prepared: dict, cfg: ValueBasisTest10Config, spec: dict, seed: int
) -> ProgressStore:
    return ProgressStore(
        prepared["out_dir"] / "progress" / prepared["config_hash"],
        RunIdentity(
            stage="final_train_test",
            model_id=spec["model_id"],
            seed=seed,
            config_hash=_arm_hash(prepared, cfg, spec, seed),
            source_revision=prepared["revision"],
            input_hash=prepared["input_hash"],
        ),
    )


def _arm_paths(prepared: dict, model_id: str, seed: int) -> dict[str, Path]:
    root = prepared["out_dir"] / "arms" / prepared["config_hash"]
    stem = f"{model_id}_s{seed}"
    return {
        "result": root / f"{stem}.json",
        "per_user": root / f"{stem}_per_user.npz",
        "checkpoint": root / f"{stem}.pt",
    }


def _train_arm(
    model,
    prepared: dict,
    cfg: ValueBasisTest10Config,
    model_id: str,
    seed: int,
    store: ProgressStore,
) -> dict:
    """Train exactly ``cfg.epochs`` epochs of single-negative BPR."""

    data = prepared["data"]
    tr_u, tr_i, positive_keys = data["tr_u"], data["tr_i"], data["pos_key"]
    n_train = len(tr_u)
    n_batches = math.ceil(n_train / cfg.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(seed)

    restored = store.restore_epoch(model, optimizer, rng)
    start_epoch = 1 if restored is None else int(restored["next_epoch"])
    history = list(restored.get("history", [])) if restored else []
    previous_wall = float(restored.get("wall_clock_sec", 0.0)) if restored else 0.0
    if restored is not None:
        print(f"  [{model_id} s{seed}] epoch {start_epoch - 1}에서 자동 재개")
    store.mark_stage(
        "running", epoch=start_epoch - 1, max_epoch=cfg.epochs, selection="none"
    )

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
            positives = torch.as_tensor(
                positives_np, dtype=torch.long, device=v3.DEVICE
            )
            negatives = torch.as_tensor(
                negatives_np, dtype=torch.long, device=v3.DEVICE
            )
            user_z, item_z = model.propagated_embeddings()
            positive_scores = (user_z[users] * item_z[positives]).sum(dim=1)
            negative_scores = (user_z[users, None, :] * item_z[negatives]).sum(dim=2)
            bpr = torch.nn.functional.softplus(
                negative_scores - positive_scores[:, None]
            ).mean()
            loss = bpr + model.sampled_l2(users, positives, negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss)
            totals["bpr"] += float(bpr)
            totals["p_correct"] += float(
                (positive_scores[:, None] > negative_scores).float().mean()
            )
            store.heartbeat(
                epoch=epoch,
                max_epoch=cfg.epochs,
                batch=batch + 1,
                batches=n_batches,
                loss=totals["loss"] / (batch + 1),
                selection="none",
            )
        record = {
            "epoch": epoch,
            "loss": totals["loss"] / n_batches,
            "bpr": totals["bpr"] / n_batches,
            "p_correct": totals["p_correct"] / n_batches,
            "epoch_sec": time.time() - epoch_started,
        }
        history.append(record)
        store.save_epoch(
            model,
            optimizer,
            rng,
            epoch=epoch,
            history=history,
            wall_clock_sec=previous_wall + time.time() - started,
            selection="none",
        )
        print(
            f"  [{model_id} s{seed}] ep {epoch:3d}/{cfg.epochs} | "
            f"loss {record['loss']:.4f} | P(pos>neg) {record['p_correct']:.3f} | "
            f"{record['epoch_sec']:.0f}s"
        )
    if last_epoch != cfg.epochs:
        raise RuntimeError(f"고정 {cfg.epochs} epoch 미완료: {last_epoch}")
    return {
        "epochs_run": cfg.epochs,
        "selection": "none",
        "early_stopping": False,
        "wall_clock_sec": previous_wall + time.time() - started,
        "resumed_from_epoch": start_epoch - 1,
        "history": history,
    }


def _run_arm(
    prepared: dict, cfg: ValueBasisTest10Config, spec: dict, seed: int
) -> dict:
    paths = _arm_paths(prepared, spec["model_id"], seed)
    cached = test10._load_cached_arm(paths)
    if cached is not None:
        return cached

    model = _build_model(prepared, cfg, spec, seed)
    store = _progress_store(prepared, cfg, spec, seed)
    training = _train_arm(model, prepared, cfg, spec["model_id"], seed, store)
    model.eval()
    paths["checkpoint"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["checkpoint"].with_suffix(".pt.tmp")
    torch.save(
        {
            "state": clone_state(model),
            "model_id": spec["model_id"],
            "seed": seed,
            "training": training,
            "config": asdict(cfg),
            "source_revision": prepared["revision"],
            "input_hash": prepared["input_hash"],
        },
        temporary,
    )
    os.replace(temporary, paths["checkpoint"])

    # The only protected-split evaluation call in this arm.  Once the atomic
    # result and per-user files exist, reconnects take the cached path above.
    metrics, per_user = moe._flat_evaluation(
        model,
        0.0,
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    public_metrics = test10._public_metrics(metrics)
    public_per_user = test10._public_per_user(per_user)
    test10._atomic_npz(paths["per_user"], public_per_user)
    payload = {
        "model_id": spec["model_id"],
        "role": spec["role"],
        "assignment": spec["assignment"],
        "seed": seed,
        "split": "test",
        "final_epoch": cfg.epochs,
        "validation_selection": False,
        "test_evaluation_count": 1,
        "test_evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": public_metrics,
        "diagnostics": model.representation_diagnostics(),
        "training": training,
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "per_user_path": str(paths["per_user"]),
    }
    test10._atomic_json(paths["result"], payload)
    payload["per_user"] = public_per_user
    store.mark_complete(
        epoch=cfg.epochs,
        max_epoch=cfg.epochs,
        selection="none",
        split="test",
        test_evaluation_count=1,
        checkpoint_path=str(paths["checkpoint"]),
        result_path=str(paths["result"]),
    )
    return payload


def _absolute_rows(arms: list[dict]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        row = {
            "model_id": arm["model_id"],
            "role": arm["role"],
            "assignment": arm["assignment"],
            "seed": arm["seed"],
        }
        row.update(arm["metrics"])
        diagnostics = arm.get("diagnostics", {})
        for key in ("n_gate_slope", "n_gate_offset", "value_strength_mean"):
            row[key] = diagnostics.get(key)
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_columns(absolute: pd.DataFrame) -> list[str]:
    return [
        column
        for column in absolute.columns
        if "@" in column
        or column == "user_value_tendency_recommended_price_alignment"
    ]


def paired_tables(
    absolute: pd.DataFrame, arms: list[dict], pairs: list[tuple[str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same-seed differences for every (model, reference) pair."""

    metrics = _metric_columns(absolute)
    arm_map = {(arm["seed"], arm["model_id"]): arm for arm in arms}
    seeds = sorted({arm["seed"] for arm in arms})
    rows = []
    for model_id, reference in pairs:
        for seed in seeds:
            compared, base = arm_map[(seed, model_id)], arm_map[(seed, reference)]
            for metric in metrics:
                rows.append(
                    {
                        "seed": seed,
                        "model_id": model_id,
                        "reference": reference,
                        "metric": metric,
                        "delta": float(
                            compared["metrics"][metric] - base["metrics"][metric]
                        ),
                    }
                )
    paired_seed = pd.DataFrame(rows)
    summary = []
    for (model_id, reference, metric), group in paired_seed.groupby(
        ["model_id", "reference", "metric"], sort=False
    ):
        summary.append(
            {
                "model_id": model_id,
                "reference": reference,
                "metric": metric,
                **test10._mean_ci(group["delta"].to_numpy()),
                "positive_seed_count": int((group["delta"] > 0).sum()),
            }
        )
    return paired_seed, pd.DataFrame(summary)


def _interval(summary: pd.DataFrame, model_id: str, reference: str, metric: str) -> dict:
    match = summary[
        summary.model_id.eq(model_id)
        & summary.reference.eq(reference)
        & summary.metric.eq(metric)
    ]
    if len(match) != 1:
        raise KeyError(f"{model_id} vs {reference}의 {metric} 요약이 없습니다")
    return match.iloc[0].to_dict()


def test10_reading(paired_summary: pd.DataFrame) -> dict:
    """Apply the pre-registered decision rule to the 10-seed intervals."""

    economic = _interval(paired_summary, M2_MODEL_ID, M1_MODEL_ID, ECONOMIC_METRIC)
    accuracy = _interval(paired_summary, M2_MODEL_ID, M1_MODEL_ID, ACCURACY_METRIC)
    attribution = _interval(
        paired_summary, M2_MODEL_ID, M2_SHUFFLE_MODEL_ID, ECONOMIC_METRIC
    )
    secondary = _interval(
        paired_summary, M2_MODEL_ID, M1_MODEL_ID, SECONDARY_ECONOMIC_METRIC
    )
    economic_gain = bool(economic["lo"] > 0.0)
    accuracy_guard = bool(accuracy["hi"] >= 0.0)
    clv_attribution = bool(attribution["lo"] > 0.0)
    passed = economic_gain and accuracy_guard and clv_attribution
    return {
        "economic_gain_over_m1": economic_gain,
        "accuracy_not_significantly_lower": accuracy_guard,
        "clv_attribution_over_shuffle": clv_attribution,
        "classification": (
            "value_basis_confirmed" if passed else "value_basis_not_confirmed"
        ),
        "intervals": {
            "m2_minus_m1_economic": economic,
            "m2_minus_m1_accuracy": accuracy,
            "m2_minus_shuffle_economic": attribution,
            "m2_minus_m1_secondary_economic_reported_only": secondary,
        },
        "selection_on_test": False,
    }


def _persist(
    prepared: dict, cfg: ValueBasisTest10Config, arms: list[dict]
) -> pd.DataFrame:
    absolute = _absolute_rows(arms)
    absolute_summary = []
    for model_id, group in absolute.groupby("model_id", sort=False):
        for metric in _metric_columns(absolute):
            absolute_summary.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    **test10._mean_ci(group[metric].to_numpy()),
                }
            )
    absolute_summary = pd.DataFrame(absolute_summary)
    paired_seed, paired_summary = paired_tables(
        absolute,
        arms,
        [
            (M2_MODEL_ID, M1_MODEL_ID),
            (M2_SHUFFLE_MODEL_ID, M1_MODEL_ID),
            (M2_MODEL_ID, M2_SHUFFLE_MODEL_ID),
        ],
    )
    reading = test10_reading(paired_summary)

    stem = f"m2_value_basis_test10_{prepared['config_hash']}"
    paths = {
        "absolute_csv": prepared["out_dir"] / f"{stem}.csv",
        "absolute_summary_csv": prepared["out_dir"] / f"{stem}_mean.csv",
        "paired_seed_csv": prepared["out_dir"] / f"{stem}_paired_seed.csv",
        "paired_summary_csv": prepared["out_dir"] / f"{stem}_paired_mean.csv",
        "json": prepared["out_dir"] / f"{stem}.json",
    }
    test10._atomic_csv(paths["absolute_csv"], absolute)
    test10._atomic_csv(paths["absolute_summary_csv"], absolute_summary)
    test10._atomic_csv(paths["paired_seed_csv"], paired_seed)
    test10._atomic_csv(paths["paired_summary_csv"], paired_summary)
    test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "source_revision": prepared["revision"],
            "input_manifest": prepared["manifest"],
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "control_diagnostics": prepared["control_diagnostics"],
            "data_stats": prepared["data"].get("data_stats", {}),
            "absolute_rows": absolute.to_dict("records"),
            "absolute_10seed_summary": absolute_summary.to_dict("records"),
            "same_seed_differences": paired_seed.to_dict("records"),
            "same_seed_10seed_summary": paired_summary.to_dict("records"),
            "reading": reading,
            "result_paths": {name: str(path) for name, path in paths.items()},
            "interpretation": {
                "selection": "none; test was not used for model or epoch selection",
                "weighted_hit": (
                    "price/purchase-amount weighted recommendation hit; not "
                    "actual incremental revenue"
                ),
                "significance": (
                    "the reported interval summarizes variation across the 10 "
                    "paired seeds"
                ),
            },
        },
    )
    absolute.attrs["absolute_summary"] = absolute_summary
    absolute.attrs["paired_seed"] = paired_seed
    absolute.attrs["paired_summary"] = paired_summary
    absolute.attrs["reading"] = reading
    absolute.attrs["result_paths"] = {
        name: str(path) for name, path in paths.items()
    }
    return absolute


def run_value_basis_test10(
    cfg: ValueBasisTest10Config | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_value_basis_test10())
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("순열 대조 점검:", json.dumps(prepared["control_diagnostics"], default=str))
    arms = []
    for seed in cfg.seeds:
        for spec in arm_specifications():
            print(f"\n===== seed {seed} | {spec['model_id']} | K=1 | 100 epoch =====")
            arms.append(_run_arm(prepared, cfg, spec, seed))
    frame = _persist(prepared, cfg, arms)
    print("\n10시드 test 절대지표 평균:")
    print(frame.attrs["absolute_summary"].to_string(index=False))
    print("\n동일 seed 차이의 10시드 평균과 95% 신뢰구간:")
    print(frame.attrs["paired_summary"].to_string(index=False))
    print("\n사전 판정:", json.dumps(frame.attrs["reading"], ensure_ascii=False, indent=2))
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


if __name__ == "__main__":
    print(
        json.dumps(
            preflight_summary(configure_value_basis_test10()),
            ensure_ascii=False,
            indent=2,
        )
    )
