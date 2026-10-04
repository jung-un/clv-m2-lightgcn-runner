"""H&M M4 reference arm at 300 epochs: binary graph + the confirmed M4 row weights.

Why a new file.  The confirmed H&M M4 module fixes ``epochs`` at 100 in
``validate_config`` and folds ``epochs`` into its config hash, so it can neither
be relaxed nor resumed at 300 without editing the module the protected-test
confirmation stands on.  That module is left untouched; this runner trains the
same loss on the same binary graph in its own folder.

Why this loop.  Candidates A/B were trained by ``budget._train_curve``, so the
reference uses that same loop with a uniform-weight graph.  ``A - reference`` is
then the M3 graph alone, measured at one epoch with everything else equal.

Staging.  Run ``train(cfg, prepared, stop_epoch=100)`` first (~12.5h), read
``reproduction_gap`` against the frozen M4@100, and only then call the same
function with 300 — it resumes from the saved epoch.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m5_m3_m4_split_nv_hm2y_screen as m5
import lightgcn_clv_hm2y_seed42_common as common
import lightgcn_clv_m2_training_budget_hm2y as budget
import lightgcn_clv_m3_centered_value_graph_hm2y as hm2y
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m5-m3-m4-split-nv-hm2y-m4-reference-300-v1"
MODEL_ID = "m4_personalized_positive_weight_actual_qc_bpr_k1_hm2y_binary_graph"
VARIANT = "original"
JUDGE_EPOCH = 300
# The frozen H&M M4 ran 100 epochs on a different module; the same loop must
# land on the same numbers there or the 300 curve is not its extension.
REPRODUCTION_RELATIVE_TOLERANCE = 1e-4


def configure(**overrides):
    root = v3.default_out_dir("hm")
    defaults = {"out_dir": f"{root}_clv_m5_m3_m4_split_nv_hm2y_s43_m4_reference_300_v1"}
    cfg = m5.configure(**(defaults | overrides))
    if cfg.epochs != 300 or JUDGE_EPOCH not in cfg.evaluation_epochs:
        raise ValueError("300 epoch 격자만 허용합니다")
    return cfg


def candidate_out_dir() -> Path:
    """Where the A/B arms wrote their curves (a separate run, read only)."""
    return Path(m5.configure().out_dir)


def binary_edge_weights(prepared: dict) -> np.ndarray:
    """Uniform edge weights, checked to rebuild M1's own binary adjacency."""
    ones = np.ones(len(prepared["signals"]["edge_users"]), dtype=np.float32)
    data = prepared["data"]
    rebuilt = v3.build_adj(prepared["signals"]["edge_users"],
                           prepared["signals"]["edge_items"], ones,
                           data["n_users"], data["n_items"])
    if not same_adjacency(rebuilt, data["adj"]):
        raise RuntimeError("균등가중 그래프가 M1 이진 인접행렬과 다릅니다 — 학습하지 않습니다")
    return ones


def same_adjacency(left, right) -> bool:
    left, right = left.coalesce(), right.coalesce()
    return bool(torch.equal(left.indices(), right.indices())
                and torch.allclose(left.values(), right.values(),
                                   rtol=1e-6, atol=0.0))


def prepare(cfg) -> dict:
    """Reuse the candidates' own preparation: same data, same audited weights."""
    prepared = m5.prepare(cfg)
    prepared["binary_weights"] = binary_edge_weights(prepared)
    return prepared


def _arm_dir(prepared: dict) -> Path:
    root = Path(prepared["out_dir"]) / "arms" / prepared["run_hash"]
    root.mkdir(parents=True, exist_ok=True)
    return root


def train(cfg, prepared: dict, stop_epoch: int = JUDGE_EPOCH) -> dict:
    if stop_epoch not in cfg.evaluation_epochs:
        raise ValueError(f"stop_epoch {stop_epoch}")
    path = _arm_dir(prepared) / f"{MODEL_ID}_to{stop_epoch}.json"
    if path.exists():
        print(f"  [cached] {MODEL_ID} → epoch {stop_epoch} 결과 재사용", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))
    spec = {"model_id": MODEL_ID, "arm": "binary_graph", "gamma": 0.0, "kind": "m4",
            "question": "H&M original M4 loss on the binary graph, 300 epochs"}
    weights = prepared["m5_weights"][VARIANT]
    arm_hash = hashlib.sha256(
        f"{prepared['run_hash']}:{CODE_VERSION}:"
        f"{prepared['m5_weight_audit'][VARIANT]['sha256']}".encode()).hexdigest()[:12]
    model = hm2y._build_model(prepared, cfg, {"weights": prepared["binary_weights"]})
    store = common.progress_store(prepared, cfg, MODEL_ID, arm_hash)
    curve = budget._train_curve(model, prepared, cfg, spec, store,
                                row_weights=weights, stop_epoch=stop_epoch)
    payload = {**spec, "variant": VARIANT, "seed": cfg.seed, "stop_epoch": stop_epoch,
               "split": hm2y.SPLIT, "weight_audit": prepared["m5_weight_audit"][VARIANT],
               "code_version": CODE_VERSION, "source_revision": prepared["revision"],
               "evaluated_at": datetime.now(timezone.utc).isoformat(), "curve": curve}
    common.atomic_json(path, payload)
    if stop_epoch == cfg.epochs:
        store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                            checkpoint_path="", result_path=str(path))
    return payload


def reproduction_gap(payload: dict, prepared: dict) -> pd.DataFrame:
    """Relative gap at epoch 100 against the frozen H&M M4@100, per metric."""
    frozen = prepared["m5_reference_curve"]
    frozen = frozen[frozen.model_id.eq(m5.M4_ID)]
    if len(frozen) != 1:
        raise RuntimeError("동결 M4@100 기준 행을 찾지 못했습니다")
    frozen = frozen.iloc[0]
    at_100 = next((r for r in payload["curve"]
                   if r["epoch"] == m5.JUDGE_EPOCH and "metrics" in r), None)
    if at_100 is None:
        raise RuntimeError("epoch 100 평가가 없습니다")
    rows = []
    for metric in list(m5.ACCURACY) + list(m5.ECONOMIC):
        base, value = float(frozen[metric]), float(at_100["metrics"][metric])
        rows.append({"metric": metric, "frozen_m4_at_100": base, "this_run_at_100": value,
                     "rel_gap": abs(value - base) / abs(base) if base else float("nan")})
    frame = pd.DataFrame(rows)
    frame.attrs["reproduces_frozen_m4_at_100"] = bool(
        (frame.rel_gap <= REPRODUCTION_RELATIVE_TOLERANCE).all())
    return frame


def candidate_payloads(prepared: dict) -> dict[str, dict]:
    """The longest finished A/B curve from the candidates' own run folder."""
    root = candidate_out_dir() / "arms" / prepared["run_hash"]
    found = {}
    for variant, model_id in m5.ARMS.items():
        done = sorted(root.glob(f"{model_id}_to*.json"))
        if done:
            found[variant] = max(
                (json.loads(p.read_text(encoding="utf-8")) for p in done),
                key=lambda a: a["stop_epoch"])
    return found


def report(cfg, prepared: dict, payload: dict) -> dict:
    """Compare A/B with this reference and with M1/M3 at every shared epoch."""
    reference = {r["epoch"]: r["metrics"] for r in payload["curve"] if "metrics" in r}
    rows = [{"model_id": MODEL_ID, "epoch": epoch, **metrics}
            for epoch, metrics in reference.items()]
    candidates = candidate_payloads(prepared)
    for arm in candidates.values():
        rows += [{"model_id": arm["model_id"], "epoch": r["epoch"], **r["metrics"]}
                 for r in arm["curve"] if "metrics" in r]
    curve = pd.concat([pd.DataFrame(rows), prepared["m5_reference_curve"]],
                      ignore_index=True)
    index = curve.set_index(["model_id", "epoch"])
    metrics = [c for c in curve.columns if "@" in c
               or c == "user_value_tendency_recommended_price_alignment"]
    comparison = []
    for arm in candidates.values():
        for epoch in sorted({r["epoch"] for r in arm["curve"] if "metrics" in r}):
            for ref in (m5.M1_ID, m5.M3_ID, MODEL_ID):
                if (ref, epoch) not in index.index:
                    continue
                left, right = index.loc[(arm["model_id"], epoch)], index.loc[(ref, epoch)]
                for metric in metrics:
                    base, value = float(right[metric]), float(left[metric])
                    comparison.append({"epoch": epoch, "model_id": arm["model_id"],
                                       "reference": ref, "metric": metric,
                                       "reference_value": base, "candidate_value": value,
                                       "delta": value - base,
                                       "ratio": value / base if base else float("nan")})
    comparison = pd.DataFrame(comparison)
    gap = reproduction_gap(payload, prepared)
    reading = {"judge_epoch": JUDGE_EPOCH,
               "epoch_100_verdict_stands_unchanged": True,
               "registered_extension_reported_either_direction": True,
               "significance_claim": False,
               "arm_selected_after_results": False,
               "clv_assignment_attribution_at_300": False,
               "reference_reproduces_frozen_m4_at_100": bool(
                   gap.attrs["reproduces_frozen_m4_at_100"]),
               "max_reproduction_rel_gap": float(gap.rel_gap.max())}
    for arm in candidates.values():
        reading[arm["model_id"]] = arm_reading(comparison, arm, JUDGE_EPOCH)
    out = Path(cfg.out_dir) / "reports"
    stem = f"{CODE_VERSION}_{prepared['run_hash']}"
    paths = {"absolute_csv": out / f"{stem}_absolute.csv",
             "comparison_csv": out / f"{stem}_comparison.csv",
             "reproduction_csv": out / f"{stem}_reproduction.csv",
             "json": out / f"{stem}.json"}
    common.atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"], "split": hm2y.SPLIT,
        "final_test": False, "holdout": False, "lambda": m5.LAMBDA,
        "reference_model_id": MODEL_ID,
        "weight_audit": prepared["m5_weight_audit"][VARIANT],
        "references": prepared["m5_reference_sources"], "reading": reading,
        "limits": "single repeatedly exposed development seed; the 300-epoch point "
                  "carries no CLV-assignment attribution (no permuted arm at 300) "
                  "and does not replace the registered epoch-100 verdict",
        "result_paths": {k: str(v) for k, v in paths.items()}})
    for key, frame in (("absolute_csv", curve), ("comparison_csv", comparison),
                       ("reproduction_csv", gap)):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(paths[key], index=False)
    print(json.dumps({"reading": reading}, ensure_ascii=False, indent=2), flush=True)
    return {"absolute": curve, "comparison": comparison, "reproduction": gap,
            "reading": reading, "paths": paths}


def arm_reading(comparison: pd.DataFrame, arm: dict, epoch: int) -> dict:
    """The three registered conditions, read at `epoch` against this reference."""
    at = comparison[comparison.epoch.eq(epoch) & comparison.model_id.eq(arm["model_id"])]
    if at.empty:
        return {"trained_to_epoch": arm["stop_epoch"], "evaluated": False}
    m1 = at[at.reference.eq(m5.M1_ID)].set_index("metric")
    m4 = at[at.reference.eq(MODEL_ID)].set_index("metric")
    guard = bool((m1.loc[list(m5.ACCURACY), "ratio"] >= 0.99).all())
    above = bool((m1.loc[list(m5.ECONOMIC), "delta"] > 0).all())
    noninferior = bool((m4.loc[list(m5.ECONOMIC), "ratio"] >= 0.99).all())
    return {"trained_to_epoch": arm["stop_epoch"], "evaluated": True,
            "accuracy_guard_vs_m1": guard,
            "both_economic_at10_above_m1": above,
            "economic_at10_noninferior_99pct_vs_m4": noninferior,
            "economic_at10_above_m4": bool((m4.loc[list(m5.ECONOMIC), "delta"] > 0).all()),
            "all_three_met": guard and above and noninferior}
