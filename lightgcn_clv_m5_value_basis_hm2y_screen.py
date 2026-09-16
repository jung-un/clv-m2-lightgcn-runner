"""H&M portability screen for the CLV-scaled value-basis M2.

Dunnhumby seed 42 showed the value basis beating M1 on both economic metrics
(+2.32% / +2.12%) and beating its degree-matched CLV shuffle (+2.01% / +2.06%).
Nothing of that transfers to another dataset on its own. This run repeats the
minimal version on the H&M two-year validation split with the same single
negative BPR:

* ``m1``           - no value block;
* ``m2_actual``    - the value block with the observed CLV assignment;
* ``m2_shuffled``  - the same block after a degree-matched permutation.

The arms are trained in that order and each one is cached on Drive, so stopping
after the first two still answers "does M2 improve H&M at all"; the third arm
answers "is the improvement due to the customer-CLV assignment".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_axis_specific_gate_hm2y as hm2y
import lightgcn_clv_axis_specific_test10 as test10
import lightgcn_clv_gradient_isolated_economic_interaction as report_helpers
import lightgcn_clv_joint_nv as joint
import lightgcn_clv_m5_clv_scaled_value_basis_k1_screen as screen
import lightgcn_clv_m5_k1_m4_improvement_screen as improvement
import lightgcn_clv_m5_n_conditioned_value_basis_controls as controls
import lightgcn_clv_v3 as v3


CODE_VERSION = "m5-value-basis-hm2y-k1-screen-v1"
M1_MODEL_ID = "m1_bpr_k1_hm2y"
M2_ACTUAL_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1_hm2y_actual"
M2_SHUFFLED_MODEL_ID = "m2_clv_scaled_value_basis_bpr_k1_hm2y_degree_matched_shuffle"
MODEL_IDS = (M1_MODEL_ID, M2_ACTUAL_MODEL_ID, M2_SHUFFLED_MODEL_ID)
ECONOMIC_METRICS = screen.ECONOMIC_METRICS
TOP10_ACCURACY_METRICS = screen.TOP10_ACCURACY_METRICS
ACCURACY_METRICS = screen.ACCURACY_METRICS
ACCURACY_GUARD = 0.99


@dataclass(frozen=True)
class M5ValueBasisHm2yConfig:
    dataset: str = "hm"
    seed: int = 42
    epochs: int = 100
    id_dim: int = 64
    n_layers: int = 2
    rho: float = 0.25
    gate_delta: float = 0.25
    basis_bandwidth: float = 0.25
    negative_count: int = 1
    batch_size: int = 131_072
    lr: float = 5e-4
    pref_reg: float = 1e-3
    input_days: int = 365
    diagnostic_max_k: int = 50
    shuffle_degree_bins: int = 10
    shuffle_seed: int = 42
    include_shuffle: bool = True
    out_dir: str = ""


def configure_hm2y_value_basis_screen(**overrides) -> M5ValueBasisHm2yConfig:
    defaults = {
        "out_dir": f"{v3.default_out_dir('hm')}_m5_value_basis_hm2y_k1_screen_v1"
    }
    return validate_config(M5ValueBasisHm2yConfig(**(defaults | overrides)))


def validate_config(cfg: M5ValueBasisHm2yConfig) -> M5ValueBasisHm2yConfig:
    fixed = {
        "dataset": "hm",
        "seed": 42,
        "epochs": 100,
        "id_dim": 64,
        "n_layers": 2,
        "rho": 0.25,
        "gate_delta": 0.25,
        "basis_bandwidth": 0.25,
        "negative_count": 1,
        "batch_size": 131_072,
        "input_days": 365,
        "shuffle_degree_bins": 10,
        "shuffle_seed": 42,
    }
    for key, expected in fixed.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"H&M 가치기저 screen은 {key}={expected!r}이어야 합니다")
    if cfg.lr <= 0 or cfg.pref_reg < 0 or not cfg.out_dir:
        raise ValueError("학습 설정 또는 out_dir가 잘못됐습니다")
    return cfg


def preflight_summary(cfg: M5ValueBasisHm2yConfig) -> dict:
    cfg = validate_config(cfg)
    trained = list(MODEL_IDS) if cfg.include_shuffle else list(MODEL_IDS[:2])
    return {
        "code_version": CODE_VERSION,
        "dataset": cfg.dataset,
        "seed": cfg.seed,
        "split": "hm two-year train through 2020-09-01, validation 2020-09-02--08",
        "trained_models": trained,
        "reused_models": [],
        "research_question": (
            "Does the CLV-scaled value basis that improved Dunnhumby also improve "
            "H&M, and is any improvement due to the customer-CLV assignment?"
        ),
        "loss": {
            "bpr": "mean softplus(s(u,j) - s(u,i+)) + batch layer-0 L2",
            "negative_count": cfg.negative_count,
            "negative_sampling": "uniform over unseen items",
            "positive_weight": "none in every arm (no M4 here)",
        },
        "m2": {
            "user_block": "sqrt(rho) * q_C * g_N(q_N) * RBF(q_V)",
            "item_block": "sqrt(rho) * RBF(train mean item price percentile)",
            "rho": cfg.rho,
            "economic_graph_propagation": False,
        },
        "control": {
            "permutation": (
                "q_N, q_V, q_C and the CLV validity flag move together inside the "
                "same binary user-degree decile"
            ),
            "degree_bins": cfg.shuffle_degree_bins,
        },
        "reading_rule": {
            "improvement": "m2_actual > m1 on both economic metrics",
            "accuracy_guard": f"m2_actual Recall@10 and NDCG@10 >= {ACCURACY_GUARD} x m1",
            "evaluable": "m2_actual changes at least one user's Top-10 set versus m1",
            "assignment_signal": (
                "m2_actual > m2_shuffled on both economic metrics, reported only "
                "when the shuffle arm is trained"
            ),
            "portability": "improvement and accuracy_guard and evaluable",
        },
        "fixed": {
            "new_item_task": True,
            "min_item_interactions": 1,
            "graph": "binary",
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "final_test_constructed": False,
            "holdout_constructed": False,
            "one_training_loop_and_optimizer_per_arm": True,
        },
        "statistical_note": (
            "one development seed on the H&M validation split; no significance or "
            "generalization claim"
        ),
        "out_dir": cfg.out_dir,
    }


def degree_deciles(train: pd.DataFrame, n_users: int, bins: int = 10) -> np.ndarray:
    """Binary user-degree decile for the degree-matched permutation."""

    pairs = train[["u_idx", "i_idx"]].drop_duplicates()
    degree = np.bincount(pairs.u_idx.to_numpy(np.int64), minlength=n_users)
    order = np.argsort(np.argsort(degree, kind="stable"), kind="stable")
    edges = (order * bins) // max(len(degree), 1)
    return np.clip(edges, 0, bins - 1).astype(np.int64)


def _config_hash(cfg: M5ValueBasisHm2yConfig, input_hash: str, revision: str) -> str:
    payload = {
        "code_version": CODE_VERSION,
        "config": asdict(cfg),
        "input_hash": input_hash,
        "source_revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def _prepare(cfg: M5ValueBasisHm2yConfig) -> dict:
    runner_cfg = hm2y.configure_axis_specific_gate_hm2y_run(
        out_dir=cfg.out_dir, batch_size=cfg.batch_size
    )
    prepared = joint._prepare(runner_cfg)
    prepared["out_dir"] = Path(cfg.out_dir)
    data = prepared["data"]
    q_n, q_v, q_c, clv_valid = report_helpers.build_clv_inputs(prepared["axes"])
    item_price, item_valid = report_helpers.build_item_price_inputs(
        data["train"], data["n_items"]
    )
    prepared.update(
        q_n=q_n,
        q_v=q_v,
        q_c=q_c,
        clv_valid=clv_valid,
        item_amount_percentile=item_price,
        item_economic_valid=item_valid,
        degree_bin=degree_deciles(
            data["train"], data["n_users"], cfg.shuffle_degree_bins
        ),
    )
    prepared["m2_actual"] = {
        "q_n": q_n,
        "q_v": q_v,
        "q_c": q_c,
        "clv_valid": clv_valid,
    }
    prepared["m2_shuffle"] = controls.degree_matched_nv_shuffle(
        prepared, seed=cfg.shuffle_seed, degree_bins=cfg.shuffle_degree_bins
    )
    source = prepared["m2_shuffle"]["source_user"]
    prepared["control_diagnostics"] = {
        "n_users": int(len(source)),
        "valid_users": int(np.asarray(clv_valid).sum()),
        "moved_user_share": float(np.mean(source != np.arange(len(source)))),
        "same_degree_bin": bool(
            np.all(prepared["degree_bin"][source] == prepared["degree_bin"])
        ),
    }
    if not prepared["control_diagnostics"]["same_degree_bin"]:
        raise RuntimeError("CLV 순열이 degree 층 밖으로 이동했습니다")
    prepared["config_hash"] = _config_hash(
        cfg, prepared["input_hash"], prepared["revision"]
    )
    return prepared


def arm_specifications(prepared: dict, cfg: M5ValueBasisHm2yConfig) -> list[dict]:
    arms = [
        {
            "model_id": M1_MODEL_ID,
            "role": "hm2y_m1",
            "rho": 0.0,
            "improvement": None,
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "inactive",
        },
        {
            "model_id": M2_ACTUAL_MODEL_ID,
            "role": "hm2y_m2_observed_clv",
            "rho": cfg.rho,
            "improvement": None,
            "m2_assignment": prepared["m2_actual"],
            "m2_assignment_name": "observed_clv",
        },
    ]
    if cfg.include_shuffle:
        arms.append(
            {
                "model_id": M2_SHUFFLED_MODEL_ID,
                "role": "hm2y_m2_assignment_control",
                "rho": cfg.rho,
                "improvement": None,
                "m2_assignment": prepared["m2_shuffle"],
                "m2_assignment_name": "degree_matched_clv_shuffle",
            }
        )
    return arms


def portability_reading(
    metric_rows: dict[str, dict], *, top10_change_shares: dict[str, float]
) -> dict:
    m1, m2 = metric_rows[M1_MODEL_ID], metric_rows[M2_ACTUAL_MODEL_ID]
    improvement_signal = all(m2[metric] > m1[metric] for metric in ECONOMIC_METRICS)
    accuracy_guard = all(
        m2[metric] >= ACCURACY_GUARD * m1[metric] for metric in TOP10_ACCURACY_METRICS
    )
    evaluable = top10_change_shares.get(M2_ACTUAL_MODEL_ID, 0.0) > 0.0
    reading = {
        "classification": (
            "not_evaluable_no_top10_change"
            if not evaluable
            else ("portable" if improvement_signal and accuracy_guard else "not_portable")
        ),
        "portability": bool(evaluable and improvement_signal and accuracy_guard),
        "evaluable": bool(evaluable),
        "m2_beats_m1_on_both_economic_metrics": bool(improvement_signal),
        "m2_accuracy_guard_vs_m1": bool(accuracy_guard),
        "top10_set_changed_user_share": dict(top10_change_shares),
        "deltas_m2_minus_m1": {
            metric: float(m2[metric] - m1[metric])
            for metric in TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
        },
        "assignment_signal_tested": M2_SHUFFLED_MODEL_ID in metric_rows,
        "statistical_note": (
            "one development seed on the H&M validation split; no significance or "
            "generalization claim"
        ),
    }
    if M2_SHUFFLED_MODEL_ID in metric_rows:
        shuffled = metric_rows[M2_SHUFFLED_MODEL_ID]
        reading["m2_beats_shuffle_on_both_economic_metrics"] = bool(
            all(m2[metric] > shuffled[metric] for metric in ECONOMIC_METRICS)
        )
        reading["deltas_m2_actual_minus_shuffled"] = {
            metric: float(m2[metric] - shuffled[metric])
            for metric in TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
        }
    return reading


def run_hm2y_value_basis_screen(
    cfg: M5ValueBasisHm2yConfig | None = None,
) -> pd.DataFrame:
    cfg = validate_config(cfg or configure_hm2y_value_basis_screen())
    summary = preflight_summary(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    prepared = _prepare(cfg)
    print("\n[순열 불변식]")
    print(json.dumps(prepared["control_diagnostics"], ensure_ascii=False, indent=2))

    arms: dict[str, dict] = {}
    models: dict[str, object] = {}
    for spec in arm_specifications(prepared, cfg):
        print(
            f"\n===== {spec['model_id']} | seed {cfg.seed} | K={cfg.negative_count} | "
            f"fixed {cfg.epochs} epochs ====="
        )
        arm, model = improvement._run_arm(prepared, cfg, spec)
        arm["m2_assignment"] = spec["m2_assignment_name"]
        arms[spec["model_id"]] = arm
        models[spec["model_id"]] = model
        print(
            f"  [{spec['model_id']}] "
            + json.dumps(
                {
                    metric: round(float(arm["metrics"][metric]), 6)
                    for metric in TOP10_ACCURACY_METRICS + ECONOMIC_METRICS
                },
                ensure_ascii=False,
            )
        )

    trained = list(arms)
    metric_rows = {model_id: arms[model_id]["metrics"] for model_id in trained}
    frame = pd.DataFrame(
        [
            {
                "model_id": model_id,
                "role": arms[model_id]["role"],
                "rho": arms[model_id]["rho"],
                "m2_assignment": arms[model_id]["m2_assignment"],
                **arms[model_id]["diagnostics"],
                **arms[model_id]["metrics"],
            }
            for model_id in trained
        ]
    )
    comparison = report_helpers._metric_comparison(
        metric_rows, references=(M1_MODEL_ID,)
    )

    topk = {
        model_id: report_helpers._masked_topk(
            models[model_id], prepared, max_k=cfg.diagnostic_max_k
        )
        for model_id in trained
    }
    segments = prepared["cache"].seg
    overlap = pd.concat(
        [
            report_helpers.topk_overlap_summary(
                topk[M1_MODEL_ID][1], topk[model_id][1], segments
            ).assign(reference=M1_MODEL_ID, model_id=model_id)
            for model_id in trained
            if model_id != M1_MODEL_ID
        ],
        ignore_index=True,
    )
    overall = overlap[overlap.group.eq("전체")].set_index("model_id")
    change_shares = {
        model_id: float(overall.at[model_id, "top10_set_changed_user_share"])
        for model_id in trained
        if model_id != M1_MODEL_ID
    }
    reading = portability_reading(metric_rows, top10_change_shares=change_shares)

    out = Path(cfg.out_dir)
    stem = f"m5_value_basis_hm2y_{prepared['config_hash']}"
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
            "control_diagnostics": prepared["control_diagnostics"],
            "absolute_rows": frame.to_dict("records"),
            "comparison_rows": comparison.to_dict("records"),
            "top10_overlap_rows": overlap.to_dict("records"),
            "portability_reading": reading,
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
            preflight_summary(configure_hm2y_value_basis_screen()),
            ensure_ascii=False,
            indent=2,
        )
    )
