"""Run two isolated personal-history M2 adjustments on one development seed.

The completed seed-42 M1 and axis_dim=4/rho=.05 M2 curves are read from the
existing capacity-search directory.  Only the following M2 arms are trained,
in this fixed order:

1. axis_dim=8, rho=.05
2. axis_dim=4, rho=.025

The historical development week (Dunnhumby days 684--690) is used.  The final
test week and holdout are neither constructed nor evaluated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_m2_capacity_search as search


CODE_VERSION = "clv-m2-axis-rho-seed42-screen-v1"
SEED = 42
REFERENCE_CONDITION = "baseline"
NEW_CONDITIONS = ("axis_wide", "weak_signal")
ALL_CONDITIONS = (REFERENCE_CONDITION,) + NEW_CONDITIONS
EPOCH = 300
PRIMARY_RECALL = "recall@50"
PRIMARY_WEIGHTED = "price_purchase_amount_weighted_hit@50"
TOP10_GUARDS = (
    "recall@10",
    "price_purchase_amount_weighted_hit@10",
    "vndcg@10",
)


@dataclass(frozen=True)
class Config:
    seed: int = SEED
    conditions: tuple[str, ...] = ALL_CONDITIONS
    epochs: int = EPOCH
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    negative_count: int = 1
    out_dir: str = ""


def configure(*, out_dir: str | None = None) -> Config:
    return validate_config(
        Config(
            out_dir=out_dir
            or f"{search.v3.default_out_dir('dunnhumby')}_clv_m2_capacity_search_v1"
        )
    )


def validate_config(cfg: Config) -> Config:
    required = {
        "seed": SEED,
        "conditions": ALL_CONDITIONS,
        "epochs": EPOCH,
        "eval_every": 25,
        "batch_size": 8192,
        "lr": 5e-4,
        "n_layers": 2,
        "negative_count": 1,
    }
    for field, expected in required.items():
        if getattr(cfg, field) != expected:
            raise ValueError(f"1시드 screen은 {field}={expected!r}이어야 합니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: Config) -> dict:
    validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "question": (
            "Can either extra N/V axis capacity or a weaker N/V score budget "
            "preserve the observed Top-10 signal while recovering deeper candidates?"
        ),
        "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "seed": cfg.seed,
        "fixed_epoch": cfg.epochs,
        "execution_order": [
            {"condition": "axis_wide", "axis_dim": 8, "rho": 0.05},
            {"condition": "weak_signal", "axis_dim": 4, "rho": 0.025},
        ],
        "reused_without_training": [
            {"condition": "baseline", "model": search.M1_MODEL_ID},
            {"condition": "baseline", "model": search.M2_MODEL_ID,
             "axis_dim": 4, "rho": 0.05},
        ],
        "new_training_fits": 2,
        "fixed": {
            "id_dim": 64,
            "pref_reg": 0.001,
            "n_layers": 2,
            "negative_count": 1,
            "graph": "binary",
            "objective": "plain_bpr",
        },
        "screen_gate": {
            "deep_vs_m1": [PRIMARY_RECALL, PRIMARY_WEIGHTED],
            "deep_vs_axis4": [PRIMARY_RECALL, PRIMARY_WEIGHTED],
            "ndcg10_ratio_vs_m1": ">=0.98",
            "top10_guards_vs_m1": list(TOP10_GUARDS),
        },
        "selection": "fixed epoch 300 only; no best-epoch selection",
        "final_test_constructed": False,
        "holdout_constructed": False,
        "limits": (
            "one screening seed on a repeatedly exposed development split; "
            "no significance, generalization, or final-model claim"
        ),
    }


def _search_config(cfg: Config) -> search.CapacitySearchConfig:
    return search.configure_capacity_search(
        conditions=cfg.conditions,
        seeds=(cfg.seed,),
        epochs=cfg.epochs,
        eval_every=cfg.eval_every,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        n_layers=cfg.n_layers,
        negative_count=cfg.negative_count,
        out_dir=cfg.out_dir,
    )


def _specs(search_cfg: search.CapacitySearchConfig) -> list[dict]:
    return search.arm_specifications(search_cfg)


def _reference_specs(search_cfg: search.CapacitySearchConfig) -> list[dict]:
    return [spec for spec in _specs(search_cfg) if spec["condition"] == REFERENCE_CONDITION]


def _new_specs(search_cfg: search.CapacitySearchConfig) -> list[dict]:
    specs = [
        spec
        for spec in _specs(search_cfg)
        if spec["model_id"] == search.M2_MODEL_ID
        and spec["condition"] in NEW_CONDITIONS
    ]
    if [spec["condition"] for spec in specs] != list(NEW_CONDITIONS):
        raise RuntimeError("신규 arm 실행 순서가 axis_wide -> weak_signal이 아닙니다")
    return specs


def _audit_arm_payload(payload: dict, spec: dict) -> None:
    for key in (
        "condition", "model_id", "shared_key", "id_dim", "axis_dim",
        "pref_reg", "rho", "seed", "code_version", "source_revision", "curve",
    ):
        if key not in payload:
            raise RuntimeError(f"기준 결과에 {key}가 없습니다")
    expected = {
        "condition": spec["condition"],
        "model_id": spec["model_id"],
        "shared_key": spec["shared_key"],
        "id_dim": spec["id_dim"],
        "axis_dim": spec["axis_dim"],
        "pref_reg": spec["pref_reg"],
        "rho": spec["rho"],
        "seed": SEED,
        "code_version": search.CODE_VERSION,
    }
    for key, value in expected.items():
        actual = payload[key]
        if isinstance(value, float):
            matches = np.isclose(float(actual), value)
        else:
            matches = actual == value
        if not matches:
            raise RuntimeError(f"기준 {key} 불일치: {actual!r} != {value!r}")
    epochs = [int(row["epoch"]) for row in payload["curve"]]
    if not epochs or epochs[-1] != EPOCH or len(set(epochs)) != len(epochs):
        raise RuntimeError("기준 곡선이 300 epoch까지 완료된 고유 평가점이 아닙니다")


def _load_references(prepared: dict, search_cfg: search.CapacitySearchConfig) -> list[dict]:
    payloads = []
    for spec in _reference_specs(search_cfg):
        path = search._arm_paths(prepared, spec, SEED)["result"]
        if not path.exists():
            raise RuntimeError(
                f"기존 seed42 기준결과가 없어 새 학습 전에 중단합니다: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        _audit_arm_payload(payload, spec)
        payloads.append(payload)
    if {payload["model_id"] for payload in payloads} != {
        search.M1_MODEL_ID, search.M2_MODEL_ID
    }:
        raise RuntimeError("기준 M1과 4차원 M2가 모두 필요합니다")
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise RuntimeError("기준 M1과 4차원 M2의 source revision이 다릅니다")
    return payloads


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column for column in frame.columns
        if "@" in column and pd.api.types.is_numeric_dtype(frame[column])
    ]


def fixed_epoch_tables(curve: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    absolute = curve[curve["epoch"].eq(EPOCH)].copy()
    expected = {
        (REFERENCE_CONDITION, search.M1_MODEL_ID),
        (REFERENCE_CONDITION, search.M2_MODEL_ID),
        ("axis_wide", search.M2_MODEL_ID),
        ("weak_signal", search.M2_MODEL_ID),
    }
    observed = set(zip(absolute["condition"], absolute["model_id"]))
    if observed != expected or len(absolute) != len(expected):
        raise RuntimeError("300 epoch absolute의 4개 arm이 누락되거나 중복됐습니다")
    metrics = _metric_columns(absolute)
    reference_rows = {
        "m1": absolute[
            absolute["condition"].eq(REFERENCE_CONDITION)
            & absolute["model_id"].eq(search.M1_MODEL_ID)
        ].iloc[0],
        "axis4_rho005": absolute[
            absolute["condition"].eq(REFERENCE_CONDITION)
            & absolute["model_id"].eq(search.M2_MODEL_ID)
        ].iloc[0],
    }
    comparison_rows = []
    for condition in NEW_CONDITIONS:
        candidate = absolute[
            absolute["condition"].eq(condition)
            & absolute["model_id"].eq(search.M2_MODEL_ID)
        ].iloc[0]
        for reference_name, reference in reference_rows.items():
            for metric in metrics:
                base, value = float(reference[metric]), float(candidate[metric])
                comparison_rows.append(
                    {
                        "condition": condition,
                        "reference": reference_name,
                        "metric": metric,
                        "reference_value": base,
                        "candidate_value": value,
                        "difference": value - base,
                        "relative_percent": (value / base - 1.0) * 100 if base else np.nan,
                    }
                )
    comparison = pd.DataFrame(comparison_rows)
    readings = []
    for condition in NEW_CONDITIONS:
        own = comparison[comparison["condition"].eq(condition)]

        def row(reference: str, metric: str) -> pd.Series:
            selected = own[own["reference"].eq(reference) & own["metric"].eq(metric)]
            if len(selected) != 1:
                raise RuntimeError(f"{condition}/{reference}/{metric} 비교가 없습니다")
            return selected.iloc[0]

        m1_ndcg = row("m1", "ndcg@10")
        deep_vs_m1 = all(row("m1", metric)["difference"] > 0 for metric in (
            PRIMARY_RECALL, PRIMARY_WEIGHTED
        ))
        deep_vs_axis4 = all(
            row("axis4_rho005", metric)["difference"] > 0
            for metric in (PRIMARY_RECALL, PRIMARY_WEIGHTED)
        )
        top10_guards = all(row("m1", metric)["difference"] >= 0 for metric in TOP10_GUARDS)
        ndcg_guard = bool(
            m1_ndcg["candidate_value"] >= 0.98 * m1_ndcg["reference_value"]
        )
        readings.append(
            {
                "condition": condition,
                "deep_vs_m1": deep_vs_m1,
                "deep_vs_axis4": deep_vs_axis4,
                "ndcg10_guard_pass": ndcg_guard,
                "top10_guards_pass": top10_guards,
                "single_seed_screen_condition_met": bool(
                    deep_vs_m1 and deep_vs_axis4 and ndcg_guard and top10_guards
                ),
                "final_test_used": False,
                "condition_selected": False,
                "significance_claimed": False,
            }
        )
    return (
        absolute.sort_values(["condition", "model_id"]).reset_index(drop=True),
        comparison,
        pd.DataFrame(readings),
    )


def run(cfg: Config, *, root: Path) -> dict:
    validate_config(cfg)
    expected_out = Path(root) / "results_v3_dunnhumby_clv_m2_capacity_search_v1"
    if Path(cfg.out_dir) != expected_out:
        raise ValueError(f"out_dir은 기존 기준결과 폴더여야 합니다: {expected_out}")
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    search_cfg = _search_config(cfg)
    prepared = search._prepare(search_cfg)
    references = _load_references(prepared, search_cfg)
    trained = []
    for spec in _new_specs(search_cfg):
        print(
            f"\n===== {spec['condition']} | {spec['model_id']} | seed {cfg.seed} "
            f"| axis {spec['axis_dim']} | rho {spec['rho']} ====="
        )
        payload = search._run_arm(prepared, search_cfg, spec, cfg.seed)
        _audit_arm_payload(payload, spec)
        trained.append(payload)
    curve = search.curve_table(references + trained)
    absolute, comparison, reading = fixed_epoch_tables(curve)

    out = Path(cfg.out_dir)
    paths = {
        "curve": out / "axis_rho_seed42_curve.csv",
        "absolute": out / "axis_rho_seed42_epoch300_absolute.csv",
        "comparison": out / "axis_rho_seed42_epoch300_comparison.csv",
        "reading": out / "axis_rho_seed42_epoch300_reading.csv",
        "json": out / "axis_rho_seed42_result.json",
    }
    search.test10._atomic_csv(paths["curve"], curve)
    search.test10._atomic_csv(paths["absolute"], absolute)
    search.test10._atomic_csv(paths["comparison"], comparison)
    search.test10._atomic_csv(paths["reading"], reading)
    search.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "source_revision": prepared["revision"],
            "reading": reading.to_dict("records"),
            "paths": {name: str(path) for name, path in paths.items()},
        },
    )
    print("\n300 epoch 비교:")
    headline = comparison[comparison["metric"].isin(
        ("recall@10", "ndcg@10", PRIMARY_RECALL, PRIMARY_WEIGHTED,
         "price_purchase_amount_weighted_hit@10", "vndcg@10")
    )]
    print(headline.to_string(index=False))
    print("\n단일 시드 screen 판독:")
    print(reading.to_string(index=False))
    return {name: str(path) for name, path in paths.items()}


if __name__ == "__main__":
    raise SystemExit("Colab에서 configure() 후 run(cfg, root=ROOT)를 호출하세요")
