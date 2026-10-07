"""Complete the 300-epoch personal-history M2 development check at five seeds.

Only Dunnhumby seeds 45 and 46 are newly trained.  Their fixed-epoch results
are combined with the already completed seeds 42--44, without constructing or
reading the final test week.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

import lightgcn_clv_m2_capacity_search as search


CODE_VERSION = "clv-m2-history-budget-dev5-completion-v1"
M1_MODEL_ID = search.M1_MODEL_ID
M2_MODEL_ID = search.M2_MODEL_ID
SEEDS = (45, 46)
ALL_SEEDS = (42, 43, 44, 45, 46)
PRIMARY_RECALL = "recall@50"
PRIMARY_WEIGHTED = "price_purchase_amount_weighted_hit@50"
GUARD_METRIC = "ndcg@10"


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = SEEDS
    conditions: tuple[str, ...] = ("baseline",)
    epochs: int = 300
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
            or f"{search.v3.default_out_dir('dunnhumby')}_clv_m2_history_budget_dev5_completion_v1"
        )
    )


def validate_config(cfg: Config) -> Config:
    required = {
        "seeds": SEEDS,
        "conditions": ("baseline",),
        "epochs": 300,
        "eval_every": 25,
        "batch_size": 8192,
        "lr": 5e-4,
        "n_layers": 2,
        "negative_count": 1,
    }
    for field, expected in required.items():
        if getattr(cfg, field) != expected:
            raise ValueError(f"5시드 보충은 {field}={expected!r}이어야 합니다")
    if not cfg.out_dir:
        raise ValueError("out_dir가 필요합니다")
    return cfg


def preflight_summary(cfg: Config) -> dict:
    cfg = validate_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "question": (
            "Does the personal-history q_N/q_V M2 retain its pre-registered "
            "deep-candidate role at 300 epochs across five development seeds?"
        ),
        "dataset": "dunnhumby",
        "split": "historical_development_days_684_690",
        "new_seeds": list(cfg.seeds),
        "combined_seeds": list(ALL_SEEDS),
        "models": [M1_MODEL_ID, M2_MODEL_ID],
        "new_training_fits": len(cfg.seeds) * 2,
        "epochs": cfg.epochs,
        "evaluated_every": cfg.eval_every,
        "final_test_constructed": False,
        "holdout_constructed": False,
        "selection": "fixed epoch 300 only; no best-epoch selection",
        "gate": {
            "recall50": "mean M2-M1 > 0 and positive in at least 4/5 seeds",
            "weighted_hit50": "mean M2-M1 > 0 and positive in at least 4/5 seeds",
            "ndcg10": "five-seed M2 mean is at least 98% of M1 mean",
        },
        "limits": (
            "repeatedly exposed development split; no final-test, significance, "
            "generalization, or CLV-attribution claim"
        ),
    }


def _metric_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"epoch", "seed"}
    return [
        column
        for column in frame.columns
        if "@" in column and column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _fixed_epoch_absolute(curve: pd.DataFrame) -> pd.DataFrame:
    required = {
        "condition", "model_id", "seed", "epoch", "id_dim", "axis_dim",
        "pref_reg", "rho", PRIMARY_RECALL, PRIMARY_WEIGHTED, GUARD_METRIC,
    }
    missing = sorted(required - set(curve.columns))
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    absolute = curve[
        curve["condition"].eq("baseline")
        & curve["epoch"].eq(300)
        & curve["seed"].isin(ALL_SEEDS)
        & curve["model_id"].isin((M1_MODEL_ID, M2_MODEL_ID))
    ].copy()
    counts = absolute.groupby(["seed", "model_id"]).size()
    expected = pd.MultiIndex.from_product(
        [ALL_SEEDS, (M1_MODEL_ID, M2_MODEL_ID)], names=["seed", "model_id"]
    )
    if not counts.index.equals(expected) or not (counts.to_numpy() == 1).all():
        raise ValueError("300 epoch의 seed-model 쌍이 누락되거나 중복됐습니다")
    for column, expected_value in {
        "id_dim": 64,
        "pref_reg": 0.001,
    }.items():
        if not np.allclose(absolute[column].astype(float), expected_value):
            raise ValueError(f"{column} 계약이 다릅니다")
    m1 = absolute[absolute.model_id.eq(M1_MODEL_ID)]
    m2 = absolute[absolute.model_id.eq(M2_MODEL_ID)]
    if not (m1.axis_dim.eq(0).all() and np.allclose(m1.rho, 0.0)):
        raise ValueError("M1 비개입 계약이 다릅니다")
    if not (m2.axis_dim.eq(4).all() and np.allclose(m2.rho, 0.05)):
        raise ValueError("M2 axis_dim/rho 계약이 다릅니다")
    return absolute.sort_values(["seed", "model_id"]).reset_index(drop=True)


def read_five_seed_result(curve: pd.DataFrame):
    absolute = _fixed_epoch_absolute(curve)
    metrics = _metric_columns(absolute)
    by_model = {
        model: absolute[absolute.model_id.eq(model)].set_index("seed")
        for model in (M1_MODEL_ID, M2_MODEL_ID)
    }
    paired_rows = []
    for seed in ALL_SEEDS:
        row = {"seed": seed}
        for metric in metrics:
            baseline = float(by_model[M1_MODEL_ID].loc[seed, metric])
            value = float(by_model[M2_MODEL_ID].loc[seed, metric])
            row[metric] = value - baseline
            row[f"{metric}_ratio"] = value / baseline if baseline else np.nan
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)

    summary_rows = []
    for metric in metrics:
        m1_values = by_model[M1_MODEL_ID][metric].astype(float)
        m2_values = by_model[M2_MODEL_ID][metric].astype(float)
        differences = m2_values - m1_values
        m1_mean, m2_mean = float(m1_values.mean()), float(m2_values.mean())
        summary_rows.append(
            {
                "metric": metric,
                "m1_mean": m1_mean,
                "m1_std": float(m1_values.std(ddof=1)),
                "m2_mean": m2_mean,
                "m2_std": float(m2_values.std(ddof=1)),
                "mean_difference": float(differences.mean()),
                "relative_percent": (m2_mean / m1_mean - 1.0) * 100 if m1_mean else np.nan,
                "positive_seeds": int((differences > 0).sum()),
            }
        )
    summary = pd.DataFrame(summary_rows)

    def summary_for(metric: str) -> pd.Series:
        rows = summary[summary.metric.eq(metric)]
        if len(rows) != 1:
            raise ValueError(f"판독 지표 {metric}이 없습니다")
        return rows.iloc[0]

    recall = summary_for(PRIMARY_RECALL)
    weighted = summary_for(PRIMARY_WEIGHTED)
    ndcg = summary_for(GUARD_METRIC)
    ndcg_ratio = float(ndcg.m2_mean / ndcg.m1_mean) if ndcg.m1_mean else float("nan")
    reading = {
        "seeds": list(ALL_SEEDS),
        "epoch": 300,
        "recall50_mean_positive": bool(recall.mean_difference > 0),
        "recall50_positive_seeds": int(recall.positive_seeds),
        "weighted_hit50_mean_positive": bool(weighted.mean_difference > 0),
        "weighted_hit50_positive_seeds": int(weighted.positive_seeds),
        "ndcg10_guard_ratio": ndcg_ratio,
        "ndcg10_guard_pass": bool(ndcg_ratio >= 0.98),
        "significance_claimed": False,
        "final_test_used": False,
    }
    reading["retained_for_final_evaluation"] = bool(
        reading["recall50_mean_positive"]
        and reading["recall50_positive_seeds"] >= 4
        and reading["weighted_hit50_mean_positive"]
        and reading["weighted_hit50_positive_seeds"] >= 4
        and reading["ndcg10_guard_pass"]
    )
    return absolute, paired, summary, reading


def _single_curve(folder: Path) -> Path:
    files = sorted(folder.glob("clv_m2_capacity_search_*_curve.csv"))
    if len(files) != 1:
        raise RuntimeError(f"{folder}의 curve CSV가 1개가 아닙니다: {files}")
    return files[0]


def collect_curves(root: Path, completion_dir: Path) -> pd.DataFrame:
    folders = {
        42: root / "results_v3_dunnhumby_clv_m2_capacity_search_v1",
        43: root / "results_v3_dunnhumby_clv_m2_training_budget_seed43_v1",
        44: root / "results_v3_dunnhumby_clv_m2_training_budget_seed44_v1",
    }
    frames = []
    for seed, folder in folders.items():
        frame = pd.read_csv(_single_curve(folder))
        frame = frame[frame.seed.eq(seed)]
        frames.append(frame)
    completion_curve = pd.read_csv(_single_curve(completion_dir))
    frames.append(completion_curve[completion_curve.seed.isin(SEEDS)])
    return pd.concat(frames, ignore_index=True)


def run(cfg: Config, *, root: Path) -> dict:
    cfg = validate_config(cfg)
    print(json.dumps(preflight_summary(cfg), ensure_ascii=False, indent=2))
    search_cfg = search.configure_capacity_search(
        conditions=cfg.conditions,
        seeds=cfg.seeds,
        epochs=cfg.epochs,
        eval_every=cfg.eval_every,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        n_layers=cfg.n_layers,
        negative_count=cfg.negative_count,
        out_dir=cfg.out_dir,
    )
    search.run_capacity_search(search_cfg)
    curve = collect_curves(Path(root), Path(cfg.out_dir))
    absolute, paired, summary, reading = read_five_seed_result(curve)
    out = Path(cfg.out_dir)
    paths = {
        "absolute": out / "dev5_epoch300_absolute.csv",
        "paired": out / "dev5_epoch300_paired.csv",
        "summary": out / "dev5_epoch300_summary.csv",
        "json": out / "dev5_epoch300_result.json",
    }
    search.test10._atomic_csv(paths["absolute"], absolute)
    search.test10._atomic_csv(paths["paired"], paired)
    search.test10._atomic_csv(paths["summary"], summary)
    search.test10._atomic_json(
        paths["json"],
        {
            "code_version": CODE_VERSION,
            "config": asdict(cfg),
            "preflight": preflight_summary(cfg),
            "reading": reading,
            "paths": {key: str(path) for key, path in paths.items()},
        },
    )
    print(json.dumps(reading, ensure_ascii=False, indent=2))
    return {key: str(path) for key, path in paths.items()}


if __name__ == "__main__":
    raise SystemExit("Colab에서 configure() 후 run(cfg, root=ROOT)를 호출하세요")
