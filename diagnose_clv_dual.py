"""재학습 없이 CLV 이중축 validation 산출물을 진단한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY_MODEL = "dual_clv_fixed"
CONTROLS = ("dual_shuffled_user", "dual_adapter_only")
ACCURACY_KEYS = tuple(
    f"{metric}@{cutoff}"
    for metric in ("recall", "ndcg")
    for cutoff in (10, 20, 50)
)


def _read_inputs(csv_path: Path, json_path: Path) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    required = {
        "model_id",
        "split",
        "gate_shape",
        "lambda",
        "effective_strength",
        "revenue@10",
        *ACCURACY_KEYS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"진단 필수 열 누락: {missing}")
    if set(frame["split"].dropna()) != {"val"}:
        raise ValueError("validation 산출물만 진단할 수 있습니다")
    keys = ["model_id", "gate_shape", "lambda"]
    duplicated = frame.loc[frame.duplicated(keys, keep=False), keys]
    if not duplicated.empty:
        raise ValueError(f"모형 운영점 중복: {duplicated.to_dict('records')[:3]}")
    selected = payload.get("selected_operating_point", {})
    if not {"gate_shape", "lambda"}.issubset(selected):
        raise ValueError("JSON에 selected_operating_point가 없습니다")
    return frame, payload


def _baseline(frame: pd.DataFrame) -> pd.Series:
    rows = frame.loc[frame.model_id.eq("m1")]
    if len(rows) != 1:
        raise ValueError(f"M1 행은 정확히 하나여야 합니다: {len(rows)}")
    return rows.iloc[0]


def _operating_row(
    frame: pd.DataFrame, model_id: str, gate_shape: str, lam: float
) -> pd.Series:
    rows = frame.loc[
        frame.model_id.eq(model_id)
        & frame.gate_shape.eq(gate_shape)
        & np.isclose(frame["lambda"].to_numpy(float), float(lam))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"운영점 행이 하나가 아닙니다: {model_id}, {gate_shape}, {lam}"
        )
    return rows.iloc[0]


def _eligible_curve(frame: pd.DataFrame, baseline: pd.Series, gate_shape: str):
    curve = frame.loc[
        frame.model_id.isin((PRIMARY_MODEL, *CONTROLS))
        & frame.gate_shape.eq(gate_shape)
        & frame["lambda"].gt(0)
    ].copy()
    main = curve.loc[curve.model_id.eq(PRIMARY_MODEL)].copy()
    eligible = np.ones(len(main), dtype=bool)
    for key in ACCURACY_KEYS:
        eligible &= main[key].to_numpy(float) >= float(baseline[key]) * 0.99
    eligible_lambda = set(main.loc[eligible, "lambda"].astype(float))
    curve["primary_accuracy_eligible"] = curve["lambda"].isin(eligible_lambda)
    curve["revenue_delta_vs_m1"] = curve["revenue@10"] - float(
        baseline["revenue@10"]
    )
    curve["revenue_change_vs_m1_pct"] = (
        curve["revenue_delta_vs_m1"] / float(baseline["revenue@10"]) * 100
    )
    return curve.sort_values(["model_id", "lambda"])


def _selected_comparison(
    frame: pd.DataFrame, baseline: pd.Series, gate_shape: str, lam: float
) -> pd.DataFrame:
    main = _operating_row(frame, PRIMARY_MODEL, gate_shape, lam)
    records = []
    for label, row in [
        ("M1", baseline),
        *[
            (control, _operating_row(frame, control, gate_shape, lam))
            for control in CONTROLS
        ],
    ]:
        delta = float(main["revenue@10"] - row["revenue@10"])
        records.append(
            {
                "comparison": label,
                "gate_shape": gate_shape,
                "lambda": lam,
                "primary_revenue@10": float(main["revenue@10"]),
                "comparison_revenue@10": float(row["revenue@10"]),
                "absolute_delta": delta,
                "relative_change_pct": delta / float(row["revenue@10"]) * 100,
            }
        )
    return pd.DataFrame(records)


def _curve_crossings(curve: pd.DataFrame) -> pd.DataFrame:
    main = curve.loc[curve.model_id.eq(PRIMARY_MODEL), ["lambda", "revenue@10"]]
    records = []
    for control in CONTROLS:
        other = curve.loc[
            curve.model_id.eq(control), ["lambda", "revenue@10"]
        ]
        paired = main.merge(other, on="lambda", suffixes=("_main", "_control"))
        paired = paired.sort_values("lambda")
        paired["difference"] = (
            paired["revenue@10_main"] - paired["revenue@10_control"]
        )
        values = paired.to_dict("records")
        for left, right in zip(values, values[1:]):
            left_diff, right_diff = left["difference"], right["difference"]
            if left_diff == 0 or right_diff == 0 or left_diff * right_diff < 0:
                records.append(
                    {
                        "control": control,
                        "lambda_interval": f"{left['lambda']:g}→{right['lambda']:g}",
                        "left_difference": float(left_diff),
                        "right_difference": float(right_diff),
                    }
                )
    return pd.DataFrame(
        records,
        columns=[
            "control",
            "lambda_interval",
            "left_difference",
            "right_difference",
        ],
    )


def _same_lambda_dominance(curve: pd.DataFrame) -> pd.DataFrame:
    main = curve.loc[
        curve.model_id.eq(PRIMARY_MODEL) & curve.primary_accuracy_eligible
    ]
    records = []
    for _, main_row in main.iterrows():
        for control in CONTROLS:
            other = curve.loc[
                curve.model_id.eq(control)
                & np.isclose(
                    curve["lambda"].to_numpy(float), float(main_row["lambda"])
                )
            ]
            if other.empty:
                continue
            control_revenue = float(other.iloc[0]["revenue@10"])
            delta = float(main_row["revenue@10"]) - control_revenue
            records.append(
                {
                    "lambda": float(main_row["lambda"]),
                    "effective_strength": float(main_row["effective_strength"]),
                    "control": control,
                    "primary_revenue@10": float(main_row["revenue@10"]),
                    "control_revenue@10": control_revenue,
                    "absolute_delta": delta,
                    "primary_wins": delta > 0,
                }
            )
    return pd.DataFrame(records)


def _matched_curve_dominance(curve: pd.DataFrame) -> pd.DataFrame:
    main = curve.loc[
        curve.model_id.eq(PRIMARY_MODEL) & curve.primary_accuracy_eligible
    ]
    records = []
    for _, main_row in main.iterrows():
        for control in CONTROLS:
            value, method = _interpolate_curve(
                curve.loc[curve.model_id.eq(control)],
                float(main_row["effective_strength"]),
            )
            delta = float(main_row["revenue@10"]) - value
            records.append(
                {
                    "lambda": float(main_row["lambda"]),
                    "effective_strength": float(main_row["effective_strength"]),
                    "control": control,
                    "primary_revenue@10": float(main_row["revenue@10"]),
                    "interpolated_control_revenue": value,
                    "absolute_delta": delta,
                    "primary_wins": delta > 0,
                    "matching_method": method,
                }
            )
    return pd.DataFrame(records)


def _primary_gate_summary(frame: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    main = frame.loc[frame.model_id.eq(PRIMARY_MODEL)].copy()
    eligible = np.ones(len(main), dtype=bool)
    for key in ACCURACY_KEYS:
        eligible &= main[key].to_numpy(float) >= float(baseline[key]) * 0.99
    main["eligible"] = eligible
    main["improves_m1"] = main["revenue@10"] > float(baseline["revenue@10"])
    records = []
    for gate_shape, group in main.groupby("gate_shape"):
        positive = group.loc[group.eligible & group.improves_m1 & group["lambda"].gt(0)]
        best = None if positive.empty else positive.loc[positive["revenue@10"].idxmax()]
        records.append(
            {
                "gate_shape": gate_shape,
                "eligible_positive_point_count": int(len(positive)),
                "best_lambda": np.nan if best is None else float(best["lambda"]),
                "best_revenue@10": (
                    np.nan if best is None else float(best["revenue@10"])
                ),
                "best_change_vs_m1_pct": (
                    np.nan
                    if best is None
                    else (
                        float(best["revenue@10"]) - float(baseline["revenue@10"])
                    )
                    / float(baseline["revenue@10"])
                    * 100
                ),
            }
        )
    return pd.DataFrame(records).sort_values("gate_shape")


def _interpolate_curve(curve: pd.DataFrame, target: float) -> tuple[float, str]:
    points = (
        curve[["effective_strength", "revenue@10"]]
        .dropna()
        .groupby("effective_strength", as_index=False)["revenue@10"]
        .mean()
        .sort_values("effective_strength")
    )
    x = points.effective_strength.to_numpy(float)
    y = points["revenue@10"].to_numpy(float)
    if not len(points):
        return float("nan"), "missing"
    if len(points) == 1 or target <= x[0]:
        return float(y[0]), "nearest_boundary"
    if target >= x[-1]:
        return float(y[-1]), "nearest_boundary"
    return float(np.interp(target, x, y)), "linear_interpolation"


def _matched_strength(
    curve: pd.DataFrame, selected_main: pd.Series
) -> pd.DataFrame:
    target = float(selected_main["effective_strength"])
    records = []
    for control in CONTROLS:
        value, method = _interpolate_curve(
            curve.loc[curve.model_id.eq(control)], target
        )
        delta = float(selected_main["revenue@10"]) - value
        records.append(
            {
                "control": control,
                "target_effective_strength": target,
                "primary_revenue@10": float(selected_main["revenue@10"]),
                "interpolated_control_revenue": value,
                "absolute_delta": delta,
                "relative_change_pct": delta / value * 100 if value else np.nan,
                "matching_method": method,
            }
        )
    return pd.DataFrame(records)


def _neighbor_stability(curve: pd.DataFrame, selected_lambda: float) -> pd.DataFrame:
    main = curve.loc[curve.model_id.eq(PRIMARY_MODEL)].sort_values("lambda")
    lambdas = main["lambda"].to_numpy(float)
    center = int(np.abs(lambdas - selected_lambda).argmin())
    selected_indices = range(max(0, center - 1), min(len(main), center + 2))
    records = []
    for index in selected_indices:
        main_row = main.iloc[index]
        for control in CONTROLS:
            control_row = curve.loc[
                curve.model_id.eq(control)
                & np.isclose(
                    curve["lambda"].to_numpy(float), float(main_row["lambda"])
                )
            ]
            if control_row.empty:
                continue
            delta = float(main_row["revenue@10"] - control_row.iloc[0]["revenue@10"])
            records.append(
                {
                    "lambda": float(main_row["lambda"]),
                    "is_selected": bool(np.isclose(main_row["lambda"], selected_lambda)),
                    "control": control,
                    "absolute_delta": delta,
                    "primary_wins": delta > 0,
                    "primary_accuracy_eligible": bool(
                        main_row["primary_accuracy_eligible"]
                    ),
                }
            )
    return pd.DataFrame(records)


def _axis_summary(selected_main: pd.Series, payload: dict) -> dict:
    keys = (
        "effective_n_ratio",
        "effective_v_ratio",
        "effective_total_ratio",
        "expert_score_corr",
        "expert_top10_jaccard",
    )
    return {
        **{key: float(selected_main[key]) for key in keys if key in selected_main},
        "axis_preflight": payload.get("axis_preflight", {}),
    }


def _segment_table(selected_main: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    records = []
    for segment in ("저CLV", "중CLV", "고CLV"):
        for metric in ("recall@10", "ndcg@10", "revenue@10"):
            key = f"{segment}_{metric}"
            if key not in selected_main or key not in baseline:
                continue
            base_value = float(baseline[key])
            model_value = float(selected_main[key])
            records.append(
                {
                    "segment": segment,
                    "metric": metric,
                    "M1": base_value,
                    PRIMARY_MODEL: model_value,
                    "absolute_delta": model_value - base_value,
                    "relative_change_pct": (
                        (model_value - base_value) / base_value * 100
                        if base_value
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(records)


def _plot_curve(curve: pd.DataFrame, x: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_id, model_curve in curve.groupby("model_id", sort=False):
        model_curve = model_curve.sort_values(x)
        ax.plot(
            model_curve[x],
            model_curve["revenue@10"],
            marker="o",
            label=model_id,
        )
    ax.set_title(title)
    ax.set_xlabel("λ" if x == "lambda" else "Effective intervention strength")
    ax.set_ylabel("Purchase-amount weighted hit@10")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def diagnose_dual_results(
    csv_path: str | Path,
    json_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict:
    """기존 CSV/JSON만 읽어 이중축 모델의 선택점과 곡선 안정성을 진단한다."""
    csv_path, json_path = Path(csv_path), Path(json_path)
    output_dir = Path(output_dir or csv_path.parent / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, payload = _read_inputs(csv_path, json_path)
    baseline = _baseline(frame)
    selected = payload["selected_operating_point"]
    gate_shape, lam = selected["gate_shape"], float(selected["lambda"])
    curve = _eligible_curve(frame, baseline, gate_shape)
    selected_main = _operating_row(frame, PRIMARY_MODEL, gate_shape, lam)
    primary_gate_summary = _primary_gate_summary(frame, baseline)
    same_lambda_dominance = _same_lambda_dominance(curve)
    matched_curve_dominance = _matched_curve_dominance(curve)
    selected_comparison = _selected_comparison(
        frame, baseline, gate_shape, lam
    )
    crossings = _curve_crossings(curve)
    matched_strength = _matched_strength(curve, selected_main)
    neighbor_stability = _neighbor_stability(curve, lam)
    segments = _segment_table(selected_main, baseline)
    axis_summary = _axis_summary(selected_main, payload)

    stem = csv_path.stem
    paths = {
        "curve_table_csv": output_dir / f"{stem}_curve_diagnostic.csv",
        "primary_gate_summary_csv": output_dir / f"{stem}_primary_gate_summary.csv",
        "same_lambda_dominance_csv": output_dir / f"{stem}_same_lambda_dominance.csv",
        "matched_curve_dominance_csv": output_dir / f"{stem}_matched_curve_dominance.csv",
        "selected_comparison_csv": output_dir / f"{stem}_selected_comparison.csv",
        "crossings_csv": output_dir / f"{stem}_crossings.csv",
        "matched_strength_csv": output_dir / f"{stem}_matched_strength.csv",
        "neighbor_stability_csv": output_dir / f"{stem}_neighbor_stability.csv",
        "segment_csv": output_dir / f"{stem}_segment_diagnostic.csv",
        "summary_json": output_dir / f"{stem}_diagnostic_summary.json",
        "lambda_curve_png": output_dir / f"{stem}_lambda_curve.png",
        "strength_curve_png": output_dir / f"{stem}_strength_curve.png",
    }
    curve.to_csv(paths["curve_table_csv"], index=False)
    primary_gate_summary.to_csv(paths["primary_gate_summary_csv"], index=False)
    same_lambda_dominance.to_csv(paths["same_lambda_dominance_csv"], index=False)
    matched_curve_dominance.to_csv(paths["matched_curve_dominance_csv"], index=False)
    selected_comparison.to_csv(paths["selected_comparison_csv"], index=False)
    crossings.to_csv(paths["crossings_csv"], index=False)
    matched_strength.to_csv(paths["matched_strength_csv"], index=False)
    neighbor_stability.to_csv(paths["neighbor_stability_csv"], index=False)
    segments.to_csv(paths["segment_csv"], index=False)
    _plot_curve(curve, "lambda", paths["lambda_curve_png"], f"{stem}: lambda curve")
    _plot_curve(
        curve,
        "effective_strength",
        paths["strength_curve_png"],
        f"{stem}: effective-strength curve",
    )

    limitations = {
        "user_quadrant_available": False,
        "reason": (
            "현재 결과 JSON에는 사용자별 q_N/q_V와 추천·정답 지표가 없으므로 "
            "N/V 사분면 paired gain은 이 1차 무학습 진단에서 계산하지 않습니다."
        ),
        "inference": "seed 42 validation 탐색 결과이며 test 확증이 아닙니다.",
    }
    summary_payload = {
        "source_csv": str(csv_path),
        "source_json": str(json_path),
        "selected_operating_point": selected,
        "original_screening_decision": payload.get("screening_decision", {}),
        "selected_point_all_comparisons_positive": bool(
            (selected_comparison.absolute_delta > 0).all()
        ),
        "matched_strength_all_comparisons_positive": bool(
            (matched_strength.absolute_delta > 0).all()
        ),
        "neighbor_all_comparisons_positive": bool(
            neighbor_stability.primary_wins.all()
        ),
        "same_lambda_failed_point_count": int(
            (~same_lambda_dominance.primary_wins).sum()
        ),
        "matched_curve_failed_point_count": int(
            (~matched_curve_dominance.primary_wins).sum()
        ),
        "crossing_count": int(len(crossings)),
        "axis_summary": axis_summary,
        "limitations": limitations,
    }
    paths["summary_json"].write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {
        "curve": curve,
        "primary_gate_summary": primary_gate_summary,
        "same_lambda_dominance": same_lambda_dominance,
        "matched_curve_dominance": matched_curve_dominance,
        "selected_comparison": selected_comparison,
        "crossings": crossings,
        "matched_strength": matched_strength,
        "neighbor_stability": neighbor_stability,
        "segments": segments,
        "axis_summary": axis_summary,
        "limitations": limitations,
        "summary": summary_payload,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = diagnose_dual_results(args.csv_path, args.json_path, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str))
    print(json.dumps(result["paths"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
