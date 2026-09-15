"""Four-arm M1/M2/M4/M5 comparison with single-negative BPR sampling."""

from __future__ import annotations

import pandas as pd

import lightgcn_clv_m5_nv_economic_positive_weight as legacy


M1_MODEL_ID = "m1_lightgcn_single_negative_bpr"
M2_MODEL_ID = "m2_explicit_nv_economic_embedding_single_negative_bpr"
M4P_MODEL_ID = "m4_clv_personalized_economic_positive_weight_single_negative_bpr"
M5_MODEL_ID = "m5_explicit_nv_personalized_economic_positive_weight_single_negative_bpr"
MODEL_IDS = (M1_MODEL_ID, M2_MODEL_ID, M4P_MODEL_ID, M5_MODEL_ID)

ACCURACY_METRICS = legacy.ACCURACY_METRICS
PRIMARY_METRIC = legacy.PRIMARY_METRIC
M5EconomicPositiveConfig = legacy.M5EconomicPositiveConfig

# Reuse the exact explicit-N/V inputs, model, optimizer, and loss implementation
# from the prior ten-seed experiment. K=1 is fixed by the new runner config.
build_nv_economic_inputs = legacy.build_nv_economic_inputs
_arm_hash = legacy._arm_hash
_arm_paths = legacy._arm_paths
_build_model = legacy._build_model
_train_arm = legacy._train_arm


def arm_specifications(prepared: dict, cfg: M5EconomicPositiveConfig) -> list[dict]:
    """Return the matched 2x2 M2-by-M4 factorial without attribution arms."""

    return [
        {
            "model_id": M1_MODEL_ID,
            "role": "single_negative_bpr_m1",
            "rho": 0.0,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "observed",
        },
        {
            "model_id": M2_MODEL_ID,
            "role": "single_negative_bpr_m2_explicit_nv",
            "rho": cfg.rho,
            "weighted": False,
            "assignment": prepared,
            "assignment_name": "observed",
        },
        {
            "model_id": M4P_MODEL_ID,
            "role": "single_negative_bpr_m4_personalized_positive_weight",
            "rho": 0.0,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed",
        },
        {
            "model_id": M5_MODEL_ID,
            "role": "single_negative_bpr_m5_explicit_nv_plus_m4",
            "rho": cfg.rho,
            "weighted": True,
            "assignment": prepared,
            "assignment_name": "observed",
        },
    ]


def interaction_rows(metric_rows: dict[str, dict]) -> pd.DataFrame:
    """Compute the matched M2-by-M4 difference-in-differences."""

    m1 = metric_rows[M1_MODEL_ID]
    m2 = metric_rows[M2_MODEL_ID]
    m4 = metric_rows[M4P_MODEL_ID]
    m5 = metric_rows[M5_MODEL_ID]
    metrics = ACCURACY_METRICS + (
        PRIMARY_METRIC,
        "price_purchase_amount_weighted_hit@10",
    )
    rows = []
    for metric in metrics:
        if not all(metric in values for values in (m1, m2, m4, m5)):
            continue
        m2_effect = float(m2[metric] - m1[metric])
        rows.append(
            {
                "metric": metric,
                "m2_effect": m2_effect,
                "m4_effect": float(m4[metric] - m1[metric]),
                "m5_effect": float(m5[metric] - m1[metric]),
                "m2_increment_given_m4": float(m5[metric] - m4[metric]),
                "interaction_effect": float((m5[metric] - m4[metric]) - m2_effect),
            }
        )
    return pd.DataFrame(rows)


def screening_reading(metric_rows: dict[str, dict]) -> dict:
    """Describe the four-arm result without selecting a model on exposed test."""

    m1 = metric_rows[M1_MODEL_ID]
    m2 = metric_rows[M2_MODEL_ID]
    m4 = metric_rows[M4P_MODEL_ID]
    m5 = metric_rows[M5_MODEL_ID]
    metrics = ACCURACY_METRICS + (
        PRIMARY_METRIC,
        "price_purchase_amount_weighted_hit@10",
    )

    def deltas(model: dict, reference: dict) -> dict[str, float]:
        return {
            metric: float(model[metric] - reference[metric])
            for metric in metrics
            if metric in model and metric in reference
        }

    return {
        "descriptive_only": True,
        "model_selection_permitted": False,
        "m2_minus_m1": deltas(m2, m1),
        "m4_minus_m1": deltas(m4, m1),
        "m5_minus_m1": deltas(m5, m1),
        "m5_minus_m4": deltas(m5, m4),
        "interaction": interaction_rows(metric_rows).set_index("metric")[
            "interaction_effect"
        ].to_dict(),
        "protocol_note": (
            "post-hoc rerun after correcting every arm from K=5 mean BPR to "
            "one uniformly sampled negative per positive; the exposed test must "
            "not be used for further tuning"
        ),
        "statistical_note": (
            "report ten-seed means and paired differences; do not claim a fresh "
            "confirmatory test because this test interval was already exposed"
        ),
    }
