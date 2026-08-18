import pandas as pd
import pytest

import lightgcn_clv_m4_interaction as M4


def test_m4_interaction_screening_config_is_validation_only_and_comparable(tmp_path):
    cfg = M4.configure_m4_interaction_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )

    assert cfg["DATASET"] == "dunnhumby"
    assert cfg["SEED_LIST"] == [42]
    assert cfg["ARCH"] == "pref_only"
    assert cfg["GRAPH_MODE"] == "binary"
    assert cfg["LOSS_MODES"] == ("user", "pair_contribution", "clv_pair")
    assert cfg["LOSS_STRENGTHS"] == {
        "user": 1.0,
        "pair_contribution": 0.25,
        "clv_pair": 0.25,
    }
    assert cfg["NEG_MODE"] == "uniform"
    assert cfg["MIN_USER_INTER"] == cfg["MIN_ITEM_INTER"] == 1
    assert cfg["EVAL_TEST"] is False
    assert cfg["EVAL_HOLDOUT"] is False


@pytest.mark.parametrize("key", ["EVAL_TEST", "EVAL_HOLDOUT"])
def test_m4_interaction_screening_rejects_protected_splits(tmp_path, key):
    cfg = M4.configure_m4_interaction_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    cfg[key] = True

    with pytest.raises(ValueError, match=key):
        M4.validate_screening_config(cfg)


def _screening_rows(*, main_top10=0.40, main_distinct=100.0):
    common = {
        "split": "val",
        "recall@10": 1.0,
        "ndcg@10": 1.0,
        "recall@20": 1.0,
        "ndcg@20": 1.0,
        "recall@50": 1.0,
        "ndcg@50": 1.0,
        "n_distinct@10": 100.0,
        "eff_catalog@10": 100.0,
        "top10_share@10": 0.40,
    }
    return pd.DataFrame(
        [
            {**common, "model_id": "m1_baseline", "revenue@10": 1.00},
            {**common, "model_id": "m4_user", "revenue@10": 0.99},
            {
                **common,
                "model_id": "m4_pair_contribution",
                "revenue@10": 1.01,
            },
            {
                **common,
                "model_id": "m4_clv_pair",
                "revenue@10": 1.02,
                "top10_share@10": main_top10,
                "n_distinct@10": main_distinct,
            },
        ]
    )


def test_m4_main_must_beat_m1_and_pair_only_under_accuracy_and_exposure_guards():
    decision = M4.screening_decision(_screening_rows())

    assert decision["arms"]["m4_clv_pair"]["passes_m1_screen"] is True
    assert decision["clv_specific_candidate"] is True
    assert decision["main_revenue@10_delta_vs_pair_control"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("main_top10", "main_distinct"), [(0.43, 100.0), (0.40, 94.0)]
)
def test_m4_main_fails_when_exposure_concentrates_beyond_fixed_guard(
    main_top10, main_distinct
):
    decision = M4.screening_decision(
        _screening_rows(main_top10=main_top10, main_distinct=main_distinct)
    )

    assert decision["arms"]["m4_clv_pair"]["passes_m1_screen"] is False
    assert decision["clv_specific_candidate"] is False
