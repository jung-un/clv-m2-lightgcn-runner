import numpy as np
import pandas as pd

import lightgcn_clv_fixed_segment_error_diagnostic_hm2y as diagnostic


def test_hm2y_purchase_occasion_axes_use_customer_date_and_amount():
    train = pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1],
            "t": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-03", "2020-01-02"]),
            "v": [2.0, 3.0, 5.0, 7.0],
        }
    )

    axes = diagnostic.build_purchase_occasion_axes(train, n_users=3)

    assert axes["n_behavior_score"].tolist() == [2.0, 1.0, 0.0]
    assert axes["v_behavior_score"].tolist() == [5.0, 7.0, 0.0]
    assert axes["clv_proxy"].tolist() == [10.0, 7.0, 0.0]
    assert axes["valid_user"].tolist() == [True, True, False]


def test_hm2y_preflight_keeps_test_and_holdout_closed():
    cfg = diagnostic.configure_hm2y_fixed_segment_error_diagnostic(
        out_dir="/tmp/diagnostic",
        m1_checkpoint_dir="/tmp/m1",
    )
    summary = diagnostic.preflight_summary(cfg)

    assert summary["dataset"] == "hm"
    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["test_executed"] is False
    assert summary["holdout_executed"] is False
    assert summary["new_item_task"] is True


def test_history_relations_use_only_requested_users_and_graph_history():
    occurrences = pd.DataFrame(
        {
            "user_idx": [1, 3],
            "item_idx": [1, 2],
            "category": ["tops", "shoes"],
        }
    )
    traits = pd.DataFrame(
        {
            "item_idx": [0, 1, 2, 3],
            "category": ["tops", "tops", "shoes", "bottoms"],
        }
    )
    # user 1 bought item 0; user 3 bought item 3.
    csr_ptr = np.array([0, 0, 1, 1, 2])
    csr_items = np.array([0, 3])
    item_embedding = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    result = diagnostic.attach_history_relations_from_graph(
        occurrences,
        csr_ptr=csr_ptr,
        csr_items=csr_items,
        item_traits=traits,
        item_embedding=item_embedding,
    )

    assert result.history_category_overlap.tolist() == [1.0, 0.0]
    assert result.history_embedding_cosine.tolist() == [1.0, 1.0]


def test_checkpoint_path_uses_exact_hm2y_m1_hash(tmp_path):
    base_cfg = {
        "OUT_DIR": str(tmp_path),
        "DATASET": "hm",
    }
    prepared = {"base_cfg": base_cfg}
    cfg = diagnostic.configure_hm2y_fixed_segment_error_diagnostic(
        out_dir=str(tmp_path / "diagnostic"),
        m1_checkpoint_dir=str(tmp_path),
    )
    original = diagnostic.v3.cfg_hash
    diagnostic.v3.cfg_hash = lambda *_args: "abc12345"
    try:
        path = diagnostic._checkpoint_path(prepared, cfg)
    finally:
        diagnostic.v3.cfg_hash = original

    assert path == tmp_path / "ckpt_pref_only_hm_s42_abc12345.pt"
