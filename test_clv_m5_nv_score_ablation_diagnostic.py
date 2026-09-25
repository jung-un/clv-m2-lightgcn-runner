import numpy as np
import pytest
import torch

import clv_m5_nv_score_ablation_diagnostic as diagnostic


class TinyM5:
    id_dim = 1
    axis_dim = 1

    def __init__(self, n_bonus=2.0):
        self.user = torch.tensor([[1.0, 1.0, 1.0]])
        self.item = torch.zeros((60, 3))
        self.item[:, 0] = torch.arange(100.0, 40.0, -1.0)
        self.item[10, 1] = n_bonus

    def eval(self):
        return self

    def embeddings(self):
        return self.user, self.item, torch.zeros((1, 1)), torch.zeros((60, 1))


def prepared():
    cache = type("Cache", (), {"users": np.array([0]), "seg": np.array(["고CLV"]),
                                  "gt": {0: [10]}, "rev": {0: [5.0]}})()
    return dict(cache=cache, data=dict(n_users=1, n_items=60,
                csr_ptr=np.array([0, 0]), csr_items=np.array([], dtype=np.int64)),
                base_cfg={"EVAL_BATCH": 16})


def test_same_checkpoint_nv_term_moves_a_truth_across_top10():
    full, id_only, candidates, error = diagnostic.rank_and_explain(TinyM5(), prepared())
    assert id_only[0, :10].tolist() == list(range(10))
    assert 10 in full[0, :10] and 9 not in full[0, :10]
    assert error < 1e-5
    changed = candidates.set_index("item")
    assert changed.loc[10, "top10_change_from_id"] == "entered_full"
    assert changed.loc[10, "n_score"] == pytest.approx(2.0)
    assert changed.loc[10, "v_score"] == pytest.approx(0.0)
    assert changed.loc[9, "top10_change_from_id"] == "exited_full"
    truth, users, summary = diagnostic._movement_tables(prepared(), id_only, full)
    assert truth.loc[0, "top10_change"] == "gained"
    assert summary.set_index("segment").loc["전체", "weighted_net"] == pytest.approx(5.0)
    assert users.loc[0, "top10_truth_gained"] == 1


def test_zero_nv_score_has_no_direct_top10_effect():
    full, id_only, candidates, error = diagnostic.rank_and_explain(TinyM5(0.0), prepared())
    np.testing.assert_array_equal(full, id_only)
    assert candidates.empty
    assert error == 0.0


def test_catalog_smaller_than_top50_is_rejected():
    prep = prepared()
    prep["data"]["n_items"] = 49
    with pytest.raises(ValueError, match="dimensions/catalog"):
        diagnostic.rank_and_explain(TinyM5(), prep)
