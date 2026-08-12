import numpy as np
import pandas as pd
import torch

from clv_moe_features import UserProfileArtifact


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(7)
        self.E_u = torch.nn.Embedding(3, 8)
        self.E_i = torch.nn.Embedding(4, 8)

    def embeddings(self, need_value=True):
        return self.E_u.weight, self.E_i.weight, None, None


def _train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2],
            "i_idx": [0, 0, 1, 0, 2, 3],
            "cat_idx": [10, 10, 10, 10, 20, 20],
            "b_raw": ["a", "b", "b", "c", "c", "d"],
            "t": [1, 4, 4, 2, 2, 3],
            "up": [1.0, 1.0, 3.0, 1.0, 5.0, 8.0],
            "v": [1.0, 1.0, 3.0, 1.0, 5.0, 8.0],
        }
    )


def _user_profile():
    names = (
        *(f"repurchase_feature_{i}" for i in range(3)),
        *(f"monetary_feature_{i}" for i in range(2)),
        "pred_log_future_transactions",
        "pred_log_transaction_value",
        "pred_log_clv_proxy",
    )
    values = np.arange(24, dtype=np.float32).reshape(3, 8) / 10
    return UserProfileArtifact(values, np.ones(3, bool), tuple(names))


def test_percentile_gate_shapes_cover_high_equal_and_low_directions():
    from clv_dual_axis_model import apply_gate_shape, fixed_percentile_ranks

    q_n, q_v = fixed_percentile_ranks(
        np.array([1.0, 3.0, 2.0]),
        np.array([30.0, 10.0, 20.0]),
        np.ones(3, bool),
    )
    high_n = apply_gate_shape(q_n, "high")
    equal_n = apply_gate_shape(q_n, "equal")
    low_n = apply_gate_shape(q_n, "low")
    assert np.argsort(q_n).tolist() == [0, 2, 1]
    assert np.argsort(q_v).tolist() == [1, 2, 0]
    np.testing.assert_allclose(equal_n, np.ones(3))
    np.testing.assert_allclose(high_n + low_n, np.full(3, 2.0))
    assert np.isclose(high_n.mean(), 1.0)
    assert np.isclose(low_n.mean(), 1.0)


def test_item_axes_have_disjoint_named_features():
    from clv_dual_axis_model import build_dual_item_profiles

    profile = build_dual_item_profiles(_train(), n_items=4, is_date=False)
    assert profile.activity_names == (
        "repeat_purchase_share",
        "log_median_repeat_gap",
        "repeat_gap_valid",
    )
    assert "price_percentile" in profile.value_names
    assert not set(profile.activity_names).intersection(profile.value_names)
    assert profile.activity.shape[0] == profile.value.shape[0] == 4
    assert np.isfinite(profile.activity).all()
    assert np.isfinite(profile.value).all()


def test_dual_model_freezes_m1_and_lambda_zero_is_exact_base():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_ranks,
    )

    base = _Base()
    profile = _user_profile()
    item = build_dual_item_profiles(_train(), 4, False)
    gates = fixed_percentile_ranks(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
        profile.valid_user,
    )
    model = CLVDualAxisEmbeddingModel(
        base, profile, item, *gates, control="dual_clv_fixed", seed=42
    )
    users = torch.arange(3)

    assert not any(parameter.requires_grad for parameter in model.base_model.parameters())
    torch.testing.assert_close(
        model.score_all(users, 0.0), model.base_score_all(users), rtol=0, atol=0
    )
    assert not torch.equal(model.score_all(users, 1.0), model.base_score_all(users))


def test_controls_jointly_shuffle_user_clv_or_remove_all_added_information():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_ranks,
    )

    profile = _user_profile()
    item = build_dual_item_profiles(_train(), 4, False)
    gates = fixed_percentile_ranks(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
        profile.valid_user,
    )
    models = {
        control: CLVDualAxisEmbeddingModel(
            _Base(), profile, item, *gates, control=control, seed=42
        )
        for control in ("dual_clv_fixed", "dual_shuffled_user", "dual_adapter_only")
    }
    signatures = {
        control: [(name, tuple(p.shape)) for name, p in model.named_parameters()]
        for control, model in models.items()
    }
    assert len({str(value) for value in signatures.values()}) == 1
    full, shuffled, adapter_only = (
        models["dual_clv_fixed"],
        models["dual_shuffled_user"],
        models["dual_adapter_only"],
    )
    full_rows = torch.cat(
        [full.user_activity, full.user_value, full.q_n[:, None], full.q_v[:, None]], 1
    )
    shuffled_rows = torch.cat(
        [
            shuffled.user_activity,
            shuffled.user_value,
            shuffled.q_n[:, None],
            shuffled.q_v[:, None],
        ],
        1,
    )
    assert not torch.equal(full_rows, shuffled_rows)
    torch.testing.assert_close(
        full_rows[torch.argsort(full_rows[:, 0])],
        shuffled_rows[torch.argsort(shuffled_rows[:, 0])],
    )
    assert torch.count_nonzero(adapter_only.user_activity) == 0
    assert torch.count_nonzero(adapter_only.user_value) == 0
    assert torch.count_nonzero(adapter_only.item_activity) == 0
    assert torch.count_nonzero(adapter_only.item_value) == 0
    for shape in ("high", "equal", "low"):
        g_n, g_v = adapter_only.gate_values(shape)
        torch.testing.assert_close(g_n, torch.ones_like(g_n))
        torch.testing.assert_close(g_v, torch.ones_like(g_v))


def test_bpr_is_plain_and_uses_frozen_base_cache():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_ranks,
    )

    base = _Base()
    base.calls = 0
    original = base.embeddings

    def counted(need_value=True):
        base.calls += 1
        return original(need_value)

    base.embeddings = counted
    profile = _user_profile()
    item = build_dual_item_profiles(_train(), 4, False)
    gates = fixed_percentile_ranks(
        np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), profile.valid_user
    )
    model = CLVDualAxisEmbeddingModel(base, profile, item, *gates)
    base.calls = 0
    loss = model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
        lam=1.0,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert base.calls == 0


def test_training_is_gate_neutral_but_evaluation_gate_changes_scores():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_ranks,
    )

    profile = _user_profile()
    model = CLVDualAxisEmbeddingModel(
        _Base(),
        profile,
        build_dual_item_profiles(_train(), 4, False),
        *fixed_percentile_ranks(
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
            profile.valid_user,
        ),
    )
    users = torch.tensor([0, 1])
    positives = torch.tensor([0, 1])
    negatives = torch.tensor([2, 3])
    model.set_gate_shape("high")
    high_loss = model.bpr_loss(users, positives, negatives)
    high_scores = model.score_all(users, 1.0)
    model.set_gate_shape("low")
    low_loss = model.bpr_loss(users, positives, negatives)
    low_scores = model.score_all(users, 1.0)

    torch.testing.assert_close(high_loss, low_loss, rtol=0, atol=0)
    assert not torch.equal(high_scores, low_scores)
    diagnostics = model.axis_diagnostics("high", max_users=3, max_items=4)
    assert diagnostics["effective_total_ratio"] > 0
    assert -1 <= diagnostics["expert_score_corr"] <= 1
    assert 0 <= diagnostics["expert_top10_jaccard"] <= 1


def test_eval_axis_mask_reuses_same_experts_without_retraining():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_ranks,
    )

    profile = _user_profile()
    model = CLVDualAxisEmbeddingModel(
        _Base(),
        profile,
        build_dual_item_profiles(_train(), 4, False),
        *fixed_percentile_ranks(
            np.array([1.0, 2.0, 3.0]),
            np.array([3.0, 2.0, 1.0]),
            profile.valid_user,
        ),
    )
    users = torch.tensor([0, 1])
    base = model.base_score_all(users)
    full = model.score_all(users, 1.0, "equal")
    model.set_eval_axes("n_only")
    n_only = model.score_all(users, 1.0, "equal")
    _, _, n_user, n_item = model.embeddings()
    assert torch.count_nonzero(n_user[:, n_user.shape[1] // 2 :]) == 0
    assert torch.count_nonzero(n_item[:, n_item.shape[1] // 2 :]) == 0
    model.set_eval_axes("v_only")
    v_only = model.score_all(users, 1.0, "equal")
    _, _, v_user, v_item = model.embeddings()
    assert torch.count_nonzero(v_user[:, : v_user.shape[1] // 2]) == 0
    assert torch.count_nonzero(v_item[:, : v_item.shape[1] // 2]) == 0

    torch.testing.assert_close(full - base, (n_only - base) + (v_only - base))
    model.set_eval_axes("n_plus_v")
    torch.testing.assert_close(model.score_all(users, 1.0, "equal"), full)
