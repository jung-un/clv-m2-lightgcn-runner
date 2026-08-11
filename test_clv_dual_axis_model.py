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


def test_fixed_gates_are_monotone_mean_one_and_jointly_shuffled():
    from clv_dual_axis_model import fixed_percentile_gates

    g_n, g_v = fixed_percentile_gates(
        np.array([1.0, 3.0, 2.0]),
        np.array([30.0, 10.0, 20.0]),
        np.ones(3, bool),
    )
    assert np.argsort(g_n).tolist() == [0, 2, 1]
    assert np.argsort(g_v).tolist() == [1, 2, 0]
    assert np.isclose(g_n.mean(), 1.0)
    assert np.isclose(g_v.mean(), 1.0)


def test_item_axes_have_disjoint_named_features():
    from clv_dual_axis_model import build_dual_item_profiles

    profile = build_dual_item_profiles(_train(), n_items=4, is_date=False)
    assert "repeat_purchase_share" in profile.activity_names
    assert "price_percentile" in profile.value_names
    assert not set(profile.activity_names).intersection(profile.value_names)
    assert profile.activity.shape[0] == profile.value.shape[0] == 4
    assert np.isfinite(profile.activity).all()
    assert np.isfinite(profile.value).all()


def test_dual_model_freezes_m1_and_lambda_zero_is_exact_base():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_gates,
    )

    base = _Base()
    profile = _user_profile()
    item = build_dual_item_profiles(_train(), 4, False)
    gates = fixed_percentile_gates(
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


def test_controls_preserve_parameter_shapes_and_change_only_planned_inputs():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_gates,
    )

    profile = _user_profile()
    item = build_dual_item_profiles(_train(), 4, False)
    gates = fixed_percentile_gates(
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 2.0, 1.0]),
        profile.valid_user,
    )
    models = {
        control: CLVDualAxisEmbeddingModel(
            _Base(), profile, item, *gates, control=control, seed=42
        )
        for control in ("dual_clv_fixed", "dual_shuffled_gate", "dual_base_only")
    }
    signatures = {
        control: [(name, tuple(p.shape)) for name, p in model.named_parameters()]
        for control, model in models.items()
    }
    assert len({str(value) for value in signatures.values()}) == 1
    full, shuffled, base_only = (
        models["dual_clv_fixed"],
        models["dual_shuffled_gate"],
        models["dual_base_only"],
    )
    assert not torch.equal(full.g_n, shuffled.g_n)
    pairs_full = sorted(zip(full.g_n.tolist(), full.g_v.tolist()))
    pairs_shuffled = sorted(zip(shuffled.g_n.tolist(), shuffled.g_v.tolist()))
    assert pairs_full == pairs_shuffled
    assert torch.count_nonzero(base_only.user_activity) == 0
    assert torch.count_nonzero(base_only.user_value) == 0
    assert torch.count_nonzero(base_only.item_activity) == 0
    assert torch.count_nonzero(base_only.item_value) == 0
    torch.testing.assert_close(base_only.g_n, torch.ones_like(base_only.g_n))
    torch.testing.assert_close(base_only.g_v, torch.ones_like(base_only.g_v))


def test_bpr_is_plain_and_uses_frozen_base_cache():
    from clv_dual_axis_model import (
        CLVDualAxisEmbeddingModel,
        build_dual_item_profiles,
        fixed_percentile_gates,
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
    gates = fixed_percentile_gates(
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
