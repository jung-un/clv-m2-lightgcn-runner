import numpy as np
import torch
import torch.nn.functional as F

from clv_moe_features import ItemProfileArtifact, UserProfileArtifact


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.E_u = torch.nn.Embedding(4, 8)
        self.E_i = torch.nn.Embedding(6, 8)
        self.register_buffer("adj", torch.zeros(10, 10))
        self.embedding_calls = 0

    def embeddings(self, need_value=True):
        self.embedding_calls += 1
        return self.E_u.weight, self.E_i.weight, None, None


class _BaseWithUnusedValueParameter(_Base):
    def __init__(self):
        super().__init__()
        self.unused_value = torch.nn.Parameter(torch.ones(17))

    def pref_params(self):
        return [self.E_u.weight, self.E_i.weight]


def _artifacts(invalid_last=False):
    rng = np.random.default_rng(4)
    valid = np.ones(4, bool)
    if invalid_last:
        valid[-1] = False
    user_values = rng.normal(size=(4, 51)).astype(np.float32)
    user_values[~valid] = 0
    user = UserProfileArtifact(user_values, valid, tuple(f"u{x}" for x in range(51)))
    item = ItemProfileArtifact(
        numeric=rng.normal(size=(6, 6)).astype(np.float32),
        category_ids=np.array([1, 1, 2, 2, 3, 0], np.int64),
        valid_item=np.ones(6, bool),
        numeric_names=tuple(f"i{x}" for x in range(6)),
        n_categories=4,
    )
    return user, item


def _model(control="clv", seed=42, invalid_last=False, base=None):
    from clv_moe_model import CLVMixtureEmbeddingModel

    user, item = _artifacts(invalid_last)
    return CLVMixtureEmbeddingModel(
        base or _Base(), user, item, control=control, seed=seed
    )


def _single(control, seed=42):
    return _model(control=control, seed=seed)


def test_single_full_is_exact_legacy_single_adapter_alias():
    legacy = _single("single_adapter")
    full = _single("single_full")
    assert legacy.single_variant == full.single_variant == "single_full"
    torch.testing.assert_close(legacy.routed_profile, full.routed_profile)
    torch.testing.assert_close(legacy.item_numeric, full.item_numeric)
    torch.testing.assert_close(
        legacy.score_all(torch.arange(4), 1.0),
        full.score_all(torch.arange(4), 1.0),
    )


def test_single_zero_user_preserves_mask_and_zeros_only_user_profile():
    full = _single("single_full")
    zero = _single("single_zero_user")
    torch.testing.assert_close(zero.routed_profile, torch.zeros_like(zero.routed_profile))
    torch.testing.assert_close(zero.item_numeric, full.item_numeric)
    assert torch.equal(zero.has_profile, full.has_profile)


def test_single_zero_item_preserves_mask_and_zeros_item_side_features():
    full = _single("single_full")
    zero = _single("single_zero_item")
    torch.testing.assert_close(zero.item_numeric, torch.zeros_like(zero.item_numeric))
    assert torch.equal(zero.item_category_ids, torch.zeros_like(zero.item_category_ids))
    assert torch.equal(zero.valid_item, full.valid_item)
    torch.testing.assert_close(zero.routed_profile, full.routed_profile)


def test_single_base_only_zeros_both_added_inputs_without_disabling_residual():
    model = _single("single_base_only")
    assert torch.count_nonzero(model.routed_profile) == 0
    assert torch.count_nonzero(model.item_numeric) == 0
    assert torch.count_nonzero(model.item_category_ids) == 0
    assert model.has_profile.all() and model.valid_item.all()
    assert not torch.equal(
        model.score_all(torch.arange(4), 1.0),
        model.base_score_all(torch.arange(4)),
    )


def test_single_shuffled_user_is_seeded_permutation_of_valid_profiles():
    full = _single("single_full")
    a = _single("single_shuffled_user", seed=42)
    b = _single("single_shuffled_user", seed=42)
    torch.testing.assert_close(a.routed_profile, b.routed_profile)
    assert not torch.equal(a.routed_profile, full.routed_profile)
    assert sorted(a.routed_profile[:, 0].tolist()) == sorted(
        full.routed_profile[:, 0].tolist()
    )


def test_all_single_variants_have_identical_parameter_names_and_shapes():
    controls = [
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_zero_item",
        "single_base_only",
    ]
    signatures = []
    for control in controls:
        model = _single(control)
        signatures.append(
            [
                (name, tuple(parameter.shape))
                for name, parameter in model.named_parameters()
            ]
        )
        assert model.expert_count == 1
        torch.testing.assert_close(
            model.score_all(torch.arange(4), 0.0),
            model.base_score_all(torch.arange(4)),
            rtol=0,
            atol=0,
        )
    assert signatures.count(signatures[0]) == len(signatures)


def test_single_variant_audit_hashes_inputs_masks_and_capacity():
    from lightgcn_clv_single import _variant_audit

    model = _single("single_shuffled_user", seed=42)
    audit = _variant_audit(model, "m1-state")
    assert audit["starting_base_state_hash"] == "m1-state"
    assert audit["routed_profile_sha256"] != audit["original_profile_sha256"]
    assert len(audit["item_numeric_sha256"]) == 64
    assert len(audit["item_category_ids_sha256"]) == 64
    assert len(audit["has_profile_sha256"]) == 64
    assert len(audit["valid_item_sha256"]) == 64
    assert audit["adapter_parameter_count"] > 0
    assert audit["joint_trainable_parameter_count"] > audit["adapter_parameter_count"]


def test_single_variant_audit_counts_only_optimizer_base_parameters():
    from lightgcn_clv_single import _variant_audit

    model = _model(control="single_full", base=_BaseWithUnusedValueParameter())
    audit = _variant_audit(model, "m1-state")
    expected_base = model.base_model.E_u.weight.numel() + model.base_model.E_i.weight.numel()
    assert audit["base_parameter_count"] == expected_base
    assert audit["joint_trainable_parameter_count"] == (
        audit["adapter_parameter_count"] + expected_base
    )


def test_three_experts_generate_user_and_item_embeddings():
    model = _model()
    user_experts, item_experts, gate = model.expert_embeddings(torch.arange(4))
    assert user_experts.shape == (4, 3, 16)
    assert item_experts.shape == (6, 3, 16)
    assert gate.shape == (4, 3)
    torch.testing.assert_close(gate.sum(1), torch.ones(4))


def test_lambda_zero_exactly_equals_external_m1_scores():
    model = _model()
    users = torch.arange(4)
    torch.testing.assert_close(
        model.score_all(users, lam=0.0),
        model.base_score_all(users),
        rtol=0,
        atol=0,
    )


def test_score_is_mixture_of_expert_embedding_inner_products():
    model = _model()
    users = torch.tensor([0, 2])
    user_experts, item_experts, gate = model.expert_embeddings(users)
    expected = model.base_score_all(users) + torch.einsum(
        "uk,ukd,ikd->ui", gate, user_experts, item_experts
    )
    torch.testing.assert_close(model.score_all(users, 1.0), expected)


def test_flattened_value_embeddings_reproduce_moe_score_for_v3_evaluator():
    model = _model()
    base_user, base_item, value_user, value_item = model.embeddings()
    direct = model.score_all(torch.arange(4), 1.0)
    flattened = base_user @ base_item.T + value_user @ value_item.T
    torch.testing.assert_close(flattened, direct)


def test_constant_gate_removes_user_specific_routing_only():
    model = _model(control="constant_gate")
    assert model.constant_gate_logits.shape == (3,)
    gate = model.routing_weights(torch.arange(4))
    torch.testing.assert_close(gate, gate[:1].expand_as(gate))


def test_shuffled_clv_is_seed_deterministic_and_nonidentity():
    a = _model(control="shuffled_clv", seed=42).routed_profile
    b = _model(control="shuffled_clv", seed=42).routed_profile
    torch.testing.assert_close(a, b)
    assert not torch.equal(a, _model(control="clv").routed_profile)


def test_user_without_clv_profile_gets_no_moe_residual():
    model = _model(invalid_last=True)
    user = torch.tensor([3])
    torch.testing.assert_close(
        model.score_all(user, 1.0), model.base_score_all(user), rtol=0, atol=0
    )


def test_bpr_is_plain_mean_without_clv_sample_weights():
    model = _model()
    users = torch.tensor([0, 1])
    positives = torch.tensor([1, 2])
    negatives = torch.tensor([3, 4])
    expected = -F.logsigmoid(
        model.score_pairs(users, positives, 1.0)
        - model.score_pairs(users, negatives, 1.0)
    ).mean()
    torch.testing.assert_close(
        model.bpr_loss(users, positives, negatives, 1.0), expected
    )


def test_bpr_computes_base_graph_embeddings_once_per_batch():
    model = _model()
    model.base_model.embedding_calls = 0
    model.bpr_loss(
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        torch.tensor([3, 4]),
        1.0,
    )
    assert model.base_model.embedding_calls == 1


def test_evaluation_embedding_export_computes_base_graph_once():
    model = _model()
    model.base_model.embedding_calls = 0
    model.embeddings()
    assert model.base_model.embedding_calls == 1


def test_model_reuses_base_graph_and_experts_have_no_graph():
    base = _Base()
    model = _model(base=base)
    assert model.base_model.adj is base.adj
    assert all(not hasattr(expert, "adj") for expert in model.experts)


def test_base_and_adapter_parameter_sets_are_disjoint():
    model = _model()
    base = {id(parameter) for parameter in model.base_parameters()}
    adapters = {id(parameter) for parameter in model.adapter_parameters()}
    assert base
    assert adapters
    assert base.isdisjoint(adapters)


def test_single_adapter_matches_moe_capacity_without_a_router():
    mixture = _model(control="clv")
    single = _model(control="single_adapter")
    mixture_count = sum(p.numel() for p in mixture.adapter_parameters())
    single_count = sum(p.numel() for p in single.adapter_parameters())
    assert 0.95 <= single_count / mixture_count <= 1.05
    assert single.expert_count == 1
    gate = single.routing_weights(torch.arange(4))
    torch.testing.assert_close(gate, torch.ones(4, 1))


def test_moe_diagnostics_report_routing_and_specialization():
    from clv_moe_model import moe_diagnostics

    diagnostics = moe_diagnostics(_model(), seed=42, max_users=4, max_items=5)
    assert set(diagnostics) >= {
        "gate_entropy_mean",
        "expert_usage_mean",
        "expert_user_cosine",
        "expert_item_cosine",
        "expert_score_correlation",
        "residual_to_base_score_std",
        "parameter_match_ratio",
    }
    assert len(diagnostics["expert_usage_mean"]) == 3
    assert np.isfinite(diagnostics["gate_entropy_mean"])
    assert np.isfinite(diagnostics["residual_to_base_score_std"])


def test_moe_diagnostics_never_scores_the_full_catalog():
    from clv_moe_model import moe_diagnostics

    model = _model()

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must use only sampled users and items")

    model.score_all = forbidden
    diagnostics = moe_diagnostics(model, seed=42, max_users=2, max_items=3)
    assert diagnostics["diagnostic_users"] == 2
    assert diagnostics["diagnostic_items"] == 3
