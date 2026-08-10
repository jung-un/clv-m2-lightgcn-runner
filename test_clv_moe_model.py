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

    def embeddings(self, need_value=True):
        return self.E_u.weight, self.E_i.weight, None, None


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


def test_constant_gate_removes_user_specific_routing_only():
    model = _model(control="constant_gate")
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
