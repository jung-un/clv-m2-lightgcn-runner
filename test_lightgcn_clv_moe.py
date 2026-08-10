import numpy as np
import torch

from clv_moe_features import ItemProfileArtifact, UserProfileArtifact
from clv_moe_model import CLVMixtureEmbeddingModel


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(9)
        self.E_u = torch.nn.Embedding(4, 8)
        self.E_i = torch.nn.Embedding(6, 8)

    def embeddings(self, need_value=True):
        return self.E_u.weight, self.E_i.weight, None, None

    def pref_params(self):
        return list(self.E_u.parameters()) + list(self.E_i.parameters())


def _model():
    rng = np.random.default_rng(10)
    users = UserProfileArtifact(
        values=rng.normal(size=(4, 51)).astype(np.float32),
        valid_user=np.ones(4, bool),
        feature_names=tuple(f"u{x}" for x in range(51)),
    )
    items = ItemProfileArtifact(
        numeric=rng.normal(size=(6, 6)).astype(np.float32),
        category_ids=np.array([1, 1, 2, 2, 3, 3], np.int64),
        valid_item=np.ones(6, bool),
        numeric_names=tuple(f"i{x}" for x in range(6)),
        n_categories=4,
    )
    return CLVMixtureEmbeddingModel(_Base(), users, items, seed=42)


def _data():
    tr_u = np.array([0, 1, 2, 3], np.int64)
    tr_i = np.array([0, 1, 2, 3], np.int64)
    return {
        "tr_u": tr_u,
        "tr_i": tr_i,
        "n_items": 6,
        "pos_key": np.sort(tr_u * 6 + tr_i),
        "item_cat": np.array([0, 0, 1, 1, 2, 2], np.int64),
        "cat_items": [
            np.array([0, 1], np.int64),
            np.array([2, 3], np.int64),
            np.array([4, 5], np.int64),
        ],
    }


def _base_cfg():
    return {"BATCH_SIZE": 2, "NEG_MODE": "uniform"}


class _IncreasingRecall:
    def __init__(self):
        self.value = 0

    def __call__(self, model):
        self.value += 1
        return float(self.value)


def test_default_screening_is_seed42_validation_only():
    import lightgcn_clv_moe as moe

    cfg = moe.configure_moe_run("dunnhumby")
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert cfg.expert_count == 3 and cfg.frozen_epochs == 5
    assert cfg.adapter_lr == 5e-4 and cfg.base_lr == 5e-5
    assert cfg.lambda_eval == (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)


def test_joint_warm_updates_only_adapters_before_epoch_six():
    import lightgcn_clv_moe as moe

    cfg = moe.configure_moe_run(
        "dunnhumby", max_epochs=6, patience=10, adapter_lr=1e-3, base_lr=1e-3
    )
    records = moe.train_moe(
        _model(), _data(), _base_cfg(), cfg, seed=42, eval_recall=_IncreasingRecall()
    )
    assert records["base_updates_by_epoch"][:5] == [0, 0, 0, 0, 0]
    assert records["base_updates_by_epoch"][5] == 2
    assert records["loss"] == "plain_bpr"


def test_frozen_moe_preserves_m1_hash():
    import lightgcn_clv_moe as moe

    model = _model()
    cfg = moe.configure_moe_run(
        "dunnhumby", frozen_epochs=0, max_epochs=3, patience=10
    )
    before = moe.state_hash(model.base_model)
    stats = moe.train_moe(
        model,
        _data(),
        _base_cfg(),
        cfg,
        seed=42,
        eval_recall=_IncreasingRecall(),
        freeze_base=True,
    )
    assert moe.state_hash(model.base_model) == before
    assert stats["base_updates"] == 0


def test_pref_continue_has_exact_matched_base_updates():
    import lightgcn_clv_moe as moe

    base = _Base()
    cfg = moe.configure_moe_run("dunnhumby", base_lr=1e-3)
    before = moe.state_hash(base)
    stats = moe.train_pref_continue(
        base, _data(), _base_cfg(), cfg, seed=42, target_base_updates=3
    )
    assert stats["base_updates"] == 3
    assert stats["loss"] == "plain_bpr"
    assert moe.state_hash(base) != before
