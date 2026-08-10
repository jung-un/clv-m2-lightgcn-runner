import json
from pathlib import Path

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
    assert records["base_updates_at_best"] == 2
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


def _baseline_metrics():
    return {
        f"{metric}@{k}": 1.0
        for metric in ("recall", "ndcg")
        for k in (10, 20, 50)
    } | {"revenue@10": 1.0}


def _lambda_row(lam, baseline, *, revenue=1.0, recall50=None):
    row = {"lambda": lam, **baseline}
    row["revenue@10"] = revenue
    if recall50 is not None:
        row["recall@50"] = recall50
    return row


def test_select_lambda_uses_all_six_accuracy_guardrails():
    import lightgcn_clv_moe as moe

    base = _baseline_metrics()
    rows = [
        _lambda_row(0.0, base),
        _lambda_row(0.5, base, revenue=1.1),
        _lambda_row(1.0, base, revenue=1.2, recall50=0.989),
    ]
    selected, table = moe.select_lambda(rows, base, tolerance=0.01)
    assert selected == 0.5
    assert not bool(table.loc[table["lambda"].eq(1.0), "eligible"].iat[0])
    assert table.attrs["success"] is True


def test_select_lambda_fallback_zero_is_not_success():
    import lightgcn_clv_moe as moe

    base = _baseline_metrics()
    selected, table = moe.select_lambda(
        [_lambda_row(0.0, base), _lambda_row(0.5, base, recall50=0.5)], base
    )
    assert selected == 0.0
    assert table.attrs["success"] is False


def test_select_lambda_prefers_smaller_positive_lambda_on_revenue_tie():
    import lightgcn_clv_moe as moe

    base = _baseline_metrics()
    selected, _ = moe.select_lambda(
        [
            _lambda_row(0.0, base),
            _lambda_row(0.25, base, revenue=1.2),
            _lambda_row(0.5, base, revenue=1.2),
        ],
        base,
    )
    assert selected == 0.25


def test_preflight_exposes_m2_boundaries_and_high_cost_settings():
    import lightgcn_clv_moe as moe

    summary = moe.preflight_summary(moe.configure_moe_run("hm"))
    assert summary["dataset"] == "hm"
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["graph_mode"] == "binary"
    assert summary["loss_mode"] == "plain"
    assert summary["expert_count"] == 3
    assert summary["window"] == "full official train (~2 years)"


def test_validate_result_metrics_requires_exposure_outputs():
    import lightgcn_clv_moe as moe

    try:
        moe.validate_result_metrics({"recall@10": 0.1}, ks=(10,))
    except KeyError as exc:
        assert "n_distinct@10" in str(exc)
        assert "exposure_entropy@10" in str(exc)
    else:
        raise AssertionError("missing exposure metrics must be rejected")


def test_confirmation_splits_require_explicit_ready_flag():
    import lightgcn_clv_moe as moe

    try:
        moe.configure_moe_run("dunnhumby", eval_test=True)
    except ValueError as exc:
        assert "confirmation_ready" in str(exc)
    else:
        raise AssertionError("test exposure must require explicit confirmation")


def test_checkpoint_paths_are_json_key_safe():
    import lightgcn_clv_moe as moe

    converted = moe.checkpoint_paths_for_json(
        {("clv_moe", 42): "/tmp/a.pt", ("frozen_moe", 42): "/tmp/b.pt"}
    )
    assert converted == {
        "clv_moe_s42": "/tmp/a.pt",
        "frozen_moe_s42": "/tmp/b.pt",
    }


def test_colab_has_fresh_clone_preflight_and_high_cost_gate():
    notebook = json.loads(Path("clv_moe_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "feat/clv-conditioned-moe" in source
    assert "sys.path.insert" in source
    assert "configure_moe_run" in source
    assert "preflight_summary" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "eval_test=False" in source
    assert "eval_holdout=False" in source
    assert "run_experiment(cfg)" in source
