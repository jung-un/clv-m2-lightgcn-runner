import json
from pathlib import Path
from types import SimpleNamespace

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
        self._pref_cache = None
        self.freeze_calls = 0

    def embeddings(self, need_value=True):
        user, item = (
            self._pref_cache
            if self._pref_cache is not None
            else (self.E_u.weight, self.E_i.weight)
        )
        return user, item, None, None

    def pref_params(self):
        return list(self.E_u.parameters()) + list(self.E_i.parameters())

    def freeze_pref_and_cache(self):
        self.freeze_calls += 1
        for parameter in self.pref_params():
            parameter.requires_grad_(False)
        self._pref_cache = (self.E_u.weight.detach(), self.E_i.weight.detach())


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


def test_source_revision_is_recordable_for_provenance():
    import lightgcn_clv_moe as moe

    revision = moe.source_revision()
    assert isinstance(revision, str) and revision


def test_file_sha256_is_content_sensitive(tmp_path):
    import lightgcn_clv_moe as moe

    path = tmp_path / "transactions.csv"
    path.write_bytes(b"a,b\n1,2\n")
    first = moe.file_sha256(path)
    path.write_bytes(b"a,b\n1,3\n")
    assert len(first) == 64
    assert moe.file_sha256(path) != first


def test_input_manifest_hashes_transactions_and_item_metadata(tmp_path):
    import lightgcn_clv_moe as moe

    tx = tmp_path / "tx.csv"
    item = tmp_path / "items.csv"
    tx.write_bytes(b"tx-v1")
    item.write_bytes(b"item-v1")
    manifest = moe.build_input_manifest(
        {"tx_path": str(tx), "item_meta_path": str(item)}
    )
    assert manifest["transactions"]["sha256"] == moe.file_sha256(tx)
    assert manifest["item_metadata"]["sha256"] == moe.file_sha256(item)
    item.write_bytes(b"item-v2")
    assert moe.build_input_manifest(
        {"tx_path": str(tx), "item_meta_path": str(item)}
    ) != manifest


def test_m1_manifest_fails_closed_when_data_identity_changes(tmp_path):
    import lightgcn_clv_moe as moe

    checkpoint = tmp_path / "m1.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    inputs = {
        "transactions": {"path": "/a", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/b", "bytes": 2, "sha256": "bb"},
    }
    moe.validate_or_write_m1_manifest(
        checkpoint,
        inputs,
        config_hash="cfg1",
        state_hash_value="state1",
        existed_before=False,
    )
    moe.validate_or_write_m1_manifest(
        checkpoint,
        inputs,
        config_hash="cfg1",
        state_hash_value="state1",
        existed_before=True,
    )
    changed = {**inputs, "item_metadata": {"path": "/b", "bytes": 3, "sha256": "cc"}}
    try:
        moe.validate_or_write_m1_manifest(
            checkpoint,
            changed,
            config_hash="cfg1",
            state_hash_value="state1",
            existed_before=True,
        )
    except RuntimeError as exc:
        assert "manifest" in str(exc)
    else:
        raise AssertionError("M1/data mismatch must fail closed")


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


def test_frozen_phase_builds_base_graph_cache_once_and_clears_on_unfreeze():
    import lightgcn_clv_moe as moe

    base = _Base()
    moe._set_base_trainable(base, False)
    moe._set_base_trainable(base, False)
    assert base.freeze_calls == 1
    assert base._pref_cache is not None
    moe._set_base_trainable(base, True)
    assert base._pref_cache is None


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


def test_select_lambda_rejects_eligible_lambda_without_economic_improvement():
    import lightgcn_clv_moe as moe

    base = _baseline_metrics()
    selected, table = moe.select_lambda(
        [
            _lambda_row(0.0, base),
            _lambda_row(0.25, base, revenue=0.99),
            _lambda_row(0.5, base, revenue=1.0),
        ],
        base,
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


def test_screening_decision_requires_main_to_outperform_all_controls():
    import lightgcn_clv_moe as moe

    selected = {
        "clv_moe": 0.5,
        "frozen_moe": 0.5,
        "constant_gate": 0.25,
        "shuffled_clv": 0.25,
        "single_adapter": 0.5,
        "pref_continue": 0.0,
    }
    rows = [
        {"seed": 42, "model_id": model_id, "split": "val", "lambda": lam,
         "revenue@10": revenue}
        for model_id, lam, revenue in [
            ("clv_moe", 0.5, 1.10),
            ("frozen_moe", 0.5, 1.05),
            ("constant_gate", 0.25, 1.02),
            ("shuffled_clv", 0.25, 1.01),
            ("single_adapter", 0.5, 1.04),
            ("pref_continue", 0.0, 1.03),
        ]
    ]
    decision = moe.screening_decision(rows, selected, {"clv_moe": True})
    assert decision["success"] is True
    rows[2]["revenue@10"] = 1.11
    decision = moe.screening_decision(rows, selected, {"clv_moe": True})
    assert decision["success"] is False
    assert "constant_gate" in decision["failed_controls"]


def test_preflight_exposes_m2_boundaries_and_high_cost_settings():
    import lightgcn_clv_moe as moe

    summary = moe.preflight_summary(moe.configure_moe_run("hm"))
    assert summary["dataset"] == "hm"
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["confirmation_ready"] is False
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


def test_screening_runner_fails_closed_even_with_confirmation_flag():
    import lightgcn_clv_moe as moe

    try:
        moe.configure_moe_run(
            "dunnhumby", eval_test=True, confirmation_ready=True
        )
    except ValueError as exc:
        assert "screening-only" in str(exc)
        assert "manifest" in str(exc)
    else:
        raise AssertionError("test must stay closed until two-dataset manifest exists")


def test_run_experiment_revalidates_direct_dataclass_before_any_data_access(
    monkeypatch,
):
    import lightgcn_clv_moe as moe

    def forbidden(*args, **kwargs):
        raise AssertionError("protected split must fail before touching data")

    monkeypatch.setattr(moe, "file_sha256", forbidden)
    monkeypatch.setattr(moe.v3, "prepare_data", forbidden)
    try:
        moe.run_experiment(moe.MoEConfig(eval_test=True, confirmation_ready=True))
    except ValueError as exc:
        assert "screening-only" in str(exc)
    else:
        raise AssertionError("direct dataclass construction must not bypass split guard")


def test_screening_runner_rejects_multi_seed_direct_config():
    import lightgcn_clv_moe as moe

    try:
        moe.validate_moe_config(moe.MoEConfig(seed_list=(42, 43, 44)))
    except ValueError as exc:
        assert "seed 42" in str(exc)
        assert "screening-only" in str(exc)
    else:
        raise AssertionError("screening must stay fixed to seed 42")


def test_checkpoint_paths_are_json_key_safe():
    import lightgcn_clv_moe as moe

    converted = moe.checkpoint_paths_for_json(
        {("clv_moe", 42): "/tmp/a.pt", ("frozen_moe", 42): "/tmp/b.pt"}
    )
    assert converted == {
        "clv_moe_s42": "/tmp/a.pt",
        "frozen_moe_s42": "/tmp/b.pt",
    }


def test_moe_checkpoint_round_trip_reproduces_scores(tmp_path):
    import lightgcn_clv_moe as moe

    model = _model()
    context = {
        "user_profile": SimpleNamespace(
            values=model.original_profile.cpu().numpy(),
            valid_user=model.has_profile.cpu().numpy(),
            feature_names=tuple(f"u{x}" for x in range(51)),
        ),
        "item_profile": SimpleNamespace(
            numeric=model.item_numeric.cpu().numpy(),
            category_ids=model.item_category_ids.cpu().numpy(),
            valid_item=model.valid_item.cpu().numpy(),
            numeric_names=tuple(f"i{x}" for x in range(6)),
            n_categories=4,
        ),
        "artifact": SimpleNamespace(ev_all=np.arange(4, dtype=np.float32)),
    }
    path = tmp_path / "model.pt"
    moe._save_model_checkpoint(path, model, context, {"best_epoch": 1}, {})
    reload_base = _Base()
    reload_base._pref_cache = (
        torch.zeros_like(reload_base.E_u.weight),
        torch.zeros_like(reload_base.E_i.weight),
    )
    loaded = moe.load_moe_checkpoint(
        path,
        reload_base,
        moe.configure_moe_run("dunnhumby"),
        control="clv",
        device=torch.device("cpu"),
    )
    assert loaded.base_model._pref_cache is None
    users = torch.arange(4)
    torch.testing.assert_close(
        loaded.score_all(users, 0.5), model.score_all(users, 0.5), rtol=0, atol=0
    )


def test_colab_has_fresh_clone_preflight_and_high_cost_gate():
    notebook = json.loads(Path("clv_moe_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "REVIEWED_COMMIT = '7f433bf712644db21d3b11b4ffd130549c11dfbc'" in source
    assert "checkout', '--detach', REVIEWED_COMMIT" in source
    assert "rev-parse', 'HEAD'" in source
    assert "sys.path.insert" in source
    assert "configure_moe_run" in source
    assert "preflight_summary" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "eval_test=False" in source
    assert "eval_holdout=False" in source
    assert "summary['confirmation_ready'] is False" in source
    assert "run_experiment(cfg)" in source
    assert "screening_decision" in source
    assert "failed_controls" in source


def test_direct_cli_is_preflight_only(monkeypatch, capsys):
    import lightgcn_clv_moe as moe

    monkeypatch.setattr(
        moe,
        "run_experiment",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("CLI must not start high-cost training")
        ),
    )
    moe.main_cli()
    output = capsys.readouterr().out
    assert '"dataset": "dunnhumby"' in output
    assert '"confirmation_ready": false' in output
