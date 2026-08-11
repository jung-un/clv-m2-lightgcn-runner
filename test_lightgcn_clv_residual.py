import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import lightgcn_clv_residual as residual


def _transactions(days=700):
    rows = []
    start = pd.Timestamp("2020-01-01")
    for day in range(days):
        # user 0 buys every ten days; user 1 buys every twenty days.
        if day % 10 == 0:
            rows.append(
                (
                    0,
                    day % 7,
                    start + pd.Timedelta(days=day),
                    10.0 + day % 5,
                    10.0 + day % 5,
                    day // 10,
                    day % 3,
                )
            )
        if day % 20 == 0:
            rows.append(
                (
                    1,
                    (day + 1) % 7,
                    start + pd.Timedelta(days=day),
                    20.0,
                    20.0,
                    10_000 + day // 20,
                    (day + 1) % 3,
                )
            )
    return pd.DataFrame(
        rows, columns=["u_idx", "i_idx", "t", "v", "up", "b_raw", "cat_idx"]
    )


def test_anchor_windows_exclude_future_rows_and_keep_nonbuyers_negative():
    train = _transactions()
    ds = residual.build_anchor_examples(train, n_users=3, is_date=True)
    assert [a.offset_days for a in ds.anchors] == [270, 180, 90]
    for anchor in ds.anchors:
        assert anchor.observation_end < anchor.target_start
        assert anchor.target_end <= ds.train_end
        assert (
            anchor.user_ids == 2
        ).sum() == 0  # no history means no fabricated sample

    last = ds.anchors[-1]
    # Add a user with observation history but no purchase in the following 90 days.
    extra = train.iloc[:1].copy()
    extra["u_idx"] = 2
    extra["t"] = last.observation_end - pd.Timedelta(days=1)
    ds2 = residual.build_anchor_examples(
        pd.concat([train, extra]), n_users=3, is_date=True
    )
    final = ds2.anchors[-1]
    row = int(np.where(final.user_ids == 2)[0][0])
    assert final.purchase_target[row] == 0.0
    assert final.amount_target[row] == 0.0


def test_anchor_features_do_not_change_when_target_values_change():
    train = _transactions()
    first = residual.build_anchor_examples(train, n_users=2, is_date=True)
    anchor = first.anchors[-1]
    changed = train.copy()
    in_target = (changed.t >= anchor.target_start) & (changed.t <= anchor.target_end)
    changed.loc[in_target, ["v", "up"]] *= 1000
    second = residual.build_anchor_examples(changed, n_users=2, is_date=True)
    np.testing.assert_allclose(first.anchors[-1].numeric, second.anchors[-1].numeric)
    assert not np.allclose(
        first.anchors[-1].amount_target, second.anchors[-1].amount_target
    )


def test_anchor_requires_full_635_day_history():
    with pytest.raises(ValueError, match="635"):
        residual.build_anchor_examples(_transactions(days=620), n_users=2, is_date=True)


def test_short_window_anchors_fit_and_have_ordered_target_windows():
    train = pd.concat(
        [
            _transactions(days=40),
            pd.DataFrame(
                [(0, 0, pd.Timestamp("2020-02-09"), 10.0, 10.0, 4, 0)],
                columns=["u_idx", "i_idx", "t", "v", "up", "b_raw", "cat_idx"],
            ),
        ],
        ignore_index=True,
    )
    ds = residual.build_anchor_examples(
        train,
        n_users=2,
        is_date=True,
        input_days=14,
        target_days=7,
        anchor_offsets=(21, 14, 7),
    )
    assert [anchor.offset_days for anchor in ds.anchors] == [21, 14, 7]
    windows = [(anchor.target_start, anchor.target_end) for anchor in ds.anchors]
    assert windows[0][1] < windows[1][0]
    assert windows[1][1] < windows[2][0]


def test_integer_day_and_datetime_use_identical_feature_schema():
    date_train = _transactions()
    int_train = date_train.copy()
    int_train["t"] = (int_train.t - int_train.t.min()).dt.days
    a = residual.build_anchor_examples(date_train, 2, True)
    b = residual.build_anchor_examples(int_train, 2, False)
    assert a.feature_names == b.feature_names == residual.NUMERIC_FEATURES
    np.testing.assert_allclose(a.anchors[-1].numeric, b.anchors[-1].numeric, atol=1e-6)


def test_vectorized_features_match_single_user_definition():
    train = _transactions()
    end = train.t.max() - pd.Timedelta(days=90)
    obs = train[(train.t > end - pd.Timedelta(days=365)) & (train.t <= end)]
    users = np.sort(obs.u_idx.unique())
    threshold = float(obs.up.quantile(0.8))
    matrix, valid = residual._feature_matrix(obs, users, end, True, threshold)
    for row, user in enumerate(users):
        expected, expected_valid = residual._feature_row(
            obs[obs.u_idx == user], end, end - pd.Timedelta(days=365), True, threshold
        )
        np.testing.assert_allclose(matrix[row], expected, atol=1e-6)
        np.testing.assert_array_equal(valid[row], expected_valid)


def test_transform_fits_only_training_anchors_and_appends_validity_masks():
    ds = residual.build_anchor_examples(_transactions(), 2, True)
    transform = residual.fit_feature_transform(ds.anchors[:2])
    x = residual.transform_features(ds.anchors[-1], transform)
    assert x.shape[1] == 2 * len(residual.NUMERIC_FEATURES)
    np.testing.assert_allclose(
        x[:, len(residual.NUMERIC_FEATURES) :], ds.anchors[-1].valid.astype(np.float32)
    )
    assert np.isfinite(x).all()


def test_future_value_encoder_outputs_finite_nonnegative_amount():
    model = residual.FutureValueEncoder(input_dim=32)
    h, purchase_logit, log_amount = model(torch.randn(4, 32))
    assert h.shape == (4, 16)
    assert purchase_logit.shape == log_amount.shape == (4,)
    assert torch.isfinite(h).all() and torch.isfinite(purchase_logit).all()
    assert (log_amount >= 0).all()


def test_encoder_loss_ignores_nonbuyer_amount_targets():
    logits = torch.tensor([0.0, 0.0])
    pred_amount = torch.tensor([1.0, 1.0])
    purchase = torch.tensor([1.0, 0.0])
    loss1 = residual.future_value_loss(
        logits, pred_amount, purchase, torch.tensor([2.0, 999.0]), pos_weight=1.0
    )
    loss2 = residual.future_value_loss(
        logits, pred_amount, purchase, torch.tensor([2.0, 0.0]), pos_weight=1.0
    )
    torch.testing.assert_close(loss1, loss2)


def test_encoder_training_is_seed_deterministic_and_returns_all_user_embedding():
    ds = residual.build_anchor_examples(_transactions(), 3, True)
    snapshot = residual.build_final_snapshot(_transactions(), 3, True)
    cfg = residual.ResidualConfig(
        encoder_epochs=2, encoder_patience=2, encoder_batch_size=8
    )
    a = residual.train_future_value_encoder(ds, snapshot, cfg, seed=7)
    b = residual.train_future_value_encoder(ds, snapshot, cfg, seed=7)
    np.testing.assert_allclose(a.h_all, b.h_all, atol=1e-7)
    np.testing.assert_allclose(a.ev_all, b.ev_all, atol=1e-7)
    assert a.h_all.shape == (3, 16)
    assert a.ev_all.shape == (3,)
    assert np.isfinite(a.h_all).all() and (a.ev_all >= 0).all()


class _Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.E_u = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), freeze=False
        )
        self.E_i = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]]),
            freeze=False,
        )

    def embeddings(self, need_value=True):
        return self.E_u.weight, self.E_i.weight, None, None


def test_residual_lambda_zero_exactly_equals_base_and_freezes_base():
    base = _Base()
    h = torch.randn(3, 16)
    model = residual.CLVResidualModel(base, h, dim=2)
    up, ip, r, _ = model.embeddings()
    g = model.gate()
    base_scores = up @ ip.T
    scores = residual.residual_scores(up, ip, r, g, 0.0)
    torch.testing.assert_close(scores, base_scores, rtol=0, atol=0)
    assert all(not p.requires_grad for p in base.parameters())


def test_residual_bpr_is_plain_unweighted_bpr():
    model = residual.CLVResidualModel(_Base(), torch.randn(3, 16), dim=2)
    u = torch.tensor([0, 1])
    i = torch.tensor([0, 1])
    j = torch.tensor([3, 0])
    loss = model.bpr_loss(u, i, j, lam=1.0)
    up, ip, r, _ = model.embeddings()
    g = model.gate()
    all_scores = residual.residual_scores(up[u], ip, r[u], g[u], 1.0)
    expected = -torch.nn.functional.logsigmoid(
        all_scores[torch.arange(2), i] - all_scores[torch.arange(2), j]
    ).mean()
    torch.testing.assert_close(loss, expected)


def test_plain_bpr_updates_only_adapter_and_gate():
    base = _Base()
    model = residual.CLVResidualModel(base, torch.randn(3, 16), dim=2)
    before = [p.detach().clone() for p in base.parameters()]
    loss = model.bpr_loss(
        torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([3, 0]), lam=1.0
    )
    loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.trainable_parameters()
    )
    for old, current in zip(before, base.parameters()):
        torch.testing.assert_close(old, current)


def test_constant_control_removes_user_specific_residual_and_gate():
    h = torch.randn(3, 16)
    model = residual.CLVResidualModel(_Base(), h, dim=2, constant=True)
    _, _, r, _ = model.embeddings()
    g = model.gate()
    torch.testing.assert_close(r[0], r[1])
    torch.testing.assert_close(r[1], r[2])
    torch.testing.assert_close(g[0], g[1])
    torch.testing.assert_close(g[1], g[2])


def test_user_without_clv_history_receives_zero_gate():
    h = torch.randn(3, 16)
    h[2].zero_()
    model = residual.CLVResidualModel(_Base(), h, dim=2)
    assert model.gate()[2].item() == 0.0


def test_runtime_lambda_zero_equivalence_checks_every_score():
    model = residual.CLVResidualModel(_Base(), torch.randn(3, 16), dim=2)
    residual.assert_lambda_zero_equivalence(model, n_check=3)


def test_select_lambda_applies_all_accuracy_guardrails_and_tie_break():
    base = {f"{m}@{k}": 1.0 for m in ("recall", "ndcg") for k in (10, 20, 50)}
    rows = []
    for lam, economic in [(0.0, 1.0), (0.05, 1.2), (0.1, 1.2), (0.25, 2.0)]:
        row = {"lambda": lam, "revenue@10": economic, **base}
        rows.append(row)
    rows[-1]["ndcg@50"] = 0.989  # >1% relative drop: reject despite best economics
    selected, table = residual.select_lambda(rows, base, tolerance=0.01)
    assert selected == 0.05
    assert table.loc[table["lambda"] == 0.25, "eligible"].item() is False


def test_select_lambda_falls_back_to_zero_without_claiming_success():
    base = {f"{m}@{k}": 1.0 for m in ("recall", "ndcg") for k in (10, 20, 50)}
    bad = [{"lambda": 0.1, "revenue@10": 2.0, **{key: 0.5 for key in base}}]
    selected, table = residual.select_lambda(bad, base)
    assert selected == 0.0
    assert not table["eligible"].any()


def test_configure_screening_is_validation_only_and_uses_two_year_window():
    cfg = residual.configure_residual_run("hm")
    assert cfg.dataset == "hm"
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert cfg.input_days == 365 and cfg.target_days == 90
    assert cfg.lambda_eval == (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)


def test_residual_colab_pins_reviewed_detached_commit_and_protected_screening():
    notebook = json.loads(Path("clv_residual_colab.ipynb").read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "REVIEWED_SHA = '3807e000df923ea857dee716cb60703b7c469902'" in code
    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", code)
    assert "'checkout', '--detach', REVIEWED_SHA" in code
    assert "'rev-parse', 'HEAD'" in code
    assert "assert actual_sha == REVIEWED_SHA" in code
    assert "GIT_REF" not in code
    assert " switch " not in code
    assert "pull --ff-only" not in code
    assert "ACKNOWLEDGE_HIGH_COST = False" in code
    assert "eval_test=False" in code and "eval_holdout=False" in code


def _provenance_base_config():
    return {
        "DIM": 64,
        "N_LAYERS": 3,
        "BATCH_SIZE": 1024,
        "LR": 0.001,
        "EPOCHS": 100,
        "EARLY_STOP": 20,
        "WINDOW_DAYS": None,
        "VAL_DAYS": 30,
        "TEST_DAYS": 30,
        "HOLDOUT_DAYS": 30,
        "MIN_USER_INTER": 1,
        "MIN_ITEM_INTER": 1,
        "NEG_MODE": "uniform",
        "GRAPH_MODE": "binary",
        "LOSS_MODE": "plain",
    }


def test_residual_result_fingerprint_isolates_data_source_and_m1_state():
    cfg = residual.configure_residual_run("dunnhumby")
    base_cfg = _provenance_base_config()
    manifest = {
        "transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/item", "bytes": 2, "sha256": "bb"},
    }

    original = residual._result_fingerprint(
        cfg, base_cfg, manifest, "source-a", {"42": "m1-a"}
    )
    changed_data = residual._result_fingerprint(
        cfg,
        base_cfg,
        manifest
        | {
            "transactions": {
                "path": "/tx",
                "bytes": 3,
                "sha256": "changed",
            }
        },
        "source-a",
        {"42": "m1-a"},
    )
    changed_source = residual._result_fingerprint(
        cfg, base_cfg, manifest, "source-b", {"42": "m1-a"}
    )
    changed_m1 = residual._result_fingerprint(
        cfg, base_cfg, manifest, "source-a", {"42": "m1-b"}
    )
    assert len({original, changed_data, changed_source, changed_m1}) == 4


def test_residual_checkpoint_filename_and_payload_round_trip_provenance(tmp_path):
    cfg = residual.configure_residual_run("dunnhumby")
    base_cfg = _provenance_base_config()
    manifest = {
        "transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/item", "bytes": 2, "sha256": "bb"},
    }
    first = residual._checkpoint_fingerprint(
        cfg, base_cfg, manifest, "source-a", 42, "m1-a"
    )
    changed = residual._checkpoint_fingerprint(
        cfg, base_cfg, manifest, "source-a", 42, "m1-b"
    )
    first_path = residual._checkpoint_path(
        tmp_path, "encoder", cfg.dataset, 42, first
    )
    changed_path = residual._checkpoint_path(
        tmp_path, "encoder", cfg.dataset, 42, changed
    )
    assert first_path != changed_path

    provenance = residual._checkpoint_provenance(
        cfg,
        base_cfg,
        manifest,
        "source-a",
        42,
        "m1-a",
        "/m1.pt",
    )
    residual._save_provenance_checkpoint(
        first_path, {"state": {"weight": torch.ones(1)}}, provenance
    )
    payload = torch.load(first_path, map_location="cpu", weights_only=False)
    assert payload["provenance"] == provenance
    assert provenance["checkpoint_fingerprint"] == first
    assert provenance["input_manifest"] == manifest
    assert provenance["source_revision"] == "source-a"
    assert provenance["baseline_state_hash"] == "m1-a"
    assert provenance["m1_checkpoint"] == "/m1.pt"


def test_residual_public_runner_persists_result_and_checkpoint_provenance(
    tmp_path, monkeypatch
):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_v3 as v3

    tx_path = tmp_path / "transactions.csv"
    item_path = tmp_path / "items.csv"
    tx_path.write_bytes(b"transactions-v1")
    item_path.write_bytes(b"items-v1")
    monkeypatch.setattr(v3, "CFG", dict(v3.CFG))
    monkeypatch.setattr(v3, "DCFG", v3.DCFG)
    schema = dict(v3.SCHEMA["dunnhumby"])
    schema.update(tx_path=str(tx_path), item_meta_path=str(item_path))
    monkeypatch.setitem(v3.SCHEMA, "dunnhumby", schema)

    cfg = residual.configure_residual_run(
        "dunnhumby",
        out_dir=str(tmp_path / "results"),
        m1_checkpoint_dir=str(tmp_path / "m1"),
        include_constant_control=False,
    )
    fake_stats = {
        "source": {"rows": 4, "time_min": 0, "time_max": 3},
        "split_evaluation_status": {
            "val": "constructed",
            "test": "not_constructed",
            "holdout": "not_constructed",
        },
    }
    data = {
        "train": object(),
        "n_users": 3,
        "n_items": 4,
        "splits": {"val": ({}, {})},
        "x_val_u": np.zeros((3, 5), dtype=np.float32),
        "csr_ptr": np.zeros(4, dtype=np.int64),
        "csr_items": np.array([], dtype=np.int32),
        "data_stats": fake_stats,
    }
    artifact = SimpleNamespace(
        model=torch.nn.Linear(2, 1),
        transform=SimpleNamespace(
            mean=np.zeros(2, dtype=np.float32),
            std=np.ones(2, dtype=np.float32),
            feature_names=("a", "b"),
        ),
        h_all=np.ones((3, 16), dtype=np.float32),
        ev_all=np.arange(3, dtype=np.float32),
        best_epoch=1,
        diagnostics={"best_val_loss": 0.5},
    )
    monkeypatch.setattr(v3, "prepare_data", lambda *args: data)
    monkeypatch.setattr(residual, "build_anchor_examples", lambda *args: object())
    monkeypatch.setattr(residual, "build_final_snapshot", lambda *args: object())
    monkeypatch.setattr(
        residual, "train_future_value_encoder", lambda *args: artifact
    )
    monkeypatch.setattr(
        v3,
        "item_value_features",
        lambda *args: (
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
        ),
    )
    monkeypatch.setattr(v3, "item_meta", lambda *args: None)
    monkeypatch.setattr(v3, "segment_thresholds", lambda *args: (0.0, 1.0))
    monkeypatch.setattr(v3, "EvalCache", lambda *args: object())

    def fake_get_or_train(arch, seed, *args):
        base_cfg = args[-1]
        checkpoint = Path(base_cfg["OUT_DIR"]) / (
            f"ckpt_pref_only_dunnhumby_s{seed}_"
            f"{v3.cfg_hash(base_cfg, v3.DCFG, 'pref_only', seed)}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": "m1"}, checkpoint)
        return _Base(), {}

    monkeypatch.setattr(v3, "get_or_train", fake_get_or_train)

    def fake_evaluate(*args, per_user=False, **kwargs):
        flat = {"revenue@10": 1.0, "arp@10": 0.2}
        for k in (10, 20, 50):
            flat[f"recall@{k}"] = 0.1
            flat[f"ndcg@{k}"] = 0.1
        if per_user:
            flat["per_user"] = {
                metric: np.zeros(2, dtype=np.float32)
                for metric in ("recall", "ndcg", "revenue", "arp")
            }
        return flat

    monkeypatch.setattr(v3, "evaluate", fake_evaluate)
    monkeypatch.setattr(v3, "flatten", lambda result: result)
    monkeypatch.setattr(residual, "validate_result_metrics", lambda *args: None)
    monkeypatch.setattr(
        residual,
        "train_residual_adapter",
        lambda *args: {"loss": "plain_bpr", "best_epoch": 1},
    )
    monkeypatch.setattr(residual, "assert_lambda_zero_equivalence", lambda *args: None)
    monkeypatch.setattr(
        residual,
        "_effective_score_ratio",
        lambda *args: {"effective_score_ratio": 0.1},
    )
    monkeypatch.setattr(
        v3,
        "paired_bootstrap",
        lambda *args: {"mean_diff": 0.0, "lo": 0.0, "hi": 0.0},
    )

    frame = residual.run_experiment(cfg)
    result_path = Path(frame.attrs["result_paths"]["json"])
    payload = json.loads(result_path.read_text())
    assert payload["result_fingerprint"] == frame.attrs["result_fingerprint"]
    assert payload["input_manifest"]["transactions"]["sha256"] == moe.file_sha256(
        tx_path
    )
    assert payload["source_revision"]
    assert payload["baseline_state_hashes"]["42"]
    assert payload["data_stats"] == fake_stats
    assert set(payload["checkpoint_paths"]) == {
        "encoder_s42",
        "m1_s42",
        "clv_residual_s42",
    }
    assert all(len(value) == 64 for value in payload["checkpoint_sha256"].values())
    assert f"data_{moe.manifest_hash(payload['input_manifest'])[:12]}" in payload[
        "checkpoint_paths"
    ]["m1_s42"]
    for key in ("encoder_s42", "clv_residual_s42"):
        checkpoint = torch.load(
            payload["checkpoint_paths"][key], map_location="cpu", weights_only=False
        )
        provenance = checkpoint["provenance"]
        assert provenance["source_revision"] == payload["source_revision"]
        assert provenance["input_manifest"] == payload["input_manifest"]
        assert provenance["baseline_state_hash"] == payload[
            "baseline_state_hashes"
        ]["42"]


def test_result_rows_require_all_exposure_and_economic_metrics():
    flat = {
        "recall@10": 0.1,
        "ndcg@10": 0.1,
        "revenue@10": 0.2,
        "coverage@10": 0.3,
        "n_distinct@10": 10,
        "exposure_entropy@10": 1.2,
        "eff_catalog@10": 3.3,
        "top10_share@10": 0.4,
        "top100_share@10": 0.8,
    }
    residual.validate_result_metrics(flat, ks=(10,))
    with pytest.raises(KeyError, match="eff_catalog"):
        residual.validate_result_metrics(
            {k: v for k, v in flat.items() if k != "eff_catalog@10"}, ks=(10,)
        )


def test_normalize_flat_metrics_uses_explicit_exposure_entropy_name():
    flat = {"entropy@10": 1.5, "recall@10": 0.2}
    normalized = residual.normalize_flat_metrics(flat)
    assert normalized["exposure_entropy@10"] == 1.5
    assert "entropy@10" not in normalized


def test_adapter_training_preserves_base_hash_and_reports_plain_bpr_updates():
    base = _Base()
    model = residual.CLVResidualModel(base, torch.randn(3, 16), dim=2)
    data = {
        "tr_u": np.array([0, 1, 2, 0]),
        "tr_i": np.array([0, 1, 2, 2]),
        "n_items": 4,
        "pos_key": np.array([0, 2, 5, 10]),
        "item_cat": np.zeros(4, dtype=np.int64),
        "cat_items": {0: np.arange(4)},
    }
    cfg = residual.ResidualConfig(adapter_epochs=2, adapter_patience=2)
    base_cfg = {"BATCH_SIZE": 2, "NEG_MODE": "uniform"}
    stats = residual.train_residual_adapter(
        model, base, data, base_cfg, cfg, seed=3, eval_recall=lambda _: 0.1
    )
    assert stats["loss"] == "plain_bpr"
    assert stats["updates"] == 4
    assert stats["base_hash_before"] == stats["base_hash_after"]
