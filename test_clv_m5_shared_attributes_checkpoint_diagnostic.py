"""Local functional checks; actual Drive checkpoint inference is not run here."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

import clv_m5_shared_attributes_checkpoint_diagnostic as d
from test_clv_m5_shared_attributes_centered_seed48 import model


def test_joint_permutation_preserves_strata_validity_and_joint_tuples():
    degree = np.repeat(np.arange(1, 11), 10)
    valid = np.ones(100, bool)
    valid[[0, 17, 99]] = False
    donor, bins, edges = d.joint_permutation(degree, valid)
    np.testing.assert_array_equal(np.sort(donor), np.arange(100))
    np.testing.assert_array_equal(bins[donor], bins)
    np.testing.assert_array_equal(valid[donor], valid)
    np.testing.assert_array_equal(donor[~valid], np.flatnonzero(~valid))
    assert (donor != np.arange(100)).sum() > 50
    np.testing.assert_array_equal(d.joint_permutation(degree, valid)[0], donor)
    assert len(edges) <= 9
    # Columns use the same donor, not independent shuffles.
    context = np.column_stack([np.arange(100), np.arange(100) * 2])
    np.testing.assert_array_equal(context[donor, 1], context[donor, 0] * 2)


def test_permutation_empty_invalid_singletons_and_bad_shapes():
    for degree, valid in (([1, 2], [False, False]), ([1], [True])):
        np.testing.assert_array_equal(d.joint_permutation(degree, valid)[0], np.arange(len(degree)))
    with pytest.raises(ValueError):
        d.joint_permutation([1, 2], [True])
    with pytest.raises(ValueError):
        d.joint_permutation([np.nan], [True])


def test_three_views_same_id_items_graph_parameters_context_restored():
    net = model()
    before = deepcopy(net.state_dict())
    with patch.object(net, "weighted_bpr_loss", side_effect=AssertionError("training forbidden")):
        views = d.embedding_views(net, np.array([1, 0, 2]))
    assert tuple(views) == d.VIEWS
    full, id_only, shuffled = (views[k] for k in d.VIEWS)
    torch.testing.assert_close(full.items, shuffled.items, rtol=0, atol=0)
    torch.testing.assert_close(full.users[:, :net.id_dim], shuffled.users[:, :net.id_dim], rtol=0, atol=0)
    assert torch.count_nonzero(id_only.users[:, net.id_dim:]) == 0
    assert torch.count_nonzero(id_only.items[:, net.id_dim:]) == 0
    torch.testing.assert_close(id_only.users[:, :net.id_dim], full.users[:, :net.id_dim])
    assert torch.count_nonzero(full.users[:2, net.id_dim:] - shuffled.users[:2, net.id_dim:]) > 0
    torch.testing.assert_close(full.users[2], shuffled.users[2])
    assert all(not value.requires_grad for v in views.values() for value in v.embeddings())
    for key, value in net.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in net.parameters())


def test_context_restored_when_swapped_forward_fails():
    net = model()
    context = net.context.clone()
    original = net.history_vectors
    calls = []
    def history_vectors(attributes):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("forward failed")
        return original(attributes)
    with patch.object(net, "history_vectors", side_effect=history_vectors), pytest.raises(RuntimeError):
        d.embedding_views(net, np.array([1, 0, 2]))
    torch.testing.assert_close(net.context, context, rtol=0, atol=0)


def test_id_recalculation_roundoff_cannot_contaminate_nv_comparison():
    """Inject the reported equality failure without requiring a CUDA device."""
    net = model()
    original = net.id_vectors
    calls = []
    def id_vectors():
        users, items = original()
        calls.append(1)
        if len(calls) > 1:
            users = users + 1e-7
        return users, items
    with patch.object(net, "id_vectors", side_effect=id_vectors):
        views = d.embedding_views(net, np.array([1, 0, 2]))
    assert len(calls) == 1
    torch.testing.assert_close(views["full"].users[:, :net.id_dim],
        views["m2_nv_permuted"].users[:, :net.id_dim], rtol=0, atol=0)


def test_cached_full_matches_original_and_identity_permutation_is_exact():
    net = model()
    expected_users, expected_items, _, _ = net.embeddings()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    views = d.embedding_views(net, np.arange(net.n_users))
    torch.testing.assert_close(views["full"].users, expected_users, rtol=0, atol=0)
    torch.testing.assert_close(views["full"].items, expected_items, rtol=0, atol=0)
    torch.testing.assert_close(views["m2_nv_permuted"].users, views["full"].users, rtol=0, atol=0)
    assert views["full"].items.data_ptr() == views["m2_nv_permuted"].items.data_ptr()
    assert torch.are_deterministic_algorithms_enabled() == deterministic
    assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only


def test_nv_delta_matches_direct_swapped_profile_and_global_setting_restored():
    net = model()
    context = net.context.clone()
    with torch.no_grad():
        net.context.copy_(context[torch.tensor([1, 0, 2])])
        direct_users, direct_items, _, _ = net.embeddings()
        net.context.copy_(context)
    views = d.embedding_views(net, np.array([1, 0, 2]))
    torch.testing.assert_close(views["m2_nv_permuted"].users, direct_users, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(views["m2_nv_permuted"].items, direct_items, rtol=0, atol=0)


def test_full_comparison_preserves_all_keys_losses_and_zero_denominator():
    metrics = {"full": {"recall@10": .9, "coverage@10": 0.},
               "joint_id_only": {"recall@10": 1., "coverage@10": 0.},
               "m2_nv_permuted": {"recall@10": .8, "coverage@10": .1}}
    result = d.comparisons(metrics)
    assert len(result) == 4
    assert result.iloc[0].delta < 0
    assert pd.isna(result.iloc[1].relative_change_pct)
    metrics["m2_nv_permuted"].pop("coverage@10")
    with pytest.raises(RuntimeError):
        d.comparisons(metrics)


def test_source_exact_identity_readback_requires_checkpoint(tmp_path):
    # Source content changes must fail before preparation or training.
    with pytest.raises(RuntimeError, match="원본"):
        d.source_artifacts(tmp_path)
    with pytest.raises(ValueError):
        d.run(tmp_path, tmp_path / "diagnostic")


def test_fixed_boundary_components_decompose_direct_score_and_joint_nv_effect():
    net = model()
    views = d.embedding_views(net, np.array([1, 0, 2]))
    truth = pd.DataFrame(dict(seed=[48], user=[0], item=[2], rank=[np.nan]))
    recs = pd.DataFrame(dict(user=[0, 0, 0], item=[3, 1, 0], rank=[10, 20, 50]))
    population = pd.DataFrame(dict(user=[0], segment=["고CLV"], degree_bin=[1], q_n=[.2], q_v=[.3]))
    with patch.object(d.screen, "_population", return_value=population):
        result = d.boundary_components(views, truth, recs, {})
    assert len(result) == 3 and result.k.tolist() == [10, 20, 50]
    for row in result.itertuples():
        full = views["full"]
        expected = float((full.users[0] * (full.items[2] - full.items[row.boundary_item])).sum())
        assert row.id_margin + row.actual_attribute_margin == pytest.approx(expected, abs=1e-7)
        assert row.actual_minus_permuted_margin == pytest.approx(
            row.actual_attribute_margin - row.permuted_attribute_margin)
        assert row.segment == "고CLV" and pd.isna(row.rank)


def test_run_stops_after_failed_full_readback_without_scoring_other_views(tmp_path):
    net = model()
    state = deepcopy(net.state_dict())
    prepared = dict(data={"csr_ptr": np.array([0, 2, 4, 5])}, clv_valid=np.array([True, True, False]),
                    cache=SimpleNamespace(users=np.array([0])))
    arm = {"curve": [{"metrics": {"recall@10": .5}}]}
    with patch.object(d, "source_artifacts", return_value=({}, arm, state, {})), \
         patch.object(d, "prepare", return_value=(None, prepared)), \
         patch.object(d.screen, "build_model", return_value=net), \
         patch.object(d.screen.capacity, "_evaluate", return_value={"recall@10": .4}) as evaluate, \
         patch.object(d.screen.ranks, "_score_truth", side_effect=AssertionError("must stop")):
        with pytest.raises(RuntimeError, match="재현"):
            d.run(tmp_path / "source", tmp_path / "out")
    assert evaluate.call_count == 1
    assert not pd.read_csv(tmp_path / "out" / "readback.csv").passed.all()


def test_complete_readonly_output_pipeline_with_mock_source(tmp_path):
    net = model()
    out = tmp_path / "out"
    prepared = dict(data={"csr_ptr": np.array([0, 2, 4, 5])}, clv_valid=np.array([True, True, False]),
                    cache=SimpleNamespace(users=np.array([0])), out_dir=out)
    metrics = {f"{name}@{k}": .1 for k in (10, 20, 50) for name in ("recall", "price_purchase_amount_weighted_hit")}
    metrics.update({f"extra_metric_{k}": .2 for k in range(136)})
    arm = {"curve": [{"metrics": metrics}]}
    truth = pd.DataFrame(dict(seed=[48], user=[0], item=[2], rank=[np.nan], weight=[1.]))
    population = pd.DataFrame(dict(user=[0], segment=["고CLV"], degree_bin=[1], q_n=[.2], q_v=[.3]))
    def score(view, prepared, seed, recommendations):
        recommendations.extend(dict(seed=48, user=0, item=i, rank=k, is_truth=False, score=0.)
                               for k, i in ((10, 3), (20, 1), (50, 0)))
        return truth.copy()
    def pair(prepared, reference, candidate, recs, seed, folder, labels):
        path = out / folder / "summary.csv"
        d.screen.io._atomic_csv(path, pd.DataFrame(dict(grouping=["all"] * 3, k=[10, 20, 50],
            mean_recall_delta=[0.] * 3, mean_weighted_hit_delta=[0.] * 3)))
        return {"summary": str(path)}
    with patch.object(d, "source_artifacts", return_value=({}, arm, deepcopy(net.state_dict()), {})), \
         patch.object(d, "prepare", return_value=(None, prepared)), \
         patch.object(d.screen, "build_model", return_value=net), \
         patch.object(d.screen.capacity, "_evaluate", return_value=metrics) as evaluate, \
         patch.object(d.screen.ranks, "_score_truth", side_effect=score), \
         patch.object(d.screen, "_save_pair", side_effect=pair), \
         patch.object(d.screen, "_population", return_value=population):
        result = d.run(tmp_path / "source", out)
    assert evaluate.call_count == 3
    assert Path(result["zip"]).is_file()
    assert len(pd.read_csv(out / "comparison.csv")) == 284
    assert pd.read_csv(out / "absolute.csv").shape == (3, 145)
    info = json.loads(Path(result["result"]).read_text())
    assert info["new_training"] == 0 and info["original_full_readback_passed"]
    assert not info["next_training_authorized"] and not info["id_only_is_m1"]
    assert pd.read_csv(out / "boundary_score_components.csv").shape[0] == 3


def test_runner_has_no_training_calls_or_protected_data_reads():
    tree = ast.parse(Path(d.__file__).read_text())
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {"run_arm", "_train_curve", "backward", "step", "Adam", "mark_complete", "save_epoch"}
    assert "diagnose_existing_c" not in calls


def test_colab_schema_ast_and_pinned_source():
    path = Path(__file__).with_name("clv_m5_shared_attributes_checkpoint_diagnostic_colab.ipynb")
    if not path.exists():
        pytest.skip("notebook added after source commit")
    notebook = json.loads(path.read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "files.download" in source and "diagnostic.run" in source
    assert "drive.mount" in source and "--detach" in source
    assert "screen.run" not in source


def test_saved_list_requires_exact_users_unique_items_and_all50_ranks():
    rows = pd.DataFrame([dict(model="m5", user=u, rank=k, item=k - 1)
                         for u in (8, 4) for k in range(1, 51)])
    top = d._saved_tops(rows, np.array([4, 8]))
    np.testing.assert_array_equal(top[0], np.arange(50))
    with pytest.raises(RuntimeError):
        d._saved_tops(rows.iloc[:-1], np.array([4, 8]))
    with pytest.raises(RuntimeError):
        d._saved_tops(pd.concat([rows, rows.iloc[:1]]), np.array([4, 8]))


def test_probe_captures_same_evaluation_topk_and_replays_without_changing_hooks():
    view = d.EmbeddingView(torch.ones(1, 4), torch.ones(60, 4), 4)
    prepared = dict(cache=SimpleNamespace(users=np.array([7])))
    current = np.arange(50).reshape(1, -1)
    saved = np.arange(1, 51).reshape(1, -1)
    matrix = torch.arange(60, dtype=torch.float32).reshape(1, -1)
    used = []
    def metric_fn(top, batch, *args, **kwargs):
        used.append(top.copy())
        return float(top.sum())
    def evaluate(view, prepared):
        d.screen.v3.combined_score_all()
        batch_top = current.copy()
        total = d.screen.v3.score_topk(batch_top, np.array([7]))
        return {"test": total, "exposure_after_callback": float(batch_top.sum())}
    with patch.object(d.screen.v3, "combined_score_all", return_value=matrix) as score_hook, \
         patch.object(d.screen.v3, "score_topk", side_effect=metric_fn) as metric_hook, \
         patch.object(d.screen.capacity, "_evaluate", side_effect=evaluate):
        values, top, scores = d._capture_evaluation(view, prepared)
        np.testing.assert_array_equal(top, current)
        np.testing.assert_array_equal(scores, current.astype(float))
        assert values["test"] == current.sum()
        replay, replay_top, replay_scores = d._capture_evaluation(view, prepared, saved)
        assert replay["test"] == saved.sum()
        assert replay["exposure_after_callback"] == saved.sum()
        np.testing.assert_array_equal(replay_top, saved)
        np.testing.assert_array_equal(replay_scores, saved.astype(float))
        assert d.screen.v3.combined_score_all is score_hook
        assert d.screen.v3.score_topk is metric_hook


def test_probe_uses_cached_context_preserves_failed_readback_and_never_prepares(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    failed_readback = out / "readback.csv"
    failed_readback.write_text("original failed readback\n")
    source = tmp_path / "source"
    recs = pd.DataFrame([dict(model="m5", user=0, rank=k, item=k - 1) for k in range(1, 51)])
    d.screen.io._atomic_csv(source / "new_pair_diagnostic" / "recommendations.csv", recs)
    prepared = dict(data={"csr_ptr": np.array([0, 1]), "csr_items": np.array([59])},
        cache=SimpleNamespace(users=np.array([0]), seg=np.array(["고CLV"]), gt={0: np.array([58])}))
    saved = np.arange(50).reshape(1, -1)
    current = saved.copy()
    current[:, [19, 20]] = current[:, [20, 19]]
    context = dict(prepared=prepared, views={"full": None}, arm={"curve": [{"metrics": {"diversity@20": .5}}]}, provenance={})
    with patch.object(d.screen.shared, "validate_prepared"), \
         patch.object(d, "prepare", side_effect=AssertionError("preparation forbidden")), \
         patch.object(d, "_capture_evaluation", side_effect=[({"diversity@20": .55}, current, np.arange(50, 0, -1).reshape(1, -1)),
                                                           ({"diversity@20": .5}, saved, saved)]):
        result = d.probe_failed_readback(source, out, context=context)
    assert failed_readback.read_text() == "original failed readback\n"
    info = json.loads((Path(result["out_dir"]) / "result.json").read_text())
    assert not info["current_probe_readback_passed"]
    assert info["saved_list_replay_matches_recorded"]
    assert info["recommendation_changes"]["20"]["changed_set_users"] == 1
    assert info["recommendation_changes"]["10"]["changed_set_users"] == 0
    assert info["recommendation_changes"]["50"]["changed_set_users"] == 0
    assert not info["strict_gate_bypassed"] and Path(result["zip"]).is_file()
    changes = pd.read_csv(Path(result["out_dir"]) / "topk_item_changes.csv")
    assert len(changes) == 2 and not changes.is_truth.any()


def test_failed_context_recovers_only_selected_runner_locals(monkeypatch):
    namespace = {}
    exec(compile("def run():\n prepared = {}\n views = {}\n arm = {}\n provenance = {}\n unrelated = 'do not expose'\n raise RuntimeError('gate')\n", d.__file__, "exec"), namespace)
    try:
        namespace["run"]()
    except RuntimeError as error:
        monkeypatch.setattr(d.sys, "last_traceback", error.__traceback__, raising=False)
        assert set(d._failed_context()) == {"prepared", "views", "arm", "provenance"}
    monkeypatch.setattr(d.sys, "last_traceback", None)
    with pytest.raises(RuntimeError, match="캐시가 런타임에 없습니다"):
        d._failed_context()
