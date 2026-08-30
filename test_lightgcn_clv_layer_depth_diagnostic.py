from types import SimpleNamespace

import numpy as np
import torch

import lightgcn_clv_layer_depth_diagnostic as diagnostic


class TinyLightGCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_users = 2
        self.n_items = 2
        self.cfg = {"N_LAYERS": 2}
        self.E_u = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]), freeze=False
        )
        self.E_i = torch.nn.Embedding.from_pretrained(
            torch.tensor([[1.0, 1.0], [2.0, 0.0]]), freeze=False
        )
        dense = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        self.adj = dense.to_sparse().coalesce()

    def propagate_pref(self):
        layers = diagnostic.propagation_layers(self)
        full = torch.stack(layers).mean(dim=0)
        return full[: self.n_users], full[self.n_users :]


def test_layer_views_use_exact_lightgcn_averages():
    model = TinyLightGCN()

    layers = diagnostic.propagation_layers(model)
    views = diagnostic.aggregate_layer_views(model, layers)

    initial = torch.cat([model.E_u.weight, model.E_i.weight])
    first = torch.sparse.mm(model.adj, initial)
    second = torch.sparse.mm(model.adj, first)
    assert torch.equal(layers[0], initial)
    assert torch.equal(layers[1], first)
    assert torch.equal(layers[2], second)
    assert torch.equal(views["layer0"][0], initial[:2])
    assert torch.equal(views["layer0_1_mean"][0], ((initial + first) / 2)[:2])
    assert torch.equal(
        views["layer0_1_2_mean"][1], ((initial + first + second) / 3)[2:]
    )


def test_full_layer_view_matches_the_models_normal_preference_embedding():
    model = TinyLightGCN()

    views = diagnostic.aggregate_layer_views(
        model, diagnostic.propagation_layers(model)
    )
    diagnostic.assert_full_view_parity(model, views, atol=0.0)


def test_group_comparison_maps_global_user_ids_not_row_positions():
    users = np.array([100, 1764, 2499])
    membership = diagnostic.membership_for_evaluation_users(
        users,
        fixed_segments=np.array(["저CLV", "고CLV", "중CLV"]),
        high_compositions=np.array(["비고CLV", "N우세 고CLV", "비고CLV"]),
    )

    assert membership.user_idx.tolist() == [100, 1764, 2499]
    assert membership.fixed_clv_segment.tolist() == ["저CLV", "고CLV", "중CLV"]
    assert membership.high_clv_composition.tolist() == [
        "비고CLV",
        "N우세 고CLV",
        "비고CLV",
    ]


def test_topk_overlap_is_set_based_and_grouped():
    users = np.array([5, 9])
    reference = np.array([[1, 2, 3], [4, 5, 6]])
    alternative = np.array([[3, 2, 1], [4, 7, 8]])
    membership = diagnostic.membership_for_evaluation_users(
        users,
        fixed_segments=np.array(["저CLV", "고CLV"]),
        high_compositions=np.array(["비고CLV", "V우세 고CLV"]),
    )

    overlap = diagnostic.topk_overlap_rows(
        users=users,
        reference_topk=reference,
        alternative_topk=alternative,
        membership=membership,
        k=3,
    )

    by_user = overlap.set_index("user_idx")
    assert by_user.at[5, "topk_set_changed"] == 0.0
    assert by_user.at[5, "topk_order_changed"] == 1.0
    assert by_user.at[5, "topk_jaccard"] == 1.0
    assert by_user.at[9, "topk_set_changed"] == 1.0
    assert np.isclose(by_user.at[9, "topk_jaccard"], 1.0 / 5.0)


def test_preflight_is_checkpoint_only_and_keeps_protected_splits_closed(tmp_path):
    cfg = diagnostic.configure_layer_depth_diagnostic(
        "dunnhumby",
        out_dir=str(tmp_path / "out"),
        baseline_result_dir=str(tmp_path / "baseline"),
    )

    summary = diagnostic.preflight_summary(cfg)

    assert summary["training"] is False
    assert summary["checkpoint_selection"] is False
    assert summary["final_test_executed"] is False
    assert summary["holdout_executed"] is False
    assert summary["views"] == [
        "layer0",
        "layer0_1_mean",
        "layer0_1_2_mean",
    ]
