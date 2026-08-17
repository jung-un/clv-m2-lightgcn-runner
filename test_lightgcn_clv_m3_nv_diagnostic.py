import lightgcn_clv_m3_nv_diagnostic as diagnostic

from test_clv_m3_nv_diagnostics import _fixture


def test_runner_uses_train_graph_without_model_training(monkeypatch, tmp_path):
    train, graph = _fixture()
    cfg = diagnostic.configure_m3_clv_nv_dunnhumby_run(
        out_dir=str(tmp_path / "dunnhumby")
    )
    monkeypatch.setattr(
        diagnostic,
        "_prepare_train_graph",
        lambda _cfg: (train, graph, 2, 3),
    )

    report = diagnostic.run_graph_diagnostics(cfg)

    assert report["summary"]["n_edges"] == 4
    assert all(tmp_path.as_posix() in path for path in report["paths"].values())
    assert not hasattr(diagnostic, "train_model")
