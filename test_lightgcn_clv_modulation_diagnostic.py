from dataclasses import replace

import numpy as np

import lightgcn_clv_modulation_diagnostic as diagnostic


def test_runner_evaluates_checkpoint_views_without_training(monkeypatch, tmp_path):
    cfg = replace(
        diagnostic.modulation.configure_modulation_dunnhumby_run(),
        out_dir=str(tmp_path),
    )
    prepared = {
        "out_dir": tmp_path,
        "input_hash": "input",
        "base_cfg": {"N_BOOT": 20},
    }

    class FakeModel:
        def set_eval_axes(self, mode):
            self.mode = mode

    model = FakeModel()
    monkeypatch.setattr(diagnostic.modulation, "_prepare", lambda _cfg: prepared)
    monkeypatch.setattr(diagnostic.modulation, "_build_model", lambda *_: model)
    monkeypatch.setattr(
        diagnostic,
        "load_modulation_checkpoint",
        lambda *_: {"state": {}},
    )
    monkeypatch.setattr(
        diagnostic,
        "modulation_structure",
        lambda _model: {"scale_both": {"user_mean": 1.0}},
    )

    values = {
        "none": 0.0,
        "n_only": 1.0,
        "v_only": 2.0,
        "both": 3.0,
        "shuffled_user": 4.0,
    }

    def fake_eval(_model, _prepared, mode):
        value = values[mode]
        metrics = {"recall@10": value, "revenue@10": value}
        per_user = {
            name: np.array([value, value + 1.0])
            for name in ("recall", "ndcg", "revenue", "arp")
        }
        return metrics, per_user

    monkeypatch.setattr(diagnostic, "_evaluate_view", fake_eval)
    monkeypatch.setattr(
        diagnostic,
        "_evaluate_shuffled_user",
        lambda model, prepared, seed: fake_eval(
            model, prepared, "shuffled_user"
        ),
    )

    frame = diagnostic.run_checkpoint_diagnostics(
        cfg, checkpoint_path=str(tmp_path / "model.pt")
    )

    assert frame["view"].tolist() == list(diagnostic.VIEW_MODES)
    assert set(frame.attrs["paired"]["view"]) == set(diagnostic.VIEW_MODES[1:])
    assert "train" not in diagnostic.run_checkpoint_diagnostics.__name__


def test_checkpoint_finder_uses_latest_modulation_checkpoint(tmp_path):
    first = tmp_path / "m2_clv_modulation_dunnhumby_s42_a.pt"
    second = tmp_path / "m2_clv_modulation_dunnhumby_s42_b.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    first.touch()
    second.touch()

    assert diagnostic.find_modulation_checkpoint(tmp_path) in {first, second}
