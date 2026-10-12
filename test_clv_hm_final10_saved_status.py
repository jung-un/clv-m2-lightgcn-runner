import json

import pytest

import clv_hm_final10_saved_status as s


def test_saved_results_and_progress_are_read_without_loading_checkpoint(tmp_path):
    root = tmp_path / "same"
    root.mkdir()
    protocol = {"code_version": s.VERSION, "source_revision": "revision", "input_hash": "input",
                "config": {"dataset": "hm", "epochs": 300, "batch_size": 131072,
                           "id_dim": 64, "n_layers": 2, "lr": .0005, "pref_reg": .001,
                           "negative_count": 1, "input_days": 365},
                "data_stats": {"split_boundaries": {
                    "train": {"end_inclusive": "2020-09-15 00:00:00"},
                    "test": {"end_inclusive": "2020-09-22 00:00:00"}}}}
    (root / "protocol.json").write_text(json.dumps(protocol))
    arms = root / "arms"
    arms.mkdir()
    identity = {"stage": s.VERSION, "model_id": s.MODELS[0], "seed": 49,
                "config_hash": "same", "source_revision": "revision", "input_hash": "input"}
    (arms / f"{s.MODELS[0]}_s49.json").write_text(json.dumps({
        "identity": identity, "epochs": 300, "metrics": {"recall@10": .01}}))
    checkpoint = root / "progress/resume" / f"{s.VERSION}_{s.MODELS[1]}_s49_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not a Torch checkpoint; metadata inspection must never load it")
    result = s.inspect_saved(str(tmp_path))
    assert len(result["status"]) == 6
    assert "완료" in result["status"][0]["status"]
    assert "재개" in result["status"][1]["status"]
    assert "새 학습" in result["status"][2]["status"]
    assert len(result["completed_metrics"]) == 1
    assert result["test_evaluations"] == result["new_training"] == 0
    assert result["checkpoint_loaded"] is False
    protocol["config"]["input_days"] = 730
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(FileNotFoundError):
        s.inspect_saved(str(tmp_path))
