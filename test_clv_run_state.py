import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from clv_run_state import ProgressStore, RunIdentity, clone_state


def identity(model_id="m1"):
    return RunIdentity(
        stage=model_id,
        model_id=model_id,
        seed=42,
        config_hash="cfg-123",
        source_revision="abc123",
        input_hash="input-123",
    )


def make_trained_model():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return model, optimizer


def assert_state_equal(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert torch.equal(left[name], right[name]), name


def test_epoch_round_trip_restores_model_optimizer_rng_and_best_state(tmp_path):
    model, optimizer = make_trained_model()
    rng = np.random.default_rng(42)
    rng.integers(0, 100, size=9)
    best_state = clone_state(model)
    store = ProgressStore(tmp_path, identity(), heartbeat_interval_sec=0)

    store.save_epoch(
        model,
        optimizer,
        rng,
        epoch=2,
        best_epoch=1,
        best_metric=0.4,
        best_state=best_state,
        bad=1,
        updates=7,
        samples=70,
        history=[{"epoch": 1}, {"epoch": 2}],
        wall_clock_sec=3.5,
    )
    saved_rng_state = rng.bit_generator.state
    expected_rng = np.random.default_rng()
    expected_rng.bit_generator.state = saved_rng_state
    expected_numbers = expected_rng.integers(0, 1000, size=5)

    new_model = torch.nn.Linear(3, 1)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-2)
    new_rng = np.random.default_rng(999)
    restored = store.restore_epoch(new_model, new_optimizer, new_rng)

    assert restored["next_epoch"] == 3
    assert restored["best_epoch"] == 1
    assert restored["bad"] == 1
    assert restored["updates"] == 7
    assert restored["samples"] == 70
    assert_state_equal(model.state_dict(), new_model.state_dict())
    assert_state_equal(best_state, restored["best_state"])
    assert np.array_equal(new_rng.integers(0, 1000, size=5), expected_numbers)
    assert new_optimizer.state_dict()["state"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_checkpoint_identity_mismatch_fails_closed(tmp_path):
    model, optimizer = make_trained_model()
    rng = np.random.default_rng(42)
    ProgressStore(tmp_path, identity()).save_epoch(
        model,
        optimizer,
        rng,
        epoch=1,
        best_epoch=1,
        best_metric=0.4,
        best_state=clone_state(model),
        bad=0,
        updates=1,
        samples=10,
        history=[{"epoch": 1}],
        wall_clock_sec=1.0,
    )
    wrong = ProgressStore(tmp_path, replace(identity(), config_hash="changed"))

    with pytest.raises(RuntimeError, match="identity"):
        wrong.restore_epoch(model, optimizer, rng)


def test_progress_and_heartbeat_are_drive_readable_without_process_state(tmp_path):
    store = ProgressStore(tmp_path, identity(), heartbeat_interval_sec=0)

    store.mark_stage("running", epoch=0, max_epoch=3)
    store.heartbeat(epoch=1, batch=2, batches=5, loss=0.6)
    progress = json.loads((tmp_path / "progress.json").read_text())
    rows = (tmp_path / "progress.csv").read_text().splitlines()

    assert progress["stage"] == "m1"
    assert progress["status"] == "running"
    assert progress["epoch"] == 1
    assert progress["batch"] == 2
    assert progress["batches"] == 5
    assert progress["last_heartbeat_at"]
    assert len(rows) == 3


def test_completed_checkpoint_is_detected(tmp_path):
    store = ProgressStore(tmp_path, identity())
    assert store.is_complete() is False

    store.mark_complete(checkpoint_sha256="deadbeef")

    assert store.is_complete() is True
    assert store.read_progress()["checkpoint_sha256"] == "deadbeef"
