from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import clv_core_features as core
import lightgcn_clv_moe as moe
import lightgcn_clv_residual as residual
import lightgcn_clv_v3 as v3
from clv_run_state import ProgressStore, RunIdentity


def store(root, stage):
    return ProgressStore(
        root,
        RunIdentity(stage, stage, 42, "cfg", "source", "input"),
        heartbeat_interval_sec=0,
    )


def assert_modules_equal(left, right):
    for name, value in left.state_dict().items():
        assert torch.equal(value, right.state_dict()[name]), name


class TinyM1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(0.0))
        self.E_u = torch.nn.Embedding(2, 2)
        self.E_i = torch.nn.Embedding(3, 2)

    def bpr_loss(self, users, positives, negatives, gate, lam, loss_weight):
        loss = (self.w - 3.0).square()
        return loss, {
            "bpr": float(loss.detach()),
            "p_correct": float(self.w.detach().sigmoid()),
        }


def m1_cfg(epochs):
    return {
        "WD": 0.0,
        "REG_MODE": "batch_l2",
        "LR": 0.01,
        "BATCH_SIZE": 2,
        "EPOCHS": epochs,
        "EARLY_STOP": 99,
        "NEG_MODE": "uniform",
        "SELECT_K": 10,
        "SELECT_METRIC": "recall",
    }


def tiny_data():
    return {
        "tr_u": np.array([0, 1, 0, 1]),
        "tr_i": np.array([0, 1, 1, 2]),
        "pos_key": np.array([0, 1, 4, 5]),
        "n_items": 3,
        "item_cat": np.zeros(3, dtype=int),
        "cat_items": {0: np.arange(3)},
        "csr_ptr": np.array([0, 2, 4]),
        "csr_items": np.array([0, 1, 1, 2]),
    }


def test_m1_resume_runs_only_missing_epoch_and_matches_uninterrupted(monkeypatch, tmp_path):
    monkeypatch.setattr(
        v3,
        "evaluate",
        lambda model, *_args, **_kwargs: {
            "overall": {10: {"recall": float(model.w.detach())}}
        },
    )
    data = tiny_data()
    torch.manual_seed(123)
    full = TinyM1()
    v3.train_phase(
        full, [full.w], data, None, 0.0, m1_cfg(3), 42, "m1", None, None,
        progress_store=store(tmp_path / "full", "m1"),
    )

    torch.manual_seed(123)
    first = TinyM1()
    first_stats = v3.train_phase(
        first, [first.w], data, None, 0.0, m1_cfg(2), 42, "m1", None, None,
        progress_store=store(tmp_path / "resume", "m1"),
    )
    torch.manual_seed(999)
    resumed = TinyM1()
    resumed_stats = v3.train_phase(
        resumed, [resumed.w], data, None, 0.0, m1_cfg(3), 42, "m1", None, None,
        progress_store=store(tmp_path / "resume", "m1"),
    )

    assert first_stats["epochs_run"] == 2
    assert resumed_stats["resumed_from_epoch"] == 2
    assert resumed_stats["new_epochs_run"] == 1
    assert_modules_equal(full, resumed)


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(0.0))
        self._pref_cache = None

    def pref_params(self):
        return [self.base]

    def freeze_pref_and_cache(self):
        self.base.requires_grad_(False)
        self._pref_cache = True


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = TinyBase()
        self.w = torch.nn.Parameter(torch.tensor(0.0))

    def adapter_parameters(self):
        return [self.w]

    def bpr_loss(self, users, positives, negatives, lam):
        return (self.w - 3.0).square()


def adapter_cfg(epochs):
    return SimpleNamespace(
        adapter_lr=0.01,
        base_lr=0.001,
        lambda_train=1.0,
        max_epochs=epochs,
        frozen_epochs=epochs,
        patience=99,
    )


def test_adapter_resume_matches_uninterrupted(tmp_path):
    data = tiny_data()
    base_cfg = {"BATCH_SIZE": 2, "NEG_MODE": "uniform"}
    full = TinyAdapter()
    moe.train_moe(
        full, data, base_cfg, adapter_cfg(3), 42,
        lambda model: float(model.w.detach()), freeze_base=True,
        progress_store=store(tmp_path / "full", "adapter"),
    )

    first = TinyAdapter()
    moe.train_moe(
        first, data, base_cfg, adapter_cfg(2), 42,
        lambda model: float(model.w.detach()), freeze_base=True,
        progress_store=store(tmp_path / "resume", "adapter"),
    )
    resumed = TinyAdapter()
    stats = moe.train_moe(
        resumed, data, base_cfg, adapter_cfg(3), 42,
        lambda model: float(model.w.detach()), freeze_base=True,
        progress_store=store(tmp_path / "resume", "adapter"),
    )

    assert stats["resumed_from_epoch"] == 2
    assert stats["new_epochs_run"] == 1
    assert_modules_equal(full, resumed)


def target_train():
    rows = []
    for day in range(1, 51):
        for user, price in ((0, 2.0), (1, 5.0)):
            if user == 1 and day % 2:
                continue
            rows.append(
                {
                    "u_idx": user,
                    "i_idx": day % 4,
                    "cat_idx": day % 2,
                    "b_raw": f"u{user}-{day}",
                    "t": day,
                    "up": price,
                    "v": price,
                }
            )
    return pd.DataFrame(rows)


def test_encoder_selection_and_final_fit_resume_from_epoch_checkpoint(tmp_path):
    train = target_train()
    anchors = residual.build_anchor_examples(
        train, 2, False, input_days=10, target_days=4,
        anchor_offsets=(12, 8, 4),
    )
    snapshot = residual.build_final_snapshot(train, 2, False, input_days=10)
    progress = {
        "select": store(tmp_path, "encoder_select"),
        "final": store(tmp_path, "encoder_final"),
    }
    core.train_clv_core_encoder(
        anchors, snapshot, encoder_epochs=1, encoder_patience=10,
        encoder_batch_size=16, encoder_lr=1e-3, seed=42,
        device=torch.device("cpu"), progress_stores=progress,
    )
    artifact = core.train_clv_core_encoder(
        anchors, snapshot, encoder_epochs=2, encoder_patience=10,
        encoder_batch_size=16, encoder_lr=1e-3, seed=42,
        device=torch.device("cpu"), progress_stores=progress,
    )

    selection = torch.load(progress["select"].latest_checkpoint, weights_only=False)
    assert selection["epoch"] == 2
    assert artifact.diagnostics["resumed_from_epoch"] == 1
