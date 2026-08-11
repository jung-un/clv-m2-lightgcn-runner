# Single-Adapter CLV Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only M2 runner that tests whether the promising single adapter's economic improvement depends on user-specific CLV-related behavior profiles rather than adapter capacity, item features, or extra fine-tuning.

**Architecture:** Preserve the existing binary-LightGCN/plain-BPR pipeline and extend `CLVMixtureEmbeddingModel` with five parameter-matched single-adapter input variants. Add a focused `lightgcn_clv_single.py` orchestrator so the reviewed MoE runner remains behaviorally unchanged; the new runner evaluates `single_full` first, conditionally runs matched controls, and may reuse the saved Dunnhumby full checkpoint only after fail-closed provenance and metric round-trip checks.

**Tech Stack:** Python 3.11, PyTorch, NumPy, pandas, pytest, ruff, Jupyter/Colab JSON.

## Global Constraints

- Research position is M2 only: graph mode stays `binary`; loss stays unweighted `plain BPR`; negative sampling stays `uniform`.
- Screening is fixed to seed `(42,)`, validation-only; test and holdout ground truth must not be constructed.
- User input stays the current 51-dimensional train-only CLV-related behavior representation.
- Item input stays the current six train-only numeric features plus category embedding.
- All five variants keep identical input dimensions, parameter shapes, initialization rule, maximum epochs, patience, and learning rates.
- `single_zero_user` and `single_base_only` zero values but preserve `has_profile`; `single_zero_item` and `single_base_only` zero item features but preserve `valid_item`.
- Lambda grid is exactly `(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)`.
- A positive lambda is eligible only when Recall/NDCG at K=`10,20,50` are each within 1% relative decline from external M1 and `revenue@10` is strictly greater than external M1.
- `single_full` must beat the independently selected `single_zero_user`, `single_shuffled_user`, and `single_base_only` values of `revenue@10`; `single_zero_item` is a mechanism ablation, not a mandatory success gate.
- All absolute lambda curves, paired deltas, exposure metrics, feature schema, hashes, checkpoints, training counts, and the authoritative screening decision must be persisted.
- Existing MoE behavior and historical result fingerprints must not change.
- No high-cost H&M or Dunnhumby training is part of implementation verification.

---

## File Structure

- Modify `clv_moe_model.py`: define parameter-matched single-adapter input variants while retaining `single_adapter` as a backward-compatible alias for `single_full`.
- Modify `test_clv_moe_model.py`: runtime invariants for zero/shuffle behavior, masks, parameter equality, alias equality, and lambda-zero identity.
- Create `lightgcn_clv_single.py`: configuration, preflight, reuse validation, validation runner, selection, persistence, and fail-safe CLI for the new experiment.
- Create `test_lightgcn_clv_single.py`: policy, provenance, orchestration, schema, and protected-split runtime tests.
- Create `clv_single_adapter_colab.ipynb`: reviewed Colab entry point with dataset presets, optional Dunnhumby reuse paths, preflight, approval gate, and final decision display.
- Modify `RESEARCH_STATUS.md` outside the Git repository at `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`: record implementation state and explicitly distinguish it from high-cost results.

### Task 1: Add matched single-adapter input variants

**Files:**
- Modify: `clv_moe_model.py:38-180`
- Test: `test_clv_moe_model.py`

**Interfaces:**
- Consumes: `UserProfileArtifact`, `ItemProfileArtifact`, existing `CLVMixtureEmbeddingModel` constructor.
- Produces: `SINGLE_VARIANTS`, `canonical_single_variant(control: str) -> str | None`, and accepted controls `single_full`, `single_zero_user`, `single_shuffled_user`, `single_zero_item`, `single_base_only` while preserving legacy `single_adapter`.

- [ ] **Step 1: Write failing tests for the variant buffers and masks**

Add this helper and tests to `test_clv_moe_model.py`:

```python
def _single(control, seed=42):
    return _model(control=control, seed=seed)


def test_single_full_is_exact_legacy_single_adapter_alias():
    legacy = _single("single_adapter")
    full = _single("single_full")
    assert legacy.single_variant == full.single_variant == "single_full"
    torch.testing.assert_close(legacy.routed_profile, full.routed_profile)
    torch.testing.assert_close(legacy.item_numeric, full.item_numeric)
    torch.testing.assert_close(
        legacy.score_all(torch.arange(4), 1.0),
        full.score_all(torch.arange(4), 1.0),
    )


def test_single_zero_user_preserves_mask_and_zeros_only_user_profile():
    full = _single("single_full")
    zero = _single("single_zero_user")
    torch.testing.assert_close(zero.routed_profile, torch.zeros_like(zero.routed_profile))
    torch.testing.assert_close(zero.item_numeric, full.item_numeric)
    assert torch.equal(zero.has_profile, full.has_profile)


def test_single_zero_item_preserves_mask_and_zeros_item_side_features():
    full = _single("single_full")
    zero = _single("single_zero_item")
    torch.testing.assert_close(zero.item_numeric, torch.zeros_like(zero.item_numeric))
    assert torch.equal(zero.item_category_ids, torch.zeros_like(zero.item_category_ids))
    assert torch.equal(zero.valid_item, full.valid_item)
    torch.testing.assert_close(zero.routed_profile, full.routed_profile)


def test_single_base_only_zeros_both_added_inputs_without_disabling_residual():
    model = _single("single_base_only")
    assert torch.count_nonzero(model.routed_profile) == 0
    assert torch.count_nonzero(model.item_numeric) == 0
    assert torch.count_nonzero(model.item_category_ids) == 0
    assert model.has_profile.all() and model.valid_item.all()
    assert not torch.equal(
        model.score_all(torch.arange(4), 1.0),
        model.base_score_all(torch.arange(4)),
    )
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run:

```bash
pytest -q \
  test_clv_moe_model.py::test_single_full_is_exact_legacy_single_adapter_alias \
  test_clv_moe_model.py::test_single_zero_user_preserves_mask_and_zeros_only_user_profile \
  test_clv_moe_model.py::test_single_zero_item_preserves_mask_and_zeros_item_side_features \
  test_clv_moe_model.py::test_single_base_only_zeros_both_added_inputs_without_disabling_residual
```

Expected: FAIL because the new controls and `single_variant` do not exist.

- [ ] **Step 3: Implement canonical variant handling**

In `clv_moe_model.py`, add:

```python
SINGLE_VARIANTS = frozenset(
    {
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_zero_item",
        "single_base_only",
    }
)


def canonical_single_variant(control: str) -> str | None:
    if control == "single_adapter":
        return "single_full"
    return control if control in SINGLE_VARIANTS else None
```

Update `CONTROLS`, then derive and apply the variant without changing tensor dimensions:

```python
self.single_variant = canonical_single_variant(control)
is_single = self.single_variant is not None
self.expert_count = 1 if is_single else requested_expert_count

routed = values.clone()
if control == "shuffled_clv" or self.single_variant == "single_shuffled_user":
    routed = _permute_valid_rows(routed, valid_user, seed)
if self.single_variant in {"single_zero_user", "single_base_only"}:
    routed.zero_()
if self.single_variant in {"single_zero_item", "single_base_only"}:
    item_numeric.zero_()
    item_categories.zero_()
```

Use `is_single` in capacity matching, gate construction, and `routing_weights`. Keep `has_profile` and `valid_item` unchanged.

- [ ] **Step 4: Add deterministic shuffle and capacity/state-shape tests**

```python
def test_single_shuffled_user_is_seeded_permutation_of_valid_profiles():
    full = _single("single_full")
    a = _single("single_shuffled_user", seed=42)
    b = _single("single_shuffled_user", seed=42)
    torch.testing.assert_close(a.routed_profile, b.routed_profile)
    assert not torch.equal(a.routed_profile, full.routed_profile)
    assert sorted(a.routed_profile[:, 0].tolist()) == sorted(full.routed_profile[:, 0].tolist())


def test_all_single_variants_have_identical_parameter_names_and_shapes():
    controls = ["single_full", "single_zero_user", "single_shuffled_user",
                "single_zero_item", "single_base_only"]
    signatures = []
    for control in controls:
        model = _single(control)
        signatures.append([(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()])
        assert model.expert_count == 1
        torch.testing.assert_close(
            model.score_all(torch.arange(4), 0.0),
            model.base_score_all(torch.arange(4)),
            rtol=0,
            atol=0,
        )
    assert signatures.count(signatures[0]) == len(signatures)
```

- [ ] **Step 5: Run model tests and lint**

Run:

```bash
pytest -q test_clv_moe_model.py
ruff check clv_moe_model.py test_clv_moe_model.py
```

Expected: all tests PASS; ruff reports no errors.

- [ ] **Step 6: Commit Task 1**

```bash
git add clv_moe_model.py test_clv_moe_model.py
git commit -m "feat: add matched single-adapter variants"
```

### Task 2: Define validation-only single-adapter policy and decision logic

**Files:**
- Create: `lightgcn_clv_single.py`
- Create: `test_lightgcn_clv_single.py`

**Interfaces:**
- Consumes: `lightgcn_clv_moe.MoEConfig`, `configure_moe_run`, `validate_moe_config`, `select_lambda`.
- Produces: `configure_single_run(dataset: str, **overrides) -> MoEConfig`, `validate_single_config(cfg: MoEConfig) -> MoEConfig`, `preflight_summary(cfg: MoEConfig) -> dict`, and `single_screening_decision(rows, selected, selection_success) -> dict`.

- [ ] **Step 1: Write failing policy tests**

Create `test_lightgcn_clv_single.py` with:

```python
import dataclasses

import pytest


def test_default_single_screening_is_seed42_validation_only():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    summary = single.preflight_summary(cfg)
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert summary["primary_model_id"] == "single_full"
    assert summary["required_controls"] == [
        "single_zero_user", "single_shuffled_user", "single_base_only"
    ]
    assert summary["mechanism_controls"] == ["single_zero_item"]
    assert summary["graph_mode"] == "binary"
    assert summary["loss_mode"] == "plain"


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_direct_dataclass_cannot_open_protected_splits(field):
    import lightgcn_clv_single as single
    import lightgcn_clv_moe as moe

    cfg = dataclasses.replace(moe.MoEConfig(), **{field: True})
    with pytest.raises(ValueError, match="screening-only"):
        single.validate_single_config(cfg)


def test_single_screening_decision_requires_full_to_beat_required_controls():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
        "pref_continue": 0.0,
    }
    rows = [
        {"seed": 42, "split": "val", "model_id": model_id,
         "lambda": selected[model_id], "revenue@10": revenue}
        for model_id, revenue in {
            "single_full": 1.10,
            "single_zero_user": 1.04,
            "single_shuffled_user": 1.03,
            "single_zero_item": 1.12,
            "single_base_only": 1.02,
            "pref_continue": 1.01,
        }.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is True
    assert decision["mechanism_comparison"]["single_zero_item"] == 1.12
    rows[1]["revenue@10"] = 1.11
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False
    assert decision["failed_controls"] == ["single_zero_user"]
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `pytest -q test_lightgcn_clv_single.py`

Expected: import failure because `lightgcn_clv_single.py` does not exist.

- [ ] **Step 3: Implement the configuration and decision module**

Create `lightgcn_clv_single.py` with these constants and functions:

```python
PRIMARY_MODEL_ID = "single_full"
REQUIRED_CONTROLS = (
    "single_zero_user",
    "single_shuffled_user",
    "single_base_only",
)
MECHANISM_CONTROLS = ("single_zero_item",)
ALL_SINGLE_MODELS = (PRIMARY_MODEL_ID, *REQUIRED_CONTROLS, *MECHANISM_CONTROLS)
CODE_VERSION = "clv-single-identification-v1.0"


def configure_single_run(dataset: str, **overrides) -> moe.MoEConfig:
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "lambda_eval": (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
        "run_controls_after_success": True,
        "out_dir": f"{v3.default_out_dir(dataset)}_clv_single",
    }
    return validate_single_config(moe.configure_moe_run(dataset, **(defaults | overrides)))


def validate_single_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    cfg = moe.validate_moe_config(cfg)
    if cfg.seed_list != (42,):
        raise ValueError("single-adapter screening-only runner requires seed 42")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("single-adapter screening-only runner cannot open test/holdout")
    if cfg.lambda_eval != (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        raise ValueError("single-adapter lambda grid is frozen by the approved design")
    return cfg
```

Implement `single_screening_decision` by reading each model's independently selected row, failing if `single_full` did not itself pass selection, and comparing only `REQUIRED_CONTROLS` for the authoritative success flag. Store `single_zero_item` under `mechanism_comparison` without adding it to `failed_controls`.

- [ ] **Step 4: Add direct-run fail-closed and strict-greater tests**

```python
def test_decision_tie_with_required_control_is_failure():
    import lightgcn_clv_single as single
    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
    }
    values = {
        "single_full": 1.10,
        "single_zero_user": 1.10,
        "single_shuffled_user": 1.03,
        "single_zero_item": 1.12,
        "single_base_only": 1.02,
    }
    rows = [
        {"seed": 42, "split": "val", "model_id": model_id,
         "lambda": selected[model_id], "revenue@10": revenue}
        for model_id, revenue in values.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False


def test_validate_rejects_changed_lambda_grid():
    import dataclasses
    import pytest
    import lightgcn_clv_single as single
    cfg = single.configure_single_run("dunnhumby")
    with pytest.raises(ValueError, match="lambda grid"):
        single.validate_single_config(dataclasses.replace(cfg, lambda_eval=(0.0, 1.0)))
```

- [ ] **Step 5: Run policy tests and lint**

Run:

```bash
pytest -q test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
```

Expected: all policy tests PASS; ruff reports no errors.

- [ ] **Step 6: Commit Task 2**

```bash
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: define single-adapter screening policy"
```

### Task 3: Add fail-closed reuse of the saved Dunnhumby full model

**Files:**
- Modify: `lightgcn_clv_single.py`
- Modify: `test_lightgcn_clv_single.py`

**Interfaces:**
- Consumes: saved MoE result JSON, `single_adapter` checkpoint, current input manifest, external M1 state, current features and validation cache.
- Produces: `ReusableSingleFull` dataclass and `load_reusable_single_full(...) -> ReusableSingleFull`; raises `RuntimeError` before reuse on any mismatch.

- [ ] **Step 1: Create the concrete reuse fixture and failing provenance tests**

Add these imports and fixture definitions to `test_lightgcn_clv_single.py`. The fixture writes a real torch payload for the encoder-value hash while stubbing only the heavyweight model rebuild and evaluator:

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass
class ReuseFixture:
    result_json: Path
    current_manifest: dict
    base_hash: str
    cfg: object
    base_cfg: dict
    context: dict
    data: dict
    rows_by_lambda: dict


def _reuse_metric_row(lam):
    row = {
        "seed": 42,
        "model_id": "single_adapter",
        "split": "val",
        "lambda": lam,
        "role": "control",
        "revenue@10": 1.0 + 0.01 * lam,
        "arp@10": 0.2,
    }
    for k in (10, 20, 50):
        row[f"recall@{k}"] = 0.1
        row[f"ndcg@{k}"] = 0.1
        row[f"n_distinct@{k}"] = 3
        row[f"exposure_entropy@{k}"] = 1.0
        row[f"eff_catalog@{k}"] = 2.7
        row[f"top10_share@{k}"] = 0.5
        row[f"top100_share@{k}"] = 1.0
    return row


@pytest.fixture
def reuse_fixture(tmp_path, monkeypatch):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    manifest = {
        "transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/item", "bytes": 2, "sha256": "bb"},
    }
    ev_all = np.array([1.0, 2.0], dtype=np.float32)
    checkpoint = tmp_path / "single_adapter.pt"
    torch.save({"ev_all": ev_all}, checkpoint)
    rows = {float(lam): _reuse_metric_row(float(lam)) for lam in cfg.lambda_eval}
    payload = {
        "source_revision": "legacy-revision",
        "input_manifest": manifest,
        "config": asdict(cfg),
        "baseline_state_hashes": {"42": "base-state"},
        "feature_schema": {
            "user": ["u0"],
            "item_numeric": ["i0"],
        },
        "checkpoint_paths": {"single_adapter_s42": str(checkpoint)},
        "absolute_rows": list(rows.values()),
        "training": {"single_adapter_s42": {"base_updates_at_best": 3}},
        "moe_diagnostics": {"single_adapter_s42": {"parameter_match_ratio": 1.0}},
    }
    result_json = tmp_path / "legacy.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    context = {
        "artifact": SimpleNamespace(ev_all=ev_all),
        "user_profile": SimpleNamespace(feature_names=("u0",)),
        "item_profile": SimpleNamespace(numeric_names=("i0",)),
        "caches": {"val": object()},
    }
    monkeypatch.setattr(moe, "load_moe_checkpoint", lambda *args, **kwargs: object())

    def fake_flat(model, lam, *args, **kwargs):
        row = rows[float(lam)]
        metrics = {
            key: value for key, value in row.items()
            if key not in {"seed", "model_id", "split", "lambda", "role"}
        }
        return metrics, None

    monkeypatch.setattr(moe, "_flat_evaluation", fake_flat)
    return ReuseFixture(
        result_json=result_json,
        current_manifest=manifest,
        base_hash="base-state",
        cfg=cfg,
        base_cfg={"K_LIST": [10, 20, 50]},
        context=context,
        data={"n_items": 2},
        rows_by_lambda=rows,
    )
```

Then add:

```python
def test_reuse_rejects_input_manifest_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    fixture = reuse_fixture
    changed = fixture.current_manifest | {
        "transactions": {"path": "/x", "bytes": 1, "sha256": "changed"}
    }
    with pytest.raises(RuntimeError, match="input manifest"):
        single.load_reusable_single_full(
            fixture.result_json,
            current_manifest=changed,
            baseline_state_hash=fixture.base_hash,
            cfg=fixture.cfg,
            base_cfg=fixture.base_cfg,
            context=fixture.context,
            data=fixture.data,
        )


def test_reuse_rejects_m1_state_or_feature_schema_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    with pytest.raises(RuntimeError, match="M1 state"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash="wrong",
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )
```

- [ ] **Step 2: Run reuse tests and confirm RED**

Run: `pytest -q test_lightgcn_clv_single.py -k reuse`

Expected: FAIL because the reuse API does not exist.

- [ ] **Step 3: Implement explicit compatibility keys and array hashing**

```python
REUSE_CONFIG_KEYS = (
    "dataset", "seed_list", "input_days", "target_days", "anchor_offsets",
    "encoder_epochs", "encoder_patience", "encoder_batch_size", "encoder_lr",
    "expert_count", "expert_hidden_dim", "expert_dim", "category_dim",
    "frozen_epochs", "max_epochs", "patience", "adapter_lr", "base_lr",
    "lambda_train", "lambda_eval", "accuracy_tolerance",
)


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    payload = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()
```

Define:

```python
@dataclass(frozen=True)
class ReusableSingleFull:
    model: CLVMixtureEmbeddingModel
    rows: tuple[dict, ...]
    training: dict
    diagnostics: dict
    result_json_sha256: str
    legacy_source_revision: str
    legacy_checkpoint: str
```

`load_reusable_single_full` must validate, in order: JSON structure; exact current input manifest; seed-42 M1 state hash; all `REUSE_CONFIG_KEYS`; user/item feature names; checkpoint existence; checkpoint `ev_all` hash against current encoder `ev_all`; successful `load_moe_checkpoint(..., control="single_adapter")`; and reevaluated validation metrics for every lambda against the JSON `single_adapter` rows using `np.isclose(rtol=0, atol=5e-8)` on all numeric metrics returned by `_flat_evaluation`.

- [ ] **Step 4: Add positive round-trip and tampered-metric tests**

```python
def test_reuse_accepts_exact_legacy_full_and_relabels_rows(reuse_fixture):
    import lightgcn_clv_single as single
    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
        current_manifest=reuse_fixture.current_manifest,
        baseline_state_hash=reuse_fixture.base_hash,
        cfg=reuse_fixture.cfg,
        base_cfg=reuse_fixture.base_cfg,
        context=reuse_fixture.context,
        data=reuse_fixture.data,
    )
    assert {row["model_id"] for row in reused.rows} == {"single_full"}
    assert tuple(row["lambda"] for row in reused.rows) == reuse_fixture.cfg.lambda_eval
    assert reused.result_json_sha256


def test_reuse_rejects_metric_round_trip_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    payload = json.loads(reuse_fixture.result_json.read_text())
    row = next(row for row in payload["absolute_rows"] if row["model_id"] == "single_adapter")
    row["revenue@10"] += 0.01
    reuse_fixture.result_json.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="metric round-trip"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )
```

- [ ] **Step 5: Run reuse tests, then all new tests**

Run:

```bash
pytest -q test_lightgcn_clv_single.py -k reuse
pytest -q test_clv_moe_model.py test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
```

Expected: all PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: validate reusable single-adapter results"
```

### Task 4: Implement conditional validation orchestration and persistence

**Files:**
- Modify: `lightgcn_clv_single.py`
- Modify: `test_lightgcn_clv_single.py`

**Interfaces:**
- Consumes: `configure_single_run`, existing feature/encoder/M1 helpers, model variants, optional reusable full result.
- Produces: `run_experiment(cfg: MoEConfig | None = None, *, reuse_full_result_json: str | Path | None = None) -> pd.DataFrame` with `frame.attrs["screening_decision"]` and three saved artifacts.
- Produces internal typed boundaries: `PreparedSingleContext`, `VariantRun`, `_prepare_validation_context(cfg) -> PreparedSingleContext`, `_train_evaluate_variant(prepared, cfg, model_id) -> VariantRun`, `_select_models(rows, baseline, model_ids) -> tuple[dict, dict, dict]`, and `_persist_result(...) -> pd.DataFrame`.

- [ ] **Step 1: Write a failing orchestration test with stubbed training**

Define this test helper before the orchestration tests. It patches the new runner's own typed boundaries, so it never reads a dataset or trains a model:

```python
def _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue):
    import lightgcn_clv_single as single

    calls = {"controls": []}
    baseline = _reuse_metric_row(0.0) | {
        "model_id": "m1", "role": "baseline", "revenue@10": 1.0
    }
    prepared = SimpleNamespace(
        out_dir=tmp_path,
        baseline_row=baseline,
        baseline_metrics=baseline,
        baseline_per_user={
            "recall": np.zeros(2), "ndcg": np.zeros(2),
            "revenue": np.zeros(2), "arp": np.zeros(2),
        },
        input_manifest={"transactions": {}, "item_metadata": {}},
        baseline_state_hash="base-state",
        base_cfg={"N_BOOT": 10, "K_LIST": [10, 20, 50]},
        data={"data_stats": {}},
        context={
            "user_profile": SimpleNamespace(feature_names=("u0",)),
            "item_profile": SimpleNamespace(numeric_names=("i0",)),
            "artifact": SimpleNamespace(diagnostics={}),
        },
        source_revision="test-revision",
    )
    monkeypatch.setattr(single, "_prepare_validation_context", lambda cfg: prepared)

    def fake_variant(prepared, cfg, model_id):
        calls["controls"].append(model_id)
        revenue = full_revenue if model_id == "single_full" else 1.01
        rows = []
        per_user = {}
        for lam in cfg.lambda_eval:
            row = _reuse_metric_row(float(lam)) | {
                "model_id": model_id,
                "role": "model" if model_id == "single_full" else "control",
                "revenue@10": revenue if lam == 1.0 else 1.0,
            }
            rows.append(row)
            per_user[float(lam)] = prepared.baseline_per_user
        return single.VariantRun(
            model_id=model_id,
            rows=tuple(rows),
            per_user=per_user,
            training={"base_updates_at_best": 3},
            diagnostics={"parameter_match_ratio": 1.0},
            checkpoint=str(tmp_path / f"{model_id}.pt"),
            reuse_provenance=None,
        )

    monkeypatch.setattr(single, "_train_evaluate_variant", fake_variant)
    pref_row = baseline | {
        "model_id": "pref_continue",
        "role": "control",
        "lambda": 0.0,
        "revenue@10": 1.0,
    }
    monkeypatch.setattr(single, "_run_pref_continue", lambda *args, **kwargs: pref_row)
    return calls
```

Use it in the test and return deterministic metrics such that `single_full` succeeds:

```python
def test_runner_trains_full_then_all_controls_only_after_success(monkeypatch, tmp_path):
    import lightgcn_clv_single as single
    calls = _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    frame = single.run_experiment(cfg)
    assert calls["controls"] == [
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_base_only",
        "single_zero_item",
    ]
    assert set(frame.model_id) >= {"m1", "single_full", *single.REQUIRED_CONTROLS,
                                   *single.MECHANISM_CONTROLS, "pref_continue"}
    assert frame.attrs["screening_decision"]["success"] is True
```

Add a second case with `full_revenue=0.99` and assert no control or `pref_continue` model is trained and the decision is false.

- [ ] **Step 2: Run orchestration tests and confirm RED**

Run: `pytest -q test_lightgcn_clv_single.py -k runner`

Expected: FAIL because `run_experiment` is not implemented.

- [ ] **Step 3: Implement the runner in explicit phases**

Use the existing MoE helpers without changing `lightgcn_clv_moe.run_experiment`:

```python
def run_experiment(cfg=None, *, reuse_full_result_json=None):
    cfg = validate_single_config(cfg or configure_single_run("dunnhumby"))
    prepared = _prepare_validation_context(cfg)
    rows = [prepared.baseline_row]
    full = _reuse_or_train_full(prepared, cfg, reuse_full_result_json)
    rows.extend(full.rows)
    selected, selection_success, selection_tables = _select_models(
        rows, prepared.baseline_metrics, (PRIMARY_MODEL_ID,)
    )
    controls = {}
    if selection_success[PRIMARY_MODEL_ID] and cfg.run_controls_after_success:
        for model_id in (*REQUIRED_CONTROLS, *MECHANISM_CONTROLS):
            controls[model_id] = _train_evaluate_variant(prepared, cfg, model_id)
            rows.extend(controls[model_id].rows)
        selected, selection_success, selection_tables = _select_models(
            rows, prepared.baseline_metrics, ALL_SINGLE_MODELS
        )
        pref_row = _run_pref_continue(prepared, cfg, full.training)
        if pref_row is not None:
            rows.append(pref_row)
        selected["pref_continue"] = 0.0
        selection_success["pref_continue"] = True
    decision = single_screening_decision(rows, selected, selection_success)
    return _persist_result(prepared, cfg, rows, selected, selection_success,
                           selection_tables, decision, full, controls)
```

`_prepare_validation_context` must construct only train and validation artifacts, use an external pure M1 with the same seed, and retain M1 per-user metrics for paired deltas. `_train_evaluate_variant` must start each variant from a fresh copy of the exact same external M1 state and call existing `train_moe` with `freeze_base=False`.

- [ ] **Step 4: Implement selected-row deltas and result schema**

Persist:

```python
stem = f"clv_single_{cfg.dataset}_{fingerprint}"
frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.8f")
pd.DataFrame(delta_records).to_csv(out_dir / f"{stem}_delta.csv", index=False)
json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
```

The JSON payload must contain: `code_version`, `source_revision`, `result_fingerprint`, `input_manifest`, `config`, `base_config`, `data_stats`, `feature_schema`, `variant_definitions`, `baseline_state_hashes`, `selected_lambda`, `lambda_selection_success`, `screening_decision`, `selection_tables`, `encoder_diagnostics`, `training`, `diagnostics`, `checkpoint_paths`, `reuse_provenance`, `absolute_rows`, `delta`, and interpretation strings for CLV and revenue.

For each selected model, store paired bootstrap deltas for `recall`, `ndcg`, `revenue`, and `arp`. Preserve all exposure metrics in the absolute rows.

- [ ] **Step 5: Add persistence and direct-config protection tests**

```python
def test_runner_persists_authoritative_json_and_exposure_metrics(monkeypatch, tmp_path):
    _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    frame = single.run_experiment(single.configure_single_run("dunnhumby", out_dir=str(tmp_path)))
    payload = json.loads(next(tmp_path.glob("clv_single_*.json")).read_text())
    assert payload["screening_decision"] == frame.attrs["screening_decision"]
    assert payload["variant_definitions"]["single_zero_user"]["user_profile"] == "zero"
    assert {"n_distinct@10", "exposure_entropy@10", "eff_catalog@10",
            "top10_share@10", "top100_share@10"}.issubset(payload["absolute_rows"][0])


def test_run_experiment_revalidates_before_data_access(monkeypatch):
    monkeypatch.setattr(single, "_prepare_validation_context",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("data touched")))
    bad = dataclasses.replace(single.configure_single_run("dunnhumby"), seed_list=(42, 43))
    with pytest.raises(ValueError, match="seed 42"):
        single.run_experiment(bad)
```

- [ ] **Step 6: Run runner tests and commit Task 4**

Run:

```bash
pytest -q test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: run single-adapter CLV identification"
```

### Task 5: Add the guarded Colab entry point

**Files:**
- Create: `clv_single_adapter_colab.ipynb`
- Modify: `test_lightgcn_clv_single.py`

**Interfaces:**
- Consumes: `configure_single_run`, `preflight_summary`, `run_experiment`.
- Produces: a fresh-clone Colab workflow for H&M 2-year or Dunnhumby seed-42 validation screening.

- [ ] **Step 1: Write a failing notebook contract test**

```python
def test_colab_has_pinned_source_preflight_gate_and_final_decision():
    notebook = json.loads(Path("clv_single_adapter_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "configure_single_run" in source
    assert "preflight_summary" in source
    assert "reuse_full_result_json" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "screening_decision" in source
    assert "eval_test=False" in source and "eval_holdout=False" in source
```

- [ ] **Step 2: Run the notebook contract test and confirm RED**

Run: `pytest -q test_lightgcn_clv_single.py::test_colab_has_pinned_source_preflight_gate_and_final_decision`

Expected: FAIL because the notebook does not exist.

- [ ] **Step 3: Create the notebook with six explicit sections**

Create valid notebook JSON with:

1. GPU/runtime and Drive mount.
2. Fresh clone and checkout of the reviewed branch commit; print and assert the exact SHA after final review.
3. Dataset preset:

```python
DATASET = "dunnhumby"  # or "hm"
cfg = configure_single_run(
    DATASET,
    seed_list=(42,),
    eval_test=False,
    eval_holdout=False,
    out_dir=f"/content/drive/MyDrive/논문/data/results_clv_single_{DATASET}",
    m1_checkpoint_dir=f"/content/drive/MyDrive/논문/data/results_v3_{DATASET}",
)
reuse_full_result_json = (
    "/content/drive/MyDrive/논문/data/results_clv_moe_dunnhumby/"
    "clv_moe_dunnhumby_6f89c6b32f.json"
    if DATASET == "dunnhumby" else None
)
```

4. Full preflight JSON and explicit note that no training has started.
5. High-cost approval cell:

```python
ACKNOWLEDGE_HIGH_COST = False
assert ACKNOWLEDGE_HIGH_COST, "설정 검토 후 True로 바꾸세요."
result_df = run_experiment(cfg, reuse_full_result_json=reuse_full_result_json)
```

6. Results table, saved paths, `selected_lambda`, `lambda_selection_success`, and authoritative `screening_decision.success/reason/failed_controls/mechanism_comparison`.

- [ ] **Step 4: Validate notebook JSON and contract**

Run:

```bash
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
pytest -q test_lightgcn_clv_single.py -k colab
```

Expected: JSON valid and test PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add clv_single_adapter_colab.ipynb test_lightgcn_clv_single.py
git commit -m "feat: add guarded single-adapter Colab"
```

### Task 6: Final verification, research-status update, review, and pinning

**Files:**
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`
- Modify after review: `clv_single_adapter_colab.ipynb`
- Test: all repository tests.

**Interfaces:**
- Consumes: completed Tasks 1-5.
- Produces: reviewed source commit, pinned notebook commit, verified local implementation, and an updated research ledger that does not claim a high-cost result.

- [ ] **Step 1: Run focused and full verification**

```bash
pytest -q test_clv_moe_model.py test_lightgcn_clv_single.py
pytest -q
ruff check .
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
git diff --check
```

Expected: all tests PASS; ruff, JSON validation, and diff check succeed.

- [ ] **Step 2: Run preset preflight smoke without training**

```bash
python - <<'PY'
import json
from lightgcn_clv_single import configure_single_run, preflight_summary

for dataset in ("dunnhumby", "hm"):
    cfg = configure_single_run(
        dataset,
        encoder_epochs=1,
        frozen_epochs=5,
        max_epochs=6,
        patience=2,
    )
    summary = preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

Expected: both presets print; no dataset file is read and no training begins.

- [ ] **Step 3: Update the research ledger**

Add an implementation subsection to `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md` recording:

- exact implementation commit and test count;
- model IDs and zero/shuffle definitions;
- validation-only/test-protection behavior;
- Dunnhumby reuse is conditional on manifest, M1 state, encoder values, feature schema, checkpoint, and metric round-trip;
- no new high-cost H&M or Dunnhumby result has been produced by implementation verification.

Do not alter or delete the 2026-08-10 observed MoE/single-adapter result.

- [ ] **Step 4: Commit implementation documentation**

The repository plan/spec are already committed. If repository documentation changed during final verification, commit only those tracked files:

```bash
git add docs clv_single_adapter_colab.ipynb
git commit -m "docs: finalize single-adapter screening workflow"
```

Do not attempt to commit `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`; it is outside the repository.

- [ ] **Step 5: Request code review and address only verified findings**

Invoke `superpowers:requesting-code-review` against the design commit `ee606d5`. Review both standards and spec compliance. For each finding, reproduce it with a failing runtime test before changing implementation unless it is a documentation-only contradiction.

- [ ] **Step 6: Re-run the full verification after review fixes**

Repeat Step 1 exactly. Record the final passing test count and source commit.

- [ ] **Step 7: Pin the notebook to the reviewed source commit**

Print the reviewed source commit first:

```bash
git rev-parse HEAD
```

Use `apply_patch` to put the literal 40-character output into the notebook clone cell as `REVIEWED_SHA`, followed by
`git checkout {REVIEWED_SHA}` and `assert actual_sha[0] == REVIEWED_SHA`. Do not commit a symbolic branch name,
abbreviated SHA, command substitution, or temporary marker.

- [ ] **Step 8: Validate and commit the pin-only change**

```bash
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
pytest -q test_lightgcn_clv_single.py -k colab
git diff --check
git add clv_single_adapter_colab.ipynb
git commit -m "chore: pin single-adapter Colab to reviewed commit"
```

- [ ] **Step 9: Publish only after user authorization**

Do not push or update a pull request unless the user asks to publish. When authorized, push the current feature branch and verify the Colab GitHub URL resolves to the committed notebook.
