# M2 CLV-Conditioned Modulation LightGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dunnhumby seed-42 validation runner that trains M1 and one CLV-conditioned, 64-dimensional modulation LightGCN with no independent N/V score.

**Architecture:** Reuse the joint-N/V runner's train-only user and item feature preparation, but replace concatenated N/V subspaces with four bias-free rank-4 modulators. Their zero-initialized output projections scale the existing 64-dimensional ID embeddings by at most ±10% before one unchanged LightGCN propagation and one dot-product score.

**Tech Stack:** Python, PyTorch, pandas, NumPy, pytest, Google Colab notebook JSON.

## Global Constraints

- One model, one optimizer, one plain BPR objective.
- Binary graph and uniform negative sampling remain unchanged.
- No gamma, lambda, learned gate, external residual, post-score reranking, CLV-weighted loss, or edge weighting.
- First run is Dunnhumby seed 42 validation-only; test and holdout must fail closed.
- M1 and M2 use 64-dimensional final embeddings and identical ID initialization for the same seed.
- `tau=0.10`, modulation rank `4`, and current v1.5 N/V feature schema are fixed.

---

### Task 1: Modulation model

**Files:**
- Create: `clv_conditioned_modulation_model.py`
- Create: `test_clv_conditioned_modulation_model.py`

**Interfaces:**
- Consumes: `DualItemProfile`, user N/V arrays and masks, binary normalized adjacency.
- Produces: `CLVConditionedModulationLightGCN(...).layer0_embeddings()`, `.propagate()`, `.embeddings()`, `.batch_l2()`, `.modulation_diagnostics()`.

- [ ] **Step 1: Write failing tests**

```python
def test_zero_output_projection_starts_exactly_at_id_embeddings():
    model = _model(tau=0.1, rank=4)
    user, item = model.layer0_embeddings()
    torch.testing.assert_close(user, model.E_u.weight)
    torch.testing.assert_close(item, model.E_i.weight)

def test_nonzero_modulation_changes_embeddings_without_changing_width():
    model = _model(tau=0.1, rank=4)
    model.user_n.output.weight.data.fill_(0.2)
    user, item = model.layer0_embeddings()
    assert user.shape[1] == item.shape[1] == 64
    assert not torch.equal(user, model.E_u.weight)

def test_invalid_axis_is_exactly_masked():
    model = _model(activity_valid=False)
    model.user_n.output.weight.data.fill_(0.2)
    user, _ = model.layer0_embeddings()
    torch.testing.assert_close(user, model.E_u.weight)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_clv_conditioned_modulation_model.py`

Expected: collection fails because `clv_conditioned_modulation_model` does not exist.

- [ ] **Step 3: Implement the minimal model**

Implement `_LowRankModulator(input_dim, rank, output_dim)` with a nonzero first projection, zero output projection, no bias, and `tanh`. Compute axis-specific modulation, apply masks, sum N/V, and use
`E_id * (1 + tau * modulation)` before unchanged sparse LightGCN propagation.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q test_clv_conditioned_modulation_model.py`

Expected: all tests pass.

### Task 2: Validation runner and persistence

**Files:**
- Create: `lightgcn_clv_modulation.py`
- Create: `test_lightgcn_clv_modulation.py`

**Interfaces:**
- Consumes: `configure_modulation_dunnhumby_run(**overrides) -> ModulationConfig`, `lightgcn_clv_joint_nv._prepare`, common evaluator/trainer.
- Produces: `run_experiment(cfg) -> pandas.DataFrame`, result CSV/delta CSV/JSON, model checkpoint and preflight summary.

- [ ] **Step 1: Write failing runner tests**

```python
def test_dunnhumby_preset_is_validation_only():
    cfg = modulation.configure_modulation_dunnhumby_run()
    summary = modulation.preflight_summary(cfg)
    assert summary["dataset"] == "dunnhumby"
    assert summary["seed"] == 42
    assert summary["models"] == ["m1", "m2_clv_modulation"]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    assert summary["tau"] == 0.10
    assert summary["rank"] == 4

def test_public_runner_rejects_protected_splits_before_prepare(monkeypatch):
    cfg = replace(modulation.configure_modulation_dunnhumby_run(), eval_test=True)
    monkeypatch.setattr(modulation, "_prepare", lambda _: pytest.fail("must not prepare"))
    with pytest.raises(ValueError, match="test"):
        modulation.run_experiment(cfg)
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_lightgcn_clv_modulation.py`

Expected: collection fails because `lightgcn_clv_modulation` does not exist.

- [ ] **Step 3: Implement the runner**

Reuse existing train-only preparation and M1 training. Train only `m2_clv_modulation`, save absolute metrics and paired deltas, and compute the six-metric 99% accuracy/economic-improvement decision. Save modulation schema, magnitude statistics, masks, source/input/config hashes, checkpoint paths and the interpretation that `revenue` is a price/purchase-amount weighted hit.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q test_lightgcn_clv_modulation.py test_clv_conditioned_modulation_model.py`

Expected: all tests pass.

### Task 3: Colab runner and handoff

**Files:**
- Create: `clv_conditioned_modulation_dunnhumby_colab.ipynb`
- Modify: `test_lightgcn_clv_modulation.py`

**Interfaces:**
- Consumes: pinned reviewed source SHA and `configure_modulation_dunnhumby_run`.
- Produces: a run-all Colab that mounts Drive, clones the pinned commit, prints preflight, runs once, and displays the absolute and M1-delta tables.

- [ ] **Step 1: Add failing notebook contract test**

```python
def test_colab_is_pinned_and_runs_modulation_once():
    payload = json.loads(Path("clv_conditioned_modulation_dunnhumby_colab.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert source.count("result_df = run_experiment(cfg)") == 1
    assert "eval_test=False" in source and "eval_holdout=False" in source
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_lightgcn_clv_modulation.py::test_colab_is_pinned_and_runs_modulation_once`

Expected: fail because the notebook is missing.

- [ ] **Step 3: Create notebook and pin after implementation commit**

Create a minimal notebook with install/clone, imports and configuration, GPU/preflight, one execution cell, and result display. Commit source files first, replace the notebook SHA with that source commit, and commit the pinned notebook separately.

- [ ] **Step 4: Verify focused scope**

Run:

```bash
pytest -q test_clv_conditioned_modulation_model.py test_lightgcn_clv_modulation.py
ruff check clv_conditioned_modulation_model.py lightgcn_clv_modulation.py test_clv_conditioned_modulation_model.py test_lightgcn_clv_modulation.py
python -m py_compile clv_conditioned_modulation_model.py lightgcn_clv_modulation.py
python -m json.tool clv_conditioned_modulation_dunnhumby_colab.ipynb >/dev/null
git diff --check
```

Expected: focused tests and static checks pass; no high-cost training runs locally.
