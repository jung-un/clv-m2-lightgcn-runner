# CLV Dual-Axis Multiseed Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the frozen Dunnhumby and H&M-60 dual-axis operating points at seeds 43 and 44, combine them with the existing seed-42 validation result, and stop after a three-seed reproducibility decision.

**Architecture:** Generalize the existing dual-axis seed preparation and single-variant evaluator with optional seed and evaluation-point arguments while preserving the seed-42 public behavior. Add a separate multiseed runner that validates the frozen protocol, re-evaluates the saved seed-42 checkpoint without training, trains only M1 and `dual_clv_fixed` for seeds 43/44, and persists three-seed results. A dedicated Colab runs Dunnhumby then H&M 60 days and contains no H&M two-year, test, holdout, control, or lambda-sweep path.

**Tech Stack:** Python 3, PyTorch, pandas, NumPy, pytest, Google Colab, existing LightGCN/CLV modules.

## Global Constraints

- Dunnhumby uses N+V, gate `equal`, lambda `2.0`.
- H&M 60 days uses N+V, gate `high`, lambda `1.0`.
- New training seeds are exactly `(43, 44)`; seed 42 is loaded from the existing result.
- Only M1 and `dual_clv_fixed` are trained for seeds 43/44.
- Validation only; test and holdout answers are never constructed.
- H&M two-year, controls, M3, M4, and M5 are not invoked.
- Gate, lambda, structure, and training budget are not selected per seed.
- Success never triggers another run automatically.

---

### Task 1: Parameterize Existing Dual Seed Internals

**Files:**
- Modify: `lightgcn_clv_dual.py`
- Test: `test_lightgcn_clv_dual.py`

**Interfaces:**
- Produces: `_prepare(cfg, seed: int = 42) -> dict`
- Produces: `_fresh_base(prepared: dict, seed: int = 42)`
- Produces: `_train_variant(..., seed: int = 42, gate_shapes=GATE_SHAPES, lambda_eval=None) -> dict`
- Preserves: `run_experiment()` seed-42 screening behavior and filenames.

- [ ] **Step 1: Write failing tests for non-42 seed propagation and one-point evaluation**

Add runtime tests that monkeypatch encoder/M1 training and assert seed `43` reaches encoder training,
M1 load/train, adapter training, checkpoint name, result rows, and per-user keys. Assert
`gate_shapes=("equal",)` and `lambda_eval=(2.0,)` produce exactly one row and do not evaluate other
gates/lambdas.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest -q test_lightgcn_clv_dual.py -k 'parameterized_seed or fixed_point'`

Expected: failure because current helpers hardcode seed 42 and full curves.

- [ ] **Step 3: Implement optional seed and evaluation-point parameters**

Replace each internal literal `42` that identifies training/evaluation state with the `seed` argument.
Default every new argument to the current seed-42 behavior. Use:

```python
shapes = tuple(gate_shapes)
lambdas = tuple(cfg.lambda_eval if lambda_eval is None else lambda_eval)
```

Keep `run_experiment()` calls explicit with `seed=42` so its research protocol remains obvious.

- [ ] **Step 4: Run focused and existing dual tests**

Run: `pytest -q test_lightgcn_clv_dual.py test_clv_dual_axis_model.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lightgcn_clv_dual.py test_lightgcn_clv_dual.py
git commit -m "refactor: parameterize dual-axis seed evaluation"
```

### Task 2: Build Frozen Multiseed Protocol and Decision Logic

**Files:**
- Create: `lightgcn_clv_dual_multiseed.py`
- Create: `test_lightgcn_clv_dual_multiseed.py`

**Interfaces:**
- Produces: `MultiSeedValidationConfig`
- Produces: `configure_multiseed_validation(dataset, seed42_result_json, *, short_hm=False, out_dir=None) -> MultiSeedValidationConfig`
- Produces: `validate_multiseed_config(cfg) -> MultiSeedValidationConfig`
- Produces: `reproducibility_decision(absolute_rows, seed_delta_rows) -> dict`
- Produces: `run_multiseed_validation(cfg) -> pandas.DataFrame`

- [ ] **Step 1: Write failing fail-closed configuration tests**

Cover rejection before data access for:

```python
MultiSeedValidationConfig(seeds=(42, 43))
MultiSeedValidationConfig(eval_test=True)
MultiSeedValidationConfig(eval_holdout=True)
MultiSeedValidationConfig(dataset="hm", window_days=None)
MultiSeedValidationConfig(dataset="dunnhumby", gate_shape="high")
MultiSeedValidationConfig(dataset="hm", gate_shape="high", fixed_lambda=2.0)
```

Also assert configured model IDs equal `("m1", "dual_clv_fixed")` and no control identifier is present.

- [ ] **Step 2: Write failing decision tests**

Construct three-seed fixture rows and assert success requires all three fixed conditions:

```python
mean_revenue_delta > 0
positive_revenue_seed_count >= 2
all six mean_accuracy_ratios >= 0.99
```

Test each condition independently fails and returns explicit `failed_conditions`.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest -q test_lightgcn_clv_dual_multiseed.py`

Expected: import failure because the module does not exist.

- [ ] **Step 4: Implement immutable presets and pure decision function**

Use a frozen dataclass with dataset-specific values derived only by
`configure_multiseed_validation`. Validate the existing seed-42 JSON has seed `(42,)`, validation-only,
the expected selected gate/lambda, matching code version, and required checkpoint hashes before any new
training begins.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `pytest -q test_lightgcn_clv_dual_multiseed.py -k 'config or decision'`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add lightgcn_clv_dual_multiseed.py test_lightgcn_clv_dual_multiseed.py
git commit -m "feat: add frozen multiseed validation protocol"
```

### Task 3: Implement Seed-42 Reuse, Seed 43/44 Training, and Persistence

**Files:**
- Modify: `lightgcn_clv_dual_multiseed.py`
- Modify: `test_lightgcn_clv_dual_multiseed.py`

**Interfaces:**
- Consumes: `dual._prepare(cfg, seed)`, `dual._fresh_base(prepared, seed)`,
  `dual._train_variant(..., seed, gate_shapes, lambda_eval)`
- Consumes: checkpoint diagnostic loader for seed-42 validation per-user arrays.
- Produces: DataFrame attrs `reproducibility_decision` and `result_paths`.

- [ ] **Step 1: Write failing orchestration tests**

Monkeypatch all expensive functions and assert:

- seed 42 is loaded/evaluated but never passed to a training function;
- seeds 43 and 44 each train exactly M1 and `dual_clv_fixed` once;
- gate/lambda are the frozen dataset preset;
- controls, full lambda grid, test, holdout, and H&M two-year are never invoked;
- a failed decision returns normally and does not call another runner.

- [ ] **Step 2: Write failing persistence tests**

Assert the runner writes:

- `<stem>.csv` with six absolute rows (M1/model × three seeds);
- `<stem>_delta.csv` with seed-level and three-seed paired bootstrap rows;
- `<stem>_decision.csv` with each fixed condition;
- `<stem>.json` with raw-input manifest, source revision, original seed-42 result identity,
  M1/encoder/adapter checkpoint hashes, training statistics, and interpretation limits.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `pytest -q test_lightgcn_clv_dual_multiseed.py -k 'orchestration or persist'`

Expected: failures because orchestration and persistence are absent.

- [ ] **Step 4: Implement seed-42 checkpoint re-evaluation**

Reuse the validation-only checkpoint diagnostic loading path to recover M1/model per-user arrays at the
frozen point. Verify checkpoint and input manifests. Do not call encoder or adapter training for seed 42.

- [ ] **Step 5: Implement seed 43/44 fixed-point runs**

For each seed call the parameterized dual helpers, evaluate only one gate/lambda, retain baseline/model
per-user arrays, and collect all required absolute metrics. Assert evaluation-user ordering and counts are
identical across seeds before calling `v3.paired_bootstrap`.

- [ ] **Step 6: Implement aggregation and persistence**

Calculate seed-level deltas, relative changes, three-seed user-paired bootstrap, per-seed means/standard
deviation, six accuracy mean ratios, and the frozen decision. Save only validation artifacts and return
the absolute DataFrame with paths/decision in attrs.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `pytest -q test_lightgcn_clv_dual_multiseed.py`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add lightgcn_clv_dual_multiseed.py test_lightgcn_clv_dual_multiseed.py
git commit -m "feat: run dual-axis multiseed validation"
```

### Task 4: Create Two-Dataset Validation-Only Colab

**Files:**
- Create: `clv_dual_multiseed_validation_colab.ipynb`
- Modify: `test_lightgcn_clv_dual_multiseed.py`

**Interfaces:**
- Consumes: `configure_multiseed_validation`, `run_multiseed_validation`
- Produces: Drive outputs for Dunnhumby and H&M 60 days.

- [ ] **Step 1: Write failing notebook contract test**

Parse notebook JSON and assert it:

- checks out one full reviewed SHA;
- references `results_clv_dual_dunnhumby` and `results_clv_dual_hm_w60`;
- invokes `run_multiseed_validation` for both datasets;
- prints seed-level metrics, three-seed decision, failed conditions, and result files;
- does not contain `window_days=None` for H&M, `eval_test=True`, `eval_holdout=True`,
  `dual_shuffled_user`, `dual_adapter_only`, or any H&M two-year run.

- [ ] **Step 2: Run notebook test and verify RED**

Run: `pytest -q test_lightgcn_clv_dual_multiseed.py -k colab`

Expected: fail because notebook is missing.

- [ ] **Step 3: Create the notebook**

Use four executable stages: Drive/reviewed checkout, preflight/source JSON discovery, sequential runs,
and result display. Clearly state that this starts four adapter trainings plus four same-seed M1/encoder
paths as needed, but never starts H&M two-year or protected splits.

- [ ] **Step 4: Validate notebook and tests**

Run:

```bash
python -m json.tool clv_dual_multiseed_validation_colab.ipynb >/dev/null
pytest -q test_lightgcn_clv_dual_multiseed.py
```

Expected: both pass.

- [ ] **Step 5: Commit implementation notebook, then pin reviewed SHA**

```bash
git add clv_dual_multiseed_validation_colab.ipynb test_lightgcn_clv_dual_multiseed.py
git commit -m "feat: add multiseed validation Colab"
```

Replace the notebook placeholder with that commit's full SHA, rerun JSON/tests, and commit:

```bash
git add clv_dual_multiseed_validation_colab.ipynb
git commit -m "chore: pin multiseed validation Colab"
```

### Task 5: Full Verification, Research Status, and Publication

**Files:**
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: all completed code and validation evidence.
- Produces: remote Colab executable from the reviewed branch.

- [ ] **Step 1: Run complete verification**

Run:

```bash
pytest -q
ruff check lightgcn_clv_dual.py lightgcn_clv_dual_multiseed.py \
  test_lightgcn_clv_dual.py test_lightgcn_clv_dual_multiseed.py
python -m json.tool clv_dual_multiseed_validation_colab.ipynb >/dev/null
git diff --check
```

Expected: all tests/checks pass; only the existing sparse-tensor warning may remain.

- [ ] **Step 2: Update research status**

Record the frozen protocol, code commits, checks, Colab path, and explicit fact that seed 43/44 training,
H&M two-year, test, holdout, and controls have not yet been executed.

- [ ] **Step 3: Push and verify remote checkout**

Push `feat/clv-conditioned-moe`, verify the remote ref, fresh-clone it, checkout the notebook's reviewed
SHA, and assert the runner exists. Do not create or merge a PR unless separately requested.

- [ ] **Step 4: Hand off**

Provide the Colab URL, expected scope/cost, exact outputs to share, and repeat that completion stops at
the three-seed validation decision.
