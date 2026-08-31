# Gradient-Isolated CLV Economic Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one seed-42 Dunnhumby M2 screen that protects the ordinary LightGCN gradient path while jointly learning a bounded three-dimensional CLV-conditioned relation and one-dimensional price relation.

**Architecture:** Reuse the existing two-layer ID-only LightGCN and historical screen runner. The auxiliary projections consume detached final ID representations, while the single BPR score concatenates the ordinary ID block, a bounded three-dimensional CLV-conditioned relation, and a bounded one-dimensional price relation.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest, Google Colab notebook JSON.

**Spec:** `docs/superpowers/specs/2026-08-31-m2-gradient-isolated-clv-economic-interaction-design.md`

## Global Constraints

- New-item evaluation excludes training `(user,item)` pairs.
- `MIN_ITEM_INTER=1`, binary graph, uniform negatives, no sample weights.
- One plain BPR plus the existing sampled ID L2; no new loss term.
- One optimizer, one training loop, no pretraining or freezing.
- ID dimension 64, relation dimension 3, price dimension 1.
- `rho=0.05`, `beta=0.25`, `delta=0.25`, `eta=0.5`, `epsilon=0.5`.
- Dunnhumby day 1–683 training and day 684–690 historical development evaluation.
- Seed 42, 100 fixed epochs, no epoch selection, no final test or holdout.

---

### Task 1: Gradient-isolated model

**Files:**
- Create: `clv_gradient_isolated_economic_interaction_model.py`
- Test: `test_clv_gradient_isolated_economic_interaction_model.py`

**Interfaces:**
- Consumes: train-only arrays `q_n`, `q_v`, `q_c`, user validity, item global price percentile and item validity, and the existing normalized binary adjacency tensor.
- Produces: `GradientIsolatedCLVEconomicInteractionLightGCN`, with `id_embeddings()`, `auxiliary_embeddings()`, `embeddings()`, `candidate_score_components()`, `bpr_loss()`, and diagnostics compatible with the common trainer/evaluator.

- [ ] **Step 1: Write failing construction and shape tests**

```python
model = _model(rho=0.05)
user, item, *_ = model.embeddings()
assert user.shape == (3, 10)  # ID6 + relation3 + price1
assert item.shape == (4, 10)
assert 1 / 3 <= model.price_calibration().item() <= 1
```

- [ ] **Step 2: Write failing gradient-isolation test**

```python
id_user, id_item = model.id_embeddings()
relation_u, relation_i, price_u, price_i = model.auxiliary_embeddings(id_user, id_item)
(relation_u.sum() + relation_i.sum()).backward()
assert model.E_u.weight.grad is None
assert model.E_i.weight.grad is None
assert model.user_projection.weight.grad.abs().sum() > 0
assert model.item_projection.weight.grad.abs().sum() > 0
```

- [ ] **Step 3: Write failing rho-zero and bounded-price tests**

```python
rho0 = _model(rho=0.0)
full_u, full_i, *_ = rho0.embeddings()
id_u, id_i = rho0.id_embeddings()
torch.testing.assert_close(full_u, id_u, atol=0, rtol=0)
torch.testing.assert_close(full_i, id_i, atol=0, rtol=0)
assert rho0.representation_diagnostics()["rho_zero_auxiliary_max_abs"] == 0
```

- [ ] **Step 4: Run tests and verify they fail before implementation**

Run: `pytest -q test_clv_gradient_isolated_economic_interaction_model.py`

Expected: import failure because the model module does not exist.

- [ ] **Step 5: Implement the model**

Implement:

```python
base_u = F.normalize(user_projection(id_user.detach()), dim=1)
base_i = F.normalize(item_projection(id_item.detach()), dim=1)
modulation = 1 + delta * torch.tanh(condition_mixer(torch.stack([q_n, q_v, q_c], 1)))
relation_u = valid[:, None] * F.normalize(base_u * modulation, dim=1)
relation_i = base_i
g_c = valid * (1 + eta * (q_c - 0.5))
kappa = (1 + epsilon * torch.tanh(raw_price_calibration)) / (1 + epsilon)
price_u = valid * g_c * kappa * (2 * q_v - 1)
price_i = item_valid * (2 * item_price_percentile - 1)
```

Concatenate the blocks with `sqrt(rho*(1-beta))` and `sqrt(rho*beta)`. For `rho=0`, return the ordinary-width ID tensors to preserve exact arithmetic parity. Reject M3/M4 inputs in `bpr_loss`.

- [ ] **Step 6: Run model tests**

Run: `pytest -q test_clv_gradient_isolated_economic_interaction_model.py`

Expected: all tests pass.

### Task 2: Historical screen runner and diagnostics

**Files:**
- Create: `lightgcn_clv_gradient_isolated_economic_interaction.py`
- Test: `test_lightgcn_clv_gradient_isolated_economic_interaction.py`

**Interfaces:**
- Consumes: the Task 1 model and existing helpers from `lightgcn_clv_conditioned_user_item_interaction.py`, `lightgcn_clv_gatefree_lowdim.py`, and `lightgcn_clv_v3.py`.
- Produces: `configure_gradient_isolated_run()`, `preflight_summary()`, `build_item_price_inputs()`, `screening_reading()`, and `run_gradient_isolated_screen()`.

- [ ] **Step 1: Write failing protocol tests**

```python
summary = preflight_summary(configure_gradient_isolated_run(out_dir="/tmp/x", baseline_result_dir="/tmp/y"))
assert summary["fixed"]["graph"] == "binary"
assert summary["m2"]["rho"] == 0.05
assert summary["m2"]["beta"] == 0.25
assert summary["historical_development_split"]["final_test_constructed"] is False
```

- [ ] **Step 2: Write failing global item-price test**

```python
price_pct, valid = build_item_price_inputs(train, n_items=4)
assert price_pct[lowest_item] < price_pct[highest_item]
assert price_pct[missing_item] == 0.5
assert not valid[missing_item]
```

- [ ] **Step 3: Write failing screen decision test**

Verify the screen requires the six 99% accuracy ratios, positive weighted-hit@10, positive high-CLV Recall/NDCG@10, high-CLV Top-10 changes, exact rho-zero parity, and jointly-trained ID-only ratios of at least 99.5%.

- [ ] **Step 4: Run runner tests and verify failure**

Run: `pytest -q test_lightgcn_clv_gradient_isolated_economic_interaction.py`

Expected: import failure because the runner does not exist.

- [ ] **Step 5: Implement preparation and matched arms**

Reuse the existing historical split, train-only CLV axes, saved M1 display row, item metadata, evaluator cache, fixed-epoch trainer, checkpoint resume, and atomic output helpers. Build global item price percentiles from train-only mean `up`. Train matched `rho=0` and active `rho=0.05` arms with identical seed and initialization order.

- [ ] **Step 6: Implement ablations and diagnostics**

Evaluate the jointly-trained ID-only view and no-price/no-relation views without retraining. Save relation, price, combined score standard deviations relative to ID, Top-10 overlap by CLV segment, learned price calibration, and component gradient norms. Do not store pandas DataFrames in `frame.attrs`; store only plain dictionaries and file paths to avoid notebook formatter errors.

- [ ] **Step 7: Run runner tests**

Run: `pytest -q test_lightgcn_clv_gradient_isolated_economic_interaction.py`

Expected: all tests pass.

### Task 3: Colab entrypoint and project status

**Files:**
- Create: `clv_m2_gradient_isolated_economic_interaction_dunnhumby_colab.ipynb`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: `configure_gradient_isolated_run`, `preflight_summary`, and `run_gradient_isolated_screen` from Task 2.
- Produces: a GPU-checked Colab notebook pinned to the final source commit and a dated research-status entry distinguishing design, implementation, and pending result.

- [ ] **Step 1: Create the notebook from the existing conditioned-interaction notebook pattern**

Use cells for repository checkout at a fixed commit, Drive mount/import, preflight display, run, and plain table/file-path display. Do not call final test or holdout evaluators.

- [ ] **Step 2: Update research status**

Record the A2 question, model formula, fixed hyperparameters, stop-gradient limitation, protected split, pre-registered screen rule, implementation files, and that no performance result exists yet.

- [ ] **Step 3: Run focused verification**

Run:

```bash
pytest -q \
  test_clv_gradient_isolated_economic_interaction_model.py \
  test_lightgcn_clv_gradient_isolated_economic_interaction.py \
  test_clv_conditioned_user_item_interaction_model.py \
  test_lightgcn_clv_conditioned_user_item_interaction.py
python -m py_compile \
  clv_gradient_isolated_economic_interaction_model.py \
  lightgcn_clv_gradient_isolated_economic_interaction.py
python -m json.tool clv_m2_gradient_isolated_economic_interaction_dunnhumby_colab.ipynb
git diff --check
```

Expected: all tests and checks pass.

- [ ] **Step 4: Commit source and status**

Commit the model, runner, tests, status, and an unpinned notebook first. Then update the notebook checkout SHA to that source commit and create a final notebook-pin commit.

- [ ] **Step 5: Push branch**

Push `feat/m2-joint-nv-lightgcn` to its configured origin and report both commits and the Colab path.
