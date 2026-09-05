# M5 CLV Economic Positive Weighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible Dunnhumby seed-42 M5 screen that jointly trains a CLV-related economic representation and CLV-conditioned positive purchase-amount weighting, with a complete 2×2 factorial and attribution controls.

**Architecture:** A 4-dimensional bounded economic block is concatenated to 64-dimensional ID embeddings at LightGCN layer 0 and jointly propagated for two layers. A separate loss helper applies a fixed positive-row weight to the mean of five per-negative BPR losses. One runner prepares train-only economic inputs, trains six independent arms, evaluates the existing new-item protocol, and writes comparisons, interactions, diagnostics, and checkpoints.

**Tech Stack:** Python 3, NumPy, pandas, PyTorch sparse tensors, pytest, Jupyter/Colab JSON.

**Spec:** `docs/superpowers/specs/2026-09-05-m5-clv-economic-representation-positive-weighting-design.md`

## Global Constraints

- New-item recommendation only; exclude every training `(user,item)` pair from evaluation truth.
- `MIN_ITEM_INTER=1`, binary graph, two propagation layers, uniform negative sampling.
- One training loop and one optimizer per arm; ID and economic projection parameters train together.
- M2-only arm has no sample weighting; M4'-only arm has no economic representation.
- No external reranking, new auxiliary loss, final test, or holdout construction.
- Freeze seed 42, 100 epochs, ID dimension 64, economic dimension 4, `rho=0.15`, `lambda=0.5`, four bins, `kappa=10`, and K=5 before execution.
- Primary metric is code key `vndcg@10`, reported as price/purchase-amount-weighted NDCG@10.
- Do not claim statistical significance from the single-seed development screen.

---

### Task 1: Train-only economic inputs and assignment controls

**Files:**
- Create: `lightgcn_clv_m5_economic_positive_weight.py`
- Test: `test_lightgcn_clv_m5_economic_positive_weight.py`

**Interfaces:**
- Produces: `build_economic_inputs(train, n_users, n_items, n_bins, shrinkage_strength, degree_bins) -> dict[str, np.ndarray | dict]`
- Produces: `joint_degree_matched_shuffle(prepared, seed, degree_bins) -> dict[str, np.ndarray]`
- Consumes: prepared training columns `u_idx`, `i_idx`, `cat_idx`, `v`, plus existing `q_v`, `q_c`, and `clv_valid` arrays.

- [ ] **Step 1: Write failing input tests**

```python
def test_economic_inputs_use_equal_item_count_bins_and_preserve_shrinkage():
    built = runner.build_economic_inputs(frame, n_users=3, n_items=8,
                                         n_bins=4, shrinkage_strength=10.0,
                                         degree_bins=2)
    assert np.bincount(built["item_bin"], minlength=4).tolist() == [2, 2, 2, 2]
    assert np.all(np.linalg.norm(built["user_economic_input"][:, :4], axis=1) < 1)
    assert built["item_economic_input"].shape == (8, 2)

def test_joint_shuffle_moves_the_whole_user_tuple_within_degree_bin():
    shuffled = runner.joint_degree_matched_shuffle(prepared, seed=42, degree_bins=2)
    for target, source in enumerate(shuffled["source_user"]):
        assert prepared["degree_bin"][target] == prepared["degree_bin"][source]
        np.testing.assert_array_equal(shuffled["user_economic_input"][target],
                                      prepared["user_economic_input"][source])
        assert shuffled["q_c"][target] == prepared["q_c"][source]
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `pytest -q test_lightgcn_clv_m5_economic_positive_weight.py -k 'economic_inputs or joint_shuffle'`

Expected: import failure because `lightgcn_clv_m5_economic_positive_weight` does not exist.

- [ ] **Step 3: Implement deterministic train-only inputs**

```python
def build_economic_inputs(train, *, n_users, n_items, q_v, q_c, clv_valid,
                          n_bins=4, shrinkage_strength=10.0, degree_bins=10):
    item_amount = train.groupby("i_idx")["v"].median().reindex(range(n_items))
    amount_rank = np.zeros(n_items, dtype=np.float32)
    amount_rank[item_amount.notna()] = item_amount[item_amount.notna()].map(
        np.log1p
    ).rank(pct=True, method="average").to_numpy(np.float32)
    # Assign stable equal-item-count bins, calculate category ranks, spend shares,
    # empirical-Bayes shrinkage, degree bins, and within-bin V ranks.
    return {
        "user_economic_input": user_input.astype(np.float32),
        "item_economic_input": item_input.astype(np.float32),
        "item_amount_percentile": amount_rank,
        "degree_bin": degree_bin,
        "economic_input_diagnostics": diagnostics,
    }
```

Invalid items receive centered coordinates zero. Invalid user rows receive all-zero economic inputs. The first four user columns are the shrunken centered spend distribution and the fifth is the centered within-degree-decile V rank.

- [ ] **Step 4: Run the input tests and verify GREEN**

Run: `pytest -q test_lightgcn_clv_m5_economic_positive_weight.py -k 'economic_inputs or joint_shuffle'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the input layer**

```bash
git add lightgcn_clv_m5_economic_positive_weight.py test_lightgcn_clv_m5_economic_positive_weight.py
git commit -m "Add M5 train-only economic inputs"
```

### Task 2: Bounded economic LightGCN model

**Files:**
- Create: `clv_m5_economic_positive_weight_model.py`
- Test: `test_clv_m5_economic_positive_weight_model.py`

**Interfaces:**
- Produces: `M5EconomicLightGCN(...).propagated_embeddings() -> tuple[Tensor, Tensor]`
- Produces: `candidate_score_components(users, items) -> {"id", "economic", "full"}`
- Produces: `training_gradient_diagnostics()` and `representation_diagnostics()`.
- Consumes: fixed user input `[n_users,5]`, item input `[n_items,2]`, sparse normalized adjacency, `rho`, dimensions, layers, and regularization.

- [ ] **Step 1: Write failing model tests**

```python
def test_bounded_projection_preserves_zero_and_caps_norm():
    model = make_model(rho=0.15, layers=0)
    user_e, item_e = model.economic_coordinates()
    assert torch.equal(user_e[2], torch.zeros(4))
    assert user_e.norm(dim=1).max() <= 1.0
    assert item_e.norm(dim=1).max() <= 1.0

def test_rho_zero_is_exact_id_lightgcn():
    model = make_model(rho=0.0, layers=2)
    full_u, full_i = model.propagated_embeddings()
    id_u, id_i = model.id_embeddings()
    torch.testing.assert_close(full_u[:, :6], id_u, atol=0, rtol=0)
    torch.testing.assert_close(full_i[:, :6], id_i, atol=0, rtol=0)
    assert full_u[:, 6:].abs().max() == 0

def test_one_ranking_loss_updates_id_and_both_economic_projections():
    loss = model.mean_multi_negative_bpr(users, positives, negatives)[0]
    loss.backward()
    assert model.E_u.weight.grad.abs().sum() > 0
    assert model.E_i.weight.grad.abs().sum() > 0
    assert model.user_economic_projection.weight.grad.abs().sum() > 0
    assert model.item_economic_projection.weight.grad.abs().sum() > 0
```

- [ ] **Step 2: Run the model tests and verify RED**

Run: `pytest -q test_clv_m5_economic_positive_weight_model.py`

Expected: import failure because the model file does not exist.

- [ ] **Step 3: Implement the minimal jointly propagated model**

```python
def economic_coordinates(self):
    user = 0.5 * torch.tanh(self.user_economic_projection(self.user_economic_input))
    item = 0.5 * torch.tanh(self.item_economic_projection(self.item_economic_input))
    user = user * self.user_economic_valid[:, None]
    item = item * self.item_economic_valid[:, None]
    return user, item

def layer0_embeddings(self):
    user_e, item_e = self.economic_coordinates()
    scale = math.sqrt(self.rho)
    return torch.cat([self.E_u.weight, scale * user_e], 1), \
           torch.cat([self.E_i.weight, scale * item_e], 1)
```

Implement two-layer sparse propagation, one final dot-product score, ID-only decomposition, sampled-ID L2, and diagnostics. Initialize ID embeddings before projections so `rho=0` preserves the M1 random stream and exact parity.

- [ ] **Step 4: Run the model tests and verify GREEN**

Run: `pytest -q test_clv_m5_economic_positive_weight_model.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the model**

```bash
git add clv_m5_economic_positive_weight_model.py test_clv_m5_economic_positive_weight_model.py
git commit -m "Add bounded M5 economic LightGCN"
```

### Task 3: Positive-row weighted per-negative BPR

**Files:**
- Modify: `clv_m5_economic_positive_weight_model.py`
- Modify: `test_clv_m5_economic_positive_weight_model.py`

**Interfaces:**
- Produces: `positive_row_weights(q_c, item_amount_percentile, train_mean_raw_weight, lambda_) -> Tensor`
- Produces: `weighted_multi_negative_bpr(positive_scores, negative_scores, row_weights) -> tuple[Tensor, dict]`.

- [ ] **Step 1: Write failing loss tests**

```python
def test_positive_weights_have_fixed_formula_and_mean_mass():
    q = torch.tensor([0.0, 1.0, 1.0])
    amount = torch.tensor([0.0, 0.5, 1.0])
    raw = 1 + 0.5 * q * (2 * amount - 1)
    weights = positive_row_weights(q, amount, raw.mean(), lambda_=0.5)
    torch.testing.assert_close(weights, raw / raw.mean())

def test_weighted_bpr_averages_losses_before_row_weighting():
    pos = torch.tensor([1.0, 0.5])
    neg = torch.tensor([[0.0, 2.0], [0.0, 1.0]])
    weights = torch.tensor([1.5, 0.5])
    expected = (weights * F.softplus(neg - pos[:, None]).mean(1)).mean()
    actual, diagnostics = weighted_multi_negative_bpr(pos, neg, weights)
    torch.testing.assert_close(actual, expected)
    assert diagnostics["negative_count"] == 2
```

- [ ] **Step 2: Run the loss tests and verify RED**

Run: `pytest -q test_clv_m5_economic_positive_weight_model.py -k 'positive_weights or weighted_bpr'`

Expected: failure because the loss helpers do not exist.

- [ ] **Step 3: Implement the fixed formula and diagnostics**

```python
def weighted_multi_negative_bpr(positive_scores, negative_scores, row_weights):
    losses = F.softplus(negative_scores - positive_scores[:, None])
    per_row = losses.mean(dim=1)
    loss = (row_weights * per_row).mean()
    return loss, {
        "negative_count": negative_scores.shape[1],
        "row_weight_mean": row_weights.mean().detach(),
        "row_weight_cv": (row_weights.std(unbiased=False) /
                          row_weights.mean().clamp_min(1e-12)).detach(),
        "p_correct": (positive_scores[:, None] > negative_scores).float().mean().detach(),
    }
```

Validate finite values, shapes, percentile range, positive weights, fixed `lambda=0.5`, and fixed global normalization mass.

- [ ] **Step 4: Run the loss tests and full model tests**

Run: `pytest -q test_clv_m5_economic_positive_weight_model.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the loss**

```bash
git add clv_m5_economic_positive_weight_model.py test_clv_m5_economic_positive_weight_model.py
git commit -m "Add CLV positive purchase weighting"
```

### Task 4: Six-arm runner, evaluation, and saved diagnostics

**Files:**
- Modify: `lightgcn_clv_m5_economic_positive_weight.py`
- Modify: `test_lightgcn_clv_m5_economic_positive_weight.py`

**Interfaces:**
- Produces: `M5EconomicPositiveConfig`, `configure_m5_economic_positive_run`, `preflight_summary`, `arm_specifications`, `interaction_rows`, `screening_reading`, and `run_m5_economic_positive_screen`.
- Consumes: existing `lightgcn_clv_v3`, evaluator, progress store, uniform K-negative sampler, and output helpers.

- [ ] **Step 1: Write failing runner tests**

```python
def test_preflight_freezes_six_arms_and_protected_splits():
    summary = runner.preflight_summary(config(tmp_path))
    assert summary["reported_models"] == list(runner.MODEL_IDS)
    assert summary["fixed"]["final_test_constructed"] is False
    assert summary["fixed"]["holdout_constructed"] is False
    assert summary["m2"]["rho"] == 0.15
    assert summary["m4_prime"]["lambda"] == 0.5
    assert summary["reading_rule"]["primary_metric"] == "vndcg@10"

def test_interaction_is_factorial_difference_in_differences():
    rows = runner.interaction_rows(metric_rows).set_index("metric")
    assert rows.at["vndcg@10", "interaction_effect"] == pytest.approx(
        (metric_rows[runner.M5]["vndcg@10"] - metric_rows[runner.M4P]["vndcg@10"])
        - (metric_rows[runner.M2]["vndcg@10"] - metric_rows[runner.M1]["vndcg@10"])
    )

def test_screen_requires_primary_attribution_interaction_accuracy_and_exposure():
    reading = runner.screening_reading(passing_rows)
    assert reading["positive_screen"] is True
    failing = copy.deepcopy(passing_rows)
    failing[runner.M5_SHUFFLE]["vndcg@10"] = failing[runner.M5]["vndcg@10"]
    assert runner.screening_reading(failing)["positive_screen"] is False
```

- [ ] **Step 2: Run the runner tests and verify RED**

Run: `pytest -q test_lightgcn_clv_m5_economic_positive_weight.py`

Expected: failures because configuration, arms, and screening functions are incomplete.

- [ ] **Step 3: Implement config, preparation, arm loop, and reporting**

```python
MODEL_IDS = (M1, M2, M4P, M5, M5_SHUFFLE, M5_DEGREE_GATE)

def arm_specifications(prepared, cfg):
    return [
        arm(M1, rho=0.0, weighted=False, assignment="observed"),
        arm(M2, rho=cfg.rho, weighted=False, assignment="observed"),
        arm(M4P, rho=0.0, weighted=True, assignment="observed"),
        arm(M5, rho=cfg.rho, weighted=True, assignment="observed"),
        arm(M5_SHUFFLE, rho=cfg.rho, weighted=True, assignment="joint_shuffle"),
        arm(M5_DEGREE_GATE, rho=cfg.rho, weighted=True, assignment="degree_gate"),
    ]
```

Train each arm from the same seed initialization in an independent optimizer. Save compact checkpoints and resumable epoch state. Evaluate with the existing new-item evaluator. Save absolute CSV, pairwise comparison CSV, interaction CSV, score/loss diagnostics CSV, Top-10 overlap CSV, and JSON with all arm payloads.

- [ ] **Step 4: Run runner and regression tests**

Run: `pytest -q test_lightgcn_clv_m5_economic_positive_weight.py test_clv_m5_economic_positive_weight_model.py test_clv_m4_clv_hard_negative_loss.py test_lightgcn_clv_m5_embedding_hard_negative.py`

Expected: all tests pass; existing M4/M5 behavior remains unchanged.

- [ ] **Step 5: Commit the runner**

```bash
git add lightgcn_clv_m5_economic_positive_weight.py test_lightgcn_clv_m5_economic_positive_weight.py
git commit -m "Add six-arm M5 economic screen"
```

### Task 5: Colab entry point and final verification

**Files:**
- Create: `clv_m5_economic_positive_weight_dunnhumby_colab.ipynb`
- Modify: `test_lightgcn_clv_m5_economic_positive_weight.py`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Notebook imports `configure_m5_economic_positive_run`, `preflight_summary`, and `run_m5_economic_positive_screen`.
- Notebook checks out the reviewed commit, reloads local project modules, prints preflight, runs once, and displays all result tables and saved paths.

- [ ] **Step 1: Write the failing notebook contract test**

```python
def test_colab_runs_only_development_seed42_and_no_protected_evaluation():
    source = notebook_source("clv_m5_economic_positive_weight_dunnhumby_colab.ipynb")
    assert "run_m5_economic_positive_screen" in source
    assert "seed=42" in source
    assert "EVAL_TEST = True" not in source
    assert "EVAL_HOLDOUT = True" not in source
    assert "REVIEWED_SHA" in source
```

- [ ] **Step 2: Run the notebook test and verify RED**

Run: `pytest -q test_lightgcn_clv_m5_economic_positive_weight.py -k colab`

Expected: failure because the notebook does not exist.

- [ ] **Step 3: Generate the notebook and update research status**

Create a compact Colab with setup, fresh checkout, CUDA assertion, preflight, one run call, table display, and Drive paths. Record implementation facts in the existing master status; do not create another analysis document.

- [ ] **Step 4: Run final verification**

Run: `python -m py_compile clv_m5_economic_positive_weight_model.py lightgcn_clv_m5_economic_positive_weight.py`

Run: `pytest -q test_clv_m5_economic_positive_weight_model.py test_lightgcn_clv_m5_economic_positive_weight.py test_clv_m4_clv_hard_negative_loss.py test_lightgcn_clv_m5_embedding_hard_negative.py`

Run: `git diff --check`

Expected: compilation succeeds, all selected tests pass, and no whitespace errors are reported.

- [ ] **Step 5: Commit and push the completed experiment**

```bash
git add clv_m5_economic_positive_weight_dunnhumby_colab.ipynb \
        test_lightgcn_clv_m5_economic_positive_weight.py
git commit -m "Add M5 economic positive-weighting Colab"
git push origin feat/m2-joint-nv-lightgcn
```
