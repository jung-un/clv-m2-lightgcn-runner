# CLV Dual-Axis Fixed-Gate Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a validation-only M2 runner that combines literature-grounded future-transaction and transaction-value user embeddings with matching item activity/economic embeddings through fixed percentile gates.

**Architecture:** Keep the external M1 LightGCN frozen and add two L2-normalized residual embedding experts. Train with the same binary graph, uniform negative sampling, and plain BPR as M1; screen only `m1`, `dual_clv_fixed`, `dual_shuffled_gate`, and `dual_base_only`.

**Tech Stack:** Python, PyTorch, NumPy, pandas, SciPy, pytest, Google Colab notebook JSON.

## Global Constraints

- Both datasets use the same model equation and four model IDs.
- H&M 60-day uses customer-date as transaction proxy; Dunnhumby uses `b_raw`/`BASKET_ID`.
- M1 is frozen for the complete adapter training run.
- Graph is binary; loss is unweighted plain BPR; negative sampling is uniform.
- Gates are fixed: `g_N=2*percentile(N_hat)`, `g_V=2*percentile(V_hat)` among valid users.
- Recall/NDCG@10/20/50 use the existing 1% validation guardrail; test and holdout remain unavailable.
- Run only M1, `dual_clv_fixed`, `dual_shuffled_gate`, and `dual_base_only`.

---

### Task 1: Two-axis CLV targets and user representation

**Files:**
- Modify: `lightgcn_clv_residual.py`
- Create: `clv_core_features.py`
- Create: `test_clv_core_features.py`

**Interfaces:**
- Produces: `AnchorExamples.transaction_target`, `AnchorExamples.mean_transaction_value_target`.
- Produces: `train_clv_core_encoder(...) -> CLVCoreArtifact` with `n_hat_all`, `v_hat_all`, and `ev_all=n_hat_all*v_hat_all`.
- Produces: `compose_clv_core_profiles(...) -> UserProfileArtifact` containing 29 literature-axis values and predictions.

- [ ] **Step 1: Add the failing target-decomposition test**

```python
def test_anchor_targets_separate_future_transaction_count_and_value():
    anchor = build_anchor_examples(...).anchors[-1]
    np.testing.assert_array_equal(anchor.transaction_target, [4.0, 2.0])
    np.testing.assert_allclose(anchor.mean_transaction_value_target, [2.0, 5.0])
    np.testing.assert_allclose(
        anchor.transaction_target * anchor.mean_transaction_value_target,
        anchor.amount_target,
    )
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_clv_core_features.py`

Expected: failure because the two target fields and `clv_core_features` do not exist.

- [ ] **Step 3: Implement count/value targets and the separated encoder**

Use `b_raw.nunique()` when available and `t.nunique()` otherwise. Train the count head against `log1p(future transaction count)` for all observed users, and train the monetary head against `log1p(future mean transaction value)` only for users with a future transaction. Core repurchase inputs are `recency_days`, `basket_count`, `observed_days`, and `gap_mean`; core monetary inputs are `avg_basket_value`, with hidden dimensions 8+8.

- [ ] **Step 4: Verify GREEN**

Run: `pytest -q test_clv_core_features.py test_lightgcn_clv_residual.py test_clv_moe_features.py`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lightgcn_clv_residual.py clv_core_features.py test_clv_core_features.py
git commit -m "feat: add literature-grounded CLV core encoder"
```

### Task 2: Dual item axes, fixed gates, and frozen-base model

**Files:**
- Create: `clv_dual_axis_model.py`
- Create: `test_clv_dual_axis_model.py`

**Interfaces:**
- Produces: `DualItemProfile(activity, value, valid_activity, valid_value, activity_names, value_names)`.
- Produces: `build_dual_item_profiles(train, n_items, is_date) -> DualItemProfile`.
- Produces: `fixed_percentile_gates(n_hat, v_hat, valid_user) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `CLVDualAxisEmbeddingModel(base_model, user_profile, item_profile, g_n, g_v, control, seed, ...)` with `score_all`, `score_pairs`, and frozen base parameters.

- [ ] **Step 1: Write failing tests for fixed gates and item-axis separation**

```python
def test_fixed_gates_are_monotone_and_mean_one():
    g_n, g_v = fixed_percentile_gates(
        np.array([1., 3., 2.]), np.array([30., 10., 20.]), np.ones(3, bool)
    )
    assert np.argsort(g_n).tolist() == [0, 2, 1]
    assert np.argsort(g_v).tolist() == [1, 2, 0]
    assert np.isclose(g_n.mean(), 1.0)
    assert np.isclose(g_v.mean(), 1.0)

def test_item_axes_have_disjoint_named_features():
    profile = build_dual_item_profiles(tiny_train, n_items=4, is_date=False)
    assert "repeat_purchase_share" in profile.activity_names
    assert "price_percentile" in profile.value_names
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_clv_dual_axis_model.py`

Expected: import failure for `clv_dual_axis_model`.

- [ ] **Step 3: Implement dual profiles and controls**

Activity item features are unique-buyer percentile, repeat-purchase share, median repeat gap plus validity, and within-category frequency percentile. Economic item features are global price percentile, within-category price percentile, log mean unit price, and mean item share of transaction value. `dual_shuffled_gate` permutes the `(g_N,g_V)` pair jointly among valid users with seed 42. `dual_base_only` zeros user and item side information but preserves dimensions and adapter parameter count. Expert outputs are L2-normalized before dot products.

- [ ] **Step 4: Prove base freezing and lambda-zero identity**

```python
def test_dual_model_freezes_m1_and_lambda_zero_is_exact_base():
    model = CLVDualAxisEmbeddingModel(...)
    assert not any(p.requires_grad for p in model.base_model.parameters())
    torch.testing.assert_close(model.score_all(users, 0.0), model.base_score_all(users))
```

- [ ] **Step 5: Verify GREEN and commit**

Run: `pytest -q test_clv_dual_axis_model.py`

```bash
git add clv_dual_axis_model.py test_clv_dual_axis_model.py
git commit -m "feat: add fixed-gate dual-axis embedding model"
```

### Task 3: Four-model validation runner and Colab

**Files:**
- Create: `lightgcn_clv_dual.py`
- Create: `test_lightgcn_clv_dual.py`
- Create: `clv_dual_axis_colab.ipynb`

**Interfaces:**
- Produces: `DualAxisConfig` or a narrow preset wrapper around `MoEConfig` with dataset, time-window, encoder, adapter, and lambda settings.
- Produces: `configure_dual_run(dataset, short_hm=False, **overrides)`.
- Produces: `run_experiment(cfg) -> pd.DataFrame` with result paths, selected lambdas, and authoritative screening decision in `attrs`.

- [ ] **Step 1: Write failing runner policy tests**

```python
def test_dual_runner_is_seed42_validation_only_and_has_four_models():
    cfg = configure_dual_run("hm", short_hm=True)
    summary = preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["models"] == [
        "m1", "dual_clv_fixed", "dual_shuffled_gate", "dual_base_only"
    ]
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q test_lightgcn_clv_dual.py`

Expected: import failure for `lightgcn_clv_dual`.

- [ ] **Step 3: Implement shared-M1 training, selection, and persistence**

Train `dual_clv_fixed` first. If it has no positive lambda satisfying all six accuracy guardrails and weighted-hit@10 above M1, return failure without training controls. If it passes, train only `dual_shuffled_gate` and `dual_base_only`. Save all lambda rows, paired bootstrap deltas, exposure metrics, N/V/gate hashes and summaries, encoder diagnostics, model/checkpoint hashes, source/data/M1 fingerprints, and the final decision.

- [ ] **Step 4: Add one dataset-toggle Colab**

The notebook defaults to H&M 60-day seed 42 validation and exposes a single `DATASET_PRESET` switch for `hm_w60` or `dunnhumby`. It mounts Drive, checks out a pinned source SHA, prints preflight, runs the experiment without a redundant acknowledgement flag, and displays absolute curves, paired deltas, selected lambdas, and the final decision.

- [ ] **Step 5: Run focused verification**

Run:

```bash
pytest -q test_clv_core_features.py test_clv_dual_axis_model.py test_lightgcn_clv_dual.py
python -m json.tool clv_dual_axis_colab.ipynb >/dev/null
ruff check clv_core_features.py clv_dual_axis_model.py lightgcn_clv_dual.py \
  test_clv_core_features.py test_clv_dual_axis_model.py test_lightgcn_clv_dual.py
git diff --check
```

Expected: all commands exit 0. Do not run raw-data training locally.

- [ ] **Step 6: Commit source, then pin notebook**

```bash
git add lightgcn_clv_dual.py test_lightgcn_clv_dual.py \
  clv_dual_axis_colab.ipynb
git commit -m "feat: add dual-axis CLV screening runner"
```

Replace `REVIEWED_SHA` in the notebook with that source commit, verify notebook JSON and the SHA assertion test, then commit the pin separately.
