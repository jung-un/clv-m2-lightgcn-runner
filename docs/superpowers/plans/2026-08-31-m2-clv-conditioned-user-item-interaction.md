# M2 CLV-Conditioned User–Item Interaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a seed-42 Dunnhumby historical-development runner that jointly learns one bounded user–item–CLV interaction beside the ordinary 64-dimensional LightGCN representation.

**Architecture:** Keep the ordinary two-layer LightGCN ID representation unchanged. Project the final user and item representations into one four-dimensional unit space, condition the user projection with fixed train-history overall CLV percentile and N/V composition, and concatenate the bounded interaction coordinates before the single BPR dot product. Train a matched `rho=0` arm and `rho=0.05` arm with identical initialization and sampling.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, SciPy, pytest, Jupyter/Colab

**Spec:** `docs/superpowers/specs/2026-08-31-m2-clv-conditioned-user-item-interaction-design.md`

## Global Constraints

- New-item recommendation only; exclude train `(user,item)` pairs at evaluation.
- Binary graph, uniform negative sampling, no BPR sample weights, no new loss term.
- One model, one optimizer, one plain BPR training loop; no frozen or external reranking stage.
- `MIN_ITEM_INTER=1`, two LightGCN layers, 100 fixed epochs, seed 42.
- Historical development split only: train through day 683 and evaluate days 684–690; do not construct final test or holdout.
- Fixed `id_dim=64`, `context_dim=4`, `rho∈{0.0,0.05}`.

---

### Task 1: CLV-conditioned interaction model

**Files:**
- Create: `clv_conditioned_user_item_interaction_model.py`
- Test: `test_clv_conditioned_user_item_interaction_model.py`

**Interfaces:**
- Consumes: `q_c: np.ndarray`, `d_nv: np.ndarray`, `user_clv_valid: np.ndarray`, normalized sparse `adj: torch.Tensor`.
- Produces: `CLVConditionedUserItemInteractionLightGCN`, with `id_embeddings()`, `interaction_embeddings()`, `embeddings()`, `bpr_loss()`, `candidate_score_diagnostics()`, and scalar diagnostic methods compatible with the shared trainer.

- [ ] **Step 1: Implement the model**

  Create free `E_u/E_i` ID tables, two-layer LightGCN propagation, bias-free projections `P_u/P_i`, and learned vectors `w_C/w_D`. Compute:

  ```python
  h_u = F.normalize(P_u(z_u), dim=1, eps=1e-12)
  h_i = F.normalize(P_i(z_i), dim=1, eps=1e-12)
  h_c = F.normalize(w_c + d_nv[:, None] * w_d, dim=1, eps=1e-12)
  a_u = q_c[:, None] * h_u * h_c
  user = torch.cat([z_u, sqrt(rho) * a_u], dim=1)
  item = torch.cat([z_i, sqrt(rho) * h_i], dim=1)
  ```

  Keep sampled ID L2 only and reject M3/M4/external-score arguments.

- [ ] **Step 2: Add focused model tests**

  Verify exact `rho=0` score parity, bounded `|R|≤q_C`, candidate-specific and user-specific interaction values, one joint BPR gradient reaching ID/projections/context vectors, invalid-user zeroing, and M4/external-score rejection.

- [ ] **Step 3: Run model tests**

  Run: `pytest -q test_clv_conditioned_user_item_interaction_model.py`

  Expected: all tests pass.

### Task 2: Matched historical-development runner and diagnostics

**Files:**
- Create: `lightgcn_clv_conditioned_user_item_interaction.py`
- Test: `test_lightgcn_clv_conditioned_user_item_interaction.py`

**Interfaces:**
- Consumes: the verified historical split helpers and compatible stored M1@64 result.
- Produces: `configure_conditioned_user_item_interaction_run()`, `preflight_summary()`, and `run_conditioned_user_item_interaction_screen()` returning a DataFrame with result paths and reading metadata.

- [ ] **Step 1: Build fixed CLV inputs and fail-closed preparation**

  Use existing `build_user_axis_inputs()` and calculate:

  ```python
  valid = valid_user & activity_valid & value_valid
  q_c, _ = fixed_percentile_ranks(clv_proxy, clv_proxy, valid)
  d_nv = np.where(valid, q_n - q_v, 0.0).astype(np.float32)
  ```

  Assert only the day-684–690 development split exists and no loss weights are present.

- [ ] **Step 2: Train matched arms and save checkpoints**

  Reset seed 42 before constructing each identical architecture. Train `m1_matched_rho0` and `m2_clv_conditioned_user_item_interaction` for 100 epochs with independent resumable progress stores.

- [ ] **Step 3: Evaluate full and ID-only views**

  Save external M1@64, matched `rho=0`, M2 full, and the jointly trained M2 ID-only evaluation in one absolute table. Write comparisons against both matched `rho=0` and external M1.

- [ ] **Step 4: Save mechanism diagnostics**

  Use deterministic evaluation-user/candidate samples for `std(rho*R)/std(S_ID)`, `mean/max(|rho*R|)`, `q_C`–mean-`|R|` Spearman, N/V-composition summaries, and `w_C/w_D` cosine. Recompute Top-10 for the matched and M2 arms and save changed-user shares overall and by CLV segment.

- [ ] **Step 5: Apply the predeclared screen**

  Require all Recall/NDCG@10/20/50 ratios versus matched `rho=0` to be at least `0.99`, high-CLV Recall@10 and NDCG@10 to rise, overall price/purchase-amount weighted hit@10 not to fall, nonzero high-CLV Top-10 changes, and exact `rho=0` non-intervention parity.

- [ ] **Step 6: Add and run runner tests**

  Run: `pytest -q test_lightgcn_clv_conditioned_user_item_interaction.py`

  Expected: configuration freezes every invariant, CLV inputs are train-only and valid-masked, screening logic passes/fails on controlled tables, and Top-10 overlap summaries are correct.

### Task 3: Colab entry point

**Files:**
- Create: `clv_m2_conditioned_user_item_interaction_dunnhumby_colab.ipynb`

**Interfaces:**
- Consumes: a pinned Git commit containing Tasks 1–2.
- Produces: one GPU Colab workflow that mounts Drive, clones the repository at the reviewed SHA, prints preflight, runs both arms, and displays the core comparisons and diagnostics.

- [ ] **Step 1: Create the notebook**

  Add markdown explaining the hypothesis and code cells for Drive mount, reviewed SHA checkout, preflight assertions, execution, and display.

- [ ] **Step 2: Pin the source revision**

  Commit Tasks 1–2, insert that commit SHA into `REVIEWED_SHA`, and verify notebook JSON parses.

### Task 4: Verification, research status, and delivery

**Files:**
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: implementation and focused test outcomes.
- Produces: a reproducible branch and Colab link; no experimental performance claim before the user runs Colab.

- [ ] **Step 1: Run focused regression suite**

  Run:

  ```bash
  pytest -q \
    test_clv_conditioned_user_item_interaction_model.py \
    test_lightgcn_clv_conditioned_user_item_interaction.py \
    test_lightgcn_clv_v3.py \
    test_lightgcn_clv_axis_specific_test10.py
  ```

  Expected: all tests pass.

- [ ] **Step 2: Validate files**

  Run `python -m py_compile` on both Python files, parse the notebook with `python -m json.tool`, and run `git diff --check`.

- [ ] **Step 3: Record implementation status**

  Add exact filenames, fixed protocol, test result, and the statement that no performance or significance claim exists before Colab execution.

- [ ] **Step 4: Commit and push**

  Commit the notebook/status update and push `feat/m2-joint-nv-lightgcn`.

