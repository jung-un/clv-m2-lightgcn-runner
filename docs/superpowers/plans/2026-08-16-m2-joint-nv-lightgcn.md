# M2 Joint N/V LightGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** CLV N/V 행동축을 LightGCN 초기 embedding에 내장하고 plain BPR로 한 번에 학습하는 M2 runner와 Colab을 추가한다.

**Architecture:** ID, N, V 분리 하위공간을 concat한 뒤 전체를 하나의 binary LightGCN으로 전파한다. 모든 파라미터는 하나의 optimizer와 plain BPR로 공동 학습한다.

**Tech Stack:** Python, PyTorch, NumPy, pandas, pytest, Google Colab

## Global Constraints

- binary graph, uniform negative sampling, uniform BPR only.
- no frozen/pretrained M1 inside M2, no encoder pretraining, no auxiliary loss, no post-score residual.
- validation-only seed 42; protected splits remain disabled.
- focused tests only during this exploration loop.

---

### Task 1: Joint N/V model

**Files:**
- Create: `clv_joint_nv_model.py`
- Create: `test_clv_joint_nv_model.py`

1. Write failing runtime tests for concatenated dimensions, one graph propagation path, BPR gradients into ID/N/V branches, fixed gates, and shuffled/constant controls.
2. Run the focused test and confirm RED.
3. Implement the minimal model and rerun to GREEN.

### Task 2: Validation runner

**Files:**
- Create: `lightgcn_clv_joint_nv.py`
- Create: `test_lightgcn_clv_joint_nv.py`

1. Write failing tests for protected-split validation, deterministic train-only profiles, plain-BPR configuration, and result schema.
2. Implement sequential H&M 60-day and full Dunnhumby seed-42 validation training/evaluation and persistence. Run only M1 and joint-NV in this fast screen.
3. Verify focused tests.

### Task 3: Colab entry point

**Files:**
- Create: `clv_joint_nv_hm_w60_colab.ipynb`

1. Add clone/import/preflight/run/result cells.
2. Ensure the notebook runs H&M 60-day seed 42 only and does not expose test/holdout.
3. Validate notebook JSON and import cells without starting training.

### Task 4: Focused verification and research record

**Files:**
- Modify: `RESEARCH_STATUS.md` after implementation status is known.

1. Run only the two new focused test files and syntax/notebook checks.
2. Record implementation and the fact that the high-cost validation run is not executed locally.
3. Commit/push only after the focused checks pass.
