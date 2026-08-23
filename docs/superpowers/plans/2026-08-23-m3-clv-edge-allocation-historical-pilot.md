# M3 Historical CLV Edge Allocation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a seed-42, fixed-100-epoch Dunnhumby historical backtest that compares M1 with a directional LightGCN graph whose item-receiving messages redistribute each user’s historical CLV proxy across price-free, popularity-discounted user–item relationships.

**Architecture:** A focused graph module computes train-only `N_u`, `V_u`, `C_u=N_u*V_u`, TF-IDF-shaped relationship shares, edge-level CLV allocations, and item-mass-preserving directional coefficients. A separate runner reconstructs DAY 1–683 training and DAY 684–690 evaluation, trains M1 and the proposal under identical fixed settings, evaluates each final checkpoint once, applies the predeclared pilot decision, and writes provenance-rich CSV/JSON outputs. A pinned Colab notebook provides the user-facing execution path.

**Tech Stack:** Python 3, NumPy, pandas, SciPy `rankdata`, PyTorch sparse tensors, pytest, Ruff, Jupyter/Colab JSON.

**Spec:** `docs/superpowers/specs/2026-08-23-m3-clv-edge-allocation-historical-pilot-design.md`

## Global Constraints

- Dunnhumby DAY 1–683 is training and DAY 684–690 is the only evaluation interval.
- Seed is exactly `42`; both arms train for exactly `100` epochs.
- No validation, early stopping, checkpoint selection, final test, or holdout is constructed.
- Evaluation remains new-item recommendation: every training `(user,item)` pair is excluded from truth and Top-K candidates.
- `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`, 64-dimensional two-layer LightGCN, plain BPR, uniform negative sampling, and equal optimizer settings are fixed across arms.
- M3 changes only graph propagation coefficients; it adds no M2 representation and no M4 sample/loss weighting.
- Public output uses `price_purchase_amount_weighted_hit`, `mean_recommended_price_percentile`, and `user_value_tendency_recommended_price_alignment`; it does not label weighted hits as revenue.
- One seed supports no standard deviation, confidence interval, significance, or generalization claim.

---

### Task 1: Train-Only Edge Allocation and Directional Operator

**Files:**
- Create: `clv_m3_edge_allocation_graph.py`
- Create: `test_clv_m3_edge_allocation_graph.py`

**Interfaces:**
- Consumes: a train `DataFrame` with `u_idx`, `i_idx`, `t`, `v`, and optional `b_raw`; integer `n_users`, `n_items`; a PyTorch device.
- Produces: `EdgeAllocatedCLVGraph`, `build_edge_allocated_clv_graph(train, n_users, n_items) -> EdgeAllocatedCLVGraph`, and `build_directional_torch_adj(graph, n_users, n_items, device) -> torch.Tensor`.

- [ ] **Step 1: Write the graph invariant tests**

```python
import numpy as np
import pandas as pd
import torch

from clv_m3_edge_allocation_graph import (
    build_directional_torch_adj,
    build_edge_allocated_clv_graph,
)


def _train():
    return pd.DataFrame(
        {
            "u_idx": [0, 0, 0, 1, 1, 2],
            "i_idx": [0, 0, 1, 0, 2, 2],
            "b_raw": [10, 11, 11, 20, 21, 30],
            "t": [1, 2, 2, 1, 2, 1],
            "v": [4.0, 5.0, 1.0, 10.0, 2.0, 3.0],
        }
    )


def test_clv_is_allocated_once_across_each_users_edges():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    allocated = np.bincount(
        graph.edge_users, weights=graph.edge_clv_allocation, minlength=3
    )
    np.testing.assert_allclose(allocated, graph.clv_proxy, rtol=0, atol=1e-10)
    shares = np.bincount(
        graph.edge_users, weights=graph.relationship_share, minlength=3
    )
    np.testing.assert_allclose(shares, np.ones(3), rtol=0, atol=1e-10)


def test_edge_set_and_item_message_mass_match_m1():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    assert list(zip(graph.edge_users, graph.edge_items)) == [
        (0, 0), (0, 1), (1, 0), (1, 2), (2, 2)
    ]
    base_mass = np.bincount(
        graph.edge_items, weights=graph.base_coefficients, minlength=3
    )
    changed_mass = np.bincount(
        graph.edge_items, weights=graph.item_user_coefficients, minlength=3
    )
    np.testing.assert_allclose(changed_mass, base_mass, rtol=0, atol=1e-10)


def test_degree_one_items_are_exactly_m1_and_all_coefficients_are_positive():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    degree = np.bincount(graph.edge_items, minlength=3)
    mask = degree[graph.edge_items] == 1
    np.testing.assert_array_equal(
        graph.item_user_coefficients[mask], graph.base_coefficients[mask]
    )
    assert np.isfinite(graph.item_user_coefficients).all()
    assert (graph.item_user_coefficients > 0).all()


def test_directional_operator_keeps_user_rows_at_m1():
    graph = build_edge_allocated_clv_graph(_train(), 3, 3)
    adj = build_directional_torch_adj(graph, 3, 3, torch.device("cpu")).to_dense()
    for edge, base in zip(
        zip(graph.edge_users, graph.edge_items), graph.base_coefficients
    ):
        user, item = edge
        assert adj[user, 3 + item].item() == base
```

- [ ] **Step 2: Run the new tests and verify the missing-module failure**

Run: `pytest -q test_clv_m3_edge_allocation_graph.py`

Expected: collection fails with `ModuleNotFoundError: No module named 'clv_m3_edge_allocation_graph'`.

- [ ] **Step 3: Implement the graph data structure and train-only statistics**

```python
@dataclass(frozen=True)
class EdgeAllocatedCLVGraph:
    edge_users: np.ndarray
    edge_items: np.ndarray
    base_coefficients: np.ndarray
    item_user_coefficients: np.ndarray
    relationship_share: np.ndarray
    edge_clv_allocation: np.ndarray
    n_hat: np.ndarray
    v_hat: np.ndarray
    clv_proxy: np.ndarray
    diagnostics: dict


def _basket_summary(train: pd.DataFrame, n_users: int):
    keys = ["u_idx", "b_raw"] if "b_raw" in train else ["u_idx", "t"]
    baskets = train.groupby(keys, sort=False)["v"].sum().rename("basket_value")
    grouped = baskets.groupby(level="u_idx", sort=False)
    summary = pd.DataFrame({"n_hat": grouped.size(), "v_hat": grouped.mean()})
    summary["clv_proxy"] = summary["n_hat"] * summary["v_hat"]
    return summary
```

In `build_edge_allocated_clv_graph`:

1. Validate required columns, non-empty train, and user/item bounds.
2. Build sorted unique edges by `u_idx, i_idx`.
3. Count `f_ui` as distinct source baskets containing each edge.
4. Count `d_i` as distinct users on each item.
5. Compute `r_ui=log1p(f_ui)*log((n_active_users+1)/(d_i+1))`.
6. Normalize `r_ui` within user; if a user sum is zero, use `1/user_degree`.
7. Compute `edge_clv_allocation=clv_proxy[edge_users]*relationship_share`.
8. Compute M1 `base=1/sqrt(user_degree*item_degree)`.
9. Convert allocations to average-rank percentiles separately within each item using `(rankdata(values, method="average")-0.5)/n` and set `c_ui=0.5+percentile`.
10. Multiply `base*c_ui` and rescale within item so its sum exactly matches the M1 base mass.
11. Store diagnostics for allocation conservation, coefficient ranges, item mass error, and Spearman correlations with item degree and train item price percentile.

- [ ] **Step 4: Implement the directional sparse operator**

```python
def build_directional_torch_adj(graph, n_users, n_items, device):
    rows = np.concatenate([graph.edge_users, graph.edge_items + n_users])
    cols = np.concatenate([graph.edge_items + n_users, graph.edge_users])
    values = np.concatenate(
        [graph.base_coefficients, graph.item_user_coefficients]
    ).astype(np.float32)
    indices = torch.from_numpy(np.stack([rows, cols]))
    return torch.sparse_coo_tensor(
        indices,
        torch.from_numpy(values),
        size=(n_users + n_items, n_users + n_items),
        check_invariants=False,
    ).coalesce().to(device)
```

- [ ] **Step 5: Run graph tests and static checks**

Run: `pytest -q test_clv_m3_edge_allocation_graph.py`

Expected: all tests pass.

Run: `ruff check clv_m3_edge_allocation_graph.py test_clv_m3_edge_allocation_graph.py`

Expected: exit code 0.

- [ ] **Step 6: Commit the graph component**

```bash
git add clv_m3_edge_allocation_graph.py test_clv_m3_edge_allocation_graph.py
git commit -m "feat: add CLV edge allocation graph"
```

---

### Task 2: Fixed-Epoch Historical Pilot Runner

**Files:**
- Create: `lightgcn_clv_m3_edge_allocation_backtest.py`
- Create: `test_lightgcn_clv_m3_edge_allocation_backtest.py`
- Reuse without modifying: `lightgcn_clv_axis_specific_test10.py`
- Reuse without modifying: `lightgcn_clv_repeatshare_backtest.py`
- Reuse without modifying: `lightgcn_clv_v3.py`

**Interfaces:**
- Consumes: `build_edge_allocated_clv_graph`, `build_directional_torch_adj`, existing `v3.prepare_data`, `v3.build_model`, fixed-epoch trainer and public metric adapters.
- Produces: `M3EdgeAllocationBacktestConfig`, `configure_m3_edge_allocation_backtest(**overrides)`, `preflight_summary(cfg)`, `_pilot_decision(frame)`, and `run_m3_edge_allocation_backtest(cfg=None) -> pd.DataFrame`.

- [ ] **Step 1: Write configuration and decision tests**

```python
import pandas as pd
import pytest

import lightgcn_clv_m3_edge_allocation_backtest as pilot


def test_pilot_is_locked_to_historical_seed42_without_selection(tmp_path):
    cfg = pilot.configure_m3_edge_allocation_backtest(out_dir=str(tmp_path))
    summary = pilot.preflight_summary(cfg)
    assert summary["models"] == ["m1_baseline", "m3_clv_edge_allocation"]
    assert summary["historical_development_split"] == {
        "train_end_inclusive": 683,
        "evaluation_start_inclusive": 684,
        "evaluation_end_inclusive": 690,
        "final_test_constructed": False,
        "holdout_constructed": False,
    }
    assert summary["seed"] == 42
    assert summary["epochs"] == 100
    assert summary["validation_or_epoch_selection"] is False


@pytest.mark.parametrize("key,value", [("seed", 43), ("epochs", 99), ("time_cutoff", 697)])
def test_pilot_config_fails_closed(tmp_path, key, value):
    with pytest.raises(ValueError):
        pilot.configure_m3_edge_allocation_backtest(
            out_dir=str(tmp_path), **{key: value}
        )


def test_pilot_decision_requires_every_predeclared_guard():
    frame = pd.DataFrame(
        [
            {
                "model_id": "m1_baseline",
                "recall@10": 1.0, "ndcg@10": 1.0,
                "recall@20": 1.0, "ndcg@20": 1.0,
                "recall@50": 1.0, "ndcg@50": 1.0,
                "price_purchase_amount_weighted_hit@10": 2.0,
                "mean_recommended_price_percentile@10": 0.25,
                "n_distinct@10": 200,
                "top10_share@10": 0.40,
            },
            {
                "model_id": "m3_clv_edge_allocation",
                "recall@10": 0.995, "ndcg@10": 1.01,
                "recall@20": 1.01, "ndcg@20": 1.01,
                "recall@50": 1.01, "ndcg@50": 1.01,
                "price_purchase_amount_weighted_hit@10": 2.1,
                "mean_recommended_price_percentile@10": 0.251,
                "n_distinct@10": 195,
                "top10_share@10": 0.405,
            },
        ]
    )
    assert pilot._pilot_decision(frame)["passes_pilot"] is True
    frame.loc[1, "recall@10"] = 0.98
    assert pilot._pilot_decision(frame)["passes_pilot"] is False
```

- [ ] **Step 2: Run runner tests and verify the missing-module failure**

Run: `pytest -q test_lightgcn_clv_m3_edge_allocation_backtest.py`

Expected: collection fails because the runner module does not exist.

- [ ] **Step 3: Implement the locked configuration and historical split preflight**

```python
@dataclass(frozen=True)
class M3EdgeAllocationBacktestConfig:
    dataset: str = "dunnhumby"
    seed: int = 42
    time_cutoff: int = 690
    evaluation_days: int = 7
    epochs: int = 100
    dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = 5e-4
    pref_reg: float = 1e-3
    out_dir: str = ""
```

Implement `validate_config` so dataset, seed, cutoff, evaluation days, epochs, dimension, and layers must equal the spec. Implement `_base_config` by calling `v3.configure_run` with `TIME_CUTOFF=690`, `TRAIN_ON_VAL=True`, `VAL_DAYS=7`, `TEST_DAYS=7`, `HOLDOUT_DAYS=0`, `EVAL_TEST=True`, `EVAL_HOLDOUT=False`, binary graph, plain loss, uniform negatives, and both interaction thresholds equal to 1. The internal `test` label represents the historical evaluation interval only; expose it publicly as `historical_development_days_684_690`.

- [ ] **Step 4: Implement data preparation and contamination checks**

`_prepare(cfg)` must:

1. Build and hash the input manifest and source revision.
2. Call `v3.prepare_data` once.
3. Require `set(data["splits"]) == {"test"}` while recording that this is the historical interval.
4. Require `train.t.max()==683`, no loss weights, no holdout, and no original final test rows.
5. Build the edge-allocation graph and require its sorted unique edge arrays to equal `data["pos_key"]` decomposed into users/items.
6. Build an `EvalCache`, item metadata, and M1 item features from the same train frame.
7. Store the graph diagnostics under `data_stats["m3_clv_edge_allocation"]`.

- [ ] **Step 5: Implement matched M1 and M3 training arms**

Use one `_build_model(prepared, cfg, model_id)` function. For M1 use `data["adj"]`; for M3 copy the data mapping and replace only `adj` with `build_directional_torch_adj(...)`. Call `v3.set_seed(42)` immediately before each model construction so both arms start from matched random initialization. Return `list(model.pref_params())` for the existing fixed-epoch trainer.

Use `ProgressStore` with separate model identities and source/input/config hashes. `_run_arm` must train with `test10._fixed_epoch_train`, save the final state atomically, evaluate once with `moe._flat_evaluation(..., per_user=False)`, and write an arm JSON containing model id, role, seed, historical split label, final epoch, public metrics, training record, graph diagnostics for M3, checkpoint path, and SHA-256.

- [ ] **Step 6: Implement the predeclared decision and outputs**

```python
ACCURACY_METRICS = (
    "recall@10", "ndcg@10", "recall@20",
    "ndcg@20", "recall@50", "ndcg@50",
)


def _pilot_decision(frame):
    rows = frame.set_index("model_id")
    base = rows.loc["m1_baseline"]
    model = rows.loc["m3_clv_edge_allocation"]
    accuracy = {metric: model[metric] / base[metric] for metric in ACCURACY_METRICS}
    checks = {
        "accuracy_guard": all(value >= 0.99 for value in accuracy.values()),
        "weighted_hit_improved": (
            model["price_purchase_amount_weighted_hit@10"]
            > base["price_purchase_amount_weighted_hit@10"]
        ),
        "price_guard": 0.97 <= (
            model["mean_recommended_price_percentile@10"]
            / base["mean_recommended_price_percentile@10"]
        ) <= 1.03,
        "catalog_guard": model["n_distinct@10"] / base["n_distinct@10"] >= 0.95,
        "exposure_guard": model["top10_share@10"] - base["top10_share@10"] <= 0.01,
    }
    return {
        "passes_pilot": all(checks.values()),
        "checks": checks,
        "accuracy_ratios": accuracy,
        "single_seed_limitation": "no variance, interval, significance, or generalization claim",
    }
```

Persist:

- absolute two-row CSV
- M3-minus-M1 metric comparison CSV with absolute and relative differences
- JSON containing config, preflight, input manifest, data stats, graph diagnostics, arm payloads, decision, and file paths

The runner must print all two rows and every failed guard, then return the absolute frame with comparison, decision, and paths in `DataFrame.attrs`.

- [ ] **Step 7: Run focused tests and relevant regressions**

Run:

```bash
pytest -q \
  test_clv_m3_edge_allocation_graph.py \
  test_lightgcn_clv_m3_edge_allocation_backtest.py \
  test_lightgcn_clv_m3_mass_preserving.py \
  test_lightgcn_clv_repeatshare_backtest.py
```

Expected: all tests pass.

Run:

```bash
ruff check \
  clv_m3_edge_allocation_graph.py \
  lightgcn_clv_m3_edge_allocation_backtest.py \
  test_clv_m3_edge_allocation_graph.py \
  test_lightgcn_clv_m3_edge_allocation_backtest.py
python -m py_compile \
  clv_m3_edge_allocation_graph.py \
  lightgcn_clv_m3_edge_allocation_backtest.py
```

Expected: both commands exit 0.

- [ ] **Step 8: Commit the runner**

```bash
git add \
  lightgcn_clv_m3_edge_allocation_backtest.py \
  test_lightgcn_clv_m3_edge_allocation_backtest.py
git commit -m "feat: add historical M3 edge allocation pilot"
```

---

### Task 3: Source-Pinned Colab Execution Notebook

**Files:**
- Create: `clv_m3_edge_allocation_backtest_colab.ipynb`

**Interfaces:**
- Consumes: the source commit produced after Tasks 1–2 and `run_m3_edge_allocation_backtest`.
- Produces: a one-click Colab flow that mounts Drive, checks out the exact source SHA, prints preflight, runs the two arms, and displays the compact result and decision.

- [ ] **Step 1: Record the exact source commit**

Run: `git rev-parse HEAD`

Expected: a 40-character commit SHA containing Tasks 1–2.

- [ ] **Step 2: Create notebook JSON with five executable cells**

Create a valid nbformat-4 notebook with:

1. Markdown: purpose, exploratory historical split, single-seed limitation, and no final test/holdout.
2. Setup: mount Drive, clone/fetch `https://github.com/jung-un/clv-m2-lightgcn-runner.git`, checkout the exact source SHA, assert `git rev-parse HEAD`, remove stale project modules from `sys.modules`, and add the repo to `sys.path`.
3. Preflight: import `configure_m3_edge_allocation_backtest` and `preflight_summary`, print JSON, and assert seed 42, epochs 100, train end 683, evaluation 684–690, and no final test/holdout.
4. Run: call `run_m3_edge_allocation_backtest(cfg)`.
5. Display: show the two-row public metric subset, the comparison table, the complete pilot decision, and result paths without displaying `DataFrame.attrs` directly.

Set the output directory to:

```python
"/content/drive/MyDrive/논문/data/results_m3_clv_edge_allocation_historical_dunnhumby"
```

- [ ] **Step 3: Validate notebook structure and pinned source**

Run:

```bash
python -m json.tool clv_m3_edge_allocation_backtest_colab.ipynb >/dev/null
python - <<'PY'
import json
from pathlib import Path

path = Path("clv_m3_edge_allocation_backtest_colab.ipynb")
nb = json.loads(path.read_text())
assert nb["nbformat"] == 4
source = "\n".join(
    "".join(cell.get("source", [])) for cell in nb["cells"]
)
assert "run_m3_edge_allocation_backtest" in source
assert "DAY 684" in source or "684" in source
assert "EVAL_HOLDOUT" not in source
PY
```

Expected: exit code 0.

- [ ] **Step 4: Commit the pinned notebook**

```bash
git add clv_m3_edge_allocation_backtest_colab.ipynb
git commit -m "chore: add Colab for M3 edge allocation pilot"
```

---

### Task 4: Final Verification, Research Status, and Remote Availability

**Files:**
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`
- Verify: all files created in Tasks 1–3

**Interfaces:**
- Consumes: passing source/tests/notebook and final commit SHAs.
- Produces: an auditable research-status entry, a pushed branch, and a Colab URL pinned to the branch notebook.

- [ ] **Step 1: Run the full relevant verification bundle**

Run:

```bash
pytest -q \
  test_clv_m3_edge_allocation_graph.py \
  test_lightgcn_clv_m3_edge_allocation_backtest.py \
  test_lightgcn_clv_m3_mass_preserving.py \
  test_lightgcn_clv_m3_behavior_diagnostic.py \
  test_lightgcn_clv_repeatshare_backtest.py
ruff check \
  clv_m3_edge_allocation_graph.py \
  lightgcn_clv_m3_edge_allocation_backtest.py \
  test_clv_m3_edge_allocation_graph.py \
  test_lightgcn_clv_m3_edge_allocation_backtest.py
python -m py_compile \
  clv_m3_edge_allocation_graph.py \
  lightgcn_clv_m3_edge_allocation_backtest.py
python -m json.tool clv_m3_edge_allocation_backtest_colab.ipynb >/dev/null
git diff --check HEAD~2..HEAD
```

Expected: all commands exit 0; only already-known warnings may appear.

- [ ] **Step 2: Update the master research status**

Add a dated entry that records:

- why the already-viewed final test is not reused;
- the DAY 1–683 / 684–690 historical split;
- the exact CLV allocation and item-mass-preserving formulas;
- M1/M3-only seed-42 fixed-epoch scope;
- predeclared pilot guards;
- source and notebook commit SHAs;
- tests completed;
- the fact that GPU training has not yet run and no performance claim exists.

Do not create a second analysis memo.

- [ ] **Step 3: Push the implementation branch and verify the remote ref**

Run:

```bash
git push origin feat/m2-joint-nv-lightgcn
git ls-remote origin refs/heads/feat/m2-joint-nv-lightgcn
git rev-parse HEAD
```

Expected: remote and local HEAD SHAs are identical.

- [ ] **Step 4: Deliver the Colab link and execution contract**

Return:

```text
https://colab.research.google.com/github/jung-un/clv-m2-lightgcn-runner/blob/feat/m2-joint-nv-lightgcn/clv_m3_edge_allocation_backtest_colab.ipynb
```

State that the notebook runs only M1 and the proposal on seed 42, that passing the pilot only opens the 10-seed/control/H&M stage, and that a one-seed result cannot support significance or generalization.
