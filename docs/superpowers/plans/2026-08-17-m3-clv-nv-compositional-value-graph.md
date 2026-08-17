# M3 CLV-NV Compositional Value Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Dunnhumby seed-42 validation runner comparing pure binary LightGCN M1 with one M3 graph whose edge weights compose historical CLV N/V user components with edge-level repeat-purchase and basket-value context.

**Architecture:** A focused `clv_m3_nv_graph.py` module computes aligned positive edge weights and diagnostics from train-only rows. `lightgcn_clv_v3.py` consumes the weights as a new `GRAPH_MODE="clv_nv"`, preserving its shared binary M1 baseline, LightGCN, plain BPR, evaluation, and persistence paths. A small preset module and pinned Colab expose only the approved two-model screen.

**Tech Stack:** Python, NumPy, pandas, SciPy rankdata, PyTorch sparse LightGCN, pytest, Google Colab.

## Global Constraints

- Dataset is Dunnhumby, seed 42, validation only.
- `EVAL_TEST=False` and `EVAL_HOLDOUT=False` before any data preparation.
- M1 and M3 use identical unique edges, dimension, layers, optimizer budget, plain BPR, and uniform negative sampling.
- M3 changes only the normalized graph edge weights.
- No alpha sweep and no N-only, V-only, or shuffled control training in this first screen.
- All user, basket, and edge quantities use train rows only.

---

### Task 1: Train-only CLV-NV edge-weight builder

**Files:**
- Create: `clv_m3_nv_graph.py`
- Create: `test_clv_m3_nv_graph.py`

**Interfaces:**
- Consumes: train `DataFrame` with `u_idx`, `i_idx`, `b_raw`, `t`, `v`; integer `n_users`, `n_items`.
- Produces: `build_clv_nv_graph(train, n_users, n_items) -> CLVNVGraphWeights` containing sorted `edge_users`, `edge_items`, `weights`, N/V components, user percentiles, and diagnostics.

- [ ] **Step 1: Write failing tests for alignment, positivity, boundedness, and semantics**

```python
graph = build_clv_nv_graph(train, n_users=2, n_items=3)
assert np.array_equal(graph.edge_users * 3 + graph.edge_items, np.unique(keys))
assert np.all(np.isfinite(graph.weights))
assert np.all((graph.weights >= 0.25) & (graph.weights <= 4.0))
assert graph.n_relation[repeated_edge] > graph.n_relation[single_edge]
assert graph.v_relation[large_basket_edge] > graph.v_relation[small_basket_edge]
```

- [ ] **Step 2: Run the focused test and confirm missing-module failure**

Run: `pytest -q test_clv_m3_nv_graph.py`

Expected: FAIL because `clv_m3_nv_graph` does not exist.

- [ ] **Step 3: Implement deterministic train-only aggregation**

```python
@dataclass(frozen=True)
class CLVNVGraphWeights:
    edge_users: np.ndarray
    edge_items: np.ndarray
    weights: np.ndarray
    n_relation: np.ndarray
    v_relation: np.ndarray
    q_n: np.ndarray
    q_v: np.ndarray
    diagnostics: dict
```

Compute repeat transaction rate and mean basket value per user, edge distinct-basket count, and mean total basket value conditional on the edge. Use average-tie percentiles, mean-one N/V components, the approved composition formula, and `[0.25, 4.0]` clipping. Fail closed on missing columns, empty input, misaligned edges, non-finite values, or non-positive weights.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q test_clv_m3_nv_graph.py`

Expected: PASS.

### Task 2: Integrate `clv_nv` as an M3 graph mode

**Files:**
- Modify: `lightgcn_clv_v3.py`
- Modify: `test_lightgcn_clv_v3.py`

**Interfaces:**
- Consumes: `build_clv_nv_graph()` result.
- Produces: `prepare_data()` with `w_edge`, weighted sparse adjacency, and `data_stats["clv_nv_graph"]`; `model_id(cfg) == "m3_clv_nv"`.

- [ ] **Step 1: Write a failing runtime integration test**

```python
cfg = {**V3.CFG, "GRAPH_MODE": "clv_nv"}
data = V3.prepare_data(cfg, fixture_schema)
assert data["w_edge"].shape == data["pos_key"].shape
assert not np.allclose(data["w_edge"], 1.0)
assert np.array_equal(data["adj"].indices(), expected_weighted_indices)
```

- [ ] **Step 2: Confirm the new mode is rejected before implementation**

Run: `pytest -q test_lightgcn_clv_v3.py -k clv_nv`

Expected: FAIL because `GRAPH_MODE="clv_nv"` is unknown.

- [ ] **Step 3: Dispatch the new mode without changing other modes**

```python
elif cfg["GRAPH_MODE"] == "clv_nv":
    graph = build_clv_nv_graph(train, n_users, n_items)
    np.testing.assert_array_equal(graph.edge_users, eu)
    np.testing.assert_array_equal(graph.edge_items, ei)
    w_edge = graph.weights
    data_stats["clv_nv_graph"] = graph.diagnostics
```

Update accepted-mode messages and `CODE_VERSION`; retain `binary_baseline()` as the external M1 comparator.

- [ ] **Step 4: Run the graph and existing v3 tests**

Run: `pytest -q test_clv_m3_nv_graph.py test_lightgcn_clv_v3.py`

Expected: PASS.

### Task 3: Safe Dunnhumby runner and pinned Colab

**Files:**
- Create: `lightgcn_clv_m3_nv.py`
- Create: `test_lightgcn_clv_m3_nv.py`
- Create: `clv_m3_nv_dunnhumby_colab.ipynb`
- Modify: `RESEARCH_STATUS.md` after implementation facts are known.

**Interfaces:**
- Produces: `configure_m3_clv_nv_dunnhumby_run()`, `preflight_summary()`, `run_experiment()`.

- [ ] **Step 1: Write failing preset and notebook safety tests**

```python
cfg = configure_m3_clv_nv_dunnhumby_run()
assert cfg["DATASET"] == "dunnhumby"
assert cfg["SEED_LIST"] == [42]
assert cfg["ARCH"] == "pref_only"
assert cfg["GRAPH_MODE"] == "clv_nv"
assert cfg["LOSS_MODE"] == "plain"
assert cfg["EVAL_TEST"] is cfg["EVAL_HOLDOUT"] is False
```

The notebook test asserts a full pinned SHA, one `run_experiment()` call, GPU check, and no approval flag.

- [ ] **Step 2: Implement preset and preflight guard**

Build from `v3.configure_run()` with an isolated result directory. `run_experiment()` revalidates all safety invariants immediately before `v3.main()`.

- [ ] **Step 3: Add one-cell execution and M1 comparison output**

The notebook prints absolute metrics, M1 deltas, decision, graph diagnostics, and result paths. It never enables test or holdout.

- [ ] **Step 4: Verify only the changed path**

Run:

```bash
pytest -q test_clv_m3_nv_graph.py test_lightgcn_clv_m3_nv.py test_lightgcn_clv_v3.py
ruff check clv_m3_nv_graph.py lightgcn_clv_m3_nv.py test_clv_m3_nv_graph.py test_lightgcn_clv_m3_nv.py lightgcn_clv_v3.py test_lightgcn_clv_v3.py
python -m json.tool clv_m3_nv_dunnhumby_colab.ipynb >/dev/null
git diff --check
```

- [ ] **Step 5: Commit source, pin notebook to that source commit, commit notebook, and push the active branch**

Record implementation and validation-not-yet-run status in `RESEARCH_STATUS.md`; do not state that M3 succeeded before Colab results exist.
