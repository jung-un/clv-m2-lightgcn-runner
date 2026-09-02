# M3 CLV Supplemental Candidate Edges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 일반 신규상품 관계 100개를 보존하고 historical CLV proxy가 고른 일반 관계 밖 후보 20개만 추가하는 M3 그래프, 사후 집합 진단, 실행 차단 Colab을 구현한다.

**Architecture:** 기존 train-only 5-fold 후보관계 추정 코드를 확장해 일반·실제 CLV·degree-matched shuffle의 보조 희소행렬을 한 번에 만든다. 세 arm은 공통 기본 블록 질량 `5/6`과 서로 다른 추가 블록 질량 `1/6`을 가지며, 기존 `CLVCandidateItemLightGCN`이 같은 plain BPR 학습 루프 안에서 이를 사용한다.

**Tech Stack:** Python 3, NumPy, pandas, PyTorch sparse COO, pytest, Jupyter/Colab

**Spec:** `docs/superpowers/specs/2026-09-02-m3-clv-supplemental-candidate-edges-design.md`

## Global Constraints

- `historical_clv_proxy = N_hat * V_hat`, train-only mid-rank percentile를 사용한다.
- `base_candidate_items=100`, `supplemental_candidate_items=20`, 기본 질량 `5/6`, 추가 질량 `1/6`, `gamma=0.075`를 고정한다.
- M1 이진 구매그래프, `MIN_ITEM_INTER=1`, uniform negative sampling, plain BPR, 64차원, 2층, 100 epoch를 유지한다.
- 상품 가격, 표본가중치, 추가 손실, 외부 재정렬, holdout을 사용하지 않는다.
- DAY 684~690은 집합 진단에만 사용하고 DAY 705~711은 명시적 평가 승인 없이 열지 않는다.
- 논문·출력에서는 실제 매출을 뜻하는 표현 대신 가격·구매금액 가중 적중값을 사용한다.

---

### Task 1: 일반 기본 블록과 CLV 추가 후보 그래프

**Files:**
- Modify: `clv_m3_clv_conditioned_candidate_item_graph.py`
- Modify: `test_clv_m3_clv_conditioned_candidate_item_graph.py`

**Interfaces:**
- Consumes: 기존 `_probabilities`, `_item_probabilities`, `_historical_clv`, `_degree_stratified_shuffle`, `CLVCandidateItemGraph`
- Produces: `RELATION_MODE_SUPPLEMENTAL`, `build_clv_conditioned_supplemental_candidate_item_graph(train, n_users, n_items, n_cat, *, category_min_support_users, category_kappa, item_min_support_users, item_kappa, shuffle_seed, shuffle_degree_bins, cross_fit_folds, max_target_categories, base_candidate_items, supplemental_candidate_items) -> CLVCandidateItemGraph`

- [ ] **Step 1: Write failing graph tests**

```python
def _supplemental_graph():
    return build_clv_conditioned_supplemental_candidate_item_graph(
        _train(), n_users=8, n_items=6, n_cat=2,
        base_candidate_items=1, supplemental_candidate_items=1,
        category_kappa=0.0, category_min_support_users=1,
        item_kappa=0.0, item_min_support_users=1,
        shuffle_degree_bins=1, cross_fit_folds=2,
        max_target_categories=2,
    )


def test_supplemental_graph_preserves_base_and_matches_block_mass():
    graph = _supplemental_graph()
    operators = graph.user_item_operators
    for operator in operators.values():
        dense = operator.to_dense().numpy()
        np.testing.assert_allclose(dense.sum(axis=1), 1.0)
        assert np.all((dense > 0).sum(axis=1) == 2)
    assert graph.diagnostics["supplemental_support"]["base_edges_identical"]
    assert graph.diagnostics["supplemental_support"]["base_mass"] == pytest.approx(0.5)
    assert graph.diagnostics["supplemental_support"]["extra_mass"] == pytest.approx(0.5)


def test_supplemental_candidates_are_outside_base_and_train_pairs():
    graph = _supplemental_graph()
    assert graph.diagnostics["supplemental_support"]["base_extra_overlap"] == 0
    assert graph.diagnostics["supplemental_support"]["train_pair_edges"] == 0


def test_supplemental_graph_fails_when_positive_excess_is_insufficient():
    with pytest.raises(RuntimeError, match="positive excess candidates"):
        build_clv_conditioned_supplemental_candidate_item_graph(
            _train(), n_users=8, n_items=6, n_cat=2,
            base_candidate_items=3, supplemental_candidate_items=3,
            category_kappa=0.0, category_min_support_users=1,
            item_kappa=0.0, item_min_support_users=1,
            shuffle_degree_bins=1, cross_fit_folds=2,
            max_target_categories=2,
        )
```

- [ ] **Step 2: Run tests and confirm the new interface is missing**

Run: `pytest -q test_clv_m3_clv_conditioned_candidate_item_graph.py`

Expected: collection or test failure because the supplemental builder does not exist.

- [ ] **Step 3: Implement deterministic row construction**

Add constants and a public builder with the exact signature listed in the
Interfaces block. The fixed defaults are:

```python
RELATION_MODE_SUPPLEMENTAL = "pooled_base_plus_clv_positive_excess"
DEFAULT_BASE_CANDIDATE_ITEMS = 100
DEFAULT_SUPPLEMENTAL_CANDIDATE_ITEMS = 20
```

For every cross-fit consumer user:

```python
base = top_indices(general_score, base_candidate_items)
actual_extra = top_indices_where_not_base(
    np.maximum(actual_score - general_score, 0.0),
    supplemental_candidate_items,
)
shuffle_extra = top_indices_where_not_base(
    np.maximum(shuffle_score - general_score, 0.0),
    supplemental_candidate_items,
)
general_extra = pooled_ranks_101_to_120(general_score)
```

Fail closed unless every arm has exactly `base_candidate_items + supplemental_candidate_items` distinct candidates. Construct weights as:

```python
base_mass = base_candidate_items / (base_candidate_items + supplemental_candidate_items)
extra_mass = supplemental_candidate_items / (base_candidate_items + supplemental_candidate_items)
base_weights = base_mass * general_score[base] / general_score[base].sum()
extra_weights = np.full(supplemental_candidate_items, extra_mass / supplemental_candidate_items)
```

Use item id as the secondary key for deterministic score ties. Save per-user base and extra edge counts, mass errors, base-extra overlap, actual/shuffle extra Jaccard, train-pair count, and shuffle CLV movement in diagnostics.

- [ ] **Step 4: Run graph tests**

Run: `pytest -q test_clv_m3_clv_conditioned_candidate_item_graph.py`

Expected: all graph tests pass.

- [ ] **Step 5: Commit graph construction**

```bash
git add clv_m3_clv_conditioned_candidate_item_graph.py test_clv_m3_clv_conditioned_candidate_item_graph.py
git commit -m "feat: add supplemental CLV candidate graph"
```

### Task 2: Runner 설정, 평가 승인 차단, 성과 판정

**Files:**
- Modify: `lightgcn_clv_m3_clv_conditioned_candidate_item.py`
- Modify: `test_lightgcn_clv_m3_clv_conditioned_candidate_item.py`

**Interfaces:**
- Consumes: `RELATION_MODE_SUPPLEMENTAL`, `build_clv_conditioned_supplemental_candidate_item_graph`
- Produces: `configure_clv_candidate_item_supplemental_run(**overrides) -> CLVCandidateItemConfig`, `run_clv_candidate_item_supplemental(cfg) -> pd.DataFrame`, supplemental arm model ids

- [ ] **Step 1: Write failing runner tests**

```python
def test_supplemental_preflight_locks_graph_and_blocks_unapproved_evaluation(tmp_path):
    cfg = runner.configure_clv_candidate_item_supplemental_run(
        out_dir=str(tmp_path / "m3"), evaluation_authorized=False,
    )
    summary = runner.preflight_summary(cfg)
    assert summary["m3"]["base_candidate_items"] == 100
    assert summary["m3"]["supplemental_candidate_items"] == 20
    assert summary["m3"]["base_mass"] == pytest.approx(5 / 6)
    assert summary["m3"]["supplemental_mass"] == pytest.approx(1 / 6)
    with pytest.raises(RuntimeError, match="evaluation protocol approval"):
        runner.run_clv_candidate_item_supplemental(cfg)


def test_supplemental_config_rejects_seen_or_unapproved_test_boundaries(tmp_path):
    with pytest.raises(ValueError):
        runner.configure_clv_candidate_item_supplemental_run(
            out_dir=str(tmp_path), time_cutoff=697,
            evaluation_authorized=True,
        )
```

- [ ] **Step 2: Run targeted runner tests and confirm failure**

Run: `pytest -q test_lightgcn_clv_m3_clv_conditioned_candidate_item.py`

Expected: supplemental configuration functions are absent.

- [ ] **Step 3: Extend the existing runner without duplicating training code**

Add supplemental code version and model ids, config fields:

```python
base_candidate_items: int = 100
supplemental_candidate_items: int = 20
evaluation_authorized: bool = False
evaluation_protocol_label: str = ""
```

The configuration validator must reject performance execution when approval is false, any holdout request, and already-seen DAY 684~704 boundaries. The graph preparation branch calls the supplemental builder. Reuse the existing model, optimizer, training, metric, and JSON/CSV writing functions.

Keep the primary attribution rule:

```python
actual_vs_m1 > 1 and actual_vs_general > 1 and actual_vs_shuffle > 1
```

where each term is the geometric mean of Recall/NDCG@10·20·50 ratios. Add `candidate_truth_hits_actual > candidate_truth_hits_shuffle` as a separate mechanistic requirement; price and exposure metrics remain descriptive.

- [ ] **Step 4: Run runner and model tests**

Run: `pytest -q test_lightgcn_clv_m3_clv_conditioned_candidate_item.py test_clv_m3_clv_conditioned_candidate_item_model.py`

Expected: all tests pass.

- [ ] **Step 5: Commit runner integration**

```bash
git add lightgcn_clv_m3_clv_conditioned_candidate_item.py test_lightgcn_clv_m3_clv_conditioned_candidate_item.py
git commit -m "feat: wire supplemental CLV candidate screen"
```

### Task 3: 학습 없는 후보 집합 진단

**Files:**
- Create: `lightgcn_clv_m3_supplemental_candidate_diagnostic.py`
- Create: `test_lightgcn_clv_m3_supplemental_candidate_diagnostic.py`

**Interfaces:**
- Consumes: prepared train/test data and supplemental graph operators
- Produces: `candidate_truth_set_reading(graph, test_pairs) -> tuple[pd.DataFrame, dict]`, `run_supplemental_candidate_precheck(cfg) -> pd.DataFrame`

- [ ] **Step 1: Write failing pure set-reading tests**

```python
from types import SimpleNamespace

import torch


def _operator(pairs):
    indices = torch.tensor(pairs, dtype=torch.long).T
    values = torch.ones(len(pairs), dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (2, 8)).coalesce()


def _tiny_graph():
    return SimpleNamespace(
        user_item_operators={
            ARM_GENERAL: _operator([(0, 5)]),
            ARM_ACTUAL: _operator([(0, 4), (0, 5), (1, 6)]),
            ARM_SHUFFLE: _operator([(0, 4), (1, 7)]),
        },
        clv_percentile=np.array([0.25, 0.75]),
    )


def test_candidate_truth_set_reading_separates_net_from_actual_only():
    frame, reading = candidate_truth_set_reading(
        graph=_tiny_graph(),
        test_pairs={(0, 4), (0, 5), (1, 6), (1, 7)},
    )
    assert reading["actual_truth_hits"] - reading["shuffle_truth_hits"] == 1
    assert reading["actual_only_truth_pairs"] == 2
    assert reading["actual_only_outside_general_truth_pairs"] == 1
    assert reading["automatic_model_selection"] is False
```

- [ ] **Step 2: Run the new diagnostic test and confirm failure**

Run: `pytest -q test_lightgcn_clv_m3_supplemental_candidate_diagnostic.py`

Expected: module import fails because the diagnostic does not exist.

- [ ] **Step 3: Implement pair-set decomposition and saved outputs**

For each arm, convert sparse indices to `(user,item)` pairs, intersect with held-out truth, and save:

```python
actual_only = actual_hits - shuffle_hits
shuffle_only = shuffle_hits - actual_hits
actual_only_inside_general = actual_only & general_hits
actual_only_outside_general = actual_only - general_hits
```

Report overall and CLV-quintile rows, candidate hit count, truth-pair coverage, actual-only/shuffle-only counts, and the outside-general split. The returned routing may recommend training only when actual extra truth hits exceed shuffle and `actual_only_outside_general` is nonzero, but it must set `automatic_model_selection=False` and label the seen interval as post-hoc.

- [ ] **Step 4: Run diagnostic tests**

Run: `pytest -q test_lightgcn_clv_m3_supplemental_candidate_diagnostic.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the diagnostic**

```bash
git add lightgcn_clv_m3_supplemental_candidate_diagnostic.py test_lightgcn_clv_m3_supplemental_candidate_diagnostic.py
git commit -m "feat: diagnose supplemental CLV candidate sets"
```

### Task 4: Colab, 마스터 상태, 최종 국소 검증

**Files:**
- Create: `clv_m3_clv_supplemental_candidate_item_dunnhumby_colab.ipynb`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: Task 2 runner and Task 3 precheck
- Produces: one-click train-free precheck notebook and a disabled performance execution cell

- [ ] **Step 1: Create the Colab notebook**

The notebook must:

1. clone/pull `feat/m2-joint-nv-lightgcn` and remove stale imported modules from `sys.modules`;
2. mount Drive and configure Dunnhumby;
3. print the fixed design and run `run_supplemental_candidate_precheck` on DAY 684~690 without training;
4. display overall/quintile set tables and the routing result;
5. keep performance execution disabled with `evaluation_authorized=False` and explain that a new approved boundary is required;
6. never construct DAY 705~711 truth or a holdout in the default run.

- [ ] **Step 2: Update the master status document**

Record the design as an approved but not yet performance-evaluated M3, link the spec, distinguish the post-hoc set precheck from future test performance, and preserve all previous failures.

- [ ] **Step 3: Validate notebook structure and targeted tests**

Run:

```bash
python -m json.tool clv_m3_clv_supplemental_candidate_item_dunnhumby_colab.ipynb >/dev/null
pytest -q \
  test_clv_m3_clv_conditioned_candidate_item_graph.py \
  test_lightgcn_clv_m3_clv_conditioned_candidate_item.py \
  test_clv_m3_clv_conditioned_candidate_item_model.py \
  test_lightgcn_clv_m3_supplemental_candidate_diagnostic.py
```

Expected: notebook JSON is valid and all targeted tests pass.

- [ ] **Step 4: Inspect the final diff and commit**

Run: `git diff --check && git status --short`

Commit:

```bash
git add clv_m3_clv_supplemental_candidate_item_dunnhumby_colab.ipynb /Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md docs/superpowers/plans/2026-09-02-m3-clv-supplemental-candidate-edges.md
git commit -m "docs: add supplemental M3 Colab and status"
```

- [ ] **Step 5: Push the feature branch**

Run: `git push origin feat/m2-joint-nv-lightgcn`

Expected: the new commits are present on the current remote feature branch.
