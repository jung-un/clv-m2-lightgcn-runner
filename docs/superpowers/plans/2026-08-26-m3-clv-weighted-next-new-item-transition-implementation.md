# M3 CLV-Weighted Next-New-Item Transition Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a locked, CPU-only, train-only Dunnhumby diagnostic that tests whether historical-CLV-weighted next-new-item transitions rank future new-item truths better than the identical unweighted and shuffled-CLV relations.

**Architecture:** Add a pure sparse-relation module for event construction, historical CLV coefficients, within-activity-stratum shuffling, row-normalized transition matrices, and candidate ranking. Wrap it with a fixed historical diagnostic runner that loads only DAY 1--669, constructs relations from DAY 1--662, evaluates DAY 663--669, enforces the predeclared pilot decision, and writes reproducible CSV/JSON artifacts. A thin Colab notebook clones the pinned branch and calls only the locked runner.

**Tech Stack:** Python 3, pandas, NumPy, SciPy sparse matrices, pytest, Google Colab.

**Spec:** `docs/superpowers/specs/2026-08-26-m3-clv-weighted-next-new-item-transition-design.md`

## Global Constraints

- This is an exploratory M3 diagnostic, not confirmatory final-test evidence.
- Read no transactions after DAY 669; relation/CLV inputs must end at DAY 662.
- Preserve the new-item task and `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`.
- Use historical CLV proxy `N_hat * V_hat`; do not use item price, N/V routing, external reranking, PPMI, pruning, learned attention, or a learned fusion coefficient.
- Compare exactly `transition_global`, `transition_clv`, and `transition_clv_shuffle` with fixed shuffle seed `20260826`.
- Do not create a holdout, validation-selection loop, or GPU model in this phase.
- Public result labels use “가격·구매금액 가중 적중값” and “추천 상품의 평균 가격 백분위”; do not label them Revenue or ARP.

---

## Task 1: Implement transition-event and graph primitives

**Files:**

- Create: `clv_m3_next_new_transition.py`
- Create: `test_clv_m3_next_new_transition.py`

- [ ] **Step 1: Write failing tests for consecutive-basket target construction**

  Add hand-built users where a later basket contains repeat and first-purchase items. Assert that only first-purchase targets remain and that users with no next-new target contribute no event.

- [ ] **Step 2: Run the focused test and confirm failure**

  Run: `pytest -q test_clv_m3_next_new_transition.py -k next_new`

  Expected: import or missing-function failure.

- [ ] **Step 3: Implement `build_user_transition_events`**

  Interface:

  ```python
  @dataclass(frozen=True)
  class TransitionEvents:
      user_idx: np.ndarray
      source_item_idx: np.ndarray
      target_item_idx: np.ndarray
      contribution: np.ndarray
      eligible_pair_count_by_user: np.ndarray

  def build_user_transition_events(
      transactions: pd.DataFrame,
      *,
      n_users: int,
  ) -> TransitionEvents:
      ...
  ```

  Require columns `u_idx`, `i_idx`, `t`, and `basket_id`. Deduplicate items within each basket, order baskets by `(t, basket_id)`, calculate `New_(u,t+1)`, allocate `1/(|B_t| |New_(t+1)|)`, then normalize all retained contributions for each user to sum to one.

- [ ] **Step 4: Add and run tests for basket-size and user-level mass normalization**

  Run: `pytest -q test_clv_m3_next_new_transition.py -k 'normalization or next_new'`

  Expected: all selected tests pass.

- [ ] **Step 5: Write failing tests for CLV coefficients and constrained shuffle**

  Assert `N_hat` equals distinct basket count, `V_hat` equals mean basket total, `CLV=N_hat*V_hat`, coefficient mean is one, and the shuffle preserves the coefficient multiset within every `N_hat` midrank decile.

- [ ] **Step 6: Implement CLV and shuffle functions**

  Interfaces:

  ```python
  @dataclass(frozen=True)
  class HistoricalCLV:
      n_hat: np.ndarray
      v_hat: np.ndarray
      clv_proxy: np.ndarray
      percentile: np.ndarray
      coefficient: np.ndarray
      activity_decile: np.ndarray

  def build_historical_clv(
      transactions: pd.DataFrame,
      *,
      n_users: int,
      shuffle_seed: int = 20260826,
  ) -> tuple[HistoricalCLV, np.ndarray]:
      ...
  ```

  Basket value is the train-only sum of `v` per `(user,basket)`. Percentiles use deterministic midranks. Shuffle only `coefficient` within `activity_decile`.

- [ ] **Step 7: Write failing tests for sparse graph aggregation and row normalization**

  Assert each nonempty source row sums to one, empty rows remain zero, CLV changes contributor allocation, and all three graphs keep the same shape.

- [ ] **Step 8: Implement `build_transition_graphs`**

  Interface:

  ```python
  @dataclass(frozen=True)
  class TransitionGraphs:
      global_relation: sparse.csr_matrix
      clv_relation: sparse.csr_matrix
      shuffled_clv_relation: sparse.csr_matrix
      edge_support: sparse.csr_matrix

  def build_transition_graphs(
      events: TransitionEvents,
      *,
      clv_coefficient: np.ndarray,
      shuffled_coefficient: np.ndarray,
      n_items: int,
  ) -> TransitionGraphs:
      ...
  ```

  Aggregate without PPMI, minimum-support filtering, or Top-M pruning. Row-normalize each relation independently.

- [ ] **Step 9: Run all core tests**

  Run: `pytest -q test_clv_m3_next_new_transition.py`

  Expected: all tests pass.

- [ ] **Step 10: Commit Task 1**

  ```bash
  git add clv_m3_next_new_transition.py test_clv_m3_next_new_transition.py
  git commit -m "feat: add CLV-weighted next-new transition graphs"
  ```

## Task 2: Implement ranking, metrics, and pilot decision

**Files:**

- Modify: `clv_m3_next_new_transition.py`
- Modify: `test_clv_m3_next_new_transition.py`

- [ ] **Step 1: Write failing tests for last-basket candidate scoring**

  Assert the user score is the mean relation row over last-basket sources, construction purchases are excluded, no popularity backfill occurs, and ties use deterministic item-index order.

- [ ] **Step 2: Implement `rank_transition_candidates`**

  Interface:

  ```python
  def rank_transition_candidates(
      relation: sparse.csr_matrix,
      *,
      last_basket_items: dict[int, np.ndarray],
      seen_items: dict[int, np.ndarray],
      eval_users: np.ndarray,
      top_k: int = 50,
  ) -> dict[int, np.ndarray]:
      ...
  ```

  Return only positive-score candidates, sorted by score descending then item index ascending.

- [ ] **Step 3: Write failing tests for evaluation and pass rule**

  Cover Recall/NDCG @10/20/50, reachable-truth share, distinct Top-10, top-10-item exposure share, and all six predeclared pass conditions, including exact boundary cases.

- [ ] **Step 4: Implement evaluation helpers and `decide_pilot`**

  Interfaces:

  ```python
  def evaluate_transition_ranking(...) -> tuple[pd.DataFrame, pd.DataFrame]: ...
  def decide_pilot(metric_table: pd.DataFrame) -> dict[str, object]: ...
  ```

  Include support strata, CLV quintiles, candidate counts, item-popularity correlation, item-price-percentile correlation, coverage, entropy, effective catalog, top-10 share, and top-100 share. Treat unavailable price data as an explicit missing diagnostic, never as zero.

- [ ] **Step 5: Run focused and full core tests**

  Run: `pytest -q test_clv_m3_next_new_transition.py`

  Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

  ```bash
  git add clv_m3_next_new_transition.py test_clv_m3_next_new_transition.py
  git commit -m "feat: add transition ranking diagnostic metrics"
  ```

## Task 3: Implement the locked Dunnhumby historical runner

**Files:**

- Create: `lightgcn_clv_m3_next_new_transition_diagnostic.py`
- Create: `test_lightgcn_clv_m3_next_new_transition_diagnostic.py`

- [ ] **Step 1: Write failing configuration and leakage-guard tests**

  Assert fixed dataset `dunnhumby`, construction end `662`, pseudo-future start/end `663/669`, shuffle seed `20260826`, `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`, no holdout, and rejection of any transaction with `t>669` entering the prepared frame.

- [ ] **Step 2: Implement locked configuration and preflight summary**

  Interfaces:

  ```python
  @dataclass(frozen=True)
  class M3NextNewTransitionDiagnosticConfig:
      dataset: str = "dunnhumby"
      construction_end_day: int = 662
      evaluation_start_day: int = 663
      evaluation_end_day: int = 669
      shuffle_seed: int = 20260826
      rank_limit: int = 50
      out_dir: str = ""

  def configure_m3_next_new_transition_diagnostic(**overrides) -> ...: ...
  def preflight_summary(cfg) -> dict: ...
  ```

- [ ] **Step 3: Write failing tests for split preparation**

  Use a small synthetic frame to assert construction/evaluation boundaries, construction-pair exclusion from truth, last construction basket extraction, and no future leakage into CLV, transition events, or item diagnostics.

- [ ] **Step 4: Implement `_prepare_transactions` and `_build_truth`**

  Load through `lightgcn_clv_v3.load_transactions`, merge Dunnhumby item metadata, apply the train-based one-pass-converged k-core with thresholds one, index users/items from the capped DAY 1--669 frame, and retain `basket_id` from `b_raw` (fall back to `t` only if the schema lacks it). The truth is unique DAY 663--669 items absent from each user's DAY 1--662 history.

- [ ] **Step 5: Write failing artifact and integrity tests**

  Assert saved configuration includes input manifest/hash/source revision, all row-mass invariants pass, no construction pair appears in truth or recommendations, metric recomputation matches, and filenames include the run hash.

- [ ] **Step 6: Implement `run_m3_next_new_transition_diagnostic`**

  Interface:

  ```python
  def run_m3_next_new_transition_diagnostic(
      cfg: M3NextNewTransitionDiagnosticConfig,
  ) -> pd.DataFrame:
      ...
  ```

  Save absolute metrics, pairwise comparison, segment metrics, support diagnostics, integrity checks, and one JSON record. Attach `quality_passed`, `pilot_decision`, and `result_paths` to the returned DataFrame attributes.

- [ ] **Step 7: Run runner tests**

  Run: `pytest -q test_lightgcn_clv_m3_next_new_transition_diagnostic.py`

  Expected: all tests pass.

- [ ] **Step 8: Run combined focused suite**

  Run: `pytest -q test_clv_m3_next_new_transition.py test_lightgcn_clv_m3_next_new_transition_diagnostic.py`

  Expected: all tests pass.

- [ ] **Step 9: Commit Task 3**

  ```bash
  git add lightgcn_clv_m3_next_new_transition_diagnostic.py test_lightgcn_clv_m3_next_new_transition_diagnostic.py
  git commit -m "feat: add locked M3 transition feasibility runner"
  ```

## Task 4: Add the pinned Colab entry point and research record

**Files:**

- Create: `clv_m3_next_new_transition_diagnostic_colab.ipynb`
- Modify: `RESEARCH_STATUS.md`

- [ ] **Step 1: Create the minimal Colab notebook**

  Cells must: mount Drive; clone or update `jung-un/clv-m2-lightgcn-runner` on `feat/m2-joint-nv-lightgcn`; install only missing dependencies; print `preflight_summary`; run the diagnostic; assert `quality_passed`; display the three-arm table, pilot decision, and result paths. Do not expose split or graph settings as editable notebook controls.

- [ ] **Step 2: Validate notebook structure without executing the full dataset**

  Run:

  ```bash
  python -m json.tool clv_m3_next_new_transition_diagnostic_colab.ipynb >/dev/null
  python - <<'PY'
  import json
  p='clv_m3_next_new_transition_diagnostic_colab.ipynb'
  nb=json.load(open(p))
  source='\n'.join(''.join(c.get('source', [])) for c in nb['cells'])
  for required in ['configure_m3_next_new_transition_diagnostic', 'preflight_summary', 'run_m3_next_new_transition_diagnostic', "quality_passed"]:
      assert required in source, required
  for forbidden in ['EVAL_HOLDOUT=True', 'TIME_CUTOFF=690', 'construction_end_day=']:
      assert forbidden not in source, forbidden
  PY
  ```

  Expected: both commands exit zero.

- [ ] **Step 3: Update `RESEARCH_STATUS.md`**

  Record the approved hypothesis, locked DAY 1--662 / DAY 663--669 exploratory interval, Phase-1-only status, fixed controls, and the rule that no neural M3 model is built unless the diagnostic passes. Mark implementation as complete but experiment result as pending.

- [ ] **Step 4: Run complete targeted verification**

  Run:

  ```bash
  pytest -q test_clv_m3_next_new_transition.py test_lightgcn_clv_m3_next_new_transition_diagnostic.py
  python -m py_compile clv_m3_next_new_transition.py lightgcn_clv_m3_next_new_transition_diagnostic.py
  git diff --check
  ```

  Expected: tests pass, compilation succeeds, and no whitespace errors are reported.

- [ ] **Step 5: Review spec coverage and placeholder hygiene**

  Run:

  ```bash
  rg -n "TODO|TBD|placeholder|NotImplemented" clv_m3_next_new_transition.py lightgcn_clv_m3_next_new_transition_diagnostic.py test_clv_m3_next_new_transition.py test_lightgcn_clv_m3_next_new_transition_diagnostic.py clv_m3_next_new_transition_diagnostic_colab.ipynb
  ```

  Expected: no matches.

  Manually verify every design-spec invariant and all six pilot conditions have at least one code path and one test assertion.

- [ ] **Step 6: Commit and push the implementation**

  ```bash
  git add clv_m3_next_new_transition_diagnostic_colab.ipynb RESEARCH_STATUS.md docs/superpowers/plans/2026-08-26-m3-clv-weighted-next-new-item-transition-implementation.md
  git commit -m "docs: add M3 transition diagnostic Colab and status"
  git push origin feat/m2-joint-nv-lightgcn
  ```

- [ ] **Step 7: Hand off the Colab URL**

  Provide:

  `https://colab.research.google.com/github/jung-un/clv-m2-lightgcn-runner/blob/feat/m2-joint-nv-lightgcn/clv_m3_next_new_transition_diagnostic_colab.ipynb`

  State explicitly that the notebook is the locked Phase-1 feasibility diagnostic, not the neural M3 model or a final-test run.
