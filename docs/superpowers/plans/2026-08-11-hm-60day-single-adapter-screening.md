# H&M 60-Day Single-Adapter Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** H&M 공식 거래자료의 마지막 60일만 사용해 seed 42 validation-only 단일 어댑터 M2 screening을 안전하게 실행하고, 2년 실행 및 기존 Dunnhumby 결과와 체크포인트·산출물을 완전히 격리한다.

**Architecture:** 공개 설정에 선택적 `window_days`를 추가하고 이를 외부 M1 설정까지 한 경로로 전달한다. H&M 60일 전용 preset은 14일 행동입력, 7일 미래예측, `(21, 14, 7)` anchor와 보호 split 차단을 고정한다. encoder anchor 품질을 결과 JSON에 감사 가능하게 저장하며, 별도 Colab notebook은 검토된 commit을 고정하고 고비용 승인 전에는 학습하지 않는다.

**Tech Stack:** Python 3.11, dataclasses, PyTorch, pandas/numpy, pytest, ruff, Jupyter notebook JSON, Git/GitHub.

## Global Constraints

- 이 작업은 M2 임베딩·표현 실험만 바꾼다. 그래프는 binary, 손실은 plain BPR, negative sampling은 uniform을 유지한다.
- H&M 60일은 탐색적 비용·파이프라인 진단이다. H&M 2년 본실험을 대체하지 않는다.
- test와 holdout 정답은 구성하지 않는다. `run_experiment()` 진입점에서도 fail closed를 유지한다.
- 선택 규칙은 Recall/NDCG@10/20/50 각각 M1의 99% 이상이면서, 양의 lambda의 가격·구매금액 가중 적중값@10이 M1보다 실제로 높아야 한다.
- 고비용 학습은 이 계획의 코드·테스트·notebook·상태문서 검토와 push가 끝난 뒤 사용자가 Colab 승인값을 바꿀 때만 시작한다.
- 모든 동작 변경은 실패하는 테스트를 먼저 확인한 뒤 최소 구현으로 통과시킨다.

---

## Task 1: Window 설정을 공개 API에서 외부 M1까지 전달

**Files:**

- Modify: `lightgcn_clv_moe.py:126-197,278-310,558-582`
- Test: `test_lightgcn_clv_moe.py`

- [ ] **Step 1: `window_days` 유효성 및 M1 전달 실패 테스트 작성**

  `test_lightgcn_clv_moe.py`에 다음 동작 테스트를 추가한다.

  ```python
  @pytest.mark.parametrize("window_days", [0, -1])
  def test_window_days_must_be_none_or_positive(window_days):
      with pytest.raises(ValueError, match="window_days"):
          moe.configure_moe_run("hm", window_days=window_days)


  def test_pure_m1_receives_screening_window(monkeypatch):
      captured = {}

      def fake_configure_run(**kwargs):
          captured.update(kwargs)
          return {
              **kwargs,
              "ARCH": "pref_only",
              "GRAPH_MODE": "binary",
              "LOSS_MODE": "plain",
              "NEG_MODE": "uniform",
          }

      monkeypatch.setattr(moe.v3, "configure_run", fake_configure_run)
      cfg = moe.configure_moe_run("hm", window_days=60)
      base_cfg = moe._pure_m1_config(cfg, "/tmp/m1-hm-w60")
      assert captured["WINDOW_DAYS"] == 60
      assert base_cfg["WINDOW_DAYS"] == 60
  ```

- [ ] **Step 2: RED 확인**

  Run: `pytest -q test_lightgcn_clv_moe.py -k 'window_days or pure_m1_receives_screening_window'`

  Expected: `MoEConfig`가 `window_days`를 받지 않거나 M1이 계속 `None`을 받아 실패한다.

- [ ] **Step 3: 최소 설정 구현**

  `MoEConfig`에 아래 필드를 추가한다.

  ```python
  window_days: int | None = None
  ```

  `validate_moe_config()`에서 `None` 또는 양의 정수만 허용한다. `bool`은 정수로 취급하지 않는다.

  ```python
  if cfg.window_days is not None and (
      isinstance(cfg.window_days, bool) or int(cfg.window_days) <= 0
  ):
      raise ValueError("window_days는 None 또는 양의 정수여야 합니다")
  ```

  `_pure_m1_config()`의 `WINDOW_DAYS=None`을 다음으로 바꾼다.

  ```python
  WINDOW_DAYS=cfg.window_days,
  ```

- [ ] **Step 4: 사람이 읽을 수 있는 preflight 테스트와 구현**

  테스트는 full H&M과 60일 H&M을 구분하고, 60일 설정에서 약 39일 train과 validation-only를 표시하는지 확인한다.

  ```python
  full = moe.preflight_summary(moe.configure_moe_run("hm"))
  short = moe.preflight_summary(moe.configure_moe_run("hm", window_days=60))
  assert "full official" in full["window"]
  assert short["window_days"] == 60
  assert short["estimated_train_days"] == 39
  assert "last 60 calendar days" in short["window"]
  ```

  구현은 현재 고정 split 예약기간 `7 + 7 + 7`을 명시적으로 빼되, 이를 실제 행 수가 아니라 사전 추정치라고 표시한다.

- [ ] **Step 5: 회귀 테스트와 commit**

  Run: `pytest -q test_lightgcn_clv_moe.py`

  Expected: PASS.

  Run: `git diff --check`

  Commit: `git commit -am "feat: propagate bounded data windows to M1"`

---

## Task 2: H&M 60일 preset과 체크포인트 격리

**Files:**

- Modify: `lightgcn_clv_single.py:30-57,157-208,284-330,470-530`
- Test: `test_lightgcn_clv_single.py`
- Test: `test_lightgcn_clv_residual.py`

- [ ] **Step 1: 안전한 전용 preset 실패 테스트 작성**

  다음 불변식을 테스트한다.

  ```python
  def test_hm_60day_preset_is_validation_only_and_fits_train_window():
      cfg = single.configure_hm_60day_run()
      assert cfg.dataset == "hm"
      assert cfg.window_days == 60
      assert cfg.seed_list == (42,)
      assert (cfg.input_days, cfg.target_days) == (14, 7)
      assert cfg.anchor_offsets == (21, 14, 7)
      assert cfg.input_days + max(cfg.anchor_offsets) == 35
      assert not cfg.eval_test
      assert not cfg.eval_holdout


  @pytest.mark.parametrize(
      ("key", "value"),
      [("window_days", 61), ("input_days", 28), ("eval_test", True)],
  )
  def test_hm_60day_preset_rejects_frozen_overrides(key, value):
      with pytest.raises(ValueError, match="H&M 60-day"):
          single.configure_hm_60day_run(**{key: value})
  ```

- [ ] **Step 2: RED 확인 후 preset 구현**

  Run: `pytest -q test_lightgcn_clv_single.py -k 'hm_60day_preset'`

  `lightgcn_clv_single.py`에 고정 연구설정과 허용 가능한 실행설정을 분리한다.

  ```python
  HM_60DAY_FROZEN = {
      "window_days": 60,
      "seed_list": (42,),
      "input_days": 14,
      "target_days": 7,
      "anchor_offsets": (21, 14, 7),
      "eval_test": False,
      "eval_holdout": False,
      "lambda_eval": LAMBDA_GRID,
      "accuracy_tolerance": ACCURACY_TOLERANCE,
  }


  def configure_hm_60day_run(**overrides) -> moe.MoEConfig:
      for key, expected in HM_60DAY_FROZEN.items():
          if key in overrides and overrides[key] != expected:
              raise ValueError(f"H&M 60-day 고정설정 변경 금지: {key}")
      return configure_single_run("hm", **(HM_60DAY_FROZEN | overrides))
  ```

  `out_dir`와 `m1_checkpoint_dir`은 notebook이 `hm_w60` 전용 경로를 명시하므로 고정 연구값에는 넣지 않는다.

- [ ] **Step 3: 실제 anchor 구성이 가능한지 runtime test 작성**

  `test_lightgcn_clv_residual.py`의 날짜 거래 fixture를 39일 이상 만들고 아래를 검증한다.

  ```python
  ds = residual.build_anchor_examples(
      train,
      n_users=2,
      is_date=True,
      input_days=14,
      target_days=7,
      anchor_offsets=(21, 14, 7),
  )
  assert [anchor.offset_days for anchor in ds.anchors] == [21, 14, 7]
  windows = [(a.target_start, a.target_end) for a in ds.anchors]
  assert windows[0][1] < windows[1][0]
  assert windows[1][1] < windows[2][0]
  ```

  Run: `pytest -q test_lightgcn_clv_residual.py -k '60_day or short_window'`

  Expected: PASS without production changes unless an off-by-one defect is exposed.

- [ ] **Step 4: 기존 재사용과 60일 재사용을 동시에 보호하는 테스트 작성**

  `REUSE_CONFIG_KEYS`에 `window_days`를 넣되 저장 JSON에 키가 없는 기존 결과는 `None`으로만 해석한다.

  ```python
  saved_value = saved_config.get(key) if key == "window_days" else saved_config[key]
  ```

  테스트 세 가지를 둔다.

  1. 저장 config에 `window_days`가 없는 기존 full-window 결과와 현재 `None` 설정은 기존 검증을 통과한다.
  2. 저장 `window_days=None`과 현재 `60`은 `saved config mismatch for window_days`로 거부한다.
  3. 저장 `window_days=60`과 현재 `60`은 정확히 일치할 때만 재사용한다.

- [ ] **Step 5: fingerprint 및 M1 hash 격리 테스트 작성**

  동일 source/data에서 `window_days=None`과 `60`으로 만든 `_result_fingerprint()`가 다르고, `_pure_m1_config()`의 `v3.cfg_hash()`도 다른지 확인한다. 이 테스트는 2년 M1을 60일 실행이 재사용하지 못하도록 보장해야 한다.

- [ ] **Step 6: 집중 테스트와 commit**

  Run: `pytest -q test_lightgcn_clv_single.py test_lightgcn_clv_residual.py`

  Expected: PASS.

  Run: `git diff --check`

  Commit: `git commit -am "feat: add isolated H&M 60-day screening preset"`

---

## Task 3: Anchor별 입력 품질 진단 저장

**Files:**

- Modify: `lightgcn_clv_single.py:92-107,487-550,856-960`
- Test: `test_lightgcn_clv_single.py`

- [ ] **Step 1: 순수 진단 함수 테스트 작성**

  `summarize_anchor_dataset()`의 출력은 anchor별로 다음 필드를 가져야 한다.

  ```python
  {
      "offset_days": 21,
      "n_users": 2,
      "observation_start": "...",
      "observation_end": "...",
      "target_start": "...",
      "target_end": "...",
      "observed_days_p10": ...,
      "observed_days_median": ...,
      "observed_days_p90": ...,
      "purchase_rate": ...,
      "future_amount_mean": ...,
  }
  ```

  synthetic `AnchorDataset`으로 사용자 수, `observed_days` 열의 분위수, 구매율, 금액 평균을 정확히 검증한다.

- [ ] **Step 2: RED 확인 후 순수 함수 구현**

  Run: `pytest -q test_lightgcn_clv_single.py -k anchor_diagnostics`

  Expected: helper가 없어 FAIL.

  구현은 `residual.NUMERIC_FEATURES.index("observed_days")`로 열을 찾고 `np.quantile`을 사용한다. 날짜는 `str()`로 직렬화한다.

- [ ] **Step 3: 준비 context와 JSON persistence 연결 테스트 작성**

  `PreparedSingleContext`에 `anchor_diagnostics: tuple[dict, ...]`를 추가한다. `_prepare_validation_context()`에서 anchor 생성 직후 계산하고, encoder checkpoint 및 최종 JSON의 최상위 `anchor_diagnostics`에 동일 내용을 저장한다.

  `_persist_result()` 단위 테스트는 세 anchor가 모두 JSON에 남고 순서가 `(21, 14, 7)`인지 확인한다.

- [ ] **Step 4: 집중 테스트와 commit**

  Run: `pytest -q test_lightgcn_clv_single.py -k 'anchor_diagnostics or persist'`

  Expected: PASS.

  Run: `ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py`

  Commit: `git commit -am "feat: persist short-window anchor diagnostics"`

---

## Task 4: 전용 Colab notebook과 고비용 차단

**Files:**

- Create: `clv_single_adapter_hm_w60_colab.ipynb`
- Modify: `test_lightgcn_clv_single.py`

- [ ] **Step 1: notebook 계약 테스트 작성**

  JSON을 읽어 모든 코드 셀 source를 합친 뒤 다음을 검사한다.

  ```python
  assert "configure_hm_60day_run" in source
  assert "results_clv_single_hm_w60" in source
  assert "results_v3_hm_w60" in source
  assert "ACKNOWLEDGE_HIGH_COST = False" in source
  assert "window_days == 60" in source
  assert "input_days == 14" in source
  assert "anchor_offsets == (21, 14, 7)" in source
  assert "eval_test" in source and "eval_holdout" in source
  assert "TO_BE_PINNED" not in source
  ```

  `REVIEWED_SHA`는 40자리 소문자 Git SHA 형식인지 정규식으로 검사한다.

- [ ] **Step 2: RED 확인 후 notebook 작성**

  Run: `pytest -q test_lightgcn_clv_single.py -k hm_w60_colab`

  notebook 셀 순서는 다음과 같이 고정한다.

  1. 연구상 지위와 60일/약 39일 train/14→7 설명.
  2. Drive mount, repository fresh clone, 검토 SHA checkout 및 HEAD assert.
  3. import와 `configure_hm_60day_run()` 호출. 출력 경로는 각각 `results_clv_single_hm_w60`, `results_v3_hm_w60`.
  4. GPU 및 모든 frozen 설정 assert, `preflight_summary()` 출력.
  5. `ACKNOWLEDGE_HIGH_COST = False` 승인 gate.
  6. `run_experiment(cfg)` 실행.
  7. 절대곡선, delta, 최종 `screening_decision`, 저장 파일 표시.

  이 단계에서는 `REVIEWED_SHA`를 바로 앞 구현 commit의 실제 SHA로 넣는다. 가변 branch pull이나 placeholder를 사용하지 않는다.

- [ ] **Step 3: notebook 정적 검증**

  Run: `python -m json.tool clv_single_adapter_hm_w60_colab.ipynb >/dev/null`

  Run: `pytest -q test_lightgcn_clv_single.py -k 'hm_w60_colab or colab'`

  Expected: PASS.

- [ ] **Step 4: notebook commit**

  Run: `git add clv_single_adapter_hm_w60_colab.ipynb test_lightgcn_clv_single.py`

  Commit: `git commit -m "feat: add pinned H&M 60-day Colab runner"`

---

## Task 5: 전체 검증, 연구상태 기록, 원격 공개

**Files:**

- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`
- Verify: all changed Python, tests, notebook, spec and plan files

- [ ] **Step 1: 실행 없는 preflight smoke**

  Run:

  ```bash
  python - <<'PY'
  import json
  from lightgcn_clv_single import configure_hm_60day_run, preflight_summary

  cfg = configure_hm_60day_run(
      out_dir="/tmp/results_clv_single_hm_w60",
      m1_checkpoint_dir="/tmp/results_v3_hm_w60",
  )
  summary = preflight_summary(cfg)
  assert summary["window_days"] == 60
  assert summary["estimated_train_days"] == 39
  assert summary["encoder_windows"] == {
      "input_days": 14,
      "target_days": 7,
      "anchor_offsets": [21, 14, 7],
  }
  assert not summary["eval_test"] and not summary["eval_holdout"]
  print(json.dumps(summary, ensure_ascii=False, indent=2))
  PY
  ```

  Expected: 설정 요약만 출력하고 데이터 접근·M1 학습·encoder 학습은 시작하지 않는다.

- [ ] **Step 2: 전체 자동검증**

  Run: `pytest -q test_lightgcn_clv_moe.py test_lightgcn_clv_single.py test_lightgcn_clv_residual.py`

  Run: `pytest -q`

  Run: `ruff check lightgcn_clv_moe.py lightgcn_clv_single.py test_lightgcn_clv_moe.py test_lightgcn_clv_single.py test_lightgcn_clv_residual.py`

  Run: `python -m json.tool clv_single_adapter_hm_w60_colab.ipynb >/dev/null`

  Run: `git diff --check`

  Expected: 모두 exit 0.

- [ ] **Step 3: 연구상태 갱신**

  `RESEARCH_STATUS.md`에 다음을 확정 사실/미실행/다음 실험으로 구분해 기록한다.

  - 구현 commit과 검증 명령·통과 건수.
  - 28일안이 `28+21=49일` 때문에 실행 전에 폐기되고 14일안이 채택된 근거.
  - H&M 60일 고비용 학습 및 결과는 아직 미실행이라는 사실.
  - 다음 실행은 seed 42 validation-only이며, test/holdout과 다중 seed는 닫혀 있다는 사실.

- [ ] **Step 4: 마지막 코드 commit과 clean 확인**

  코드 저장소에 남은 변경이 있으면 목적에 맞는 하나의 commit으로 묶는다.

  Run: `git status --short`

  Expected: code worktree clean. `RESEARCH_STATUS.md`는 저장소 밖의 상위 연구상태 문서이므로 별도 저장 완료를 확인한다.

- [ ] **Step 5: GitHub push 전 최종 검토**

  notebook의 `REVIEWED_SHA`가 실제 존재하는 구현 commit이며 현재 notebook을 제외한 실행 코드를 가리키는지 확인한다.

  Run: `git rev-parse HEAD`

  Run: `git log --oneline --decorate -6`

  Run: `git push origin feat/clv-conditioned-moe`

  Push 뒤 원격 ref가 갱신됐는지 확인한다.

  Run: `git ls-remote --heads origin feat/clv-conditioned-moe`

- [ ] **Step 6: 사용자에게 실행 경계 인계**

  다운로드할 notebook의 절대 경로와 GitHub/Colab 링크를 제공한다. `ACKNOWLEDGE_HIGH_COST=False` 상태에서는 학습이 시작되지 않으며, 사용자가 preflight 전체를 한 번 검토한 뒤에만 `True`로 바꾸도록 안내한다. 고비용 결과가 없으므로 성능 개선 여부는 보고하지 않는다.
