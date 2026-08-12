# CLV Dual-Axis Checkpoint Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 `dual_clv_fixed` 체크포인트를 재학습 없이 재평가하여 N-only/V-only/N+V, 선택점 주변 lambda, 사용자 N/V 4분면의 validation 성과를 저장한다.

**Architecture:** `CLVDualAxisEmbeddingModel`에 평가 전용 axis mask를 추가하고, 별도 `lightgcn_clv_dual_checkpoint_diagnostic.py`가 원본 결과 manifest를 검증한 뒤 기존 데이터·M1·encoder·dual checkpoint를 복원한다. 동일 validation cache에서 세 축 조합을 평가하고, 집계·사용자 대응 delta·4분면·실효강도 비교를 저장한다. Colab은 Dunnhumby와 H&M 60일 원본 JSON을 순차 처리하며 학습 함수를 호출하지 않는다.

**Tech Stack:** Python, PyTorch, pandas, NumPy, matplotlib, pytest, Google Colab

## Global Constraints

- seed 42 validation-only이며 test·holdout을 생성하거나 평가하지 않는다.
- 기존 M1·encoder·`dual_clv_fixed` checkpoint를 재사용하고 학습 optimizer/epoch 함수를 호출하지 않는다.
- Dunnhumby는 gate `equal`, H&M 60일은 gate `high`를 사용한다.
- N-only, V-only, N+V는 새 학습모형이 아니라 동일 checkpoint의 사후 메커니즘 ablation이다.
- 결과는 기존 screening 파일을 덮어쓰지 않고 `checkpoint_diagnostics/`에 저장한다.
- 원본 source/data/M1/checkpoint hash가 맞지 않으면 평가 전에 중단한다.

---

### Task 1: 평가 전용 축 마스킹

**Files:**
- Modify: `clv_dual_axis_model.py`
- Test: `test_clv_dual_axis_model.py`

**Interfaces:**
- Consumes: `CLVDualAxisEmbeddingModel.score_all(users, lam, gate_shape)` 및 `score_pairs(...)`
- Produces: `set_eval_axes(axis_mode: str) -> None`, 지원값 `"n_only"`, `"v_only"`, `"n_plus_v"`

- [ ] **Step 1: 축 마스킹 공개 동작의 실패 테스트 작성**

```python
def test_eval_axis_mask_reuses_same_experts_without_retraining():
    model = make_dual_model()
    users = torch.tensor([0, 1])
    full = model.score_all(users, 1.0, "equal")
    model.set_eval_axes("n_only")
    n_only = model.score_all(users, 1.0, "equal")
    model.set_eval_axes("v_only")
    v_only = model.score_all(users, 1.0, "equal")
    base = model.base_score_all(users)
    torch.testing.assert_close(full - base, (n_only - base) + (v_only - base))
    assert all(parameter.requires_grad is False for parameter in model.parameters()) is False
```

- [ ] **Step 2: 실패 확인**

Run: `pytest -q test_clv_dual_axis_model.py::test_eval_axis_mask_reuses_same_experts_without_retraining`

Expected: FAIL because `set_eval_axes` does not exist.

- [ ] **Step 3: 최소 축 마스킹 구현**

```python
EVAL_AXIS_MODES = ("n_only", "v_only", "n_plus_v")

def set_eval_axes(self, axis_mode: str) -> None:
    if axis_mode not in EVAL_AXIS_MODES:
        raise ValueError(f"지원하지 않는 axis mode: {axis_mode}")
    self.eval_axis_mode = axis_mode
```

`score_all`과 `score_pairs`에서 N/V residual을 각각 계산한 후 `eval_axis_mode`에 따라 필요한 축만 합산한다. 기본값은 기존 결과와 동일한 `n_plus_v`로 둔다. `bpr_loss`는 항상 두 축을 사용하여 학습 의미가 바뀌지 않게 한다.

- [ ] **Step 4: 모델 테스트 통과 확인**

Run: `pytest -q test_clv_dual_axis_model.py`

Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add clv_dual_axis_model.py test_clv_dual_axis_model.py
git commit -m "feat: add dual-axis evaluation masks"
```

---

### Task 2: Checkpoint validation 재평가 runner

**Files:**
- Create: `lightgcn_clv_dual_checkpoint_diagnostic.py`
- Create: `test_lightgcn_clv_dual_checkpoint_diagnostic.py`

**Interfaces:**
- Consumes: 원본 dual 결과 JSON 경로, 원 데이터, JSON의 `checkpoint_paths`와 `checkpoint_sha256`
- Produces: `run_checkpoint_diagnostic(result_json: str | Path, output_dir: str | Path | None = None) -> pd.DataFrame`

- [ ] **Step 1: 보호 split·hash·학습 금지 실패 테스트 작성**

```python
def test_checkpoint_diagnostic_rejects_non_validation_before_prepare(monkeypatch, tmp_path):
    payload = valid_payload() | {"config": valid_payload()["config"] | {"eval_test": True}}
    path = write_payload(tmp_path, payload)
    called = False
    monkeypatch.setattr(dual.v3, "prepare_data", lambda *_: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(ValueError, match="validation-only"):
        dual.run_checkpoint_diagnostic(path)

def test_checkpoint_diagnostic_checks_every_checkpoint_hash(tmp_path):
    path = write_payload(tmp_path, valid_payload_with_wrong_dual_hash())
    with pytest.raises(RuntimeError, match="checkpoint hash"):
        dual.run_checkpoint_diagnostic(path)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest -q test_lightgcn_clv_dual_checkpoint_diagnostic.py`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: public runner 입력검증·복원 구현**

```python
def run_checkpoint_diagnostic(result_json, output_dir=None):
    payload = load_and_validate_payload(result_json)
    verify_checkpoint_hashes(payload)
    prepared = restore_validation_context(payload)
    model = restore_dual_model(prepared, payload)
    return evaluate_axis_modes(model, prepared, payload, output_dir)
```

`load_and_validate_payload`은 dataset, seed 42, `eval_test=False`, `eval_holdout=False`, 원본 model ID와 체크포인트 키를 확인한다. `restore_validation_context`는 원 실행과 같은 `configure_dual_run`·`_prepare` 경로를 사용하되 저장 encoder checkpoint의 state와 예측값을 재사용하도록 하여 encoder와 adapter를 다시 학습하지 않는다. 데이터·M1 state hash가 원본과 다르면 중단한다.

- [ ] **Step 4: 세 축·세밀한 lambda 절대지표 테스트 작성**

```python
def test_evaluation_emits_all_axis_modes_and_dataset_lambda_grid(monkeypatch, tmp_path):
    result = run_with_small_runtime_fixture(monkeypatch, tmp_path, dataset="dunnhumby")
    assert set(result.axis_mode) == {"n_only", "v_only", "n_plus_v"}
    assert set(result["lambda"]) == {1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0}
    assert set(result.split) == {"val"}
```

- [ ] **Step 5: 축별 평가 구현**

각 axis mode와 lambda에서 기존 `_flat_evaluation(..., per_user=True)`를 호출한다. 각 행에는 정확도·경제·ARP·Coverage·추천상품 수·노출 entropy/effective catalog·상위 노출점유율·value alignment와 `effective_strength`를 저장한다. M1은 한 번만 평가한다.

- [ ] **Step 6: 4분면 paired gain 실패 테스트 작성**

```python
def test_quadrants_use_train_only_q_threshold_and_keep_empty_groups_missing():
    table = quadrant_metrics(
        q_n=np.array([.2, .8, .2, .8]),
        q_v=np.array([.2, .2, .8, .8]),
        valid=np.ones(4, bool),
        model_per_user=per_user([2, 3, 4, 5]),
        baseline_per_user=per_user([1, 1, 1, 1]),
        n_boot=100,
    )
    assert set(table.quadrant) == {"low_low", "activity", "value", "core"}
    assert table.set_index("quadrant").loc["activity", "user_count"] == 1
    assert table.set_index("quadrant").loc["core", "revenue_improved_user_share"] == 1.0
```

- [ ] **Step 7: 4분면·paired bootstrap 구현**

평가 cache의 사용자 순서와 동일한 user index를 확보해 `q_N/q_V`를 정렬한다. 각 4분면에서 metric별 모델−M1 사용자 차이의 평균, 중앙값, 양수 비율, 기존 `paired_bootstrap` CI를 계산한다. 집단 정의와 user count를 모든 행에 기록한다.

- [ ] **Step 8: 동일 실효강도·plateau 비교 구현**

단독축과 N+V의 sampled score 표준편차 비율을 axis mode별로 계산한다. 같은 lambda 표와 N+V 각 점에 가장 가까운 단독축 점 및 선형보간 표를 저장한다. 선택점 최고 경제지표의 99% 이상이면서 정확도 6개 가드레일을 통과하는 연속 lambda 범위를 `descriptive_plateau`로 기록하되 새 성공판정으로 사용하지 않는다.

- [ ] **Step 9: 산출물·round-trip 테스트 작성 및 구현**

```python
def test_diagnostic_persists_auditable_outputs(tmp_path):
    frame = run_with_small_runtime_fixture(...)
    paths = frame.attrs["result_paths"]
    assert set(paths) == {"csv", "delta_csv", "quadrant_csv", "strength_csv", "json"}
    payload = json.loads(Path(paths["json"]).read_text())
    assert payload["training_performed"] is False
    assert payload["source_result_fingerprint"] == "fixture"
    assert payload["checkpoint_sha256"]
```

CSV 네 개와 JSON을 `checkpoint_diagnostics/`에 저장한다. JSON에는 원본 결과 ID, 설정, 축·lambda, source/data/M1/checkpoint hash, 절대행, delta, 4분면, 실효강도, plateau, 해석 한계를 포함한다.

- [ ] **Step 10: runner 테스트 통과 및 커밋**

Run: `pytest -q test_lightgcn_clv_dual_checkpoint_diagnostic.py test_clv_dual_axis_model.py`

Expected: PASS.

```bash
git add lightgcn_clv_dual_checkpoint_diagnostic.py test_lightgcn_clv_dual_checkpoint_diagnostic.py
git commit -m "feat: add dual-axis checkpoint diagnostics"
```

---

### Task 3: 두 데이터셋 순차 Colab과 연구상태

**Files:**
- Create: `clv_dual_checkpoint_diagnostic_colab.ipynb`
- Modify: `test_lightgcn_clv_dual_checkpoint_diagnostic.py`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: Google Drive의 최신 Dunnhumby/H&M 60일 원본 dual JSON
- Produces: 두 데이터셋의 진단표·그래프·저장 경로

- [ ] **Step 1: 노트북 실행경계 실패 테스트 작성**

```python
def test_colab_runs_both_checkpoint_diagnostics_without_training():
    source = notebook_source("clv_dual_checkpoint_diagnostic_colab.ipynb")
    assert "run_checkpoint_diagnostic" in source
    assert "run_experiment" not in source
    assert "train_" not in source
    assert "results_clv_dual_dunnhumby" in source
    assert "results_clv_dual_hm_w60" in source
```

- [ ] **Step 2: 실패 확인**

Run: `pytest -q test_lightgcn_clv_dual_checkpoint_diagnostic.py::test_colab_runs_both_checkpoint_diagnostics_without_training`

Expected: FAIL because notebook does not exist.

- [ ] **Step 3: Colab 생성**

노트북은 Drive mount, 고정 source checkout, 두 원본 결과 JSON 탐색, `run_checkpoint_diagnostic` 순차 실행, 절대지표·4분면·실효강도·plateau 표시 순서로 구성한다. GPU가 없으면 실행 전에 중단하며 별도 고비용 승인 셀은 두지 않는다. 체크포인트 재평가이므로 모델학습보다 훨씬 짧다는 점을 첫 셀에 표시한다.

- [ ] **Step 4: 전체 검증**

Run:

```bash
ruff check clv_dual_axis_model.py lightgcn_clv_dual_checkpoint_diagnostic.py \
  test_clv_dual_axis_model.py test_lightgcn_clv_dual_checkpoint_diagnostic.py
python -m json.tool clv_dual_checkpoint_diagnostic_colab.ipynb >/dev/null
pytest -q
git diff --check
```

Expected: Ruff PASS, notebook JSON valid, full pytest PASS, no whitespace errors.

- [ ] **Step 5: 연구상태 갱신**

`RESEARCH_STATUS.md`에 코드 완료, 검증 개수, 고비용 재학습 없음, 실데이터 checkpoint 진단 미실행을 확정 사실·다음 실행으로 구분해 기록한다.

- [ ] **Step 6: 커밋·푸시**

```bash
git add clv_dual_checkpoint_diagnostic_colab.ipynb \
  test_lightgcn_clv_dual_checkpoint_diagnostic.py
git commit -m "feat: add checkpoint diagnostic Colab"
git push origin feat/clv-conditioned-moe
```
