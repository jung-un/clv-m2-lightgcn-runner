# H&M 2년 M2 4모형 고속·자동재개 실행 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** H&M 전체기간 seed 42 validation에서 M1과 세 M2 모형을 한 번에 실행하고, epoch별 Drive 체크포인트와 heartbeat로 중단 후 자동 재개한다.

**Architecture:** 공통 실행상태 모듈이 원자적 JSON·Torch 저장, 실행 identity 검증, epoch 복원을 담당한다. 기존 M1·CLV encoder·adapter 학습 루프에는 선택적 resume hook만 추가하고, 전용 suite runner가 대용량 배치 사전점검·단계 순서·평가·결과 저장을 조정한다.

**Tech Stack:** Python 3, PyTorch, NumPy, pandas, pytest, Google Colab

## Global Constraints

- H&M 전체기간(`window_days=None`), seed `42`, validation only
- test·holdout 정답 구성 금지
- binary LightGCN, plain BPR, uniform negative sampling 유지
- 네 모형: `m1`, `dual_clv_fixed`, `dual_shuffled_user`, `dual_adapter_only`
- gate=`high`, target `rho=0.2`, validation을 사용한 lambda 재탐색 금지
- batch 후보는 `131072`, `65536`, `32768`; 사전점검 후 하나를 네 모형에 고정
- epoch checkpoint는 model·optimizer·NumPy/PyTorch/CUDA RNG·best state·early-stop 상태를 포함
- 기존 M2/M3/M4 연구축 정의와 `revenue` 해석 규칙을 변경하지 않음

---

### Task 1: 원자적 실행상태와 epoch resume 저장소

**Files:**
- Create: `clv_run_state.py`
- Create: `test_clv_run_state.py`

**Interfaces:**
- Produces: `RunIdentity(stage, model_id, seed, config_hash, source_revision, input_hash)`
- Produces: `ProgressStore(root, identity, heartbeat_interval_sec=60.0)`
- Produces: `ProgressStore.save_epoch(model, optimizer, rng, **epoch_state)`
- Produces: `ProgressStore.restore_epoch(model, optimizer, rng) -> dict | None`
- Produces: `ProgressStore.mark_stage(status: str, **fields) -> None`
- Produces: `ProgressStore.heartbeat(**fields) -> None`

- [ ] **Step 1: 중단·복원과 identity 불일치 실패 테스트를 작성한다.**

```python
def test_epoch_round_trip_restores_optimizer_rng_and_best_state(tmp_path):
    store = ProgressStore(tmp_path, identity("m1"), heartbeat_interval_sec=0)
    store.save_epoch(model, optimizer, rng, epoch=2, best_epoch=1,
                     best_metric=0.4, best_state=clone(model), bad=1,
                     updates=7, samples=70, history=[{"epoch": 2}])
    restored = store.restore_epoch(new_model, new_optimizer, new_rng)
    assert restored["next_epoch"] == 3
    assert restored["bad"] == 1
    assert torch.equal(new_model.weight, model.weight)

def test_checkpoint_identity_mismatch_fails_closed(tmp_path):
    first = ProgressStore(tmp_path, identity("m1"))
    first.save_epoch(model, optimizer, rng, epoch=1, best_epoch=1,
                     best_metric=0.4, best_state=clone(model), bad=0,
                     updates=1, samples=10, history=[{"epoch": 1}])
    with pytest.raises(RuntimeError, match="identity"):
        ProgressStore(tmp_path, identity("dual_clv_fixed")).restore_epoch(
            new_model, new_optimizer, new_rng
        )
```

- [ ] **Step 2: 테스트를 실행해 모듈 부재로 실패하는지 확인한다.**

Run: `pytest -q test_clv_run_state.py`
Expected: FAIL with `ModuleNotFoundError: clv_run_state`

- [ ] **Step 3: 원자적 저장과 복원을 구현한다.**

```python
@dataclass(frozen=True)
class RunIdentity:
    stage: str
    model_id: str
    seed: int
    config_hash: str
    source_revision: str
    input_hash: str

class ProgressStore:
    def save_epoch(self, model, optimizer, rng, **state):
        payload = {
            "identity": asdict(self.identity),
            "model_state": clone_state(model),
            "optimizer_state": optimizer.state_dict(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            **state,
        }
        atomic_torch_save(payload, self.latest_checkpoint)
```

- [ ] **Step 4: checkpoint round-trip, atomic JSON, heartbeat throttling 테스트를 통과시킨다.**

Run: `pytest -q test_clv_run_state.py`
Expected: PASS

- [ ] **Step 5: Task 1을 커밋한다.**

```bash
git add clv_run_state.py test_clv_run_state.py
git commit -m "feat: add atomic resumable run state"
```

---

### Task 2: 기존 세 학습 루프에 선택적 resume hook 추가

**Files:**
- Modify: `lightgcn_clv_v3.py`
- Modify: `lightgcn_clv_moe.py`
- Modify: `clv_core_features.py`
- Create: `test_clv_resumable_training.py`

**Interfaces:**
- Consumes: `ProgressStore`
- Produces: 기존 `train_phase`에 `progress_store: ProgressStore | None` 선택 인자
- Produces: 기존 `get_or_train`에 `progress_store: ProgressStore | None` 선택 인자
- Produces: 기존 `train_moe`에 `progress_store: ProgressStore | None` 선택 인자
- Produces: 기존 `train_clv_core_encoder`에 `progress_stores: dict[str, ProgressStore] | None` 선택 인자

- [ ] **Step 1: M1이 epoch 2에서 중단된 뒤 epoch 3만 추가하는 실패 테스트를 작성한다.**

```python
def test_train_phase_resumes_at_next_epoch(monkeypatch, tmp_path):
    first = run_tiny_m1(tmp_path, epochs=2)
    second = run_tiny_m1(tmp_path, epochs=3)
    assert first["epochs_run"] == 2
    assert second["resumed_from_epoch"] == 2
    assert second["epochs_run"] == 3
    assert second["new_epochs_run"] == 1
```

- [ ] **Step 2: encoder와 adapter도 optimizer·RNG·best-state를 복원하는 실패 테스트를 작성한다.**

```python
@pytest.mark.parametrize("trainer", ["encoder", "adapter"])
def test_training_resume_matches_uninterrupted(trainer, tmp_path):
    uninterrupted = run_tiny(trainer, epochs=3, root=tmp_path / "full")
    run_tiny(trainer, epochs=2, root=tmp_path / "resume")
    resumed = run_tiny(trainer, epochs=3, root=tmp_path / "resume")
    assert_state_dict_equal(uninterrupted.state_dict(), resumed.state_dict())
```

- [ ] **Step 3: 실패 테스트를 실행한다.**

Run: `pytest -q test_clv_resumable_training.py`
Expected: FAIL because the training functions do not accept `progress_store`

- [ ] **Step 4: M1 학습 루프에 epoch 복원·저장·batch heartbeat를 연결한다.**

```python
resume = progress_store.restore_epoch(model, opt, rng) if progress_store else None
start_epoch = 1 if resume is None else resume["next_epoch"]
for ep in range(start_epoch, cfg["EPOCHS"] + 1):
    order = rng.permutation(n_train)
    for b in range(n_batch):
        idx = order[b * batch_size:(b + 1) * batch_size]
        loss = train_batch(idx)
        progress_store.heartbeat(epoch=ep, batch=b + 1, batches=n_batch)
    progress_store.save_epoch(
        model, opt, rng, epoch=ep, best_epoch=best_ep,
        best_metric=best, best_state=best_state, bad=bad,
        updates=updates, samples=samples, history=history,
    )
```

- [ ] **Step 5: encoder selection/final-fit와 adapter 루프에 동일한 hook을 연결한다.**

Encoder는 `encoder_select`와 `encoder_final`을 별도 stage checkpoint로 저장한다. Adapter는 각 model id별 store를 사용하며, 완료 시 best state를 최종 checkpoint에 저장한다.

- [ ] **Step 6: focused tests와 기존 회귀 테스트를 통과시킨다.**

Run: `pytest -q test_clv_run_state.py test_clv_resumable_training.py test_lightgcn_clv_v3.py test_lightgcn_clv_dual.py`
Expected: PASS

- [ ] **Step 7: Task 2를 커밋한다.**

```bash
git add lightgcn_clv_v3.py lightgcn_clv_moe.py clv_core_features.py test_clv_resumable_training.py
git commit -m "feat: resume M1 encoder and adapter epochs"
```

---

### Task 3: H&M 2년 4모형 suite runner와 대용량 배치 사전점검

**Files:**
- Create: `lightgcn_clv_dual_hm2y_suite.py`
- Create: `test_lightgcn_clv_dual_hm2y_suite.py`
- Modify: `lightgcn_clv_dual.py`

**Interfaces:**
- Produces: `configure_hm2y_suite(**overrides) -> MoEConfig`
- Produces: `choose_batch_size(candidates, probe) -> int`
- Produces: `run_hm2y_suite(cfg=None) -> pandas.DataFrame`
- Produces: `read_progress(out_dir) -> dict`

- [ ] **Step 1: 프로토콜·4모형·batch fallback·완료 stage skip 실패 테스트를 작성한다.**

```python
def test_suite_protocol_and_models(tmp_path):
    cfg = configure_hm2y_suite(out_dir=tmp_path)
    assert cfg.dataset == "hm" and cfg.window_days is None
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is cfg.eval_holdout is False
    assert MODELS == ("m1", "dual_clv_fixed", "dual_shuffled_user", "dual_adapter_only")

def test_choose_batch_size_uses_first_passing_candidate():
    assert choose_batch_size((131072, 65536, 32768), lambda n: n <= 65536) == 65536

def test_completed_stage_is_not_retrained(monkeypatch, completed_manifest):
    run_hm2y_suite(completed_manifest.cfg)
    assert completed_manifest.train_calls == []
```

- [ ] **Step 2: 실패 테스트를 실행한다.**

Run: `pytest -q test_lightgcn_clv_dual_hm2y_suite.py`
Expected: FAIL with missing module/functions

- [ ] **Step 3: H&M full-period 전용 설정 검증과 M1 throwaway 1-step batch probe를 구현한다.**

Probe는 모델·optimizer 상태를 저장하지 않는 새 M1 인스턴스에서 forward/backward 한 번만 수행한다. CUDA OOM만 다음 후보로 낮추며 다른 예외는 즉시 전달한다. 선택한 batch는 manifest에 저장하고 재개 시 바꾸지 않는다.

- [ ] **Step 4: 단계 오케스트레이션과 네 모형 공통 학습을 구현한다.**

```python
MODELS = ("m1", dual.PRIMARY_MODEL, *dual.CONTROLS)
for model_id in MODELS:
    if manifest.completed_with_matching_hash(model_id):
        continue
    manifest.start(model_id)
    run_model_with_store(model_id, stores[model_id])
    manifest.complete(model_id, checkpoint_sha256)
```

- [ ] **Step 5: 각 M2 arm을 `high`, `rho=0.2`의 자체 실효강도 정규화점에서 평가한다.**

```python
ratio = diagnostics["gate_shape_diagnostics"]["high"]["effective_total_ratio"]
point = operating_point(ratio)
model.set_gate_shape("high")
flat, per_user = moe._flat_evaluation(
    model, point["lambda"], prepared["cache"], prepared["meta"],
    prepared["data"], prepared["base_cfg"], per_user=True,
)
```

- [ ] **Step 6: 절대지표·paired delta·대조군 판정·provenance 저장을 구현한다.**

`dual_clv_fixed`는 M1 정확도 6개 99% guardrail, M1보다 높은 `revenue@10`, 두 대조군보다 높은 `revenue@10`을 모두 통과해야 성공이다. guardrail은 점수를 조작하지 않고 사후 판정만 한다.

- [ ] **Step 7: suite와 기존 runner 회귀 테스트를 통과시킨다.**

Run: `pytest -q test_lightgcn_clv_dual_hm2y_suite.py test_lightgcn_clv_dual_hm2y_seed42.py test_lightgcn_clv_dual.py`
Expected: PASS

- [ ] **Step 8: Task 3을 커밋한다.**

```bash
git add lightgcn_clv_dual_hm2y_suite.py lightgcn_clv_dual.py test_lightgcn_clv_dual_hm2y_suite.py
git commit -m "feat: add resumable H&M 2y M2 suite"
```

---

### Task 4: 즉시 실행 가능한 Colab과 최종 검증

**Files:**
- Create: `clv_dual_hm2y_suite_colab.ipynb`
- Modify: `test_lightgcn_clv_dual_hm2y_suite.py`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

**Interfaces:**
- Consumes: `configure_hm2y_suite`, `run_hm2y_suite`, `read_progress`
- Produces: GitHub SHA에 고정된 Colab notebook

- [ ] **Step 1: notebook 구조 실패 테스트를 작성한다.**

```python
def test_colab_is_pinned_and_has_run_and_status_cells():
    source = notebook_source("clv_dual_hm2y_suite_colab.ipynb")
    assert re.search(r"REVIEWED_SHA = '[0-9a-f]{40}'", source)
    assert "run_hm2y_suite(cfg)" in source
    assert "read_progress(cfg.out_dir)" in source
    assert "eval_test=True" not in source
    assert "eval_holdout=True" not in source
```

- [ ] **Step 2: clone·SHA 확인, GPU 확인, 설정 요약, 실행, 독립 상태확인, 결과표 셀을 만든다.**

노트북 실행 셀은 네 모형을 한 번에 실행한다. 별도의 상태 셀은 새 런타임에서도 Drive의 `progress.json`만 읽어 현재 stage·epoch·마지막 heartbeat·예상 잔여시간을 표시한다.

- [ ] **Step 3: 전체 관련 테스트·Ruff·notebook JSON을 검증한다.**

```bash
pytest -q test_clv_run_state.py test_clv_resumable_training.py test_lightgcn_clv_dual_hm2y_suite.py test_lightgcn_clv_dual.py
ruff check clv_run_state.py lightgcn_clv_v3.py lightgcn_clv_moe.py clv_core_features.py lightgcn_clv_dual.py lightgcn_clv_dual_hm2y_suite.py
python -m json.tool clv_dual_hm2y_suite_colab.ipynb >/dev/null
```

- [ ] **Step 4: 연구상태에 구현 commit·검증·고비용 미실행을 기록한다.**

실제 H&M 2년 학습은 사용자가 Colab에서 시작하며, 코드 구현·테스트 중에는 test·holdout과 고비용 학습을 실행하지 않는다.

- [ ] **Step 5: 코드 commit 후 notebook SHA를 최종 commit에 핀하고 push한다.**

```bash
git add clv_dual_hm2y_suite_colab.ipynb test_lightgcn_clv_dual_hm2y_suite.py
git commit -m "feat: add H&M 2y resumable suite Colab"
git push origin feat/clv-conditioned-moe
```
