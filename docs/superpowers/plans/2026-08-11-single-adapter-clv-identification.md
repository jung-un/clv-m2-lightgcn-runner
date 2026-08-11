# 단일 어댑터 CLV 효과 식별 구현계획

> **에이전트 작업 필수사항:** 이 계획을 작업별로 구현할 때는 `superpowers:subagent-driven-development`(권장) 또는 `superpowers:executing-plans` 스킬을 사용한다. 진행 상태는 체크박스(`- [ ]`)로 관리한다.

**목표:** 유망한 단일 어댑터의 경제지표 개선이 어댑터 용량, 아이템 특성 또는 추가 미세조정이 아니라 사용자별 CLV 관련 행동표현에 의존하는지 검증하는 검증자료 전용 M2 실행기를 만든다.

**구조:** 기존 binary-LightGCN/plain-BPR 파이프라인은 유지하고 `CLVMixtureEmbeddingModel`에 파라미터 수를 맞춘 단일 어댑터 입력 변형 다섯 개를 추가한다. 검토가 끝난 기존 MoE 실행기의 동작을 바꾸지 않도록 `lightgcn_clv_single.py`를 별도 실행기로 만든다. 새 실행기는 `single_full`을 먼저 평가하고 성공한 경우에만 대조군을 실행한다. 기존 Dunnhumby 체크포인트는 출처·설정 검증과 지표 재현 검사를 모두 통과할 때만 재사용한다.

**기술 구성:** Python 3.11, PyTorch, NumPy, pandas, pytest, ruff, Jupyter/Colab JSON.

## 전체 제약조건

- 연구 위치는 M2로 고정한다. 그래프는 `binary`, 손실은 가중치 없는 `plain BPR`, 음성 샘플링은 `uniform`을 유지한다.
- 선별실험은 seed `(42,)`와 검증자료로 고정하며 시험자료·추가 보류자료의 정답은 만들지 않는다.
- 사용자 입력은 현재처럼 학습구간에서만 계산한 51차원 CLV 관련 행동표현을 유지한다.
- 아이템 입력은 현재처럼 학습구간에서만 계산한 수치특성 6개와 카테고리 임베딩을 유지한다.
- 다섯 변형의 입력 차원, 파라미터 형태, 초기화 규칙, 최대 에포크, 조기 종료 인내횟수, 학습률은 동일하게 유지한다.
- `single_zero_user`와 `single_base_only`는 값을 0으로 만들되 `has_profile`을 유지한다. `single_zero_item`과 `single_base_only`도 아이템 특성만 0으로 만들고 `valid_item`은 유지한다.
- λ 후보군은 정확히 `(0.0, 0.1, 0.25, 0.5, 1.0, 2.0)`으로 고정한다.
- 양의 λ는 K=`10,20,50`의 Recall/NDCG가 각각 외부 M1 대비 상대하락 1% 이내이고 `revenue@10`이 외부 M1보다 실제로 높을 때만 후보로 인정한다.
- `single_full`의 `revenue@10`은 각자 최선 λ를 선택한 `single_zero_user`, `single_shuffled_user`, `single_base_only`보다 높아야 한다. `single_zero_item`은 메커니즘 분석용이며 필수 성공조건에는 넣지 않는다.
- 모든 λ 절대곡선, 대응 차이, 노출지표, 특징 스키마, 해시, 체크포인트, 학습횟수, 최종 선별 판정을 저장한다.
- 기존 MoE 동작과 과거 결과 지문은 변경하지 않는다.
- 구현 검증에서는 H&M이나 Dunnhumby의 고비용 학습을 실행하지 않는다.

---

## 파일 구성

- `clv_moe_model.py` 수정: `single_adapter`를 `single_full`의 하위호환 별칭으로 유지하면서 파라미터 수가 동일한 단일 어댑터 입력 변형을 정의한다.
- `test_clv_moe_model.py` 수정: zero/shuffle 동작, mask 보존, 파라미터 동일성, 별칭 동일성, lambda=0 동일성을 실행 테스트로 검증한다.
- `lightgcn_clv_single.py` 생성: 설정, 사전점검, 검증자료 실행, 선택, 재사용 검증, 저장, 안전한 명령행 실행을 담당한다.
- `test_lightgcn_clv_single.py` 생성: 정책, 출처, 실행순서, 결과 스키마, 보호자료 차단을 검증한다.
- `clv_single_adapter_colab.ipynb` 생성: 데이터셋 사전설정, 선택적 Dunnhumby 결과 재사용, 사전점검, 고비용 승인 장치, 최종 판정 출력을 제공한다.
- Git 저장소 밖의 `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md` 수정: 구현 상태를 기록하고 고비용 실행 결과와 명확히 구분한다.

### 작업 1: 동일 용량 단일 어댑터 입력 변형 추가

**파일:**
- 수정: `clv_moe_model.py:38-180`
- 테스트: `test_clv_moe_model.py`

**인터페이스:**
- 입력: `UserProfileArtifact`, `ItemProfileArtifact`, 기존 `CLVMixtureEmbeddingModel` 생성자.
- 출력: `SINGLE_VARIANTS`, `canonical_single_variant(control: str) -> str | None`, 새 control `single_full`, `single_zero_user`, `single_shuffled_user`, `single_zero_item`, `single_base_only`. 기존 `single_adapter`는 유지한다.

- [ ] **1단계: 변형별 buffer와 mask에 대한 실패 테스트 작성**

다음 helper와 테스트를 `test_clv_moe_model.py`에 추가한다.

```python
def _single(control, seed=42):
    return _model(control=control, seed=seed)


def test_single_full_is_exact_legacy_single_adapter_alias():
    legacy = _single("single_adapter")
    full = _single("single_full")
    assert legacy.single_variant == full.single_variant == "single_full"
    torch.testing.assert_close(legacy.routed_profile, full.routed_profile)
    torch.testing.assert_close(legacy.item_numeric, full.item_numeric)
    torch.testing.assert_close(
        legacy.score_all(torch.arange(4), 1.0),
        full.score_all(torch.arange(4), 1.0),
    )


def test_single_zero_user_preserves_mask_and_zeros_only_user_profile():
    full = _single("single_full")
    zero = _single("single_zero_user")
    torch.testing.assert_close(zero.routed_profile, torch.zeros_like(zero.routed_profile))
    torch.testing.assert_close(zero.item_numeric, full.item_numeric)
    assert torch.equal(zero.has_profile, full.has_profile)


def test_single_zero_item_preserves_mask_and_zeros_item_side_features():
    full = _single("single_full")
    zero = _single("single_zero_item")
    torch.testing.assert_close(zero.item_numeric, torch.zeros_like(zero.item_numeric))
    assert torch.equal(zero.item_category_ids, torch.zeros_like(zero.item_category_ids))
    assert torch.equal(zero.valid_item, full.valid_item)
    torch.testing.assert_close(zero.routed_profile, full.routed_profile)


def test_single_base_only_zeros_both_added_inputs_without_disabling_residual():
    model = _single("single_base_only")
    assert torch.count_nonzero(model.routed_profile) == 0
    assert torch.count_nonzero(model.item_numeric) == 0
    assert torch.count_nonzero(model.item_category_ids) == 0
    assert model.has_profile.all() and model.valid_item.all()
    assert not torch.equal(
        model.score_all(torch.arange(4), 1.0),
        model.base_score_all(torch.arange(4)),
    )
```

- [ ] **2단계: 집중 테스트를 실행해 RED 확인**

실행:

```bash
pytest -q \
  test_clv_moe_model.py::test_single_full_is_exact_legacy_single_adapter_alias \
  test_clv_moe_model.py::test_single_zero_user_preserves_mask_and_zeros_only_user_profile \
  test_clv_moe_model.py::test_single_zero_item_preserves_mask_and_zeros_item_side_features \
  test_clv_moe_model.py::test_single_base_only_zeros_both_added_inputs_without_disabling_residual
```

예상 결과: 새 control과 `single_variant`가 없으므로 FAIL.

- [ ] **3단계: 표준 변형 처리 구현**

`clv_moe_model.py`에 다음을 추가한다.

```python
SINGLE_VARIANTS = frozenset(
    {
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_zero_item",
        "single_base_only",
    }
)


def canonical_single_variant(control: str) -> str | None:
    if control == "single_adapter":
        return "single_full"
    return control if control in SINGLE_VARIANTS else None
```

`CONTROLS`를 갱신하고 tensor 차원을 바꾸지 않은 채 변형을 적용한다.

```python
self.single_variant = canonical_single_variant(control)
is_single = self.single_variant is not None
self.expert_count = 1 if is_single else requested_expert_count

routed = values.clone()
if control == "shuffled_clv" or self.single_variant == "single_shuffled_user":
    routed = _permute_valid_rows(routed, valid_user, seed)
if self.single_variant in {"single_zero_user", "single_base_only"}:
    routed.zero_()
if self.single_variant in {"single_zero_item", "single_base_only"}:
    item_numeric.zero_()
    item_categories.zero_()
```

용량 정합, gate 생성, `routing_weights`에서 `is_single`을 사용한다. `has_profile`과 `valid_item`은 변경하지 않는다.

- [ ] **4단계: 결정적 shuffle 및 용량·state 형태 테스트 추가**

```python
def test_single_shuffled_user_is_seeded_permutation_of_valid_profiles():
    full = _single("single_full")
    a = _single("single_shuffled_user", seed=42)
    b = _single("single_shuffled_user", seed=42)
    torch.testing.assert_close(a.routed_profile, b.routed_profile)
    assert not torch.equal(a.routed_profile, full.routed_profile)
    assert sorted(a.routed_profile[:, 0].tolist()) == sorted(full.routed_profile[:, 0].tolist())


def test_all_single_variants_have_identical_parameter_names_and_shapes():
    controls = ["single_full", "single_zero_user", "single_shuffled_user",
                "single_zero_item", "single_base_only"]
    signatures = []
    for control in controls:
        model = _single(control)
        signatures.append([(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()])
        assert model.expert_count == 1
        torch.testing.assert_close(
            model.score_all(torch.arange(4), 0.0),
            model.base_score_all(torch.arange(4)),
            rtol=0,
            atol=0,
        )
    assert signatures.count(signatures[0]) == len(signatures)
```

- [ ] **5단계: 모델 테스트와 lint 실행**

실행:

```bash
pytest -q test_clv_moe_model.py
ruff check clv_moe_model.py test_clv_moe_model.py
```

예상 결과: 모든 테스트 PASS, ruff 오류 없음.

- [ ] **6단계: 작업 1 커밋**

```bash
git add clv_moe_model.py test_clv_moe_model.py
git commit -m "feat: add matched single-adapter variants"
```

### 작업 2: 검증자료 전용 단일 어댑터 정책과 판정 로직 정의

**파일:**
- 생성: `lightgcn_clv_single.py`
- 생성: `test_lightgcn_clv_single.py`

**인터페이스:**
- 입력: `lightgcn_clv_moe.MoEConfig`, `configure_moe_run`, `validate_moe_config`, `select_lambda`.
- 출력: `configure_single_run(dataset: str, **overrides) -> MoEConfig`, `validate_single_config(cfg: MoEConfig) -> MoEConfig`, `preflight_summary(cfg: MoEConfig) -> dict`, `single_screening_decision(rows, selected, selection_success) -> dict`.

- [ ] **1단계: 정책 실패 테스트 작성**

다음 내용으로 `test_lightgcn_clv_single.py`를 생성한다.

```python
import dataclasses

import pytest


def test_default_single_screening_is_seed42_validation_only():
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby")
    summary = single.preflight_summary(cfg)
    assert cfg.seed_list == (42,)
    assert cfg.eval_test is False and cfg.eval_holdout is False
    assert summary["primary_model_id"] == "single_full"
    assert summary["required_controls"] == [
        "single_zero_user", "single_shuffled_user", "single_base_only"
    ]
    assert summary["mechanism_controls"] == ["single_zero_item"]
    assert summary["graph_mode"] == "binary"
    assert summary["loss_mode"] == "plain"


@pytest.mark.parametrize("field", ["eval_test", "eval_holdout"])
def test_direct_dataclass_cannot_open_protected_splits(field):
    import lightgcn_clv_single as single
    import lightgcn_clv_moe as moe

    cfg = dataclasses.replace(moe.MoEConfig(), **{field: True})
    with pytest.raises(ValueError, match="screening-only"):
        single.validate_single_config(cfg)


def test_single_screening_decision_requires_full_to_beat_required_controls():
    import lightgcn_clv_single as single

    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
        "pref_continue": 0.0,
    }
    rows = [
        {"seed": 42, "split": "val", "model_id": model_id,
         "lambda": selected[model_id], "revenue@10": revenue}
        for model_id, revenue in {
            "single_full": 1.10,
            "single_zero_user": 1.04,
            "single_shuffled_user": 1.03,
            "single_zero_item": 1.12,
            "single_base_only": 1.02,
            "pref_continue": 1.01,
        }.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is True
    assert decision["mechanism_comparison"]["single_zero_item"] == 1.12
    rows[1]["revenue@10"] = 1.11
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False
    assert decision["failed_controls"] == ["single_zero_user"]
```

- [ ] **2단계: 테스트를 실행해 RED 확인**

실행: `pytest -q test_lightgcn_clv_single.py`

예상 결과: `lightgcn_clv_single.py`가 없으므로 import 실패.

- [ ] **3단계: 설정 및 판정 모듈 구현**

다음 상수와 함수로 `lightgcn_clv_single.py`를 생성한다.

```python
PRIMARY_MODEL_ID = "single_full"
REQUIRED_CONTROLS = (
    "single_zero_user",
    "single_shuffled_user",
    "single_base_only",
)
MECHANISM_CONTROLS = ("single_zero_item",)
ALL_SINGLE_MODELS = (PRIMARY_MODEL_ID, *REQUIRED_CONTROLS, *MECHANISM_CONTROLS)
CODE_VERSION = "clv-single-identification-v1.0"


def configure_single_run(dataset: str, **overrides) -> moe.MoEConfig:
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "lambda_eval": (0.0, 0.1, 0.25, 0.5, 1.0, 2.0),
        "run_controls_after_success": True,
        "out_dir": f"{v3.default_out_dir(dataset)}_clv_single",
    }
    return validate_single_config(moe.configure_moe_run(dataset, **(defaults | overrides)))


def validate_single_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    cfg = moe.validate_moe_config(cfg)
    if cfg.seed_list != (42,):
        raise ValueError("single-adapter screening-only runner requires seed 42")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("single-adapter screening-only runner cannot open test/holdout")
    if cfg.lambda_eval != (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        raise ValueError("single-adapter lambda grid is frozen by the approved design")
    return cfg
```

`single_screening_decision`은 모형별 독립 선택행을 읽는다. `single_full` 자체가 선택조건을 통과하지 못하면 실패 처리하고, 최종 성공 여부는 `REQUIRED_CONTROLS`와의 비교로만 판정한다. `single_zero_item`은 `failed_controls`에 넣지 않고 `mechanism_comparison`에 저장한다.

- [ ] **4단계: 직접 실행 차단과 엄격한 초과조건 테스트 추가**

```python
def test_decision_tie_with_required_control_is_failure():
    import lightgcn_clv_single as single
    selected = {
        "single_full": 1.0,
        "single_zero_user": 1.0,
        "single_shuffled_user": 0.5,
        "single_zero_item": 1.0,
        "single_base_only": 0.5,
    }
    values = {
        "single_full": 1.10,
        "single_zero_user": 1.10,
        "single_shuffled_user": 1.03,
        "single_zero_item": 1.12,
        "single_base_only": 1.02,
    }
    rows = [
        {"seed": 42, "split": "val", "model_id": model_id,
         "lambda": selected[model_id], "revenue@10": revenue}
        for model_id, revenue in values.items()
    ]
    success = {model_id: True for model_id in selected}
    decision = single.single_screening_decision(rows, selected, success)
    assert decision["success"] is False


def test_validate_rejects_changed_lambda_grid():
    import dataclasses
    import pytest
    import lightgcn_clv_single as single
    cfg = single.configure_single_run("dunnhumby")
    with pytest.raises(ValueError, match="lambda grid"):
        single.validate_single_config(dataclasses.replace(cfg, lambda_eval=(0.0, 1.0)))
```

- [ ] **5단계: 정책 테스트와 lint 실행**

실행:

```bash
pytest -q test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
```

예상 결과: 모든 정책 테스트 PASS, ruff 오류 없음.

- [ ] **6단계: 작업 2 커밋**

```bash
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: define single-adapter screening policy"
```

### 작업 3: 저장된 Dunnhumby 전체 모형의 안전한 재사용 추가

**파일:**
- 수정: `lightgcn_clv_single.py`
- 수정: `test_lightgcn_clv_single.py`

**인터페이스:**
- 입력: 저장된 MoE 결과 JSON, `single_adapter` 체크포인트, 현재 입력 명세, 외부 M1 상태, 현재 특징과 검증자료 캐시.
- 출력: `ReusableSingleFull` dataclass와 `load_reusable_single_full(...) -> ReusableSingleFull`. 하나라도 불일치하면 재사용 전에 `RuntimeError`를 발생시킨다.

- [ ] **1단계: 구체적인 재사용 fixture와 출처 검증 실패 테스트 작성**

다음 가져오기 구문과 테스트용 픽스처를 `test_lightgcn_clv_single.py`에 추가한다. 인코더 값 해시 검증에는 실제 PyTorch 데이터를 사용하고, 고비용 모형 재구성과 평가기만 가벼운 대체 구현으로 처리한다.

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@dataclass
class ReuseFixture:
    result_json: Path
    current_manifest: dict
    base_hash: str
    cfg: object
    base_cfg: dict
    context: dict
    data: dict
    rows_by_lambda: dict


def _reuse_metric_row(lam):
    row = {
        "seed": 42,
        "model_id": "single_adapter",
        "split": "val",
        "lambda": lam,
        "role": "control",
        "revenue@10": 1.0 + 0.01 * lam,
        "arp@10": 0.2,
    }
    for k in (10, 20, 50):
        row[f"recall@{k}"] = 0.1
        row[f"ndcg@{k}"] = 0.1
        row[f"n_distinct@{k}"] = 3
        row[f"exposure_entropy@{k}"] = 1.0
        row[f"eff_catalog@{k}"] = 2.7
        row[f"top10_share@{k}"] = 0.5
        row[f"top100_share@{k}"] = 1.0
    return row


@pytest.fixture
def reuse_fixture(tmp_path, monkeypatch):
    import lightgcn_clv_moe as moe
    import lightgcn_clv_single as single

    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    manifest = {
        "transactions": {"path": "/tx", "bytes": 2, "sha256": "aa"},
        "item_metadata": {"path": "/item", "bytes": 2, "sha256": "bb"},
    }
    ev_all = np.array([1.0, 2.0], dtype=np.float32)
    checkpoint = tmp_path / "single_adapter.pt"
    torch.save({"ev_all": ev_all}, checkpoint)
    rows = {float(lam): _reuse_metric_row(float(lam)) for lam in cfg.lambda_eval}
    payload = {
        "source_revision": "legacy-revision",
        "input_manifest": manifest,
        "config": asdict(cfg),
        "baseline_state_hashes": {"42": "base-state"},
        "feature_schema": {
            "user": ["u0"],
            "item_numeric": ["i0"],
        },
        "checkpoint_paths": {"single_adapter_s42": str(checkpoint)},
        "absolute_rows": list(rows.values()),
        "training": {"single_adapter_s42": {"base_updates_at_best": 3}},
        "moe_diagnostics": {"single_adapter_s42": {"parameter_match_ratio": 1.0}},
    }
    result_json = tmp_path / "legacy.json"
    result_json.write_text(json.dumps(payload), encoding="utf-8")
    context = {
        "artifact": SimpleNamespace(ev_all=ev_all),
        "user_profile": SimpleNamespace(feature_names=("u0",)),
        "item_profile": SimpleNamespace(numeric_names=("i0",)),
        "caches": {"val": object()},
    }
    monkeypatch.setattr(moe, "load_moe_checkpoint", lambda *args, **kwargs: object())

    def fake_flat(model, lam, *args, **kwargs):
        row = rows[float(lam)]
        metrics = {
            key: value for key, value in row.items()
            if key not in {"seed", "model_id", "split", "lambda", "role"}
        }
        return metrics, None

    monkeypatch.setattr(moe, "_flat_evaluation", fake_flat)
    return ReuseFixture(
        result_json=result_json,
        current_manifest=manifest,
        base_hash="base-state",
        cfg=cfg,
        base_cfg={"K_LIST": [10, 20, 50]},
        context=context,
        data={"n_items": 2},
        rows_by_lambda=rows,
    )
```

이어서 다음 테스트를 추가한다.

```python
def test_reuse_rejects_input_manifest_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    fixture = reuse_fixture
    changed = fixture.current_manifest | {
        "transactions": {"path": "/x", "bytes": 1, "sha256": "changed"}
    }
    with pytest.raises(RuntimeError, match="input manifest"):
        single.load_reusable_single_full(
            fixture.result_json,
            current_manifest=changed,
            baseline_state_hash=fixture.base_hash,
            cfg=fixture.cfg,
            base_cfg=fixture.base_cfg,
            context=fixture.context,
            data=fixture.data,
        )


def test_reuse_rejects_m1_state_or_feature_schema_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    with pytest.raises(RuntimeError, match="M1 state"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash="wrong",
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )
```

- [ ] **2단계: 재사용 테스트를 실행해 RED 확인**

실행: `pytest -q test_lightgcn_clv_single.py -k reuse`

예상 결과: 재사용 API가 없으므로 FAIL.

- [ ] **3단계: 명시적 호환성 key와 배열 hash 구현**

```python
REUSE_CONFIG_KEYS = (
    "dataset", "seed_list", "input_days", "target_days", "anchor_offsets",
    "encoder_epochs", "encoder_patience", "encoder_batch_size", "encoder_lr",
    "expert_count", "expert_hidden_dim", "expert_dim", "category_dim",
    "frozen_epochs", "max_epochs", "patience", "adapter_lr", "base_lr",
    "lambda_train", "lambda_eval", "accuracy_tolerance",
)


def array_sha256(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    payload = array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(payload).hexdigest()
```

다음을 정의한다.

```python
@dataclass(frozen=True)
class ReusableSingleFull:
    model: CLVMixtureEmbeddingModel
    rows: tuple[dict, ...]
    training: dict
    diagnostics: dict
    result_json_sha256: str
    legacy_source_revision: str
    legacy_checkpoint: str
```

`load_reusable_single_full`은 다음 순서로 검증한다. JSON 구조, 현재 입력 명세의 완전 일치, seed 42 M1 상태 해시, 모든 `REUSE_CONFIG_KEYS`, 사용자·아이템 특징명, 체크포인트 존재 여부, 체크포인트와 현재 인코더의 `ev_all` 해시, `load_moe_checkpoint(..., control="single_adapter")` 성공 여부, 모든 λ의 검증자료 재평가 지표. 재평가 수치지표는 JSON의 `single_adapter` 행과 `np.isclose(rtol=0, atol=5e-8)`로 비교한다.

- [ ] **4단계: 정상 round-trip과 지표 변조 테스트 추가**

```python
def test_reuse_accepts_exact_legacy_full_and_relabels_rows(reuse_fixture):
    import lightgcn_clv_single as single
    reused = single.load_reusable_single_full(
        reuse_fixture.result_json,
        current_manifest=reuse_fixture.current_manifest,
        baseline_state_hash=reuse_fixture.base_hash,
        cfg=reuse_fixture.cfg,
        base_cfg=reuse_fixture.base_cfg,
        context=reuse_fixture.context,
        data=reuse_fixture.data,
    )
    assert {row["model_id"] for row in reused.rows} == {"single_full"}
    assert tuple(row["lambda"] for row in reused.rows) == reuse_fixture.cfg.lambda_eval
    assert reused.result_json_sha256


def test_reuse_rejects_metric_round_trip_mismatch(reuse_fixture):
    import lightgcn_clv_single as single
    payload = json.loads(reuse_fixture.result_json.read_text())
    row = next(row for row in payload["absolute_rows"] if row["model_id"] == "single_adapter")
    row["revenue@10"] += 0.01
    reuse_fixture.result_json.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="metric round-trip"):
        single.load_reusable_single_full(
            reuse_fixture.result_json,
            current_manifest=reuse_fixture.current_manifest,
            baseline_state_hash=reuse_fixture.base_hash,
            cfg=reuse_fixture.cfg,
            base_cfg=reuse_fixture.base_cfg,
            context=reuse_fixture.context,
            data=reuse_fixture.data,
        )
```

- [ ] **5단계: 재사용 테스트와 전체 신규 테스트 실행**

실행:

```bash
pytest -q test_lightgcn_clv_single.py -k reuse
pytest -q test_clv_moe_model.py test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
```

예상 결과: 모두 PASS.

- [ ] **6단계: 작업 3 커밋**

```bash
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: validate reusable single-adapter results"
```

### 작업 4: 조건부 validation 실행과 결과 저장 구현

**파일:**
- 수정: `lightgcn_clv_single.py`
- 수정: `test_lightgcn_clv_single.py`

**인터페이스:**
- 입력: `configure_single_run`, 기존 특징·encoder·M1 helper, 모형 변형, 선택적 기존 전체모형 결과.
- 출력: `frame.attrs["screening_decision"]`과 세 결과파일을 만드는 `run_experiment(cfg: MoEConfig | None = None, *, reuse_full_result_json: str | Path | None = None) -> pd.DataFrame`.
- 내부 타입 경계: `PreparedSingleContext`, `VariantRun`, `_prepare_validation_context(cfg) -> PreparedSingleContext`, `_train_evaluate_variant(prepared, cfg, model_id) -> VariantRun`, `_select_models(rows, baseline, model_ids) -> tuple[dict, dict, dict]`, `_persist_result(...) -> pd.DataFrame`.

- [ ] **1단계: 학습을 stub 처리한 실행순서 실패 테스트 작성**

실행순서 테스트 앞에 다음 helper를 정의한다. 새 실행기의 타입 경계를 patch하므로 데이터셋을 읽거나 모형을 학습하지 않는다.

```python
def _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue):
    import lightgcn_clv_single as single

    calls = {"controls": []}
    baseline = _reuse_metric_row(0.0) | {
        "model_id": "m1", "role": "baseline", "revenue@10": 1.0
    }
    prepared = SimpleNamespace(
        out_dir=tmp_path,
        baseline_row=baseline,
        baseline_metrics=baseline,
        baseline_per_user={
            "recall": np.zeros(2), "ndcg": np.zeros(2),
            "revenue": np.zeros(2), "arp": np.zeros(2),
        },
        input_manifest={"transactions": {}, "item_metadata": {}},
        baseline_state_hash="base-state",
        base_cfg={"N_BOOT": 10, "K_LIST": [10, 20, 50]},
        data={"data_stats": {}},
        context={
            "user_profile": SimpleNamespace(feature_names=("u0",)),
            "item_profile": SimpleNamespace(numeric_names=("i0",)),
            "artifact": SimpleNamespace(diagnostics={}),
        },
        source_revision="test-revision",
    )
    monkeypatch.setattr(single, "_prepare_validation_context", lambda cfg: prepared)

    def fake_variant(prepared, cfg, model_id):
        calls["controls"].append(model_id)
        revenue = full_revenue if model_id == "single_full" else 1.01
        rows = []
        per_user = {}
        for lam in cfg.lambda_eval:
            row = _reuse_metric_row(float(lam)) | {
                "model_id": model_id,
                "role": "model" if model_id == "single_full" else "control",
                "revenue@10": revenue if lam == 1.0 else 1.0,
            }
            rows.append(row)
            per_user[float(lam)] = prepared.baseline_per_user
        return single.VariantRun(
            model_id=model_id,
            rows=tuple(rows),
            per_user=per_user,
            training={"base_updates_at_best": 3},
            diagnostics={"parameter_match_ratio": 1.0},
            checkpoint=str(tmp_path / f"{model_id}.pt"),
            reuse_provenance=None,
        )

    monkeypatch.setattr(single, "_train_evaluate_variant", fake_variant)
    pref_row = baseline | {
        "model_id": "pref_continue",
        "role": "control",
        "lambda": 0.0,
        "revenue@10": 1.0,
    }
    monkeypatch.setattr(single, "_run_pref_continue", lambda *args, **kwargs: pref_row)
    return calls
```

`single_full`이 성공하도록 결정적인 지표를 반환하는 테스트에서 이 helper를 사용한다.

```python
def test_runner_trains_full_then_all_controls_only_after_success(monkeypatch, tmp_path):
    import lightgcn_clv_single as single
    calls = _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    cfg = single.configure_single_run("dunnhumby", out_dir=str(tmp_path))
    frame = single.run_experiment(cfg)
    assert calls["controls"] == [
        "single_full",
        "single_zero_user",
        "single_shuffled_user",
        "single_base_only",
        "single_zero_item",
    ]
    assert set(frame.model_id) >= {"m1", "single_full", *single.REQUIRED_CONTROLS,
                                   *single.MECHANISM_CONTROLS, "pref_continue"}
    assert frame.attrs["screening_decision"]["success"] is True
```

`full_revenue=0.99`인 두 번째 경우를 추가해 대조군과 `pref_continue`가 학습되지 않고 최종 판정이 false인지 확인한다.

- [ ] **2단계: 실행순서 테스트를 실행해 RED 확인**

실행: `pytest -q test_lightgcn_clv_single.py -k runner`

예상 결과: `run_experiment`가 구현되지 않아 FAIL.

- [ ] **3단계: 실행기를 명시적 단계로 구현**

`lightgcn_clv_moe.run_experiment`를 변경하지 않고 기존 MoE helper를 사용한다.

```python
def run_experiment(cfg=None, *, reuse_full_result_json=None):
    cfg = validate_single_config(cfg or configure_single_run("dunnhumby"))
    prepared = _prepare_validation_context(cfg)
    rows = [prepared.baseline_row]
    full = _reuse_or_train_full(prepared, cfg, reuse_full_result_json)
    rows.extend(full.rows)
    selected, selection_success, selection_tables = _select_models(
        rows, prepared.baseline_metrics, (PRIMARY_MODEL_ID,)
    )
    controls = {}
    if selection_success[PRIMARY_MODEL_ID] and cfg.run_controls_after_success:
        for model_id in (*REQUIRED_CONTROLS, *MECHANISM_CONTROLS):
            controls[model_id] = _train_evaluate_variant(prepared, cfg, model_id)
            rows.extend(controls[model_id].rows)
        selected, selection_success, selection_tables = _select_models(
            rows, prepared.baseline_metrics, ALL_SINGLE_MODELS
        )
        pref_row = _run_pref_continue(prepared, cfg, full.training)
        if pref_row is not None:
            rows.append(pref_row)
        selected["pref_continue"] = 0.0
        selection_success["pref_continue"] = True
    decision = single_screening_decision(rows, selected, selection_success)
    return _persist_result(prepared, cfg, rows, selected, selection_success,
                           selection_tables, decision, full, controls)
```

`_prepare_validation_context`는 학습자료와 검증자료 산출물만 만들고 같은 seed의 외부 순수 M1을 사용하며 대응 차이 계산을 위한 M1 사용자별 지표를 보존한다. `_train_evaluate_variant`는 모든 변형을 동일한 외부 M1 상태의 새 복사본에서 시작하고 기존 `train_moe`를 `freeze_base=False`로 호출한다.

- [ ] **4단계: 선택행 delta와 결과 schema 구현**

다음 형식으로 저장한다.

```python
stem = f"clv_single_{cfg.dataset}_{fingerprint}"
frame.to_csv(out_dir / f"{stem}.csv", index=False, float_format="%.8f")
pd.DataFrame(delta_records).to_csv(out_dir / f"{stem}_delta.csv", index=False)
json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
```

JSON에는 `code_version`, `source_revision`, `result_fingerprint`, `input_manifest`, `config`, `base_config`, `data_stats`, `feature_schema`, `variant_definitions`, `baseline_state_hashes`, `selected_lambda`, `lambda_selection_success`, `screening_decision`, `selection_tables`, `encoder_diagnostics`, `training`, `diagnostics`, `checkpoint_paths`, `reuse_provenance`, `absolute_rows`, `delta`, CLV와 revenue 해석문을 저장한다.

선택된 모형마다 `recall`, `ndcg`, `revenue`, `arp`의 사용자 대응 부트스트랩 차이를 저장한다. 절대 지표행에는 모든 노출지표를 유지한다.

- [ ] **5단계: 결과 저장과 직접 설정 우회 차단 테스트 추가**

```python
def test_runner_persists_authoritative_json_and_exposure_metrics(monkeypatch, tmp_path):
    _install_tiny_runner_stubs(monkeypatch, tmp_path, full_revenue=1.10)
    frame = single.run_experiment(single.configure_single_run("dunnhumby", out_dir=str(tmp_path)))
    payload = json.loads(next(tmp_path.glob("clv_single_*.json")).read_text())
    assert payload["screening_decision"] == frame.attrs["screening_decision"]
    assert payload["variant_definitions"]["single_zero_user"]["user_profile"] == "zero"
    assert {"n_distinct@10", "exposure_entropy@10", "eff_catalog@10",
            "top10_share@10", "top100_share@10"}.issubset(payload["absolute_rows"][0])


def test_run_experiment_revalidates_before_data_access(monkeypatch):
    monkeypatch.setattr(single, "_prepare_validation_context",
                        lambda cfg: (_ for _ in ()).throw(AssertionError("data touched")))
    bad = dataclasses.replace(single.configure_single_run("dunnhumby"), seed_list=(42, 43))
    with pytest.raises(ValueError, match="seed 42"):
        single.run_experiment(bad)
```

- [ ] **6단계: 실행기 테스트 후 작업 4 커밋**

실행:

```bash
pytest -q test_lightgcn_clv_single.py
ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py
git add lightgcn_clv_single.py test_lightgcn_clv_single.py
git commit -m "feat: run single-adapter CLV identification"
```

### 작업 5: 보호장치가 적용된 Colab 실행기 추가

**파일:**
- 생성: `clv_single_adapter_colab.ipynb`
- 수정: `test_lightgcn_clv_single.py`

**인터페이스:**
- 입력: `configure_single_run`, `preflight_summary`, `run_experiment`.
- 출력: 새로 복제한 저장소에서 H&M 2년 또는 Dunnhumby seed 42 검증 선별실험을 실행하는 Colab 절차.

- [ ] **1단계: 노트북 계약 실패 테스트 작성**

```python
def test_colab_has_pinned_source_preflight_gate_and_final_decision():
    notebook = json.loads(Path("clv_single_adapter_colab.ipynb").read_text())
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "configure_single_run" in source
    assert "preflight_summary" in source
    assert "reuse_full_result_json" in source
    assert "ACKNOWLEDGE_HIGH_COST = False" in source
    assert "assert ACKNOWLEDGE_HIGH_COST" in source
    assert "screening_decision" in source
    assert "eval_test=False" in source and "eval_holdout=False" in source
```

- [ ] **2단계: 노트북 계약 테스트를 실행해 RED 확인**

실행: `pytest -q test_lightgcn_clv_single.py::test_colab_has_pinned_source_preflight_gate_and_final_decision`

예상 결과: 노트북이 없으므로 FAIL.

- [ ] **3단계: 여섯 개 명시적 구역으로 노트북 생성**

다음 구역을 포함하는 유효한 노트북 JSON을 생성한다.

1. GPU 런타임 확인과 Google Drive 연결.
2. 저장소를 새로 복제하고 검토된 커밋으로 이동한다. 최종 검토 후 정확한 SHA를 출력하고 일치 여부를 검증한다.
3. 데이터셋 사전설정:

```python
DATASET = "dunnhumby"  # or "hm"
cfg = configure_single_run(
    DATASET,
    seed_list=(42,),
    eval_test=False,
    eval_holdout=False,
    out_dir=f"/content/drive/MyDrive/논문/data/results_clv_single_{DATASET}",
    m1_checkpoint_dir=f"/content/drive/MyDrive/논문/data/results_v3_{DATASET}",
)
reuse_full_result_json = (
    "/content/drive/MyDrive/논문/data/results_clv_moe_dunnhumby/"
    "clv_moe_dunnhumby_6f89c6b32f.json"
    if DATASET == "dunnhumby" else None
)
```

4. 전체 사전점검 JSON과 아직 학습이 시작되지 않았다는 명시적 안내.
5. 고비용 실행 승인 셀:

```python
ACKNOWLEDGE_HIGH_COST = False
assert ACKNOWLEDGE_HIGH_COST, "설정 검토 후 True로 바꾸세요."
result_df = run_experiment(cfg, reuse_full_result_json=reuse_full_result_json)
```

6. 결과표, 저장경로, `selected_lambda`, `lambda_selection_success`, 최종 `screening_decision.success/reason/failed_controls/mechanism_comparison`.

- [ ] **4단계: 노트북 JSON과 계약 검증**

실행:

```bash
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
pytest -q test_lightgcn_clv_single.py -k colab
```

예상 결과: JSON 유효, 테스트 PASS.

- [ ] **5단계: 작업 5 커밋**

```bash
git add clv_single_adapter_colab.ipynb test_lightgcn_clv_single.py
git commit -m "feat: add guarded single-adapter Colab"
```

### 작업 6: 최종 검증, 연구상태 갱신, 코드리뷰, commit 고정

**파일:**
- 수정: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`
- 리뷰 후 수정: `clv_single_adapter_colab.ipynb`
- 테스트: 저장소 전체 테스트.

**인터페이스:**
- 입력: 완료된 작업 1~5.
- 출력: 리뷰된 소스 커밋, 고정된 노트북 커밋, 검증된 로컬 구현, 고비용 결과를 주장하지 않는 갱신된 연구상태 문서.

- [ ] **1단계: 집중 검증과 전체 검증 실행**

```bash
pytest -q test_clv_moe_model.py test_lightgcn_clv_single.py
pytest -q
ruff check .
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
git diff --check
```

예상 결과: 모든 테스트 PASS, ruff·JSON·diff 검사 성공.

- [ ] **2단계: 학습 없이 데이터셋 사전설정 사전점검 실행**

```bash
python - <<'PY'
import json
from lightgcn_clv_single import configure_single_run, preflight_summary

for dataset in ("dunnhumby", "hm"):
    cfg = configure_single_run(
        dataset,
        encoder_epochs=1,
        frozen_epochs=5,
        max_epochs=6,
        patience=2,
    )
    summary = preflight_summary(cfg)
    assert summary["seed_list"] == [42]
    assert summary["eval_test"] is False
    assert summary["eval_holdout"] is False
    print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
```

예상 결과: 두 preset이 모두 출력되고 데이터 파일을 읽거나 학습을 시작하지 않음.

- [ ] **3단계: 연구상태 문서 갱신**

`/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`에 다음 내용을 기록하는 구현 소절을 추가한다.

- 정확한 구현 commit과 테스트 수.
- model ID와 zero/shuffle 정의.
- validation-only 및 test 보호 동작.
- Dunnhumby 재사용 조건: manifest, M1 state, encoder 값, 특징 schema, checkpoint, 지표 round-trip.
- 구현 검증에서 새 H&M·Dunnhumby 고비용 결과를 만들지 않았다는 사실.

2026-08-10에 관찰된 MoE/single-adapter 결과는 수정하거나 삭제하지 않는다.

- [ ] **4단계: 구현 문서 커밋**

저장소의 plan/spec은 이미 커밋됐다. 최종 검증 중 저장소 문서가 바뀐 경우 추적되는 문서만 커밋한다.

```bash
git add docs clv_single_adapter_colab.ipynb
git commit -m "docs: finalize single-adapter screening workflow"
```

`/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`는 저장소 밖에 있으므로 커밋하지 않는다.

- [ ] **5단계: 코드리뷰 요청 및 검증된 지적만 반영**

설계 커밋 `ee606d5`를 기준으로 `superpowers:requesting-code-review`를 실행한다. 코딩 표준과 설계 준수 여부를 모두 검토한다. 문서 모순이 아닌 코드 지적은 수정 전에 실패하는 실행 테스트로 재현한다.

- [ ] **6단계: 리뷰 수정 후 전체 검증 재실행**

1단계 명령을 그대로 반복하고 최종 통과 테스트 수와 소스 커밋을 기록한다.

- [ ] **7단계: 노트북을 검토된 소스 커밋에 고정**

먼저 검토된 소스 커밋을 출력한다.

```bash
git rev-parse HEAD
```

`apply_patch`로 출력된 40자 SHA를 노트북의 저장소 복제 셀에 있는 `REVIEWED_SHA`에 그대로 넣는다. 이어서 `git checkout {REVIEWED_SHA}`와 `assert actual_sha[0] == REVIEWED_SHA`를 실행하게 한다. 브랜치 이름, 축약 SHA, 명령 치환, 임시 표시는 커밋하지 않는다.

- [ ] **8단계: commit 고정 변경만 검증하고 커밋**

```bash
python -m json.tool clv_single_adapter_colab.ipynb >/dev/null
pytest -q test_lightgcn_clv_single.py -k colab
git diff --check
git add clv_single_adapter_colab.ipynb
git commit -m "chore: pin single-adapter Colab to reviewed commit"
```

- [ ] **9단계: 사용자 승인 후에만 배포**

사용자가 배포를 요청하기 전에는 원격 저장소에 올리거나 pull request를 갱신하지 않는다. 승인을 받으면 현재 기능 브랜치를 원격 저장소에 올리고 Colab GitHub URL이 커밋된 노트북을 정상적으로 여는지 확인한다.
