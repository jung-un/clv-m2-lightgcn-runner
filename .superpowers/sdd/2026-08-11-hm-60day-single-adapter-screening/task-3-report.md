# Task 3 보고: H&M 60일 anchor 입력 진단 저장

## 구현 결과

- `summarize_anchor_dataset()`를 추가했다. 각 train-only encoder anchor에 대해 offset, 관찰·목표 기간, 관찰 사용자 수, `observed_days` p10/중앙값/p90, 미래 구매율, 미래 구매금액 평균을 JSON-safe 값으로 요약한다.
- `PreparedSingleContext.anchor_diagnostics`에 요약을 보관한다.
- encoder checkpoint와 최종 결과 JSON의 최상위 `anchor_diagnostics`에 동일한 anchor 순서의 진단을 저장한다.

## 검증

- RED: `pytest -q test_lightgcn_clv_single.py -k 'anchor_diagnostics or persists_authoritative'`에서 helper 부재와 JSON 키 부재로 2개 실패를 확인했다.
- GREEN/집중: `pytest -q test_lightgcn_clv_single.py -k 'anchor_diagnostics or persist'` → 2 passed.
- 전체 단일 실행기: `pytest -q test_lightgcn_clv_single.py` → 34 passed.
- 정적 검사: `ruff check lightgcn_clv_single.py test_lightgcn_clv_single.py` 및 `git diff --check` 통과.

## 범위 및 해석

이 진단은 짧은 행동창의 입력 품질을 투명하게 남기기 위한 train-only 메타데이터다. `future_amount_mean`은 anchor 목표기간의 관측 구매금액 평균이며, 실현 CLV 또는 추천으로 인한 증분매출을 뜻하지 않는다.
