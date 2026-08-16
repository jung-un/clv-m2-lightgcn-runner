# H&M 2년 M2 Seed 42 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** H&M 2년 validation에서 동결된 dual-axis M2를 seed 42·rho 0.2 한 점으로 실행한다.

**Architecture:** 기존 `lightgcn_clv_dual`의 준비·학습·평가 함수를 재사용한다. 전용 runner는 주모형 한 개만 학습하고 진단된 residual 비율로 lambda를 한 번 계산하며, Colab은 이 runner만 호출한다.

**Tech Stack:** Python, PyTorch, pandas, Colab

## Global Constraints

- H&M 전체기간, seed 42, validation only
- gate=`high`, rho=`0.2`
- 대조군·sweep·test·holdout·seed 43·44 미실행

---

### Task 1: 고정점 runner

**Files:**
- Create: `lightgcn_clv_dual_hm2y_seed42.py`
- Create: `test_lightgcn_clv_dual_hm2y_seed42.py`

- [ ] 설정이 H&M 전체기간·seed42·validation only·rho0.2를 강제하는 실패 테스트를 작성한다.
- [ ] 주모형만 한 번 학습하고 `lambda=rho/raw_effective_ratio`로 한 점만 평가하는 실패 테스트를 작성한다.
- [ ] 기존 dual helper를 조합하는 최소 runner를 구현한다.
- [ ] 집중 테스트와 Ruff를 실행한다.

### Task 2: Colab과 배포

**Files:**
- Create: `clv_dual_hm2y_seed42_colab.ipynb`
- Modify: `test_lightgcn_clv_dual_hm2y_seed42.py`
- Modify: `RESEARCH_STATUS.md` (workspace document)

- [ ] 노트북이 GPU·전체기간·seed42·validation-only 설정과 출력표를 제공하는지 테스트한다.
- [ ] 노트북을 구현하고 검토 커밋 SHA에 핀한다.
- [ ] JSON·pytest·Ruff를 검증하고 연구 브랜치에 push한다.
- [ ] 실행 전 상태와 다음 판정 규칙을 연구상태에 기록한다.
