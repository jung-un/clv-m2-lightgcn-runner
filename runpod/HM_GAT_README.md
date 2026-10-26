# H&M GAT seed49 — M1~M4 Runpod 실행 준비

기본 범위는 seed49 M1→M2→M3→M4. M5_A/B/C/D는 연구계획에 남기되 자동 실행하지 않습니다.
기존 Dunnhumby GAT·H&M NGCF·Colab을 변경하지 않는 새 Pod에서 사용합니다.
코드 준비 완료와 실제 GPU 메모리 적합성/전체 데이터 검증 완료는 다릅니다. 이 패키지는 아직 전체 H&M GPU 실행을 하지 않았습니다.

## 실행
1. 신규 Pod에 영구 저장공간을 `/workspace`로 연결합니다. GPU 모델은 실제 메모리 점검 전 확정하지 않습니다. 전체 그래프를 사용하는 GAT이므로 48GB에 반드시 들어간다고 보장하지 않습니다.
2. `hm_gat_start.zip`을 `/workspace`에 업로드하고 Terminal에서 `unzip -n /workspace/hm_gat_start.zip -d /workspace` 실행.
3. `hm_gat_start/HM_GAT_시작.ipynb`를 열어 순서대로 실행합니다. 설치·검사 셀에서는 학습하지 않습니다.
4. 개인 Drive에서 이미 검증한 로컬 `transactions_train.parquet`를 새 서버의 `/workspace/transactions_train.parquet`에 업로드합니다. 같은 영구볼륨에 검증된 `/workspace/clv/data/raw/hm/transactions_train.parquet`가 있으면 재업로드하지 않습니다. 다른 서버에 볼륨을 연결할 때 기존 실행을 중단하지 않도록 별도로 확인해야 합니다.
5. 원본·코드·CUDA 검사가 통과한 뒤, 다른 서버에 H&M GAT seed49가 실행 중이지 않은지 확인하고 시작 셀의 `CONFIRMED_NO_DUPLICATE=True`로 바꿉니다.
6. 기본 MODELS를 유지하면 M1~M4만 실행됩니다. 첫 배치와 첫 epoch 완료·저장을 확인해야 실제 학습이 시작된 것입니다.

## 보존·재개
- 출력 `/workspace/clv/results/hm-gat`는 NGCF/GAT(DH)와 분리됩니다.
- 같은 볼륨·동일 GPU/환경/고정 코드에서 재개합니다. 완료 결과는 재학습·재평가하지 않고, 미완료는 모델·Adam·난수를 복구합니다.
- 동일 볼륨 hm-gat 작업 잠금으로 이중 실행을 막습니다. 다른 서버/계정의 중복은 자동 감지하지 못합니다.
- 마지막 epoch 저장 직전본 보존, 손상파일·저장공간 부족 시 중단. 최종평가가 시작됐는데 결과가 없으면 자동 재평가하지 않습니다.
- 결과 `/workspace/clv/exports/hm-gat_results_partial.zip`; 8모형이 모두 완료돼야 최종 ZIP 표시입니다. M1~M4 완료 시 partial은 정상입니다.
- 브라우저 종료와 학습 종료는 GPU 과금을 중지하지 않습니다. 자동 서버 정지는 포함하지 않습니다.

## 고정 연구 조건
H&M ~2020-09-15 학습/2020-09-16~22 test, 신규상품 추천, k=1, seed49, 300epoch 마지막점 1회 평가, validation/holdout/조기종료 없음. H&M 기존 batch131072/lr.0005/ID64/2층/uniform K1 유지. 기존 GAT 단일 head·self-loop 없음·feature/attention dropout0·층평균 사용. M2는 같은 추천손실로 공동학습, M3는 원래 거래가중치를 attention prior에 반영, M4는 원래 BPR 가중치. M3의 GAT 행 정규화는 LightGCN 가중차수 보존과 같지 않습니다. 새 하이퍼파라미터 탐색/테스트 기반 선택 없음.

원본 SHA256: `0d90884fc1bc400161e6aeba238cba94183998bedfeb4ebbf4962e17d0c8489b`.
M3 near-unit 평균의 순위 진단은 기존 허용 기준을 유지하고 참조/현재/차이를 기록합니다. 원본·특징·M4 해시 및 기타 그래프 진단 수치 일치 검사를 유지합니다.
