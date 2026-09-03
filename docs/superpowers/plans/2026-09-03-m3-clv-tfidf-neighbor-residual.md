# M3 TF-IDF neighbor residual implementation plan

1. TF-IDF Top-K, M1 2-hop, degree-matched random 사용자 관계를 만드는 순수
   함수를 추가하고 관계·후보 예산·누수 방지 테스트를 먼저 작성한다.
2. 학습기간 내부 다섯 시점에서 신규상품 Candidate Recall@100을 계산하고,
   전체·고CLV·degree 층화·시점 과반 조건을 저장하는 진단 실행기를 만든다.
3. M1 표현, 유사사용자 메시지, 정확한 직교 residual, 자연크기 보존,
   사용자 norm 보존을 구현한 공동학습 LightGCN을 추가한다.
4. `rho=0`, 무효 이웃, 0 메시지, 직교 오차, norm 보존, gradient 연결,
   실제/shuffle 동일 gate 다중집합을 국소 테스트로 고정한다.
5. M1·관계대조·실제 CLV·degree-matched shuffle·degree gate를 같은 100 epoch
   루프로 실행하고 비교·개입량 진단을 저장하는 runner를 만든다.
6. Colab notebook에서 소스 커밋을 checkout하고 기존 Python 모듈 캐시를
   제거한 뒤, 관계 진단 통과 시에만 고비용 학습을 실행한다.
7. 국소 테스트, Ruff, compile, notebook 정적검사를 통과한 뒤
   `RESEARCH_STATUS.md`를 갱신하고 현재 feature branch에 커밋·push한다.
