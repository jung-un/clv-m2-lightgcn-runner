# M3: historical CLV-conditioned TF-IDF neighbor residual

## 연구 위치와 질문

이 실험은 M3(그래프 구조·전파) 축이다. M1의 이진 사용자–상품 그래프와
plain BPR은 유지하고, TF-IDF 구매취향으로 만든 사용자–사용자 그래프의
메시지를 historical CLV 수준에 따라 사용자 표현에 추가한다.

질문은 다음과 같다.

> 취향이 비슷한 사용자의 추가 협업정보를 historical CLV가 높은 사용자에게
> 더 크게 배분하면 신규상품 추천이 개선되는가?

CLV는 이웃이나 상품을 고르지 않는다. 취향 그래프가 정보의 방향을 만들고,
historical CLV percentile은 그 방향을 반영하는 양만 정한다.

## 학습 전 관계 진단

공식 평가구간을 쓰지 않고 학습기간 내부의 마지막 다섯 개 비중첩 7일 창을
사용한다. 각 기준시점에서 과거 구매만으로 다음 세 관계를 만들며 후보 예산은
사용자당 100개로 동일하다.

- `tfidf_topk_neighbor`: 이진 구매 TF-IDF cosine 상위 20명
- `ordinary_copurchase_propagation`: M1 대칭 정규화 그래프의 사용자 2-hop
- `degree_matched_random_neighbor`: 사용자 degree 10분위 내 무작위 20명

정답은 기준시점 이후 7일의 신규상품이며, 기준시점 이전 사용자–상품 쌍과
기준시점까지 등장하지 않은 상품은 제외한다. 사용자별 Candidate Recall@100의
대응 차이로 판정한다. TF-IDF가 두 대조군보다 전체 및 고CLV에서 높고,
고CLV 개선이 전체 개선 이상이며, 다섯 시점의 과반에서 고CLV 차이가 양수이고,
degree 층화 평균에서도 양의 방향일 때만 고비용 학습으로 진행한다.

## 모델 수식

학습기간 고유구매 행렬을 `A`라 하고

`p_ui = A_ui log((|U|+1)/(deg(i)+1))`

로 둔다. 자기 자신을 제외한 양의 cosine 상위 20명을 행 정규화한 연산자를
`S`라 한다. M1의 1층 사용자 표현을 `e_v^(1)`이라 할 때

`m_u = sum_v S_uv e_v^(1)`

이다. M1 최종 사용자 표현 `z_u`와 평행한 부분을 제거한다.

`r_u = m_u - (m_u^T z_u / ||z_u||^2) z_u`

유효 이웃이 없거나 `||z_u||`, `||m_u||`가 임계값보다 작으면 `r_u=0`이다.
유효 분모에는 epsilon을 더하지 않아 정상 사용자의 직교성을 보존한다.

`eta_u = ||r_u|| / ||m_u||`

는 진단값이며,

`rtilde_u = ||z_u|| r_u / ||m_u||`

로 residual의 자연스러운 상대크기를 보존한다. arm별 gate `g_u`와 고정
`rho=0.075`를 사용한다.

`h_u = z_u + rho g_u rtilde_u`

`z_u^M3 = ||z_u|| h_u / ||h_u||`

유효하지 않은 사용자는 정확히 `z_u`를 반환하며, `rho=0` 또는 `g=0`도
정확히 M1이다. 상품 표현은 M1과 같다.

## 대조군

- M1: `g=0`
- 관계 대조군: 모든 사용자 `g=mean(q_C)`
- 실제 CLV: `g=q_C`, `q_C=percentile(N_hat*V_hat)`
- CLV shuffle: binary degree 10분위 안에서 `q_C`를 순열
- degree gate: binary user-degree percentile

모든 arm은 같은 TF-IDF 관계, 초기화, 이진 M1 그래프, uniform negative,
plain BPR, 100 epoch, 하나의 optimizer를 사용한다. 표본가중·새 손실·외부
재정렬·사전학습·동결은 없다.

## 기록과 판정

시작·epoch별·최종 `B_all=mean(g*eta)`와 유효사용자 기준 `B_eligible`을
기록한다. actual과 shuffle의 `B_eligible` 상대차가 10%를 넘으면 순위 방향의
순수 귀속으로 단독 해석하지 않는다는 경고를 낸다. 단일 seed 42 결과는
탐색 결과이며 유의성·일반화를 주장하지 않는다.

