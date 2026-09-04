# M3: 구매취향 후보 안의 historical CLV 조건부 이웃 선택

## 1. 연구 위치와 질문

이 실험은 M3 그래프 구조·전파 축이다. M1의 이진 사용자–상품 그래프,
uniform negative sampling, plain BPR은 유지한다. 새 질문은 다음과 같다.

> 구매취향이 비슷한 사용자 후보 안에서 historical CLV 수준과 N/V 구성이
> 비슷한 사용자를 최종 이웃으로 선택하면 신규상품 추천이 개선되는가?

CLV는 전체 사용자 중 이웃을 단독으로 만들지 않는다. 구매취향이 먼저 후보를
제한하고, CLV는 그 후보 안에서 연결의 방향과 질량 배분을 바꾼다. 고CLV
사용자에게 무조건 더 많은 메시지를 주지 않으며 저·중·고CLV 사용자는 모두
같은 규칙을 적용받는다.

이 구조는 사용자–사용자 관계를 중간 계산으로 사용하지만, 최종적으로

\[
B_{ui}^{arm}=\sum_v W_{uv}^{arm}\widehat A_{vi}
\]

라는 사용자–상품 전파 연산자를 만든다. 따라서 M3의 개입은 CLV 조건부로
유도된 사용자–상품 그래프의 추가 전파다. 기존의 관측 user-item edge 재가중
계열과 다른 구조라는 사실은 결과 문서에 함께 기록한다.

## 2. 고정 데이터와 과업

- 데이터: Dunnhumby
- seed: 42
- 학습정보: `DAY 1~683`
- 탐색 성능평가: `DAY 684~690`
- 평가 정답: 학습기간에 없었던 `(user,item)` 신규상품 쌍만 사용
- `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`
- validation 선택 없음, 고정 100 epoch
- final test와 holdout은 구성하지 않음
- 단일 seed 역사적 개발실험이므로 유의성·일반화를 주장하지 않음

고비용 학습 전 관계 진단은 공식 성능평가구간을 읽지 않고 `DAY 1~683`
내부의 마지막 다섯 개 비중첩 7일 창을 사용한다.

## 3. 구매취향 후보 관계

기준시점까지의 고유 구매 여부를 `A_ui`라고 한다. 거래횟수와 구매금액은
취향 유사도에 사용하지 않는다.

\[
p_{ui}=A_{ui}\log\frac{|\mathcal U|+1}{\deg(i)+1}
\]

\[
s_{uv}^{pref}=\max\{0,\cos(p_u,p_v)\}
\]

자기 자신을 제외하고 `s_pref`가 큰 사용자 100명을 후보 이웃으로 고정한다.
양의 유사도 이웃이 100명보다 적으면 존재하는 이웃만 사용한다. 양의 이웃이
없으면 추가 관계를 만들지 않고 해당 사용자는 정확히 M1로 복귀한다.

## 4. Historical CLV 수준과 구성

학습시점까지의 고유 장바구니 수와 평균 장바구니 금액을 사용한다.

\[
\widehat N_u=\text{number of distinct baskets},\qquad
\widehat V_u=\text{mean basket value}
\]

\[
q_C(u)=\operatorname{Percentile}(\widehat N_u\widehat V_u)
\]

\[
q_N(u)=\operatorname{Percentile}(\widehat N_u),\qquad
q_V(u)=\operatorname{Percentile}(\widehat V_u)
\]

\[
d_u=q_N(u)-q_V(u),\qquad x_D(u)=\frac{d_u+1}{2}
\]

`q_C`는 CLV 총수준, `x_D`는 같은 CLV 안에서 빈도 우세와 거래가치 우세를
구별하는 구성좌표다. 둘 다 `[0,1]` 범위다.

## 5. CLV 유사성과 이력 신뢰도

학습형 bandwidth와 축별 가중치를 두지 않는다. 범위가 정규화된 수치변수의
Gower 유사도를 사용한다.

\[
s_{uv}^{CLV}=1-\frac{|q_C(u)-q_C(v)|+|x_D(u)-x_D(v)|}{2}
\]

희소 사용자에게 불안정한 CLV 구성을 강하게 적용하지 않도록 학습시점까지의
고유 구매상품 수 `n_u`로 공통 축소계수를 만든다.

\[
r_u=\frac{n_u}{n_u+5}
\]

`5`는 이번 실험 전에 고정한 prior strength이며 결과를 보고 조정하지 않는다.
모든 arm은 같은 `r_u`를 사용한다.

## 6. 최종 이웃 관계

TF-IDF 상위 100명 안에서만 다음 affinity를 계산한다.

\[
a_{uv}^{CLV}=s_{uv}^{pref}
\left[(1-r_ur_v)+r_ur_vs_{uv}^{CLV}\right]
\]

이를 기준으로 상위 20명을 남기고 행합 1로 정규화한다.

\[
W_{uv}^{CLV}=\operatorname{RowNormalize}
\left[\operatorname{Top20}_{v\in\operatorname{Top100}^{pref}(u)}
a_{uv}^{CLV}\right]
\]

관계 대조군은 같은 후보 100명 중 `s_pref` 상위 20명을 사용한다.

\[
W_{uv}^{pref}=\operatorname{RowNormalize}
\left[\operatorname{Top20}_{v\in\operatorname{Top100}^{pref}(u)}
s_{uv}^{pref}\right]
\]

모든 유효 행에서 두 연산자의 행합은 1이므로 다음이 성립한다.

\[
\sum_v(W_{uv}^{CLV}-W_{uv}^{pref})=0
\]

따라서 CLV는 총 이웃 메시지 계수질량을 늘리지 않고 누구의 정보를 받는지만
재배분한다.

## 7. 대조군

모든 M3 arm은 같은 TF-IDF 후보 100명, 최종 이웃 20명, 축소계수, 학습
설정과 초기화를 사용한다.

| arm | 최종 이웃 선택 |
|---|---|
| M1 | 사용자–사용자 추가 관계 없음 |
| 관계 대조군 | `s_pref` 상위 20명 |
| 실제 CLV | 실제 `(q_C,q_N,q_V,x_D)`로 계산한 CLV 유사성 적용 |
| CLV shuffle | binary user-degree 10분위 안에서 `(q_C,q_N,q_V,x_D)` 튜플 전체를 사용자 단위로 순열 |
| Degree 관계 | `s_degree=1-|q_degree(u)-q_degree(v)|`를 같은 축소식에 적용 |

CLV shuffle은 CLV 수준과 N/V 구성의 내부 일관성을 보존한다. Degree 관계는
비슷한 활동량의 사용자를 고른 효과와 CLV 의미의 효과를 구별한다.

## 8. LightGCN 내부 적용

M1의 방향별 전파를 다음처럼 쓴다.

\[
e_u^{(1)}=\widehat A_{UI}e_i^{(0)},\qquad
e_i^{(1)}=\widehat A_{IU}e_u^{(0)}
\]

\[
e_u^{(2),M1}=\widehat A_{UI}e_i^{(1)},\qquad
e_i^{(2),M1}=\widehat A_{IU}e_u^{(1)}
\]

각 arm의 추가 상품정보 메시지는 다음과 같다.

\[
m_u^{arm}=\sum_vW_{uv}^{arm}e_v^{(1)}
\]

사용자 2층 표현만 고정 `gamma=0.075`로 결합한다.

\[
e_u^{(2),arm}=0.925e_u^{(2),M1}+0.075m_u^{arm}
\]

상품 1·2층 전파는 M1과 정확히 같다. 최종 표현과 점수는 다음과 같다.

\[
z_u^{arm}=\frac{e_u^{(0)}+e_u^{(1)}+e_u^{(2),arm}}{3}
\]

\[
z_i^{arm}=\frac{e_i^{(0)}+e_i^{(1)}+e_i^{(2),M1}}{3},\qquad
S(u,i)=z_u^{arm\top}z_i^{arm}
\]

`gamma=0` 또는 유효 이웃 없음이면 정확히 M1이다. 외부 점수 합산·재정렬,
사전학습·동결, 표본가중, 새 손실항은 없다. 사용자·상품 ID embedding은
하나의 plain BPR과 optimizer로 처음부터 함께 학습한다.

관계 대조군과 실제 CLV의 직접 차이는 다음으로 기록한다.

\[
0.075(W^{CLV}-W^{pref})e^{(1)}
\]

## 9. 학습 전 다중시점 관계 진단

각 기준시점 `a`에서 `t<=a`만으로 모든 관계를 만들고, `(a,a+7]`의 신규상품을
임시 정답으로 사용한다. 기준시점까지 등장하지 않은 상품과 이미 구매한
`(user,item)` 쌍은 정답과 후보에서 제외한다.

각 arm의 후보점수는 다음과 같다.

\[
C_{ui}^{arm}=\sum_vW_{uv}^{arm}A_{vi}
\]

자기 과거 구매상품을 제거한 뒤 동일하게 Top-100을 선택하고 사용자별
Candidate Recall@100을 계산한다.

고비용 학습 통과조건은 다음과 같다.

1. 실제 CLV의 전체 사용자 평균 Candidate Recall@100이 관계 대조군보다 높다.
2. 실제 CLV가 degree-matched CLV shuffle보다 높다.
3. 실제 CLV가 Degree 관계보다 높다.
4. 위 세 대응 차이가 각각 5개 시점 중 최소 3개에서 양수다.
5. 실제 CLV와 shuffle의 최종 Top-20 이웃집합이 다른 사용자 비율이 10% 이상이다.
6. 모든 arm에서 사용자별 최종 이웃 수와 행별 총질량 조건이 일치한다.

이번 구조는 CLV에 따라 메시지 양을 단조 증가시키지 않으므로
`고CLV 개선폭 >= 전체 개선폭`은 판정조건으로 사용하지 않는다. 저·중·고CLV
결과와 CLV percentile–대응차이 Spearman은 기술 진단으로 모두 보고한다.

하나라도 실패하면 100 epoch 성능학습을 실행하지 않는다. 이는 앞선
`q_C` 비례 gate의 판정조건을 완화한 것이 아니라, CLV의 개입 역할이
메시지 양에서 이웃 정체성으로 바뀐 별도 가설이다.

## 10. 성능학습과 판정

학습 전 진단을 통과할 때만 관계 대조군·실제 CLV·CLV shuffle·Degree 관계를
각각 seed 42, 고정 100 epoch로 학습한다. 호환되는 M1을 같은 표에 둔다.

주 판정은 실제 CLV의 Recall/NDCG@10·20·50 여섯 지표 기하평균이 M1,
관계 대조군, CLV shuffle, Degree 관계를 각각 초과하는지다. 개별 여섯 지표와
가격·구매금액 가중 적중값, 추천 상품 평균 가격 백분위, coverage와 노출
집중도, CLV 세그먼트 지표를 함께 보고하되 사후 성공조건을 추가하지 않는다.

실제 CLV가 M1만 이기고 관계 대조군 또는 shuffle을 이기지 못하면 CLV
고유효과로 해석하지 않는다. Degree 관계를 이기지 못하면 구매활동량 유사성
효과와 분리하지 못한 것으로 판정한다. 단일 seed에서는 유의성을 주장하지
않고, 양성일 때만 추가 seed 또는 H&M으로 확장한다.

## 11. 불변조건과 구현 검증

- TF-IDF 후보는 train-only 고유 구매쌍으로만 계산한다.
- 자기 자신과 사용자 자신의 과거 구매상품을 후보에서 제외한다.
- 실제·shuffle·Degree 관계는 동일한 TF-IDF 후보집합 안에서만 선택한다.
- 모든 유효 관계행의 합은 1이고 최종 이웃 수는 최대 20이다.
- 실제 CLV와 shuffle은 같은 CLV 튜플 다중집합을 degree 층 안에서 보존한다.
- `gamma=0`과 무효 이웃 사용자는 정확히 M1이다.
- 상품 전파와 상품 최종 표현은 M1과 동일하다.
- 학습 파라미터는 사용자·상품 embedding뿐이며 같은 BPR gradient를 받는다.
- test는 고정 100 epoch 뒤 arm당 한 번만 평가한다.
- holdout 정답·지표·파일을 만들지 않는다.

Colab은 고정 source commit을 checkout하고 기존 `lightgcn_clv*`, `clv_m3*`,
`clv_run_state` 모듈 캐시를 제거한 뒤 실행한다.

