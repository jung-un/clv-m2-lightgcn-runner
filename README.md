# CLV-LightGCN runner

LightGCN의 M1 기준모형과 CLV·가치정보 개입 위치(M2/M3/M4)를 같은 분할과 평가체계에서 비교하는 연구 코드다.

## CLV-Residual M2

`lightgcn_clv_residual.py`는 365일 과거행동으로 향후 90일 구매여부와 구매금액을 예측하도록 학습한 16차원 고객가치 임베딩을 사용한다. 동결된 M1 사용자 표현에 작은 residual과 사용자 gate를 더하며, 추천 손실은 M1과 같은 plain BPR만 사용한다. 따라서 CLV 가중 BPR을 연구하는 M4와 구분된다.

Colab에서는 `clv_residual_colab.ipynb`를 연다. 기본 설정은 다음과 같다.

- Dunnhumby, seed 42, validation only
- H&M과 Dunnhumby 모두 전체 약 2년 사용
- `lambda=[0, 0.05, 0.1, 0.25, 0.5, 1.0]`
- Recall/NDCG@10·20·50이 M1의 99% 이상인 후보 중 가격·구매금액 가중 적중값@10 최대화
- 제안모형과 constant-CLV 대조군 동시 실행
- `ACKNOWLEDGE_HIGH_COST=True`로 바꾸기 전에는 학습하지 않음

Drive 입력 경로는 기존 v3 스키마를 따른다.

```text
/content/drive/MyDrive/논문/data/raw/hm/transactions_train.parquet
/content/drive/MyDrive/논문/data/raw/hm/articles.csv
/content/drive/MyDrive/논문/data/raw/dunnhumby/transaction_data.csv
/content/drive/MyDrive/논문/data/raw/dunnhumby/product.csv
```

로컬 단위 테스트:

```bash
pytest -q
```

`revenue`/`PWGain`은 가격·구매금액 가중 추천 적중값이며 실제 증분매출이 아니다. 새 encoder의 `EV`도 향후 90일 기대구매금액 예측값이지 생애전체 CLV가 아니다.
