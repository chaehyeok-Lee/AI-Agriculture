# 셀프 피드백 루프 프롬프트 — "검증된 신호가 test 외삽에서 죽는 문제" 탐색

## 배경 (오늘 EDA에서 확인한 사실, 전부 재현 가능)

1. `circ_fan_mean`(순환팬)은 train에서 `soil_ec`와 상관계수 -0.49로, **day_num(선형상관 0.10)보다 훨씬 강한 실제 신호**다. 환기 ON/OFF가 EC 구간(A/B/C/D)을 구별하는 능력(between/within variance ratio)도 1.35로, 다른 어떤 X변수보다 압도적으로 높다 (내부습도 0.15, 내부온도 0.014, 외부온도·습도는 0.0002~0.0004로 거의 0).
2. 그런데 **현재 `output/submission.csv`(최신 train.py로 생성)를 까보면, test 135~146일의 soil_ec 예측이 사실상 평평하다** (일평균 0.406~0.410 사이, 138/140/141/143일은 소수점까지 완전히 동일한 값). 실제로 test의 circ_fan은 135~137일·144~146일엔 ON(0.4~0.9대), 138~143일엔 OFF(0.01 이하)로 뚜렷이 다른데도 예측값엔 전혀 반영되지 않고 있다.
3. 즉 **`circ_fan`이 이미 피처로 들어가 있고 train에서 검증된 강한 신호인데도, day_num이 학습 범위(109~134) 밖으로 나가는 test에서는 그 신호가 죽어버리는 것으로 보인다.** 아마 트리 분기 우선순위에서 day_num(feature importance 1위)이 먼저 걸려 "범위 밖"으로 분류된 뒤에는 circ_fan 분기까지 못 내려가거나, 그 리프에 도달하는 학습 샘플 자체가 별로 없어서일 가능성이 있다.

## 미션

**"train에서는 검증된 진짜 신호인데 test 외삽 상황에서는 죽어버리는 피처가 circ_fan 말고 또 있는지 전부 찾고, circ_fan을 포함해서 이 문제를 실제로 완화하는 방법을 찾아 적용하라.**

### 반드시 확인할 것 (순서대로)

1. **재현부터**: 위 1~3번 내용을 직접 코드로 재현해서 사실인지 먼저 확인한다 (`output/submission.csv` 재생성 후 일별 EC 통계 확인).
2. **원인 진단**: LightGBM의 `feature_importances_`와 실제 트리 분기 구조(`booster_.trees_to_dataframe()` 등)를 까서, test 샘플들이 실제로 어떤 리프에 도달하는지, day_num이 정말 circ_fan보다 먼저 분기되는지 확인한다. 막연히 추측하지 말고 직접 확인할 것.
3. **다른 후보 신호도 스캔**: circ_fan처럼 "train 내에서 EC/moisture/temp와 상관관계는 강한데, X변수 자체의 train/test 분포는 겹치는(=외삽 위험 없는) 피처"가 더 있는지 체계적으로 찾는다 (구간 구별력 점수 계산 방식은 `python/18_regime_predictors.py` 참고). 후보: `greenhouse_roof_vent1/2_mean`, `humidity_mean`, 이들의 rolling/lag 버전 등.
4. **해결 방법 시도** (아이디어 예시, 각각 4-fold CV로 검증):
   - day_num의 상대적 중요도를 낮추기 위해 `max_depth`/`min_child_samples`를 조정해서 circ_fan 분기가 더 일찍 걸리게 유도
   - day_num을 아예 빼고 circ_fan 계열 피처만으로 별도 모델을 만들어, day_num 모델과 앙상블(가중 평균)
   - circ_fan의 "연속 OFF 지속일수" 같은 파생 피처를 새로 만들어서(단, `fan_cumoff`는 이미 실패 이력 있음 — 왜 실패했는지 FEEDBACK.md에서 먼저 확인하고 그것과 무엇이 다른지 명확히 한 뒤 시도)
   - monotonic constraint(`monotone_constraints`)로 circ_fan↓→EC↑ 관계를 모델에 강제 힌트로 주기

### 절대 하지 말 것 (이미 실패 확인된 것들, FEEDBACK.md 참고)

- `cycle_phase`(14일 주기), `fan_day_interact`, `fan_cumoff`, EC 자기회귀 재귀 lag, huber loss, early stopping 제거 — 전부 이미 시도했다 실패함. **같은 걸 다시 시도하지 말고**, 왜 실패했는지 이해한 뒤 그것과 근본적으로 다른 접근인지 확인하고 진행할 것.
- 내부습도로 "정확한 급액 타이밍"을 추정하려는 시도 — 오늘 train 122~129일 구간에서 실제 EC와 대조 검증한 결과 상관계수 -0.036으로 **반박됨**. 내부습도는 큰 틀(높다/낮다) 참고 정도로만 쓰고, 세부 타이밍 예측에는 쓰지 말 것.

### 검증 기준

- 반드시 **4-fold 시계열 교차검증**(`FOLD_CUTOFFS`)으로 확인하고, 폴드 1개짜리 결과로 성급하게 결론 내리지 말 것.
- 개선됐다고 판단했으면, **실제 `output/submission.csv`의 soil_ec 일별 통계(mean/nunique)가 circ_fan ON/OFF 구간에 따라 실제로 달라지는지** 반드시 재확인할 것 (val RMSE만 보고 끝내지 말 것 — val 점수가 좋아도 실제 test 예측이 그대로 평평하면 의미 없음, 이건 이미 한 번 있었던 함정임).
- 모든 시도(성공/실패)는 FEEDBACK.md에 이어서 기록하고, PLAN.md의 "soil_ec" 항목도 최종 결과로 갱신할 것.

## 최종 산출물

- `train.py`/`inference.py` 수정 (개선이 확인된 경우만)
- `FEEDBACK.md`에 이번 루프 기록 추가
- `PLAN.md`, `SUMMARY.md`의 soil_ec 섹션 갱신
- 개선이 안 됐다면 "왜 안 됐는지"도 반드시 기록 (구조적 한계 재확인이라도 그 자체가 유의미한 결과임)
