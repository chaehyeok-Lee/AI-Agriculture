# 셀프 피드백 루프 — 26.07.13

## 기준선 (실험 전)
| 타깃 | 4-fold mean | 4-fold std |
|---|---|---|
| soil_moisture | 1.1199 | 0.3768 |
| soil_ec | 0.3106 | 0.2586 |
| soil_temp | 1.0771 | 0.3349 |

---

## 루프 1: lag 피처 추가 (1h/3h/6h 전 X변수값)

**변경**: `add_lag_features()` — temperature_mean, humidity_mean, circ_fan_mean, greenhouse_roof_vent1_mean, co2_mean의 lag12/36/72(1h/3h/6h) 추가

**결과**:
| 타깃 | 변경 전 | 변경 후 | 변화 |
|---|---|---|---|
| soil_moisture | 1.1199 | 1.1451 | +2.2% 악화 |
| soil_ec | 0.3106 | 0.3114 | +0.3% 무변화 |
| soil_temp | 1.0771 | 0.8488 | **-21.2% 개선** |

**판단**: soil_temp는 내부온도 열관성(1~6h)을 lag 피처가 잘 포착. soil_moisture는 day_num(0.72 상관) 중심이라 lag가 노이즈로 작용 → soil_moisture에서 lag 피처 제외(DROP_COLS_PER_TARGET).

---

## 루프 2: 하이퍼파라미터 튜닝 (n_estimators=2000, lr=0.03, num_leaves=63, subsample/colsample=0.8, reg_lambda=0.1)

**결과**: 전체 악화 (soil_moisture +11.4%, soil_ec +13.3%, soil_temp +0.1%)

**판단**: 즉시 롤백. 조기종료 50 rounds와 lr=0.03 조합이 충분히 수렴 안 된 것 추정. 기본 파라미터(lr=0.05, n_est=1000)가 이 데이터셋 규모에 더 적합.

---

## 루프 3: rolling window 피처 (1일/2일 평균) + per-target 제외

**변경**: `add_rolling_features()` — circ_fan_mean, greenhouse_roof_vent1_mean, temperature_mean, humidity_mean의 roll288(1일)/roll576(2일) 추가
- soil_temp에서 rolling 피처 제외 (노이즈로 작용, lag 피처와 중복)
- soil_moisture에서 lag 피처 제외 유지

**결과**:
| 타깃 | 기준선 | 현재 | 누적 변화 |
|---|---|---|---|
| soil_moisture | 1.1199 | **1.1069** | **-1.2% 개선** |
| soil_ec | 0.3106 | **0.3077** | **-0.9% 개선** |
| soil_temp | 1.0771 | **0.8488** | **-21.2% 개선** |

**판단**: rolling 피처로 soil_moisture/soil_ec 소폭 추가 개선. soil_temp는 lag 피처만 사용하는 것이 최적. 모든 타깃 개선 → 유지.

---

## 현재 최적 설정 (3루프 후)
| 타깃 | soil_moisture | soil_ec | soil_temp |
|---|---|---|---|
| lag 피처 | ❌ 제외 | ✅ 포함 | ✅ 포함 |
| rolling 피처 | ✅ 포함 | ✅ 포함 | ❌ 제외 |
| day_num | ✅ 포함 | ✅ 포함 | ❌ 제외 |
| 1d trend 피처 | ✅ 포함 | ✅ 포함 | ✅ 포함 |

---

---

## 루프 4: hour sin/cos 인코딩

**변경**: `add_cyclic_features()` — hour_sin, hour_cos 추가 (자정 연속성 표현)

**결과**: 전체 미미한 변화 (noise 수준). 유지.

---

## 루프 5: early_stopping 50→100 rounds

**결과**: soil_moisture -0.1%, 나머지 동일. 유지.

---

## 루프 6: rolling std 피처 + per-target 조정

**변경**: `add_rolling_features()`에 rollstd288 추가 (팬/온도 1일 변동성). soil_temp rolling mean 제외 유지.

**결과**:
- soil_moisture: 1.1032 → **1.0853** (-1.6% 추가 개선)
- soil_temp: 0.8570 → **0.8451** (-1.4% 추가 개선)
- soil_ec: 거의 동일

---

## 루프 7: lag step 확장 (12h, 1일 추가)

**변경**: `LAG_STEPS = [12, 36, 72, 144, 288]`

**결과**: soil_ec -0.6% 소폭 개선, soil_temp 거의 동일.

---

## 루프 8: circ_fan_accel 피처 (1일-2일 rolling 차이)

**변경**: `add_rolling_features()`에 `circ_fan_accel = roll288 - roll576` 추가

**결과**:
- soil_moisture: 1.0853 → **1.0638** (-2.0% 추가 개선)
- soil_temp: 0.8451 → **0.8392** (-0.7% 추가 개선)

---

## 루프 9: vent1_accel + VPD → 롤백

**결과**: soil_moisture +4.7% 악화. 즉시 롤백.

---

## 루프 10: per-target 모델 파라미터

**변경**: soil_ec에 lr=0.03, n_est=2000 적용 (best_iteration이 66으로 낮아 더 정밀 탐색)

**결과**: soil_ec 미미 개선. 유지.

---

## 루프 11: 4-fold 전체 항상-0 피처 제거

**변경**: 피처 중요도 분석으로 모든 fold에서 importance=0인 피처 식별 → 타깃별 DROP 목록에 추가
- soil_moisture: 16개 제거
- soil_ec: 41개 제거  
- soil_temp: 19개 제거

**결과**: 변화 미미 (noise). 코드 정리 효과. 유지.

---

## 루프 12: humidity_mean_roll864 (3일 rolling)

**결과**: soil_temp -0.9% 추가 개선.

---

## 루프 13: 하이퍼파라미터 시도들 (num_leaves=63, fan_day_interact, cycle_phase)

**결과**: 모두 4-fold에서 악화 → 전부 롤백.
- num_leaves=63 for soil_ec: 소폭 악화
- fan_day_interact: soil_moisture +4.7% 악화
- cycle_phase(14일 주기): soil_moisture +1.3% 악화 (단일 val은 개선이었으나 4-fold 더 신뢰)

---

## 루프 14: greenhouse_roof_vent1_mean_accel 추가

**결과**: soil_moisture **-0.8%** 추가 개선.

---

## 최종 결과 (vs 기준선)

| 타깃 | 기준선 | 최종 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.1199 | **1.0564** | **-5.7%** |
| soil_ec | 0.3106 | **0.3078** | **-0.9%** |
| soil_temp | 1.0771 | **0.8338** | **-22.6%** |

---

## 최종 피처 구성 요약

### 추가된 피처
| 피처 그룹 | 컬럼 수 | 효과 |
|---|---|---|
| lag 1h/3h/6h/12h/1d | 25개 | soil_temp -21%, soil_ec -0.6% |
| rolling 1d/2d mean | 8개 | soil_moisture -1.2%, soil_ec -0.9% |
| rolling 1d std | 4개 | soil_moisture -1.6%, soil_temp -1.4% |
| circ_fan_accel | 1개 | soil_moisture -2.0%, soil_temp -0.7% |
| vent1_accel | 1개 | soil_moisture -0.8% |
| humidity_roll864 | 1개 | soil_temp -0.9% |
| hour_sin/cos | 2개 | 미미 |

### 타깃별 피처 전략
| | soil_moisture | soil_ec | soil_temp |
|---|---|---|---|
| lag 피처 | ❌ 전부 제외 | ✅ | ✅ |
| rolling mean (1d/2d) | ✅ | ✅ | ❌ 제외 |
| rolling std | ✅ | ✅ | ✅ (soil_temp에 유일하게 허용된 rolling) |
| accel 피처 | ✅ | ✅ | ✅ |
| day_num | ✅ | ✅ | ❌ 제외 |
| 0-importance 피처 | ❌ 제외 | ❌ 제외 | ❌ 제외 |

---

---

## 루프 15: Ridge + LightGBM 앙상블 (soil_temp 한정)

**동기**: SUMMARY.md에서 EC 대상 하이브리드가 실패(음수 EC)했지만, soil_temp는 temperature_mean과 선형 상관이 강해서 Ridge의 외삽 능력이 도움될 수 있다는 가설.

**실험 방법**: 
- alpha ∈ {0.1, 1.0, 10.0, 100.0}, w_ridge ∈ 0.0~0.70 그리드 탐색
- 4-fold 교차검증으로 최적 조합 선택

**최적 조합**: alpha=0.1, w_ridge=0.40 (Ridge 40% + LightGBM 60%)

**결과**:
| | LightGBM 단독 | Ridge+LightGBM | 개선 |
|---|---|---|---|
| soil_temp 4-fold mean | 0.8338 | **0.7636** | **-8.4%** |
| fold1 | 0.7895 | 0.6940 | -12.1% |
| fold2 | 0.4665 | 0.3731 | -20.0% |
| fold3 | 0.8955 | 0.8536 | -4.7% |
| fold4 | 1.1835 | 1.1336 | -4.2% |

**모든 fold에서 균등 개선** — 특정 구간 과적합 아님.

**구현**: `BlendModel` 래퍼 클래스로 `.predict()` 인터페이스 유지 → inference.py 구조 변경 없음.
Ridge에서 NaN(lag 초기 구간)은 train 컬럼 median으로 대체 후 StandardScaler 적용.

**검증**: `python3 inference.py` → 3456행 정상 출력, 값범위 통과.

---

## 최종 누적 결과 (기준선 대비)

| 타깃 | 기준선 | 최종 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.1199 | **1.0564** | **-5.7%** |
| soil_ec | 0.3106 | **0.3078** | **-0.9%** |
| soil_temp | 1.0771 | **0.7636** | **-29.1%** |

---

## 미해결 / 향후 시도 여지

1. **soil_ec 계단 구조**: 근본적으로 외삽 문제. rolling/lag으로 최선 다함.
2. **soil_moisture 고분산 fold**: fold1(1.39), fold3(1.42)이 레벨 전환 구간이라 높음. 관리 이벤트 예측은 불가능.
3. **다분광 이미지 피처**: 파장 713~920nm 한계로 EC/수분 직접 측정 불가 확정.

---

## 루프 16: soil_ec day_num "날짜 암기" 과적합 의심 검증 (26.07.13, 3인 전문가 검토)

**동기**: day_num을 트리에 넣으면 특정 날짜 구간을 그대로 암기(과적합)해서, 학습 범위 안(작은 갭)에서는
잘 맞지만 실제 test처럼 갭이 큰 구간에서는 오히려 해로울 수 있다는 가설 제기.

**실험**: 기존 FOLD_CUTOFFS(전부 갭 0~4일)만으로는 이 가설을 검증 못 함 — 일부러 학습 구간을 짧게 끊고
검증 구간을 훨씬 뒤(갭 8~17일, 실제 test 갭과 비슷한 수준)로 배치한 폴드 5개를 추가로 만들어 재검증.

| 갭 | day_num 있음 | day_num 없음 | 승자 |
|---|---|---|---|
| 8~11일 (val126-130) | 0.965 | 0.964 | 거의 동일 |
| 8~11일 (val130-134) | 0.350 | 0.275 | 없음 승(-21%) |
| 11~14일 (val126-130) | 0.915 | 0.915 | 동일 |
| 14~17일 (val128-132) | 0.615 | 0.615 | 동일 |
| 12~15일 (val129-133) | 0.139 | 0.260 | 있음 승(+87%) |
| **평균** | **0.597** | **0.606** | **사실상 무승부** |

**판단**: 초반 2개 폴드만 봤을 땐 "갭 크면 day_num이 해롭다"는 가설이 맞는 것처럼 보였으나(-21% 개선),
5개로 늘리자 방향성이 사라짐(마지막 폴드는 오히려 day_num이 +87% 더 좋음). **폴드 1~2개의 함정을 또 한 번
확인한 사례** — 가설 자체는 합리적이었지만 표본 부족으로 인한 노이즈였음. → **day_num 유지, 변경 없음.**

---

# 셀프 피드백 루프 2 — 26.07.13 (1시간)

## 기준선 (루프 시작 시점)
| 타깃 | 4-fold mean |
|---|---|
| soil_moisture | 1.0564 |
| soil_ec | 0.3065 |
| soil_temp | 0.7636 (Ridge+LightGBM 블렌드 포함) |

---

## [A] soil_moisture Ridge+LightGBM 블렌딩

**결과**: 모든 (alpha, w_ridge) 조합에서 +6~63% 악화. fold3 (126-130일 구간) 최대 RMSE 2.95까지 폭발.

**판단**: soil_moisture는 계단형 비선형 구조라 Ridge 선형 성분이 전환점에서 역효과. 폐기 확정.

---

## [B] 캐노피 밀도 proxy (raw/masked 비율, 새로운 간접 경로)

**아이디어**: 기존 4가지 다분광 실패(절대 밴드값/마스킹/비율/NDRE)는 "밝기 수준"을 봤음.
이번은 (masked_mean / raw_mean) → 어두운 픽셀 비율을 반영하는 새로운 구조 정보.

**결과**: moisture +4.6% ❌, ec 0.0% —, temp +10.6% ❌  →  폐기

---

## [C] EC 전용 상호작용 피처 (humidity×co2, vent1×vent2, fan×vent1)

**결과**: moisture +1.3% ❌, ec 0.0% —, temp +9.4% ❌  →  폐기

---

## [D] B+C 결합

**결과**: 전체 악화 →  폐기

---

## [E] polynomial 피처 (temp², temp×hour, day×humidity)

**결과**: moisture −0.2% —, ec −0.1% —, temp +4.0% ❌

**판단**: moisture/ec는 사실상 중립이지만 soil_temp의 Ridge 성분과 상호작용하여 악화. 폐기.

---

## [F] soil_ec Ridge+LightGBM 블렌딩 (clip≥0 추가)

**결과**: 모든 (alpha, w) 조합에서 +8~77% 악화. EC 계단 구조에 선형 외삽은 무용.

**판단**: SUMMARY.md의 "음수EC 발생" 문제는 clip으로 해결됐지만, 근본적으로 EC가 선형 패턴이 아니라 완전 폐기 확정. 향후 재시도 금지.

---

## [G] Otsu 이진화 기반 캐노피 커버 (plant_fraction, 새로운 접근)

**아이디어**: 픽셀 밝기 15th~90th 퍼센타일 구간 = 식물 픽셀, 그 비율 = 캐노피 커버
기존 방식(밴드 평균값)과 근본적으로 다른 "픽셀 카운트 구조 정보"

**처리**: train 575장, test 240장 (33.8초 + 13.1초). plant_fraction 평균: train=test=0.750

**EC 상관**: loc3_plant_fraction = 0.277 (기존 방법 최고 0.229보다 높음)

**결과**: moisture +1.7% ❌, ec 0.0% —, temp +3.5% ❌

**판단**: EC 상관이 0.277로 높지만 4-fold에서 개선 없음. day_num과 공선형(시간 추세 효과). 폐기.

---

## 루프 2 결론

이번 루프 7가지 실험 모두 채택 실패. **현재 모델이 데이터 구조적 한계에 도달**.

**확인된 사실**:
- soil_ec: 선형 컴포넌트(Ridge) 불가, 상호작용 피처 불가, 이미지 피처 불가 → 현 4-fold 0.3065가 실질적 천장
- soil_moisture: Ridge 블렌딩 불가(계단구조), 이미지 피처 불가 → 현 1.0564 유지
- soil_temp: 이미 최적화됨 (0.7636)
- EC test 예측 다양성: **1,567 고유값** (원본 104 → 현재 1,567), 최다값 1.8% → 실질적 뭉침 해소

**코드 상태**: train.py 수정 없음. 기준선 유지.

---

## 루프 3 추가 실험 — 26.07.13 (루프 2 이후)

### [H] inference.py cold start 버그 수정 (채택)

**발견**: test_X를 단독 처리하면 test DAT135 첫 1일(288행)의 lag/rolling/trend 피처가 NaN
- lag288(1일 lag): 첫 288행 전부 NaN → LightGBM이 훈련 마지막 날 컨텍스트 없이 예측
- trend1d(1일 차분): 첫 289행 NaN
- 4-fold CV는 영향 없음 (train 내부는 연속 데이터라 문제 없음), **test 시점에만 발생**

**수정**: `inference.py`에서 train_X + test_X 연결 후 피처 계산, test 구간만 슬라이싱
```python
combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
combined_feat = build_features(combined_raw) → ... → test_feat = combined_feat[index >= DAT135]
```

**효과**:
- lag 피처 첫 1일 NaN: **38.6% → 0.0%**
- soil_temp 첫 1일 예측 MAE 차이: **0.71** (temperature_mean_lag288 등 핵심 피처 채워짐)
- soil_moisture 첫 1일 MAE 차이: 0.30
- EC: 소폭 변화 (0.014) — lag 의존도 낮아 영향 작음

**검증**: `python3 inference.py` → 3456행, 검증 통과. EC 고유값 1,567→1,542 (거의 동일)

---

### [I] 멀티시드 앙상블 (3/5/7 seeds) — 폐기

**결과**: seed 3/5/7개 전부 단일 seed와 **완전히 동일** (소수점 4자리까지)

**판단**: LightGBM early stopping 적용 시 동일 데이터에서 수렴점이 seed 무관하게 거의 같음. 분산 감소 효과 없음. 폐기.

---

### [J] EC 로그 변환 (log(EC) 학습 → exp 복원) — 폐기

**결과**: EC **+3.3% 악화** (0.3065 → 0.3165), fold3 특히 0.5951→0.6317

**판단**: EC 계단형 구조는 선형 스케일에서 트리가 더 잘 포착. 로그 공간으로 변환하면 계단 구조가 왜곡됨. 폐기.

---

## EC 구조적 분석 (26.07.13 최종)

| fold | val 구간 | EC 상황 | RMSE | 개선 가능성 |
|---|---|---|---|---|
| 1 | 118~122 | 0.55→**1.7 급등** (day 121) | 0.531 | **없음** — 관리 이벤트, X변수 신호 없음 |
| 2 | 122~126 | 1.7 안정 | 0.078 | 이미 우수 |
| 3 | 126~130 | **1.7→0.4 급락** (day 129, 팬 재가동) | 0.595 | **있음** — 팬 ON 신호 존재 |
| 4 | 130~134 | 0.4 안정 | 0.022 | 이미 우수 |

→ fold 3 개선이 유일한 현실적 목표. 팬 재가동 후 EC 급락을 더 잘 포착하는 피처 필요.

---

## 최종 누적 결과 (26.07.12 기준선 대비)

| 타깃 | 기준선 | 최종 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.1199 | **1.0564** | **-5.7%** |
| soil_ec | 0.3106 | **0.3065** | **-1.3%** |
| soil_temp | 1.0771 | **0.7636** | **-29.1%** |

**EC test 예측 고유값**: 104 → **1,567개** (최다값 1.8%), 실질적 뭉침 해소
**inference.py cold start 수정 완료** (lag 피처 첫 1일 NaN 38.6% → 0%)

---

# 셀프 피드백 루프 4 — 26.07.13 (목표: soil_ec 4-fold mean 0.2 달성 시도)

**동기**: 사용자 요청으로 soil_ec mean을 0.2까지 낮출 수 있는지 재시도. 기존 루프에서
"구조적 한계"로 결론났지만, 아직 안 써본 3가지 새 접근을 검증.

## [K] 자기회귀(EC 자신의 과거값) lag 피처, 재귀적(recursive) 추론으로 구현 — 폐기

**동기**: 기존엔 "test에 EC 실측값이 없어서 lag 계산 불가"로 시도조차 안 했음(⚠️ 피처
설계 함정 항목 참고). 하지만 실제 시계열 예측 표준 기법인 **재귀 예측**(1스텝씩 예측 →
그 예측값을 다음 스텝의 lag 입력으로 재사용)을 쓰면 test에서도 계산 가능함을 확인.

**검증 방법**: 4-fold 각각에서, val 구간을 행 단위로 순회하며 soil_ec_lag{36,144,288}
피처를 "이미 예측된 값"으로 재귀적으로 채워가며 예측(진짜 실전과 동일한 조건 재현).

**결과**: fold별 -0.0001~+0.0007, mean 0.3065→0.3062 (사실상 무변화)

**판단**: 급등/급락 당일은 재귀 lag도 "직전 값"만 반영하므로 점프를 예측 못 함(당연히
lag는 과거값이지 미래 신호가 아님). 폐기.

## [L] Huber loss / 최근일 가중치(recency sample weight) — 폐기

**결과**:
| variant | mean |
|---|---|
| baseline | 0.3065 |
| huber loss | 0.3068 |
| recency weight (half-life 5일) | 0.3088 |
| recency weight (half-life 2일) | 0.3170 |

**판단**: 최근일 가중치는 오히려 악화 — fold1/3처럼 "이번이 사실상 처음 보는 전환"인
경우, 오래된 데이터(이전 전환 1회)의 신호를 줄이면 그나마 있던 유일한 참고 사례마저
약화됨. 폐기.

## [M] early stopping 제거(고정 n_estimators) — 폐기, 그러나 메커니즘 규명

**동기**: fold3(cutoff=126) 모델의 best_iteration이 단 28이라 관찰 → in-sample(학습에
쓰인 121~125일)조차 실제 1.7보다 낮은 1.35~1.40으로 과소적합 중임을 발견. early
stopping이 "안정 구간 정확도"를 희생하며 너무 일찍 멈추는 게 아닌지 검증.

**결과**:
| n_estimators | fold1 | fold2 | fold3 | fold4 | mean |
|---|---|---|---|---|---|
| early_stop(현재) | 0.5312 | 0.0780 | 0.5951 | 0.0216 | 0.3065 |
| 고정 2000 (조기종료 없음) | 0.5776 | 0.0778 | 0.6857 | 0.0233 | 0.3411 |

**day별 재확인 (cutoff=126, 고정 2000)**: 126~128일(안정 구간) 예측이 1.73~1.76으로
실제(1.69~1.79)에 훨씬 근접(기존 1.40에서 개선) — 그러나 **129일(급락 당일)** 예측이
1.73(기존 1.40보다 더 나쁨, 실제 0.37)까지 상승 → 안정구간 적합도를 올릴수록 급락
당일 확신(=오답 크기)도 같이 커지는 **편향-확신 트레이드오프**를 확인.

**판단**: early stopping이 "덜 확신하는 모델"을 만들어 급변일의 최악 오차를 억제하는
암묵적 정규화 역할을 하고 있었음. 안정구간 정확도와 급변일 안전마진은 이 데이터로는
동시에 못 잡음 — 구조적으로 얽혀 있어 하이퍼파라미터로 분리 불가. 폐기(현재 설정 유지).

## 루프 4 결론 — 0.2 목표는 현재 데이터/피처로 도달 불가

**fold1(day121)·fold3(day129) 오차의 실체 확인** (day별 예측 vs 실제 직접 대조):
```
cutoff=118 예측: 118~120일(0.639)과 121일(0.639)이 거의 동일 → 121일 급등(1.69)을
                 전혀 못 봄. 모델은 val 구간 내내 "평평한" 값 하나만 예측.
cutoff=126 예측: 126~128일 과소적합(1.40 vs 실제 1.7~1.8) + 129일 급락(0.37)을
                 전혀 못 봄(예측 1.40 유지).
```
→ 두 폴드 모두 "**단 하루짜리, 훈련 데이터에 전례가 최대 1회뿐인 관리/전환 이벤트**"가
val 구간 4일 중 마지막 날에 위치 — RMSE 기여분의 대부분이 이 단 하루에서 나옴
(fold1: day121 하루가 fold RMSE의 약 98% 설명, 계산: sqrt(0.25×1.05²)≈0.52 ≈ 관측 0.531).

**결론**: 이번 루프 3가지 신규 시도(재귀 lag, loss/가중치 변경, 조기종료 제거) 전부
개선 실패 + 메커니즘까지 규명됨(위 [M]) → **0.2 mean은 지금 데이터(X변수만, 실제
시비/관수 이벤트 로그 없음)로는 도달 불가능한 목표**로 최종 판단. 코드 변경 없음
(train.py 프로덕션 설정 그대로 유지, 4-fold mean 0.3065).

**0.2에 도달하려면 필요한 것** (현재 데이터로는 불가):
1. 실제 시비/양액 농도 조정 이벤트 로그 (day121 급등의 유일한 선행 신호일 가능성)
2. 팬 재가동-EC 반응 사이클 다회 관측치 (현재 26일엔 재가동 1회뿐, 재현성 확인 불가)
3. 대안: test 기간이 안정 레짐(day130~134 패턴 지속)이면 실제 test RMSE는 4-fold
   mean(0.31)보다 fold2/4 수준(0.02~0.08)에 훨씬 가까울 가능성 (SUMMARY.md 기존 관찰과 일치)

---

# 셀프 피드백 루프 5 — 26.07.13 (미사용 정형 데이터 컬럼 점검)

**동기**: 사용자 요청으로 train_X.csv 원본 19개 컬럼 중 지금까지 lag/rolling/trend
등 심화 피처가 전혀 적용 안 된 컬럼이 있는지 점검 → 5가지 후보 도출 후 4-fold 검증.

| 후보 | 대상 타깃 | 결과 | 판단 |
|---|---|---|---|
| **wind_direction_outside 순환 인코딩(sin/cos)** | 전체 | moisture -0.29%, temp -0.72%, ec 무변화 | ✅ **채택** |
| solar_radiation lag(1h~1일) | soil_temp | +1.8% 악화 | 폐기 |
| thermal_curtain lag+rolling | soil_temp | +3.1% 악화 | 폐기 |
| greenhouse_roof_vent2 lag+rolling | soil_ec/moisture | ec 무변화, moisture +0.9% 악화 | 폐기 |
| co2_supply/fcu_fan/fogging lag+rolling | soil_ec | 무변화 | 폐기 |

**발견**: `wind_direction_outside`(0~360도)가 지금까지 `hour`(과거엔 마찬가지 문제로
sin/cos 처리됨)와 똑같은 원형 변수 문제를 갖고 있었는데 방치돼 있었음 — 원본 각도값을
그대로 mean/max/min/std/last 집계하면 예를 들어 350도와 10도의 평균이 180도(정반대
방향)로 계산되는 오류가 생김. `add_cyclic_features()`에 `wind_dir_sin`/`wind_dir_cos`
(hour_sin/cos와 동일한 방식, `wind_direction_outside_mean` 기준) 추가로 해결.

**채택 결과**: train.py `add_cyclic_features()`에 반영, 전체 재실행으로 재확인.
`python3 inference.py`도 정상 동작(3456행, 검증 통과) 확인 완료.

| 타깃 | 이전 | 현재 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.0564 | **1.0534** | -0.29% |
| soil_ec | 0.3065 | 0.3065 | 무변화 |
| soil_temp | 0.7636 | **0.7581** | -0.72% |

**결론**: 나머지 4개 컬럼(solar_radiation, thermal_curtain, greenhouse_roof_vent2,
co2_supply/fcu_fan/fogging)은 심화 피처 없이 raw 집계값(mean/max/min/std/last)만으로도
이미 트리가 필요한 정보를 충분히 뽑아내고 있는 것으로 판단 — 추가 lag/rolling은 대부분
노이즈로 작용. 원본 컬럼 중 실제로 처리 방식 자체가 잘못돼 있던 건 wind_direction뿐.



---

# 셀프 피드백 루프 6 - 26.07.14 (circ_fan 진단 + submission 개선)

## 진단 결과

- circ_fan_mean_roll576: 전체 트리(113개) 최초 분기 + gain 압도적 1위(28,049)
- 위상 지연 문제: roll576은 현재 상태 아닌 1-2일 전 상태 반영
- day_num: 100% OOD (test 135-146 모두 train 범위 109-134 밖)
- 날씨: 3일 주기로 2가지 패턴(DAT116, DAT127) 반복

## 실험: circ_fan_mean _ZERO_EC 복원 + 이진 피처 추가

변경: (1) _ZERO_EC에서 circ_fan_mean 제거, (2) circ_fan_on_binary + circ_fan_regime 추가

4-fold CV: moisture=1.0535 / ec=0.3066 / temp=0.7583 (기준선 대비 무변화)

submission 변화:
- EC 고유값: 104개(38.8%) -> 1,410개(2.7%)
- EC 패턴: 팬OFF(138-143일) 0.47->1.67 상승, 팬ON(144+) 0.48 하락
- 물리적으로 올바른 패턴 포착 성공

판단: 유지 (4-fold 무변화 + submission 13.6배 다양화)

---

## 루프 6-B: temperature_mean_roll288 for soil_temp (selective rolling)

변경: DROP_COLS_PER_TARGET["soil_temp"]에서 temperature_mean_roll288/576 제외
- 기존: ROLL_FEATURE_NAMES 전체 드롭 (circ_fan/vent1/temp/humidity 모두)
- 변경: "temperature_mean" 포함된 rolling만 유지 (열관성 신호)

4-fold CV: moisture=1.0535 / ec=0.3066 / temp=**0.7084** (-6.6%)
- fold별: [0.61, 0.364, 0.782, 1.077] ← 전 fold 개선

판단: 유지. temperature_mean_roll288 (24시간 평균)은 temperature_mean_lag36 (3시간 스냅샷)과
보완적으로 작동 — 열관성의 다른 시간 스케일 포착.

---

## 루프 6-C: EC day_num 제거 재시도 (circ_fan 수정 후 재검증)

변경 시도: soil_ec DROP에 day_num 추가

4-fold CV: ec=0.5131 (+67.4% 악화, fold2: 0.078→0.991!)
판단: circ_fan 수정 후에도 day_num이 EC에 필수 (EC 레짐 전환 타이밍 암기). 즉시 롤백.

---

## 루프 6-D: thermal_curtain + solar_radiation trend1d → EC 전용

변경: TREND_SRC_COLS에 thermal_curtain_mean, solar_radiation_mean 추가
     + _ZERO_MOISTURE/_ZERO_TEMP에 두 피처 추가 (EC 전용 격리)

4-fold CV: moisture=1.0535 / ec=**0.3038** (-0.9%) / temp=0.7084
- EC fold3: 0.5951→0.5828 (-2.1%)
- submission EC 고유값: 1,410→1,493

판단: 유지. 온실 관리 신호(커튼 작동, 일사량 변화)가 EC 변화와 직접 연결.

---

## 루프 6-E: ROLL_WINDOWS에 144(12시간) 추가 + 모델별 격리

변경: ROLL_WINDOWS = [144, 288, 576]
      temperature_mean_roll144 → temp 유지, roll144 전체 → moisture 제외

4-fold CV: moisture=1.0535 / ec=0.3038 / temp=**0.6716** (-5.2%)
- temp fold별: [0.624, 0.343, 0.729, 0.991] ← fold4가 1.077→0.991 (-8%)
- submission EC 고유값: 1,493→3,061 (2배!)

판단: 유지. 12시간 온도 rolling이 24시간/48시간 rolling과 보완적으로 온도 모델 개선.

---

## 루프 6-F: temperature_outside_mean lag → EC 전용

변경: LAG_SRC_COLS에 temperature_outside_mean 추가
      temperature_outside_mean_lag* → _ZERO_TEMP (EC 전용 격리)

4-fold CV: moisture=1.0535 / ec=**0.3033** (-0.2%) / temp=0.6716
- submission EC 고유값: 3,061→3,250

판단: 유지. 외기온도 지연값이 EC(EC는 관수/증산과 연동) 소폭 개선.

---

## 루프 6 누적 결과

| 타깃 | 루프5 기준선 | 현재 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.0534 | **1.0535** | ±0% (구조적 한계: 관리 이벤트) |
| soil_ec | 0.3065 | **0.3033** | **-1.0%** |
| soil_temp | 0.7581 | **0.6716** | **-11.4%** |
| EC 고유값 | 1,567 | **3,250** | **+107%** |

실패 실험 목록 (루프6):
- fcu_pump rolling: moisture +19% 악화 (관수 이력 롤링은 노이즈)
- CO2 rolling: moisture +3.5%, temp +2.3% 악화
- humidity rolling for temp: 소폭 악화
- 3-day rolling (864 window): 전체 소폭 악화
- min_child_samples for moisture: 10/30/50 모두 악화
- EC day_num 제거: +67% 악화 (circ_fan 수정 후에도 동일)
- trend clip (OOD 방지): 트리 모델에서 효과 없음 (경계 리프에서 자동 처리됨)

---

# 셀프 피드백 루프 7 — 26.07.14 (디스크 복구 후 LightGBM 하이퍼파라미터 탐색)

## 기준선 (루프 시작 시점 — 루프6 최종 + 이전 세션 wind 개선 포함)
| 타깃 | 4-fold mean |
|---|---|
| soil_moisture | 1.0535 |
| soil_ec | 0.3033 |
| soil_temp | 0.6232 |

(이전 세션에서 wind_speed_outside lags: -1.3%, wind rolling(roll144/288): -4.8%, lag576 for temp: -0.7%, W_RIDGE=0.30: -0.5%가 적용된 상태)

---

## [피처 실험] 실패 실험 목록

| 실험 | 결과 |
|---|---|
| fcu_fan_mean lags → temp | +0.24% 악화 |
| humidity_outside_mean lags → temp | +7.1% 악화 |
| solar_radiation rolling(roll144/288) → temp | +2.9% 악화 |
| vent1 rolling 유지 → temp | +8.4% 악화 |
| co2_mean lags 제거 → temp | +0.5% 악화 |
| circ_fan_mean lags 제거 → temp | +0.3% 악화 |
| temperature_outside lag288 재시도 → temp | +4.1% 악화 |
| temperature_mean_roll864 → temp | +4.7% 악화 |

---

## [하이퍼파라미터 탐색] soil_temp — 핵심 성과

### A. num_leaves=8 채택 (-3.1%)

**실험**: num_leaves=31(기본) → 15 → 10 → 8 → 6 순차 탐색
**결과**:
| num_leaves | soil_temp CV |
|---|---|
| 31 (기본) | 0.6232 |
| 15 | 0.6039 |
| 10 | 0.5914 |
| 8 | **0.5826** |
| 6 | 0.5899 (악화) |

**채택**: num_leaves=8. fold4(0.9203→0.8676) 대폭 개선 — 리프 수 제한이 fold가 짧을수록 효과적.

---

### B. feature_fraction=0.8 채택 (-2.1%)

**실험**: 0.7 / 0.8 / 0.9 탐색
**결과**:
| feature_fraction | soil_temp CV |
|---|---|
| 0.7 | 0.5790 |
| 0.8 | **0.5701** |
| 0.9 | 0.5777 |

**채택**: 0.8. 트리별 피처 무작위 서브샘플링이 토양온도 예측의 다중공선성 노이즈 억제.

---

### C. reg_lambda=3.0 채택 (-1.6%)

**실험**: 1.0 → 2.0 → 3.0 → 4.0 → 5.0 탐색
**결과**:
| reg_lambda | soil_temp CV |
|---|---|
| 0 (기본) | 0.5701 |
| 1.0 | 0.5663 |
| 2.0 | 0.5624 |
| 3.0 | **0.5588** |
| 4.0 | 0.5654 |
| 5.0 | 0.5710 |

**채택**: 3.0. L2 정규화로 리프 예측값 축소 → fold4 고분산 구간 안정화.

---

### D. 실패한 하이퍼파라미터 (num_leaves=8 + feature_fraction=0.8 + reg_lambda=3.0 상태에서)

| 실험 | 결과 |
|---|---|
| subsample=0.8 | +1.0% 악화 |
| min_child_samples=50 | +3.8% 악화 |
| reg_alpha=0.1 | +0.5% 악화 |
| max_bin=63 | +2.2% 악화 |
| learning_rate=0.03 | +2.7% 악화 |
| learning_rate=0.08 | +5.4% 악화 |
| W_RIDGE=0.20 | +1.5% 악화 |
| num_leaves=15 for moisture | +2.9% 악화 |
| feature_fraction=0.8 for moisture | +10.9% 악화 |
| reg_lambda=1.0 for moisture | +2.0% 악화 |
| num_leaves=15 for EC | n_estimators 85↓, +0.6% 악화 |
| feature_fraction=0.8 for EC | n_estimators 264↓, +3.6% 악화 |
| reg_lambda=1.0 for EC | n_estimators 173↓, +0.4% 악화 |
| EC min_child_samples=30/100 | n_estimators 급감, 악화 |

---

## 루프 7 누적 결과

| 타깃 | 루프6 기준선 | 현재 | 개선율 |
|---|---|---|---|
| soil_moisture | 1.0535 | **1.0535** | ±0% |
| soil_ec | 0.3033 | **0.3033** | ±0% |
| soil_temp | 0.6232 | **0.5588** | **-10.3%** |

**soil_temp 전 세션 누적**: 1.0771(최초) → 0.5588 현재 = **-48.1%**

**핵심 발견**: soil_temp는 `num_leaves=8, feature_fraction=0.8, reg_lambda=3.0` 조합으로
fold4(짧은 학습 구간)의 과적합이 대폭 억제됨. EC/moisture에는 모두 역효과 — 타깃별
모델 복잡도 최적 지점이 전혀 다름 확인.

**EC 고유값**: 3,227개 (정상, 팬OFF→팬ON→팬OFF 물리적 EC 패턴 유지)


