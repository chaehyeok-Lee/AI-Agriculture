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

