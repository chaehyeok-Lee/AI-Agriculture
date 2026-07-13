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

## 다음 시도 예정
1. 시간 주기성 인코딩 (hour_sin, hour_cos) — 일주기 패턴 포착
2. circ_fan 누적 상태 피처 (최근 N시간 동안 팬 가동 비율) — EC regime 포착 강화
3. 피처 중요도 분석 후 노이즈 피처 제거
