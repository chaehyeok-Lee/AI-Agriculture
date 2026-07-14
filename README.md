# 온라인테스트1 — 생육환경 예측 (26.07.13 최종)

```bash
# 실행 순서
python3 train.py       # 모델 학습 → model/model.pkl
python3 inference.py   # 예측    → output/submission.csv

# Docker 제출 (경로 전환 필요 — PLAN.md 참조)
docker build -t submission .
docker run --rm -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" \
  submission sh -c "python train.py && python inference.py"
```

## 과제 개요
테스트 구간(DAT135~146, 12일)의 soil_moisture / soil_ec / soil_temp 예측.
제출 파일은 time 컬럼 기준으로 정답 행과 매칭. 평가: 항목별 raw RMSE → 순위 점수 변환.

---

## 최종 성능 (26.07.13)

| 타깃 | 기준선 4-fold | 최종 4-fold | 개선율 | 비고 |
|---|---|---|---|---|
| soil_moisture | 1.1199 | **1.0564** | -5.7% | rolling/accel 피처 |
| soil_ec | 0.3106 | **0.3065** | -1.3% | 고유값 104→1,567개 |
| soil_temp | 1.0771 | **0.7636** | **-29.1%** | lag + Ridge 앙상블 |

---

## 모델 구조 요약

### 피처 파이프라인 (`train.py` → `inference.py` 동일하게 적용)
```
build_features(raw_X)           # preprocess.py: 1분→5분 집계, clip, day_num/hour
→ add_trend_features()          # X변수 1일 전 대비 변화량 (7개 컬럼)
→ add_lag_features()            # 1h/3h/6h/12h/1일 전 값 (5컬럼 × 5lag = 25개)
→ add_rolling_features()        # 1일/2일 rolling mean+std + accel + humidity 3일
→ add_cyclic_features()         # hour sin/cos 인코딩
```

**⚠️ inference.py 주의**: train_X + test_X를 연결해서 피처 계산 후 test 부분만 슬라이싱.
단독 처리하면 test 첫 1일 lag 피처 38.6%가 NaN — cold start 버그.

### 타깃별 모델
| 타깃 | 모델 | 특이사항 |
|---|---|---|
| soil_moisture | LightGBM (n_est=1000, lr=0.05) | lag 피처 전부 제외 (day_num 중심) |
| soil_ec | LightGBM (n_est=2000, lr=0.03, min_child=50) | 외삽 구조적 한계 |
| soil_temp | **Ridge(α=0.1) × 0.40 + LightGBM × 0.60** | BlendModel 클래스 |

### 타깃별 피처 제외 규칙 (`DROP_COLS_PER_TARGET`)
- **soil_moisture**: lag 피처 25개 전부 제외 + 0-importance 16개
- **soil_ec**: 0-importance 41개만 제거 (day_num/lag/rolling 전부 유지)
- **soil_temp**: day_num 제외 + rolling mean 8개 제외 + 0-importance 19개

---

## 피처별 채택 근거 및 효과

### ✅ add_trend_features (채택)
**이유**: EC/moisture가 환기 스케줄 변화에 계단형 반응. "EC 추세"는 test에 y값 없어 누수 함정 → X변수 변화량으로 대체.  
**효과**: moisture -4.2%, temp -5.5%, ec 무변화(해롭지 않음).

### ✅ add_lag_features (채택, soil_temp 핵심)
**이유**: 토양이 환경 변화에 즉각 반응하지 않음. temperature_mean_lag12/36/72가 배지온도의 "열관성" 포착. 배지온도 = 내부온도에서 열이 전달되는 데 수 시간 걸림.  
**효과**: soil_temp -21.2% (가장 큰 단일 개선). soil_moisture는 오히려 +2.2% 악화 → LAG_FEATURE_NAMES로 제외.  
**lag steps**: 12(1h) / 36(3h) / 72(6h) / 144(12h) / 288(1일) — 5분 격자 기준.

### ✅ add_rolling_features (채택)
**이유**: lag("딱 한 시점")와 달리 rolling은 "최근 N일 평균 상태" — 순간 노이즈에 덜 민감.  
circ_fan_mean_roll288/576으로 환기 "체제(regime)"를 안정적 포착.  
rollstd로 팬/온도 변동성 포착 (0이면 체제 안정, 크면 전환 중).  
accel(1일-2일 차이) = 최근 환기 추세 방향 (양수→팬 가동 증가→EC 하락 기대).  
**효과**: circ_fan_accel로 moisture -2.0%, vent1_accel로 moisture -0.8%, rollstd로 -1.4~1.6%.  
soil_temp는 rolling mean이 노이즈(lag와 중복) → ROLL_FEATURE_NAMES로 제외. rollstd는 포함.

### ✅ add_cyclic_features (채택)
**이유**: hour를 0~24 선형값으로 쓰면 23시→0시가 가장 먼 값. sin/cos로 원형 연속성 표현.
wind_direction_outside(0~360도)도 같은 원형 문제라 26.07.13에 sin/cos 추가.
**효과**: 미미(noise), 해롭지 않아 유지.

### ⚠️ soil_moisture 재현성 재검증 (26.07.13 루프6, FEEDBACK.md 상세)
위 trend/rolling/cyclic의 **soil_moisture 개선폭(-5.7~-5.9%)은 cutoff 세트 하나로만 검증된
수치**였음. cutoff을 ±1~2일 옮긴 4세트로 재확인하니 "4-fold 평균" 기준으로는 셋 다 재현
안 됨(4세트 중 0개 통과 — 원래 세트조차 자체 판정기준 미달). 다만 **leave-one-out**(지금
조합에서 하나씩 빼보는 검증)은 cyclic을 빼면 4개 세트 **전부**에서 더 나빠짐으로 일관됨 →
"확실한 개선"은 아니지만 "지금 조합을 건드리면 검증 가능하게 나빠진다"는 근거로 **코드는
그대로 유지**(train.py 피처/로직 변경 없음). soil_ec/soil_temp는 이 재검증 대상 아님.

### ✅ Ridge + LightGBM 앙상블 for soil_temp (채택, BlendModel)
**이유**: soil_temp가 temperature_mean과 선형 관계가 강해 Ridge의 선형 외삽이 유효. LightGBM은 비선형/시간대 패턴 담당.  
**탐색**: alpha ∈ {0.1, 1.0, 10.0}, w_ridge ∈ 0.0~0.70 그리드 → alpha=0.1, w=0.40 최적.  
**효과**: LightGBM 단독 0.8338 → **0.7636 (-8.4%)**. 모든 fold 균등 개선.  
**구현**: BlendModel 클래스 (predict() 인터페이스 동일 → inference.py 수정 불필요).  
Ridge NaN 처리: train column median으로 대체 후 StandardScaler.

---

## 폐기된 접근법 전체 (재시도 금지)

### 모델/앙상블 관련
| 방법 | 결과 | 이유 |
|---|---|---|
| soil_moisture Ridge 블렌딩 | fold3 RMSE 2.9 폭발 (+46%) | EC 계단구조처럼 moisture도 step 패턴 → 선형화 불가 |
| soil_ec Ridge 블렌딩 (clip≥0) | +8~77% 악화 | EC 4개 레짐(0.4/0.55/0.7/1.7)은 선형 외삽 불가 |
| LightGBM num_leaves=63 전체 적용 | 전체 +9~13% 악화 | 소규모 데이터(7488행)에 과대 복잡도 |
| LightGBM subsample=0.7+colsample=0.7 | EC +9.5% | 배깅이 EC 학습 방해 |
| 멀티시드 앙상블 (3/5/7 seeds) | 완전 동일 (0.0%) | early stopping으로 수렴점이 seed 무관 |
| EC 로그 변환 log(EC) | EC +3.3% | 계단구조는 선형 스케일에서 더 잘 포착됨 |
| HistGradientBoosting + LightGBM | 전체 악화 | HistGBT 단독이 7~13% 열세 |

### 피처 관련
| 방법 | 결과 | 이유 |
|---|---|---|
| cycle_phase (14일 주기 가설) | moisture +1.3% | 단일val만 개선 = 과적합, 4-fold 신뢰 |
| fan_day_interact (circ_fan×day_num) | moisture +4.7% | 교호작용 피처가 noise로 작용 |
| humidity lag만 허용 (moisture) | moisture +1.8% | lag 전부 제거가 더 좋음 |
| VPD 프록시 (temp×humidity) | moisture +1.0% 악화 | 기존 피처와 중복 |
| 팬 5/7일 rolling (1440/2016 steps) | EC fold3 불변 | 훈련에 팬→EC 급락 사이클 없어 학습 불가 |
| 팬 전환 감지 (fan_transition) | EC fold3 불변 | 동일 이유 |
| 팬 누적 OFF 기간 (fan_cumoff) | EC 중립, moisture 소폭 악화 | 동일 이유 |
| polynomial (temp², temp×hour) | soil_temp +4% | Ridge 성분과 충돌 |
| EC 상호작용 (humidity×co2, vent1×vent2) | EC 0.0% 중립 | 기존 피처 이미 포착 |

### 다분광 이미지 (전부 폐기)
**근본 한계**: 카메라 파장 713~920nm (적색에지~근적외선만). 가시광(400~700nm)이나 단파적외선(SWIR, 1200nm+) 없음.

| 방법 | 결과 | 이유 |
|---|---|---|
| 원본 밴드 평균값 | EC/moisture 무변화, temp 악화 | 조명 세기 영향, 배경 미분리 |
| 하위 15% 마스킹 후 평균 | EC/moisture 무변화, temp 악화 | 기준판(밝은 비식물) 미제거 |
| 밴드 비율(적색에지/NIR) | 동일 | 정보량 동일, 스케일만 달라짐 |
| NDRE/CRE/Stress 식생지수 | EC 0.0% 중립 | day_num과 공선형(계절 추세) |
| Otsu 이진화 캐노피 커버 | EC 0.0% 중립 | day_num과 공선형 |
| 캐노피 밀도 proxy (raw/masked 비율) | moisture/temp 악화 | 잡음 추가 |

**알 수 있는 것**: 엽록소 함량(NDRE), 엽면적지수(LAI), 생육 단계. 생육 모니터링 용도에 적합.  
**알 수 없는 것**: EC/염류 스트레스(가시광·SWIR 필요), 수분 스트레스(970nm+).

---

## EC 구조적 한계 (핵심)

EC 4-fold 분포: **[0.53, 0.08, 0.60, 0.02]** — mean 0.3065.

| fold | val 구간 | EC 패턴 | RMSE | 개선 가능성 |
|---|---|---|---|---|
| 1 | 118~122 | **0.55→1.7 급등** (day121) | 0.531 | ❌ 없음 — 시비 결정, X변수 신호 없음 |
| 2 | 122~126 | 1.7 안정 | 0.078 | 이미 우수 |
| 3 | 126~130 | **1.7→0.4 급락** (day129 팬ON) | 0.595 | ❌ 없음 — 훈련에 "팬ON→EC급락" 사이클 없음 |
| 4 | 130~134 | 0.4 안정 | 0.022 | 이미 우수 |

**4-fold mean 0.31은 과대추정**: test(DAT135~146)가 day134의 EC~0.4, 팬ON 상태에서 이어지는 안정 레짐이면 실제 test RMSE ≈ fold4 수준(0.02~0.08).

### 🆕 소프트 블렌드로 test 141~143일 고EC 후보 대응 (26.07.14 루프9)

test_X를 실측 스캔한 결과, **DAT141~143일**에 train의 실제 고EC 구간(121~128일)과
동일한 물리 신호(보온커튼=차광커튼 완전동기화 AND 순환팬 거의 정지)가 재현됨을 발견.
`preprocess.py`의 `compute_ec_high_confidence()`가 이 신호를 연속 신뢰도[0,1]로
계산하고, `train.py`의 `ECBlendModel`이 `예측 = 신뢰도×고EC레짐평균 + (1-신뢰도)×LightGBM예측`
형태로 소프트 블렌드 — 4-fold EC mean 0.3033 → **0.2899(-4.4%)**.

수학적으로 이 블렌드 형태(확률가중평균)는 "참값이 두 레벨 중 하나일 확률분포일 때
오차제곱을 최소화하는 최적해"라 임의의 절충이 아니지만, 게이트 규칙 자체가 train에서
**단 1번(121~128일)만 관측된 패턴(n=1)**이라 test에서 진짜로 재현되는지는 확정 검증
불가 — "확신도만큼만 반영"해 틀렸을 때 손해를 제한하는 리스크 관리 장치이지 정답을
보장하지 않음. 상세: FEEDBACK.md 루프9.

---

작업 요약 — 센서 EDA → 다분광 EDA → 전처리 파이프라인 → Docker/git 설정, 전체 흐름
데이터 구조 — env(1분/5분) + ms(다분광 이미지) 구조 요약
핵심 분석 인사이트 및 주의사항 (제일 비중 크게)
⚠️ 주의해야 할 점 (위험 신호)
greenhouse_roof_vent1: train 범위 030인데 test는 059까지 감 → train이 한 번도 못 본 값 구간. 트리 모델은 이 구간을 제대로 못 다루고 근처 학습값으로 뭉갤 수 있음
wind_speed_outside: train 최대 6.54, test 최대 8.28 → train이 못 본 강풍 구간
내부 temperature: train 최소 1.50, test 최소 -0.20 → train이 못 본 저온 구간. 이 컬럼이 soil_temp와 직접 연관 가능성 높아서 셋 중 가장 위험한 신호
solar_radiation, soil_ec: 원래 분포 자체가 심하게 치우쳐 있어서, IQR 같은 통계 기법을 무심코 적용하면 "맑은 날 정오"처럼 정상적인 값을 전부 이상치로 잘못 판단할 위험이 큼
다분광 위치0: train 7일/test 3일치뿐인 적은 표본이라, 상관관계(0.30~0.37)가 실제보다 과대평가됐을 가능성 있음 — 과신 금지
5분 집계 시 인과관계 방향: pandas resample을 기본값 그대로 쓰면 "지금 시점" 피처에 미래 데이터가 섞여 들어가는 버그가 생김 (이미 발견해서 고쳤지만, 유사한 시계열 집계 코드를 새로 짤 때마다 항상 재확인 필요)
dataset/ vs input/ 경로 혼동: 로컬 개발(dataset/)과 실제 채점 환경(input/dataset/)의 경로가 다름 — train.py/inference.py는 반드시 input/dataset/ 기준으로 짜야 함
💡 중요한 점 (핵심 발견 — 모델링에 꼭 반영)
day_num(경과일수)이 soil_moisture와 상관관계 0.72 — 다분광 이미지보다도 강력한 단일 신호. 반드시 피처로 넣어야 함
soil_moisture 이상치 4개는 사실 하나의 사건이었음 — 같은 날 저녁(DAT128, 19:55~21:40) 수분 급감 + EC 급등 + 온도 상승이 동시에 나타남 → 시비/관수 이벤트로 추정되는 정상 물리 현상, 삭제하면 안 됨
CO2, solar_radiation, 내부온도의 "통계적 이상치"는 전부 정상 물리 현상 (CO2 공급장치 작동, 낮/밤 주기, 계절 추세) — IQR 기준으로 걸러도 실제로 삭제하면 신호의 5~13%를 날리는 꼴이라 오히려 성능이 나빠짐
다분광 이미지가 day_num 대조실험을 통과함 — 위치0의 상관관계가 단순 날짜 우연이 아니라 실제 이미지 정보를 담고 있다는 게 확인됨 (같은 기간 안에서 날짜 자체는 상관관계 거의 0인데 이미지 값은 상관관계 있었음)
구동기 컬럼 정리 규칙 확정: on/off 5개는 0/201→0/1 매핑만 하면 되고, tube_rail_valve는 분산이 0인 죽은 컬럼이라 제거 가능
train_X/train_y의 시간 구조가 완전한 격자(빠진 시각 없음)로 확인되어, 리샘플링을 걱정 없이 안전하게 할 수 있음
🔍 확인해봐야 할 점 (아직 결론 안 남)
다분광 이미지를 최종 모델에 실제로 넣을지 여부 — 상관관계는 확인됐지만 사용 여부는 미결정
위치0 상관관계의 통계적 유의성(작은 표본 대비) 재검증
avg_band/std_band 요약 피처의 실제 결측률 — 위치 2개 이상 동시에 값이 있어야 계산되는데 위치별 결측률이 70~92%라 대부분 NaN일 가능성
카메라가 10비트 센서(최대 1023)로 추정되는데, 실제로 포화(클리핑)된 이미지가 몇 장인지 세어보지 않음
베이스라인 모델 라이브러리 선택 (scikit-learn / LightGBM / numpy 직접구현) — 아직 미정

💡 추가 발견 (26.07.10 오후, 🟡 항목 처리 중)
5분 집계 인과관계 버그 실제로 수정 완료 — resample 결과를 5분 뒤로 shift + 첫 행 reindex 처리. 라벨 T 피처가 이제 (T-5,T] 구간만 담음. python preprocess.py로 train(7488,92)/test(3456,92) 둘 다 재검증 통과
merge_image_features 함수 실행 검증 완료 — dtype 에러 없이 정상 작동 (shape 7488×152)
test 다분광 이미지가 train보다 평균 28% 더 밝음 (train 128.75 vs test 164.98) — 01_eda.py에서 이미 확인한 "test 구간 solar_radiation 평균이 train보다 높다"는 사실과 정확히 교차검증됨 (겨울이라 기온은 낮지만 맑은 날이 많아 일사량 자체는 더 강함). 이미지 피처를 모델에 쓸 경우 train/test 밝기 스케일 차이도 다른 range-mismatch 컬럼들처럼 주의 필요
카메라 최대값이 train/test 둘 다 정확히 1023 — 10비트 센서 추정에 대한 재확인

💡 추가 발견 (26.07.11, 실무자 조언으로 "불량 데이터 제외" 대신 "재활용/보정" 정책 전환 후 확인)
밝기 상/하위 1% 이상치 사진(train 저녁 어두운 사진들, train 정오 밝은 사진, test 위치0 아침 사진) 실제로 열어본 결과 전부 정상 — 저조도에서는 노이즈만 늘 뿐 잎 형태가 살아있고, 손상된 사진은 0장. 제외하지 않고 min-max 정규화로 재활용하기로 결론.
캘리브레이션 기준판 정체 검증 완료 — 진짜 반사 기준판 맞음. 프레임 테두리(마운트 부분)를 제외한 패널면만 정확히 잘라서 위치3 사진 39장(하루 전체 시간대 분포)의 평균 밝기를 전체 이미지 평균 밝기와 비교했더니 상관계수 0.958로 조명 세기에 따라 같이 밝아지고 어두워짐(정상 반사판 거동). 처음 프레임 테두리까지 포함해서 쟀을 때는 "패널이 오히려 새까맣다"는 반대 결과가 나왔었는데, 테두리와 패널면을 분리하니 해소됨 — 이 패널을 기준으로 밝기 이상치 사진을 실제로 보정(정규화)할 수 있다는 근거 확보.

⚠️ 베이스라인 모델 3인 검토로 발견한 위험 신호 (26.07.11) — day_num 외삽 문제
day_num(경과일수)이 soil_moisture와 상관 0.72로 EDA 때부터 핵심 피처로 꼽혔고 실제로 baseline 모델의 feature importance 1위(3개 타깃 전부)였음. 그런데 train(DAT109~134)과 test(DAT135~146) 날짜가 완전히 안 겹침 — 트리 모델은 학습 때 못 본 day_num 범위를 못 다룸(외삽 불가).
1차로 폴드 1개짜리 "먼 갭" 실험만 보고 day_num을 전부 빼기로 했었는데, 4-fold 시계열 교차검증으로 재확인하니 결론이 일부 뒤집힘 — 1차 실험은 갭 크기 효과와 학습데이터 부족 효과가 섞여있던 왜곡된 비교였음. 4-fold 평균 기준 soil_moisture/soil_ec는 day_num이 있는 쪽이 확실히 낫고, soil_temp만 없는 쪽이 나음(day_num 의존도가 낮고 실제 온도 센서값이 더 잘 설명). → 최종적으로 타깃별로 다르게 적용(soil_moisture/soil_ec는 유지, soil_temp만 제외). 교훈: 중요한 피처 결정은 폴드 1개가 아니라 여러 폴드로 재확인해야 함.

🚨 미해결 문제 — soil_ec 실제 test 예측이 심하게 뭉침 (26.07.11, inference.py 첫 실행에서 발견)
val RMSE(0.03~0.06)는 괜찮아 보였는데, 실제 test_X(3456행)로 예측해보니 soil_ec만 고유값이 104개뿐이고 그중 한 값이 전체의 39%(1341행)를 차지, 같은 값이 12일 전체에 걸쳐 반복됨. 실제 정답(train_y)의 soil_ec는 하루 288개가 거의 다 고유값인 매끄러운 연속값이라 이 정도 뭉침은 확실히 비정상. day_num을 빼도 여전히 뭉쳐서(고유값 28개) day_num이 원인은 아님. soil_moisture/soil_temp는 이 문제가 없음. 원인 미상 — val 점수가 좋아도 실제 test에서 이런 문제가 숨어있을 수 있다는 교훈. 예측 다양성(고유값 비율) 체크를 val RMSE와 함께 항상 같이 볼 것.

📡 다분광 밴드(파장)별 의미 — 도메인 지식 (26.07.10 정리)
10개 밴드 파장: 713, 736, 759, 782, 805, 828, 851, 874, 897, 920nm. 전부 적색 경계(red edge)~근적외선(NIR) 영역이고 일반 RGB가 아님 — 작물 생리 상태 관찰용 특화 파장대.
713, 736nm (적색 경계 초입) — 엽록소 흡수대(~680nm)에서 빠져나오는 구간. 엽록소 농도·질소 상태·초기 스트레스에 민감. 상용 농업센서(RapidEye 690~730nm 등)가 흔히 쓰는 red edge 밴드와 겹침.
759nm (적색 경계 → NIR 전환부) — 반사율이 급격히 오르는 변곡점(red edge inflection point) 근처. 이 변곡점 위치 자체가 스트레스 지표 — 건강할수록 긴 파장 쪽, 스트레스받으면 짧은 파장 쪽으로 이동(blue shift).
782~920nm (NIR 평탄부, 8개 밴드) — 색소가 아니라 잎 내부 세포구조·조직 온전성(turgor)을 반영. 897·920nm은 수분 흡수대(970nm 부근)에 가까워 잎 수분함량 영향이 약간 더 섞일 수 있음.
핵심 시사점 — 10개 밴드 중 8개(782~920)가 전부 같은 "NIR 평탄부"라 서로 다른 정보라기보다 같은 신호(잎 구조)를 반복 측정한 것에 가까움. 이게 바로 02_ms_eda.py 상관관계표에서 band1~band10_mean 상관계수가 거의 다 비슷하게 나왔던 이유(물리적으로 설명됨, 우연 아님).
모델링 제안 — 나중에 이미지 피처 쓸 때 10개 밴드를 다 따로 넣기보다 ①713·736(색소/스트레스 그룹) vs ②782~920 평균(구조 그룹) 2그룹으로 압축하거나, 두 그룹 비율(red-edge 지수)을 만드는 게 정보 중복을 줄임. 이 데이터엔 진짜 red(~650nm) 밴드가 없어서 일반 NDVI는 직접 계산 불가.

✅ 위 2그룹 가설 실측 검증 결과 (26.07.10, eda_outputs/correlation_vs_wavelength.png · reflectance_spectrum.png)
반사 곡선(파장별 평균 밴드값)은 이론과 완벽히 일치 — 713→782nm 급상승(적색경계), 782~874nm 평탄부, 897~920nm 재하강(수분흡수대 인접). 전형적인 건강한 식물 분광 곡선.
상관계수는 타깃마다 다름: soil_moisture·soil_ec는 가설대로 713~759nm에서 오르다 782nm부터 완전히 평평(2그룹 구분 유효). soil_temp는 782nm 이후로도 920nm까지 계속 완만히 상승 — 평탄부 안에서도 파장 길수록 더 강해짐. 즉 "2그룹"이 아니라 "적색경계(713~759, 상승) / NIR평탄부(782~874, 평평) / 끝단(897~920, soil_temp에 특히 중요)" 3구간으로 보는 게 더 정확.
→ soil_temp 예측 시엔 band10(920nm)을 NIR 그룹 평균에 묻지 말고 별도 피처로 남기는 걸 권장.

완료된 파일 목록 — 01_eda.py, 02_ms_eda.py, preprocess.py, Dockerfile/requirements.txt/.dockerignore, PLAN.md
작업 방식 메모 갱신 — 기존 "코드 직접 수정 금지" 원칙 유지 확인

세션 재개용 메모 (26.07.10 기준, 대화 기록 없이 이어서 작업할 때 참고)

새 세션 시작하면 제일 먼저 PLAN.md부터 읽을 것 — 진행 상황(트리 구조), 우선순위 액션 리스트(🔴🟡🟢)가 실시간으로 관리되는 파일임. 이 README는 스냅샷이고 PLAN.md가 최신 상태.

지금까지 만들어진 파일과 역할
- 01_eda.py — 센서(env) 데이터 EDA. dt 인덱스 활성화됨, 구동기 분류/이상치 확인 완료
- 02_ms_eda.py — 다분광(ms) 이미지 EDA. 폴더 구조 파싱, 5분 그리드 시간매칭(forward-fill), 화질 점검, day_num 대조실험까지 완료. 실행 결과 캐시는 cache/, 시각화 PNG는 eda_outputs/에 저장됨
- preprocess.py — train.py/inference.py가 공유할 전처리 함수 모음. dt변환/구동기매핑/클리핑/day_num,hour/1분→5분 집계/train-val 시간순 분할까지 구현 및 검증 완료 (python preprocess.py로 재검증 가능)
- train.py, inference.py — 아직 hello-world 샘플 상태. 다음 작업이 여기(4~5단계, 베이스라인 모델)
- Dockerfile, requirements.txt, .dockerignore — python:3.13-slim 기준으로 실제 docker build/run 검증 완료 (hello-world 기준)
- PLAN.md — 전체 진행 조직도 + 우선순위 액션 리스트 (제일 먼저 확인할 파일)

환경 정보
- 로컬 파이썬: .venv 안에 pandas 3.0.3 / numpy 2.5.1 / matplotlib 3.11.0 / seaborn 0.13.2 설치됨
  (26.07.11: .venv 재생성 + lightgbm 4.6.0 / scikit-learn 1.9.0 추가 설치, requirements.txt 반영됨)
- Docker Desktop, Git 설치 완료 (버전은 각각 docker --version, git --version으로 확인)
- git 저장소 초기화 및 첫 커밋 완료. GitHub 원격 저장소: https://github.com/chaehyeok-Lee/AI-Agriculture.git (푸시는 본인이 수기로 진행)

데이터 경로 규칙 (헷갈리기 쉬운 부분, 26.07.11 정정)
- dataset/... — 로컬 전체 실데이터(38일치, 실제 위치명). EDA·preprocess.py·train.py·inference.py 전부 지금 이 경로로 개발/검증 중
- input/dataset/... — 현재 비어있음. 원래 "대회측 샘플(1일치, P1_ 위치명)"이라고 알고 있었으나, 실제로 이 경로에 들어있던 데이터가 위 dataset/의 전체 실데이터였던 것으로 확인되어 dataset/로 옮김. Docker 채점 시 그라더가 이 경로에 자기 데이터를 마운트하는 구조로 추정 — train.py/inference.py는 Docker 제출 직전에 이 경로 기준으로 바꿔야 함(아직 미착수, PLAN.md 4/5단계 참고)

다음에 시작할 지점 (26.07.11 기준 갱신)
- 4단계(베이스라인 train.py)와 5단계(inference.py) 둘 다 1차 완료. LightGBM, 타깃별 3모델,
  val RMSE: soil_moisture 1.0122 / soil_ec 0.0337 / soil_temp 1.2489
- 다분광(ms) 이미지는 실제로 넣어서 val 비교까지 해봤고, 개선 없어서 미반영으로 확정됨(재시도는 마스킹/보정 이후)
- 미해결로 남은 것: soil_ec가 실제 test 예측에서 심하게 뭉치는 문제(원인: train/test 계절차로 인한
  광범위 분포 이동으로 추정, 수정 시도 2번 다 실패) — 알려진 한계로 문서화, 정답 확보 전까지 보류
- 다음 할 일: submission.csv 포맷 assert 검증, input/dataset/ 경로 전환, Docker 재현 검증(6단계)

작업 규칙 추가 (26.07.10) — 앞으로 모델링에 유의미한 분석 결과가 새로 나올 때마다, 이 README(핵심 분석 인사이트 섹션)에 자동으로 반영하기로 함. PLAN.md는 작업 진행상황 추적용, README는 데이터 인사이트 축적용으로 역할 분리.