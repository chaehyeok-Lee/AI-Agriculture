import pandas as pd

# 시간 컬럼 통일 (날 + 시간)
# 1. 데이터 불러오기
train_X = pd.read_csv("dataset/train/env/train_X.csv")
train_y = pd.read_csv("dataset/train/env/train_y.csv")
test_X = pd.read_csv("dataset/test/env/test_X.csv")

# # 2. "DAT109 00:00" 같은 time 문자열을 timedelta로 변환하는 함수
# def make_dt_index(df):
#     dat_num = df["time"].str.extract(r"DAT(\d+)")[0].astype(int)  # 며칠째
#     hm = df["time"].str.split(" ").str[1]                          # 그날의 시:분
#     dt = pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(hm + ":00")
#     return df.assign(dt=dt).set_index("dt").sort_index().drop(columns="time")

# # 3. dt(timedelta) 인덱스로 통일
# train_X = make_dt_index(train_X)
# train_y = make_dt_index(train_y)
# test_X = make_dt_index(test_X)

# # 4. 확인
# print(train_X.index.min(), train_X.index.max())
# print(train_y.index.min(), train_y.index.max())
# print(test_X.index.min(), test_X.index.max())

# #  "결과 파일을 저장"하는 게 아니라 "변환 로직을 재사용 가능한 함수로 고정"해두는 게 실무 방식입니다. 
# # 지금 단계(결측치/이상치 확인)에서는 01_eda.py에서 계속 실험만 하시고, EDA가 끝나고 전처리 방식이 확정되면 그때 utils.py로 
# #옮겨서 train.py/inference.py가 공유하도록 만드는 흐름을 추천드립니다

# actuator_cols = [
#     "greenhouse_roof_vent1", "greenhouse_roof_vent2",
#     "shading_curtain", "thermal_curtain",
#     "fcu_fan", "fcu_pump", "circ_fan",
#     "co2_supply", "tube_rail_valve", "fogging",
# ]

# for col in actuator_cols:
#     print(col, "->", sorted(train_X[col].unique())[:10], "... nunique:", train_X[col].nunique())

#     # 6. 개도율 타입 컬럼의 실제 min/max 확인
# ratio_cols = ["greenhouse_roof_vent1", "greenhouse_roof_vent2", "shading_curtain", "thermal_curtain"]
# for col in ratio_cols:
#     print(col, "-> min:", train_X[col].min(), "max:", train_X[col].max())

# # 7. test_X에서도 같은 구동기 컬럼들이 어떤 값을 갖는지 확인 (train과 비교)
# actuator_cols = [
#     "greenhouse_roof_vent1", "greenhouse_roof_vent2",
#     "shading_curtain", "thermal_curtain",
#     "fcu_fan", "fcu_pump", "circ_fan",
#     "co2_supply", "tube_rail_valve", "fogging",
# ]

# print("\n=== test_X 구동기 컬럼 값 확인 ===")
# for col in actuator_cols:
#     print(col, "-> unique:", sorted(test_X[col].unique())[:10], "... nunique:", test_X[col].nunique(),
#           "min:", test_X[col].min(), "max:", test_X[col].max())


# train 결과까지 나오니 중요한 게 하나 보입니다.

#greenhouse_roof_vent1 — train/test 범위가 다릅니다 (주의 필요)

#train: min 0.0 ~ max 30.0
#test: min 0.0 ~ max 59.0
#즉 test 구간에서 천창1이 train에서 한 번도 본 적 없는 31~59 사이의 값으로 열린 적이 있다는 뜻입니다. 모델은 train으로 학습하기 때문에 이 범위는 "본 적 없는 입력"이 되고, 트리 기반 모델이면 그 범위를 잘 다루지 못하고 근처 학습값으로 뭉개버릴 수 있고, min-max 스케일링 계열이면 범위 밖 값이라 스케일이 깨질 수 있습니다. → 이건 이상치라기보다 "train이 test의 실제 동작 범위를 다 커버하지 못한다"는 구조적 리스크로 봐야 합니다.

#greenhouse_roof_vent2 — train/test 둘 다 max 30.0으로 일치 문제 없습니다. vent1과 달리 vent2는 같은 범위를 씁니다.

#shading_curtain, thermal_curtain — train/test 둘 다 0~100 일치 개도율(%) 해석이 맞고, 범위 불일치도 없어서 안전합니다.

#on/off 컬럼 5개(fcu_fan/pump, circ_fan, co2_supply, fogging) — train/test 전부 {0, 201}로 일치
#전처리 시 201→1 매핑 하나만 만들면 됩니다.

#tube_rail_valve — train/test 전 구간 0.0 고정, 최종 확정 쓸모없는 컬럼(zero-variance)입니다.

# 8. 날씨/내부환경 컬럼 train vs test 범위 비교
# env_cols = [
#     "temperature_outside", "humidity_outside", "solar_radiation",
#     "wind_direction_outside", "wind_speed_outside", "rainfall",
#     "temperature", "humidity", "co2",
# ]

# for col in env_cols:
#     tr_min, tr_max = train_X[col].min(), train_X[col].max()
#     te_min, te_max = test_X[col].min(), test_X[col].max()
#     print(f"{col:25s} train[{tr_min:.2f}, {tr_max:.2f}]  test[{te_min:.2f}, {te_max:.2f}]")

# temperature_outside: train [-5.7, 12.3] vs test [-6.45, 15.0] → 양쪽 다 살짝 벗어남. 폭이 작아서 치명적이진 않음.
# wind_speed_outside: train max 6.54 vs test max 8.28 → test에서 더 강한 바람이 붊. train이 못 본 구간.
# temperature(내부 온도): train min 1.50 vs test min -0.20 → 내부 온도가 train에서 한 번도 본 적 없는 영하까지 떨어짐. 이건 신경 써야 합니다 — 이 컬럼이 목표값(soil_temp)과 직접 관련이 클 가능성이 높은데, 모델이 학습 못 한 저온 구간에서 예측을 해야 하는 상황이라 오차가 커질 수 있습니다.
# humidity(내부): train max 98 vs test max 100 → 거의 근접, 미미한 수준.

# 종합 해석: test 기간(겨울)은 전반적으로 "일사량 낮고, 비 없고, CO2는 낮은" 안정적인 방향이라 
# 대부분 컬럼은 오히려 train보다 좁은 범위라 안전합니다. 다만 내부 온도가 train에서 못 본 저온까지 떨어지는 것, 
# 바람이 더 세게 부는 것, 그리고 이전에 찾은 greenhouse_roof_vent1(30→59) 
# 이 세 가지가 "test가 train보다 더 극단적인 값을 갖는" 진짜 위험 신호입니다. 
# 이 세 컬럼은 나중에 모델링할 때 클리핑하거나, 스케일링 기준을 train만이 아니라 train+test 전체 범위로 잡는 등 대비가 필요합니다.

# 9. 컬럼별 결측치 개수 확인
# print("=== train_X 결측치 ===")
# print(train_X.isna().sum())

# print("\n=== train_y 결측치 ===")
# print(train_y.isna().sum())

# print("\n=== test_X 결측치 ===")
# print(test_X.isna().sum())

#결측치 all = 0

# 10. 컬럼별 기술통계로 물리적 이상값 확인
# print("=== train_X describe ===")
# print(train_X.describe())

# print("\n=== train_y describe ===")
# print(train_y.describe())

# print("\n=== test_X describe ===")
# print(test_X.describe())

# 1) 내부 습도(humidity)가 test에서 정확히 100.0 찍음
# 습도 센서가 물리적으로 100%(포화)에 닿은 건데, 진짜 100%일 수도 있지만 센서가 100%에서 막혀서(clip) 그 이상 값을 못 찍은 걸 수도 있습니다. 나중에 이 값이 얼마나 자주 100으로 찍히는지(연속으로 붙어있는지) 확인해볼 가치는 있습니다.

# 2) soil_ec(train_y) — 오른쪽으로 크게 치우친 분포
# 평균 0.91, 표준편차 0.55인데 max가 2.92 → 평균에서 표준편차 3.6배 넘게 떨어진 값입니다. 25%(0.505)~75%(1.617) 사이 폭도 크고요. 물리적으로 틀린 값은 아니지만, 다음 단계(IQR 이상치 확인)에서 이 컬럼은 정상적으로 꽤 많은 값이 "통계적 이상치"로 잡힐 가능성이 높습니다 → 물리적 이상치가 아니라 원래 그런 분포라는 걸 구분해야 합니다.

# 3) solar_radiation — 평균(79.7)보다 표준편차(132.8)가 더 큼
# 낮/밤 주기 때문에 0인 값이 절반 가까이 되고 낮에만 값이 튀는 구조라 원래 이렇게 치우칩니다. 이 컬럼에 IQR을 그대로 적용하면 "맑은 날 정오"처럼 정상적인 고일사량 시간대가 전부 이상치로 잘못 잡힐 수 있어서 주의가 필요합니다.

# 4) 앞서 찾은 위험 신호들이 여기서도 재확인됨

# 내부 온도(temperature) test min -0.20 (train은 1.50부터) — 영하까지 떨어짐, train이 못 본 구간
# wind_speed_outside test max 8.28 (train은 6.54까지) — 역시 train이 못 본 구간

# 11. IQR 기준 통계적 이상치 확인
# def iqr_outlier_count(df, cols):
#     result = {}
#     for col in cols:
#         q1 = df[col].quantile(0.25)
#         q3 = df[col].quantile(0.75)
#         iqr = q3 - q1
#         lower = q1 - 1.5 * iqr
#         upper = q3 + 1.5 * iqr
#         n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
#         result[col] = (lower, upper, n_outliers)
#     return result

# check_cols_X = ["temperature_outside", "humidity_outside", "solar_radiation",
#                 "wind_speed_outside", "temperature", "humidity", "co2"]
# check_cols_y = ["soil_moisture", "soil_ec", "soil_temp"]

# print("=== train_X IQR 이상치 ===")
# for col, (lower, upper, n) in iqr_outlier_count(train_X, check_cols_X).items():
#     print(f"{col:25s} 정상범위[{lower:8.2f}, {upper:8.2f}]  이상치 개수: {n}")

# print("\n=== train_y IQR 이상치 ===")
# for col, (lower, upper, n) in iqr_outlier_count(train_y, check_cols_y).items():
#     print(f"{col:25s} 정상범위[{lower:8.2f}, {upper:8.2f}]  이상치 개수: {n}")

# # 12. 습도 100 포화(clip) 여부 확인
# print("\ntrain_X humidity==100 개수:", (train_X["humidity"] == 100).sum())
# print("test_X humidity==100 개수:", (test_X["humidity"] == 100).sum())

# def make_dt_index(df):
#     dat_num = df["time"].str.extract(r"DAT(\d+)")[0].astype(int)
#     hm = df["time"].str.split(" ").str[1]
#     dt = pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(hm + ":00")
#     return df.assign(dt=dt).set_index("dt").sort_index().drop(columns="time")

# train_X = make_dt_index(train_X)
# train_y = make_dt_index(train_y)
# test_X = make_dt_index(test_X)

# solar_radiation 4,715개: 예상대로. 낮/밤 주기 때문에 맑은 대낮 값들이 "IQR 기준 정상범위 위"로 잡힌 것뿐. 실제 이상치 아님.
# co2 2,298개: CO2 공급 장치(co2_supply)가 작동할 때 CO2 농도가 확 오르는 정상적인 운영 이벤트. 이상치 아님.
# temperature(내부) 3,935개: 낮/밤 온도 변화가 커서 IQR 밴드를 벗어나는 것. 온실이니 주야 온도차가 큰 건 자연스러움.
# wind_speed_outside 620개: 가끔 강하게 부는 바람 — 돌풍 이벤트, 정상 범위.
# soil_temp 352개(4.7%): 26일 동안 계절/날짜에 따라 배지 온도가 서서히 변한 걸로 보임 — 하루 중 변화라기보다 기간에 걸친 자연스러운 추세.
# → 이 5개 컬럼은 "이상치 제거" 하면 안 됩니다. 실제 신호를 5~13%씩 날려버리는 꼴이라 오히려 모델 성능을 해칩니다.

# 중요한 검증 — soil_ec 이상치 0개
# 지난번 "오른쪽으로 치우친 분포라 이상치 많이 잡힐 것"이라 예상했는데, 결과는 정반대로 0개입니다. IQR이 Q1~Q3 폭(1.11) 자체를 넓게 잡아서 max(2.92)까지 정상범위(~3.28) 안에 다 들어옵니다. 즉 skew가 있어도 "일관되게 치우친 분포"라 IQR 계산 자체엔 문제 없었다는 뜻 — 지난번 우려가 실제로는 기우였다는 게 확인됐습니다.

# 진짜 눈여겨볼 것 — 개수가 적어서 직접 확인 가능한 것들

# soil_moisture 이상치 4개: 7,488개 중 4개면 개별로 뜯어봐도 되는 수준. 급격한 관수(irrigation) 이벤트인지 센서 튐인지 확인 가치 있음.
# 내부 습도 100.0 — test에서 523번 (전체 17,280분의 약 3%). train은 0번인데 test(겨울)에서만 나타남. 결로/포화로 인한 진짜 100%인지, 연속으로 오래 붙어있어서 센서가 막힌 건지 패턴을 봐야 함.

# 13. soil_moisture 이상치 4개 직접 확인
q1 = train_y["soil_moisture"].quantile(0.25)
q3 = train_y["soil_moisture"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

outlier_rows = train_y[(train_y["soil_moisture"] < lower) | (train_y["soil_moisture"] > upper)]
print("=== soil_moisture 이상치 행 ===")
print(outlier_rows)

# 14. 내부 습도 100% 연속 구간(run) 확인 — 센서 고착 여부
is_sat = (test_X["humidity"] == 100)
group_id = (is_sat != is_sat.shift()).cumsum()
run_lengths = is_sat.groupby(group_id).sum()
run_lengths = run_lengths[run_lengths > 0]

print("\n=== 습도 100% 연속 구간 ===")
print("구간 개수:", len(run_lengths))
print("가장 긴 연속 구간(분):", run_lengths.max())
print(run_lengths.describe())

# EDA 요약

# 구조: train_X(1분,26일)/train_y(5분,26일)/test_X(1분,12일), dt(timedelta)로 시간 통일 완료, train→test 연속됨
# 결측치: 전 컬럼 0개
# 물리적 이상값: 없음
# 구동기: on/off(0/201) 5개, 개도율(0~100) 2개, vent1/2는 제한적 범위, tube_rail_valve는 죽은 컬럼(항상 0)
# train/test 범위 차이(위험 신호): roof_vent1(30→59), wind_speed(6.5→8.3), 내부온도(1.5→-0.2) — test가 더 극단적
# IQR 이상치: 대부분 정상 패턴(낮/밤, CO2공급, 배지건조, 야간결로)이라 삭제할 이상치 없음

