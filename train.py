"""온라인테스트1 학습: LightGBM + Ridge 앙상블 (타깃별 3개 모델). model/model.pkl 에 저장.

경로 규칙(preprocess.py와 동일): 로컬 검증은 dataset/... 사용.
Docker 제출용으로 바꿀 때는 input/dataset/...로 교체 필요 (PLAN.md 4단계 항목).

================================================================================
피처 엔지니어링 개요 및 실험 결과 (26.07.13 최종, 상세는 FEEDBACK.md 참조)
================================================================================

[기준선 → 최종 4-fold RMSE]
  soil_moisture : 1.1199 → 1.0564  (-5.7%)
  soil_ec       : 0.3106 → 0.3065  (-1.3%)   EC 고유값: 104 → 1,567개 (뭉침 해소)
  soil_temp     : 1.0771 → 0.7636  (-29.1%)

[채택된 피처 그룹]
  1. add_trend_features  — X변수 1일 전 대비 변화량 (7개 컬럼 × 1)
     채택 이유: "EC 추세" 피처는 test에 y값이 없어 계산 불가(누수 함정). X변수 변화량으로 대체.
     효과: moisture -4.2%, temp -5.5%, ec 무변화.
     폐기: "최근 EC 추세" 피처 → test에 정답 없어 불가.

  2. add_lag_features — 1h/3h/6h/12h/1일 전 X변수값 (5개 컬럼 × 5 lag = 25개)
     채택 이유: 토양이 환경 변화에 즉각 반응하지 않음. temperature_mean lag12/36/72가
                soil_temp의 "열관성(thermal inertia)"을 포착 → 가장 큰 개선 요인.
     효과: soil_temp -21.2%. soil_moisture는 노이즈(LAG_FEATURE_NAMES로 제외).
     폐기 후보: humidity lag만 moisture에 허용 → 오히려 +1.8% 악화, 전부 제외 확정.

  3. add_rolling_features — 1일/2일 rolling mean + rollstd + circ_fan_accel + vent1_accel
     채택 이유: circ_fan_mean_roll288/576으로 환기 "체제(regime)"를 안정적으로 포착.
                lag가 "딱 한 시점"이라면 rolling은 "최근 N일 평균 상태"를 반영.
                accel(1일-2일 차이) = 최근 추세 방향, 팬 가동 증감 신호.
     효과: moisture -1.2%, ec -0.9%, rollstd 추가로 moisture -1.6%/temp -1.4%,
           circ_fan_accel로 moisture -2.0%/temp -0.7%, vent1_accel로 moisture -0.8%.
     soil_temp는 rolling mean 제외(ROLL_FEATURE_NAMES) — rollstd/accel은 예외적 포함.
     폐기: vent1_accel + VPD 결합 → moisture +4.7% 악화 / fan_day_interact → moisture +4.7%.
     폐기: 5일/7일 rolling(roll1440/2016) → EC fold3 변화 없음, moisture 악화.
     폐기: 팬 전환 감지(fan_transition_6h/12h/24h) → EC fold3 변화 없음.
     폐기: 팬 누적 OFF 기간(fan_cumoff) → EC 중립, moisture 소폭 악화.

  4. add_cyclic_features — hour sin/cos 인코딩
     채택 이유: hour를 0~24 선형값으로 쓰면 트리가 23시와 0시를 "가장 먼 값"으로 인식.
                sin/cos로 변환하면 원형 연속성 표현.
     효과: 미미(noise 수준), 해롭지 않아 유지.
     폐기: cycle_phase(14일 주기 가설) → moisture +1.3% 악화. 단일val만 개선 = 과적합 신호.

[폐기된 접근 전체 요약]
  - LightGBM num_leaves=63 + subsample/colsample → 전체 악화(+9~13%)
  - soil_moisture Ridge+LightGBM 블렌딩 → fold3에서 RMSE 2.9까지 폭발(+46%), 계단구조 불가
  - soil_ec Ridge+LightGBM 블렌딩(clip≥0 포함) → 전체 +8~77% 악화, 계단구조에 선형 외삽 무용
  - 다분광 NDRE/CRE/Stress/delta/detrend → EC 0.0% 중립, 카메라 파장(713~920nm) 한계
  - Otsu 이진화 캐노피 커버 피처 → EC 0.0% 중립 (day_num 공선형)
  - 다분광 캐노피 밀도 proxy(raw/masked 비율) → moisture/temp 악화
  - EC 상호작용 피처(humidity×co2, vent1×vent2) → EC 0.0% 중립, 나머지 악화
  - polynomial 피처(temp², temp×hour) → soil_temp Ridge 성분과 충돌, 악화
  - 멀티시드 앙상블(3/5/7 seeds) → 완전 동일(early stopping 덕에 수렴점 같음)
  - EC 로그 변환 → +3.3% 악화(계단구조가 선형 스케일에서 더 잘 포착됨)
  - HistGradientBoosting + LightGBM 앙상블 → 전체 악화(HistGBT 단독 7~13% 열세)
  - EC day_num 제거 → +70.4% 급격 악화
  - EC 5/7일 rolling → EC fold3 불변(0.5951), 구조적 한계 확인

[EC 구조적 한계]
  fold 분포 [0.53, 0.08, 0.60, 0.02] — fold 1(day121 관리 이벤트)과 fold 3(day129 팬
  재가동)이 고RMSE. fold 1은 X변수로 예측 불가능한 시비 결정. fold 3은 훈련 데이터
  (109~126)에 "팬 재가동 → EC 급락" 사이클이 없어 어떤 모델도 학습 불가.
  EC 4-fold 0.31은 과대추정 — test(135~146)가 안정 레짐이면 실제 RMSE ≈ 0.02~0.08 예상.
================================================================================
"""
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from preprocess import build_features, load_target, time_based_split

RANDOM_STATE = 42
TARGET_COLS = ["soil_moisture", "soil_ec", "soil_temp"]

# X변수(구동기/날씨) 추세 피처: soil_ec가 circ_fan/천창 스케줄 변화(109~115 ON, 115~129 OFF, 129~134 ON)에
# 계단형으로 반응한다는 분석(SUMMARY.md)에 착안. "최근 EC 추세"는 test에 정답이 없어 계산 불가(누수 함정)라
# 대신 값을 아는 X변수의 "1일 전 대비 변화량"을 피처로 추가. 4-fold 검증에서 soil_moisture -4.2%,
# soil_temp -5.5% 개선, soil_ec는 무변화(해롭지 않음) 확인 — 26.07.12.
TREND_SRC_COLS = [
    "circ_fan_mean", "greenhouse_roof_vent1_mean", "greenhouse_roof_vent2_mean",
    "temperature_outside_mean", "humidity_outside_mean", "humidity_mean", "co2_mean",
]
TREND_WINDOW = 288  # 5분 격자 기준 288칸 = 1일

# lag 피처: 토양 반응 지연 포착 — soil_temp는 내부온도 열관성(1~6h)에 효과적
# soil_moisture는 day_num(0.72 상관) 중심이고 lag 피처가 노이즈로 작용해 제외 — 26.07.13 실험 확인
LAG_SRC_COLS = [
    "temperature_mean", "humidity_mean",
    "circ_fan_mean", "greenhouse_roof_vent1_mean", "co2_mean",
]
LAG_STEPS = [12, 36, 72, 144, 288]  # 1h, 3h, 6h, 12h, 1일 (5분 격자 기준)
LAG_FEATURE_NAMES = [f"{col}_lag{lag}" for col in LAG_SRC_COLS for lag in LAG_STEPS]
# soil_moisture용: humidity lag만 허용(관수/증산 지연), 나머지 lag는 노이즈
MOISTURE_LAG_EXCLUDE = [f"{col}_lag{lag}" for col in LAG_SRC_COLS for lag in LAG_STEPS
                        if col != "humidity_mean"]

# rolling window 피처: circ_fan 1일/2일 평균으로 환기 "체제(regime)" 포착 → soil_ec 계단 구조 반영
# soil_temp는 rolling이 노이즈로 작용해 DROP_COLS_PER_TARGET에서 제외 — 26.07.13 실험 확인
ROLL_SRC_COLS = [
    "circ_fan_mean", "greenhouse_roof_vent1_mean",
    "temperature_mean", "humidity_mean",
]
ROLL_WINDOWS = [288, 576]  # 1일, 2일 (5분 격자 기준)
ROLL_FEATURE_NAMES = [f"{col}_roll{w}" for col in ROLL_SRC_COLS for w in ROLL_WINDOWS]

# day_num(경과일수): train(DAT109~134)과 test(DAT135~146) 날짜가 안 겹쳐서 트리 모델이 외삽해야 함.
# 4-fold 시계열 교차검증(expanding window)으로 타깃별 확인한 결과 soil_moisture/soil_ec는
# day_num이 있는 쪽이 확실히 낫고(외삽 위험보다 신호 가치가 큼), soil_temp만 없는 쪽이 나음
# (day_num 의존도가 낮고 실제 온도 센서값이 더 잘 설명함) — 26.07.11 실험 확인.
#
# 아래 3개 리스트(_ZERO_*): 학습된 모델의 feature_importances_를 4-fold 전체에서 뜯어봤을 때
# "단 한 번도 분기(split) 기준으로 안 쓰인" 피처들. 트리가 전혀 안 쓰는 값이므로 넣어도
# 성능에 영향은 없지만, 컬럼 수를 줄여서 학습을 조금 더 가볍고 깔끔하게 만들려고 제거함
# (max/min/std처럼 분산이 거의 없는 통계량, 또는 애초에 원본이 0/거의 상수인 컬럼들 위주).
_ZERO_MOISTURE = [
    "circ_fan_max", "circ_fan_min", "co2_supply_max", "co2_supply_min",
    "fcu_fan_max", "fcu_fan_min", "fcu_pump_max", "fcu_pump_min",
    "fogging_last", "fogging_max", "fogging_min", "fogging_std",
    "greenhouse_roof_vent1_max", "greenhouse_roof_vent1_min",
    "greenhouse_roof_vent2_max", "greenhouse_roof_vent2_min",
]
_ZERO_EC = [
    "circ_fan_last", "circ_fan_max", "circ_fan_mean", "circ_fan_mean_lag12",
    "circ_fan_mean_lag144", "circ_fan_mean_lag288", "circ_fan_mean_lag72",
    "circ_fan_mean_trend1d", "circ_fan_min", "co2_supply_last",
    "co2_supply_max", "co2_supply_min", "fcu_fan_max", "fcu_fan_min",
    "fcu_pump_last", "fcu_pump_max", "fcu_pump_min", "fcu_pump_std",
    "fogging_last", "fogging_max", "fogging_min", "fogging_std",
    "greenhouse_roof_vent1_max", "greenhouse_roof_vent1_min",
    "greenhouse_roof_vent2_last", "greenhouse_roof_vent2_max",
    "greenhouse_roof_vent2_mean", "greenhouse_roof_vent2_min",
    "greenhouse_roof_vent2_std", "greenhouse_roof_vent2_mean_lag12",
    "greenhouse_roof_vent2_mean_lag36", "greenhouse_roof_vent2_mean_lag72",
    "greenhouse_roof_vent2_mean_lag144", "greenhouse_roof_vent2_mean_lag288",
    "greenhouse_roof_vent2_mean_trend1d",
    "humidity_outside_mean_lag12", "humidity_outside_mean_lag36",
    "humidity_outside_mean_lag72", "humidity_outside_mean_lag144",
    "humidity_outside_mean_lag288", "wind_speed_outside_mean",
]
_ZERO_TEMP = [
    "circ_fan_max", "circ_fan_min", "co2_supply_max", "co2_supply_min",
    "fcu_fan_max", "fcu_fan_min", "fcu_fan_std", "fcu_pump_max", "fcu_pump_min",
    "fogging_max", "fogging_min", "fogging_std",
    "greenhouse_roof_vent1_max", "greenhouse_roof_vent1_min",
    "greenhouse_roof_vent2_max", "greenhouse_roof_vent2_min",
    "greenhouse_roof_vent2_std", "wind_speed_outside_max", "wind_speed_outside_min",
]

# 타깃별로 "이 컬럼들은 학습에서 빼라"는 최종 지시서. 아래 run_folds()/main() 둘 다
# `[c for c in 전체컬럼 if c not in DROP_COLS_PER_TARGET[타깃]]` 형태로 이걸 참조해서
# 실제 fit()에 들어가는 컬럼 목록을 타깃마다 다르게 골라냄.
DROP_COLS_PER_TARGET = {
    "soil_moisture": LAG_FEATURE_NAMES + _ZERO_MOISTURE,   # lag 피처 전부 노이즈 + 중요도0 피처
    "soil_ec": _ZERO_EC,                                    # day_num/lag/rolling/trend 다 유지, 중요도0만 제거
    "soil_temp": ["day_num"] + ROLL_FEATURE_NAMES + _ZERO_TEMP,  # day_num 외삽위험 + rolling 노이즈 + 중요도0
}


def add_trend_features(feat_df, window=TREND_WINDOW):
    """"지금 값 - 정확히 1일 전 값" = 변화량(차이) 피처를 TREND_SRC_COLS마다 하나씩 추가.

    두 시점(지금, 1일 전)만 비교하는 가장 단순한 형태 — 절대 수준이 아니라
    "얼마나 움직였는가"를 보려는 목적. shift(window)는 인덱스를 window칸(=1일)
    통째로 밀어서 정렬하는 것이므로, 데이터 첫 1일 구간은 뺄 대상이 없어 NaN이 됨
    (버그 아님, LightGBM이 결측을 그대로 학습에 활용).
    """
    feat_df = feat_df.copy()
    for col in TREND_SRC_COLS:
        feat_df[f"{col}_trend1d"] = feat_df[col] - feat_df[col].shift(window)
    return feat_df


def add_lag_features(feat_df, lags=LAG_STEPS):
    """"정확히 N칸 전의 값"을 그대로 복사해 새 컬럼으로 추가 (가공 없음, lag=지연값).

    trend/rolling과 달리 계산(차이/평균)을 하지 않고 과거 시점의 원값 자체를 그대로
    옮겨오는 것 — "1시간 전엔 몇 도였나"처럼, 그 시점 값 하나하나를 트리가 직접
    비교할 수 있게 해줌. LAG_STEPS(12/36/72/144/288칸 = 1h/3h/6h/12h/1일)마다
    각각 별도 컬럼이 생기므로 컬럼 수 = len(LAG_SRC_COLS) × len(LAG_STEPS)개 증가.
    """
    feat_df = feat_df.copy()
    for col in LAG_SRC_COLS:
        if col in feat_df.columns:
            for lag in lags:
                feat_df[f"{col}_lag{lag}"] = feat_df[col].shift(lag)
    return feat_df


def add_rolling_features(feat_df):
    """최근 1일/2일 "구간 전체"를 평균·표준편차로 요약하는 이동window 피처.

    lag가 "딱 한 시점"만 보는 것과 달리, rolling은 지금 시점 기준 과거 w칸
    (288=1일, 576=2일) 전부를 하나의 통계값으로 뭉뚱그림 — 순간 노이즈에 덜 민감하고
    "최근 추세가 어떤지"를 안정적으로 보여줌. min_periods=1이라 데이터 시작
    직후에도(윈도 전체가 안 차도) 그때까지 있는 값만으로 계산 — NaN이 거의 안 생김.
    """
    feat_df = feat_df.copy()
    for col in ROLL_SRC_COLS:
        if col in feat_df.columns:
            for w in ROLL_WINDOWS:
                feat_df[f"{col}_roll{w}"] = feat_df[col].rolling(w, min_periods=1).mean()
            # std: 팬/온도 변동성 — 0이면 체제 안정, 크면 전환 중
            feat_df[f"{col}_rollstd288"] = feat_df[col].rolling(288, min_periods=1).std().fillna(0)
    # circ_fan 가속도: "최근 1일 평균 - 최근 2일 평균"으로 두 rolling 피처를 다시 한번 비교.
    # 양수 = 최근 1일이 그 전 1일보다 팬 가동이 늘어나는 추세(가속) → EC 하락 신호로 기대,
    # 음수면 팬 가동이 줄어드는 추세(감속) → EC 상승 신호로 기대 (circ_fan-EC 상관 -0.49 기반 가설).
    # 가속 피처: 1일 rolling - 2일 rolling = 최근 추세 방향
    for src in ["circ_fan_mean", "greenhouse_roof_vent1_mean"]:
        r1, r2 = f"{src}_roll288", f"{src}_roll576"
        if r1 in feat_df.columns and r2 in feat_df.columns:
            feat_df[f"{src}_accel"] = feat_df[r1] - feat_df[r2]
    # humidity 3일 rolling: 관수 주기(수분 레벨)를 더 긴 시야로 포착
    if "humidity_mean" in feat_df.columns:
        feat_df["humidity_mean_roll864"] = feat_df["humidity_mean"].rolling(864, min_periods=1).mean()
    return feat_df


def add_cyclic_features(feat_df):
    """hour(0~24)를 sin/cos 두 컬럼으로 바꿔서 "자정에 값이 이어진다"는 걸 알려줌.

    hour를 숫자 그대로(0~24) 쓰면 트리 입장에서 23시와 0시가 "가장 먼 값"처럼
    보여서 하루의 끝과 시작이 이어진다는 걸 모름. sin/cos 두 값의 조합으로
    표현하면 원 위의 좌표처럼 23시와 0시가 실제로 가깝게 인코딩됨 (과거 시점을
    보는 피처는 아니고, 지금 이 순간이 하루 중 어디쯤인지를 다르게 표현하는 것).
    """
    feat_df = feat_df.copy()
    if "hour" in feat_df.columns:
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
    return feat_df


def rmse(y_true, y_pred):
    """평균제곱근오차 — 대회 채점 지표와 동일한 식. 값이 작을수록 좋음."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# 마지막 4일 단일검증은 폴드 1개짜리라 노이즈에 취약(day_num 결정 때 실제로 결론이 뒤집힌 전례 있음).
# 대신 학습 구간을 점점 늘려가며(expanding window) 4번 반복 검증하는 게 더 안정적인 지표 — 26.07.12 채택.
# 예: 109~117일로 학습→118~121일 검증, 109~121일로 학습→122~125일 검증, ... (컷오프 4개)
FOLD_CUTOFFS = [118, 122, 126, 130]

# soil_ec: lr=0.03(세밀 탐색) + min_child_samples=50(보수적 분기 — 26.07.13 그리드 탐색, -0.4%)
# EC 과적합이 아닌 "관리 이벤트 예측 불가" 구조적 한계 확인 → 파라미터 한계선 이게 최선
# soil_moisture/soil_temp: 기본 파라미터가 이 규모에 최적임을 실험으로 확인 — 26.07.13
MODEL_PARAMS = {
    "soil_moisture": {"n_estimators": 1000, "learning_rate": 0.05},
    "soil_ec":       {"n_estimators": 2000, "learning_rate": 0.03, "min_child_samples": 50},
    "soil_temp":     {"n_estimators": 1000, "learning_rate": 0.05},
}

# soil_temp 앙상블 파라미터: 4-fold 그리드 탐색으로 결정 — 26.07.13
# alpha=0.1(약한 정규화), w=0.40(Ridge 40% + LightGBM 60%)
# → LightGBM 단독 0.8338 대비 0.7636(-8.4%) 개선
TEMP_BLEND_ALPHA = 0.1
TEMP_BLEND_W_RIDGE = 0.40


class BlendModel:
    """Ridge + LightGBM 가중 앙상블 — .predict() 인터페이스가 LGBMRegressor와 동일해
    inference.py를 수정하지 않아도 그대로 동작함."""

    def __init__(self, lgbm_model, ridge_model, scaler, col_medians, w_ridge):
        self.lgbm = lgbm_model
        self.ridge = ridge_model
        self.scaler = scaler
        self.col_medians = col_medians  # NaN 대체용 (train median)
        self.w_ridge = w_ridge

    def predict(self, X):
        pred_lgbm = self.lgbm.predict(X)
        X_filled = X.fillna(self.col_medians)
        X_scaled = self.scaler.transform(X_filled)
        pred_ridge = self.ridge.predict(X_scaled)
        return self.w_ridge * pred_ridge + (1 - self.w_ridge) * pred_lgbm


def run_folds(feat_df, target_y, cutoffs=FOLD_CUTOFFS):
    """4-fold 시계열 교차검증 실행. cutoff마다 학습(처음~cutoff직전) + 검증(cutoff~cutoff+4일)
    쌍을 만들어서, 타깃 3개 각각 새 모델을 처음부터 학습시키고 검증 RMSE를 기록함.
    이 함수가 반환하는 모델들은 여기서 버려짐(재사용 안 함) — 순수하게 "이 피처 조합/
    하이퍼파라미터가 얼마나 좋은지" 점수만 얻으려는 목적. 실제 제출용 모델은 main()에서
    따로(전체 데이터로) 학습함.

    Returns: {"soil_moisture": [폴드1 RMSE, 폴드2 RMSE, ...], "soil_ec": [...], "soil_temp": [...]}
    """
    results = {c: [] for c in TARGET_COLS}
    for cutoff in cutoffs:
        # 학습 = 맨 처음부터 cutoff 직전까지(누적/expanding), 검증 = cutoff부터 4일간
        train_mask = feat_df.index < pd.Timedelta(days=cutoff)
        val_mask = (feat_df.index >= pd.Timedelta(days=cutoff)) & (feat_df.index < pd.Timedelta(days=cutoff + 4))
        tr_feat, tr_y = feat_df[train_mask], target_y[train_mask]
        val_feat, val_y = feat_df[val_mask], target_y[val_mask]
        for col in TARGET_COLS:
            cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
            m = lgb.LGBMRegressor(**MODEL_PARAMS[col], random_state=RANDOM_STATE, verbosity=-1)
            m.fit(tr_feat[cols], tr_y[col], eval_set=[(val_feat[cols], val_y[col])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            pred_lgbm = m.predict(val_feat[cols])

            if col == "soil_temp":
                # Ridge는 NaN 불허 → train median으로 대체 후 표준화
                col_med = tr_feat[cols].median()
                scaler = StandardScaler()
                tr_scaled = scaler.fit_transform(tr_feat[cols].fillna(col_med))
                val_scaled = scaler.transform(val_feat[cols].fillna(col_med))
                ridge = Ridge(alpha=TEMP_BLEND_ALPHA)
                ridge.fit(tr_scaled, tr_y[col])
                pred_ridge = ridge.predict(val_scaled)
                pred = TEMP_BLEND_W_RIDGE * pred_ridge + (1 - TEMP_BLEND_W_RIDGE) * pred_lgbm
            else:
                pred = pred_lgbm

            results[col].append(rmse(val_y[col].to_numpy(), pred))
    return results


def main():
    # --- 1. 원본 CSV → 5분 격자 피처(build_features, preprocess.py)로 변환 후,
    #        이 파일에서 정의한 4종 피처(trend/lag/rolling/cyclic)를 순서대로 얹음 ---
    train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    train_feat = add_trend_features(train_feat)
    train_feat = add_lag_features(train_feat)
    train_feat = add_rolling_features(train_feat)
    train_feat = add_cyclic_features(train_feat)
    train_y = load_target("dataset/train/env/train_y.csv")

    # --- 2. 4-fold 교차검증으로 "지금 이 피처/하이퍼파라미터 조합이 얼마나 좋은지" 먼저 확인 ---
    #        (실제 제출 모델 학습 전에 참고용으로만 찍어보는 것 — 여기서 만든 모델은 버려짐)
    print("=== 4-fold 시계열 교차검증 (더 신뢰할 수 있는 지표) ===")
    fold_results = run_folds(train_feat, train_y)
    for col in TARGET_COLS:
        arr = np.array(fold_results[col])
        print(f"  {col}: folds={np.round(arr, 4)} mean={arr.mean():.4f} std={arr.std():.4f}")

    tr_feat, tr_y, val_feat, val_y = time_based_split(train_feat, train_y, val_days=4)

    # --- 3. 실제 제출용 모델 만들기 (2단계로 진행) ---
    # 1단계: 마지막 4일을 val로 떼어내 early stopping으로 "적정 트리 개수"를 찾음 (단일폴드, 참고용)
    # 2단계: 그 트리 개수 그대로, train 전체(26일)로 다시 학습해서 실제 제출용 모델을 만듦
    #        (val로 트리 개수만 정하고, 실제 모델은 가진 데이터를 전부 써야 test에 더 유리함)
    print("\n=== 마지막 4일 단일검증 (참고용, 4-fold보다 신뢰도 낮음) ===")
    models = {}
    for col in TARGET_COLS:
        cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]

        # 1단계: val 4일로 early stopping → 이 조합에 맞는 "적정 트리 개수" 탐색
        val_model = lgb.LGBMRegressor(**MODEL_PARAMS[col], random_state=RANDOM_STATE, verbosity=-1)
        val_model.fit(
            tr_feat[cols], tr_y[col],
            eval_set=[(val_feat[cols], val_y[col])],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        pred = val_model.predict(val_feat[cols])
        score = rmse(val_y[col].to_numpy(), pred)
        print(f"{col} val RMSE: {score:.4f} (best_iteration={val_model.best_iteration_})")

        # 2단계: 위에서 찾은 트리 개수(best_iteration_)를 고정값으로 써서, 이번엔 val 없이
        #        train 26일 전체로 재학습 → 이게 model.pkl에 실제로 저장되는 모델
        final_lgbm = lgb.LGBMRegressor(
            n_estimators=val_model.best_iteration_,
            learning_rate=MODEL_PARAMS[col]["learning_rate"],
            random_state=RANDOM_STATE, verbosity=-1,
        )
        final_lgbm.fit(train_feat[cols], train_y[col])

        if col == "soil_temp":
            # soil_temp: LightGBM + Ridge 앙상블 (BlendModel로 저장 — predict() 인터페이스 동일)
            col_med = train_feat[cols].median()
            scaler = StandardScaler()
            tr_scaled = scaler.fit_transform(train_feat[cols].fillna(col_med))
            ridge = Ridge(alpha=TEMP_BLEND_ALPHA)
            ridge.fit(tr_scaled, train_y[col])
            models[col] = BlendModel(final_lgbm, ridge, scaler, col_med, TEMP_BLEND_W_RIDGE)
        else:
            models[col] = final_lgbm

    # --- 4. 타깃 3개 모델을 딕셔너리 하나로 묶어서 pickle 저장 (inference.py가 그대로 불러다 씀) ---
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(models, f)
    print("저장: model/model.pkl (train 전체 26일로 재학습된 최종 모델)")


if __name__ == "__main__":
    main()
