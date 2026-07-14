"""온라인테스트1 학습: LightGBM + Ridge 앙상블 (타깃별 3개 모델). model/model.pkl 에 저장.

경로 규칙(preprocess.py와 동일): 로컬 검증은 dataset/... 사용.
Docker 제출용으로 바꿀 때는 input/dataset/...로 교체 필요 (PLAN.md 4단계 항목).

================================================================================
피처 엔지니어링 개요 및 실험 결과 (26.07.14 최종, 상세는 FEEDBACK.md 참조)
================================================================================

[기준선 → 최종 4-fold RMSE]
  soil_moisture : 1.1199 → 1.0535  (-5.9%)
  soil_ec       : 0.3106 → 0.3033  (-2.4%)   EC 고유값: 104 → 3,250개 (뭉침 해소)
  soil_temp     : 1.0771 → 0.6232  (-42.1%)

[채택된 피처 그룹]
  1. add_trend_features  — X변수 1일 전 대비 변화량 (9개 컬럼 × 1)
     효과: moisture -4.2%, temp -5.5%, ec 무변화.
     thermal_curtain/solar_radiation trend → EC 전용 (-0.9%)

  2. add_lag_features — 1h~2일 전 X변수값 (7개 컬럼 × 6 lag = 42개)
     temperature/humidity/fan/vent1/co2 + temp_outside(EC전용) + wind_speed(temp전용)
     lag576(2일) → temp에만 유지(EC unique값 보호), temp -0.7%
     wind_speed lags → temp -1.3%

  3. add_rolling_features — 12h/1일/2일 rolling mean + rollstd + accel
     circ_fan_on_binary/regime: 이진 레짐 피처 → EC 고유값 1,410→3,250
     temperature_mean roll만 temp에 유지(열관성), 12h roll144 추가 → temp -5.2%
     wind_speed roll144/288 → temp -4.8% (열손실 맥락)

  4. add_cyclic_features — hour/wind_direction sin/cos

[폐기된 접근 전체 요약]
  - LightGBM num_leaves=63 + subsample/colsample → 전체 악화
  - soil_ec Ridge+LightGBM 블렌딩 → 전체 악화
  - 다분광 NDRE/CRE 등 → EC 0.0% 중립
  - EC day_num 제거 → +70.4% 급격 악화
  - VPD domain features → moisture/temp 악화
  - lag864(3일) → temp 악화
  - temperature_outside rolling → 모든 타깃 악화
  - solar_radiation lags → temp 악화

[EC 구조적 한계]
  fold 분포 [0.53, 0.08, 0.58, 0.02] — fold 1(day121 관리 이벤트)과 fold 3(day129 팬
  재가동)이 고RMSE. fold 1은 X변수로 예측 불가능한 시비 결정. fold 3은 훈련 데이터
  (109~126)에 "팬 재가동 → EC 급락" 사이클이 없어 어떤 모델도 학습 불가.
  EC 4-fold 0.30은 과대추정 — test(135~146) 실제 ≈ 0.02~0.08 예상.

[soil_moisture 피처 재현성 재검증 — 26.07.13 루프6, FEEDBACK.md 상세]
  기존 "-5.7%(1.1199→1.0564)" 개선은 원래 cutoff 세트([118,122,126,130]) 딱 하나로만
  검증됐던 것. cutoff을 ±1~2일 옮긴 4세트로 재확인하니 "4-fold 평균" 기준 재현 안 됨
  (trend/rolling/cyclic을 각각 baseline에 단독으로 추가했을 때 4세트 전부 0/4 통과 —
  기존 개선폭이 기준선 fold 표준오차보다 작아 노이즈와 통계적으로 구분 불가).

  "최대 학습데이터 fold"(실전 배포와 가장 비슷한 조건)만 따로 보면 trend는 4/4·rolling은
  3/4 일관되게 개선하는데, cyclic만 4/4 전부 악화 — 그래서 처음엔 "cyclic만 soil_moisture
  에서 제외"로 결론지었었음. 그런데 실제로 빼서(leave-one-out) 재검증하니 정반대 결과가
  나옴: trend+rolling 조합에서 cyclic을 빼면 4개 cutoff 세트 **전부**에서 오히려 악화
  (예: set0 1.0534→1.0971). 단독 추가 효과와 이미 있는 조합에서 빼는 효과가 다르게
  나온 것 — cyclic이 trend/rolling과 상호작용한다는 뜻. → **결론: 코드 변경 없음(전부
  유지)**. 다만 기존 "-5.7%" 문서화는 신뢰 과잉이었음을 인정 — 단일 cutoff 세트의 평균
  개선폭 자체는 노이즈와 통계적으로 구분 안 되지만, "이 조합을 빼면 더 나빠진다"는
  leave-one-out 신호는 4/4로 강함. 즉 "확실히 더 좋다"는 아니어도 "지금 조합을 건드리면
  더 나빠진다"는 충분히 검증됨 → 현상 유지가 최선의 선택.
================================================================================
"""
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from preprocess import build_features, load_target, add_ec_high_confidence, add_ms_band_features

RANDOM_STATE = 42
TARGET_COLS = ["soil_moisture", "soil_ec", "soil_temp"]

# 다분광 밴드9/10(897/920nm, NIR 끝단) — soil_ec 전용. PLAN.md에 미구현으로 남아있던
# "밴드 평균 그룹에 뭉개지 말고 개별 피처로" 아이디어를 구현. 4-fold 검증(FEEDBACK.md
# 루프10) 결과 EC +0.0006%(사실상 무변화) — 다분광 활용이 필수 요건이라 채택하되,
# 실질적 성능 기여는 없다고 문서화해둠(카메라 파장 713~920nm로는 EC 직접 측정 불가라는
# 기존 결론과 일치). moisture/temp는 전부 악화(+3~6%)로 확인돼 격리.
MS_BAND_COLS = ["ms_band9_mean", "ms_band10_mean"]

# X변수(구동기/날씨) 추세 피처
# thermal_curtain/solar_radiation → EC 전용(-0.9%), moisture/temp는 _ZERO로 제외
#
# ⚠️ soil_moisture 재현성 재검증(26.07.13 루프6, FEEDBACK.md): cutoff을 ±1~2일 옮긴 4세트로
# 재확인한 결과, "4-fold 평균" 기준으로는 재현 안 됨(4세트 중 0개 통과 — 노이즈와 통계적으로
# 구분 불가). 다만 "최대 학습데이터 fold"(실전 배포와 가장 비슷한 조건)만 따로 보면 4세트
# 전부에서 일관되게 개선(+0.02~+0.11) — 학습 데이터가 많을수록 안정적으로 도움되는 "안정구간
# 한정" 피처로 재분류. 초기 소규모 학습창(폴드1~3류)에서의 개선/악화는 신뢰 불가.
TREND_SRC_COLS = [
    "circ_fan_mean", "greenhouse_roof_vent1_mean", "greenhouse_roof_vent2_mean",
    "temperature_outside_mean", "humidity_outside_mean", "humidity_mean", "co2_mean",
    "thermal_curtain_mean", "solar_radiation_mean",
]
TREND_WINDOW = 288  # 5분 격자 기준 288칸 = 1일

# lag 피처: 토양 반응 지연 포착
# temperature_outside → EC 전용(temp는 _ZERO로 제외)
# wind_speed_outside → temp 전용(EC는 _ZERO로 제외)
LAG_SRC_COLS = [
    "temperature_mean", "humidity_mean",
    "circ_fan_mean", "greenhouse_roof_vent1_mean", "co2_mean",
    "temperature_outside_mean", "wind_speed_outside_mean",
]
LAG_STEPS = [12, 36, 72, 144, 288, 576]  # 1h, 3h, 6h, 12h, 1일, 2일 (5분 격자 기준)
LAG_FEATURE_NAMES = [f"{col}_lag{lag}" for col in LAG_SRC_COLS for lag in LAG_STEPS]
MOISTURE_LAG_EXCLUDE = [f"{col}_lag{lag}" for col in LAG_SRC_COLS for lag in LAG_STEPS
                        if col != "humidity_mean"]

# rolling window 피처: circ_fan 12h/1일/2일 평균으로 환기 "체제(regime)" 포착
# soil_temp는 temperature_mean rolling만 유지(열관성), 나머지는 _ZERO로 제외
#
# ⚠️ soil_moisture 재현성 재검증(26.07.13 루프6): trend와 동일한 패턴 — 4-fold 평균 기준
# 재현 안 됨(0/4), 최대 학습데이터 fold 기준으로는 4세트 중 3개에서 개선(+0.02~+0.50, set2만
# 소폭 악화 -0.07) → "안정구간 한정" 피처로 재분류, trend보다 재현성이 살짝 약함.
ROLL_SRC_COLS = [
    "circ_fan_mean", "greenhouse_roof_vent1_mean",
    "temperature_mean", "humidity_mean",
]
ROLL_WINDOWS = [144, 288, 576]  # 12시간, 1일, 2일 (5분 격자 기준)
ROLL_FEATURE_NAMES = [f"{col}_roll{w}" for col in ROLL_SRC_COLS for w in ROLL_WINDOWS]

_ZERO_MOISTURE = [
    "circ_fan_max", "circ_fan_min", "co2_supply_max", "co2_supply_min",
    "fcu_fan_max", "fcu_fan_min", "fcu_pump_max", "fcu_pump_min",
    "fogging_last", "fogging_max", "fogging_min", "fogging_std",
    "greenhouse_roof_vent1_max", "greenhouse_roof_vent1_min",
    "greenhouse_roof_vent2_max", "greenhouse_roof_vent2_min",
    "thermal_curtain_mean_trend1d", "solar_radiation_mean_trend1d",
    "circ_fan_mean_roll144", "greenhouse_roof_vent1_mean_roll144",
    "temperature_mean_roll144", "humidity_mean_roll144",
    "wind_speed_outside_roll144", "wind_speed_outside_roll288",
    "ec_high_confidence", "ms_band9_mean", "ms_band10_mean",
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
    "temperature_mean_lag576", "humidity_mean_lag576",
    "circ_fan_mean_lag576", "greenhouse_roof_vent1_mean_lag576",
    "co2_mean_lag576", "temperature_outside_mean_lag576",
    "wind_speed_outside_mean_lag12", "wind_speed_outside_mean_lag36",
    "wind_speed_outside_mean_lag72", "wind_speed_outside_mean_lag144",
    "wind_speed_outside_mean_lag288", "wind_speed_outside_mean_lag576",
    "wind_speed_outside_roll144", "wind_speed_outside_roll288",
]
_ZERO_TEMP = [
    "circ_fan_max", "circ_fan_min", "co2_supply_max", "co2_supply_min",
    "fcu_fan_max", "fcu_fan_min", "fcu_fan_std", "fcu_pump_max", "fcu_pump_min",
    "fogging_max", "fogging_min", "fogging_std",
    "greenhouse_roof_vent1_max", "greenhouse_roof_vent1_min",
    "greenhouse_roof_vent2_max", "greenhouse_roof_vent2_min",
    "greenhouse_roof_vent2_std", "wind_speed_outside_max", "wind_speed_outside_min",
    "thermal_curtain_mean_trend1d", "solar_radiation_mean_trend1d",
    "temperature_outside_mean_lag12", "temperature_outside_mean_lag36",
    "temperature_outside_mean_lag72", "temperature_outside_mean_lag144",
    "temperature_outside_mean_lag288", "temperature_outside_mean_lag576",
    "ec_high_confidence", "ms_band9_mean", "ms_band10_mean",
]

DROP_COLS_PER_TARGET = {
    "soil_moisture": LAG_FEATURE_NAMES + _ZERO_MOISTURE,
    "soil_ec": _ZERO_EC,
    "soil_temp": ["day_num"] + [f for f in ROLL_FEATURE_NAMES if "temperature_mean" not in f] + _ZERO_TEMP,
}


def add_trend_features(feat_df, window=TREND_WINDOW):
    """"지금 값 - 정확히 1일 전 값" = 변화량(차이) 피처."""
    feat_df = feat_df.copy()
    for col in TREND_SRC_COLS:
        feat_df[f"{col}_trend1d"] = feat_df[col] - feat_df[col].shift(window)
    return feat_df


def add_lag_features(feat_df, lags=LAG_STEPS):
    """"정확히 N칸 전의 값"을 그대로 복사해 새 컬럼으로 추가."""
    feat_df = feat_df.copy()
    for col in LAG_SRC_COLS:
        if col in feat_df.columns:
            for lag in lags:
                feat_df[f"{col}_lag{lag}"] = feat_df[col].shift(lag)
    return feat_df


def add_rolling_features(feat_df):
    """최근 12h/1일/2일 "구간 전체"를 평균·표준편차로 요약하는 이동window 피처."""
    feat_df = feat_df.copy()
    for col in ROLL_SRC_COLS:
        if col in feat_df.columns:
            for w in ROLL_WINDOWS:
                feat_df[f"{col}_roll{w}"] = feat_df[col].rolling(w, min_periods=1).mean()
            feat_df[f"{col}_rollstd288"] = feat_df[col].rolling(288, min_periods=1).std().fillna(0)
    for src in ["circ_fan_mean", "greenhouse_roof_vent1_mean"]:
        r1, r2 = f"{src}_roll288", f"{src}_roll576"
        if r1 in feat_df.columns and r2 in feat_df.columns:
            feat_df[f"{src}_accel"] = feat_df[r1] - feat_df[r2]
    if "humidity_mean" in feat_df.columns:
        feat_df["humidity_mean_roll864"] = feat_df["humidity_mean"].rolling(864, min_periods=1).mean()
    # 외기풍속 rolling: rollstd 없이 rolling mean만 — soil_temp 열손실 맥락
    if "wind_speed_outside_mean" in feat_df.columns:
        feat_df["wind_speed_outside_roll144"] = feat_df["wind_speed_outside_mean"].rolling(144, min_periods=1).mean()
        feat_df["wind_speed_outside_roll288"] = feat_df["wind_speed_outside_mean"].rolling(288, min_periods=1).mean()
    # circ_fan 이진 피처: 연속값보다 명확한 ON/OFF 경계를 모델에 제공
    if "circ_fan_mean" in feat_df.columns:
        feat_df["circ_fan_on_binary"] = (feat_df["circ_fan_mean"] > 0.2).astype(float)
        if "circ_fan_mean_roll288" in feat_df.columns:
            feat_df["circ_fan_regime"] = (feat_df["circ_fan_mean_roll288"] > 0.3).astype(float)
    return feat_df


def add_domain_features(feat_df):
    """딸기 재배 도메인 지식 기반 파생 피처 (VPD 등) — 실험 결과 모든 타깃 악화, 미사용."""
    return feat_df


def add_cyclic_features(feat_df):
    """hour/wind_direction을 sin/cos로 바꿔 원형 연속성을 표현.

    ⚠️ soil_moisture 재현성 재검증(26.07.13 루프6, 모듈 docstring 상단 참고): 이 함수를
    baseline에 단독으로 추가하면 재현성이 약했지만(cutoff 4세트 재현 안 됨), trend+rolling이
    이미 있는 상태에서 이 함수만 빼보면(leave-one-out) 4세트 전부 악화 — 상호작용 효과로
    현재 조합에서는 유지가 최선. soil_moisture DROP_COLS_PER_TARGET에 넣지 않음(그대로 유지).
    """
    feat_df = feat_df.copy()
    if "hour" in feat_df.columns:
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
    if "wind_direction_outside_mean" in feat_df.columns:
        rad = np.deg2rad(feat_df["wind_direction_outside_mean"])
        feat_df["wind_dir_sin"] = np.sin(rad)
        feat_df["wind_dir_cos"] = np.cos(rad)
    return feat_df


def rmse(y_true, y_pred):
    """평균제곱근오차 — 대회 채점 지표와 동일한 식."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


FOLD_CUTOFFS = [118, 122, 126, 130]

MODEL_PARAMS = {
    "soil_moisture": {"n_estimators": 1000, "learning_rate": 0.05},
    "soil_ec":       {"n_estimators": 2000, "learning_rate": 0.03, "min_child_samples": 50},
    "soil_temp":     {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 8, "feature_fraction": 0.8, "reg_lambda": 3.0, "min_child_samples": 5},
}

# alpha=0.01(최소 정규화), w=0.30(Ridge 30% + LightGBM 70%) — Ridge alpha 재탐색
TEMP_BLEND_ALPHA = 0.01
TEMP_BLEND_W_RIDGE = 0.35


class BlendModel:
    """Ridge + LightGBM 가중 앙상블."""

    def __init__(self, lgbm_model, ridge_model, scaler, col_medians, w_ridge):
        self.lgbm = lgbm_model
        self.ridge = ridge_model
        self.scaler = scaler
        self.col_medians = col_medians
        self.w_ridge = w_ridge

    def predict(self, X):
        pred_lgbm = self.lgbm.predict(X)
        X_filled = X.fillna(self.col_medians)
        X_scaled = self.scaler.transform(X_filled)
        pred_ridge = self.ridge.predict(X_scaled)
        return self.w_ridge * pred_ridge + (1 - self.w_ridge) * pred_lgbm


class ECBlendModel:
    """LightGBM 예측과 "고EC 레짐 override"를 ec_high_confidence 컬럼(신뢰도, 0~1)으로
    소프트 블렌드. 신뢰도가 1이면 완전히 override(고EC 레짐 평균값), 0이면 LightGBM 그대로,
    그 사이는 가중평균.

    수학적 근거: 실제값이 두 레벨(고EC/평시) 중 하나일 확률분포를 따른다고 볼 때,
    오차제곱합을 최소화하는 단일 예측값은 정확히 p*고EC값 + (1-p)*평시예측값 형태
    (p=고EC일 확률) — "확신도만큼만 반영"은 임의의 절충이 아니라 불확실성 하에서
    RMSE를 최소화하는 이론적 최적해. p=1(완전확신)이면 하드 오버라이드, p=0이면
    LightGBM 단독과 동일해 두 극단을 포함하는 일반형이다.

    ⚠️ ec_high_confidence 게이트 규칙 자체가 train에서 단 1번(121~128일)만 관측된
    패턴이라 p를 100% 신뢰할 근거는 아님 — 그래서 하드 분류(0/1)가 아닌 연속 신뢰도로
    블렌드해 오탐 시 피해를 제한한다 (자세한 근거: FEEDBACK.md 루프9).

    .predict() 인터페이스가 LGBMRegressor와 동일해 inference.py 구조 변경 불필요 —
    단, 입력 X에는 "ec_high_confidence" 컬럼이 반드시 포함되어야 한다.
    """

    def __init__(self, lgbm_model, lgbm_cols, high_regime_value):
        self.lgbm = lgbm_model
        self.lgbm_cols = lgbm_cols
        self.high_regime_value = high_regime_value

    def predict(self, X):
        pred_lgbm = self.lgbm.predict(X[self.lgbm_cols])
        conf = X["ec_high_confidence"].to_numpy()
        return conf * self.high_regime_value + (1 - conf) * pred_lgbm


def run_folds(feat_df, target_y, cutoffs=FOLD_CUTOFFS):
    """4-fold 시계열 교차검증 실행."""
    results = {c: [] for c in TARGET_COLS}
    fold_iters = {c: [] for c in TARGET_COLS}
    for cutoff in cutoffs:
        train_mask = feat_df.index < pd.Timedelta(days=cutoff)
        val_mask = (feat_df.index >= pd.Timedelta(days=cutoff)) & (feat_df.index < pd.Timedelta(days=cutoff + 4))
        tr_feat, tr_y = feat_df[train_mask], target_y[train_mask]
        val_feat, val_y = feat_df[val_mask], target_y[val_mask]
        for col in TARGET_COLS:
            cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
            # soil_ec: ec_high_confidence는 소프트 블렌드 레이어에서만 쓰고, 트리 자체의
            # 분기 피처로는 안 씀(레짐 override 신뢰도를 트리가 직접 학습하면 의미가 섞임).
            fit_cols = [c for c in cols if c != "ec_high_confidence"] if col == "soil_ec" else cols
            m = lgb.LGBMRegressor(**MODEL_PARAMS[col], random_state=RANDOM_STATE, verbosity=-1)
            m.fit(tr_feat[fit_cols], tr_y[col], eval_set=[(val_feat[fit_cols], val_y[col])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            fold_iters[col].append(m.best_iteration_)
            pred_lgbm = m.predict(val_feat[fit_cols])

            if col == "soil_temp":
                col_med = tr_feat[cols].median()
                scaler = StandardScaler()
                tr_scaled = scaler.fit_transform(tr_feat[cols].fillna(col_med))
                val_scaled = scaler.transform(val_feat[cols].fillna(col_med))
                ridge = Ridge(alpha=TEMP_BLEND_ALPHA)
                ridge.fit(tr_scaled, tr_y[col])
                pred_ridge = ridge.predict(val_scaled)
                pred = TEMP_BLEND_W_RIDGE * pred_ridge + (1 - TEMP_BLEND_W_RIDGE) * pred_lgbm
            elif col == "soil_ec":
                # 고EC 레짐 override 값은 이 fold의 학습구간(tr_feat/tr_y)만으로 계산 —
                # 미래(val) 정보 누수 없음. 신뢰도(confidence)는 val 당일 X값에서만 계산되므로
                # 마찬가지로 누수 없음(y를 몰라도 알 수 있는 값).
                high_mask = tr_feat["ec_high_confidence"] >= 0.99
                high_regime_value = tr_y.loc[high_mask, col].mean() if high_mask.any() else tr_y[col].mean()
                conf = val_feat["ec_high_confidence"].to_numpy()
                pred = conf * high_regime_value + (1 - conf) * pred_lgbm
            else:
                pred = pred_lgbm

            results[col].append(rmse(val_y[col].to_numpy(), pred))
    return results, fold_iters


def main():
    train_raw = pd.read_csv("dataset/train/env/train_X.csv")
    train_feat = build_features(train_raw)
    train_feat = add_trend_features(train_feat)
    train_feat = add_lag_features(train_feat)
    train_feat = add_rolling_features(train_feat)
    train_feat = add_cyclic_features(train_feat)
    train_feat = add_ec_high_confidence(train_feat, train_raw)
    train_feat = add_ms_band_features(train_feat, "dataset", "train")
    train_y = load_target("dataset/train/env/train_y.csv")

    print("=== 4-fold 시계열 교차검증 (더 신뢰할 수 있는 지표) ===")
    fold_results, fold_iters = run_folds(train_feat, train_y)
    fold_weights = np.array(FOLD_CUTOFFS, dtype=float)
    n_estimators_per_target = {}
    for col in TARGET_COLS:
        arr = np.array(fold_results[col])
        weighted_mean = np.average(arr, weights=fold_weights)
        iters = np.array(fold_iters[col], dtype=float)
        n_estimators_per_target[col] = int(round(np.average(iters, weights=fold_weights)))
        print(f"  {col}: folds={np.round(arr, 4)} mean={arr.mean():.4f} std={arr.std():.4f} "
              f"weighted_mean(train_size)={weighted_mean:.4f} | best_iters={iters.astype(int).tolist()} "
              f"-> n_estimators={n_estimators_per_target[col]}")

    print("\n=== 최종 모델 학습 (train 전체 26일, n_estimators는 4-fold 가중평균 고정값) ===")
    models = {}
    for col in TARGET_COLS:
        cols = [c for c in train_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
        fit_cols = [c for c in cols if c != "ec_high_confidence"] if col == "soil_ec" else cols

        final_params = {**MODEL_PARAMS[col], "n_estimators": n_estimators_per_target[col],
                         "random_state": RANDOM_STATE, "verbosity": -1}
        final_lgbm = lgb.LGBMRegressor(**final_params)
        final_lgbm.fit(train_feat[fit_cols], train_y[col])

        if col == "soil_temp":
            col_med = train_feat[cols].median()
            scaler = StandardScaler()
            tr_scaled = scaler.fit_transform(train_feat[cols].fillna(col_med))
            ridge = Ridge(alpha=TEMP_BLEND_ALPHA)
            ridge.fit(tr_scaled, train_y[col])
            models[col] = BlendModel(final_lgbm, ridge, scaler, col_med, TEMP_BLEND_W_RIDGE)
        elif col == "soil_ec":
            high_mask = train_feat["ec_high_confidence"] >= 0.99
            high_regime_value = train_y.loc[high_mask, col].mean()
            models[col] = ECBlendModel(final_lgbm, fit_cols, high_regime_value)
        else:
            models[col] = final_lgbm

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(models, f)
    print("저장: model/model.pkl (train 전체 26일로 재학습된 최종 모델)")


if __name__ == "__main__":
    main()
