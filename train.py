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
  재가동)이 고RMSE. EC 4-fold 0.30은 과대추정 — test(135~146) 실제 ≈ 0.02~0.08 예상.
================================================================================
"""
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from preprocess import build_features, load_target

RANDOM_STATE = 42
TARGET_COLS = ["soil_moisture", "soil_ec", "soil_temp"]

# X변수(구동기/날씨) 추세 피처
# thermal_curtain/solar_radiation → EC 전용(-0.9%), moisture/temp는 _ZERO로 제외
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
    """hour/wind_direction을 sin/cos로 바꿔 원형 연속성을 표현."""
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
    "soil_temp":     {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 8, "feature_fraction": 0.8, "reg_lambda": 3.0},
}

# alpha=0.05(약한 정규화), w=0.30(Ridge 30% + LightGBM 70%) — lag576+wind+정규화 최적화
TEMP_BLEND_ALPHA = 0.05
TEMP_BLEND_W_RIDGE = 0.30


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
            m = lgb.LGBMRegressor(**MODEL_PARAMS[col], random_state=RANDOM_STATE, verbosity=-1)
            m.fit(tr_feat[cols], tr_y[col], eval_set=[(val_feat[cols], val_y[col])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            fold_iters[col].append(m.best_iteration_)
            pred_lgbm = m.predict(val_feat[cols])

            if col == "soil_temp":
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
    return results, fold_iters


def main():
    train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    train_feat = add_trend_features(train_feat)
    train_feat = add_lag_features(train_feat)
    train_feat = add_rolling_features(train_feat)
    train_feat = add_cyclic_features(train_feat)
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

        final_params = {**MODEL_PARAMS[col], "n_estimators": n_estimators_per_target[col],
                         "random_state": RANDOM_STATE, "verbosity": -1}
        final_lgbm = lgb.LGBMRegressor(**final_params)
        final_lgbm.fit(train_feat[cols], train_y[col])

        if col == "soil_temp":
            col_med = train_feat[cols].median()
            scaler = StandardScaler()
            tr_scaled = scaler.fit_transform(train_feat[cols].fillna(col_med))
            ridge = Ridge(alpha=TEMP_BLEND_ALPHA)
            ridge.fit(tr_scaled, train_y[col])
            models[col] = BlendModel(final_lgbm, ridge, scaler, col_med, TEMP_BLEND_W_RIDGE)
        else:
            models[col] = final_lgbm

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(models, f)
    print("저장: model/model.pkl (train 전체 26일로 재학습된 최종 모델)")


if __name__ == "__main__":
    main()
