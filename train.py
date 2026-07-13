"""온라인테스트1 학습: LightGBM 베이스라인 (타깃별 3개 모델). model/model.pkl 에 저장.

경로 규칙(preprocess.py와 동일): 로컬 검증은 dataset/... 사용.
Docker 제출용으로 바꿀 때는 input/dataset/...로 교체 필요 (PLAN.md 4단계 항목).
"""
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

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
DROP_COLS_PER_TARGET = {
    "soil_moisture": LAG_FEATURE_NAMES,  # lag 피처 모두 노이즈 (humidity도 포함) — 26.07.13
    "soil_ec": [],
    "soil_temp": ["day_num"] + ROLL_FEATURE_NAMES,  # rolling mean 노이즈 — 26.07.13
}


def add_trend_features(feat_df, window=TREND_WINDOW):
    feat_df = feat_df.copy()
    for col in TREND_SRC_COLS:
        feat_df[f"{col}_trend1d"] = feat_df[col] - feat_df[col].shift(window)
    return feat_df


def add_lag_features(feat_df, lags=LAG_STEPS):
    feat_df = feat_df.copy()
    for col in LAG_SRC_COLS:
        if col in feat_df.columns:
            for lag in lags:
                feat_df[f"{col}_lag{lag}"] = feat_df[col].shift(lag)
    return feat_df


def add_rolling_features(feat_df):
    feat_df = feat_df.copy()
    for col in ROLL_SRC_COLS:
        if col in feat_df.columns:
            for w in ROLL_WINDOWS:
                feat_df[f"{col}_roll{w}"] = feat_df[col].rolling(w, min_periods=1).mean()
            # std: 팬/온도 변동성 — 0이면 체제 안정, 크면 전환 중
            feat_df[f"{col}_rollstd288"] = feat_df[col].rolling(288, min_periods=1).std().fillna(0)
    # circ_fan 가속: 1일 평균 - 2일 평균 → 양수면 최근 팬 가동↑(EC 하락 신호), 음수면 ↓(EC 상승 신호)
    if "circ_fan_mean_roll288" in feat_df.columns and "circ_fan_mean_roll576" in feat_df.columns:
        feat_df["circ_fan_accel"] = feat_df["circ_fan_mean_roll288"] - feat_df["circ_fan_mean_roll576"]
    return feat_df


def add_cyclic_features(feat_df):
    """hour를 sin/cos로 인코딩 — 트리가 23→0 연속성(자정 연속)을 학습하게"""
    feat_df = feat_df.copy()
    if "hour" in feat_df.columns:
        feat_df["hour_sin"] = np.sin(2 * np.pi * feat_df["hour"] / 24)
        feat_df["hour_cos"] = np.cos(2 * np.pi * feat_df["hour"] / 24)
    return feat_df


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# 마지막 4일 단일검증은 폴드 1개짜리라 노이즈에 취약(day_num 결정 때 실제로 결론이 뒤집힌 전례 있음).
# 대신 학습 구간을 점점 늘려가며(expanding window) 4번 반복 검증하는 게 더 안정적인 지표 — 26.07.12 채택.
# 예: 109~117일로 학습→118~121일 검증, 109~121일로 학습→122~125일 검증, ... (컷오프 4개)
FOLD_CUTOFFS = [118, 122, 126, 130]

# soil_ec: 4-fold best_iteration 평균이 ~66으로 낮음 → lr을 절반으로 줄여 더 세밀하게 학습
# soil_moisture/soil_temp: 기본 파라미터가 이 규모에 최적임을 실험으로 확인 — 26.07.13
MODEL_PARAMS = {
    "soil_moisture": {"n_estimators": 1000, "learning_rate": 0.05},
    "soil_ec":       {"n_estimators": 2000, "learning_rate": 0.03, "num_leaves": 63},
    "soil_temp":     {"n_estimators": 1000, "learning_rate": 0.05},
}


def run_folds(feat_df, target_y, cutoffs=FOLD_CUTOFFS):
    results = {c: [] for c in TARGET_COLS}
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
            pred = m.predict(val_feat[cols])
            results[col].append(rmse(val_y[col].to_numpy(), pred))
    return results


def main():
    train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    train_feat = add_trend_features(train_feat)
    train_feat = add_lag_features(train_feat)
    train_feat = add_rolling_features(train_feat)
    train_feat = add_cyclic_features(train_feat)
    train_y = load_target("dataset/train/env/train_y.csv")

    print("=== 4-fold 시계열 교차검증 (더 신뢰할 수 있는 지표) ===")
    fold_results = run_folds(train_feat, train_y)
    for col in TARGET_COLS:
        arr = np.array(fold_results[col])
        print(f"  {col}: folds={np.round(arr, 4)} mean={arr.mean():.4f} std={arr.std():.4f}")

    tr_feat, tr_y, val_feat, val_y = time_based_split(train_feat, train_y, val_days=4)

    # 1단계: 마지막 4일을 val로 떼어내 early stopping으로 "적정 트리 개수"를 찾음 (단일폴드, 참고용)
    # 2단계: 그 트리 개수 그대로, train 전체(26일)로 다시 학습해서 실제 제출용 모델을 만듦
    #        (val로 트리 개수만 정하고, 실제 모델은 가진 데이터를 전부 써야 test에 더 유리함)
    print("\n=== 마지막 4일 단일검증 (참고용, 4-fold보다 신뢰도 낮음) ===")
    models = {}
    for col in TARGET_COLS:
        cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]

        val_model = lgb.LGBMRegressor(**MODEL_PARAMS[col], random_state=RANDOM_STATE, verbosity=-1)
        val_model.fit(
            tr_feat[cols], tr_y[col],
            eval_set=[(val_feat[cols], val_y[col])],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
        pred = val_model.predict(val_feat[cols])
        score = rmse(val_y[col].to_numpy(), pred)
        print(f"{col} val RMSE: {score:.4f} (best_iteration={val_model.best_iteration_})")

        final_model = lgb.LGBMRegressor(
            n_estimators=val_model.best_iteration_,
            learning_rate=MODEL_PARAMS[col]["learning_rate"],
            random_state=RANDOM_STATE, verbosity=-1,
        )
        final_model.fit(train_feat[cols], train_y[col])
        models[col] = final_model

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(models, f)
    print("저장: model/model.pkl (train 전체 26일로 재학습된 최종 모델)")


if __name__ == "__main__":
    main()
