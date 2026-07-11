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
# day_num(경과일수): train(DAT109~134)과 test(DAT135~146) 날짜가 안 겹쳐서 트리 모델이 외삽해야 함.
# 4-fold 시계열 교차검증(expanding window)으로 타깃별 확인한 결과 soil_moisture/soil_ec는
# day_num이 있는 쪽이 확실히 낫고(외삽 위험보다 신호 가치가 큼), soil_temp만 없는 쪽이 나음
# (day_num 의존도가 낮고 실제 온도 센서값이 더 잘 설명함) — 26.07.11 실험 확인.
DROP_COLS_PER_TARGET = {
    "soil_moisture": [],
    "soil_ec": [],
    "soil_temp": ["day_num"],
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main():
    train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    train_y = load_target("dataset/train/env/train_y.csv")

    tr_feat, tr_y, val_feat, val_y = time_based_split(train_feat, train_y, val_days=4)

    models = {}
    for col in TARGET_COLS:
        cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
        model.fit(
            tr_feat[cols], tr_y[col],
            eval_set=[(val_feat[cols], val_y[col])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        pred = model.predict(val_feat[cols])
        score = rmse(val_y[col].to_numpy(), pred)
        print(f"{col} val RMSE: {score:.4f} (best_iteration={model.best_iteration_})")
        models[col] = model

    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(models, f)
    print("저장: model/model.pkl")


if __name__ == "__main__":
    main()
