"""실험: soil_ec/soil_moisture를 '이미 안다'고 가정하고(train은 실제값, test는 모델의
예측값), 전체 환경 피처(zone/clim 포함) + oof_ec + oof_moisture로 soil_temp를 새로 예측.

핵심: train에서 soil_temp 모델을 학습할 때 진짜 soil_ec/soil_moisture 값을 그대로 피처로
쓰면 안 됨(test엔 그 값이 없음) -> OOF(out-of-fold) 예측값을 대신 써서 train/test 조건을
동일하게 맞춘다 (누수 없는 stacking).
"""
import os
import sys
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/ 의 상위 = 프로젝트 루트
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src import features as FT
from src import msfeat as MS
from src import structure as ST

SEED = 42
N_SPLITS = 6
DET = dict(deterministic=True, force_row_wise=True, n_jobs=4)
PARAMS = {
    "soil_moisture": dict(n_estimators=1200, learning_rate=0.03, num_leaves=31,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=1.0),
    "soil_ec": dict(n_estimators=2000, learning_rate=0.03, num_leaves=7,
                    min_child_samples=80, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.6, reg_lambda=5.0),
    "soil_temp": dict(n_estimators=1200, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0),
}
ALL_Y = ["soil_moisture", "soil_ec", "soil_temp"]

print("=== 데이터 로드 ===")
trX = ST.parse_time(pd.read_csv("dataset/train/env/train_X.csv"))
trY = ST.parse_time(pd.read_csv("dataset/train/env/train_y.csv"))
teX = ST.parse_time(pd.read_csv("dataset/test/env/test_X.csv"))

st, art = ST.discover(trX, teX, seed=SEED)
n_zones = art["n_zones"]
print(f"  구획 {n_zones}개")

sess = MS.extract_split("dataset/train/ms")
ms_daily = MS.daily_summary(sess)
ms_cols = [c for c in ms_daily.columns if c != "dat"]
ms_fill = ms_daily[ms_cols].median()

X = FT.build(trX, st, ms_daily, ms_fill, n_zones)
D = X.merge(trY.sort_values(["dat", "mod"])[["dat", "mod"] + ALL_Y],
            on=["dat", "mod"], how="inner").reset_index(drop=True)
base_cols = [c for c in FT.feature_columns(D) if c not in ALL_Y]
groups = D["dat"].to_numpy()
gkf = GroupKFold(n_splits=N_SPLITS)

print("\n=== 1) soil_ec/soil_moisture OOF 예측 생성 ===")
oof = {"soil_ec": np.zeros(len(D)), "soil_moisture": np.zeros(len(D))}
for tgt in ["soil_ec", "soil_moisture"]:
    cols = FT.select(base_cols, tgt) + [f"clim_{t}" for t in ALL_Y]
    for tr_i, va_i in gkf.split(D, D[tgt], groups):
        dtr, dva = D.iloc[tr_i], D.iloc[va_i]
        enc = FT.fit_climatology(dtr, ALL_Y)
        xtr = FT.apply_climatology(dtr, enc)[cols]
        xva = FT.apply_climatology(dva, enc)[cols]
        m = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS[tgt])
        m.fit(xtr, dtr[tgt])
        oof[tgt][va_i] = m.predict(xva)
    rmse = float(np.sqrt(np.mean((D[tgt] - oof[tgt]) ** 2)))
    print(f"  {tgt} OOF RMSE = {rmse:.4f}  (참고: 전체 train으로 한 번에 학습한 CV와 비슷해야 정상)")

D["oof_ec"] = oof["soil_ec"]
D["oof_moisture"] = oof["soil_moisture"]

print("\n=== 2) soil_temp: 전체 env 피처 + oof_ec + oof_moisture로 예측 (baseline과 CV 비교) ===")
temp_cols_stack = FT.select(base_cols, "soil_temp") + [f"clim_{t}" for t in ALL_Y] + ["oof_ec", "oof_moisture"]
temp_cols_base = FT.select(base_cols, "soil_temp") + [f"clim_{t}" for t in ALL_Y]

for label, cols in [("baseline(env+zone+clim, ec/moisture 없음)", temp_cols_base),
                     ("stacked(env+zone+clim+oof_ec+oof_moisture)", temp_cols_stack)]:
    scores = []
    for tr_i, va_i in gkf.split(D, D["soil_temp"], groups):
        dtr, dva = D.iloc[tr_i], D.iloc[va_i]
        enc = FT.fit_climatology(dtr, ALL_Y)
        xtr = FT.apply_climatology(dtr, enc)[cols]
        xva = FT.apply_climatology(dva, enc)[cols]
        m = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS["soil_temp"])
        m.fit(xtr, dtr["soil_temp"])
        scores.append(float(np.sqrt(np.mean((dva["soil_temp"] - m.predict(xva)) ** 2))))
    print(f"  [{label}] folds={np.round(scores,4)} mean={np.mean(scores):.4f}")

print("\n=== 3) 최종 학습 + test 예측 (그래프용) ===")
# ec/moisture: 전체 train으로 재학습 -> test 예측 (output/submission.csv와 동일 로직)
enc_full = FT.fit_climatology(D, ALL_Y)
Dc = FT.apply_climatology(D, enc_full)

st_te = st[st.split == "test"]
ms_daily_te = MS.daily_summary(MS.extract_split("dataset/test/ms"))
Xte = FT.build(teX, st_te, ms_daily_te, ms_fill, n_zones)
Xte = FT.apply_climatology(Xte, enc_full).sort_values(["dat", "mod"]).reset_index(drop=True)

ec_moist_models = {}
for tgt in ["soil_ec", "soil_moisture"]:
    cols = FT.select(base_cols, tgt) + [f"clim_{t}" for t in ALL_Y]
    m = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS[tgt])
    m.fit(Dc[cols], Dc[tgt])
    ec_moist_models[tgt] = m
    Xte[f"oof_{tgt.split('_')[1]}"] = m.predict(Xte[cols])  # oof_ec / oof_moisture (test용은 실제 예측치)

# soil_temp: stacked 피처로 최종 학습
m_temp = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS["soil_temp"])
m_temp.fit(Dc[temp_cols_stack], Dc["soil_temp"])
temp_pred = m_temp.predict(Xte[temp_cols_stack])

out = pd.DataFrame({
    "time": Xte["dat"].map(lambda d: f"DAT{int(d)}") + " "
            + Xte["mod"].map(lambda mm: f"{int(mm)//60:02d}:{int(mm)%60:02d}"),
    "dat": Xte["dat"].astype(int),
    "soil_temp_stacked": temp_pred,
})
out.to_csv(os.path.join(ROOT, "python", "temp_stacked_test_pred.csv"), index=False)
print("저장: temp_stacked_test_pred.csv")
print(out[["soil_temp_stacked"]].describe().loc[["mean","min","max"]].round(3).to_string())
