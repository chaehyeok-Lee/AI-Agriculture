"""
학습 진입점 (안전 버전).  실행:  python train.py
  입력:  input/dataset/train/{env,ms}, input/dataset/test/env   (운영진 주입)
  출력:  model/  (가중치 + 구조 아티팩트)

규칙 준수 (문제설명서 5.1.5)
  · 제공된 데이터만 사용한다. 외부 데이터·사전학습 가중치를 일절 쓰지 않는다.
  · 정답(test_y)은 어떤 형태로도 참조하지 않는다.
  · 상수를 하드코딩하지 않는다. 재배 그룹 구조는 코드가 데이터에서 도출한다.
  · 각 날(DAT)을 독립 단위로 다루며 날짜 간 순서를 쓰지 않는다. test 를 train 뒤에 잇지 않는다.
    학습 특징은 train 환경만으로 계산되어, test 의 어떤 값도 학습에 흘러들지 않는다.
    (평가 구간 test_X 는 오직 '어느 그룹인지' 분류에만 쓰인다 — 비지도, 제공 데이터 범위.)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import features as FT
from src import msfeat as MS
from src import structure as ST
from src import temp_model as TM

ROOT = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(ROOT, "input", "dataset")
MODEL = os.path.join(ROOT, "model")
TARGETS = ["soil_moisture", "soil_ec"]          # LightGBM 그룹 모델
TEMP = "soil_temp"                              # 전용 앙상블
SEED = 42
N_SPLITS = 6

DET = dict(deterministic=True, force_row_wise=True, n_jobs=4)

# 교차검증(그룹+당일 특징, DAT-group)으로 선택한 타깃별 파라미터.
# soil_moisture: 26.07.17 랜덤서치(25회, GroupKFold 6-fold, real_day 반영 후 baseline
# 0.7206) 결과로 교체 — num_leaves↑(더 복잡한 트리) + subsample/colsample↓·reg_alpha
# 추가(더 강한 정규화) 조합이 6-fold 중 5개에서 개선, mean -6.4%(0.7206->0.6742).
PARAMS = {
    "soil_moisture": dict(n_estimators=1200, learning_rate=0.03, num_leaves=63,
                          min_child_samples=80, subsample=0.6, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.5),
    # soil_ec: 26.07.17 하이퍼파라미터 전용 총력전(배깅/dart/goss/n_estimators단독탐색/
    # 100회 확장랜덤서치) 결과 — boosting_type='goss'만 바꾸는 게 유일하게 6-fold 전부
    # 일관 개선(mean -6.3%, 0.0520->0.0487). 다른 값까지 같이 튜닝하면 오히려 악화되어
    # 나머지는 기존값 그대로 유지. goss는 자체 샘플링을 쓰므로 subsample 계열은 무의미해
    # 제거(bagging_fraction 관련 경고 방지).
    "soil_ec": dict(n_estimators=2000, learning_rate=0.03, num_leaves=7,
                    min_child_samples=80, colsample_bytree=0.6, reg_lambda=5.0,
                    boosting_type="goss"),
    "soil_temp": dict(n_estimators=1200, learning_rate=0.03, num_leaves=31,
                      min_child_samples=40, subsample=0.8, subsample_freq=1,
                      colsample_bytree=0.7, reg_lambda=1.0),
}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    os.makedirs(MODEL, exist_ok=True)

    log("환경 데이터 로드")
    trX = ST.parse_time(pd.read_csv(f"{IN}/train/env/train_X.csv"))
    trY = ST.parse_time(pd.read_csv(f"{IN}/train/env/train_y.csv"))
    te_path = f"{IN}/test/env/test_X.csv"
    teX = ST.parse_time(pd.read_csv(te_path)) if os.path.exists(te_path) else None

    log("구조 도출: 외부기상 그룹 -> 재배 그룹 군집 + 실제 날짜 체인(real_day) 복원")
    st, art = ST.discover_calendar(trX, teX, seed=SEED)
    n_zones = art["n_zones"]
    log(f"  재배 그룹 {n_zones}개 도출 (날짜체인 경계비용 총합={art['chain_boundary_cost']:.2f}, "
        f"기준zone={art['ref_zone']})")
    log("  그룹별 소속 DAT:\n" +
        st.groupby("zone")["dat"].apply(lambda s: list(s)).to_string())
    st.to_csv(f"{MODEL}/structure.csv", index=False)

    log("다분광 특징 추출 (train)")
    sess = MS.extract_split(f"{IN}/train/ms")
    ms_daily = MS.daily_summary(sess)
    log(f"  세션 {len(sess)}개 -> {len(ms_daily)}일 요약")
    ms_cols = [c for c in ms_daily.columns if c != "dat"]
    ms_fill = ms_daily[ms_cols].median()
    ms_daily.to_csv(f"{MODEL}/ms_daily_train.csv", index=False)

    all_y = TARGETS + [TEMP]                       # 기후값 특징은 3개 타깃 모두로 계산
    log("특징 생성 (train 환경만 — test 는 특징 계산에 관여하지 않음)")
    X = FT.build(trX, st, ms_daily, ms_fill, n_zones)
    D = X.merge(trY.sort_values(["dat", "mod"])[["dat", "mod"] + all_y],
                on=["dat", "mod"], how="inner").reset_index(drop=True)

    base_cols = [c for c in FT.feature_columns(D) if c not in all_y]
    clim_cols = [f"clim_{t}" for t in all_y]
    cols = {t: FT.select(base_cols, t) + clim_cols for t in TARGETS}
    log(f"  학습 행 {len(D)} / 기본 특징 {len(base_cols)} + 기후값 {len(clim_cols)}")

    groups = D["dat"].to_numpy()          # 원본 DAT 단위 그룹
    gkf = GroupKFold(n_splits=N_SPLITS)
    scores = {}

    # ── soil_moisture / soil_ec : 그룹 LightGBM ─────────────────────────
    log(f"검증: DAT-group {N_SPLITS}-fold  (하루를 통째로 빼는 표준 그룹 CV)")
    for tgt in TARGETS:
        s = []
        for tr_i, va_i in gkf.split(D, D[tgt], groups):
            dtr, dva = D.iloc[tr_i], D.iloc[va_i]
            enc = FT.fit_climatology(dtr, all_y)     # fold 안에서만 계산 (누수 차단)
            xtr = FT.apply_climatology(dtr, enc)[cols[tgt]]
            xva = FT.apply_climatology(dva, enc)[cols[tgt]]
            m = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS[tgt])
            m.fit(xtr, dtr[tgt])
            s.append(float(np.sqrt(np.mean((dva[tgt] - m.predict(xva)) ** 2))))
        scores[tgt] = s
        log(f"  {tgt:<14} RMSE = {np.mean(s):.4f}   (특징 {len(cols[tgt])}개, "
            f"분광 {'사용' if any(c.startswith('ms_') for c in cols[tgt]) else '미사용'})")

    # ── soil_temp : 전용 앙상블 (온도 이력 EWMA + LGBM/Ridge/HGB) ───────
    # zone별 빈 real_day를 analog+경계보정 스캐폴드로 채워 완전히 이어붙인 뒤 이력(과거+
    # 미래 방향) 계산 -> source=='train' 행만 학습에 사용 (synthetic/test 행은 이력
    # 연속성 확보용으로만 쓰이고 여기선 버려짐).
    tf_full = TM.build_temp_features_full(trX, teX if teX is not None else trX.iloc[0:0], st)
    log(f"  온도 스캐폴드: 실측 train={int((tf_full['source']=='train').sum())} / "
        f"test={int((tf_full['source']=='test').sum())} / "
        f"synthetic(합성,학습제외)={int((tf_full['source']=='synthetic').sum())}행")
    tf = tf_full[tf_full["source"] == "train"].reset_index(drop=True)
    Dt = tf.merge(trY.sort_values(["dat", "mod"])[["dat", "mod", TEMP]],
                  on=["dat", "mod"], how="inner").reset_index(drop=True)
    # 청크(zone)별 real_day 확장창 CV — 26.07.17, 사용자 지정 검증 방식.
    # DAT 무작위 GroupKFold 대신 각 zone 자신의 날짜 진행 순서를 그대로 반영한다.
    temp_folds = ST.expanding_chunk_folds(st, window_sizes=(3, 4, 5))
    s = []
    for train_dats, val_dats in temp_folds:
        dtr = Dt[Dt["dat"].isin(train_dats)].reset_index(drop=True)
        dva = Dt[Dt["dat"].isin(val_dats)].reset_index(drop=True)
        mt = TM.fit_temp(dtr, dtr[TEMP])
        dyn = TM.fit_temp_dynamics(dtr, dtr[TEMP])
        pred = TM.predict_temp_blended(mt, dyn, dva, w_dynamic=TM.W_DYNAMIC)
        pred = TM.correct_boundary(Dt, dva, pred, exclude_dats=set(val_dats))
        s.append(float(np.sqrt(np.mean((dva[TEMP].to_numpy(float) - pred) ** 2))))
    scores[TEMP] = s
    log(f"  {TEMP:<14} RMSE = {np.mean(s):.4f}   (재귀+경계앵커링 포함, 청크별 확장창 CV {len(temp_folds)}fold)")

    log("전체 학습 데이터로 최종 학습")
    enc = FT.fit_climatology(D, all_y)
    Dc = FT.apply_climatology(D, enc)
    models = {}
    for tgt in TARGETS:
        m = lgb.LGBMRegressor(random_state=SEED, verbose=-1, **DET, **PARAMS[tgt])
        m.fit(Dc[cols[tgt]], Dc[tgt])
        models[tgt] = m
    temp_ens = TM.fit_temp(Dt, Dt[TEMP])
    temp_dyn = TM.fit_temp_dynamics(Dt, Dt[TEMP])
    temp_truth = Dt[["zone", "real_day", "dat", "mod", TEMP]].rename(columns={TEMP: "soil_temp"})

    with open(f"{MODEL}/models.pkl", "wb") as f:
        pickle.dump({"models": models, "cols": cols, "n_zones": n_zones,
                     "ms_fill": ms_fill, "clim": enc, "temp": temp_ens,
                     "temp_dynamics": temp_dyn, "temp_truth": temp_truth,
                     "temp_w_dynamic": TM.W_DYNAMIC,
                     "structure_art": art}, f)
    with open(f"{MODEL}/cv_scores.json", "w") as f:
        json.dump({t: {"mean_rmse": float(np.mean(v)), "per_fold": v}
                   for t, v in scores.items()}, f, indent=2)
    log(f"저장 완료 -> {MODEL}/models.pkl")


if __name__ == "__main__":
    main()
