"""
답안 생성 진입점 (안전 버전).  실행:  python inference.py
  입력:  input/dataset/  (운영진 주입) + model/  (train.py 산출)
  출력:  output/submission.csv   (형식은 train_y.csv 와 동일)

test 특징은 test 환경만으로 계산한다 (train 뒤에 잇지 않는다).
학습 때 저장한 그룹 기후값(clim)을 test 의 (그룹, 시각)에 그대로 적용한다.
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import features as FT
from src import msfeat as MS
from src import structure as ST
from src import temp_model as TM

ROOT = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(ROOT, "input", "dataset")
OUT = os.path.join(ROOT, "output")
MODEL = os.path.join(ROOT, "model")
TARGETS = ["soil_moisture", "soil_ec"]      # LightGBM 그룹 모델
TEMP = "soil_temp"                          # 전용 앙상블
ALL_TARGETS = ["soil_moisture", "soil_ec", "soil_temp"]
SEED = 42
CLIP_PAD = 0.10          # 학습에서 관측된 물리적 범위 + 10% 여유로 클립


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{MODEL}/models.pkl", "rb") as f:
        B = pickle.load(f)
    models, cols = B["models"], B["cols"]
    n_zones, ms_fill, enc, temp_ens = B["n_zones"], B["ms_fill"], B["clim"], B["temp"]
    temp_dyn, temp_truth, temp_w = B["temp_dynamics"], B["temp_truth"], B["temp_w_dynamic"]

    log("환경 데이터 로드")
    trX = ST.parse_time(pd.read_csv(f"{IN}/train/env/train_X.csv"))
    trY = pd.read_csv(f"{IN}/train/env/train_y.csv")
    teX = ST.parse_time(pd.read_csv(f"{IN}/test/env/test_X.csv"))

    log("구조 도출 (train.py 와 동일한 결정적 절차) — test 를 재배 그룹에 분류 + 날짜체인 복원")
    st, _ = ST.discover_calendar(trX, teX, seed=SEED)
    st_te = st[st.split == "test"]
    log("  그룹별 test DAT:\n" +
        st_te.groupby("zone")["dat"].apply(lambda s: list(s)).to_string())

    log("다분광 특징 추출 (test)")
    ms_daily = MS.daily_summary(MS.extract_split(f"{IN}/test/ms"))
    log(f"  {len(ms_daily)}일 요약")

    log("특징 생성 (test 환경만으로 계산)")
    X = FT.build(teX, st_te, ms_daily, ms_fill, n_zones)
    X = FT.apply_climatology(X, enc)
    X = X.sort_values(["dat", "mod"]).reset_index(drop=True)

    needed = sorted({c for cs in cols.values() for c in cs})
    for c in needed:
        if c not in X.columns:
            X[c] = float(ms_fill[c]) if c in ms_fill.index else np.nan
    X = X.copy()

    log("예측")
    out = pd.DataFrame({
        "time": X["dat"].map(lambda d: f"DAT{int(d)}") + " "
                + X["mod"].map(lambda m: f"{int(m) // 60:02d}:{int(m) % 60:02d}")
    })
    # soil_temp: 전용 앙상블 (train 관측치 참고—X만, 정답 아님—로 청크경계·미래방향까지
    # 포함한 완전한 이력 계산 후 test 행만 사용, key=[dat,mod]로 정렬 매칭)
    # + 26.07.17 재귀(당일 첫 값에서 시작하는 열역학 재귀) 블렌드 + 경계 앵커링(같은 zone
    # 내 바로 이전/다음 real_day의 실측 train soil_temp 쪽으로 곡선을 당겨 붙임).
    tf_full = TM.build_temp_features_full(trX, teX, st)
    tf = tf_full[tf_full["source"] == "test"].reset_index(drop=True)
    tf_pred = TM.predict_temp_blended(temp_ens, temp_dyn, tf, w_dynamic=temp_w)
    tf_pred = TM.correct_boundary(temp_truth, tf, tf_pred, exclude_dats=set())
    tf["p_temp"] = tf_pred
    pred = {(int(d), int(m)): float(p)
            for d, m, p in zip(tf["dat"], tf["mod"], tf["p_temp"])}
    tpred = np.array([pred[(int(d), int(m))] for d, m in zip(X["dat"], X["mod"])])

    def clip(tgt, p):
        lo, hi = float(trY[tgt].min()), float(trY[tgt].max())
        pad = (hi - lo) * CLIP_PAD
        p = np.clip(p, lo - pad, hi + pad)
        bad = ~np.isfinite(p)            # NaN/inf 는 채점 패널티 -> 절대 남기지 않는다
        if bad.any():
            log(f"  경고: {tgt} 비유효값 {int(bad.sum())}개 -> 학습 중앙값 대체")
            p[bad] = float(trY[tgt].median())
        return p

    for tgt in TARGETS:
        out[tgt] = clip(tgt, models[tgt].predict(X[cols[tgt]]))
    out[TEMP] = clip(TEMP, tpred)
    out = out[["time"] + ALL_TARGETS]          # train_y.csv 열 순서와 동일하게

    path = os.path.join(OUT, "submission.csv")
    out.to_csv(path, index=False)
    log(f"저장 완료 -> {path}  ({len(out)} 행)")
    log("  요약:\n" +
        out[ALL_TARGETS].describe().loc[["mean", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
