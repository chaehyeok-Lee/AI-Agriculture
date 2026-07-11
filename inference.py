"""온라인테스트1 추론: model/model.pkl 로 test_X 예측, output/submission.csv 생성.

경로 규칙(train.py와 동일): 로컬 검증은 dataset/... 사용.
Docker 제출용으로 바꿀 때는 input/dataset/...로 교체 필요 (PLAN.md 4단계 항목).
"""
import os
import pickle

import pandas as pd

from preprocess import build_features
from train import DROP_COLS_PER_TARGET, TARGET_COLS

COLUMNS = ["time", "soil_moisture", "soil_ec", "soil_temp"]


def format_time_index(index):
    """timedelta 인덱스를 'DAT109 00:00' 형식 문자열로 되돌림 (train_y.csv와 동일 포맷)."""
    days = (index // pd.Timedelta("1D")).astype(int)
    remainder = index - pd.to_timedelta(days, unit="D")
    total_minutes = (remainder / pd.Timedelta("1min")).round().astype(int)
    hh, mm = total_minutes // 60, total_minutes % 60
    return [f"DAT{d} {h:02d}:{m:02d}" for d, h, m in zip(days, hh, mm)]


def main():
    with open("model/model.pkl", "rb") as f:
        models = pickle.load(f)

    test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))

    submission = pd.DataFrame({"time": format_time_index(test_feat.index)})
    for col in TARGET_COLS:
        cols = [c for c in test_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
        submission[col] = models[col].predict(test_feat[cols])

    os.makedirs("output", exist_ok=True)
    submission[COLUMNS].to_csv("output/submission.csv", index=False)
    print(f"저장: output/submission.csv (행: {len(submission)})")


if __name__ == "__main__":
    main()
