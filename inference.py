"""온라인테스트1 추론: model/model.pkl 로 test_X 예측, output/submission.csv 생성.

경로 규칙(train.py와 동일): 로컬 검증은 dataset/... 사용.
Docker 제출용으로 바꿀 때는 input/dataset/...로 교체 필요 (PLAN.md 4단계 항목).
"""
import os
import pickle

import pandas as pd

from preprocess import build_features
from train import DROP_COLS_PER_TARGET, TARGET_COLS, add_trend_features

COLUMNS = ["time", "soil_moisture", "soil_ec", "soil_temp"]
EXPECTED_ROWS = 3456  # test 12일 * 5분 간격(288/일)
# 물리적으로 말이 되는 범위(train_y 실측 29.6~43.5 / 0.34~2.92 / 3.8~17.7 대비 넉넉한 여유).
# 모델 성능엔 관여하지 않고, 파이프라인 버그·극단 외삽(예: 이전에 실패한 실험에서 나온 음수 EC)만 잡는 안전장치.
SANITY_RANGES = {
    "soil_moisture": (0, 100),
    "soil_ec": (0, 10),
    "soil_temp": (-10, 45),
}


def validate_submission(df):
    assert list(df.columns) == COLUMNS, f"컬럼 불일치: {list(df.columns)}"
    assert len(df) == EXPECTED_ROWS, f"행 개수 불일치: {len(df)} (기대: {EXPECTED_ROWS})"
    assert not df.isna().any().any(), "NaN이 포함된 예측값이 있음"
    assert df["time"].is_unique, "time 컬럼에 중복된 시각이 있음"
    for col, (lo, hi) in SANITY_RANGES.items():
        assert df[col].between(lo, hi).all(), f"{col} 값이 정상 범위[{lo},{hi}]를 벗어남: min={df[col].min()}, max={df[col].max()}"
    print("검증 통과: 컬럼/행개수/NaN/중복시각/값범위 전부 정상")


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
    test_feat = add_trend_features(test_feat)

    submission = pd.DataFrame({"time": format_time_index(test_feat.index)})
    for col in TARGET_COLS:
        cols = [c for c in test_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
        submission[col] = models[col].predict(test_feat[cols])

    submission = submission[COLUMNS]
    validate_submission(submission)

    os.makedirs("output", exist_ok=True)
    submission.to_csv("output/submission.csv", index=False)
    print(f"저장: output/submission.csv (행: {len(submission)})")


if __name__ == "__main__":
    main()
