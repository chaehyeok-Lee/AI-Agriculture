"""공유 전처리 함수 — train.py / inference.py 에서 재사용

경로 규칙:
- 이 파일(EDA/로컬 검증용): dataset/... (로컬에 있는 전체 실데이터)
- train.py / inference.py (Docker 실행용): input/dataset/... (채점 시 실제 마운트되는 경로)
  input/dataset/는 대회 측이 준 샘플(1일치)이라 전체 검증엔 못 씀 — 구조 확인용
"""
import pandas as pd


ACTUATOR_ONOFF_COLS = ["fcu_fan", "fcu_pump", "circ_fan", "co2_supply", "fogging"]
DEAD_COLS = ["tube_rail_valve"]

# EDA에서 확인된 train/test 범위 불일치 컬럼 — train+test를 합친 범위로 안전장치를 걸어둠
# (지금 있는 데이터는 이미 이 범위 안이라 값이 바뀌진 않고, 나중에 스케일링 기준을 잡거나
#  더 극단적인 미지값이 들어왔을 때 모델이 깨지지 않게 막아주는 용도)
CLIP_RANGES = {
    "greenhouse_roof_vent1": (0.0, 59.0),
    "wind_speed_outside": (0.0, 8.28),
    "temperature": (-0.2, 26.1),  # 내부 온도
    "humidity": (39.0, 100.0),  # 내부 습도
}


def make_dt_index(df):
    """'DAT109 00:00' 같은 time 문자열을 timedelta 인덱스로 변환"""
    dat_num = df["time"].str.extract(r"DAT(\d+)")[0].astype(int)
    hm = df["time"].str.split(" ").str[1]
    dt = pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(hm + ":00")
    return df.assign(dt=dt).set_index("dt").sort_index().drop(columns="time")


def map_actuator_onoff(df):
    """on/off 액추에이터 컬럼: 0/201 -> 0/1"""
    df = df.copy()
    for col in ACTUATOR_ONOFF_COLS:
        df[col] = (df[col] > 0).astype(int)
    return df


def drop_dead_columns(df):
    """분산이 0인(항상 같은 값) 컬럼 제거 — tube_rail_valve"""
    return df.drop(columns=[c for c in DEAD_COLS if c in df.columns])


def clip_out_of_range(df):
    df = df.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def add_time_features(df):
    """day_num(경과일수), hour(시간대) 추가 — day_num이 soil_moisture와 상관 0.72로 특히 중요"""
    df = df.copy()
    df["day_num"] = df.index // pd.Timedelta("1D")
    df["hour"] = (df.index % pd.Timedelta("1D")) / pd.Timedelta("1h")
    return df


def aggregate_to_5min(df_1min):
    """1분 단위 데이터를 5분 그리드로 집계 (평균/최댓값/최솟값/표준편차/마지막값).

    resample 기본값은 [T, T+5분) 구간을 T로 라벨링하는데, 그러면 y(T) 실측 시점보다
    "미래"인 T+1~T+4 데이터가 T 라벨의 피처에 섞여 들어가 인과관계가 거꾸로 됨.
    여기서는 먼저 기본 방식으로 묶은 뒤 라벨을 5분 뒤로 밀어서, 라벨 T의 피처가
    (T-5분, T] 즉 "T 이전 5분" 데이터만 담도록 만든다 (y(T)를 예측하기 직전까지의 정보).
    """
    agg = df_1min.resample("5min").agg(["mean", "max", "min", "std", "last"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg.index = agg.index + pd.Timedelta("5min")
    return agg


def make_target_grid(df_1min):
    """df_1min과 같은 기간의 5분 그리드(day 시작 00:00부터) 생성.
    aggregate_to_5min 결과를 이 그리드에 맞춰 reindex하면 행 개수/라벨이 정확히 맞음.
    맨 첫 구간(예: 기간 첫날 00:00)은 '이전 5분' 데이터가 아예 없어서 reindex 후 NaN이 됨 —
    버그가 아니라 실제로 그 시점엔 과거 데이터가 없다는 뜻.
    """
    first_day = int(df_1min.index.min() // pd.Timedelta("1D"))
    start = pd.Timedelta(days=first_day)
    end = df_1min.index.max()
    return pd.timedelta_range(start=start, end=end, freq="5min")


def merge_image_features(feat_5min, ms_matched_path):
    """(선택) 02_ms_eda.py에서 만든 train_ms_matched.pkl / test_ms_matched.pkl 병합"""
    ms = pd.read_pickle(ms_matched_path).set_index("dt")
    return feat_5min.join(ms, how="left")


def build_features(raw_df, ms_matched_path=None):
    """train_X/test_X 원본 -> 모델 입력 피처까지 한 번에"""
    df = make_dt_index(raw_df)
    df = map_actuator_onoff(df)
    df = drop_dead_columns(df)
    df = clip_out_of_range(df)

    numeric_1min = df.select_dtypes("number")
    feat_5min = aggregate_to_5min(numeric_1min)
    feat_5min = feat_5min.reindex(make_target_grid(numeric_1min))
    feat_5min = add_time_features(feat_5min)

    if ms_matched_path is not None:
        feat_5min = merge_image_features(feat_5min, ms_matched_path)

    return feat_5min


def load_target(y_path):
    raw_y = pd.read_csv(y_path)
    return make_dt_index(raw_y)


def time_based_split(feat_df, target_df, val_days=4):
    """시계열 데이터라 랜덤 분할 금지 — 마지막 val_days일을 시간순으로 검증셋 분리"""
    last_day = feat_df.index.max() // pd.Timedelta("1D")
    cutoff = pd.Timedelta(days=int(last_day) - val_days + 1)
    train_mask = feat_df.index < cutoff
    return (
        feat_df[train_mask], target_df[train_mask],
        feat_df[~train_mask], target_df[~train_mask],
    )


if __name__ == "__main__":
    train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    train_y = load_target("dataset/train/env/train_y.csv")
    print("=== train ===")
    print("train_feat shape:", train_feat.shape, "(기대: 7488행)")
    print("train_y shape:", train_y.shape, "(기대: 7488행)")
    print("인덱스 일치:", (train_feat.index == train_y.index).all())
    print("맨 첫 행 NaN 개수(경계, 정상):", train_feat.iloc[0].isna().sum(), "/", train_feat.shape[1])

    test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))
    print("\n=== test ===")
    print("test_feat shape:", test_feat.shape, "(기대: 3456행)")
    print("맨 첫 행 NaN 개수(경계, 정상):", test_feat.iloc[0].isna().sum(), "/", test_feat.shape[1])

    print("\n컬럼 목록:")
    print(list(train_feat.columns))
