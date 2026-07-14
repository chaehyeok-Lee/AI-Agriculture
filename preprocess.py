"""공유 전처리 함수 — train.py / inference.py 에서 재사용

경로 규칙:
- 이 파일(EDA/로컬 검증용): dataset/... (로컬에 있는 전체 실데이터)
- train.py / inference.py (Docker 실행용): input/dataset/... (채점 시 실제 마운트되는 경로)
  input/dataset/는 대회 측이 준 샘플(1일치)이라 전체 검증엔 못 씀 — 구조 확인용
"""
import os
import re

import numpy as np
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


def compute_ec_high_confidence(raw_df):
    """"고EC 관리이벤트(시비 농도 상향)" 소프트 신뢰도[0,1]를 일 단위로 계산.

    train(DAT121~128, soil_ec ~1.7)에서 관측된 패턴: 보온커튼(thermal_curtain)과
    차광커튼(shading_curtain)이 하루 동안 거의 완전히 동기화되어 움직이고(agreement>=98%,
    둘 다 완전개방/완전폐쇄 지점을 하루 중 한 번씩은 지남) AND 순환팬(circ_fan)이 대체로
    꺼져있으면(일평균<15, raw 0~201 스케일) 고EC 레짐일 가능성이 높다고 판단.

    ⚠️ 이 규칙은 train 전체에서 단 1번(121~128일)만 관측된 패턴이라 100% 확신할 근거는
    아님 — test(135~146일)에서 동일 신호가 DAT141~143에 재현되는 것은 확인했으나(circ_fan
    거의 OFF + 커튼 완전동기), 외부 날씨가 test 구간 내 게이트ON/OFF 사이에 전혀 차이가
    없어(3일 주기 반복 템플릿) 실제 시비 결정과 무관한 우연일 가능성도 배제 못 함 —
    그래서 0/1 하드 분류가 아니라 연속값 신뢰도로 리턴, 소프트 블렌드로만 사용할 것.
    (자세한 근거: FEEDBACK.md 루프 9)

    Returns: {day_num(int): confidence(float, 0~1)} 딕셔너리.
    circ_fan은 map_actuator_onoff로 0/1 이진화되기 *전* raw 스케일(0~201)을 써야
    임계값(<15)이 의미가 있음 — 이 함수는 raw_df를 직접 받아 자체적으로 dt 인덱스만 만들고
    이진화는 하지 않는다.
    """
    df = make_dt_index(raw_df.copy())
    df["day_num"] = df.index // pd.Timedelta("1D")
    conf = {}
    for d, g in df.groupby("day_num"):
        th, sh, fan = g["thermal_curtain"], g["shading_curtain"], g["circ_fan"]
        curtain_both_extremes = ((th >= 99) & (sh >= 99)).any() and ((th <= 1) & (sh <= 1)).any()
        agreement = (th.sub(sh).abs() <= 0.5).mean() >= 0.98
        gate = curtain_both_extremes and agreement
        fan_conf = float(np.clip((20 - fan.mean()) / 15, 0, 1))
        conf[int(d)] = fan_conf if gate else 0.0
    return conf


def add_ec_high_confidence(feat_df, raw_df):
    """build_features() 결과(5분 그리드)에 ec_high_confidence 컬럼을 day_num 매핑으로 추가.
    soil_ec 예측에서만 소프트 블렌드 용도로 사용 — 다른 타깃은 이 컬럼을 제외하고 학습한다."""
    conf = compute_ec_high_confidence(raw_df)
    feat_df = feat_df.copy()
    feat_df["ec_high_confidence"] = feat_df["day_num"].map(conf).fillna(0.0)
    return feat_df


MS_CUBE_BANDS, MS_CUBE_LINES, MS_CUBE_SAMPLES = 10, 1024, 1280  # ENVI BSQ, uint16, 713~920nm 10밴드


def _read_ms_cube(session_folder, stride=4):
    """ms 세션 폴더의 cube.raw(ENVI BSQ, uint16, 10밴드)를 읽어 (10, L/stride, S/stride) 배열로 반환.
    stride로 다운샘플링해 속도 확보(픽셀별 정밀도가 필요 없는 밴드평균 용도라 무손실일 필요 없음)."""
    raw = np.fromfile(os.path.join(session_folder, "cube.raw"), dtype="<u2")
    return raw.reshape(MS_CUBE_BANDS, MS_CUBE_LINES, MS_CUBE_SAMPLES).astype(np.float32)[:, ::stride, ::stride]


def compute_daily_band_means(root, split, bands=(9, 10)):
    """다분광 원본 큐브에서 지정 밴드(1-indexed, 기본 9·10번=897·920nm NIR 끝단)의
    일별 평균 반사값을 계산 — 그날 촬영된 모든 세션(위치 무관)의 평균.

    PLAN.md에 "897·920nm를 NIR 평균 그룹에 뭉개지 말고 별도 피처로 남기는 걸 권장"이라고
    적혀 있었으나 미구현이던 아이디어를 실제로 구현. 4-fold 검증 결과(FEEDBACK.md 루프10)
    soil_ec에 +0.0006%(사실상 무변화, 노이즈 이하) — 유의미한 개선은 아니지만 악화도
    아니라서, 다분광 데이터 활용이 필수 요건일 때 안전하게 포함 가능한 최선의 선택지.
    day_num(int) -> {band: 평균값} 딕셔너리 반환. 세션이 없는 날은 결과에서 제외(NaN 처리
    는 호출부에서 day_num 매핑 시 자동으로 됨)."""
    base = os.path.join(root, split, "ms")
    if not os.path.isdir(base):
        return {}
    daily = {}
    for dat_folder in sorted(os.listdir(base)):
        dd = os.path.join(base, dat_folder)
        m = re.match(r"DAT(\d+)", dat_folder)
        if not os.path.isdir(dd) or not m:
            continue
        day = int(m.group(1))
        vals = {b: [] for b in bands}
        for sess in sorted(os.listdir(dd)):
            sess_path = os.path.join(dd, sess)
            if not os.path.isdir(sess_path) or not os.path.exists(os.path.join(sess_path, "cube.raw")):
                continue
            cube = _read_ms_cube(sess_path)
            for b in bands:
                vals[b].append(float(cube[b - 1].mean()))
        if any(vals.values()):
            daily[day] = {b: (float(np.mean(v)) if v else np.nan) for b, v in vals.items()}
    return daily


def add_ms_band_features(feat_df, root, split, bands=(9, 10)):
    """build_features() 결과(5분 그리드)에 다분광 밴드 평균 컬럼(ms_bandN_mean)을
    day_num 매핑으로 추가. 세션이 없는 날/그리드 시작 전 구간은 NaN(그날 촬영이 없거나
    ms 폴더 자체가 없는 경우 — LightGBM이 결측을 그대로 학습에 활용)."""
    daily = compute_daily_band_means(root, split, bands)
    feat_df = feat_df.copy()
    for b in bands:
        col = f"ms_band{b}_mean"
        feat_df[col] = feat_df["day_num"].map(lambda d: daily.get(int(d), {}).get(b, np.nan))
    return feat_df


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
