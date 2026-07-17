from pathlib import Path
import pickle

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

TRAIN_X_PATH = ROOT / "input" / "dataset" / "train" / "env" / "train_X.csv"
TRAIN_Y_PATH = ROOT / "input" / "dataset" / "train" / "env" / "train_y.csv"
TEST_X_PATH = ROOT / "input" / "dataset" / "test" / "env" / "test_X.csv"

OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


TARGET_COLS = ["soil_moisture", "soil_ec", "soil_temp"]


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[읽기 성공] {path.name} / encoding={enc} / shape={df.shape}")
            return df
        except Exception:
            pass
    raise RuntimeError(f"CSV 읽기 실패: {path}")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["dat_id"] = df["time"].astype(str).str.extract(r"(DAT\d+)")
    df["dat_num"] = df["dat_id"].str.replace("DAT", "", regex=False).astype(int)

    hhmm = df["time"].astype(str).str.extract(r"(\d{2}:\d{2})")[0]
    hm = hhmm.str.split(":", expand=True)

    df["hour"] = hm[0].astype(int)
    df["minute"] = hm[1].astype(int)
    df["minute_of_day"] = df["hour"] * 60 + df["minute"]

    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)
    df["is_night"] = 1 - df["is_day"]

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)

    return df


def calc_vpd(temp: pd.Series, humidity: pd.Series) -> pd.Series:
    temp = temp.astype(float)
    humidity = humidity.astype(float).clip(0, 100)

    saturation_vapor_pressure = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    actual_vapor_pressure = saturation_vapor_pressure * humidity / 100.0
    vpd = saturation_vapor_pressure - actual_vapor_pressure

    return vpd.clip(lower=0)


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    주의:
    temperature_outside 음수는 겨울철 실제 외부기온일 수 있으므로 제거하지 않는다.
    test의 실내 temperature 음수도 일단 삭제하지 않는다.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df.loc[df[col] == 9999, col] = np.nan
        df.loc[df[col] >= 9000, col] = np.nan

    return df


def add_basic_agri_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "temperature" in df.columns and "humidity" in df.columns:
        df["vpd_inside"] = calc_vpd(df["temperature"], df["humidity"])

    if "temperature_outside" in df.columns and "humidity_outside" in df.columns:
        df["vpd_outside"] = calc_vpd(df["temperature_outside"], df["humidity_outside"])

    if "temperature" in df.columns and "temperature_outside" in df.columns:
        df["temp_gap_inside_outside"] = df["temperature"] - df["temperature_outside"]

    if "humidity" in df.columns and "humidity_outside" in df.columns:
        df["humidity_gap_inside_outside"] = df["humidity"] - df["humidity_outside"]

    if "greenhouse_roof_vent1" in df.columns and "greenhouse_roof_vent2" in df.columns:
        df["roof_vent_mean"] = (
            df["greenhouse_roof_vent1"] + df["greenhouse_roof_vent2"]
        ) / 2

    control_cols = [
        "greenhouse_roof_vent1",
        "greenhouse_roof_vent2",
        "shading_curtain",
        "thermal_curtain",
        "fcu_fan",
        "fcu_pump",
        "circ_fan",
        "co2_supply",
        "tube_rail_valve",
        "fogging",
    ]

    for col in control_cols:
        if col in df.columns:
            df[f"{col}_on"] = (df[col] > 0).astype(int)

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1분 데이터에서 과거 흐름을 만든다.
    미래값은 사용하지 않는다.
    """
    df = df.copy()
    df = df.sort_values(["dat_num", "minute_of_day"]).reset_index(drop=True)

    rolling_cols = [
        "temperature_outside",
        "humidity_outside",
        "solar_radiation",
        "wind_speed_outside",
        "rainfall",
        "greenhouse_roof_vent1",
        "greenhouse_roof_vent2",
        "shading_curtain",
        "thermal_curtain",
        "fcu_fan",
        "fcu_pump",
        "circ_fan",
        "co2_supply",
        "fogging",
        "temperature",
        "humidity",
        "co2",
        "vpd_inside",
        "vpd_outside",
        "temp_gap_inside_outside",
        "humidity_gap_inside_outside",
    ]

    rolling_cols = [c for c in rolling_cols if c in df.columns]

    for col in rolling_cols:
        g = df.groupby("dat_num")[col]

        df[f"{col}_diff1"] = g.diff(1).fillna(0)
        df[f"{col}_diff5"] = g.diff(5).fillna(0)

        df[f"{col}_roll5_mean"] = g.transform(
            lambda s: s.rolling(5, min_periods=1).mean()
        )
        df[f"{col}_roll15_mean"] = g.transform(
            lambda s: s.rolling(15, min_periods=1).mean()
        )
        df[f"{col}_roll60_mean"] = g.transform(
            lambda s: s.rolling(60, min_periods=1).mean()
        )

    return df


def make_env_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print("\n" + "=" * 80)
    print(f"[{name}] ENV 피처 생성")
    print("=" * 80)

    df = add_time_features(df)
    df = clean_values(df)
    df = add_basic_agri_features(df)
    df = add_rolling_features(df)

    df_5min = df[df["minute_of_day"] % 5 == 0].copy()
    df_5min = df_5min.sort_values(["dat_num", "minute_of_day"]).reset_index(drop=True)

    print(f"{name} 원본 shape       : {df.shape}")
    print(f"{name} 5분 피처 shape  : {df_5min.shape}")
    print(f"{name} 5분 time head")
    print(df_5min["time"].head())
    print(f"{name} 5분 time tail")
    print(df_5min["time"].tail())

    return df_5min


def fill_missing_by_train_median(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    데이터 누수 방지를 위해 중앙값은 train에서만 계산하고 test에 적용한다.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    medians = {}

    for col in numeric_cols:
        med = train_df[col].median()

        if pd.isna(med):
            med = 0

        medians[col] = med
        train_df[col] = train_df[col].fillna(med)

        if col in test_df.columns:
            test_df[col] = test_df[col].fillna(med)

    return train_df, test_df, medians


def main():
    print("=" * 80)
    print("features_env.py 시작 - ENV 1분 데이터 → 5분 피처 생성")
    print("=" * 80)

    train_x = read_csv_safely(TRAIN_X_PATH)
    train_y = read_csv_safely(TRAIN_Y_PATH)
    test_x = read_csv_safely(TEST_X_PATH)

    train_env = make_env_features(train_x, "train_X")
    test_env = make_env_features(test_x, "test_X")

    print("\n" + "=" * 80)
    print("[train_X / train_y] time merge 확인")
    print("=" * 80)

    merged = train_env.merge(train_y, on="time", how="inner")

    print(f"train_env rows: {len(train_env)}")
    print(f"train_y rows  : {len(train_y)}")
    print(f"merge rows    : {len(merged)}")

    if len(merged) != len(train_y):
        raise RuntimeError("train_env와 train_y merge 행 수가 맞지 않습니다.")

    missing_targets = [c for c in TARGET_COLS if c not in merged.columns]
    if missing_targets:
        raise RuntimeError(f"타깃 컬럼 없음: {missing_targets}")

    y_train = merged[["time"] + TARGET_COLS].copy()
    x_train = merged.drop(columns=TARGET_COLS)

    x_train, test_env, medians = fill_missing_by_train_median(x_train, test_env)

    print("\n" + "=" * 80)
    print("[최종 저장 shape]")
    print("=" * 80)
    print(f"X_train_env shape: {x_train.shape}")
    print(f"y_train shape    : {y_train.shape}")
    print(f"X_test_env shape : {test_env.shape}")

    x_train.to_pickle(OUTPUT_DIR / "train_env_features.pkl")
    y_train.to_pickle(OUTPUT_DIR / "train_y_aligned.pkl")
    test_env.to_pickle(OUTPUT_DIR / "test_env_features.pkl")

    with open(OUTPUT_DIR / "env_medians.pkl", "wb") as f:
        pickle.dump(medians, f)

    x_train.head(20).to_csv(
        OUTPUT_DIR / "train_env_features_head.csv",
        index=False,
        encoding="utf-8-sig",
    )
    y_train.head(20).to_csv(
        OUTPUT_DIR / "train_y_aligned_head.csv",
        index=False,
        encoding="utf-8-sig",
    )
    test_env.head(20).to_csv(
        OUTPUT_DIR / "test_env_features_head.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[저장 완료]")
    print(OUTPUT_DIR / "train_env_features.pkl")
    print(OUTPUT_DIR / "train_y_aligned.pkl")
    print(OUTPUT_DIR / "test_env_features.pkl")
    print(OUTPUT_DIR / "env_medians.pkl")

    print("\n" + "=" * 80)
    print("features_env.py 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()