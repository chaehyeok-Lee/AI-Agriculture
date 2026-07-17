from pathlib import Path
import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "model"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

FEATURES_ENV_SCRIPT = ROOT / "features_env.py"

TRAIN_FEATURE_PATH = OUTPUT_DIR / "train_env_features.pkl"
TEST_FEATURE_PATH = OUTPUT_DIR / "test_env_features.pkl"
TRAIN_TARGET_PATH = OUTPUT_DIR / "train_y_aligned.pkl"

MODEL_PATH = MODEL_DIR / "env_model.pkl"
FEATURE_LIST_PATH = MODEL_DIR / "env_feature_columns.json"
METRIC_PATH = OUTPUT_DIR / "env_train_metrics.csv"

# [v3 추가] soil_temp 전용 단일 타깃 모델 저장 경로
SOIL_TEMP_MODEL_PATH = MODEL_DIR / "soil_temp_extra_trees.pkl"

TARGET_COLS = ["soil_moisture", "soil_ec", "soil_temp"]

# 최종 제출용 train.py는 inference.py가 사용하는 ENV 모델을 만든다.
# ENV+MS 모델은 실험 결과상 최종 후보에서 제외했으므로 여기서는 사용하지 않는다.
REBUILD_ENV_FEATURES = True


def run_features_env():
    """
    최종 재현성을 위해 train.py 실행 시 ENV 피처를 다시 만든다.
    이 덕분에 운영진 환경에서도 train.py -> inference.py 순서가 이어진다.
    """
    print("\n[ENV 피처 생성 단계]")

    if not FEATURES_ENV_SCRIPT.exists():
        raise FileNotFoundError(f"features_env.py 없음: {FEATURES_ENV_SCRIPT}")

    if REBUILD_ENV_FEATURES:
        print("features_env.py를 실행해 ENV 피처를 생성/갱신합니다.")
        subprocess.run(
            [sys.executable, str(FEATURES_ENV_SCRIPT)],
            cwd=ROOT,
            check=True,
        )
    else:
        print("REBUILD_ENV_FEATURES=False 이므로 기존 ENV 피처를 사용합니다.")


def check_required_files():
    print("\n[필수 파일 확인]")

    required_paths = [
        TRAIN_FEATURE_PATH,
        TEST_FEATURE_PATH,
        TRAIN_TARGET_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"필수 파일 없음: {path}")
        print(f"[확인] {path.name}")


def load_data():
    print("\n[데이터 불러오기]")

    train_df = pd.read_pickle(TRAIN_FEATURE_PATH)
    test_df = pd.read_pickle(TEST_FEATURE_PATH)
    y_df = pd.read_pickle(TRAIN_TARGET_PATH)

    print(f"train_env_features shape: {train_df.shape}")
    print(f"test_env_features shape : {test_df.shape}")
    print(f"train_y_aligned shape   : {y_df.shape}")

    if len(train_df) != len(y_df):
        raise RuntimeError(
            f"train feature와 y 행 수가 다릅니다: {len(train_df)} vs {len(y_df)}"
        )

    missing_targets = [col for col in TARGET_COLS if col not in y_df.columns]
    if missing_targets:
        raise RuntimeError(f"타깃 컬럼 없음: {missing_targets}")

    if "dat_num" not in train_df.columns:
        raise RuntimeError("train_env_features에 dat_num 컬럼이 없습니다.")

    return train_df, test_df, y_df


def prepare_features(train_df, test_df, y_df):
    """
    sklearn 모델은 숫자형 컬럼만 사용한다.
    dat_num은 DAT 번호를 직접 외우는 위험이 있어 feature에서 제외한다.
    """
    print("\n[학습 피처 준비]")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()

    drop_cols = ["dat_num"]
    feature_cols = [col for col in numeric_cols if col not in drop_cols]

    missing_in_test = [col for col in feature_cols if col not in test_df.columns]
    if missing_in_test:
        raise RuntimeError(f"test에 없는 feature 컬럼: {missing_in_test[:20]}")

    x = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()
    y = y_df[TARGET_COLS].copy()

    medians = x.median(numeric_only=True)

    x = x.replace([np.inf, -np.inf], np.nan)
    x_test = x_test.replace([np.inf, -np.inf], np.nan)

    x = x.fillna(medians)
    x_test = x_test.fillna(medians)

    x = x.fillna(0)
    x_test = x_test.fillna(0)

    print(f"X_train shape : {x.shape}")
    print(f"X_test shape  : {x_test.shape}")
    print(f"y shape       : {y.shape}")
    print(f"feature 개수  : {len(feature_cols)}")

    return x, x_test, y, feature_cols


def make_time_split(train_df):
    """
    랜덤 분할 대신 DAT 기준 시간 검증을 사용한다.
    마지막 약 20% DAT를 validation으로 둔다.
    """
    print("\n[DAT 기준 validation split]")

    dats = sorted(train_df["dat_num"].unique().tolist())
    n_valid = max(1, int(round(len(dats) * 0.2)))

    valid_dats = dats[-n_valid:]
    train_dats = dats[:-n_valid]

    train_mask = train_df["dat_num"].isin(train_dats).to_numpy()
    valid_mask = train_df["dat_num"].isin(valid_dats).to_numpy()

    print(f"전체 DAT: {dats}")
    print(f"학습 DAT: {train_dats}")
    print(f"검증 DAT: {valid_dats}")
    print(f"학습 행 수: {train_mask.sum()}")
    print(f"검증 행 수: {valid_mask.sum()}")

    return train_mask, valid_mask


def make_model():
    """
    ENV baseline 모델 (soil_moisture, soil_ec, soil_temp 다중 타깃).
    inference.py가 soil_ec 예측에 이 모델을 사용한다.
    """
    return ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )


def make_soil_temp_model():
    """
    [v3 추가] soil_temp 전용 단일 타깃 모델.
    experiment_model_candidates.py 12일 rolling 검증에서
    env_raw보다 좋았던 extra_trees 후보와 동일한 설정을 사용한다.
    """
    return ExtraTreesRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )


def calc_metrics(y_true, y_pred):
    rows = []

    y_true_np = y_true.to_numpy()

    for i, target in enumerate(TARGET_COLS):
        true = y_true_np[:, i]
        pred = y_pred[:, i]

        error = true - pred
        mse = float(np.mean(error ** 2))
        rmse = float(np.sqrt(mse))

        sse = float(np.sum(error ** 2))
        sst = float(np.sum((true - true.mean()) ** 2))
        r2 = float(1 - sse / max(sst, 1e-12))

        rows.append(
            {
                "target": target,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
            }
        )

    metric_df = pd.DataFrame(rows)

    mean_row = {
        "target": "mean",
        "mse": metric_df["mse"].mean(),
        "rmse": metric_df["rmse"].mean(),
        "r2": metric_df["r2"].mean(),
    }

    metric_df = pd.concat(
        [metric_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )

    return metric_df


def train_validation_model(x, y, train_mask, valid_mask):
    print("\n[검증용 ENV 모델 학습]")

    x_train = x.iloc[train_mask]
    y_train = y.iloc[train_mask]

    x_valid = x.iloc[valid_mask]
    y_valid = y.iloc[valid_mask]

    model = make_model()
    model.fit(x_train, y_train)

    pred = model.predict(x_valid)
    metric_df = calc_metrics(y_valid, pred)

    print("\n[Validation 성능 - 다중 타깃 ENV 모델]")
    print(metric_df)

    metric_df.to_csv(METRIC_PATH, index=False, encoding="utf-8-sig")
    print(f"\n성능 저장: {METRIC_PATH}")

    return metric_df


def validate_soil_temp_model(x, y, train_mask, valid_mask):
    """
    [v3 추가] soil_temp 단일 타깃 모델의 검증 성능을 확인한다.
    다중 타깃 모델의 soil_temp 성능과 비교하기 위한 로그 목적이다.
    """
    print("\n[검증용 soil_temp ExtraTrees 단일 타깃 모델 학습]")

    x_train = x.iloc[train_mask]
    y_train = y["soil_temp"].iloc[train_mask]

    x_valid = x.iloc[valid_mask]
    y_valid = y["soil_temp"].iloc[valid_mask].to_numpy()

    model = make_soil_temp_model()
    model.fit(x_train, y_train)

    pred = model.predict(x_valid)

    error = y_valid - pred
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    sse = float(np.sum(error ** 2))
    sst = float(np.sum((y_valid - y_valid.mean()) ** 2))
    r2 = float(1 - sse / max(sst, 1e-12))

    print(f"soil_temp 단일 타깃 검증: mse={mse:.6f} / rmse={rmse:.6f} / r2={r2:.6f}")


def train_final_model(x, y):
    print("\n[최종 ENV 모델 학습]")
    print("전체 train 데이터로 inference.py가 사용할 env_model.pkl을 생성합니다.")

    model = make_model()
    model.fit(x, y)

    return model


def train_final_soil_temp_model(x, y):
    """
    [v3 추가] 전체 train 데이터로 soil_temp 전용 최종 모델을 학습한다.
    inference.py가 soil_temp 예측에 이 모델을 사용한다.
    """
    print("\n[최종 soil_temp ExtraTrees 모델 학습]")
    print("전체 train 데이터로 soil_temp_extra_trees.pkl을 생성합니다.")

    model = make_soil_temp_model()
    model.fit(x, y["soil_temp"])

    return model


def save_model(model, soil_temp_model, feature_cols):
    print("\n[모델 저장]")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(soil_temp_model, SOIL_TEMP_MODEL_PATH)  # [v3 추가]

    with open(FEATURE_LIST_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    print(f"모델 저장: {MODEL_PATH}")
    print(f"soil_temp 모델 저장: {SOIL_TEMP_MODEL_PATH}")
    print(f"피처 목록 저장: {FEATURE_LIST_PATH}")


def main():
    print("=" * 80)
    print("train.py 시작 - 최종 제출 재현용 ENV 모델 학습 (v3: soil_temp 단일 타깃 추가)")
    print("=" * 80)

    run_features_env()
    check_required_files()

    train_df, test_df, y_df = load_data()
    x, _x_test, y, feature_cols = prepare_features(train_df, test_df, y_df)

    train_mask, valid_mask = make_time_split(train_df)

    train_validation_model(
        x=x,
        y=y,
        train_mask=train_mask,
        valid_mask=valid_mask,
    )

    # [v3 추가] soil_temp 단일 타깃 모델 검증 성능 확인
    validate_soil_temp_model(
        x=x,
        y=y,
        train_mask=train_mask,
        valid_mask=valid_mask,
    )

    final_model = train_final_model(x, y)
    final_soil_temp_model = train_final_soil_temp_model(x, y)  # [v3 추가]

    save_model(final_model, final_soil_temp_model, feature_cols)

    print("\n[최종 산출물]")
    print(f"- {MODEL_PATH}")
    print(f"- {SOIL_TEMP_MODEL_PATH}")
    print(f"- {FEATURE_LIST_PATH}")
    print(f"- {METRIC_PATH}")
    print("\n이제 inference.py를 실행하면 env_model.pkl + soil_temp_extra_trees.pkl로 submission.csv를 생성합니다.")

    print("\n" + "=" * 80)
    print("train.py 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()