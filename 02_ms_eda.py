import re
import pandas as pd
from pathlib import Path


def parse_ms_folder(base_dir):
    """base_dir 아래의 DATxxx/{위치}_{DAT}_{HHMMSS} 폴더들을 표로 정리"""
    rows = []
    base = Path(base_dir)
    for dat_dir in sorted(base.iterdir()):
        if not dat_dir.is_dir():
            continue
        dat_num = int(re.match(r"DAT(\d+)", dat_dir.name).group(1))
        for session_dir in sorted(dat_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            m = re.match(r"(\d+)_DAT(\d+)_(\d{6})", session_dir.name)
            if not m:
                print("이름 형식 이상:", session_dir)
                continue
            loc = int(m.group(1))
            hhmmss = m.group(3)
            hh, mm, ss = hhmmss[:2], hhmmss[2:4], hhmmss[4:6]
            dt = pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(f"{hh}:{mm}:{ss}")
            hdr_path = session_dir / "cube.hdr"
            raw_path = session_dir / "cube.raw"
            rows.append({
                "dat": dat_num,
                "location": loc,
                "dt": dt,
                "hdr_path": str(hdr_path),
                "raw_path": str(raw_path),
                "hdr_exists": hdr_path.exists(),
                "raw_exists": raw_path.exists(),
            })
    return pd.DataFrame(rows)


# 1. 전체 인덱스 테이블화
train_ms = parse_ms_folder("dataset/train/ms")
test_ms = parse_ms_folder("dataset/test/ms")

print("train_ms shape:", train_ms.shape)
print("test_ms shape:", test_ms.shape)

# 2. 위치(location)별 하루당 세션 수 전수조사 (0번 위치가 정말 일부 날짜에만 있는지 확인)
print("\n=== train: DAT별 위치별 세션 수 ===")
print(train_ms.groupby(["dat", "location"]).size().unstack(fill_value=0))

print("\n=== test: DAT별 위치별 세션 수 ===")
print(test_ms.groupby(["dat", "location"]).size().unstack(fill_value=0))

# hdr/raw 파일 누락 체크
missing_train = train_ms[~train_ms["hdr_exists"] | ~train_ms["raw_exists"]]
missing_test = test_ms[~test_ms["hdr_exists"] | ~test_ms["raw_exists"]]
print("\ntrain 누락 파일 개수(hdr,row):", len(missing_train))
print("test 누락 파일 개수(hdr,row):", len(missing_test))


# 3. 전체 hdr 스펙이 정말 모든 폴더에서 동일한지 전수 검증
#    (지금까지는 샘플 몇 개만 手동으로 열어봤음 — 예외가 있는지 전부 확인)
def read_hdr_spec(path):
    spec = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                spec[k.strip()] = v.strip()
    return spec


for name, df in [("train", train_ms), ("test", test_ms)]:
    specs = df["hdr_path"].apply(read_hdr_spec)
    key_tuple = specs.apply(
        lambda s: (s.get("samples"), s.get("lines"), s.get("bands"),
                   s.get("data type"), s.get("interleave"))
    )

# 결과가 "1개" — 즉 639개 폴더 전부, 265개(다분광 이미지 개수) 폴더 전부가 예외 없이 (1280, 1024, 10, 12, bsq)라는 완전히 동일한 스펙이라는 뜻
#촬영 규격 동일, (가로 픽셀,세로 픽셀,밴드 파장 개수,ㅇdata type, interleave=reshape) 순서로 튜플화해서 비교
    print(f"\n{name} 고유 스펙 조합 개수:", key_tuple.nunique())
    print(key_tuple.value_counts())

# 4. env dt와 매칭 준비: ms의 dt를 env dt 인덱스와 같은 timedelta 단위로 맞춰둠 (위 dt 컬럼)
#    -> 다음 단계에서 train_ms/test_ms의 dt를 env의 dt에 merge_asof 등으로 매칭 가능

import os

DTYPE_SIZE = {12: 2}  # ENVI data type 12 = uint16 (2바이트) — 이 데이터셋엔 12만 나왔음

def check_raw_size(row):
    spec = read_hdr_spec(row["hdr_path"])
    samples = int(spec["samples"])
    lines = int(spec["lines"])
    bands = int(spec["bands"])
    dtype = int(spec["data type"])
    expected = samples * lines * bands * DTYPE_SIZE[dtype]
    actual = os.path.getsize(row["raw_path"]) if row["raw_exists"] else None
    return pd.Series({"expected_bytes": expected, "actual_bytes": actual})

for name, df in [("train", train_ms), ("test", test_ms)]:
    sizes = df.apply(check_raw_size, axis=1)
    df["expected_bytes"] = sizes["expected_bytes"]
    df["actual_bytes"] = sizes["actual_bytes"]
    mismatch = df[df["expected_bytes"] != df["actual_bytes"]]
    print(f"\n{name}: raw 크기 불일치 개수: {len(mismatch)}")
    if len(mismatch):
        print(mismatch[["dat", "location", "dt", "raw_path", "expected_bytes", "actual_bytes"]])

        # expected = samples × lines × bands × (dtype당 바이트 수)
        #  = 1280 × 1024 × 10 × 2   (dtype 12=uint16이라 픽셀 1개당 2바이트)
        #  = 26,214,400 바이트

import numpy as np
import time

BAND_COUNT = 10
BAND_COLS = [f"band{i+1}_mean" for i in range(BAND_COUNT)]


# 1. 이미지 1장에서 밴드별 평균 반사값(10개 숫자) 뽑기
def read_band_means(raw_path):
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280)  # BSQ: band, line, sample 순서
    return cube.mean(axis=(1, 2))  # 밴드별로 사진 전체 평균


# 2. 모든 이미지에 대해 위 계산을 돌리고, 한 번 계산한 건 파일로 저장(캐시)해서 재실행 시 재계산 방지
def extract_features(df, cache_path):
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    n = len(df)
    means = []
    start = time.time()
    for i, p in enumerate(df["raw_path"]):
        means.append(read_band_means(p))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (n - (i + 1)) / rate if rate > 0 else 0
            print(f"  {i+1}/{n} 처리 완료 ({elapsed:.1f}초 경과, 예상 남은 시간 {remaining:.1f}초)")

    means = np.stack(means)
    feat = pd.DataFrame(means, columns=BAND_COLS)
    feat = pd.concat([df.reset_index(drop=True), feat], axis=1)
    feat.to_pickle(cache_path)
    return feat


train_ms_feat = extract_features(train_ms, "cache/train_ms_features.pkl")
test_ms_feat = extract_features(test_ms, "cache/test_ms_features.pkl")
print("이미지 피처 추출 완료:", train_ms_feat.shape, test_ms_feat.shape)


# 3. train_y와 동일한 5분 격자(기준 틀) 만들기
def build_grid(start_day, n_days, freq="5min"):
    start = pd.to_timedelta(start_day, unit="D")
    periods = n_days * 24 * 60 // 5
    return pd.timedelta_range(start=start, periods=periods, freq=freq)


train_grid = pd.DataFrame({"dt": build_grid(109, 26)})
test_grid = pd.DataFrame({"dt": build_grid(135, 12)})
print("\ntrain_grid:", train_grid.shape, "(기대: 7488행)")
print("test_grid:", test_grid.shape, "(기대: 3456행)")


# 4. 위치별로 "직전 촬영값을 다음 촬영 전까지 유지"(forward-fill) 매칭
#    3시간 넘게 사진이 없으면 억지로 채우지 않고 결측으로 둠
FILL_LIMIT = pd.Timedelta("3h")


def attach_location_features(grid_df, feat_df, location):
    grid_df = grid_df.copy()
    grid_df["dt"] = grid_df["dt"].astype("timedelta64[ns]")

    loc_df = feat_df[feat_df["location"] == location].sort_values("dt").copy()
    if loc_df.empty:
        for col in BAND_COLS:
            grid_df[f"loc{location}_{col}"] = np.nan
        return grid_df

    loc_df["dt"] = loc_df["dt"].astype("timedelta64[ns]")
    merged = pd.merge_asof(
        grid_df,
        loc_df[["dt"] + BAND_COLS],
        on="dt",
        direction="backward",   # 과거 방향으로 가장 가까운 값 = forward-fill
        tolerance=FILL_LIMIT,
    )
    return merged.rename(columns={c: f"loc{location}_{c}" for c in BAND_COLS})




for location in [0, 1, 2, 3]:
    train_grid = attach_location_features(train_grid, train_ms_feat, location)
    test_grid = attach_location_features(test_grid, test_ms_feat, location)


# 5. 위치 간 평균/표준편차 요약 컬럼 추가 (개별 위치 컬럼은 그대로 유지)
for band_col in BAND_COLS:
    loc_cols = [f"loc{loc}_{band_col}" for loc in [0, 1, 2, 3]]
    train_grid[f"avg_{band_col}"] = train_grid[loc_cols].mean(axis=1)
    train_grid[f"std_{band_col}"] = train_grid[loc_cols].std(axis=1)
    test_grid[f"avg_{band_col}"] = test_grid[loc_cols].mean(axis=1)
    test_grid[f"std_{band_col}"] = test_grid[loc_cols].std(axis=1)


# 6. 위치별로 실제 얼마나 채워졌는지(결측 비율) 확인
print("\n=== train: 위치별 결측 비율 ===")
for location in [0, 1, 2, 3]:
    ratio = train_grid[f"loc{location}_{BAND_COLS[0]}"].isna().mean()
    print(f"위치{location}: 결측 {ratio:.1%}")

print("\n=== test: 위치별 결측 비율 ===")
for location in [0, 1, 2, 3]:
    ratio = test_grid[f"loc{location}_{BAND_COLS[0]}"].isna().mean()
    print(f"위치{location}: 결측 {ratio:.1%}")

train_grid.to_pickle("cache/train_ms_matched.pkl")
test_grid.to_pickle("cache/test_ms_matched.pkl")
print("\n저장 완료: train_ms_matched.pkl, test_ms_matched.pkl")

# 에러 없이 완료됐습니다. 위치0 결측 ~91%, 위치1/2/3 결측 7074% — 둘 다 "촬영 안 한 시간대가 많아서" 생기는 정상적인 결과입니다. 
# train_ms_matched.pkl / test_ms_matched.pkl 저장 완료.

import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt


# 7. 이미지 피처 vs 정답(y) 상관관계
train_y = pd.read_csv("dataset/train/env/train_y.csv")
dat_num = train_y["time"].str.extract(r"DAT(\d+)")[0].astype(int)
hm = train_y["time"].str.split(" ").str[1]
train_y["dt"] = (pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(hm + ":00")).astype("timedelta64[ns]")
train_y = train_y.set_index("dt").sort_index().drop(columns="time")

merged = train_grid.set_index("dt").join(train_y, how="inner")
feature_cols = [c for c in train_grid.columns if "band" in c]
target_cols = ["soil_moisture", "soil_ec", "soil_temp"]

corr_table = merged[feature_cols + target_cols].corr().loc[feature_cols, target_cols]
print("=== 이미지 피처 vs 정답 상관계수 (절댓값 최댓값 기준 상위 10개) ===")
print(corr_table.abs().max(axis=1).sort_values(ascending=False).head(10))
print("\n전체 상관계수 표:")
print(corr_table)


# 8. 실제 사진 눈으로 확인 (같은 날, 위치별로 1장씩 저장) 상관관계 확인 및 이미지 시각화 진행
def save_sample_image(raw_path, out_path, band_index=5):
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280)
    band = cube[band_index].astype(float)
    band_norm = (band - band.min()) / (band.max() - band.min())
    plt.figure(figsize=(8, 6))
    plt.imshow(band_norm, cmap="gray")
    plt.title(f"{Path(raw_path).parent.name} - band{band_index+1}")
    plt.colorbar(label="normalized reflectance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print("저장:", out_path)


sample_day = 123  # 위치 0/1/2/3이 전부 있는 날
sample_rows = (
    train_ms[train_ms["dat"] == sample_day]
    .sort_values("location")
    .groupby("location")
    .first()
    .reset_index()
)
for _, row in sample_rows.iterrows():
    save_sample_image(row["raw_path"], f"eda_outputs/sample_dat{sample_day}_loc{row['location']}.png")


#이미지 상관관걔전 화질부터 정검해야함

# 9. 전체 이미지 화질 점검 (노출 상태, 촬영 시각별 밝기 편차)
def image_quality_stats(raw_path):
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280).astype(float)
    return pd.Series({
        "img_mean": cube.mean(),
        "img_std": cube.std(),
        "img_min": cube.min(),
        "img_max": cube.max(),
    })


def check_all_quality(df, cache_path):
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    n = len(df)
    rows = []
    start = time.time()
    for i, p in enumerate(df["raw_path"]):
        rows.append(image_quality_stats(p))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  품질체크 {i+1}/{n} ({time.time()-start:.1f}초 경과)")
    quality = pd.DataFrame(rows)
    quality = pd.concat([df.reset_index(drop=True), quality], axis=1)
    quality.to_pickle(cache_path)
    return quality


train_quality = check_all_quality(train_ms, "cache/train_ms_quality.pkl")
test_quality = check_all_quality(test_ms, "cache/test_ms_quality.pkl")

print("\n=== train 이미지 밝기/노출 분포 요약 ===")
print(train_quality[["img_mean", "img_std", "img_min", "img_max"]].describe())

# 유난히 어둡거나 밝은 이미지(전체 평균 기준 상하위 1%) 찾기
low_cut = train_quality["img_mean"].quantile(0.01)
high_cut = train_quality["img_mean"].quantile(0.99)
print(f"\n하위 1% 밝기 기준: {low_cut:.1f}, 상위 1% 밝기 기준: {high_cut:.1f}")
print(train_quality[train_quality["img_mean"] <= low_cut][["dat", "location", "dt", "img_mean"]])
print(train_quality[train_quality["img_mean"] >= high_cut][["dat", "location", "dt", "img_mean"]])

# 촬영 시각(시간대)에 따라 밝기가 얼마나 다른지 - 조명 영향 확인용
train_quality["hour"] = (train_quality["dt"] % pd.Timedelta("1D")) / pd.Timedelta("1h")
print("\n=== 시간대별 평균 밝기 ===")
print(train_quality.groupby(train_quality["hour"].round())["img_mean"].mean())

# 10. 위치0의 상관관계가 진짜 이미지 신호인지, 그냥 날짜(계절) 효과인지 확인
merged["day_num"] = merged.index // pd.Timedelta("1D")
print("=== 그냥 '경과일수'와 정답의 상관관계 ===")
print(merged[["day_num"] + target_cols].corr().loc["day_num"])

# 위치0이 있는 날짜만 따로 비교
loc0_days = train_ms[train_ms["location"] == 0]["dat"].unique()
mask = merged["day_num"].isin(loc0_days)
print(f"\n위치0이 존재하는 날짜(dat): {sorted(loc0_days)}")
print("\n=== 위치0 존재 구간만 필터링한 '경과일수' 상관관계 ===")
print(merged.loc[mask, ["day_num"] + target_cols].corr().loc["day_num"])

# 9. 전체 이미지 화질 점검 (노출 상태, 촬영 시각별 밝기 편차,오직 점검만)
def image_quality_stats(raw_path):
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280).astype(float)
    return pd.Series({
        "img_mean": cube.mean(),
        "img_std": cube.std(),
        "img_min": cube.min(),
        "img_max": cube.max(),
    })


def check_all_quality(df, cache_path):
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    n = len(df)
    rows = []
    start = time.time()
    for i, p in enumerate(df["raw_path"]):
        rows.append(image_quality_stats(p))
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"  품질체크 {i+1}/{n} ({time.time()-start:.1f}초 경과)")
    quality = pd.DataFrame(rows)
    quality = pd.concat([df.reset_index(drop=True), quality], axis=1)
    quality.to_pickle(cache_path)
    return quality


train_quality = check_all_quality(train_ms, "cache/train_ms_quality.pkl")
test_quality = check_all_quality(test_ms, "cache/test_ms_quality.pkl")

print("\n=== train 이미지 밝기/노출 분포 요약 ===")
print(train_quality[["img_mean", "img_std", "img_min", "img_max"]].describe())

# 유난히 어둡거나 밝은 이미지(전체 평균 기준 상하위 1%) 찾기
low_cut = train_quality["img_mean"].quantile(0.01)
high_cut = train_quality["img_mean"].quantile(0.99)
print(f"\n하위 1% 밝기 기준: {low_cut:.1f}, 상위 1% 밝기 기준: {high_cut:.1f}")
print(train_quality[train_quality["img_mean"] <= low_cut][["dat", "location", "dt", "img_mean"]])
print(train_quality[train_quality["img_mean"] >= high_cut][["dat", "location", "dt", "img_mean"]])

# 촬영 시각(시간대)에 따라 밝기가 얼마나 다른지 - 조명 영향 확인용
train_quality["hour"] = (train_quality["dt"] % pd.Timedelta("1D")) / pd.Timedelta("1h")
print("\n=== 시간대별 평균 밝기 ===")
print(train_quality.groupby(train_quality["hour"].round())["img_mean"].mean())

# , 1023 = 2¹⁰-1, 즉 이 카메라 센서가 10비트(0~1023 범위)로 찍는 장비일 가능성이 높습니다. 
# 그렇다면 값이 1023에 딱 걸리는 픽셀은 "더 밝은데 카메라가 못 찍고 잘라낸(포화/클리핑)" 픽셀일 수 있어요
#  그냥 "며칠째인지"(day_num)만으로도 soil_moisture와 상관관계 0.717이 나왔습니다. 이건 이미지 없이 날짜 숫자 하나만 갖고도 이미 강한 예측력
# 그런데 위치0이 존재하는 그 특정 7일(DAT121,123~128)만 따로 떼서 보면, day_num과 soil_moisture 상관관계는 -0.0096

