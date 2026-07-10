import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 화면 없이 파일로 저장
import matplotlib.pyplot as plt


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


# =========================================
# 1. 전체 인덱스 테이블화
# =========================================
train_ms = parse_ms_folder("dataset/train/ms")
test_ms = parse_ms_folder("dataset/test/ms")

print("train_ms shape:", train_ms.shape)
print("test_ms shape:", test_ms.shape)


# =========================================
# 2. 위치(location)별 하루당 세션 수 전수조사 (0번 위치가 정말 일부 날짜에만 있는지 확인)
# =========================================
print("\n=== train: DAT별 위치별 세션 수 ===")
print(train_ms.groupby(["dat", "location"]).size().unstack(fill_value=0))

print("\n=== test: DAT별 위치별 세션 수 ===")
print(test_ms.groupby(["dat", "location"]).size().unstack(fill_value=0))

# hdr/raw 파일 누락 체크
missing_train = train_ms[~train_ms["hdr_exists"] | ~train_ms["raw_exists"]]
missing_test = test_ms[~test_ms["hdr_exists"] | ~test_ms["raw_exists"]]
print("\ntrain 누락 파일 개수(hdr,row):", len(missing_train))
print("test 누락 파일 개수(hdr,row):", len(missing_test))


# =========================================
# 3. 전체 hdr 스펙이 정말 모든 폴더에서 동일한지 전수 검증
#    (지금까지는 샘플 몇 개만 手동으로 열어봤음 — 예외가 있는지 전부 확인)
# =========================================
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

    # 결과가 "1개" — 즉 639개 폴더 전부, 265개(다분광 이미지 개수) 폴더 전부가 예외 없이
    # (1280, 1024, 10, 12, bsq)라는 완전히 동일한 스펙이라는 뜻
    # 촬영 규격 동일, (가로 픽셀,세로 픽셀,밴드 파장 개수,data type, interleave) 순서로 튜플화해서 비교
    print(f"\n{name} 고유 스펙 조합 개수:", key_tuple.nunique())
    print(key_tuple.value_counts())

# env dt와 매칭 준비: ms의 dt를 env dt 인덱스와 같은 timedelta 단위로 맞춰둠 (위 dt 컬럼)
# -> 다음 단계에서 train_ms/test_ms의 dt를 env의 dt에 merge_asof 등으로 매칭 가능


# =========================================
# 4. raw 파일 크기 전수 검증 (hdr 스펙대로 정확한 바이트 수인지)
# =========================================
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
#          = 1280 × 1024 × 10 × 2   (dtype 12=uint16이라 픽셀 1개당 2바이트)
#          = 26,214,400 바이트


BAND_COUNT = 10
BAND_COLS = [f"band{i+1}_mean" for i in range(BAND_COUNT)]


def read_band_means(raw_path):
    """이미지 1장에서 밴드별 평균 반사값(10개 숫자) 뽑기"""
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280)  # BSQ: band, line, sample 순서
    return cube.mean(axis=(1, 2))  # 밴드별로 사진 전체 평균


def extract_features(df, cache_path):
    """모든 이미지에 대해 위 계산을 돌리고, 한 번 계산한 건 파일로 저장(캐시)해서 재실행 시 재계산 방지"""
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


# =========================================
# 5. 이미지 → 수치 피처 변환 (밴드별 평균)
# =========================================
train_ms_feat = extract_features(train_ms, "cache/train_ms_features.pkl")
test_ms_feat = extract_features(test_ms, "cache/test_ms_features.pkl")
print("이미지 피처 추출 완료:", train_ms_feat.shape, test_ms_feat.shape)


def build_grid(start_day, n_days, freq="5min"):
    """train_y와 동일한 5분 격자(기준 틀) 만들기"""
    start = pd.to_timedelta(start_day, unit="D")
    periods = n_days * 24 * 60 // 5
    return pd.timedelta_range(start=start, periods=periods, freq=freq)


train_grid = pd.DataFrame({"dt": build_grid(109, 26)})
test_grid = pd.DataFrame({"dt": build_grid(135, 12)})
print("\ntrain_grid:", train_grid.shape, "(기대: 7488행)")
print("test_grid:", test_grid.shape, "(기대: 3456행)")


# =========================================
# 6. 위치별로 "직전 촬영값을 다음 촬영 전까지 유지"(forward-fill) 매칭
#    3시간 넘게 사진이 없으면 억지로 채우지 않고 결측으로 둠
# =========================================
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
    merged_loc = pd.merge_asof(
        grid_df,
        loc_df[["dt"] + BAND_COLS],
        on="dt",
        direction="backward",   # 과거 방향으로 가장 가까운 값 = forward-fill
        tolerance=FILL_LIMIT,
    )
    return merged_loc.rename(columns={c: f"loc{location}_{c}" for c in BAND_COLS})


for location in [0, 1, 2, 3]:
    train_grid = attach_location_features(train_grid, train_ms_feat, location)
    test_grid = attach_location_features(test_grid, test_ms_feat, location)


# 위치 간 평균/표준편차 요약 컬럼 추가 (개별 위치 컬럼은 그대로 유지)
for band_col in BAND_COLS:
    loc_cols = [f"loc{loc}_{band_col}" for loc in [0, 1, 2, 3]]
    train_grid[f"avg_{band_col}"] = train_grid[loc_cols].mean(axis=1)
    train_grid[f"std_{band_col}"] = train_grid[loc_cols].std(axis=1)
    test_grid[f"avg_{band_col}"] = test_grid[loc_cols].mean(axis=1)
    test_grid[f"std_{band_col}"] = test_grid[loc_cols].std(axis=1)


# 위치별로 실제 얼마나 채워졌는지(결측 비율) 확인
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

# 에러 없이 완료됐습니다. 위치0 결측 ~91%, 위치1/2/3 결측 70~74% — 둘 다 "촬영 안 한 시간대가 많아서"
# 생기는 정상적인 결과입니다.


# =========================================
# 7. 이미지 피처 vs 정답(y) 상관관계
# =========================================
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


# =========================================
# 8. 실제 사진 눈으로 확인 (같은 날, 위치별로 1장씩 저장)
# =========================================
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

# 위치1/2/3: 딸기로 추정되는 톱니모양 잎사귀 근접 촬영, 캘리브레이션 기준판 동반
# 위치0: 더 넓고 흐릿한 범위 (고정 카메라로 추정), 촬영 스케줄도 하루종일 30분 간격으로 다름


# =========================================
# 9. 전체 이미지 화질 점검 (노출 상태, 촬영 시각별 밝기 편차) — train/test 둘 다 확인
#    (이미지 상관관계 분석 전에 화질부터 점검해야 함)
# =========================================
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

for name, quality in [("train", train_quality), ("test", test_quality)]:
    print(f"\n=== {name} 이미지 밝기/노출 분포 요약 ===")
    print(quality[["img_mean", "img_std", "img_min", "img_max"]].describe())

    low_cut = quality["img_mean"].quantile(0.01)
    high_cut = quality["img_mean"].quantile(0.99)
    print(f"\n{name} 하위 1% 밝기 기준: {low_cut:.1f}, 상위 1% 밝기 기준: {high_cut:.1f}")
    print(quality[quality["img_mean"] <= low_cut][["dat", "location", "dt", "img_mean"]])
    print(quality[quality["img_mean"] >= high_cut][["dat", "location", "dt", "img_mean"]])

    quality["hour"] = (quality["dt"] % pd.Timedelta("1D")) / pd.Timedelta("1h")
    print(f"\n=== {name} 시간대별 평균 밝기 ===")
    print(quality.groupby(quality["hour"].round())["img_mean"].mean())

# test 이미지가 train보다 평균 28% 더 밝음 (train 128.75 vs test 164.98) — 01_eda.py에서 확인한
# "test 구간 solar_radiation 평균이 train보다 높다"는 사실과 교차검증됨 (겨울이라 기온은 낮지만
# 맑은 날이 많아 일사량 자체는 더 강함). 이미지 피처를 모델에 쓸 경우 train/test 밝기 스케일 차이 주의.


# =========================================
# 10. 위치0의 상관관계가 진짜 이미지 신호인지, 그냥 날짜(계절) 효과인지 확인
# =========================================
merged["day_num"] = merged.index // pd.Timedelta("1D")
print("\n=== 그냥 '경과일수'와 정답의 상관관계 ===")
print(merged[["day_num"] + target_cols].corr().loc["day_num"])

# 위치0이 있는 날짜만 따로 비교
loc0_days = train_ms[train_ms["location"] == 0]["dat"].unique()
mask = merged["day_num"].isin(loc0_days)
print(f"\n위치0이 존재하는 날짜(dat): {sorted(loc0_days)}")
print("\n=== 위치0 존재 구간만 필터링한 '경과일수' 상관관계 ===")
print(merged.loc[mask, ["day_num"] + target_cols].corr().loc["day_num"])


# =========================================
# 11. 파장(밴드)별 패턴 검증 — "적색 경계(713,736nm) vs NIR 평탄부(759~920nm)" 그룹 가설을
#     실제 데이터로 확인 (README.md의 도메인 지식 설명이 이 데이터에서도 맞는지 검증)
# =========================================
def get_wavelengths(hdr_path):
    spec = read_hdr_spec(hdr_path)
    wl_str = spec["wavelength"].strip("{}")
    return [float(w.strip()) for w in wl_str.split(",")]


wavelengths = get_wavelengths(train_ms["hdr_path"].iloc[0])
print("\n밴드별 파장(nm):", wavelengths)

band_idx = range(1, BAND_COUNT + 1)

# ① 파장별 상관계수 그래프 — 위치 평균(avg_band) 기준, 타깃 3개 각각
plt.figure(figsize=(10, 6))
for target in target_cols:
    rows = [f"avg_band{i}_mean" for i in band_idx]
    plt.plot(wavelengths, corr_table.loc[rows, target].values, marker="o", label=target)
plt.axvline(759, color="gray", linestyle="--", alpha=0.6, label="red edge 변곡점 추정(~759nm)")
plt.xlabel("파장 (nm)")
plt.ylabel("상관계수 (avg_band vs 정답)")
plt.title("파장별 상관계수 — red edge vs NIR 평탄부 그룹 가설 검증")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eda_outputs/correlation_vs_wavelength.png", dpi=120)
plt.close()
print("저장: eda_outputs/correlation_vs_wavelength.png")

# ② 실제 평균 분광 반사 곡선 — train 전체 이미지 평균 (밴드값 자체가 파장에 따라 어떻게 변하는지)
mean_reflectance = train_ms_feat[BAND_COLS].mean()
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, mean_reflectance.values, marker="o", color="darkgreen")
plt.axvline(759, color="gray", linestyle="--", alpha=0.6, label="red edge 변곡점 추정(~759nm)")
plt.xlabel("파장 (nm)")
plt.ylabel("평균 반사값 (raw DN)")
plt.title("평균 분광 반사 곡선 (train 전체 이미지 평균)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("eda_outputs/reflectance_spectrum.png", dpi=120)
plt.close()
print("저장: eda_outputs/reflectance_spectrum.png")

# ③ 숫자로도 그룹 가설 확인 — 713,736 그룹 vs 759~920 그룹 상관계수 요약
redge_rows = [f"avg_band{i}_mean" for i in [1, 2]]       # 713, 736nm
nir_rows = [f"avg_band{i}_mean" for i in range(3, 11)]   # 759~920nm
print("\n=== 그룹별 상관계수 요약 ===")
print("적색 경계(713,736nm) 그룹:")
print(corr_table.loc[redge_rows, target_cols])
print("\nNIR 평탄부(759~920nm) 그룹 (평균):")
print(corr_table.loc[nir_rows, target_cols].mean())


# =========================================
# EDA 요약
# =========================================
# - 위치 0/1/2/3 존재, 촬영 스펙(1280x1024x10밴드, uint16, bsq) 전 폴더 100% 동일, raw 파일 크기도 전부 정상
# - img_max 최댓값이 1023 = 2^10-1 → 이 카메라 센서가 10비트(0~1023) 장비로 추정됨.
#   1023에 딱 걸리는 픽셀은 포화(클리핑)됐을 가능성 있음 (실제 비율은 아직 안 셈, PLAN.md 참고)
# - day_num(경과일수)만으로도 soil_moisture와 상관관계 0.717 → 매우 강한 단일 신호
# - 위치0 존재 구간(DAT121,123~128)만 떼어보면 day_num-soil_moisture 상관관계는 -0.0096으로 거의 0
#   → 위치0 이미지의 상관관계(0.30~0.37)가 날짜 우연이 아니라 실제 신호라는 근거
# - test 이미지가 train보다 평균 28% 더 밝음 — solar_radiation 센서값 차이와 교차검증됨
#
# 자세한 내용/최신 상태는 README.md(핵심 분석 인사이트), PLAN.md(진행상황) 참고.
