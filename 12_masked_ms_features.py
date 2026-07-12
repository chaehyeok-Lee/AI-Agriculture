"""다분광 이미지에 간단한 배경 마스킹(하위 밝기 퍼센타일 제외)을 적용해 밴드 평균을 다시 뽑고,
이 피처를 env 베이스라인에 추가했을 때 val RMSE가 실제로 좋아지는지 비교한다.

마스킹 방식: 픽셀별 10개 밴드 평균 밝기를 구해서, 하위 DARK_PCT%를 배경/그림자로 보고 제외한 뒤
나머지 픽셀만으로 밴드별 평균을 다시 계산 (전문 세그멘테이션 없이 가장 단순한 근사).
"""
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"  # 이 mac 기준 한글 깨짐 방지 (다른 OS에서는 폰트명 교체 필요)
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import lightgbm as lgb

from preprocess import build_features, load_target
from train import DROP_COLS_PER_TARGET, TARGET_COLS, RANDOM_STATE

FOLD_CUTOFFS = [118, 122, 126, 130]  # train < cutoff, val = 다음 4일 (expanding window)
COLOR_ENV = "#2a78d6"      # 팔레트 slot1 blue
COLOR_MASKED = "#1baf7a"   # 팔레트 slot2 aqua

BAND_COUNT = 10
BAND_COLS = [f"band{i+1}_mean" for i in range(BAND_COUNT)]
DARK_PCT = 15  # 하위 15% 밝기 픽셀을 배경으로 간주하고 제외


def parse_ms_folder(base_dir):
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
                continue
            loc = int(m.group(1))
            hh, mm, ss = m.group(3)[:2], m.group(3)[2:4], m.group(3)[4:6]
            dt = pd.to_timedelta(dat_num, unit="D") + pd.to_timedelta(f"{hh}:{mm}:{ss}")
            rows.append({
                "dat": dat_num, "location": loc, "dt": dt,
                "raw_path": str(session_dir / "cube.raw"),
            })
    return pd.DataFrame(rows)


def read_band_means_masked(raw_path, dark_pct=DARK_PCT):
    arr = np.fromfile(raw_path, dtype=np.uint16)
    cube = arr.reshape(BAND_COUNT, 1024, 1280).astype(np.float32)
    brightness = cube.mean(axis=0)
    thresh = np.percentile(brightness, dark_pct)
    mask = brightness >= thresh
    return cube[:, mask].mean(axis=1)


def extract_masked_features(df, cache_path):
    if Path(cache_path).exists():
        return pd.read_pickle(cache_path)
    n = len(df)
    means = []
    start = time.time()
    for i, p in enumerate(df["raw_path"]):
        means.append(read_band_means_masked(p))
        if (i + 1) % 100 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} ({time.time()-start:.1f}초)")
    feat = pd.DataFrame(np.stack(means), columns=BAND_COLS)
    feat = pd.concat([df.reset_index(drop=True), feat], axis=1)
    feat.to_pickle(cache_path)
    return feat


def build_grid(start_day, n_days, freq="5min"):
    start = pd.to_timedelta(start_day, unit="D")
    periods = n_days * 24 * 60 // 5
    return pd.timedelta_range(start=start, periods=periods, freq=freq)


def attach_location_features(grid_df, feat_df, location, fill_limit=pd.Timedelta("3h")):
    grid_df = grid_df.copy()
    grid_df["dt"] = grid_df["dt"].astype("timedelta64[ns]")
    loc_df = feat_df[feat_df["location"] == location].sort_values("dt").copy()
    if loc_df.empty:
        for col in BAND_COLS:
            grid_df[f"loc{location}_{col}"] = np.nan
        return grid_df
    loc_df["dt"] = loc_df["dt"].astype("timedelta64[ns]")
    merged = pd.merge_asof(
        grid_df, loc_df[["dt"] + BAND_COLS], on="dt",
        direction="backward", tolerance=fill_limit,
    )
    return merged.rename(columns={c: f"loc{location}_{c}" for c in BAND_COLS})


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_folds(feat_df, train_y):
    """4-fold 시계열 교차검증(expanding window). 단일 폴드는 노이즈에 취약해서
    day_num 실험 때처럼 성급한 결론을 막기 위해 기본으로 이 방식을 씀."""
    results = {c: [] for c in TARGET_COLS}
    for cutoff in FOLD_CUTOFFS:
        train_mask = feat_df.index < pd.Timedelta(days=cutoff)
        val_mask = (feat_df.index >= pd.Timedelta(days=cutoff)) & (feat_df.index < pd.Timedelta(days=cutoff + 4))
        tr_feat, tr_y = feat_df[train_mask], train_y[train_mask]
        val_feat, val_y = feat_df[val_mask], train_y[val_mask]
        for col in TARGET_COLS:
            cols = [c for c in tr_feat.columns if c not in DROP_COLS_PER_TARGET[col]]
            m = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, random_state=RANDOM_STATE, verbosity=-1)
            m.fit(tr_feat[cols], tr_y[col], eval_set=[(val_feat[cols], val_y[col])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = m.predict(val_feat[cols])
            results[col].append(rmse(val_y[col].to_numpy(), pred))
    return results


def plot_comparison(results_env, results_masked, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(TARGET_COLS))
    width = 0.32

    means_env = [np.mean(results_env[c]) for c in TARGET_COLS]
    std_env = [np.std(results_env[c]) for c in TARGET_COLS]
    means_masked = [np.mean(results_masked[c]) for c in TARGET_COLS]
    std_masked = [np.std(results_masked[c]) for c in TARGET_COLS]

    ax.bar(x - width / 2, means_env, width, yerr=std_env, capsize=4,
           color=COLOR_ENV, label="env-only", zorder=3)
    ax.bar(x + width / 2, means_masked, width, yerr=std_masked, capsize=4,
           color=COLOR_MASKED, label="env+ms(마스킹)", zorder=3)

    for xi, m in zip(x - width / 2, means_env):
        ax.text(xi, m + 0.03, f"{m:.3f}", ha="center", fontsize=9, color="#52514e")
    for xi, m in zip(x + width / 2, means_masked):
        ax.text(xi, m + 0.03, f"{m:.3f}", ha="center", fontsize=9, color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_COLS)
    ax.set_ylabel("val RMSE (4-fold 평균 ± 표준편차)")
    ax.set_title("마스킹 이미지 피처 추가 효과 — 4-fold 시계열 교차검증")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n저장: {out_path}")


def main():
    train_ms = parse_ms_folder("dataset/train/ms")
    test_ms = parse_ms_folder("dataset/test/ms")

    print("=== 마스킹 밴드 평균 추출 (train) ===")
    train_ms_feat = extract_masked_features(train_ms, "cache/train_ms_features_masked.pkl")
    print("=== 마스킹 밴드 평균 추출 (test) ===")
    test_ms_feat = extract_masked_features(test_ms, "cache/test_ms_features_masked.pkl")

    train_grid = pd.DataFrame({"dt": build_grid(109, 26)})
    test_grid = pd.DataFrame({"dt": build_grid(135, 12)})
    for loc in [0, 1, 2, 3]:
        train_grid = attach_location_features(train_grid, train_ms_feat, loc)
        test_grid = attach_location_features(test_grid, test_ms_feat, loc)
    train_grid.to_pickle("cache/train_ms_matched_masked.pkl")
    test_grid.to_pickle("cache/test_ms_matched_masked.pkl")
    print("저장: cache/train_ms_matched_masked.pkl, cache/test_ms_matched_masked.pkl")

    # ---- val RMSE 비교: env-only vs env+마스킹이미지 (4-fold) ----
    train_y = load_target("dataset/train/env/train_y.csv")
    feat_env = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    feat_masked = build_features(
        pd.read_csv("dataset/train/env/train_X.csv"),
        ms_matched_path="cache/train_ms_matched_masked.pkl",
    )

    print("\n=== env-only, 4-fold ===")
    results_env = run_folds(feat_env, train_y)
    for col in TARGET_COLS:
        arr = np.array(results_env[col])
        print(f"  {col}: folds={np.round(arr, 3)} mean={arr.mean():.4f} std={arr.std():.4f}")

    print("\n=== env+ms(마스킹), 4-fold ===")
    results_masked = run_folds(feat_masked, train_y)
    for col in TARGET_COLS:
        arr = np.array(results_masked[col])
        print(f"  {col}: folds={np.round(arr, 3)} mean={arr.mean():.4f} std={arr.std():.4f}")

    print("\n=== 비교 (4-fold 평균) ===")
    for col in TARGET_COLS:
        m1, m2 = np.mean(results_env[col]), np.mean(results_masked[col])
        verdict = "개선" if m2 < m1 * 0.98 else ("악화" if m2 > m1 * 1.02 else "무의미(노이즈 범위)")
        print(f"{col}: env-only={m1:.4f}  env+ms(마스킹)={m2:.4f}  -> {verdict}")

    plot_comparison(results_env, results_masked, "eda_outputs/masked_ms_comparison.png")


if __name__ == "__main__":
    main()
