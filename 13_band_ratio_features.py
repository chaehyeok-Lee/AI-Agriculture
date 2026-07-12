"""밴드 절대값 대신 밴드 그룹 간 '비율' 피처를 만들어서 env 베이스라인에 도움되는지 확인.

아이디어: 조명이 세지면 모든 밴드값이 같이 오르고 약해지면 같이 내려간다(기준판 상관 0.958로 이미 확인됨).
그래서 두 밴드 그룹의 절대 평균 대신 "그룹 A ÷ 그룹 B" 비율을 쓰면 조명 세기가 분자·분모에서
어느 정도 상쇄돼서, 기준판 보정 없이도 조명 영향을 줄인 피처를 만들 수 있다는 가설.

밴드 그룹 (README/PLAN.md 도메인 정리 기준, 713~920nm 10개):
- red_edge  (713·736·759nm, band1~3): 엽록소/스트레스 반응 구간, 파장 늘수록 반사율 급상승
- nir_plateau (782~874nm, band4~8): 잎 구조/turgor 반영, 평탄부
- nir_tail  (897·920nm, band9~10): 수분흡수대 인접, soil_temp에 특히 중요하다고 알려짐

피처: redge_ratio = red_edge / nir_plateau, tail_ratio = nir_tail / nir_plateau (위치별로 4개씩, 총 8개)
이미 캐시된 원본 밴드평균(cache/*_ms_features.pkl)에서 바로 계산 — 이미지 재처리 불필요.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from preprocess import build_features, load_target
from train import DROP_COLS_PER_TARGET, TARGET_COLS, RANDOM_STATE

RED_EDGE_BANDS = ["band1_mean", "band2_mean", "band3_mean"]      # 713, 736, 759nm
NIR_PLATEAU_BANDS = ["band4_mean", "band5_mean", "band6_mean", "band7_mean", "band8_mean"]  # 782~874nm
NIR_TAIL_BANDS = ["band9_mean", "band10_mean"]                    # 897, 920nm

FOLD_CUTOFFS = [118, 122, 126, 130]
COLOR_ENV = "#2a78d6"
COLOR_RATIO = "#eda100"  # 팔레트 slot3 yellow (마스킹 실험의 aqua와 구분)


def add_ratio_cols(feat_df):
    feat_df = feat_df.copy()
    red_edge = feat_df[RED_EDGE_BANDS].mean(axis=1)
    nir_plateau = feat_df[NIR_PLATEAU_BANDS].mean(axis=1)
    nir_tail = feat_df[NIR_TAIL_BANDS].mean(axis=1)
    feat_df["redge_ratio"] = red_edge / nir_plateau
    feat_df["tail_ratio"] = nir_tail / nir_plateau
    return feat_df


def build_grid(start_day, n_days, freq="5min"):
    start = pd.to_timedelta(start_day, unit="D")
    periods = n_days * 24 * 60 // 5
    return pd.timedelta_range(start=start, periods=periods, freq=freq)


def attach_ratio_features(grid_df, feat_df, location, fill_limit=pd.Timedelta("3h")):
    grid_df = grid_df.copy()
    grid_df["dt"] = grid_df["dt"].astype("timedelta64[ns]")
    ratio_cols = ["redge_ratio", "tail_ratio"]
    loc_df = feat_df[feat_df["location"] == location].sort_values("dt").copy()
    if loc_df.empty:
        for col in ratio_cols:
            grid_df[f"loc{location}_{col}"] = np.nan
        return grid_df
    loc_df["dt"] = loc_df["dt"].astype("timedelta64[ns]")
    merged = pd.merge_asof(
        grid_df, loc_df[["dt"] + ratio_cols], on="dt",
        direction="backward", tolerance=fill_limit,
    )
    return merged.rename(columns={c: f"loc{location}_{c}" for c in ratio_cols})


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run_folds(feat_df, train_y):
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


def plot_comparison(results_env, results_ratio, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(TARGET_COLS))
    width = 0.32

    means_env = [np.mean(results_env[c]) for c in TARGET_COLS]
    std_env = [np.std(results_env[c]) for c in TARGET_COLS]
    means_ratio = [np.mean(results_ratio[c]) for c in TARGET_COLS]
    std_ratio = [np.std(results_ratio[c]) for c in TARGET_COLS]

    ax.bar(x - width / 2, means_env, width, yerr=std_env, capsize=4, color=COLOR_ENV, label="env-only", zorder=3)
    ax.bar(x + width / 2, means_ratio, width, yerr=std_ratio, capsize=4, color=COLOR_RATIO, label="env+밴드비율", zorder=3)

    for xi, m in zip(x - width / 2, means_env):
        ax.text(xi, m + 0.03, f"{m:.3f}", ha="center", fontsize=9, color="#52514e")
    for xi, m in zip(x + width / 2, means_ratio):
        ax.text(xi, m + 0.03, f"{m:.3f}", ha="center", fontsize=9, color="#52514e")

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_COLS)
    ax.set_ylabel("val RMSE (4-fold 평균 ± 표준편차)")
    ax.set_title("밴드 비율 피처 추가 효과 — 4-fold 시계열 교차검증")
    ax.grid(axis="y", color="#e1e0d9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n저장: {out_path}")


def main():
    train_ms_feat = add_ratio_cols(pd.read_pickle("cache/train_ms_features.pkl"))

    train_grid = pd.DataFrame({"dt": build_grid(109, 26)})
    for loc in [0, 1, 2, 3]:
        train_grid = attach_ratio_features(train_grid, train_ms_feat, loc)
    train_grid.to_pickle("cache/train_ms_matched_ratio.pkl")
    print("저장: cache/train_ms_matched_ratio.pkl")
    print(train_grid[[c for c in train_grid.columns if "ratio" in c]].describe())

    train_y = load_target("dataset/train/env/train_y.csv")
    feat_env = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
    feat_ratio = build_features(
        pd.read_csv("dataset/train/env/train_X.csv"),
        ms_matched_path="cache/train_ms_matched_ratio.pkl",
    )

    print("\n=== env-only, 4-fold ===")
    results_env = run_folds(feat_env, train_y)
    for col in TARGET_COLS:
        arr = np.array(results_env[col])
        print(f"  {col}: folds={np.round(arr, 3)} mean={arr.mean():.4f} std={arr.std():.4f}")

    print("\n=== env+밴드비율, 4-fold ===")
    results_ratio = run_folds(feat_ratio, train_y)
    for col in TARGET_COLS:
        arr = np.array(results_ratio[col])
        print(f"  {col}: folds={np.round(arr, 3)} mean={arr.mean():.4f} std={arr.std():.4f}")

    print("\n=== 비교 (4-fold 평균) ===")
    for col in TARGET_COLS:
        m1, m2 = np.mean(results_env[col]), np.mean(results_ratio[col])
        verdict = "개선" if m2 < m1 * 0.98 else ("악화" if m2 > m1 * 1.02 else "무의미(노이즈 범위)")
        print(f"{col}: env-only={m1:.4f}  env+밴드비율={m2:.4f}  -> {verdict}")

    plot_comparison(results_env, results_ratio, "eda_outputs/band_ratio_comparison.png")


if __name__ == "__main__":
    main()
