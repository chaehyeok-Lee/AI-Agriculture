import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import build_features, load_target

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
train_y = load_target("dataset/train/env/train_y.csv")
test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))

# 1. 타깃 3종 시계열 (train, 26일 전체)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for ax, col, label in zip(axes, ["soil_moisture", "soil_ec", "soil_temp"],
                           ["배지수분 (%)", "배지 EC", "배지온도 (°C)"]):
    ax.plot(train_y.index.total_seconds() / 86400, train_y[col], color=TRAIN_COLOR, linewidth=0.8)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("경과일 (day_num)")
axes[0].set_title("타깃 3종 시계열 (train, DAT109~134)")
fig.tight_layout()
fig.savefig("eda_outputs/01_target_timeseries.png", dpi=150)

# 2. train vs test 분포 비교 (위험 신호 컬럼 3개)
risky_cols = ["greenhouse_roof_vent1_mean", "wind_speed_outside_mean", "temperature_mean"]
risky_labels = ["천창1 개도율 (%)", "외부 풍속", "내부 온도 (°C)"]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col, label in zip(axes, risky_cols, risky_labels):
    ax.hist(train_feat[col].dropna(), bins=30, alpha=0.6, color=TRAIN_COLOR, label="train", density=True)
    ax.hist(test_feat[col].dropna(), bins=30, alpha=0.6, color=TEST_COLOR, label="test", density=True)
    ax.set_title(label)
    ax.legend()
fig.suptitle("Train vs Test 분포 비교 (위험 신호 컬럼)")
fig.tight_layout()
fig.savefig("eda_outputs/02_train_test_distribution.png", dpi=150)

# 3. day_num vs soil_moisture 산점도
fig, ax = plt.subplots(figsize=(8, 5))
day_num = train_y.index // pd.Timedelta("1D")
corr = np.corrcoef(day_num, train_y["soil_moisture"])[0, 1]
ax.scatter(day_num, train_y["soil_moisture"], s=4, alpha=0.3, color=TRAIN_COLOR)
ax.set_xlabel("day_num (경과일)")
ax.set_ylabel("soil_moisture (%)")
ax.set_title(f"day_num vs soil_moisture (상관계수 {corr:.2f})")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("eda_outputs/03_daynum_vs_moisture.png", dpi=150)

# 4. 상관관계 히트맵 (주요 피처 vs 타깃 3종)
merged = train_feat.join(train_y, how="inner")
key_cols = [c for c in merged.columns if c.endswith("_mean")] + ["day_num", "hour"]
target_cols = ["soil_moisture", "soil_ec", "soil_temp"]
corr_mat = merged[key_cols + target_cols].corr().loc[key_cols, target_cols]

fig, ax = plt.subplots(figsize=(4, max(6, len(key_cols) * 0.3)))
im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(target_cols))); ax.set_xticklabels(target_cols, rotation=45, ha="right")
ax.set_yticks(range(len(key_cols))); ax.set_yticklabels(key_cols, fontsize=8)
fig.colorbar(im, ax=ax, label="상관계수")
ax.set_title("피처 vs 타깃 상관관계")
fig.tight_layout()
fig.savefig("eda_outputs/04_correlation_heatmap.png", dpi=150)

print("4개 그래프 저장 완료: eda_outputs/")
plt.show()
