import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from preprocess import load_target  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

y = load_target("dataset/train/env/train_y.csv")
day_num = (y.index // pd.Timedelta("1D")).astype(int)
y = y.assign(day_num=day_num)

REGIMES = [(109, 115, "A(109~114)"), (115, 121, "B(115~120)"),
           (121, 129, "C(121~128)"), (129, 135, "D(129~134)")]
regime_colors = {"A(109~114)": "#4C72B0", "B(115~120)": "#55A868",
                  "C(121~128)": "#C44E52", "D(129~134)": "#8172B2"}


def regime_label(d):
    for start, end, label in REGIMES:
        if start <= d < end:
            return label
    return "기타"


daily = y.groupby("day_num").agg(moisture=("soil_moisture", "mean"),
                                   ec=("soil_ec", "mean")).reset_index()
daily["regime"] = daily["day_num"].apply(regime_label)

fig = plt.figure(figsize=(15, 11))
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.4])

# 1. 일평균 수분
ax1 = fig.add_subplot(gs[0])
ax1.plot(daily["day_num"], daily["moisture"], color="#4C72B0", marker="o")
for d, note in [(121, "121일"), (128, "128일")]:
    ax1.axvline(d, color="red" if d == 121 else "purple", linestyle="--", alpha=0.6)
ax1.set_ylabel("일평균 soil_moisture(%)")
ax1.set_title("일평균 수분·EC + 구간별 관계 종합 (121일/128일 주목)")
ax1.grid(alpha=0.3)

# 2. 일평균 EC
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(daily["day_num"], daily["ec"], color="#DD8452", marker="s")
for d in [121, 128]:
    ax2.axvline(d, color="red" if d == 121 else "purple", linestyle="--", alpha=0.6)
ax2.set_ylabel("일평균 soil_ec")
ax2.set_xlabel("경과일 (day_num)")
ax2.grid(alpha=0.3)

# 3. 수분 vs EC 산점도 (구간별 색, 121/128 강조)
ax3 = fig.add_subplot(gs[2])
for start, end, label in REGIMES:
    sub = daily[(daily["day_num"] >= start) & (daily["day_num"] < end)]
    ax3.scatter(sub["moisture"], sub["ec"], color=regime_colors[label], label=label, s=70, zorder=2)
for _, row in daily.iterrows():
    ax3.annotate(str(int(row["day_num"])), (row["moisture"], row["ec"]), fontsize=7,
                 xytext=(3, 3), textcoords="offset points")
# 121일, 128일 강조 테두리
for d, c in [(121, "red"), (128, "purple")]:
    row = daily[daily["day_num"] == d].iloc[0]
    ax3.scatter([row["moisture"]], [row["ec"]], s=250, facecolors="none",
                edgecolors=c, linewidths=2.5, zorder=3)
ax3.set_xlabel("soil_moisture 일평균(%)")
ax3.set_ylabel("soil_ec 일평균")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("eda_outputs/20_combined_121_128.png", dpi=150)
print("저장 완료: eda_outputs/20_combined_121_128.png")
