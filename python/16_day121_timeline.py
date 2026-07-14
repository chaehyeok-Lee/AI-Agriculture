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

train_y = load_target("dataset/train/env/train_y.csv")
day = train_y.index.total_seconds() / 86400

mask = (day >= 120.5) & (day <= 122.0)
sub_day = day[mask]
moisture = train_y["soil_moisture"][mask]
ec = train_y["soil_ec"][mask]

fig, ax1 = plt.subplots(figsize=(14, 7))

l1, = ax1.plot(sub_day, moisture, color="#4C72B0", linewidth=1.2, label="soil_moisture (%)")
ax1.set_ylabel("soil_moisture (%)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_xlabel("경과일 (day_num)")
ax1.grid(alpha=0.3)
ax1.set_title("day121 사건 타임라인 — 수분·EC 겹쳐보기 (동시 vs 시차 발생 구간)")

ax2 = ax1.twinx()
l2, = ax2.plot(sub_day, ec, color="#DD8452", linewidth=1.2, label="soil_ec")
ax2.set_ylabel("soil_ec", color="#DD8452")
ax2.tick_params(axis="y", labelcolor="#DD8452")

ax1.legend(handles=[l1, l2], loc="lower right")

# 자정 1차 점프 (수분+EC 동시)
ax1.axvline(121.0, color="red", linestyle="--", alpha=0.7)
ax1.annotate("00:00 1차 점프\n(수분↑ + EC↑ 동시)", xy=(121.0, 40.0), xytext=(121.05, 37),
             fontsize=9, color="red",
             arrowprops=dict(arrowstyle="->", color="red", alpha=0.7))

# 13:30 2차 점프 (수분만 상승, EC는 하락)
t2 = 121 + 13.5 / 24
ax1.axvline(t2, color="green", linestyle="--", alpha=0.7)
ax1.annotate("13:30 2차 점프\n(수분↑, EC는 오히려 하락)", xy=(t2, 41.4), xytext=(t2 + 0.03, 42),
             fontsize=9, color="green",
             arrowprops=dict(arrowstyle="->", color="green", alpha=0.7))

fig.tight_layout()
fig.savefig("eda_outputs/16_day121_timeline.png", dpi=150)
print("저장 완료: eda_outputs/16_day121_timeline.png")
