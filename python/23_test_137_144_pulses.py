import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from preprocess import build_features  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))
day = test_feat.index.total_seconds() / 86400
mask = (day >= 137) & (day <= 144)
sub = test_feat[mask]
sub_day = day[mask]

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

axes[0].plot(sub_day, sub["humidity_mean"], color="#DD8452", linewidth=1.1)
axes[0].axhline(78.4, color="gray", linestyle=":", label="train 기준선(78.4)")
axes[0].set_ylabel("내부습도")
axes[0].set_title("test 137~144일 원본(5분 단위) — 2파동(작은 급액 139일 + 큰 급액 142일) 가설 확인")
axes[0].grid(alpha=0.3)

# 두 파동 표시
axes[0].annotate("① 작은 급액 추정\n(139일 소폭 상승)", xy=(139.3, 88), xytext=(137.6, 92),
                  fontsize=9, color="#C44E52", arrowprops=dict(arrowstyle="->", color="#C44E52"))
axes[0].annotate("② 큰 급액 추정\n(142일 최고점)", xy=(142.3, 91), xytext=(142.6, 94),
                  fontsize=9, color="#C44E52", arrowprops=dict(arrowstyle="->", color="#C44E52"))
axes[0].annotate("사이 회복(희석)\n140일 저점", xy=(140.3, 50), xytext=(140.6, 40),
                  fontsize=9, color="#4C72B0", arrowprops=dict(arrowstyle="->", color="#4C72B0"))
axes[0].legend(loc="lower right")

axes[1].plot(sub_day, sub["temperature_mean"], color="#C44E52", linewidth=1.1)
axes[1].set_ylabel("내부온도")
axes[1].grid(alpha=0.3)

axes[2].plot(sub_day, sub["circ_fan_mean"], color="#4C72B0", linewidth=1.1)
axes[2].set_ylabel("circ_fan")
axes[2].set_xlabel("경과일 (day_num)")
axes[2].grid(alpha=0.3)

for d in range(137, 145):
    for ax in axes:
        ax.axvline(d, color="black", linestyle="--", alpha=0.2)

fig.tight_layout()
fig.savefig("eda_outputs/23_test_137_144_pulses.png", dpi=150)
print("저장 완료: eda_outputs/23_test_137_144_pulses.png")
