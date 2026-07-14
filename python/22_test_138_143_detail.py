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
mask = (day >= 137.5) & (day <= 144)
sub = test_feat[mask]
sub_day = day[mask]

fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

axes[0].plot(sub_day, sub["humidity_mean"], color="#DD8452", linewidth=1.0)
axes[0].axhline(78.4, color="gray", linestyle=":", label="기준선(78.4)")
axes[0].set_ylabel("내부습도")
axes[0].legend()
axes[0].set_title("test 138~143일 원본(5분 단위) — 내부습도/온도/순환팬")
axes[0].grid(alpha=0.3)

axes[1].plot(sub_day, sub["temperature_mean"], color="#C44E52", linewidth=1.0)
axes[1].set_ylabel("내부온도")
axes[1].grid(alpha=0.3)

axes[2].plot(sub_day, sub["circ_fan_mean"], color="#4C72B0", linewidth=1.0)
axes[2].set_ylabel("circ_fan")
axes[2].set_xlabel("경과일 (day_num)")
axes[2].grid(alpha=0.3)

for d in range(138, 144):
    for ax in axes:
        ax.axvline(d, color="black", linestyle="--", alpha=0.2)

fig.tight_layout()
fig.savefig("eda_outputs/22_test_138_143_detail.png", dpi=150)
print("저장 완료: eda_outputs/22_test_138_143_detail.png")

# 시간대별(6시간 단위)로 좀 더 세밀하게 습도 추이 출력
print("\n=== 6시간 단위 내부습도 평균 ===")
sub2 = sub.copy()
sub2["day_num"] = sub_day.astype(int)
sub2["quarter"] = ((sub_day % 1) * 4).astype(int)  # 0~5,6~11,12~17,18~23시
q_table = sub2.groupby(["day_num", "quarter"])["humidity_mean"].mean().unstack()
print(q_table.round(1))
