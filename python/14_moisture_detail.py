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
moisture = train_y["soil_moisture"]

day_min, day_max = int(day.min()), int(day.max()) + 1

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(day, moisture, color="#4C72B0", linewidth=0.9)

# 하루 단위로 눈금 + 세로선 전부 표시
ax.set_xticks(range(day_min, day_max + 1))
ax.set_xticklabels(range(day_min, day_max + 1), rotation=90, fontsize=8)
for d in range(day_min, day_max + 1):
    ax.axvline(d, color="gray", linewidth=0.5, alpha=0.3)

ax.set_xlim(day_min, day_max)
ax.set_xlabel("경과일 (day_num)")
ax.set_ylabel("soil_moisture (%)")
ax.set_title("soil_moisture 26일 상세 시계열 (1일 단위 눈금)")
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig("eda_outputs/14_moisture_detail.png", dpi=150)
print("저장 완료: eda_outputs/14_moisture_detail.png")
