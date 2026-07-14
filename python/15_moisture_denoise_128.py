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

# "128일 사건 = 노이즈"라고 가정하고 처리하는 방법: rolling median 필터.
# 짧게 튀는 값(몇 개 포인트짜리 스파이크)은 중앙값으로 눌러 없애면서, 계단 전환처럼
# 오래 지속되는 진짜 레벨 변화는 그대로 남기는 방식 — 윈도우를 1시간(12칸, 5분격자)으로 설정.
WINDOW = 12
moisture_denoised = moisture.rolling(WINDOW, center=True, min_periods=1).median()

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 왼쪽: 26일 전체 비교
axes[0].plot(day, moisture, color="#4C72B0", linewidth=0.6, alpha=0.4, label="원본")
axes[0].plot(day, moisture_denoised, color="#DD8452", linewidth=1.2, label=f"노이즈 제거 가정(rolling median, {WINDOW*5}분 창)")
axes[0].set_xlabel("경과일 (day_num)")
axes[0].set_ylabel("soil_moisture (%)")
axes[0].set_title("26일 전체 비교")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

# 오른쪽: 128일 근처만 확대
mask = (day >= 127) & (day <= 130)
axes[1].plot(day[mask], moisture[mask], color="#4C72B0", linewidth=1.0, marker=".", markersize=3, label="원본")
axes[1].plot(day[mask], moisture_denoised[mask], color="#DD8452", linewidth=1.8, label="노이즈 제거 가정")
axes[1].set_xlabel("경과일 (day_num)")
axes[1].set_title("128일 사건 확대 (127~130일)")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.suptitle("soil_moisture 원본 vs \"128일 사건=노이즈\" 가정 시 (rolling median 필터)")
fig.tight_layout()
fig.savefig("eda_outputs/15_moisture_denoise_128.png", dpi=150)
print("저장 완료: eda_outputs/15_moisture_denoise_128.png")
