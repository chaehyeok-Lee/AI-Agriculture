import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
from preprocess import build_features, load_target  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
train_y = load_target("dataset/train/env/train_y.csv")
merged = train_feat.join(train_y, how="inner")
day = merged.index.total_seconds() / 86400

# 23번(test 137~144, 환기 OFF 구간+전환)과 같은 구조 — train의 C구간 후반+D 전환(122~129일)
mask = (day >= 122) & (day <= 129)
sub = merged[mask]
sub_day = day[mask]

fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

axes[0].plot(sub_day, sub["humidity_mean"], color="#DD8452", linewidth=1.1)
axes[0].axhline(78.4, color="gray", linestyle=":", label="test에 썼던 기준선(78.4)")
axes[0].set_ylabel("내부습도")
axes[0].set_title("train 122~129일 원본(5분 단위) — 내부습도 파동이 실제 EC 파동과 일치하는지 검증")
axes[0].legend(loc="lower right")
axes[0].grid(alpha=0.3)

axes[1].plot(sub_day, sub["temperature_mean"], color="#C44E52", linewidth=1.1)
axes[1].set_ylabel("내부온도")
axes[1].grid(alpha=0.3)

axes[2].plot(sub_day, sub["circ_fan_mean"], color="#4C72B0", linewidth=1.1)
axes[2].set_ylabel("circ_fan")
axes[2].grid(alpha=0.3)

axes[3].plot(sub_day, sub["soil_ec"], color="black", linewidth=1.1)
axes[3].set_ylabel("실제 soil_ec (정답)")
axes[3].set_xlabel("경과일 (day_num)")
axes[3].grid(alpha=0.3)

for d in range(122, 130):
    for ax in axes:
        ax.axvline(d, color="black", linestyle="--", alpha=0.2)

fig.tight_layout()
fig.savefig("eda_outputs/24_train_122_129_pulses.png", dpi=150)
print("저장 완료: eda_outputs/24_train_122_129_pulses.png")

# 상관계수로도 확인 (5분 단위 원본 기준)
corr = sub[["humidity_mean", "soil_ec"]].corr().iloc[0, 1]
print(f"\n이 구간(122~129일) 내부습도 vs 실제EC 상관계수: {corr:.3f}")
