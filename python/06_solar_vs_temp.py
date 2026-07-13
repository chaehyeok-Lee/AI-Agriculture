import os

import matplotlib.pyplot as plt
import pandas as pd

from preprocess import build_features

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

TRAIN_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. 산점도: 일사량 vs 내부온도 (train/test 겹쳐서)
axes[0].scatter(train_feat["solar_radiation_mean"], train_feat["temperature_mean"],
                 s=4, alpha=0.3, color=TRAIN_COLOR, label="train")
axes[0].scatter(test_feat["solar_radiation_mean"], test_feat["temperature_mean"],
                 s=4, alpha=0.3, color=TEST_COLOR, label="test")
axes[0].set_xlabel("solar_radiation (5분 평균)")
axes[0].set_ylabel("내부온도 (°C)")
axes[0].set_title("일사량 vs 내부온도")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. 야간(일사량 <= 5, 기저값 포함) 만 따로 — "맑은 날 밤이 더 춥다" 가설 확인용
# 주의: 원본 solar_radiation은 밤에도 정확히 0이 아니라 ~4.98 근처 기저값을 찍음 (센서 노이즈 바닥)
train_night = train_feat[train_feat["solar_radiation_mean"] <= 5]
test_night = test_feat[test_feat["solar_radiation_mean"] <= 5]
axes[1].hist(train_night["temperature_mean"].dropna(), bins=30, alpha=0.6, density=True,
             color=TRAIN_COLOR, label="train (야간)")
axes[1].hist(test_night["temperature_mean"].dropna(), bins=30, alpha=0.6, density=True,
             color=TEST_COLOR, label="test (야간)")
axes[1].set_xlabel("내부온도 (°C)")
axes[1].set_title("야간(일사량<=5)만 내부온도 분포")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("일사량 vs 내부온도 교차분석")
fig.tight_layout()
fig.savefig("eda_outputs/06_solar_vs_temp.png", dpi=150)
print("저장 완료: eda_outputs/06_solar_vs_temp.png")

print("\ntrain 상관계수(solar_radiation vs 내부온도):",
      train_feat[["solar_radiation_mean", "temperature_mean"]].corr().iloc[0, 1])
print("test 상관계수(solar_radiation vs 내부온도):",
      test_feat[["solar_radiation_mean", "temperature_mean"]].corr().iloc[0, 1])

print("\ntrain 야간(<=5) 표본 수:", len(train_night), " 평균 내부온도:", train_night["temperature_mean"].mean())
print("test 야간(<=5) 표본 수:", len(test_night), " 평균 내부온도:", test_night["temperature_mean"].mean())

plt.show()
