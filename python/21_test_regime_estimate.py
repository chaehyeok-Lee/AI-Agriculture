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

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
test_feat = build_features(pd.read_csv("dataset/test/env/test_X.csv"))

FAN_ON_THRESHOLD = 0.2
# train B/C 구간(환기 OFF)의 내부습도 중앙값을 그대로 기준선으로 재사용 (test는 정답이 없어 test 자체 기준을 못 만듦)
train_day = (train_feat.index // pd.Timedelta("1D")).astype(int)
train_feat_tmp = train_feat.assign(day_num=train_day)
bc_mask = (train_feat_tmp["day_num"] >= 115) & (train_feat_tmp["day_num"] < 129)
HUMIDITY_SPLIT = train_feat_tmp.loc[bc_mask & (train_feat_tmp["circ_fan_mean"] <= FAN_ON_THRESHOLD), "humidity_mean"].median()
print(f"내부습도 기준선(train B/C 구간 중앙값): {HUMIDITY_SPLIT:.2f}")


def build_daily(feat_df):
    day_num = (feat_df.index // pd.Timedelta("1D")).astype(int)
    feat_df = feat_df.assign(day_num=day_num)
    daily = feat_df.groupby("day_num").agg(
        circ_fan=("circ_fan_mean", "mean"),
        vent1=("greenhouse_roof_vent1_mean", "mean"),
        humidity_in=("humidity_mean", "mean"),
        temp_in=("temperature_mean", "mean"),
    ).reset_index()
    daily["fan_on"] = daily["circ_fan"] > FAN_ON_THRESHOLD

    def stage(row):
        if row["fan_on"]:
            return "환기 ON (저EC 추정)"
        elif row["humidity_in"] >= HUMIDITY_SPLIT:
            return "환기 OFF+습도높음 (고EC 추정)"
        else:
            return "환기 OFF+습도낮음 (저EC 추정)"

    daily["estimated_stage"] = daily.apply(stage, axis=1)
    return daily


train_daily = build_daily(train_feat)
test_daily = build_daily(test_feat)

print("\n=== test 일별 추정 ===")
print(test_daily.to_string(index=False))

# 전환 지점(추정 단계가 바뀌는 날) 표시
test_daily["changed"] = test_daily["estimated_stage"] != test_daily["estimated_stage"].shift(1)
transitions = test_daily.loc[test_daily["changed"], "day_num"].tolist()
print(f"\ntest 구간 내 추정 전환일: {transitions}")

stage_colors = {
    "환기 ON (저EC 추정)": "#4C72B0",
    "환기 OFF+습도높음 (고EC 추정)": "#C44E52",
    "환기 OFF+습도낮음 (저EC 추정)": "#55A868",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

ax1 = axes[0]
for _, row in train_daily.iterrows():
    ax1.axvspan(row["day_num"], row["day_num"] + 1, color=stage_colors[row["estimated_stage"]], alpha=0.3)
ax1.set_xlim(train_daily["day_num"].min(), train_daily["day_num"].max() + 1)
ax1.set_title("train — 추정 구간 (정답 EC로 이미 검증됨)")
ax1.set_yticks([])

ax2 = axes[1]
for _, row in test_daily.iterrows():
    ax2.axvspan(row["day_num"], row["day_num"] + 1, color=stage_colors[row["estimated_stage"]], alpha=0.3)
    ax2.text(row["day_num"] + 0.5, 0.58, str(int(row["day_num"])), ha="center", va="center", fontsize=9)
    ax2.text(row["day_num"] + 0.5, 0.40, f"습도{row['humidity_in']:.0f}", ha="center", va="center", fontsize=7, color="#555555")
ax2.set_xlim(test_daily["day_num"].min(), test_daily["day_num"].max() + 1)
ax2.set_title("test — 순환팬+내부습도만으로 추정 (정답 없음, X데이터로만 유추)")
ax2.set_xlabel("경과일 (day_num)")
ax2.set_yticks([])

from matplotlib.patches import Patch
handles = [Patch(color=c, alpha=0.3, label=l) for l, c in stage_colors.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("eda_outputs/21_test_regime_estimate.png", dpi=150)
print("\n저장 완료: eda_outputs/21_test_regime_estimate.png")
