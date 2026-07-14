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
y = load_target("dataset/train/env/train_y.csv")
merged = train_feat.join(y, how="inner")
day = merged.index.total_seconds() / 86400

# 일별로 집계 (구간 판별은 하루 단위가 더 안정적)
day_num = (merged.index // pd.Timedelta("1D")).astype(int)
merged = merged.assign(day_num=day_num)
daily = merged.groupby("day_num").agg(
    circ_fan=("circ_fan_mean", "mean"),
    vent1=("greenhouse_roof_vent1_mean", "mean"),
    vent2=("greenhouse_roof_vent2_mean", "mean"),
    humidity_in=("humidity_mean", "mean"),
    ec=("soil_ec", "mean"),
    moisture=("soil_moisture", "mean"),
).reset_index()

# --- 2단계 추정 로직 ---
# 1단계: 순환팬(+천창)으로 큰 틀 구분 -> 팬 켜짐(換氣 ON) vs 꺼짐(OFF)
FAN_ON_THRESHOLD = 0.2
daily["fan_on"] = daily["circ_fan"] > FAN_ON_THRESHOLD

# 2단계: 팬 꺼짐 구간 안에서 내부습도로 B(회복기, 저EC) vs C(숙성기, 고EC) 세분화
HUMIDITY_SPLIT = daily.loc[~daily["fan_on"], "humidity_in"].median()

def estimate_stage(row):
    if row["fan_on"]:
        return "환기 ON (A or D 추정)"
    elif row["humidity_in"] >= HUMIDITY_SPLIT:
        return "환기 OFF + 내부습도 높음 (C 추정, 고EC)"
    else:
        return "환기 OFF + 내부습도 낮음 (B 추정, 저EC)"

daily["estimated_stage"] = daily.apply(estimate_stage, axis=1)

print("=== 일별 추정 결과 vs 실제 EC ===")
print(daily[["day_num", "circ_fan", "humidity_in", "estimated_stage", "ec"]].to_string(index=False))

stage_colors = {
    "환기 ON (A or D 추정)": "#4C72B0",
    "환기 OFF + 내부습도 높음 (C 추정, 고EC)": "#C44E52",
    "환기 OFF + 내부습도 낮음 (B 추정, 저EC)": "#55A868",
}

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# 위: 실제 EC + 추정 단계를 배경색으로
ax1 = axes[0]
for _, row in daily.iterrows():
    ax1.axvspan(row["day_num"], row["day_num"] + 1, color=stage_colors[row["estimated_stage"]], alpha=0.25)
ax1.plot(day, merged["soil_ec"], color="black", linewidth=0.8)
ax1.set_ylabel("실제 soil_ec")
ax1.set_title("실제 EC (검은선) vs 순환팬+내부습도 기반 추정 구간 (배경색)")
ax1.grid(alpha=0.3)

# 아래: 추정에 쓴 원재료 신호들
ax2 = axes[1]
ax2b = ax2.twinx()
l1, = ax2.plot(daily["day_num"], daily["circ_fan"], color="#4C72B0", marker="o", label="일평균 circ_fan")
ax2.axhline(FAN_ON_THRESHOLD, color="#4C72B0", linestyle=":", alpha=0.6)
l2, = ax2b.plot(daily["day_num"], daily["humidity_in"], color="#DD8452", marker="s", label="일평균 내부습도")
ax2b.axhline(HUMIDITY_SPLIT, color="#DD8452", linestyle=":", alpha=0.6)
ax2.set_ylabel("circ_fan (일평균)", color="#4C72B0")
ax2b.set_ylabel("내부습도 (일평균)", color="#DD8452")
ax2.set_xlabel("경과일 (day_num)")
ax2.legend(handles=[l1, l2], loc="upper left")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("eda_outputs/19_ec_proxy_estimate.png", dpi=150)
print("\n저장 완료: eda_outputs/19_ec_proxy_estimate.png")

# 정확도 요약: 추정 단계별 실제 EC 평균이 잘 갈리는지
print("\n=== 추정 단계별 실제 EC 통계 ===")
print(daily.groupby("estimated_stage")["ec"].agg(["mean", "std", "count"]))
