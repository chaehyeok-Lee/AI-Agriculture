import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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

# 지금까지 분석한 4개 EC 레벨 구간(=추정 양액 레시피 구간)
# growth_stage: 딸기 재배 도메인 지식 기반 "추정" 생육단계 — 실제 재배일지로 확인된 건 아님(가설)
REGIMES = [
    (109, 115, "A: 저농도 초기 (109~114일)", "추정: 이전 화방 수확 후기\n(영양생장 위주)"),
    (115, 121, "B: 저농도 회복기 (115~120일)", "추정: 다음 화방 준비기\n(저농도 영양생장)"),
    (121, 129, "C: 고농도 숙성기 (121~128일)", "추정: 과실 비대·숙성기\n(당도 관리용 고농도)"),
    (129, 135, "D: 저농도 재조정 (129~134일)", "추정: 수확기 + 다음 화방\n회복 시작 (저농도 전환)"),
]


def regime_label(d):
    for start, end, label, _stage in REGIMES:
        if start <= d < end:
            return label
    return "기타"


y["regime"] = y["day_num"].apply(regime_label)

# 일별 집계
daily = y.groupby("day_num").agg(
    moisture_mean=("soil_moisture", "mean"),
    moisture_min=("soil_moisture", "min"),
    moisture_max=("soil_moisture", "max"),
    ec_mean=("soil_ec", "mean"),
    ec_min=("soil_ec", "min"),
    ec_max=("soil_ec", "max"),
).reset_index()
daily["regime"] = daily["day_num"].apply(regime_label)

print("=== 구간(양액 레시피 추정)별 요약 ===")
regime_summary = y.groupby("regime")[["soil_moisture", "soil_ec"]].agg(["mean", "std"])
print(regime_summary)

print("\n=== 일별 상세 ===")
print(daily.to_string(index=False))

corr_all = daily[["moisture_mean", "ec_mean"]].corr().iloc[0, 1]
print(f"\n일평균 기준 전체 상관계수(수분 vs EC): {corr_all:.3f}")
for start, end, label, stage in REGIMES:
    sub = daily[(daily["day_num"] >= start) & (daily["day_num"] < end)]
    if len(sub) >= 3:
        c = sub[["moisture_mean", "ec_mean"]].corr().iloc[0, 1]
        print(f"  {label} 구간 내 상관계수: {c:.3f}  |  {stage.replace(chr(10), ' ')}")

# ---- 그래프 ----
regime_colors = {"A: 저농도 초기 (109~114일)": "#4C72B0", "B: 저농도 회복기 (115~120일)": "#55A868",
                  "C: 고농도 숙성기 (121~128일)": "#C44E52", "D: 저농도 재조정 (129~134일)": "#8172B2"}

fig, axes = plt.subplots(2, 1, figsize=(14, 9))

# 위: 일별 평균 수분·EC 겹쳐보기(twin axis) + 구간 배경색 + 추정 생육단계 텍스트
ax1 = axes[0]
for start, end, label, stage in REGIMES:
    ax1.axvspan(start, end, color=regime_colors[label], alpha=0.12)
    ax1.text((start + end) / 2, 1.03, stage, transform=ax1.get_xaxis_transform(),
              ha="center", va="bottom", fontsize=8, color=regime_colors[label])
l1, = ax1.plot(daily["day_num"], daily["moisture_mean"], color="#4C72B0", marker="o", markersize=4, label="일평균 수분(%)")
ax1.set_ylabel("soil_moisture 일평균(%)", color="#4C72B0")
ax1.tick_params(axis="y", labelcolor="#4C72B0")
ax1.set_xlabel("경과일 (day_num)")
ax2 = ax1.twinx()
l2, = ax2.plot(daily["day_num"], daily["ec_mean"], color="#DD8452", marker="s", markersize=4, label="일평균 EC")
ax2.set_ylabel("soil_ec 일평균", color="#DD8452")
ax2.tick_params(axis="y", labelcolor="#DD8452")
ax1.legend(handles=[l1, l2], loc="upper left")
ax1.grid(alpha=0.3)
fig.suptitle("일별 평균 수분·EC 비교 (배경색 = 추정 양액 구간, 위쪽 텍스트 = 추정 생육단계)", y=1.0)

# 아래: 구간별 색으로 산점도 (일평균 수분 vs 일평균 EC)
ax3 = axes[1]
for start, end, label, stage in REGIMES:
    sub = daily[(daily["day_num"] >= start) & (daily["day_num"] < end)]
    ax3.scatter(sub["moisture_mean"], sub["ec_mean"], color=regime_colors[label],
                label=f"{label}\n{stage}", s=60)
    for _, row in sub.iterrows():
        ax3.annotate(str(int(row["day_num"])), (row["moisture_mean"], row["ec_mean"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
ax3.set_xlabel("soil_moisture 일평균(%)")
ax3.set_ylabel("soil_ec 일평균")
ax3.set_title(f"구간별 일평균 수분 vs EC 관계 (전체 상관계수 {corr_all:.2f})")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("eda_outputs/17_regime_comparison.png", dpi=150)
print("\n저장 완료: eda_outputs/17_regime_comparison.png")
