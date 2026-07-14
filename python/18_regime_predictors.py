import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
day_num = (train_feat.index // pd.Timedelta("1D")).astype(int)
train_feat = train_feat.assign(day_num=day_num)

REGIMES = [
    (109, 115, "A(109~114)"),
    (115, 121, "B(115~120)"),
    (121, 129, "C(121~128)"),
    (129, 135, "D(129~134)"),
]


def regime_label(d):
    for start, end, label in REGIMES:
        if start <= d < end:
            return label
    return "기타"


train_feat["regime"] = train_feat["day_num"].apply(regime_label)
regime_order = [r[2] for r in REGIMES]
regime_colors = {"A(109~114)": "#4C72B0", "B(115~120)": "#55A868",
                  "C(121~128)": "#C44E52", "D(129~134)": "#8172B2"}

# 후보 변수: 날씨(외부, 복제 의심) + 내부환경(진짜 신호) + 구동기(참고용, 이미 강한 예측력 확인됨)
CANDIDATES = [
    ("temperature_outside_mean", "외부온도"),
    ("humidity_outside_mean", "외부습도"),
    ("temperature_mean", "내부온도"),
    ("humidity_mean", "내부습도"),
    ("circ_fan_mean", "순환팬(참고)"),
]

print("=== 구간별 변수 평균 ± 표준편차 (구별력 확인용) ===")
summary = train_feat.groupby("regime")[[c for c, _ in CANDIDATES]].agg(["mean", "std"])
print(summary.reindex(regime_order))

# 구간 간 분리도 대략 확인: (구간간 분산) / (구간내 분산) — 값이 클수록 그 변수로 구간을 잘 구별할 수 있음
print("\n=== 구간 구별력 점수 (between/within variance ratio, 높을수록 구별 잘 됨) ===")
for col, label in CANDIDATES:
    grp_means = train_feat.groupby("regime")[col].mean()
    grand_mean = train_feat[col].mean()
    between = ((grp_means - grand_mean) ** 2).mean()
    within = train_feat.groupby("regime")[col].var().mean()
    ratio = between / within if within > 0 else np.nan
    print(f"  {label:10s} ({col}): {ratio:.4f}")

fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(4 * len(CANDIDATES), 5))
for ax, (col, label) in zip(axes, CANDIDATES):
    data = [train_feat.loc[train_feat["regime"] == r, col].dropna() for r in regime_order]
    bp = ax.boxplot(data, labels=regime_order, patch_artist=True)
    for patch, r in zip(bp["boxes"], regime_order):
        patch.set_facecolor(regime_colors[r])
        patch.set_alpha(0.6)
    ax.set_title(label)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")

fig.suptitle("구간(A/B/C/D)별 X변수 분포 — 날씨/온습도로 양액 구간을 구별할 수 있는지 확인")
fig.tight_layout()
fig.savefig("eda_outputs/18_regime_predictors.png", dpi=150)
print("\n저장 완료: eda_outputs/18_regime_predictors.png")
