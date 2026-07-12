import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

sub = pd.read_csv("output/submission.csv")
ec = sub["soil_ec"]

n_total = len(ec)
n_unique = ec.nunique()
top_value = ec.value_counts().idxmax()
top_count = ec.value_counts().max()
top_ratio = top_count / n_total

print(f"전체 예측 개수: {n_total}")
print(f"고유값 개수: {n_unique}")
print(f"가장 많이 나온 값: {top_value:.4f} ({top_count}개, {top_ratio:.1%})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1. 예측값 시계열 (test 12일)
axes[0].plot(range(n_total), ec.values, color="#DD8452", linewidth=0.8)
axes[0].axhline(top_value, color="red", linestyle="--", alpha=0.6,
                 label=f"최다값 {top_value:.3f} ({top_ratio:.0%})")
axes[0].set_xlabel("test 시점 순서 (5분 단위, 0~3455)")
axes[0].set_ylabel("예측된 soil_ec")
axes[0].set_title("test 구간 soil_ec 예측값 시계열")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 2. 예측값 빈도 상위 15개
top15 = ec.value_counts().head(15).sort_values()
colors = ["red" if v == top_value else "#4C72B0" for v in top15.index]
axes[1].barh(range(len(top15)), top15.values, color=colors)
axes[1].set_yticks(range(len(top15)))
axes[1].set_yticklabels([f"{v:.4f}" for v in top15.index], fontsize=8)
axes[1].set_xlabel("등장 횟수")
axes[1].set_title(f"예측값 빈도 상위 15개 (전체 고유값 {n_unique}개 중)")
axes[1].grid(alpha=0.3, axis="x")

fig.suptitle(f"soil_ec 예측 뭉침 현황 — 전체 {n_total}개 중 고유값 {n_unique}개, 최다값 {top_ratio:.0%} 차지")
fig.tight_layout()
fig.savefig("eda_outputs/10_ec_clumping.png", dpi=150)
print("저장 완료: eda_outputs/10_ec_clumping.png")
