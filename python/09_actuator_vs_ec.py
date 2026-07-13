import os

import matplotlib.pyplot as plt
import pandas as pd

from preprocess import build_features, load_target

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
train_y = load_target("dataset/train/env/train_y.csv")
merged = train_feat.join(train_y, how="inner")
day = merged.index.total_seconds() / 86400

plot_cols = [
    ("circ_fan_mean", "순환팬 (circ_fan)", "#4C72B0"),
    ("greenhouse_roof_vent1_mean", "천창1 개도율", "#55A868"),
    ("greenhouse_roof_vent2_mean", "천창2 개도율", "#8172B2"),
    ("soil_ec", "soil_ec (타깃)", "#DD8452"),
]
transition_days = [115, 121, 129]

fig, axes = plt.subplots(len(plot_cols), 1, figsize=(13, 9), sharex=True)
for ax, (col, label, color) in zip(axes, plot_cols):
    ax.plot(day, merged[col], color=color, linewidth=0.8)
    ax.set_ylabel(label)
    ax.grid(alpha=0.3)
    for td in transition_days:
        ax.axvline(td, color="red", linestyle="--", alpha=0.4)

axes[-1].set_xlabel("경과일 (day_num)")
axes[0].set_title("순환팬 / 천창1 / 천창2 / soil_ec 비교 (빨간 점선 = 전환일 115/121/129)")
fig.tight_layout()
fig.savefig("eda_outputs/09_actuator_vs_ec.png", dpi=150)
print("저장 완료: eda_outputs/09_actuator_vs_ec.png")
