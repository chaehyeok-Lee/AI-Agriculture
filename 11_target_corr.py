import os

import matplotlib.pyplot as plt
import pandas as pd

from preprocess import load_target

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

y = load_target("dataset/train/env/train_y.csv")
cols = ["soil_moisture", "soil_ec", "soil_temp"]
corr = y[cols].corr()

fig, ax = plt.subplots(figsize=(5, 4.5))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, rotation=30, ha="right")
ax.set_yticks(range(len(cols)))
ax.set_yticklabels(cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                 color="white" if abs(corr.values[i, j]) > 0.5 else "black", fontsize=11)
fig.colorbar(im, ax=ax, label="상관계수")
ax.set_title("타깃 3종 상호 상관관계")
fig.tight_layout()
fig.savefig("eda_outputs/11_target_corr.png", dpi=150)
print("저장 완료: eda_outputs/11_target_corr.png")
