import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import load_target

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

train_y = load_target("dataset/train/env/train_y.csv")
day_num = train_y.index // pd.Timedelta("1D")

corr = np.corrcoef(day_num, train_y["soil_ec"])[0, 1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(day_num, train_y["soil_ec"], s=4, alpha=0.3, color="#DD8452")
ax.set_xlabel("day_num (경과일)")
ax.set_ylabel("soil_ec")
ax.set_title(f"day_num vs soil_ec (상관계수 {corr:.2f})")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("eda_outputs/07_daynum_vs_ec.png", dpi=150)
print("저장 완료: eda_outputs/07_daynum_vs_ec.png")
print(f"상관계수: {corr:.4f}")
