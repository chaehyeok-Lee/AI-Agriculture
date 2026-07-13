import os

import matplotlib.pyplot as plt

from preprocess import load_target

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
os.makedirs("eda_outputs", exist_ok=True)

train_y = load_target("dataset/train/env/train_y.csv")
day = train_y.index.total_seconds() / 86400

transition_days = [115, 121, 129]
fig, axes = plt.subplots(len(transition_days), 2, figsize=(12, 10))
for i, td in enumerate(transition_days):
    mask = (day >= td - 1) & (day <= td + 1)

    axes[i, 0].plot(day[mask], train_y["soil_moisture"][mask], color="#4C72B0", marker=".", markersize=2)
    axes[i, 0].axvline(td, color="red", linestyle="--", alpha=0.5)
    axes[i, 0].set_title(f"{td}일 경계 — soil_moisture")
    axes[i, 0].grid(alpha=0.3)

    axes[i, 1].plot(day[mask], train_y["soil_ec"][mask], color="#DD8452", marker=".", markersize=2)
    axes[i, 1].axvline(td, color="red", linestyle="--", alpha=0.5)
    axes[i, 1].set_title(f"{td}일 경계 — soil_ec")
    axes[i, 1].grid(alpha=0.3)

axes[-1, 0].set_xlabel("경과일 (day)")
axes[-1, 1].set_xlabel("경과일 (day)")
fig.suptitle("전환 구간 확대 (±1일)")
fig.tight_layout()
fig.savefig("eda_outputs/05_transition_zoom.png", dpi=150)
print("저장 완료: eda_outputs/05_transition_zoom.png")
plt.show()
