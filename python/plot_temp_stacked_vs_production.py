import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/ 의 상위 = 프로젝트 루트
os.chdir(ROOT)

# temp_stacked_experiment.py 를 먼저 실행해서 만들어둔 예측 파일이 필요함
new = pd.read_csv("python/temp_stacked_test_pred.csv")
old = pd.read_csv("output/submission.csv")
old_dat = old["time"].str.extract(r"DAT(\d+)")[0].astype(int)
old["dat"] = old_dat

zone_map = {}
for d in [135,136,137]: zone_map[d]=0
for d in [138,139,140]: zone_map[d]=2
for d in [141,142,143]: zone_map[d]=3
for d in [144,145,146]: zone_map[d]=1
zone_colors = {0:"#2a78d6",1:"#eda100",2:"#1baf7a",3:"#e34948"}

fig, ax = plt.subplots(figsize=(15,4.5))
x_offset = 0
xticks, xticklabels = [], []
test_days = sorted(new["dat"].unique())
for d in test_days:
    n = new[new.dat==d]
    o = old[old.dat==d]
    x = range(x_offset, x_offset+len(n))
    z = zone_map[d]
    ax.plot(x, o["soil_temp"].to_numpy(), color="gray", linewidth=1.0, linestyle="--",
            label="production(기존)" if d==135 else None)
    ax.plot(x, n["soil_temp_stacked"].to_numpy(), color=zone_colors[z], linewidth=1.2,
            label="stacked(신규)" if d in [135,138,141,144] else None)
    xticks.append(x_offset+len(n)//2)
    xticklabels.append(f"D{d}(z{z})")
    x_offset += len(n)
    ax.axvline(x_offset, color="#cccccc", linewidth=0.5, linestyle=":")
ax.set_xticks(xticks); ax.set_xticklabels(xticklabels, fontsize=8)
ax.set_ylabel("Soil Temp (C)")
ax.set_title("soil_temp: production(회색 점선) vs stacked(env+oof_ec+oof_moisture, 색깔 실선)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)
plt.tight_layout()
out = "eda_outputs/36_temp_stacked_vs_production.png"
plt.savefig(out, dpi=140)
print("saved:", out)

import numpy as np
diff = new["soil_temp_stacked"].to_numpy() - old.sort_values("dat")["soil_temp"].to_numpy()
print(f"두 예측 차이: mean={diff.mean():.3f} std={diff.std():.3f} max_abs={np.abs(diff).max():.3f}")
