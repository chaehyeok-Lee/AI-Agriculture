import pandas as pd

from preprocess import build_features

train_feat = build_features(pd.read_csv("dataset/train/env/train_X.csv"))
day = train_feat.index.total_seconds() / 86400

transition_days = [115, 121, 129]
mean_cols = [c for c in train_feat.columns if c.endswith("_mean")]
std_all = train_feat[mean_cols].std()

for td in transition_days:
    before = train_feat[(day >= td - 1) & (day < td)][mean_cols].mean()
    after = train_feat[(day >= td) & (day < td + 1)][mean_cols].mean()
    diff = after - before
    z = (diff / std_all).sort_values(key=abs, ascending=False)

    print(f"\n=== {td}일 전환 — 전날 대비 다음날 변화 (표준화 점수 큰 순, 상위 8개) ===")
    for col in z.index[:8]:
        print(f"  {col:30s} 변화량(z) = {z[col]:+.2f}   (전 {before[col]:.2f} -> 후 {after[col]:.2f})")
