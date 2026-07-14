"""죽어있는 신호 전체 스캔 — train vs test 피처 분포 OOD 분석
루프1: val은 좋은데 submission이 평평한 원인 피처 전체 점검
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from preprocess import build_features
from train import (
    add_trend_features, add_lag_features, add_rolling_features,
    add_cyclic_features, BlendModel,  # noqa: F401
)

train_raw = pd.read_csv('dataset/train/env/train_X.csv')
test_raw  = pd.read_csv('dataset/test/env/test_X.csv')
combined  = pd.concat([train_raw, test_raw], ignore_index=True)
feat = build_features(combined)
feat = add_trend_features(feat)
feat = add_lag_features(feat)
feat = add_rolling_features(feat)
feat = add_cyclic_features(feat)

train_feat = feat[feat.index < pd.Timedelta(days=135)]
test_feat  = feat[feat.index >= pd.Timedelta(days=135)]

# ── OOD 비율 ──────────────────────────────────────────────────────────────────
print("=== OOD (Out-of-Distribution): test가 train 범위를 벗어나는 비율 ===\n")
results = []
for col in train_feat.columns:
    tr = train_feat[col].dropna()
    te = test_feat[col].dropna()
    if len(tr) == 0 or len(te) == 0:
        continue
    tr_min, tr_max = tr.min(), tr.max()
    ood_frac = ((te < tr_min) | (te > tr_max)).mean()
    results.append({
        'feature': col,
        'ood_%': round(ood_frac * 100, 1),
        'tr_min': round(tr_min, 3),
        'tr_max': round(tr_max, 3),
        'te_min': round(te.min(), 3),
        'te_max': round(te.max(), 3),
        'te_mean': round(te.mean(), 3),
        'tr_mean': round(tr.mean(), 3),
    })

ood_df = (pd.DataFrame(results)
          .sort_values('ood_%', ascending=False)
          .reset_index(drop=True))

print(ood_df[ood_df['ood_%'] > 1.0].to_string(index=False))
print(f"\n총 {(ood_df['ood_%'] > 1.0).sum()}개 피처 OOD 비율 >1%")
print(f"총 {(ood_df['ood_%'] > 5.0).sum()}개 피처 OOD 비율 >5%")
print(f"총 {(ood_df['ood_%'] > 20.0).sum()}개 피처 OOD 비율 >20%")

# ── day_num 외삽 확인 ─────────────────────────────────────────────────────────
print("\n=== day_num 외삽 정도 ===")
tr_day_max = train_feat['day_num'].max()
te_day_min = test_feat['day_num'].min()
te_day_max = test_feat['day_num'].max()
print(f"  train 최대 day_num: {tr_day_max:.0f}")
print(f"  test  범위:         {te_day_min:.0f} ~ {te_day_max:.0f}")
print(f"  외삽 간격:          +{te_day_min - tr_day_max:.0f}일 (0이면 연속, 양수면 갭)")

# ── circ_fan 관련 피처 분포 ─────────────────────────────────────────────────
print("\n=== circ_fan 관련 피처: 구간별 평균 (train / test_off(135~137) / test_on(138~143)) ===")
m_off = (test_feat.index >= pd.Timedelta(days=135)) & (test_feat.index < pd.Timedelta(days=138))
m_on  = (test_feat.index >= pd.Timedelta(days=138)) & (test_feat.index < pd.Timedelta(days=144))
circ_cols = [c for c in feat.columns if 'circ_fan' in c]
print(f"{'피처':45s}  train_mean  off_mean  on_mean  diff(on-off)")
for col in circ_cols:
    tr_m = train_feat[col].mean()
    of_m = test_feat.loc[m_off, col].mean()
    on_m = test_feat.loc[m_on,  col].mean()
    print(f"  {col:45s}  {tr_m:10.4f}  {of_m:8.4f}  {on_m:7.4f}  {on_m-of_m:11.4f}")

# ── 날씨 복제 패턴 확인 ──────────────────────────────────────────────────────
print("\n=== 날씨 복제 패턴 확인: test 날씨가 train 어느 날과 동일한지 ===")
weather_cols = [
    'temperature_outside_mean', 'humidity_outside_mean',
    'solar_radiation_mean', 'wind_speed_outside_mean',
]
# 일별 평균으로 비교
def day_avg(df, cols):
    df = df.copy()
    df['day'] = (df.index // pd.Timedelta('1D')).astype(int)
    return df.groupby('day')[cols].mean().round(3)

tr_daily = day_avg(train_feat, weather_cols)
te_daily = day_avg(test_feat, weather_cols)

for te_day, te_row in te_daily.iterrows():
    diffs = (tr_daily - te_row).abs().sum(axis=1)
    best_match = diffs.idxmin()
    best_diff = diffs.min()
    print(f"  test DAT{te_day}: ← train DAT{best_match} 와 유사 (절대차합={best_diff:.3f})")
