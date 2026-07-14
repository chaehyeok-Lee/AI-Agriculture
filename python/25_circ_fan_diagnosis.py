"""circ_fan 평평 예측 진단 — 트리 구조 + 피처 분포 + submission 분석
루프1: circ_fan 구별력 1.35인데 test 예측이 평평한 원인 추적
"""
import pickle
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np

from preprocess import build_features
from train import (
    add_trend_features, add_lag_features, add_rolling_features,
    add_cyclic_features, DROP_COLS_PER_TARGET, BlendModel,  # noqa: F401
)

# ── 1. 모델 로드 ──────────────────────────────────────────────────────────────
with open('model/model.pkl', 'rb') as f:
    models = pickle.load(f)
ec_model = models['soil_ec']

# ── 2. 피처 준비 (inference.py와 완전 동일) ───────────────────────────────────
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
ec_cols = [c for c in test_feat.columns if c not in DROP_COLS_PER_TARGET['soil_ec']]
ec_test_X = test_feat[ec_cols]

# ── 3. 트리 구조: 최상위 분기 피처 ─────────────────────────────────────────────
tree_df = ec_model.booster_.trees_to_dataframe()
print("=== [A] 최상위 분기 피처 (depth=0), n_trees 기준 빈도 ===")
top0 = tree_df[tree_df['node_depth'] == 1]['split_feature'].value_counts().head(15)
print(top0.to_string())
print()

print("=== [B] 깊이별 circ_fan 등장 횟수 ===")
circ_mask = tree_df['split_feature'].str.contains('circ_fan', na=False)
depth_dist = tree_df[circ_mask]['node_depth'].value_counts().sort_index()
print(depth_dist.to_string())
print()

# ── 4. Feature Importance (split / gain) ───────────────────────────────────
fi_split = pd.Series(ec_model.booster_.feature_importance('split'),
                     index=ec_model.feature_name_).sort_values(ascending=False)
fi_gain  = pd.Series(ec_model.booster_.feature_importance('gain'),
                     index=ec_model.feature_name_).sort_values(ascending=False)
print("=== [C] Feature Importance (split 기준 상위 25) ===")
print(fi_split.head(25).to_string())
print()
print("=== [D] Feature Importance (gain 기준 상위 25) ===")
print(fi_gain.head(25).to_string())
print()

# ── 5. circ_fan 관련 피처 EC 모델 포함 여부 확인 ────────────────────────────
all_circ = [c for c in feat.columns if 'circ_fan' in c]
in_model  = [c for c in all_circ if c in ec_cols]
excluded  = [c for c in all_circ if c not in ec_cols]
print("=== [E] circ_fan 피처 중 EC 모델에 포함된 것 ===")
print(in_model)
print()
print("=== [F] circ_fan 피처 중 EC 모델에서 제외된 것 (_ZERO_EC) ===")
print(excluded)
print()

# ── 6. 135~137(OFF) vs 138~143(ON) 피처값 비교 ──────────────────────────────
m_off = (test_feat.index >= pd.Timedelta(days=135)) & (test_feat.index < pd.Timedelta(days=138))
m_on  = (test_feat.index >= pd.Timedelta(days=138)) & (test_feat.index < pd.Timedelta(days=144))
m_off2= (test_feat.index >= pd.Timedelta(days=144))

print("=== [G] circ_fan 관련 피처 값: 구간별 평균 ===")
print(f"{'피처':45s}  OFF(135~137)  ON(138~143)  OFF2(144~)")
for col in in_model:
    if 'circ_fan' in col:
        v_off  = test_feat.loc[m_off,  col].mean()
        v_on   = test_feat.loc[m_on,   col].mean()
        v_off2 = test_feat.loc[m_off2, col].mean()
        print(f"  {col:45s}  {v_off:12.4f}  {v_on:11.4f}  {v_off2:9.4f}")
print()

# ── 7. day_num 분포 ──────────────────────────────────────────────────────────
print(f"=== [H] day_num 범위 ===")
print(f"  train: {train_feat['day_num'].min():.0f} ~ {train_feat['day_num'].max():.0f}")
print(f"  test:  {test_feat['day_num'].min():.0f} ~ {test_feat['day_num'].max():.0f}")
print()

# ── 8. 실제 예측값 — OFF/ON 구간별 분산 ─────────────────────────────────────
sub_path = 'output/submission.csv'
if os.path.exists(sub_path):
    sub = pd.read_csv(sub_path)
    sub['day'] = sub['time'].str.extract(r'DAT(\d+)')[0].astype(int)
    print("=== [I] submission EC 일별 통계 ===")
    print(sub.groupby('day')['soil_ec'].agg(['mean', 'std', 'min', 'max']).round(5).to_string())
    print()
    print(f"  EC 전체 고유값 수: {sub['soil_ec'].nunique()}")
    print(f"  EC 최다값 비중: {sub['soil_ec'].value_counts(normalize=True).iloc[0]:.1%}")
else:
    print("[I] submission.csv 없음 — python inference.py 먼저 실행 필요")
print()

# ── 9. 부분 의존성 분석: circ_fan_mean_roll288 변화 시 EC 예측 변화 확인 ─────
print("=== [J] 부분 의존성: circ_fan_mean_roll288를 0→1로 바꿨을 때 EC 변화 ===")
if 'circ_fan_mean_roll288' in ec_cols:
    # test 138~143일 샘플에서 circ_fan_mean_roll288만 변조
    sample = ec_test_X.loc[m_on].copy()
    pred_original = ec_model.predict(sample)
    sample_low = sample.copy()
    sample_low['circ_fan_mean_roll288'] = 0.0
    pred_low = ec_model.predict(sample_low)
    sample_high = sample.copy()
    sample_high['circ_fan_mean_roll288'] = 0.9
    pred_high = ec_model.predict(sample_high)
    print(f"  원본(실제값 평균={sample['circ_fan_mean_roll288'].mean():.3f}): "
          f"EC 예측 평균={pred_original.mean():.4f} std={pred_original.std():.4f}")
    print(f"  roll288=0.0(팬OFF)로 변조:  EC 예측 평균={pred_low.mean():.4f} std={pred_low.std():.4f}")
    print(f"  roll288=0.9(팬ON)로 변조:   EC 예측 평균={pred_high.mean():.4f} std={pred_high.std():.4f}")
    print(f"  → EC 변화폭: {pred_low.mean() - pred_high.mean():.4f} (OFF-ON 차이, 양수면 팬OFF일때 EC높음)")
else:
    print("  circ_fan_mean_roll288가 EC 모델 피처에 없음!")
