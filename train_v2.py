"""학습 진입점: 모델을 적합하여 model/ 에 저장.

타깃별 전략
- soil_temp : 커튼레짐별 (Ridge+HGB 블렌드) 전문가, env + lag/dewpoint 피처
- soil_moisture : 최근3일 레벨(persistence) + 일사구동 within-day shape(HGB 잔차)
- soil_ec : 강화레짐(커튼 agreement>=0.98 AND 순환팬 OFF) 고전문가(env+다분광 융합)
             + 저레짐 persistence(최근3 저일 EC). 소프트 신뢰도로 블렌드.
"""
from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import features_v2 as F
from train import FOLD_CUTOFFS, rmse

MODEL_DIR = 'model'


def run_folds_v2(FULL, feats, lagf, regEC, cutoffs=FOLD_CUTOFFS):
    """train.py의 4-fold 시계열 교차검증(누적 학습구간 + 4일 검증, 동일 FOLD_CUTOFFS)을
    레짐 라우팅 파이프라인에 그대로 적용. cutoff마다 학습(처음~cutoff직전)만으로
    레짐별 전문가/persistence를 처음부터 다시 만들고, 검증(cutoff~cutoff+4일)에서 RMSE를 잰다.
    FULL은 NaN을 채우지 않은 원본 병합 프레임이어야 함 — fold마다 그 fold의 학습구간
    median으로만 결측을 채워야 미래 정보 누수가 없음."""
    results = {'soil_moisture': [], 'soil_ec': [], 'soil_temp': []}
    for cutoff in cutoffs:
        tr_mask = FULL['dat'] < cutoff
        val_mask = (FULL['dat'] >= cutoff) & (FULL['dat'] < cutoff + 4)
        TR, VAL = FULL[tr_mask].copy(), FULL[val_mask].copy()
        if len(VAL) == 0:
            continue

        med = TR[feats + lagf].median()
        TR[feats + lagf] = TR[feats + lagf].fillna(med)
        VAL[feats + lagf] = VAL[feats + lagf].fillna(med)
        med_ms = TR[F.MSC].median()
        TR[F.MSC] = TR[F.MSC].fillna(med_ms)
        VAL[F.MSC] = VAL[F.MSC].fillna(med_ms)

        # soil_temp: 커튼 레짐별 전문가
        temp_experts = {}
        for rv in sorted(TR['reg'].dropna().unique()):
            sub = TR[TR['reg'] == rv]
            if len(sub) >= 100:
                temp_experts[rv] = F.fit_expert(sub, feats + lagf, 'soil_temp')
        if not temp_experts:
            temp_experts[0] = F.fit_expert(TR, feats + lagf, 'soil_temp')
        temp_default = sorted(temp_experts)[0]

        pred_temp = np.zeros(len(VAL))
        for rv, exp in temp_experts.items():
            m = (VAL['reg'] == rv).to_numpy()
            if m.any():
                pred_temp[m] = F.pred_expert(exp, VAL[m], feats + lagf)
        unmatched = ~VAL['reg'].isin(temp_experts.keys()).to_numpy()
        if unmatched.any():
            pred_temp[unmatched] = F.pred_expert(temp_experts[temp_default], VAL[unmatched], feats + lagf)
        results['soil_temp'].append(rmse(VAL['soil_temp'].to_numpy(), pred_temp))

        # soil_moisture: 최근3일 baseline + within-day shape
        last3 = sorted(TR['dat'].unique())[-3:]
        moist_baseline = float(TR[TR['dat'].isin(last3)]['soil_moisture'].mean())
        dm = TR.groupby('dat')['soil_moisture'].transform('mean')
        moist_shape = HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=0).fit(TR[feats], TR['soil_moisture'] - dm)
        pred_moist = moist_baseline + moist_shape.predict(VAL[feats])
        results['soil_moisture'].append(rmse(VAL['soil_moisture'].to_numpy(), pred_moist))

        # soil_ec: 고레짐 전문가 + 저레짐 persistence
        hi = TR[TR['regEC'] == 1]
        ec_high = F.fit_expert(hi, feats + F.MSC, 'soil_ec') if len(hi) >= 100 else None
        low_days = [d for d in sorted(TR['dat'].unique()) if regEC.get(d, 0) == 0]
        ec_low = float(TR[TR['dat'].isin(low_days[-3:])]['soil_ec'].mean()) if low_days else float(TR['soil_ec'].mean())

        pred_ec = np.full(len(VAL), ec_low)
        hi_mask = (VAL['regEC'] == 1).to_numpy()
        if hi_mask.any() and ec_high is not None:
            pred_ec[hi_mask] = F.pred_expert(ec_high, VAL[hi_mask], feats + F.MSC)
        results['soil_ec'].append(rmse(VAL['soil_ec'].to_numpy(), pred_ec))
    return results


def main():
    root = F.find_root()
    print(f'[train] DATA_ROOT = {os.path.abspath(root)}')
    Xtr, Xte, Y = F.load_env(root)
    common = [c for c in Xtr.columns if c in Xte.columns]
    allX, feats, lagf = F.build(pd.concat([Xtr[common], Xte[common]], ignore_index=True))
    reg = F.curtain_regime(allX)
    regEC = F.refined_regime(allX)

    Y['dat'] = Y['time'].map(F._dat); Y['min'] = Y['time'].map(F._min)

    # 다분광: 학습 일별 캐노피 요약
    dayMS = F.daily_ms(F.extract_ms(root, 'train'))
    dayMS = dayMS.reindex(range(int(Y.dat.min()), int(Y.dat.max()) + 1)).interpolate().ffill().bfill()

    train = allX[allX.dat <= int(Y.dat.max())].copy()
    allf = feats + lagf
    TR_raw = Y.merge(train[['dat', 'min'] + allf], on=['dat', 'min'], how='left')
    TR_raw['reg'] = TR_raw['dat'].map(reg); TR_raw['regEC'] = TR_raw['dat'].map(regEC)
    for c in F.MSC:
        TR_raw[c] = TR_raw['dat'].map(dayMS[c])

    # --- 4-fold 시계열 교차검증 (train.py와 동일 FOLD_CUTOFFS 기준) ---
    print("=== v2 (레짐 라우팅) 4-fold 시계열 교차검증 ===")
    fold_results = run_folds_v2(TR_raw, feats, lagf, regEC)
    for col in ['soil_moisture', 'soil_ec', 'soil_temp']:
        arr = np.array(fold_results[col])
        print(f'  {col}: folds={np.round(arr, 4)} mean={arr.mean():.4f} std={arr.std():.4f}')

    # --- 최종 제출용 모델은 전체 26일로 재학습 (fillna는 전체 median 사용) ---
    med = TR_raw[allf].median()
    TR = TR_raw.copy()
    TR[allf] = TR[allf].fillna(med)
    med_ms = TR[F.MSC].median()
    TR[F.MSC] = TR[F.MSC].fillna(med_ms)

    bundle = {'feats': feats, 'lagf': lagf, 'MSC': F.MSC,
              'med': med, 'med_ms': med_ms}

    # --- soil_temp : 커튼레짐별 전문가 (env+lag) ---
    temp_experts = {}
    for rv in sorted(TR['reg'].unique()):
        sub = TR[TR['reg'] == rv]
        if len(sub) >= 100:
            temp_experts[rv] = F.fit_expert(sub, feats + lagf, 'soil_temp')
    if not temp_experts:
        temp_experts[0] = F.fit_expert(TR, feats + lagf, 'soil_temp')
    bundle['temp_experts'] = temp_experts
    bundle['temp_default'] = sorted(temp_experts)[0]

    # --- soil_moisture : 최근3일 baseline + 일사구동 within-day shape ---
    last3 = sorted(TR['dat'].unique())[-3:]
    bundle['moist_baseline'] = float(Y[Y.dat.isin(last3)]['soil_moisture'].mean())
    dm = TR.groupby('dat')['soil_moisture'].transform('mean')
    bundle['moist_shape'] = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=0).fit(TR[feats], TR['soil_moisture'] - dm)

    # --- soil_ec : 강화레짐 고전문가(env+MS) + 저 persistence ---
    hi = TR[TR['regEC'] == 1]
    bundle['ec_high'] = F.fit_expert(hi, feats + F.MSC, 'soil_ec') if len(hi) >= 100 else None
    low_days = [d for d in sorted(TR['dat'].unique()) if regEC[d] == 0]
    bundle['ec_low'] = float(Y[Y.dat.isin(low_days[-3:])]['soil_ec'].mean())

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, os.path.join(MODEL_DIR, 'model.pkl'))
    print(f'[train] 저장 완료: {os.path.join(MODEL_DIR, "model.pkl")}')
    print(f'[train] temp 레짐={list(temp_experts)}, EC 고레짐 학습일={sorted(hi.dat.unique()) if len(hi) else "없음"}')
    print(f'[train] moisture baseline={bundle["moist_baseline"]:.3f}, EC 저 persistence={bundle["ec_low"]:.3f}')


if __name__ == '__main__':
    main()
