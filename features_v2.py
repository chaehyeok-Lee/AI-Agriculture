"""공통 모듈: 피처 생성 · 커튼 레짐 판별 · 다분광 큐브 추출 · 전문가 모델.

온라인테스트 1 (멀티모달 근권부 환경 예측) 파이프라인의 공통 로직.
train.py / inference.py 가 이 모듈을 공유하여 피처·전처리를 완전히 일치시킨다.
"""
from __future__ import annotations
import os, re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# ---- 컬럼 정의 ----
CONT = ['temperature', 'humidity', 'co2', 'temperature_outside',
        'humidity_outside', 'solar_radiation', 'wind_speed_outside']
ACT = ['greenhouse_roof_vent1', 'greenhouse_roof_vent2', 'shading_curtain',
       'thermal_curtain', 'fcu_fan', 'fcu_pump', 'circ_fan', 'co2_supply',
       'fogging', 'rainfall']
LAG_COLS = ['temperature', 'temperature_outside', 'humidity', 'co2',
            'solar_radiation', 'circ_fan']
MSC = ['ms_veg_frac', 'ms_bright', 'ms_ndre', 'ms_ndvi_re', 'ms_rep', 'ms_nir_slope']
TARGETS = ['soil_moisture', 'soil_ec', 'soil_temp']

# 다분광 큐브 사양 (ENVI BSQ, uint16, 10밴드 713~920nm)
WL = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920.])
B, L, S = 10, 1024, 1280


# ---- 데이터 위치 탐색 ----
def find_root():
    """환경변수 DATA_ROOT → input/ → dataset/ 순으로 데이터 루트를 찾는다."""
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.environ.get('DATA_ROOT'), 'input', 'dataset',
             os.path.join(here, '..', 'dataset'), os.path.join(here, 'input'), '.']
    for c in cands:
        if c and os.path.isdir(os.path.join(c, 'train', 'env')):
            return c
    raise FileNotFoundError('데이터 루트를 찾을 수 없음 (DATA_ROOT 환경변수로 지정하세요)')


def _find_csv(folder, *names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f'{folder} 에서 {names} 중 하나도 못 찾음')


def load_env(root):
    """train_X, test_X, train_y 를 읽어 반환. 파일명 변형(train_X_1.csv 등) 자동 탐색."""
    tre = os.path.join(root, 'train', 'env')
    tee = os.path.join(root, 'test', 'env')
    Xtr = pd.read_csv(_find_csv(tre, 'train_X_1.csv', 'train_X.csv'))
    Y = pd.read_csv(_find_csv(tre, 'train_y.csv'))
    Xte = pd.read_csv(_find_csv(tee, 'test_X.csv', 'test_X_1.csv'))
    return Xtr, Xte, Y


def _dat(s):  return int(s.split(' ')[0][3:])
def _min(s):  return int(s.split(' ')[1][:2]) * 60 + int(s.split(' ')[1][3:5])


# ---- 피처 생성 ----
def build(df):
    """env tabular → 피처. (feat=공통, lagf=temp 전용 lag/dewpoint) 반환.
    rolling/lag 은 시간정렬 후 계산하므로 train+test 를 이어붙여 넣으면 test 도 이력을 갖는다."""
    df = df.copy()
    df['dat'] = df['time'].map(_dat)
    df['min'] = df['time'].map(_min)
    df = df.sort_values(['dat', 'min']).reset_index(drop=True)
    feat = []
    df['time_sin'] = np.sin(2 * np.pi * df['min'] / 1440)
    df['time_cos'] = np.cos(2 * np.pi * df['min'] / 1440)
    feat += ['time_sin', 'time_cos']
    for c in CONT:
        for w in (60, 240, 1440):
            df[f'{c}_ma{w}'] = df[c].rolling(w, min_periods=1).mean()
        df[f'{c}_diff60'] = df[c] - df[c].shift(60)
        feat += [c, f'{c}_ma60', f'{c}_ma240', f'{c}_ma1440', f'{c}_diff60']
    for c in ACT:
        for w in (60, 240):
            df[f'{c}_ma{w}'] = df[c].rolling(w, min_periods=1).mean()
        feat += [c, f'{c}_ma60', f'{c}_ma240']
    df['rad_day_cum'] = df.groupby('dat')['solar_radiation'].cumsum()
    feat += ['rad_day_cum']
    # temp 전용: env lag(열관성) + dewpoint_gap(냉각). 상대값이라 외삽 안전.
    lagf = []
    for c in LAG_COLS:
        for Lg in (60, 180, 360, 720, 1440):
            df[f'{c}_lag{Lg}'] = df[c].shift(Lg)
            lagf.append(f'{c}_lag{Lg}')
    df['dewpoint_gap'] = df['temperature'] - df['temperature_outside']
    lagf.append('dewpoint_gap')
    for w in (60, 240, 720):
        df[f'dpgap_m{w}'] = df['dewpoint_gap'].rolling(w, min_periods=1).mean()
        lagf.append(f'dpgap_m{w}')
    return df, feat, lagf


# ---- 커튼 레짐 판별 ----
def curtain_regime(df):
    """기본 커튼 레짐(soil_temp 전문가 분리용)."""
    r = {}
    for d, g in df.groupby('dat'):
        th = g['thermal_curtain'].values; sh = g['shading_curtain'].values
        r[d] = int(((th >= 99) & (sh >= 99)).any() and ((th <= 1) & (sh <= 1)).any()
                   and np.mean(np.abs(th - sh) < 5) >= 0.90)
    return r


def _high_gate(g):
    """고EC 게이트: 보온·차광 둘다 완전 전개/개방 도달 AND 동기율 agreement>=0.98.
    (독립적으로 circ_fan OFF 와 동일한 날을 지목 — 이중 확인)."""
    th = g['thermal_curtain'].values; sh = g['shading_curtain'].values
    curt1 = ((th >= 99) & (sh >= 99)).any() and ((th <= 1) & (sh <= 1)).any()
    agr098 = np.mean(np.abs(th - sh) <= 0.5) >= 0.98
    return curt1 and agr098


def refined_regime(df):
    """강화 고EC 레짐: agreement>=0.98 AND 순환팬 OFF (두 독립신호 모두 요구)."""
    return {d: int(_high_gate(g) and g['circ_fan'].mean() < 15)
            for d, g in df.groupby('dat')}


def high_conf(df):
    """고EC 소프트 신뢰도[0,1]: 게이트 통과 + 팬 확실 OFF → 1, 애매하면 중간(폴백)."""
    return {d: (1.0 if _high_gate(g) else 0.0) * float(np.clip((20 - g['circ_fan'].mean()) / 15, 0, 1))
            for d, g in df.groupby('dat')}


# ---- 다분광 큐브 추출 ----
def _read_cube(folder, stride=4):
    raw = np.fromfile(os.path.join(folder, 'cube.raw'), dtype='<u2')
    return raw.reshape(B, L, S).astype(np.float32)[:, ::stride, ::stride]


def _cube_features(c):
    m = c.mean(0) + 1e-6
    n = c / m
    veg = (n[4] - n[0]) > 0.35
    out = {'veg_frac': float(veg.mean()), 'bright': float(c.mean())}
    if veg.sum() < 50:
        out.update({'ndre': np.nan, 'ndvi_re': np.nan, 'rep': np.nan, 'nir_slope': np.nan})
        return out
    rv = c.reshape(B, -1)[:, veg.ravel()].mean(1)
    out['ndre'] = float((rv[4] - rv[0]) / (rv[4] + rv[0] + 1e-6))
    out['ndvi_re'] = float((rv[9] - rv[0]) / (rv[9] + rv[0] + 1e-6))
    out['rep'] = float(WL[np.argmax(np.diff(rv))])
    out['nir_slope'] = float((rv[9] - rv[4]) / rv[4])
    return out


def extract_ms(root, split):
    """{root}/{split}/ms 의 모든 큐브 세션에서 캐노피 지표 추출 → DataFrame."""
    base = os.path.join(root, split, 'ms')
    rows = []
    if not os.path.isdir(base):
        return pd.DataFrame(columns=['dat', 'pos', 'minute'] + [c[3:] for c in MSC])
    for dat in sorted(os.listdir(base)):
        dd = os.path.join(base, dat)
        if not os.path.isdir(dd):
            continue
        for sess in sorted(os.listdir(dd)):
            mt = re.match(r'(\d+)_DAT(\d+)_(\d+)', sess)
            if not mt:
                continue
            t = mt.group(3)
            f = _cube_features(_read_cube(os.path.join(dd, sess)))
            f.update(dat=int(mt.group(2)), pos=int(mt.group(1)),
                     minute=int(t[:2]) * 60 + int(t[2:4]))
            rows.append(f)
    return pd.DataFrame(rows)


def daily_ms(msdf):
    """작물 위치(pos!=0) 일평균 캐노피 요약 → 컬럼 MSC."""
    if len(msdf) == 0:
        return pd.DataFrame(columns=MSC)
    crop = msdf[msdf['pos'] != 0]
    g = crop.groupby('dat')[['veg_frac', 'bright', 'ndre', 'ndvi_re', 'rep', 'nir_slope']].mean()
    g.columns = MSC
    return g


# ---- 전문가 모델(Ridge + HGB 블렌드) ----
def fit_expert(trX, feats, targ):
    """그룹 마지막날로 블렌드 가중치 보정 후 전체 refit. (sc, ridge, hgb, w) 반환."""
    days = sorted(trX['dat'].unique())
    if len(days) >= 3:
        cal = days[-1]; base = trX[trX.dat != cal]; cl = trX[trX.dat == cal]
        sc = StandardScaler().fit(base[feats])
        r = Ridge(alpha=5).fit(sc.transform(base[feats]), base[targ])
        h = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=0).fit(base[feats], base[targ])
        pr = r.predict(sc.transform(cl[feats])); ph = h.predict(cl[feats]); yt = cl[targ].values
        ws = np.linspace(0, 1, 11)
        w = ws[int(np.argmin([np.sqrt(np.mean((yt - ((1 - a) * pr + a * ph)) ** 2)) for a in ws]))]
    else:
        w = 0.5
    sc = StandardScaler().fit(trX[feats])
    r = Ridge(alpha=5).fit(sc.transform(trX[feats]), trX[targ])
    h = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                      l2_regularization=1.0, random_state=0).fit(trX[feats], trX[targ])
    return (sc, r, h, float(w))


def pred_expert(exp, Xdf, feats):
    sc, r, h, w = exp
    return (1 - w) * r.predict(sc.transform(Xdf[feats])) + w * h.predict(Xdf[feats])
