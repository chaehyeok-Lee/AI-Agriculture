"""
다분광 하이퍼큐브(ENVI) 특징 추출.

주의 — 이 문제의 구조적 제약:
  · 제출은 5분 격자 전체(야간 포함, 하루 288행)를 채워야 한다.
  · 그런데 영상은 낮 12~17시에만 존재하고, 5분 격자의 4%에만 대응된다.
  => 따라서 분광을 '매 시점 필수 입력'으로 쓸 수 없다.
     하루·그룹 단위로 요약한 '그날 그 그룹의 캐노피 상태'로 만들어 브로드캐스트한다.

전처리는 밴드축 1차 미분(DN_D1)을 주 특징으로 사용한다. 세션 간 조도 차이로 인한
기준선 오프셋을 억제하고 밴드 간 변화율을 강조하는 효과가 있으며, 교차검증에서
가장 안정적인 특징 공간을 주었다. 회색 기준판 정규화는 보조 특징으로만 소량 사용한다.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

BANDS = 10
LINES, SAMPLES = 1024, 1280
WAVELENGTH = np.array([713, 736, 759, 782, 805, 828, 851, 874, 897, 920], dtype=np.float64)
STRIDE = 4          # 공간 다운샘플 (속도)
DARK_PCT = 0.5      # 밴드별 암전류 오프셋 추정 분위수
VEG_NDVI = 0.25     # 식생 픽셀 임계
PANEL_PCT = 98      # 기준판 후보: 713nm 상위 밝기 분위수


def _read_cube(session_dir: str) -> np.ndarray | None:
    raw = os.path.join(session_dir, "cube.raw")
    if not os.path.exists(raw):
        return None
    a = np.fromfile(raw, dtype=np.uint16)
    if a.size != BANDS * LINES * SAMPLES:
        return None
    return a.reshape(BANDS, LINES, SAMPLES)[:, ::STRIDE, ::STRIDE].astype(np.float32)


def session_features(session_dir: str) -> dict | None:
    cube = _read_cube(session_dir)
    if cube is None:
        return None
    # 암전류 보정: 보정하지 않으면 저조도(늦은 오후) 큐브에서 지수가 붕괴한다
    dark = np.percentile(cube, DARK_PCT, axis=(1, 2)).reshape(BANDS, 1, 1)
    c = np.clip(cube - dark, 0.0, None)

    b713, b805 = c[0], c[4]
    ndvi = (b805 - b713) / (b805 + b713 + 1e-6)
    veg = ndvi > VEG_NDVI
    if veg.sum() < 50:
        return None
    bright = c[0] > np.percentile(c[0], PANEL_PCT)
    panel = bright & (ndvi < 0.10)          # 밝고 분광이 평평 = 회색 기준판

    f = {"veg_frac": float(veg.mean()), "ndvi_veg": float(ndvi[veg].mean())}
    dn = np.array([float(c[b][veg].mean()) for b in range(BANDS)])
    for b in range(BANDS):
        f[f"dn{b}"] = dn[b]
    # DN_D1: 파장축 1차 미분 (주 전처리 조건)
    d1 = np.diff(dn) / np.diff(WAVELENGTH)
    for b in range(BANDS - 1):
        f[f"d1_{b}"] = float(d1[b])
    # 기준판 정규화 상대반사율 (보조)
    if panel.sum() >= 50:
        pan = np.array([float(c[b][panel].mean()) for b in range(BANDS)])
        ref = dn / np.maximum(pan, 1e-6)
        f["ref_re"] = float((ref[4] - ref[0]) / (ref[4] + ref[0] + 1e-6))
        f["ref_w920"] = float((ref[4] - ref[9]) / (ref[4] + ref[9] + 1e-6))
    else:
        f["ref_re"] = np.nan
        f["ref_w920"] = np.nan
    return f


def extract_split(ms_root: str) -> pd.DataFrame:
    """ms_root = .../dataset/{train|test}/ms  ->  세션별 특징 테이블"""
    rows = []
    for sess in sorted(glob.glob(os.path.join(ms_root, "DAT*", "*_DAT*_*"))):
        name = os.path.basename(sess)
        m = re.match(r"(\d+)_DAT(\d+)_(\d{6})", name)
        if not m:
            continue
        f = session_features(sess)
        if f is None:
            continue
        f["pos"] = int(m.group(1))
        f["dat"] = int(m.group(2))
        f["hhmmss"] = m.group(3)
        f["mod"] = int(m.group(3)[:2]) * 60 + int(m.group(3)[2:4])
        rows.append(f)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


FEATCOLS = (["veg_frac", "ndvi_veg"] + [f"dn{b}" for b in range(BANDS)]
            + [f"d1_{b}" for b in range(BANDS - 1)] + ["ref_re", "ref_w920"])


def daily_summary(sess: pd.DataFrame) -> pd.DataFrame:
    """세션 -> (날짜) 단위 캐노피 상태 요약. 5분 격자 전체에 브로드캐스트할 값."""
    if sess.empty:
        return pd.DataFrame()
    g = sess.groupby("dat")[FEATCOLS].mean()
    g.columns = ["ms_" + c for c in g.columns]
    g["ms_n_sessions"] = sess.groupby("dat").size()
    return g.reset_index()
