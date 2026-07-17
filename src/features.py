"""
5분 격자 특징 생성.

■ 설계 원칙
  · 각 날(DAT)을 독립 단위로 다룬다. 날짜 간 순서나 자정을 넘는 이력을 만들지 않는다.
  · test 를 train 뒤에 이어 붙이지 않는다.
  · 오직 train 에 나타나는 4개 재배 그룹(zone) 패턴만 test 로 전이한다:
      - 그룹 원핫
      - 그룹 × 하루 중 시각 기후값(target encoding)
      - 그날 그 그룹의 운영 요약(제어 이력 가동률·내부 기후)
      - 당일(자정 리셋) 이동평균/누적 이력
      - 다분광 캐노피 요약 (함수율에만)

  모든 이력은 DAT(하루) 경계에서 리셋된다. 서로 다른 그룹·서로 다른 날의
  데이터가 섞이지 않으며, test 의 어떤 값도 train 특징에 흘러들지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENV_NUM = ["temperature_outside", "humidity_outside", "solar_radiation",
           "wind_direction_outside", "wind_speed_outside", "rainfall",
           "greenhouse_roof_vent1", "greenhouse_roof_vent2", "shading_curtain",
           "thermal_curtain", "fcu_fan", "fcu_pump", "circ_fan", "co2_supply",
           "fogging", "temperature", "humidity", "co2"]

# 당일 안에서만 굴리는 이동평균 시정수 (5분 단위): 30m 1h 2h 3h 4h 6h
MA_WIN = [6, 12, 24, 36, 48, 72]
ROLL_VARS = ["temperature", "temperature_outside", "humidity", "solar_radiation",
             "vpd", "fcu_fan", "co2"]


def _vpd(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """포화수증기압차 (kPa). 증산 수요의 물리적 대리변수."""
    es = 0.6108 * np.exp(17.27 * t_c / (t_c + 237.3))
    return es * (1.0 - np.clip(rh, 0, 100) / 100.0)


def to_5min(env: pd.DataFrame) -> pd.DataFrame:
    e = env.copy()
    e["mod5"] = (e["mod"] // 5) * 5
    g = e.groupby(["dat", "mod5"], as_index=False)[ENV_NUM].mean()
    return g.rename(columns={"mod5": "mod"})


def build(env: pd.DataFrame, structure: pd.DataFrame, ms_daily: pd.DataFrame | None,
          ms_fill: pd.Series | None, n_zones: int) -> pd.DataFrame:
    """5분 격자 특징. env 는 parse_time 이 적용된 1분 원본 (train 또는 test)."""
    df = to_5min(env)
    st = structure.set_index("dat")
    df["zone"] = df["dat"].map(st["zone"]).astype(int)
    df = df.sort_values(["dat", "mod"]).reset_index(drop=True)

    # real_day(구조.discover_calendar()가 외부기상 경계연속성만으로 도출한, 하드코딩
    # 없는 실제 경과일)가 있으면 피처로 붙인다 — 26.07.17 검증: leave-one-real-day-out
    # CV에서 soil_moisture는 real_day(+zone 상호작용)로 -2.9%(4/6 fold 일관 개선),
    # soil_ec는 오히려 +6.0% 악화(특히 학습 불가능한 관리이벤트 fold에서 더 나빠짐) ->
    # select()에서 soil_ec/soil_temp는 이 컬럼들을 제외한다.
    if "real_day" in st.columns:
        df["real_day"] = df["dat"].map(st["real_day"]).astype(float)
        for z in range(n_zones):
            df[f"real_day_zone{z}"] = df["real_day"] * (df["zone"].to_numpy() == z)

    # ── 물리 파생
    df["vpd"] = _vpd(df["temperature"].to_numpy(), df["humidity"].to_numpy())
    df["vpd_out"] = _vpd(df["temperature_outside"].to_numpy(),
                         df["humidity_outside"].to_numpy())
    df["dtemp_in_out"] = df["temperature"] - df["temperature_outside"]
    df["vent_total"] = df["greenhouse_roof_vent1"] + df["greenhouse_roof_vent2"]

    new = {}
    # ── 당일(자정 리셋) 이력: 이동평균 + 편차 + 당일 누적
    gday = df.groupby("dat", sort=False)
    for v in ROLL_VARS:
        for w in MA_WIN:
            new[f"{v}_ma{w}"] = gday[v].transform(
                lambda s, w=w: s.rolling(w, min_periods=1).mean()).to_numpy()
        new[f"{v}_d24"] = df[v].to_numpy() - new[f"{v}_ma24"]
    for v in ["solar_radiation", "vpd"]:
        new[f"{v}_cum"] = gday[v].transform(lambda s: s.expanding().mean()).to_numpy()
    new["solar_cumsum"] = gday["solar_radiation"].transform(
        lambda s: s.cumsum()).to_numpy() * 5 / 60.0
    new["fcu_duty_cum"] = gday["fcu_fan"].transform(
        lambda s: (s > 0).expanding().mean()).to_numpy()

    # ── 시각 (하루 중 시각, 일주기)
    tod = df["mod"].to_numpy() / 1440.0 * 2 * np.pi
    for k in (1, 2, 3):
        new[f"tod_sin{k}"] = np.sin(k * tod)
        new[f"tod_cos{k}"] = np.cos(k * tod)
    new["mod_frac"] = df["mod"].to_numpy() / 1440.0

    # ── 그룹 원핫
    for z in range(n_zones):
        new[f"zone_{z}"] = (df["zone"].to_numpy() == z).astype(np.int8)

    df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    # ── 그날 그 그룹의 운영 요약 (그룹 지문을 모델이 직접 볼 수 있게)
    day = env.groupby("dat")
    add = {}
    for c in ["circ_fan", "co2_supply", "fcu_fan", "greenhouse_roof_vent1",
              "shading_curtain", "thermal_curtain", "fogging"]:
        duty = day[c].apply(lambda s: float((s > 0).mean() * 100))
        add[f"day_{c}_duty"] = df["dat"].map(duty).to_numpy()
    for c in ["temperature", "humidity", "co2"]:
        add[f"day_{c}_mean"] = df["dat"].map(day[c].mean()).to_numpy()
    df = pd.concat([df, pd.DataFrame(add, index=df.index)], axis=1)

    # ── 다분광: 하루·그룹 단위 캐노피 상태를 전 시각에 브로드캐스트
    if ms_daily is not None and not ms_daily.empty:
        ms_cols = [c for c in ms_daily.columns if c != "dat"]
        df = df.merge(ms_daily, on="dat", how="left")
        df["ms_available"] = (~df[ms_cols[0]].isna()).astype(np.int8)
        for c in ms_cols:
            fill = float(ms_fill[c]) if (ms_fill is not None and c in ms_fill.index) \
                else float(df[c].median())
            df[c] = df[c].fillna(fill)

    df = df.replace([np.inf, -np.inf], np.nan)
    return df.sort_values(["dat", "mod"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"dat", "mod", "zone"}
    return [c for c in df.columns if c not in drop and df[c].dtype != object]


# ── 그룹 × 하루 중 시각 기후값 (target encoding) ─────────────────────────
# 근권부 상태의 일중 곡선이 그룹마다 다르므로,
# 그 곡선을 학습 라벨에서 추정해 특징으로 준다.
SMOOTH = 20.0


def fit_climatology(d: pd.DataFrame, targets: list[str]) -> dict:
    enc = {}
    for t in targets:
        g = d.groupby(["zone", "mod"])[t].agg(["mean", "count"])
        gm = float(d[t].mean())
        enc[t] = {"table": ((g["mean"] * g["count"] + gm * SMOOTH)
                            / (g["count"] + SMOOTH)).to_dict(),
                  "global": gm}
    return enc


def apply_climatology(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    keys = list(zip(df["zone"].to_numpy(), df["mod"].to_numpy()))
    add = {}
    for t, e in enc.items():
        tab, gm = e["table"], e["global"]
        add[f"clim_{t}"] = np.array([tab.get(k, gm) for k in keys], dtype=np.float64)
    return pd.concat([df, pd.DataFrame(add, index=df.index)], axis=1)


def select(cols: list[str], target: str) -> list[str]:
    """타깃별 특징 집합. 교차검증으로 실측:
    - 분광(ms_*)은 함수율에만 도움되고 EC·배지온도에는 잡음으로 작용해 제외.
    - real_day(+구획 상호작용)는 soil_moisture만 개선(-2.9%), soil_ec는 악화(+6.0%,
      특히 학습 불가능한 관리이벤트 fold에서 더 나빠짐) -> soil_ec에서 제외."""
    if target == "soil_moisture":
        return list(cols)
    filtered = [c for c in cols if not c.startswith("ms_")]
    if target == "soil_ec":
        filtered = [c for c in filtered if c != "real_day" and not c.startswith("real_day_zone")]
    return filtered
