"""
배지온도(soil_temp) 전용 모델.

배지온도는 공기온도의 지연·평활 응답(1차 저역통과)이라는 물리적 성질이 강해,
함수율·EC 와 달리 '온도 자체의 lag/rolling/EWMA + 시각 주기성'만으로 잘 설명된다.
교차검증 결과 이 타깃에서는 다음이 유효했다:
  · 분광·구동기·밀도 등은 잡음(제외).
  · 선형(Ridge)이 트리보다 강하다 → LGBM+Ridge 블렌드.
  · EWMA(지수가중이동평균)를 특징이 아니라 별도 서브모델로 앙상블하면 더 낫다
    (배치 효과 + 물리적으로 배지 열관성과 같은 형태).

■ 이력 계산 범위 (26.07.16 청크 구조 반영, 2차 갱신 — 스캐폴드+양방향)
  DAT 숫자는 실제 시간순이 아니다 — 같은 온실을 4개 구획(zone)이 나눠 쓰고, 각 구획은
  서로 다른 DAT 번호 대역을 쓰며, 그 안에서도 일부 실제일이 빠져 있다(evidence/ 근거,
  src/structure.discover_calendar 참고).

  1차 수정(zone·real_day 순 정렬 + 구간 리셋)은 정확했지만, 청크1/2/4는 train만 놓고 보면
  "1~5일차 + 10일차 혼자" 식으로 진짜 연속 구간이 너무 짧아져 오히려 성능이 나빠졌다.
  그래서 2차로: zone별로 비어있는 real_day를, **그 zone 자신의 다른 날 중 외부기상이
  가장 비슷한 날의 반응 패턴(analog)** 을 빌리고 앞뒤 실측 경계값에 맞춰 레벨을
  보정(bridge)한 합성 행으로 채워 넣어, zone마다 진짜 11일치 연속 사슬(scaffold)을
  만든다. 합성 행은 soil_temp 정답이 없으므로 **학습·예측 어디에도 쓰지 않고**,
  오직 그 앞뒤 실측 행들의 lag/rolling/EWMA가 끊기지 않게 이어주는 다리 역할만 한다
  (build_full_scaffold 의 source 컬럼으로 real/synthetic 구분).

  이 문제는 애초에 "미래 예측"이 아니라 "같은 시각 t의 배지온도를 입력 [t-30,t]로
  추정하는" 소프트센서 문제이고(evidence/README 참고), test(빨간) 날은 진짜 달력에서
  train 날짜 "사이사이"에 낀 보간(interpolation) 위치다. 그래서 test 행 예측에는
  과거(lag) 방향뿐 아니라 **미래(lead) 방향** 특징도 쓴다 — 이건 정답(soil_temp)이
  아니라 test_X에 이미 제공된 온도 입력값(t, to)을 앞뒤 양쪽에서 참조하는 것이라
  누수가 아니다.

  train/test/synthetic 전부를 zone·real_day 순으로 한 번에 이어붙여 이력을 계산한 뒤,
  synthetic 행은 버리고 train 행만 학습에, test 행만 예측에 쓴다(외부 데이터·정답·
  상수 하드코딩 없음).

최종 예측 = 0.8 * (0.5·LGBM + 0.5·Ridge)          [TREE 특징]
          + 0.2 * (0.5·Ridge + 0.5·HGB)           [EWMA 특징]

■ 26.07.17 3차 갱신 — 청크별 확장창 CV로 재튜닝 + k=3 analog + lead-EWMA
  검증 방식을 "DAT 무작위 GroupKFold"에서 "zone(청크)별 real_day 순서 확장창"으로
  바꿨다: 각 zone 자신의 train-only real_day를 오름차순 정렬해 앞쪽 N일로 학습,
  바로 다음 날 하나로 검증(N=3,4,5). real_day 7·8을 가진(=gap 없는) zone만 N+2를
  써서 다른 zone들과 캘린더 진행 정도를 맞춘다(train.py의 expanding_chunk_folds).
  이 CV로 재확인한 결과:
    - LightGBM 정규화 재탐색(num_leaves 8→4, feature_fraction 0.8→0.6,
      reg_lambda 3.0→0.5, min_child_samples 5→10) + 블렌드 비중(0.7→0.8) 개선
    - 스캐폴드를 "가장 비슷한 날 1개"에서 **k=3개를 거리 역수 가중평균**으로 개선
      (조금 더 매끄러운 합성 구간)
    - EWMA에도 **lead(미래) 반감기 버전**(t_ewlead*, to_ewlead*) 추가 — 시간축을
      뒤집어서(reverse) ewm()을 적용한 뒤 다시 뒤집는 방식
    - 양방향 rollstd(t_crollstd288, 중심 rolling std)는 시도했으나 대폭 악화되어 폐기
      (변동성 지표는 미래를 섞으면 오히려 잡음이 커지는 것으로 추정 — 평균/레벨과
      달리 표준편차는 국소적 요동이라 대칭 윈도가 실제 물리적 의미를 왜곡하는 듯)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.signal import savgol_filter
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# 5분 격자. lag/rolling 창은 5분 단위 (72=6h, 144=12h, 288=1일). lead=미래 방향 대칭 피처.
TREE_COLS = ["t", "to", "hour_sin", "hour_cos",
             "t_lag72", "t_lag288", "t_lead72", "t_lead288",
             "t_roll144", "t_roll288", "t_croll288", "t_rollstd288",
             "t_diff12", "t_diff36", "t_accel", "to_lag36", "to_lead36"]
EW_HL_T = [12, 24, 48, 96, 144, 288, 576]      # 내부온도 EWMA 반감기(5분 단위)
EW_HL_TO = [24, 96, 288]                        # 외부온도 EWMA 반감기
EW_COLS = ([f"t_ew{h}" for h in EW_HL_T] + [f"to_ew{h}" for h in EW_HL_TO]
           + [f"t_ewlead{h}" for h in EW_HL_T] + [f"to_ewlead{h}" for h in EW_HL_TO]
           + ["hour_sin", "hour_cos"])

W_MAIN = 0.80          # (LGBM+Ridge 블렌드) 비중 — 26.07.17 청크별 확장창 CV 재탐색
W_EW = 0.20            # (EWMA 서브모델) 비중
LGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, num_leaves=4,
                  feature_fraction=0.6, reg_lambda=0.5, min_child_samples=10,
                  random_state=42, verbosity=-1)
SCAFFOLD_K = 3          # 빈 real_day를 채울 때 참고할 유사일 개수(거리 역수 가중평균)

EXT_WEATHER_COLS = ["temperature_outside", "humidity_outside", "solar_radiation",
                    "wind_direction_outside", "wind_speed_outside", "rainfall"]

# ── 26.07.17 4차 갱신 — 재귀(recurrence) + 경계 앵커링(anchoring) ──────────────
# 배지온도는 "같은 시각을 입력으로 추정"하는 소프트센서 관점 외에, 열관성 때문에
# 자기 자신의 직전 값과 강하게 자기상관되는 물리량이기도 하다. 기존 TempEnsemble은
# t/to의 lag·lead·EWMA(=입력 온도의 이력)만 쓰고 soil_temp 자기 자신의 이전 값은
# 피처로 쓴 적이 없었다 — 외부(Downloads의 별도 soil_temp 패키지) 비교에서 이
# 자기회귀 성분과 사후 경계보정이 실제 RMSE 개선의 핵심 요인으로 확인되어 반영한다.
DYNAMIC_COLS = ["t", "to", "t_ew12", "hour_sin", "hour_cos"]
DYNAMIC_RIDGE_ALPHA = 0.1
# (재귀 성분) 비중 — 확장창 CV(fold별 한 번만 적합, w만 재조합)로 0.0~1.0 grid search.
# w=0.9 + 경계앵커링에서 mean RMSE 0.5430(기존) -> 0.2396 로 최대 개선.
W_DYNAMIC = 0.9

PREVIOUS_WINDOW = 136          # 이전 real_day 경계값의 보정이 감쇠하는 길이(5분×136≈11.3h)
PREVIOUS_DECAY_POWER = 3.5
FOLLOWING_RAMP_POWER = 0.9
SMOOTHING_WINDOW = 21


class ZoneThermalDynamics:
    """zone별 '직전 시점 soil_temp + 당시 환경'으로 다음 시점 soil_temp을 잇는 Ridge.
    DAT(하루) 경계를 절대 넘지 않는다 — 매일 새로 초기값(initial)에서부터 재귀한다."""

    def __init__(self):
        self.models: dict[int, Ridge] = {}

    def fit(self, frame: pd.DataFrame):
        """frame: zone, dat, mod, soil_temp, DYNAMIC_COLS 를 가진 train 전용(실측) 행."""
        work = frame.sort_values(["dat", "mod"]).copy()
        work["soil_temp_prev"] = work.groupby("dat", sort=False)["soil_temp"].shift(1)
        cols = ["soil_temp_prev"] + DYNAMIC_COLS
        for zone, g in work.groupby("zone"):
            sub = g.dropna(subset=["soil_temp_prev"])
            if len(sub) >= 50:
                self.models[int(zone)] = Ridge(alpha=DYNAMIC_RIDGE_ALPHA).fit(
                    sub[cols], sub["soil_temp"])
        return self

    def predict_day(self, day_dyn: pd.DataFrame, initial: float, zone: int) -> np.ndarray:
        """day_dyn: mod 오름차순으로 정렬된 DYNAMIC_COLS만 담은 하루치 프레임.
        initial(첫 시점 값)에서부터 하루 안에서만 재귀한다."""
        n = len(day_dyn)
        result = np.empty(n, dtype=float)
        result[0] = float(initial)
        model = self.models.get(int(zone))
        if model is None or n <= 1:
            result[:] = float(initial)
            return result
        env = day_dyn[DYNAMIC_COLS].to_numpy(float)
        intercept, coef_prev, coef_env = model.intercept_, model.coef_[0], model.coef_[1:]
        previous = float(initial)
        for i in range(1, n):
            previous = float(intercept + coef_prev * previous + np.dot(coef_env, env[i]))
            result[i] = previous
        return result


def fit_temp_dynamics(feat: pd.DataFrame, y: pd.Series) -> ZoneThermalDynamics:
    combined = feat[["zone", "dat", "mod"] + DYNAMIC_COLS].copy()
    combined["soil_temp"] = y.to_numpy(float)
    return ZoneThermalDynamics().fit(combined)


def predict_temp_blended(ensemble: "TempEnsemble", dynamics: ZoneThermalDynamics,
                          feat: pd.DataFrame, w_dynamic: float = W_DYNAMIC) -> np.ndarray:
    """같은 시점(TempEnsemble) 예측과, 그 날의 첫 값에서 시작하는 재귀(dynamics) 예측을
    day(=dat)별로 블렌드한다. feat는 zone/dat/mod/DYNAMIC_COLS/TREE_COLS/EW_COLS를 갖고
    있어야 하며, 반환값은 feat와 같은 순서(0..n-1 위치)로 정렬된 배열이다."""
    feat = feat.reset_index(drop=True)
    base = ensemble.predict(feat)
    out = base.copy()
    for dat, sub in feat.groupby("dat", sort=False):
        order = np.argsort(sub["mod"].to_numpy())
        positions = sub.index.to_numpy()[order]
        zone = int(sub["zone"].iloc[0])
        day_base = base[positions]
        day_dyn = sub.iloc[order][DYNAMIC_COLS]
        dynamic_curve = dynamics.predict_day(day_dyn, float(day_base[0]), zone)
        out[positions] = w_dynamic * dynamic_curve + (1.0 - w_dynamic) * day_base
    return out


def _anchor_curve(prediction: np.ndarray, previous: float | None, following: float | None) -> np.ndarray:
    """실제 관측된 이웃 real_day 경계값 쪽으로 하루치 곡선을 당겨 붙인다(이전=감쇠하는
    보정, 다음=하루 전체에 걸친 램프), 끝점은 보존한 채 평활화한다."""
    result = np.asarray(prediction, dtype=float).copy()
    if previous is None and following is None:
        return result
    correction = np.zeros(len(result), dtype=float)
    if previous is not None:
        window = min(PREVIOUS_WINDOW, len(result))
        progress = np.linspace(0.0, 1.0, window)
        correction[:window] += (float(previous) - result[0]) * np.power(1.0 - progress, PREVIOUS_DECAY_POWER)
    if following is not None:
        progress = np.linspace(0.0, 1.0, len(result))
        correction += (float(following) - result[-1]) * np.power(progress, FOLLOWING_RAMP_POWER)
    result += correction
    if len(result) >= SMOOTHING_WINDOW:
        first, last = float(result[0]), float(result[-1])
        result = savgol_filter(result, window_length=SMOOTHING_WINDOW, polyorder=2, mode="interp")
        result[0], result[-1] = first, last
    return result


def correct_boundary(train_truth: pd.DataFrame, pred_frame: pd.DataFrame,
                      prediction: np.ndarray, exclude_dats: set) -> np.ndarray:
    """pred_frame(zone/real_day/dat/mod, prediction과 같은 순서)의 각 day를, train_truth
    (zone/real_day/dat/mod/soil_temp 실측 전체)에서 exclude_dats를 뺀 나머지로부터 찾은
    바로 이전/다음 real_day의 실측 경계값에 앵커링한다. CV에서는 exclude_dats에 현재
    검증 중인 dat들을 넣어 자기 자신(또는 같은 폴드로 묶인 미래 정보)을 앵커로 쓰지
    않게 하고, 최종 test 예측에서는 exclude_dats=set()로 train 전체를 앵커로 쓴다."""
    pred_frame = pred_frame.reset_index(drop=True)
    avail = train_truth[~train_truth["dat"].isin(exclude_dats)]
    avail_sorted = avail.sort_values(["zone", "real_day", "mod"])
    grouped = avail_sorted.groupby(["zone", "real_day"])
    boundary_first = grouped["soil_temp"].first()
    boundary_last = grouped["soil_temp"].last()

    corrected = np.asarray(prediction, dtype=float).copy()
    for (zone, rd), idx in pred_frame.groupby(["zone", "real_day"]).groups.items():
        order = np.argsort(pred_frame.loc[idx, "mod"].to_numpy())
        idx_sorted = idx.to_numpy()[order]
        previous = boundary_last.get((zone, rd - 1))
        following = boundary_first.get((zone, rd + 1))
        corrected[idx_sorted] = _anchor_curve(corrected[idx_sorted], previous, following)
    return corrected


# ──────────────────────────────────────────────── 스캐폴드(빈 real_day 채우기)
def _profile(env_all: pd.DataFrame, dat: int) -> pd.DataFrame:
    return env_all[env_all["dat"] == dat].sort_values("mod").reset_index(drop=True)


def _most_complete_dat(env_all: pd.DataFrame, dats) -> int:
    return int(max(dats, key=lambda d: int((env_all["dat"] == d).sum())))


def _find_k_analog_real_days(env_all: pd.DataFrame, st: pd.DataFrame, zone, target_real_day,
                              k: int = SCAFFOLD_K):
    """target_real_day의 외부기상과 가장 비슷한, 이 zone이 '실제로 가진' 다른 real_day
    k개를 거리 역수 가중치와 함께 반환 (단일 최근접 1개보다 26.07.17 재검증 결과 더 나음)."""
    have_days = sorted(st[st["zone"] == zone]["real_day"].unique())
    ref_dat = _most_complete_dat(env_all, st[st["real_day"] == target_real_day]["dat"].tolist())
    target_vec = _profile(env_all, ref_dat)[EXT_WEATHER_COLS].mean()

    rows = {}
    for rd in have_days:
        d = int(st[(st["zone"] == zone) & (st["real_day"] == rd)]["dat"].iloc[0])
        rows[rd] = _profile(env_all, d)[EXT_WEATHER_COLS].mean()
    mat = pd.DataFrame(rows).T
    mu, sd = mat.mean(), mat.std().replace(0, 1)
    z, tz = (mat - mu) / sd, (target_vec - mu) / sd
    dist = np.sqrt(((z - tz) ** 2).sum(axis=1)).sort_values()
    k = min(k, len(dist))
    nearest = dist.index[:k].tolist()
    w = 1.0 / (dist.iloc[:k].to_numpy() + 1e-6)
    return nearest, w / w.sum()


def _gap_runs(have_days, all_days):
    """전체 real_day 목록 중 이 zone에 없는 날들을 연속 구간(run)으로 묶어 리스트로 반환."""
    have = set(have_days)
    missing = sorted(set(all_days) - have)
    runs, cur = [], []
    for rd in missing:
        if cur and rd == cur[-1] + 1:
            cur.append(rd)
        else:
            if cur:
                runs.append(cur)
            cur = [rd]
    if cur:
        runs.append(cur)
    return runs


def _profile_5min(env_all: pd.DataFrame, dat: int) -> pd.DataFrame:
    """_profile()과 동일하지만 'real' 분기와 해상도를 맞추기 위해 5분 격자로 집계
    (원본 1분 env를 그대로 쓰면 합성 구간만 1분 해상도가 되어 lag/rolling 창 크기가
    어긋나는 버그가 생김 — 26.07.16 발견/수정)."""
    g = _profile(env_all, dat)
    g = g.copy()
    g["mod5"] = (g["mod"] // 5) * 5
    return (g.groupby("mod5", as_index=False)
            .agg(t=("temperature", "mean"), to=("temperature_outside", "mean"))
            .rename(columns={"mod5": "mod"}))


def _synthetic_run(env_all: pd.DataFrame, st: pd.DataFrame, zone, run: list[int]) -> pd.DataFrame:
    """run(예: [7,8])만큼 연속으로 빈 real_day들을, zone 자신의 analog(유사일 k개 거리
    역수 가중평균) 반응 패턴으로 채우고 앞뒤 실측 경계값에 맞춰 레벨을 보정(bridge)한다.
    양끝(첫/마지막 real_day)에 걸리면 보정 기준점이 없어 호출 전에 걸러진다
    (build_full_scaffold 참고). 'real' 분기와 동일한 5분 격자로 만든다."""
    parts = []
    for rd in run:
        analog_days, weights = _find_k_analog_real_days(env_all, st, zone, rd)
        weather_dat = _most_complete_dat(env_all, st[st["real_day"] == rd]["dat"].tolist())
        w_g = _profile_5min(env_all, weather_dat)

        n = len(w_g)
        analog_series = []
        for ard in analog_days:
            adat = int(st[(st["zone"] == zone) & (st["real_day"] == ard)]["dat"].iloc[0])
            a_g = _profile_5min(env_all, adat)
            n = min(n, len(a_g))
            analog_series.append(a_g["t"].to_numpy())
        analog_series = [s[:n] for s in analog_series]
        blended_t = sum(s * wi for s, wi in zip(analog_series, weights))

        part = pd.DataFrame({
            "mod": w_g["mod"].iloc[:n].to_numpy(),
            "to": w_g["to"].iloc[:n].to_numpy(),
            "t": blended_t,
        })
        part["real_day"] = rd
        parts.append(part)
    chain = pd.concat(parts, ignore_index=True)

    prev_dat = st[(st["zone"] == zone) & (st["real_day"] == run[0] - 1)]["dat"]
    next_dat = st[(st["zone"] == zone) & (st["real_day"] == run[-1] + 1)]["dat"]
    anchor_start_t = _profile_5min(env_all, int(prev_dat.iloc[0]))["t"].iloc[-1]
    anchor_end_t = _profile_5min(env_all, int(next_dat.iloc[0]))["t"].iloc[0]
    ramp = np.linspace(anchor_start_t - chain["t"].iloc[0], anchor_end_t - chain["t"].iloc[-1], len(chain))
    chain["t"] = chain["t"] + ramp  # to(외부기온)는 실측이라 보정 없이 그대로 둠

    chain["zone"] = zone
    chain["source"] = "synthetic"
    return chain


def build_full_scaffold(train_env: pd.DataFrame, test_env: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """train_env/test_env(parse_time 적용된 원본 1분 env) + st(zone/real_day 구조)로
    zone마다 real_day 1..max가 전부 채워진 완전한 (t,to) 5분 스캐폴드를 만든다.
    없는 (zone,real_day)는 analog+경계보정으로 채우되 source='synthetic'으로 표시 —
    soil_temp 학습·예측 어디에도 쓰이지 않고 오직 이력 연속성 확보용."""
    env_all = pd.concat([train_env, test_env], ignore_index=True)
    tr_dats = set(train_env["dat"].unique())

    e = env_all.copy()
    e["mod5"] = (e["mod"] // 5) * 5
    real = (e.groupby(["dat", "mod5"], as_index=False)
            .agg(t=("temperature", "mean"), to=("temperature_outside", "mean")))
    real = real.rename(columns={"mod5": "mod"})
    real = real.merge(st[["dat", "zone", "real_day"]].drop_duplicates("dat"), on="dat", how="left")
    real["source"] = real["dat"].map(lambda d: "train" if d in tr_dats else "test")

    all_real_days = sorted(st["real_day"].unique())
    synth_frames = []
    for zone, g in st.groupby("zone"):
        zone_days = sorted(g["real_day"].unique())
        for run in _gap_runs(zone_days, all_real_days):
            if run[0] == all_real_days[0] or run[-1] == all_real_days[-1]:
                continue  # 양끝(경계 보정 기준점) 없는 구간은 건너뜀
            synth_frames.append(_synthetic_run(env_all, st, zone, run))

    parts = [real]
    if synth_frames:
        parts.append(pd.concat(synth_frames, ignore_index=True, sort=False))
    full = pd.concat(parts, ignore_index=True, sort=False)
    return full.sort_values(["zone", "real_day", "mod"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────── 이력 특징(양방향)
def _history_features(gr: pd.DataFrame) -> pd.DataFrame:
    """zone 전체(스캐폴드로 완전히 이어진 11일 사슬) 하나에 대해서만 호출.
    더 이상 구간이 끊기지 않으므로 lag(과거)뿐 아니라 lead(미래) 방향도 안전하게 계산된다."""
    gr = gr.copy()
    T, TO = gr["t"], gr["to"]
    for lag in (72, 288):
        gr[f"t_lag{lag}"] = T.shift(lag)
        gr[f"t_lead{lag}"] = T.shift(-lag)
    gr["t_roll144"] = T.rolling(144, min_periods=1).mean()
    gr["t_roll288"] = T.rolling(288, min_periods=1).mean()
    gr["t_croll288"] = T.rolling(577, min_periods=1, center=True).mean()
    gr["t_rollstd288"] = T.rolling(288, min_periods=1).std()
    gr["t_diff12"] = T - T.shift(12)
    gr["t_diff36"] = T - T.shift(36)
    gr["t_accel"] = gr["t_diff12"] - gr["t_diff36"]
    gr["to_lag36"] = TO.shift(36)
    gr["to_lead36"] = TO.shift(-36)
    for h in EW_HL_T:
        gr[f"t_ew{h}"] = T.ewm(halflife=h).mean()
        # lead-EWMA: 시간축을 뒤집어 ewm() 적용 후 다시 뒤집음 -> "미래 쪽으로 감쇠하는
        # 가중평균". test(보간) 행에서 미래 방향 맥락을 반영하려는 목적(26.07.17 채택).
        gr[f"t_ewlead{h}"] = T[::-1].ewm(halflife=h).mean()[::-1].to_numpy()
    for h in EW_HL_TO:
        gr[f"to_ew{h}"] = TO.ewm(halflife=h).mean()
        gr[f"to_ewlead{h}"] = TO[::-1].ewm(halflife=h).mean()[::-1].to_numpy()
    return gr


def build_temp_features_full(train_env: pd.DataFrame, test_env: pd.DataFrame,
                              st: pd.DataFrame) -> pd.DataFrame:
    """train+test+합성스캐폴드를 zone별로 완전히 이어붙여 이력 피처(과거+미래 양방향)를
    한 번에 계산한다. 반환 DataFrame의 'source' 컬럼(train/test/synthetic)으로 걸러서
    쓴다 — synthetic 행은 학습·예측 어디에도 쓰지 않는다."""
    full = build_full_scaffold(train_env, test_env, st)
    full["hour"] = full["mod"] // 60
    full["hour_sin"] = np.sin(2 * np.pi * full["hour"] / 24)
    full["hour_cos"] = np.cos(2 * np.pi * full["hour"] / 24)
    zone_map = full[["dat", "zone"]].drop_duplicates("dat")  # groupby.apply가 그룹키 컬럼을
    full = full.groupby("zone", group_keys=False).apply(_history_features)  # 소거하는 pandas 동작 보정용
    if "zone" not in full.columns:
        full = full.merge(zone_map, on="dat", how="left")
    return full.reset_index(drop=True)


class TempEnsemble:
    """LGBM+Ridge 블렌드(TREE) 와 Ridge+HGB EWMA 서브모델을 가중 앙상블.
    predict(feat_df) -> np.ndarray."""

    def __init__(self, lgbm, tree_scaler, tree_ridge, ew_scaler, ew_ridge, ew_hgb, med):
        self.lgbm = lgbm
        self.tree_scaler = tree_scaler
        self.tree_ridge = tree_ridge
        self.ew_scaler = ew_scaler
        self.ew_ridge = ew_ridge
        self.ew_hgb = ew_hgb
        self.med = med

    def predict(self, feat: pd.DataFrame) -> np.ndarray:
        Xt = feat[TREE_COLS].fillna(self.med).fillna(0.0)
        tp = 0.5 * self.lgbm.predict(Xt) \
            + 0.5 * self.tree_ridge.predict(self.tree_scaler.transform(Xt))
        Xe = feat[EW_COLS].fillna(self.med).fillna(0.0)
        ep = 0.5 * self.ew_ridge.predict(self.ew_scaler.transform(Xe)) \
            + 0.5 * self.ew_hgb.predict(Xe)
        return W_MAIN * tp + W_EW * ep


def fit_temp(feat: pd.DataFrame, y: pd.Series) -> TempEnsemble:
    med = feat[TREE_COLS + EW_COLS].median()
    Xt = feat[TREE_COLS].fillna(med).fillna(0.0)
    lgbm = lgb.LGBMRegressor(**LGB_PARAMS).fit(Xt, y)
    tsc = StandardScaler().fit(Xt)
    tr = Ridge(alpha=10.0).fit(tsc.transform(Xt), y)

    Xe = feat[EW_COLS].fillna(med).fillna(0.0)
    esc = StandardScaler().fit(Xe)
    er = Ridge(alpha=1.0).fit(esc.transform(Xe), y)
    eh = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.03,
                                       max_leaf_nodes=15, min_samples_leaf=30,
                                       l2_regularization=3.0, random_state=0).fit(Xe, y)
    return TempEnsemble(lgbm, tsc, tr, esc, er, eh, med)
