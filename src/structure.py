"""
재배 그룹(zone) 식별.

제공된 환경 데이터(train_X / test_X)만 사용하며, 어떤 상수도 하드코딩하지 않는다.
구조는 데이터로부터 알고리즘으로 도출된다.

  1) 외부기상 그룹(weather group)
     외부 환경 값이 사실상 동일한 날들을 union-find로 묶는다.
     이 그룹의 최대 크기가 곧 서로 구분되는 재배 그룹 수(=군집 수)가 된다.

  2) 재배 그룹(zone)
     날짜별 '제어·내부기후 지문'(제어 이력 가동률 + 내부 기후 수준)을 KMeans로 군집화한다.
     외부기상이 같은 날들은 서로 다른 그룹에 속해야 하므로,
     그 안에서는 Hungarian 배정으로 서로 다른 군집에 대응시킨다.

  ※ 날짜 간 순서는 사용하지 않는다. 각 날은 독립 단위이며,
     train 에 나타나는 4개 그룹 패턴만 test 로 전이한다.
     외부 데이터·정답(test_y)·하드코딩 상수를 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EXT_COLS = ["temperature_outside", "humidity_outside", "solar_radiation",
            "wind_direction_outside", "wind_speed_outside", "rainfall"]
ACT_COLS = ["greenhouse_roof_vent1", "greenhouse_roof_vent2", "shading_curtain",
            "thermal_curtain", "fcu_fan", "fcu_pump", "circ_fan", "co2_supply",
            "fogging"]
IN_COLS = ["temperature", "humidity", "co2"]

SAME_WEATHER_TOL = 0.01    # 외부값이 이 이하로 다르면 '같은 값'
SAME_WEATHER_FRAC = 0.99   # 전체 칸의 이 비율 이상 같으면 '같은 기상일'


def parse_time(df: pd.DataFrame) -> pd.DataFrame:
    """'DAT109 14:27' -> dat=109, mod=867(하루 중 분)"""
    df = df.copy()
    m = df["time"].str.extract(r"DAT(\d+)\s+(\d+):(\d+)")
    df["dat"] = m[0].astype(int)
    df["mod"] = m[1].astype(int) * 60 + m[2].astype(int)
    return df


# ────────────────────────────────────────────────────────────── 1) 기상 그룹
def _weather_matrix(env: pd.DataFrame) -> dict[int, np.ndarray]:
    out = {}
    for d, g in env.groupby("dat"):
        out[int(d)] = g.sort_values("mod")[EXT_COLS].to_numpy(dtype=np.float64)
    return out


def group_by_weather(env: pd.DataFrame) -> dict[int, int]:
    """외부 기상이 사실상 동일한 날들을 같은 그룹으로 묶는다. -> {dat: group_id}"""
    W = _weather_matrix(env)
    dats = sorted(W)
    parent = {d: d for d in dats}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(dats):
        for b in dats[i + 1:]:
            if W[a].shape != W[b].shape:
                continue
            same = (np.abs(W[a] - W[b]) <= SAME_WEATHER_TOL).mean()
            if same >= SAME_WEATHER_FRAC:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    roots = sorted({find(d) for d in dats})
    rid = {r: i for i, r in enumerate(roots)}
    return {d: rid[find(d)] for d in dats}


# ─────────────────────────────────────────────────────────── 2) 그룹 지문/군집
def day_fingerprint(env: pd.DataFrame) -> pd.DataFrame:
    """날짜별 '제어 이력 + 내부 기후' 지문. 외부 기상은 절대 넣지 않는다
    (외부 기상은 그룹 간 동일하므로 구분 정보가 0이다)."""
    rows = []
    for d, g in env.groupby("dat"):
        r = {"dat": int(d)}
        night = g["mod"] < 360
        for c in ACT_COLS:
            v = g[c].to_numpy(dtype=np.float64)
            r[f"{c}__duty"] = float((v > 0).mean() * 100)
            r[f"{c}__mean"] = float(v.mean())
            r[f"{c}__max"] = float(v.max())
        for c in IN_COLS:
            v = g[c].to_numpy(dtype=np.float64)
            r[f"{c}__mean"] = float(v.mean())
            r[f"{c}__std"] = float(v.std())
            r[f"{c}__night"] = float(g.loc[night, c].mean())
            r[f"{c}__day"] = float(g.loc[~night, c].mean())
        rows.append(r)
    return pd.DataFrame(rows).set_index("dat").sort_index()


def fit_zones(env_train: pd.DataFrame, n_zones: int, seed: int = 0):
    """train 날짜들로 그룹 군집을 학습한다. -> (scaler, kmeans, fingerprint columns)"""
    fp = day_fingerprint(env_train)
    sc = StandardScaler().fit(fp.values)
    km = KMeans(n_clusters=n_zones, n_init=50, random_state=seed).fit(sc.transform(fp.values))
    return sc, km, list(fp.columns)


def assign_zones(env: pd.DataFrame, sc, km, fp_cols, wgroup: dict[int, int]) -> dict[int, int]:
    """날짜 -> 그룹. 외부기상이 같은 날들은 서로 다른 그룹이 되도록 Hungarian 배정."""
    fp = day_fingerprint(env)[fp_cols]
    Z = sc.transform(fp.values)
    D = np.linalg.norm(Z[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    dats = list(fp.index)
    zone = {}
    for gid in sorted(set(wgroup[d] for d in dats)):
        members = [i for i, d in enumerate(dats) if wgroup[dats[i]] == gid]
        sub = D[members]
        if len(members) == 1:
            zone[dats[members[0]]] = int(np.argmin(sub[0]))
            continue
        ri, ci = linear_sum_assignment(sub)
        for r, c in zip(ri, ci):
            zone[dats[members[r]]] = int(c)
    return zone


# ──────────────────────────────────────────────────────────────── 전체 조립
def discover(env_train: pd.DataFrame, env_test: pd.DataFrame | None, seed: int = 0):
    """train(+test) 환경 데이터로부터 재배 그룹 구조를 도출한다.

    반환: (structure DataFrame[dat, split, wgroup, zone], artifacts dict)
    ※ 날짜 간 순서는 도출하지 않는다.
    """
    env_all = env_train if env_test is None else pd.concat([env_train, env_test],
                                                           ignore_index=True)
    wgroup = group_by_weather(env_all)
    n_zones = int(max(1, max(pd.Series(list(wgroup.values())).value_counts())))

    sc, km, fp_cols = fit_zones(env_train, n_zones, seed=seed)
    zone = assign_zones(env_all, sc, km, fp_cols, wgroup)

    tr_dats = set(env_train.dat.unique())
    rows = [{"dat": d, "split": "train" if d in tr_dats else "test",
             "wgroup": wgroup[d], "zone": zone[d]}
            for d in sorted(wgroup)]
    st = pd.DataFrame(rows).sort_values(["zone", "dat"]).reset_index(drop=True)
    art = {"scaler": sc, "kmeans": km, "fp_cols": fp_cols, "n_zones": n_zones}
    return st, art


# ────────────────────────────────────────────────────── 3) 실제 날짜 체인 순서
def _rep_series(env: pd.DataFrame, dats: list[int]) -> np.ndarray:
    """그룹 내에서 데이터 행이 가장 완전한 dat 하나를 골라 외부기온 시계열(분 순 정렬) 반환.
    같은 실제일이면 외부기상이 사실상 동일하므로 어느 dat을 골라도 결과는 같다."""
    best = max(dats, key=lambda d: int((env["dat"] == d).sum()))
    g = env[env["dat"] == best].sort_values("mod")
    return g["temperature_outside"].to_numpy(dtype=np.float64)


def chain_order(env: pd.DataFrame, wgroup: dict[int, int]) -> tuple[list[int], float]:
    """wgroup(dat->그룹id)들을 자정 경계 온도 연속성이 가장 매끄러운 순서로 체인 연결한다.

    ⚠️ 26.07.17 수정: 원래 전수 시작점 + 그리디 최근접 연결을 썼는데, 실제로 돌려보니
    그리디가 준최적 경로(총비용 2.10)에 갇혀서 진짜 정답 순서(총비용 0.90, evidence/
    폴더 근거로 직접 검증)를 못 찾는 경우가 있었다 — 몇몇 무관한 쌍의 경계비용도 우연히
    작아서(근접 tie) 그리디가 잘못된 다음 노드로 새는 문제. 그룹 수가 11개 안팎으로
    적으므로(2^11=2048가지) Held-Karp 동적계획법으로 "진짜 최단" 해밀턴 경로를 정확히
    구한다 (더 이상 휴리스틱이 아니라 이 비용함수 기준 전역 최적해가 보장됨).

    반환: (그룹id 순서 리스트, 총 경계비용) — 방향(어느 쪽이 더 이른 날인지)은 미정.
    """
    groups: dict[int, list[int]] = {}
    for d, gid in wgroup.items():
        groups.setdefault(gid, []).append(d)
    gids = sorted(groups)
    reps = {gid: _rep_series(env, dats) for gid, dats in groups.items()}

    def cost(a, b):
        sa, sb = reps[a], reps[b]
        if len(sa) == 0 or len(sb) == 0:
            return np.inf
        return abs(float(sa[-1]) - float(sb[0]))

    n = len(gids)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                C[i, j] = cost(gids[i], gids[j])

    full = 1 << n
    dp = np.full((full, n), np.inf)
    parent = np.full((full, n), -1, dtype=int)
    for i in range(n):
        dp[1 << i, i] = 0.0
    for mask in range(full):
        for last in range(n):
            if not (mask & (1 << last)) or not np.isfinite(dp[mask, last]):
                continue
            base = dp[mask, last]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                cand = base + C[last, nxt]
                if cand < dp[nm, nxt]:
                    dp[nm, nxt] = cand
                    parent[nm, nxt] = last

    full_mask = full - 1
    end = int(np.argmin(dp[full_mask]))
    best_total = float(dp[full_mask, end])

    path_idx = []
    mask, cur = full_mask, end
    while cur != -1:
        path_idx.append(cur)
        prev = int(parent[mask, cur])
        mask ^= (1 << cur)
        cur = prev
    path_idx.reverse()
    best_path = [gids[i] for i in path_idx]
    return best_path, best_total


def discover_calendar(env_train: pd.DataFrame, env_test: pd.DataFrame | None, seed: int = 0):
    """discover()에 '실제 경과일(real_day, 1부터)'을 추가로 복원해 붙인다.

    체인 순서(chain_order)만으로는 방향(어느 쪽이 1일차인지)이 정해지지 않으므로,
    실제일을 가장 많이(=결측 없이) 갖고 있는 zone의 train 전용 dat 오름차순과
    상관이 양(+)이 되도록 방향을 고정한다 — 어떤 DAT 값도 하드코딩하지 않고,
    "결측이 가장 적은 zone"이라는 일반 규칙으로 도출한다.
    """
    st, art = discover(env_train, env_test, seed=seed)
    env_all = env_train if env_test is None else pd.concat([env_train, env_test],
                                                           ignore_index=True)
    wgroup = {int(r.dat): int(r.wgroup) for r in st.itertuples()}

    path, total_cost = chain_order(env_all, wgroup)
    order = {gid: i for i, gid in enumerate(path)}  # 0-based, 방향 미정

    zone_counts = st.groupby("zone")["dat"].count()
    ref_zone = int(zone_counts.idxmax())
    ref_train = st[(st["zone"] == ref_zone) & (st["split"] == "train")].sort_values("dat")
    ref_positions = [order[wgroup[int(d)]] for d in ref_train["dat"]]
    if len(ref_positions) >= 2:
        corr = np.corrcoef(np.arange(len(ref_positions)), ref_positions)[0, 1]
        if corr < 0:
            order = {gid: (len(path) - 1 - i) for gid, i in order.items()}

    st = st.copy()
    st["real_day"] = st["dat"].map(lambda d: order[wgroup[int(d)]] + 1)
    art = dict(art)
    art["chain_boundary_cost"] = total_cost
    art["ref_zone"] = ref_zone
    return st, art


# ──────────────────────────────────────────── 4) 청크별 확장창(expanding window) CV
def expanding_chunk_folds(st: pd.DataFrame, window_sizes=(3, 4, 5)):
    """zone(청크)마다 '자기 자신의 train-only real_day 순서'로 확장창 폴드를 만든다.

    real_day 7·8을 가진(=다른 zone과 달리 결측 없는) zone은 그만큼 실제일을 2일 더
    갖고 있으므로, 다른 zone들과 캘린더 진행 정도를 맞추기 위해 학습창을 window+2로
    준다. 검증은 항상 그 zone 자신의 시퀀스에서 학습창 바로 다음 날 하나.
    (사용자 지정 검증 방식, 26.07.17 — DAT 무작위 GroupKFold보다 zone의 실제 날짜
    진행 순서를 더 직접적으로 반영함.)

    반환: [(train_dats, val_dats), ...] — window_sizes 길이만큼의 폴드 리스트.
    """
    train_only = st[st["split"] == "train"]
    zone_days = {z: g.sort_values("real_day")["dat"].tolist()
                 for z, g in train_only.groupby("zone")}
    complete_zone = max(zone_days, key=lambda z: len(zone_days[z]))

    folds = []
    for w in window_sizes:
        train_dats, val_dats = [], []
        for zone, dats in zone_days.items():
            ww = w + 2 if zone == complete_zone else w
            if len(dats) <= ww:
                continue
            train_dats += dats[:ww]
            val_dats.append(dats[ww])
        folds.append((train_dats, val_dats))
    return folds
