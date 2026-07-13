# 온라인테스트 1 — 멀티모달(환경·다분광) 근권부 환경 예측

온실 환경 데이터(tabular)와 딸기 다분광 하이퍼큐브(10밴드 713~920nm)를 융합하여
5분 단위 근권부 환경(**배지 함수율·EC·온도**)을 예측한다.

## 실행 방법

### Docker (제출 규격)
```bash
docker build -t online1 .
docker run --rm -e DATA_ROOT=/app/input -v /경로/데이터:/app/input -v /경로/출력:/app/output online1
# 컨테이너 내부에서 train.py → inference.py 실행 → output/submission.csv 생성
```

### 로컬
```bash
export DATA_ROOT=./dataset      # train/env, test/env, train/ms, test/ms 를 포함
python train.py                 # 모델 학습 → model/model.pkl
python inference.py             # 예측 → output/submission.csv
```

- **입력 위치**: `DATA_ROOT` 미지정 시 `input/` → `dataset/` 순으로 자동 탐색.
  하위에 `train/env/train_X(_1).csv`, `train/env/train_y.csv`, `test/env/test_X.csv`,
  `train/ms/`, `test/ms/` 가 있다고 가정. 파일명 변형은 자동 탐색한다.
- **출력**: `output/submission.csv` (`OUTPUT_DIR` 로 변경 가능), 형식은 `train_y.csv` 와 동일.

## 파일 구성
| 파일 | 역할 |
|---|---|
| `features.py` | 공통 모듈: 피처 생성 · 커튼 레짐 판별 · 다분광 큐브 추출 · 전문가 모델 |
| `train.py` | 학습 진입점 → `model/model.pkl` |
| `inference.py` | 추론 진입점 → `output/submission.csv` |
| `Dockerfile`, `requirements.txt` | 실행 환경 |

## 핵심 아이디어 — 타깃별 특성에 맞춘 분리 모델링

세 타깃은 물리적으로 성격이 완전히 달라, 단일 회귀 대신 **타깃별 전략**을 적용했다.

### soil_temp (배지 온도) — 근권가온 구동
- **커튼 레짐별 (Ridge + HistGradientBoosting) 블렌드 전문가**, 환경 + **lag(1h~1일)·dewpoint_gap** 피처.
- 배지 온도는 난방·외기에 의해 결정되며 **열관성**이 커서, 과거값 lag 이 유효(외삽 안전).
- 급한파(외기 급강하) 구간은 데이터 부족으로 예측 오차가 커지는 한계가 있다.

### soil_moisture (배지 함수율) — 관수 구동
- **최근 3일 레벨(persistence)** + **일사구동 within-day shape**(HGB 잔차 모델).
- 함수율은 between-day 성분이 지배적이고 **미래 레벨 외삽이 근본적으로 어렵다**. 선형추세 외삽은
  최대 12일 앞에서 발산하므로, "최근 레벨 유지 + 일사비례 관수 톱니 복원"이 가장 로버스트했다.

### soil_ec (배지 EC) — 운영(시비) 결정
- **강화 레짐 라우팅**: 고EC 날은 전문가(환경+**다분광 융합**), 그 외는 **최근 저레벨 persistence**,
  소프트 신뢰도로 블렌드.
- **왜 레짐 분리인가**: EC 는 환경으로 예측되지 않는 **농가의 시비 결정**이다. 본 데이터에서
  *외부 기상이 완전히 동일한 날들*의 EC 가 0.4~1.7 로 갈렸다(자연실험). 즉 연속 환경변수 회귀로는
  잡히지 않고, **급격한 고농도 시비 에피소드**를 관리행동의 흔적으로 감지해야 한다.
- **고EC 레짐 판별(핵심 발견)**: 겨울 집중 수확관리 기간에는 보온·차광 커튼을 완전 동기화로
  전개(**agreement ≥ 0.98**)하고 **순환팬을 끈다(circ_fan ≈ 0)**. 이 **두 독립 신호를 모두 요구**하면
  학습 데이터의 고EC 날을 100% 정밀·재현으로 지목한다(커튼만으로는 초기 한파일 오탐 발생).
- **재배학적 근거**: 겨울철 딸기는 과실 품질을 위해 급액 EC 를 상향(겨울 ~2.2 vs 봄 ~1.3 mS/cm)하고,
  근권 EC 가 임계(~1.2)를 넘으면 플러시한다 — 관측된 이산적 봉우리와 부합.

## 다분광(멀티모달) 활용
- test 큐브에서 작물 위치의 캐노피 지표(veg_frac, NDRE, red-edge 등)를 당일 요약해
  **soil_ec 고전문가에 융합**한다.
- 검증 결과 다분광은 함수율·온도 예측엔 유의미한 이득이 없었다(성숙 수확기라 캐노피가
  정체 + 713~920nm 에 물 흡수 밴드 없음). 잡음이 되는 곳엔 넣지 않고, 약한 신호가 있는 **EC 에만
  융합**하여 멀티모달을 구현하되 RMSE 를 방어했다.

## 검증 (walk-forward)
test 가 미래 구간이므로 **시간순 확장창(walk-forward)** 이 가장 객관적이다. 학습이 고EC 에피소드를
포함하는 최신 구간 기준 실측 RMSE:

| 타깃 | RMSE |
|---|---|
| soil_temp | ~0.85 |
| soil_moisture | ~1.42 |
| soil_ec | ~0.15 |

- 여러 검증 스킴(walk-forward / LODO / 블록 / 랜덤 day)에서 EC 는 0.13~0.20 으로 수렴.
- **한계**: 고EC 에피소드가 데이터에 1개뿐이라, test 의 고EC 가 학습된 에피소드와 질적으로 다르면
  어떤 검증으로도 보증할 수 없다. 이에 대비해 **소프트 신뢰도 폴백**(애매하면 중간값)을 내장했다.

## 설계 원칙
- **day_number(정식후 일수) 미사용**: 훈련 범위 밖 미래(test)에서 외삽 시 트리가 붕괴하는 누수 위험을 제거.
- 피처는 train+test 를 이어붙여 계산해 test 의 rolling/lag cold-start NaN 을 방지(타깃 누수 없음).
