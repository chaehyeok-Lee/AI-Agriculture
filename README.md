# 온라인테스트1 제출 샘플 (재현 검증용)

hello world

```bash
docker build -t submission .
docker run --rm -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" \
  submission sh -c "python train.py && python inference.py"
```
생육 환경 예측

참가 팀은 테스트 구간의 soil_moisture, soil_ec, soil_temp 값을 예측한다. 제출 파일은 time 컬럼 기준으로 정답 행과 매칭한다.

평가 항목은 배지수분 RMSE, 배지 EC RMSE, 배지온도 RMSE이다. 온라인1 점수는 각 항목 raw RMSE를 순위 점수로 변환해 산출한다.

주의사항 이 작업할거고 너가 앞으로 전부 임의로 코드는 수정하지 마 그냥 나한테 허락만 받거나 코드를 내가 캡쳐할수 있게 해

작업 요약 — 센서 EDA → 다분광 EDA → 전처리 파이프라인 → Docker/git 설정, 전체 흐름
데이터 구조 — env(1분/5분) + ms(다분광 이미지) 구조 요약
핵심 분석 인사이트 및 주의사항 (제일 비중 크게)
⚠️ 주의해야 할 점 (위험 신호)
greenhouse_roof_vent1: train 범위 030인데 test는 059까지 감 → train이 한 번도 못 본 값 구간. 트리 모델은 이 구간을 제대로 못 다루고 근처 학습값으로 뭉갤 수 있음
wind_speed_outside: train 최대 6.54, test 최대 8.28 → train이 못 본 강풍 구간
내부 temperature: train 최소 1.50, test 최소 -0.20 → train이 못 본 저온 구간. 이 컬럼이 soil_temp와 직접 연관 가능성 높아서 셋 중 가장 위험한 신호
solar_radiation, soil_ec: 원래 분포 자체가 심하게 치우쳐 있어서, IQR 같은 통계 기법을 무심코 적용하면 "맑은 날 정오"처럼 정상적인 값을 전부 이상치로 잘못 판단할 위험이 큼
다분광 위치0: train 7일/test 3일치뿐인 적은 표본이라, 상관관계(0.30~0.37)가 실제보다 과대평가됐을 가능성 있음 — 과신 금지
5분 집계 시 인과관계 방향: pandas resample을 기본값 그대로 쓰면 "지금 시점" 피처에 미래 데이터가 섞여 들어가는 버그가 생김 (이미 발견해서 고쳤지만, 유사한 시계열 집계 코드를 새로 짤 때마다 항상 재확인 필요)
dataset/ vs input/ 경로 혼동: 로컬 개발(dataset/)과 실제 채점 환경(input/dataset/)의 경로가 다름 — train.py/inference.py는 반드시 input/dataset/ 기준으로 짜야 함
💡 중요한 점 (핵심 발견 — 모델링에 꼭 반영)
day_num(경과일수)이 soil_moisture와 상관관계 0.72 — 다분광 이미지보다도 강력한 단일 신호. 반드시 피처로 넣어야 함
soil_moisture 이상치 4개는 사실 하나의 사건이었음 — 같은 날 저녁(DAT128, 19:55~21:40) 수분 급감 + EC 급등 + 온도 상승이 동시에 나타남 → 시비/관수 이벤트로 추정되는 정상 물리 현상, 삭제하면 안 됨
CO2, solar_radiation, 내부온도의 "통계적 이상치"는 전부 정상 물리 현상 (CO2 공급장치 작동, 낮/밤 주기, 계절 추세) — IQR 기준으로 걸러도 실제로 삭제하면 신호의 5~13%를 날리는 꼴이라 오히려 성능이 나빠짐
다분광 이미지가 day_num 대조실험을 통과함 — 위치0의 상관관계가 단순 날짜 우연이 아니라 실제 이미지 정보를 담고 있다는 게 확인됨 (같은 기간 안에서 날짜 자체는 상관관계 거의 0인데 이미지 값은 상관관계 있었음)
구동기 컬럼 정리 규칙 확정: on/off 5개는 0/201→0/1 매핑만 하면 되고, tube_rail_valve는 분산이 0인 죽은 컬럼이라 제거 가능
train_X/train_y의 시간 구조가 완전한 격자(빠진 시각 없음)로 확인되어, 리샘플링을 걱정 없이 안전하게 할 수 있음
🔍 확인해봐야 할 점 (아직 결론 안 남)
다분광 이미지를 최종 모델에 실제로 넣을지 여부 — 상관관계는 확인됐지만 사용 여부는 미결정
위치0 상관관계의 통계적 유의성(작은 표본 대비) 재검증
avg_band/std_band 요약 피처의 실제 결측률 — 위치 2개 이상 동시에 값이 있어야 계산되는데 위치별 결측률이 70~92%라 대부분 NaN일 가능성
카메라가 10비트 센서(최대 1023)로 추정되는데, 실제로 포화(클리핑)된 이미지가 몇 장인지 세어보지 않음
베이스라인 모델 라이브러리 선택 (scikit-learn / LightGBM / numpy 직접구현) — 아직 미정

💡 추가 발견 (26.07.10 오후, 🟡 항목 처리 중)
5분 집계 인과관계 버그 실제로 수정 완료 — resample 결과를 5분 뒤로 shift + 첫 행 reindex 처리. 라벨 T 피처가 이제 (T-5,T] 구간만 담음. python preprocess.py로 train(7488,92)/test(3456,92) 둘 다 재검증 통과
merge_image_features 함수 실행 검증 완료 — dtype 에러 없이 정상 작동 (shape 7488×152)
test 다분광 이미지가 train보다 평균 28% 더 밝음 (train 128.75 vs test 164.98) — 01_eda.py에서 이미 확인한 "test 구간 solar_radiation 평균이 train보다 높다"는 사실과 정확히 교차검증됨 (겨울이라 기온은 낮지만 맑은 날이 많아 일사량 자체는 더 강함). 이미지 피처를 모델에 쓸 경우 train/test 밝기 스케일 차이도 다른 range-mismatch 컬럼들처럼 주의 필요
카메라 최대값이 train/test 둘 다 정확히 1023 — 10비트 센서 추정에 대한 재확인

📡 다분광 밴드(파장)별 의미 — 도메인 지식 (26.07.10 정리)
10개 밴드 파장: 713, 736, 759, 782, 805, 828, 851, 874, 897, 920nm. 전부 적색 경계(red edge)~근적외선(NIR) 영역이고 일반 RGB가 아님 — 작물 생리 상태 관찰용 특화 파장대.
713, 736nm (적색 경계 초입) — 엽록소 흡수대(~680nm)에서 빠져나오는 구간. 엽록소 농도·질소 상태·초기 스트레스에 민감. 상용 농업센서(RapidEye 690~730nm 등)가 흔히 쓰는 red edge 밴드와 겹침.
759nm (적색 경계 → NIR 전환부) — 반사율이 급격히 오르는 변곡점(red edge inflection point) 근처. 이 변곡점 위치 자체가 스트레스 지표 — 건강할수록 긴 파장 쪽, 스트레스받으면 짧은 파장 쪽으로 이동(blue shift).
782~920nm (NIR 평탄부, 8개 밴드) — 색소가 아니라 잎 내부 세포구조·조직 온전성(turgor)을 반영. 897·920nm은 수분 흡수대(970nm 부근)에 가까워 잎 수분함량 영향이 약간 더 섞일 수 있음.
핵심 시사점 — 10개 밴드 중 8개(782~920)가 전부 같은 "NIR 평탄부"라 서로 다른 정보라기보다 같은 신호(잎 구조)를 반복 측정한 것에 가까움. 이게 바로 02_ms_eda.py 상관관계표에서 band1~band10_mean 상관계수가 거의 다 비슷하게 나왔던 이유(물리적으로 설명됨, 우연 아님).
모델링 제안 — 나중에 이미지 피처 쓸 때 10개 밴드를 다 따로 넣기보다 ①713·736(색소/스트레스 그룹) vs ②782~920 평균(구조 그룹) 2그룹으로 압축하거나, 두 그룹 비율(red-edge 지수)을 만드는 게 정보 중복을 줄임. 이 데이터엔 진짜 red(~650nm) 밴드가 없어서 일반 NDVI는 직접 계산 불가.

✅ 위 2그룹 가설 실측 검증 결과 (26.07.10, eda_outputs/correlation_vs_wavelength.png · reflectance_spectrum.png)
반사 곡선(파장별 평균 밴드값)은 이론과 완벽히 일치 — 713→782nm 급상승(적색경계), 782~874nm 평탄부, 897~920nm 재하강(수분흡수대 인접). 전형적인 건강한 식물 분광 곡선.
상관계수는 타깃마다 다름: soil_moisture·soil_ec는 가설대로 713~759nm에서 오르다 782nm부터 완전히 평평(2그룹 구분 유효). soil_temp는 782nm 이후로도 920nm까지 계속 완만히 상승 — 평탄부 안에서도 파장 길수록 더 강해짐. 즉 "2그룹"이 아니라 "적색경계(713~759, 상승) / NIR평탄부(782~874, 평평) / 끝단(897~920, soil_temp에 특히 중요)" 3구간으로 보는 게 더 정확.
→ soil_temp 예측 시엔 band10(920nm)을 NIR 그룹 평균에 묻지 말고 별도 피처로 남기는 걸 권장.

완료된 파일 목록 — 01_eda.py, 02_ms_eda.py, preprocess.py, Dockerfile/requirements.txt/.dockerignore, PLAN.md
작업 방식 메모 갱신 — 기존 "코드 직접 수정 금지" 원칙 유지 확인

세션 재개용 메모 (26.07.10 기준, 대화 기록 없이 이어서 작업할 때 참고)

새 세션 시작하면 제일 먼저 PLAN.md부터 읽을 것 — 진행 상황(트리 구조), 우선순위 액션 리스트(🔴🟡🟢)가 실시간으로 관리되는 파일임. 이 README는 스냅샷이고 PLAN.md가 최신 상태.

지금까지 만들어진 파일과 역할
- 01_eda.py — 센서(env) 데이터 EDA. dt 인덱스 활성화됨, 구동기 분류/이상치 확인 완료
- 02_ms_eda.py — 다분광(ms) 이미지 EDA. 폴더 구조 파싱, 5분 그리드 시간매칭(forward-fill), 화질 점검, day_num 대조실험까지 완료. 실행 결과 캐시는 cache/, 시각화 PNG는 eda_outputs/에 저장됨
- preprocess.py — train.py/inference.py가 공유할 전처리 함수 모음. dt변환/구동기매핑/클리핑/day_num,hour/1분→5분 집계/train-val 시간순 분할까지 구현 및 검증 완료 (python preprocess.py로 재검증 가능)
- train.py, inference.py — 아직 hello-world 샘플 상태. 다음 작업이 여기(4~5단계, 베이스라인 모델)
- Dockerfile, requirements.txt, .dockerignore — python:3.13-slim 기준으로 실제 docker build/run 검증 완료 (hello-world 기준)
- PLAN.md — 전체 진행 조직도 + 우선순위 액션 리스트 (제일 먼저 확인할 파일)

환경 정보
- 로컬 파이썬: .venv 안에 pandas 3.0.3 / numpy 2.5.1 / matplotlib 3.11.0 / seaborn 0.13.2 설치됨
- Docker Desktop, Git 설치 완료 (버전은 각각 docker --version, git --version으로 확인)
- git 저장소 초기화 및 첫 커밋 완료. GitHub 원격 저장소: https://github.com/chaehyeok-Lee/AI-Agriculture.git (푸시는 본인이 수기로 진행)

데이터 경로 규칙 (헷갈리기 쉬운 부분)
- dataset/... — 로컬에 있는 전체 실데이터. EDA·01_eda.py·02_ms_eda.py·preprocess.py 로컬 검증에서 사용
- input/dataset/... — 대회 측이 준 샘플(1일치, 위치명도 P1_ 등으로 실데이터와 다름). 실제 채점 시 docker run이 이 경로로 데이터를 마운트하므로, train.py/inference.py(Docker 안에서 실행되는 최종 제출 코드)는 반드시 이 경로 기준으로 작성해야 함

다음에 시작할 지점
- PLAN.md의 "🔴 베이스라인 시작 전 반드시 처리"와 "🟡 여유 있을 때 처리" 항목 전부 완료된 상태 (2026-07-10 기준)
- 다음은 4단계: 베이스라인 모델 라이브러리 결정(scikit-learn / numpy 직접구현 / LightGBM) → train.py 구현 → inference.py 구현 → Docker 재검증 순서
- 다분광(ms) 이미지를 실제 모델에 넣을지는 아직 미결정 — PLAN.md의 🟢 항목(표본크기 재검증 등) 먼저 확인 후 결정 권장

작업 규칙 추가 (26.07.10) — 앞으로 모델링에 유의미한 분석 결과가 새로 나올 때마다, 이 README(핵심 분석 인사이트 섹션)에 자동으로 반영하기로 함. PLAN.md는 작업 진행상황 추적용, README는 데이터 인사이트 축적용으로 역할 분리.