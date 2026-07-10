# 온라인테스트1 진행 현황

마지막 업데이트: 2026-07-10 (🔴 우선순위 5개 전부 완료)

```
온라인테스트1 (생육환경 예측)
│
├─ 1. 센서(env) EDA [01_eda.py]                         ✅ 완료
│   ├─ 구조·결측치·이상치 확인                            ✅
│   ├─ train/test 분포 차이 파악                          ✅
│   │   └─ roof_vent1(30→59), wind_speed(6.5→8.3), 내부온도(1.5→-0.2)
│   └─ dt(시간) 인덱스 활성화, 중복 제거                   ✅
│
├─ 2. 다분광(ms) EDA [02_ms_eda.py]                      ✅ 완료 (정리·검증 일부 남음)
│   ├─ 폴더 구조·촬영 스펙 전수 검증                       ✅
│   │   └─ 위치 0/1/2/3 존재, 촬영 스펙 100% 동일 확인
│   ├─ 5분 그리드 시간 매칭 (forward-fill, 3시간 제한)      ✅
│   ├─ 이미지→수치 피처 변환 (밴드별 평균)                  ✅
│   ├─ 화질 점검 (밝기/시간대 영향)                         ✅
│   │   └─ 낮 시간대 밝고 아침저녁 어두운 정상 패턴 확인
│   ├─ 활용 가치 검증 (day_num 대조실험)                    ✅
│   │   └─ 이미지 신호가 단순 날짜효과 아님을 확인
│   ├─ ⚠️ "9. 화질점검" 블록 통째로 중복 — 캐시 없으면       🔲 미정리 (🟡)
│   │      904장을 두 번 훑는 실질적 시간 낭비
│   ├─ ⚠️ 위치0 상관관계(0.30~0.37), 표본 크기(n) 확인 안 됨  🔲 미완료 (🟢)
│   │      → 7일치뿐이라 우연한 상관일 가능성 배제 안 됨
│   ├─ ⚠️ avg_band/std_band 결측률 미확인                    🔲 미완료 (🟢)
│   │      → std_band는 위치 2개 이상 동시 존재해야 계산되는데
│   │        위치별 결측률 70~92%라 대부분 NaN일 가능성 높음
│   ├─ ⚠️ 포화(1023) 픽셀 비율 — 발견만 하고 실제로 안 셈    🔲 미완료 (🟢)
│   ├─ ⚠️ test_quality 계산만 하고 결과 확인 안 함           🔲 미완료 (🟡)
│   └─ ⚠️ train vs test 이미지 피처 범위 비교 안 함          🔲 미완료 (🟢)
│
├─ 3. 전처리 파이프라인 [preprocess.py]                    ✅ 완료 및 검증됨
│   ├─ dt/actuator매핑/day_num,hour 함수화                  ✅
│   ├─ 1분→5분 그리드 집계 (mean/max/min/std/last)          ✅ 인과관계 방향 수정 완료
│   │      resample 결과를 5분 뒤로 shift + reindex 처리:
│   │      라벨 T의 피처가 (T-5,T] 구간(=T 이전 5분)만 담도록 수정
│   │      → y(T) 예측 시점에 미래 데이터가 안 섞이게 고침
│   │      맨 첫 행은 "이전 5분" 데이터가 없어 NaN (정상, 버그 아님)
│   ├─ CLIP_RANGES에 내부습도(humidity) 누락                🔲 미완료 (🟡)
│   │      train max98/test max100, 작지만 일관성 위해 추가 권장
│   ├─ (선택) ms 이미지 피처 병합 함수                       ⚠️ 실행 검증 안 됨 (🟡)
│   │      train_ms_matched.pkl의 dt(timedelta64[ns])와
│   │      5분 그리드 인덱스 단위가 또 안 맞을 위험 있음
│   │      (예전에 겪은 <m8[s]> vs <m8[us]> 에러 재발 가능)
│   ├─ Train/Val 시간순 분할 (마지막 4일)                    ✅
│   ├─ __main__ 검증 코드 실제 실행                          ✅ 완료
│   │      train_feat (7488,92) / train_y (7488,3) / 인덱스 일치 True
│   ├─ test_X에 대해서도 build_features 검증                 ✅ 완료
│   │      test_feat (3456,92) 정확히 일치 확인
│   └─ (참고자료의 "8개월 스팬"은 오류 — 실제론 38일로 정정)  ✅
│
├─ [requirements.txt / Dockerfile / .dockerignore]         ✅ 완료
│   ├─ pandas/numpy/matplotlib/seaborn 버전 고정            ✅
│   ├─ .dockerignore로 .venv/dataset/.env/캐시 제외          ✅
│   ├─ python:3.11-slim → 3.13-slim (numpy 2.5.1 호환)      ✅
│   ├─ 실제 docker build/run 성공 확인 (hello-world 기준)    ✅
│   └─ dataset/ vs input/ 경로 규칙 확정                     ✅
│         input/dataset/는 대회 측 샘플(1일치, P1_ 위치명)이라
│         실제 채점 시 마운트되는 경로 형식임을 확인.
│         → train.py/inference.py(Docker용): input/dataset/... 사용
│         → EDA·preprocess.py 로컬 검증: dataset/...(전체 실데이터) 유지
│
├─ [폴더 정리]                                             ✅ 완료
│   ├─ eda_outputs/ — EDA 시각화 PNG 9개                    ✅
│   └─ cache/ — ms 피처/매칭 결과 pkl 6개                   ✅
│
├─ [버전 관리]                                             ✅ 완료
│   └─ git init 완료, .gitignore 작성(.venv/dataset/input/output/
│       model/cache/eda_outputs/*.pkl/.claude 등 제외), 첫 커밋 진행
│
├─ 4. 베이스라인 모델 [train.py]                           ⏳ 대기 (모델 수준 고민 중)
│   ├─ 모델 라이브러리 결정 (scikit-learn / numpy직접 / LightGBM)  🔲 미결정
│   ├─ random_state 전체 고정                              🔲 미착수
│   ├─ input/dataset/... 경로로 읽도록 작성                 🔲 미착수
│   └─ 다분광(ms) 피처 실제 반영 여부 결정                   🔲 미결정
│
├─ 5. 추론·제출 [inference.py]                             ⏳ 대기
│   └─ submission.csv 포맷 assert 검증 (컬럼/행개수/NaN/timestamp 포맷)  🔲 미착수
│
└─ 6. Docker 재현 검증                                     ⚠️ 1차 완료 (hello-world 기준일 뿐)
    └─ 4단계 실제 모델 완성 후 재검증 필수                   🔲 대기
```

## 우선순위 액션 리스트

**✅ 완료 (2026-07-10)**
1. ~~`python preprocess.py` 실제 실행 → 검증 통과 확인~~
2. ~~`aggregate_to_5min` 인과관계 방향 버그 수정~~
3. ~~test_X도 `build_features` 검증 (3456행 확인)~~
4. ~~dataset/ vs input/ 경로 확정 (Docker는 input/dataset/, 로컬은 dataset/)~~
5. ~~`git init` + `.gitignore` + 첫 커밋~~

**🟡 여유 있을 때 처리**
6. `merge_image_features` 실행해서 dtype 에러 여부 확인
7. CLIP_RANGES에 humidity 추가
8. 02_ms_eda.py "9. 화질점검" 중복 블록 제거
9. test_quality 결과 확인 및 해석

**🟢 이미지 피처를 실제로 모델에 쓰기로 결정했을 때만**
10. 위치0 상관관계 표본크기(n)/통계적 유의성 재검증
11. std_band 결측률 확인
12. 포화(1023) 픽셀 비율 확인
13. train vs test 이미지 피처 범위 비교

## 참고
- 최종 제출 형식은 train_y.csv와 동일 (5분 간격 timestamp,soil_moisture,soil_ec,soil_temp)
- 평가: 항목별 raw RMSE → 순위 점수 변환
