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

EDA 정리 (01_eda.py) 26.07.10.01:29
완료한 것
데이터 구조: train_X(1분,26일,19컬럼) / train_y(5분,26일,3컬럼) / test_X(1분,12일,19컬럼)
time("DAT109 00:00") → dt(timedelta) 인덱스로 통일, train→test 시간 연속 확인
결측치: 전 컬럼 0개
기술통계(describe): 물리적으로 잘못된 값 없음
구동기 컬럼 분류: on/off(0/201) 5개(fcu_fan,fcu_pump,circ_fan,co2_supply,fogging), 개도율(0~100) 2개(shading_curtain,thermal_curtain), 제한적 범위 2개(greenhouse_roof_vent1/2), 죽은 컬럼 1개(tube_rail_valve, 항상 0)
IQR 통계적 이상치 확인 → 전부 정상 물리 현상(낮/밤 주기, CO2 공급, 배지 건조, 야간 결로)으로 판단, 삭제한 값 없음
주의할 점
greenhouse_roof_vent1: train 범위 030인데 test는 059까지 감 (train이 못 본 값)
wind_speed_outside: train 최대 6.54, test 최대 8.28 (train이 못 본 값)
내부 temperature: train 최소 1.50, test 최소 -0.20 (test 겨울 구간이 더 추움, train이 못 본 저온)
내부 humidity 100% 포화: test에서 7번, 최대 182분간 지속 (야간 결로로 추정, 센서 오류 아님)
solar_radiation, soil_ec는 원래 분포가 치우쳐 있어 통계 기법(IQR 등) 적용 시 주의 필요
현재 01_eda.py는 실험용 스크립트라 dt 인덱스 변환 등이 주석 처리/누락된 상태일 수 있음 — 실행 전 확인 필요
다음에 할 일 (추천)
확정된 전처리 로직(dt 인덱스 변환, 201→1 매핑, 컬럼 정리)을 preprocess.py로 분리해 train.py/inference.py가 공유하도록 구성
train_X(1분)와 train_y(5분) 시간 간격 정렬/리샘플링 방식 결정
train이 못 본 범위(roof_vent1, wind_speed, 내부 저온)에 대한 모델 대응 방안 검토 (클리핑, train+test 통합 스케일링 등)
tube_rail_valve(죽은 컬럼) 제외 여부 결정
베이스라인 모델 학습 (train.py/inference.py를 hello-world에서 실제 파이프라인으로 교체)
랩실에서 이어서 작업하기 (현재 상태 / 부족한 부분)
01_eda.py 현재 상태: 지금까지 실행했던 코드가 전부 주석(#)으로 남아있는 "로그" 형태이고, 실제로 살아서 실행되는 코드는 파일 맨 아래 두 블록(soil_moisture 이상치 확인, 내부 습도 100% 연속 구간 확인)뿐입니다. 그 위의 데이터 불러오기(pd.read_csv 3줄)만 살아있고, 나머지(결측치 확인, describe, IQR 등)는 전부 주석 처리된 상태로 남아있습니다.
dt(timedelta) 인덱스 변환 함수(make_dt_index)가 현재 주석 처리되어 비활성 상태입니다 (파일 164~172번째 줄 근처). 리샘플링/시간 정렬 등 시간축이 필요한 작업을 이어가려면 이 부분을 다시 살려야 합니다. 지금 상태로도 파일 자체는 에러 없이 끝까지 실행됩니다 (맨 아래 두 블록은 dt 없이도 동작함).
실행 환경: 그동안 C:\Users\chaeh\AppData\Local\Programs\Python\Python314\python.exe로 실행함. 랩실 컴퓨터에도 Python + pandas가 설치되어 있어야 합니다.
데이터 위치: dataset/train/env/{train_X.csv, train_y.csv}, dataset/test/env/test_X.csv 경로 기준으로 코드를 짰습니다. 랩실에서도 같은 상대 경로에 데이터가 있어야 그대로 돌아갑니다 (OneDrive 동기화로 파일 자체는 같이 이동할 가능성 높음).
아직 파일로 따로 저장된 전처리 코드가 없습니다. 위 "완료한 것" 섹션은 결과 요약이고, 실제 재사용 가능한 코드(make_dt_index, 201→1 매핑 등)는 이 README의 대화 기록이나 01_eda.py 주석 속에만 남아있는 상태라, preprocess.py로 아직 옮기지 않았습니다.
작업 방식 메모: 지금까지는 AI가 코드를 파일에 직접 쓰지 않고 채팅으로 코드만 주면 사용자가 직접 붙여넣는 방식으로 진행했습니다. 이어서 작업할 때도 같은 방식을 원하면 새 세션에서 다시 알려줘야 합니다 (자동으로 기억되지 않음).


