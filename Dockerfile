# 재현성을 위해 베이스 이미지와 의존성 버전을 모두 고정한다.
FROM python:3.11-slim

# 필수: 작업 디렉터리를 제출물 루트로 고정
WORKDIR /app

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.13.1 \
        scikit-learn==1.4.2 \
        lightgbm==4.3.0

COPY . /app

# 운영진이 볼륨으로 주입하는 경로 (코드는 이 상대 경로를 고정 사용)
RUN mkdir -p /app/input /app/output /app/model

# 스레드 수를 고정해 부동소수 누적 순서를 결정적으로 만든다
ENV OMP_NUM_THREADS=4 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1

# 검증: docker run ... sh -c "python train.py && python inference.py"
CMD ["sh", "-c", "python train.py && python inference.py"]
