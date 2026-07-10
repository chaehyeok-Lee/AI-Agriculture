# 베이스 이미지 자유(재현성을 위해 버전 고정 권장)
FROM python:3.13-slim

# 필수: 작업 디렉터리 'app' 은 제출물 폴더 루트로 고정
WORKDIR /app
COPY . /app

# 아래는 자유롭게 작성 (의존성 설치 등). 본 샘플은 표준 라이브러리만 사용.
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# 진입점: train.py -> inference.py (운영진 검증 시 override 가능)
CMD ["sh", "-c", "python train.py && python inference.py"]
