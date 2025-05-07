# 1. CUDA 11.8 베이스 이미지 사용 (PyTorch 호환성을 위해 CUDA 11.8 선택)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 2. 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Whisper 모델 관련 환경 변수
ENV MODEL_SIZE="medium"
ENV DEVICE="cuda"
ENV COMPUTE_TYPE="float16"
ENV BATCH_SIZE=16

# 4. 필수 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 5. python → python3 심볼릭 링크
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 6. 작업 디렉토리
WORKDIR /app

# 7. 의존성 복사 및 설치 (CUDA 11.8 휠 인덱스 추가)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r requirements.txt --no-cache-dir

# 8. 모델 저장을 위한 디렉토리 생성
RUN mkdir -p /app/models /app/temp_audio

# 9. 애플리케이션 코드 복사
COPY . .

# 10. FastAPI 서버 실행 (포트 8001 사용)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]