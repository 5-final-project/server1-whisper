# 1. CUDA 런타임이 포함된 베이스 이미지
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# 2. 파이썬 캐시 비활성화 및 비버퍼 출력
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. 필수 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

# 4. python → python3 심볼릭 링크
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 5. 작업 디렉토리
WORKDIR /app

# 6. 의존성 복사 및 설치 (CUDA 휠 인덱스 추가)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r requirements.txt --no-cache-dir

# 7. 애플리케이션 코드 복사
COPY . .

# 8. FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]