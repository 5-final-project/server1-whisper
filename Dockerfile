#1.베이스 이미지 설정
FROM python:3.9-slim-buster

#2.환경 변수 설정 (파이썬 캐시 비활성화)
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

#3.필수 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#4. 작업 디렉토리 설정
WORKDIR /app

#5. 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt --no-cache-dir

#6. 앱 코드 복사
COPY . .

#7. FastAPI 서버 실행 명령어 설정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]