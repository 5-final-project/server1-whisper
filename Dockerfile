# 베이스 이미지 설정: 파이썬 3.9 슬림 버전 사용
FROM python:3.9-slim-buster

# 작업 디렉토리 설정: 컨테이너 내에서 /app으로 작업
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

# 프로젝트 파일 복사
COPY . .

# FastAPI 애플리케이션 실행 명령어 설정
# --host 0.0.0.0: 외부에서의 접근을 허용
# --port 8000: 컨테이너 내부에서 사용할 포트
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
