# Whisper STT API 서버 및 클라이언트

오디오 파일을 텍스트로 만드는 STT API 서비스입니다. 
Faster-Whisper의 BatchedInferencePipeline을 사용해 긴 오디오 파일도 효율적이고 빠르게 처리할 수 있습니다.

## 프로젝트 구성

```
프로젝트 구조
|-- main.py             # 서버 메인 파일
|-- stt_client.py       # API 테스트용 클라이언트
|-- requirements.txt    # 의존성 파일
|-- Dockerfile          # 도커 이미지 빌드 파일
|-- temp_audio/         # 변환 과정 임시 오디오 파일 저장 디렉토리
```

## 주요 기능

- 클라이언트로부터 오디오 파일(.mp3, .wav, .m4a 등) 수신
- Faster-Whisper 기반 음성 인식 모델 사용 (medium 사이즈)
- BatchedInferencePipeline을 통한 효율적인 긴 오디오 처리
- 한국어/영어 등 다양한 언어 지원 (language 파라미터 지정 가능)

## 시스템 요구사항

### 서버 요구사항

- Python 3.8 이상
- CPU 실행 가능 (GPU 사용 가능 시 기본적으로 GPU 사용)
- Docker 환경 지원
- CUDA 12.6 지원 (선택사항 - GPU 사용 시)
- FFmpeg (오디오 전처리용)
- 최소 4GB RAM, 권장 8GB 이상


## 설치 및 실행 방법

### 시스템 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 디렉토리 생성
mkdir -p temp_audio
```

### 서버 실행

```bash
# CPU 모드
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker로 실행

```bash
# 도커 이미지 빌드
docker build -t whisper-stt-server .

# CPU 모드로 실행
docker run -p 8000:8000 whisper-stt-server

# GPU 모드로 실행 (NVIDIA Docker 필요)
docker run --gpus all -p 8000:8000 whisper-stt-server
```

## API 사용법

### Upload API 사용법

**Endpoint**: `POST /upload-audio`

**Parameters**:
- `file`: Audio file (mp3, wav, etc)
- `meeting_info`: 회의 정보 (선택)
- `language`: 언어 코드 (e.g., 'ko', 'en', 선택)

**예제**:
```bash
curl -X 'POST' \
  'http://cuvtgv0ku7.ap.loclx.io/upload-audio' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@LLM 설명 (요약버전) [HnvitMTkXro].mp3;type=audio/mpeg' \
  -F 'meeting_info=test' \
  -F 'language=ko'
```

### 클라이언트 사용법 (stt_client.py)

```bash
# 기본 사용법
python stt_client.py --server-url http://localhost:8000/upload-audio --audio-file "회의록_테스트.wav" --meeting-info "회의 테스트"

# 언어 지정
python stt_client.py --server-url http://localhost:8000/upload-audio --audio-file "회의록_테스트.wav" --meeting-info "회의 테스트" --language ko

# 결과 JSON 저장
python stt_client.py --server-url http://localhost:8000/upload-audio --audio-file "회의록_테스트.wav" --meeting-info "회의 테스트" --language ko --save-json result.json
```

## 응답 형식

```json
{
  "text": "오늘 회의에서 논의할 주제는 인공지능 활용 방안입니다.",
  "meeting_info": "회의 테스트",
  "processing_time_sec": 5.67
}
```

## 주의사항

- 긴 오디오 파일의 경우 처리 시간이 길어질 수 있습니다.
- GPU가 있는 환경에서 효율적으로 작동합니다.
- 한국어 오디오 파일의 경우 `--language ko` 옵션을 사용하는 것이 좋습니다.

## 버전 정보

- 업데이트: 2025년 5월 7일
