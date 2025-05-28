from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import uuid
import time
import shutil
import logging
import logging.handlers # 추가
import subprocess
import wave
import math
from pythonjsonlogger import jsonlogger # 추가

# --- FastAPI 앱 생성 및 CORS 설정 ---
app = FastAPI(
    title="Whisper STT API Server",
    description="음성 파일을 텍스트로 변환하는 API (Batched Inference Pipeline 사용)",
    version="1.0.0"
)

# --- Global Exception Handler for Logging ---
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error({
        "event": "unhandled_exception",
        "path": str(request.url),
        "error": str(exc),
        "type": type(exc).__name__
    })
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error({
        "event": "http_exception",
        "path": str(request.url),
        "error": str(exc.detail),
        "status_code": exc.status_code
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(FastAPIRequestValidationError)
async def validation_exception_handler(request: Request, exc: FastAPIRequestValidationError):
    logger.error({
        "event": "validation_exception",
        "path": str(request.url),
        "errors": exc.errors(),
        "body": await request.body()
    })
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 설정 ---
MODEL_SIZE = "medium"
if torch.cuda.is_available():
  DEVICE = "cuda"
  COMPUTE_TYPE = "float16"
else:
  DEVICE = "cpu"
  COMPUTE_TYPE = "int8"

UPLOAD_DIR = "temp_audio" # 업로드된 오디오 임시 저장 폴더
LOGS_DIR = "logs"         # 로그 파일 저장 폴더
SERVICE_NAME = "whisper-stt-server" # ELK에서 이 서비스 식별 이름
BATCH_SIZE = 16
NUM_WORKERS = min(4, os.cpu_count() or 4)


# --- 로깅 설정 ---
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.DEBUG) # 개발 시에는 DEBUG, 실제 운영 시 INFO 등으로 조정
logger.propagate = False # 루트 로거로의 전파 방지 (중복 로깅 방지)

# 로그 폴더 생성
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "whisper_server.log")

# JSON 포맷터
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('@timestamp'): # 이미 @timestamp가 있다면 사용 (Filebeat 등에서 설정 가능)
            log_record['@timestamp'] = logging.Formatter().formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S.%fZ')
        if record.levelname:
            log_record['log.level'] = record.levelname.upper()
        else:
            log_record['log.level'] = 'INFO' # 기본값
        log_record['service.name'] = SERVICE_NAME
        # transaction.id는 로깅 호출 시 extra로 전달받아 자동으로 포함됨

# 포맷터 인스턴스 생성
formatter = CustomJsonFormatter()

# 파일 핸들러 (RotatingFileHandler)
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 콘솔 핸들러 (개발 시 확인용 - JSON 포맷터 동일하게 적용)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler) # 개발 중에는 콘솔 출력도 같이 보면 편합니다.

# --- 기존 logging.basicConfig(level=logging.INFO) 부분은 삭제 또는 주석 처리 ---
# logging.basicConfig(level=logging.INFO) # 이 줄은 주석 처리하거나 삭제합니다.


# --- 모델 로드 ---
# 모델 로드 시점에 대한 로그 (기본 정보 포함)
base_log_extra_model_load = {"event.module": "initialization", "event.action": "load_model"}
logger.info(
    f"Attempting to load Whisper model '{MODEL_SIZE}' on device '{DEVICE}' with compute_type '{COMPUTE_TYPE}'",
    extra={**base_log_extra_model_load, "model.size": MODEL_SIZE, "model.device": DEVICE, "model.compute_type": COMPUTE_TYPE}
)
try:
  base_model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    cpu_threads=NUM_WORKERS,
  )
  model = BatchedInferencePipeline(model=base_model)
  logger.info("Whisper model with batched pipeline loaded successfully.", extra=base_log_extra_model_load)
except Exception as e:
  logger.critical(f"CRITICAL: Error loading Whisper model. Application may not work correctly.", exc_info=True, extra=base_log_extra_model_load)
  model = None
  base_model = None


# --- 배치 처리 함수 ---
def process_audio(audio_path: str, request_id: str, batch_size: int = BATCH_SIZE, language: str = None):
    """배치 처리를 활용해 오디오 파일을 한 번에 처리하는 함수. 요청 ID를 받아 로깅에 활용."""
    # 오디오 길이(초) 계산
    try:
        with wave.open(audio_path, 'rb') as wf:
            audio_duration_sec = wf.getnframes() / wf.getframerate()
    except Exception:
        audio_duration_sec = None

    log_extra_base = {"transaction.id": request_id, "audio.path": audio_path, "stt.batch_size": batch_size, "audio.duration_sec": round(audio_duration_sec, 2) if audio_duration_sec else "N/A"}
    logger.info("Starting audio processing (transcription)", extra=log_extra_base)
    start_time = time.time()

    try:
        transcription_options = {
            "beam_size": 5, "vad_filter": True, "word_timestamps": True,
            "condition_on_previous_text": True, "batch_size": batch_size,
            "task": "transcribe", "best_of": 5, "temperature": 0,
            "initial_prompt": "This is a business meeting recording."
        }
        # 🔽 이 아랫부분이 이전 제 코드에서 제공된 상세 로깅 및 STT 처리 로직입니다.
        # 이 부분을 채워주시면 됩니다.

        log_extra_transcribe = {**log_extra_base, "stt.options_preview": {k:v for k,v in transcription_options.items() if k != "initial_prompt"}} # 프롬프트는 길 수 있으므로 제외 또는 일부만

        if language:
            transcription_options["language"] = language
            log_extra_transcribe["stt.language.user_specified"] = language
            if language == "ko":
                transcription_options["initial_prompt"] = "This is a Korean business meeting recording. The primary language is Korean with occasional English terms or phrases. Please accurately transcribe the Korean speech while maintaining English terms that might be present. Focus on properly capturing Korean sentences with their natural flow and structure."
                # 프롬프트 내용도 중요 정보이므로 로깅 (길다면 일부만)
                log_extra_transcribe["stt.initial_prompt_used"] = transcription_options["initial_prompt"][:100] + "..." if len(transcription_options["initial_prompt"]) > 100 else transcription_options["initial_prompt"]
                logger.info("Transcribing with specific Korean language settings and prompt", extra=log_extra_transcribe)
            else:
                log_extra_transcribe["stt.initial_prompt_used"] = transcription_options["initial_prompt"][:100] + "..." if len(transcription_options["initial_prompt"]) > 100 else transcription_options["initial_prompt"]
                logger.info(f"Transcribing with specified language: {language}", extra=log_extra_transcribe)
        else:
            log_extra_transcribe["stt.initial_prompt_used"] = transcription_options["initial_prompt"][:100] + "..." if len(transcription_options["initial_prompt"]) > 100 else transcription_options["initial_prompt"]
            logger.info("Transcribing with automatic language detection", extra=log_extra_transcribe)

        # model 객체가 None이 아닌지 확인 (애플리케이션 시작 시 로드 실패 경우 대비)
        if model is None:
            logger.error("Whisper model is not loaded. Cannot perform transcription.", extra=log_extra_base)
            raise RuntimeError("Whisper model not available for transcription.")


        segments_iterable, info = model.transcribe(audio_path, **transcription_options)
        segments_list = list(segments_iterable) # 제너레이터 소모 및 결과 로깅을 위해 리스트 변환

        # 최종 텍스트 통계
        transcript_word_count = sum(len(segment.text.strip().split()) for segment in segments_list)

        end_time = time.time()
        processing_time_sec = end_time - start_time

        final_log_extra = {
            **log_extra_base,
            "stt.processing_time_sec": round(processing_time_sec, 2),
            "stt.language.detected": info.language if info else "N/A",
            "stt.language.probability": round(info.language_probability, 4) if info else "N/A",
            "stt.num_segments": len(segments_list),
            "transcript.word_count": transcript_word_count,
            "stt.throughput_ratio": round(processing_time_sec / audio_duration_sec, 2) if audio_duration_sec else "N/A",
            "stt.words_per_sec": round(transcript_word_count / processing_time_sec, 2) if processing_time_sec else "N/A",
        }
        logger.info("Transcription completed successfully", extra=final_log_extra)
        return segments_list, info # 정상 완료 시 결과 반환

    except Exception as e:
        elapsed_time_sec = time.time() - start_time # 에러 발생 시점까지의 시간
        logger.error(
            "Error during STT batch processing",
            exc_info=True, # 스택 트레이스 포함
            extra={**log_extra_base, "error.message_detail": str(e), "stt.processing_time_sec_before_error": round(elapsed_time_sec, 2)}
        )
        raise # 예외를 다시 발생시켜 FastAPI가 처리하도록 함 (예: /upload-audio 핸들러의 except 블록)


# --- FastAPI 엔드포인트 ---
from typing import Optional

@app.post("/upload-audio")
async def upload_audio(request: Request, file: UploadFile = File(...), meeting_info: str = Form("N/A"), language: Optional[str] = Form(None)):
    """
    오디오 파일을 STT로 변환하여 전체 텍스트를 JSON 으로 반환합니다.
    """
    import tempfile
    start_time = time.time()
    # 요청 ID 가져오기
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    # 임시 파일 저장
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    suffix = os.path.splitext(file.filename)[-1] if file.filename else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=suffix) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_path = temp_file.name

    # wav 변환 (필요 시)
    converted_wav_path = temp_path
    if not temp_path.lower().endswith(".wav"):
        converted_wav_path = temp_path + ".wav"
        try:
            subprocess.run([
                "ffmpeg", "-y", "-i", temp_path,
                "-ar", "16000", "-ac", "1", "-f", "wav", converted_wav_path
            ], check=True, capture_output=True)
        except Exception as e:
            os.remove(temp_path)
            raise HTTPException(status_code=500, detail=f"ffmpeg 변환 실패: {e}")
        os.remove(temp_path)

    try:
        segments, info = process_audio(converted_wav_path, request_id, BATCH_SIZE, language)
        sorted_segments = sorted(segments, key=lambda s: s.start)
        full_text = "\n".join([segment.text.strip() for segment in sorted_segments])
        return {
            "text": full_text,
            "meeting_info": meeting_info,
            "processing_time_sec": round(time.time() - start_time, 2)
        }
    finally:
        if os.path.exists(converted_wav_path):
            os.remove(converted_wav_path)


@app.get("/")
async def read_root():
    """
    Root health check.
    """
    return {"message": "Whisper STT API Server with BatchedInferencePipeline is running"}

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)