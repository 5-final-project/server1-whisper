# main.py - 확실한 정보만 모니터링하는 버전

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import torch
import os
import uuid
import time
import shutil
import logging
import logging.handlers
import subprocess
import wave
import math
from pythonjsonlogger import jsonlogger
from prometheus_client import Gauge, Counter, Histogram
import psutil
import asyncio
from contextlib import asynccontextmanager
import threading
import re

# ✅ 확실한 메트릭만 정의 (추정 메트릭 제거)
team5_gpu_utilization = Gauge('team5_gpu_utilization_percent', 'Team5 GPU utilization', ['service'])
team5_gpu_memory_used = Gauge('team5_gpu_memory_used_mb', 'Team5 GPU memory used', ['service'])
team5_stt_requests = Counter('team5_stt_requests_total', 'Total STT requests', ['service'])
team5_stt_duration = Histogram('team5_stt_processing_seconds', 'STT processing time', ['service'])

# ❌ 제거된 메트릭들 (WSL2에서 정확 측정 불가)
# team5_process_gpu_memory = Gauge('team5_process_gpu_memory_mb', 'Estimated process GPU memory', ['service', 'container_id'])
# team5_process_gpu_utilization = Gauge('team5_process_gpu_utilization_percent', 'Estimated process GPU utilization', ['service', 'container_id'])

def get_reliable_gpu_info():
    """확실한 GPU 정보만 수집 (nvidia-smi 기반)"""
    try:
        # 전체 GPU 상태만 수집 (100% 신뢰 가능)
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode != 0:
            return None
            
        gpu_line = result.stdout.strip()
        if not gpu_line:
            return None
            
        # GPU 상태 파싱
        gpu_parts = [x.strip() for x in gpu_line.split(',')]
        utilization = int(gpu_parts[0]) if gpu_parts[0] != '[Not Supported]' else 0
        memory_used_mb = float(gpu_parts[1]) if gpu_parts[1] != '[Not Supported]' else 0
        memory_total_mb = float(gpu_parts[2]) if gpu_parts[2] != '[Not Supported]' else 10240
        temperature_c = float(gpu_parts[3]) if gpu_parts[3] != '[Not Supported]' else 0
        power_draw_w = float(gpu_parts[4]) if gpu_parts[4] != '[Not Supported]' else 0
        
        return {
            'utilization_percent': utilization,
            'memory_used_mb': memory_used_mb,
            'memory_total_mb': memory_total_mb,
            'temperature_celsius': temperature_c,
            'power_draw_watts': power_draw_w
        }
        
    except Exception as e:
        logger.debug(f"GPU 정보 수집 실패: {e}")
        return None

def update_reliable_gpu_metrics():
    """신뢰할 수 있는 GPU 메트릭만 업데이트"""
    try:
        gpu_info = get_reliable_gpu_info()
        
        if gpu_info is None:
            logger.debug("GPU 정보 수집 실패, 메트릭 업데이트 건너뜀")
            return
        
        # ✅ 확실한 전체 GPU 메트릭만 업데이트
        team5_gpu_utilization.labels(service='whisper-stt').set(gpu_info['utilization_percent'])
        team5_gpu_memory_used.labels(service='whisper-stt').set(gpu_info['memory_used_mb'])
        
        # 📊 의미 있는 변화만 로깅
        if gpu_info['utilization_percent'] > 10:
            logger.info(
                f"🖥️ GPU 활성 상태: {gpu_info['utilization_percent']}% 사용률, "
                f"{gpu_info['memory_used_mb']:.0f}MB/{gpu_info['memory_total_mb']:.0f}MB 메모리"
            )
        else:
            # 유휴 상태 로깅 (덜 빈번하게)
            if hasattr(update_reliable_gpu_metrics, '_idle_counter'):
                update_reliable_gpu_metrics._idle_counter += 1
            else:
                update_reliable_gpu_metrics._idle_counter = 1
                
            if update_reliable_gpu_metrics._idle_counter % 6 == 0:  # 1분마다
                logger.info(f"🏠 GPU 유휴 상태: {gpu_info['utilization_percent']}% 사용률")
                
    except Exception as e:
        logger.error(f"GPU 메트릭 업데이트 실패: {e}")

# lifespan 함수 (주기적 모니터링)
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI 애플리케이션 시작 - 신뢰할 수 있는 GPU 모니터링 시작")
    
    async def periodic_gpu_monitoring():
        monitor_count = 0
        while True:
            try:
                monitor_count += 1
                update_reliable_gpu_metrics()
                
                if monitor_count % 6 == 0:  # 1분마다
                    logger.info(f"✅ 신뢰성 기반 GPU 모니터링 {monitor_count}회 완료")
                    
            except Exception as e:
                logger.debug(f"주기적 GPU 모니터링 에러: {e}")
            await asyncio.sleep(10)  # 10초마다 실행
    
    gpu_task = asyncio.create_task(periodic_gpu_monitoring())
    logger.info("🎯 백그라운드 GPU 모니터링 태스크 생성됨")
    
    yield
    
    # 앱 종료 시 태스크 취소
    logger.info("🛑 FastAPI 애플리케이션 종료 - GPU 모니터링 중단")
    gpu_task.cancel()

# --- FastAPI 앱 생성 ---
app = FastAPI(
    title="Whisper STT API Server",
    description="음성 파일을 텍스트로 변환하는 API (신뢰할 수 있는 GPU 모니터링)",
    version="3.1.0",
    lifespan=lifespan
)

# --- Global Exception Handler for Logging ---
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# GPU 모니터링 미들웨어 (확실한 메트릭만)
@app.middleware("http")
async def reliable_gpu_monitoring_middleware(request: Request, call_next):
    if request.url.path == "/upload-audio":
        start_time = time.time()
        response = await call_next(request)
        processing_time = time.time() - start_time
        
        # 백그라운드에서 비동기 메트릭 업데이트
        asyncio.create_task(update_metrics_background(processing_time))
        return response
    else:
        return await call_next(request)

async def update_metrics_background(processing_time: float):
    """백그라운드 메트릭 업데이트 (확실한 메트릭만)"""
    try:
        # ✅ 확실한 STT 메트릭
        team5_stt_requests.labels(service='whisper-stt').inc()
        team5_stt_duration.labels(service='whisper-stt').observe(processing_time)
        
        # ✅ 확실한 GPU 메트릭 (즉시 업데이트)
        update_reliable_gpu_metrics()
        
    except Exception as e:
        logger.debug(f"백그라운드 메트릭 업데이트 실패: {e}")

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

UPLOAD_DIR = "temp_audio"
LOGS_DIR = "logs"
SERVICE_NAME = "whisper-stt-server"
BATCH_SIZE = 16
NUM_WORKERS = min(4, os.cpu_count() or 4)

# --- 로깅 설정 ---
logger = logging.getLogger(SERVICE_NAME)
logger.setLevel(logging.INFO)
logger.propagate = False

os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, "whisper_server.log")

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        if not log_record.get('@timestamp'):
            log_record['@timestamp'] = logging.Formatter().formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S.%fZ')
        if record.levelname:
            log_record['log.level'] = record.levelname.upper()
        else:
            log_record['log.level'] = 'INFO'
        log_record['service.name'] = SERVICE_NAME

formatter = CustomJsonFormatter()

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# --- 모델 로드 ---
base_log_extra_model_load = {"event.module": "initialization", "event.action": "load_model"}
logger.info(
    f"🚀 Whisper 모델 로딩 시작: '{MODEL_SIZE}' on '{DEVICE}' with '{COMPUTE_TYPE}'",
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
    
    # 모델 로딩 완료 후 신뢰할 수 있는 GPU 상태 확인
    logger.info("🎯 신뢰할 수 있는 GPU 모니터링 시작")
    update_reliable_gpu_metrics()
    
    logger.info("✅ Whisper 모델 로딩 완료!", extra=base_log_extra_model_load)
except Exception as e:
    logger.critical(f"❌ Whisper 모델 로딩 실패: {e}", exc_info=True, extra=base_log_extra_model_load)
    model = None
    base_model = None

# --- 배치 처리 함수 ---
def process_audio(audio_path: str, request_id: str, batch_size: int = BATCH_SIZE, language: str = None, initial_prompt: Optional[str] = None):
    """배치 처리를 활용해 오디오 파일을 한 번에 처리하는 함수. 요청 ID를 받아 로깅에 활용."""
    try:
        with wave.open(audio_path, 'rb') as wf:
            audio_duration_sec = wf.getnframes() / wf.getframerate()
    except Exception:
        audio_duration_sec = None

    log_extra_base = {
        "service.name": "whisper-stt",
        "request.id": request_id,
        "audio.path": os.path.basename(audio_path),
        "stt.batch_size_configured": batch_size,
        "stt.language_requested": language if language else "auto",
        "stt.initial_prompt_provided": bool(initial_prompt),
        "audio.duration_sec": round(audio_duration_sec, 2) if audio_duration_sec else "N/A"
    }
    
    logger.info(f"🎤 STT 배치 처리 시작 (요청 ID: {request_id})", extra=log_extra_base)
    start_time = time.time()

    try:
        # decode_options 구성
        decode_options = {
            "language": language,
            "word_timestamps": True,
            "beam_size": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "task": "transcribe",
            "best_of": 5,
            "temperature": 0
        }
        
        # initial_prompt 설정
        if initial_prompt:
            decode_options["initial_prompt"] = initial_prompt
        elif language == "ko":
            decode_options["initial_prompt"] = "이것은 한국어 비즈니스 회의 녹음입니다. 한국어를 정확하게 전사해주세요."
        else:
            decode_options["initial_prompt"] = "This is a business meeting recording."

        # Filter out None values
        decode_options = {k: v for k, v in decode_options.items() if v is not None}

        if model is None:
            logger.critical("❌ Whisper 모델이 로드되지 않음", extra=log_extra_base)
            raise RuntimeError("Whisper model not available for transcription.")

        # GPU 모니터링 (처리 시작 시)
        update_reliable_gpu_metrics()

        # model.transcribe 호출 시 batch_size 인자 명시적 전달
        segments_iterable, info = model.transcribe(audio_path, batch_size=batch_size, **decode_options)
        segments_list = list(segments_iterable)

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
        
        logger.info("✅ STT 처리 완료", extra=final_log_extra)
        
        # GPU 모니터링 (처리 완료 시)
        update_reliable_gpu_metrics()
        
        return segments_list, info

    except Exception as e:
        elapsed_time_sec = time.time() - start_time
        logger.error(
            "❌ STT 배치 처리 실패",
            exc_info=True,
            extra={**log_extra_base, "error.message_detail": str(e), "stt.processing_time_sec_before_error": round(elapsed_time_sec, 2)}
        )
        raise

@app.post("/upload-audio")
async def upload_audio(
    request: Request, 
    file: UploadFile = File(...), 
    meeting_info: str = Form("N/A"), 
    language: Optional[str] = Form(None), 
    title: str = Form(...), 
    meeting_attendees: List[str] = Form([]), 
    writer: str = Form(...)
):
    """
    오디오 파일을 STT로 변환하여 전체 텍스트를 JSON으로 반환합니다.
    """
    import tempfile
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Construct initial_prompt from form parameters
    if language == "en":
        prompt_attendees_en = ", ".join(meeting_attendees) if meeting_attendees else "No attendee information"
        initial_prompt_text = f"This recording is about the following meeting:\nMeeting Title: {title}\nAttendees: {prompt_attendees_en}\nRecorder: {writer}\n"
    else:  # Default to Korean for 'ko' or if language is not specified (None) or other values
        prompt_attendees_ko = ", ".join(meeting_attendees) if meeting_attendees else "참석자 정보 없음"
        initial_prompt_text = f"이 녹음은 다음 회의에 관한 것입니다:\n회의 제목: {title}\n참석자: {prompt_attendees_ko}\n작성자: {writer}\n"
    
    base_log_extra_upload = {"request_id": request_id, "service.name": "whisper-stt", "api.endpoint": "/upload-audio"}
    logger.info(
        "🎤 Initial prompt 생성 완료",
        extra={
            **base_log_extra_upload,
            "meeting.title": title,
            "meeting.attendees_count": len(meeting_attendees) if meeting_attendees else 0,
            "meeting.writer": writer,
            "initial_prompt.length": len(initial_prompt_text),
            "initial_prompt.language_used": "en" if language == "en" else "ko"
        }
    )

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
        segments, info = process_audio(converted_wav_path, request_id, BATCH_SIZE, language, initial_prompt=initial_prompt_text)
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
    """Root health check"""
    return {"message": "🚀 Whisper STT API Server with Reliable GPU Monitoring is running"}

@app.get("/gpu-status")
async def get_gpu_status():
    """신뢰할 수 있는 GPU 상태 확인 엔드포인트"""
    gpu_info = get_reliable_gpu_info()
    if gpu_info:
        return {
            "status": "available",
            "utilization_percent": gpu_info['utilization_percent'],
            "memory_used_mb": gpu_info['memory_used_mb'],
            "memory_total_mb": gpu_info['memory_total_mb'],
            "memory_usage_ratio_percent": round(gpu_info['memory_used_mb'] / gpu_info['memory_total_mb'] * 100, 1),
            "temperature_celsius": gpu_info['temperature_celsius'],
            "power_draw_watts": gpu_info['power_draw_watts'],
            "note": "This shows total GPU status. Process-level breakdown not available in WSL2 environment."
        }
    else:
        return {"status": "unavailable", "error": "Unable to query GPU information"}

# Prometheus 메트릭 노출
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)