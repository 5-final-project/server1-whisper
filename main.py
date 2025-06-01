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
import pynvml
from prometheus_client import Gauge, Counter, Histogram
import psutil
import asyncio
from contextlib import asynccontextmanager
import threading
import re

# GPU 메트릭 정의
team5_gpu_utilization = Gauge('team5_gpu_utilization_percent', 'Team5 GPU utilization', ['service'])
team5_gpu_memory_used = Gauge('team5_gpu_memory_used_mb', 'Team5 GPU memory used', ['service'])
team5_stt_requests = Counter('team5_stt_requests_total', 'Total STT requests', ['service'])
team5_stt_duration = Histogram('team5_stt_processing_seconds', 'STT processing time', ['service'])

# 프로세스별 GPU 메트릭 (정확한 측정 기반)
team5_process_gpu_memory = Gauge('team5_process_gpu_memory_mb', 'Estimated process GPU memory', ['service', 'container_id'])
team5_process_gpu_utilization = Gauge('team5_process_gpu_utilization_percent', 'Estimated process GPU utilization', ['service', 'container_id'])

def get_container_id():
    """컨테이너 ID 획득 (여러 방법 시도)"""
    try:
        # 방법 1: /proc/self/cgroup
        with open('/proc/self/cgroup', 'r') as f:
            for line in f:
                if 'docker' in line:
                    container_id = line.strip().split('/')[-1][:12]
                    if len(container_id) >= 12:
                        return container_id
        
        # 방법 2: hostname
        with open('/etc/hostname', 'r') as f:
            hostname = f.read().strip()
            if len(hostname) >= 12:
                return hostname[:12]
                
        # 방법 3: 환경변수
        return os.environ.get('HOSTNAME', 'unknown')[:12]
        
    except:
        return 'unknown'

def get_accurate_gpu_info():
    """nvidia-smi를 직접 사용한 정확한 GPU 정보 수집"""
    try:
        # 1. 전체 GPU 상태 (utilization, memory)
        gpu_query = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if gpu_query.returncode != 0:
            return None
            
        gpu_line = gpu_query.stdout.strip()
        if not gpu_line:
            return None
            
        # GPU 상태 파싱
        gpu_parts = [x.strip() for x in gpu_line.split(',')]
        total_utilization = int(gpu_parts[0]) if gpu_parts[0] != '[Not Supported]' else 0
        total_memory_mb = float(gpu_parts[1]) if gpu_parts[1] != '[Not Supported]' else 0
        total_memory_capacity = float(gpu_parts[2]) if gpu_parts[2] != '[Not Supported]' else 10240
        temperature = float(gpu_parts[3]) if gpu_parts[3] != '[Not Supported]' else 0
        power_draw = float(gpu_parts[4]) if gpu_parts[4] != '[Not Supported]' else 0
        
        # 2. 프로세스별 정보 (PID, 메모리 사용량)
        process_query = subprocess.run([
            'nvidia-smi',
            '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        whisper_processes = []
        total_process_memory = 0
        
        if process_query.returncode == 0 and process_query.stdout.strip():
            for line in process_query.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        parts = [x.strip() for x in line.split(',')]
                        if len(parts) >= 2 and parts[1] != '[Not Supported]':
                            pid = int(parts[0])
                            memory_mb = float(parts[1])
                            
                            # PID로 프로세스 정보 확인
                            if is_whisper_process(pid):
                                whisper_processes.append({
                                    'pid': pid,
                                    'memory_mb': memory_mb
                                })
                            total_process_memory += memory_mb
                    except (ValueError, IndexError):
                        continue
        
        # 3. Whisper 프로세스 정보 계산
        whisper_memory_mb = sum(p['memory_mb'] for p in whisper_processes)
        whisper_process_count = len(whisper_processes)
        
        # 4. Whisper 프로세스 GPU 사용률 추정 (메모리 비율 기반)
        if whisper_memory_mb > 0 and total_memory_mb > 0:
            # 메모리 비율 기반 사용률 추정
            memory_ratio = whisper_memory_mb / total_memory_mb
            estimated_whisper_utilization = total_utilization * memory_ratio
        else:
            estimated_whisper_utilization = 0
        
        return {
            'total_utilization': total_utilization,
            'total_memory_mb': total_memory_mb,
            'total_memory_capacity_mb': total_memory_capacity,
            'temperature_c': temperature,
            'power_draw_w': power_draw,
            'whisper_processes': whisper_processes,
            'whisper_memory_mb': whisper_memory_mb,
            'whisper_process_count': max(whisper_process_count, 1),  # 최소 1개
            'estimated_whisper_utilization': round(estimated_whisper_utilization, 1),
            'container_id': get_container_id()
        }
        
    except Exception as e:
        logger.debug(f"nvidia-smi GPU 정보 수집 실패: {e}")
        return None

def is_whisper_process(pid):
    """PID가 Whisper 관련 프로세스인지 확인"""
    try:
        proc = psutil.Process(pid)
        cmdline = ' '.join(proc.cmdline()).lower()
        name = proc.name().lower()
        
        # Whisper/FastAPI 관련 키워드
        whisper_keywords = [
            'whisper', 'fastapi', 'uvicorn', 'main.py',
            'faster-whisper', 'whisper-stt', 'stt'
        ]
        
        return any(keyword in cmdline or keyword in name for keyword in whisper_keywords)
        
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False

def update_accurate_gpu_metrics():
    """정확한 GPU 메트릭 업데이트"""
    try:
        gpu_info = get_accurate_gpu_info()
        
        if gpu_info is None:
            logger.debug("GPU 정보 수집 실패, 메트릭 업데이트 건너뜀")
            return
        
        # 1. 전체 GPU 메트릭 (nvidia-smi 직접 값)
        team5_gpu_utilization.labels(service='whisper-stt').set(gpu_info['total_utilization'])
        team5_gpu_memory_used.labels(service='whisper-stt').set(gpu_info['total_memory_mb'])
        
        # 2. Whisper 프로세스별 메트릭 (실제 측정값)
        whisper_memory = gpu_info['whisper_memory_mb']
        whisper_utilization = gpu_info['estimated_whisper_utilization']
        
        team5_process_gpu_memory.labels(
            service='whisper-stt', 
            container_id=gpu_info['container_id']
        ).set(whisper_memory)
        
        team5_process_gpu_utilization.labels(
            service='whisper-stt', 
            container_id=gpu_info['container_id']
        ).set(whisper_utilization)
        
        # 3. 로깅 (중요한 변화만)
        if gpu_info['total_utilization'] > 5 or whisper_memory > 500:
            logger.info(
                f"🎯 실시간 GPU 추적: "
                f"전체={gpu_info['total_memory_mb']:.1f}MB({gpu_info['total_utilization']}%), "
                f"프로세스={whisper_memory:.1f}MB({whisper_utilization}%), "
                f"비율={whisper_memory/gpu_info['total_memory_mb']*100:.1f}%"
            )
        else:
            # 유휴 상태 로깅 (덜 빈번하게)
            if hasattr(update_accurate_gpu_metrics, '_idle_log_counter'):
                update_accurate_gpu_metrics._idle_log_counter += 1
            else:
                update_accurate_gpu_metrics._idle_log_counter = 1
                
            if update_accurate_gpu_metrics._idle_log_counter % 6 == 0:  # 1분마다 (10초 * 6)
                logger.info(f"🏠 유휴상태 ({gpu_info['total_utilization']}%): {whisper_memory:.1f}MB (베이스라인 {gpu_info['total_memory_mb']:.1f}MB의 {whisper_memory/gpu_info['total_memory_mb']*100:.1f}%)")
                
    except Exception as e:
        logger.error(f"GPU 메트릭 업데이트 실패: {e}")

# lifespan 함수 (주기적 모니터링)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 백그라운드 GPU 모니터링 시작
    logger.info("🚀 FastAPI 애플리케이션 시작 - 백그라운드 GPU 모니터링 시작")
    
    async def periodic_gpu_monitoring():
        monitor_count = 0
        while True:
            try:
                monitor_count += 1
                update_accurate_gpu_metrics()
                
                if monitor_count % 6 == 0:  # 1분마다
                    logger.info(f"✅ GPU 모니터링 {monitor_count}회 완료")
                    
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
    description="음성 파일을 텍스트로 변환하는 API (정확한 GPU 모니터링)",
    version="3.0.0",
    lifespan=lifespan
)

# --- Global Exception Handler for Logging ---
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 최적화된 GPU 모니터링 미들웨어
@app.middleware("http")
async def enhanced_gpu_monitoring_middleware(request: Request, call_next):
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
    """백그라운드 메트릭 업데이트"""
    try:
        # 기본 STT 메트릭
        team5_stt_requests.labels(service='whisper-stt').inc()
        team5_stt_duration.labels(service='whisper-stt').observe(processing_time)
        
        # GPU 메트릭 (즉시 업데이트)
        update_accurate_gpu_metrics()
        
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
    
    # 모델 로딩 완료 후 정확한 GPU 상태 확인
    logger.info("🎯 GPU 베이스라인 메모리 설정: 모델 로딩 완료")
    update_accurate_gpu_metrics()
    
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

        # GPU 모니터링 강화 (처리 시작 시)
        update_accurate_gpu_metrics()

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
        
        # GPU 모니터링 강화 (처리 완료 시)
        update_accurate_gpu_metrics()
        
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
    return {"message": "🚀 Whisper STT API Server with Accurate GPU Monitoring is running"}

@app.get("/gpu-status")
async def get_gpu_status():
    """개선된 GPU 상태 확인 엔드포인트"""
    gpu_info = get_accurate_gpu_info()
    if gpu_info:
        return {
            "status": "available",
            "total_utilization_percent": gpu_info['total_utilization'],
            "total_memory_used_mb": gpu_info['total_memory_mb'],
            "total_memory_capacity_mb": gpu_info['total_memory_capacity_mb'],
            "temperature_celsius": gpu_info['temperature_c'],
            "power_draw_watts": gpu_info['power_draw_w'],
            "whisper_processes": gpu_info['whisper_processes'],
            "whisper_memory_mb": gpu_info['whisper_memory_mb'],
            "whisper_utilization_percent": gpu_info['estimated_whisper_utilization'],
            "whisper_process_count": gpu_info['whisper_process_count'],
            "container_id": gpu_info['container_id'],
            "memory_usage_ratio": round(gpu_info['total_memory_mb'] / gpu_info['total_memory_capacity_mb'] * 100, 1)
        }
    else:
        return {"status": "unavailable", "error": "Unable to query GPU information"}

# Prometheus 메트릭 노출
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)