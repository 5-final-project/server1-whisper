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
import math

# GPU 메트릭 정의
team5_gpu_utilization = Gauge('team5_gpu_utilization_percent', 'Team5 GPU utilization', ['service'])
team5_gpu_memory_used = Gauge('team5_gpu_memory_used_mb', 'Team5 GPU memory used', ['service'])
team5_stt_requests = Counter('team5_stt_requests_total', 'Total STT requests', ['service'])
team5_stt_duration = Histogram('team5_stt_processing_seconds', 'STT processing time', ['service'])

# 프로세스별 GPU 메트릭 (추정 기반)
team5_process_gpu_memory = Gauge('team5_process_gpu_memory_mb', 'Estimated process GPU memory', ['service', 'container_id'])
team5_process_gpu_utilization = Gauge('team5_process_gpu_utilization_percent', 'Estimated process GPU utilization', ['service', 'container_id'])

# GPU 메모리 추적기 (Windows WSL2 환경 대응)
gpu_memory_tracker = {
    'baseline_memory_mb': 0,  # 모델 로딩 시 베이스라인
    'peak_memory_mb': 0,      # 최대 사용량
    'last_high_usage_time': 0,  # 마지막 고사용량 시간
    'estimated_per_process_mb': 0,  # 추정 프로세스별 메모리
    'whisper_process_count': 2,  # 기본 Whisper 프로세스 수
    'model_loaded': False
}

# GPU 핸들 캐싱
_gpu_handle = None
_gpu_handle_lock = threading.Lock()

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

def get_cached_gpu_handle():
    """GPU 핸들 캐시 (재초기화 최소화)"""
    global _gpu_handle
    if _gpu_handle is None:
        with _gpu_handle_lock:
            if _gpu_handle is None:
                try:
                    pynvml.nvmlInit()
                    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                except Exception as e:
                    logger.debug(f"GPU 핸들 초기화 실패: {e}")
                    return None
    return _gpu_handle

def get_gpu_info_enhanced():
    """Windows WSL2 환경에 최적화된 GPU 정보 수집"""
    global gpu_memory_tracker
    
    try:
        handle = get_cached_gpu_handle()
        if handle is None:
            return None
            
        # 1. 전체 GPU 상태 수집
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_mb = mem.used / 1024 / 1024
        
        # 2. 베이스라인 메모리 설정 (모델 로딩 완료 후)
        if not gpu_memory_tracker['model_loaded'] and total_memory_mb > 1000:
            gpu_memory_tracker['baseline_memory_mb'] = total_memory_mb
            gpu_memory_tracker['model_loaded'] = True
            logger.info(f"🎯 GPU 베이스라인 메모리 설정: {total_memory_mb:.1f}MB")
        
        # 3. GPU 프로세스 카운트 (실제 확인)
        whisper_process_count = count_whisper_processes()
        if whisper_process_count > 0:
            gpu_memory_tracker['whisper_process_count'] = whisper_process_count
        
        # 4. 프로세스별 메모리 추정
        estimated_process_memory = estimate_process_gpu_memory_smart(
            total_memory_mb, util.gpu
        )
        
        # 5. 피크 메모리 추적
        if total_memory_mb > gpu_memory_tracker['peak_memory_mb']:
            gpu_memory_tracker['peak_memory_mb'] = total_memory_mb
        
        return {
            'total_utilization': util.gpu,
            'total_memory_mb': total_memory_mb,
            'estimated_process_memory_mb': estimated_process_memory,
            'process_count': gpu_memory_tracker['whisper_process_count'],
            'container_id': get_container_id(),
            'baseline_memory_mb': gpu_memory_tracker['baseline_memory_mb'],
            'peak_memory_mb': gpu_memory_tracker['peak_memory_mb']
        }
        
    except Exception as e:
        logger.debug(f"GPU 정보 수집 실패: {e}")
        return None

def count_whisper_processes():
    """Whisper 관련 프로세스 수 카운트"""
    try:
        whisper_count = 0
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                name = proc.info['name'].lower()
                
                # Whisper/FastAPI 관련 프로세스 탐지
                keywords = ['uvicorn', 'main.py', 'whisper', 'fastapi', 'faster-whisper']
                if any(keyword in cmdline.lower() or keyword in name for keyword in keywords):
                    whisper_count += 1
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        return max(whisper_count, 1)  # 최소 1개는 보장
        
    except Exception:
        return 2  # 기본값

def estimate_process_gpu_memory_smart(total_memory_mb: float, gpu_utilization: int) -> float:
    """실시간 반응하는 프로세스별 GPU 메모리 추정"""
    global gpu_memory_tracker
    
    current_time = time.time()
    baseline = gpu_memory_tracker['baseline_memory_mb']
    process_count = max(gpu_memory_tracker['whisper_process_count'], 1)
    
    # 🔥 방법 1: 높은 GPU 사용률 (50% 이상) - 실시간 반응
    if gpu_utilization >= 50:
        # 베이스라인 + 사용률에 비례한 동적 메모리
        if baseline > 0:
            # 사용률에 따른 추가 메모리 계산
            utilization_factor = gpu_utilization / 100.0
            dynamic_memory = baseline * (0.3 + utilization_factor * 0.7)  # 30% ~ 100%
            
            # 전체 메모리 고려
            max_dynamic = total_memory_mb * (0.4 + utilization_factor * 0.4)  # 40% ~ 80%
            estimated = min(dynamic_memory, max_dynamic)
        else:
            # 베이스라인이 없으면 사용률 기반으로만
            estimated = total_memory_mb * (0.3 + gpu_utilization / 100.0 * 0.5)  # 30% ~ 80%
        
        # 값 저장 및 시간 기록
        gpu_memory_tracker['estimated_per_process_mb'] = estimated
        gpu_memory_tracker['last_high_usage_time'] = current_time
        
        logger.info(f"🔥 고사용률 ({gpu_utilization}%): 프로세스 메모리 {estimated:.1f}MB 추정")
        return estimated
    
    # ⚡ 방법 2: 중간 GPU 사용률 (20-49%) - 점진적 증가
    elif gpu_utilization >= 20:
        if baseline > 0:
            # 사용률에 비례한 점진적 증가
            utilization_factor = gpu_utilization / 100.0
            estimated = baseline * (0.25 + utilization_factor * 0.5)  # 25% ~ 75%
        else:
            estimated = total_memory_mb * (0.2 + utilization_factor * 0.4)  # 20% ~ 60%
        
        gpu_memory_tracker['estimated_per_process_mb'] = estimated
        gpu_memory_tracker['last_high_usage_time'] = current_time
        
        logger.info(f"⚡ 중간사용률 ({gpu_utilization}%): 프로세스 메모리 {estimated:.1f}MB 추정")
        return estimated
    
    # 📊 방법 3: 최근 고사용률의 감쇠 (5분간)
    time_since_high = current_time - gpu_memory_tracker['last_high_usage_time']
    if time_since_high < 300:  # 5분 이내
        recent_estimate = gpu_memory_tracker['estimated_per_process_mb']
        if recent_estimate > 0:
            # 시간에 따른 지수적 감쇠 (빠른 감소)
            decay_factor = max(0.2, math.exp(-time_since_high / 120))  # 2분 시상수
            decayed_estimate = recent_estimate * decay_factor
            
            logger.debug(f"📊 감쇠 적용: {decayed_estimate:.1f}MB (원래: {recent_estimate:.1f}MB, {time_since_high:.0f}초 전)")
            return decayed_estimate
    
    # 🏠 방법 4: 유휴 상태 베이스라인 (GPU < 20%)
    if baseline > 0:
        # 🔧 유휴 상태에서는 베이스라인의 20-30% 사용
        idle_ratio = 0.2 + (gpu_utilization / 100.0) * 0.1  # 20% ~ 30%
        idle_estimate = baseline * idle_ratio
        
        # 전체 메모리의 40%를 넘지 않도록
        max_idle = total_memory_mb * 0.4
        idle_estimate = min(idle_estimate, max_idle)
        
        logger.debug(f"🏠 유휴상태 ({gpu_utilization}%): {idle_estimate:.1f}MB (베이스라인 {idle_ratio*100:.0f}%)")
        return idle_estimate
    
    # 🎯 방법 5: 최종 기본값 (베이스라인 없음)
    if total_memory_mb > 1000:
        # GPU 사용률에 따른 기본 추정
        base_ratio = 0.15 + (gpu_utilization / 100.0) * 0.25  # 15% ~ 40%
        default_estimate = total_memory_mb * base_ratio
        
        logger.debug(f"🎯 기본 추정 ({gpu_utilization}%): {default_estimate:.1f}MB")
        return default_estimate
    
    return 0.0


def try_nvidia_smi_parsing() -> float:
    """nvidia-smi 파싱으로 프로세스 메모리 획득 시도"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=3
        )
        
        if result.returncode == 0 and result.stdout.strip():
            total_memory = 0
            process_count = 0
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    try:
                        parts = line.split(', ')
                        if len(parts) >= 2 and parts[1] != '[Not Supported]':
                            memory_mb = float(parts[1])
                            total_memory += memory_mb
                            process_count += 1
                    except (ValueError, IndexError):
                        continue
            
            if process_count > 0:
                return total_memory / process_count
                
    except Exception as e:
        logger.debug(f"nvidia-smi 파싱 실패: {e}")
    
    return 0.0

def update_realistic_gpu_metrics():
    """실시간 반응 GPU 메트릭 업데이트"""
    try:
        gpu_info = get_gpu_info_enhanced()
        
        if gpu_info is None:
            logger.debug("GPU 정보 수집 실패")
            return
        
        # 전체 GPU 메트릭
        team5_gpu_utilization.labels(service='whisper-stt').set(gpu_info['total_utilization'])
        team5_gpu_memory_used.labels(service='whisper-stt').set(gpu_info['total_memory_mb'])
        
        # 🔥 실시간 반응하는 프로세스별 메트릭
        estimated_memory = gpu_info['estimated_process_memory_mb']
        
        # 프로세스 사용률도 실시간 계산
        gpu_util = gpu_info['total_utilization']
        if gpu_util >= 50:
            # 고사용률: 프로세스가 GPU의 60-80% 책임
            estimated_utilization = min(gpu_util * 0.7, 100)
        elif gpu_util >= 20:
            # 중간사용률: 프로세스가 GPU의 40-60% 책임
            estimated_utilization = min(gpu_util * 0.5, 100)
        else:
            # 저사용률: 프로세스가 GPU의 20-40% 책임
            estimated_utilization = min(gpu_util * 0.3, 100)
        
        # 최소값 보장
        if estimated_memory > 500:  # 500MB 이상이면 최소 사용률 보장
            estimated_utilization = max(estimated_utilization, 3)
        
        # 메트릭 업데이트
        team5_process_gpu_memory.labels(
            service='whisper-stt', 
            container_id=gpu_info['container_id']
        ).set(estimated_memory)
        
        team5_process_gpu_utilization.labels(
            service='whisper-stt', 
            container_id=gpu_info['container_id']
        ).set(estimated_utilization)
        
        # 변화 감지 로깅
        memory_ratio = estimated_memory / gpu_info['total_memory_mb'] if gpu_info['total_memory_mb'] > 0 else 0
        
        # GPU 사용률이 높거나 메모리 변화가 클 때 로깅
        if gpu_util > 30 or abs(estimated_memory - gpu_memory_tracker.get('last_logged_memory', 0)) > 500:
            logger.info(
                f"🎯 실시간 GPU 추적: "
                f"전체={gpu_info['total_memory_mb']:.1f}MB({gpu_util}%), "
                f"프로세스={estimated_memory:.1f}MB({estimated_utilization:.1f}%), "
                f"비율={memory_ratio*100:.1f}%"
            )
            gpu_memory_tracker['last_logged_memory'] = estimated_memory
            
    except Exception as e:
        logger.error(f"GPU 메트릭 업데이트 실패: {e}")
# lifespan 함수 (주기적 모니터링)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 백그라운드 GPU 모니터링 시작
    async def periodic_gpu_monitoring():
        while True:
            try:
                update_realistic_gpu_metrics()
            except Exception as e:
                logger.debug(f"주기적 GPU 모니터링 에러: {e}")
            await asyncio.sleep(10)  # 10초마다 실행
    
    gpu_task = asyncio.create_task(periodic_gpu_monitoring())
    yield
    # 앱 종료 시 태스크 취소
    gpu_task.cancel()

# --- FastAPI 앱 생성 ---
app = FastAPI(
    title="Whisper STT API Server",
    description="음성 파일을 텍스트로 변환하는 API (Windows WSL2 GPU 모니터링 최적화)",
    version="2.0.0",
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
        update_realistic_gpu_metrics()
        
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
logger.setLevel(logging.INFO)  # INFO로 변경 (DEBUG는 너무 많은 로그)
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
    
    # 모델 로딩 완료 후 GPU 상태 확인
    update_realistic_gpu_metrics()
    
    logger.info("✅ Whisper 모델 로딩 완료!", extra=base_log_extra_model_load)
except Exception as e:
    logger.critical(f"❌ Whisper 모델 로딩 실패: {e}", exc_info=True, extra=base_log_extra_model_load)
    model = None
    base_model = None

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
        update_realistic_gpu_metrics()

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
        update_realistic_gpu_metrics()
        
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
    return {"message": "🚀 Whisper STT API Server with Enhanced GPU Monitoring is running"}

@app.get("/gpu-status")
def get_gpu_info_enhanced():
    """향상된 GPU 정보 수집 (현실성 검증 추가)"""
    global gpu_memory_tracker
    
    try:
        handle = get_cached_gpu_handle()
        if handle is None:
            return None
            
        # 1. 전체 GPU 상태 수집
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_mb = mem.used / 1024 / 1024
        
        # 2. 베이스라인 메모리 설정 (더 보수적)
        if not gpu_memory_tracker['model_loaded'] and total_memory_mb > 1000:
            gpu_memory_tracker['baseline_memory_mb'] = total_memory_mb
            gpu_memory_tracker['model_loaded'] = True
            logger.info(f"🎯 GPU 베이스라인 메모리 설정: {total_memory_mb:.1f}MB")
        
        # 3. GPU 프로세스 카운트
        whisper_process_count = count_whisper_processes()
        if whisper_process_count > 0:
            gpu_memory_tracker['whisper_process_count'] = whisper_process_count
        
        # 4. 프로세스별 메모리 추정 (현실적 제한)
        estimated_process_memory = estimate_process_gpu_memory_smart(
            total_memory_mb, util.gpu
        )
        
        # 🔧 최종 안전장치: 프로세스 메모리가 전체의 80%를 넘지 않도록
        max_process_memory = total_memory_mb * 0.8
        if estimated_process_memory > max_process_memory:
            logger.warning(f"⚠️ 프로세스 메모리 추정값이 너무 높음: {estimated_process_memory:.1f}MB -> {max_process_memory:.1f}MB로 제한")
            estimated_process_memory = max_process_memory
        
        # 5. 피크 메모리 추적
        if total_memory_mb > gpu_memory_tracker['peak_memory_mb']:
            gpu_memory_tracker['peak_memory_mb'] = total_memory_mb
        
        return {
            'total_utilization': util.gpu,
            'total_memory_mb': total_memory_mb,
            'estimated_process_memory_mb': estimated_process_memory,
            'process_count': gpu_memory_tracker['whisper_process_count'],
            'container_id': get_container_id(),
            'baseline_memory_mb': gpu_memory_tracker['baseline_memory_mb'],
            'peak_memory_mb': gpu_memory_tracker['peak_memory_mb'],
            'memory_ratio': estimated_process_memory / total_memory_mb if total_memory_mb > 0 else 0
        }
        
    except Exception as e:
        logger.debug(f"GPU 정보 수집 실패: {e}")
        return None


# Prometheus 메트릭 노출
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
##