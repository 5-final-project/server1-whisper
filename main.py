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
# 기존 코드 맨 위에 추가
import pynvml
from prometheus_client import Gauge, Counter, Histogram
# 새로 추가되는 import들
import psutil
import asyncio
from contextlib import asynccontextmanager
import threading

# 기존 GPU 메트릭 정의
team5_gpu_utilization = Gauge('team5_gpu_utilization_percent', 'Team5 GPU utilization', ['service'])
team5_gpu_memory_used = Gauge('team5_gpu_memory_used_mb', 'Team5 GPU memory used', ['service'])
team5_stt_requests = Counter('team5_stt_requests_total', 'Total STT requests', ['service'])
team5_stt_duration = Histogram('team5_stt_processing_seconds', 'STT processing time', ['service'])

# 새로운 프로세스별 GPU 메트릭 정의
team5_process_gpu_memory = Gauge('team5_process_gpu_memory_mb', 'Process-specific GPU memory usage', ['service', 'container_id'])
team5_process_gpu_utilization = Gauge('team5_process_gpu_utilization_percent', 'Process-specific GPU utilization estimation', ['service', 'container_id'])

# GPU 핸들 캐싱을 위한 전역 변수
_gpu_handle = None
_gpu_handle_lock = threading.Lock()

def get_container_id():
    """현재 실행 중인 컨테이너 ID 획득"""
    try:
        with open('/proc/self/cgroup', 'r') as f:
            for line in f:
                if 'docker' in line:
                    return line.strip().split('/')[-1][:12]
        return 'unknown'
    except:
        return 'unknown'

def get_cached_gpu_handle():
    """GPU 핸들을 캐시하여 매번 초기화하지 않음"""
    global _gpu_handle
    if _gpu_handle is None:
        with _gpu_handle_lock:
            if _gpu_handle is None:
                try:
                    pynvml.nvmlInit()
                    _gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                except Exception as e:
                    return None
    return _gpu_handle

def get_whisper_process_gpu_usage():
    """Whisper 프로세스의 GPU 사용량 조회"""
    try:
        handle = get_cached_gpu_handle()
        if handle is None:
            return {'memory_mb': 0, 'utilization_percent': 0, 'container_id': get_container_id()}
        
        current_pid = os.getpid()
        container_id = get_container_id()
        
        # GPU에서 실행 중인 모든 프로세스 조회
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        process_memory = 0
        found_process = False
        
        for proc in processes:
            try:
                # 현재 프로세스인지 확인
                if proc.pid == current_pid:
                    process_memory = proc.usedGpuMemory / 1024 / 1024  # MB로 변환
                    found_process = True
                    break
                    
                # 같은 컨테이너의 Python 프로세스 확인
                try:
                    proc_info = psutil.Process(proc.pid)
                    cmdline = ' '.join(proc_info.cmdline()).lower()
                    if ('python' in proc_info.name().lower() and 
                        ('whisper' in cmdline or 'main.py' in cmdline)):
                        process_memory += proc.usedGpuMemory / 1024 / 1024
                        found_process = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
            except Exception:
                continue
        
        # GPU 사용률 추정
        estimated_utilization = 0
        if found_process and process_memory > 0:
            total_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            if total_memory.used > 0:
                memory_ratio = process_memory / (total_memory.used / 1024 / 1024)
                estimated_utilization = min(total_util.gpu * memory_ratio, 100)
            
        return {
            'memory_mb': process_memory,
            'utilization_percent': estimated_utilization,
            'container_id': container_id
        }
            
    except Exception as e:
        return {
            'memory_mb': 0,
            'utilization_percent': 0,
            'container_id': get_container_id()
        }

def get_gpu_metrics_fast():
    """최적화된 전체 GPU 메트릭 수집 - 캐시된 핸들 사용"""
    handle = get_cached_gpu_handle()
    if handle is None:
        return None
    
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'utilization': util.gpu,
            'memory_used': mem.used / 1024 / 1024
        }
    except Exception as e:
        # 핸들 무효화하여 다음에 재초기화
        global _gpu_handle
        _gpu_handle = None
        return None

async def update_process_gpu_metrics_background(processing_time):
    """백그라운드에서 프로세스별 GPU 메트릭 업데이트"""
    try:
        # 기본 메트릭 (빠른 처리)
        team5_stt_requests.labels(service='whisper-stt').inc()
        team5_stt_duration.labels(service='whisper-stt').observe(processing_time)
        
        # 프로세스별 GPU 메트릭 (느릴 수 있음)
        process_metrics = get_whisper_process_gpu_usage()
        
        team5_process_gpu_memory.labels(
            service='whisper-stt', 
            container_id=process_metrics['container_id']
        ).set(process_metrics['memory_mb'])
        
        team5_process_gpu_utilization.labels(
            service='whisper-stt', 
            container_id=process_metrics['container_id']
        ).set(process_metrics['utilization_percent'])
        
        # 전체 GPU 메트릭도 유지 (기존 대시보드 호환성)
        gpu_metrics = get_gpu_metrics_fast()
        if gpu_metrics:
            team5_gpu_utilization.labels(service='whisper-stt').set(gpu_metrics['utilization'])
            team5_gpu_memory_used.labels(service='whisper-stt').set(gpu_metrics['memory_used'])
            
    except Exception as e:
        pass  # 백그라운드 작업이므로 에러 무시

# 주기적 프로세스 모니터링
async def periodic_gpu_monitoring():
    """5초마다 프로세스 GPU 사용량 업데이트"""
    while True:
        try:
            process_metrics = get_whisper_process_gpu_usage()
            
            team5_process_gpu_memory.labels(
                service='whisper-stt', 
                container_id=process_metrics['container_id']
            ).set(process_metrics['memory_mb'])
            
            team5_process_gpu_utilization.labels(
                service='whisper-stt', 
                container_id=process_metrics['container_id']
            ).set(process_metrics['utilization_percent'])
            
        except Exception as e:
            pass  # 주기적 작업이므로 에러 무시
        
        await asyncio.sleep(5)

# lifespan 함수 추가
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 백그라운드 GPU 모니터링 시작
    gpu_task = asyncio.create_task(periodic_gpu_monitoring())
    yield
    # 앱 종료 시 태스크 취소
    gpu_task.cancel()

# --- FastAPI 앱 생성 및 CORS 설정 ---
app = FastAPI(
    title="Whisper STT API Server",
    description="음성 파일을 텍스트로 변환하는 API (프로세스별 GPU 모니터링)",
    version="1.0.0",
    lifespan=lifespan  # lifespan 추가
)

# --- Global Exception Handler for Logging ---
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

# 최적화된 GPU 모니터링 미들웨어 (백그라운드 처리)
@app.middleware("http")
async def optimized_gpu_monitoring_middleware(request: Request, call_next):
    if request.url.path == "/upload-audio":
        start_time = time.time()
        response = await call_next(request)  # 먼저 응답 처리
        processing_time = time.time() - start_time
        
        # 백그라운드에서 비동기 실행 - 응답 즉시 반환!
        asyncio.create_task(update_process_gpu_metrics_background(processing_time))
        return response
    else:
        return await call_next(request)

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
def process_audio(audio_path: str, request_id: str, batch_size: int = BATCH_SIZE, language: Optional[str] = None, initial_prompt: Optional[str] = None):
    """배치 처리를 활용해 오디오 파일을 한 번에 처리하는 함수. 요청 ID를 받아 로깅에 활용."""
    start_time = time.time()
    audio_duration_sec = 0 # 초기화

    log_extra_base = {
        "service.name": "whisper-stt",
        "request.id": request_id,
        "audio.path": os.path.basename(audio_path),
        "stt.batch_size_configured": batch_size,
        "stt.language_requested": language if language else "auto",
        "stt.initial_prompt_provided": bool(initial_prompt),
    }

    try:
        logger.info(f"Starting STT batch processing for request {request_id}", extra=log_extra_base)

        # 오디오 파일 길이 가져오기 (로깅 및 통계용)
        try:
            with wave.open(audio_path, 'rb') as wf:
                audio_duration_sec = wf.getnframes() / wf.getframerate()
        except Exception:
            audio_duration_sec = None

        log_extra_base["audio.duration_sec"] = round(audio_duration_sec, 2) if audio_duration_sec else "N/A"

        # transcription_options 를 decode_options 로 명칭 변경하여 역할 명확화
        decode_options = {
            "language": language,
            "word_timestamps": True, # 세그먼트 시간 정보 포함
            # "condition_on_previous_text": False, # 테스트용: 문맥 의존성 낮춤
            # "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), # 다양성 증가 (기본값)
        }
        if initial_prompt:
            decode_options["initial_prompt"] = initial_prompt

        # Filter out None values to avoid passing them explicitly if they are meant to be default
        decode_options = {k: v for k, v in decode_options.items() if v is not None}

        if not model: # 모델 로드 실패 시 예외 발생
            logger.critical(
                "Whisper model is not loaded. Cannot perform transcription.",
                extra=log_extra_base
            )
            raise RuntimeError("Whisper model not available for transcription.")

        # model.transcribe 호출 시 batch_size 인자 명시적 전달
        segments_iterable, info = model.transcribe(audio_path, batch_size=batch_size, **decode_options)
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

@app.post("/upload-audio")
async def upload_audio(request: Request, file: UploadFile = File(...), meeting_info: str = Form("N/A"), language: Optional[str] = Form(None), title: str = Form(...), meeting_attendees: List[str] = Form([]), writer: str = Form(...)):
    """
    오디오 파일을 STT로 변환하여 전체 텍스트를 JSON 으로 반환합니다.
    """
    import tempfile
    start_time = time.time()
    # 요청 ID 가져오기
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Construct initial_prompt from new form parameters
    if language == "en":
        prompt_attendees_en = ", ".join(meeting_attendees) if meeting_attendees else "No attendee information"
        initial_prompt_text = f"This recording is about the following meeting:\nMeeting Title: {title}\nAttendees: {prompt_attendees_en}\nRecorder: {writer}\n"
    else:  # Default to Korean for 'ko' or if language is not specified (None) or other values
        prompt_attendees_ko = ", ".join(meeting_attendees) if meeting_attendees else "참석자 정보 없음"
        initial_prompt_text = f"이 녹음은 다음 회의에 관한 것입니다:\n회의 제목: {title}\n참석자: {prompt_attendees_ko}\n작성자: {writer}\n"
    
    base_log_extra_upload = {"request_id": request_id, "service.name": "whisper-stt", "api.endpoint": "/upload-audio"}
    logger.info(
        "Initial prompt constructed for STT.",
        extra={
            **base_log_extra_upload,
            "meeting.title": title,
            "meeting.attendees_count": len(meeting_attendees) if meeting_attendees else 0,
            "meeting.writer": writer,
            "initial_prompt.length": len(initial_prompt_text),
            "initial_prompt.language_used": "en" if language == "en" else "ko" # 프롬프트 언어 로깅
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
    """
    Root health check.
    """
    return {"message": "Whisper STT API Server with BatchedInferencePipeline is running"}

from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)