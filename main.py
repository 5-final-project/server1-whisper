from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from faster_whisper import WhisperModel, BatchedInferencePipeline
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import uuid
import time
import shutil
import logging
import subprocess
import math

# --- 설정 ---
MODEL_SIZE = "medium"
if torch.cuda.is_available():
  DEVICE = "cuda"
  COMPUTE_TYPE = "float16"
else:
  DEVICE = "cpu"
  COMPUTE_TYPE = "int8"
UPLOAD_DIR = "temp_audio"
BATCH_SIZE = 16 
NUM_WORKERS = min(4, os.cpu_count() or 4)

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 모델 로드 ---
logger.info(f"Loading model '{MODEL_SIZE}' on device '{DEVICE}' with compute_type '{COMPUTE_TYPE}'...")
try:
  # 기본 모델 로드
  base_model = WhisperModel(
    MODEL_SIZE, 
    device=DEVICE, 
    compute_type=COMPUTE_TYPE,
    cpu_threads=NUM_WORKERS,  # CPU 스레드 수 지정
  )
  
  # BatchedInferencePipeline을 사용하여 모델 래핑
  model = BatchedInferencePipeline(model=base_model)
  logger.info("Whisper model with batched pipeline loaded successfully.")
except Exception as e:
  logger.error(f"Error loading model: {e}")
  model = None
  base_model = None

# --- 배치 처리 함수 ---
def process_audio(audio_path, batch_size=BATCH_SIZE, language=None):
    """배치 처리를 활용해 오디오 파일을 한 번에 처리하는 함수"""
    try:
        logger.info(f"Processing audio file: {audio_path} with batch_size={batch_size}")
        start_time = time.time()
        
        # BatchedInferencePipeline을 통한 배치 처리
        
        # 기본 옵션 설정 - 모든 언어에 대해 높은 정확도 옵션 적용
        transcription_options = {
            "beam_size": 5,
            "vad_filter": True,
            "word_timestamps": True,
            "condition_on_previous_text": True,
            "batch_size": batch_size,
            "task": "transcribe",  # 전체 스크립트 형식
            "best_of": 5,         # 더 정확한 결과를 위한 beam search
            "temperature": 0,       # 정확도 상실 방지
            "initial_prompt" : """This is a business meeting recording."""
        }
        
        if language:
            transcription_options["language"] = language
            
            # 한국어인 경우 추가 설정
            if language == "ko":
                # 한국어 프롬프트 적용
                transcription_options["initial_prompt"] = """This is a Korean business meeting recording. The primary language is Korean with occasional English terms or phrases. Please accurately transcribe the Korean speech while maintaining English terms that might be present. Focus on properly capturing Korean sentences with their natural flow and structure."""
                logger.info(f"Transcribing with Korean language settings and prompt")
            else:
                logger.info(f"Transcribing with specified language: {language}")
        else:
            logger.info("Transcribing with automatic language detection")
        
        # 모델에 옵션 전달
        segments, info = model.transcribe(
            audio_path,
            **transcription_options
        )
        
        # BatchedInferencePipeline이 이미 내부적으로 병렬화 처리 제공
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # segments는 제너레이터이므로 리스트로 변환
        segments_list = list(segments)
        
        logger.info(f"Transcription completed in {processing_time:.2f} seconds. "  
                  f"Detected language: {info.language}, Segments: {len(segments_list)}")
        
        return segments_list, info
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}", exc_info=True)
        raise

# --- FastAPI 앱 생성 ---
app = FastAPI()

# --- CORS 설정 ---
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
)

# --- 임시 디렉토리 생성 ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), meeting_info: str = Form(...), language: str = Form(None)):
  """
  오디오 파일을 STT로 변환하여 전체 텍스트를 JSON 으로 반환합니다.
  • Hub(WebSocket 서버)가 후속 API2/3/4를 호출합니다.
  """
  if model is None:
    raise HTTPException(status_code=503, detail="Model is not available")

  request_id = str(uuid.uuid4())
  original_filename = f"{request_id}{os.path.splitext(file.filename)[1]}"
  original_filepath = os.path.join(UPLOAD_DIR, original_filename)
  converted_wav_path = None

  try:
    logger.info(f"[{request_id}] Receiving file: {file.filename}, meeting_info: '{meeting_info}'")
    with open(original_filepath, "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
    logger.info(f"[{request_id}] File saved successfully: {original_filepath}")

    converted_wav_filename = f"{request_id}_converted.wav"
    converted_wav_path = os.path.join(UPLOAD_DIR, converted_wav_filename)
    logger.info(f"[{request_id}] Converting to WAV (16kHz, Mono) -> '{converted_wav_path}'")

    try:
      command = ["ffmpeg", "-y", "-i", original_filepath, "-ar", "16000", "-ac", "1", "-vn", converted_wav_path]
      process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
      if not os.path.exists(converted_wav_path):
        logger.error(f"[{request_id}] ffmpeg output missing: {converted_wav_path}")
        raise RuntimeError("Audio conversion failed: output file missing.")
      logger.info(f"[{request_id}] Conversion successful: '{converted_wav_path}'")
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
      logger.error(f"[{request_id}] ffmpeg error: {e}")
      raise HTTPException(status_code=400, detail=f"Audio conversion failed: {e}")

    stt_input_path = converted_wav_path
    logger.info(f"[{request_id}] Starting chunked transcription for '{stt_input_path}'...")
    start_time = time.time()
    
    # BatchedInferencePipeline을 활용한 배치 처리 (language 매개변수 전달)
    segments, info = process_audio(stt_input_path, batch_size=BATCH_SIZE, language=language)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 모든 세그먼트를 타임스탬프 순서대로 정렬
    sorted_segments = sorted(segments, key=lambda s: s.start)
    full_text = "\n".join([segment.text.strip() for segment in sorted_segments])
    
    logger.info(f"[{request_id}] Transcription finished in {processing_time:.2f} seconds. Language: {info.language}, Chunks processed: {math.ceil(len(sorted_segments) / 3)}")

    return {
      "text": full_text,
      "meeting_info": meeting_info,
      "processing_time_sec": round(processing_time, 2),
    }

  except HTTPException as he:
    raise he
  except Exception as e:
    logger.error(f"[{request_id or 'N/A'}] Unhandled exception: {e}", exc_info=True)
    raise HTTPException(status_code=500, detail="Internal server error during processing.")

  finally:
    # 임시 파일 삭제
    if os.path.exists(original_filepath):
      try:
        os.remove(original_filepath)
        logger.info(f"[{request_id or 'N/A'}] Deleted original file: {original_filepath}")
      except Exception as e_del:
        logger.error(f"[{request_id or 'N/A'}] Error deleting original file {original_filepath}: {e_del}")
    if converted_wav_path and os.path.exists(converted_wav_path):
      try:
        os.remove(converted_wav_path)
        logger.info(f"[{request_id or 'N/A'}] Deleted converted file: {converted_wav_path}")
      except Exception as e_del:
        logger.error(f"[{request_id or 'N/A'}] Error deleting converted file {converted_wav_path}: {e_del}")
    # 파일 객체 닫기 (안전하게)
    if "file" in locals() and file:
      await file.close()

@app.get("/")
async def read_root():
  return {"message": "Whisper STT API Server with BatchedInferencePipeline is running"}