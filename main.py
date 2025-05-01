from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from faster_whisper import WhisperModel
import requests
import io
from fastapi.middleware.cors import CORSMiddleware
import torch
import os
import uuid
import time
import shutil
import logging
import subprocess

# --- 설정 ---
MODEL_SIZE = "small"
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
UPLOAD_DIR = "temp_audio"

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 모델 로드 ---
logger.info(f"Loading faster-whisper model '{MODEL_SIZE}' on device '{DEVICE}' with compute_type '{COMPUTE_TYPE}'...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

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

# --- 서버2 URL (회의 정보 포함하여 POST하도록 수정) ---
SERVER2_URL = "http://192.168.1.244:8001/chunk-summarize-search"

@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...), meeting_info: str = Form(...)):
    """
    오디오 파일을 받아 WAV 변환 후 STT를 수행하고,
    변환된 텍스트와 회의 정보를 서버2로 POST 요청을 보냅니다.
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
        logger.info(f"[{request_id}] Starting transcription for '{stt_input_path}'...")
        start_time = time.time()
        segments, info = model.transcribe(stt_input_path, language="ko", beam_size=5, vad_filter=True)
        end_time = time.time()
        processing_time = end_time - start_time
        full_text = "\n".join([segment.text.strip() for segment in segments])
        logger.info(f"[{request_id}] Transcription finished in {processing_time:.2f} seconds. Language: {info.language}")

        logger.info(f"[{request_id}] Sending STT result and meeting info to '{SERVER2_URL}'...")
        try:
            response_server2 = requests.post(
                SERVER2_URL,
                json={"text": full_text, "meeting_info": meeting_info},
                timeout=10
            )
            response_server2.raise_for_status()
            server2_result = response_server2.json()
            logger.info(f"[{request_id}] Server2 response: {server2_result}")
            return {"message": "STT 변환 및 서버2로 전송 완료", "server2_result": server2_result, "stt_text": full_text}
        except requests.exceptions.RequestException as e:
            logger.error(f"[{request_id}] Error sending data to Server2: {e}")
            return {"error": f"서버2 요청 실패: {e}", "stt_text": full_text}

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
        if 'file' in locals() and file:
            await file.close()

@app.get("/")
async def read_root():
    return {"message": "Faster-Whisper STT API (Server 1) is running"}