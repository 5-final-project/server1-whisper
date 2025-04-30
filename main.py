import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from faster_whisper import WhisperModel
import os
import uuid  # 고유 파일 이름 생성을 위해
import time
import shutil # 파일 저장을 위해
import logging # 로깅 추가
import subprocess
# --- 설정 ---
MODEL_SIZE = "large-v3"
# 장치 설정 (GPU 우선, 없으면 CPU)
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_TYPE = "float16"
else:
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"
# 임시 파일 저장 경로
UPLOAD_DIR = "temp_audio"

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 모델 로드 ---
# 서버 시작 시 한 번만 로드하여 효율성 증대
logger.info(f"Loading faster-whisper model '{MODEL_SIZE}' on device '{DEVICE}' with compute_type '{COMPUTE_TYPE}'...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # 실제 운영 환경에서는 모델 로드 실패 시 서버를 시작하지 않거나
    # 적절한 오류 처리가 필요합니다. 여기서는 None으로 설정합니다.
    model = None

# --- FastAPI 앱 생성 ---
app = FastAPI()

# --- 임시 디렉토리 생성 ---
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 엔드포인트 정의 ---
# main.py 파일 내부에 이 함수를 통째로 교체하거나 수정하세요.

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    오디오 파일을 받아 WAV(16kHz, Mono)로 변환 후,
    faster-whisper large-v3 모델로 STT를 수행하고 결과를 반환합니다.
    """
    if model is None:
        # 서버 시작 시 모델 로딩 실패한 경우
        raise HTTPException(status_code=503, detail="Model is not available")

    # 1. 원본 파일 임시 저장
    # 고유 ID를 사용하여 파일 이름 충돌 방지 및 추적 용이
    request_id = str(uuid.uuid4())
    original_filename = f"{request_id}{os.path.splitext(file.filename)[1]}"
    original_filepath = os.path.join(UPLOAD_DIR, original_filename)

    converted_wav_path = None # 변환된 파일 경로 초기화 (finally에서 사용하기 위함)

    try:
        # 원본 파일 저장
        logger.info(f"[{request_id}] Receiving file: {file.filename}, saving to {original_filepath}")
        with open(original_filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"[{request_id}] File saved successfully: {original_filepath}")

        # 2. 표준 WAV 포맷으로 변환 (16kHz, Mono)
        converted_wav_filename = f"{request_id}_converted.wav"
        converted_wav_path = os.path.join(UPLOAD_DIR, converted_wav_filename)
        logger.info(f"[{request_id}] Converting '{original_filepath}' to WAV (16kHz, Mono) -> '{converted_wav_path}'")

        try:
            # ffmpeg 명령어 실행
            command = [
                "ffmpeg", "-y",             # 덮어쓰기 허용
                "-i", original_filepath,    # 입력: 원본 파일
                "-ar", "16000",             # 오디오 샘플링 레이트 16kHz
                "-ac", "1",                 # 오디오 채널 1 (모노)
                "-vn",                      # 비디오 스트림 무시
                converted_wav_path          # 출력: 변환된 WAV 파일
            ]
            # 타임아웃 설정 (예: 5분). 긴 파일 처리 시 필요에 따라 조절
            process = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)

            # 변환된 파일이 실제로 생성되었는지 확인
            if not os.path.exists(converted_wav_path):
                 # 이 경우는 ffmpeg 명령은 성공했으나 파일이 없는 드문 케이스
                 logger.error(f"[{request_id}] ffmpeg command ran but output file missing: {converted_wav_path}")
                 raise RuntimeError("Audio conversion seemed successful but output file is missing.")
            logger.info(f"[{request_id}] Conversion successful: '{converted_wav_path}'")

        # ffmpeg 관련 오류 처리
        except subprocess.TimeoutExpired:
             logger.error(f"[{request_id}] ffmpeg conversion timed out for {original_filepath}")
             raise HTTPException(status_code=408, detail="Audio conversion process timed out.")
        except subprocess.CalledProcessError as e:
            # ffmpeg 명령 자체가 오류를 반환한 경우
            logger.error(f"[{request_id}] ffmpeg conversion failed for {original_filepath}. Return code: {e.returncode}, Error: {e.stderr}")
            # 사용자에게는 간단한 오류 메시지 전달, 로그에는 상세 내용 기록
            raise HTTPException(status_code=400, detail=f"Failed to convert audio file. It might be corrupted or in an unsupported format.")
        except FileNotFoundError:
            # 서버 환경에 ffmpeg가 설치되지 않은 경우
             logger.error("'ffmpeg' command not found. Ensure ffmpeg is installed on the server environment.")
             raise HTTPException(status_code=500, detail="Server configuration error: ffmpeg is missing.")
        except Exception as e:
            # 기타 변환 중 예외 처리
             logger.error(f"[{request_id}] Unexpected error during audio conversion: {e}")
             raise HTTPException(status_code=500, detail=f"Unexpected error during audio processing.")

        # 3. STT 수행 (★★★★★ 변환된 WAV 파일 사용 ★★★★★)
        stt_input_path = converted_wav_path # STT에는 변환된 파일을 사용
        logger.info(f"[{request_id}] Starting transcription for CONVERTED file {stt_input_path} using model '{MODEL_SIZE}'...")
        start_time = time.time()

        segments, info = model.transcribe(
            stt_input_path,
            language="ko",
            beam_size=5,
            vad_filter=True
        )

        end_time = time.time()
        processing_time = end_time - start_time
        logger.info(f"[{request_id}] Transcription finished in {processing_time:.2f} seconds. Detected language: {info.language}")

        # 4. 결과 처리
        results = []
        full_text = []
        # segments는 제너레이터이므로 반복 처리 필요
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text.append(segment.text.strip())

        logger.info(f"[{request_id}] Transcription result processing complete.")

        # 5. 결과 반환
        return {
            "request_id": request_id, # 추적을 위한 ID 포함
            "detected_language": info.language,
            "language_probability": info.language_probability,
            "processing_time_seconds": processing_time,
            "full_text": "\n".join(full_text),
            "segments": results
        }

    except Exception as e:
        # 전체 요청 처리 중 예상치 못한 오류 발생 시
        logger.error(f"[{request_id or 'N/A'}] Unhandled exception during /transcribe: {e}", exc_info=True)
        # 이미 HTTPException으로 처리된 경우는 그대로 전달, 아닌 경우는 500 에러 발생
        if not isinstance(e, HTTPException):
             raise HTTPException(status_code=500, detail="An internal server error occurred.")
        else:
             raise e

    finally:
        # 6. 임시 파일 삭제 (try 블록 성공/실패 여부와 관계없이 항상 실행)
        # 원본 업로드 파일 삭제
        if os.path.exists(original_filepath):
            try:
                os.remove(original_filepath)
                logger.info(f"[{request_id or 'N/A'}] Original temporary file deleted: {original_filepath}")
            except Exception as e_del:
                logger.error(f"[{request_id or 'N/A'}] Error deleting original temporary file {original_filepath}: {e_del}")
        # 변환된 WAV 파일 삭제 (생성된 경우에만)
        if converted_wav_path and os.path.exists(converted_wav_path):
            try:
                os.remove(converted_wav_path)
                logger.info(f"[{request_id or 'N/A'}] Converted temporary file deleted: {converted_wav_path}")
            except Exception as e_del:
                logger.error(f"[{request_id or 'N/A'}] Error deleting converted temporary file {converted_wav_path}: {e_del}")
        # 업로드 파일 객체 닫기
        await file.close()

# (선택 사항) 서버 상태 확인용 기본 엔드포인트
@app.get("/")
async def read_root():
    return {"message": "Faster-Whisper STT API is running with large-v3 model!"}