import requests
import os
import json # JSON 파싱 보기 좋게 하기 위해 추가

# --- 클라이언트 설정 ---
# FastAPI 서버 주소와 포트 (★★★★★ 중요: 실제 서버 주소로 변경해야 함 ★★★★★)
SERVER_URL = "http://192.168.0.61:8000/upload-audio" # <--- /upload-audio 엔드포인트로 수정!

# 변환할 오디오 파일 경로 (★★★★★ 중요: 실제 오디오 파일 경로로 변경해야 함 ★★★★★)
AUDIO_FILE_PATH = "회의록_테스트.wav" # <--- 이 부분을 실제 파일 경로로 수정하세요!

# --- 회의 정보 추가 ---
MEETING_INFO = "2025년 5월 1일 팀 회의 내용" # 실제 회의 정보를 넣어주세요!

# --- 오디오 파일 존재 확인 ---
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"오류: 오디오 파일을 찾을 수 없습니다 - {AUDIO_FILE_PATH}")
else:
    # --- 파일 열기 및 요청 보내기 ---
    try:
        # 오디오 파일을 바이너리 읽기 모드('rb')로 열기
        with open(AUDIO_FILE_PATH, "rb") as audio_file:
            files = {'file': (os.path.basename(AUDIO_FILE_PATH), audio_file, 'audio/wav')}
            data = {'meeting_info': MEETING_INFO} # Form 데이터에 회의 정보 추가

            print(f"'{AUDIO_FILE_PATH}' 파일을 서버({SERVER_URL})로 전송하여 STT 요청 중...")
            response = requests.post(SERVER_URL, files=files, data=data, timeout=600) # data 파라미터 추가

            # --- 응답 처리 ---
            response.raise_for_status()
            print("요청 성공!")
            result = response.json()

            print("\n--- 서버 응답 ---")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # 최종 결과를 JSON 파일로 저장 (예시)
            output_file_path = "final_response.json"
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\n최종 결과가 '{output_file_path}'에 저장되었습니다.")

    except requests.exceptions.ConnectionError as e:
        print(f"오류: 서버({SERVER_URL})에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요. - {e}")
    except requests.exceptions.Timeout as e:
        print(f"오류: 서버 응답 시간 초과 - {e}")
    except requests.exceptions.RequestException as e:
        print(f"서버 요청 중 오류 발생: {e}")
    except Exception as e:
        print(f"처리 중 예외 발생: {e}")