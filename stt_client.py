import requests
import os
import json # JSON 파싱 보기 좋게 하기 위해 추가

# --- 클라이언트 설정 ---
# FastAPI 서버 주소와 포트 (★★★★★ 중요: 실제 서버 주소로 변경해야 함 ★★★★★)
# 예: 서버가 IP 192.168.0.10 에서 실행 중이면 "http://192.168.0.10:8000/transcribe"
# 예: 서버와 클라이언트가 같은 컴퓨터면 "http://127.0.0.1:8000/transcribe"
SERVER_URL = "http://127.0.0.1:8000/transcribe" # <--- 이 부분을 서버 주소에 맞게 수정하세요!

# 변환할 오디오 파일 경로 (★★★★★ 중요: 실제 오디오 파일 경로로 변경해야 함 ★★★★★)
AUDIO_FILE_PATH = "회의록_테스트.wav" # <--- 이 부분을 실제 파일 경로로 수정하세요!

# --- 오디오 파일 존재 확인 ---
if not os.path.exists(AUDIO_FILE_PATH):
    print(f"오류: 오디오 파일을 찾을 수 없습니다 - {AUDIO_FILE_PATH}")
else:
    # --- 파일 열기 및 요청 보내기 ---
    try:
        # 오디오 파일을 바이너리 읽기 모드('rb')로 열기
        with open(AUDIO_FILE_PATH, "rb") as audio_file:
            # 서버의 FastAPI 엔드포인트에서 정의한 파라미터 이름('file')과 동일하게 파일 지정
            # 'files' 딕셔너리: {'form_field_name': (filename, file_object, content_type)}
            files = {'file': (os.path.basename(AUDIO_FILE_PATH), audio_file, 'audio/wav')} # 필요시 content_type 변경

            print(f"'{AUDIO_FILE_PATH}' 파일을 서버({SERVER_URL})로 전송하여 STT 요청 중...")
            # 서버에 POST 요청 전송 (timeout 설정 권장)
            response = requests.post(SERVER_URL, files=files, timeout=600) # 10분 타임아웃

            # --- 응답 처리 ---
            # HTTP 오류(4xx, 5xx)가 발생했는지 확인하고, 오류 시 예외 발생
            response.raise_for_status()

            # 성공 시 (2xx 상태 코드)
            print("요청 성공!")
            result = response.json() # 서버로부터 받은 JSON 응답 파싱

            print("\n--- STT 결과 ---")
            # JSON 데이터 보기 좋게 출력
            print(json.dumps(result, indent=2, ensure_ascii=False))

            # # 필요한 정보만 따로 출력할 수도 있습니다.
            # print(f"\n감지된 언어: {result.get('detected_language')} (확률: {result.get('language_probability'):.2f})")
            # print(f"처리 시간: {result.get('processing_time_seconds'):.2f} 초")
            # print("\n[전체 텍스트]")
            # print(result.get('full_text'))

    except requests.exceptions.ConnectionError:
        print(f"오류: 서버({SERVER_URL})에 연결할 수 없습니다. 서버가 실행 중인지, 주소가 정확한지 확인하세요.")
    except requests.exceptions.Timeout:
        print("오류: 서버 응답 시간 초과 (Timeout)")
    except requests.exceptions.RequestException as e:
        print(f"서버 요청 중 오류 발생: {e}")
    except Exception as e:
        print(f"처리 중 예외 발생: {e}")