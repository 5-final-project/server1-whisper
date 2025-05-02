#!/usr/bin/env python3
"""
STT‑FASTAPI 클라이언트

- 오디오 파일과 회의 정보를 Server‑1(/upload‑audio) 엔드포인트로 전송
- 응답(JSON)을 예쁘게 출력하고, 원하면 파일로 저장
- 명령줄 인자로 서버 주소·오디오 파일·회의 정보를 지정할 수 있음
"""

import argparse
import json
import os
import sys
import requests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whisper STT‑FASTAPI 클라이언트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8100/upload-audio",
        help="Server‑1 /upload‑audio 엔드포인트 URL (기본값: localhost:8100)",
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        help="전송할 오디오 파일 경로 (wav/mp3 등)",
    )
    parser.add_argument(
        "--meeting-info",
        default="회의 제목 또는 메타정보",
        help="회의 정보(제목·날짜 등) 텍스트",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="요청 타임아웃(초)",
    )
    parser.add_argument(
        "--save-json",
        metavar="FILE",
        help="응답 JSON을 저장할 경로 (지정 안 하면 저장하지 않음)",
    )
    args = parser.parse_args()

    # --- 파일 존재 확인 ---
    if not os.path.exists(args.audio_file):
        sys.exit(f"오디오 파일을 찾을 수 없습니다: {args.audio_file}")

    try:
        with open(args.audio_file, "rb") as audio_fp:
            files = {
                "file": (
                    os.path.basename(args.audio_file),
                    audio_fp,
                    "audio/wav",
                )
            }
            data = {"meeting_info": args.meeting_info}

            print(
                f"'{args.audio_file}' 전송 → {args.server_url} (timeout={args.timeout}s)"
            )
            resp = requests.post(
                args.server_url, files=files, data=data, timeout=args.timeout
            )
            resp.raise_for_status()

            result = resp.json()
            print("\n요청 성공! --- 서버 응답 ---")
            print(json.dumps(result, indent=2, ensure_ascii=False))

            if args.save_json:
                with open(args.save_json, "w", encoding="utf-8") as fp:
                    json.dump(result, fp, indent=2, ensure_ascii=False)
                print(f"\n결과 JSON 저장 완료 → {args.save_json}")

    except requests.exceptions.Timeout:
        sys.exit("오류: 서버 응답 시간 초과")
    except requests.exceptions.ConnectionError:
        sys.exit("오류: 서버에 연결할 수 없습니다. 주소·포트·방화벽 확인")
    except requests.exceptions.RequestException as e:
        sys.exit(f"서버 요청 오류: {e}")
    except Exception as e:
        sys.exit(f"처리 중 예외 발생: {e}")


if __name__ == "__main__":
    main()