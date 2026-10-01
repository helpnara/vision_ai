"""웹 서버 진입점.

    python serve.py                  # http://localhost:8000 (브라우저가 열린다)
    python serve.py --port 8501      # 포트 바꾸기
    python serve.py --no-browser     # 원격/CI
    python serve.py --reload         # 서버 코드를 고치며 쓸 때

화면(`web/dist`)은 저장소에 빌드된 채로 들어 있으므로 Node 없이 이 파일만 실행하면 된다.
화면 코드를 고쳤을 때만 `cd web && npm run build`가 필요하다.
"""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="표면 결함 탐지 파이프라인 웹 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="브라우저를 열지 않는다")
    parser.add_argument("--reload", action="store_true", help="코드 변경 시 자동 재시작 (개발용)")
    args = parser.parse_args()

    import socket

    import uvicorn

    # 포트가 이미 쓰이고 있으면 uvicorn은 «[WinError 10048]» 한 줄만 남기고 끝난다.
    # 무슨 뜻인지, 무엇을 하면 되는지를 먼저 말해 준다 (앱을 두 번 띄운 경우가 가장 흔하다).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if probe.connect_ex((args.host, args.port)) == 0:
            print(
                f"포트 {args.port}을(를) 이미 다른 프로그램이 쓰고 있습니다. "
                f"앱이 이미 켜져 있다면 브라우저에서 http://localhost:{args.port} 을 여세요. "
                f"다른 포트로 띄우려면:  run.bat 8501  (또는 python serve.py --port 8501)"
            )
            raise SystemExit(1)

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        # 서버가 뜬 뒤에 열어야 «연결할 수 없음»이 먼저 뜨지 않는다.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"→ {url}")
    uvicorn.run("server.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
