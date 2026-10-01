#!/usr/bin/env bash
# macOS / Linux 용 — Windows 는 run.bat 을 더블클릭한다.
cd "$(dirname "$0")"
export PYTHONUTF8=1
[ -x .venv/bin/python ] || { echo "먼저 ./setup.sh 를 실행하세요."; exit 1; }
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
  echo "의존성이 바뀌었습니다. 패키지를 다시 설치합니다..."
  .venv/bin/python -m pip install -r requirements.txt && cp requirements.txt .venv/installed-requirements.txt
fi
PORT="${1:-8000}"
echo "http://localhost:$PORT — 끄려면 Ctrl+C"
exec .venv/bin/python serve.py --port "$PORT"
