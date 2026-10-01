#!/usr/bin/env bash
# macOS / Linux 용 — Windows 는 setup.bat 을 더블클릭한다.
set -e
cd "$(dirname "$0")"
export PYTHONUTF8=1
echo "[1/3] 파이썬 확인"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then echo "파이썬 3.10 이상을 설치하세요."; exit 1; fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { echo "파이썬 3.10 이상이 필요합니다: $("$PY" --version)"; exit 1; }
"$PY" --version
echo "[2/3] 가상환경 만들기 (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
echo "[3/3] 패키지 설치 (처음에는 몇 분 걸립니다)"
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -r requirements.txt
cp requirements.txt .venv/installed-requirements.txt
echo "설치 완료. ./run.sh 로 실행하세요."
