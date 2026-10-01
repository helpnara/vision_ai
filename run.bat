@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem 파이썬을 UTF-8 모드로 돌린다. 한국어 Windows의 기본 인코딩(cp949)이 끼어들지 않게 한다.
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
title 표면 결함 탐지 파이프라인

if not exist ".venv\Scripts\python.exe" (
    echo 먼저 setup.bat 을 실행해 설치하세요.
    pause
    exit /b 1
)

rem 새 판을 덮어써서 requirements.txt 가 바뀌었으면 패키지를 다시 깐다 (같으면 건너뛴다).
fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>&1
if errorlevel 1 (
    echo 의존성이 바뀌었습니다. 패키지를 다시 설치합니다...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo 패키지 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 실행하세요.
        pause
        exit /b 1
    )
    copy /y requirements.txt ".venv\installed-requirements.txt" >nul
)

set "PORT=%~1"
if "%PORT%"=="" set "PORT=8000"

echo ============================================================
echo  http://localhost:%PORT%  — 브라우저가 자동으로 열립니다.
echo  끄려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo  다른 포트로 띄우려면:  run.bat 8501
echo ============================================================
".venv\Scripts\python.exe" serve.py --port %PORT%
if errorlevel 1 (
    echo.
    echo 서버가 오류로 멈췄습니다. 위 메시지를 확인하세요.
    pause
)
