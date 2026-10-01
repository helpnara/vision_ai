@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 표면 결함 탐지 파이프라인 — 설치

echo ============================================================
echo  표면 결함 탐지 파이프라인 — 처음 한 번 설치 (setup.bat)
echo  이 폴더 안에 가상환경(.venv)을 만들고 파이썬 패키지를 깝니다.
echo  Node.js 는 필요 없습니다 — 화면은 web\dist 에 빌드돼 있습니다.
echo ============================================================
echo.

echo [1/3] 파이썬 확인
set "PY="
py -3 --version >nul 2>&1 && set "PY=py -3"
if not defined PY ( python --version >nul 2>&1 && set "PY=python" )
if not defined PY (
    echo.
    echo   파이썬을 찾지 못했습니다. https://www.python.org/downloads/ 에서 3.10 이상을 설치하세요.
    echo   설치할 때 "Add python.exe to PATH" 를 반드시 켭니다.
    echo.
    pause
    exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo   파이썬 3.10 이상이 필요합니다. 지금 버전:
    %PY% --version
    echo.
    pause
    exit /b 1
)
%PY% --version

echo.
echo [2/3] 가상환경 만들기 (.venv)
if exist ".venv\Scripts\python.exe" (
    echo   이미 있습니다 — 그대로 씁니다.
) else (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   가상환경을 만들지 못했습니다.
        pause
        exit /b 1
    )
)

echo.
echo [3/3] 패키지 설치 (처음에는 몇 분 걸립니다)
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>&1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo   패키지 설치에 실패했습니다. 인터넷 연결을 확인하고 다시 실행하세요.
    pause
    exit /b 1
)
rem run.bat 이 «의존성이 바뀌었는가»를 비교할 기준 사본. 새 판을 덮어쓰면 requirements.txt 만
rem 바뀌고 이 사본은 그대로라, 둘이 다르면 run.bat 이 알아서 다시 설치한다.
copy /y requirements.txt ".venv\installed-requirements.txt" >nul

echo.
echo ============================================================
echo  설치 완료. 이제 run.bat 을 실행하면 브라우저에서 앱이 열립니다.
echo ============================================================
pause
