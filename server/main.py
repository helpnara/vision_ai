"""FastAPI 앱. 실행: `python serve.py` (또는 `uvicorn server.main:app`).

`/api/...`는 JSON API, 그 밖의 경로는 `web/dist`의 빌드된 화면을 돌려준다. 화면은
클라이언트 라우팅(React Router)을 쓰므로 모르는 경로는 전부 `index.html`로 보낸다 —
그래야 `/labeling`을 새로고침해도 404가 아니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
_SRC = ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_ai import projects  # noqa: E402

from .routers import files, home, ingest, jobs, labeling, modeling, operations, settings  # noqa: E402
from .routers import projects as projects_router  # noqa: E402

WEB_DIST = ROOT / "web" / "dist"


def create_app() -> FastAPI:
    app = FastAPI(title="표면 결함 탐지 파이프라인", docs_url="/api/docs", openapi_url="/api/openapi.json")

    # 저장된 활성 프로젝트를 실제 경로에 적용한다. 이후 모든 요청이 이 프로젝트를 본다.
    projects.bootstrap()

    for router in (
        projects_router.router, home.router, settings.router, files.router, jobs.router,
        ingest.router, labeling.router, modeling.router, operations.router,
    ):
        app.include_router(router, prefix="/api")

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        # 코어는 잘못된 입력을 ValueError로 말한다. 500이 아니라 400으로 화면에 전한다.
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(OSError)
    async def _os_error(_: Request, exc: OSError) -> JSONResponse:
        # 파일을 못 읽고 못 쓴 것. 그냥 두면 화면에는 «500 Internal Server Error»만 보인다.
        # Windows에서 가장 흔한 원인은 **다른 프로그램이 파일을 쥐고 있는 것**이다 —
        # manifest.csv·labels.csv를 엑셀로 열어 두면 앱이 그 파일에 쓰지 못한다(WinError 32).
        message = str(exc) or exc.__class__.__name__
        if isinstance(exc, PermissionError):
            message = (
                f"파일을 쓸 수 없습니다 — 다른 프로그램(엑셀·탐색기 미리보기·재생 중인 영상 등)이 "
                f"열고 있는지 확인하고 닫은 뒤 다시 시도하세요. ({message})"
            )
        return JSONResponse(status_code=500, content={"detail": message})

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "project": projects.active().slug}

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def no_build() -> JSONResponse:
            return JSONResponse(
                status_code=503,
                content={"detail": "web/dist가 없습니다. `cd web && npm install && npm run build`를 먼저 실행하세요."},
            )

    return app


app = create_app()
