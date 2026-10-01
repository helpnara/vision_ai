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
