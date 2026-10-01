"""(이식 중) ingest 라우터 자리."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])
