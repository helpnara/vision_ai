"""(이식 중) operations 라우터 자리."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/operations", tags=["operations"])
