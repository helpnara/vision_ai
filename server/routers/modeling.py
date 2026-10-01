"""(이식 중) modeling 라우터 자리."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/modeling", tags=["modeling"])
