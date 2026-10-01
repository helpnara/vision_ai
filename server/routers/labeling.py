"""(이식 중) labeling 라우터 자리."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/labeling", tags=["labeling"])
