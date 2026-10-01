"""백그라운드 잡 조회."""

from __future__ import annotations

from fastapi import APIRouter

from ..common import not_found
from ..jobs import runner

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = runner.get(job_id)
    if job is None:
        raise not_found(f"잡을 찾을 수 없습니다: {job_id}")
    return job.as_dict()
