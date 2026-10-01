"""프로젝트(작업공간) 목록·전환·관리."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from vision_ai import config, projects

from .. import state
from ..common import bad_request

router = APIRouter(prefix="/projects", tags=["projects"])


def _listing() -> dict:
    registry = projects.load()
    current = projects.active()
    return {
        "active": current.slug,
        "projects": [
            {"slug": p.slug, "name": p.name, "note": getattr(p, "note", ""),
             "created_at": getattr(p, "created_at", "")}
            for p in registry.projects
        ],
        "data_root": str(config.data_root()),
        "artifact_root": str(config.artifact_root()),
    }


@router.get("")
def list_projects() -> dict:
    return _listing()


class SlugBody(BaseModel):
    slug: str


class NameBody(BaseModel):
    name: str


@router.post("/use")
def use_project(body: SlugBody) -> dict:
    if projects.get(body.slug) is None:
        raise bad_request(f"없는 프로젝트입니다: {body.slug}")
    projects.use(body.slug)
    state.clear()   # 다른 프로젝트의 학습 결과·추론 결과가 남아 있으면 안 된다
    return _listing()


@router.post("")
def create_project(body: NameBody) -> dict:
    made = projects.create(body.name)
    state.clear()
    return {"created": {"slug": made.slug, "name": made.name}, **_listing()}


@router.post("/{slug}/rename")
def rename_project(slug: str, body: NameBody) -> dict:
    projects.rename(slug, body.name)
    return _listing()


@router.delete("/{slug}")
def remove_project(slug: str) -> dict:
    projects.remove(slug)
    state.clear()
    return _listing()


@router.get("/{slug}/summary")
def project_summary(slug: str) -> dict:
    return projects.summary(slug)
