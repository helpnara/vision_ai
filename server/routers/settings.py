"""설정: 판정 기준 · 현장 가정 · 재학습 기준 · 캐시."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter
from pydantic import BaseModel

from vision_ai import config, evaluate, feature_cache, patch_cache, settings

from ..common import jsonable

router = APIRouter(prefix="/settings", tags=["settings"])

# 설정 미리보기에 대입하는 VisA 실측값. 슬라이더 숫자만으로는 결과가 안 보인다.
MEASURED_RECALL = 0.925
MEASURED_FPR = 0.455


def _preview(candidate: settings.Settings) -> dict:
    impact = evaluate.business_impact(
        MEASURED_RECALL, MEASURED_FPR, prevalence=candidate.prevalence, volume=candidate.volume
    )
    passes_recall = MEASURED_RECALL >= candidate.target_recall
    passes_reduction = impact["reduction_ratio"] >= candidate.min_reduction
    missing = []
    if not passes_recall:
        missing.append(f"재현율 {MEASURED_RECALL:.1%} < 목표 {candidate.target_recall:.0%}")
    if not passes_reduction:
        missing.append(f"검수량 절감 {impact['reduction_ratio']:.0%} < 기준 {candidate.min_reduction:.0%}")
    return jsonable({
        "measured_recall": MEASURED_RECALL, "measured_fpr": MEASURED_FPR,
        "impact": impact, "passes": passes_recall and passes_reduction, "missing": missing,
    })


def _payload() -> dict:
    current = settings.load()
    diff = settings.changed(current)
    return jsonable({
        "values": asdict(current),
        "defaults": asdict(settings.DEFAULTS),
        "bounds": {k: list(v) for k, v in settings.BOUNDS.items()},
        "labels": settings.LABELS,
        "help": settings.HELP,
        "changed": {k: {"value": v[0], "default": v[1]} for k, v in diff.items()},
        "path": str(config.data_root() / settings.SETTINGS_FILE),
        "preview": _preview(current),
        "cache": {
            "features": feature_cache.summary() | {"path": str(feature_cache.cache_path())},
            "patches": patch_cache.summary(),
        },
    })


@router.get("")
def get_settings() -> dict:
    return _payload()


class SettingsBody(BaseModel):
    target_recall: float
    min_reduction: float
    prevalence: float
    volume: int
    new_label_threshold: int
    recall_margin: float


@router.post("/preview")
def preview(body: SettingsBody) -> dict:
    return _preview(settings.Settings(**body.model_dump()))


@router.put("")
def save_settings(body: SettingsBody) -> dict:
    candidate = settings.Settings(**body.model_dump())
    stored = settings.save(candidate)
    return {"clamped": stored != candidate, **_payload()}


@router.post("/reset")
def reset_settings() -> dict:
    settings.reset()
    return _payload()


@router.post("/cache/clear")
def clear_feature_cache() -> dict:
    feature_cache.clear()
    return _payload()


@router.post("/cache/patches/clear")
def clear_patch_cache() -> dict:
    patch_cache.clear()
    return _payload()
