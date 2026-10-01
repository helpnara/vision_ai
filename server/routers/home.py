"""홈 화면: 데이터 현황 · 진행 순서 · 실측 성능 · 안내문."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from vision_ai import config, datasets, evaluate, glossary, guide, labeling, quickstart, storage

from ..common import jsonable
from ..jobs import runner

router = APIRouter(prefix="/home", tags=["home"])

VALIDATION_PATH = config.PROJECT_ROOT / "docs" / "results" / "visa_validation.json"

# 실제 생산 라인 불량률은 시험 구성(1:1)보다 훨씬 낮다. 그 차이를 눈에 보이게 두려고 함께 쓴다.
ASSUMED_PREVALENCE = 0.01

STAGES = [
    {"no": 1, "name": "데이터 수집 (입력)", "icon": "📥",
     "detail": "오픈 데이터셋 카탈로그 · 로컬 폴더 임포트 · 이미지 업로드 · 영상 프레임 추출 · 합성 샘플 생성 · 품질 점검"},
    {"no": 2, "name": "라벨링", "icon": "🏷️",
     "detail": "라벨 검수 큐 · 영상 구간 라벨링 · 폴더 라벨 검증 · 결함 유형 정규화 · 다중 박스 · 층화 분할"},
    {"no": 3, "name": "모델 개발 · 평가", "icon": "🧠",
     "detail": "베이스라인 · 패치 이상탐지(위치 히트맵) · Claude 2차 판정 · 임계값 조정 · 실험 기록"},
    {"no": 4, "name": "운영 관리 (MLOps)", "icon": "⚙️",
     "detail": "모델 레지스트리(롤백) · 배치 추론 로그 · 판정 영상 · 영상 비교 · 드리프트 감시 · 성능 추이 · 재학습 판단 · 판정 이력"},
]


def _load_validation(path: Path = VALIDATION_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _validation_block() -> dict:
    report = _load_validation()
    if report is None:
        return {"available": False}
    rows = report.get("categories") or []
    scored = [r for r in rows if "recall" in (r.get("held_out") or {})]
    block: dict = {
        "available": True,
        "mean_auroc": report.get("mean_auroc"),
        "mean_ap": report.get("mean_average_precision"),
        "source": f"{report.get('dataset', 'VisA')} · {report.get('protocol', '')} · {report.get('threshold_policy', '')}",
        "categories": [
            {
                "category": r["category"],
                "auroc": r.get("auroc"),
                "ap": r.get("average_precision"),
                "recall": (r.get("held_out") or {}).get("recall"),
                "precision": (r.get("held_out") or {}).get("precision"),
                "false_alarm_rate": (r.get("held_out") or {}).get("false_alarm_rate"),
            }
            for r in rows
        ],
    }
    if scored:
        mean_recall = sum(r["held_out"]["recall"] for r in scored) / len(scored)
        mean_fpr = sum(r["held_out"]["false_alarm_rate"] for r in scored) / len(scored)
        impact = evaluate.business_impact(mean_recall, mean_fpr, prevalence=ASSUMED_PREVALENCE, volume=1000)
        block["mean_recall"] = mean_recall
        block["impact"] = jsonable(impact)
        block["assumed_prevalence"] = ASSUMED_PREVALENCE
    return jsonable(block)


@router.get("/overview")
def overview() -> dict:
    df = storage.load_manifest()
    stats = storage.summarize(df)
    label_stats = labeling.stats(labeling.resolve(df)) if stats["total"] else None
    steps = guide.pipeline_steps(manifest=df)
    done, total = guide.progress(steps)
    upcoming = guide.next_step(steps)
    default = datasets.default_dataset()
    return jsonable({
        "stats": stats,
        "label_stats": label_stats,
        "default_dataset": {"name": default.name, "license": default.license},
        "progress": {"done": done, "total": total},
        "next_step": (
            {"key": upcoming.key, "stage": upcoming.stage, "title": upcoming.title,
             "detail": upcoming.detail, "action": upcoming.action, "where": upcoming.where,
             "route": guide.route_of(upcoming.page)}
            if upcoming else None
        ),
        "steps": [
            {"key": s.key, "stage": s.stage, "title": s.title, "done": s.done,
             "detail": s.detail, "where": s.where, "route": guide.route_of(s.page)}
            for s in steps
        ],
        "stages": STAGES,
        "validation": _validation_block(),
        "data_root": str(config.data_root()),
        "glossary": [{"term": k, "text": v} for k, v in glossary.TERMS.items()],
        "defect_types": [{"key": k, "label": v} for k, v in config.DEFECT_TYPES.items()],
        "quickstart_available": df.empty,
    })


@router.post("/quickstart")
def start_quickstart() -> dict:
    """데모 한 바퀴. 1분쯤 걸리므로 잡으로 돌린다."""
    def run(job):
        result = quickstart.run(progress=job.progress3)
        return {
            "ok": result.ok, "version": result.version, "n_images": result.n_images,
            "n_train": result.n_train, "warnings": list(result.warnings),
        }
    job = runner.submit("데모 만드는 중", run, note="합성 샘플 생성 → 분할 → 학습 → 승격. 1분쯤 걸립니다.")
    return {"job_id": job.id}
