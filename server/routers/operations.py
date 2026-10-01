"""4단계: 사후 운영관리(MLOps) API.

Streamlit 화면(`app_pages/p4_operations.py`)의 9탭을 그대로 옮겼다. 코어를 부르고 JSON으로
바꾸는 얇은 층이며, 판단 로직은 전부 코어(`registry` `serving` `monitoring` `segments`
`compare` `playback` `scenario`)에 있다.

세션 상태(`st.session_state`)에 두던 것 중 다시 계산하면 비싼 것 — 마지막 배치 추론의 특징
행렬(`state.KEY_BATCH`)과 시나리오 결과 객체(`state.KEY_SCENARIO`) — 는 `server.state`에 둔다.
드리프트 감시·재학습 판단은 그 특징 행렬을 다시 계산하지 않고 그대로 쓴다.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vision_ai import (
    charts,
    compare,
    config,
    evaluate,
    experiments,
    features,
    glossary,
    labeling,
    models,
    monitoring,
    playback,
    registry,
    scenario,
    segments,
    serving,
    settings as user_settings,
    storage,
    video,
    viz,
)

from .. import state
from ..common import bad_request, inside, jsonable, not_found, table, typed_path, upload_name
from ..jobs import runner

router = APIRouter(prefix="/operations", tags=["operations"])


# --- 공통 ---------------------------------------------------------------------

def _resolved() -> pd.DataFrame:
    df = labeling.resolve()
    if df.empty:
        return df
    return df.assign(path_abs=[str(storage.resolve_path(p)) for p in df["path"]])


def _row(series: pd.Series | None) -> dict | None:
    return None if series is None else jsonable(series.to_dict())


def _figure(value, digits: int = 2) -> str:
    """잴 수 없는 값을 0으로 적지 않는다 — 0은 «못 잡았다»로 읽힌다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}" if np.isfinite(number) else "—"


def _column_help(frame: pd.DataFrame) -> dict[str, str]:
    """`ui.table_columns`의 자리 — 열 이름을 용어 사전에서 찾아 설명을 붙인다."""
    found = {}
    for name in frame.columns:
        text = glossary.column_help(str(name))
        if text:
            found[str(name)] = text
    return found


def _production_block() -> dict:
    """`_production_banner`의 자리. 화면이 성공/경고 배너를 고른다."""
    prod = registry.production()
    return {
        "production": _row(prod),
        "runs_empty": bool(experiments.load_runs().empty),
    }


# --- 1) 운영 시나리오 시연 ---------------------------------------------------

def _scenario_summary(result) -> dict:
    return jsonable({
        "version": result.version,
        "warnings": list(result.warnings),
        "total_logged": result.total_logged,
        "total_verified": result.total_verified,
        "days": sum(p.phase.days for p in result.phases),
        "table": table(pd.DataFrame(
            [
                {
                    "구간": p.phase.title,
                    "환경": p.phase.environment.describe(),
                    "검사": p.logged,
                    "결함 판정 비율": round(p.defect_rate, 3),
                    "평균 점수": round(p.mean_score, 2),
                    "재현율": round(p.recall, 2),
                    "검수된 결함": p.verified_defects,
                    "드리프트": p.drift_level,
                    "변화 특징 수": p.drift_changed,
                }
                for p in result.phases
            ]
        )) if result.phases else None,
        "top_drift": [
            {"title": p.phase.title, "features": [{"name": n, "psi": v} for n, v in p.top_drift]}
            for p in result.phases if p.top_drift
        ],
    })


@router.get("/scenario")
def scenario_view() -> dict:
    df = _resolved()
    prod = registry.production()
    result = state.get(state.KEY_SCENARIO)
    first = scenario.DEFAULT_TIMELINE[0]
    return jsonable({
        "production": _row(prod),
        "n_images": int(len(df)),
        "timeline": [
            {"title": p.title, "environment": p.environment.describe(),
             "narration": p.narration, "watch": p.watch}
            for p in scenario.DEFAULT_TIMELINE
        ],
        "n_per_phase": first.n_images,
        "verify_ratio": first.verify_ratio,
        "labeled_by": scenario.LABELED_BY,
        "result": _scenario_summary(result) if result is not None else None,
    })


@router.post("/scenario/run")
def scenario_run() -> dict:
    prod = registry.production()
    if prod is None:
        raise bad_request(
            "서비스 중인 모델이 없습니다. **모델 레지스트리** 탭에서 3단계 실행을 등록하고 승격해야 "
            "시나리오를 돌릴 수 있습니다."
        )
    df = _resolved()
    if df.empty:
        raise bad_request("등록된 이미지가 없습니다. 1단계에서 데이터를 먼저 등록하세요.")
    version = str(prod["version"])

    def run(job):
        result = scenario.run(version, df, progress=job.progress3)
        state.put(state.KEY_SCENARIO, result)
        return _scenario_summary(result)

    job = runner.submit("추론과 검수를 재생하는 중", run, note="구간 3개 × 200장을 판정하고 사후 검수까지 기록합니다.")
    return {"job_id": job.id}


@router.post("/scenario/clear")
def scenario_clear() -> dict:
    removed = scenario.clear()
    state.pop(state.KEY_SCENARIO)
    return {"removed": int(removed)}


# --- 2) 모델 레지스트리 -------------------------------------------------------

REGISTRY_COLUMNS = ["version", "상태", "kind", "model", "threshold", "recall", "precision",
                    "auroc", "created_at", "promoted_at", "note"]


def _run_label(runs: pd.DataFrame, run_id: str) -> str:
    row = runs[runs["run_id"].astype(str) == run_id]
    if row.empty:
        return run_id
    row = row.iloc[0]
    recall = row.get("recall")
    suffix = f" · 재현율 {float(recall):.3f}" if pd.notna(recall) else ""
    return f"{run_id} · {row.get('kind', '')}/{row.get('model', '')}{suffix}"


@router.get("/registry")
def registry_view() -> dict:
    frame = registry.load_registry()
    runs = experiments.load_runs()
    view = None
    if not frame.empty:
        view = frame.copy()
        view["상태"] = view["status"].map(lambda s: registry.STATUS_KO.get(str(s), s))
        view = view[REGISTRY_COLUMNS]
    registered = set(frame["run_id"].astype(str)) if not frame.empty else set()
    options = (
        [] if runs.empty
        else [r for r in runs["run_id"].astype(str) if r not in registered]
    )
    target = registry.rollback_target(frame) if not frame.empty else None
    archivable = (
        frame[frame["status"] != registry.STATUS_ARCHIVED]["version"].astype(str).tolist()
        if not frame.empty else []
    )
    return jsonable({
        "table": table(view) if view is not None else None,
        "versions": frame["version"].astype(str).tolist() if not frame.empty else [],
        "archivable": archivable,
        "runs_empty": bool(runs.empty),
        "options": [{"run_id": r, "label": _run_label(runs, r)} for r in options],
        "default_promote_now": bool(frame.empty),
        "rollback_target": (
            None if target is None else {
                "version": str(target["version"]),
                "recall": target.get("recall"),
                "promoted_at": target.get("promoted_at"),
            }
        ),
    })


class RegisterBody(BaseModel):
    run_id: str
    note: str = ""
    promote_now: bool = False


@router.post("/registry/register")
def registry_register(body: RegisterBody) -> dict:
    """실행을 등록하고, 현재 학습 분할로 드리프트 기준선을 만든다.

    기준선 계산은 학습 분할 전체의 특징을 뽑는 일이라 오래 걸릴 수 있어 잡으로 돌린다.
    """
    df = _resolved()

    def run(job):
        baseline = None
        labeled = (
            df[df["label"].astype(str).isin([config.LABEL_NORMAL, config.LABEL_DEFECT])]
            if not df.empty else df
        )
        train = (
            labeled[labeled["split"].astype(str) == config.SPLIT_TRAIN]
            if not labeled.empty else labeled
        )
        if not train.empty:
            vectors = []
            total = len(train)
            for index, path in enumerate(train["path_abs"], start=1):
                image = viz.load_rgb(path)
                if image is not None:
                    vectors.append(features.image_features(image))
                job.progress(index, total, f"드리프트 기준선 계산 중... {index:,}/{total:,}")
            if vectors:
                matrix = np.stack(vectors)
                labels = (train["label"].astype(str) == config.LABEL_DEFECT).astype(int)
                baseline = registry.make_baseline(
                    matrix, features.FEATURE_NAMES,
                    quality=train[["blur_score", "brightness"]],
                    label_ratio=float(labels.mean()),
                )
        try:
            result = registry.register(
                body.run_id, baseline=baseline, note=body.note, promote_now=body.promote_now
            )
        except ValueError as exc:
            raise RuntimeError(f"등록 실패: {exc}") from exc
        return {"version": result.version, "warnings": list(result.warnings)}

    job = runner.submit(
        "드리프트 기준선 계산 중", run,
        note="학습 분할의 특징 분위 경계를 저장합니다. 이 기준선이 없으면 입력 분포 변화를 감지할 수 없습니다.",
    )
    return {"job_id": job.id}


@router.get("/registry/gate")
def registry_gate(version: str) -> dict:
    """승격 전에 D5 기준으로 점검한다. 미달이면 확인을 받는다.

    **막지는 않는다.** 기준은 아직 승인 전 제안값이고, 비교나 시연 목적으로 일부러 낮은
    모델을 올릴 수도 있다. 다만 초보자가 성능 미달 모델을 아무 신호 없이 올리면, 그 뒤의
    드리프트 감시와 재학습 판단이 전부 그 모델을 기준으로 돌아간다.
    """
    frame = registry.load_registry()
    row = frame[frame["version"].astype(str) == str(version)] if not frame.empty else frame
    if row.empty:
        return {"passed": True, "problems": [], "notes": []}
    row = row.iloc[0]

    metrics = {"recall": row.get("recall")}
    impact = None
    recall, false_alarm = row.get("recall"), row.get("false_alarm_rate")
    if pd.notna(recall) and pd.notna(false_alarm):
        impact = evaluate.business_impact(float(recall), float(false_alarm))
    elif pd.notna(recall) and pd.notna(row.get("precision")):
        # 레지스트리에는 오탐률이 없다. 정밀도와 재현율로 되짚어 계산한다
        # (평가 구성이 1:1이라는 가정이 들어가므로 근사치다).
        precision = float(row["precision"])
        if precision > 0:
            false_alarm = float(recall) * (1.0 - precision) / precision
            impact = evaluate.business_impact(float(recall), min(false_alarm, 1.0))

    check = glossary.promotion_check(metrics, impact)
    return {"passed": bool(check.passed), "problems": list(check.problems), "notes": list(check.notes)}


class VersionBody(BaseModel):
    version: str


@router.post("/registry/promote")
def registry_promote(body: VersionBody) -> dict:
    registry.promote(body.version)
    return {"message": f"{body.version}을(를) 서비스 중으로 전환했습니다."}


@router.post("/registry/rollback")
def registry_rollback() -> dict:
    restored = registry.rollback()
    if restored is None:
        raise bad_request("되돌릴 이전 서비스 버전이 없습니다. (한 번도 교체하지 않았습니다)")
    return {
        "restored": restored,
        "message": f"{restored}(으)로 되돌렸습니다. 새 모델이 더 나빴다면 이렇게 즉시 복구합니다.",
    }


@router.post("/registry/archive")
def registry_archive(body: VersionBody) -> dict:
    registry.archive(body.version)
    return {"message": f"{body.version}을(를) 보관 처리했습니다."}


@router.get("/registry/detail")
def registry_detail(version: str) -> dict:
    run = registry.load_run(version)
    baseline = registry.load_baseline(version)
    return jsonable({
        "run": run,
        "baseline_samples": int(baseline.get("n_samples", 0)) if baseline else None,
    })


# --- 3) 배치 추론 -------------------------------------------------------------

SCOPE_ALL = "전체"
SCOPE_TEST = "test 분할"
SCOPE_UNLABELED = "미라벨"
SCOPE_CATEGORY = "특정 카테고리"
SCOPE_VIDEO = "영상 하나"
SCOPES = (SCOPE_ALL, SCOPE_TEST, SCOPE_UNLABELED, SCOPE_CATEGORY, SCOPE_VIDEO)

LOG_TAIL = 200
HEAD_ROWS = 50


def _video_subset(df: pd.DataFrame, name: str | None, *, limit: int):
    """평가할 영상을 고르고 그 프레임을 **영상 전체에 고르게** 뽑는다.

    «영상 A로 만든 모델이 영상 B에서 얼마나 잡는가»를 재려면 영상 하나를 지목할 수 있어야
    한다. 카테고리로는 안 된다 — 같은 라인에서 찍은 영상이 여럿이면 다 섞인다.

    **앞에서부터 자르지 않는다.** 10분 영상의 앞 200장만 보면 앞 2분을 잰 것이지 그 영상을
    잰 것이 아니다. 조명이 바뀌거나 물건이 달라지는 뒤쪽을 통째로 놓친다.

    Returns:
        (부분집합 또는 None, 화면에 보일 안내문 목록, 프레임이 없을 때의 안내)
    """
    frames = labeling.video_frames(df)
    if frames.empty:
        return None, [], "영상에서 뽑은 프레임이 없습니다. 1단계 **영상에서 프레임 추출**을 먼저 하세요."
    counts = frames["group"].astype(str).value_counts()
    picked = name if name in counts.index else str(counts.index[0])
    chosen = frames[frames["group"].astype(str) == picked]
    notes = []
    if len(chosen) > limit:
        chosen = labeling.even_sample(chosen, limit)
        notes.append(
            f"이 영상 {counts[picked]:,}장 중 **{len(chosen):,}장을 영상 전체에 고르게** "
            "뽑아 판정합니다. 앞에서부터 자르면 뒷부분을 통째로 놓칩니다."
        )
    else:
        notes.append(f"이 영상의 프레임 {len(chosen):,}장을 **전부** 판정합니다.")
    labeled = int((chosen["label"].astype(str) != config.LABEL_UNLABELED).sum())
    if labeled:
        notes.append(f"이 중 {labeled:,}장에 라벨이 있어 **성능 추이** 탭에서 재현율을 잴 수 있습니다.")
    else:
        notes.append(
            "이 영상에는 아직 정답 라벨이 없습니다. 판정은 되지만 **얼마나 맞았는지는 "
            "알 수 없습니다** — 2단계 **영상 구간 라벨링**에서 정답을 붙이세요."
        )
    return chosen, notes, None


def _subset(df: pd.DataFrame, scope: str, limit: int, category: str | None, video_name: str | None):
    """대상 선택. `(부분집합, 안내문, 막는 안내)`."""
    if scope == SCOPE_VIDEO:
        return _video_subset(df, video_name, limit=limit)
    subset = df
    if scope == SCOPE_TEST:
        subset = df[df["split"].astype(str) == config.SPLIT_TEST]
    elif scope == SCOPE_UNLABELED:
        subset = df[df["label"].astype(str) == config.LABEL_UNLABELED]
    elif scope == SCOPE_CATEGORY:
        categories = sorted(df["category"].dropna().astype(str).unique().tolist())
        picked = category if category in categories else (categories[0] if categories else None)
        subset = df[df["category"].astype(str) == picked] if picked else df.iloc[0:0]
    return subset.head(int(limit)), [], None


def _threshold_default(version: str) -> float:
    row = registry.get(version)
    return float(row["threshold"]) if row is not None and pd.notna(row.get("threshold")) else 0.5


def _batch_block(batch: dict | None, version: str | None) -> dict | None:
    """최근 실행 요약. 화면이 고른 버전과 같을 때만 보인다 (원문과 같다)."""
    if not batch or not batch.get("records") or (version and batch.get("version") != version):
        return None
    frame = pd.DataFrame(batch["records"])
    counts = frame["decision"].value_counts()
    return jsonable({
        "version": batch["version"],
        "n": int(len(frame)),
        "defect": int(counts.get(config.LABEL_DEFECT, 0)),
        "normal": int(counts.get(config.LABEL_NORMAL, 0)),
        "latency_mean": float(frame["latency_ms"].mean()),
        "head": table(frame.head(HEAD_ROWS)),
        "failed": int(batch.get("failed", 0)),
    })


def _log_block() -> dict:
    log = monitoring.load_log()
    summary = monitoring.log_summary(log)
    return jsonable({
        "summary": summary,
        "table": table(log.tail(LOG_TAIL).iloc[::-1]) if not log.empty else None,
    })


@router.get("/inference")
def inference_view(version: str | None = None) -> dict:
    df = _resolved()
    usable = list(serving.selectable_versions())
    prod = registry.production()
    default = (
        str(prod["version"]) if prod is not None and str(prod["version"]) in usable
        else (usable[0] if usable else None)
    )
    picked = version if version in usable else default
    frames = labeling.video_frames(df) if not df.empty else df
    videos = (
        [{"name": str(n), "count": int(c)} for n, c in frames["group"].astype(str).value_counts().items()]
        if not frames.empty else []
    )
    return jsonable({
        "empty": bool(df.empty),
        "usable": usable,
        "default_version": default,
        "scopes": list(SCOPES),
        "categories": sorted(df["category"].dropna().astype(str).unique().tolist()) if not df.empty else [],
        "videos": videos,
        "min_drift_samples": monitoring.MIN_DRIFT_SAMPLES,
        "threshold_default": _threshold_default(picked) if picked else 0.5,
        "last": _batch_block(state.get(state.KEY_BATCH), picked),
        "log": _log_block(),
    })


@router.get("/inference/target")
def inference_target(
    version: str, scope: str = SCOPE_ALL, limit: int = Query(200, ge=1, le=5000),
    category: str | None = None, video: str | None = None,
) -> dict:
    """대상 건수를 미리 센다 — 실행 전에 «몇 건을 판정하는지»가 보여야 한다."""
    df = _resolved()
    if df.empty:
        return {"count": 0, "notes": [], "blocked": "수집된 이미지가 없습니다.", "threshold_default": 0.5}
    subset, notes, blocked = _subset(df, scope, limit, category, video)
    return jsonable({
        "count": 0 if subset is None else int(len(subset)),
        "notes": notes,
        "blocked": blocked,
        "threshold_default": _threshold_default(version),
    })


class InferenceBody(BaseModel):
    version: str
    scope: str = SCOPE_ALL
    limit: int = 200
    category: str | None = None
    video: str | None = None
    threshold: float | None = None


@router.post("/inference/run")
def inference_run(body: InferenceBody) -> dict:
    df = _resolved()
    if df.empty:
        raise bad_request("수집된 이미지가 없습니다.")
    subset, _, blocked = _subset(df, body.scope, body.limit, body.category, body.video)
    if blocked:
        raise bad_request(blocked)
    if subset is None or subset.empty:
        raise bad_request("대상 이미지가 없습니다.")
    try:
        model = serving.load_version(body.version)
    except (ValueError, FileNotFoundError) as exc:
        raise bad_request(f"모델 로드 실패: {exc}") from exc
    version = body.version
    threshold = body.threshold

    def run(job):
        result = serving.run_batch(
            model, subset, threshold=threshold,
            progress=lambda d, t: job.progress(d, t, f"추론 중... {d:,}/{t:,}"),
        )
        if not result.records:
            raise RuntimeError("판정한 이미지가 없습니다.")
        monitoring.log_inference(result.records)
        batch = {
            "version": version,
            "records": result.records,
            "features": result.features,
            "failed": len(result.failed),
        }
        state.put(state.KEY_BATCH, batch)
        return {
            "last": _batch_block(batch, version),
            "failed": len(result.failed),
            "warning": f"읽기 실패 {len(result.failed)}건은 제외했습니다." if result.failed else None,
        }

    job = runner.submit(
        "추론 중", run, note=f"{len(subset):,}건을 판정하고 추론 로그에 남깁니다.",
    )
    return {"job_id": job.id}


@router.get("/inference/log/csv")
def inference_log_csv() -> dict:
    log = monitoring.load_log()
    return {"csv": log.to_csv(index=False) if not log.empty else ""}


@router.post("/inference/clear_log")
def inference_clear_log() -> dict:
    monitoring.clear_log()
    state.pop(state.KEY_BATCH)
    return {"ok": True}


# --- 영상 고르기 (ui.video_source의 자리) -------------------------------------
#
# 1단계(추출)와 4단계(판정 영상)가 같은 일을 한다 — «어떤 영상을 쓸 것인가». 1단계 쪽
# 엔드포인트가 함께 만들어지는 중이라 여기서는 자기 라우터 아래에 둔다. 나중에 하나로 합친다.

def _describe_video(path: Path) -> str:
    """목록에 보일 이름. 크기를 붙여야 같은 이름의 다른 영상을 구분할 수 있다.

    영상마다 열어서 길이를 재면 목록을 그릴 때마다 파일을 전부 여는 셈이 되므로 크기만
    쓴다. MB로만 적으면 작은 영상이 전부 `0MB`가 되어 구분이 되지 않는다.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return path.name
    if size >= 1_048_576:
        return f"{path.name} · {size / 1_048_576:,.1f}MB"
    return f"{path.name} · {max(size // 1024, 1):,}KB"


@router.get("/videos")
def videos(scan: str = "") -> dict:
    scan = typed_path(scan)
    found = video.listed(*([scan] if scan else []))
    return {
        "videos": [{"path": str(p), "label": _describe_video(p)} for p in found],
        "video_dir": str(video.video_dir()),
        "scan_missing": bool(scan) and not Path(scan).expanduser().is_dir(),
        "extensions": sorted(video.VIDEO_EXTENSIONS),
    }


@router.post("/videos/upload")
async def videos_upload(file: UploadFile = File(...)) -> dict:
    name = upload_name(file.filename, "upload.mp4")
    if not video.is_video(name):
        raise bad_request(f"영상 파일이 아닙니다: `{Path(name).suffix}`")
    target = video.video_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return {"path": str(target), "message": f"저장 위치: `{target}` — 다음부터는 **목록에서** 고를 수 있습니다."}


@router.get("/videos/check")
def videos_check(path: str) -> dict:
    candidate = Path(typed_path(path)).expanduser()
    if not candidate.is_file():
        return {"ok": False, "error": f"파일을 찾을 수 없습니다: `{candidate}`"}
    if not video.is_video(candidate):
        return {"ok": False, "error": f"영상 파일이 아닙니다: `{candidate.suffix}`"}
    return {"ok": True, "path": str(candidate), "error": None}


# --- 4) 판정 영상 (V7) --------------------------------------------------------

def _playback_block(result) -> dict:
    """판정본 한 벌을 화면에 보일 모양으로. 영상이 위, 대표 판정 결과가 아래."""
    stills = {int(p.stem): p for p in result.stills}
    highlights = []
    for judged in result.highlights():
        picture = stills.get(judged.index)
        if picture is not None:
            highlights.append({"path": str(picture), "caption": judged.caption()})
    frame = result.frame()
    return jsonable({
        "source_name": result.source_name,
        "version": result.version,
        "note": result.note(),
        "video_path": str(result.video),
        "video_name": result.video.name,
        "mime": result.mime,
        "summary": result.summary(),
        "scanned": result.scanned,
        "fps": result.fps,
        "has_defects": bool(result.defects),
        "highlights": highlights,
        "frames": table(frame),
        "csv": frame.to_csv(index=False),
        "directory": str(result.directory),
    })


@router.get("/playback")
def playback_view(version: str | None = None) -> dict:
    usable = list(serving.selectable_versions())
    prod = registry.production()
    default = (
        str(prod["version"]) if prod is not None and str(prod["version"]) in usable
        else (usable[0] if usable else None)
    )
    picked = version if version in usable else default
    row = registry.get(picked) if picked else None
    return jsonable({
        "usable": usable,
        "default_version": default,
        "kind": str(row.get("kind", "")) if row is not None else "",
        "default_limit": playback.DEFAULT_LIMIT,
    })


@router.get("/playback/find")
def playback_find(source: str, version: str) -> dict:
    """이미 만들어 둔 판정본이 있으면 되찾아 온다. 화면을 새로 열 때마다 몇 분짜리 작업을
    다시 시키면 아무도 두 번 쓰지 않는다."""
    result = playback.find(Path(typed_path(source)).expanduser(), version)
    return {"result": _playback_block(result) if result is not None else None}


class PlaybackBody(BaseModel):
    source: str
    version: str
    limit: int = playback.DEFAULT_LIMIT


@router.post("/playback/render")
def playback_render(body: PlaybackBody) -> dict:
    source = Path(typed_path(body.source)).expanduser()
    if not source.is_file():
        raise bad_request(f"파일을 찾을 수 없습니다: `{source}`")
    try:
        model = serving.load_version(body.version)
    except (ValueError, FileNotFoundError) as exc:
        raise bad_request(f"모델 로드 실패: {exc}") from exc
    limit = int(body.limit)

    def run(job):
        try:
            result = playback.render(
                source, model, limit=limit,
                progress=lambda d, t: job.progress(d, t, f"판정 중... {d:,}/{t:,}"),
            )
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(f"판정 영상을 만들지 못했습니다: {exc}") from exc
        if not len(result):
            raise RuntimeError("판정한 프레임이 없습니다. 영상을 읽을 수 있는지 확인해 주세요.")
        return _playback_block(result)

    job = runner.submit(
        "판정 중", run,
        note="영상 전체에서 고르게 뽑은 프레임마다 판정하고, 판정을 그려 넣은 영상을 새로 씁니다.",
    )
    return {"job_id": job.id}


def _judged_video(path: str) -> Path:
    """판정본 영상 경로 검사. 프로젝트 자료 폴더 안의 파일만 돌려준다."""
    target = Path(path).expanduser().resolve()
    if not inside(target, (config.DATA_HOME, config.ARTIFACT_HOME)):
        raise not_found("프로젝트 폴더 밖의 파일은 열 수 없습니다")
    if not target.is_file():
        raise not_found(f"파일이 없습니다: {target}")
    return target


def _video_media(target: Path) -> str:
    return "video/webm" if target.suffix.lower() == ".webm" else "video/mp4"


@router.get("/playback/video")
def playback_video(path: str) -> FileResponse:
    """판정 영상 스트리밍 (`<video>`용).

    `files.video_file`을 쓰지 않는 이유 — 그쪽은 `video.is_video()`로 확장자를 거르는데
    `VIDEO_EXTENSIONS`에 `.webm`이 없다. 판정본은 브라우저가 재생할 수 있는 WebM/VP8로
    먼저 만들므로(`playback.CODECS`) 그대로 두면 재생 요청이 404가 난다.
    """
    target = _judged_video(path)
    return FileResponse(target, media_type=_video_media(target))


@router.get("/playback/download")
def playback_download(path: str) -> FileResponse:
    """판정 영상 내려받기."""
    target = _judged_video(path)
    return FileResponse(target, media_type=_video_media(target), filename=target.name)


# --- 5) 영상 비교 (V4) --------------------------------------------------------

def _feature_matrix(resolved: pd.DataFrame, image_ids, progress=None) -> np.ndarray:
    """이 프레임들의 특징 행렬. 3단계와 같은 캐시를 쓰므로 두 번째부터는 즉시 나온다."""
    rows = resolved[resolved["image_id"].isin(set(image_ids))]
    if rows.empty:
        return np.empty((0, len(features.FEATURE_NAMES)), dtype=np.float32)
    dataset = models.build_dataset(
        rows["path_abs"].tolist(), [0] * len(rows), rows["image_id"].tolist(), progress=progress
    )
    return dataset.X


def _feedback(df: pd.DataFrame) -> pd.DataFrame:
    log = monitoring.load_log()
    return monitoring.feedback_frame(log, df) if not log.empty and not df.empty else pd.DataFrame()


def _video_names(feedback: pd.DataFrame) -> list[str]:
    """`segments.videos_in`의 결과에서 빈 묶음을 거른다.

    pandas 3에서는 `astype(str)`이 NaN을 문자열로 바꾸지 않고 그대로 두므로, 낱장 사진의
    빈 `group`이 목록에 `nan`으로 섞여 들어온다. 영상이 아닌 것을 영상 목록에 보이면 안 된다.
    """
    return [name for name in segments.videos_in(feedback) if isinstance(name, str) and name.strip()]


@router.get("/compare")
def compare_view() -> dict:
    df = _resolved()
    names = _video_names(_feedback(df)) if not df.empty else []
    return {"empty": bool(df.empty), "names": names}


@router.get("/compare/check")
def compare_check(left: str, right: str, fps: float = 30.0) -> dict:
    """두 영상 모두 정답과 대조할 프레임이 있어야 비교할 수 있다."""
    df = _resolved()
    feedback = _feedback(df)
    frames = labeling.video_frames(df) if not df.empty else df
    reports = {
        name: segments.from_feedback(feedback, frames, name, fps=float(fps)) for name in (left, right)
    }
    return {"ok": all(report is not None for report in reports.values())}


class CompareBody(BaseModel):
    left: str
    right: str
    fps: float = 30.0


@router.post("/compare/run")
def compare_run(body: CompareBody) -> dict:
    df = _resolved()
    feedback = _feedback(df)
    frames = labeling.video_frames(df) if not df.empty else df
    left, right, fps = body.left, body.right, float(body.fps)
    reports = {
        name: segments.from_feedback(feedback, frames, name, fps=fps) for name in (left, right)
    }
    if any(report is None for report in reports.values()):
        raise bad_request("두 영상 모두 정답과 대조할 프레임이 있어야 비교할 수 있습니다.")

    def run(job):
        matrices = {}
        for name in (left, right):
            picked = feedback[feedback["group"].astype(str) == name]["image_id"]
            matrices[name] = _feature_matrix(
                df, picked,
                progress=lambda d, t: job.progress(d, t, f"특징 읽는 중... {d:,}/{t:,}"),
            )
        verdict = compare.Comparison(
            reference=compare.side_from(reports[left], left, frames=len(matrices[left])),
            candidate=compare.side_from(reports[right], right, frames=len(matrices[right])),
            drift=compare.drift_between(matrices[left], matrices[right], features.FEATURE_NAMES),
        )
        drift = verdict.drift
        drift_text = (
            f"입력 분포: 평균 PSI {drift.psi_mean:.3f} ({drift.level}) · 비교 표본 "
            f"{drift.n_samples:,}장"
            if np.isfinite(drift.psi_mean)
            else f"입력 분포: 표본이 부족해 비교하지 않았습니다 ({drift.n_samples:,}장)."
        )
        return jsonable({
            "table": table(verdict.table()),
            "cause": verdict.cause(),
            "verdict": verdict.verdict(),
            "advice": verdict.advice(),
            "drift_text": drift_text,
            "causes": {"none": compare.CAUSE_NONE, "unknown": compare.CAUSE_UNKNOWN},
        })

    job = runner.submit(
        "특징 읽는 중", run, note="입력 분포까지 비교하려면 두 영상의 특징을 다시 읽어야 합니다.",
    )
    return {"job_id": job.id}


# --- 6) 드리프트 감시 ---------------------------------------------------------

def _drift_causes(drift: pd.DataFrame) -> list[str]:
    """분포가 변한 특징을 현장에서 확인할 것으로 옮겨 준다.

    `gray_mean`, `lap_p99` 같은 이름만 보고 "조명이 바뀌었나?"까지 연결하려면 각 특징이
    무엇을 재는지 알아야 하는데, 그건 이 파이프라인을 만든 사람만 안다.
    """
    shifted = drift[drift["level"].astype(str) == "변화"]
    if shifted.empty:
        return []
    names = shifted.sort_values("psi", ascending=False)["feature"].astype(str).tolist()
    return glossary.drift_causes(names)[:4]


@router.get("/drift")
def drift_view() -> dict:
    block: dict = {
        **_production_block(),
        "baseline": None, "has_batch": False, "summary": None,
        "thresholds": {"stable": monitoring.PSI_STABLE, "shifted": monitoring.PSI_SHIFTED,
                       "min_samples": monitoring.MIN_DRIFT_SAMPLES,
                       "insufficient": monitoring.LEVEL_INSUFFICIENT},
        "glossary": {"psi_caption": glossary.caption("psi"), "psi_detail": glossary.detail("psi"),
                     "arbitrary_psi": glossary.ARBITRARY["psi"],
                     "arbitrary_samples": glossary.ARBITRARY["drift_samples"]},
    }
    prod = registry.production()
    if prod is None:
        return jsonable(block)
    version = str(prod["version"])
    baseline = registry.load_baseline(version)
    if baseline is None:
        return jsonable(block)
    block["baseline"] = {
        "n_samples": int(baseline.get("n_samples", 0)),
        "features": len(baseline.get("feature_names", [])),
        "bins": baseline.get("bins", 0),
    }
    batch = state.get(state.KEY_BATCH)
    if not batch or batch.get("features") is None or len(batch["features"]) == 0:
        return jsonable(block)
    block["has_batch"] = True

    current = batch["features"]
    drift = monitoring.feature_drift(baseline, current, features.FEATURE_NAMES)
    summary = monitoring.drift_summary(drift)
    block["summary"] = summary
    block["causes"] = _drift_causes(drift)

    display = drift.head(20)[
        ["feature", "psi", "level", "baseline_mean", "current_mean", "shift_sigma"]
    ].copy()
    # 내부 특징명만으로는 무엇이 변했는지 알 수 없다. 뜻을 나란히 붙인다.
    display.insert(1, "무엇을 재는가", [glossary.feature_meaning(f) for f in display["feature"]])
    display = display.round(4)
    block["table"] = table(display)
    block["help"] = _column_help(display)

    bars = drift.head(15).set_index("feature")["psi"].dropna()
    block["bars"] = [{"name": str(k), "psi": float(v)} for k, v in bars.items()]

    scores = np.array([r["score"] for r in batch["records"]], dtype=float)
    block["score"] = monitoring.score_drift(baseline, scores)
    return jsonable(block)


# --- 7) 성능 추이 -------------------------------------------------------------

@router.get("/performance")
def performance_view() -> dict:
    df = _resolved()
    log = monitoring.load_log()
    if log.empty:
        return {"log_empty": True, "feedback_empty": True}
    feedback = monitoring.feedback_frame(log, df)
    if feedback.empty:
        return {"log_empty": False, "feedback_empty": True}

    metrics = monitoring.performance_metrics(feedback)
    block: dict = {"log_empty": False, "feedback_empty": False, "metrics": metrics, "gap": None}

    prod = registry.production()
    if prod is not None and pd.notna(prod.get("recall")):
        registered = float(prod["recall"])
        actual = metrics["recall"]
        if np.isfinite(actual):
            block["gap"] = {"registered": registered, "actual": actual, "gap": registered - actual}

    misses = feedback[feedback["outcome"] == "FN"].head(8)
    known = set(df["image_id"].astype(str)) if not df.empty else set()
    block["misses"] = [
        {"imageId": str(row["image_id"]), "caption": f"{row['image_id']} · 점수 {float(row['score']):.3f}"}
        for _, row in misses.iterrows() if str(row["image_id"]) in known
    ]

    block["videos"] = _video_names(feedback)
    block["segment_defaults"] = {"fps": 30.0, "min_hits": segments.DEFAULT_MIN_HITS}

    by_version = monitoring.performance_by_version(feedback)
    if not by_version.empty:
        by_version = by_version.round(4)
        block["by_version"] = table(by_version)
        block["by_version_help"] = _column_help(by_version)

    trend = monitoring.performance_trend(feedback)
    if trend.empty or len(trend) < 2:
        block["trend"] = None
    else:
        rows = [
            {"bucket": ts.strftime("%Y-%m-%d"), "recall": r, "precision": p}
            for ts, r, p in zip(trend["bucket"], trend["recall"], trend["precision"])
        ]
        rows = jsonable(rows)
        spec = charts.line_chart(rows, x="bucket", ys=["recall", "precision"], x_title="날짜")
        # 날짜를 이름(ordinal)으로 두면 90일치 눈금이 전부 찍혀 축이 읽히지 않는다.
        # `st.line_chart`가 날짜 인덱스를 시간축으로 그리던 것과 같게 맞춘다.
        if spec is not None:
            spec["encoding"]["x"]["type"] = "temporal"
            spec["encoding"]["x"]["axis"] = {"title": "날짜", "format": "%m-%d"}
        block["trend"] = spec

    block["detail"] = table(
        feedback[["image_id", "version", "score", "decision", "label", "outcome", "logged_at"]]
    )
    return jsonable(block)


@router.get("/performance/segments")
def performance_segments(
    video: str, fps: float = 30.0, min_hits: int = Query(segments.DEFAULT_MIN_HITS, ge=1, le=20),
) -> dict:
    """구간 단위 검출률 (V3).

    프레임 단위 숫자는 결함이 여러 장에 걸칠 때 **실제보다 나쁘게** 나온다. 100프레임짜리
    결함에서 3장만 잡아도 알람은 울렸는데 재현율은 3%로 찍힌다. 두 숫자를 나란히 놓고
    서로 다르면 그 사실을 말해 준다.
    """
    df = _resolved()
    feedback = _feedback(df)
    report = segments.from_feedback(
        feedback, labeling.video_frames(df) if not df.empty else df, video,
        fps=float(fps), min_hits=int(min_hits),
    )
    if report is None:
        return {"available": False}
    # V6 — 미탐이 몇 건인지보다 **어디서** 놓쳤는지가 다음에 무엇을 라벨링할지 정해 준다.
    timeline = charts.segment_timeline(
        segments.bands(report), duration=max(report.duration_sec, 0.001)
    )
    return jsonable({
        "available": True,
        "metrics": {
            "segment_recall": _figure(report.segment_recall),
            "frame_recall": _figure(report.frame_recall),
            "missed": len(report.missed),
            "false_alarms_per_min": _figure(report.false_alarms_per_min),
        },
        "summary": report.summary(),
        "contrast": report.contrast(),
        "timeline": timeline,
        "missed_text": ", ".join(s.span_text() for s in report.missed[:10]) if report.missed else "",
        "false_text": ", ".join(s.span_text() for s in report.false_alarms[:10]) if report.false_alarms else "",
    })


# --- 8) 재학습 판단 -----------------------------------------------------------

def _decision_block(new_label_threshold: int, recall_margin: float) -> dict:
    df = _resolved()
    prod = registry.production()
    log = monitoring.load_log()
    feedback = monitoring.feedback_frame(log, df) if not log.empty else pd.DataFrame()

    drift = None
    batch = state.get(state.KEY_BATCH)
    if prod is not None and batch and batch.get("features") is not None and len(batch["features"]):
        baseline = registry.load_baseline(str(prod["version"]))
        if baseline:
            drift = monitoring.feature_drift(baseline, batch["features"], features.FEATURE_NAMES)

    new_labels = monitoring.new_labels_since(prod.get("promoted_at")) if prod is not None else 0
    decision = monitoring.retraining_signals(
        production=prod, log=log, feedback=feedback, drift=drift,
        new_labels=new_labels, new_label_threshold=int(new_label_threshold),
        recall_margin=float(recall_margin),
    )
    return {
        "recommended": bool(decision.recommended),
        "signals": [
            {"key": s.key, "level": s.level, "icon": s.icon, "title": s.title, "detail": s.detail}
            for s in decision.signals
        ],
    }


@router.get("/retraining")
def retraining_view() -> dict:
    saved = user_settings.load()
    return jsonable({
        "defaults": {
            "new_label_threshold": saved.new_label_threshold,
            "recall_margin": saved.recall_margin,
        },
        "help": {
            "new_label_threshold": user_settings.HELP["new_label_threshold"],
            "recall_margin": user_settings.HELP["recall_margin"],
        },
        "caption": (
            f"{glossary.ARBITRARY['new_labels']} {glossary.ARBITRARY['recall_margin']} "
            "위 두 값은 화면에서 바로 바꿔 볼 수 있습니다."
        ),
        "decision": _decision_block(saved.new_label_threshold, saved.recall_margin),
    })


class RetrainingBody(BaseModel):
    new_label_threshold: int
    recall_margin: float


@router.post("/retraining")
def retraining_decide(body: RetrainingBody) -> dict:
    return jsonable(_decision_block(body.new_label_threshold, body.recall_margin))


# --- 9) 판정 이력 -------------------------------------------------------------

@router.get("/trace")
def trace_view() -> dict:
    df = _resolved()
    if df.empty:
        return {"empty": True, "candidates": []}
    log = monitoring.load_log()
    candidates = (
        log["image_id"].astype(str).unique().tolist() if not log.empty
        else df["image_id"].astype(str).tolist()
    )
    return {"empty": False, "candidates": candidates}


@router.get("/trace/{image_id}")
def trace_image(image_id: str) -> dict:
    trace = monitoring.trace_image(image_id)
    manifest = trace["manifest"]
    effective = trace["effective"]
    events = trace["label_events"]
    inferences = trace["inferences"]
    reviews = trace["claude_reviews"]
    return jsonable({
        "image_id": image_id,
        "known": manifest is not None,
        "manifest": (
            f"출처 `{manifest.get('source')}` · 카테고리 `{manifest.get('category')}` · "
            f"{manifest.get('width')}×{manifest.get('height')} · "
            f"선명도 {manifest.get('blur_score')} · 밝기 {manifest.get('brightness')}"
            if manifest else None
        ),
        "effective": (
            f"{config.LABEL_KO.get(str(effective.get('label')), effective.get('label'))} / "
            f"{config.defect_type_label(str(effective.get('defect_type')))} · "
            f"근거 `{effective.get('label_source')}` · 분할 `{effective.get('split')}`"
            if effective else None
        ),
        "events": table(events) if events is not None and not events.empty else None,
        "inferences": (
            table(inferences[["logged_at", "version", "score", "threshold", "decision", "latency_ms"]])
            if inferences is not None and not inferences.empty else None
        ),
        "reviews": (
            [
                {
                    "title": (
                        f"**{config.defect_type_label(str(review['defect_type']))}** · "
                        f"심각도 {review['severity']} · 확신도 {review['confidence']} · "
                        f"모델 `{review['model']}`"
                    ),
                    "reason": str(review["reason"]),
                }
                for _, review in reviews.iterrows()
            ]
            if reviews is not None and not reviews.empty else []
        ),
    })
