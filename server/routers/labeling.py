"""2단계: 라벨링 — 검수 큐 · 영상 구간 · 폴더 라벨 검증 · 유형 정규화 · 분할 · 현황 · 박스 내보내기.

Streamlit 화면(`app_pages/p2_labeling.py`)을 그대로 옮겼다. 판단 로직은 전부 코어
(`labeling` `boxes` `detection`)에 있고 여기서는 JSON으로 바꾸기만 한다.

검수 큐의 **커서는 서버가 들고 있지 않는다.** 예전 `st.session_state`의 커서는 화면 로컬
state로 옮겼다 — 큐 자체(`review_queue`)는 저장된 라벨에서 매번 다시 계산되므로 서버가
기억할 것이 없다.
"""

from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, File, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from vision_ai import boxes as box_store
from vision_ai import config, detection, labeling, storage, viz

from ..common import bad_request, counts, jsonable, not_found, table
from ..jobs import runner
from .files import MAX_WIDTH, image_response

router = APIRouter(prefix="/labeling", tags=["labeling"])

# 라벨을 기록하면 큐에서 빠지는 모드 (커서를 그대로 두면 다음 항목이 올라온다)
DRAINING_MODES = frozenset({"unlabeled", "unspecified", "unmapped"})

QUEUE_MODES = {
    "unspecified": "결함이지만 유형 미지정 (VisA는 대부분 여기)",
    "unlabeled": "라벨 없음",
    "unverified": "폴더 라벨 미확인",
    "unmapped": "표준 유형으로 정규화되지 않은 결함",
    "all": "전체",
}

DEFECT_TYPE_CHOICES = list(config.DEFECT_TYPES)

ALL_CATEGORIES = "(전체)"


def _defect_types() -> list[dict]:
    return [{"key": key, "label": config.defect_type_label(key)} for key in DEFECT_TYPE_CHOICES]


def _resolved_row(resolved: pd.DataFrame, image_id: str) -> pd.Series:
    rows = resolved[resolved["image_id"].astype(str) == str(image_id)]
    if rows.empty:
        raise not_found(f"이미지가 없습니다: {image_id}")
    return rows.iloc[0]


# --- 공통 --------------------------------------------------------------------

@router.get("/overview")
def overview() -> dict:
    """탭 표식·상단 안내에 필요한 것만.

    2단계에서 반드시 해야 하는 것은 **데이터 분할**이다. 나머지는 데이터에 따라 건너뛸 수
    있는데, 분할을 빼먹으면 3단계가 통째로 막힌다. 그래서 그 탭만 상태를 표시한다.
    """
    resolved = labeling.resolve()
    split_done = (
        not resolved.empty
        and (resolved["split"].astype(str) != config.SPLIT_NONE).any()
    )
    return {
        "total": int(len(resolved)),
        "split_done": bool(split_done),
        "defect_types": _defect_types(),
        "labels_ko": dict(config.LABEL_KO),
        "queue_modes": [{"key": k, "label": v} for k, v in QUEUE_MODES.items()],
        "draining_modes": sorted(DRAINING_MODES),
    }


# --- 라벨 검수 큐 ------------------------------------------------------------

@router.get("/queue")
def queue(mode: str = "unspecified", category: str = ALL_CATEGORIES) -> dict:
    if mode not in QUEUE_MODES:
        raise bad_request(f"모르는 검수 대상입니다: {mode}")
    resolved = labeling.resolve()
    if resolved.empty:
        return {"total_images": 0, "categories": [ALL_CATEGORIES], "ids": []}
    categories = [ALL_CATEGORIES] + sorted(
        resolved["category"].dropna().astype(str).unique().tolist()
    )
    picked = labeling.review_queue(
        resolved, mode=mode, category=None if category == ALL_CATEGORIES else category
    )
    return {
        "total_images": int(len(resolved)),
        "categories": categories,
        "ids": picked["image_id"].astype(str).tolist(),
    }


@router.get("/item/{image_id}")
def item(image_id: str) -> dict:
    """한 항목의 정보 전부 — 라벨·근거·이미지 크기·자동 ROI 유무·저장된 박스.

    처음 열 때 **이미 저장된 박스**를 함께 보낸다. 저장된 것을 안 보여주면 다시 열었을 때
    빈 화면이 나와 "지워졌나?" 하게 된다.
    """
    resolved = labeling.resolve()
    if resolved.empty:
        raise not_found(f"이미지가 없습니다: {image_id}")
    row = _resolved_row(resolved, image_id)
    path = storage.resolve_path(row["path"])
    rgb = viz.load_rgb(path)
    auto_roi = labeling.roi_from_image_path(str(row["path"]))
    saved = box_store.to_boxes(box_store.for_image(image_id))
    return jsonable({
        "image_id": image_id,
        "category": row["category"],
        "path": str(path),
        "readable": rgb is not None,
        "width": int(rgb.shape[1]) if rgb is not None else None,
        "height": int(rgb.shape[0]) if rgb is not None else None,
        "label": str(row["label"]),
        "label_ko": config.LABEL_KO.get(str(row["label"]), str(row["label"])),
        "defect_type": str(row["defect_type"]),
        "defect_type_label": config.defect_type_label(str(row["defect_type"])),
        "source": row["source"],
        "label_source": row["label_source"],
        "raw_defect_type": row["raw_defect_type"],
        "verified": bool(row["verified"]),
        "auto_roi": list(auto_roi) if auto_roi else None,
        "boxes": [
            {"x": b.x, "y": b.y, "w": b.w, "h": b.h, "label": b.label, "source": b.source}
            for b in saved
        ],
    })


class BoxBody(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str = config.DEFECT_TYPE_UNSPECIFIED
    source: str = box_store.SOURCE_HUMAN


class SaveBody(BaseModel):
    image_id: str
    label: str
    defect_type: str | None = None
    boxes: list[BoxBody] = []
    note: str = ""


@router.post("/save")
def save(body: SaveBody) -> dict:
    if body.label not in config.LABELS:
        raise bad_request(f"모르는 판정입니다: {body.label}")
    is_defect = body.label == config.LABEL_DEFECT
    if is_defect and not body.defect_type:
        raise bad_request("결함으로 판정했으면 유형을 선택해야 저장할 수 있다.")
    drawn = [
        box_store.Box(
            image_id=body.image_id, x=b.x, y=b.y, w=b.w, h=b.h,
            label=b.label or config.DEFECT_TYPE_UNSPECIFIED, source=b.source,
        )
        for b in body.boxes
    ] if is_defect else []
    # 단일 ROI 열은 그대로 유지한다 — 3단계 Claude 크롭과 예전 이력이 이 값을 본다.
    # 여러 개를 그렸으면 첫 박스를 대표로 남기고, 전부는 박스 표에 저장한다.
    labeling.record_label(
        body.image_id,
        label=body.label,
        defect_type=body.defect_type if is_defect else config.DEFECT_TYPE_NONE,
        roi=(drawn[0].x, drawn[0].y, drawn[0].w, drawn[0].h) if drawn else None,
        verified=True,
        note=body.note,
    )
    box_store.replace(body.image_id, drawn)
    return {"ok": True, "image_id": body.image_id, "boxes": len(drawn)}


def parse_boxes(text: str) -> list[tuple[int, int, int, int, str]]:
    """`x,y,w,h,label;x,y,w,h,label` 꼴의 query 값. 라벨은 없어도 된다."""
    found = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(",")
        if len(parts) < 4:
            raise bad_request(f"박스 형식이 잘못되었습니다: {chunk}")
        try:
            x, y, w, h = (int(float(v)) for v in parts[:4])
        except ValueError as exc:
            raise bad_request(f"박스 좌표가 숫자가 아닙니다: {chunk}") from exc
        label = ",".join(parts[4:]).strip() if len(parts) > 4 else ""
        found.append((x, y, w, h, label))
    return found


@router.get("/preview/{image_id}")
def preview(image_id: str, boxes: str = Query("")) -> Response:
    """이미지 위에 박스를 전부 그린다. 번호가 있어야 오른쪽 목록과 대조된다."""
    row = _resolved_row(labeling.resolve(), image_id)
    rgb = viz.load_rgb(storage.resolve_path(row["path"]))
    if rgb is None:
        raise not_found(f"이미지를 읽을 수 없습니다: {row['path']}")
    out = rgb
    for order, (x, y, w, h, label) in enumerate(parse_boxes(boxes), start=1):
        out = viz.draw_roi(
            out, (x, y, w, h),
            label=f"{order}. {config.defect_type_label(label or config.DEFECT_TYPE_UNSPECIFIED)}",
        )
    return image_response(out, width=MAX_WIDTH)


# --- 영상 구간 라벨링 (H4) --------------------------------------------------

@router.get("/video-frames")
def video_frames() -> dict:
    """영상별 프레임 목록. 초 계산(fps)은 화면이 한다 — fps는 사용자가 바꾸는 값이라서."""
    resolved = labeling.resolve()
    frames = labeling.video_frames(resolved)
    if frames.empty:
        return {"videos": []}
    videos = []
    for name in sorted(frames["group"].astype(str).unique()):
        subset = frames[frames["group"].astype(str) == name]
        videos.append({
            "name": name,
            "frames": [
                {
                    "image_id": str(row["image_id"]),
                    "frame_index": int(row["frame_index"]),
                    "label": str(row["label"]),
                    "label_ko": config.LABEL_KO.get(str(row["label"]), str(row["label"])),
                }
                for _, row in subset.iterrows()
            ],
        })
    return {"videos": videos}


class SegmentBody(BaseModel):
    video: str
    start: float
    end: float
    image_ids: list[str]
    label: str
    defect_type: str | None = None


@router.post("/segments/apply")
def apply_segment(body: SegmentBody) -> dict:
    if body.label not in (config.LABEL_DEFECT, config.LABEL_NORMAL):
        raise bad_request(f"모르는 판정입니다: {body.label}")
    is_defect = body.label == config.LABEL_DEFECT
    if is_defect and not body.defect_type:
        raise bad_request("결함으로 판정했으면 유형을 선택해야 저장할 수 있다.")
    count = labeling.record_labels([
        {
            "image_id": str(image_id),
            "label": body.label,
            "defect_type": body.defect_type if is_defect else config.DEFECT_TYPE_NONE,
            "verified": True,
            "labeled_by": "구간 라벨링",
            "note": f"{body.video} {body.start:.1f}~{body.end:.1f}초",
        }
        for image_id in body.image_ids
    ])
    return {"count": int(count)}


# --- 폴더 라벨 검증 ----------------------------------------------------------

def _pending_folder_labels(resolved: pd.DataFrame) -> pd.DataFrame:
    return resolved[
        (resolved["label_source"].astype(str) == "folder")
        & (~resolved["verified"].fillna(False).astype(bool))
    ]


@router.get("/verify")
def verify_sample(size: int = Query(8, ge=1, le=48), seed: int = Query(0, ge=0)) -> dict:
    resolved = labeling.resolve()
    if resolved.empty:
        return {"total_images": 0, "pending": 0, "items": []}
    pending = _pending_folder_labels(resolved)
    if pending.empty:
        return {"total_images": int(len(resolved)), "pending": 0, "items": []}
    sample = pending.sample(min(size, len(pending)), random_state=int(seed))
    return jsonable({
        "total_images": int(len(resolved)),
        "pending": int(len(pending)),
        "items": [
            {
                "image_id": str(row["image_id"]),
                "category": row["category"],
                "raw_defect_type": row["raw_defect_type"],
                "label": str(row["label"]),
                "defect_type": str(row["defect_type"]),
            }
            for _, row in sample.iterrows()
        ],
    })


class Decision(BaseModel):
    image_id: str
    label: str


class VerifyBody(BaseModel):
    decisions: list[Decision]


@router.post("/verify")
def verify(body: VerifyBody) -> dict:
    resolved = labeling.resolve()
    if resolved.empty:
        raise bad_request("수집된 이미지가 없습니다.")
    lookup = resolved.set_index(resolved["image_id"].astype(str))
    items = []
    changed = 0
    for decision in body.decisions:
        if decision.label not in (config.LABEL_NORMAL, config.LABEL_DEFECT):
            raise bad_request(f"모르는 판정입니다: {decision.label}")
        if decision.image_id not in lookup.index:
            raise not_found(f"이미지가 없습니다: {decision.image_id}")
        original = lookup.loc[decision.image_id]
        if decision.label != str(original["label"]):
            changed += 1
        items.append({
            "image_id": decision.image_id,
            "label": decision.label,
            "defect_type": (
                str(original["defect_type"])
                if decision.label == config.LABEL_DEFECT
                else config.DEFECT_TYPE_NONE
            ),
            "verified": True,
            "note": "folder_label_review",
        })
    count = labeling.record_labels(items)
    return {"count": int(count), "changed": changed}


# --- 결함 유형 정규화 --------------------------------------------------------

@router.get("/mapping")
def mapping() -> dict:
    resolved = labeling.resolve()
    unmapped = labeling.unmapped_defect_types(resolved)
    current = labeling.load_type_map()
    frequencies = (
        resolved[resolved["label"] == config.LABEL_DEFECT]["defect_type"].astype(str).value_counts()
        if not resolved.empty else pd.Series(dtype=int)
    )
    current_table = pd.DataFrame(
        [{"원본 유형": k, "표준 유형": v, "표준 유형(한글)": config.defect_type_label(v)}
         for k, v in sorted(current.items())]
    )
    return jsonable({
        "unmapped": [{"raw": raw, "count": int(frequencies.get(raw, 0))} for raw in unmapped],
        "mapping": table(current_table),
        "defect_types": _defect_types(),
    })


class MappingBody(BaseModel):
    choices: dict[str, str]


@router.post("/mapping")
def save_mapping(body: MappingBody) -> dict:
    for raw, standard in body.choices.items():
        if standard not in DEFECT_TYPE_CHOICES:
            raise bad_request(f"표준 유형이 아닙니다: {raw} → {standard}")
    current = labeling.load_type_map()
    current.update(body.choices)
    labeling.save_type_map(current)
    return {"count": len(body.choices)}


# --- 데이터 분할 -------------------------------------------------------------

@router.get("/split")
def split_state() -> dict:
    resolved = labeling.resolve()
    if resolved.empty:
        return {"total_images": 0}
    groups = labeling.image_groups(resolved)
    ids = resolved["image_id"].astype(str)
    grouped = int((groups != ids).sum())
    gap = labeling.split_gap(resolved)

    current = resolved["split"].astype(str)
    has_split = bool(
        current.isin([config.SPLIT_TRAIN, config.SPLIT_VAL, config.SPLIT_TEST]).any()
    )
    crosstab = None
    if has_split:
        crosstab = table(
            pd.crosstab([resolved["category"].astype(str), resolved["label"].astype(str)], current),
            index=True,
        )
    return jsonable({
        "total_images": int(len(resolved)),
        "grouped": grouped,
        "videos": int(groups[groups != ids].nunique()) if grouped else 0,
        "split_modes": [
            {"key": key, "label": labeling.SPLIT_MODE_LABELS[key]} for key in labeling.SPLIT_MODES
        ],
        "gap": {
            "total": gap.total, "assigned": gap.assigned, "unassigned": gap.unassigned,
            "labeled_unassigned": gap.labeled_unassigned,
            "boxed_unassigned": gap.boxed_unassigned,
            "complete": gap.complete, "message": gap.message(), "advice": gap.advice(),
        },
        "has_split": has_split,
        "crosstab": crosstab,
    })


class SplitBody(BaseModel):
    train: float = 0.6
    val: float = 0.2
    test: float = 0.2
    seed: int = 42
    labeled_only: bool = True
    mode: str = labeling.SPLIT_BY_GROUP


@router.post("/split/run")
def run_split(body: SplitBody) -> dict:
    if body.test <= 0:
        raise bad_request("test 비율이 0 이하다. train/val 비율을 줄여야 한다.")
    resolved = labeling.resolve()
    mapping = labeling.assign_splits(
        resolved, train=body.train, val=body.val, test=body.test, seed=int(body.seed),
        labeled_only=body.labeled_only, mode=body.mode,
    )
    if not mapping:
        return {"count": 0}
    labeling.save_splits(mapping)
    return {"count": len(mapping)}


@router.post("/split/import")
def import_split(file: UploadFile = File(...)) -> dict:
    try:
        mapping, unmatched = labeling.import_split_csv(file.file, storage.load_manifest())
    except (ValueError, KeyError) as exc:
        raise bad_request(f"임포트 실패: {exc}") from exc
    if mapping:
        labeling.save_splits(mapping)
    return {"count": len(mapping), "unmatched": int(unmatched)}


@router.post("/split/clear")
def clear_split() -> dict:
    labeling.clear_splits()
    return {"ok": True}


# --- 라벨 현황 ---------------------------------------------------------------

@router.get("/status")
def status() -> dict:
    resolved = labeling.resolve()
    stats = labeling.stats(resolved)
    if stats["total"] == 0:
        return {"stats": stats}
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    events = labeling.load_events()
    return jsonable({
        "stats": stats,
        "defect_types": counts(defects["defect_type"]) if not defects.empty else [],
        "label_sources": counts(resolved["label_source"]),
        "events_total": int(len(events)),
        "events": table(events.tail(200).iloc[::-1]),
    })


@router.get("/export/{name}")
def export_text(name: str) -> dict:
    """작은 텍스트 내려받기 — 화면이 `downloadText()`로 저장한다."""
    if name == "resolved":
        return {"filename": "resolved_labels.csv", "mime": "text/csv",
                "text": labeling.resolve().to_csv(index=False)}
    if name == "events":
        events = labeling.load_events()
        if events.empty:
            raise not_found("아직 기록된 라벨 이벤트가 없습니다.")
        return {"filename": "labels.csv", "mime": "text/csv", "text": events.to_csv(index=False)}
    if name == "boxes":
        return {"filename": "boxes.csv", "mime": "text/csv",
                "text": box_store.load().to_csv(index=False)}
    if name == "coco":
        payload = box_store.to_coco(box_store.load(), storage.load_manifest())
        return {"filename": "annotations_coco.json", "mime": "application/json",
                "text": json.dumps(payload, ensure_ascii=False, indent=2)}
    raise not_found(f"모르는 내려받기입니다: {name}")


@router.get("/export-yolo.zip")
def export_yolo_zip() -> Response:
    data = box_store.to_yolo_zip(box_store.load(), storage.load_manifest())
    return Response(content=data, media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=labels_yolo.zip"})


# --- 결함 박스 ---------------------------------------------------------------

@router.get("/boxes")
def boxes_state() -> dict:
    """박스 요약 · 마스크 자동 생성 가능 수 · 예전 ROI · 검출 준비 상태를 한 번에."""
    resolved = labeling.resolve()
    frame = box_store.load()
    info = box_store.summary(frame)
    manifest = storage.load_manifest()
    mask = box_store.mask_candidates(resolved) if not resolved.empty else {
        "ready": 0, "without_mask": 0, "already": 0,
    }
    unspecified = int(
        (
            (resolved["label"].astype(str) == config.LABEL_DEFECT)
            & (resolved["defect_type"].astype(str) == config.DEFECT_TYPE_UNSPECIFIED)
        ).sum()
    ) if not resolved.empty else 0
    texts = box_store.to_yolo(frame, manifest) if info["boxes"] else {}
    return jsonable({
        "summary": info,
        "mask": mask,
        "unspecified": unspecified,
        "adoptable": len(box_store.from_manifest_roi(resolved)) if not info["boxes"] else 0,
        "class_names": box_store.class_names(frame),
        "yolo_images": len(texts),
        "readiness": detection.readiness(resolved, frame),
    })


@router.post("/boxes/from-masks")
def boxes_from_masks() -> dict:
    """마스크에서 덩어리마다 박스를 만든다. 이미지마다 마스크를 읽으므로 잡으로 돌린다."""
    def run(job):
        resolved = labeling.resolve()
        return box_store.from_masks(resolved, progress=job.progress)
    job = runner.submit("마스크에서 박스 만드는 중", run, note="이미지마다 마스크를 읽습니다.")
    return {"job_id": job.id}


@router.post("/boxes/adopt")
def adopt_rois() -> dict:
    moved = box_store.adopt_manifest_rois(labeling.resolve())
    return {"moved": int(moved)}


class DetectionBody(BaseModel):
    copy_images: bool = False


@router.post("/detection/export")
def detection_export(body: DetectionBody) -> dict:
    """검출 학습 폴더를 **디스크에** 만든다 — 이미지 수천 장은 브라우저로 내려받을 물건이 아니다."""
    resolved = labeling.resolve()
    frame = box_store.load()
    try:
        result = detection.export_yolo(resolved=resolved, frame=frame, copy_images=body.copy_images)
    except OSError as exc:
        raise bad_request(f"내보내기 실패: {exc}") from exc
    return {
        "message": result.as_message(),
        "blocking": result.blocking_note(),
        "notes": result.skipped_notes(),
        "root": str(result.root),
        "command": (
            "pip install ultralytics\n"
            f"PYTHONPATH=src python scripts/train_detector.py --data {result.root / detection.DATA_FILE}"
        ),
    }
