"""1단계: 데이터 수집 — 카탈로그 · 폴더 임포트 · 업로드 · 영상 프레임 추출 · 합성 샘플 · 수집 현황.

`app_pages/p1_ingest.py`의 배선을 그대로 옮긴 것이다. 판단 로직은 코어(`ingest` `video`
`framing` …)에 있고 여기서는 코어를 부르고 JSON으로 바꾼다.

영상 선택기(`/videos` `/videos/upload` `/videos/check`)는 **4단계 판정 영상 탭도 같이 쓴다.**
1단계(추출)와 4단계(판정 영상)가 같은 일을 한다 — «어떤 영상을 쓸 것인가». 화면마다
따로 만들면 한쪽만 고쳐지고 다른 쪽은 옛날 방식으로 남는다 (예전 `ui.video_source`).
"""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, Form, Query, UploadFile
from pydantic import BaseModel

from vision_ai import boxes, config, datasets, framing, glossary, ingest, quality, storage, video

from ..common import bad_request, counts, jsonable, not_found, table
from ..jobs import runner

router = APIRouter(prefix="/ingest", tags=["ingest"])

FIT_STARS = {5: "★★★★★", 4: "★★★★☆", 3: "★★★☆☆", 2: "★★☆☆☆", 1: "★☆☆☆☆"}

MOSTLY_DROPPED = 0.8
"""이 비율을 넘게 버렸으면 사용자에게 알린다. 잘 걸러진 것일 수도, 전멸한 것일 수도 있다."""

SYNTHETIC_SIZES = (128, 192, 256, 320, 512)

PREVIEW_ROWS = 10
"""폴더 구조 해석 미리보기 건수. 라벨 추론이 맞는지 보는 데는 10건이면 된다."""

MANIFEST_ROWS = 2000
"""manifest 원본 표로 보내는 최대 행 수. CSV는 전체를 보내고, 표는 화면이 감당할 만큼만."""

CATALOG_HELP = {
    "일상 적합도": "일상 사물 촬영본으로 이 앱을 시험해 볼 때의 적합도 (별 5개가 가장 적합)",
    "상업적 이용": "회사 업무로 연장할 수 있는지. 원본 배포 페이지에서 재확인할 것",
    "폴더 구조": "'로컬 폴더 임포트' 탭이 이 구조로 폴더를 읽는다",
    "key": "스크립트에서 데이터셋을 가리킬 때 쓰는 식별자",
}


# --- 공통 선택지 ---------------------------------------------------------------

@router.get("/options")
def options() -> dict:
    """화면이 처음 그릴 때 필요한 선택지 전부. 탭 표식(👉/✅)의 근거인 `has_data`도 여기."""
    default = datasets.default_dataset()
    return jsonable({
        "has_data": not storage.load_manifest().empty,
        "default_dataset": {"key": default.key, "name": default.name, "license": default.license,
                            "layout": default.layout},
        "layouts": list(datasets.LAYOUT_OPTIONS),
        "layout_help": dict(datasets.LAYOUT_HELP),
        "labels": [{"key": k, "label": config.LABEL_KO.get(k, k)}
                   for k in (config.LABEL_UNLABELED, config.LABEL_NORMAL, config.LABEL_DEFECT)],
        "defect_types": [{"key": k, "label": config.defect_type_label(k)}
                         for k in (config.DEFECT_TYPE_NONE, *config.DEFECT_TYPES)],
        "image_extensions": sorted(config.IMAGE_EXTENSIONS),
        "video_extensions": sorted(video.VIDEO_EXTENSIONS),
        "surface_styles": list(ingest.SURFACE_STYLES),
        "synthetic_defects": list(ingest.SYNTHETIC_DEFECTS),
        "synthetic_sizes": list(SYNTHETIC_SIZES),
        "model_inputs": [{"name": k, "px": v} for k, v in framing.MODEL_INPUT_PX.items()],
        "safe_px": framing.SAFE_PX,
        "floor_px": framing.FLOOR_PX,
        "default_min_change": video.DEFAULT_MIN_CHANGE,
        "default_frames_per_object": video.DEFAULT_FRAMES_PER_OBJECT,
    })


# --- 🗂️ 오픈 데이터셋 카탈로그 ---------------------------------------------------

@router.get("/catalog")
def catalog() -> dict:
    default = datasets.default_dataset()
    listed = datasets.recommended()
    frame = pd.DataFrame([
        {
            "데이터셋": ("⭐ " if d.is_default else "") + d.name,
            "일상 적합도": FIT_STARS.get(d.everyday_fit, ""),
            "상업적 이용": d.commercial_use,
            "라이선스": d.license,
            "카테고리 수": len(d.categories),
            "폴더 구조": d.layout,
            "key": d.key,
        }
        for d in listed
    ])
    return jsonable({
        "default": {"name": default.name, "license": default.license},
        "table": table(frame),
        "help": CATALOG_HELP,
        "datasets": [
            {
                "key": d.key, "name": d.name, "summary": d.summary, "url": d.url,
                "license": d.license, "commercial_use": d.commercial_use,
                "layout": d.layout, "layout_note": d.layout_note,
                "categories": list(d.categories), "fit_stars": FIT_STARS.get(d.everyday_fit, ""),
                "everyday_note": d.everyday_note, "download_note": d.download_note,
                "license_note": d.license_note, "tags": list(d.tags), "is_default": d.is_default,
            }
            for d in listed
        ],
    })


# --- 📁 로컬 폴더 임포트 ----------------------------------------------------------

def _folder_root(root: str) -> Path:
    path = Path(root).expanduser()
    if not root.strip() or not path.is_dir():
        raise not_found(f"폴더를 찾을 수 없습니다: `{path}`")
    return path


@router.get("/folder/preview")
def folder_preview(root: str, layout: str = "visa") -> dict:
    """경로를 넣으면 이미지 수와 라벨 추론 결과(상위 10건)를 보여준다 — 실제 등록 전에."""
    path = _folder_root(root)
    total = ingest.count_image_files(path)
    rows = []
    for item in islice(ingest.iter_image_files(path), PREVIEW_ROWS):
        parsed = datasets.parse_path(item.relative_to(path), layout)
        rows.append({
            "상대경로": str(item.relative_to(path)),
            "카테고리": parsed["category"],
            "분할": parsed["split"],
            "라벨": config.LABEL_KO.get(parsed["label"], parsed["label"]),
            "결함유형": parsed["defect_type"],
            "마스크": "예" if parsed["is_mask"] else "",
        })
    return jsonable({"root": str(path), "total": total, "preview": table(pd.DataFrame(rows))})


class FolderBody(BaseModel):
    root: str
    layout: str = "visa"
    source: str = "local"
    include_masks: bool = False
    limit: int | None = None


@router.post("/folder/ingest")
def start_folder_ingest(body: FolderBody) -> dict:
    path = _folder_root(body.root)
    if body.layout not in datasets.LAYOUT_OPTIONS:
        raise bad_request(f"알 수 없는 구조 해석 방식: {body.layout}")

    def run(job):
        try:
            result = ingest.ingest_folder(
                path, source=body.source or "local", layout=body.layout,
                include_masks=body.include_masks,
                limit=int(body.limit) if body.limit else None,
                progress=lambda d, t: job.progress(d, t, f"등록 중... {d:,}/{t:,}"),
            )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"등록 실패: {exc}") from exc
        return {"message": result.as_message(), "added": result.added, "duplicates": result.duplicates,
                "skipped_masks": result.skipped_masks, "failed": list(result.failed[:50]),
                "n_failed": len(result.failed)}

    job = runner.submit("등록 중", run, note="파일마다 해시와 품질 지표를 계산합니다. 원본은 복사하지 않습니다.")
    return {"job_id": job.id}


# --- ⬆️ 이미지 업로드 -------------------------------------------------------------

@router.post("/upload")
async def upload_images(
    files: list[UploadFile] = File(...),
    category: str = Form(""),
    label: str = Form(config.LABEL_UNLABELED),
    defect_type: str = Form(config.DEFECT_TYPE_NONE),
) -> dict:
    if label not in config.LABELS:
        raise bad_request(f"알 수 없는 라벨: {label}")
    payload = [(f.filename or "upload", await f.read()) for f in files]
    result = ingest.ingest_uploads(
        payload, category=category.strip(), label=label,
        defect_type=defect_type if label == config.LABEL_DEFECT else config.DEFECT_TYPE_NONE,
    )
    return {"message": result.as_message(), "added": result.added, "duplicates": result.duplicates,
            "failed": list(result.failed[:10]), "n_failed": len(result.failed)}


# --- 영상 선택기 (1단계·4단계 공용) ---------------------------------------------------

def _describe_video(path: Path) -> str:
    """목록에 보일 이름. 크기를 붙여야 같은 이름의 다른 영상을 구분할 수 있다.

    영상마다 열어서 길이를 재면 목록을 그릴 때마다 파일을 전부 여는 셈이 되므로 크기만
    쓴다. MB로만 적으면 작은 영상이 전부 `0MB`가 되어 구분이 되지 않는다.
    (예전 `ui.describe_video` — ui는 Streamlit을 부르므로 여기서 다시 쓴다.)
    """
    try:
        size = path.stat().st_size
    except OSError:
        return path.name
    if size >= 1_048_576:
        return f"{path.name} · {size / 1_048_576:,.1f}MB"
    return f"{path.name} · {max(size // 1024, 1):,}KB"


def _video_entry(path: Path) -> dict:
    return {"path": str(path), "name": path.name, "label": _describe_video(path)}


@router.get("/videos")
def list_videos(scan: str = Query("")) -> dict:
    """고를 수 있는 영상 목록. `scan`에 적은 폴더(하위 폴더 포함)도 함께 훑는다.

    **경로를 전부 타이핑하게 하면 안 된다.** 오타 하나로 실패하고, 무엇보다 어떤 영상이
    이미 올라와 있는지 화면에서 알 수가 없다.
    """
    extra = scan.strip()
    found = video.listed(*([extra] if extra else []))
    return {
        "videos": [_video_entry(p) for p in found],
        "video_dir": str(video.video_dir()),
        "scan_dir": extra,
        "scan_dir_missing": bool(extra) and not Path(extra).expanduser().is_dir(),
    }


@router.post("/videos/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """영상을 프로젝트 영상 폴더에 저장한다. 다음부터는 목록에서 고를 수 있다."""
    name = Path(file.filename or "upload.mp4").name
    if not video.is_video(name):
        raise bad_request(f"영상 파일이 아닙니다: `{Path(name).suffix}`")
    target = video.video_dir() / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return _video_entry(target)


@router.get("/videos/check")
def check_video(path: str) -> dict:
    """직접 입력한 경로가 쓸 수 있는 영상인지. 문장은 화면에 그대로 보인다."""
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise not_found(f"파일을 찾을 수 없습니다: `{candidate}`")
    if not video.is_video(candidate):
        raise bad_request(f"영상 파일이 아닙니다: `{candidate.suffix}`")
    return _video_entry(candidate)


@router.post("/videos/sample")
def make_sample_video() -> dict:
    """물체가 3초마다 하나씩 지나가고 일부에 흠집이 있는 10초짜리 시험용 영상."""
    try:
        made = video.make_sample()
    except OSError as exc:
        raise bad_request(str(exc)) from exc
    return _video_entry(made)


# --- 🎞️ 영상에서 프레임 추출 -------------------------------------------------------

def _probe(path: str) -> video.VideoInfo:
    try:
        info = video.probe(path)
    except OSError as exc:
        raise bad_request(str(exc)) from exc
    if not info.usable:
        raise bad_request("fps나 프레임 수를 읽을 수 없는 영상입니다.")
    return info


@router.get("/video/probe")
def probe_video(path: str) -> dict:
    info = _probe(path)
    return {"path": str(info.path), "fps": info.fps, "frame_count": info.frame_count,
            "width": info.width, "height": info.height, "duration_sec": info.duration_sec}


class FramingBody(BaseModel):
    sensor_px: int
    fov_m: float = 1.0
    defect_mm: float = 10.0
    model: str = next(iter(framing.MODEL_INPUT_PX))
    tiles: int = 1


@router.post("/video/framing")
def framing_check(body: FramingBody) -> dict:
    """이 촬영으로 결함이 보이기는 하는가 (V0).

    **모델을 고르기 전에 답해야 하는 질문이다.** 결함이 3픽셀로 잡히면 어떤 모델을 써도
    안 되는데, 보통은 프레임을 다 뽑고 라벨링을 하고 학습을 돌린 뒤에야 그 사실이 드러난다.
    그래서 화면은 이것을 **추출 버튼 위에** 둔다 — 며칠 뒤에 알 일을 지금 알려주자는 것.
    """
    if body.model not in framing.MODEL_INPUT_PX:
        raise bad_request(f"알 수 없는 모델 입력: {body.model}")
    view = framing.Framing(
        defect_mm=float(body.defect_mm), fov_mm=float(body.fov_m) * 1000,
        sensor_px=int(body.sensor_px), model_px=framing.MODEL_INPUT_PX[body.model],
        tiles=max(1, int(body.tiles)),
    )
    return jsonable({
        "native_px": view.native_px,
        "model_input_px": view.model_input_px,
        "verdict": view.verdict,
        "ok": view.verdict == framing.VERDICT_OK,
        "hard": view.verdict == framing.VERDICT_HARD,
        "smallest_mm": view.smallest_catchable_mm(),
        "safe_px": framing.SAFE_PX,
        "floor_px": framing.FLOOR_PX,
        "advice": framing.advice(view),
        "levers": [{"name": lv.name, "change": lv.change, "cost": lv.cost, "reachable": lv.reachable}
                   for lv in framing.levers(view)],
        "hint": _framing_hint_from_boxes(),
    })


def _framing_hint_from_boxes() -> dict | None:
    """결함 크기를 모를 때, 이미 그려 둔 박스로 어림한다.

    실물 크기를 자로 재 본 사람은 드물지만 라벨링한 박스는 있다. 다만 **비율은 그 촬영을
    어떻게 했느냐의 결과이지 물리 상수가 아니므로**, 그 촬영이 담던 폭을 함께 물어야 한다.
    """
    frame = boxes.load()
    if frame.empty:
        return None
    fraction = framing.fraction_from_boxes(frame, storage.load_manifest())
    if not fraction:
        return None
    return {
        "fraction": fraction,
        "source_fov_m": framing.TYPICAL_SOURCE_FOV_MM / 1000,
        "defect_mm": framing.defect_mm_from_fraction(fraction, framing.TYPICAL_SOURCE_FOV_MM),
    }


class PlanBody(BaseModel):
    path: str
    mode: str = "rate"                      # "rate" (초당 장수) | "process" (공정 값으로 계산)
    per_second: float = 2.0
    field_of_view_m: float = 0.30
    speed_mps: float = 0.5
    frames_per_object: int = video.DEFAULT_FRAMES_PER_OBJECT


def _plan_dict(plan: video.Plan) -> dict:
    return {"stride": plan.stride, "expected_frames": plan.expected_frames, "per_second": plan.per_second,
            "frames_per_object": plan.frames_per_object, "required_fps": plan.required_fps,
            "feasible": plan.feasible, "reason": plan.reason}


@router.post("/video/plan")
def plan_extraction(body: PlanBody) -> dict:
    """간격을 정한다. CCTV처럼 물체가 불규칙하게 지나가면 라인 속도로 계산할 수 없으므로
    초당 장수 지정이 기본이다. 코어의 ValueError는 400으로 화면에 그대로 보인다."""
    info = _probe(body.path)
    if body.mode == "process":
        plan = video.plan_from_process(
            info, field_of_view_m=body.field_of_view_m, speed_mps=body.speed_mps,
            frames_per_object=int(body.frames_per_object),
        )
    else:
        plan = video.plan_from_rate(info, per_second=body.per_second)
    return jsonable(_plan_dict(plan))


class ExtractBody(BaseModel):
    path: str
    stride: int
    category: str = "video"
    dedupe: bool = True
    check_quality: bool = False


@router.post("/video/extract")
def start_extract(body: ExtractBody) -> dict:
    info = _probe(body.path)
    stride = max(1, int(body.stride))

    def run(job):
        try:
            result, extracted = ingest.ingest_video(
                info.path, stride=stride, category=body.category or "video",
                min_change=video.DEFAULT_MIN_CHANGE if body.dedupe else None,
                check_quality=body.check_quality,
                progress=lambda d, t: job.progress(d, t, f"프레임 추출 중 {d:,}/{t:,}"),
            )
        except OSError as exc:
            raise RuntimeError(str(exc)) from exc
        return jsonable({
            "message": result.as_message(),
            "extract_message": extracted.as_message(),
            "video_id": extracted.video_id,
            "added": result.added,
            "duplicates": result.duplicates,
            "scanned": extracted.scanned,
            "kept": extracted.kept,
            "drop_rate": extracted.drop_rate,
            "forced": extracted.forced,
            # 대부분이 버려졌으면 그 사실을 말한다. "N장 추출"만 찍고 넘어가면 40장을 기대했는데
            # 3장이 나온 것을 **2단계에 가서야** 안다. 다만 꼭 오류는 아니다 — CCTV처럼 빈
            # 장면이 대부분이면 많이 버려지는 게 정상이다. 그래서 막지 않고 판단 근거만 준다.
            "mostly_dropped": extracted.scanned >= 5 and extracted.drop_rate >= MOSTLY_DROPPED,
            "samples": [str(p) for p in extracted.saved[:4]],
        })

    job = runner.submit(
        "프레임 추출 중", run,
        note=(f"{info.frame_count:,}프레임을 훑으며 {stride}프레임마다 한 장씩 저장합니다. "
              "건너뛰는 프레임은 디코딩하지 않습니다."),
    )
    return {"job_id": job.id}


# --- 🧪 합성 샘플 생성 -------------------------------------------------------------

class SyntheticBody(BaseModel):
    categories: list[str]
    n_normal: int = 40
    n_defect: int = 20
    size: int = 256
    seed: int = 42
    layout: str = "visa"
    overwrite: bool = True


@router.post("/synthetic")
def start_synthetic(body: SyntheticBody) -> dict:
    if not body.categories:
        raise bad_request("표면 종류를 1개 이상 선택하세요.")
    unknown = [c for c in body.categories if c not in ingest.SURFACE_STYLES]
    if unknown:
        raise bad_request(f"알 수 없는 표면 종류: {', '.join(unknown)}")
    if body.layout not in ("visa", "mvtec"):
        raise bad_request(f"지원하지 않는 layout: {body.layout} (visa 또는 mvtec)")

    def run(job):
        # 생성 단계는 진행률 콜백이 없다. 막대 대신 문구로 «멈춘 것이 아님»을 알린다.
        job.progress(0, 0, "이미지 생성 중...")
        out_dir = ingest.generate_synthetic(
            categories=body.categories, n_normal=int(body.n_normal), n_defect=int(body.n_defect),
            size=int(body.size), seed=int(body.seed), overwrite=body.overwrite, layout=body.layout,
        )
        if body.overwrite:
            ingest.remove_source(ingest.SYNTHETIC_SOURCE)
        result = ingest.ingest_folder(
            out_dir, source=ingest.SYNTHETIC_SOURCE, layout=body.layout,
            progress=lambda d, t: job.progress(d, t, f"등록 중... {d:,}/{t:,}"),
        )
        return {"out_dir": str(out_dir), "message": result.as_message(), "added": result.added}

    total = len(body.categories) * (body.n_normal + body.n_defect)
    job = runner.submit("합성 샘플 생성 중", run, note=f"이미지 {total:,}장을 그리고 등록합니다.")
    return {"job_id": job.id}


# --- 📊 수집 현황 ------------------------------------------------------------------

@router.get("/status")
def status() -> dict:
    df = storage.load_manifest()
    if df.empty:
        return {"empty": True}
    stats = storage.summarize(df)

    # 클래스 불균형은 이후 학습 전략을 좌우하므로 눈에 띄게 알린다
    ratio = None
    if stats["normal"] and stats["defect"]:
        value = stats["normal"] / stats["defect"]
        if value >= 5 or value <= 0.2:
            ratio = value

    crosstab = pd.crosstab(df["source"], df["label"])
    crosstab.columns = [config.LABEL_KO.get(c, c) for c in crosstab.columns]

    flagged = df[df["note"].astype(str).str.contains("blurry|exposed|low_resolution", na=False)]
    flagged_view = None
    if not flagged.empty:
        view = flagged[["image_id", "category", "label", "width", "height", "blur_score", "brightness", "note"]].copy()
        view["점검 결과"] = view["note"].map(quality.describe_flags)
        flagged_view = table(view.drop(columns=["note"]))

    return jsonable({
        "empty": False,
        "stats": stats,
        "ratio": ratio,
        "by_category": counts(df["category"]),
        "by_defect_type": counts(df["defect_type"]),
        "crosstab": table(crosstab, index=True),
        "quality": {
            "flagged": int(len(flagged)),
            "mean_blur": df["blur_score"].mean(),
            "mean_brightness": df["brightness"].mean(),
            "help_blur": glossary.detail("blur_score"),
            "help_brightness": glossary.detail("brightness"),
            "caption": f"{glossary.caption('blur_score')} {glossary.caption('brightness')} "
                       f"{glossary.ARBITRARY['quality']}",
            "flagged_table": flagged_view,
        },
        "categories": sorted(df["category"].dropna().astype(str).unique().tolist()),
        "labels": [{"key": k, "label": config.LABEL_KO.get(k, k)}
                   for k in sorted(df["label"].dropna().astype(str).unique().tolist())],
        "sources": sorted(df["source"].dropna().astype(str).unique().tolist()),
    })


@router.get("/preview")
def preview_images(category: str = "", label: str = "", count: int = Query(8, ge=1, le=48)) -> dict:
    """필터에 맞는 이미지의 무작위 표본. `random_state=0`이라 같은 조건이면 같은 표본이 나온다."""
    df = storage.load_manifest()
    subset = df
    if category:
        subset = subset[subset["category"].astype(str) == category]
    if label:
        subset = subset[subset["label"].astype(str) == label]
    if subset.empty:
        return {"items": [], "total": 0}
    sample = subset.sample(min(count, len(subset)), random_state=0)
    items = []
    for _, row in sample.iterrows():
        caption = f"{row['category']} · {config.LABEL_KO.get(row['label'], row['label'])}"
        if row["defect_type"] not in (config.DEFECT_TYPE_NONE, None) and not pd.isna(row["defect_type"]):
            caption += f" ({row['defect_type']})"
        items.append({"image_id": str(row["image_id"]), "caption": caption,
                      "exists": storage.resolve_path(row["path"]).exists()})
    return {"items": items, "total": int(len(subset))}


@router.get("/manifest")
def manifest() -> dict:
    """manifest 원본 표(앞 MANIFEST_ROWS행)와 내려받기용 CSV 전체."""
    df = storage.load_manifest()
    return jsonable({"table": table(df, limit=MANIFEST_ROWS), "csv": df.to_csv(index=False)})


@router.delete("/sources")
def remove_source(name: str) -> dict:
    """출처별 레코드 삭제. manifest에서 레코드만 제거하며, 원본 이미지 파일은 지우지 않는다."""
    removed = ingest.remove_source(name)
    return {"removed": removed, "message": f"{name} 레코드 {removed:,}건을 제거했습니다."}
