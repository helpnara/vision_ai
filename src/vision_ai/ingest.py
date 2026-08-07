"""1단계: 데이터 수집(입력).

세 가지 경로로 이미지를 수집해 manifest에 등록한다.
1. 오픈 데이터셋을 내려받아 압축 해제한 **로컬 폴더 임포트** (주 경로)
2. 일상에서 직접 촬영한 이미지 **업로드**
3. 파이프라인 검증용 **합성 샘플 생성** (외부 다운로드 없이 전 단계를 돌려보기 위함)
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from . import config, datasets, quality, storage

ProgressCallback = Callable[[int, int], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_image_files(root: Path) -> Iterator[Path]:
    """폴더 아래의 이미지 파일을 정렬된 순서로 순회한다."""
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS:
            yield path


def count_image_files(root: Path) -> int:
    """폴더 아래 이미지 개수를 센다."""
    return sum(1 for _ in iter_image_files(root))


def _build_record(
    path: Path,
    *,
    source: str,
    category: str,
    split: str,
    label: str,
    defect_type: str,
    note: str = "",
) -> dict | None:
    """파일 하나에 대한 manifest 레코드를 만든다. 디코드 실패 시 None."""
    metrics = quality.assess_file(path)
    if metrics is None:
        return None
    sha1 = storage.sha1_of_file(path)
    flag_note = ",".join(metrics.flags)
    return {
        "image_id": storage.image_id_from_sha1(sha1),
        "path": storage.to_manifest_path(path),
        "source": source,
        "category": category,
        "split": split,
        "label": label,
        "defect_type": defect_type,
        "width": metrics.width,
        "height": metrics.height,
        "channels": metrics.channels,
        "filesize": path.stat().st_size,
        "sha1": sha1,
        "blur_score": metrics.blur_score,
        "brightness": metrics.brightness,
        "ingested_at": _now(),
        "note": "; ".join(x for x in (note, flag_note) if x),
    }


class IngestResult:
    """수집 결과 요약."""

    def __init__(self) -> None:
        self.added = 0
        self.duplicates = 0
        self.skipped_masks = 0
        self.failed: list[str] = []

    @property
    def scanned(self) -> int:
        return self.added + self.duplicates + self.skipped_masks + len(self.failed)

    def as_message(self) -> str:
        parts = [f"신규 등록 {self.added}건"]
        if self.duplicates:
            parts.append(f"중복 제외 {self.duplicates}건")
        if self.skipped_masks:
            parts.append(f"마스크 제외 {self.skipped_masks}건")
        if self.failed:
            parts.append(f"읽기 실패 {len(self.failed)}건")
        return ", ".join(parts)


def ingest_folder(
    root: Path,
    *,
    source: str,
    layout: str = "mvtec",
    include_masks: bool = False,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> IngestResult:
    """로컬 폴더의 이미지를 manifest에 등록한다 (파일은 이동/복사하지 않음).

    Args:
        root: 데이터셋 루트 폴더
        source: manifest에 기록할 출처 이름 (보통 데이터셋 key)
        layout: "visa" | "mvtec" | "flat" | "custom" — 경로에서 라벨을 추론하는 방식
        include_masks: ground_truth 마스크 이미지도 등록할지 여부
        limit: 최대 등록 건수 (미리보기용)
        progress: (처리한 수, 전체 수) 콜백
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"폴더를 찾을 수 없습니다: {root}")

    paths = list(iter_image_files(root))
    if limit is not None:
        paths = paths[:limit]
    total = len(paths)

    result = IngestResult()
    records: list[dict] = []

    for index, path in enumerate(paths, start=1):
        parsed = datasets.parse_path(path.relative_to(root), layout)
        if parsed["is_mask"] and not include_masks:
            result.skipped_masks += 1
        else:
            record = _build_record(
                path,
                source=source,
                category=parsed["category"],
                split=parsed["split"],
                label=parsed["label"],
                defect_type=parsed["defect_type"],
                note="mask" if parsed["is_mask"] else "",
            )
            if record is None:
                result.failed.append(str(path))
            else:
                records.append(record)
        if progress is not None:
            progress(index, total)

    added, duplicates = storage.append_records(records)
    result.added = added
    result.duplicates = duplicates
    return result


def ingest_uploads(
    files: Sequence[tuple[str, bytes]],
    *,
    category: str,
    label: str = config.LABEL_UNLABELED,
    defect_type: str = config.DEFECT_TYPE_NONE,
    source: str = "upload",
) -> IngestResult:
    """업로드된 이미지를 data/raw/<source>/<category>/ 에 저장하고 등록한다."""
    target_dir = config.raw_dir() / source / (category or "uncategorized")
    target_dir.mkdir(parents=True, exist_ok=True)

    result = IngestResult()
    records: list[dict] = []

    for name, data in files:
        if quality.decode(data) is None:
            result.failed.append(name)
            continue
        sha1 = storage.sha1_of_bytes(data)
        suffix = Path(name).suffix.lower() or ".png"
        if suffix not in config.IMAGE_EXTENSIONS:
            suffix = ".png"
        destination = target_dir / f"{storage.image_id_from_sha1(sha1)}{suffix}"
        if not destination.exists():
            destination.write_bytes(data)
        record = _build_record(
            destination,
            source=source,
            category=category or "uncategorized",
            split=config.SPLIT_NONE,
            label=label,
            defect_type=defect_type,
            note=f"original_name={name}",
        )
        if record is None:
            result.failed.append(name)
        else:
            records.append(record)

    added, duplicates = storage.append_records(records)
    result.added = added
    result.duplicates = duplicates
    return result


def remove_source(source: str) -> int:
    """특정 출처의 레코드를 manifest에서 제거한다 (파일은 삭제하지 않음)."""
    df = storage.load_manifest()
    if df.empty:
        return 0
    keep = df["source"].astype(str) != source
    removed = int((~keep).sum())
    storage.save_manifest(df[keep].reset_index(drop=True))
    return removed


# --- 합성 샘플 생성 --------------------------------------------------------
# 오픈 데이터셋 다운로드 전에도 수집→라벨링→학습→운영 전 과정을 돌려볼 수 있도록,
# 실제 데이터셋과 같은 폴더 구조(기본 VisA)로 가짜 표면 이미지와 결함 마스크를 만든다.

SYNTHETIC_SOURCE = "synthetic"

# 색상은 OpenCV 관례에 따라 BGR 순서다.
SURFACE_STYLES: dict[str, dict] = {
    # 제조업 PoC 대상 — 기판 위 규칙적 트레이스 패턴. 정렬된 구조라
    # 위치별 이상탐지(models.PatchAnomalyModel의 per_position)에 적합하다.
    "pcb_green": {"base": (60, 110, 45), "grain": "trace", "noise": 6},
    "pcb_blue": {"base": (135, 75, 45), "grain": "trace", "noise": 6},
    "wood_panel": {"base": (150, 180, 205), "grain": "stripe", "noise": 8},
    "fabric": {"base": (170, 165, 160), "grain": "weave", "noise": 12},
    "painted_metal": {"base": (195, 195, 193), "grain": "smooth", "noise": 6},
    # 밝기를 과다 노출 임계값(quality.BRIGHT_THRESHOLD) 아래로 유지한다.
    "ceramic_plate": {"base": (208, 208, 205), "grain": "speckle", "noise": 5},
}

SYNTHETIC_DEFECTS = ("scratch", "dent", "crack", "stain")


def _make_surface(rng: np.random.Generator, style: dict, size: int) -> np.ndarray:
    """스타일에 따른 배경 표면 텍스처를 생성한다."""
    base = np.array(style["base"], dtype=np.float32)
    image = np.tile(base, (size, size, 1))

    grain = style["grain"]
    if grain == "stripe":  # 목재 결
        freq = rng.uniform(0.05, 0.12)
        phase = rng.uniform(0, np.pi)
        wave = np.sin(np.arange(size) * freq + phase) * 12.0
        image += wave[None, :, None]
    elif grain == "weave":  # 직물 짜임
        axis = np.arange(size)
        pattern = (np.sin(axis * 0.6)[:, None] + np.sin(axis * 0.6)[None, :]) * 5.0
        image += pattern[:, :, None]
    elif grain == "trace":  # PCB 기판 위 배선/패드
        pitch = max(size // 8, 16)
        offset = int(rng.integers(0, 5))
        layer = np.zeros_like(image)
        thickness = max(size // 90, 2)
        for position in range(offset, size, pitch):
            cv2.line(layer, (position, 0), (position, size - 1), (58, 48, 42), thickness)
            cv2.line(layer, (0, position), (size - 1, position), (58, 48, 42), thickness)
        radius = max(size // 50, 3)
        for y in range(offset + pitch // 2, size, pitch):
            for x in range(offset + pitch // 2, size, pitch):
                cv2.circle(layer, (x, y), radius, (74, 64, 56), -1, lineType=cv2.LINE_AA)
        image += layer
    elif grain == "speckle":
        speckle = rng.normal(0, 3, (size, size, 1))
        image += speckle

    # 조명 불균일 (실제 촬영 환경 모사)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    center = rng.uniform(0.3, 0.7, 2)
    falloff = 1.0 - 0.25 * ((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    image *= falloff[:, :, None]

    image += rng.normal(0, style["noise"], image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def _feathered_mask(
    size: int, draw: Callable[[np.ndarray], object], sigma: float
) -> np.ndarray:
    """draw로 그린 영역을 경계가 부드러운 0~1 알파 마스크로 만든다."""
    mask = np.zeros((size, size), dtype=np.float32)
    draw(mask)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    peak = float(mask.max())
    return mask / peak if peak > 0 else mask


def _composite(base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """알파 마스크로 overlay를 base에 국소 합성한다.

    결함은 국소 현상이므로 이미지 전체를 흐리게 만들지 않는 것이 중요하다.
    (전체 블렌딩을 쓰면 주변 표면까지 뭉개져 선명도 지표가 함께 떨어진다.)
    """
    weight = alpha[:, :, None].astype(np.float32)
    blended = base.astype(np.float32) * (1.0 - weight) + overlay.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def _mask_from_difference(
    before: np.ndarray, after: np.ndarray, threshold: int = 2
) -> np.ndarray:
    """결함을 그리기 전/후 차이에서 ground truth 마스크를 만든다.

    도형을 마스크에 다시 그리면 페더링·안티에일리어싱 때문에 실제 변경 영역과
    어긋난다. 변화량에서 직접 뽑으면 이미지와 마스크가 항상 일치한다.
    """
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).max(axis=2)
    return ((diff > threshold).astype(np.uint8)) * 255


def _draw_defect(
    rng: np.random.Generator, image: np.ndarray, defect: str
) -> tuple[np.ndarray, np.ndarray]:
    """표면 이미지에 결함을 그려 넣고 (결함 이미지, 결함 마스크)를 반환한다.

    마스크는 ground truth 역할을 한다 — 2단계에서 이 마스크로 ROI를 자동 추출하고,
    3단계에서 위치 예측 성능을 평가할 때 정답으로 쓴다. 그래서 이미지에 그린 것과
    정확히 같은 도형을 마스크에도 그린다.
    """
    out = image.copy()
    size = out.shape[0]

    if defect == "scratch":
        for _ in range(rng.integers(1, 3)):
            start = rng.integers(0, size, 2)
            angle = rng.uniform(0, np.pi)
            length = rng.integers(size // 4, size // 2)
            end = (
                int(np.clip(start[0] + np.cos(angle) * length, 0, size - 1)),
                int(np.clip(start[1] + np.sin(angle) * length, 0, size - 1)),
            )
            shade = int(rng.integers(40, 100))
            thickness = int(rng.integers(1, 3))
            origin = tuple(start.tolist())
            cv2.line(out, origin, end, (shade, shade, shade), thickness, cv2.LINE_AA)

    elif defect == "dent":
        center = tuple(rng.integers(size // 5, size * 4 // 5, 2).tolist())
        radius = int(rng.integers(size // 14, size // 7))
        # 그림자(테두리) + 하이라이트(중심)로 눌린 느낌을 만든다
        shading = np.zeros_like(out)
        cv2.circle(shading, center, radius, (85, 85, 90), -1, lineType=cv2.LINE_AA)
        cv2.circle(shading, center, max(radius - 3, 1), (150, 150, 155), -1, lineType=cv2.LINE_AA)
        shading = cv2.GaussianBlur(shading, (9, 9), 0)
        mask = _feathered_mask(size, lambda m: cv2.circle(m, center, radius, 1.0, -1, lineType=cv2.LINE_AA),
                              sigma=max(radius * 0.25, 1.0))
        out = _composite(out, shading, mask * 0.8)

    elif defect == "crack":
        point = rng.integers(size // 5, size * 4 // 5, 2).astype(np.int32)
        points = [point.copy()]
        direction = rng.uniform(0, 2 * np.pi)
        for _ in range(int(rng.integers(6, 12))):
            direction += rng.uniform(-0.7, 0.7)
            step = rng.integers(6, 18)
            point = np.clip(
                point + (np.array([np.cos(direction), np.sin(direction)]) * step).astype(np.int32),
                0, size - 1,
            )
            points.append(point.copy())
        polyline = [np.array(points, dtype=np.int32)]
        thickness = int(rng.integers(1, 3))
        cv2.polylines(out, polyline, False, (35, 35, 40), thickness, cv2.LINE_AA)

    elif defect == "stain":
        center = tuple(rng.integers(size // 5, size * 4 // 5, 2).tolist())
        axes = tuple(rng.integers(size // 12, size // 5, 2).tolist())
        angle = float(rng.uniform(0, 180))
        tint = tuple(int(v) for v in rng.integers(70, 150, 3))
        tint_layer = np.full_like(out, tint)
        mask = _feathered_mask(
            size,
            lambda m: cv2.ellipse(m, center, axes, angle, 0, 360, 1.0, -1),
            sigma=max(max(axes) * 0.3, 1.0),
        )
        out = _composite(out, tint_layer, mask * 0.6)

    return out, _mask_from_difference(image, out)


def generate_synthetic(
    *,
    categories: Sequence[str] | None = None,
    n_normal: int = 40,
    n_defect: int = 20,
    size: int = 256,
    seed: int = 42,
    out_dir: Path | None = None,
    overwrite: bool = False,
    layout: str = "visa",
) -> Path:
    """합성 데이터셋을 생성하고 경로를 반환한다.

    기본 예시 데이터셋이 VisA이므로 기본 layout도 `visa`로 두어, VisA를 내려받기 전에도
    같은 폴더 구조·마스크 규칙으로 2단계 라벨링(마스크 기반 ROI 추출)까지 시험할 수 있게 한다.

    layout="visa":
        <out_dir>/<category>/Data/Images/Normal/*.png
        <out_dir>/<category>/Data/Images/Anomaly/*.png
        <out_dir>/<category>/Data/Masks/Anomaly/*.png
    layout="mvtec":
        <out_dir>/<category>/train/good/*.png
        <out_dir>/<category>/test/{good,<defect>}/*.png
        <out_dir>/<category>/ground_truth/<defect>/*_mask.png
    """
    if layout not in ("visa", "mvtec"):
        raise ValueError(f"지원하지 않는 layout: {layout} (visa 또는 mvtec)")

    categories = tuple(categories or SURFACE_STYLES.keys())
    out_dir = Path(out_dir) if out_dir else config.raw_dir() / SYNTHETIC_SOURCE
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)

    rng = np.random.default_rng(seed)
    per_defect = max(1, n_defect // len(SYNTHETIC_DEFECTS))

    for category in categories:
        style = SURFACE_STYLES.get(category, next(iter(SURFACE_STYLES.values())))

        if layout == "visa":
            normal_dir = out_dir / category / "Data" / "Images" / "Normal"
            anomaly_dir = out_dir / category / "Data" / "Images" / "Anomaly"
            mask_dir = out_dir / category / "Data" / "Masks" / "Anomaly"
            for directory in (normal_dir, anomaly_dir, mask_dir):
                directory.mkdir(parents=True, exist_ok=True)

            for i in range(n_normal):
                cv2.imwrite(str(normal_dir / f"{i:04d}.png"), _make_surface(rng, style, size))

            # VisA는 결함 유형을 폴더로 나누지 않는다 — 한 폴더에 모아 쓴다
            index = 0
            for defect in SYNTHETIC_DEFECTS:
                for _ in range(per_defect):
                    surface = _make_surface(rng, style, size)
                    image, mask = _draw_defect(rng, surface, defect)
                    cv2.imwrite(str(anomaly_dir / f"{index:04d}.png"), image)
                    cv2.imwrite(str(mask_dir / f"{index:04d}.png"), mask)
                    index += 1
            continue

        # mvtec
        train_good = out_dir / category / "train" / "good"
        test_good = out_dir / category / "test" / "good"
        train_good.mkdir(parents=True, exist_ok=True)
        test_good.mkdir(parents=True, exist_ok=True)

        # 정상: 80%는 학습용, 20%는 테스트용
        n_test_normal = max(1, n_normal // 5)
        for i in range(n_normal):
            surface = _make_surface(rng, style, size)
            target = test_good if i < n_test_normal else train_good
            cv2.imwrite(str(target / f"{i:04d}.png"), surface)

        for defect in SYNTHETIC_DEFECTS:
            defect_dir = out_dir / category / "test" / defect
            truth_dir = out_dir / category / "ground_truth" / defect
            defect_dir.mkdir(parents=True, exist_ok=True)
            truth_dir.mkdir(parents=True, exist_ok=True)
            for i in range(per_defect):
                surface = _make_surface(rng, style, size)
                image, mask = _draw_defect(rng, surface, defect)
                cv2.imwrite(str(defect_dir / f"{i:04d}.png"), image)
                cv2.imwrite(str(truth_dir / f"{i:04d}_mask.png"), mask)

    return out_dir
