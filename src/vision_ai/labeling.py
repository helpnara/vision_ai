"""2단계: 라벨링.

manifest(1단계 수집 사실)를 직접 수정하지 않고, 사람의 판정을 **덮어쓰기 계층**으로 분리한다.

- `labels.csv`  : 사람이 남긴 라벨 이벤트 (append-only 이력 — 누가 언제 무엇을 바꿨는지 추적)
- `splits.csv`  : image_id → train/val/test (층화 분할로 재생성 가능한 파생 데이터)
- `defect_type_map.csv` : 데이터셋 고유 결함 유형명 → 프로젝트 표준 유형 매핑

`resolve()`가 이 세 계층을 manifest 위에 얹어 "현재 유효한 라벨"을 만든다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, storage

# --- 파일 경로 -------------------------------------------------------------
# 경로는 호출 시점에 config에서 읽는다 (테스트에서 데이터 루트를 바꿔 끼울 수 있도록).


def _labels_path() -> Path:
    return config.labels_path()


def _splits_path() -> Path:
    return config.data_root() / "splits.csv"


def _type_map_path() -> Path:
    return config.data_root() / "defect_type_map.csv"


# --- 라벨 이벤트 이력 -------------------------------------------------------
EVENT_COLUMNS: tuple[str, ...] = (
    "image_id",
    "label",         # normal / defect / unlabeled
    "defect_type",   # 표준 결함 유형 키
    "roi_x", "roi_y", "roi_w", "roi_h",   # 결함 위치 (원본 픽셀 기준, 없으면 결측)
    "verified",      # 폴더 라벨을 사람이 확인했는지
    "labeled_by",
    "labeled_at",
    "note",
)

_ROI_COLUMNS = ("roi_x", "roi_y", "roi_w", "roi_h")


def empty_events() -> pd.DataFrame:
    return pd.DataFrame({col: pd.Series(dtype="object") for col in EVENT_COLUMNS})


def load_events() -> pd.DataFrame:
    """라벨 이벤트 이력 전체를 시간순으로 읽는다."""
    path = _labels_path()
    if not path.exists():
        return empty_events()
    df = pd.read_csv(path, dtype={"image_id": "str"})
    for col in EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in _ROI_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["verified"] = df["verified"].astype("object").map(_as_bool)
    return df[list(EVENT_COLUMNS)]


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "y")


def record_label(
    image_id: str,
    *,
    label: str,
    defect_type: str = config.DEFECT_TYPE_NONE,
    roi: tuple[int, int, int, int] | None = None,
    verified: bool = True,
    labeled_by: str = "human",
    note: str = "",
) -> None:
    """라벨 이벤트를 한 건 추가한다 (기존 이력은 지우지 않는다)."""
    record_labels([
        {
            "image_id": image_id,
            "label": label,
            "defect_type": defect_type,
            "roi": roi,
            "verified": verified,
            "labeled_by": labeled_by,
            "note": note,
        }
    ])


def record_labels(items: list[dict]) -> int:
    """여러 라벨 이벤트를 한 번에 추가한다. 추가된 건수를 반환한다."""
    if not items:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for item in items:
        roi = item.get("roi")
        row = {
            "image_id": str(item["image_id"]),
            "label": item["label"],
            "defect_type": item.get("defect_type", config.DEFECT_TYPE_NONE),
            "verified": bool(item.get("verified", True)),
            "labeled_by": item.get("labeled_by", "human"),
            "labeled_at": item.get("labeled_at", now),
            "note": item.get("note", ""),
        }
        for col, value in zip(_ROI_COLUMNS, roi if roi else (None,) * 4):
            row[col] = value
        rows.append(row)

    config.ensure_dirs()
    incoming = pd.DataFrame(rows)[list(EVENT_COLUMNS)]
    path = _labels_path()
    header = not path.exists() or path.stat().st_size == 0
    incoming.to_csv(path, mode="a", header=header, index=False)
    return len(incoming)


def latest_events() -> pd.DataFrame:
    """image_id별 가장 최근 라벨 이벤트만 남긴다 (기록 순서 기준)."""
    events = load_events()
    if events.empty:
        return events
    return events.drop_duplicates(subset="image_id", keep="last").set_index("image_id")


# --- 결함 유형 정규화 매핑 --------------------------------------------------
# 데이터셋마다 결함 유형명이 다르다. 표준 유형(config.DEFECT_TYPES)으로 모아야
# 여러 데이터셋을 함께 학습·평가할 수 있다.
TYPE_MAP_COLUMNS: tuple[str, ...] = ("raw_type", "standard_type")

# MVTec AD 등에서 흔히 쓰이는 이름의 기본 매핑. 나머지는 UI에서 사람이 지정한다.
DEFAULT_TYPE_MAP: dict[str, str] = {
    "scratch": "scratch",
    "scratch_head": "scratch",
    "scratch_neck": "scratch",
    "rough": "scratch",
    "crack": "crack",
    "cracked": "crack",
    "broken": "broken",
    "broken_large": "broken",
    "broken_small": "broken",
    "broken_teeth": "broken",
    "contamination": "contamination",
    "metal_contamination": "contamination",
    "glue": "contamination",
    "glue_strip": "contamination",
    "oil": "stain",
    "liquid": "stain",
    "stain": "stain",
    "color": "discoloration",
    "discolor": "discoloration",
    "print": "discoloration",
    "faulty_imprint": "discoloration",
    "bent": "deformation",
    "bent_lead": "deformation",
    "bent_wire": "deformation",
    "squeeze": "deformation",
    "squeezed_teeth": "deformation",
    "fold": "deformation",
    "flip": "deformation",
    "hole": "hole",
    "poke": "hole",
    "cut": "hole",
    "torn": "hole",
    "tear": "hole",
    "thread": "other",
    "combined": "other",
    "defective": "other",
    "misplaced": "other",
}


def load_type_map() -> dict[str, str]:
    """결함 유형 매핑을 읽는다. 파일이 없으면 기본 매핑을 반환한다."""
    path = _type_map_path()
    if not path.exists():
        return dict(DEFAULT_TYPE_MAP)
    df = pd.read_csv(path)
    if not set(TYPE_MAP_COLUMNS) <= set(df.columns):
        return dict(DEFAULT_TYPE_MAP)
    mapping = dict(DEFAULT_TYPE_MAP)
    mapping.update(
        {str(r.raw_type): str(r.standard_type) for r in df.itertuples() if pd.notna(r.standard_type)}
    )
    return mapping


def save_type_map(mapping: dict[str, str]) -> None:
    """결함 유형 매핑을 저장한다."""
    config.ensure_dirs()
    rows = [{"raw_type": k, "standard_type": v} for k, v in sorted(mapping.items())]
    pd.DataFrame(rows, columns=list(TYPE_MAP_COLUMNS)).to_csv(_type_map_path(), index=False)


def normalize_defect_type(raw: str, mapping: dict[str, str]) -> str:
    """데이터셋 고유 결함 유형명을 표준 유형으로 변환한다.

    매핑에 없고 표준 유형도 아니면 원본 값을 그대로 남긴다 — 사람이 매핑해야 한다는
    사실이 드러나야 하므로 임의로 `other`로 뭉개지 않는다.
    """
    key = str(raw)
    if key in mapping:
        return mapping[key]
    if config.is_standard_defect_type(key):
        return key
    return key


def unmapped_defect_types(resolved: pd.DataFrame) -> list[str]:
    """표준 유형으로 정규화되지 않은 결함 유형명을 반환한다."""
    if resolved.empty:
        return []
    defects = resolved[resolved["label"] == config.LABEL_DEFECT]
    raw = defects["defect_type"].dropna().astype(str).unique()
    return sorted(t for t in raw if not config.is_standard_defect_type(t))


# --- 데이터 분할 -----------------------------------------------------------
SPLIT_COLUMNS: tuple[str, ...] = ("image_id", "split")


def load_splits() -> dict[str, str]:
    """image_id → split 매핑을 읽는다."""
    path = _splits_path()
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype={"image_id": "str"})
    if not set(SPLIT_COLUMNS) <= set(df.columns):
        return {}
    return {str(r.image_id): str(r.split) for r in df.itertuples()}


def save_splits(mapping: dict[str, str]) -> None:
    """image_id → split 매핑을 저장한다 (전체 덮어쓰기)."""
    config.ensure_dirs()
    rows = [{"image_id": k, "split": v} for k, v in sorted(mapping.items())]
    pd.DataFrame(rows, columns=list(SPLIT_COLUMNS)).to_csv(_splits_path(), index=False)


def clear_splits() -> None:
    """분할 정보를 삭제한다."""
    _splits_path().unlink(missing_ok=True)


def _allocate(total: int, ratios: tuple[float, ...]) -> list[int]:
    """비율에 맞춰 total을 정수로 배분한다 (최대 잉여법 — 합이 정확히 total)."""
    weights = np.array(ratios, dtype=float)
    weights = weights / weights.sum()
    exact = weights * total
    counts = np.floor(exact).astype(int)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(exact - counts))
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1
    return counts.tolist()


SPLIT_BY_RANDOM = "random"
SPLIT_BY_GROUP = "group"
SPLIT_BY_TIME = "time"
SPLIT_MODES = (SPLIT_BY_GROUP, SPLIT_BY_TIME, SPLIT_BY_RANDOM)

SPLIT_MODE_LABELS = {
    SPLIT_BY_GROUP: "영상(그룹) 단위로 나누기",
    SPLIT_BY_TIME: "시간 순서로 나누기",
    SPLIT_BY_RANDOM: "무작위로 나누기",
}


def image_groups(resolved: pd.DataFrame) -> pd.Series:
    """이미지가 속한 그룹. 영상에서 뽑은 프레임은 영상 id, 나머지는 자기 자신.

    그룹이 필요한 이유는 하나다 — **같은 영상의 프레임은 서로 너무 비슷하다.** 무작위로
    나누면 학습에 쓴 것과 거의 같은 장면이 평가에 들어가 성능이 실제보다 높게 나온다.
    영상 데이터로 넘어갈 때 가장 흔하고 가장 비싼 실수다.

    `group` 열이 비어 있는 예전 데이터는 이미지 하나가 곧 그룹이므로 지금 동작 그대로다.
    """
    ids = resolved["image_id"].astype(str)
    if "group" not in resolved.columns:
        return ids
    groups = resolved["group"].astype(str).replace({"nan": "", "None": ""}).fillna("")
    return groups.where(groups.str.strip() != "", ids)


def assign_splits(
    resolved: pd.DataFrame,
    *,
    train: float = 0.6,
    val: float = 0.2,
    test: float = 0.2,
    seed: int = 42,
    labeled_only: bool = True,
    mode: str = SPLIT_BY_GROUP,
) -> dict[str, str]:
    """카테고리 × 라벨로 층화하여 train/val/test를 배정한다.

    층화하지 않으면 특정 카테고리나 결함 클래스가 한쪽 분할에만 몰려
    평가 결과를 신뢰할 수 없게 된다.

    ``mode``는 **무엇을 하나로 묶어 옮길지**를 정한다.

    * ``group`` (기본) — 같은 영상의 프레임은 통째로 같은 분할로 간다. 영상이 없는
      데이터에서는 이미지 하나가 곧 그룹이라 무작위와 결과가 같다.
    * ``time`` — 수집 시각 순으로 앞은 학습, 뒤는 평가. 영상이 하나뿐이라 그룹으로는
      나눌 수 없을 때 쓴다. 경계 부근 프레임은 여전히 비슷해 누수가 조금 남는다.
    * ``random`` — 이미지 단위 무작위. **영상 프레임에는 쓰면 안 된다.**
    """
    if resolved.empty:
        return {}
    if mode not in SPLIT_MODES:
        raise ValueError(f"지원하지 않는 분할 방식: {mode} (가능: {SPLIT_MODES})")

    subset = resolved
    if labeled_only:
        subset = subset[subset["label"].isin([config.LABEL_NORMAL, config.LABEL_DEFECT])]
    if subset.empty:
        return {}

    rng = np.random.default_rng(seed)
    ratios = (train, val, test)
    names = (config.SPLIT_TRAIN, config.SPLIT_VAL, config.SPLIT_TEST)
    mapping: dict[str, str] = {}

    for _, stratum in subset.groupby(["category", "label"], dropna=False, sort=True):
        if mode == SPLIT_BY_TIME:
            # 시간 순으로 **이어서** 자른다. 영상이 하나뿐이면 그룹으로는 못 나누므로
            # 앞부분으로 배우고 뒷부분으로 평가하는 것이 그나마 가까운 대용이다.
            order = "ingested_at" if "ingested_at" in stratum.columns else "image_id"
            ids = stratum.sort_values([order, "image_id"], kind="stable")["image_id"].astype(str)
            _assign_in_order(list(ids), _allocate(len(ids), ratios), names, mapping)
        elif mode == SPLIT_BY_RANDOM:
            ids = stratum["image_id"].astype(str).to_numpy()
            ids = ids[rng.permutation(len(ids))]
            _assign_in_order(list(ids), _allocate(len(ids), ratios), names, mapping)
        else:
            members = (
                stratum.assign(_group=image_groups(stratum).to_numpy())
                .groupby("_group")["image_id"]
                .apply(lambda s: [str(v) for v in s])
                .to_dict()
            )
            keys = sorted(members)
            keys = [keys[i] for i in rng.permutation(len(keys))]
            for key, bucket in _fit_groups(keys, members, ratios).items():
                for image_id in members[key]:
                    mapping[image_id] = names[bucket]
    return mapping


def _assign_in_order(ids, counts, names, mapping) -> None:
    start = 0
    for name, count in zip(names, counts):
        for image_id in ids[start:start + count]:
            mapping[str(image_id)] = name
        start += count


def _fit_groups(keys, members, ratios) -> dict[str, int]:
    """그룹을 통째로 분할에 배정한다. 한 그룹이 두 분할에 걸치면 누수가 생긴다.

    **큰 그룹부터 가장 덜 찬 분할에 넣는다.** 섞인 순서대로 앞에서 잘라 나가면, 큰 영상이
    마지막에 오는 순간 평가 분할이 학습보다 커지는 일이 생긴다(실제로 40:2:2:2 구성에서
    학습 4장 / 평가 40장이 나왔다).

    나누는 기준은 그룹 **개수**가 아니라 그룹이 담은 **이미지 수**다. 영상마다 프레임 수가
    크게 다르기 때문이다. 크기가 같은 그룹끼리는 섞인 순서를 그대로 따르므로 시드가 바뀌면
    결과도 바뀐다.
    """
    if not keys:
        return {}

    sizes = {key: len(members.get(key, [])) or 1 for key in keys}
    total = sum(sizes.values())
    targets = [total * ratio for ratio in ratios]
    usable = [index for index, ratio in enumerate(ratios) if ratio > 0] or [0]

    filled = [0.0, 0.0, 0.0]
    placed: dict[str, int] = {}
    order = sorted(range(len(keys)), key=lambda i: (-sizes[keys[i]], i))

    for step, position in enumerate(order):
        key = keys[position]
        empty = [index for index in usable if index not in set(placed.values())]
        remaining = len(order) - step
        if empty and remaining <= len(empty):
            # 남은 그룹으로는 빈 분할을 채우는 것이 우선이다 — 평가 분할이 비면
            # 3단계 학습 자체가 막힌다.
            bucket = empty[0]
        else:
            bucket = min(
                usable,
                key=lambda index: (
                    (filled[index] + sizes[key] / 2) / targets[index]
                    if targets[index] > 0
                    else float("inf")
                ),
            )
        placed[key] = bucket
        filled[bucket] += sizes[key]
    return placed


def import_split_csv(csv_source, manifest: pd.DataFrame) -> tuple[dict[str, str], int]:
    """외부 CSV의 공식 분할 정의를 가져온다 (예: VisA `split_csv/1cls.csv`).

    이미지 경로 컬럼과 split 컬럼을 자동으로 찾아 manifest의 image_id에 대응시킨다.

    매칭은 **경로 접미사** 기준이다. manifest는 데이터 루트 기준 경로를, CSV는 데이터셋 루트
    기준 경로를 담기 때문에 앞부분이 다를 수 있다. 파일명만으로 매칭하면 안 되는데,
    VisA는 한 카테고리 안에서 `Normal/0000.JPG`와 `Anomaly/0000.JPG`처럼 같은 파일명을
    재사용하므로 파일명이 고유하지 않다.

    Args:
        csv_source: 파일 경로 또는 파일 객체 (업로드된 파일도 그대로 받는다)
        manifest: 대응시킬 manifest

    Returns:
        (image_id → split 매핑, 매칭되지 않은 CSV 행 수)
    """
    df = pd.read_csv(csv_source)
    lowered = {str(c).lower().strip(): c for c in df.columns}

    split_col = next((lowered[c] for c in ("split", "set", "subset", "phase") if c in lowered), None)
    image_col = next(
        (lowered[c] for c in ("image", "image_path", "path", "filename", "file", "img") if c in lowered),
        None,
    )
    if split_col is None or image_col is None:
        raise ValueError(
            f"CSV에서 분할/이미지 컬럼을 찾을 수 없습니다. 컬럼: {list(df.columns)}"
        )

    # manifest 경로의 모든 접미사를 색인해 둔다 (뒤에서부터 k개 조각)
    suffix_index: dict[tuple[str, ...], list[str]] = {}
    for path_value, image_id in zip(
        manifest["path"].astype(str), manifest["image_id"].astype(str)
    ):
        parts = _path_parts(path_value)
        for k in range(1, len(parts) + 1):
            suffix_index.setdefault(parts[-k:], []).append(image_id)

    mapping: dict[str, str] = {}
    unmatched = 0
    # 컬럼명에 공백·특수문자가 있어도 안전하도록 Series를 직접 zip한다
    for raw_image, raw_split in zip(df[image_col].astype(str), df[split_col].astype(str)):
        split = _normalize_split(raw_split)
        if split is None:
            unmatched += 1
            continue
        image_id = _match_by_suffix(_path_parts(raw_image), suffix_index)
        if image_id is None:
            unmatched += 1
        else:
            mapping[image_id] = split
    return mapping, unmatched


def _path_parts(path_value: str) -> tuple[str, ...]:
    """경로 문자열을 조각으로 나눈다 (윈도우식 구분자도 처리)."""
    return Path(str(path_value).replace("\\", "/")).parts


def _match_by_suffix(
    parts: tuple[str, ...], suffix_index: dict[tuple[str, ...], list[str]]
) -> str | None:
    """가장 긴(=가장 구체적인) 접미사로 유일하게 매칭되는 image_id를 찾는다."""
    for k in range(len(parts), 0, -1):
        candidates = suffix_index.get(parts[-k:])
        if candidates:
            # 존재하는 가장 긴 접미사가 모호하면 더 짧은 접미사는 더 모호하다
            return candidates[0] if len(candidates) == 1 else None
    return None


# --- ground truth 마스크에서 ROI 추출 --------------------------------------
# VisA·MVTec AD 모두 결함 픽셀 마스크를 제공한다. 이를 활용하면 결함 위치를
# 손으로 그리지 않고 자동으로 얻을 수 있다.

def find_mask_path(image_path: Path) -> Path | None:
    """이미지에 대응하는 결함 마스크 경로를 찾는다. 없으면 None.

    - VisA : <물체>/Data/Images/Anomaly/000.JPG -> <물체>/Data/Masks/Anomaly/000.png
    - MVTec: <카테고리>/test/<유형>/000.png     -> <카테고리>/ground_truth/<유형>/000_mask.png
    """
    image_path = Path(image_path)
    parts = list(image_path.parts)
    candidates: list[Path] = []

    # VisA: 경로의 Images -> Masks
    for index, part in enumerate(parts):
        if part.lower() == "images":
            for suffix in (".png", ".PNG", image_path.suffix):
                swapped = parts.copy()
                swapped[index] = "Masks"
                candidates.append(Path(*swapped).with_suffix(suffix))
            break

    # MVTec: test -> ground_truth, 파일명에 _mask 접미사
    for index, part in enumerate(parts):
        if part.lower() == "test":
            swapped = parts.copy()
            swapped[index] = "ground_truth"
            base = Path(*swapped)
            candidates.append(base.with_name(f"{image_path.stem}_mask.png"))
            candidates.append(base.with_name(f"{image_path.stem}.png"))
            break

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def roi_from_mask(mask_path: Path) -> tuple[int, int, int, int] | None:
    """마스크의 결함 픽셀을 감싸는 바운딩박스 (x, y, w, h)를 계산한다."""
    import cv2  # 지연 임포트 — 라벨 저장 로직은 OpenCV 없이도 동작해야 한다

    data = np.frombuffer(Path(mask_path).read_bytes(), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def roi_from_image_path(path_value: str) -> tuple[int, int, int, int] | None:
    """manifest의 path 값으로 마스크를 찾아 ROI를 계산한다."""
    mask_path = find_mask_path(storage.resolve_path(path_value))
    return roi_from_mask(mask_path) if mask_path else None


def _normalize_split(value: str) -> str | None:
    key = value.strip().lower()
    if key in ("train", "training"):
        return config.SPLIT_TRAIN
    if key in ("val", "valid", "validation"):
        return config.SPLIT_VAL
    if key in ("test", "testing"):
        return config.SPLIT_TEST
    return None


# --- 유효 라벨 계산 --------------------------------------------------------

def resolve(manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """manifest에 라벨 이벤트·분할·유형 매핑을 얹어 현재 유효한 라벨을 만든다.

    추가 컬럼:
        raw_defect_type : 데이터셋 원본 결함 유형명
        label_source    : "human" (사람이 지정) / "folder" (폴더 구조 추론) / "none"
        verified        : 사람이 확인했는지
        roi_*           : 결함 위치
    """
    df = (storage.load_manifest() if manifest is None else manifest).copy()
    if df.empty:
        for col in ("raw_defect_type", "label_source", "verified", *_ROI_COLUMNS):
            df[col] = pd.Series(dtype="object")
        return df

    df["image_id"] = df["image_id"].astype(str)
    df["raw_defect_type"] = df["defect_type"].astype(str)

    mapping = load_type_map()
    df["defect_type"] = [normalize_defect_type(v, mapping) for v in df["raw_defect_type"]]

    df["label_source"] = np.where(
        df["label"].astype(str) == config.LABEL_UNLABELED, "none", "folder"
    )
    df["verified"] = False
    for col in _ROI_COLUMNS:
        df[col] = pd.NA

    # 사람 라벨 이벤트를 병합해 덮어쓴다 (행 단위 루프 대신 merge — 대용량에서도 빠르게)
    events = latest_events()
    if not events.empty:
        overrides = events.reset_index()[
            ["image_id", "label", "defect_type", "verified", *_ROI_COLUMNS]
        ].add_prefix("ev_").rename(columns={"ev_image_id": "image_id"})
        df = df.merge(overrides, on="image_id", how="left")

        has_event = df["ev_label"].notna()
        df["label"] = df["ev_label"].where(has_event, df["label"])
        df["defect_type"] = np.where(
            has_event,
            [normalize_defect_type(v, mapping) for v in df["ev_defect_type"].astype(str)],
            df["defect_type"],
        )
        df["label_source"] = np.where(has_event, "human", df["label_source"])
        df["verified"] = df["ev_verified"].where(has_event, False).fillna(False).astype(bool)
        for col in _ROI_COLUMNS:
            df[col] = df[f"ev_{col}"]
        df = df.drop(columns=[c for c in df.columns if c.startswith("ev_")])

    splits = load_splits()
    if splits:
        df["split"] = [splits.get(i, config.SPLIT_NONE) for i in df["image_id"]]

    return df


def stats(resolved: pd.DataFrame) -> dict:
    """2단계 진행 상황 요약."""
    if resolved.empty:
        return {
            "total": 0, "normal": 0, "defect": 0, "unlabeled": 0,
            "human": 0, "verified": 0, "unspecified_type": 0,
            "unmapped_types": 0, "with_roi": 0, "split_assigned": 0,
        }
    labels = resolved["label"].astype(str)
    types = resolved["defect_type"].astype(str)
    is_defect = labels == config.LABEL_DEFECT
    return {
        "total": len(resolved),
        "normal": int((labels == config.LABEL_NORMAL).sum()),
        "defect": int(is_defect.sum()),
        "unlabeled": int((labels == config.LABEL_UNLABELED).sum()),
        "human": int((resolved["label_source"].astype(str) == "human").sum()),
        "verified": int(resolved["verified"].fillna(False).astype(bool).sum()),
        "unspecified_type": int((is_defect & (types == config.DEFECT_TYPE_UNSPECIFIED)).sum()),
        "unmapped_types": len(unmapped_defect_types(resolved)),
        "with_roi": int(resolved["roi_w"].notna().sum()),
        "split_assigned": int(
            resolved["split"].astype(str).isin(
                [config.SPLIT_TRAIN, config.SPLIT_VAL, config.SPLIT_TEST]
            ).sum()
        ),
    }


def review_queue(
    resolved: pd.DataFrame,
    *,
    mode: str = "unlabeled",
    category: str | None = None,
) -> pd.DataFrame:
    """검수해야 할 이미지 목록을 우선순위대로 반환한다.

    mode:
        unlabeled  — 라벨이 없는 이미지
        unspecified— 결함이지만 유형이 미지정인 이미지 (VisA가 여기에 해당)
        unverified — 폴더 라벨을 아직 사람이 확인하지 않은 이미지
        unmapped   — 표준 유형으로 정규화되지 않은 결함 유형
        all        — 전체
    """
    if resolved.empty:
        return resolved

    df = resolved
    if category:
        df = df[df["category"].astype(str) == category]

    labels = df["label"].astype(str)
    types = df["defect_type"].astype(str)

    if mode == "unlabeled":
        df = df[labels == config.LABEL_UNLABELED]
    elif mode == "unspecified":
        df = df[(labels == config.LABEL_DEFECT) & (types == config.DEFECT_TYPE_UNSPECIFIED)]
    elif mode == "unverified":
        df = df[~df["verified"].fillna(False).astype(bool)]
    elif mode == "unmapped":
        unmapped = set(unmapped_defect_types(resolved))
        df = df[(labels == config.LABEL_DEFECT) & types.isin(unmapped)]

    return df.sort_values(["category", "image_id"]).reset_index(drop=True)
