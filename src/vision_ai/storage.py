"""수집 이미지 인덱스(manifest) 저장소.

manifest는 "어떤 이미지가 어디서 왔고 어떤 상태인지"를 담는 단일 진실 원천이다.
1단계(수집)에서 생성하고, 2단계(라벨링) 이후 단계가 이를 읽어 사용한다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config

# manifest 컬럼 정의 (순서 고정)
MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",      # sha1 앞 16자 — 이미지 고유 식별자
    "path",          # DATA_ROOT 기준 상대경로 (외부 경로는 절대경로)
    "source",        # 출처: 데이터셋 key / "upload" / "synthetic" / "local"
    "category",      # 대상 물건 카테고리 (bottle, wood, ...)
    "split",         # train / val / test / unassigned
    "label",         # normal / defect / unlabeled
    "defect_type",   # 결함 유형 키 (없으면 none)
    "width",
    "height",
    "channels",
    "filesize",
    "sha1",
    "blur_score",    # Laplacian 분산 — 낮으면 흐림
    "brightness",    # 평균 밝기 0-255
    "ingested_at",   # ISO8601
    "note",
)

_NUMERIC_COLUMNS = ("width", "height", "channels", "filesize", "blur_score", "brightness")


def empty_manifest() -> pd.DataFrame:
    """빈 manifest DataFrame을 반환한다."""
    return pd.DataFrame({col: pd.Series(dtype="object") for col in MANIFEST_COLUMNS})


def load_manifest() -> pd.DataFrame:
    """manifest를 읽는다. 파일이 없으면 빈 DataFrame을 반환한다."""
    if not config.MANIFEST_PATH.exists():
        return empty_manifest()
    df = pd.read_csv(config.MANIFEST_PATH, dtype={"image_id": "str", "sha1": "str"})
    # 컬럼 스키마가 바뀐 경우에도 깨지지 않도록 보정
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in _NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[list(MANIFEST_COLUMNS)]


def save_manifest(df: pd.DataFrame) -> None:
    """manifest를 저장한다."""
    config.ensure_dirs()
    df.to_csv(config.MANIFEST_PATH, index=False)


def append_records(records: Iterable[dict]) -> tuple[int, int]:
    """레코드를 manifest에 추가한다.

    sha1이 동일한 이미지는 중복으로 보고 건너뛴다.

    Returns:
        (추가된 건수, 중복으로 건너뛴 건수)
    """
    records = list(records)
    if not records:
        return 0, 0

    incoming = pd.DataFrame(records)
    for col in MANIFEST_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = pd.NA
    incoming = incoming[list(MANIFEST_COLUMNS)]

    # 입력 배치 내부 중복 제거
    before = len(incoming)
    incoming = incoming.drop_duplicates(subset="sha1", keep="first")
    skipped = before - len(incoming)

    current = load_manifest()
    if not current.empty:
        known = set(current["sha1"].dropna().astype(str))
        mask = ~incoming["sha1"].astype(str).isin(known)
        skipped += int((~mask).sum())
        incoming = incoming[mask]

    if incoming.empty:
        return 0, skipped

    merged = pd.concat([current, incoming], ignore_index=True) if not current.empty else incoming
    save_manifest(merged)
    return len(incoming), skipped


def sha1_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """파일의 sha1 해시를 계산한다."""
    digest = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha1_of_bytes(data: bytes) -> str:
    """바이트열의 sha1 해시를 계산한다."""
    return hashlib.sha1(data).hexdigest()


def image_id_from_sha1(sha1: str) -> str:
    """sha1에서 image_id(앞 16자)를 만든다."""
    return sha1[:16]


def resolve_path(path_value: str) -> Path:
    """manifest의 path 값을 실제 파일 경로로 해석한다."""
    path = Path(str(path_value))
    return path if path.is_absolute() else config.DATA_ROOT / path


def to_manifest_path(path: Path) -> str:
    """실제 경로를 manifest에 저장할 문자열로 변환한다 (가능하면 상대경로)."""
    path = path.resolve()
    try:
        return str(path.relative_to(config.DATA_ROOT))
    except ValueError:
        return str(path)


def summarize(df: pd.DataFrame) -> dict:
    """manifest 요약 통계를 반환한다."""
    if df.empty:
        return {
            "total": 0,
            "sources": 0,
            "categories": 0,
            "labeled": 0,
            "normal": 0,
            "defect": 0,
            "unlabeled": 0,
        }
    label_counts = df["label"].value_counts()
    normal = int(label_counts.get(config.LABEL_NORMAL, 0))
    defect = int(label_counts.get(config.LABEL_DEFECT, 0))
    return {
        "total": len(df),
        "sources": int(df["source"].nunique()),
        "categories": int(df["category"].nunique()),
        "labeled": normal + defect,
        "normal": normal,
        "defect": defect,
        "unlabeled": int(label_counts.get(config.LABEL_UNLABELED, 0)),
    }
