"""프로젝트 전역 설정: 경로, 라벨/결함 분류 체계, 상수.

경로는 환경변수로 재정의할 수 있다.
- VISION_AI_DATA_ROOT: 데이터 루트 (기본 <repo>/data)
- VISION_AI_ARTIFACT_ROOT: 모델/리포트 산출물 루트 (기본 <repo>/artifacts)
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(key: str, default: Path) -> Path:
    value = os.environ.get(key)
    return Path(value).expanduser().resolve() if value else default


# --- 경로 -----------------------------------------------------------------
DATA_ROOT = _env_path("VISION_AI_DATA_ROOT", PROJECT_ROOT / "data")
RAW_DIR = DATA_ROOT / "raw"          # 수집 원본 이미지
INTERIM_DIR = DATA_ROOT / "interim"  # 전처리 중간 산출물
ARTIFACT_ROOT = _env_path("VISION_AI_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")
MODEL_DIR = ARTIFACT_ROOT / "models"
REPORT_DIR = ARTIFACT_ROOT / "reports"

MANIFEST_PATH = DATA_ROOT / "manifest.csv"   # 수집 이미지 인덱스 (1단계)
LABELS_PATH = DATA_ROOT / "labels.csv"       # 라벨링 결과 (2단계)

ALL_DIRS = (DATA_ROOT, RAW_DIR, INTERIM_DIR, ARTIFACT_ROOT, MODEL_DIR, REPORT_DIR)


def ensure_dirs() -> None:
    """필요한 디렉터리를 생성한다 (idempotent)."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


# --- 이미지 ---------------------------------------------------------------
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

# --- 라벨 체계 -------------------------------------------------------------
LABEL_NORMAL = "normal"
LABEL_DEFECT = "defect"
LABEL_UNLABELED = "unlabeled"
LABELS = (LABEL_NORMAL, LABEL_DEFECT, LABEL_UNLABELED)

LABEL_KO = {
    LABEL_NORMAL: "정상",
    LABEL_DEFECT: "결함",
    LABEL_UNLABELED: "미라벨",
}

# 데이터 분할
SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_NONE = "train", "val", "test", "unassigned"
SPLITS = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST, SPLIT_NONE)

# --- 결함 유형 (일상 생활 물건 기준) --------------------------------------
# 산업용 세부 분류(용접 불량 등) 대신, 일상 물건에서 관찰 가능한 표면 이상으로 구성.
DEFECT_TYPES: dict[str, str] = {
    "scratch": "스크래치 / 긁힘",
    "dent": "찍힘 / 눌림",
    "crack": "균열 / 갈라짐",
    "stain": "얼룩 / 오염",
    "discoloration": "변색 / 색편차",
    "contamination": "이물 부착",
    "broken": "파손 / 깨짐",
    "hole": "구멍 / 찢김",
    "deformation": "변형 / 휘어짐",
    "other": "기타",
}

DEFECT_TYPE_NONE = "none"                # 정상 이미지의 결함 유형 값
DEFECT_TYPE_UNSPECIFIED = "unspecified"  # 결함이지만 유형이 아직 지정되지 않음

_SPECIAL_DEFECT_LABELS = {
    DEFECT_TYPE_NONE: "해당 없음",
    DEFECT_TYPE_UNSPECIFIED: "유형 미지정",
}


def defect_type_label(key: str) -> str:
    """결함 유형 키를 한글 표시명으로 변환한다."""
    if key in _SPECIAL_DEFECT_LABELS:
        return _SPECIAL_DEFECT_LABELS[key]
    return DEFECT_TYPES.get(key, key)


def is_standard_defect_type(key: str) -> bool:
    """프로젝트 표준 결함 유형인지 판단한다 (정규화 대상 판별용)."""
    return key in DEFECT_TYPES or key in _SPECIAL_DEFECT_LABELS
