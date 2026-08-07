"""프로젝트 전역 설정: 경로, 라벨/결함 분류 체계, 상수.

## 경로가 함수인 이유

경로는 **활성 프로젝트에 따라 달라지므로 상수가 아니라 함수**다. 현장·라인마다 데이터와
라벨을 섞지 않으려면 작업공간이 나뉘어야 하고, 그러면 "manifest가 어디 있는가"의 답이
지금 어느 프로젝트를 보고 있느냐에 달린다.

    data/projects/<slug>/     manifest.csv · labels.csv · splits.csv · raw/ · interim/
    artifacts/projects/<slug>/ models/ · reports/ · cache/ · registry.csv · runs.csv

사전학습 CNN 모델(45MB)만 예외로 프로젝트 밖(`shared_model_dir()`)에 둔다. 프로젝트마다
같은 파일을 다시 내려받을 이유가 없다.

활성 프로젝트는 **프로세스 전역**이다. 이 앱에는 로그인 개념이 없고 이미 한 컨테이너의
CSV를 모두가 공유하므로(README 참고), 여기만 세션별로 나누는 것은 일관되지 않는다.

경로 컨테이너는 환경변수로 재정의할 수 있다.
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


# --- 경로 컨테이너 ---------------------------------------------------------
DATA_HOME = _env_path("VISION_AI_DATA_ROOT", PROJECT_ROOT / "data")
ARTIFACT_HOME = _env_path("VISION_AI_ARTIFACT_ROOT", PROJECT_ROOT / "artifacts")

PROJECTS_DIR = "projects"
DEFAULT_PROJECT = "default"

_active_project = DEFAULT_PROJECT


def active_project() -> str:
    """지금 보고 있는 프로젝트의 slug."""
    return _active_project


def use_project(slug: str) -> None:
    """활성 프로젝트를 바꾼다. 이후 모든 경로가 이 프로젝트를 가리킨다."""
    global _active_project
    _active_project = str(slug)


# --- 프로젝트별 경로 -------------------------------------------------------

def data_root() -> Path:
    return DATA_HOME / PROJECTS_DIR / _active_project


def raw_dir() -> Path:
    """수집 원본 이미지."""
    return data_root() / "raw"


def interim_dir() -> Path:
    """전처리 중간 산출물."""
    return data_root() / "interim"


def artifact_root() -> Path:
    return ARTIFACT_HOME / PROJECTS_DIR / _active_project


def model_dir() -> Path:
    return artifact_root() / "models"


def report_dir() -> Path:
    return artifact_root() / "reports"


def cache_dir() -> Path:
    """다시 계산하면 되는 것들 (지워도 무방)."""
    return artifact_root() / "cache"


def manifest_path() -> Path:
    """수집 이미지 인덱스 (1단계)."""
    return data_root() / "manifest.csv"


def labels_path() -> Path:
    """라벨링 결과 (2단계)."""
    return data_root() / "labels.csv"


def shared_model_dir() -> Path:
    """프로젝트 공통 모델 보관소.

    사전학습 CNN(45MB)처럼 **프로젝트와 무관하게 같은 파일**은 여기 둔다. 프로젝트를
    만들 때마다 다시 내려받게 하면 디스크와 시간을 그냥 버리는 셈이다.
    """
    return ARTIFACT_HOME / "shared" / "models"


def all_dirs() -> tuple[Path, ...]:
    return (
        data_root(), raw_dir(), interim_dir(),
        artifact_root(), model_dir(), report_dir(), cache_dir(),
        shared_model_dir(),
    )


def ensure_dirs() -> None:
    """필요한 디렉터리를 생성한다 (idempotent)."""
    for path in all_dirs():
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
