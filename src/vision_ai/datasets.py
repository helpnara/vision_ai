"""오픈 표면 결함 데이터셋 카탈로그와 폴더 구조 파서.

사내 데이터를 쓰지 않는 것이 이 프로젝트의 전제이므로, 공개 데이터셋이 유일한
학습 데이터 공급원이다. 여기서는 각 데이터셋의 성격·라이선스·폴더 구조를 정리해
1단계(수집) 화면에서 선택·임포트할 수 있게 한다.

주의: 라이선스와 URL은 문서화 시점 기준 정보다. 실제 사용 전에 반드시 원본
배포 페이지에서 조건을 직접 확인해야 한다 (`license_note` 참고).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config

# MVTec AD 계열에서 "정상"을 뜻하는 폴더명
_GOOD_DIR_NAMES = frozenset({"good", "ok", "normal", "non_defective"})


@dataclass(frozen=True)
class OpenDataset:
    """공개 데이터셋 메타데이터."""

    key: str
    name: str
    summary: str
    url: str
    license: str
    commercial_use: str          # "가능" / "불가(비상업)" / "확인 필요"
    layout: str                  # "mvtec" | "flat" | "custom"
    layout_note: str
    categories: tuple[str, ...]
    everyday_fit: int            # 일상 물건 적합도 1~5 (5가 가장 적합)
    everyday_note: str
    download_note: str
    license_note: str = "사용 전 원본 배포 페이지에서 라이선스 조건을 직접 확인할 것."
    tags: tuple[str, ...] = field(default_factory=tuple)


CATALOG: tuple[OpenDataset, ...] = (
    OpenDataset(
        key="mvtec_ad",
        name="MVTec AD",
        summary=(
            "이상탐지 분야 표준 벤치마크. 15개 카테고리에 대해 정상 이미지만으로 학습하고 "
            "테스트에서 결함을 찾는 구조이며, 결함 픽셀 마스크(ground truth)까지 제공한다."
        ),
        url="https://www.mvtec.com/company/research/datasets/mvtec-ad",
        license="CC BY-NC-SA 4.0",
        commercial_use="불가(비상업)",
        layout="mvtec",
        layout_note=(
            "<category>/train/good/*.png, <category>/test/<defect_type>/*.png, "
            "<category>/ground_truth/<defect_type>/*_mask.png"
        ),
        categories=(
            "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
            "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
        ),
        everyday_fit=5,
        everyday_note=(
            "bottle(병), toothbrush(칫솔), wood(목재), leather(가죽), carpet(카펫), "
            "zipper(지퍼), tile(타일), hazelnut(견과) 등 일상에서 볼 수 있는 품목이 많다."
        ),
        download_note=(
            "배포 페이지에서 약관 동의 후 다운로드(약 5GB, 카테고리별 개별 다운로드도 가능). "
            "다운로드·압축 해제 후 로컬 폴더 임포트를 사용한다."
        ),
        tags=("이상탐지", "픽셀 마스크", "벤치마크"),
    ),
    OpenDataset(
        key="visa",
        name="VisA (SPot-the-difference)",
        summary=(
            "Amazon이 공개한 12개 물체 대상 이상탐지 데이터셋. MVTec AD보다 규모가 크고 "
            "한 이미지에 여러 개체가 함께 놓인 복잡한 장면을 포함한다."
        ),
        url="https://github.com/amazon-science/spot-diff",
        license="CC BY 4.0",
        commercial_use="가능(출처 표기 조건)",
        layout="custom",
        layout_note=(
            "원본은 <object>/Data/Images/{Normal,Anomaly}/ + CSV 분할 정의. "
            "저장소의 변환 스크립트로 MVTec 형식으로 바꿀 수 있다."
        ),
        categories=(
            "candle", "capsules", "cashew", "chewinggum", "fryum",
            "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
        ),
        everyday_fit=4,
        everyday_note="candle(양초), chewinggum(껌), cashew(캐슈너트), macaroni(마카로니) 등 생활 품목 중심.",
        download_note="GitHub 저장소 안내에 따라 다운로드 후, MVTec 형식으로 변환하면 임포트가 쉽다.",
        tags=("이상탐지", "다중 개체", "CC BY"),
    ),
    OpenDataset(
        key="mpdd",
        name="MPDD",
        summary="금속 가공 부품 표면 결함 데이터셋. 촬영 각도·조명 변화가 크게 포함되어 있다.",
        url="https://github.com/stepanje/MPDD",
        license="확인 필요",
        commercial_use="확인 필요",
        layout="mvtec",
        layout_note="MVTec AD와 동일한 폴더 구조.",
        categories=("bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"),
        everyday_fit=2,
        everyday_note="산업 금속 부품 중심으로, 일상 물건과는 거리가 있다.",
        download_note="GitHub 저장소 링크에서 다운로드 후 로컬 폴더 임포트.",
        tags=("이상탐지", "금속"),
    ),
    OpenDataset(
        key="btad",
        name="BTAD (beanTech AD)",
        summary="3종 산업 제품의 표면·형상 이상 데이터셋. 소규모라 빠른 실험에 적합하다.",
        url="http://avires.dimi.uniud.it/papers/btad/btad.zip",
        license="확인 필요",
        commercial_use="확인 필요",
        layout="mvtec",
        layout_note="MVTec AD와 유사한 구조(product/train/ok, product/test/ko 등).",
        categories=("01", "02", "03"),
        everyday_fit=2,
        everyday_note="산업 제품 중심. 파이프라인 검증용 소규모 데이터로는 유용하다.",
        download_note="URL에서 zip 직접 다운로드.",
        tags=("이상탐지", "소규모"),
    ),
    OpenDataset(
        key="magnetic_tile",
        name="Magnetic Tile Defect",
        summary="마그네틱 타일 표면 결함(균열·기포·마모 등) 6종 분류 데이터셋.",
        url="https://github.com/abin24/Magnetic-tile-defect-datasets.",
        license="확인 필요",
        commercial_use="확인 필요",
        layout="flat",
        layout_note="결함 유형별 폴더(MT_Blowhole, MT_Crack, MT_Free, ...) 안에 이미지와 마스크가 함께 있다.",
        categories=("MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free"),
        everyday_fit=2,
        everyday_note="산업 부품이지만 균열·마모 등 결함 유형 자체는 일상 물건과 통한다.",
        download_note="GitHub 저장소를 clone 또는 zip 다운로드.",
        tags=("분류", "균열"),
    ),
    OpenDataset(
        key="dagm2007",
        name="DAGM 2007",
        summary="약지도 학습용 인공 생성 텍스처 표면 결함 데이터셋. 10개 클래스로 구성된다.",
        url="https://hci.iwr.uni-heidelberg.de/content/weakly-supervised-learning-industrial-optical-inspection",
        license="연구 목적 사용 (확인 필요)",
        commercial_use="불가로 가정(확인 필요)",
        layout="flat",
        layout_note="클래스별 폴더에 이미지와 약한 라벨(타원 영역) 정보가 함께 제공된다.",
        categories=tuple(f"Class{i}" for i in range(1, 11)),
        everyday_fit=1,
        everyday_note="인공 텍스처로, 일상 물건 사진과는 성격이 다르다. 알고리즘 검증용.",
        download_note="배포 페이지 조건 확인 후 다운로드.",
        tags=("텍스처", "약지도"),
    ),
    OpenDataset(
        key="neu_det",
        name="NEU-DET",
        summary="열연 강판 표면 결함 6종 검출 데이터셋. 바운딩박스 어노테이션 제공.",
        url="https://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/",
        license="확인 필요",
        commercial_use="확인 필요",
        layout="custom",
        layout_note="IMAGES/ + ANNOTATIONS/(Pascal VOC XML) 구조.",
        categories=("crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"),
        everyday_fit=1,
        everyday_note="철강 표면 전용. 객체 검출(바운딩박스) 학습 연습용으로 참고.",
        download_note="배포 페이지에서 다운로드.",
        tags=("객체 검출", "철강"),
    ),
)


def get(key: str) -> OpenDataset | None:
    """key로 데이터셋 메타데이터를 조회한다."""
    for dataset in CATALOG:
        if dataset.key == key:
            return dataset
    return None


def keys() -> tuple[str, ...]:
    """카탈로그의 모든 key를 반환한다."""
    return tuple(dataset.key for dataset in CATALOG)


def recommended() -> tuple[OpenDataset, ...]:
    """일상 물건 적합도가 높은 순으로 데이터셋을 반환한다."""
    return tuple(sorted(CATALOG, key=lambda d: -d.everyday_fit))


# --- 폴더 구조 파서 --------------------------------------------------------

def parse_mvtec_path(relative_path: Path) -> dict:
    """MVTec AD 형식 상대경로에서 category/split/label/defect_type을 추론한다.

    예) bottle/test/broken_large/000.png
        -> category=bottle, split=test, label=defect, defect_type=broken_large

    구조를 인식할 수 없으면 label=unlabeled로 두고 최선의 추정값을 채운다.
    """
    parts = [p for p in relative_path.parts if p not in (".", "")]
    result = {
        "category": "unknown",
        "split": config.SPLIT_NONE,
        "label": config.LABEL_UNLABELED,
        "defect_type": config.DEFECT_TYPE_NONE,
        "is_mask": False,
    }
    if len(parts) < 2:
        return result

    # 파일명 제외
    dirs = parts[:-1]
    lowered = [d.lower() for d in dirs]

    # split 위치 탐색 (train / test / validation)
    split_idx = None
    for idx, name in enumerate(lowered):
        if name in ("train", "test", "val", "validation"):
            split_idx = idx
            break

    if split_idx is None:
        # ground_truth 마스크 폴더만 있는 경우 등
        if "ground_truth" in lowered:
            result["is_mask"] = True
        result["category"] = dirs[0] if dirs else "unknown"
        return result

    if split_idx > 0:
        result["category"] = dirs[split_idx - 1]
    split_name = lowered[split_idx]
    result["split"] = config.SPLIT_VAL if split_name in ("val", "validation") else split_name

    # split 아래 폴더가 정상/결함 유형을 나타낸다
    remainder = dirs[split_idx + 1:]
    if not remainder:
        return result

    leaf = remainder[-1]
    if leaf.lower() in _GOOD_DIR_NAMES:
        result["label"] = config.LABEL_NORMAL
        result["defect_type"] = config.DEFECT_TYPE_NONE
    else:
        result["label"] = config.LABEL_DEFECT
        result["defect_type"] = leaf

    if "ground_truth" in lowered:
        result["is_mask"] = True
    return result


def parse_flat_path(relative_path: Path) -> dict:
    """클래스 폴더 하나만 있는 단순 구조를 해석한다.

    예) MT_Crack/exp1.jpg -> category=MT_Crack, label=defect, defect_type=MT_Crack
        good/001.png      -> label=normal
    """
    parts = [p for p in relative_path.parts if p not in (".", "")]
    if len(parts) < 2:
        return {
            "category": "unknown",
            "split": config.SPLIT_NONE,
            "label": config.LABEL_UNLABELED,
            "defect_type": config.DEFECT_TYPE_NONE,
            "is_mask": False,
        }
    folder = parts[-2]
    is_good = folder.lower() in _GOOD_DIR_NAMES or folder.lower().endswith("_free")
    return {
        "category": folder,
        "split": config.SPLIT_NONE,
        "label": config.LABEL_NORMAL if is_good else config.LABEL_DEFECT,
        "defect_type": config.DEFECT_TYPE_NONE if is_good else folder,
        "is_mask": False,
    }


LAYOUT_PARSERS = {
    "mvtec": parse_mvtec_path,
    "flat": parse_flat_path,
}


def parse_path(relative_path: Path, layout: str) -> dict:
    """layout에 맞는 파서로 경로를 해석한다. 알 수 없으면 미라벨로 둔다."""
    parser = LAYOUT_PARSERS.get(layout)
    if parser is None:
        return {
            "category": "unknown",
            "split": config.SPLIT_NONE,
            "label": config.LABEL_UNLABELED,
            "defect_type": config.DEFECT_TYPE_NONE,
            "is_mask": False,
        }
    return parser(relative_path)
