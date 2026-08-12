"""라벨을 지도학습 **검출** 모델이 읽는 폴더로 내보낸다.

## 왜 이 단계가 따로 있는가

지금까지의 모델(`models.PatchAnomalyModel`)은 **이상탐지**다 — 정상만 학습하고 "정상과
얼마나 다른가"를 잰다. 결함이 어디에 있는지는 대략의 열지도로만 알 수 있고, "스크래치
2개와 찍힘 1개"처럼 **무엇이 몇 개인지는 말할 수 없다.**

거기까지 가려면 지도학습 검출 모델(YOLO 계열)이 필요하고, 그건 결함 위치가 박스로
붙은 데이터를 요구한다. H4에서 그 박스를 모을 수 있게 됐으니 이제 넘겨줄 수 있다.

## 왜 학습은 여기서 돌리지 않는가

검출 모델 학습은 GPU가 사실상 필수다. 이 앱은 브라우저에서 도는 PoC이고 배포본 컨테이너는
일회성이라, 학습을 여기서 시작하면 몇 시간 뒤에 결과를 잃는다. **폴더까지 만들어 주고
학습은 로컬호스트에서** 돌리는 것이 맞다 (`scripts/train_detector.py`).

## 폴더 구조

YOLO 계열이 공통으로 읽는 배치다. `data.yaml`이 클래스 이름과 분할 위치를 가리킨다.

    <루트>/
      data.yaml
      images/train/<image_id>.jpg      labels/train/<image_id>.txt
      images/val/...                   labels/val/...
      images/test/...                  labels/test/...

## 정상 이미지도 넣는다

박스가 있는 이미지만 넣으면 모델은 **"화면에는 항상 결함이 있다"**를 배운다. 정상으로
라벨된 이미지를 빈 txt와 함께 넣어 "여기에는 아무것도 없다"를 함께 가르친다. 검출에서
배경(negative) 표본은 선택이 아니라 요건이다.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import boxes as box_store
from . import config, labeling, storage

SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGES_DIR = "images"
LABELS_DIR = "labels"
DATA_FILE = "data.yaml"


@dataclass
class ExportResult:
    """무엇을 내보냈고 무엇을 왜 뺐는지.

    뺀 것을 조용히 넘기면 "박스를 300장에 그렸는데 왜 210장이지?"가 된다.
    """

    root: Path
    classes: list[str] = field(default_factory=list)
    per_split: dict[str, int] = field(default_factory=dict)
    positives: int = 0
    backgrounds: int = 0
    skipped_unlabeled: int = 0
    skipped_unassigned: int = 0
    skipped_missing: int = 0
    skipped_no_size: int = 0

    @property
    def total(self) -> int:
        return sum(self.per_split.values())

    @property
    def usable(self) -> bool:
        """이 폴더로 학습하면 뭐라도 배울 수 있는가.

        **결함이 한 장도 없으면 학습은 멀쩡히 돌고 "결함은 없다"만 배운다.** 지표까지
        훌륭하게 나온다(아무것도 못 찾아도 틀린 적이 없으니까). 폴더가 만들어졌다는
        사실만으로는 이것을 알 수 없으므로 여기서 밝힌다.
        """
        return self.positives > 0 and self.per_split.get("train", 0) > 0

    def blocking_note(self) -> str:
        """학습을 시작하면 안 되는 이유. 없으면 빈 문자열."""
        if self.total == 0:
            return "내보낼 이미지가 없습니다."
        if not self.positives:
            return (
                "**결함 이미지가 한 장도 없습니다.** 이대로 학습하면 모델은 "
                "'결함은 없다'만 배우고, 아무것도 못 찾으면서 지표는 훌륭하게 나옵니다. "
                "박스를 그린 이미지가 분할에 배정되어 있는지 확인하세요."
            )
        if not self.per_split.get("train"):
            return "train에 이미지가 없습니다. 데이터 분할을 다시 확인하세요."
        return ""

    def as_message(self) -> str:
        parts = [
            f"{self.total:,}장 내보냄 "
            f"(결함 {self.positives:,} · 정상 {self.backgrounds:,})",
            " / ".join(f"{name} {self.per_split.get(name, 0):,}" for name in SPLITS),
            f"클래스 {len(self.classes)}종",
        ]
        return " · ".join(parts)

    def skipped_notes(self) -> list[str]:
        notes = []
        if self.skipped_unlabeled:
            notes.append(f"라벨이 없어 {self.skipped_unlabeled:,}장 제외 — 2단계에서 판정하세요.")
        if self.skipped_unassigned:
            notes.append(
                f"분할이 배정되지 않아 {self.skipped_unassigned:,}장 제외 — "
                "2단계 **데이터 분할**을 먼저 실행하세요."
            )
        if self.skipped_missing:
            notes.append(f"파일을 찾을 수 없어 {self.skipped_missing:,}장 제외.")
        if self.skipped_no_size:
            notes.append(
                f"폭·높이를 몰라 {self.skipped_no_size:,}장 제외 — "
                "YOLO 좌표는 이미지 크기로 나눠야 해서 크기 없이는 만들 수 없습니다."
            )
        return notes


def default_root() -> Path:
    """내보낼 기본 위치. 프로젝트 안에 둬야 현장이 섞이지 않는다."""
    return config.artifact_root() / "detection"


def _link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    """심볼릭 링크를 우선 쓰되 안 되면 복사한다.

    4,500장을 복사하면 2GB가 또 생긴다. 다만 다른 기계로 폴더째 옮겨 학습할 생각이라면
    링크는 끊어지므로 그때는 복사해야 한다.
    """
    if target.exists() or target.is_symlink():
        target.unlink()
    if not copy:
        try:
            target.symlink_to(source.resolve())
            return
        except (OSError, NotImplementedError):
            pass                       # Windows 등 링크를 못 만드는 환경
    shutil.copy2(source, target)


def export_yolo(
    root: Path | None = None,
    *,
    resolved: pd.DataFrame | None = None,
    frame: pd.DataFrame | None = None,
    copy_images: bool = False,
    include_backgrounds: bool = True,
) -> ExportResult:
    """박스 라벨을 YOLO 학습 폴더로 내보낸다.

    Args:
        root: 내보낼 폴더. 기본은 `artifacts/<프로젝트>/detection`.
        resolved: 유효 라벨이 얹힌 manifest. 없으면 새로 만든다.
        frame: 박스 표. 없으면 새로 읽는다.
        copy_images: 원본을 복사할지. 기본은 심볼릭 링크(용량을 두 배로 쓰지 않는다).
        include_backgrounds: 정상 이미지를 빈 라벨로 함께 넣을지. **기본은 넣는다** —
            결함이 있는 이미지만 학습하면 모델은 "항상 결함이 있다"를 배운다.
    """
    root = Path(root) if root else default_root()
    resolved = labeling.resolve() if resolved is None else resolved
    frame = box_store.load() if frame is None else frame

    result = ExportResult(root=root, classes=box_store.class_names(frame))
    if resolved.empty:
        return result

    for split in SPLITS:
        (root / IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (root / LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
        result.per_split[split] = 0

    texts = box_store.to_yolo(frame, resolved)
    with_boxes = set(frame["image_id"].astype(str)) if not frame.empty else set()

    for _, row in resolved.iterrows():
        image_id = str(row["image_id"])
        label = str(row["label"])
        has_boxes = image_id in with_boxes

        if label == config.LABEL_UNLABELED:
            result.skipped_unlabeled += 1
            continue
        if label == config.LABEL_NORMAL and not include_backgrounds:
            continue
        if label == config.LABEL_DEFECT and not has_boxes:
            # 결함이라고만 하고 위치를 안 그린 이미지. 배경으로 넣으면 "여기엔 결함이
            # 없다"고 가르치는 셈이라 **거짓을 학습시킨다.** 그냥 뺀다.
            result.skipped_unlabeled += 1
            continue

        split = str(row.get("split") or "")
        if split not in SPLITS:
            result.skipped_unassigned += 1
            continue

        source = storage.resolve_path(row["path"])
        if not source.is_file():
            result.skipped_missing += 1
            continue
        if has_boxes and image_id not in texts:
            result.skipped_no_size += 1     # to_yolo가 크기를 몰라 건너뛴 것
            continue

        target = root / IMAGES_DIR / split / f"{image_id}{source.suffix.lower()}"
        _link_or_copy(source, target, copy=copy_images)
        (root / LABELS_DIR / split / f"{image_id}.txt").write_text(
            texts.get(image_id, ""), encoding="utf-8"
        )

        result.per_split[split] += 1
        if has_boxes:
            result.positives += 1
        else:
            result.backgrounds += 1

    (root / DATA_FILE).write_text(_data_yaml(root, result.classes), encoding="utf-8")
    return result


def _data_yaml(root: Path, classes: list[str]) -> str:
    """`data.yaml`. 클래스 번호는 `boxes.class_names()`의 순서와 **반드시** 같아야 한다.

    번호가 어긋나면 학습은 멀쩡히 돌고 결과만 틀린다 — 스크래치를 찍힘이라고 부른다.
    """
    lines = [
        "# vision_ai가 만든 검출 학습 설정",
        "# 클래스 번호는 boxes.csv의 클래스 이름을 정렬한 순서다.",
        f"path: {root.resolve()}",
        f"train: {IMAGES_DIR}/train",
        f"val: {IMAGES_DIR}/val",
        f"test: {IMAGES_DIR}/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    for index, name in enumerate(classes):
        lines.append(f"  {index}: {name}   # {config.defect_type_label(name)}")
    return "\n".join(lines) + "\n"


def readiness(resolved: pd.DataFrame | None = None, frame: pd.DataFrame | None = None) -> dict:
    """검출 학습을 시작할 만한 상태인지 본다.

    **박스가 몇 개 있으면 되는지**는 이 앱이 대신 정해 줄 수 없지만, 자릿수는 말할 수
    있다. 클래스당 수십 장으로는 검출 모델이 학습되지 않는다 — 시작해 놓고 몇 시간 뒤에
    아는 것보다 미리 아는 편이 싸다.
    """
    resolved = labeling.resolve() if resolved is None else resolved
    frame = box_store.load() if frame is None else frame

    per_class = (
        frame["label"].astype(str).value_counts().to_dict() if not frame.empty else {}
    )
    images = int(frame["image_id"].nunique()) if not frame.empty else 0
    assigned = (
        int(resolved["split"].isin(SPLITS).sum()) if "split" in resolved.columns else 0
    )
    return {
        "images_with_boxes": images,
        "boxes": len(frame),
        "per_class": per_class,
        "thin_classes": sorted(name for name, count in per_class.items() if count < 50),
        "splits_assigned": assigned,
        "ready": images >= 50 and assigned > 0,
    }
