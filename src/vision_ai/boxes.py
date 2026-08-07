"""이미지 한 장의 결함 박스 여러 개.

## 왜 별도 파일인가

manifest에는 박스가 `roi_x/roi_y/roi_w/roi_h` 네 칸, 즉 **한 개만** 들어간다. CCTV 프레임
한 장에 결함이 여럿이면 그 구조로는 담을 수 없다. 그렇다고 manifest에 열을 계속 늘릴 수도
없어서(박스 5개면 20칸) 박스는 **한 줄이 박스 하나**인 별도 표로 뺀다.

## 왜 COCO/YOLO가 아니라 CSV인가

둘 다 학습 프레임워크가 바로 읽는 표준이지만, **고칠 때가 문제다.** COCO는 박스 하나를
고쳐도 JSON 전체를 다시 써야 하고, YOLO는 이미지마다 txt가 생겨 파일이 수천 개가 되며
클래스가 번호로만 남아 사람이 읽을 수 없다. 이 앱의 나머지(manifest·라벨 이력·실험 기록)가
전부 CSV라 도구도 일관된다.

학습은 **내보낼 때** 변환한다(`to_coco()` / `to_yolo()`). 변환은 언제든 다시 할 수 있지만
편집 이력은 한 번 잃으면 끝이다.

## 기존 단일 ROI와의 관계

`roi_x/y/w/h`로 저장된 예전 라벨은 **박스 1개짜리**로 본다(`from_manifest_roi()`).
새 구조로 넘어가면서 이미 해 둔 라벨링을 잃지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import config

BOXES_FILE = "boxes.csv"

COLUMNS: tuple[str, ...] = (
    "box_id",      # image_id + 순번 — 같은 이미지 안에서만 고유하면 된다
    "image_id",
    "x", "y", "w", "h",   # 원본 픽셀 기준 좌상단과 크기
    "label",       # 결함 유형 키 (config.DEFECT_TYPES) 또는 프로젝트가 정한 클래스
    "source",      # 누가 그렸는가: "사람" / "마스크" / "가져오기"
    "created_at",
    "note",
)

SOURCE_HUMAN = "사람"
SOURCE_MASK = "마스크"
SOURCE_IMPORT = "가져오기"


@dataclass(frozen=True)
class Box:
    image_id: str
    x: int
    y: int
    w: int
    h: int
    label: str
    source: str = SOURCE_HUMAN
    note: str = ""

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.x + self.w, self.y + self.h


def boxes_path() -> Path:
    return config.data_root() / BOXES_FILE


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNS))


def load() -> pd.DataFrame:
    path = boxes_path()
    if not path.exists():
        return empty()
    try:
        frame = pd.read_csv(path, dtype={"box_id": "str", "image_id": "str"})
    except (OSError, ValueError):
        return empty()
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = "" if column not in ("x", "y", "w", "h") else 0
    return frame[list(COLUMNS)]


def save(frame: pd.DataFrame) -> None:
    path = boxes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(COLUMNS)].to_csv(path, index=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def for_image(image_id: str, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = load() if frame is None else frame
    if frame.empty:
        return frame
    return frame[frame["image_id"].astype(str) == str(image_id)].reset_index(drop=True)


def replace(image_id: str, drawn: list[Box]) -> pd.DataFrame:
    """이 이미지의 박스를 통째로 갈아 끼운다.

    **한 장을 라벨링하는 행위는 "이 장의 박스는 이것들이다"** 이지 "박스를 하나 더한다"가
    아니다. 더하기만 되면 지운 박스가 남아 되돌릴 방법이 없다.
    """
    frame = load()
    if not frame.empty:
        frame = frame[frame["image_id"].astype(str) != str(image_id)]

    rows = []
    stamp = _now()
    for order, box in enumerate(drawn, start=1):
        if box.w < 1 or box.h < 1:
            continue   # 끌지 않고 누르기만 해서 생긴 0픽셀 영역
        rows.append(
            {
                "box_id": f"{image_id}#{order}",
                "image_id": str(image_id),
                "x": int(box.x), "y": int(box.y), "w": int(box.w), "h": int(box.h),
                "label": str(box.label),
                "source": str(box.source),
                "created_at": stamp,
                "note": str(box.note),
            }
        )

    updated = pd.concat([frame, pd.DataFrame(rows, columns=list(COLUMNS))], ignore_index=True)
    save(updated)
    return updated


def to_boxes(frame: pd.DataFrame) -> list[Box]:
    return [
        Box(
            image_id=str(row["image_id"]),
            x=int(row["x"]), y=int(row["y"]), w=int(row["w"]), h=int(row["h"]),
            label=str(row["label"]),
            source=str(row.get("source") or SOURCE_HUMAN),
            note=str(row.get("note") or ""),
        )
        for _, row in frame.iterrows()
    ]


def from_manifest_roi(resolved: pd.DataFrame) -> list[Box]:
    """예전 단일 ROI 라벨을 박스 1개짜리로 읽어온다. 해 둔 라벨링을 잃지 않기 위해서다."""
    needed = {"image_id", "roi_x", "roi_y", "roi_w", "roi_h"}
    if resolved.empty or not needed <= set(resolved.columns):
        return []

    found: list[Box] = []
    for _, row in resolved.iterrows():
        values = [row.get(name) for name in ("roi_x", "roi_y", "roi_w", "roi_h")]
        if any(pd.isna(value) for value in values):
            continue
        x, y, w, h = (int(value) for value in values)
        if w < 1 or h < 1:
            continue
        found.append(
            Box(
                image_id=str(row["image_id"]), x=x, y=y, w=w, h=h,
                label=str(row.get("defect_type") or config.DEFECT_TYPE_UNSPECIFIED),
                source=SOURCE_MASK if row.get("label_source") == "folder" else SOURCE_HUMAN,
            )
        )
    return found


def adopt_manifest_rois(resolved: pd.DataFrame) -> int:
    """예전 ROI를 박스 표로 옮긴다. **이미 박스가 있는 이미지는 건드리지 않는다.**

    사람이 새로 그린 박스를 옛 ROI로 덮어쓰면 작업을 잃는다.
    """
    candidates = from_manifest_roi(resolved)
    if not candidates:
        return 0

    frame = load()
    taken = set(frame["image_id"].astype(str)) if not frame.empty else set()
    fresh = [box for box in candidates if box.image_id not in taken]
    if not fresh:
        return 0

    stamp = _now()
    rows = [
        {
            "box_id": f"{box.image_id}#1",
            "image_id": box.image_id,
            "x": box.x, "y": box.y, "w": box.w, "h": box.h,
            "label": box.label, "source": box.source,
            "created_at": stamp, "note": "단일 ROI에서 옮김",
        }
        for box in fresh
    ]
    save(pd.concat([frame, pd.DataFrame(rows, columns=list(COLUMNS))], ignore_index=True))
    return len(rows)


def summary(frame: pd.DataFrame | None = None) -> dict:
    frame = load() if frame is None else frame
    if frame.empty:
        return {"boxes": 0, "images": 0, "labels": 0, "per_image": 0.0}
    images = frame["image_id"].nunique()
    return {
        "boxes": len(frame),
        "images": int(images),
        "labels": int(frame["label"].nunique()),
        "per_image": len(frame) / max(int(images), 1),
    }


# --- 내보내기 ---------------------------------------------------------------

def class_names(frame: pd.DataFrame) -> list[str]:
    """등장하는 클래스를 정렬해 돌려준다. 번호는 이 순서로 매겨진다."""
    if frame.empty:
        return []
    return sorted(frame["label"].astype(str).unique())


def to_coco(frame: pd.DataFrame, images: pd.DataFrame) -> dict:
    """COCO 형식으로 바꾼다.

    `images`는 manifest다 — 폭·높이가 필요하다. COCO의 bbox는 [x, y, w, h]로 이 표와 같다.
    """
    names = class_names(frame)
    category_id = {name: index + 1 for index, name in enumerate(names)}

    sizes = {}
    if not images.empty:
        for _, row in images.iterrows():
            sizes[str(row["image_id"])] = (row.get("width"), row.get("height"), row.get("path"))

    used = sorted(frame["image_id"].astype(str).unique()) if not frame.empty else []
    image_id = {name: index + 1 for index, name in enumerate(used)}

    coco_images = []
    for name in used:
        width, height, path = sizes.get(name, (0, 0, name))
        coco_images.append(
            {
                "id": image_id[name],
                "file_name": str(path),
                "width": int(width) if pd.notna(width) else 0,
                "height": int(height) if pd.notna(height) else 0,
            }
        )

    annotations = []
    for order, (_, row) in enumerate(frame.iterrows(), start=1):
        annotations.append(
            {
                "id": order,
                "image_id": image_id[str(row["image_id"])],
                "category_id": category_id[str(row["label"])],
                "bbox": [int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])],
                "area": int(row["w"]) * int(row["h"]),
                "iscrowd": 0,
            }
        )

    return {
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": category_id[name], "name": name} for name in names],
    }


def to_yolo(frame: pd.DataFrame, images: pd.DataFrame) -> dict[str, str]:
    """YOLO 형식으로 바꾼다 — `{이미지id: txt 내용}`.

    YOLO는 **이미지 크기로 나눈 중심 좌표와 크기**를 쓴다. 폭·높이를 모르면 변환할 수
    없으므로 그런 이미지는 건너뛴다(잘못된 좌표를 내보내는 것보다 낫다).
    """
    names = class_names(frame)
    index_of = {name: index for index, name in enumerate(names)}

    sizes = {}
    if not images.empty:
        for _, row in images.iterrows():
            sizes[str(row["image_id"])] = (row.get("width"), row.get("height"))

    output: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        name = str(row["image_id"])
        width, height = sizes.get(name, (None, None))
        if not width or not height or pd.isna(width) or pd.isna(height):
            continue
        width, height = float(width), float(height)
        cx = (float(row["x"]) + float(row["w"]) / 2) / width
        cy = (float(row["y"]) + float(row["h"]) / 2) / height
        output.setdefault(name, []).append(
            f"{index_of[str(row['label'])]} "
            f"{cx:.6f} {cy:.6f} {float(row['w']) / width:.6f} {float(row['h']) / height:.6f}"
        )
    return {name: "\n".join(lines) + "\n" for name, lines in output.items()}
