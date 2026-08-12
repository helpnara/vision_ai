"""다중 박스 라벨 테스트.

박스는 **사람이 손으로 그린 결과물**이다. 잘못 덮어쓰면 들인 시간이 그대로 사라진다.
그래서 저장·불러오기보다 **무엇을 덮어쓰고 무엇을 지키는지**를 먼저 확인한다.

내보내기(COCO/YOLO)는 좌표 규약이 서로 다르다 — COCO는 좌상단+크기, YOLO는 이미지 크기로
나눈 중심+크기다. 규약을 틀리면 학습이 조용히 엉뚱한 곳을 배운다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_ai import boxes, config, storage


def _box(image_id="img1", x=10, y=20, w=30, h=40, label="scratch", **kwargs):
    return boxes.Box(image_id=image_id, x=x, y=y, w=w, h=h, label=label, **kwargs)


# --- 저장과 불러오기 --------------------------------------------------------

def test_empty_when_nothing_saved(sandbox):
    assert boxes.load().empty


def test_boxes_round_trip(sandbox):
    boxes.replace("img1", [_box(), _box(x=100, label="dent")])
    frame = boxes.for_image("img1")
    assert len(frame) == 2
    assert set(frame["label"]) == {"scratch", "dent"}


def test_one_image_can_hold_many_boxes(sandbox):
    """이게 이 모듈의 존재 이유다 — manifest의 roi 네 칸으로는 담을 수 없다."""
    boxes.replace("img1", [_box(x=index * 50) for index in range(5)])
    assert len(boxes.for_image("img1")) == 5


def test_saving_replaces_only_that_image(sandbox):
    boxes.replace("img1", [_box("img1")])
    boxes.replace("img2", [_box("img2"), _box("img2", x=99)])
    assert len(boxes.for_image("img1")) == 1
    assert len(boxes.for_image("img2")) == 2


def test_relabelling_an_image_drops_the_boxes_that_were_removed(sandbox):
    """한 장을 라벨링하는 행위는 '이 장의 박스는 이것들이다'이지 '하나 더한다'가 아니다."""
    boxes.replace("img1", [_box(), _box(x=100), _box(x=200)])
    boxes.replace("img1", [_box()])
    assert len(boxes.for_image("img1")) == 1


def test_zero_pixel_boxes_are_not_stored(sandbox):
    """끌지 않고 누르기만 해도 선택이 생긴다. 0픽셀 영역을 저장하면 안 된다."""
    boxes.replace("img1", [_box(w=0), _box(h=0), _box()])
    assert len(boxes.for_image("img1")) == 1


def test_a_corrupt_file_reads_as_empty(sandbox):
    boxes.boxes_path().write_text("이건 csv가 아니다\x00", encoding="utf-8")
    assert boxes.load().empty


# --- 예전 단일 ROI 이어받기 -------------------------------------------------

def _resolved(**overrides):
    row = {
        "image_id": "old1", "roi_x": 5, "roi_y": 6, "roi_w": 70, "roi_h": 80,
        "defect_type": "dent", "label_source": "human",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_old_single_roi_is_read_as_one_box(sandbox):
    found = boxes.from_manifest_roi(_resolved())
    assert len(found) == 1
    assert found[0].as_xyxy() == (5, 6, 75, 86)
    assert found[0].label == "dent"


def test_rows_without_a_roi_are_skipped(sandbox):
    assert boxes.from_manifest_roi(_resolved(roi_x=None)) == []


def test_adopting_old_rois_fills_the_box_table(sandbox):
    assert boxes.adopt_manifest_rois(_resolved()) == 1
    assert len(boxes.for_image("old1")) == 1


def test_adopting_never_overwrites_hand_drawn_boxes(sandbox):
    """사람이 새로 그린 박스를 옛 ROI로 덮어쓰면 작업을 잃는다."""
    boxes.replace("old1", [_box("old1", x=999, label="crack")])
    assert boxes.adopt_manifest_rois(_resolved()) == 0

    kept = boxes.for_image("old1")
    assert len(kept) == 1 and int(kept.iloc[0]["x"]) == 999


def test_adopting_twice_adds_nothing_the_second_time(sandbox):
    boxes.adopt_manifest_rois(_resolved())
    assert boxes.adopt_manifest_rois(_resolved()) == 0


# --- 현황 ------------------------------------------------------------------

def test_summary_counts_boxes_and_images(sandbox):
    boxes.replace("img1", [_box(), _box(x=60, label="dent")])
    boxes.replace("img2", [_box("img2")])
    info = boxes.summary()
    assert info == {"boxes": 3, "images": 2, "labels": 2, "per_image": pytest.approx(1.5)}


# --- 내보내기 ---------------------------------------------------------------

@pytest.fixture
def labelled(sandbox):
    boxes.replace("img1", [_box(label="scratch"), _box(x=100, label="dent")])
    boxes.replace("img2", [_box("img2", x=1, y=2, w=10, h=20, label="scratch")])
    manifest = pd.DataFrame(
        [
            {"image_id": "img1", "width": 200, "height": 100, "path": "a/img1.png"},
            {"image_id": "img2", "width": 200, "height": 100, "path": "a/img2.png"},
        ]
    )
    return boxes.load(), manifest


def test_coco_keeps_the_top_left_and_size(labelled):
    frame, manifest = labelled
    coco = boxes.to_coco(frame, manifest)
    first = next(a for a in coco["annotations"] if a["bbox"][0] == 10)
    assert first["bbox"] == [10, 20, 30, 40]
    assert first["area"] == 1200


def test_coco_numbers_categories_from_one(labelled):
    """COCO의 category_id는 1부터다. 0부터 매기면 학습 쪽에서 클래스가 하나 밀린다."""
    frame, manifest = labelled
    coco = boxes.to_coco(frame, manifest)
    assert sorted(c["id"] for c in coco["categories"]) == [1, 2]
    assert {c["name"] for c in coco["categories"]} == {"scratch", "dent"}


def test_yolo_uses_normalised_centres(labelled):
    """YOLO는 이미지 크기로 나눈 중심 좌표를 쓴다. 규약을 틀리면 엉뚱한 곳을 배운다."""
    frame, manifest = labelled
    texts = boxes.to_yolo(frame, manifest)
    line = next(l for l in texts["img1"].splitlines() if l.split()[1].startswith("0.125"))
    _, cx, cy, w, h = line.split()
    assert float(cx) == pytest.approx((10 + 30 / 2) / 200)
    assert float(cy) == pytest.approx((20 + 40 / 2) / 100)
    assert float(w) == pytest.approx(30 / 200)
    assert float(h) == pytest.approx(40 / 100)


def test_yolo_skips_images_with_unknown_size(sandbox):
    """크기를 모르면 정규화할 수 없다. 틀린 좌표를 내보내느니 건너뛴다."""
    boxes.replace("img1", [_box()])
    texts = boxes.to_yolo(boxes.load(), pd.DataFrame([{"image_id": "img1"}]))
    assert texts == {}


def test_yolo_zip_holds_one_file_per_image(labelled):
    """YOLO는 이미지 한 장당 txt 한 개를 요구한다. 한 파일로 이어 붙이면 받는 쪽이
    직접 쪼개야 하고, 쪼개는 규칙은 어디에도 적혀 있지 않다."""
    import io
    import zipfile

    frame, manifest = labelled
    with zipfile.ZipFile(io.BytesIO(boxes.to_yolo_zip(frame, manifest))) as archive:
        names = set(archive.namelist())
        assert names == {"classes.txt", "labels/img1.txt", "labels/img2.txt"}
        assert len(archive.read("labels/img1.txt").decode().strip().splitlines()) == 2


def test_yolo_zip_carries_the_class_names(labelled):
    """txt 안에는 번호뿐이다. 번호와 이름을 잇는 표가 없으면 나중에 뜻을 알 수 없다."""
    import io
    import zipfile

    frame, manifest = labelled
    with zipfile.ZipFile(io.BytesIO(boxes.to_yolo_zip(frame, manifest))) as archive:
        listed = archive.read("classes.txt").decode().strip().splitlines()
    assert listed == boxes.class_names(frame)


def test_yolo_zip_is_still_a_valid_zip_when_nothing_is_labelled(sandbox):
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(boxes.to_yolo_zip(boxes.empty(), pd.DataFrame()))) as archive:
        assert archive.namelist() == ["classes.txt"]


def test_class_numbers_are_stable_across_exports(labelled):
    """번호가 실행마다 달라지면 이미 학습한 모델과 어긋난다."""
    frame, manifest = labelled
    assert boxes.class_names(frame) == boxes.class_names(frame)
    assert boxes.class_names(frame) == sorted(boxes.class_names(frame))


def test_exports_are_empty_when_nothing_is_labelled(sandbox):
    empty = boxes.empty()
    assert boxes.to_coco(empty, pd.DataFrame())["annotations"] == []
    assert boxes.to_yolo(empty, pd.DataFrame()) == {}


def test_boxes_live_inside_the_project(sandbox):
    assert boxes.boxes_path().parent == config.data_root()


# --- 마스크에서 일괄 생성 (검출 학습용) --------------------------------------

def _mask_dataset(sandbox, blobs, *, label="scratch"):
    """VisA 배치(Images/Masks)로 이미지 한 장과 마스크를 만든다."""
    import cv2
    import numpy as np

    image = sandbox / "raw" / "pcb1" / "Data" / "Images" / "Anomaly" / "000.JPG"
    mask = sandbox / "raw" / "pcb1" / "Data" / "Masks" / "Anomaly" / "000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image), np.full((100, 200, 3), 60, np.uint8))

    canvas = np.zeros((100, 200), np.uint8)
    for x, y, w, h in blobs:
        canvas[y:y + h, x:x + w] = 255
    cv2.imwrite(str(mask), canvas)

    resolved = pd.DataFrame([{
        "image_id": "img1",
        "path": storage.to_manifest_path(image),
        "label": config.LABEL_DEFECT,
        "defect_type": label,
    }])
    return resolved


def test_separate_defects_become_separate_boxes(sandbox):
    """실측으로 VisA 결함의 58%가 떨어진 덩어리 2개 이상이다. 한 박스로 묶으면
    면적이 평균 2.5배가 되고 그 절반이 배경이라 모델이 배경을 결함이라고 배운다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20), (150, 70, 20, 20)])
    counts = boxes.from_masks(resolved)

    assert counts["images"] == 1 and counts["boxes"] == 2
    made = boxes.for_image("img1")
    assert set(zip(made["x"], made["y"])) == {(10, 10), (150, 70)}


def test_boxes_are_tight_around_each_blob(sandbox):
    resolved = _mask_dataset(sandbox, [(10, 20, 30, 40)])
    boxes.from_masks(resolved)
    row = boxes.for_image("img1").iloc[0]
    assert (row["x"], row["y"], row["w"], row["h"]) == (10, 20, 30, 40)


def test_specks_are_dropped(sandbox):
    """VisA 마스크 덩어리 하위 5%는 1~2픽셀 — 결함이 아니라 가장자리 계단 자국이다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 30, 30), (150, 80, 1, 1)])
    assert boxes.from_masks(resolved)["boxes"] == 1


def test_the_speck_floor_can_be_lowered(sandbox):
    resolved = _mask_dataset(sandbox, [(10, 10, 30, 30), (150, 80, 1, 1)])
    assert boxes.from_masks(resolved, min_area=1)["boxes"] == 2


def test_human_boxes_are_never_overwritten(sandbox):
    """자동으로 만든 것이 손으로 고친 것을 덮으면 그 작업을 되돌릴 방법이 없다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20), (150, 70, 20, 20)])
    boxes.replace("img1", [_box("img1", x=5, y=5, w=9, h=9, label="dent")])

    counts = boxes.from_masks(resolved)
    assert counts["images"] == 0 and counts["skipped_existing"] == 1
    kept = boxes.for_image("img1")
    assert len(kept) == 1 and kept.iloc[0]["label"] == "dent"


def test_overwrite_replaces_the_whole_image(sandbox):
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20), (150, 70, 20, 20)])
    boxes.replace("img1", [_box("img1", x=5, y=5, w=9, h=9, label="dent")])

    boxes.from_masks(resolved, overwrite=True)
    made = boxes.for_image("img1")
    assert len(made) == 2 and set(made["label"]) == {"scratch"}


def test_generated_boxes_are_marked_as_coming_from_a_mask(sandbox):
    """사람이 그린 것과 구분이 안 되면 어디까지 검수했는지 알 수 없다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20)])
    boxes.from_masks(resolved)
    assert boxes.for_image("img1").iloc[0]["source"] == boxes.SOURCE_MASK


def test_the_image_defect_type_becomes_the_box_class(sandbox):
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20)], label="dent")
    boxes.from_masks(resolved)
    assert boxes.for_image("img1").iloc[0]["label"] == "dent"


def test_images_without_a_mask_are_counted(sandbox):
    """VisA 결함 461장 중 13장은 마스크가 없었다. 조용히 빠지면 왜 적은지 알 수 없다."""
    resolved = pd.DataFrame([{
        "image_id": "img1", "path": "raw/plain/007.jpg",
        "label": config.LABEL_DEFECT, "defect_type": "scratch",
    }])
    counts = boxes.from_masks(resolved)
    assert counts["skipped_no_mask"] == 1 and counts["boxes"] == 0


def test_an_empty_mask_is_counted_separately(sandbox):
    """마스크가 없는 것과 있는데 다 잡티였던 것은 원인이 다르다."""
    resolved = _mask_dataset(sandbox, [])
    counts = boxes.from_masks(resolved)
    assert counts["skipped_empty"] == 1 and counts["skipped_no_mask"] == 0


def test_normal_images_are_left_alone(sandbox):
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20)])
    resolved["label"] = config.LABEL_NORMAL
    assert boxes.from_masks(resolved)["boxes"] == 0


def test_candidates_are_counted_before_running(sandbox):
    """수백 장을 훑기 전에 얼마나 채워지는지 알려줘야 누를지 정할 수 있다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20)])
    assert boxes.mask_candidates(resolved) == {"ready": 1, "without_mask": 0, "already": 0}

    boxes.from_masks(resolved)
    assert boxes.mask_candidates(resolved)["already"] == 1


def test_box_ids_stay_unique_within_an_image(sandbox):
    """같은 id가 둘이면 내보낼 때 하나가 조용히 사라진다."""
    resolved = _mask_dataset(sandbox, [(10, 10, 20, 20), (60, 10, 20, 20), (150, 70, 20, 20)])
    boxes.from_masks(resolved)
    made = boxes.for_image("img1")
    assert len(set(made["box_id"])) == len(made) == 3
