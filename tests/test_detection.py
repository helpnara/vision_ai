"""검출 학습 폴더 내보내기 테스트 (H5).

이 단계의 실패는 **조용하다.** 폴더는 만들어지고 학습도 돌아가는데 결과만 틀린다 —
클래스 번호가 밀렸거나, 결함만 학습해서 "화면에는 항상 결함이 있다"를 배웠거나,
위치를 안 그린 결함 이미지를 배경으로 넣어 거짓을 가르쳤거나. 그래서 여기서 잡는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_ai import boxes as box_store
from vision_ai import config, detection, storage


@pytest.fixture
def dataset(sandbox):
    """정상 2장 · 박스 있는 결함 2장 · 박스 없는 결함 1장 · 라벨 없음 1장."""
    import cv2

    rows = []
    for image_id, label, split in [
        ("n1", config.LABEL_NORMAL, "train"),
        ("n2", config.LABEL_NORMAL, "val"),
        ("d1", config.LABEL_DEFECT, "train"),
        ("d2", config.LABEL_DEFECT, "test"),
        ("d3", config.LABEL_DEFECT, "train"),      # 결함이지만 박스를 안 그렸다
        ("u1", config.LABEL_UNLABELED, "train"),
    ]:
        path = config.raw_dir() / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), np.full((100, 200, 3), 128, np.uint8))
        rows.append(
            {
                "image_id": image_id,
                "path": storage.to_manifest_path(path),
                "label": label,
                "split": split,
                "width": 200,
                "height": 100,
                "category": "demo",
            }
        )

    box_store.replace("d1", [box_store.Box("d1", 10, 20, 30, 40, "scratch")])
    box_store.replace("d2", [
        box_store.Box("d2", 0, 0, 20, 20, "dent"),
        box_store.Box("d2", 50, 50, 20, 20, "scratch"),
    ])
    return pd.DataFrame(rows), box_store.load()


def _export(dataset, tmp_path, **kwargs):
    resolved, frame = dataset
    return detection.export_yolo(tmp_path / "det", resolved=resolved, frame=frame, **kwargs)


# --- 무엇이 들어가고 무엇이 빠지는가 -----------------------------------------

def test_images_with_boxes_become_positives(dataset, tmp_path):
    result = _export(dataset, tmp_path)
    assert result.positives == 2


def test_normal_images_go_in_as_backgrounds(dataset, tmp_path):
    """결함만 학습하면 모델은 **'화면에는 항상 결함이 있다'** 를 배운다."""
    result = _export(dataset, tmp_path)
    assert result.backgrounds == 2
    assert (tmp_path / "det" / "labels" / "train" / "n1.txt").read_text() == ""


def test_backgrounds_can_be_left_out(dataset, tmp_path):
    result = _export(dataset, tmp_path, include_backgrounds=False)
    assert result.backgrounds == 0


def test_a_defect_without_boxes_is_not_used_as_background(dataset, tmp_path):
    """결함이라고만 하고 위치를 안 그린 이미지를 빈 라벨로 넣으면 **거짓을 가르친다.**"""
    result = _export(dataset, tmp_path)
    assert not (tmp_path / "det" / "labels" / "train" / "d3.txt").exists()
    assert result.skipped_unlabeled >= 1


def test_unlabelled_images_are_left_out(dataset, tmp_path):
    result = _export(dataset, tmp_path)
    assert not (tmp_path / "det" / "images" / "train" / "u1.jpg").exists()


def test_images_without_a_split_are_reported_not_guessed(sandbox, tmp_path):
    """분할을 임의로 정하면 같은 영상 프레임이 학습과 평가에 섞여 성능이 부풀려진다."""
    resolved = pd.DataFrame(
        [{"image_id": "x", "path": "none.jpg", "label": config.LABEL_NORMAL, "split": "unassigned"}]
    )
    result = detection.export_yolo(tmp_path / "det", resolved=resolved, frame=box_store.empty())
    assert result.skipped_unassigned == 1
    assert any("데이터 분할" in note for note in result.skipped_notes())


def test_missing_files_are_counted_not_crashed(sandbox, tmp_path):
    resolved = pd.DataFrame(
        [{"image_id": "x", "path": "gone.jpg", "label": config.LABEL_NORMAL, "split": "train"}]
    )
    result = detection.export_yolo(tmp_path / "det", resolved=resolved, frame=box_store.empty())
    assert result.skipped_missing == 1


# --- 분할 배치 ---------------------------------------------------------------

def test_each_image_lands_in_its_own_split(dataset, tmp_path):
    _export(dataset, tmp_path)
    root = tmp_path / "det"
    assert (root / "images" / "train" / "d1.jpg").exists()
    assert (root / "images" / "test" / "d2.jpg").exists()
    assert not (root / "images" / "train" / "d2.jpg").exists()


def test_every_image_has_a_label_file_beside_it(dataset, tmp_path):
    """YOLO는 이미지와 같은 이름의 txt를 찾는다. 하나라도 없으면 그 장은 조용히 빠진다."""
    root = tmp_path / "det"
    _export(dataset, tmp_path)
    for split in detection.SPLITS:
        images = {p.stem for p in (root / "images" / split).glob("*")}
        labels = {p.stem for p in (root / "labels" / split).glob("*.txt")}
        assert images == labels


# --- data.yaml ---------------------------------------------------------------

def test_class_numbers_match_the_label_files(dataset, tmp_path):
    """번호가 어긋나면 학습은 멀쩡히 돌고 **스크래치를 찍힘이라고 부른다.**"""
    result = _export(dataset, tmp_path)
    text = (tmp_path / "det" / detection.DATA_FILE).read_text()

    for index, name in enumerate(result.classes):
        assert f"  {index}: {name}" in text

    written = (tmp_path / "det" / "labels" / "train" / "d1.txt").read_text()
    assert written.split()[0] == str(result.classes.index("scratch"))


def test_data_yaml_declares_the_class_count(dataset, tmp_path):
    result = _export(dataset, tmp_path)
    assert f"nc: {len(result.classes)}" in (tmp_path / "det" / detection.DATA_FILE).read_text()


def test_data_yaml_points_at_an_absolute_root(dataset, tmp_path):
    """학습은 다른 폴더에서 실행되는 것이 보통이다. 상대경로면 못 찾는다."""
    _export(dataset, tmp_path)
    text = (tmp_path / "det" / detection.DATA_FILE).read_text()
    assert f"path: {(tmp_path / 'det').resolve()}" in text


# --- 링크와 복사 -------------------------------------------------------------

def test_images_are_linked_by_default(dataset, tmp_path):
    """4,500장을 복사하면 2GB가 또 생긴다."""
    _export(dataset, tmp_path)
    assert (tmp_path / "det" / "images" / "train" / "d1.jpg").is_symlink()


def test_copying_makes_the_folder_portable(dataset, tmp_path):
    """폴더째 다른 기계로 옮겨 학습할 생각이면 링크는 끊어진다."""
    _export(dataset, tmp_path, copy_images=True)
    target = tmp_path / "det" / "images" / "train" / "d1.jpg"
    assert target.is_file() and not target.is_symlink()


def test_exporting_twice_does_not_pile_up(dataset, tmp_path):
    """다시 내보낼 때마다 파일이 쌓이면 예전 라벨이 남아 학습에 섞인다."""
    first = _export(dataset, tmp_path)
    second = _export(dataset, tmp_path)
    assert first.total == second.total


# --- 시작할 만한 상태인가 ----------------------------------------------------

def test_readiness_says_no_when_there_are_too_few_boxes(dataset, tmp_path):
    """시작해 놓고 몇 시간 뒤에 아는 것보다 미리 아는 편이 싸다."""
    resolved, frame = dataset
    state = detection.readiness(resolved, frame)
    assert state["images_with_boxes"] == 2
    assert not state["ready"]


def test_readiness_names_the_thin_classes(dataset, tmp_path):
    resolved, frame = dataset
    assert set(detection.readiness(resolved, frame)["thin_classes"]) == {"scratch", "dent"}


def test_readiness_counts_assigned_splits(dataset, tmp_path):
    resolved, frame = dataset
    assert detection.readiness(resolved, frame)["splits_assigned"] == 6


def test_export_lives_inside_the_project(sandbox):
    """프로젝트를 나눈 이유가 현장을 섞지 않기 위해서다. 학습 폴더도 같다."""
    assert config.artifact_root() in detection.default_root().parents


# --- 쓸 수 있는 폴더인가 ----------------------------------------------------

def test_an_export_without_any_defect_is_refused(dataset, tmp_path):
    """**정상만 있는 학습 폴더는 조용한 함정이다.** 학습은 멀쩡히 돌고 모델은 '결함은
    없다'만 배우는데, 아무것도 못 찾아도 틀린 적이 없으니 지표까지 훌륭하게 나온다.
    """
    resolved, frame = dataset
    normal_only = resolved[resolved["label"] == config.LABEL_NORMAL]
    result = detection.export_yolo(
        tmp_path / "det", resolved=normal_only, frame=box_store.empty()
    )
    assert result.total > 0
    assert not result.usable
    assert "결함 이미지가 한 장도 없습니다" in result.blocking_note()


def test_a_normal_export_has_nothing_blocking(dataset, tmp_path):
    result = _export(dataset, tmp_path)
    assert result.usable
    assert result.blocking_note() == ""


def test_an_empty_export_says_so(sandbox, tmp_path):
    result = detection.export_yolo(
        tmp_path / "det", resolved=pd.DataFrame(), frame=box_store.empty()
    )
    assert not result.usable
    assert result.blocking_note()
