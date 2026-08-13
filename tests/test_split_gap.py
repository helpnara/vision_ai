"""분할 미배정을 화면이 알고 있는가 (S4).

분할을 한 번 돌리고 나서 데이터를 더 모으면, 새로 들어온 것은 배정 없이 남는다. 그 상태로
검출 학습 폴더를 내보내면 그만큼이 **통째로 빠지는데** 화면 어디에도 그 사실이 없었다.

실측에서 마스크로 박스를 447장 만들어 놓고 그중 400장이 미배정이라 학습 폴더에 47장만
들어간 적이 있다 — 내보내기 메시지를 읽고서야 알았다. 그때는 이미 «박스가 부족한가»를
의심하며 시간을 쓴 뒤였다.
"""

from __future__ import annotations

import pandas as pd

from vision_ai import config, labeling


def _images(*rows) -> pd.DataFrame:
    """(image_id, split, label) 목록."""
    return pd.DataFrame(
        [{"image_id": i, "split": s, "label": l} for i, s, l in rows]
    )


def _boxes(*image_ids) -> pd.DataFrame:
    return pd.DataFrame([{"image_id": i} for i in image_ids])


DEFECT, NORMAL, UNLABELED = config.LABEL_DEFECT, config.LABEL_NORMAL, config.LABEL_UNLABELED
TRAIN, TEST, NONE = config.SPLIT_TRAIN, config.SPLIT_TEST, config.SPLIT_NONE


# --- 세는 것 ------------------------------------------------------------------

def test_everything_assigned_reads_as_complete():
    gap = labeling.split_gap(_images(("a", TRAIN, DEFECT), ("b", TEST, NORMAL)), _boxes())
    assert gap.complete and gap.unassigned == 0
    assert "모두" in gap.message()


def test_unassigned_images_are_counted():
    gap = labeling.split_gap(
        _images(("a", TRAIN, DEFECT), ("b", NONE, DEFECT), ("c", NONE, NORMAL)), _boxes()
    )
    assert gap.unassigned == 2 and gap.assigned == 1


def test_the_message_says_how_many_are_missing():
    gap = labeling.split_gap(_images(("a", TRAIN, DEFECT), ("b", NONE, DEFECT)), _boxes())
    assert "1장이 분할 미배정" in gap.message()


def test_boxed_but_unassigned_images_are_counted_separately():
    """**이것이 이 항목의 숫자다.** 검출 학습에서 조용히 사라지는 정확한 장수다."""
    gap = labeling.split_gap(
        _images(("a", TRAIN, DEFECT), ("b", NONE, DEFECT), ("c", NONE, DEFECT)),
        _boxes("b", "c"),
    )
    assert gap.boxed_unassigned == 2


def test_a_boxed_image_that_is_assigned_is_not_counted_as_lost():
    gap = labeling.split_gap(_images(("a", TRAIN, DEFECT)), _boxes("a"))
    assert gap.boxed_unassigned == 0 and gap.complete


def test_labeled_but_unassigned_images_are_counted():
    gap = labeling.split_gap(
        _images(("a", NONE, DEFECT), ("b", NONE, UNLABELED)), _boxes()
    )
    assert gap.labeled_unassigned == 1


# --- 무엇을 하라고 말하는가 ---------------------------------------------------

def test_boxes_left_out_are_called_out_as_the_urgent_case():
    """박스를 그려 놓고 학습에 안 들어가는 것은 **이미 들인 노동이 버려지는** 경우다."""
    advice = labeling.split_gap(
        _images(("a", TRAIN, DEFECT), ("b", NONE, DEFECT)), _boxes("b")
    ).advice()
    assert "박스가 그려져 있습니다" in advice
    assert "층화 분할을 다시 실행" in advice


def test_labeled_leftovers_are_told_to_re_split():
    advice = labeling.split_gap(_images(("a", NONE, DEFECT)), _boxes()).advice()
    assert "층화 분할을 다시 실행" in advice
    assert "박스" not in advice, "박스가 없는데 박스 얘기를 하면 엉뚱한 곳을 보게 된다"


def test_unlabeled_leftovers_are_told_to_label_first():
    """미라벨은 분할을 다시 돌려도 안 들어간다(기본이 «라벨된 것만»). 라벨이 먼저다."""
    advice = labeling.split_gap(_images(("a", NONE, UNLABELED)), _boxes()).advice()
    assert "라벨을 먼저" in advice


def test_nothing_to_say_when_everything_is_assigned():
    assert labeling.split_gap(_images(("a", TRAIN, DEFECT)), _boxes()).advice() == ""


# --- 없는 열·빈 표 -------------------------------------------------------------

def test_an_empty_catalog_is_not_an_error():
    gap = labeling.split_gap(pd.DataFrame(), _boxes())
    assert gap.total == 0 and gap.complete


def test_a_catalog_without_a_split_column_counts_everything_as_unassigned():
    """분할 기능 이전에 모은 데이터에는 열 자체가 없다. 그때 «전부 배정됨»으로 보이면 안 된다."""
    rows = pd.DataFrame([{"image_id": "a", "label": DEFECT}])
    assert labeling.split_gap(rows, _boxes()).unassigned == 1


def test_an_unknown_split_value_is_not_treated_as_assigned():
    """오타나 옛 값이 «배정됨»으로 통과하면 빠진 장수가 과소집계된다."""
    rows = _images(("a", "훈련", DEFECT))
    assert labeling.split_gap(rows, _boxes()).unassigned == 1
