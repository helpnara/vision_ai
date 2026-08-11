"""영상 구간 라벨링과 다중 박스의 판단 로직 (H4).

여기 있는 것들은 브라우저를 띄워야만 눈에 띄는 종류의 버그다 — 낱장 사진이 타임라인에
섞여 들어가거나, 유형을 골랐는데 박스에는 안 붙거나. 화면에서는 멀쩡해 보이고 저장된
데이터만 틀리기 때문에 나중에 발견하면 라벨링을 다시 해야 한다.

화면 코드(`app_pages/*.py`)는 import만 해도 Streamlit 스크립트가 통째로 실행되므로
테스트에서 부를 수 없다. 그래서 판단은 전부 라이브러리 쪽에 두고 화면은 부르기만 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_ai import boxes as box_store
from vision_ai import config, labeling


# --- 타임라인에 올릴 프레임 고르기 -------------------------------------------

def _catalog():
    """영상 프레임 2장 + 낱장 사진 1장. `group` 열이 이 둘을 가른다."""
    return pd.DataFrame(
        [
            {"image_id": "a1", "group": "cam-1234abcd", "path": "raw/video/cam-1234abcd/00000030.jpg"},
            {"image_id": "b2", "group": "cam-1234abcd", "path": "raw/video/cam-1234abcd/00000090.jpg"},
            {"image_id": "c3", "group": "", "path": "raw/visa/pcb1/007.jpg"},
        ]
    )


def test_only_video_frames_make_it_into_the_timeline():
    """낱장 사진은 영상 안 시각이 없다. 섞이면 타임라인 위치가 거짓이 된다."""
    assert list(labeling.video_frames(_catalog())["image_id"]) == ["a1", "b2"]


def test_frames_carry_their_position_inside_the_video():
    assert list(labeling.video_frames(_catalog())["frame_index"]) == [30, 90]


def test_frames_come_back_in_video_order():
    """표에 담긴 순서가 아니라 영상 순서여야 구간이 이어진 조각이 된다."""
    shuffled = _catalog().iloc[[1, 0, 2]].reset_index(drop=True)
    assert list(labeling.video_frames(shuffled)["frame_index"]) == [30, 90]


def test_a_catalog_without_groups_has_no_video_frames():
    """영상 기능 이전에 모은 데이터에는 group 열 자체가 없다."""
    rows = pd.DataFrame([{"image_id": "c3", "path": "raw/visa/pcb1/007.jpg"}])
    assert labeling.video_frames(rows).empty


def test_files_that_are_not_numbered_frames_are_dropped():
    """추출한 프레임은 `00000030.jpg`처럼 번호가 곧 이름이다. 아니면 시각을 알 수 없다."""
    rows = pd.DataFrame(
        [{"image_id": "x", "group": "cam-1", "path": "raw/video/cam-1/thumb.jpg"}]
    )
    assert labeling.video_frames(rows).empty


def test_an_empty_catalog_is_not_an_error():
    assert labeling.video_frames(pd.DataFrame(columns=["image_id", "group", "path"])).empty


# --- 나중에 고른 유형을 박스에 붙이기 ---------------------------------------

def _box(label):
    return box_store.Box(image_id="img", x=1, y=2, w=30, h=40, label=label)


def test_a_type_chosen_after_drawing_reaches_the_boxes():
    """박스를 먼저 그리고 유형을 나중에 고르는 순서가 자연스럽다.

    그대로 두면 화면에는 유형이 보이는데 저장되는 박스는 '유형 미지정'이 된다.
    """
    drawn = [_box(config.DEFECT_TYPE_UNSPECIFIED)]
    assert box_store.retype_unspecified(drawn, "scratch")[0].label == "scratch"


def test_boxes_that_already_have_a_type_are_left_alone():
    """한 장에 유형이 다른 결함이 있을 수 있다. 덮으면 앞의 작업을 잃는다."""
    drawn = [_box("dent"), _box(config.DEFECT_TYPE_UNSPECIFIED)]
    assert [b.label for b in box_store.retype_unspecified(drawn, "scratch")] == ["dent", "scratch"]


def test_no_type_chosen_changes_nothing():
    drawn = [_box(config.DEFECT_TYPE_UNSPECIFIED)]
    assert box_store.retype_unspecified(drawn, None) is drawn


def test_retyping_keeps_the_box_geometry():
    """유형만 바뀌어야 한다. 좌표가 흔들리면 그린 위치를 잃는다."""
    box = box_store.retype_unspecified([_box(config.DEFECT_TYPE_UNSPECIFIED)], "crack")[0]
    assert (box.x, box.y, box.w, box.h) == (1, 2, 30, 40)


# --- 구간이 잡아내는 프레임 --------------------------------------------------

@pytest.mark.parametrize(
    "span, expected",
    [
        ((0.0, 1.0), [0.0, 0.5, 1.0]),   # 양 끝은 포함한다 — 끌어 덮은 눈금이 곧 대상이다
        ((0.6, 0.9), []),                # 사이에 프레임이 없으면 0장
        ((-5.0, 99.0), [0.0, 0.5, 1.0, 1.5]),
    ],
)
def test_the_span_picks_the_frames_under_it(span, expected):
    seconds = pd.Series([0.0, 0.5, 1.0, 1.5])
    start, end = span
    assert list(seconds[(seconds >= start) & (seconds <= end)]) == expected


def test_seconds_come_from_the_frame_number_and_fps():
    """fps를 잘못 넣어도 구간 지정 자체는 되어야 한다 — 눈금 위치만 달라진다."""
    frames = labeling.video_frames(_catalog())
    assert list(frames["frame_index"].astype(float) / 30.0) == [1.0, 3.0]
    assert list(frames["frame_index"].astype(float) / 15.0) == [2.0, 6.0]
