"""촬영 화각 점검 테스트 (V0).

이 점검이 하는 일은 **며칠 뒤에 알 일을 지금 알려주는 것**이다. 결함이 3픽셀로 잡히는
촬영은 어떤 모델을 써도 안 되는데, 보통은 영상을 다 뽑고 라벨링을 하고 학습을 돌린
뒤에야 그 사실이 드러난다.

그래서 여기서 가장 중요한 것은 «되는데 안 된다고 하는 것»보다 **«안 되는데 된다고 하는
것»을 막는 일**이다. 후자는 사람을 며칠 헛되이 일하게 만든다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_ai import framing


def _framing(**overrides):
    """1m 컨베이어를 1920px로 찍고 YOLO 640에 태우는 기본 구성."""
    base = {"defect_mm": 30.0, "fov_mm": 1000.0, "sensor_px": 1920, "model_px": 640}
    base.update(overrides)
    return framing.Framing(**base)


# --- 몇 픽셀로 잡히는가 -----------------------------------------------------

def test_native_pixels_follow_the_camera():
    """1920px가 1m를 담으면 1mm가 1.92px다."""
    assert _framing(defect_mm=10).native_px == pytest.approx(19.2)


def test_the_model_sees_less_than_the_camera_captured():
    """**원본에서 100px여도 640으로 줄이면 그만큼 작아진다.** 판정은 이쪽으로 해야 한다."""
    view = _framing(defect_mm=10)
    assert view.native_px == pytest.approx(19.2)
    assert view.model_input_px == pytest.approx(6.4)
    assert view.model_input_px < view.native_px


def test_a_bigger_model_input_sees_more():
    assert _framing(model_px=1280).model_input_px == 2 * _framing(model_px=640).model_input_px


def test_tiling_makes_the_defect_bigger_to_the_model():
    """조각이 담는 폭이 줄면 같은 결함이 입력에서 커진다 — 소프트웨어만 고치면 된다."""
    whole = _framing(defect_mm=10)
    split = _framing(defect_mm=10, tiles=2)
    assert split.model_input_px == pytest.approx(whole.model_input_px * 2)


def test_tiling_cannot_invent_detail_the_camera_never_captured():
    """**여기가 이 모듈의 핵심이다.** 확대는 정보를 만들지 못한다.

    이걸 놓치면 "타일을 잘게 나누면 다 잡힌다"는 거짓 결론이 나오고, 사람은 안 되는
    촬영으로 며칠을 쓴다.
    """
    view = _framing(defect_mm=10, tiles=99)
    assert view.model_input_px == pytest.approx(view.native_px)


# --- 판정 -------------------------------------------------------------------

@pytest.mark.parametrize(
    "defect_mm, expected",
    [
        (50.0, framing.VERDICT_OK),      # 입력에서 32px
        (25.0, framing.VERDICT_HARD),    # 입력에서 16px
        (5.0, framing.VERDICT_NO),       # 입력에서 3px
    ],
)
def test_verdict_follows_the_size_at_the_model_input(defect_mm, expected):
    assert _framing(defect_mm=defect_mm).verdict == expected


def test_a_three_pixel_defect_is_never_called_possible():
    """사람도 못 보는 크기다. 여기서 «가능»이 나오면 이 모듈은 해로운 것이 된다."""
    assert _framing(defect_mm=5).verdict == framing.VERDICT_NO
    assert not _framing(defect_mm=5).ok


def test_a_broken_field_of_view_does_not_crash():
    """0을 넣어도 화면이 죽으면 안 된다."""
    assert _framing(fov_mm=0).native_px == 0.0
    assert _framing(fov_mm=0).verdict == framing.VERDICT_NO


# --- 잡을 수 있는 가장 작은 결함 ---------------------------------------------

def test_the_smallest_catchable_defect_is_reported():
    """«우리 결함은 20mm쯤»은 알아도 «그래서 되나»는 모른다. 이 값과 비교하면 답이 나온다."""
    assert _framing().smallest_catchable_mm() == pytest.approx(31.25, rel=0.01)


def test_tiling_lowers_the_smallest_catchable_defect():
    assert _framing(tiles=2).smallest_catchable_mm() < _framing().smallest_catchable_mm()


def test_the_camera_sets_a_floor_that_software_cannot_pass():
    """1920px가 1m를 담으면 20px는 10.4mm다. **타일을 아무리 잘게 나눠도 그 아래는 없다.**"""
    floor = _framing(tiles=1).smallest_catchable_mm()
    fine = _framing(tiles=32).smallest_catchable_mm()
    assert fine == pytest.approx(20 * 1000 / 1920, rel=0.01)
    assert fine < floor


def test_a_better_camera_lowers_the_floor():
    assert (
        _framing(sensor_px=3840, tiles=32).smallest_catchable_mm()
        < _framing(sensor_px=1920, tiles=32).smallest_catchable_mm()
    )


# --- 무엇을 바꾸면 되는가 ----------------------------------------------------

def test_no_levers_are_offered_when_it_already_works():
    assert framing.levers(_framing(defect_mm=50)) == []


def test_the_cheapest_lever_comes_first():
    """화각과 장비는 현장을 바꿔야 하지만 타일 분할은 오늘 해 볼 수 있다."""
    found = framing.levers(_framing(defect_mm=25))
    assert found[0].name == "타일 분할"
    assert [item.name for item in found][-1] == "화각을 좁힌다"


def test_software_levers_are_marked_unreachable_when_the_camera_is_the_limit():
    """**소프트웨어로 못 넘는 벽을 넘을 수 있다고 말하면 안 된다.**

    원본이 이미 20px를 못 담고 있으면 타일도 입력 크기도 소용없다.
    """
    found = {item.name: item for item in framing.levers(_framing(defect_mm=5))}
    assert not found["타일 분할"].reachable
    assert not found["모델 입력 크기"].reachable
    assert found["카메라 해상도"].reachable
    assert found["화각을 좁힌다"].reachable


def test_narrowing_the_view_says_how_many_cameras_it_takes():
    """폭을 반으로 줄이면 같은 라인을 덮는 데 카메라가 두 대 든다 — 비용이 여기서 갈린다."""
    found = next(i for i in framing.levers(_framing(defect_mm=25)) if i.name == "화각을 좁힌다")
    assert "대" in found.cost


@pytest.mark.parametrize(
    "defect_mm, phrase",
    [
        (50.0, "그대로 진행"),
        (25.0, "타일 분할이 가장 쌉니다"),
        (5.0, "촬영 계획을 고쳐야"),
    ],
)
def test_advice_changes_with_the_verdict(defect_mm, phrase):
    assert phrase in framing.advice(_framing(defect_mm=defect_mm))


# --- 결함 크기를 모를 때 -----------------------------------------------------

def test_the_fraction_is_read_from_boxes_that_are_already_drawn():
    """결함 실물 크기를 재 본 사람은 드물지만 라벨링한 박스는 있다."""
    frame = pd.DataFrame(
        [
            {"image_id": "a", "w": 10, "h": 20},     # 짧은 변 10/1000 = 1%
            {"image_id": "b", "w": 60, "h": 30},     # 짧은 변 30/500 = 6%
        ]
    )
    images = pd.DataFrame(
        [
            {"image_id": "a", "width": 1000, "height": 500},
            {"image_id": "b", "width": 1000, "height": 500},
        ]
    )
    assert framing.fraction_from_boxes(frame, images) == pytest.approx(0.035)


def test_boxes_without_a_known_image_size_are_skipped():
    frame = pd.DataFrame([{"image_id": "a", "w": 10, "h": 20}])
    images = pd.DataFrame([{"image_id": "a", "width": None, "height": None}])
    assert framing.fraction_from_boxes(frame, images) is None


def test_no_boxes_means_no_estimate():
    assert framing.fraction_from_boxes(pd.DataFrame(), pd.DataFrame()) is None


def test_the_fraction_needs_the_source_framing_to_become_millimetres():
    """**비율은 물리 상수가 아니다.** 그 촬영이 담던 폭을 함께 놓아야 mm가 나온다.

    이걸 헷갈려 «지금 화각»을 넣으면 화각을 넓힐수록 결함도 커지는 결론이 나온다.
    """
    assert framing.defect_mm_from_fraction(0.0178, 300) == pytest.approx(5.34)
    assert framing.defect_mm_from_fraction(0.0178, 1000) == pytest.approx(17.8)


def test_the_documented_default_matches_the_measured_boxes():
    """문서에 적은 1.78%는 실제로 잰 값이다. 바뀌면 문서도 함께 바뀌어야 한다."""
    assert framing.TYPICAL_DEFECT_FRACTION == pytest.approx(0.0178)
