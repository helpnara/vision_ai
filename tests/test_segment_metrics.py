"""구간 단위 검출률 (V3).

여기서 확인하는 것은 «지표가 사실과 같은 방향으로 틀리는가»이다. 프레임 재현율은 결함이
여러 장에 걸칠 때 **실제보다 나쁘게** 나오고, 오경보 건수는 영상 길이를 무시해서 **짧은
영상을 유리하게** 만든다. 둘 다 도입 판단을 그르치는 방향이다.
"""

from __future__ import annotations

import math

import pytest

from vision_ai import segments


def _flags(pattern: str) -> list[bool]:
    """`x`는 결함, `.`은 정상."""
    return [ch == "x" for ch in pattern]


def _seconds(count: int, step: float = 0.1) -> list[float]:
    return [i * step for i in range(count)]


def _report(truth: str, predicted: str, *, step: float = 0.1, **kwargs):
    return segments.evaluate(
        _flags(truth), _flags(predicted), _seconds(len(truth), step), **kwargs
    )


# --- 이어진 구간 찾기 --------------------------------------------------------

def test_a_run_of_defect_frames_is_one_segment():
    found = segments.runs(_flags("..xxx..."), _seconds(8))
    assert len(found) == 1 and found[0].length == 3


def test_two_separated_runs_are_two_segments():
    assert len(segments.runs(_flags("xx...xx.."), _seconds(9))) == 2


def test_a_segment_reports_the_seconds_it_covers():
    found, = segments.runs(_flags("..xxx..."), _seconds(8))
    assert found.span_text() == "0.2~0.4초"


def test_a_one_frame_segment_reports_one_moment():
    found, = segments.runs(_flags("..x....."), _seconds(8))
    assert found.span_text() == "0.2초"


def test_a_gap_can_be_forgiven():
    """결함이 지나가는 중에 한 장 흔들려 놓치면 사람 눈에는 한 구간인데 기계는 둘로 센다.

    구간 수가 부풀면 **오경보율이 실제보다 크게** 나온다.
    """
    assert len(segments.runs(_flags("xx.xx"), _seconds(5), gap=1)) == 1
    assert len(segments.runs(_flags("xx.xx"), _seconds(5), gap=0)) == 2


def test_a_gap_wider_than_allowed_still_splits():
    assert len(segments.runs(_flags("xx...xx"), _seconds(7), gap=1)) == 2


def test_a_segment_touching_the_end_is_not_lost():
    """마지막 구간을 닫는 것을 잊으면 **영상 끝에서 잡은 결함이 통째로 사라진다.**"""
    found = segments.runs(_flags("...xx"), _seconds(5))
    assert len(found) == 1 and found[0].end == 4


def test_nothing_true_means_no_segments():
    assert segments.runs(_flags("....."), _seconds(5)) == []


def test_mismatched_lengths_are_refused():
    """길이가 어긋난 채로 계산하면 엉뚱한 시각이 붙은 구간이 조용히 나온다."""
    with pytest.raises(ValueError):
        segments.runs(_flags("xx"), _seconds(5))


# --- 구간을 잡았는가 ---------------------------------------------------------

def test_catching_a_few_frames_of_a_long_defect_counts_as_catching_it():
    """**이것이 V3의 핵심이다.**

    100프레임짜리 결함에서 3장만 잡아도 알람은 울렸고 그 물건은 걸러진다. 사람에게는 잡은
    것인데 프레임 재현율은 3%로 찍힌다 — 그대로 보고하면 «쓸 수 없는 모델»이 된다.
    """
    truth = "." * 5 + "x" * 100 + "." * 5
    predicted = "." * 50 + "xxx" + "." * 57
    report = _report(truth, predicted)
    assert report.segment_recall == 1.0
    assert report.frame_recall < 0.05


def test_the_report_says_out_loud_that_the_two_numbers_disagree():
    """숫자만 나란히 두면 어느 쪽을 믿을지 모른다. 차이가 크면 말로 알려 줘야 한다."""
    truth = "." * 5 + "x" * 100 + "." * 5
    predicted = "." * 50 + "xxx" + "." * 57
    assert "실제보다 나쁘게" in _report(truth, predicted).contrast()


def test_no_warning_when_the_two_numbers_agree():
    assert _report("..xxx..", "..xxx..").contrast() == ""


def test_missing_a_short_defect_entirely_is_pointed_out():
    """긴 구간만 잘 잡고 짧게 지나가는 결함을 통째로 놓치는 것은 프레임 숫자에 안 드러난다."""
    truth = "xxxxxxxxxx" + "...." + "x" + "....."
    predicted = "xxxxxxxxxx" + "...." + "." + "....."
    report = _report(truth, predicted)
    assert report.segment_recall == 0.5
    assert "통째로 놓치고" in report.contrast()


def test_a_missed_segment_is_reported_with_its_time():
    report = _report("xx...x..", "xx......")
    assert [s.span_text() for s in report.missed] == ["0.5초"]


def test_requiring_more_hits_makes_a_single_blip_not_count():
    """사람이 확인하러 가는 비용이 큰 현장에서는 «한 장은 튄 것일 수 있다»를 반영해야 한다."""
    assert _report("..xxxx..", "...x....", min_hits=1).segment_recall == 1.0
    assert _report("..xxxx..", "...x....", min_hits=2).segment_recall == 0.0


def test_min_hits_below_one_is_treated_as_one():
    """0을 넣으면 «아무것도 안 잡아도 잡은 것»이 되어 지표가 무의미해진다."""
    assert _report("..xx..", "......", min_hits=0).segment_recall == 0.0


def test_with_no_truth_segments_the_recall_is_not_a_number():
    """정답이 없는데 «재현율 0»이라고 적으면 못 잡은 것처럼 읽힌다. 잴 것이 없는 것이다."""
    report = _report("......", "..x...")
    assert math.isnan(report.segment_recall)
    assert "잴 수 없습니다" in report.summary()


# --- 헛알람 ------------------------------------------------------------------

def test_an_alarm_far_from_any_defect_is_a_false_alarm():
    assert len(_report("xx......", "xx....x.").false_alarms) == 1


def test_an_alarm_that_slightly_misses_the_defect_is_not_punished():
    """정답 구간을 조금 빗맞힌 것까지 헛알람으로 세면, 잡았는데 벌점을 받는 셈이 된다."""
    assert _report("..xxx...", "...xxxx.").false_alarms == ()


def test_false_alarms_are_counted_per_minute_not_per_video():
    """30초에서 3번과 10분에서 3번은 전혀 다른 얘기인데 건수로만 적으면 같아 보인다."""
    dense = _report("." * 20, "x..x..x." + "." * 12, step=0.5)      # 약 9.5초에 3번
    sparse = _report("." * 20, "x..x..x." + "." * 12, step=10.0)    # 약 190초에 3번
    assert dense.false_alarms_per_min > sparse.false_alarms_per_min * 10


def test_a_single_frame_video_has_no_alarm_rate():
    """길이를 모르는데 분당 횟수를 적으면 지어낸 숫자가 된다."""
    report = segments.evaluate([False], [True], [0.0])
    assert math.isnan(report.false_alarms_per_min)


def test_a_model_that_shouts_defect_at_everything_does_not_score_perfectly():
    """**실측에서 실제로 걸린 구멍이다.**

    전 구간에 알람을 켜 놓으면 예측 구간은 하나뿐이고 그것이 모든 정답 구간과 겹친다.
    «정답과 겹치지 않은 예측 구간»으로 헛알람을 세면 **가장 나쁜 모델이 만점을 받는다** —
    구간 재현율 1.00에 헛알람 0건.
    """
    report = _report("..xx....xx....xx....", "x" * 20)
    assert report.segment_recall == 1.0
    assert report.false_alarms, "전부 결함이라고 했는데 헛알람이 0건이다"


def test_an_always_on_alarm_is_called_out_in_the_summary():
    """숫자가 좋아 보이면 사람은 숫자를 믿는다. 그럴 때일수록 말로 짚어 줘야 한다."""
    report = _report("..xx....xx....xx....", "x" * 20)
    assert report.flooding
    assert "켜져 있습니다" in report.summary()


def test_a_normal_model_is_not_accused_of_flooding():
    report = _report("..xx....xx....xx....", "..xx....xx..........")
    assert not report.flooding
    assert "켜져 있습니다" not in report.summary()


def test_a_long_alarm_next_to_a_defect_is_still_a_false_alarm():
    """봐주는 것은 **한두 장**의 넘겨 잡기지, 그 뒤로 계속 이어지는 알람이 아니다."""
    report = _report("xx..................", "xxxxxxxxxxxx........")
    assert report.false_alarms


def test_the_alarm_time_ratio_is_reported():
    assert _report("." * 10, "xxxxx.....").false_alarm_time_ratio == 0.5


def test_the_summary_carries_both_numbers():
    text = _report("..xxx...xx..", "..x.....xx..").summary()
    assert "2곳" in text and "구간 재현율" in text and "분당" in text


# --- 프레임 지표도 같이 들고 있다 --------------------------------------------

def test_frame_recall_is_still_reported():
    """구간만 보면 «몇 장이나 잡았나»를 잃는다. 임계값을 조정할 때 필요한 것은 그쪽이다."""
    assert _report("xxxx....", "xx......").frame_recall == 0.5


def test_frame_precision_is_still_reported():
    assert _report("xx......", "xxxx....").frame_precision == 0.5


def test_precision_without_any_alarm_is_not_a_number():
    assert math.isnan(_report("xx......", "........").frame_precision)


def test_lengths_that_do_not_line_up_are_refused():
    with pytest.raises(ValueError):
        segments.evaluate(_flags("xx"), _flags("xxx"), _seconds(2))


# --- 판정 로그에서 성적표 만들기 ---------------------------------------------

def _feedback(group: str, rows: list[tuple[str, int, int]]):
    """(image_id, 정답, 판정) 목록을 로그 모양의 표로."""
    import pandas as pd

    return pd.DataFrame(
        [{"image_id": i, "group": group, "y_true": t, "y_pred": p} for i, t, p in rows]
    )


def _frames(pairs: list[tuple[str, int]]):
    import pandas as pd

    return pd.DataFrame([{"image_id": i, "frame_index": n} for i, n in pairs])


def test_a_report_is_built_from_the_log_and_the_truth():
    report = segments.from_feedback(
        _feedback("cam-1", [("a", 0, 0), ("b", 1, 1), ("c", 1, 0), ("d", 0, 0)]),
        _frames([("a", 0), ("b", 30), ("c", 60), ("d", 90)]),
        "cam-1", fps=30.0,
    )
    assert report is not None
    assert len(report.truth) == 1 and report.segment_recall == 1.0


def test_the_log_is_put_back_into_video_order_first():
    """**이것이 이 함수의 전부다.**

    로그는 판정한 순서대로 쌓이는데 그 순서는 영상 순서가 아니다. 순서가 어긋난 채로
    «이어짐»을 따지면 한 구간이 여러 조각으로 부서져 오경보율이 실제보다 크게 나온다.
    """
    shuffled = _feedback("cam-1", [("c", 1, 1), ("a", 0, 0), ("d", 0, 0), ("b", 1, 1)])
    report = segments.from_feedback(
        shuffled, _frames([("a", 0), ("b", 30), ("c", 60), ("d", 90)]), "cam-1", fps=30.0
    )
    assert len(report.truth) == 1, "구간이 부서졌다 — 영상 순서로 세우지 않았다"
    assert report.truth[0].span_text() == "1.0~2.0초"


def test_other_videos_do_not_leak_in():
    """«영상 A로 만든 모델이 영상 B에서 얼마나 잡나»를 재는 기능이다. 섞이면 뜻이 없다."""
    mixed = _feedback("cam-1", [("a", 1, 1), ("b", 1, 1)])
    other = _feedback("cam-2", [("x", 1, 0), ("y", 1, 0)])
    import pandas as pd

    both = pd.concat([mixed, other], ignore_index=True)
    frames = _frames([("a", 0), ("b", 30), ("x", 0), ("y", 30)])
    assert segments.from_feedback(both, frames, "cam-1", fps=30.0).segment_recall == 1.0
    assert segments.from_feedback(both, frames, "cam-2", fps=30.0).segment_recall == 0.0


def test_frames_without_a_position_are_dropped():
    """번호를 모르는 프레임을 0초로 두면 없는 구간이 영상 앞머리에 생긴다."""
    report = segments.from_feedback(
        _feedback("cam-1", [("a", 1, 1), ("b", 1, 1), ("ghost", 1, 0)]),
        _frames([("a", 0), ("b", 30)]),
        "cam-1", fps=30.0,
    )
    assert len(report.truth) == 1 and report.truth[0].length == 2


def test_a_video_with_almost_no_frames_gives_nothing():
    """한 장으로는 구간도 길이도 알 수 없다. 지어내느니 «못 잰다»가 맞다."""
    assert segments.from_feedback(
        _feedback("cam-1", [("a", 1, 1)]), _frames([("a", 0)]), "cam-1"
    ) is None


def test_an_unknown_video_gives_nothing():
    assert segments.from_feedback(
        _feedback("cam-1", [("a", 1, 1), ("b", 1, 1)]),
        _frames([("a", 0), ("b", 30)]),
        "cam-9",
    ) is None


def test_an_empty_log_gives_nothing():
    import pandas as pd

    assert segments.from_feedback(pd.DataFrame(), _frames([("a", 0)]), "cam-1") is None


def test_fps_only_changes_the_clock_not_the_segments():
    """fps를 잘못 넣어도 «몇 곳을 잡았나»는 같아야 한다 — 눈금만 달라진다."""
    log = _feedback("cam-1", [("a", 1, 1), ("b", 0, 0), ("c", 1, 1)])
    frames = _frames([("a", 0), ("b", 30), ("c", 60)])
    fast = segments.from_feedback(log, frames, "cam-1", fps=30.0)
    slow = segments.from_feedback(log, frames, "cam-1", fps=15.0)
    assert len(fast.truth) == len(slow.truth) == 2
    assert slow.duration_sec == fast.duration_sec * 2


def test_a_bad_fps_does_not_crash_the_report():
    """0을 넣으면 초가 무한대가 된다. 화면의 숫자 입력은 0도 받을 수 있다."""
    report = segments.from_feedback(
        _feedback("cam-1", [("a", 1, 1), ("b", 1, 1)]),
        _frames([("a", 0), ("b", 30)]),
        "cam-1", fps=0.0,
    )
    assert report is not None and report.duration_sec > 0


def test_videos_are_listed_for_the_picker():
    import pandas as pd

    both = pd.concat(
        [_feedback("cam-1", [("a", 1, 1)]), _feedback("cam-2", [("b", 1, 1)])], ignore_index=True
    )
    assert segments.videos_in(both) == ["cam-1", "cam-2"]


def test_single_photos_are_not_offered_as_videos():
    """낱장 사진에는 «몇 초»가 없다. 목록에 섞이면 고르는 순간 빈 화면이 된다."""
    assert segments.videos_in(_feedback("", [("a", 1, 1)])) == []


def test_listing_videos_of_an_empty_log_is_not_an_error():
    import pandas as pd

    assert segments.videos_in(pd.DataFrame()) == []
