"""두 영상 비교 리포트 (V4).

여기서 지키려는 것은 **결론이 다음 행동을 바꾼다**는 점이다. «촬영 탓»이면 화각을 맞추는
것이 맞고, «모델 탓»이면 재촬영은 시간 낭비다. 둘을 바꿔 말하면 사람을 엉뚱한 곳으로
보내는 것이므로, 판단이 뒤집히는 경계를 하나씩 못 박아 둔다.

동시에 **없는 확신을 만들지 않는 것**도 같이 지킨다. 표본이 얇으면 «판단 보류»여야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_ai import compare, features, monitoring, segments


def _report(truth: str, predicted: str, step: float = 0.5):
    return segments.evaluate(
        [c == "x" for c in truth],
        [c == "x" for c in predicted],
        [i * step for i in range(len(truth))],
    )


def _side(name: str, truth: str, predicted: str) -> compare.Side:
    report = _report(truth, predicted)
    return compare.side_from(report, name, frames=len(truth))


def _drift(psi: float, level: str, *, moved=(), n: int = 200) -> compare.Drift:
    return compare.Drift(psi_mean=psi, level=level, moved=tuple(moved), n_samples=n)


# 정답 구간 4곳짜리 영상. 잡는 정도만 바꿔 가며 쓴다.
FOUR = "xx..xx..xx..xx.."
ALL_CAUGHT = "xx..xx..xx..xx.."
HALF_CAUGHT = "xx..xx.........."
NONE_CAUGHT = "................"


# --- 네 갈래 판단 ------------------------------------------------------------

def test_performance_held_means_nothing_to_fix():
    verdict = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, ALL_CAUGHT), _drift(0.03, "안정")
    )
    assert verdict.cause() == compare.CAUSE_NONE
    assert "유지됩니다" in verdict.verdict()


def test_a_drop_with_a_changed_input_points_at_the_camera():
    verdict = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, HALF_CAUGHT), _drift(0.42, "변화")
    )
    assert verdict.cause() == compare.CAUSE_FRAMING
    assert "촬영" in verdict.verdict()


def test_a_drop_with_an_unchanged_input_points_at_the_model():
    """**이 갈래를 놓치면 사람을 재촬영하러 보낸다.** 같은 그림을 보고도 못 잡는 것이다."""
    verdict = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, HALF_CAUGHT), _drift(0.02, "안정")
    )
    assert verdict.cause() == compare.CAUSE_MODEL
    assert "재촬영으로는 낫지 않습니다" in verdict.advice()


def test_the_camera_verdict_does_not_tell_you_to_retrain_first():
    """촬영이 원인인데 재학습부터 하면, 잘못 찍힌 그림을 학습해 굳혀 버린다."""
    advice = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, HALF_CAUGHT), _drift(0.42, "변화")
    ).advice()
    assert advice.index("촬영 조건") < advice.index("다시 학습")


def test_the_moved_features_are_named_in_the_advice():
    """«분포가 변했다»만으로는 무엇을 맞춰야 할지 모른다. 밝기인지 초점인지가 필요하다."""
    advice = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, HALF_CAUGHT),
        _drift(0.42, "변화", moved=[("brightness", 0.5), ("blur", 0.3)]),
    ).advice()
    assert "brightness" in advice and "blur" in advice


def test_a_small_dip_is_not_called_a_drop():
    """지표는 늘 조금씩 흔들린다. 흔들릴 때마다 «성능 저하»라고 하면 아무도 안 믿게 된다."""
    verdict = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, "xx..xx..xx..xxx."), _drift(0.02, "안정")
    )
    assert verdict.performance_held and verdict.cause() == compare.CAUSE_NONE


def test_improving_is_not_a_drop():
    verdict = compare.Comparison(
        _side("A", FOUR, HALF_CAUGHT), _side("B", FOUR, ALL_CAUGHT), _drift(0.42, "변화")
    )
    assert verdict.recall_drop < 0
    assert verdict.cause() == compare.CAUSE_NONE


def test_holding_up_under_a_changed_camera_is_said_out_loud():
    """버틴 것도 정보다 — 다만 더 벌어지면 무너질 수 있다는 것을 같이 말해야 한다."""
    text = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, ALL_CAUGHT), _drift(0.42, "변화")
    ).verdict()
    assert "촬영은 달라졌는데도 버텼습니다" in text


def test_an_alarm_that_never_turns_off_is_not_called_healthy():
    """**새 영상에서 알람이 계속 켜지면 구간 재현율은 오히려 올라간다.**

    다 결함이라고 했으니 다 잡은 것이 된다. 재현율만 보면 «좋아졌다»가 되므로, 이 실패
    방식은 따로 잡아내지 않으면 그대로 통과한다 — 실측에서 실제로 그렇게 나왔다.
    """
    quiet = compare.Side("A", 240, 4, 0.75, 0.7, 1.0, false_alarm_time_ratio=0.15)
    shouting = compare.Side("B", 240, 4, 1.00, 1.0, 8.0, false_alarm_time_ratio=0.95)
    verdict = compare.Comparison(quiet, shouting, _drift(0.9, "변화"))
    assert verdict.started_flooding
    assert not verdict.performance_held
    assert verdict.cause() == compare.CAUSE_FRAMING
    assert "95%" in verdict.verdict()


def test_a_model_that_already_shouted_is_not_newly_blamed():
    """원래도 알람이 잦았다면 그것은 이 비교의 발견이 아니다 — 새로 생긴 문제만 짚는다."""
    loud = compare.Side("A", 240, 4, 1.0, 1.0, 8.0, false_alarm_time_ratio=0.9)
    also_loud = compare.Side("B", 240, 4, 1.0, 1.0, 8.0, false_alarm_time_ratio=0.9)
    verdict = compare.Comparison(loud, also_loud, _drift(0.02, "안정"))
    assert not verdict.started_flooding and verdict.performance_held


# --- 없는 확신을 만들지 않는다 -----------------------------------------------

def test_too_few_truth_segments_means_no_verdict():
    """구간이 1곳이면 재현율은 0 아니면 1이라 «떨어졌다»가 늘 100% 낙폭으로 보인다."""
    verdict = compare.Comparison(
        _side("A", "xx......", "xx......"), _side("B", "xx......", "........"),
        _drift(0.02, "안정"),
    )
    assert verdict.cause() == compare.CAUSE_UNKNOWN
    assert "정답 구간이 1곳뿐" in verdict.verdict()


def test_the_blocker_names_which_video_is_short():
    verdict = compare.Comparison(
        _side("영상A", FOUR, ALL_CAUGHT), _side("영상B", "xx......", "........"),
        _drift(0.02, "안정"),
    )
    assert "영상B" in verdict.verdict()


def test_a_drop_without_enough_samples_to_judge_the_input_stays_undecided():
    """원인을 못 가리는데 하나를 골라 말하면, 절반의 확률로 사람을 헛수고시킨다."""
    verdict = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, NONE_CAUGHT),
        _drift(float("nan"), monitoring.LEVEL_INSUFFICIENT, n=12),
    )
    assert verdict.cause() == compare.CAUSE_UNKNOWN
    assert "원인은 아직 가릴 수 없습니다" in verdict.verdict()


def test_an_undecided_verdict_says_how_to_unblock_it():
    verdict = compare.Comparison(
        _side("A", "xx......", "xx......"), _side("B", "xx......", "........"),
        _drift(0.02, "안정"),
    )
    assert "구간 라벨링" in verdict.advice()


# --- 입력 분포를 실제로 재 본다 ----------------------------------------------

def _matrix(rows: int, shift: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(shift, 1.0, size=(rows, len(features.FEATURE_NAMES)))


def test_the_same_distribution_reads_as_stable():
    drift = compare.drift_between(
        _matrix(300, seed=1), _matrix(300, seed=2), features.FEATURE_NAMES
    )
    assert drift.level == "안정" and not drift.changed


def test_a_shifted_distribution_reads_as_changed():
    drift = compare.drift_between(
        _matrix(300, seed=1), _matrix(300, shift=2.5, seed=2), features.FEATURE_NAMES
    )
    assert drift.changed and drift.psi_mean > monitoring.PSI_STABLE


def test_a_shifted_distribution_names_which_features_moved():
    drift = compare.drift_between(
        _matrix(300, seed=1), _matrix(300, shift=2.5, seed=2), features.FEATURE_NAMES
    )
    assert drift.moved and all(psi > monitoring.PSI_STABLE for _, psi in drift.moved)


def test_only_a_handful_of_features_are_named():
    """14개를 다 늘어놓으면 아무것도 안 읽힌다."""
    drift = compare.drift_between(
        _matrix(300, seed=1), _matrix(300, shift=2.5, seed=2), features.FEATURE_NAMES
    )
    assert len(drift.moved) <= compare.NAMED_FEATURES


def test_a_thin_sample_is_not_called_stable():
    """표본이 적으면 PSI가 무작위로 커진다. 그것을 «변화»로도 «안정»으로도 읽으면 안 된다."""
    drift = compare.drift_between(_matrix(300, seed=1), _matrix(10, seed=2), features.FEATURE_NAMES)
    assert not drift.measurable
    assert drift.level == monitoring.LEVEL_INSUFFICIENT


def test_an_empty_side_is_not_an_error():
    drift = compare.drift_between(
        np.empty((0, len(features.FEATURE_NAMES))), _matrix(100), features.FEATURE_NAMES
    )
    assert not drift.measurable


# --- 화면에 놓을 표 ----------------------------------------------------------

def test_the_table_puts_both_videos_on_the_same_row():
    """따로 그리면 눈으로 대 볼 수가 없다 — 대 보라고 만드는 화면이다."""
    table = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", FOUR, HALF_CAUGHT), _drift(0.02, "안정")
    ).table()
    assert list(table.columns) == ["항목", "A", "B"]
    assert "구간 재현율" in set(table["항목"])


def test_values_that_cannot_be_measured_are_not_printed_as_zero():
    """0은 «못 잡았다»로 읽힌다. 못 잰 것과 0은 다르다."""
    table = compare.Comparison(
        _side("A", FOUR, ALL_CAUGHT), _side("B", "........", "........"), _drift(0.02, "안정")
    ).table()
    assert "—" in set(table["B"])


@pytest.mark.parametrize("recall_gap", [0.0, compare.RECALL_DROP])
def test_the_drop_threshold_boundary_is_not_a_drop(recall_gap):
    side_a = compare.Side("A", 100, 10, 1.00, 0.9, 0.0)
    side_b = compare.Side("B", 100, 10, 1.00 - recall_gap, 0.9, 0.0)
    assert compare.Comparison(side_a, side_b, _drift(0.02, "안정")).performance_held
