"""지표 도움말과 결과 판정 테스트."""

from __future__ import annotations

import pytest

from vision_ai import evaluate, glossary


# --- 도움말 구성 -----------------------------------------------------------

def test_primary_metrics_have_visible_captions():
    """핵심 지표는 눌러야 보이면 안 된다 — 캡션이 반드시 있어야 한다."""
    for entry in glossary.METRICS.values():
        if entry.primary:
            assert entry.short, f"{entry.key}에 캡션이 없습니다"


def test_every_metric_has_some_explanation():
    for entry in glossary.METRICS.values():
        assert glossary.detail(entry.key), f"{entry.key}에 설명이 없습니다"


def test_detail_merges_caption_for_non_primary_metrics():
    """캡션이 없는 지표는 ? 아이콘에 전부 담겨야 한다."""
    assert glossary.caption("f1") == ""
    assert "조화평균" in glossary.detail("f1")


def test_unknown_key_returns_empty_instead_of_raising():
    assert glossary.caption("없는지표") == ""
    assert glossary.detail("없는지표") == ""


def test_precision_help_warns_about_evaluation_composition():
    """정밀도는 오해하기 가장 쉬운 지표라 경고가 캡션에 있어야 한다."""
    assert "현장과 다릅니다" in glossary.caption("precision")


def test_arbitrary_values_are_documented_with_reasons():
    for key, note in glossary.ARBITRARY.items():
        assert note.strip(), f"{key}에 근거 설명이 없습니다"
    # 측정으로 정한 값과 관습값을 구분해서 밝힌다
    assert "관습" in glossary.ARBITRARY["psi"]
    assert "측정으로 정했습니다" in glossary.ARBITRARY["drift_samples"]


# --- 결과 판정 -------------------------------------------------------------

def _impact(recall, fpr, prevalence=0.01):
    return evaluate.business_impact(recall, fpr, prevalence=prevalence, volume=1000)


def test_strong_model_is_called_good():
    metrics = {"auroc": 0.96, "recall": 0.97}
    result = glossary.verdict(metrics, _impact(0.97, 0.05))
    assert result.level == glossary.LEVEL_GOOD
    assert "좋습니다" in result.headline


def test_middling_model_is_called_screening_only():
    metrics = {"auroc": 0.83, "recall": 0.93}
    result = glossary.verdict(metrics, _impact(0.93, 0.45))
    assert result.level == glossary.LEVEL_USABLE
    assert "스크리닝" in result.headline


def test_weak_model_gets_improvement_directions():
    metrics = {"auroc": 0.62, "recall": 0.70}
    result = glossary.verdict(metrics, _impact(0.70, 0.60))
    assert result.level == glossary.LEVEL_WEAK
    assert any("개선 방향" in a for a in result.actions)


def test_recall_below_target_is_called_out():
    metrics = {"auroc": 0.93, "recall": 0.88}
    result = glossary.verdict(metrics, _impact(0.88, 0.10), target_recall=0.95)
    assert any("목표 95%에 못 미칩니다" in a for a in result.actions)


def test_no_saving_is_reported_as_no_benefit():
    """모든 것을 결함으로 올리면 검수량이 줄지 않는다 — 도입 효과가 없다고 말해야 한다."""
    metrics = {"auroc": 0.80, "recall": 1.0}
    result = glossary.verdict(metrics, _impact(1.0, 1.0))
    assert any("검수량이 전혀 줄지 않습니다" in a for a in result.actions)


def test_low_field_precision_warns_against_auto_decision():
    metrics = {"auroc": 0.93, "recall": 0.96}
    result = glossary.verdict(metrics, _impact(0.96, 0.45))
    assert any("자동 판정에는 쓸 수 없" in a for a in result.actions)


def test_nan_auroc_explains_the_cause_and_fix():
    result = glossary.verdict({"auroc": float("nan"), "recall": 0.0}, _impact(0.0, 0.0))
    assert result.level == glossary.LEVEL_WEAK
    assert "한 종류의 라벨만" in result.headline
    assert any("데이터 분할" in a for a in result.actions)


def test_healthy_model_still_gets_a_next_action():
    """지적할 것이 없어도 다음에 할 일은 알려줘야 한다."""
    metrics = {"auroc": 0.97, "recall": 0.99}
    result = glossary.verdict(metrics, _impact(0.99, 0.01))
    assert result.actions
    assert any("4단계" in a for a in result.actions)


def test_measured_visa_numbers_land_in_usable_band():
    """실측값(AUROC 0.830, 재현율 0.925, 오탐률 0.455)이 '스크리닝용' 판정에 들어가는지 고정한다."""
    result = glossary.verdict(
        {"auroc": 0.830, "recall": 0.925}, _impact(0.925, 0.455)
    )
    assert result.level == glossary.LEVEL_USABLE


# --- 드리프트 원인 평문화 (A3) ---------------------------------------------

def test_every_real_feature_name_maps_to_plain_language():
    """실제 특징 67개가 전부 현장 언어로 옮겨져야 한다 — 하나라도 빠지면 화면에 내부명이 샌다."""
    from vision_ai import features

    unmapped = [n for n in features.FEATURE_NAMES if glossary.feature_family(n) is None]
    assert not unmapped, f"매핑되지 않은 특징: {unmapped}"


def test_brightness_features_point_to_lighting():
    assert "조명" in glossary.feature_family("gray_mean").cause
    assert glossary.feature_meaning("gray_p50") == "이미지 밝기"


def test_sharpness_features_point_to_focus():
    assert "초점" in glossary.feature_family("lap_p99").cause


def test_longer_prefix_wins_over_shorter():
    """`lap_var`가 `lap`보다 먼저 잡혀야 하듯, 접두사 충돌이 없어야 한다."""
    assert glossary.feature_meaning("lap_var") == glossary.feature_meaning("lap_p99")
    assert glossary.feature_meaning("hf_p99") == "미세한 무늬 성분"


def test_drift_causes_deduplicates_same_root_cause():
    """조명이 바뀌면 밝기 계열이 한꺼번에 뜬다 — 같은 원인을 반복하면 안 된다."""
    causes = glossary.drift_causes(["gray_mean", "gray_p50", "gray_p75"])
    assert len(causes) == 1


def test_drift_causes_keeps_order_of_severity():
    causes = glossary.drift_causes(["lap_p99", "gray_mean"])
    assert "초점" in causes[0]      # PSI가 가장 큰 것이 먼저 나온다
    assert "조명" in causes[1]


def test_drift_causes_ignores_unknown_names():
    assert glossary.drift_causes(["존재하지않는특징"]) == []


# --- 승격 전 점검 (A2) ------------------------------------------------------

def test_good_model_passes_promotion_check():
    check = glossary.promotion_check({"recall": 0.97}, _impact(0.97, 0.05))
    assert check.passed
    assert not check.problems


def test_low_recall_blocks_with_reason():
    check = glossary.promotion_check({"recall": 0.84}, _impact(0.84, 0.20))
    assert not check.passed
    assert any("재현율" in p and "놓친 채로" in p for p in check.problems)


def test_no_review_saving_is_flagged():
    check = glossary.promotion_check({"recall": 0.99}, _impact(0.99, 0.95))
    assert not check.passed
    assert any("검수량 절감률" in p for p in check.problems)


def test_missing_metrics_are_noted_not_treated_as_failure():
    """지표가 없는 것과 미달인 것은 다르다 — 없는 것을 실패로 처리하면 안 된다."""
    check = glossary.promotion_check({}, None)
    assert check.passed
    assert any("점검할 수 없" in n for n in check.notes)
    assert any("계산하지 못했" in n for n in check.notes)


def test_check_always_says_criteria_are_provisional():
    """기준이 승인 전 제안값임을 항상 밝혀야 한다."""
    for metrics in ({"recall": 0.99}, {"recall": 0.50}):
        check = glossary.promotion_check(metrics, _impact(0.9, 0.2))
        assert any("승인 전" in n for n in check.notes)


def test_measured_visa_model_would_be_flagged_on_promotion():
    """실측 모델(재현율 0.925)은 기준 미달이므로 경고가 떠야 한다."""
    check = glossary.promotion_check({"recall": 0.925}, _impact(0.925, 0.455))
    assert not check.passed
