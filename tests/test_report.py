"""결과 보고서 테스트.

보고서는 화면과 달리 **맥락 없이 전달된다.** 그래서 오해를 부를 수 있는 값에는 반드시
설명이 붙어야 하고, 경고는 인용되기 전에 눈에 들어오는 위치에 있어야 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_ai import config, report


def _resolved(source: str = "visa", n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(n)],
            "label": [config.LABEL_NORMAL] * (n - 2) + [config.LABEL_DEFECT] * 2,
            "split": [config.SPLIT_TRAIN] * (n - 4) + [config.SPLIT_TEST] * 4,
            "source": [source] * n,
            "category": ["pcb1"] * n,
        }
    )


_RESULT = {"kind": "anomaly", "model": "mahalanobis", "n_train": 900, "split": "test",
           "settings": {"per_position": True, "target_recall": 0.95}}
_METRICS = {"recall": 0.925, "precision": 0.683, "auroc": 0.830,
            "average_precision": 0.811, "fn": 3, "fp": 45,
            "threshold": 4.67, "false_alarm_rate": 0.455}


def test_report_contains_the_main_sections():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS)
    for heading in ("## 1. 데이터", "## 2. 모델", "## 3. 성능",
                    "## 4. 도입하면", "## 5. 판정", "## 6. 운영 현황", "## 7. 읽을 때"):
        assert heading in text, f"{heading} 누락"


def test_precision_always_carries_the_composition_warning():
    """보고서에서 정밀도만 떼어 인용되면 곧바로 오해가 된다."""
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS)
    assert "평가 데이터 구성 기준" in text
    assert "과대평가" in text


def test_savings_always_shown_with_the_cost():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS)
    assert "검수량 절감" in text
    assert "놓치는 결함" in text
    assert "대가로" in text


def test_synthetic_warning_appears_before_any_number():
    """맥락 없이 전달되므로 경고가 아래에 있으면 못 보고 인용된다."""
    text = report.build(resolved=_resolved(source="synthetic"), result=_RESULT, metrics=_METRICS)
    assert "합성 샘플" in text
    assert text.index("합성 샘플") < text.index("## 3. 성능")


def test_real_data_report_has_no_synthetic_warning():
    text = report.build(resolved=_resolved(source="visa"), result=_RESULT, metrics=_METRICS)
    assert "합성 샘플로 낸 것입니다" not in text


def test_mixed_sources_are_not_flagged_as_synthetic():
    """합성이 섞여 있을 뿐이면 '전부 합성'이라고 하면 안 된다."""
    frame = _resolved(source="visa")
    frame.loc[0, "source"] = "synthetic"
    text = report.build(resolved=frame, result=_RESULT, metrics=_METRICS)
    assert "합성 샘플로 낸 것입니다" not in text


def test_measured_numbers_are_rendered_as_documented():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS,
                        prevalence=0.01, volume=1000)
    assert "92.5%" in text          # 재현율
    assert "0.830" in text          # AUROC
    assert "54%" in text            # 검수량 절감 (문서에 적은 값)


def test_verdict_is_included_with_next_actions():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS)
    assert "스크리닝" in text        # 실측 수준이면 스크리닝용 판정
    assert "다음에 할 일" in text


def test_operations_section_states_when_nothing_is_running():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS, production=None)
    assert "서비스 중인 모델이 없습니다" in text


def test_operations_section_reports_running_version_and_drift():
    production = pd.Series({"version": "v003", "promoted_at": "2026-07-30T10:00:00+00:00"})
    log = pd.DataFrame({"image_id": ["a"] * 42})
    text = report.build(
        resolved=_resolved(), result=_RESULT, metrics=_METRICS,
        production=production, log=log,
        drift_summary={"level": "주의", "shifted": 2, "checked": 67},
    )
    assert "v003" in text
    assert "42건" in text
    assert "주의" in text


def test_empty_data_does_not_raise():
    text = report.build(resolved=pd.DataFrame(), result={}, metrics={})
    assert "등록된 이미지가 없습니다" in text


def test_missing_metrics_render_as_dash_not_error():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics={"recall": float("nan")})
    assert "—" in text


def test_provisional_criteria_are_disclosed():
    text = report.build(resolved=_resolved(), result=_RESULT, metrics=_METRICS)
    assert "승인 전 제안값" in text


def test_filename_has_timestamp_and_markdown_suffix():
    name = report.filename()
    assert name.startswith("defect-report-")
    assert name.endswith(".md")
