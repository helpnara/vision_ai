"""초보자 안내 로직 테스트.

이 모듈의 목적이 "초보자가 막히지 않게 하는 것"이므로, 테스트도 막히는 상황을 재현해
올바른 다음 걸음과 경고가 나오는지 확인한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from vision_ai import config, guide


def _resolved(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """(label, split) 목록으로 resolve() 형태의 프레임을 만든다."""
    return pd.DataFrame(
        {
            "image_id": [f"i{i}" for i in range(len(rows))],
            "label": [label for label, _ in rows],
            "split": [split for _, split in rows],
        }
    )


def _steps(**kwargs) -> list[guide.Step]:
    defaults = {
        "manifest": pd.DataFrame(),
        "resolved": pd.DataFrame(),
        "runs": pd.DataFrame(),
        "production": None,
        "log": pd.DataFrame(),
    }
    defaults.update(kwargs)
    return guide.pipeline_steps(**defaults)


# --- 진행 순서 -------------------------------------------------------------

def test_empty_project_points_to_collecting_images():
    step = guide.next_step(_steps())
    assert step is not None
    assert step.key == "collect"
    assert step.stage == 1
    assert "합성 샘플" in step.action     # 다운로드 없이 시작하는 길을 알려준다
    assert step.page == guide.PAGE_INGEST


def test_collected_but_unlabeled_points_to_labeling():
    manifest = pd.DataFrame({"image_id": ["a", "b"]})
    resolved = _resolved([(config.LABEL_UNLABELED, config.SPLIT_NONE)] * 2)
    step = guide.next_step(_steps(manifest=manifest, resolved=resolved))
    assert step.key == "label"
    assert step.page == guide.PAGE_LABELING


def test_labeled_but_unsplit_points_to_splitting():
    manifest = pd.DataFrame({"image_id": ["a", "b"]})
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_NONE)] * 2)
    step = guide.next_step(_steps(manifest=manifest, resolved=resolved))
    assert step.key == "split"
    assert "평가를 믿을 수 있" in step.action   # 왜 나눠야 하는지 이유를 준다


def test_split_done_points_to_training():
    manifest = pd.DataFrame({"image_id": ["a", "b"]})
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * 2)
    step = guide.next_step(_steps(manifest=manifest, resolved=resolved))
    assert step.key == "train"
    assert step.page == guide.PAGE_MODELING


def test_trained_but_not_promoted_points_to_registry():
    manifest = pd.DataFrame({"image_id": ["a"]})
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)])
    runs = pd.DataFrame({"run_id": ["r1"]})
    step = guide.next_step(_steps(manifest=manifest, resolved=resolved, runs=runs))
    assert step.key == "promote"
    assert step.stage == 4


def test_promoted_but_no_log_points_to_scenario():
    manifest = pd.DataFrame({"image_id": ["a"]})
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)])
    runs = pd.DataFrame({"run_id": ["r1"]})
    production = pd.Series({"version": "v001"})
    step = guide.next_step(
        _steps(manifest=manifest, resolved=resolved, runs=runs, production=production)
    )
    assert step.key == "operate"
    assert "운영 시나리오" in step.action


def test_everything_done_returns_none():
    manifest = pd.DataFrame({"image_id": ["a"]})
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)])
    steps = _steps(
        manifest=manifest, resolved=resolved,
        runs=pd.DataFrame({"run_id": ["r1"]}),
        production=pd.Series({"version": "v001"}),
        log=pd.DataFrame({"image_id": ["a"]}),
    )
    assert guide.next_step(steps) is None
    assert guide.progress(steps) == (6, 6)


def test_finished_steps_carry_no_action():
    manifest = pd.DataFrame({"image_id": ["a"]})
    steps = _steps(manifest=manifest)
    collect = next(s for s in steps if s.key == "collect")
    assert collect.done
    assert collect.action == ""
    assert "1장" in collect.detail


def test_every_step_names_where_to_go():
    for step in _steps():
        assert step.page
        assert step.title
        assert step.where


# --- 모델 선택 안내 ---------------------------------------------------------

def test_visa_official_split_blocks_supervised_baseline():
    """VisA 공식 분할은 train이 정상뿐이라 지도학습이 불가능하다 — 미리 알려야 한다."""
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * 50
                         + [(config.LABEL_DEFECT, config.SPLIT_TEST)] * 20)
    advice = guide.model_advice(resolved)

    assert advice.baseline_blocked
    assert advice.recommended == guide.KIND_ANOMALY
    assert any("한 장도 없" in note for note in advice.notes)


def test_balanced_data_recommends_baseline_first():
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * 50
                         + [(config.LABEL_DEFECT, config.SPLIT_TRAIN)] * 30)
    advice = guide.model_advice(resolved)

    assert not advice.baseline_blocked
    assert advice.recommended == guide.KIND_BASELINE
    assert "하한선" in advice.reason


def test_few_defects_recommends_anomaly_detection():
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * 50
                         + [(config.LABEL_DEFECT, config.SPLIT_TRAIN)] * 3)
    advice = guide.model_advice(resolved)

    assert not advice.baseline_blocked      # 불가능하지는 않다
    assert advice.recommended == guide.KIND_ANOMALY
    assert any("3장뿐" in note for note in advice.notes)


def test_no_normals_blocks_anomaly_detection():
    resolved = _resolved([(config.LABEL_DEFECT, config.SPLIT_TRAIN)] * 30)
    advice = guide.model_advice(resolved)

    assert advice.anomaly_blocked
    assert advice.recommended == guide.KIND_BASELINE


def test_too_few_normals_warns_about_per_position():
    n = guide.MIN_NORMAL_FOR_ANOMALY - 2
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * n
                         + [(config.LABEL_DEFECT, config.SPLIT_TRAIN)] * 20)
    advice = guide.model_advice(resolved)

    assert not advice.anomaly_blocked
    assert any("위치별 분포 학습" in note for note in advice.notes)


def test_advice_ignores_rows_outside_train_split():
    """평가 분할의 결함을 학습 가능성으로 오인하면 안 된다."""
    resolved = _resolved([(config.LABEL_NORMAL, config.SPLIT_TRAIN)] * 40
                         + [(config.LABEL_DEFECT, config.SPLIT_VAL)] * 40)
    assert guide.model_advice(resolved).baseline_blocked


def test_advice_on_empty_data_does_not_raise():
    advice = guide.model_advice(pd.DataFrame())
    assert advice.baseline_blocked and advice.anomaly_blocked
