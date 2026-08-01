"""사용자 설정 테스트.

설정은 **여러 화면이 같은 값을 읽는 것**이 핵심이다. 3단계에서 목표를 올려놓고 4단계가
옛 값으로 판단하면 같은 모델을 두 화면이 다르게 평가하게 된다.
"""

from __future__ import annotations

import json

import pytest

from vision_ai import config, glossary, settings


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_ROOT", data_root)
    monkeypatch.setattr(config, "ALL_DIRS", (data_root,))
    config.ensure_dirs()
    return data_root


# --- 저장과 불러오기 --------------------------------------------------------

def test_defaults_when_nothing_saved(sandbox):
    assert settings.load() == settings.DEFAULTS


def test_saved_values_come_back(sandbox):
    settings.save(settings.Settings(target_recall=0.98, prevalence=0.03))
    loaded = settings.load()
    assert loaded.target_recall == pytest.approx(0.98)
    assert loaded.prevalence == pytest.approx(0.03)


def test_unset_fields_fall_back_to_defaults(sandbox):
    """설정 파일에 일부 항목만 있어도 나머지는 기본값으로 채워야 한다."""
    (sandbox / settings.SETTINGS_FILE).write_text(
        json.dumps({"target_recall": 0.97}), encoding="utf-8"
    )
    loaded = settings.load()
    assert loaded.target_recall == pytest.approx(0.97)
    assert loaded.new_label_threshold == settings.DEFAULTS.new_label_threshold


def test_corrupt_file_falls_back_instead_of_crashing(sandbox):
    """설정이 깨졌다고 앱이 멈추면 안 된다 — 기본값으로 돌아가면 그만이다."""
    (sandbox / settings.SETTINGS_FILE).write_text("{ 망가진 json", encoding="utf-8")
    assert settings.load() == settings.DEFAULTS


def test_non_dict_file_falls_back(sandbox):
    (sandbox / settings.SETTINGS_FILE).write_text("[1, 2, 3]", encoding="utf-8")
    assert settings.load() == settings.DEFAULTS


def test_reset_restores_defaults(sandbox):
    settings.save(settings.Settings(target_recall=0.99))
    assert settings.reset() == settings.DEFAULTS
    assert settings.load() == settings.DEFAULTS


# --- 범위 강제 --------------------------------------------------------------

def test_recall_target_cannot_reach_one(sandbox):
    """재현율 1.0이면 '전부 결함'이 유일한 해가 되어 도구가 무의미해진다."""
    saved = settings.save(settings.Settings(target_recall=1.0))
    assert saved.target_recall < 1.0
    assert saved.target_recall == settings.BOUNDS["target_recall"][1]


def test_recall_target_cannot_go_absurdly_low(sandbox):
    saved = settings.save(settings.Settings(target_recall=0.1))
    assert saved.target_recall == settings.BOUNDS["target_recall"][0]


def test_prevalence_cannot_be_zero(sandbox):
    """불량률 0이면 정밀도 환산이 성립하지 않는다."""
    saved = settings.save(settings.Settings(prevalence=0.0))
    assert saved.prevalence > 0


def test_clamp_keeps_integer_fields_integer(sandbox):
    saved = settings.save(settings.Settings(volume=999_999_999))
    assert isinstance(saved.volume, int)
    assert saved.volume == settings.BOUNDS["volume"][1]


def test_garbage_value_falls_back_to_default():
    assert settings.clamp("target_recall", "이건 숫자가 아님") == settings.DEFAULTS.target_recall


def test_unbounded_key_passes_through():
    assert settings.clamp("존재하지않는항목", 123) == 123


# --- 화면과의 연결 ----------------------------------------------------------

def test_changed_reports_only_differences(sandbox):
    settings.save(settings.Settings(target_recall=0.98))
    diff = settings.changed()
    assert "target_recall" in diff
    assert diff["target_recall"] == (pytest.approx(0.98), settings.DEFAULTS.target_recall)
    assert "volume" not in diff


def test_promotion_check_uses_saved_setting(sandbox):
    """설정에서 목표를 올리면 승격 점검이 곧바로 그 값으로 판단해야 한다."""
    from vision_ai import evaluate

    metrics = {"recall": 0.96}
    impact = evaluate.business_impact(0.96, 0.05, prevalence=0.01, volume=1000)

    assert glossary.promotion_check(metrics, impact).passed      # 기본 0.95 → 통과

    settings.save(settings.Settings(target_recall=0.98))
    assert not glossary.promotion_check(metrics, impact).passed  # 0.98 → 미달


def test_every_field_has_a_label_and_help():
    """설정 화면에 이름만 있고 설명이 없으면 초보자는 바꿀 수 없다."""
    from dataclasses import fields

    for entry in fields(settings.Settings):
        assert settings.LABELS.get(entry.name), f"{entry.name} 라벨 없음"
        assert settings.HELP.get(entry.name), f"{entry.name} 설명 없음"
