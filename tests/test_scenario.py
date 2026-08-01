"""운영 시나리오 시뮬레이터 테스트.

시뮬레이터의 존재 이유가 "4단계 화면이 작동하는 모습을 보이게 하는 것"이므로,
테스트도 그 관점에서 확인한다 — 로그가 과거 시각으로 쌓이는가, 환경을 바꾸면
드리프트가 실제로 잡히는가, 검수 라벨이 사람 라벨과 구분되는가.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_ai import config, labeling, monitoring, registry, scenario


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(config, "DATA_ROOT", data_root)
    monkeypatch.setattr(config, "RAW_DIR", data_root / "raw")
    monkeypatch.setattr(config, "INTERIM_DIR", data_root / "interim")
    monkeypatch.setattr(config, "MANIFEST_PATH", data_root / "manifest.csv")
    monkeypatch.setattr(config, "LABELS_PATH", data_root / "labels.csv")
    monkeypatch.setattr(config, "ARTIFACT_ROOT", artifacts)
    monkeypatch.setattr(config, "MODEL_DIR", artifacts / "models")
    monkeypatch.setattr(config, "REPORT_DIR", artifacts / "reports")
    monkeypatch.setattr(
        config, "ALL_DIRS",
        (data_root, data_root / "raw", artifacts, artifacts / "models", artifacts / "reports"),
    )
    config.ensure_dirs()
    return tmp_path


# --- Environment -----------------------------------------------------------

def test_environment_without_change_returns_same_image():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    result = scenario.Environment().apply(image)
    assert result is image


def test_brightness_raises_mean_and_clips_at_255():
    image = np.full((16, 16, 3), 250, dtype=np.uint8)
    result = scenario.Environment(brightness=40).apply(image)
    assert result.max() == 255          # 넘치지 않는다
    assert result.dtype == np.uint8


def test_brightness_shift_moves_mean():
    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    result = scenario.Environment(brightness=30).apply(image)
    assert result.mean() == pytest.approx(130, abs=1)


def test_blur_reduces_edge_sharpness():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:] = 255                  # 뚜렷한 경계
    blurred = scenario.Environment(blur=2.0).apply(image)
    # 경계가 뭉개지면 인접 픽셀 차이의 최대값이 줄어든다
    assert np.abs(np.diff(blurred[16, :, 0].astype(int))).max() < 255


def test_describe_mentions_only_active_changes():
    assert scenario.Environment().describe() == "변화 없음"
    assert "밝기" in scenario.Environment(brightness=10).describe()
    assert "흐림" not in scenario.Environment(brightness=10).describe()


# --- 표본 추출 -------------------------------------------------------------

def _rows(n_normal: int, n_defect: int) -> pd.DataFrame:
    labels = [config.LABEL_NORMAL] * n_normal + [config.LABEL_DEFECT] * n_defect
    return pd.DataFrame({
        "image_id": [f"i{i}" for i in range(len(labels))],
        "label": labels,
        "path_abs": [f"/tmp/i{i}.png" for i in range(len(labels))],
    })


def test_sample_respects_defect_rate():
    rows = _rows(500, 500)
    picked = scenario._sample(rows, 100, 0.10, np.random.default_rng(0))
    defects = (picked["label"] == config.LABEL_DEFECT).sum()
    assert len(picked) == 100
    assert defects == 10


def test_sample_without_defects_still_returns_rows():
    """결함이 없는 데이터에서도 멈추지 않아야 한다 (VisA 공식 train 분할이 이런 형태)."""
    picked = scenario._sample(_rows(50, 0), 20, 0.10, np.random.default_rng(0))
    assert len(picked) == 20
    assert (picked["label"] == config.LABEL_NORMAL).all()


def test_sample_returns_all_when_pool_is_small():
    picked = scenario._sample(_rows(5, 5), 100, 0.10, np.random.default_rng(0))
    assert len(picked) == 10


# --- 시각 배치 -------------------------------------------------------------

def test_timestamps_span_the_phase_and_are_ordered():
    from datetime import datetime, timezone

    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    stamps = scenario._timestamps(start, days=30, count=10)
    assert len(stamps) == 10
    assert stamps == sorted(stamps)
    parsed = pd.to_datetime(pd.Series(stamps))
    assert (parsed.max() - parsed.min()).days >= 26   # 구간 전체에 퍼져 있다


def test_timestamps_empty_for_zero_count():
    from datetime import datetime, timezone

    assert scenario._timestamps(datetime(2026, 1, 1, tzinfo=timezone.utc), 30, 0) == []


# --- 실행 ------------------------------------------------------------------

def test_run_without_rows_reports_warning_instead_of_raising(sandbox):
    result = scenario.run("v001", pd.DataFrame())
    assert result.total_logged == 0
    assert result.warnings and "1단계" in result.warnings[0]


def test_default_timeline_starts_stable_and_then_changes():
    phases = scenario.DEFAULT_TIMELINE
    assert not phases[0].environment.changed          # 기준선 구간
    assert all(p.environment.changed for p in phases[1:])
    # 구간마다 무슨 일이 있었고 어디를 봐야 하는지 설명이 붙어 있어야 한다
    assert all(p.narration and p.watch for p in phases)


def test_default_phase_size_meets_drift_sample_requirement():
    """구간 표본이 드리프트 판정 최소치보다 작으면 화면이 '표본 부족'만 띄운다."""
    assert all(p.n_images >= monitoring.MIN_DRIFT_SAMPLES for p in scenario.DEFAULT_TIMELINE)


def test_verification_labels_are_marked_as_scenario(sandbox):
    rows = _rows(8, 2)
    rng = np.random.default_rng(0)
    picked = scenario._record_verification(rows, list(rows["image_id"]), 1.0, rng)

    assert len(picked) == 10
    events = labeling.latest_events()
    assert (events["labeled_by"] == scenario.LABELED_BY).all()
    # 사후 검수 정답으로 쓰이려면 verified여야 한다
    assert events["verified"].astype(bool).all()


def test_verification_uses_manifest_truth_not_model_score(sandbox):
    """정답은 manifest 라벨에서 온다 — 모델 점수를 정답으로 쓰면 자기 채점이 된다."""
    rows = _rows(3, 3)
    scenario._record_verification(rows, list(rows["image_id"]), 1.0, np.random.default_rng(0))
    events = labeling.latest_events()   # 이미 image_id로 색인되어 있다
    for _, row in rows.iterrows():
        assert events.loc[row["image_id"], "label"] == row["label"]


def test_verification_skipped_when_ratio_is_zero(sandbox):
    picked = scenario._record_verification(_rows(5, 5), ["i0", "i1"], 0.0, np.random.default_rng(0))
    assert picked == set()
    assert labeling.latest_events().empty


# --- 정리 ------------------------------------------------------------------

def test_clear_removes_only_scenario_rows(sandbox):
    monitoring.log_inference([
        {"version": "v001", "image_id": "a", "score": 1.0, "threshold": 0.5, "decision": "normal"},
    ], note="실제 배치")
    monitoring.log_inference([
        {"version": "v001", "image_id": "b", "score": 2.0, "threshold": 0.5, "decision": "defect"},
    ], note=scenario.NOTE)

    removed = scenario.clear()
    remaining = monitoring.load_log()

    assert removed == 1
    assert len(remaining) == 1
    assert remaining.iloc[0]["image_id"] == "a"


def test_clear_on_empty_log_is_noop(sandbox):
    assert scenario.clear() == 0
