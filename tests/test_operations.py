"""4단계(운영관리) 코어 로직 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_ai import (
    config,
    experiments,
    features,
    labeling,
    models,
    monitoring,
    registry,
    serving,
)


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


def _metrics(recall: float = 0.9) -> dict:
    return {
        "threshold": 0.5, "recall": recall, "precision": 0.8, "f1": 0.85,
        "auroc": 0.95, "average_precision": 0.9, "tp": 9, "fp": 2, "fn": 1, "tn": 20,
    }


def _make_run(kind: str = "baseline", recall: float = 0.9, with_model: bool = True) -> tuple[str, str | None]:
    """3단계 실행 하나와 모델 파일을 만든다."""
    artifact = None
    if with_model:
        rng = np.random.default_rng(1)
        X = np.vstack([rng.normal(0, 1, (20, len(features.FEATURE_NAMES))),
                       rng.normal(3, 1, (20, len(features.FEATURE_NAMES)))])
        y = np.array([0] * 20 + [1] * 20)
        model = models.BaselineModel(models.BaselineConfig(kind="logreg")).fit(X, y)
        artifact = config.MODEL_DIR / "baseline_logreg.joblib"
        model.save(artifact)

    run_id = experiments.record_run(
        kind=kind, model="logreg", split="test", metrics=_metrics(recall),
        settings={"kind": "logreg"}, n_train=40, n_eval=32,
        artifacts={"model": str(artifact) if artifact else None},
    )
    return run_id, (str(artifact) if artifact else None)


# --- 레지스트리 -------------------------------------------------------------

def test_register_creates_version_and_copies_model(sandbox):
    run_id, artifact = _make_run()
    result = registry.register(run_id, artifact=artifact)

    assert result.version == "v001"
    assert result.artifact is not None
    from pathlib import Path
    assert Path(result.artifact).exists()
    # 3단계 산출물이 아니라 버전 폴더에 복사되어야 롤백이 가능하다
    assert "registry" in result.artifact and result.artifact != artifact


def test_registered_model_survives_stage3_overwrite(sandbox):
    """3단계가 같은 경로에 덮어써도 등록된 버전은 살아남아야 한다."""
    from pathlib import Path

    run_id, artifact = _make_run()
    result = registry.register(run_id, artifact=artifact)
    Path(artifact).write_bytes(b"overwritten-by-next-training")

    assert Path(result.artifact).read_bytes() != b"overwritten-by-next-training"
    loaded = serving.load_version(result.version)
    assert loaded.kind == "baseline"


def test_register_without_model_warns(sandbox):
    run_id, _ = _make_run(with_model=False)
    result = registry.register(run_id)
    assert result.artifact is None
    assert any("모델 파일" in w for w in result.warnings)
    assert result.version not in serving.selectable_versions()


def test_register_unknown_run_raises(sandbox):
    with pytest.raises(ValueError, match="실행 기록"):
        registry.register("does-not-exist")


def test_versions_increment(sandbox):
    first, _ = _make_run()
    second, _ = _make_run()
    assert registry.register(first).version == "v001"
    assert registry.register(second).version == "v002"


def test_promote_demotes_previous(sandbox):
    run_a, art_a = _make_run(recall=0.90)
    run_b, art_b = _make_run(recall=0.95)
    v1 = registry.register(run_a, artifact=art_a, promote_now=True).version
    v2 = registry.register(run_b, artifact=art_b).version

    assert registry.production()["version"] == v1
    registry.promote(v2)
    assert registry.production()["version"] == v2
    assert registry.get(v1)["status"] == registry.STATUS_ARCHIVED


def test_rollback_by_repromoting(sandbox):
    """이전 버전이 보관 상태로 남아 있으므로 다시 승격하면 롤백이 된다."""
    run_a, art_a = _make_run(recall=0.95)
    run_b, art_b = _make_run(recall=0.60)
    v1 = registry.register(run_a, artifact=art_a, promote_now=True).version
    v2 = registry.register(run_b, artifact=art_b, promote_now=True).version
    assert registry.production()["version"] == v2

    registry.promote(v1)
    assert registry.production()["version"] == v1
    assert registry.get(v2)["status"] == registry.STATUS_ARCHIVED


def test_promote_unknown_version_raises(sandbox):
    with pytest.raises(ValueError, match="등록되지 않은"):
        registry.promote("v999")


def test_archive(sandbox):
    run_id, artifact = _make_run()
    version = registry.register(run_id, artifact=artifact, promote_now=True).version
    registry.archive(version)
    assert registry.get(version)["status"] == registry.STATUS_ARCHIVED
    assert registry.production() is None


def test_production_none_when_empty(sandbox):
    assert registry.production() is None
    assert registry.get("v001") is None
    assert registry.load_registry().empty


def test_baseline_roundtrip(sandbox):
    run_id, artifact = _make_run()
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, len(features.FEATURE_NAMES)))
    baseline = registry.make_baseline(X, features.FEATURE_NAMES, scores=rng.random(50), label_ratio=0.3)

    version = registry.register(run_id, artifact=artifact, baseline=baseline).version
    loaded = registry.load_baseline(version)
    assert loaded["n_samples"] == 200
    assert loaded["label_ratio"] == 0.3
    assert len(loaded["edges"]) == len(features.FEATURE_NAMES)
    assert "score" in loaded


def test_load_baseline_missing(sandbox):
    run_id, artifact = _make_run()
    version = registry.register(run_id, artifact=artifact).version
    assert registry.load_baseline(version) is None


# --- 추론 -------------------------------------------------------------------

def test_load_version_missing_artifact_raises(sandbox):
    run_id, _ = _make_run(with_model=False)
    version = registry.register(run_id).version
    with pytest.raises(FileNotFoundError, match="모델 파일"):
        serving.load_version(version)


def test_load_unknown_version_raises(sandbox):
    with pytest.raises(ValueError, match="등록되지 않은"):
        serving.load_version("v999")


def test_run_batch_produces_log_records(sandbox, tmp_path):
    import cv2

    run_id, artifact = _make_run()
    version = registry.register(run_id, artifact=artifact, promote_now=True).version
    model = serving.load_version(version)

    paths = []
    for index in range(3):
        path = tmp_path / f"img{index}.png"
        cv2.imwrite(str(path), np.full((64, 64, 3), 100 + index * 20, dtype=np.uint8))
        paths.append(str(path))
    rows = pd.DataFrame({
        "image_id": ["a", "b", "c"], "path_abs": paths,
        "source": ["s"] * 3, "category": ["pcb1"] * 3,
    })

    result = serving.run_batch(model, rows)
    assert len(result) == 3
    assert result.features.shape == (3, len(features.FEATURE_NAMES))
    assert not result.failed
    for record in result.records:
        assert record["version"] == version
        assert record["decision"] in (config.LABEL_NORMAL, config.LABEL_DEFECT)
        assert record["latency_ms"] >= 0


def test_run_batch_skips_unreadable(sandbox, tmp_path):
    run_id, artifact = _make_run()
    version = registry.register(run_id, artifact=artifact).version
    model = serving.load_version(version)

    rows = pd.DataFrame({"image_id": ["missing"], "path_abs": [str(tmp_path / "nope.png")]})
    result = serving.run_batch(model, rows)
    assert len(result) == 0
    assert result.failed == ["missing"]


def test_run_batch_threshold_override(sandbox, tmp_path):
    import cv2

    run_id, artifact = _make_run()
    version = registry.register(run_id, artifact=artifact).version
    model = serving.load_version(version)

    path = tmp_path / "x.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 120, dtype=np.uint8))
    rows = pd.DataFrame({"image_id": ["x"], "path_abs": [str(path)]})

    lenient = serving.run_batch(model, rows, threshold=0.0).records[0]
    strict = serving.run_batch(model, rows, threshold=1.1).records[0]
    assert lenient["decision"] == config.LABEL_DEFECT
    assert strict["decision"] == config.LABEL_NORMAL


# --- 추론 로그 --------------------------------------------------------------

def test_log_roundtrip(sandbox):
    records = [
        {"version": "v001", "image_id": "a", "source": "s", "category": "pcb1",
         "score": 0.9, "threshold": 0.5, "decision": config.LABEL_DEFECT, "latency_ms": 12.0},
        {"version": "v001", "image_id": "b", "source": "s", "category": "pcb1",
         "score": 0.1, "threshold": 0.5, "decision": config.LABEL_NORMAL, "latency_ms": 11.0},
    ]
    assert monitoring.log_inference(records) == 2

    log = monitoring.load_log()
    assert len(log) == 2
    summary = monitoring.log_summary(log)
    assert summary["total"] == 2 and summary["defect"] == 1
    assert summary["defect_rate"] == 0.5


def test_log_appends(sandbox):
    record = {"version": "v001", "image_id": "a", "score": 0.5, "threshold": 0.5,
              "decision": config.LABEL_NORMAL, "latency_ms": 1.0}
    monitoring.log_inference([record])
    monitoring.log_inference([{**record, "image_id": "b"}])
    assert len(monitoring.load_log()) == 2


def test_log_empty_and_clear(sandbox):
    assert monitoring.log_inference([]) == 0
    assert monitoring.load_log().empty
    assert monitoring.log_summary(monitoring.load_log())["total"] == 0

    monitoring.log_inference([{"version": "v1", "image_id": "a", "score": 0.1,
                               "threshold": 0.5, "decision": config.LABEL_NORMAL, "latency_ms": 1}])
    monitoring.clear_log()
    assert monitoring.load_log().empty


# --- PSI / 드리프트 ---------------------------------------------------------

def test_psi_is_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 2000)
    baseline = registry.make_baseline(data[:, None], ["f"])
    assert monitoring.psi_from_edges(baseline["edges"]["f"], data) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    baseline = registry.make_baseline(rng.normal(0, 1, (3000, 1)), ["f"])
    edges = baseline["edges"]["f"]
    small = monitoring.psi_from_edges(edges, rng.normal(0.3, 1, 3000))
    large = monitoring.psi_from_edges(edges, rng.normal(1.5, 1, 3000))
    assert 0 < small < large
    assert large > monitoring.PSI_SHIFTED


def test_psi_constant_feature_is_nan():
    assert np.isnan(monitoring.psi_from_edges([1.0, 1.0, 1.0], np.array([1.0, 1.0])))


def test_psi_empty_input_is_nan():
    assert np.isnan(monitoring.psi_from_edges([0.0, 1.0, 2.0], np.array([])))


def test_psi_rejects_mismatched_expected():
    with pytest.raises(ValueError, match="기준 비율 개수"):
        monitoring.psi_from_edges([0.0, 1.0, 2.0, 3.0], np.array([0.5, 1.5]), expected=np.array([1.0]))


def test_coarsen_edges_returns_matching_proportions():
    edges = list(np.linspace(0, 10, 11))  # 10구간
    merged, proportions = monitoring.coarsen_edges(edges, n=90)  # 90//30 = 3구간
    assert len(merged) - 1 == len(proportions)
    assert proportions.sum() == pytest.approx(1.0)
    assert len(merged) - 1 <= 3


def test_coarsen_keeps_all_bins_when_samples_plenty():
    edges = list(np.linspace(0, 10, 11))
    merged, proportions = monitoring.coarsen_edges(edges, n=10_000)
    assert len(merged) == len(edges)
    assert proportions == pytest.approx(np.full(10, 0.1))


def test_uneven_merge_uses_correct_expected_proportions():
    """데실을 불균등하게 묶으면 기준 비율도 불균등해야 한다 — 균등 가정 시 PSI가 부풀려진다."""
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 5000)
    baseline = registry.make_baseline(reference[:, None], ["f"])
    merged, proportions = monitoring.coarsen_edges(baseline["edges"]["f"], n=200)

    sample = rng.normal(0, 1, 4000)  # 같은 분포
    correct = monitoring.psi_from_edges(merged, sample, expected=proportions)
    naive = monitoring.psi_from_edges(merged, sample)  # 균등 가정
    assert correct < 0.05
    if not np.allclose(proportions, proportions[0]):
        assert naive > correct


def test_baseline_stores_actual_bin_proportions():
    """값에 동점이 많으면 분위 경계가 겹쳐 구간이 줄고, 남은 구간의 비율은 균등이 아니다."""
    tied = np.concatenate([np.full(1200, 5.0), np.random.default_rng(0).normal(5, 1, 800)])
    baseline = registry.make_baseline(tied[:, None], ["tied"])

    props = baseline["props"]["tied"]
    assert len(props) == len(baseline["edges"]["tied"]) - 1
    assert sum(props) == pytest.approx(1.0)
    assert max(props) > 0.5, "동점 구간이 대부분의 질량을 차지해야 한다"


def test_tied_values_do_not_produce_false_drift():
    """회귀: 동점이 많은 특징에서 분포가 그대로인데 '변화'로 잘못 판정되던 문제."""
    rng = np.random.default_rng(1)
    reference = np.concatenate([np.full(1200, 5.0), rng.normal(5, 1, 800)])
    baseline = registry.make_baseline(reference[:, None], ["tied"])
    sample = np.concatenate([np.full(300, 5.0), rng.normal(5, 1, 200)])

    edges, props = monitoring.coarsen_edges(
        baseline["edges"]["tied"], 500, props=baseline["props"]["tied"]
    )
    correct = monitoring.psi_from_edges(edges, sample, expected=props)
    assert monitoring.drift_level(correct) == "안정", correct

    # 저장된 비율을 무시하고 등분위로 가정하면 같은 분포인데도 '변화'가 된다
    naive_edges, naive_props = monitoring.coarsen_edges(baseline["edges"]["tied"], 500)
    naive = monitoring.psi_from_edges(naive_edges, sample, expected=naive_props)
    assert naive > correct


def test_tied_values_still_detect_real_shift():
    rng = np.random.default_rng(2)
    reference = np.concatenate([np.full(1200, 5.0), rng.normal(5, 1, 800)])
    baseline = registry.make_baseline(reference[:, None], ["tied"])
    shifted = np.concatenate([np.full(300, 6.5), rng.normal(6.5, 1, 200)])

    drift = monitoring.feature_drift(baseline, shifted[:, None], ["tied"])
    assert drift.iloc[0]["level"] == "변화"


def test_feature_drift_uses_stored_proportions():
    """feature_drift가 저장된 비율을 실제로 쓰는지 (균등 가정으로 되돌아가지 않는지)."""
    rng = np.random.default_rng(3)
    reference = np.concatenate([np.full(1500, 2.0), rng.normal(2, 0.5, 500)])
    baseline = registry.make_baseline(reference[:, None], ["tied"])
    sample = np.concatenate([np.full(150, 2.0), rng.normal(2, 0.5, 50)])

    with_props = monitoring.feature_drift(baseline, sample[:, None], ["tied"])
    stripped = {**baseline, "props": {}}
    without = monitoring.feature_drift(stripped, sample[:, None], ["tied"])
    assert with_props.iloc[0]["psi"] < without.iloc[0]["psi"]


def test_small_sample_is_withheld_not_flagged():
    """소표본에서는 같은 분포도 PSI가 커진다 — 드리프트로 단정하지 않고 보류해야 한다."""
    rng = np.random.default_rng(2)
    baseline = registry.make_baseline(rng.normal(0, 1, (2000, 5)), [f"f{i}" for i in range(5)])
    tiny = rng.normal(0, 1, (monitoring.MIN_DRIFT_SAMPLES - 1, 5))

    drift = monitoring.feature_drift(baseline, tiny, [f"f{i}" for i in range(5)])
    assert (drift["level"] == monitoring.LEVEL_INSUFFICIENT).all()
    assert drift["psi"].isna().all()
    assert monitoring.drift_summary(drift)["level"] == monitoring.LEVEL_INSUFFICIENT


def test_same_distribution_does_not_raise_false_alarm():
    rng = np.random.default_rng(7)
    names = [f"f{i}" for i in range(8)]
    baseline = registry.make_baseline(rng.normal(0, 1, (3000, 8)), names)
    current = rng.normal(0, 1, (500, 8))

    summary = monitoring.drift_summary(monitoring.feature_drift(baseline, current, names))
    assert summary["level"] == "안정", summary


def test_shifted_distribution_is_detected():
    rng = np.random.default_rng(8)
    names = [f"f{i}" for i in range(6)]
    baseline = registry.make_baseline(rng.normal(0, 1, (3000, 6)), names)
    current = rng.normal(0, 1, (500, 6))
    current[:, 0] += 2.0  # 한 특징만 크게 이동

    drift = monitoring.feature_drift(baseline, current, names)
    assert monitoring.drift_summary(drift)["level"] == "변화"
    assert drift.iloc[0]["feature"] == "f0"
    assert abs(drift.iloc[0]["shift_sigma"]) > 1.0


def test_drift_level_labels():
    assert monitoring.drift_level(0.05) == "안정"
    assert monitoring.drift_level(0.15) == "주의"
    assert monitoring.drift_level(0.40) == "변화"
    assert monitoring.drift_level(float("nan")) == monitoring.LEVEL_INSUFFICIENT


def test_score_drift(sandbox):
    rng = np.random.default_rng(9)
    baseline = registry.make_baseline(
        rng.normal(0, 1, (100, 2)), ["a", "b"], scores=rng.normal(0.3, 0.1, 100)
    )
    result = monitoring.score_drift(baseline, np.full(50, 0.8))
    assert result["available"] is True
    assert result["shift_sigma"] > 1.0

    assert monitoring.score_drift({"edges": {}}, np.array([0.1]))["available"] is False


# --- 성능 드리프트 ----------------------------------------------------------

def _seed_feedback(sandbox, *, decisions, truths) -> pd.DataFrame:
    """추론 로그와 사람 확인 라벨을 만들어 대조 프레임을 얻는다."""
    manifest_rows = []
    records = []
    for index, (decision, truth) in enumerate(zip(decisions, truths)):
        image_id = f"img{index}"
        manifest_rows.append({
            "image_id": image_id, "path": f"{image_id}.png", "source": "s", "category": "pcb1",
            "split": config.SPLIT_TEST, "label": config.LABEL_UNLABELED,
            "defect_type": config.DEFECT_TYPE_NONE,
        })
        records.append({
            "version": "v001", "image_id": image_id, "source": "s", "category": "pcb1",
            "score": 0.9 if decision == config.LABEL_DEFECT else 0.1,
            "threshold": 0.5, "decision": decision, "latency_ms": 5.0,
        })
        labeling.record_label(image_id, label=truth, verified=True)

    from vision_ai import storage

    storage.save_manifest(pd.DataFrame(manifest_rows))
    monitoring.log_inference(records)
    return monitoring.feedback_frame(monitoring.load_log())


def test_feedback_frame_matches_human_labels(sandbox):
    feedback = _seed_feedback(
        sandbox,
        decisions=[config.LABEL_DEFECT, config.LABEL_NORMAL, config.LABEL_DEFECT, config.LABEL_NORMAL],
        truths=[config.LABEL_DEFECT, config.LABEL_DEFECT, config.LABEL_NORMAL, config.LABEL_NORMAL],
    )
    assert len(feedback) == 4
    assert feedback.set_index("image_id")["outcome"].to_dict() == {
        "img0": "TP", "img1": "FN", "img2": "FP", "img3": "TN"
    }

    metrics = monitoring.performance_metrics(feedback)
    assert metrics["tp"] == 1 and metrics["fn"] == 1 and metrics["fp"] == 1 and metrics["tn"] == 1
    assert metrics["recall"] == 0.5


def test_feedback_ignores_unverified_folder_labels(sandbox):
    """폴더 구조에서 추론한 라벨을 정답으로 쓰면 자기 채점이 된다."""
    from vision_ai import storage

    storage.save_manifest(pd.DataFrame([{
        "image_id": "a", "path": "a.png", "source": "s", "category": "pcb1",
        "split": config.SPLIT_TEST, "label": config.LABEL_DEFECT,
        "defect_type": config.DEFECT_TYPE_UNSPECIFIED,
    }]))
    monitoring.log_inference([{
        "version": "v001", "image_id": "a", "score": 0.9, "threshold": 0.5,
        "decision": config.LABEL_DEFECT, "latency_ms": 1.0,
    }])
    assert monitoring.feedback_frame(monitoring.load_log()).empty


def test_performance_by_version(sandbox):
    feedback = _seed_feedback(
        sandbox,
        decisions=[config.LABEL_DEFECT, config.LABEL_NORMAL],
        truths=[config.LABEL_DEFECT, config.LABEL_DEFECT],
    )
    by_version = monitoring.performance_by_version(feedback)
    assert len(by_version) == 1
    assert by_version.iloc[0]["version"] == "v001"
    assert by_version.iloc[0]["recall"] == 0.5


def test_performance_metrics_on_empty():
    assert monitoring.performance_metrics(pd.DataFrame()) is None
    assert monitoring.performance_by_version(pd.DataFrame()).empty
    assert monitoring.performance_trend(pd.DataFrame()).empty


def test_feedback_frame_without_log(sandbox):
    assert monitoring.feedback_frame(monitoring.empty_log()).empty


# --- 재학습 판단 ------------------------------------------------------------

def test_no_production_blocks_monitoring(sandbox):
    decision = monitoring.retraining_signals(production=None)
    assert decision.recommended is False
    assert decision.signals[0].key == "no_production"
    assert decision.signals[0].level == monitoring.LEVEL_ALERT


def test_recall_drop_recommends_retraining(sandbox):
    feedback = _seed_feedback(
        sandbox,
        decisions=[config.LABEL_NORMAL, config.LABEL_NORMAL, config.LABEL_DEFECT],
        truths=[config.LABEL_DEFECT, config.LABEL_DEFECT, config.LABEL_DEFECT],
    )
    run_id, artifact = _make_run(recall=0.95)
    registry.register(run_id, artifact=artifact, promote_now=True)

    decision = monitoring.retraining_signals(
        production=registry.production(), log=monitoring.load_log(), feedback=feedback,
    )
    assert decision.recommended is True
    keys = {s.key for s in decision.alerts}
    assert "recall_drop" in keys
    assert "miss" in keys  # 미탐 발생도 별도 경보


def test_stable_performance_does_not_recommend(sandbox):
    feedback = _seed_feedback(
        sandbox,
        decisions=[config.LABEL_DEFECT, config.LABEL_DEFECT, config.LABEL_NORMAL],
        truths=[config.LABEL_DEFECT, config.LABEL_DEFECT, config.LABEL_NORMAL],
    )
    run_id, artifact = _make_run(recall=0.95)
    registry.register(run_id, artifact=artifact, promote_now=True)

    rng = np.random.default_rng(4)
    names = list(features.FEATURE_NAMES)
    baseline = registry.make_baseline(rng.normal(0, 1, (2000, len(names))), names)
    drift = monitoring.feature_drift(baseline, rng.normal(0, 1, (400, len(names))), names)

    decision = monitoring.retraining_signals(
        production=registry.production(), log=monitoring.load_log(),
        feedback=feedback, drift=drift, new_labels=3, new_label_threshold=50,
    )
    assert decision.recommended is False
    assert not decision.alerts


def test_data_drift_triggers_alert(sandbox):
    run_id, artifact = _make_run(recall=0.9)
    registry.register(run_id, artifact=artifact, promote_now=True)

    rng = np.random.default_rng(6)
    names = [f"f{i}" for i in range(5)]
    baseline = registry.make_baseline(rng.normal(0, 1, (3000, 5)), names)
    current = rng.normal(0, 1, (400, 5))
    current[:, 0] += 2.5
    drift = monitoring.feature_drift(baseline, current, names)

    decision = monitoring.retraining_signals(
        production=registry.production(), log=monitoring.load_log(), drift=drift,
    )
    assert decision.recommended is True
    assert "data_drift" in {s.key for s in decision.alerts}


def test_small_sample_drift_warns_not_alerts(sandbox):
    """표본 부족은 경보가 아니라 경고여야 한다 — 오경보로 재학습을 유발하면 안 된다."""
    run_id, artifact = _make_run(recall=0.9)
    registry.register(run_id, artifact=artifact, promote_now=True)

    rng = np.random.default_rng(10)
    names = [f"f{i}" for i in range(4)]
    baseline = registry.make_baseline(rng.normal(0, 1, (2000, 4)), names)
    drift = monitoring.feature_drift(baseline, rng.normal(0, 1, (20, 4)), names)

    decision = monitoring.retraining_signals(
        production=registry.production(), log=monitoring.load_log(), drift=drift,
    )
    keys = {s.key for s in decision.warnings}
    assert "drift_insufficient" in keys
    assert "data_drift" not in {s.key for s in decision.alerts}


def test_new_labels_recommend_retraining(sandbox):
    run_id, artifact = _make_run(recall=0.9)
    registry.register(run_id, artifact=artifact, promote_now=True)

    decision = monitoring.retraining_signals(
        production=registry.production(), log=monitoring.load_log(),
        new_labels=120, new_label_threshold=50,
    )
    assert decision.recommended is True
    assert "new_labels" in {s.key for s in decision.signals}


def test_new_labels_since_counts_after_promotion(sandbox):
    """승격 이후 쌓인 라벨을 센다. 타임스탬프가 초 단위라 경계는 포함해서 센다."""
    from datetime import datetime, timedelta, timezone

    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="seconds")
    labeling.record_labels([
        {"image_id": "old", "label": config.LABEL_NORMAL, "labeled_at": past},
    ])
    run_id, artifact = _make_run()
    registry.register(run_id, artifact=artifact, promote_now=True)
    promoted_at = registry.production()["promoted_at"]

    labeling.record_label("new1", label=config.LABEL_DEFECT, defect_type="scratch")
    labeling.record_label("new2", label=config.LABEL_NORMAL)

    assert monitoring.new_labels_since(promoted_at) == 2
    assert monitoring.new_labels_since(None) == 0


def test_new_labels_since_includes_same_second_boundary(sandbox):
    """승격과 같은 초에 남긴 라벨을 빠뜨리면 재학습 시점이 늦어진다."""
    stamp = "2026-07-30T12:00:00+00:00"
    labeling.record_labels([{"image_id": "a", "label": config.LABEL_NORMAL, "labeled_at": stamp}])
    assert monitoring.new_labels_since(stamp) == 1


# --- 판정 이력 조회 ---------------------------------------------------------

def test_trace_image_collects_all_stages(sandbox):
    from vision_ai import storage

    storage.save_manifest(pd.DataFrame([{
        "image_id": "trace1", "path": "trace1.png", "source": "visa", "category": "pcb1",
        "split": config.SPLIT_TEST, "label": config.LABEL_DEFECT,
        "defect_type": config.DEFECT_TYPE_UNSPECIFIED, "width": 256, "height": 256,
    }]))
    labeling.record_label("trace1", label=config.LABEL_DEFECT, defect_type="scratch", verified=True)
    monitoring.log_inference([{
        "version": "v001", "image_id": "trace1", "score": 0.87, "threshold": 0.5,
        "decision": config.LABEL_DEFECT, "latency_ms": 8.0,
    }])
    from vision_ai import claude_review

    claude_review.save_reviews([claude_review.ReviewResult(
        image_id="trace1", defect=True, defect_type="scratch", severity=3,
        confidence="high", reason="선형 흠", model="claude-opus-5",
    )])

    trace = monitoring.trace_image("trace1")
    assert trace["manifest"]["source"] == "visa"
    assert trace["effective"]["defect_type"] == "scratch"
    assert len(trace["label_events"]) == 1
    assert len(trace["inferences"]) == 1
    assert len(trace["claude_reviews"]) == 1


def test_trace_unknown_image_is_empty(sandbox):
    trace = monitoring.trace_image("nope")
    assert trace["manifest"] is None
    assert trace["inferences"].empty


# --- 롤백 (A4) --------------------------------------------------------------

def _register_versions(sandbox, count: int) -> list[str]:
    versions = []
    for index in range(count):
        run_id = experiments.record_run(
            kind="anomaly", model="patch",
            metrics={"recall": 0.9 - index * 0.05, "threshold": 1.0},
            settings={"n_train": 10}, n_train=10,
        )
        versions.append(registry.register(run_id, note=f"v{index}").version)
    return versions


def test_no_rollback_target_before_any_promotion(sandbox):
    _register_versions(sandbox, 2)
    assert registry.rollback_target() is None
    assert registry.rollback() is None


def test_rollback_returns_to_previous_production(sandbox):
    first, second = _register_versions(sandbox, 2)
    registry.promote(first)
    registry.promote(second)

    assert str(registry.production()["version"]) == second
    assert registry.rollback() == first
    assert str(registry.production()["version"]) == first


def test_rollback_picks_most_recently_promoted_not_newest_registered(sandbox):
    """나중에 등록한 것을 먼저 승격했다가 되돌리는 경우가 있다 — 등록 순서로 고르면 틀린다."""
    first, second, third = _register_versions(sandbox, 3)
    registry.promote(third)      # 가장 나중에 등록된 것을 먼저 씀
    registry.promote(first)      # 그 다음 첫 번째로 바꿈
    registry.promote(second)     # 지금은 두 번째가 서비스 중

    # 직전에 쓰던 것은 first (third가 아니다)
    assert str(registry.rollback_target()["version"]) == first


def test_rollback_is_repeatable(sandbox):
    """되돌린 뒤 다시 되돌리면 그 전 버전으로 가야 한다."""
    first, second = _register_versions(sandbox, 2)
    registry.promote(first)
    registry.promote(second)
    assert registry.rollback() == first
    assert registry.rollback() == second


def test_candidate_never_becomes_a_rollback_target(sandbox):
    """한 번도 서비스한 적 없는 후보로 '되돌릴' 수는 없다."""
    first, second = _register_versions(sandbox, 2)
    registry.promote(first)
    registry.promote(second)
    target = registry.rollback_target()
    assert str(target["version"]) == first
    assert target["status"] == registry.STATUS_ARCHIVED
