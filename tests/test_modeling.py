"""3단계(모델 개발 · 평가) 코어 로직 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_ai import (
    claude_review,
    config,
    evaluate,
    experiments,
    features,
    ingest,
    labeling,
    models,
    storage,
    viz,
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


@pytest.fixture
def pcb_dataset(sandbox):
    """PCB 합성 데이터를 만들고 수집·분할까지 마친 상태를 준비한다."""
    out = ingest.generate_synthetic(
        categories=["pcb_green"], n_normal=40, n_defect=16, size=128, seed=5, layout="visa"
    )
    ingest.ingest_folder(out, source="synthetic", layout="visa")
    resolved = labeling.resolve()
    labeling.save_splits(labeling.assign_splits(resolved, train=0.6, val=0.2, test=0.2, seed=3))
    resolved = labeling.resolve()
    resolved = resolved.assign(
        path_abs=[str(storage.resolve_path(p)) for p in resolved["path"]],
        y=(resolved["label"].astype(str) == config.LABEL_DEFECT).astype(int),
    )
    return resolved


# --- 특징 추출 --------------------------------------------------------------

def test_preprocess_resizes_and_expands_gray():
    gray = np.full((100, 60), 120, dtype=np.uint8)
    out = features.preprocess(gray, size=64)
    assert out.shape == (64, 64, 3)


def test_preprocess_is_noop_at_target_size():
    rgb = np.full((features.IMAGE_SIZE, features.IMAGE_SIZE, 3), 90, dtype=np.uint8)
    assert features.preprocess(rgb) is rgb


def test_image_features_length_matches_names():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    vector = features.image_features(rgb)
    assert vector.shape == (len(features.FEATURE_NAMES),)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()


def test_image_features_are_deterministic():
    rgb = np.full((96, 96, 3), 100, dtype=np.uint8)
    rgb[30:50, 30:50] = 40
    assert np.array_equal(features.image_features(rgb), features.image_features(rgb))


def test_image_features_distinguish_defect_from_flat():
    flat = np.full((128, 128, 3), 150, dtype=np.uint8)
    scratched = flat.copy()
    scratched[60:63, 20:110] = 40
    assert not np.allclose(features.image_features(flat), features.image_features(scratched))


def test_patch_features_shape():
    rgb = np.full((features.IMAGE_SIZE, features.IMAGE_SIZE, 3), 120, dtype=np.uint8)
    grid = features.patch_features(rgb)
    rows, cols = features.patch_grid_shape()
    assert grid.shape == (rows, cols, features.PATCH_FEATURE_DIM)


def test_patch_grid_and_centers_agree():
    rows, cols = features.patch_grid_shape()
    ys, xs = features.patch_centers()
    assert len(ys) == rows and len(xs) == cols


def test_upsample_grid_returns_image_size():
    grid = np.random.default_rng(1).random((8, 8)).astype(np.float32)
    heatmap = features.upsample_grid(grid, size=64)
    assert heatmap.shape == (64, 64)


# --- 베이스라인 -------------------------------------------------------------

@pytest.mark.parametrize("kind", models.BASELINE_KINDS)
def test_baseline_fits_and_scores(kind):
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(0, 1, (30, 6)), rng.normal(3, 1, (20, 6))])
    y = np.array([0] * 30 + [1] * 20)

    model = models.BaselineModel(models.BaselineConfig(kind=kind)).fit(X, y)
    scores = model.score(X)
    assert scores.shape == (50,)
    assert ((scores >= 0) & (scores <= 1)).all()
    # 분리 가능한 데이터이므로 결함 쪽 평균 점수가 더 높아야 한다
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_baseline_rejects_single_class():
    X = np.zeros((10, 4))
    with pytest.raises(ValueError, match="두 클래스"):
        models.BaselineModel().fit(X, np.zeros(10, dtype=int))


def test_baseline_rejects_unknown_kind():
    with pytest.raises(ValueError, match="지원하지 않는"):
        models.BaselineModel(models.BaselineConfig(kind="nope")).fit(
            np.zeros((4, 2)), np.array([0, 0, 1, 1])
        )


def test_baseline_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        models.BaselineModel().score(np.zeros((2, 3)))


def test_baseline_roundtrip(sandbox):
    rng = np.random.default_rng(2)
    X = np.vstack([rng.normal(0, 1, (20, 5)), rng.normal(4, 1, (20, 5))])
    y = np.array([0] * 20 + [1] * 20)
    model = models.BaselineModel(models.BaselineConfig(kind="logreg")).fit(X, y)

    path = config.MODEL_DIR / "bl.joblib"
    model.save(path)
    loaded = models.BaselineModel.load(path)
    assert loaded.config.kind == "logreg"
    assert np.allclose(model.score(X), loaded.score(X))


def test_gradient_boosting_balances_via_sample_weight():
    """GradientBoosting은 class_weight를 지원하지 않아 sample_weight로 보정해야 한다."""
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(0, 1, (60, 4)), rng.normal(2.5, 1, (6, 4))])
    y = np.array([0] * 60 + [1] * 6)
    model = models.BaselineModel(
        models.BaselineConfig(kind="gradient_boosting", balanced=True)
    ).fit(X, y)
    scores = model.score(X)
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_build_dataset_skips_unreadable(sandbox, tmp_path):
    import cv2

    good = tmp_path / "good.png"
    cv2.imwrite(str(good), np.full((64, 64, 3), 120, dtype=np.uint8))
    missing = tmp_path / "missing.png"

    dataset = models.build_dataset([str(good), str(missing)], [0, 1], ["a", "b"])
    assert len(dataset) == 1
    assert dataset.image_ids == ["a"]
    assert dataset.failed == [str(missing)]


# --- 이상탐지 ---------------------------------------------------------------

def _normal_images(count: int, seed: int = 0, size: int = 64) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [
        np.clip(rng.normal(140, 6, (size, size, 3)), 0, 255).astype(np.uint8)
        for _ in range(count)
    ]


def test_anomaly_scores_defect_higher_than_normal():
    config_ = models.AnomalyConfig(per_position=False, image_score="max", size=64)
    model = models.PatchAnomalyModel(config_).fit(_normal_images(30))

    normal = _normal_images(1, seed=99)[0]
    defective = normal.copy()
    defective[20:40, 20:26] = 20  # 국소적인 어두운 흠

    assert model.image_score(defective) > model.image_score(normal)


def test_anomaly_score_map_shape_and_peak():
    model = models.PatchAnomalyModel(
        models.AnomalyConfig(per_position=False, size=features.IMAGE_SIZE)
    ).fit(_normal_images(30, size=features.IMAGE_SIZE))

    image = _normal_images(1, seed=5, size=features.IMAGE_SIZE)[0]
    image[120:150, 120:150] = 15
    heatmap = model.score_map(image)
    assert heatmap.shape == (features.IMAGE_SIZE, features.IMAGE_SIZE)

    peak_y, peak_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    assert 90 <= peak_y <= 180 and 90 <= peak_x <= 180, (peak_y, peak_x)


def test_anomaly_per_position_requires_enough_samples():
    with pytest.raises(ValueError, match="특징 차원"):
        models.PatchAnomalyModel(
            models.AnomalyConfig(per_position=True, size=64)
        ).fit(_normal_images(5, size=64))


def test_anomaly_rejects_empty_training_set():
    with pytest.raises(ValueError, match="정상 이미지가 없습니다"):
        models.PatchAnomalyModel().fit([])


def test_anomaly_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        models.PatchAnomalyModel().score_grid(np.zeros((64, 64, 3), dtype=np.uint8))


@pytest.mark.parametrize("mode", models.IMAGE_SCORE_MODES)
def test_anomaly_aggregation_modes(mode):
    model = models.PatchAnomalyModel(models.AnomalyConfig(per_position=False, image_score=mode))
    grid = np.arange(100, dtype=float).reshape(10, 10)
    value = model.aggregate(grid)
    assert 0.0 <= value <= 99.0


def test_anomaly_rejects_unknown_aggregation():
    model = models.PatchAnomalyModel(models.AnomalyConfig(image_score="bogus"))
    with pytest.raises(ValueError, match="집계 방식"):
        model.aggregate(np.zeros((4, 4)))


def test_anomaly_roundtrip(sandbox):
    model = models.PatchAnomalyModel(
        models.AnomalyConfig(per_position=False, size=64)
    ).fit(_normal_images(30, size=64))

    path = config.MODEL_DIR / "anomaly.npz"
    model.save(path)
    loaded = models.PatchAnomalyModel.load(path)
    assert loaded.n_train == model.n_train
    assert loaded.config.per_position is False

    image = _normal_images(1, seed=42, size=64)[0]
    assert np.isclose(model.image_score(image), loaded.image_score(image))


# --- 평가 -------------------------------------------------------------------

def test_metrics_at_threshold_counts():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    m = evaluate.metrics_at_threshold(y, scores, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5
    assert m["miss_rate"] == 0.5


def test_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="길이"):
        evaluate.metrics_at_threshold([0, 1], [0.5], 0.5)


def test_metrics_handle_all_negative_predictions():
    m = evaluate.metrics_at_threshold([0, 1], [0.0, 0.1], 0.9)
    assert m["tp"] == 0 and m["precision"] == 0.0 and m["recall"] == 0.0


def test_auroc_perfect_and_single_class():
    assert evaluate.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert np.isnan(evaluate.auroc([1, 1], [0.3, 0.7]))
    assert np.isnan(evaluate.average_precision([0, 0], [0.3, 0.7]))


def test_threshold_for_target_recall_prefers_fewest_false_positives():
    y = np.array([0] * 8 + [1] * 4)
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0])
    threshold = evaluate.threshold_for_target_recall(y, scores, 1.0)
    assert threshold is not None
    achieved = evaluate.metrics_at_threshold(y, scores, threshold)
    assert achieved["recall"] == 1.0
    # 재현율을 만족하는 임계값 중 오탐이 최소여야 한다
    assert achieved["fp"] <= evaluate.metrics_at_threshold(y, scores, 0.0)["fp"]


def test_threshold_for_target_recall_returns_none_when_unreachable():
    """달성 불가한 목표나 빈 입력에서는 None을 돌려줘야 한다 (호출부가 대비해야 하는 경로)."""
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    assert evaluate.threshold_for_target_recall(y, scores, 1.01) is None
    assert evaluate.threshold_for_target_recall(np.array([]), np.array([]), 0.9) is None


def test_threshold_for_full_recall_is_always_reachable():
    """최저 점수를 임계값으로 쓰면 전부 결함으로 판정되므로 재현율 1.0은 항상 달성된다.

    결함 점수가 정상보다 낮게 나오는 최악의 경우에도 성립한다 — 대신 오탐이 늘어난다.
    """
    y = np.array([0, 0, 1])
    scores = np.array([0.9, 0.8, 0.1])
    threshold = evaluate.threshold_for_target_recall(y, scores, 1.0)
    assert threshold is not None
    achieved = evaluate.metrics_at_threshold(y, scores, threshold)
    assert achieved["recall"] == 1.0
    assert achieved["fp"] == 2  # 대가로 정상 2건이 오탐이 된다


def test_threshold_sweep_is_monotonic_in_recall():
    rng = np.random.default_rng(4)
    y = np.array([0] * 30 + [1] * 20)
    scores = np.concatenate([rng.normal(0.3, 0.1, 30), rng.normal(0.7, 0.1, 20)])
    sweep = evaluate.threshold_sweep(y, scores)
    assert not sweep.empty
    ordered = sweep.sort_values("threshold")
    assert (ordered["recall"].diff().dropna() <= 1e-9).all(), "임계값이 오르면 재현율은 낮아져야 한다"


def test_error_frame_labels_outcomes():
    frame = evaluate.error_frame(["a", "b", "c", "d"], [1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
    assert frame.set_index("image_id")["outcome"].to_dict() == {
        "a": "TP", "b": "FN", "c": "FP", "d": "TN"
    }


def test_summarize_includes_threshold_free_metrics():
    result = evaluate.summarize([0, 1], [0.2, 0.8], 0.5)
    assert "auroc" in result and "average_precision" in result


def test_localization_metrics_hit_and_miss():
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[30:35, 30:35] = 10.0
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:38, 28:38] = 255
    hit = evaluate.localization_metrics(heatmap, mask)
    assert hit["hit"] is True
    assert hit["pixel_auroc"] > 0.5

    far = np.zeros((64, 64), dtype=np.uint8)
    far[0:6, 0:6] = 255
    assert evaluate.localization_metrics(heatmap, far)["hit"] is False


def test_localization_metrics_empty_mask():
    result = evaluate.localization_metrics(np.ones((32, 32), np.float32), np.zeros((32, 32), np.uint8))
    assert result["hit"] is False
    assert np.isnan(result["iou"])


def test_localization_metrics_resizes_mask():
    heatmap = np.zeros((64, 64), dtype=np.float32)
    heatmap[40:50, 40:50] = 5.0
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[20:25, 20:25] = 255
    assert evaluate.localization_metrics(heatmap, mask)["hit"] is True


# --- 실험 기록 --------------------------------------------------------------

def test_record_and_load_runs(sandbox):
    metrics = {"threshold": 0.5, "recall": 0.9, "precision": 0.8, "f1": 0.85,
               "auroc": 0.95, "average_precision": 0.9, "tp": 9, "fp": 2, "fn": 1, "tn": 20}
    run_id = experiments.record_run(
        kind="baseline", model="logreg", split="test", metrics=metrics,
        settings={"kind": "logreg"}, n_train=50, n_eval=32, note="첫 실행",
    )
    runs = experiments.load_runs()
    assert len(runs) == 1
    assert runs.iloc[0]["run_id"] == run_id
    assert runs.iloc[0]["recall"] == 0.9

    detail = experiments.load_run_detail(run_id)
    assert detail["settings"]["kind"] == "logreg"
    assert detail["metrics"]["auroc"] == 0.95


def test_runs_are_returned_newest_first(sandbox):
    base = {"threshold": 0.5, "recall": 0.5, "precision": 0.5, "f1": 0.5,
            "auroc": 0.5, "average_precision": 0.5, "tp": 1, "fp": 1, "fn": 1, "tn": 1}
    first = experiments.record_run(kind="a", metrics=base, settings={})
    second = experiments.record_run(kind="b", metrics=base, settings={})
    assert experiments.load_runs()["run_id"].tolist()[0] == second
    assert first in experiments.load_runs()["run_id"].tolist()


def test_load_runs_empty(sandbox):
    assert experiments.load_runs().empty
    assert experiments.load_run_detail("nope") is None
    assert experiments.comparison_frame().empty


def test_run_ids_are_unique(sandbox):
    ids = {experiments.new_run_id() for _ in range(50)}
    assert len(ids) == 50


# --- Claude 2차 판정 --------------------------------------------------------

def test_response_schema_matches_project_defect_types():
    allowed = claude_review.RESPONSE_SCHEMA["properties"]["defect_type"]["enum"]
    assert set(config.DEFECT_TYPES) <= set(allowed)
    assert config.DEFECT_TYPE_NONE in allowed
    # 구조화 출력은 숫자 제약을 지원하지 않으므로 enum으로 범위를 표현해야 한다
    assert claude_review.RESPONSE_SCHEMA["properties"]["severity"]["enum"] == [1, 2, 3, 4, 5]
    assert claude_review.RESPONSE_SCHEMA["additionalProperties"] is False


def test_prepare_image_crops_to_roi():
    rgb = np.full((200, 200, 3), 100, dtype=np.uint8)
    full = claude_review.prepare_image(rgb)
    cropped = claude_review.prepare_image(rgb, (80, 80, 20, 20), margin=5)
    assert len(cropped) < len(full)


def test_prepare_image_downscales_large_input():
    import cv2

    rgb = np.full((3000, 3000, 3), 120, dtype=np.uint8)
    data = claude_review.prepare_image(rgb, max_side=512)
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == 512


def test_review_without_sdk_returns_unavailable(sandbox, monkeypatch):
    monkeypatch.setattr(claude_review, "sdk_installed", lambda: False)
    result = claude_review.review_image(
        np.full((64, 64, 3), 100, dtype=np.uint8), image_id="x"
    )
    assert result.status == "unavailable"
    assert result.defect is False


def test_review_handles_client_failure(sandbox, monkeypatch):
    monkeypatch.setattr(claude_review, "sdk_installed", lambda: True)

    def boom():
        raise RuntimeError("자격 증명 없음")

    monkeypatch.setattr(claude_review, "_client", boom)
    result = claude_review.review_image(np.zeros((64, 64, 3), np.uint8), image_id="y")
    assert result.status == "unavailable"
    assert "자격 증명" in result.reason


def test_review_cache_roundtrip(sandbox):
    ok = claude_review.ReviewResult(
        image_id="a", defect=True, defect_type="scratch", severity=3,
        confidence="high", reason="선형 흠", model="claude-opus-5",
    )
    failed = claude_review.ReviewResult(
        image_id="b", defect=False, defect_type="none", severity=1,
        confidence="low", reason="오류", model="claude-opus-5", status="error",
    )
    assert claude_review.save_reviews([ok, failed]) == 2
    assert claude_review.reviewed_ids() == {"a"}  # 실패는 재시도 대상으로 남는다

    # 같은 image_id를 다시 저장하면 최신 값으로 대체된다
    claude_review.save_reviews([
        claude_review.ReviewResult(
            image_id="a", defect=False, defect_type="none", severity=1,
            confidence="low", reason="재판정", model="claude-opus-5",
        )
    ])
    reviews = claude_review.load_reviews()
    assert len(reviews[reviews["image_id"] == "a"]) == 1
    assert reviews.set_index("image_id").loc["a", "reason"] == "재판정"


def test_save_reviews_empty_is_noop(sandbox):
    assert claude_review.save_reviews([]) == 0
    assert claude_review.load_reviews().empty


def test_select_uncertain_prefers_scores_near_threshold():
    errors = pd.DataFrame(
        {
            "image_id": ["far_low", "near", "far_high"],
            "label": [0, 1, 1],
            "score": [0.0, 0.5, 1.0],
            "predicted": [0, 1, 1],
            "outcome": ["TN", "TP", "TP"],
        }
    )
    picked = claude_review.select_uncertain(errors, threshold=0.5, limit=1)
    assert picked.iloc[0]["image_id"] == "near"


def test_select_uncertain_on_empty_frame():
    assert claude_review.select_uncertain(pd.DataFrame(), threshold=0.5).empty


def test_estimate_cost_scales_linearly():
    assert claude_review.estimate_cost(0) == 0
    assert claude_review.estimate_cost(20) == pytest.approx(claude_review.estimate_cost(10) * 2)


def test_default_model_is_current_opus():
    assert claude_review.DEFAULT_MODEL == "claude-opus-5"


# --- 통합: PCB 합성 데이터로 전 과정 -----------------------------------------

def test_pcb_surface_style_registered():
    assert "pcb_green" in ingest.SURFACE_STYLES
    assert ingest.SURFACE_STYLES["pcb_green"]["grain"] == "trace"


def test_pcb_surface_passes_quality_check():
    from vision_ai import quality

    rng = np.random.default_rng(3)
    surface = ingest._make_surface(rng, ingest.SURFACE_STYLES["pcb_green"], 256)
    assert quality.assess_array(surface).flags == ()


def test_end_to_end_baseline_on_pcb(pcb_dataset):
    """수집 → 라벨 → 분할 → 특징 → 학습 → 평가가 이어지는지 확인한다."""
    labeled = pcb_dataset[pcb_dataset["label"].isin([config.LABEL_NORMAL, config.LABEL_DEFECT])]
    dataset = models.build_dataset(
        labeled["path_abs"].tolist(), labeled["y"].tolist(), labeled["image_id"].tolist()
    )
    assert len(dataset) == len(labeled)

    lookup = labeled.set_index(labeled["image_id"].astype(str))["split"].to_dict()
    assigned = np.array([lookup[i] for i in dataset.image_ids])
    train, test = assigned == config.SPLIT_TRAIN, assigned == config.SPLIT_TEST
    assert train.sum() > 0 and test.sum() > 0

    model = models.BaselineModel(models.BaselineConfig(kind="logreg")).fit(
        dataset.X[train], dataset.y[train]
    )
    scores = model.score(dataset.X[test])
    auroc = evaluate.auroc(dataset.y[test], scores)
    assert auroc > 0.7, f"합성 PCB 결함은 분리 가능해야 한다 (AUROC={auroc})"


def test_end_to_end_anomaly_localizes_on_pcb(pcb_dataset):
    """정상만 학습한 이상탐지가 결함 위치를 마스크 안에서 찾아내는지 확인한다."""
    import cv2

    labeled = pcb_dataset
    train_normal = labeled[
        (labeled["split"] == config.SPLIT_TRAIN) & (labeled["y"] == 0)
    ]
    images = [viz.load_rgb(p) for p in train_normal["path_abs"]]
    images = [i for i in images if i is not None]

    model = models.PatchAnomalyModel(
        models.AnomalyConfig(per_position=False, image_score="max")
    ).fit(images)

    defects = labeled[(labeled["split"] == config.SPLIT_TEST) & (labeled["y"] == 1)]
    assert not defects.empty

    hits = []
    for _, row in defects.iterrows():
        mask_path = labeling.find_mask_path(storage.resolve_path(row["path"]))
        image = viz.load_rgb(row["path_abs"])
        if mask_path is None or image is None:
            continue
        heatmap = model.score_map(features.preprocess(image))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        hits.append(evaluate.localization_metrics(heatmap, mask)["hit"])

    assert hits, "마스크가 있는 결함 이미지가 있어야 한다"
    assert np.mean(hits) >= 0.5, f"최고점 명중률이 너무 낮다: {np.mean(hits)}"
