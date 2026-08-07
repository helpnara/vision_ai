"""사전학습 CNN 특징 테스트.

모델 파일(약 45MB)은 저장소에 없고 내려받아야 하므로, 파일이 있을 때만 도는 시험과
**없을 때 어떻게 실패하는지**를 나눠서 확인한다. 선택 기능이 없다고 앱이 멈추면 안 된다.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_ai import cnn_features, config, models




needs_model = pytest.mark.skipif(
    not cnn_features.available(), reason="사전학습 모델 파일이 없습니다 (선택 기능)"
)


# --- 모델이 없을 때 -------------------------------------------------------

def test_not_available_in_clean_environment(sandbox):
    assert not cnn_features.available()


def test_describe_tells_user_what_is_missing(sandbox):
    text = cnn_features.describe()
    assert "없습니다" in text
    assert str(cnn_features.MODEL_SIZE_MB) in text


def test_load_without_file_raises_clear_error(sandbox):
    with pytest.raises(FileNotFoundError) as excinfo:
        cnn_features.load()
    assert "내려받" in str(excinfo.value)


def test_truncated_file_is_not_treated_as_available(sandbox):
    """받다 만 파일을 정상으로 보면 로드 단계에서 알 수 없는 오류가 난다."""
    path = cnn_features.model_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * 1024)
    assert not cnn_features.available()


# --- 설정 연결 -------------------------------------------------------------

def test_default_backend_is_classic():
    """기본값은 의존성 없이 도는 쪽이어야 한다. CNN은 모델 파일이 있어야 한다."""
    assert models.AnomalyConfig().backend == models.BACKEND_CLASSIC


def test_backend_survives_config_roundtrip():
    payload = models.AnomalyConfig(backend=models.BACKEND_CNN).to_dict()
    assert payload["backend"] == models.BACKEND_CNN


def test_every_backend_has_a_label():
    for name in models.BACKENDS:
        assert models.BACKEND_LABELS.get(name)


def test_cnn_backend_without_model_fails_at_fit(sandbox):
    """모델이 없으면 학습 시점에 분명히 실패해야 한다 — 조용히 고전 특징으로 돌면 안 된다."""
    images = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
    model = models.PatchAnomalyModel(models.AnomalyConfig(backend=models.BACKEND_CNN))
    with pytest.raises(FileNotFoundError):
        model.fit(images)


# --- 모델이 있을 때 -------------------------------------------------------

@needs_model
def test_extractor_reports_expected_shape():
    extractor = cnn_features.load()
    assert extractor.grid == cnn_features.EXPECTED_GRID
    assert extractor.channels == cnn_features.EXPECTED_CHANNELS
    assert extractor.dim == cnn_features.REDUCED_DIM


@needs_model
def test_patch_grid_matches_declared_shape():
    extractor = cnn_features.load()
    image = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype(np.uint8)
    grid = extractor.patch_grid(image)
    assert grid.shape == (extractor.n_positions, extractor.dim)


@needs_model
def test_same_image_gives_same_features():
    """차원 축소에 시드를 고정했으므로 같은 입력은 같은 특징을 내야 한다."""
    extractor = cnn_features.load()
    image = (np.random.default_rng(1).random((256, 256, 3)) * 255).astype(np.uint8)
    assert np.allclose(extractor.patch_grid(image), extractor.patch_grid(image))


@needs_model
def test_different_images_give_different_features():
    extractor = cnn_features.load()
    rng = np.random.default_rng(2)
    first = extractor.patch_grid((rng.random((256, 256, 3)) * 255).astype(np.uint8))
    second = extractor.patch_grid((rng.random((256, 256, 3)) * 255).astype(np.uint8))
    assert not np.allclose(first, second)
