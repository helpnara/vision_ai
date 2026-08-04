"""특징 캐시 테스트.

캐시의 위험은 하나다 — **틀린 값을 빠르게 주는 것.** 느린 것보다 나쁘다. 그래서 이 테스트는
"빨라졌는가"보다 **"언제 캐시를 버리는가"**에 집중한다. 파일이 바뀌었을 때, 특징 정의가
바뀌었을 때, 캐시 파일이 깨졌을 때 모두 다시 계산해야 한다.
"""

from __future__ import annotations

import numpy as np
import pytest

from vision_ai import config, feature_cache, features, models


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@pytest.fixture
def image_file(tmp_path):
    import cv2

    path = tmp_path / "sample.png"
    rgb = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(path), rgb)
    return path


# --- 저장과 불러오기 --------------------------------------------------------

def test_empty_when_nothing_saved(sandbox):
    assert len(feature_cache.load()) == 0


def test_roundtrip_preserves_values(sandbox, image_file):
    cache = feature_cache.FeatureCache()
    vector = np.arange(len(features.FEATURE_NAMES), dtype=np.float32)
    cache.put("img-1", image_file, vector)
    feature_cache.save(cache)

    restored = feature_cache.load()
    assert np.array_equal(restored.get("img-1", image_file), vector)


def test_saving_an_empty_cache_writes_nothing(sandbox):
    feature_cache.save(feature_cache.FeatureCache())
    assert not feature_cache.cache_path().exists()


# --- 언제 다시 계산해야 하는가 ----------------------------------------------

def test_changed_file_invalidates_its_entry(sandbox, image_file):
    """같은 경로에 다시 촬영한 사진을 덮어쓰는 일이 있다. 옛 특징을 주면 안 된다."""
    import cv2

    cache = feature_cache.FeatureCache()
    cache.put("img-1", image_file, np.zeros(len(features.FEATURE_NAMES), dtype=np.float32))
    assert cache.get("img-1", image_file) is not None

    cv2.imwrite(str(image_file), np.full((96, 96, 3), 7, dtype=np.uint8))
    assert cache.get("img-1", image_file) is None


def test_missing_file_invalidates_its_entry(sandbox, image_file):
    cache = feature_cache.FeatureCache()
    cache.put("img-1", image_file, np.zeros(len(features.FEATURE_NAMES), dtype=np.float32))
    image_file.unlink()
    assert cache.get("img-1", image_file) is None


def test_unknown_id_is_a_miss(sandbox, image_file):
    assert feature_cache.FeatureCache().get("없는id", image_file) is None


def test_changed_feature_definition_throws_the_whole_cache_away(sandbox, image_file):
    """특징을 추가하면 예전 벡터는 길이부터 다르다. 섞이면 학습이 조용히 망가진다."""
    cache = feature_cache.FeatureCache()
    cache.put("img-1", image_file, np.zeros(len(features.FEATURE_NAMES), dtype=np.float32))
    feature_cache.save(cache)
    assert len(feature_cache.load()) == 1

    original = features.FEATURE_NAMES
    try:
        features.FEATURE_NAMES = original + ("새특징",)
        assert len(feature_cache.load()) == 0
    finally:
        features.FEATURE_NAMES = original


def test_a_corrupt_cache_file_is_ignored(sandbox):
    """저장 도중 죽으면 잘린 파일이 남는다. 캐시는 다시 만들면 되므로 조용히 버린다."""
    feature_cache.cache_path().write_bytes(b"not an npz file")
    assert len(feature_cache.load()) == 0


# --- build_dataset과의 연결 --------------------------------------------------

def _tiny_dataset(tmp_path, count=6):
    import cv2

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    paths, labels, ids = [], [], []
    for index in range(count):
        path = tmp_path / f"i{index}.png"
        cv2.imwrite(str(path), rng.integers(0, 255, (48, 48, 3), dtype=np.uint8))
        paths.append(str(path))
        labels.append(index % 2)
        ids.append(f"id-{index}")
    return paths, labels, ids


def test_second_run_reuses_everything(sandbox, tmp_path):
    paths, labels, ids = _tiny_dataset(tmp_path)
    first = models.build_dataset(paths, labels, ids)
    second = models.build_dataset(paths, labels, ids)

    assert first.extracted == len(paths) and first.reused == 0
    assert second.extracted == 0 and second.reused == len(paths)
    assert np.allclose(first.X, second.X), "캐시가 다른 값을 주면 캐시가 아니라 버그다"


def test_only_the_new_images_are_extracted(sandbox, tmp_path):
    """운영 중에는 데이터가 조금씩 늘어난다. 늘어난 만큼만 계산해야 의미가 있다."""
    paths, labels, ids = _tiny_dataset(tmp_path, count=4)
    models.build_dataset(paths, labels, ids)

    more_paths, more_labels, more_ids = _tiny_dataset(tmp_path / "more", count=3)
    combined = models.build_dataset(
        paths + more_paths, labels + more_labels, ids + [f"new-{i}" for i in range(3)]
    )
    assert combined.extracted == 3
    assert combined.reused == 4


def test_cache_can_be_turned_off(sandbox, tmp_path):
    paths, labels, ids = _tiny_dataset(tmp_path)
    models.build_dataset(paths, labels, ids)
    plain = models.build_dataset(paths, labels, ids, use_cache=False)
    assert plain.reused == 0 and plain.extracted == len(paths)


def test_unreadable_images_are_still_reported(sandbox, tmp_path):
    paths, labels, ids = _tiny_dataset(tmp_path, count=3)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    dataset = models.build_dataset(
        paths + [str(broken)], labels + [0], ids + ["broken"]
    )
    assert dataset.failed == [str(broken)]
    assert len(dataset) == 3


def test_clear_removes_the_cache(sandbox, tmp_path):
    paths, labels, ids = _tiny_dataset(tmp_path)
    models.build_dataset(paths, labels, ids)
    assert feature_cache.summary()["count"] == len(paths)

    feature_cache.clear()
    assert feature_cache.summary()["count"] == 0
    assert models.build_dataset(paths, labels, ids).extracted == len(paths)


def test_more_files_than_the_directory_holds_is_fine(sandbox, tmp_path):
    """캐시가 없어도 앱은 돌아가야 한다 — 디렉터리가 없을 때 예외가 나면 안 된다."""
    import shutil

    paths, labels, ids = _tiny_dataset(tmp_path)
    shutil.rmtree(sandbox)
    dataset = models.build_dataset(paths, labels, ids)
    assert len(dataset) == len(paths)
