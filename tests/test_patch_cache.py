"""이상탐지 격자 특징 캐시 테스트 (G14).

캐시의 실패는 두 방향이다. **안 쓰면** 매번 110초를 다시 쓰고, **잘못 쓰면** 바뀐
이미지의 옛 특징으로 학습해 놓고 아무도 눈치채지 못한다. 뒤쪽이 훨씬 나쁘므로
"언제 캐시를 믿지 않는가"를 집중적으로 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vision_ai import models, patch_cache


@pytest.fixture
def anomaly_config():
    return models.AnomalyConfig()


@pytest.fixture
def image_file(sandbox):
    """캐시가 파일의 수정 시각과 크기를 보므로 진짜 파일이 필요하다."""
    path = sandbox / "raw" / "a.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"first")
    return path


def _grid(value=1.0, shape=(3, 3, 4)):
    return np.full(shape, value, np.float32)


# --- 담고 꺼내기 -------------------------------------------------------------

def test_nothing_is_cached_at_first(sandbox, anomaly_config, image_file):
    cache = patch_cache.load(anomaly_config)
    assert cache.get("a", image_file) is None
    assert len(cache) == 0


def test_what_goes_in_comes_back(sandbox, anomaly_config, image_file):
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid(2.5))
    assert np.allclose(cache.get("a", image_file), 2.5)


def test_it_survives_a_restart(sandbox, anomaly_config, image_file):
    """세션에만 담아두면 새로고침 한 번에 110초가 다시 든다."""
    first = patch_cache.load(anomaly_config)
    first.put("a", image_file, _grid(2.5))
    patch_cache.save(first)

    again = patch_cache.load(anomaly_config)
    assert np.allclose(again.get("a", image_file), 2.5)


def test_grids_come_back_as_float32(sandbox, anomaly_config, image_file):
    """float16으로 저장하지만 계산은 float32로 한다 — 부르는 쪽이 신경 쓸 일이 아니다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid(2.5))
    patch_cache.save(cache)
    assert patch_cache.load(anomaly_config).get("a", image_file).dtype == np.float32


# --- 언제 캐시를 믿지 않는가 -------------------------------------------------

def test_a_changed_image_is_not_served_from_cache(sandbox, anomaly_config, image_file):
    """같은 경로에 다시 촬영한 사진을 덮어쓰는 일이 있다. 옛 특징으로 학습하면 끝이다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid(2.5))
    patch_cache.save(cache)

    image_file.write_bytes(b"second-and-longer")
    assert patch_cache.load(anomaly_config).get("a", image_file) is None


def test_a_missing_file_is_not_served_from_cache(sandbox, anomaly_config, image_file):
    """파일이 지워졌는데 특징만 돌려주면 없는 이미지를 학습에 넣게 된다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid())
    patch_cache.save(cache)

    image_file.unlink()
    assert patch_cache.load(anomaly_config).get("a", image_file) is None


def test_a_missing_file_is_not_cached_either(sandbox, anomaly_config):
    cache = patch_cache.load(anomaly_config)
    cache.put("gone", sandbox / "raw" / "없음.png", _grid())
    assert len(cache) == 0


def test_changing_the_backend_gives_a_separate_cache(sandbox, image_file):
    """고전 CV와 CNN은 격자 자체가 다른 것이라 섞으면 안 된다."""
    classic = models.AnomalyConfig(backend="classic")
    cnn = models.AnomalyConfig(backend="cnn")

    cache = patch_cache.load(classic)
    cache.put("a", image_file, _grid(2.5))
    patch_cache.save(cache)

    assert patch_cache.load(cnn).get("a", image_file) is None
    assert patch_cache.load(classic).get("a", image_file) is not None


def test_going_back_to_a_previous_setting_finds_its_cache(sandbox, image_file):
    """지문을 폴더 이름으로 쓰므로 백엔드를 바꿨다 되돌려도 앞서 뽑은 것이 남아 있다."""
    classic, cnn = models.AnomalyConfig(), models.AnomalyConfig(backend="cnn")

    first = patch_cache.load(classic)
    first.put("a", image_file, _grid(1.0))
    patch_cache.save(first)

    second = patch_cache.load(cnn)
    second.put("a", image_file, _grid(9.0))
    patch_cache.save(second)

    assert np.allclose(patch_cache.load(classic).get("a", image_file), 1.0)


def test_a_different_patch_size_is_a_different_cache(sandbox, image_file):
    small = models.AnomalyConfig(patch=16, stride=8)
    large = models.AnomalyConfig(patch=32, stride=8)
    assert patch_cache.schema_fingerprint(small) != patch_cache.schema_fingerprint(large)


def test_mixed_grid_shapes_are_refused(sandbox, anomaly_config, image_file):
    """모양이 섞이면 배열 하나에 담을 수 없다. 조용히 깨지느니 담지 않는다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid(shape=(3, 3, 4)))
    cache.put("b", image_file, _grid(shape=(5, 5, 4)))
    assert len(cache) == 1


# --- 늘어나는 데이터 ---------------------------------------------------------

def test_adding_an_image_keeps_the_old_ones(sandbox, anomaly_config):
    """운영 중에는 데이터가 조금씩 는다. 10장 늘 때마다 4,584장을 다시 뽑으면 안 된다."""
    paths = []
    for name in "abc":
        path = sandbox / "raw" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        paths.append(path)

    first = patch_cache.load(anomaly_config)
    for index, path in enumerate(paths[:2]):
        first.put(path.stem, path, _grid(index))
    patch_cache.save(first)

    second = patch_cache.load(anomaly_config)
    assert second.get("a", paths[0]) is not None      # 다시 뽑을 필요가 없다
    second.put("c", paths[2], _grid(2))
    patch_cache.save(second)

    third = patch_cache.load(anomaly_config)
    assert [np.mean(third.get(p.stem, p)) for p in paths] == [0.0, 1.0, 2.0]


def test_nothing_new_means_nothing_is_rewritten(sandbox, anomaly_config, image_file):
    """두 번째 실행부터는 읽기만 한다. 0.12GB를 쓸 이유가 없다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid())
    patch_cache.save(cache)

    data = patch_cache.cache_dir(cache.schema) / patch_cache.DATA_NAME
    before = data.stat().st_mtime_ns

    reopened = patch_cache.load(anomaly_config)
    assert reopened.get("a", image_file) is not None
    assert reopened.added == 0
    patch_cache.save(reopened)
    assert data.stat().st_mtime_ns == before


# --- 망가진 캐시 -------------------------------------------------------------

def test_a_truncated_cache_is_dropped_not_crashed(sandbox, anomaly_config, image_file):
    """저장 도중 죽으면 잘린 파일이 남는다. 캐시는 다시 만들면 되므로 조용히 버린다."""
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid())
    patch_cache.save(cache)

    (patch_cache.cache_dir(cache.schema) / patch_cache.DATA_NAME).write_bytes(b"broken")
    assert patch_cache.load(anomaly_config).get("a", image_file) is None


def test_a_broken_index_is_dropped_not_crashed(sandbox, anomaly_config, image_file):
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid())
    patch_cache.save(cache)

    (patch_cache.cache_dir(cache.schema) / patch_cache.INDEX_NAME).write_text("{oops")
    assert patch_cache.load(anomaly_config).get("a", image_file) is None


def test_clearing_removes_everything(sandbox, anomaly_config, image_file):
    cache = patch_cache.load(anomaly_config)
    cache.put("a", image_file, _grid())
    patch_cache.save(cache)

    patch_cache.clear()
    assert patch_cache.summary()["count"] == 0
    assert patch_cache.load(anomaly_config).get("a", image_file) is None


def test_summary_counts_every_setting(sandbox, image_file):
    for backend in ("classic", "cnn"):
        cache = patch_cache.load(models.AnomalyConfig(backend=backend))
        cache.put("a", image_file, _grid())
        patch_cache.save(cache)

    info = patch_cache.summary()
    assert info["variants"] == 2 and info["count"] == 2 and info["size_mb"] > 0


def test_the_cache_lives_inside_the_project(sandbox, anomaly_config):
    from vision_ai import config

    schema = patch_cache.schema_fingerprint(anomaly_config)
    assert config.cache_dir() in patch_cache.cache_dir(schema).parents


# --- 모델과 이어지는 부분 ----------------------------------------------------

def _rows(paths):
    return pd.DataFrame(
        [{"image_id": p.stem, "path_abs": str(p), "y": 0} for p in paths]
    )


@pytest.fixture
def photos(sandbox):
    import cv2

    made = []
    rng = np.random.default_rng(0)
    for name in "abcd":
        path = sandbox / "raw" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        made.append(path)
    return made


def test_second_run_reads_no_image_files(sandbox, photos, monkeypatch):
    """캐시가 맞으면 **이미지 파일을 열지도 않는다** — 읽기 12.3ms가 통째로 빠진다."""
    model = models.PatchAnomalyModel()
    rows = _rows(photos)

    cache = patch_cache.load(model.config)
    list(patch_cache.grids_for(rows, model, cache=cache))
    patch_cache.save(cache)

    from vision_ai import viz

    opened = []
    monkeypatch.setattr(viz, "load_rgb", lambda path: opened.append(path) or None)
    again = patch_cache.load(model.config)
    got = list(patch_cache.grids_for(rows, model, cache=again))

    assert len(got) == len(photos)
    assert opened == [], "캐시가 있으면 파일을 열 이유가 없다"


def test_cached_and_fresh_features_agree(sandbox, photos):
    """담는 것과 쓰는 것이 갈리면 캐시가 있을 때와 없을 때 결과가 달라진다."""
    model = models.PatchAnomalyModel()
    rows = _rows(photos)

    cache = patch_cache.load(model.config)
    fresh = {i: g for i, g in patch_cache.grids_for(rows, model, cache=cache)}
    patch_cache.save(cache)

    reloaded = patch_cache.load(model.config)
    for image_id, grid in patch_cache.grids_for(rows, model, cache=reloaded):
        assert np.allclose(grid, fresh[image_id], rtol=2e-3, atol=1e-3)


def test_unreadable_images_are_skipped_not_crashed(sandbox, photos):
    model = models.PatchAnomalyModel()
    broken = sandbox / "raw" / "broken.png"
    broken.write_bytes(b"not an image")
    rows = _rows([*photos, broken])

    got = list(patch_cache.grids_for(rows, model))
    assert len(got) == len(photos)


def test_progress_reaches_the_end(sandbox, photos):
    model = models.PatchAnomalyModel()
    seen: list[tuple[int, int]] = []
    list(patch_cache.grids_for(_rows(photos), model, progress=lambda d, t: seen.append((d, t))))
    assert seen[-1] == (len(photos), len(photos))


def test_a_model_can_be_fitted_from_cached_grids(sandbox, photos):
    """캐시가 있으면 이미지를 읽지 않고 학습까지 간다."""
    model = models.PatchAnomalyModel(models.AnomalyConfig(per_position=False))
    grids = [grid for _, grid in patch_cache.grids_for(_rows(photos), model)]
    model.fit_grids(grids)

    assert model.is_fitted
    assert model.score_grid_features(grids[0]).shape == grids[0].shape[:2]
