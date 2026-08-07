"""빠른 시작 테스트.

목적이 "처음 연 사람이 운영 화면까지 한 번에 간다"는 것이므로, 한 번 실행한 뒤
**4단계 운영 시나리오를 곧바로 누를 수 있는 상태**가 되었는지를 확인한다.
"""

from __future__ import annotations

import pytest

from vision_ai import config, guide, labeling, quickstart, registry, storage




@pytest.fixture
def demo(sandbox):
    return quickstart.run(n_normal=12, n_defect=5, seed=1)


def test_run_promotes_a_version(demo):
    assert demo.ok
    assert demo.version == "v001"
    production = registry.production()
    assert production is not None
    assert str(production["version"]) == demo.version


def test_run_leaves_pipeline_ready_for_operations(demo):
    """빠른 시작의 목적은 4단계 시연을 곧바로 누를 수 있게 하는 것이다."""
    step = guide.next_step()
    assert step is not None
    assert step.key == "operate"     # 앞의 모든 걸음이 끝나 있어야 한다


def test_run_registers_images_and_splits(demo):
    assert demo.n_images > 0
    resolved = labeling.resolve()
    assigned = (resolved["split"].astype(str) != config.SPLIT_NONE).sum()
    assert assigned == len(resolved)


def test_run_trains_on_normal_images_only(demo):
    """이상탐지는 정상만으로 학습한다 — 결함이 섞이면 정상 분포가 오염된다."""
    resolved = labeling.resolve()
    train_normal = resolved[
        (resolved["split"].astype(str) == config.SPLIT_TRAIN)
        & (resolved["label"].astype(str) == config.LABEL_NORMAL)
    ]
    assert demo.n_train == len(train_normal)


def test_registered_version_has_drift_baseline(demo):
    """기준선이 없으면 4단계 드리프트 감시가 아무것도 못 한다."""
    assert registry.load_baseline(demo.version) is not None


def test_registered_version_can_be_used_for_inference(demo):
    from vision_ai import serving

    assert demo.version in serving.selectable_versions()


def test_progress_reports_every_step(sandbox):
    seen = []
    quickstart.run(n_normal=12, n_defect=5, seed=2,
                   progress=lambda i, n, name: seen.append((i, n, name)))
    assert [name for _, _, name in seen] == list(quickstart.STEPS)
    assert all(total == len(quickstart.STEPS) for _, total, _ in seen)


def test_too_few_images_warns_instead_of_raising(sandbox):
    """시연 중에 화면이 멈추는 것보다 무엇이 부족한지 보이는 편이 낫다."""
    result = quickstart.run(n_normal=1, n_defect=1, seed=3)
    assert not result.ok
    assert any("정상 이미지가" in w for w in result.warnings)


def test_running_twice_does_not_duplicate_images(sandbox):
    """같은 시드로 다시 돌리면 sha1 중복 제거가 걸려 이미지가 늘지 않아야 한다."""
    first = quickstart.run(n_normal=12, n_defect=5, seed=7)
    before = len(storage.load_manifest())
    quickstart.run(n_normal=12, n_defect=5, seed=7)
    assert len(storage.load_manifest()) == before
    assert first.ok


def test_second_run_creates_a_new_version(sandbox):
    quickstart.run(n_normal=12, n_defect=5, seed=8)
    second = quickstart.run(n_normal=12, n_defect=5, seed=8)
    assert second.version == "v002"
    assert str(registry.production()["version"]) == "v002"
