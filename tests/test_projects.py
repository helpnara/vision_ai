"""프로젝트(작업공간) 테스트.

프로젝트를 나누는 이유는 하나다 — **현장·라인마다 데이터와 라벨을 섞지 않는 것.** 그래서
"목록이 저장되는가"보다 **"정말로 격리되는가"**를 먼저 확인한다. 한 프로젝트에서 넣은
이미지가 다른 프로젝트에서 보이면 나눈 의미가 없다.

이관(migration)은 사용자의 실데이터를 옮기는 일이라 되돌릴 수 없는 실수가 나오면 안 된다.
**아무것도 덮어쓰지 않는지**, 그리고 **빠뜨리지 않는지**를 양쪽 다 확인한다.
"""

from __future__ import annotations

import pytest

from vision_ai import config, projects, storage


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """프로젝트 격리를 보려면 sandbox보다 낮은 층(컨테이너)에서 시작해야 한다."""
    monkeypatch.setattr(config, "DATA_HOME", tmp_path / "data")
    monkeypatch.setattr(config, "ARTIFACT_HOME", tmp_path / "artifacts")
    monkeypatch.setattr(config, "_active_project", config.DEFAULT_PROJECT)
    return tmp_path


# --- 목록과 전환 -----------------------------------------------------------

def test_starts_with_one_default_project(homes):
    registry = projects.load()
    assert [p.slug for p in registry.projects] == [config.DEFAULT_PROJECT]
    assert registry.active == config.DEFAULT_PROJECT


def test_created_project_becomes_active(homes):
    made = projects.create("2공장 도장라인")
    assert projects.active().slug == made.slug
    assert config.active_project() == made.slug


def test_switching_changes_where_files_go(homes):
    first = projects.create("가라인")
    second = projects.create("나라인")
    projects.use(first.slug)
    assert config.manifest_path().parent.name == first.slug
    projects.use(second.slug)
    assert config.manifest_path().parent.name == second.slug


def test_unknown_project_cannot_be_activated(homes):
    with pytest.raises(ValueError):
        projects.use("없는프로젝트")


def test_active_project_survives_a_restart(homes):
    made = projects.create("야간조")
    config.use_project(config.DEFAULT_PROJECT)   # 프로세스가 새로 뜬 상황을 흉내
    projects.bootstrap()
    assert config.active_project() == made.slug


# --- 이름과 slug -----------------------------------------------------------

def test_korean_names_get_a_usable_folder_name(homes):
    """한글 폴더명은 OS·인코딩에 따라 말썽이 난다. slug는 영숫자만 남긴다."""
    made = projects.create("도장 2라인")
    assert made.slug.isascii() and made.slug
    assert made.name == "도장 2라인"


def test_same_name_twice_does_not_share_a_folder(homes):
    """이름이 같다고 데이터를 섞으면 프로젝트를 나눈 의미가 없다."""
    first = projects.create("라인")
    second = projects.create("라인")
    assert first.slug != second.slug


def test_renaming_keeps_the_folder(homes):
    """폴더까지 바꾸면 쌓인 데이터를 통째로 옮겨야 한다. 이름만 바꾼다."""
    made = projects.create("옛이름")
    renamed = projects.rename(made.slug, "새이름")
    assert renamed.slug == made.slug and renamed.name == "새이름"


def test_empty_name_is_refused(homes):
    with pytest.raises(ValueError):
        projects.create("   ")


# --- 격리 -----------------------------------------------------------------

def test_data_does_not_leak_between_projects(homes, tmp_path):
    """이 테스트가 프로젝트 기능의 존재 이유다."""
    import cv2
    import numpy as np

    image = tmp_path / "a.png"
    cv2.imwrite(str(image), np.full((32, 32, 3), 120, dtype=np.uint8))

    first = projects.create("가공장")
    storage.append_records([{
        "image_id": "img1", "path": str(image), "source": "test",
        "category": "c", "sha1": "abc",
    }])
    assert len(storage.load_manifest()) == 1

    second = projects.create("나공장")
    assert len(storage.load_manifest()) == 0, "다른 프로젝트의 이미지가 보이면 안 된다"

    projects.use(first.slug)
    assert len(storage.load_manifest()) == 1, "돌아오면 원래 데이터가 있어야 한다"
    assert second.slug != first.slug


def test_removing_a_project_keeps_its_files(homes):
    """실수로 지웠을 때 라벨링에 들인 시간이 통째로 사라지면 안 된다."""
    made = projects.create("지울것")
    folder = config.data_root()
    assert folder.exists()

    projects.remove(made.slug)
    assert folder.exists(), "목록에서만 빼고 파일은 남겨야 한다"
    assert made.slug not in {p.slug for p in projects.load().projects}


def test_the_last_project_cannot_be_removed(homes):
    with pytest.raises(ValueError):
        projects.remove(config.DEFAULT_PROJECT)


# --- 예전 배치에서 옮겨오기 -------------------------------------------------

def _legacy_layout(homes):
    data, artifacts = config.DATA_HOME, config.ARTIFACT_HOME
    (data / "raw" / "synthetic").mkdir(parents=True)
    (data / "raw" / "synthetic" / "a.png").write_bytes(b"image")
    (data / "manifest.csv").write_text("image_id\n1\n", encoding="utf-8")
    (artifacts / "models").mkdir(parents=True)
    (artifacts / "models" / "m.joblib").write_bytes(b"model")


def test_plan_reports_what_would_move_without_touching_anything(homes):
    _legacy_layout(homes)
    plan = projects.migration_plan()
    assert plan["needed"]
    assert (config.DATA_HOME / "manifest.csv").exists(), "미리보기가 파일을 옮기면 안 된다"


def test_migration_moves_everything_into_the_default_project(homes):
    _legacy_layout(homes)
    projects.migrate_legacy()

    assert config.manifest_path().exists()
    assert (config.raw_dir() / "synthetic" / "a.png").read_bytes() == b"image"
    assert (config.model_dir() / "m.joblib").exists()
    assert not (config.DATA_HOME / "manifest.csv").exists()


def test_migration_merges_into_directories_that_already_exist(homes):
    """앱이 시작할 때 `ensure_dirs()`가 빈 디렉터리를 미리 만들어 둔다.

    그것을 충돌로 보고 통째로 건너뛰면, 정작 안에 든 이미지가 옛 자리에 남는다.
    실제로 이 실수를 한 번 했고 216장이 뒤에 남았다.
    """
    _legacy_layout(homes)
    config.ensure_dirs()                      # 빈 raw/ · models/ 를 미리 만든다
    assert config.raw_dir().exists()

    result = projects.migrate_legacy()
    assert (config.raw_dir() / "synthetic" / "a.png").exists(), "빈 폴더 때문에 빠뜨렸다"
    assert (config.model_dir() / "m.joblib").exists()
    assert not result["skipped"]


def test_migration_never_overwrites_a_file(homes):
    """덮어쓰면 되돌릴 수 없다. 충돌은 건너뛰고 알려준다."""
    _legacy_layout(homes)
    config.ensure_dirs()
    config.manifest_path().write_text("이미 있던 것\n", encoding="utf-8")

    result = projects.migrate_legacy()
    assert config.manifest_path().read_text(encoding="utf-8") == "이미 있던 것\n"
    assert any("manifest.csv" in item for item in result["skipped"])


def test_migration_is_safe_to_run_twice(homes):
    _legacy_layout(homes)
    projects.migrate_legacy()
    again = projects.migrate_legacy()
    assert not again["moved"] and not again["skipped"]


def test_pretrained_model_goes_to_the_shared_place(homes):
    """프로젝트마다 45MB를 다시 받게 하면 안 된다."""
    onnx = config.ARTIFACT_HOME / "models" / "onnx"
    onnx.mkdir(parents=True)
    (onnx / "resnet.onnx").write_bytes(b"weights")

    projects.migrate_legacy()
    assert (config.shared_model_dir() / "onnx" / "resnet.onnx").exists()
    assert not (config.model_dir() / "onnx").exists(), "프로젝트 안으로 딸려 들어갔다"


def test_registry_absolute_paths_are_rewritten(homes):
    """폴더를 옮기면 절대경로는 깨진다 — 승격 모델을 못 찾아 배치 추론이 멈춘다."""
    from vision_ai import registry as model_registry

    version_dir = config.ARTIFACT_HOME / "registry" / "v001"
    version_dir.mkdir(parents=True)
    (version_dir / "model.joblib").write_bytes(b"model")
    (config.ARTIFACT_HOME / "registry.csv").write_text(
        f"version,artifact\nv001,{version_dir / 'model.joblib'}\n", encoding="utf-8"
    )

    result = projects.migrate_legacy()
    assert result["rewritten"] == 1

    row = model_registry.load_registry().iloc[0]
    resolved = model_registry.artifact_path(row["artifact"])
    assert resolved is not None and resolved.exists()


# --- 화면의 프로젝트 선택 상자 ----------------------------------------------

def _app(homes):
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file("app.py")


def test_picker_shows_every_project(homes):
    projects.create("가라인")
    projects.create("나라인")
    at = _app(homes)
    at.run()
    assert not at.exception
    assert len(at.sidebar.selectbox[0].options) == 3


def test_picker_follows_a_change_made_elsewhere(homes):
    """프로젝트는 사이드바와 설정 화면 **두 곳**에서 바뀐다.

    설정에서 새 프로젝트를 만들면 파일은 새 것을 가리키는데 선택 상자의 세션 값은 옛
    것으로 남는다. 그 차이를 "사용자가 옛 것을 골랐다"로 오해하면 **방금 만든 프로젝트에서
    도로 튕겨 나온다.** 실제로 그 증상이 있었다.
    """
    at = _app(homes)
    at.run()
    assert at.sidebar.selectbox[0].value == config.DEFAULT_PROJECT

    made = projects.create("설정에서 만든 것")   # 화면 밖에서 바뀐 상황
    at.run()
    assert at.sidebar.selectbox[0].value == made.slug
    assert projects.active().slug == made.slug, "만든 프로젝트에서 튕겨 나왔다"


def test_choosing_another_project_switches_to_it(homes):
    other = projects.create("나라인", activate=False)
    at = _app(homes)
    at.run()
    at.sidebar.selectbox[0].set_value(other.slug).run()
    assert projects.active().slug == other.slug


def test_picker_is_hidden_in_rail_mode(homes):
    """74px 폭에 선택 상자를 밀어넣으면 읽을 수 없게 뭉갠다. 머리글자만 남긴다."""
    at = _app(homes)
    at.run()
    at.sidebar.button[0].click().run()          # 레일로 접기
    assert not at.sidebar.selectbox
