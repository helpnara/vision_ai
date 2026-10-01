"""Windows·한국어 PC에서만 깨지는 것들 (전수 조사 결과를 잠근다).

`test_cvio.py`(OpenCV 한글 경로)와 같은 계열이다 — **리눅스에서는 멀쩡하고 Windows에서만,
대개 조용히 깨진다.** 이 컨테이너는 리눅스라 Windows의 행동을 흉내 내는 가짜를 끼워 확인한다.

| 증상 (Windows) | 원인 | 고친 곳 |
|---|---|---|
| 격자 캐시가 영영 갱신 안 되고 .tmp.npy 가 쌓임 | 열린 memmap 위로 os.replace → WinError 32, 조용히 삼킴 | patch_cache.save |
| 판정 영상을 다시 만들면 실패 | 브라우저가 재생 중인 파일이 든 폴더를 rmtree | playback.render / find |
| «경로로 복사»로 붙여 넣은 폴더를 못 찾음 | 탐색기가 경로를 큰따옴표로 감싼다 | common.typed_path |
| 엑셀로 저장한 분할 CSV 가 안 열림 | 한국어 엑셀 CSV 는 cp949 | labeling._read_user_csv |
| `컵:빨강` 카테고리로 올리면 실패 | `:` 는 Windows 파일 이름에 못 쓴다 | fsutil.safe_name |
| 화면에 «500 Internal Server Error» 만 | 엑셀이 연 CSV 에 쓰기 → PermissionError | main 의 OSError 처리 |
| 파일 경로 검사가 정상 경로를 막음 / 형제 폴더를 통과시킴 | 문자열 startswith 비교 | common.inside |
"""

from __future__ import annotations

import gc
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vision_ai import config, feature_cache, fsutil, ingest, labeling, models, patch_cache, playback, storage


@pytest.fixture
def client(sandbox):
    from server import state
    from server.main import app

    state.clear()
    return TestClient(app, raise_server_exceptions=False)


# --- 열린 파일은 덮어쓰거나 지울 수 없다 -------------------------------------

def _open_memmaps() -> set[str]:
    """지금 살아 있는 memmap 들이 가리키는 파일."""
    names = set()
    for obj in gc.get_objects():
        if isinstance(obj, np.memmap) and getattr(obj, "filename", None):
            names.add(os.path.normcase(os.path.abspath(obj.filename)))
    return names


@pytest.fixture
def windows_replace(monkeypatch):
    """Windows처럼: 누군가 memmap 으로 열어 둔 파일 위로는 os.replace 할 수 없다."""
    real = os.replace

    def replace(src, dst, *args, **kwargs):
        gc.collect()
        if os.path.normcase(os.path.abspath(dst)) in _open_memmaps():
            raise PermissionError(32, "다른 프로세스가 파일을 사용 중이기 때문에 액세스할 수 없습니다", str(dst))
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(patch_cache.os, "replace", replace)


def _image(sandbox, name: str) -> Path:
    path = sandbox / "raw" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(name.encode())
    return path


def test_the_patch_cache_grows_even_when_the_old_array_is_open(sandbox, windows_replace):
    """두 번째 학습부터 새 이미지가 캐시에 쌓여야 한다. 예전에는 Windows에서 영영 안 쌓였다."""
    settings = models.AnomalyConfig()
    first = patch_cache.load(settings)
    first.put("a", _image(sandbox, "a.png"), np.ones((3, 3, 4), np.float32))
    patch_cache.save(first)
    del first          # 학습 잡 하나가 끝난 것 — 잡마다 캐시를 새로 열고, 끝나면 놓는다
    gc.collect()

    second = patch_cache.load(settings)            # 옛 배열을 memmap 으로 연 상태
    assert second.stored is not None
    second.put("b", _image(sandbox, "b.png"), np.full((3, 3, 4), 2.0, np.float32))
    patch_cache.save(second)

    third = patch_cache.load(settings)
    assert len(third) == 2, "새로 뽑은 것이 저장되지 않았다"
    assert float(third.get("b", sandbox / "raw" / "b.png").mean()) == 2.0


def test_saving_the_patch_cache_leaves_no_temporary_files(sandbox, windows_replace):
    settings = models.AnomalyConfig()
    for name in ("a", "b"):
        cache = patch_cache.load(settings)
        cache.put(name, _image(sandbox, f"{name}.png"), np.ones((3, 3, 4), np.float32))
        patch_cache.save(cache)
    leftovers = list(patch_cache.cache_dir(patch_cache.schema_fingerprint(settings)).glob("*.tmp*"))
    assert not leftovers, f"임시 파일이 남았다: {leftovers}"


def test_the_same_cache_object_is_still_usable_after_saving(sandbox, windows_replace):
    settings = models.AnomalyConfig()
    cache = patch_cache.load(settings)
    path = _image(sandbox, "a.png")
    cache.put("a", path, np.ones((3, 3, 4), np.float32))
    patch_cache.save(cache)
    assert cache.get("a", path) is not None and len(cache) == 1


def test_clearing_a_locked_cache_says_so(sandbox, monkeypatch):
    """예전에는 실패를 삼켜 «비웠습니다»라고 해 놓고 숫자는 그대로였다."""
    monkeypatch.setattr(fsutil, "remove_tree", lambda path: False)
    with pytest.raises(OSError):
        patch_cache.clear()


def test_clearing_a_locked_feature_cache_says_so(sandbox, monkeypatch):
    def locked(self, *args, **kwargs):
        raise PermissionError(32, "사용 중")

    feature_cache.cache_path().parent.mkdir(parents=True, exist_ok=True)
    feature_cache.cache_path().write_bytes(b"x")
    monkeypatch.setattr(Path, "unlink", locked)
    with pytest.raises(OSError):
        feature_cache.clear()


def test_remove_tree_reports_what_it_could_not_remove(tmp_path, monkeypatch):
    folder = tmp_path / "locked"
    folder.mkdir()
    (folder / "x").write_text("x")
    monkeypatch.setattr(fsutil.shutil, "rmtree", lambda *a, **k: None)   # 아무것도 못 지운다
    monkeypatch.setattr(fsutil, "RETRY_DELAY", 0.0)
    assert fsutil.remove_tree(folder) is False
    assert fsutil.remove_tree(tmp_path / "없음") is True


class _FakeModel:
    kind = "anomaly"
    version = "vtest"
    threshold = 5.0

    def score_image(self, rgb) -> float:
        return 9.0 if rgb[:, :, 0].mean() > 100 else 1.0

    def score_map(self, rgb):
        return np.ones((16, 16), np.float32)


@pytest.fixture
def clip(tmp_path) -> Path:
    import cv2

    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for i in range(12):
        writer.write(np.full((48, 64, 3), 200 if 4 <= i < 6 else 30, np.uint8))
    writer.release()
    return path


def test_a_playback_can_be_remade_while_the_old_one_is_being_watched(clip, tmp_path, monkeypatch):
    """브라우저가 옛 판정 영상을 재생하는 중이면 Windows는 그 폴더를 못 지운다(WinError 32)."""
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")
    first = playback.render(clip, _FakeModel(), limit=12)

    monkeypatch.setattr(playback.fsutil, "remove_tree", lambda path: False)   # 잠겨 있다
    second = playback.render(clip, _FakeModel(), limit=6)
    assert second.directory != first.directory
    assert second.video.exists()

    monkeypatch.undo()
    monkeypatch.setattr(config, "interim_dir", lambda: tmp_path / "interim")
    found = playback.find(clip, "vtest")
    assert found is not None and len(found) == 6, "가장 최근 판정본을 보여야 한다"


# --- 사용자가 친 경로·이름·파일 ------------------------------------------------

@pytest.mark.parametrize("typed", ['"{p}"', "'{p}'", "  {p}  ", "“{p}”"])
def test_a_folder_path_copied_from_explorer_is_accepted(client, tmp_path, typed):
    """탐색기 «경로로 복사»는 경로를 큰따옴표로 감싼다. 그대로 붙여 넣어도 찾아야 한다."""
    folder = ingest.generate_synthetic(categories=["metal"], n_normal=2, n_defect=0, size=32,
                                       out_dir=tmp_path / "데이터 셋")
    response = client.get("/api/ingest/folder/preview", params={"root": typed.format(p=folder), "layout": "visa"})
    assert response.status_code == 200, response.json()
    assert response.json()["total"] > 0


def test_a_video_path_copied_from_explorer_is_accepted(client, tmp_path):
    from vision_ai import video

    made = video.make_sample(tmp_path / "영상.mp4", seconds=1, fps=5, size=(32, 24))
    quoted = f'"{made}"'
    assert client.get("/api/ingest/videos/check", params={"path": quoted}).status_code == 200
    body = client.get("/api/operations/videos/check", params={"path": quoted}).json()
    assert body["ok"] and body["path"] == str(made)


def test_a_split_csv_saved_by_korean_excel_is_read(sandbox):
    """한국어 Windows 엑셀의 «CSV(쉼표로 분리)»는 cp949 다."""
    out = ingest.generate_synthetic(categories=["metal"], n_normal=2, n_defect=0, size=32)
    ingest.ingest_folder(out, source="synthetic", layout="visa")
    manifest = storage.load_manifest()
    rows = pd.DataFrame({"이미지": manifest["path"], "split": "train", "비고": "한글 메모"})
    rows = rows.rename(columns={"이미지": "image"})
    raw = rows.to_csv(index=False).encode("cp949")
    mapping, unmatched = labeling.import_split_csv(io.BytesIO(raw), manifest)
    assert len(mapping) == len(manifest) and unmatched == 0


def test_a_utf8_split_csv_with_bom_still_works(sandbox):
    out = ingest.generate_synthetic(categories=["metal"], n_normal=2, n_defect=0, size=32)
    ingest.ingest_folder(out, source="synthetic", layout="visa")
    manifest = storage.load_manifest()
    raw = pd.DataFrame({"image": manifest["path"], "split": "test"}).to_csv(index=False).encode("utf-8-sig")
    mapping, _ = labeling.import_split_csv(io.BytesIO(raw), manifest)
    assert set(mapping.values()) == {"test"}


@pytest.mark.parametrize("name, expected", [
    ("컵:빨강", "컵_빨강"),
    ("../../밖", ".._.._밖"),
    ("a/b\\c", "a_b_c"),
    ("CON", "_CON"),
    ("nul.txt", "_nul.txt"),
    ("끝에 점.", "끝에 점"),
    ("", "unnamed"),
    ("..", "unnamed"),
])
def test_names_are_made_safe_for_windows(name, expected):
    assert fsutil.safe_name(name) == expected


def test_an_upload_category_cannot_escape_or_break_the_folder(sandbox):
    from vision_ai import cvio

    data = cvio.write_image(sandbox / "x.png", np.zeros((40, 40, 3), np.uint8)).read_bytes()
    result = ingest.ingest_uploads([("사진.png", data)], category="../컵:빨강")
    assert result.added == 1
    row = storage.load_manifest().iloc[0]
    saved = storage.resolve_path(row["path"]).resolve()
    assert config.raw_dir().resolve() in saved.parents, f"raw 밖에 저장됐다: {saved}"
    assert row["category"] == "../컵:빨강", "manifest 에는 사용자가 친 그대로 남는다"


def test_an_uploaded_video_name_cannot_escape_the_video_folder(client):
    from vision_ai import video

    files = {"file": ("..\\..\\evil:name.mp4", b"not really a video", "video/mp4")}
    body = client.post("/api/ingest/videos/upload", files=files).json()
    saved = Path(body["path"]).resolve()
    assert saved.parent == video.video_dir().resolve()
    assert ":" not in saved.name


# --- 경로 검사 ---------------------------------------------------------------

def test_a_sibling_folder_with_the_same_prefix_is_not_inside(tmp_path):
    from server.common import inside

    (tmp_path / "data").mkdir()
    (tmp_path / "data_backup").mkdir()
    assert inside(tmp_path / "data" / "a.png", [tmp_path / "data"])
    assert not inside(tmp_path / "data_backup" / "a.png", [tmp_path / "data"])
    assert not inside(tmp_path / "data" / ".." / "secret.txt", [tmp_path / "data"])


def test_files_in_a_look_alike_folder_are_refused(client):
    sibling = Path(str(config.DATA_HOME) + "_backup")
    sibling.mkdir(parents=True, exist_ok=True)
    secret = sibling / "secret.png"
    secret.write_bytes(b"x")
    assert client.get("/api/files/path", params={"path": str(secret)}).status_code == 404


# --- 오류는 말로 --------------------------------------------------------------

def test_a_locked_file_becomes_a_readable_message(client, monkeypatch):
    """엑셀로 manifest.csv 를 열어 두면 쓰기가 PermissionError 로 실패한다."""
    from server.routers import settings as settings_router

    def locked():
        raise PermissionError(13, "Permission denied", "manifest.csv")

    monkeypatch.setattr(settings_router.feature_cache, "clear", locked)
    response = client.post("/api/settings/cache/clear")
    assert response.status_code == 500
    assert "다른 프로그램" in response.json()["detail"]


# --- 학습 폴더 설정 ------------------------------------------------------------

def test_the_detection_yaml_path_survives_windows_style_folders(tmp_path):
    from vision_ai import detection

    yaml = pytest.importorskip("yaml")
    root = tmp_path / "학습 #1: 폴더"
    root.mkdir()
    text = detection._data_yaml(root, ["scratch"])
    parsed = yaml.safe_load(text)
    assert parsed["path"] == root.resolve().as_posix()
    assert parsed["names"] == {0: "scratch"}
