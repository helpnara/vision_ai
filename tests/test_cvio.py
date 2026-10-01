"""Windows 한글 경로에서도 이미지·영상을 읽고 쓰는가 (cvio).

## 무엇에 데였는가

사용자가 PC(Windows)에 내려받아 «데모 한 바퀴 만들기»를 누르자 «합성 샘플을 만들지
못했습니다»로 끝났다. Windows용 OpenCV는 경로에 한글이 있으면 `cv2.imwrite`가 **예외 없이
False만 돌려주고 파일을 쓰지 않는다** — 폴더만 생기고 이미지는 0장이었다. 리눅스에서는 멀쩡해
아무 테스트도 잡지 못했다.

이 컨테이너는 리눅스라 진짜 Windows OpenCV를 돌릴 수 없다. 그래서 **Windows OpenCV처럼
행동하는 가짜**를 끼운다: 비ASCII 경로를 받으면 imwrite/imread는 실패, VideoCapture/VideoWriter는
열리지 않는다. 그리고 `cvio._windows()`를 참으로, 8.3 짧은 이름은 «없음»으로 둔다(짧은 이름이 꺼진
드라이브 — 가장 나쁜 경우). 이 조건에서 앱의 경로가 전부 동작해야 한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from vision_ai import config, cvio, ingest, quickstart, storage, video

ROOT = Path(__file__).resolve().parents[1]

KOREAN = "바탕 화면"


@pytest.fixture
def windows_opencv(monkeypatch, tmp_path):
    """Windows OpenCV처럼 행동하게 만든다. ASCII 임시 폴더는 tmp_path 아래로 돌린다."""
    real_capture, real_writer = cv2.VideoCapture, cv2.VideoWriter
    calls = {"imwrite_refused": 0}

    def imwrite(path, *args, **kwargs):
        if not str(path).isascii():
            calls["imwrite_refused"] += 1
            return False                      # Windows: 예외 없이 False, 파일 없음
        return real_imwrite(path, *args, **kwargs)

    def imread(path, *args, **kwargs):
        return None if not str(path).isascii() else real_imread(path, *args, **kwargs)

    class Capture:
        def __new__(cls, path, *args):
            if not str(path).isascii():
                return real_capture()         # 열리지 않은 캡처
            return real_capture(path, *args)

    class Writer:
        def __new__(cls, path, *args):
            if not str(path).isascii():
                return real_writer()          # 열리지 않은 라이터
            return real_writer(path, *args)

    real_imwrite, real_imread = cv2.imwrite, cv2.imread
    monkeypatch.setattr(cv2, "imwrite", imwrite)
    monkeypatch.setattr(cv2, "imread", imread)
    monkeypatch.setattr(cv2, "VideoCapture", Capture)
    monkeypatch.setattr(cv2, "VideoWriter", Writer)
    monkeypatch.setattr(cvio, "_windows", lambda: True)
    monkeypatch.setattr(cvio, "_short_path", lambda path: None)

    scratch = tmp_path / "ascii-scratch"
    scratch.mkdir()
    monkeypatch.setattr(cvio, "scratch_dir", lambda: scratch)
    return calls


@pytest.fixture
def korean_homes(tmp_path, monkeypatch, windows_opencv):
    """데이터·산출물 폴더가 한글 경로 안에 있다 — 한국어 Windows의 바탕 화면처럼."""
    base = tmp_path / KOREAN / "vision_ai"
    monkeypatch.setattr(config, "DATA_HOME", base / "data")
    monkeypatch.setattr(config, "ARTIFACT_HOME", base / "artifacts")
    monkeypatch.setattr(config, "_active_project", config.DEFAULT_PROJECT)
    config.ensure_dirs()
    return base


# --- 가짜가 정말 Windows처럼 실패하는지부터 본다 --------------------------------

def test_the_fake_reproduces_the_silent_failure(korean_homes, windows_opencv):
    """가짜가 실패를 재현하지 못하면 아래 테스트는 아무것도 증명하지 못한다."""
    target = korean_homes / "x.png"
    assert cv2.imwrite(str(target), np.zeros((4, 4, 3), np.uint8)) is False
    assert not target.exists()
    assert not cv2.VideoWriter(str(korean_homes / "x.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (8, 8)).isOpened()


# --- 이미지 -----------------------------------------------------------------

def test_images_round_trip_through_a_korean_path(korean_homes):
    image = np.random.default_rng(0).integers(0, 255, (16, 20, 3), dtype=np.uint8)
    path = cvio.write_image(korean_homes / "한글 이름.png", image)
    assert path.exists()
    back = cvio.read_image(path)
    assert back is not None and np.array_equal(back, image)


def test_a_failed_encode_is_an_error_not_silence(tmp_path):
    with pytest.raises(OSError):
        cvio.write_image(tmp_path / "x.unknown-format", np.zeros((4, 4, 3), np.uint8))


def test_reading_a_missing_image_is_none(tmp_path):
    assert cvio.read_image(tmp_path / "없음.png") is None


def test_synthetic_samples_are_written_under_a_korean_path(korean_homes, windows_opencv):
    folder = ingest.generate_synthetic(categories=["metal"], n_normal=3, n_defect=4, size=32)
    assert ingest.count_image_files(folder) > 0
    assert windows_opencv["imwrite_refused"] == 0, "cv2.imwrite 를 직접 불렀다"


def test_the_demo_round_trip_works_under_a_korean_path(korean_homes):
    """사용자가 실제로 누른 버튼. 예전에는 «합성 샘플을 만들지 못했습니다»로 끝났다."""
    result = quickstart.run(n_normal=20, n_defect=4)
    assert result.ok, result.warnings
    assert result.version == "v001"
    assert not storage.load_manifest().empty


def test_the_demo_names_the_reason_when_it_cannot_write(sandbox, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("디스크가 가득 찼습니다")

    monkeypatch.setattr(ingest, "generate_synthetic", refuse)
    result = quickstart.run()
    assert not result.ok
    assert "디스크가 가득 찼습니다" in result.warnings[0]


# --- 영상 -------------------------------------------------------------------

def test_sample_video_probe_and_extract_work_under_a_korean_path(korean_homes, tmp_path):
    made = video.make_sample(korean_homes / "영상" / "시험.mp4", seconds=2, fps=10, size=(64, 48))
    assert made.exists() and made.stat().st_size > 0

    info = video.probe(made)
    assert info.usable and info.frame_count > 0

    result = video.extract(made, stride=5, out_dir=korean_homes / "프레임")
    assert result.saved and all(p.exists() for p in result.saved)
    assert not list((tmp_path / "ascii-scratch").iterdir()), "우회용 임시 파일이 남았다"


def test_an_unopened_writer_leaves_nothing_behind(korean_homes, tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a, **k: _Closed())
    writer = cvio.VideoWriter(korean_homes / "x.webm", "VP80", 5, (8, 8))
    assert not writer.isOpened()
    writer.release()
    assert not (korean_homes / "x.webm").exists()
    assert not list((tmp_path / "ascii-scratch").iterdir())


class _Closed:
    def isOpened(self):  # noqa: N802
        return False

    def release(self):
        pass


# --- 규칙: OpenCV 파일 입출력은 cvio 만 거친다 ---------------------------------

def test_nothing_else_calls_opencv_file_io_directly():
    """한 곳이라도 cv2.imwrite 를 직접 부르면 Windows 한글 경로에서 조용히 실패한다."""
    pattern = re.compile(r"cv2\.(imwrite|imread|VideoCapture|VideoWriter)\(")
    offenders = []
    for folder in ("src", "server", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            if path.name == "cvio.py":
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert not offenders, "cvio 를 거치지 않은 OpenCV 파일 입출력: " + ", ".join(offenders)
