"""OpenCV 파일 입출력 — Windows 한글 경로에서도 동작하게.

## 왜 이 모듈이 있는가 (실제로 데인 것)

Windows용 OpenCV는 파일 이름을 시스템 코드페이지(ANSI)로 넘긴다. 그래서 경로에 한글 같은
비ASCII 문자가 있으면 — `C:\\Users\\홍길동\\…`, `OneDrive\\바탕 화면\\…` 처럼 한국어 PC에서는
아주 흔하다 — **`cv2.imwrite`는 예외 없이 `False`만 돌려주고 파일을 쓰지 않는다.**
`cv2.imread`·`cv2.VideoCapture`·`cv2.VideoWriter`도 같다.

리눅스(예전 배포 환경)에서는 멀쩡해서 몰랐다. 사용자가 PC에 내려받아 «데모 한 바퀴 만들기»를
누르자 합성 이미지 폴더만 생기고 이미지는 0장 → manifest가 비어 «합성 샘플을 만들지
못했습니다»로 끝났다. 읽기는 이미 바이트로 읽고 있었는데(`viz.load_rgb`) 쓰기만 빠져 있었다.

## 규칙

**`src/`·`server/`에서 `cv2.imwrite / cv2.imread / cv2.VideoCapture / cv2.VideoWriter`를
직접 부르지 않는다.** 이 모듈을 거친다. `tests/test_cvio.py`가 그것을 검사한다.

* 이미지: 메모리에서 인코딩/디코딩하고 파일은 파이썬이 읽고 쓴다 (`imencode`/`imdecode`).
  파이썬의 파일 API는 유니코드 경로를 그대로 다룬다.
* 영상: 디코더·인코더가 파일을 직접 열어야 하므로 그렇게 할 수 없다. 대신 OpenCV가 열 수 있는
  **ASCII 경로**를 만들어 넘긴다 — 8.3 짧은 이름(`C:\\Users\\5F6D~1\\…`)을 먼저 쓰고, 그것도
  안 되면(짧은 이름이 꺼진 드라이브) ASCII 임시 폴더에 링크/복사해서 연다. 쓰기는 임시 폴더에
  쓴 뒤 제자리로 옮긴다.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

SCRATCH_NAME = "vision_ai_cv"
"""ASCII 임시 폴더 안에 만드는 하위 폴더 이름."""


def _windows() -> bool:
    """Windows인가. 테스트가 바꿔 끼울 수 있게 함수로 둔다."""
    return sys.platform == "win32"


def opencv_safe(path) -> bool:
    """OpenCV에 그대로 넘겨도 되는 경로인가. Windows가 아니면 언제나 된다."""
    return not _windows() or str(path).isascii()


# --- 이미지 -----------------------------------------------------------------

def write_image(path, image: np.ndarray, params: Sequence[int] = ()) -> Path:
    """이미지를 파일로 쓴다. 실패하면 `OSError` — 조용히 넘어가지 않는다.

    형식은 확장자로 정한다 (`.png`·`.jpg` …). `params`는 `cv2.imwrite`와 같다
    (예: `[cv2.IMWRITE_JPEG_QUALITY, 90]`).
    """
    path = Path(path)
    suffix = path.suffix.lower() or ".png"
    try:
        ok, buffer = cv2.imencode(suffix, image, list(params))
    except cv2.error as exc:              # 모르는 확장자 등 — 호출한 쪽은 OSError 하나만 잡으면 된다
        raise OSError(f"이미지를 {suffix} 형식으로 바꾸지 못했습니다: {path} ({exc.err})") from exc
    if not ok:
        raise OSError(f"이미지를 {suffix} 형식으로 바꾸지 못했습니다: {path}")
    path.write_bytes(buffer.tobytes())
    return path


def read_image(path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """이미지 파일을 읽는다 (BGR 또는 `flags`대로). 없거나 못 읽으면 None."""
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


# --- 영상 -------------------------------------------------------------------

def _short_path(path: str) -> str | None:
    """Windows 8.3 짧은 이름. **이미 있는** 경로만 된다. 못 얻으면 None."""
    if not _windows():
        return None
    try:
        import ctypes
        from ctypes import wintypes

        get = ctypes.windll.kernel32.GetShortPathNameW  # type: ignore[attr-defined]
        get.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get.restype = wintypes.DWORD
        size = get(path, None, 0)
        if not size:
            return None
        buffer = ctypes.create_unicode_buffer(size)
        if not get(path, buffer, size):
            return None
        return buffer.value
    except Exception:  # noqa: BLE001 — 어떤 이유든 «짧은 이름 없음»으로 본다
        return None


def _ascii_short(path: Path) -> str | None:
    """OpenCV가 열 수 있는 ASCII 경로 (원래 경로가 ASCII거나 짧은 이름이 ASCII일 때)."""
    text = str(path)
    if opencv_safe(text):
        return text
    short = _short_path(text)
    return short if short and short.isascii() else None


def scratch_dir() -> Path:
    """OpenCV에 넘길 수 있는 ASCII 임시 폴더. 없으면 `OSError`.

    사용자 임시 폴더도 `C:\\Users\\홍길동\\AppData\\Local\\Temp`라 한글일 수 있다. 그래서
    짧은 이름 → 공용 폴더(`C:\\Users\\Public`) → 드라이브 루트 순으로 찾는다.
    """
    candidates: list[str] = []
    temp = tempfile.gettempdir()
    candidates += [temp, _short_path(temp) or ""]
    for key in ("PUBLIC", "ProgramData"):
        if os.environ.get(key):
            candidates.append(os.environ[key])
    candidates.append(os.environ.get("SystemDrive", "C:") + os.sep)
    for base in candidates:
        if not base or not opencv_safe(base):
            continue
        target = Path(base) / SCRATCH_NAME
        try:
            target.mkdir(parents=True, exist_ok=True)
            probe = target / f".probe-{uuid.uuid4().hex}"
            probe.write_bytes(b"")
            probe.unlink()
            return target
        except OSError:
            continue
    raise OSError(
        "영상 경로에 한글 등 영문이 아닌 글자가 있어 OpenCV가 열 수 없고, 우회할 임시 폴더도 "
        "찾지 못했습니다. 앱 폴더를 영문 경로(예: C:\\work\\vision_ai)로 옮겨 실행하세요."
    )


@contextmanager
def video_capture(path) -> Iterator["cv2.VideoCapture"]:
    """`cv2.VideoCapture`를 연다. 끝나면 닫고, 우회용 임시 파일도 치운다.

    열렸는지는 호출한 쪽이 `capture.isOpened()`로 본다 (예전과 같다).
    """
    path = Path(path)
    usable = _ascii_short(path)
    temporary: Path | None = None
    if usable is None and path.is_file():
        temporary = scratch_dir() / f"in-{uuid.uuid4().hex}{path.suffix.lower()}"
        try:
            os.link(path, temporary)              # 같은 드라이브면 즉시, 용량도 안 쓴다
        except OSError:
            shutil.copy2(path, temporary)         # 다른 드라이브면 복사
        usable = str(temporary)
    capture = cv2.VideoCapture(usable if usable is not None else str(path))
    try:
        yield capture
    finally:
        capture.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class VideoWriter:
    """`cv2.VideoWriter`와 같게 쓰되, 한글 경로면 ASCII 임시 파일에 쓰고 `release()` 때 옮긴다.

    `isOpened()` · `write()` · `release()`만 쓴다. 열리지 않았으면 `release()`가 임시 파일을
    치우고 아무것도 옮기지 않는다.
    """

    def __init__(self, path, fourcc: str, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = _ascii_short(self.path.parent)
        if parent is not None and self.path.name.isascii():
            self._target = os.path.join(parent, self.path.name)
            self._temporary: Path | None = None
        else:
            self._temporary = scratch_dir() / f"out-{uuid.uuid4().hex}{self.path.suffix.lower()}"
            self._target = str(self._temporary)
        self._writer = cv2.VideoWriter(self._target, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        self._released = False

    def isOpened(self) -> bool:  # noqa: N802 — cv2.VideoWriter 와 같은 이름을 쓴다
        return bool(self._writer.isOpened())

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        opened = self._writer.isOpened()
        self._writer.release()
        if self._temporary is None:
            return
        if opened and self._temporary.exists():
            self.path.unlink(missing_ok=True)
            shutil.move(str(self._temporary), str(self.path))
        else:
            self._temporary.unlink(missing_ok=True)
