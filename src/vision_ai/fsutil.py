"""파일·폴더 다루기 — Windows에서 «열려 있는 파일»과 «쓸 수 없는 이름»을 견디게.

## 왜 이 모듈이 있는가

리눅스는 열려 있는 파일도 지우고 덮어쓸 수 있다. **Windows는 안 된다** — 다른 곳(브라우저가
재생 중인 영상, 엑셀로 열어 둔 CSV, 아직 닫지 않은 memmap)이 파일을 쥐고 있으면
`PermissionError: [WinError 32]`가 난다. 리눅스에서만 돌던 시절에는 드러나지 않았다
(같은 계열의 사고: `cvio` 독스트링).

또 Windows 파일 이름에는 `\\ / : * ? " < > |`를 쓸 수 없고 `CON`·`NUL` 같은 예약어도 안 된다.
사용자가 입력한 카테고리 이름이 그대로 폴더 이름이 되면 거기서 깨진다.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import time
from pathlib import Path

RETRIES = 5
RETRY_DELAY = 0.2
"""잠깐 쥐고 있다 놓는 경우(백신 검사·탐색기 미리보기·끝나 가는 스트리밍)를 기다리는 시간."""

_INVALID = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def remove_tree(path) -> bool:
    """폴더를 지운다. 다 지웠으면 True, 무엇인가 남았으면 False (예외를 올리지 않는다).

    읽기 전용 파일은 쓰기 가능으로 바꿔 지우고, 잠긴 파일은 잠깐 기다렸다 다시 시도한다.
    """
    path = Path(path)

    def make_writable_and_retry(function, target, _info):
        try:
            os.chmod(target, stat.S_IWRITE)
            function(target)
        except OSError:
            pass

    for attempt in range(RETRIES):
        if not path.exists():
            return True
        try:
            shutil.rmtree(path, onerror=make_writable_and_retry)
        except OSError:
            pass
        if not path.exists():
            return True
        time.sleep(RETRY_DELAY * (attempt + 1))
    return not path.exists()


def safe_name(text: str, *, fallback: str = "unnamed") -> str:
    """폴더·파일 이름 한 조각으로 쓸 수 있게 다듬는다. 한글은 그대로 둔다.

    `a/b`·`..`처럼 경로를 벗어나는 이름도 여기서 막힌다 (구분자를 `_`로 바꾸므로).
    """
    cleaned = _INVALID.sub("_", str(text)).strip().rstrip(". ")
    if not cleaned or set(cleaned) <= {".", "_"}:
        return fallback
    if cleaned.split(".")[0].lower() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:120]
