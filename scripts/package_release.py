"""배포 zip 만들기 — `느린 눈-YYYYMMDD-vN.zip`.

    python scripts/package_release.py                 # 배포 브랜치 → release/ 아래에 zip
    python scripts/package_release.py --out <폴더>    # 다른 곳에 만들기
    python scripts/package_release.py --dry-run       # 이름만 보고 아무것도 안 만든다

사용자는 zip을 내려받아 **기존 폴더에 덮어쓰고** `run.bat`을 누르는 방식으로 쓴다. 그래서

* 파일 이름은 **사용자가 정한 규칙**을 따른다 — `느린 눈-날짜-v번호.zip`.
  날짜는 한국 시간 기준 `YYYYMMDD`(예: `20261002`), 번호는 **그날 몇 번째 판인가**다 —
  하루에 여러 번 올리면 v1, v2, v3 …, 날짜가 바뀌면 다시 v1. 자릿수를 채우지 않는다(`v01` 아님).
  (2026-10-02에 처음 정할 때는 날짜와 무관하게 이어 세는 `v01`이었다. 2026-10-17 사용자가
  «하루에 수회 올라가면 v1, v2, v3 순으로»라고 바로잡았다. 최신은 날짜 → 번호 순으로 비교한다.)
* zip 안의 최상위 폴더 이름은 **언제나 `vision_ai/`** 로 둔다. 판마다 바뀌면 «같은 폴더에
  덮어쓰기»가 안 된다.
* 무엇이 몇 번 판인지는 `docs/releases.md`에 남긴다. 그날의 다음 번호도 거기서 정한다 — zip은
  사용자 PC에만 있고 저장소에는 없으므로, 기록이 없으면 같은 날 같은 이름이 두 번 나온다.

배포 브랜치(README 배포표)를 묶는다. 작업 브랜치를 묶으면 아직 배포하지 않은 것이 섞인다.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_REF = "origin/claude/vision-surface-defect-detection-j16xhj"
PRODUCT = "느린 눈"
PREFIX = "vision_ai/"
LOG = ROOT / "docs" / "releases.md"
TIMEZONE = ZoneInfo("Asia/Seoul")

_RELEASED = re.compile(r"-(\d{8})-v(\d+)\.zip")
"""기록의 파일 이름에서 (날짜, 번호)를 읽는다. 옛 `v01` 표기도 같은 숫자로 읽힌다."""

LOG_HEADER = """# 배포 기록

`scripts/package_release.py`가 zip을 만들 때마다 한 줄씩 붙인다. **다음 번호는 «그날 마지막 번호 + 1»**이고,
날짜가 바뀌면 v1 이다 — 손으로 줄을 지우거나 고치면 번호가 겹친다.

| 판 | 날짜 (KST) | 파일 | 커밋 | 내용 |
|---|---|---|---|---|
"""


def last_version(day: str, log: Path = LOG) -> int:
    """`day`(YYYYMMDD)에 이미 나간 마지막 판 번호. 그날 나간 것이 없으면 0."""
    if not log.exists():
        return 0
    numbers = [
        int(m.group(2))
        for m in _RELEASED.finditer(log.read_text(encoding="utf-8"))
        if m.group(1) == day
    ]
    return max(numbers, default=0)


def release_name(version: int, when: datetime) -> str:
    return f"{PRODUCT}-{when.strftime('%Y%m%d')}-v{version}.zip"


def next_release(when: datetime, log: Path = LOG) -> tuple[int, str]:
    """(그날의 다음 번호, 파일 이름)."""
    version = last_version(when.strftime("%Y%m%d"), log) + 1
    return version, release_name(version, when)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def record(version: int, when: datetime, name: str, commit: str, summary: str, log: Path = LOG) -> None:
    text = log.read_text(encoding="utf-8") if log.exists() else LOG_HEADER
    if not text.endswith("\n"):
        text += "\n"
    summary = summary.replace("|", "/").strip() or "-"
    text += f"| v{version} | {when.strftime('%Y-%m-%d %H:%M')} | `{name}` | `{commit}` | {summary} |\n"
    log.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="배포 zip 만들기")
    parser.add_argument("--ref", default=DEPLOY_REF, help="묶을 git 참조 (기본: 배포 브랜치)")
    parser.add_argument("--out", type=Path, default=ROOT / "release", help="zip을 둘 폴더")
    parser.add_argument("--summary", default="", help="배포 기록에 남길 한 줄 설명")
    parser.add_argument("--dry-run", action="store_true", help="이름만 보여 준다")
    args = parser.parse_args()

    commit = _git("rev-parse", "--short", args.ref)
    when = datetime.now(TIMEZONE)
    version, name = next_release(when)
    if args.dry_run:
        print(f"{name}  ←  {args.ref} ({commit})")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / name
    subprocess.run(
        ["git", "archive", "--format=zip", f"--prefix={PREFIX}", "-o", str(target), args.ref],
        cwd=ROOT, check=True,
    )
    record(version, when, name, commit, args.summary or _git("log", "-1", "--format=%s", args.ref))
    print(target)


if __name__ == "__main__":
    sys.exit(main())
