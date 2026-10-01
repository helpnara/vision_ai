"""배포 zip 만들기 — `느린 눈-YYYYMMDD-vNN.zip`.

    python scripts/package_release.py                 # 배포 브랜치 → release/ 아래에 zip
    python scripts/package_release.py --out <폴더>    # 다른 곳에 만들기
    python scripts/package_release.py --dry-run       # 이름만 보고 아무것도 안 만든다

사용자는 zip을 내려받아 **기존 폴더에 덮어쓰고** `run.bat`을 누르는 방식으로 쓴다. 그래서

* 파일 이름은 **사용자가 정한 규칙**을 따른다 — `느린 눈-날짜-v번호.zip` (2026-10-02 요청).
  날짜는 한국 시간 기준 `YYYYMMDD`, 번호는 **날짜와 무관하게 하나씩 오른다**(v01, v02, …).
  날짜가 바뀌어도 번호를 다시 0부터 세지 않는 이유: «v03이 v02보다 새것»이 언제나 참이어야
  여러 날 받은 zip 중 무엇이 최신인지 이름만 보고 안다.
* zip 안의 최상위 폴더 이름은 **언제나 `vision_ai/`** 로 둔다. 판마다 바뀌면 «같은 폴더에
  덮어쓰기»가 안 된다.
* 무엇이 몇 번 판인지는 `docs/releases.md`에 남긴다. 다음 번호도 거기서 정한다 — zip은
  사용자 PC에만 있고 저장소에는 없으므로, 기록이 없으면 번호가 겹친다.

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

_ROW = re.compile(r"^\|\s*v(\d+)\s*\|")

LOG_HEADER = """# 배포 기록

`scripts/package_release.py`가 zip을 만들 때마다 한 줄씩 붙인다. **다음 번호는 이 표의 마지막
번호 + 1** 이다 — 손으로 줄을 지우거나 고치면 번호가 겹친다.

| 판 | 날짜 (KST) | 파일 | 커밋 | 내용 |
|---|---|---|---|---|
"""


def last_version(log: Path = LOG) -> int:
    """기록된 마지막 판 번호. 기록이 없으면 0."""
    if not log.exists():
        return 0
    numbers = [int(m.group(1)) for line in log.read_text(encoding="utf-8").splitlines()
               if (m := _ROW.match(line))]
    return max(numbers, default=0)


def release_name(version: int, when: datetime) -> str:
    return f"{PRODUCT}-{when.strftime('%Y%m%d')}-v{version:02d}.zip"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def record(version: int, when: datetime, name: str, commit: str, summary: str, log: Path = LOG) -> None:
    text = log.read_text(encoding="utf-8") if log.exists() else LOG_HEADER
    if not text.endswith("\n"):
        text += "\n"
    summary = summary.replace("|", "/").strip() or "-"
    text += f"| v{version:02d} | {when.strftime('%Y-%m-%d %H:%M')} | `{name}` | `{commit}` | {summary} |\n"
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
    version = last_version() + 1
    name = release_name(version, when)
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
