"""배포 zip 이름 규칙 — `느린 눈-YYYYMMDD-vNN.zip` (사용자 요청, 2026-10-02)."""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_release", ROOT / "scripts" / "package_release.py")
package_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_release)


def test_the_name_follows_the_users_pattern():
    when = datetime(2026, 10, 2, 9, 30, tzinfo=package_release.TIMEZONE)
    assert package_release.release_name(1, when) == "느린 눈-20261002-v01.zip"
    assert package_release.release_name(12, when) == "느린 눈-20261002-v12.zip"


def test_numbers_keep_rising_across_days(tmp_path):
    """날짜가 바뀌어도 번호를 다시 세지 않는다 — 이름만 보고 최신을 알 수 있어야 한다."""
    log = tmp_path / "releases.md"
    assert package_release.last_version(log) == 0
    day1 = datetime(2026, 10, 2, tzinfo=package_release.TIMEZONE)
    day2 = datetime(2026, 10, 3, tzinfo=package_release.TIMEZONE)
    package_release.record(1, day1, package_release.release_name(1, day1), "abc1234", "첫 판", log)
    package_release.record(2, day1, package_release.release_name(2, day1), "abc1235", "a | b", log)
    assert package_release.last_version(log) == 2
    assert package_release.release_name(package_release.last_version(log) + 1, day2) == "느린 눈-20261003-v03.zip"
    assert "a / b" in log.read_text(encoding="utf-8"), "표가 깨지지 않게 | 를 바꾼다"
