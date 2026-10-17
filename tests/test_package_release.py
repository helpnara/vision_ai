"""배포 zip 이름 규칙 — `느린 눈-YYYYMMDD-vN.zip`.

사용자 요청(2026-10-17): 날짜는 `20261002` 형태, 하루에 여러 번 올라가면 v1 → v2 → v3,
날짜가 바뀌면 다시 v1. 자릿수를 채우지 않는다.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("package_release", ROOT / "scripts" / "package_release.py")
package_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_release)

TZ = package_release.TIMEZONE


def _day(d: int, hour: int = 9) -> datetime:
    return datetime(2026, 10, d, hour, 0, tzinfo=TZ)


def test_the_name_follows_the_users_pattern():
    assert package_release.release_name(1, _day(2)) == "느린 눈-20261002-v1.zip"
    assert package_release.release_name(12, _day(2)) == "느린 눈-20261002-v12.zip"


def test_the_first_release_of_a_day_is_v1(tmp_path):
    assert package_release.next_release(_day(17), tmp_path / "releases.md") == (1, "느린 눈-20261017-v1.zip")


def test_several_releases_in_one_day_count_up(tmp_path):
    log = tmp_path / "releases.md"
    for expected in (1, 2, 3):
        version, name = package_release.next_release(_day(17, 8 + expected), log)
        assert (version, name) == (expected, f"느린 눈-20261017-v{expected}.zip")
        package_release.record(version, _day(17), name, "abc1234", f"{expected}번째", log)


def test_a_new_day_starts_again_at_v1(tmp_path):
    log = tmp_path / "releases.md"
    for version in (1, 2):
        package_release.record(version, _day(17), package_release.release_name(version, _day(17)), "abc", "", log)
    assert package_release.next_release(_day(18), log) == (1, "느린 눈-20261018-v1.zip")
    assert package_release.next_release(_day(17), log)[0] == 3, "그날의 번호는 이어진다"


def test_the_old_zero_padded_entry_is_still_counted(tmp_path):
    """10-02의 `v01`은 규칙을 바로잡기 전 표기다. 같은 날 다시 올리면 v2가 나와야 한다."""
    log = tmp_path / "releases.md"
    log.write_text(
        package_release.LOG_HEADER
        + "| v01 | 2026-10-02 08:51 | `느린 눈-20261002-v01.zip` | `2ca7237` | 첫 판 |\n",
        encoding="utf-8",
    )
    assert package_release.next_release(_day(2), log) == (2, "느린 눈-20261002-v2.zip")


def test_a_pipe_in_the_summary_does_not_break_the_table(tmp_path):
    log = tmp_path / "releases.md"
    package_release.record(1, _day(17), "x.zip", "abc", "a | b", log)
    assert "a / b" in log.read_text(encoding="utf-8")
