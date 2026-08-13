"""프로젝트 폴더 이름 (S1).

한글 이름을 지워 버리면 «라인1 검사»가 통째로 `project`가 된다. 프로젝트가 서너 개만
돼도 `project-2`, `project-3`이 쌓여서 **폴더만 봐서는 어느 현장인지 알 수 없다.**
설정 화면의 "데이터 위치"도 마찬가지고, 백업을 뒤질 때는 더 곤란하다.

그래서 지우지 말고 로마자로 옮긴다. 여기서 지키는 것은 «표기법이 맞는가»가 아니라
**«읽어서 알아볼 수 있고, 서로 겹치지 않는가»**이다.
"""

from __future__ import annotations

import pytest

from vision_ai import config, projects


# --- 읽어서 알아볼 수 있는가 --------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        ("라인1 검사", "rain1-geomsa"),
        ("A라인 표면검사", "arain-pyomyeongeomsa"),
        ("Line 1 QA", "line-1-qa"),
        ("용접부 외관검사 2026", "yongjeobbu-oegwangeomsa-2026"),
    ],
)
def test_a_korean_name_survives_as_readable_letters(name, expected):
    assert projects.slugify(name) == expected


def test_a_korean_only_name_does_not_collapse_to_the_fallback():
    """**이것이 이 항목의 전부다.** 예전에는 한글만 있는 이름이 전부 `project`가 됐다."""
    assert projects.slugify("검사") != projects.SLUG_FALLBACK


def test_two_different_korean_names_get_different_folders():
    assert projects.slugify("1라인") != projects.slugify("2라인")


def test_the_ascii_part_of_a_mixed_name_is_kept():
    """«V3 확인용»에서 V3만 남기고 버리면 어느 V3인지 알 수 없다."""
    slug = projects.slugify("V3 확인용")
    assert slug.startswith("v3") and len(slug) > len("v3")


# --- 폴더 이름으로 쓸 수 있는가 -----------------------------------------------

def test_a_slug_is_always_safe_for_a_folder_name():
    """OS·인코딩에 따라 깨지는 글자가 남으면 데이터를 쓰다가 실패한다."""
    for name in ["라인1 검사", "漢字", "🙂 라인", "a/b\\c", "..", "  공백  "]:
        slug = projects.slugify(name)
        assert slug and slug.strip("-") == slug
        assert all(ch.isalnum() and ch.isascii() or ch == "-" for ch in slug), slug


def test_a_slug_does_not_grow_without_bound():
    """로마자로 옮기면 글자 수가 두세 배로 늘어난다. 폴더 이름 길이 제한에 걸리면 안 된다."""
    assert len(projects.slugify("표면결함검사" * 20)) <= 40


def test_a_name_with_nothing_usable_still_gets_a_distinct_folder():
    """한자·이모지만 있는 이름도 서로 구분돼야 한다 — 안 그러면 `project-2`가 다시 쌓인다."""
    assert projects.slugify("漢字") != projects.slugify("🙂")
    assert projects.slugify("漢字").startswith(projects.SLUG_FALLBACK)


def test_the_same_name_always_gives_the_same_folder():
    """실수로 지운 프로젝트를 같은 이름으로 다시 만들면 데이터가 그대로 돌아와야 한다."""
    assert projects.slugify("漢字") == projects.slugify("漢字")


def test_an_empty_name_falls_back():
    assert projects.slugify("") == projects.SLUG_FALLBACK


# --- 로마자 옮김 자체 ---------------------------------------------------------

def test_a_final_consonant_is_not_dropped():
    """받침을 빼면 «검사»와 «거사»가 같은 폴더가 된다."""
    assert projects.romanize("검") != projects.romanize("거")


def test_non_hangul_characters_are_left_alone():
    assert projects.romanize("QA 1-2") == "QA 1-2"


def test_a_silent_initial_is_not_written_out():
    assert projects.romanize("아") == "a"


# --- 이미 있는 프로젝트는 건드리지 않는다 -------------------------------------

def test_the_default_project_keeps_its_folder():
    """기본 프로젝트의 slug가 바뀌면 이미 쌓인 데이터가 통째로 떨어져 나간다."""
    registry = projects._default_registry()
    assert registry.projects[0].slug == config.DEFAULT_PROJECT
    assert registry.active == config.DEFAULT_PROJECT
