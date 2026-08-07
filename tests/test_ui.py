"""화면 밀도와 사이드바 레일 테스트.

레일 모드의 존재 이유는 하나다 — **메뉴를 접어도 지금 몇 단계인지 보여야 한다**. 그래서
"접으면 폭이 줄어든다"만 확인해서는 부족하고, 접은 상태에서도 페이지 링크가 그대로 남아
있는지, 현재 위치를 알려줄 툴팁이 붙는지까지 확인한다.

CSS는 ``data-testid``만 써야 한다. Streamlit의 ``st-emotion-cache-...`` 클래스는 빌드마다
바뀌므로, 거기에 기대면 버전을 올리는 순간 조용히 깨진다. 이 규칙도 테스트로 잠근다.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from vision_ai import ui


# --- 레일 상태 -------------------------------------------------------------

def test_rail_is_off_by_default():
    """처음 열면 단계 이름이 보여야 한다. 초보자가 아이콘만 보고 시작할 수는 없다."""
    at = AppTest.from_file("app.py")
    at.run()
    assert ui.RAIL_KEY not in at.session_state
    assert all(link.label for link in _page_links(at))


def test_toggle_switches_to_rail_and_back():
    at = AppTest.from_file("app.py")
    at.run()
    at.sidebar.button[0].click().run()
    assert at.session_state[ui.RAIL_KEY] is True
    at.sidebar.button[0].click().run()
    assert at.session_state[ui.RAIL_KEY] is False


# --- 사이드바 내비게이션 ----------------------------------------------------

def _page_links(at):
    return [e for e in at.sidebar if e.type == "page_link"]


def test_every_page_is_reachable_from_the_sidebar():
    at = AppTest.from_file("app.py")
    at.run()
    labels = [link.label for link in _page_links(at)]
    assert "1. 데이터 수집" in labels
    assert "4. 운영 관리 (MLOps)" in labels
    assert len(labels) == 6


def test_rail_keeps_every_page_link():
    """접었을 때 링크가 사라지면 '현재 단계 확인'이라는 목적 자체가 없어진다."""
    at = AppTest.from_file("app.py")
    at.run()
    before = len(_page_links(at))
    at.sidebar.button[0].click().run()
    after = _page_links(at)
    assert len(after) == before
    assert all(link.label == "" for link in after), "레일에서는 라벨이 숨겨져야 한다"


def test_rail_puts_the_step_name_in_a_tooltip():
    """라벨을 숨기는 대신 이름을 알 방법은 남겨둬야 한다."""
    at = AppTest.from_file("app.py")
    at.run()
    at.sidebar.button[0].click().run()
    helps = [link.help for link in _page_links(at)]
    assert "3. 모델 개발 · 평가" in helps


def test_captions_are_dropped_in_rail_mode():
    """74px 폭에 두 줄짜리 설명을 밀어넣으면 읽을 수 없게 뭉갠다."""
    at = AppTest.from_file("app.py")
    at.run()
    assert any("사내 데이터 미사용" in c.value for c in at.sidebar.caption)
    at.sidebar.button[0].click().run()
    assert not any("사내 데이터 미사용" in c.value for c in at.sidebar.caption)


# --- CSS 규칙 --------------------------------------------------------------

def test_rail_css_is_only_emitted_in_rail_mode():
    at = AppTest.from_file("app.py")
    at.run()
    expanded = "".join(m.value for m in at.main.markdown)
    assert "stSidebarResizeHandle" not in expanded
    at.sidebar.button[0].click().run()
    railed = "".join(m.value for m in at.main.markdown)
    assert f"{ui.RAIL_WIDTH_PX}px" in railed


def test_density_css_is_always_applied():
    """글자를 줄여도 여백이 그대로면 화면에 들어오는 정보량은 그대로다."""
    at = AppTest.from_file("app.py")
    at.run()
    css = "".join(m.value for m in at.main.markdown)
    assert "stMainBlockContainer" in css


@pytest.mark.parametrize("css", [ui._DENSITY_CSS, ui._NAV_CSS, ui._RAIL_CSS])
def test_css_never_depends_on_emotion_class_names(css):
    """st-emotion-cache-* 는 빌드마다 바뀌는 값이라 선택자로 쓰면 안 된다."""
    assert "st-emotion-cache" not in css


def test_css_selectors_use_only_stable_hooks():
    """data-testid와 위젯 key가 만드는 st-key-* 만 허용한다. 둘 다 Streamlit이 밖으로
    약속한 표식이고, 그 밖의 선택자는 버전이 올라가면 조용히 어긋난다."""
    combined = ui._DENSITY_CSS + ui._NAV_CSS + ui._RAIL_CSS
    selectors = re.findall(r"^\s*([^@{}\n][^{\n]*)\{", combined, flags=re.MULTILINE)
    assert selectors, "선택자를 하나도 못 찾았다면 테스트가 잘못된 것이다"
    for selector in selectors:
        assert "data-testid" in selector or "st-key-" in selector, (
            f"안정적이지 않은 선택자: {selector.strip()}"
        )


# --- 접기 버튼은 하나뿐 -----------------------------------------------------

def test_only_one_collapse_button_is_rendered():
    """기본 « 버튼과 레일 토글이 나란히 보이면 같은 자리에 기능이 둘이라 헷갈린다.
    넓은 화면에서는 기본 버튼을 숨기고, 좁은 화면에서는 반대로 레일 토글을 숨긴다."""
    wide = f"@media (min-width: {ui.MOBILE_MAX_PX + 1}px)"
    narrow = f"@media (max-width: {ui.MOBILE_MAX_PX}px)"
    assert wide in ui._NAV_CSS and narrow in ui._NAV_CSS
    wide_block = ui._NAV_CSS.split(wide)[1].split(narrow)[0]
    narrow_block = ui._NAV_CSS.split(narrow)[1]
    assert "stSidebarCollapseButton" in wide_block
    assert f"st-key-{ui.TOGGLE_KEY}" in narrow_block


def test_the_toggle_keeps_its_key():
    """CSS가 st-key-* 로 이 버튼을 집어내므로 key가 바뀌면 자리 잡기가 통째로 깨진다."""
    at = AppTest.from_file("app.py")
    at.run()
    assert at.sidebar.button[0].key == ui.TOGGLE_KEY
    assert f"st-key-{ui.TOGGLE_KEY}" in ui._NAV_CSS


def test_the_toggle_has_no_text_label():
    """헤더 자리에 올라가므로 아이콘만 남아야 한다. 설명은 툴팁으로 준다."""
    at = AppTest.from_file("app.py")
    at.run()
    button = at.sidebar.button[0]
    assert button.label == ""
    assert button.help


# --- 좁은 화면 (G5·G6) ------------------------------------------------------

def test_tabs_wrap_instead_of_scrolling():
    """4단계는 탭이 7개다. 가로 스크롤이면 뒤쪽 탭이 화면 밖으로 밀려 '없는 것'이 된다."""
    assert "flex-wrap: wrap" in ui._DENSITY_CSS
    assert "stTabsScrollRight" in ui._DENSITY_CSS


def test_narrow_screens_get_smaller_side_padding():
    assert "@media (max-width: 900px)" in ui._DENSITY_CSS


# --- 표의 열 도움말 (G8) ----------------------------------------------------

def test_known_metric_columns_get_help_from_the_glossary():
    """지표를 지표 카드에서는 설명하고 표에서는 안 하면 같은 값을 다르게 만나게 된다."""
    frame = pd.DataFrame({"recall": [0.9], "precision": [0.4], "fn": [3]})
    config = ui.table_columns(frame)
    assert set(config) == {"recall", "precision", "fn"}


def test_columns_without_anything_to_explain_are_left_alone():
    """뜻이 분명한 열에 굳이 설명을 다는 것은 소음이다."""
    frame = pd.DataFrame({"version": ["v001"], "created_at": ["2026-08-03"]})
    assert ui.table_columns(frame) == {}


def test_overrides_win_over_the_glossary():
    frame = pd.DataFrame({"recall": [0.9]})
    mine = st.column_config.TextColumn(help="직접 쓴 설명")
    assert ui.table_columns(frame, overrides={"recall": mine})["recall"] is mine


# --- 좁은 화면의 표 (G9) ----------------------------------------------------

def _table_script():
    import pandas as pd
    import streamlit as st

    from vision_ai import ui

    ui.responsive_table(
        pd.DataFrame({"이름": ["가", "나"], "값": [1, 2]}),
        key="demo",
        title_column="이름",
        hide_index=True,
    )
    st.write("끝")


def test_responsive_table_draws_both_the_table_and_the_cards():
    """서버는 브라우저 폭을 모른다. 둘 다 그려 놓고 CSS로 하나만 보여준다."""
    at = AppTest.from_function(_table_script)
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 1
    body = "".join(m.value for m in at.markdown)
    assert "**가**" in body and "**나**" in body


def test_responsive_table_emits_the_swap_rules():
    at = AppTest.from_function(_table_script)
    at.run()
    css = "".join(m.value for m in at.markdown if "<style>" in m.value)
    assert f"max-width: {ui.CARD_BREAKPOINT_PX}px" in css
    assert "st-key-demo__table" in css
    assert "st-key-demo__cards" in css


def test_responsive_table_skips_an_empty_frame():
    at = AppTest.from_function(
        lambda: ui.responsive_table(pd.DataFrame(), key="empty")
    )
    at.run()
    assert not at.exception
    assert len(at.dataframe) == 0


def _long_table_script():
    import pandas as pd

    from vision_ai import ui

    ui.responsive_table(
        pd.DataFrame({"이름": [str(i) for i in range(40)], "값": range(40)}),
        key="long",
        card_limit=5,
    )


def test_long_tables_do_not_become_endless_card_stacks():
    at = AppTest.from_function(_long_table_script)
    at.run()
    titles = [m.value for m in at.markdown if m.value.startswith("**")]
    assert len(titles) == 5
    assert any("40건 중 5건" in c.value for c in at.caption)


# --- 테마 설정 -------------------------------------------------------------

def test_heading_sizes_come_from_config_not_css():
    """글자 크기는 Streamlit이 공식 지원하는 테마 옵션으로 다뤄야 버전 업에 안전하다."""
    assert "font-size" not in ui._DENSITY_CSS
    sizes = st.get_option("theme.headingFontSizes")
    assert sizes, "config.toml에 headingFontSizes가 없다"
    assert sizes[0] != "2.75rem", "h1이 기본값 그대로다"


def test_theme_follows_the_viewers_light_dark_setting():
    """base를 고정하면 어둡게 쓰는 사람에게도 흰 화면만 나온다 (G7)."""
    assert not st.get_option("theme.base")


def test_dark_mode_brightens_the_accent_color():
    """#2563eb는 어두운 배경에서 묻힌다. 어두운 테마용 값이 따로 있어야 한다."""
    light = st.get_option("theme.primaryColor")
    dark = st.get_option("theme.dark.primaryColor")
    assert dark and dark != light


# --- 화면을 여는 값 (G11) ---------------------------------------------------

@pytest.mark.parametrize(
    "page",
    [
        "app_pages/home.py",
        "app_pages/p1_ingest.py",
        "app_pages/p2_labeling.py",
        "app_pages/p3_modeling.py",
        "app_pages/p4_operations.py",
        "app_pages/p5_settings.py",
    ],
)
def test_opening_a_page_does_no_heavy_work(page):
    """화면을 여는 것만으로 전량 특징 추출이 일어나면 안 된다.

    3단계 베이스라인 탭이 버튼을 누르기도 전에 라벨된 이미지 전량의 특징을 뽑고 있었다.
    4,584장 기준 화면 하나 여는 데 92초가 걸렸고, 탭을 누를 때마다 다시 걸렸다.
    무거운 계산은 사용자가 버튼을 누른 뒤에만 해야 한다.
    """
    import time

    at = AppTest.from_file("app.py", default_timeout=120)
    at.run()
    started = time.time()
    at.switch_page(page)
    at.run()
    elapsed = time.time() - started
    assert not at.exception
    assert elapsed < 10, f"{page} 여는 데 {elapsed:.1f}초 — 버튼 누르기 전에 무거운 계산을 하고 있다"


# --- 오래 걸리는 작업 알리기 (G12) ------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0초"), (9.4, "9초"), (59, "59초"), (60, "1분"), (92.3, "1분 32초"), (600, "10분")],
)
def test_durations_are_written_the_way_people_say_them(seconds, expected):
    """'92.3초'보다 '1분 32초'가 기다릴지 말지 판단하기 쉽다."""
    assert ui._duration(seconds) == expected


def test_no_estimate_until_there_is_enough_to_measure():
    """처음 몇 건은 편차가 커서, 그걸로 계산한 남은 시간은 사람을 속인다."""
    assert ui._eta(ui.MIN_SAMPLES_FOR_ETA - 1, 1000, elapsed=1.0) is None


def test_estimate_scales_with_what_is_left():
    """100건에 10초 걸렸으면 남은 900건은 90초다."""
    assert ui._eta(100, 1000, elapsed=10.0) == pytest.approx(90.0)


def test_no_estimate_once_the_job_is_done():
    assert ui._eta(1000, 1000, elapsed=10.0) is None


def test_no_estimate_before_any_time_has_passed():
    """경과 0초로 나누면 0초 남았다고 말하게 된다."""
    assert ui._eta(100, 1000, elapsed=0.0) is None


def _progress_script():
    from vision_ai import ui

    bar = ui.Progress("특징 추출 중", note="처음 계산하는 이미지라 시간이 걸립니다.")
    for done in range(1, 31):
        bar.update(done, 30)


def test_progress_says_why_it_is_slow_while_it_runs():
    """막대만 채우면 화면이 멈춘 것처럼 보인다. 이유를 먼저 말해야 한다."""
    at = AppTest.from_function(_progress_script)
    at.run()
    assert not at.exception
    assert any("시간이 걸립니다" in c.value for c in at.caption)


def test_progress_reports_how_long_it_actually_took():
    def script():
        from vision_ai import ui

        bar = ui.Progress("특징 추출 중", note="계산 중")
        bar.update(5, 5)
        bar.done("특징 준비 완료")

    at = AppTest.from_function(script)
    at.run()
    captions = [c.value for c in at.caption]
    assert any("특징 준비 완료" in c and "걸림" in c for c in captions)
    assert not any("계산 중" == c for c in captions), "끝나면 진행 안내는 치워야 한다"


def test_progress_survives_a_zero_length_job():
    at = AppTest.from_function(lambda: ui.Progress("작업 중").update(0, 0))
    at.run()
    assert not at.exception


# --- 이미지 위에서 영역 지정 (G15) ------------------------------------------

def _state(xs, ys):
    return {"selection": {"roi": {"x": list(xs), "y": list(ys)}}}


def test_no_box_before_anything_is_drawn():
    st.session_state.clear()
    assert ui.roi_box("없는키", 100, 100) is None


def test_drag_becomes_pixel_coordinates():
    st.session_state["k"] = _state([10.4, 60.6], [20.2, 50.9])
    assert ui.roi_box("k", 200, 200) == (10, 20, 50, 31)


def test_dragging_right_to_left_gives_the_same_box():
    """오른쪽에서 왼쪽으로 끌면 좌표가 뒤집혀 온다. 폭이 음수가 되면 안 된다."""
    st.session_state["k"] = _state([80, 20], [70, 30])
    assert ui.roi_box("k", 200, 200) == (20, 30, 60, 40)


def test_dragging_past_the_edge_is_clipped_to_the_image():
    """Vega는 축 밖으로도 끌 수 있다. 이미지 밖 좌표를 저장하면 크롭이 깨진다."""
    st.session_state["k"] = _state([-30, 250], [-10, 400])
    assert ui.roi_box("k", 200, 150) == (0, 0, 200, 150)


def test_a_click_without_dragging_is_not_a_box():
    """살짝 누르기만 해도 선택이 생긴다. 0픽셀짜리 영역을 저장하면 안 된다."""
    st.session_state["k"] = _state([40.1, 40.2], [50.0, 50.4])
    assert ui.roi_box("k", 200, 200) is None


def test_a_cleared_selection_reads_as_nothing():
    st.session_state["k"] = {"selection": {"roi": {}}}
    assert ui.roi_box("k", 200, 200) is None


def _picker_script():
    import numpy as np

    from vision_ai import ui

    rgb = np.zeros((120, 200, 3), dtype=np.uint8)
    ui.roi_picker(rgb, key="pick", width=200, height=120)


def test_picker_renders_without_a_new_dependency():
    """Vega-Lite 구간 선택은 Streamlit에 들어 있다. 캔버스 컴포넌트를 새로 깔지 않는다."""
    at = AppTest.from_function(_picker_script)
    at.run()
    assert not at.exception


def test_picker_keeps_coordinates_in_original_pixels():
    """화면에는 줄여 그리더라도 좌표는 원본 픽셀이어야 한다 — 축 도메인을 원본으로 둔다."""
    import numpy as np

    rgb = np.zeros((1070, 1404, 3), dtype=np.uint8)
    uri = ui._data_uri(rgb)
    assert uri.startswith("data:image/jpeg;base64,")


def test_transport_image_is_shrunk():
    """원본을 그대로 요청에 실으면 무거워진다."""
    import numpy as np

    big = np.zeros((2000, 4000, 3), dtype=np.uint8)
    small = np.zeros((100, 200, 3), dtype=np.uint8)
    assert len(ui._data_uri(big)) < len(ui._data_uri(big, max_width=4000))
    assert ui._data_uri(small)  # 상한보다 작으면 그대로
