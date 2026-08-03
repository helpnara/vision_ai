"""화면 밀도와 사이드바 내비게이션.

Streamlit 기본 테마는 발표 슬라이드에 가까운 크기라, 데이터를 보면서 실제로 작업할 때는
한 화면에 들어오는 정보가 너무 적다. 이 앱은 "초보자가 계속 쓸 수 있는 환경"이 목표이므로
화면 밀도를 작업용에 맞춘다.

역할 분담:

* **글자 크기**는 ``.streamlit/config.toml``의 테마 옵션으로 줄인다. Streamlit이 공식
  지원하는 설정이라 버전이 올라가도 깨지지 않는다.
* **여백·레일 폭·탭 줄바꿈**은 테마 옵션으로 다룰 수 없어 이 모듈에서 CSS로 보정한다.

선택자 규칙: ``data-testid`` 또는 위젯 key가 만드는 ``st-key-*`` 클래스만 쓴다. 둘 다
Streamlit이 밖으로 약속한 표식이다. emotion 해시 클래스(``st-emotion-cache-...``)는
빌드마다 바뀌므로 절대 쓰지 않는다 — 쓰면 버전을 올리는 순간 조용히 깨진다.

내비게이션은 Streamlit 기본 메뉴(``position="sidebar"``) 대신 ``position="hidden"`` +
``st.page_link``로 직접 그린다. 기본 메뉴는 사이드바를 접으면 통째로 사라져서 "지금 몇
단계인지"를 잃어버리는데, 직접 그리면 접었을 때 아이콘만 남기는 레일 모드를 만들 수 있다.

접기 버튼은 하나뿐이다. Streamlit 기본 « 버튼은 패널을 통째로 숨겨서 레일 모드와 목적이
겹치므로 숨기고, 그 자리에 레일 토글을 놓는다. 사용자 입장에서는 "« 를 누르면 아이콘만
남는다" 하나의 규칙만 남는다.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd
import streamlit as st

from vision_ai import glossary

RAIL_KEY = "nav_rail"
"""사이드바가 레일(아이콘 전용) 모드인지 나타내는 세션 상태 키."""

RAIL_WIDTH_PX = 74
"""레일 모드 사이드바 폭. 아이콘 버튼과 좌우 여백이 겨우 들어가는 최소치."""

TOGGLE_KEY = "nav_rail_toggle"
"""접기 토글 버튼의 key. ``st-key-`` 클래스로 CSS에서 이 버튼만 집어낸다."""


def rail_enabled() -> bool:
    """사이드바가 레일 모드인지."""
    return bool(st.session_state.get(RAIL_KEY, False))


def _toggle_rail() -> None:
    st.session_state[RAIL_KEY] = not rail_enabled()


# 본문 여백과 공통 밀도. 기본값은 상단 90px / 하단 150px로, 스크롤을 한 번 더 하게 만든다.
_DENSITY_CSS = """
[data-testid="stMainBlockContainer"] {
  padding-top: 2.5rem;
  padding-bottom: 3rem;
}
/* 구분선이 위아래로 2rem씩 먹는다. 섹션 구분은 유지하되 절반으로 줄인다. */
[data-testid="stMainBlockContainer"] hr {
  margin-top: 1rem;
  margin-bottom: 1rem;
}
/* 지표 라벨과 값 사이 간격 축소 — 지표를 여러 개 나란히 놓는 화면이 많다. */
[data-testid="stMetricLabel"] {
  margin-bottom: 0;
}
/* 탭이 폭을 넘으면 Streamlit은 가로 스크롤 화살표를 붙인다. 4단계는 탭이 7개라 좁은
   화면에서 뒤쪽 탭이 밀려나 "없는 것"처럼 보인다. 줄바꿈으로 바꿔 항상 전부 보이게 한다. */
[data-testid="stTabs"] [role="tablist"] {
  flex-wrap: wrap;
  row-gap: 0.25rem;
  overflow-x: visible;
}
/* 줄바꿈하면 스크롤 화살표는 가릴 것이 없어 자리만 차지한다. */
[data-testid="stTabsScrollLeft"],
[data-testid="stTabsScrollRight"] {
  display: none;
}
/* 태블릿·모바일: 좌우 75px 여백은 이 폭에서 본문을 절반으로 깎는다. */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] {
    padding-left: 1rem;
    padding-right: 1rem;
    padding-top: 1.5rem;
  }
}
"""

MOBILE_MAX_PX = 640
"""이 폭 이하에서는 사이드바를 오버레이로 쓴다. 레일(74px)도 화면의 1/5을 먹기 때문이다."""

# 접기 버튼 자리 잡기. 레일이든 아니든 항상 적용된다.
#
# 화면 폭에 따라 접기 버튼의 의미가 다르다. 넓은 화면에서는 "레일로 줄이기"가 맞지만,
# 폰에서는 사이드바가 본문을 덮으므로 "완전히 치우기"가 맞다. 그래서 넓은 화면에서만
# 기본 « 버튼을 숨기고 레일 토글로 대체하고, 좁은 화면에서는 반대로 한다. 어느 쪽이든
# 화면에 보이는 접기 버튼은 항상 하나뿐이다.
_NAV_CSS = f"""
@media (min-width: {MOBILE_MAX_PX + 1}px) {{
  /* 기본 « 버튼은 패널을 통째로 숨겨 레일 모드와 목적이 겹친다. 버튼은 하나만 남긴다. */
  [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
    display: none;
  }}
  /* 레일 토글을 기본 « 버튼이 있던 헤더 자리로 올린다. 흐름에서 빼야 메뉴 항목이
     헤더 아래부터 곧바로 시작한다. 사이드바에는 position:relative가 걸려 있다. */
  [data-testid="stSidebar"] .st-key-{TOGGLE_KEY} {{
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    left: auto;
    width: auto;
    z-index: 2;
  }}
}}
@media (max-width: {MOBILE_MAX_PX}px) {{
  /* 폰에서는 레일로 줄여봐야 본문이 좁다. 기본 « (완전히 치우기)를 그대로 쓰고
     레일 토글은 감춘다 — 치운 사이드바를 다시 여는 버튼은 Streamlit이 본문 쪽에 그린다. */
  [data-testid="stSidebar"] .st-key-{TOGGLE_KEY} {{
    display: none;
  }}
}}
"""

# 레일 모드에서만 적용. 사이드바 폭은 인라인 style로 붙어 있어 !important가 필요하다.
#
# 링크를 정사각형으로 늘리는 것이 핵심이다. 라벨을 비우면 링크가 아이콘 폭으로 줄어드는데,
# 현재 페이지 표시(배경색)는 Streamlit이 링크에 칠하므로 링크가 줄면 표시도 같이 줄어
# 아이콘 옆 얇은 조각만 남는다. "접어도 현재 단계가 보인다"가 레일 모드의 목적이므로
# 링크를 폭 전체로 펴서 표시가 아이콘을 감싸게 한다.
_RAIL_CSS = f"""
[data-testid="stSidebar"] {{
  width: {RAIL_WIDTH_PX}px !important;
  min-width: {RAIL_WIDTH_PX}px !important;
}}
/* 폭을 고정했으므로 사이드바 드래그 손잡이는 숨긴다. */
[data-testid="stSidebar"] [data-testid="stSidebarResizeHandle"] {{
  display: none;
}}
/* 사이드바 기본 좌우 여백은 300px 폭 기준이라(합쳐서 52px) 레일에서는 내용이 들어갈
   자리가 남지 않는다. 여백을 줄여야 아이콘 버튼이 폭 전체로 펴진다. */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
  padding-left: 8px;
  padding-right: 8px;
}}
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
  padding-left: 0;
  padding-right: 0;
  padding-bottom: 1rem;
}}
/* 좁아진 폭에서는 토글을 오른쪽 끝이 아니라 가운데 둔다. */
[data-testid="stSidebar"] .st-key-{TOGGLE_KEY} {{
  left: 0;
  right: 0;
  margin: 0 auto;
}}
/* 아이콘만 남으면 기본 1rem 간격이 지나치게 벌어진다. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: 0.25rem;
}}
[data-testid="stSidebar"] [data-testid="stPageLink-NavLink"] {{
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 42px;
  padding: 0;
  border-radius: 8px;
}}
/* 폭이 좁아진 만큼 아이콘을 키워 알아보기 쉽게 한다. */
[data-testid="stSidebar"] [data-testid="stIconEmoji"] {{
  font-size: 1.3rem;
}}
"""


def apply_chrome() -> None:
    """화면 밀도 CSS를 적용한다. 페이지 본문을 그리기 전에 한 번 호출한다."""
    css = _DENSITY_CSS + _NAV_CSS + (_RAIL_CSS if rail_enabled() else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_nav(pages: Sequence[st.Page], *, captions: Sequence[str] = ()) -> None:
    """사이드바에 내비게이션을 그린다.

    펼친 모드에서는 아이콘 + 단계 이름, 레일 모드에서는 아이콘만 남는다. 레일 모드에서도
    현재 페이지를 알 수 있어야 하므로, 마우스를 올리면 단계 이름이 툴팁으로 뜬다.
    """
    rail = rail_enabled()
    with st.sidebar:
        st.button(
            "",
            icon=(
                ":material/keyboard_double_arrow_right:"
                if rail
                else ":material/keyboard_double_arrow_left:"
            ),
            on_click=_toggle_rail,
            help="메뉴를 펼칩니다." if rail else "메뉴를 아이콘만 남기고 접습니다.",
            type="tertiary",
            key=TOGGLE_KEY,
        )
        for page in pages:
            st.page_link(
                page,
                label="" if rail else page.title,
                icon=page.icon or None,
                help=page.title if rail else None,
                width="stretch",
            )
        if not rail and captions:
            st.divider()
            for line in captions:
                st.caption(line)


# --- 표 (G8·G9) -------------------------------------------------------------

def table_columns(
    frame: pd.DataFrame, *, overrides: Mapping[str, object] | None = None
) -> dict:
    """표의 열 도움말을 용어 사전에서 자동으로 붙인다.

    지표를 지표 카드에서는 캡션으로 설명해 놓고 표에서는 맨 이름만 내보내면, 같은 값을
    화면마다 다르게 만나게 된다. 열 이름을 사전에서 찾아 설명이 있으면 붙이고, 없으면
    건드리지 않는다 — 뜻이 분명한 열에 굳이 설명을 다는 것은 소음이다.

    ``overrides``로 넘긴 열 설정은 그대로 우선한다.
    """
    config: dict[str, object] = {}
    for name in frame.columns:
        text = glossary.column_help(str(name))
        if not text:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            config[name] = st.column_config.NumberColumn(help=text)
        else:
            config[name] = st.column_config.TextColumn(help=text)
    config.update(overrides or {})
    return config


CARD_BREAKPOINT_PX = 640
"""이 폭 이하에서는 표 대신 카드로 보여준다."""

CARD_LIMIT = 12
"""카드로 펼칠 최대 줄 수. 긴 표를 카드로 늘어놓으면 스크롤이 끝없이 길어진다."""


def responsive_table(
    frame: pd.DataFrame,
    *,
    key: str,
    title_column: str | None = None,
    column_config: Mapping[str, object] | None = None,
    card_limit: int = CARD_LIMIT,
    **dataframe_kwargs,
) -> None:
    """넓은 화면에서는 표, 좁은 화면에서는 카드로 보여준다.

    폰에서 표는 가로 스크롤로 볼 수는 있지만, 한 줄을 읽으려면 좌우로 오가야 해서 실제로는
    잘 안 보게 된다. 줄 하나를 통째로 세워 보여주는 편이 낫다.

    표와 카드를 둘 다 그려 놓고 CSS로 하나만 보인다. Streamlit은 서버에서 화면을 그리므로
    브라우저 폭을 모르기 때문이다. 줄 수가 적은 표에만 쓸 것 — 로그처럼 긴 표에 쓰면
    쓰지도 않을 카드를 수백 개 그리게 된다.
    """
    if frame.empty:
        return
    title_column = title_column or str(frame.columns[0])

    with st.container(key=f"{key}__table"):
        st.dataframe(frame, column_config=dict(column_config or {}), **dataframe_kwargs)

    with st.container(key=f"{key}__cards"):
        shown = frame.head(card_limit)
        for _, row in shown.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row[title_column]}**")
                lines = [
                    f"- {name}: {row[name]}"
                    for name in frame.columns
                    if name != title_column and str(row[name]).strip()
                ]
                st.markdown("\n".join(lines))
        if len(frame) > card_limit:
            st.caption(
                f"{len(frame):,}건 중 {card_limit}건만 카드로 봅니다. "
                "화면을 넓히면 전체가 표로 보입니다."
            )

    st.markdown(
        f"""<style>
@media (max-width: {CARD_BREAKPOINT_PX}px) {{
  .st-key-{key}__table {{ display: none; }}
}}
@media (min-width: {CARD_BREAKPOINT_PX + 1}px) {{
  .st-key-{key}__cards {{ display: none; }}
}}
</style>""",
        unsafe_allow_html=True,
    )
