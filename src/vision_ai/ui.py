"""화면 밀도와 사이드바 내비게이션.

Streamlit 기본 테마는 발표 슬라이드에 가까운 크기라, 데이터를 보면서 실제로 작업할 때는
한 화면에 들어오는 정보가 너무 적다. 이 앱은 "초보자가 계속 쓸 수 있는 환경"이 목표이므로
화면 밀도를 작업용에 맞춘다.

역할 분담:

* **글자 크기**는 ``.streamlit/config.toml``의 테마 옵션으로 줄인다. Streamlit이 공식
  지원하는 설정이라 버전이 올라가도 깨지지 않는다.
* **여백과 레일 폭**은 테마 옵션으로 다룰 수 없어 이 모듈에서 CSS로 보정한다. 선택자는
  ``data-testid``만 사용한다 — emotion 해시 클래스(``st-emotion-cache-...``)는 빌드마다
  바뀌므로 절대 쓰지 않는다.

내비게이션은 Streamlit 기본 메뉴(``position="sidebar"``) 대신 ``position="hidden"`` +
``st.page_link``로 직접 그린다. 기본 메뉴는 사이드바를 접으면 통째로 사라져서 "지금 몇
단계인지"를 잃어버리는데, 직접 그리면 접었을 때 아이콘만 남기는 레일 모드를 만들 수 있다.
"""

from __future__ import annotations

from typing import Sequence

import streamlit as st

RAIL_KEY = "nav_rail"
"""사이드바가 레일(아이콘 전용) 모드인지 나타내는 세션 상태 키."""

RAIL_WIDTH_PX = 74
"""레일 모드 사이드바 폭. 아이콘 버튼과 좌우 여백이 겨우 들어가는 최소치."""


def rail_enabled() -> bool:
    """사이드바가 레일 모드인지."""
    return bool(st.session_state.get(RAIL_KEY, False))


def _toggle_rail() -> None:
    st.session_state[RAIL_KEY] = not rail_enabled()


# 본문 여백. 기본값은 상단 90px / 하단 150px로, 스크롤을 한 번 더 하게 만드는 주범이다.
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
/* 아이콘만 남으면 기본 1rem 간격이 지나치게 벌어진다. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
  gap: 0.25rem;
}}
/* 토글 버튼과 메뉴 사이에는 구분을 위해 간격을 남긴다. */
[data-testid="stSidebar"] [data-testid="stButton"] {{
  margin-bottom: 0.75rem;
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
/* 토글 버튼도 같은 정사각형으로 맞춰 세로 리듬을 유지한다. */
[data-testid="stSidebar"] [data-testid="stButton"] button {{
  height: 42px;
  padding: 0;
}}
/* 폭이 좁아진 만큼 아이콘을 키워 알아보기 쉽게 한다. */
[data-testid="stSidebar"] [data-testid="stIconEmoji"] {{
  font-size: 1.3rem;
}}
"""


def apply_chrome() -> None:
    """화면 밀도 CSS를 적용한다. 페이지 본문을 그리기 전에 한 번 호출한다."""
    css = _DENSITY_CSS + (_RAIL_CSS if rail_enabled() else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_nav(pages: Sequence[st.Page], *, captions: Sequence[str] = ()) -> None:
    """사이드바에 내비게이션을 그린다.

    펼친 모드에서는 아이콘 + 단계 이름, 레일 모드에서는 아이콘만 남는다. 레일 모드에서도
    현재 페이지를 알 수 있어야 하므로, 마우스를 올리면 단계 이름이 툴팁으로 뜬다.
    """
    rail = rail_enabled()
    with st.sidebar:
        st.button(
            "" if rail else "메뉴 접기",
            icon=":material/left_panel_open:" if rail else ":material/left_panel_close:",
            on_click=_toggle_rail,
            help="메뉴를 아이콘만 남기고 접습니다." if not rail else "메뉴를 펼칩니다.",
            width="stretch",
            key="nav_rail_toggle",
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
