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

import base64
import time
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


# --- 오래 걸리는 작업 알리기 (G12) ------------------------------------------

MIN_SAMPLES_FOR_ETA = 12
"""남은 시간을 말하기 전에 재어 볼 최소 건수. 처음 몇 건은 들쭉날쭉해 값을 못 믿는다."""


class Progress:
    """오래 걸리는 작업의 진행을 알린다.

    막대만 채우면 "얼마나 더 기다려야 하는지"를 알 수 없다. 특징 추출은 4,584장 기준
    70초가 넘고, 그동안 화면이 멈춘 것처럼 보이면 사람은 새로고침을 누른다. 그래서
    **왜 오래 걸리는지**를 먼저 말하고, 진행하면서 **남은 시간**을 실측으로 갱신한다.

    남은 시간은 지금까지의 평균 속도로 추정한다. 처음 몇 건은 편차가 커서 믿을 수 없으므로
    ``MIN_SAMPLES_FOR_ETA``건을 처리하기 전에는 건수만 보여준다.
    """

    def __init__(self, label: str, *, note: str = "") -> None:
        self.label = label
        self.note = note
        self._bar = st.progress(0.0, text=self._text(0, 0, None))
        self._note_slot = st.caption(note) if note else None
        self._started = time.monotonic()

    def _text(self, done: int, total: int, remaining: float | None) -> str:
        head = f"{self.label} {done:,}/{total:,}" if total else self.label
        if remaining is None:
            return head
        return f"{head} · 남은 시간 약 {_duration(remaining)}"

    def update(self, done: int, total: int) -> None:
        if total <= 0:
            return
        remaining = _eta(done, total, time.monotonic() - self._started)
        self._bar.progress(min(done / total, 1.0), text=self._text(done, total, remaining))

    def done(self, message: str = "") -> None:
        """막대를 치우고, 실제로 걸린 시간을 남긴다."""
        self._bar.empty()
        if self._note_slot is not None:
            self._note_slot.empty()
        elapsed = time.monotonic() - self._started
        if message:
            st.caption(f"{message} ({_duration(elapsed)} 걸림)")


def _eta(done: int, total: int, elapsed: float) -> float | None:
    """지금까지의 평균 속도로 본 남은 시간(초). 아직 말할 수 없으면 None.

    처음 몇 건은 편차가 커서, 그걸로 계산한 남은 시간은 사람을 속인다. 차라리 말하지 않는다.
    """
    if done < MIN_SAMPLES_FOR_ETA or done >= total or elapsed <= 0:
        return None
    return elapsed / done * (total - done)


def _duration(seconds: float) -> str:
    """사람이 읽는 시간. '92.3초'보다 '1분 32초'가 기다릴지 말지 판단하기 쉽다."""
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}초"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}분 {rest}초" if rest else f"{minutes}분"


# --- 프로젝트 전환 (H0) -----------------------------------------------------

PROJECT_PICKER_KEY = "nav_project"
PROJECT_SEEN_KEY = "nav_project_seen"
"""고른 값 위젯과 파일에 저장된 활성 프로젝트가 **서로 다른 진실**이 되지 않게 하는 열쇠.

프로젝트는 두 곳에서 바뀐다 — 이 선택 상자와 설정 화면(만들기·목록에서 빼기)이다.
설정 화면에서 바꾸면 파일은 새 프로젝트를 가리키는데 선택 상자의 세션 값은 옛 프로젝트로
남아 있고, 그러면 "선택이 바뀌었다"고 오해해 **방금 만든 프로젝트에서 도로 튕겨 나온다.**
실제로 그 증상을 보고 이 키를 넣었다. 마지막으로 본 값을 기억해 두면 어느 쪽이 바뀐
것인지 구분할 수 있다.
"""


def project_picker() -> None:
    """사이드바 아래쪽에 지금 보고 있는 프로젝트를 띄우고 바꿀 수 있게 한다.

    **어느 프로젝트를 보고 있는지 항상 보여야 한다.** 프로젝트를 나눈 이유가 데이터를
    섞지 않는 것인데, 지금 어디에 라벨을 쌓고 있는지 모르면 나눈 의미가 없다. 그래서
    레일 모드에서도 머리글자를 남긴다.
    """
    from vision_ai import projects

    registry = projects.load()
    slugs = [p.slug for p in registry.projects]
    names = {p.slug: p.name for p in registry.projects}
    current = registry.active if registry.active in slugs else slugs[0]

    with st.sidebar:
        if rail_enabled():
            st.divider()
            st.markdown(
                f"<div title='프로젝트: {names[current]}' style='text-align:center;"
                f"font-size:0.75rem;opacity:0.7'>{names[current][:2]}</div>",
                unsafe_allow_html=True,
            )
            return

        # 밖에서 활성 프로젝트가 바뀌었으면(설정 화면에서 만들기 등) 선택 상자를 맞춰 준다.
        if st.session_state.get(PROJECT_SEEN_KEY) != current:
            st.session_state[PROJECT_SEEN_KEY] = current
            st.session_state[PROJECT_PICKER_KEY] = current

        st.divider()
        picked = st.selectbox(
            "프로젝트",
            slugs,
            format_func=lambda slug: names.get(slug, slug),
            key=PROJECT_PICKER_KEY,
            help="현장·라인마다 데이터와 라벨을 섞지 않으려면 프로젝트를 나눕니다.",
        )
        if picked != current:
            projects.use(picked)
            st.session_state[PROJECT_SEEN_KEY] = picked
            st.rerun()


# --- 이미지 위에서 영역 지정 (G15) ------------------------------------------

ROI_DISPLAY_WIDTH = 640
"""영역 지정 화면에서 이미지를 그릴 폭(px). 좌표는 원본 픽셀 기준으로 돌려준다."""

ROI_TRANSPORT_WIDTH = 1024
"""브라우저로 보낼 때 이미지를 줄이는 상한. 원본을 그대로 보내면 요청이 무거워진다."""


def _data_uri(rgb, *, max_width: int = ROI_TRANSPORT_WIDTH) -> str:
    """RGB 배열을 data URI로 바꾼다.

    Vega-Lite의 image 마크는 URL을 요구하는데, 이 앱은 이미지를 웹으로 서빙하지 않는다.
    data URI로 그림을 요청에 실어 보내면 파일 서버 없이 해결된다.
    """
    import cv2
    import numpy as np

    array = np.asarray(rgb)
    if array.shape[1] > max_width:
        scale = max_width / array.shape[1]
        array = cv2.resize(
            array, (max_width, max(1, round(array.shape[0] * scale))), interpolation=cv2.INTER_AREA
        )
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


def roi_box(key: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    """드래그로 지정된 영역을 (x, y, w, h) 정수로 읽는다. 없으면 None.

    ``roi_picker``를 그리기 **전에도** 부를 수 있다 — Streamlit이 선택 결과를 위젯 key로
    세션 상태에 남기기 때문이다. 화면 배치상 이미지(왼쪽)보다 저장 버튼(오른쪽)이 먼저
    실행되므로 이 성질이 필요하다.
    """
    state = st.session_state.get(key) or {}
    picked = (state.get("selection") or {}).get("roi") or {}
    xs, ys = picked.get("x"), picked.get("y")
    if not xs or not ys:
        return None

    x0, x1 = sorted(float(v) for v in xs[:2])
    y0, y1 = sorted(float(v) for v in ys[:2])
    # Vega는 축 밖으로도 끌 수 있다. 이미지 안으로 잘라 넣는다.
    x0, x1 = max(0.0, x0), min(float(width), x1)
    y0, y1 = max(0.0, y0), min(float(height), y1)
    w, h = round(x1 - x0), round(y1 - y0)
    if w < 1 or h < 1:
        return None
    return round(x0), round(y0), w, h


def roi_picker(
    rgb,
    *,
    key: str,
    width: int,
    height: int,
    display_width: int = ROI_DISPLAY_WIDTH,
) -> None:
    """이미지 위에서 드래그해 결함 영역을 지정하게 한다.

    좌표를 숫자로 입력하는 것보다 빠르고, 무엇보다 **보고 있는 그림 위에서** 지정하게 된다.
    슬라이더로 x/y/폭/높이를 맞추려면 값을 옮길 때마다 그림을 다시 봐야 한다.

    새 의존성은 쓰지 않는다. Vega-Lite의 구간 선택(interval)이 곧 사각형 드래그이고,
    Streamlit이 그 결과를 파이썬으로 돌려준다. 선택 영역 **안쪽을 끌면 그대로 옮겨지므로**
    지정한 뒤 위치를 고치는 것도 된다 (Vega의 기본 동작).

    화면 크기와 무관하게 **좌표는 원본 픽셀 기준**으로 나온다. 축 도메인을 원본 크기로
    두고 그림만 줄여 그리기 때문이다.

    결과는 ``roi_box(key, width, height)``로 읽는다.
    """
    uri = _data_uri(rgb)
    if not uri:
        st.error("이미지를 화면용으로 변환하지 못했습니다.")
        return

    display_height = max(1, round(display_width * height / max(width, 1)))
    axes = {
        "x": {"field": "x", "type": "quantitative", "scale": {"domain": [0, width]}, "axis": None},
        "y": {"field": "y", "type": "quantitative", "scale": {"domain": [height, 0]}, "axis": None},
    }
    spec = {
        "width": display_width,
        "height": display_height,
        "layer": [
            {
                "data": {"values": [{"x": 0, "y": 0, "url": uri}]},
                "mark": {
                    "type": "image", "width": display_width, "height": display_height,
                    "align": "left", "baseline": "top",
                },
                "encoding": {**axes, "url": {"field": "url", "type": "nominal"}},
            },
            {
                # 선택을 붙일 자리만 필요하다. 보이지 않는 점 하나로 충분하다.
                "data": {"values": [{"x": 0, "y": 0}]},
                "mark": {"type": "point", "opacity": 0},
                "encoding": axes,
                "params": [
                    {"name": "roi", "select": {"type": "interval", "encodings": ["x", "y"]}}
                ],
            },
        ],
    }
    st.vega_lite_chart(spec, on_select="rerun", key=key, use_container_width=False)


# --- 타임라인에서 구간 지정 (H4) --------------------------------------------

TIMELINE_HEIGHT = 170
"""타임라인 차트의 **전체** 높이(px).

Streamlit은 이 값을 그림틀 전체 크기로 주고 Vega는 거기에 맞춰 줄인다. 즉 축 눈금과
축 제목이 먼저 자리를 가져가고 **남은 만큼만** 끌 수 있는 영역이 된다. 90px으로 두었더니
남은 영역이 1픽셀이라 드래그가 아예 시작되지 않았다. 축이 약 55px을 쓰므로 넉넉히 준다.
"""

TIMELINE_FIELD = "t"
"""타임라인의 x축 필드 이름. 선택 결과도 **이 이름**으로 돌아온다."""


def range_box(key: str, low: float, high: float) -> tuple[float, float] | None:
    """타임라인에서 끌어 고른 구간 (시작, 끝). 안 골랐으면 None.

    `roi_box`와 같은 방식이다 — 화면을 그리기 전에도 세션 상태에서 읽을 수 있다.

    Vega는 선택 범위를 **필드 이름**으로 담아 돌려준다. `roi_box`가 `"x"`로 읽는 것은
    그 차트의 필드 이름이 마침 `x`이기 때문이지 축 이름이어서가 아니다.
    """
    state = st.session_state.get(key) or {}
    picked = (state.get("selection") or {}).get("span") or {}
    xs = picked.get(TIMELINE_FIELD)
    if not xs or len(xs) < 2:
        return None
    start, end = sorted(float(value) for value in xs[:2])
    start, end = max(low, start), min(high, end)
    return (start, end) if end > start else None


def timeline_picker(
    seconds: Sequence[float],
    marks: Sequence[str],
    *,
    key: str,
    duration: float,
    width: int = 720,
) -> None:
    """영상 타임라인 위에서 시간 구간을 끌어 고르게 한다.

    결함 위치를 이미지 위에서 끄는 것(`roi_picker`)과 **같은 도구**다. Vega-Lite 구간
    선택에서 x축 하나만 쓰면 그대로 시간 구간이 된다. 새로 만들 것이 없고, 사용자가
    배우는 조작도 하나로 남는다.

    검사원은 프레임을 한 장씩 보지 않는다. 영상을 돌려 보며 "3분 12초부터 3분 20초까지
    불량"이라고 짚는다. 한 장씩 라벨하면 3,000장을 한 장씩 봐야 해서 영상을 쓰는 이유가
    사라진다.
    """
    values = [
        {TIMELINE_FIELD: float(second), "mark": str(mark)}
        for second, mark in zip(seconds, marks)
    ]
    axis = {
        "x": {
            "field": TIMELINE_FIELD, "type": "quantitative",
            "scale": {"domain": [0, max(duration, 0.001)]},
            "axis": {"title": "영상 시각 (초)"},
        }
    }
    spec = {
        "width": width,
        "height": TIMELINE_HEIGHT,
        "data": {"values": values or [{TIMELINE_FIELD: 0.0, "mark": "없음"}]},
        "layer": [
            {
                "mark": {"type": "tick", "thickness": 3, "size": 46},
                "encoding": {
                    **axis,
                    "color": {
                        "field": "mark", "type": "nominal",
                        "legend": {"title": "라벨"},
                    },
                },
            },
            {
                "mark": {"type": "point", "opacity": 0},
                "encoding": axis,
                "params": [
                    {"name": "span", "select": {"type": "interval", "encodings": ["x"]}}
                ],
            },
        ],
    }
    st.vega_lite_chart(spec, on_select="rerun", key=key, use_container_width=False)


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
