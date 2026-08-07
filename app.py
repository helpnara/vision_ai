"""Streamlit 진입점.

실행: streamlit run app.py

4단계 파이프라인(수집 → 라벨링 → 모델 개발/평가 → 운영관리)을 하나의 앱에서 순차적으로
다룬다. 각 단계는 app_pages/ 아래의 개별 스크립트로 분리되어 있다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 편집 가능 설치(pip install -e .) 없이도 동작하도록 src를 import 경로에 추가한다.
_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_ai import projects, ui  # noqa: E402

st.set_page_config(
    page_title="표면 결함 탐지 파이프라인",
    page_icon="🔍",
    layout="wide",
    # "auto" — 넓은 화면에서는 펼치고, 폰처럼 좁은 화면에서는 접은 채로 연다.
    # "expanded"로 못박으면 폰에서 300px 사이드바가 본문을 거의 다 덮는다.
    initial_sidebar_state="auto",
)

# 저장된 활성 프로젝트를 실제 경로에 적용한다. 이후 모든 화면이 이 프로젝트를 본다.
projects.bootstrap()

PAGES = [
    st.Page("app_pages/home.py", title="홈 / 진행 현황", icon="🏠", default=True),
    st.Page("app_pages/p1_ingest.py", title="1. 데이터 수집", icon="📥"),
    st.Page("app_pages/p2_labeling.py", title="2. 라벨링", icon="🏷️"),
    st.Page("app_pages/p3_modeling.py", title="3. 모델 개발 · 평가", icon="🧠"),
    st.Page("app_pages/p4_operations.py", title="4. 운영 관리 (MLOps)", icon="⚙️"),
    st.Page("app_pages/p5_settings.py", title="설정", icon="🛠️"),
]

CAPTIONS = (
    "비전 기반 표면 결함 탐지 (제조업 PoC)",
    "데이터: 오픈 데이터셋 + 직접 촬영 (사내 데이터 미사용)",
)

# 기본 메뉴를 숨기고 직접 그린다. 접었을 때 아이콘만 남는 레일 모드를 쓰기 위해서다.
page = st.navigation(PAGES, position="hidden")
ui.apply_chrome()
ui.sidebar_nav(PAGES, captions=CAPTIONS)
ui.project_picker()
page.run()
