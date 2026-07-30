"""2단계: 라벨링 화면 (구현 예정)."""

from __future__ import annotations

import streamlit as st

from vision_ai import config, storage

PLANNED = [
    (
        "라벨 검수 큐",
        "manifest에서 미라벨 이미지를 한 장씩 띄우고 정상/결함을 지정한다. "
        "키보드 단축키로 빠르게 넘기는 흐름을 목표로 한다.",
    ),
    (
        "폴더 라벨 검증",
        "오픈 데이터셋은 폴더 구조로 라벨이 이미 주어진다. 이를 그대로 신뢰하지 않고 "
        "샘플링해 육안 검수하여 오라벨을 잡아낸다.",
    ),
    (
        "결함 유형 지정",
        "결함으로 판정한 이미지에 프로젝트 표준 결함 유형을 매핑한다. "
        "데이터셋별 고유 유형명(예: broken_large)을 표준 유형으로 정규화한다.",
    ),
    (
        "관심 영역(ROI) 표시",
        "결함 위치를 사각형으로 표시해 저장한다. 이상탐지 결과 평가와 "
        "Claude 2차 판정용 크롭 생성에 사용한다.",
    ),
    (
        "학습·검증·테스트 분할",
        "카테고리·라벨 비율을 유지한 층화 분할. 동일 원본에서 파생된 이미지가 "
        "서로 다른 분할에 섞이지 않도록 한다(데이터 누수 방지).",
    ),
    (
        "라벨 이력 관리",
        f"라벨 변경 이력을 `{config.LABELS_PATH.name}`에 누적 기록해 누가 언제 무엇을 "
        "바꿨는지 추적한다.",
    ),
]


def render() -> None:
    st.title("🏷️ 2단계 · 라벨링")
    st.info("다음 구현 대상입니다. 아래는 확정된 설계 범위입니다.", icon="🚧")

    df = storage.load_manifest()
    stats = storage.summarize(df)

    cols = st.columns(4)
    cols[0].metric("수집 이미지", f"{stats['total']:,}")
    cols[1].metric("라벨 있음", f"{stats['labeled']:,}")
    cols[2].metric("미라벨", f"{stats['unlabeled']:,}")
    coverage = (stats["labeled"] / stats["total"] * 100) if stats["total"] else 0.0
    cols[3].metric("라벨 커버리지", f"{coverage:.0f}%")

    if stats["total"] == 0:
        st.warning("먼저 **1. 데이터 수집**에서 이미지를 등록해야 합니다.", icon="📥")

    st.divider()
    st.subheader("구현 예정 기능")
    for title, detail in PLANNED:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(detail)

    st.divider()
    st.subheader("라벨 체계")
    st.markdown("**판정 라벨**: " + " · ".join(f"`{k}`({v})" for k, v in config.LABEL_KO.items()))
    st.markdown("**결함 유형**")
    st.markdown("\n".join(f"- `{k}` — {v}" for k, v in config.DEFECT_TYPES.items()))


render()
