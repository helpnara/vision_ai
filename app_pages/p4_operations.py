"""4단계: 사후 운영관리(MLOps) 화면 (구현 예정)."""

from __future__ import annotations

import streamlit as st

from vision_ai import config

PLANNED = [
    (
        "모델 레지스트리",
        "학습된 모델을 버전·지표·학습 데이터 스냅샷과 함께 등록하고, "
        "현재 서비스 중인 버전을 명시한다. 롤백 대상을 즉시 찾을 수 있어야 한다.",
    ),
    (
        "추론 로그 수집",
        "실제 판정 요청의 입력 요약, 결함 점수, 최종 판정, 처리 시간을 남긴다. "
        "성능 저하를 발견하는 유일한 근거가 되는 데이터다.",
    ),
    (
        "데이터 드리프트 감시",
        "학습 데이터와 최근 입력 이미지의 통계 분포(밝기·선명도·해상도)를 비교한다. "
        "촬영 환경이 바뀌면 모델을 건드리지 않아도 성능이 떨어진다.",
    ),
    (
        "성능 드리프트 감시",
        "사후 검수로 확인된 정답과 모델 판정을 비교해 재현율 추이를 추적하고, "
        "기준선 아래로 내려가면 경고한다.",
    ),
    (
        "재학습 트리거",
        "신규 라벨 누적량 또는 성능 저하 조건을 만족하면 재학습을 제안한다. "
        "자동 배포가 아니라 사람이 승인하는 흐름으로 둔다.",
    ),
    (
        "판정 이력 조회 · 이의 처리",
        "특정 이미지가 왜 그렇게 판정됐는지 되짚어 볼 수 있게 한다. "
        "Claude 2차 판정의 설명 텍스트를 함께 보관해 근거로 활용한다.",
    ),
]


def render() -> None:
    st.title("⚙️ 4단계 · 사후 운영관리 (MLOps)")
    st.info("3단계 완료 후 착수합니다. 아래는 확정된 설계 범위입니다.", icon="🚧")

    st.markdown(
        "모델을 만드는 것보다 **만든 뒤 성능이 떨어지는 것을 알아채는 일**이 어렵다. "
        "이 단계의 목표는 성능 저하를 조기에 감지하고 재학습으로 연결하는 것이다."
    )

    st.divider()
    st.subheader("구현 예정 기능")
    for title, detail in PLANNED:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(detail)

    st.divider()
    st.subheader("산출물 경로")
    st.markdown(
        f"- 모델: `{config.MODEL_DIR}`\n"
        f"- 리포트: `{config.REPORT_DIR}`\n"
        f"- 데이터 인덱스: `{config.MANIFEST_PATH}`"
    )
    st.caption("모두 git 추적 대상이 아니다. 재현에 필요한 설정과 코드만 저장소에 남긴다.")


render()
