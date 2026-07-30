"""3단계: 모델 개발 · 평가 화면 (구현 예정)."""

from __future__ import annotations

import streamlit as st

from vision_ai import storage

APPROACHES = [
    (
        "베이스라인 (고전 CV + 분류기)",
        "밝기·엣지·텍스처 통계 특징을 뽑아 간단한 분류기로 학습한다. "
        "무거운 모델 없이 곧바로 돌아가므로 이후 모델의 성능 하한선 역할을 한다.",
        "1순위",
    ),
    (
        "이상탐지 (정상만 학습)",
        "정상 이미지만으로 정상 분포를 학습하고 이탈 정도를 결함 점수로 쓴다. "
        "결함 샘플이 적은 상황에 맞고, MVTec AD가 전제하는 방식이다.",
        "2순위",
    ),
    (
        "지도학습 검출",
        "결함 라벨과 위치가 충분히 모이면 검출 모델로 위치와 유형을 함께 예측한다.",
        "데이터 확보 후",
    ),
    (
        "Claude 비전 2차 판정",
        "1차 모델이 애매하다고 본 이미지만 Claude에 넘겨 결함 유형과 판단 근거를 "
        "구조화된 JSON으로 받는다. 사람이 읽을 수 있는 설명을 얻는 것이 목적이다.",
        "1차 모델 이후",
    ),
]

METRICS = [
    ("Recall (재현율)", "실제 결함 중 놓치지 않고 잡아낸 비율. 미탐이 가장 큰 리스크이므로 최우선 지표."),
    ("Precision (정밀도)", "결함이라 판정한 것 중 실제 결함 비율. 오탐으로 인한 재검사 비용과 연결."),
    ("F1 / AUROC", "임계값 선택과 종합 성능 비교용."),
    ("Image AUROC / Pixel AUROC", "이상탐지에서 이미지 단위·픽셀 단위 성능을 나눠 본다."),
    ("추론 시간", "장당 처리 시간. 실시간 적용 여부 판단 근거."),
]


def render() -> None:
    st.title("🧠 3단계 · 모델 개발 · 평가")
    st.info("2단계 라벨링 완료 후 착수합니다. 아래는 확정된 설계 범위입니다.", icon="🚧")

    df = storage.load_manifest()
    stats = storage.summarize(df)
    cols = st.columns(4)
    cols[0].metric("학습 가능 이미지", f"{stats['labeled']:,}")
    cols[1].metric("정상", f"{stats['normal']:,}")
    cols[2].metric("결함", f"{stats['defect']:,}")
    ratio = f"{stats['normal'] / stats['defect']:.1f}:1" if stats["defect"] else "—"
    cols[3].metric("정상:결함 비율", ratio)

    st.divider()
    st.subheader("접근 방식과 순서")
    for title, detail, priority in APPROACHES:
        with st.container(border=True):
            head, tail = st.columns([4, 1])
            head.markdown(f"**{title}**")
            tail.caption(priority)
            st.caption(detail)

    st.divider()
    st.subheader("평가 지표")
    for name, detail in METRICS:
        st.markdown(f"- **{name}** — {detail}")

    st.divider()
    st.subheader("평가 화면에 담을 내용")
    st.markdown(
        "- 혼동행렬과 임계값 조정 슬라이더 (미탐/오탐 트레이드오프를 눈으로 확인)\n"
        "- **오탐·미탐 샘플 갤러리** — 숫자만으로는 무엇이 잘못됐는지 알 수 없다\n"
        "- 결함 점수 히트맵 오버레이\n"
        "- 실험 간 지표 비교표와 실행 설정 기록"
    )


render()
