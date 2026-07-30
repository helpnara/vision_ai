"""홈 화면: 프로젝트 개요와 단계별 진행 현황."""

from __future__ import annotations

import streamlit as st

from vision_ai import config, datasets, labeling, storage

STAGES = [
    {
        "no": 1,
        "name": "데이터 수집 (입력)",
        "status": "구현 완료",
        "icon": "📥",
        "detail": "오픈 데이터셋 카탈로그 · 로컬 폴더 임포트 · 이미지 업로드 · 합성 샘플 생성 · 품질 점검",
    },
    {
        "no": 2,
        "name": "라벨링",
        "status": "구현 완료",
        "icon": "🏷️",
        "detail": "라벨 검수 큐 · 폴더 라벨 검증 · 결함 유형 정규화 · 마스크 기반 ROI · 층화 분할",
    },
    {
        "no": 3,
        "name": "모델 개발 · 평가",
        "status": "예정",
        "icon": "🧠",
        "detail": "베이스라인 학습, 이상탐지 모델, Claude 비전 2차 판정, 지표·오탐/미탐 리포트",
    },
    {
        "no": 4,
        "name": "운영 관리 (MLOps)",
        "status": "예정",
        "icon": "⚙️",
        "detail": "모델 레지스트리, 추론 로그, 데이터/성능 드리프트 감시, 재학습 트리거",
    },
]

STATUS_BADGE = {"구현 완료": "✅", "진행 중": "🚧", "예정": "⬜"}


def render() -> None:
    st.title("🔍 비전 기반 표면 결함 탐지 파이프라인")
    st.markdown(
        "일상 생활에서 접하는 물건의 **표면 결함**을 탐지하는 모델을 오픈 데이터셋 기반으로 "
        "개발한다. 수집부터 운영까지 4단계를 이 앱에서 순차적으로 다룬다."
    )

    df = storage.load_manifest()
    stats = storage.summarize(df)

    st.subheader("데이터 현황")
    cols = st.columns(5)
    cols[0].metric("전체 이미지", f"{stats['total']:,}")
    cols[1].metric("정상", f"{stats['normal']:,}")
    cols[2].metric("결함", f"{stats['defect']:,}")
    cols[3].metric("미라벨", f"{stats['unlabeled']:,}")
    cols[4].metric("카테고리", f"{stats['categories']:,}")

    if stats["total"] == 0:
        default = datasets.default_dataset()
        st.info(
            f"아직 수집된 이미지가 없습니다. **1. 데이터 수집** 화면에서 시작하세요.\n\n"
            f"기본 예시 데이터셋은 **{default.name}**({default.license})입니다. "
            "아직 내려받지 않았다면 *합성 샘플 생성* 탭으로 전체 파이프라인을 먼저 시험해 볼 수 있습니다.",
            icon="👉",
        )
    else:
        label_stats = labeling.stats(labeling.resolve(df))
        st.subheader("라벨링 현황")
        cols = st.columns(4)
        cols[0].metric("사람이 라벨/확인", f"{label_stats['human']:,}")
        cols[1].metric("유형 미지정 결함", f"{label_stats['unspecified_type']:,}")
        cols[2].metric("ROI 지정", f"{label_stats['with_roi']:,}")
        cols[3].metric("분할 배정", f"{label_stats['split_assigned']:,}")

    st.divider()
    st.subheader("단계별 진행 현황")
    for stage in STAGES:
        badge = STATUS_BADGE.get(stage["status"], "⬜")
        with st.container(border=True):
            head, tail = st.columns([4, 1])
            head.markdown(f"**{stage['icon']} {stage['no']}단계 · {stage['name']}**")
            tail.markdown(f"{badge} {stage['status']}")
            st.caption(stage["detail"])

    st.divider()
    with st.expander("데이터 사용 원칙", expanded=False):
        st.markdown(
            "- **사내(회사) 결함 데이터는 사용하지 않는다.** 공개 데이터셋과 직접 촬영 이미지만 사용한다.\n"
            "- 오픈 데이터셋은 라이선스가 각기 다르다. 비상업 조건(CC BY-NC-SA 등)이 붙은 것이 많으므로 "
            "사용 전 원본 배포 페이지에서 조건을 직접 확인한다.\n"
            f"- 데이터 루트: `{config.DATA_ROOT}` (git 추적 대상 아님)"
        )

    with st.expander("결함 유형 분류 체계 (일상 물건 기준)", expanded=False):
        for key, label in config.DEFECT_TYPES.items():
            st.markdown(f"- `{key}` — {label}")


render()
