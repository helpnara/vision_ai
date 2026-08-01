"""설정: 판정 기준을 업무에 맞게 바꾼다."""

from __future__ import annotations

from dataclasses import fields

import streamlit as st

from vision_ai import config, evaluate, settings


def _slider(name: str, current, *, percent: bool) -> float:
    low, high = settings.BOUNDS[name]
    step = 0.01 if percent else 0.005
    return st.slider(
        settings.LABELS[name],
        float(low), float(high), float(current), step,
        format="%.3f" if not percent else "%.2f",
        help=settings.HELP[name],
        key=f"p5_{name}",
    )


def render() -> None:
    st.title("⚙️ 설정")
    st.markdown(
        "여기 값들은 **업무 판단**이지 물리 상수가 아니다. 미탐 1건의 비용이 큰 라인이면 "
        "재현율 목표를 올려야 하고, 라인마다 불량률도 다르다. 라인에 맞게 바꿔 쓴다."
    )
    st.info(
        "바꾼 값은 **3단계 평가와 4단계 승격 점검이 함께** 사용합니다. "
        "한 곳에서만 바꾸면 같은 모델을 두 화면이 다르게 평가하게 되므로 여기에 모아 두었습니다.",
        icon="ℹ️",
    )

    current = settings.load()
    diff = settings.changed(current)
    if diff:
        st.caption(f"기본값과 다른 항목 {len(diff)}개: " + ", ".join(
            f"{settings.LABELS[name]} {values[0]} (기본 {values[1]})" for name, values in diff.items()
        ))
    else:
        st.caption("현재 전부 기본값입니다.")

    st.divider()
    st.subheader("판정 기준 (D5)")
    st.caption("모델을 서비스에 올려도 되는지 판단하는 기준입니다.")

    col1, col2 = st.columns(2)
    with col1:
        target_recall = _slider("target_recall", current.target_recall, percent=True)
    with col2:
        min_reduction = _slider("min_reduction", current.min_reduction, percent=True)

    st.divider()
    st.subheader("현장 가정")
    st.caption("성능 지표를 업무 언어(검수량·놓치는 결함)로 환산할 때 쓰는 값입니다.")

    col3, col4 = st.columns(2)
    with col3:
        prevalence = col3.number_input(
            settings.LABELS["prevalence"] + " (%)",
            settings.BOUNDS["prevalence"][0] * 100, settings.BOUNDS["prevalence"][1] * 100,
            float(current.prevalence * 100), 0.1,
            help=settings.HELP["prevalence"], key="p5_prevalence",
        ) / 100.0
    with col4:
        volume = int(col4.number_input(
            settings.LABELS["volume"],
            int(settings.BOUNDS["volume"][0]), int(settings.BOUNDS["volume"][1]),
            int(current.volume), 100,
            help=settings.HELP["volume"], key="p5_volume",
        ))

    st.divider()
    st.subheader("재학습 판단")
    col5, col6 = st.columns(2)
    with col5:
        new_labels = int(col5.number_input(
            settings.LABELS["new_label_threshold"],
            int(settings.BOUNDS["new_label_threshold"][0]),
            int(settings.BOUNDS["new_label_threshold"][1]),
            int(current.new_label_threshold), 10,
            help=settings.HELP["new_label_threshold"], key="p5_new_labels",
        ))
    with col6:
        recall_margin = _slider("recall_margin", current.recall_margin, percent=False)

    candidate = settings.Settings(
        target_recall=target_recall, min_reduction=min_reduction,
        prevalence=prevalence, volume=volume,
        new_label_threshold=new_labels, recall_margin=recall_margin,
    )

    st.divider()
    _preview(candidate)

    st.divider()
    save, reset = st.columns([1, 1])
    if save.button("💾 저장", type="primary", key="p5_save"):
        stored = settings.save(candidate)
        if stored != candidate:
            st.warning("일부 값이 허용 범위를 벗어나 잘렸습니다.", icon="⚠️")
        st.success("저장했습니다. 3·4단계가 이 기준으로 판단합니다.", icon="✅")
        st.rerun()
    if reset.button("↩️ 기본값으로 되돌리기", key="p5_reset"):
        settings.reset()
        st.success("기본값으로 되돌렸습니다.", icon="✅")
        st.rerun()

    st.caption(f"저장 위치: `{config.DATA_ROOT / settings.SETTINGS_FILE}` (git 추적 대상 아님)")


def _preview(candidate: settings.Settings) -> None:
    """설정을 바꾸면 무엇이 달라지는지 실측 성능에 대입해 보여준다.

    슬라이더 숫자만으로는 결과가 안 보인다. 실측값(재현율 0.925 / 오탐률 0.455)에 넣어
    검수량과 놓치는 결함이 어떻게 변하는지 함께 보여야 판단할 수 있다.
    """
    st.subheader("이 기준이면 어떻게 되는가")
    st.caption(
        "VisA 실측 성능(재현율 92.5% · 오탐률 45.5%)에 지금 설정을 대입한 결과입니다. "
        "설정을 바꾸면 이 숫자가 함께 움직입니다."
    )
    impact = evaluate.business_impact(
        0.925, 0.455, prevalence=candidate.prevalence, volume=candidate.volume
    )
    cols = st.columns(3)
    cols[0].metric("검수량 절감", f"{impact['reduction_ratio']:.0%}")
    cols[1].metric("놓치는 결함", f"{impact['missed']:.0f}건")
    cols[2].metric("현장 기대 정밀도", f"{impact['precision']:.1%}")

    passes_recall = 0.925 >= candidate.target_recall
    passes_reduction = impact["reduction_ratio"] >= candidate.min_reduction
    if passes_recall and passes_reduction:
        st.success("실측 모델이 이 기준을 **통과**합니다.", icon="✅")
    else:
        missing = []
        if not passes_recall:
            missing.append(f"재현율 92.5% < 목표 {candidate.target_recall:.0%}")
        if not passes_reduction:
            missing.append(
                f"검수량 절감 {impact['reduction_ratio']:.0%} < 기준 {candidate.min_reduction:.0%}"
            )
        st.warning(
            "실측 모델은 이 기준에 **미달**합니다 — " + " · ".join(missing) + ". "
            "승격할 때 확인을 요구받게 됩니다.",
            icon="⚠️",
        )


render()
