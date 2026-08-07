"""홈 화면: 프로젝트 개요와 단계별 진행 현황."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from vision_ai import (
    config, datasets, evaluate, glossary, guide, labeling, quickstart, storage,
)

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
        "status": "구현 완료",
        "icon": "🧠",
        "detail": "베이스라인 · 패치 이상탐지(위치 히트맵) · Claude 2차 판정 · 임계값 조정 · 실험 기록",
    },
    {
        "no": 4,
        "name": "운영 관리 (MLOps)",
        "status": "구현 완료",
        "icon": "⚙️",
        "detail": "모델 레지스트리(롤백) · 배치 추론 로그 · 드리프트 감시 · 성능 추이 · 재학습 판단 · 판정 이력",
    },
]

STATUS_BADGE = {"구현 완료": "✅", "진행 중": "🚧", "예정": "⬜"}

# 실데이터 측정 결과는 저장소에 함께 담아 둔다. 데이터(`data/`)는 용량 때문에 git 대상이
# 아니므로, 배포본에서도 볼 수 있으려면 결과 파일만은 추적 대상이어야 한다.
VALIDATION_PATH = config.PROJECT_ROOT / "docs" / "results" / "visa_validation.json"

# 실제 생산 라인 불량률은 시험 구성(1:1)보다 훨씬 낮다. 그 차이를 눈에 보이게 두려고 함께 쓴다.
ASSUMED_PREVALENCE = 0.01


def _load_validation() -> dict | None:
    if not VALIDATION_PATH.exists():
        return None
    try:
        return json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _render_quickstart(manifest: pd.DataFrame) -> None:
    """한 번에 데모 상태까지 만든다.

    처음 연 사람이 운영 화면을 보려면 1단계 안쪽 탭부터 4단계 승격까지 순서대로 찾아
    들어가야 한다. 시연이 목적이라면 그 자체가 장벽이다.
    """
    if not manifest.empty:
        return   # 이미 데이터가 있으면 진행 순서 안내만으로 충분하다

    with st.container(border=True):
        st.markdown("#### ⚡ 처음이라면 — 한 번에 시작하기")
        st.caption(
            "합성 샘플 생성 → 분할 → 학습 → 승격까지 한 번에 밟아, "
            "**4단계 운영 시연을 곧바로 누를 수 있는 상태**로 만듭니다. "
            "다운로드도 설정도 필요 없습니다."
        )
        if st.button("⚡ 데모 한 바퀴 만들기", type="primary", key="home_quickstart"):
            bar = st.progress(0.0, text="준비 중...")

            def on_progress(index: int, total: int, name: str) -> None:
                bar.progress(index / total, text=f"{name}... ({index}/{total})")

            with st.spinner("데모를 만드는 중입니다. 1분쯤 걸립니다..."):
                result = quickstart.run(progress=on_progress)
            bar.empty()

            for warning in result.warnings:
                st.warning(warning, icon="⚠️")
            if result.ok:
                st.success(
                    f"준비 완료 — **{result.version}** 을 서비스 중으로 올렸습니다 "
                    f"(이미지 {result.n_images:,}장 / 학습 {result.n_train:,}장). "
                    "이제 **4단계 → 운영 시나리오 시연**을 눌러 보세요.",
                    icon="✅",
                )
                st.rerun()
        st.caption(
            "여기서 만든 것은 **합성 샘플**입니다. 결함이 인위적으로 뚜렷해 지표가 실제보다 "
            "높게 나오므로 성능 근거로 쓰면 안 됩니다."
        )


def _render_deployment_note() -> None:
    """배포본에서 데이터가 사라진다는 점을 앱 안에서도 알린다 (README에만 있으면 못 본다)."""
    st.caption(
        f"데이터 저장 위치: `{config.data_root()}` — git 추적 대상이 아닙니다. "
        "**무료 배포 환경에서는 컨테이너가 재시작되면 수집 이미지·라벨·모델·판정 이력이 "
        "모두 사라집니다.** 실제 라벨링과 학습은 내려받아 로컬에서 실행하세요."
    )


def _render_next_step(manifest: pd.DataFrame) -> None:
    """지금 어디까지 왔고 다음에 무엇을 할지 보여준다.

    처음 들어온 사람이 페이지 5개와 탭 20여 개 중 어디부터 눌러야 할지 알 수 있게 하는 것이
    목적이다. 판단은 `guide`가 하고 여기서는 표시만 한다.
    """
    steps = guide.pipeline_steps(manifest=manifest)
    done, total = guide.progress(steps)
    upcoming = guide.next_step(steps)

    st.subheader("진행 순서")
    st.progress(done / total, text=f"{done}/{total} 완료")

    if upcoming is None:
        st.success(
            "전 과정을 한 바퀴 돌았습니다. 이제 **4단계**에서 드리프트와 성능 추이를 살펴보거나, "
            "**3단계**로 돌아가 모델을 개선할 수 있습니다.",
            icon="🎉",
        )
    else:
        with st.container(border=True):
            st.markdown(f"#### 👉 다음: {upcoming.title}")
            st.caption(f"{upcoming.where} · 현재 상태: {upcoming.detail}")
            if upcoming.action:
                st.markdown(upcoming.action)
            st.page_link(upcoming.page, label=f"{upcoming.stage}단계로 이동", icon="➡️")

    with st.expander("전체 순서 보기", expanded=upcoming is None):
        for step in steps:
            mark = "✅" if step.done else ("👉" if step is upcoming else "⬜")
            st.markdown(f"{mark} **{step.title}** — {step.detail}  \n　　<sub>{step.where}</sub>",
                        unsafe_allow_html=True)


def _render_validation() -> None:
    """VisA 실데이터 측정 결과. 합성 데이터 수치와 혼동하지 않도록 출처를 함께 밝힌다."""
    report = _load_validation()
    st.subheader("실측 성능 (VisA PCB 4종)")
    if report is None:
        st.caption(
            "아직 실데이터 측정 결과가 없습니다. "
            "`PYTHONPATH=src python scripts/validate_visa.py`로 생성합니다."
        )
        return

    rows = report.get("categories") or []
    scored = [r for r in rows if "recall" in (r.get("held_out") or {})]

    cols = st.columns(3)
    cols[0].metric("평균 AUROC", f"{report.get('mean_auroc', float('nan')):.3f}")
    cols[1].metric("평균 AP", f"{report.get('mean_average_precision', float('nan')):.3f}")
    if scored:
        mean_recall = sum(r["held_out"]["recall"] for r in scored) / len(scored)
        mean_fpr = sum(r["held_out"]["false_alarm_rate"] for r in scored) / len(scored)
        cols[2].metric("평균 재현율", f"{mean_recall:.3f}")
    else:
        mean_recall = mean_fpr = float("nan")

    if rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "카테고리": r["category"],
                        "AUROC": round(r["auroc"], 3),
                        "AP": round(r["average_precision"], 3),
                        "재현율": round((r.get("held_out") or {}).get("recall", float("nan")), 3),
                        "정밀도": round((r.get("held_out") or {}).get("precision", float("nan")), 3),
                        "오탐률": round(
                            (r.get("held_out") or {}).get("false_alarm_rate", float("nan")), 3
                        ),
                    }
                    for r in rows
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    if scored:
        # 환산 로직은 evaluate에 한 곳으로 모아 뒀다 (3단계 평가 화면과 같은 계산).
        impact = evaluate.business_impact(
            mean_recall, mean_fpr, prevalence=ASSUMED_PREVALENCE, volume=1000
        )
        cols = st.columns(2)
        cols[0].metric(
            "검수량 절감", f"{impact['reduction_ratio']:.0%}",
            help=f"불량률 {ASSUMED_PREVALENCE:.0%} 가정. 전수 검수 대비 사람이 안 봐도 되는 비율입니다.",
        )
        cols[1].metric(
            "1,000장당 놓치는 결함", f"{impact['missed']:.0f}건",
            help="절감의 대가입니다. 이 값을 받아들일 수 있는지가 도입 판단의 핵심입니다.",
        )
        st.warning(
            f"위 표의 정밀도는 **정상:결함 = 1:1인 시험 구성** 기준이라 현장 기대치가 아닙니다. "
            f"불량률을 {ASSUMED_PREVALENCE:.0%}로 가정하면 기대 정밀도는 "
            f"**{impact['precision']:.1%}** 로 떨어집니다. 따라서 이 모델은 자동 판정용이 아니라 "
            f"**1차 스크리닝용**입니다 — 사람이 볼 물량을 줄여 주는 것이 실제 효용입니다.",
            icon="⚠️",
        )
    st.caption(
        f"출처: {report.get('dataset', 'VisA')} · {report.get('protocol', '')} · "
        f"{report.get('threshold_policy', '')}"
    )


def render() -> None:
    st.title("🔍 비전 기반 표면 결함 탐지 파이프라인")
    st.markdown(
        "**제조업 기준 PoC** — 물건 표면의 결함을 탐지하는 모델을 오픈 데이터셋 기반으로 개발한다. "
        "수집부터 운영까지 4단계를 이 앱에서 순차적으로 다룬다."
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
        st.caption(
            f"기본 예시 데이터셋은 **{default.name}**({default.license})입니다. "
            "내려받지 않았어도 합성 샘플로 전 과정을 시험할 수 있습니다."
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
    _render_quickstart(df)
    _render_next_step(df)

    st.divider()
    st.subheader("기능별 구현 현황")
    st.caption("아래는 앱에 구현된 기능 목록이다. 내가 어디까지 했는지는 위의 진행 순서를 본다.")
    for stage in STAGES:
        badge = STATUS_BADGE.get(stage["status"], "⬜")
        with st.container(border=True):
            head, tail = st.columns([4, 1])
            head.markdown(f"**{stage['icon']} {stage['no']}단계 · {stage['name']}**")
            tail.markdown(f"{badge} {stage['status']}")
            st.caption(stage["detail"])

    st.divider()
    _render_validation()

    st.divider()
    _render_deployment_note()

    with st.expander("데이터 사용 원칙", expanded=False):
        st.markdown(
            "- **사내(회사) 결함 데이터는 사용하지 않는다.** 공개 데이터셋과 직접 촬영 이미지만 사용한다.\n"
            "- 오픈 데이터셋은 라이선스가 각기 다르다. 비상업 조건(CC BY-NC-SA 등)이 붙은 것이 많으므로 "
            "사용 전 원본 배포 페이지에서 조건을 직접 확인한다.\n"
            f"- 데이터 루트: `{config.data_root()}` (git 추적 대상 아님)"
        )

    with st.expander("용어 사전 — 화면에 나오는 말이 낯설다면", expanded=False):
        st.caption("이 앱에 나오는 전문 용어를 한 줄씩 풀어 썼습니다.")
        for name, text in glossary.TERMS.items():
            st.markdown(f"- **{name}** — {text}")

    with st.expander("결함 유형 분류 체계 (표준 10종)", expanded=False):
        for key, label in config.DEFECT_TYPES.items():
            st.markdown(f"- `{key}` — {label}")


render()
