"""4단계: 사후 운영관리(MLOps) 화면."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from vision_ai import (
    claude_review,
    config,
    evaluate,
    experiments,
    features,
    glossary,
    guide,
    labeling,
    monitoring,
    registry,
    scenario,
    serving,
    storage,
    viz,
)

_BATCH_KEY = "p4_batch"
_SCENARIO_KEY = "p4_scenario"


def _resolved() -> pd.DataFrame:
    df = labeling.resolve()
    if df.empty:
        return df
    return df.assign(path_abs=[str(storage.resolve_path(p)) for p in df["path"]])


def _production_banner() -> pd.Series | None:
    prod = registry.production()
    if prod is None:
        st.warning(
            "서비스 중인 모델이 없습니다. **모델 레지스트리** 탭에서 3단계 실행을 등록하고 승격하세요.",
            icon="📚",
        )
        if experiments.load_runs().empty:
            st.caption("등록할 실행이 아직 없습니다. 3단계에서 모델을 먼저 학습하세요.")
            st.page_link(guide.PAGE_MODELING, label="3단계 모델 개발·평가로 이동", icon="➡️")
        return None
    threshold = prod.get("threshold")
    st.success(
        f"서비스 중: **{prod['version']}** ({prod.get('kind', '')}/{prod.get('model', '')}) · "
        f"임계값 {float(threshold):.4f} · 승격 {prod.get('promoted_at', '—')}"
        if pd.notna(threshold) else f"서비스 중: **{prod['version']}**",
        icon="🟢",
    )
    return prod


# --- 1) 모델 레지스트리 -------------------------------------------------------

def _registry_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "3단계 실험 기록은 *무엇을 시도했는지*, 레지스트리는 *무엇을 쓰고 있는지*의 기록이다. "
        "승격할 때 모델 파일을 **버전 폴더로 복사**하므로, 이후 학습이 덮어써도 롤백할 수 있다."
    )

    frame = registry.load_registry()
    if frame.empty:
        st.info("등록된 버전이 없습니다. 아래에서 3단계 실행을 등록하세요.", icon="📚")
    else:
        view = frame.copy()
        view["상태"] = view["status"].map(lambda s: registry.STATUS_KO.get(str(s), s))
        st.dataframe(
            view[["version", "상태", "kind", "model", "threshold", "recall", "precision",
                  "auroc", "created_at", "promoted_at", "note"]],
            hide_index=True, width="stretch",
        )

    st.divider()
    st.markdown("### 3단계 실행 등록")
    runs = experiments.load_runs()
    if runs.empty:
        st.info("등록할 실행이 없습니다. 3단계에서 모델을 먼저 학습하세요.", icon="🧪")
        st.page_link(guide.PAGE_MODELING, label="3단계 모델 개발·평가로 이동", icon="➡️")
    else:
        registered = set(frame["run_id"].astype(str)) if not frame.empty else set()
        options = [r for r in runs["run_id"].astype(str) if r not in registered]
        if not options:
            st.caption("모든 실행이 이미 등록되어 있습니다.")
        else:
            col1, col2 = st.columns([2, 1])
            picked = col1.selectbox(
                "실행", options, key="p4_reg_run",
                format_func=lambda rid: _run_label(runs, rid),
            )
            note = col2.text_input("메모", key="p4_reg_note")
            promote_now = st.checkbox(
                "등록 후 바로 서비스 중으로 승격", value=frame.empty, key="p4_reg_promote"
            )
            st.caption(
                "승격 시 현재 학습 분할로 **드리프트 기준선**(특징 분위 경계)을 함께 저장합니다. "
                "이 기준선이 없으면 입력 분포 변화를 감지할 수 없습니다."
            )
            if st.button("📚 레지스트리에 등록", type="primary", key="p4_reg_do"):
                _register_run(df, picked, note=note, promote_now=promote_now)

    if frame.empty:
        return

    st.divider()
    st.markdown("### 버전 전환")
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox(
            "승격할 버전", frame["version"].astype(str).tolist(), key="p4_promote_pick"
        )
        blocked = _promotion_gate(frame, target)
        if st.button("🟢 서비스 중으로 승격", key="p4_promote_do", disabled=blocked):
            registry.promote(target)
            st.success(f"{target}을(를) 서비스 중으로 전환했습니다.", icon="✅")
            st.rerun()
    with col2:
        archivable = frame[frame["status"] != registry.STATUS_ARCHIVED]["version"].astype(str).tolist()
        if archivable:
            to_archive = st.selectbox("보관할 버전", archivable, key="p4_archive_pick")
            if st.button("🗄️ 보관 처리", key="p4_archive_do"):
                registry.archive(to_archive)
                st.success(f"{to_archive}을(를) 보관 처리했습니다.", icon="✅")
                st.rerun()

    with st.expander("버전 상세"):
        detail_version = st.selectbox(
            "버전", frame["version"].astype(str).tolist(), key="p4_detail_pick"
        )
        run = registry.load_run(detail_version)
        baseline = registry.load_baseline(detail_version)
        st.caption(
            f"드리프트 기준선: {'있음 (' + str(baseline.get('n_samples', 0)) + '개 표본)' if baseline else '없음'}"
        )
        if run:
            st.json(run)


def _promotion_gate(frame: pd.DataFrame, version: str) -> bool:
    """승격 전에 D5 기준으로 점검한다. 미달이면 확인을 받는다.

    **막지는 않는다.** 기준은 아직 승인 전 제안값이고, 비교나 시연 목적으로 일부러 낮은
    모델을 올릴 수도 있다. 다만 초보자가 성능 미달 모델을 아무 신호 없이 올리면, 그 뒤의
    드리프트 감시와 재학습 판단이 전부 그 모델을 기준으로 돌아간다.

    Returns:
        승격 버튼을 잠가야 하면 True.
    """
    row = frame[frame["version"].astype(str) == str(version)]
    if row.empty:
        return False
    row = row.iloc[0]

    metrics = {"recall": row.get("recall")}
    impact = None
    recall, false_alarm = row.get("recall"), row.get("false_alarm_rate")
    if pd.notna(recall) and pd.notna(false_alarm):
        impact = evaluate.business_impact(float(recall), float(false_alarm))
    elif pd.notna(recall) and pd.notna(row.get("precision")):
        # 레지스트리에는 오탐률이 없다. 정밀도와 재현율로 되짚어 계산한다
        # (평가 구성이 1:1이라는 가정이 들어가므로 근사치다).
        precision = float(row["precision"])
        if precision > 0:
            false_alarm = float(recall) * (1.0 - precision) / precision
            impact = evaluate.business_impact(float(recall), min(false_alarm, 1.0))

    check = glossary.promotion_check(metrics, impact)
    if check.passed:
        st.caption("✅ 성능 기준을 만족합니다.")
        return False

    st.warning(
        "**이 버전은 권장 기준에 못 미칩니다.**\n\n" + "\n\n".join(f"- {p}" for p in check.problems),
        icon="⚠️",
    )
    for note in check.notes:
        st.caption(note)
    return not st.checkbox(
        "기준 미달을 확인했고 그래도 승격합니다", key=f"p4_promote_ack_{version}"
    )


def _run_label(runs: pd.DataFrame, run_id: str) -> str:
    row = runs[runs["run_id"].astype(str) == run_id]
    if row.empty:
        return run_id
    row = row.iloc[0]
    recall = row.get("recall")
    suffix = f" · 재현율 {float(recall):.3f}" if pd.notna(recall) else ""
    return f"{run_id} · {row.get('kind', '')}/{row.get('model', '')}{suffix}"


def _register_run(df: pd.DataFrame, run_id: str, *, note: str, promote_now: bool) -> None:
    """실행을 등록하고, 현재 학습 분할로 드리프트 기준선을 만든다."""
    baseline = None
    labeled = df[df["label"].astype(str).isin([config.LABEL_NORMAL, config.LABEL_DEFECT])] if not df.empty else df
    train = labeled[labeled["split"].astype(str) == config.SPLIT_TRAIN] if not labeled.empty else labeled

    if not train.empty:
        with st.spinner("드리프트 기준선 계산 중..."):
            vectors = []
            for path in train["path_abs"]:
                image = viz.load_rgb(path)
                if image is not None:
                    vectors.append(features.image_features(image))
            if vectors:
                matrix = np.stack(vectors)
                labels = (train["label"].astype(str) == config.LABEL_DEFECT).astype(int)
                baseline = registry.make_baseline(
                    matrix, features.FEATURE_NAMES,
                    quality=train[["blur_score", "brightness"]],
                    label_ratio=float(labels.mean()),
                )
    try:
        result = registry.register(run_id, baseline=baseline, note=note, promote_now=promote_now)
    except ValueError as exc:
        st.error(f"등록 실패: {exc}")
        return

    st.success(f"{result.version} 등록 완료.", icon="✅")
    for warning in result.warnings:
        st.warning(warning, icon="⚠️")
    st.rerun()


# --- 2) 배치 추론 -------------------------------------------------------------

def _inference_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "등록된 버전으로 이미지를 판정하고 로그를 남긴다. **로그가 없으면 성능 저하를 감지할 "
        "근거 자체가 없다** — 운영 감시의 출발점이다."
    )
    if df.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    usable = serving.selectable_versions()
    if not usable:
        st.warning(
            "추론에 쓸 수 있는 버전이 없습니다. 모델 파일이 함께 등록된 버전이 필요합니다.",
            icon="📚",
        )
        return

    prod = registry.production()
    default_index = usable.index(str(prod["version"])) if prod is not None and str(prod["version"]) in usable else 0

    col1, col2, col3 = st.columns(3)
    version = col1.selectbox("버전", usable, index=default_index, key="p4_inf_version")
    scope = col2.selectbox(
        "대상", ["전체", "test 분할", "미라벨", "특정 카테고리"], key="p4_inf_scope"
    )
    limit = col3.number_input("최대 건수", 1, 5000, 200, 10, key="p4_inf_limit")

    subset = df
    if scope == "test 분할":
        subset = df[df["split"].astype(str) == config.SPLIT_TEST]
    elif scope == "미라벨":
        subset = df[df["label"].astype(str) == config.LABEL_UNLABELED]
    elif scope == "특정 카테고리":
        categories = sorted(df["category"].dropna().astype(str).unique().tolist())
        picked = st.selectbox("카테고리", categories, key="p4_inf_category")
        subset = df[df["category"].astype(str) == picked]
    subset = subset.head(int(limit))

    row = registry.get(version)
    threshold_default = float(row["threshold"]) if row is not None and pd.notna(row.get("threshold")) else 0.5
    override = st.checkbox("임계값 직접 지정", value=False, key="p4_inf_override")
    threshold = st.number_input(
        "임계값", value=threshold_default, step=0.01, format="%.4f",
        key="p4_inf_threshold", disabled=not override,
    )

    st.caption(f"대상 {len(subset):,}건 · 임계값 {threshold if override else threshold_default:.4f}")
    if len(subset) < monitoring.MIN_DRIFT_SAMPLES:
        st.info(
            f"드리프트 판정에는 최소 {monitoring.MIN_DRIFT_SAMPLES}건이 필요합니다. "
            "표본이 적으면 같은 분포에서도 지표가 커져 오경보가 됩니다.",
            icon="📏",
        )

    if subset.empty:
        st.warning("대상 이미지가 없습니다.")
        return

    if st.button("▶️ 배치 추론 실행", type="primary", key="p4_inf_run"):
        _run_inference(version, subset, threshold if override else None)

    batch = st.session_state.get(_BATCH_KEY)
    if batch and batch.get("version") == version:
        frame = pd.DataFrame(batch["records"])
        st.success(f"최근 실행: {len(frame):,}건", icon="✅")
        counts = frame["decision"].value_counts()
        cols = st.columns(3)
        cols[0].metric("결함 판정", f"{int(counts.get(config.LABEL_DEFECT, 0)):,}")
        cols[1].metric("정상 판정", f"{int(counts.get(config.LABEL_NORMAL, 0)):,}")
        cols[2].metric("평균 처리시간", f"{frame['latency_ms'].mean():.1f} ms")
        st.dataframe(frame.head(50), hide_index=True, width="stretch")

    st.divider()
    st.markdown("### 누적 추론 로그")
    log = monitoring.load_log()
    summary = monitoring.log_summary(log)
    cols = st.columns(4)
    cols[0].metric("누적 건수", f"{summary['total']:,}")
    cols[1].metric("버전 수", f"{summary['versions']:,}")
    cols[2].metric("결함 판정 비율", f"{summary['defect_rate']:.1%}")
    cols[3].metric("처리시간 p95", f"{summary['latency_p95']:.1f} ms")

    if not log.empty:
        st.dataframe(log.tail(200).iloc[::-1], hide_index=True, width="stretch", height=260)
        st.download_button(
            "추론 로그 내려받기",
            log.to_csv(index=False).encode("utf-8-sig"),
            file_name="inference_log.csv", mime="text/csv",
        )
        with st.expander("⚠️ 로그 초기화"):
            st.caption("누적 로그를 모두 지웁니다. 드리프트·성능 추이 근거가 사라집니다.")
            if st.button("로그 삭제", key="p4_log_clear"):
                monitoring.clear_log()
                st.session_state.pop(_BATCH_KEY, None)
                st.rerun()


def _run_inference(version: str, subset: pd.DataFrame, threshold: float | None) -> None:
    try:
        model = serving.load_version(version)
    except (ValueError, FileNotFoundError) as exc:
        st.error(f"모델 로드 실패: {exc}")
        return

    bar = st.progress(0.0, text="추론 중...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / max(total, 1), text=f"추론 중... {done:,}/{total:,}")

    result = serving.run_batch(model, subset, threshold=threshold, progress=on_progress)
    bar.empty()

    if not result.records:
        st.error("판정한 이미지가 없습니다.")
        return

    monitoring.log_inference(result.records)
    st.session_state[_BATCH_KEY] = {
        "version": version,
        "records": result.records,
        "features": result.features,
    }
    if result.failed:
        st.warning(f"읽기 실패 {len(result.failed)}건은 제외했습니다.", icon="⚠️")
    st.rerun()


# --- 3) 드리프트 감시 ---------------------------------------------------------

def _drift_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "입력 분포가 변하면 모델을 건드리지 않아도 성능이 떨어진다. **정답 라벨 없이** 감지할 수 있는 "
        "것이 장점이라, 사후 검수보다 먼저 경고를 준다."
    )
    prod = _production_banner()
    if prod is None:
        return

    version = str(prod["version"])
    baseline = registry.load_baseline(version)
    if baseline is None:
        st.error(
            f"{version}에 드리프트 기준선이 없습니다. 등록 시 학습 분할이 비어 있었을 수 있습니다. "
            "다시 등록하면 기준선이 저장됩니다.",
            icon="📉",
        )
        return

    st.caption(
        f"기준선: {baseline.get('n_samples', 0):,}개 표본 · 특징 {len(baseline.get('feature_names', []))}개 · "
        f"구간 {baseline.get('bins', 0)}개"
    )

    batch = st.session_state.get(_BATCH_KEY)
    if not batch or batch.get("features") is None or len(batch["features"]) == 0:
        st.info(
            "**배치 추론** 탭에서 추론을 한 번 실행하면 그 입력으로 드리프트를 계산합니다.",
            icon="▶️",
        )
        return

    current = batch["features"]
    drift = monitoring.feature_drift(baseline, current, features.FEATURE_NAMES)
    summary = monitoring.drift_summary(drift)

    cols = st.columns(4)
    cols[0].metric("검사 표본", f"{summary['n_current']:,}")
    cols[1].metric("판정", summary["level"])
    cols[2].metric("변화 특징", f"{summary['shifted']:,}", help=glossary.detail("psi"))
    cols[3].metric(
        "평균 PSI", f"{summary['mean_psi']:.3f}" if np.isfinite(summary["mean_psi"]) else "—"
    )

    if summary["level"] == monitoring.LEVEL_INSUFFICIENT:
        st.warning(
            f"표본이 {summary['n_current']}건으로 최소 {monitoring.MIN_DRIFT_SAMPLES}건에 못 미칩니다. "
            "소표본에서는 같은 분포에서도 PSI가 커져 오경보가 되므로 판정을 보류합니다.",
            icon="📏",
        )
    elif summary["level"] == "변화":
        st.error(
            f"입력 분포가 유의미하게 변했습니다 (PSI > {monitoring.PSI_SHIFTED}인 특징 "
            f"{summary['shifted']}개). 촬영 환경·공정 변경을 먼저 확인하세요.",
            icon="🚨",
        )
    elif summary["level"] == "주의":
        st.warning(f"일부 특징이 주의 구간입니다 ({summary['watch']}개). 추이를 지켜보세요.", icon="⚠️")
    else:
        st.success("입력 분포가 안정적입니다.", icon="✅")

    st.caption(glossary.caption("psi"))
    st.caption(
        f"PSI 해석: {monitoring.PSI_STABLE} 이하 안정 · "
        f"{monitoring.PSI_STABLE}~{monitoring.PSI_SHIFTED} 주의 · {monitoring.PSI_SHIFTED} 초과 변화"
    )
    with st.expander("이 기준은 어디서 왔나"):
        st.markdown(f"- {glossary.ARBITRARY['psi']}")
        st.markdown(f"- {glossary.ARBITRARY['drift_samples']}")

    st.divider()
    _drift_causes(drift)

    st.markdown("**특징별 분포 이동 (PSI 상위)**")
    display = drift.head(20)[
        ["feature", "psi", "level", "baseline_mean", "current_mean", "shift_sigma"]
    ].copy()
    # 내부 특징명만으로는 무엇이 변했는지 알 수 없다. 뜻을 나란히 붙인다.
    display.insert(1, "무엇을 재는가", [glossary.feature_meaning(f) for f in display["feature"]])
    st.dataframe(display.round(4), hide_index=True, width="stretch")

    if drift["psi"].notna().any():
        st.bar_chart(
            drift.head(15).set_index("feature")["psi"].dropna(), horizontal=True
        )

    st.divider()
    st.markdown("**결함 점수 분포 이동**")
    scores = np.array([r["score"] for r in batch["records"]], dtype=float)
    score = monitoring.score_drift(baseline, scores)
    if not score.get("available"):
        st.caption("기준선에 점수 분포가 없어 비교할 수 없습니다.")
    else:
        cols = st.columns(3)
        cols[0].metric("기준 평균", f"{score['baseline_mean']:.4f}")
        cols[1].metric("현재 평균", f"{score['current_mean']:.4f}")
        cols[2].metric(
            "이동 (σ)", f"{score['shift_sigma']:.2f}" if np.isfinite(score["shift_sigma"]) else "—"
        )
        st.caption(
            "입력 특징은 그대로인데 점수 분포만 이동했다면, 재학습보다 **임계값 재조정**이 먼저다."
        )


# --- 4) 성능 추이 -------------------------------------------------------------

def _performance_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "사후 검수로 확인된 정답과 모델 판정을 비교한다. 정답의 출처는 2단계에서 **사람이 확인한** "
        "라벨뿐이다 — 폴더 구조에서 추론한 라벨을 정답으로 쓰면 자기 채점이 된다."
    )
    log = monitoring.load_log()
    if log.empty:
        st.info("추론 로그가 없습니다. **배치 추론**을 먼저 실행하세요.", icon="▶️")
        return

    feedback = monitoring.feedback_frame(log, df)
    if feedback.empty:
        st.warning(
            "사후 검수 정답이 없습니다. **2. 라벨링 → 라벨 검수**에서 사람이 확인한 라벨이 쌓여야 "
            "실제 성능을 측정할 수 있습니다.",
            icon="🏷️",
        )
        return

    metrics = monitoring.performance_metrics(feedback)
    cols = st.columns(5)
    cols[0].metric("검수 대조 건수", f"{metrics['n']:,}")
    cols[1].metric("재현율", f"{metrics['recall']:.3f}" if np.isfinite(metrics["recall"]) else "—")
    cols[2].metric("정밀도", f"{metrics['precision']:.3f}" if np.isfinite(metrics["precision"]) else "—")
    cols[3].metric("미탐", f"{metrics['fn']:,}")
    cols[4].metric("오탐", f"{metrics['fp']:,}")

    prod = registry.production()
    if prod is not None and pd.notna(prod.get("recall")):
        registered = float(prod["recall"])
        actual = metrics["recall"]
        if np.isfinite(actual):
            gap = registered - actual
            if gap > 0.05:
                st.error(
                    f"등록 시 재현율 {registered:.3f} → 실측 {actual:.3f} (낙폭 {gap:.3f}). "
                    "임계값 재조정 또는 재학습이 필요합니다.",
                    icon="🚨",
                )
            else:
                st.success(f"등록 시 {registered:.3f} → 실측 {actual:.3f}. 유지되고 있습니다.", icon="✅")

    if metrics["fn"]:
        st.error(
            f"미탐 {metrics['fn']}건 — 결함을 놓친 사례입니다. 이 프로젝트에서 가장 큰 리스크이므로 "
            "개별 리뷰가 필요합니다.",
            icon="🚨",
        )
        misses = feedback[feedback["outcome"] == "FN"]
        _gallery(df, misses.head(8))

    st.divider()
    st.markdown("**버전별 실측 성능**")
    by_version = monitoring.performance_by_version(feedback)
    if not by_version.empty:
        st.dataframe(by_version.round(4), hide_index=True, width="stretch")

    st.markdown("**기간별 추이**")
    trend = monitoring.performance_trend(feedback)
    if trend.empty or len(trend) < 2:
        st.caption("추이를 그리려면 서로 다른 날짜의 추론 로그가 필요합니다.")
    else:
        st.line_chart(trend.set_index("bucket")[["recall", "precision"]])

    with st.expander("검수 대조 상세"):
        st.dataframe(
            feedback[["image_id", "version", "score", "decision", "label", "outcome", "logged_at"]],
            hide_index=True, width="stretch",
        )


def _gallery(df: pd.DataFrame, subset: pd.DataFrame) -> None:
    if subset.empty or df.empty:
        return
    lookup = df.set_index(df["image_id"].astype(str))
    columns = st.columns(4)
    for position, (_, row) in enumerate(subset.iterrows()):
        image_id = str(row["image_id"])
        target = columns[position % 4]
        if image_id not in lookup.index:
            continue
        image = viz.load_rgb(lookup.loc[image_id]["path_abs"])
        if image is not None:
            target.image(image, caption=f"{image_id} · 점수 {float(row['score']):.3f}", width="stretch")


# --- 5) 재학습 판단 -----------------------------------------------------------

def _retraining_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "운영 신호를 모아 재학습이 필요한지 판단한다. **자동 배포가 아니라 사람이 승인하는 제안**을 "
        "만드는 것이 목적이다."
    )
    prod = registry.production()
    log = monitoring.load_log()
    feedback = monitoring.feedback_frame(log, df) if not log.empty else pd.DataFrame()

    drift = None
    batch = st.session_state.get(_BATCH_KEY)
    if prod is not None and batch and batch.get("features") is not None and len(batch["features"]):
        baseline = registry.load_baseline(str(prod["version"]))
        if baseline:
            drift = monitoring.feature_drift(baseline, batch["features"], features.FEATURE_NAMES)

    new_labels = monitoring.new_labels_since(prod.get("promoted_at")) if prod is not None else 0

    col1, col2 = st.columns(2)
    threshold = col1.number_input(
        "재학습 기준 신규 라벨 수", 10, 5000, 50, 10, key="p4_rt_labels",
        help="승격 이후 이만큼 라벨이 쌓이면 재학습을 제안한다.",
    )
    margin = col2.slider(
        "허용 재현율 낙폭", 0.01, 0.30, 0.05, 0.01, key="p4_rt_margin",
        help="등록 시 재현율 대비 이보다 더 떨어지면 경보를 낸다.",
    )

    st.caption(
        f"{glossary.ARBITRARY['new_labels']} {glossary.ARBITRARY['recall_margin']} "
        "위 두 값은 화면에서 바로 바꿔 볼 수 있습니다."
    )

    decision = monitoring.retraining_signals(
        production=prod, log=log, feedback=feedback, drift=drift,
        new_labels=new_labels, new_label_threshold=int(threshold), recall_margin=float(margin),
    )

    if decision.recommended:
        st.error("**재학습을 제안합니다.** 아래 근거를 확인하고 사람이 승인하세요.", icon="🔁")
    else:
        st.success("지금은 재학습이 필요해 보이지 않습니다.", icon="✅")

    st.divider()
    st.markdown("### 판단 근거")
    for signal in decision.signals:
        with st.container(border=True):
            st.markdown(f"{signal.icon} **{signal.title}**")
            st.caption(signal.detail)

    if decision.recommended:
        st.divider()
        st.markdown("### 재학습 절차")
        st.markdown(
            "1. **2. 라벨링** — 미탐 사례를 검수하고 신규 라벨을 정리한다.\n"
            "2. **3. 모델 개발·평가** — 같은 설정으로 다시 학습하고 지표를 비교한다.\n"
            "3. **4. 모델 레지스트리** — 새 실행을 등록하고, 지표가 나아졌을 때만 승격한다.\n"
            "4. 승격 후 **배치 추론**을 다시 돌려 드리프트 기준선 대비 상태를 확인한다."
        )
        st.caption("승격은 되돌릴 수 있다 — 이전 버전이 보관 상태로 남아 있어 다시 승격하면 롤백된다.")


# --- 6) 판정 이력 조회 --------------------------------------------------------

def _trace_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "이미지 하나가 **왜 그렇게 판정됐는지**를 되짚는다. 1~4단계 산출물이 모두 `image_id`로 "
        "연결되어 있어 한 화면에 모을 수 있다."
    )
    if df.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    log = monitoring.load_log()
    candidates = (
        log["image_id"].astype(str).unique().tolist() if not log.empty
        else df["image_id"].astype(str).tolist()
    )
    if not candidates:
        st.info("조회할 이미지가 없습니다.", icon="🔍")
        return

    image_id = st.selectbox(
        "이미지", candidates, key="p4_trace_pick",
        help="추론 로그가 있으면 판정된 이미지만 보여준다.",
    )
    trace = monitoring.trace_image(image_id)

    left, right = st.columns([2, 3])
    with left:
        row = df[df["image_id"].astype(str) == str(image_id)]
        if not row.empty:
            image = viz.load_rgb(row.iloc[0]["path_abs"])
            if image is not None:
                st.image(image, caption=str(image_id), width="stretch")

    with right:
        manifest = trace["manifest"]
        effective = trace["effective"]
        if manifest:
            st.markdown("**① 수집 (1단계)**")
            st.caption(
                f"출처 `{manifest.get('source')}` · 카테고리 `{manifest.get('category')}` · "
                f"{manifest.get('width')}×{manifest.get('height')} · "
                f"선명도 {manifest.get('blur_score')} · 밝기 {manifest.get('brightness')}"
            )
        if effective:
            st.markdown("**② 유효 라벨 (2단계)**")
            st.caption(
                f"{config.LABEL_KO.get(str(effective.get('label')), effective.get('label'))} / "
                f"{config.defect_type_label(str(effective.get('defect_type')))} · "
                f"근거 `{effective.get('label_source')}` · 분할 `{effective.get('split')}`"
            )

    st.divider()
    st.markdown("**③ 라벨 이력**")
    events = trace["label_events"]
    if events is None or events.empty:
        st.caption("사람이 남긴 라벨 이벤트가 없습니다.")
    else:
        st.dataframe(events, hide_index=True, width="stretch")

    st.markdown("**④ 모델 판정 이력 (4단계 추론 로그)**")
    inferences = trace["inferences"]
    if inferences is None or inferences.empty:
        st.caption("이 이미지에 대한 추론 기록이 없습니다.")
    else:
        st.dataframe(
            inferences[["logged_at", "version", "score", "threshold", "decision", "latency_ms"]],
            hide_index=True, width="stretch",
        )

    st.markdown("**⑤ Claude 2차 판정 (3단계)**")
    reviews = trace["claude_reviews"]
    if reviews is None or reviews.empty:
        st.caption("2차 판정 기록이 없습니다.")
    else:
        for _, review in reviews.iterrows():
            with st.container(border=True):
                st.markdown(
                    f"**{config.defect_type_label(str(review['defect_type']))}** · "
                    f"심각도 {review['severity']} · 확신도 {review['confidence']} · "
                    f"모델 `{review['model']}`"
                )
                st.caption(str(review["reason"]))


# --- 페이지 -------------------------------------------------------------------

def _drift_causes(drift: pd.DataFrame) -> None:
    """분포가 변한 특징을 현장에서 확인할 것으로 옮겨 준다.

    `gray_mean`, `lap_p99` 같은 이름만 보고 "조명이 바뀌었나?"까지 연결하려면 각 특징이
    무엇을 재는지 알아야 하는데, 그건 이 파이프라인을 만든 사람만 안다.
    """
    shifted = drift[drift["level"].astype(str) == "변화"]
    if shifted.empty:
        return

    names = shifted.sort_values("psi", ascending=False)["feature"].astype(str).tolist()
    causes = glossary.drift_causes(names)
    if not causes:
        return

    st.markdown("**무엇을 확인해야 하나**")
    st.caption(
        "분포가 변한 특징으로부터 추정한 것입니다. 모델을 다시 학습하기 전에 "
        "**현장 조건이 바뀌지 않았는지 먼저 확인**하는 편이 빠릅니다."
    )
    for cause in causes[:4]:
        st.markdown(f"- {cause}")
    st.divider()


def _scenario_tab(df: pd.DataFrame) -> None:
    """운영 몇 달치를 재생해 나머지 탭이 작동하는 모습을 보이게 한다."""
    st.markdown(
        "드리프트 감시 · 성능 추이 · 재학습 판단은 **시간이 흐르고 데이터가 쌓여야** 의미가 생긴다. "
        "방금 만든 앱에서는 이 화면들이 전부 비어 있어 운영이 되는지 확인할 수가 없다. "
        "여기서 **운영 3개월치를 몇 초 만에 재생**하면 나머지 탭에 실제로 값이 채워진다."
    )
    st.info(
        "새 기능을 흉내 내는 것이 아닙니다. 배치 추론 · 드리프트 감시 · 성능 측정은 "
        "다른 탭에서 쓰는 것과 **같은 코드**를 그대로 실행합니다. 다른 것은 추론 시각을 "
        "과거로 채우고, 조명·초점 변화를 이미지에 입힌다는 점뿐입니다.",
        icon="ℹ️",
    )

    prod = registry.production()
    if prod is None:
        st.warning(
            "서비스 중인 모델이 없습니다. **모델 레지스트리** 탭에서 3단계 실행을 등록하고 승격해야 "
            "시나리오를 돌릴 수 있습니다.",
            icon="⚠️",
        )
        st.page_link(guide.PAGE_MODELING, label="3단계에서 모델 학습하기", icon="➡️")
        return
    if df.empty:
        st.warning("등록된 이미지가 없습니다. 1단계에서 데이터를 먼저 등록하세요.", icon="⚠️")
        return

    st.caption(f"대상 모델: **{prod['version']}** · 이미지 {len(df):,}건")

    with st.expander("재생할 시나리오", expanded=True):
        for index, phase in enumerate(scenario.DEFAULT_TIMELINE, start=1):
            st.markdown(
                f"**{index}. {phase.title}** — 환경: `{phase.environment.describe()}`  \n"
                f"{phase.narration}  \n"
                f"➡️ {phase.watch}"
            )
    st.caption(
        f"구간마다 {scenario.DEFAULT_TIMELINE[0].n_images}장을 검사하고 그중 "
        f"{scenario.DEFAULT_TIMELINE[0].verify_ratio:.0%}를 사후 검수합니다. "
        "검수 정답은 manifest 라벨에서 오고 모델 점수와 무관하므로 자기 채점이 아닙니다. "
        f"기록되는 라벨은 `{scenario.LABELED_BY}` 출처로 남아 사람이 검수한 라벨과 구분됩니다."
    )

    col1, col2 = st.columns([1, 1])
    if col1.button("🎬 운영 3개월치 재생", type="primary", key="p4_scen_run"):
        bar = st.progress(0.0, text="준비 중...")

        def on_progress(index: int, total: int, title: str) -> None:
            bar.progress(index / total, text=f"{title} 재생 중... ({index}/{total})")

        with st.spinner("추론과 검수를 재생하는 중..."):
            result = scenario.run(str(prod["version"]), df, progress=on_progress)
        bar.empty()
        st.session_state[_SCENARIO_KEY] = result
        st.rerun()

    if col2.button("🧹 시나리오 로그 지우기", key="p4_scen_clear"):
        removed = scenario.clear()
        st.session_state.pop(_SCENARIO_KEY, None)
        st.success(f"시나리오가 만든 추론 로그 {removed:,}건을 지웠습니다.", icon="✅")
        st.rerun()

    result = st.session_state.get(_SCENARIO_KEY)
    if result is None:
        return

    for warning in result.warnings:
        st.warning(warning, icon="⚠️")
    if not result.phases:
        return

    st.divider()
    st.subheader("재생 결과")
    cols = st.columns(3)
    cols[0].metric("쌓인 추론 로그", f"{result.total_logged:,}건")
    cols[1].metric("사후 검수", f"{result.total_verified:,}건")
    cols[2].metric("재생 기간", f"{sum(p.phase.days for p in result.phases)}일")

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "구간": p.phase.title,
                    "환경": p.phase.environment.describe(),
                    "검사": p.logged,
                    "결함 판정 비율": round(p.defect_rate, 3),
                    "평균 점수": round(p.mean_score, 2),
                    "재현율": round(p.recall, 2),
                    "검수된 결함": p.verified_defects,
                    "드리프트": p.drift_level,
                    "변화 특징 수": p.drift_changed,
                }
                for p in result.phases
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "재현율은 **검수된 결함 건수**가 적으면 크게 흔들린다. 옆 칸의 건수를 함께 보고 판단하세요."
    )

    for p in result.phases:
        if p.top_drift:
            st.markdown(
                f"**{p.phase.title}** — PSI 상위: "
                + ", ".join(f"`{name}` {value:.2f}" for name, value in p.top_drift)
            )

    st.success(
        "이제 **드리프트 감시 · 성능 추이 · 재학습 판단 · 판정 이력** 탭에 값이 채워져 있습니다. "
        "차례로 열어 확인하세요.",
        icon="✅",
    )
    st.warning(
        "여기서 나온 수치는 **화면 시연용이지 모델 성능 근거가 아닙니다.** "
        "환경 변화를 인위적으로 넣은 결과이므로 재현율·PSI를 성능으로 인용하면 안 됩니다.",
        icon="⚠️",
    )


def render() -> None:
    st.title("⚙️ 4단계 · 사후 운영관리 (MLOps)")
    st.caption(
        "모델을 만드는 것보다 만든 뒤 성능이 떨어지는 것을 알아채는 일이 어렵다. "
        "데이터 드리프트(정답 없이 감지)와 성능 드리프트(정답 필요)를 나눠 본다."
    )

    df = _resolved()
    tabs = st.tabs(
        ["🎬 운영 시나리오 시연", "📚 모델 레지스트리", "▶️ 배치 추론", "📉 드리프트 감시",
         "📈 성능 추이", "🔁 재학습 판단", "🔍 판정 이력"]
    )
    with tabs[0]:
        _scenario_tab(df)
    with tabs[1]:
        _registry_tab(df)
    with tabs[2]:
        _inference_tab(df)
    with tabs[3]:
        _drift_tab(df)
    with tabs[4]:
        _performance_tab(df)
    with tabs[5]:
        _retraining_tab(df)
    with tabs[6]:
        _trace_tab(df)


render()
