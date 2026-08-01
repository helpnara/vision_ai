"""3단계: 모델 개발 · 평가 화면."""

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
    models,
    storage,
    viz,
)

_DATASET_KEY = "p3_dataset"
_RESULT_KEY = "p3_result"
_ANOMALY_KEY = "p3_anomaly_model"


# --- 공통 --------------------------------------------------------------------

def _resolved() -> pd.DataFrame:
    df = labeling.resolve()
    if df.empty:
        return df
    df = df.copy()
    df["path_abs"] = [str(storage.resolve_path(p)) for p in df["path"]]
    df["y"] = (df["label"].astype(str) == config.LABEL_DEFECT).astype(int)
    return df


def _labeled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label"].astype(str).isin([config.LABEL_NORMAL, config.LABEL_DEFECT])]


def _split_signature(df: pd.DataFrame) -> str:
    """데이터가 바뀌면 캐시된 특징을 무효화하기 위한 지문."""
    if df.empty:
        return "empty"
    return f"{len(df)}:{hash(tuple(sorted(df['image_id'].astype(str))))}"


def _readiness(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """학습을 시작할 수 있는 상태인지 점검한다."""
    problems: list[str] = []
    labeled = _labeled(df)
    if labeled.empty:
        problems.append("라벨된 이미지가 없습니다. 1~2단계를 먼저 진행하세요.")
        return False, problems

    splits = labeled["split"].astype(str)
    for name in (config.SPLIT_TRAIN, config.SPLIT_TEST):
        if not (splits == name).any():
            problems.append(f"`{name}` 분할에 이미지가 없습니다. 2단계 → 데이터 분할에서 배정하세요.")

    train = labeled[splits == config.SPLIT_TRAIN]
    if not train.empty and train["y"].nunique() < 2:
        problems.append("학습 분할에 정상·결함이 모두 있어야 베이스라인을 학습할 수 있습니다.")
    return not problems, problems


def _ensure_dataset(labeled: pd.DataFrame) -> models.Dataset | None:
    """특징 행렬을 만들고 세션에 캐시한다."""
    signature = _split_signature(labeled)
    cached = st.session_state.get(_DATASET_KEY)
    if cached and cached.get("signature") == signature:
        return cached["dataset"]

    bar = st.progress(0.0, text="특징 추출 중...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / max(total, 1), text=f"특징 추출 중... {done:,}/{total:,}")

    dataset = models.build_dataset(
        labeled["path_abs"].tolist(),
        labeled["y"].tolist(),
        labeled["image_id"].astype(str).tolist(),
        progress=on_progress,
    )
    bar.empty()
    if len(dataset) == 0:
        st.error("특징을 추출할 수 있는 이미지가 없습니다.")
        return None
    st.session_state[_DATASET_KEY] = {"signature": signature, "dataset": dataset}
    if dataset.failed:
        st.warning(f"읽기 실패 {len(dataset.failed)}건은 제외했습니다.", icon="⚠️")
    return dataset


def _split_masks(dataset: models.Dataset, labeled: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = labeled.set_index(labeled["image_id"].astype(str))["split"].astype(str).to_dict()
    assigned = np.array([lookup.get(i, config.SPLIT_NONE) for i in dataset.image_ids])
    return {name: assigned == name for name in config.SPLITS}


def _store_result(**payload) -> None:
    st.session_state[_RESULT_KEY] = payload


def _synthetic_warning(labeled: pd.DataFrame) -> None:
    sources = set(labeled["source"].astype(str))
    if sources <= {"synthetic"}:
        st.warning(
            "지금 데이터는 **합성 샘플뿐**입니다. 결함이 인위적으로 뚜렷해 지표가 실제보다 높게 나옵니다. "
            "파이프라인 배선 확인용으로만 보고, 성능 근거로 삼지 마세요.",
            icon="🧪",
        )


# --- 1) 학습 데이터 -----------------------------------------------------------

def _data_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("수집된 이미지가 없습니다. 1단계에서 시작하세요.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    labeled = _labeled(df)
    ready, problems = _readiness(df)

    cols = st.columns(4)
    cols[0].metric("라벨된 이미지", f"{len(labeled):,}")
    cols[1].metric("정상", f"{int((labeled['y'] == 0).sum()):,}")
    cols[2].metric("결함", f"{int((labeled['y'] == 1).sum()):,}")
    cols[3].metric("카테고리", f"{labeled['category'].nunique():,}")

    for problem in problems:
        st.error(problem, icon="🚧")
    if ready:
        st.success("학습을 시작할 수 있습니다.", icon="✅")
    _synthetic_warning(labeled)

    if labeled.empty:
        return

    st.divider()
    st.markdown("**분할 × 라벨**")
    crosstab = pd.crosstab(
        labeled["split"].astype(str),
        labeled["label"].astype(str).map(lambda x: config.LABEL_KO.get(x, x)),
    )
    st.dataframe(crosstab, width="stretch")

    st.divider()
    st.markdown("### 특징 추출")
    st.caption(
        f"이미지 1장 → {len(features.FEATURE_NAMES)}차원 벡터 (밝기·엣지·색상·고주파·텍스처 통계). "
        f"이미지는 {features.IMAGE_SIZE}×{features.IMAGE_SIZE}로 맞춘 뒤 계산합니다."
    )
    if st.button("🧮 특징 추출 / 갱신", type="primary", key="p3_extract"):
        st.session_state.pop(_DATASET_KEY, None)
        _ensure_dataset(labeled)

    cached = st.session_state.get(_DATASET_KEY)
    if cached and cached.get("signature") == _split_signature(labeled):
        dataset = cached["dataset"]
        st.success(f"특징 행렬 준비됨: {dataset.X.shape[0]:,}행 × {dataset.X.shape[1]}열", icon="✅")
        with st.expander("특징 목록"):
            st.write(", ".join(features.FEATURE_NAMES))
    else:
        st.info("아직 특징을 추출하지 않았습니다.", icon="🧮")


# --- 2) 베이스라인 ------------------------------------------------------------

def _advice_banner(df: pd.DataFrame, kind: str) -> bool:
    """이 탭의 방식이 지금 데이터로 가능한지, 권장되는지 미리 알려준다.

    선택지만 주고 무엇을 골라야 할지 알려주지 않으면 초보자는 막힌다. 특히 VisA 공식 분할은
    학습 분할에 결함이 없어 지도학습이 시작조차 안 되는데, 화면은 선택지를 똑같이 보여준다.
    """
    advice = guide.model_advice(df)
    blocked = advice.baseline_blocked if kind == guide.KIND_BASELINE else advice.anomaly_blocked

    if blocked:
        st.error(
            "**지금 데이터로는 이 방식을 쓸 수 없습니다.**\n\n"
            + "\n\n".join(advice.notes),
            icon="🚧",
        )
        other = "이상탐지" if kind == guide.KIND_BASELINE else "베이스라인"
        st.info(f"대신 **{other}** 탭을 사용하세요. {advice.reason}", icon="👉")
        return True

    if advice.recommended == kind:
        st.success(f"**이 데이터에는 이 방식을 권합니다.** {advice.reason}", icon="✅")
    else:
        other = "이상탐지" if advice.recommended == guide.KIND_ANOMALY else "베이스라인"
        st.info(f"쓸 수는 있지만 이 데이터에는 **{other}** 를 권합니다. {advice.reason}", icon="💡")

    for note in advice.notes:
        st.warning(note, icon="⚠️")
    return False


def _empty_state_links(df: pd.DataFrame) -> None:
    """막혔을 때 어디로 가야 하는지 링크로 알려준다. 문구만 띄우면 초보자는 길을 잃는다."""
    if df.empty:
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return
    labeled = _labeled(df)
    if labeled.empty:
        st.page_link(guide.PAGE_LABELING, label="2단계 라벨 검수로 이동", icon="➡️")
    else:
        st.page_link(guide.PAGE_LABELING, label="2단계 데이터 분할로 이동", icon="➡️")


def _baseline_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "고전 CV 특징 + 분류기로 **성능 하한선**을 만든다. 무거운 모델 없이 곧바로 돌아가므로, "
        "이후 모델이 이보다 나은지 판단하는 기준이 된다."
    )
    if df.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    if _advice_banner(df, guide.KIND_BASELINE):
        return   # 이 데이터로는 불가능하다 — 안내에서 이유와 대안을 이미 설명했다

    ready, problems = _readiness(df)
    if not ready:
        for problem in problems:
            st.error(problem, icon="🚧")
        _empty_state_links(df)
        return

    labeled = _labeled(df)
    dataset = _ensure_dataset(labeled)
    if dataset is None:
        return
    masks = _split_masks(dataset, labeled)

    col1, col2, col3 = st.columns(3)
    kind = col1.selectbox(
        "분류기", list(models.BASELINE_KINDS),
        format_func=lambda k: models.BASELINE_KIND_LABELS.get(k, k), key="p3_bl_kind",
    )
    balanced = col2.checkbox(
        "클래스 불균형 보정", value=True, key="p3_bl_balanced",
        help="정상이 결함보다 훨씬 많을 때 결함 쪽에 가중치를 준다.",
    )
    eval_split = col3.selectbox(
        "평가 분할", [config.SPLIT_TEST, config.SPLIT_VAL], key="p3_bl_split",
        help="임계값 탐색은 val에서 하고 최종 보고는 test로 하는 것이 정석이다.",
    )

    target_recall = st.slider(
        "목표 재현율 (미탐 최소화 기준)", 0.50, 1.00, 0.95, 0.01, key="p3_bl_recall",
        help="이 재현율을 만족하는 임계값 중 오탐이 가장 적은 값을 자동 선택한다.",
    )

    n_train = int(masks[config.SPLIT_TRAIN].sum())
    n_eval = int(masks[eval_split].sum())
    st.caption(f"학습 {n_train:,}장 · 평가({eval_split}) {n_eval:,}장")
    if n_eval == 0:
        st.error(f"`{eval_split}` 분할에 이미지가 없습니다.", icon="🚧")
        return

    if not st.button("🧠 학습 후 평가", type="primary", key="p3_bl_run"):
        return

    train_mask, eval_mask = masks[config.SPLIT_TRAIN], masks[eval_split]
    try:
        with st.spinner("학습 중..."):
            model = models.BaselineModel(
                models.BaselineConfig(kind=kind, balanced=balanced)
            ).fit(dataset.X[train_mask], dataset.y[train_mask])
            scores = model.score(dataset.X[eval_mask])
    except (ValueError, RuntimeError) as exc:
        st.error(f"학습 실패: {exc}")
        return

    y_eval = dataset.y[eval_mask]
    threshold = evaluate.threshold_for_target_recall(y_eval, scores, target_recall)
    if threshold is None:
        threshold = float(np.median(scores))
        st.warning(
            f"목표 재현율 {target_recall:.0%}를 만족하는 임계값이 없습니다. 중앙값을 임시로 사용합니다.",
            icon="⚠️",
        )

    metrics = evaluate.summarize(y_eval, scores, threshold)
    ids = [dataset.image_ids[i] for i in np.flatnonzero(eval_mask)]

    _store_result(
        kind="baseline", model=kind, split=eval_split, scores=scores, y=y_eval,
        image_ids=ids, threshold=threshold, metrics=metrics,
        settings={"kind": kind, "balanced": balanced, "target_recall": target_recall},
        n_train=n_train,
    )

    importance = model.feature_importance()
    artifact_path = config.MODEL_DIR / f"baseline_{kind}.joblib"
    try:
        model.save(artifact_path)
    except Exception as exc:  # 저장 실패가 평가를 막지는 않게 한다
        st.warning(f"모델 저장 실패: {exc}", icon="⚠️")
        artifact_path = None

    run_id = experiments.record_run(
        kind="baseline", model=kind, split=eval_split, metrics=metrics,
        settings={"kind": kind, "balanced": balanced, "target_recall": target_recall},
        n_train=n_train, n_eval=n_eval,
        artifacts={"model": str(artifact_path) if artifact_path else None},
    )
    st.success(f"학습·평가 완료 — 실험 기록 `{run_id}`", icon="✅")
    _metric_row(metrics)

    if importance is not None and len(importance) == len(features.FEATURE_NAMES):
        with st.expander("특징 중요도 상위 15개"):
            order = np.argsort(-importance)[:15]
            st.bar_chart(
                pd.Series(importance[order], index=[features.FEATURE_NAMES[i] for i in order]),
                horizontal=True,
            )

    st.info("자세한 임계값 조정과 오탐/미탐 샘플은 **평가 리포트** 탭에서 확인합니다.", icon="📈")


# --- 3) 이상탐지 --------------------------------------------------------------

def _anomaly_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "**정상 이미지만** 학습해 정상 분포를 만들고, 이탈 정도를 결함 점수로 쓴다. "
        "결함 샘플이 적은 상황(VisA의 기본 전제)에 맞고, 결함 **위치**까지 히트맵으로 낸다."
    )
    if df.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    if _advice_banner(df, guide.KIND_ANOMALY):
        return

    labeled = _labeled(df)
    train_normal = labeled[
        (labeled["split"].astype(str) == config.SPLIT_TRAIN) & (labeled["y"] == 0)
    ]
    col1, col2, col3 = st.columns(3)
    eval_split = col1.selectbox(
        "평가 분할", [config.SPLIT_TEST, config.SPLIT_VAL], key="p3_an_split"
    )
    per_position = col2.checkbox(
        "위치별 분포 학습", value=True, key="p3_an_perpos",
        help=(
            "격자 위치마다 별도 정상 분포를 학습한다. PCB처럼 정렬된 이미지에 유리하지만 "
            f"정상 이미지가 특징 차원({features.PATCH_FEATURE_DIM})보다 많아야 한다."
        ),
    )
    score_mode = col3.selectbox(
        "이미지 점수 집계", list(models.IMAGE_SCORE_MODES),
        index=list(models.IMAGE_SCORE_MODES).index("p99"),
        format_func=models.score_mode_label, key="p3_an_mode",
        help="격자 점수들을 이미지 1장의 점수로 합치는 방식. 결함이 작으면 max 쪽이 민감하다.",
    )
    target_recall = st.slider(
        "목표 재현율", 0.50, 1.00, 0.95, 0.01, key="p3_an_recall"
    )

    eval_rows = labeled[labeled["split"].astype(str) == eval_split]
    st.caption(
        f"정상 학습 {len(train_normal):,}장 · 평가({eval_split}) {len(eval_rows):,}장 "
        f"(결함 {int((eval_rows['y'] == 1).sum()):,}장)"
    )
    if train_normal.empty or eval_rows.empty:
        st.error("학습용 정상 이미지와 평가 분할이 모두 필요합니다.", icon="🚧")
        return
    if per_position and len(train_normal) <= features.PATCH_FEATURE_DIM:
        st.warning(
            f"정상 이미지가 {len(train_normal)}장으로 특징 차원({features.PATCH_FEATURE_DIM})보다 "
            "많지 않습니다. '위치별 분포 학습'을 끄거나 정상 이미지를 늘리세요.",
            icon="⚠️",
        )

    if not st.button("🔎 이상탐지 학습 후 평가", type="primary", key="p3_an_run"):
        _heatmap_section(df)
        return

    bar = st.progress(0.0, text="정상 분포 학습 중...")
    try:
        images = []
        for index, path in enumerate(train_normal["path_abs"], start=1):
            image = viz.load_rgb(path)
            if image is not None:
                images.append(image)
            bar.progress(index / len(train_normal) * 0.5, text=f"정상 학습 {index}/{len(train_normal)}")
        model = models.PatchAnomalyModel(
            models.AnomalyConfig(per_position=per_position, image_score=score_mode)
        ).fit(images)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        bar.empty()
        st.error(f"학습 실패: {exc}")
        return

    scores, ys, ids = [], [], []
    for index, (_, row) in enumerate(eval_rows.iterrows(), start=1):
        image = viz.load_rgb(row["path_abs"])
        if image is None:
            continue
        scores.append(model.image_score(image))
        ys.append(int(row["y"]))
        ids.append(str(row["image_id"]))
        bar.progress(0.5 + index / len(eval_rows) * 0.5, text=f"평가 {index}/{len(eval_rows)}")
    bar.empty()

    if not scores:
        st.error("평가할 이미지를 읽지 못했습니다.")
        return

    scores_array, y_array = np.asarray(scores), np.asarray(ys)
    threshold = evaluate.threshold_for_target_recall(y_array, scores_array, target_recall)
    if threshold is None:
        threshold = float(np.median(scores_array))
        st.warning(f"목표 재현율 {target_recall:.0%} 달성 임계값이 없어 중앙값을 사용합니다.", icon="⚠️")

    metrics = evaluate.summarize(y_array, scores_array, threshold)
    settings = {
        "per_position": per_position, "image_score": score_mode,
        "patch": model.config.patch, "stride": model.config.stride,
        "target_recall": target_recall,
    }
    st.session_state[_ANOMALY_KEY] = model
    _store_result(
        kind="anomaly", model="mahalanobis", split=eval_split, scores=scores_array,
        y=y_array, image_ids=ids, threshold=threshold, metrics=metrics,
        settings=settings, n_train=model.n_train,
    )

    artifact_path = config.MODEL_DIR / "anomaly_patch.npz"
    try:
        model.save(artifact_path)
    except Exception as exc:
        st.warning(f"모델 저장 실패: {exc}", icon="⚠️")
        artifact_path = None

    run_id = experiments.record_run(
        kind="anomaly", model="mahalanobis", split=eval_split, metrics=metrics,
        settings=settings, n_train=model.n_train, n_eval=len(ids),
        artifacts={"model": str(artifact_path) if artifact_path else None},
    )
    st.success(f"학습·평가 완료 — 실험 기록 `{run_id}`", icon="✅")
    _metric_row(metrics)
    _heatmap_section(df)


def _heatmap_section(df: pd.DataFrame) -> None:
    """학습된 이상탐지 모델로 결함 위치 히트맵을 보여준다."""
    model = st.session_state.get(_ANOMALY_KEY)
    if model is None:
        return

    st.divider()
    st.markdown("### 결함 위치 히트맵")
    defects = _labeled(df)
    defects = defects[defects["y"] == 1]
    if defects.empty:
        st.caption("결함 이미지가 없습니다.")
        return

    options = defects["image_id"].astype(str).tolist()
    picked = st.selectbox("이미지", options, key="p3_heat_pick")
    row = defects[defects["image_id"].astype(str) == picked].iloc[0]
    image = viz.load_rgb(row["path_abs"])
    if image is None:
        st.error("이미지를 읽을 수 없습니다.")
        return

    resized = features.preprocess(image)
    heatmap = model.score_map(resized)
    normalized = heatmap - heatmap.min()
    normalized = normalized / max(float(normalized.max()), 1e-6)

    columns = st.columns(3)
    columns[0].image(resized, caption="입력", width="stretch")
    columns[1].image(normalized, caption="결함 점수 히트맵", width="stretch", clamp=True)
    columns[2].image(
        viz.overlay_mask(resized, (normalized > 0.6).astype(np.uint8) * 255, alpha=0.45),
        caption="상위 영역 오버레이", width="stretch",
    )

    mask_path = labeling.find_mask_path(storage.resolve_path(row["path"]))
    if mask_path is None:
        st.caption("정답 마스크가 없어 위치 정확도는 계산할 수 없습니다.")
        return

    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    scores = evaluate.localization_metrics(heatmap, mask)
    cols = st.columns(3)
    cols[0].metric("최고점 명중", "예" if scores["hit"] else "아니오")
    cols[1].metric("픽셀 AUROC", f"{scores['pixel_auroc']:.3f}" if scores["pixel_auroc"] == scores["pixel_auroc"] else "—")
    cols[2].metric("IoU@p99", f"{scores['iou']:.3f}" if scores["iou"] == scores["iou"] else "—")


# --- 4) Claude 2차 판정 -------------------------------------------------------

def _claude_tab(df: pd.DataFrame) -> None:
    st.markdown(
        "1차 모델이 **애매하다고 본 이미지만** Claude에 넘겨 결함 유형과 판단 근거를 받는다. "
        "전량을 보내면 비용이 선형으로 늘기 때문에, 임계값 근처만 골라 보내는 것이 핵심이다."
    )

    if not claude_review.sdk_installed():
        st.error(
            "`anthropic` SDK가 설치되지 않았습니다. `pip install anthropic` 후 다시 시도하세요.",
            icon="📦",
        )
    hint = claude_review.credentials_hint()
    if hint:
        st.warning(hint, icon="🔑")

    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info(
            "먼저 **베이스라인** 또는 **이상탐지** 탭에서 1차 모델을 실행하세요. "
            "그 결과에서 애매한 이미지를 골라옵니다.",
            icon="🧠",
        )
        return

    errors = evaluate.error_frame(
        result["image_ids"], result["y"], result["scores"], result["threshold"]
    )
    col1, col2, col3 = st.columns(3)
    limit = col1.number_input("최대 판정 건수", 1, 100, 8, 1, key="p3_cl_limit")
    effort = col2.selectbox(
        "추론 강도(effort)", ["low", "medium", "high"], key="p3_cl_effort",
        help="단일 이미지 판정은 짧은 과업이라 low로도 충분한 편이다. 비용·지연에 직접 영향을 준다.",
    )
    skip_reviewed = col3.checkbox(
        "이미 판정한 이미지 제외", value=True, key="p3_cl_skip",
        help="같은 이미지를 두 번 청구하지 않는다.",
    )

    candidates = claude_review.select_uncertain(
        errors, threshold=result["threshold"], limit=int(limit) * 3
    )
    if skip_reviewed:
        done = claude_review.reviewed_ids()
        candidates = candidates[~candidates["image_id"].isin(done)]
    candidates = candidates.head(int(limit))

    if candidates.empty:
        st.success("2차 판정이 필요한 이미지가 없습니다.", icon="🎉")
    else:
        st.caption(
            f"1차 모델 임계값 {result['threshold']:.4f} 근처의 {len(candidates)}건을 선정했습니다. "
            f"예상 비용 약 ${claude_review.estimate_cost(len(candidates)):.3f} "
            f"(모델 {claude_review.DEFAULT_MODEL}, 이미지 해상도에 따라 달라짐)"
        )
        st.dataframe(
            candidates[["image_id", "label", "score", "outcome", "uncertainty"]],
            hide_index=True, width="stretch",
        )

        disabled = not claude_review.sdk_installed()
        if st.button("🤖 Claude 2차 판정 실행", type="primary", disabled=disabled, key="p3_cl_run"):
            _run_claude(df, candidates, effort=effort)

    st.divider()
    st.markdown("### 판정 이력")
    reviews = claude_review.load_reviews()
    if reviews.empty:
        st.caption("아직 판정 기록이 없습니다.")
        return

    ok = reviews[reviews["status"] == "ok"]
    cols = st.columns(4)
    cols[0].metric("전체 판정", f"{len(reviews):,}")
    cols[1].metric("성공", f"{len(ok):,}")
    cols[2].metric("결함 판정", f"{int(ok['defect'].astype(str).isin(['True', 'true']).sum()):,}")
    total_tokens = pd.to_numeric(reviews["input_tokens"], errors="coerce").fillna(0).sum()
    cols[3].metric("입력 토큰 합", f"{int(total_tokens):,}")

    st.dataframe(
        reviews[["image_id", "defect", "defect_type", "severity", "confidence", "reason", "status", "model"]],
        hide_index=True, width="stretch", height=280,
    )
    st.download_button(
        "판정 결과 내려받기",
        reviews.to_csv(index=False).encode("utf-8-sig"),
        file_name="claude_reviews.csv", mime="text/csv",
    )


def _run_claude(df: pd.DataFrame, candidates: pd.DataFrame, *, effort: str) -> None:
    lookup = df.set_index(df["image_id"].astype(str))
    bar = st.progress(0.0, text="Claude 판정 중...")
    results = []

    for index, (_, candidate) in enumerate(candidates.iterrows(), start=1):
        image_id = str(candidate["image_id"])
        if image_id not in lookup.index:
            continue
        row = lookup.loc[image_id]
        image = viz.load_rgb(row["path_abs"])
        if image is None:
            continue

        roi = None
        values = [row.get(c) for c in ("roi_x", "roi_y", "roi_w", "roi_h")]
        if all(v is not None and not pd.isna(v) for v in values):
            roi = tuple(int(v) for v in values)

        results.append(
            claude_review.review_image(
                image,
                image_id=image_id,
                roi=roi,
                category=str(row["category"]),
                candidate_type=str(row["defect_type"]),
                model_score=float(candidate["score"]),
                effort=effort,
            )
        )
        bar.progress(index / len(candidates), text=f"Claude 판정 중... {index}/{len(candidates)}")
    bar.empty()

    if not results:
        st.error("판정할 이미지를 준비하지 못했습니다.")
        return

    claude_review.save_reviews(results)
    statuses = pd.Series([r.status for r in results]).value_counts().to_dict()
    if statuses.get("ok"):
        st.success(f"판정 완료: {statuses}", icon="✅")
    else:
        st.error(f"모든 판정이 실패했습니다: {statuses}")
        st.caption(f"첫 실패 사유: {results[0].reason}")
    st.rerun()


# --- 5) 평가 리포트 -----------------------------------------------------------

def _metric_row(metrics: dict) -> None:
    """지표를 표시한다.

    핵심 지표는 캡션으로 **바로 보이게** 하고, 부가 설명은 `?` 아이콘에 둔다.
    눌러야 보이는 설명은 초보자가 그냥 지나치기 때문이다. 문구는 `glossary`에 모아 두어
    다른 화면과 어긋나지 않게 한다.
    """
    auroc = metrics.get("auroc")
    items = [
        ("recall", "재현율", f"{metrics['recall']:.3f}"),
        ("precision", "정밀도", f"{metrics['precision']:.3f}"),
        ("f1", "F1", f"{metrics['f1']:.3f}"),
        ("auroc", "AUROC", f"{auroc:.3f}" if auroc == auroc else "—"),
        ("miss_fp", "미탐 / 오탐", f"{metrics['fn']} / {metrics['fp']}"),
    ]
    cols = st.columns(len(items))
    for col, (key, label, value) in zip(cols, items):
        col.metric(label, value, help=glossary.detail(key))
        caption = glossary.caption(key)
        if caption:
            col.caption(caption)


def _business_impact(metrics: dict) -> dict:
    """성능 지표를 "사람이 볼 물량이 얼마나 줄어드는가"로 바꿔 보여준다.

    AUROC나 정밀도만 보면 이 도구를 도입할지 판단할 수 없다. 게다가 여기 평가 데이터는
    정상:결함 비율이 실제 라인과 달라서, 화면의 정밀도를 그대로 믿으면 크게 과대평가한다.
    """
    recall = float(metrics.get("recall", 0.0))
    fpr = float(metrics.get("false_alarm_rate", 0.0))

    st.divider()
    st.markdown("### 이 모델을 쓰면 무엇이 좋아지는가")

    col1, col2, col3 = st.columns([1, 1, 2])
    prevalence = col1.number_input(
        "가정 불량률 (%)", 0.1, 50.0, evaluate.DEFAULT_PREVALENCE * 100, 0.1,
        key="p3_rep_prevalence",
        help="실제 라인에서 100개 중 몇 개가 불량인지. 모르면 1%로 두고 본다.",
    ) / 100.0
    volume = int(col2.number_input(
        "검사 물량 (장)", 100, 1_000_000, 1000, 100, key="p3_rep_volume",
        help="이 물량을 기준으로 건수를 환산한다.",
    ))

    impact = evaluate.business_impact(recall, fpr, prevalence=prevalence, volume=volume)

    cols = st.columns(3)
    cols[0].metric("검수량 절감", f"{impact['reduction_ratio']:.0%}", help=glossary.detail("reduction"))
    cols[0].caption(glossary.caption("reduction"))
    cols[1].metric("놓치는 결함", f"{impact['missed']:.0f}건", help=glossary.detail("missed"))
    cols[1].caption(glossary.caption("missed"))
    cols[2].metric(
        "현장 기대 정밀도", f"{impact['precision']:.1%}",
        help="위 표의 정밀도는 평가 데이터 구성 기준이라 현장과 다릅니다.",
    )
    cols[2].caption("위에서 본 정밀도를 실제 불량률로 환산한 값입니다.")

    st.info(
        f"**{volume:,}장을 검사하면** — 결함 {impact['defects']:.0f}건 중 "
        f"**{impact['caught']:.0f}건을 잡고 {impact['missed']:.0f}건을 놓친다.** "
        f"사람이 볼 물량은 {volume:,}장에서 **{impact['reviewed']:.0f}장으로 줄어든다** "
        f"(그중 오탐 {impact['false_alarms']:.0f}건을 걸러내야 한다).",
        icon="💡",
    )

    if impact["precision"] < 0.5:
        st.warning(
            f"불량률 {prevalence:.1%}에서 기대 정밀도가 **{impact['precision']:.1%}** 다. "
            "즉 **자동 판정용으로는 쓸 수 없고**, 사람 검수 부하를 줄이는 "
            "**1차 스크리닝용**으로 봐야 한다. 모델이 올린 것은 사람이 다시 확인해야 한다.",
            icon="⚠️",
        )

    with st.expander("불량률이 달라지면 (정밀도가 왜 이렇게 떨어지는가)"):
        st.caption(
            "재현율과 오탐률은 불량률과 무관한 모델 고유 특성이다. 반면 정밀도는 불량률에 따라 "
            "크게 달라진다. 결함이 드물수록 정상품이 압도적으로 많아져, 같은 오탐률이라도 "
            "오탐 건수가 진짜 결함 건수를 쉽게 넘어서기 때문이다."
        )
        table = evaluate.impact_by_prevalence(recall, fpr, volume=volume)
        display = table.copy()
        display["불량률"] = display["불량률"].map(lambda v: f"{v:.0%}")
        for column in ("기대 정밀도", "검수 비율", "검수량 절감률"):
            display[column] = display[column].map(lambda v: f"{v:.1%}")
        for column in display.columns:
            if "장당" in column:
                display[column] = display[column].map(lambda v: f"{v:,.0f}")
        st.dataframe(display, hide_index=True, width="stretch")
    return impact


def _verdict(metrics: dict, impact: dict, target_recall: float) -> None:
    """숫자를 보여주고 끝내지 않고, 쓸 만한지와 다음에 무엇을 할지 문장으로 말해준다."""
    result = glossary.verdict(metrics, impact, target_recall=target_recall)
    render = {
        glossary.LEVEL_GOOD: (st.success, "✅"),
        glossary.LEVEL_USABLE: (st.info, "💡"),
        glossary.LEVEL_WEAK: (st.warning, "⚠️"),
    }[result.level]
    render[0](f"**{result.headline}**", icon=render[1])
    if result.actions:
        st.markdown("**다음에 할 일**")
        for action in result.actions:
            st.markdown(f"- {action}")


def _report_tab(df: pd.DataFrame) -> None:
    result = st.session_state.get(_RESULT_KEY)
    if result is None:
        st.info(
            "먼저 **베이스라인** 또는 **이상탐지** 탭에서 모델을 실행하세요. "
            "어느 쪽을 골라야 할지는 각 탭이 데이터를 보고 알려줍니다.",
            icon="🧠",
        )
        return

    scores, y = result["scores"], result["y"]
    st.markdown(
        f"**{result['kind']} / {result['model']}** · 평가 분할 `{result['split']}` · "
        f"학습 {result['n_train']:,}장 / 평가 {len(y):,}장"
    )
    _synthetic_warning(_labeled(df))

    st.divider()
    st.markdown("### 임계값 조정")
    st.caption(
        "임계값은 **어느 점수부터 결함으로 볼지** 정하는 값입니다. "
        "낮추면 결함을 더 많이 잡지만(재현율 ↑) 정상품도 함께 걸립니다(오탐 ↑). "
        "한쪽만 좋게 만들 수는 없습니다 — 어디서 타협할지를 고르는 것입니다."
    )
    low, high = float(np.min(scores)), float(np.max(scores))
    if high <= low:
        high = low + 1e-6
    threshold = st.slider(
        "판정 임계값", low, high, float(np.clip(result["threshold"], low, high)),
        (high - low) / 200, key="p3_rep_thr",
    )
    metrics = evaluate.summarize(y, scores, threshold)
    _metric_row(metrics)

    impact = _business_impact(metrics)
    _verdict(metrics, impact, float(result.get("settings", {}).get("target_recall", 0.95)))

    left, right = st.columns(2)
    with left:
        st.markdown("**혼동행렬**")
        st.dataframe(
            pd.DataFrame(
                [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                index=["실제 정상", "실제 결함"], columns=["예측 정상", "예측 결함"],
            ),
            width="stretch",
        )
        st.caption(f"미탐률 {metrics['miss_rate']:.1%} · 오경보율 {metrics['false_alarm_rate']:.1%}")
    with right:
        st.markdown("**임계값별 재현율 / 정밀도**")
        sweep = evaluate.threshold_sweep(y, scores)
        if not sweep.empty:
            st.line_chart(sweep.set_index("threshold")[["recall", "precision"]])

    with st.expander("ROC 곡선"):
        roc = evaluate.roc_curve_frame(y, scores)
        if roc.empty:
            st.caption("한 클래스만 있어 ROC를 그릴 수 없습니다.")
        else:
            st.line_chart(roc.set_index("fpr")[["tpr"]])

    st.divider()
    st.markdown("### 오탐 · 미탐 샘플")
    st.caption("숫자만으로는 무엇이 잘못됐는지 알 수 없다. 실제 이미지를 봐야 다음 조치가 정해진다.")

    errors = evaluate.error_frame(result["image_ids"], y, scores, threshold)
    reviews = claude_review.load_reviews()
    if not reviews.empty:
        errors = errors.merge(
            reviews[["image_id", "defect_type", "reason", "status"]].rename(
                columns={"defect_type": "claude_type", "reason": "claude_reason"}
            ),
            on="image_id", how="left",
        )

    outcome = st.radio(
        "표시할 종류", ["FN", "FP", "TP", "TN"], horizontal=True, key="p3_rep_outcome",
        format_func=lambda x: {
            "FN": "미탐 (놓친 결함)", "FP": "오탐 (정상을 결함으로)",
            "TP": "정탐", "TN": "정상 판정",
        }[x],
    )
    subset = errors[errors["outcome"] == outcome]
    st.caption(f"{outcome} {len(subset)}건")
    if subset.empty:
        st.success(f"{outcome} 사례가 없습니다.", icon="🎉")
    else:
        _sample_gallery(df, subset.head(8))
        with st.expander("표로 보기"):
            st.dataframe(subset, hide_index=True, width="stretch")

    st.download_button(
        "판정 결과 내려받기 (errors.csv)",
        errors.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"eval_{result['kind']}_{result['split']}.csv", mime="text/csv",
    )


def _sample_gallery(df: pd.DataFrame, subset: pd.DataFrame) -> None:
    lookup = df.set_index(df["image_id"].astype(str))
    columns = st.columns(4)
    for position, (_, row) in enumerate(subset.iterrows()):
        image_id = str(row["image_id"])
        target = columns[position % 4]
        if image_id not in lookup.index:
            target.warning(f"{image_id}: 원본 없음")
            continue
        source = lookup.loc[image_id]
        image = viz.load_rgb(source["path_abs"])
        if image is None:
            target.error(f"{image_id}: 읽기 실패")
            continue
        caption = f"{source['category']} · 점수 {row['score']:.3f}"
        claude_type = row.get("claude_type")
        if isinstance(claude_type, str) and claude_type:
            caption += f" · Claude: {claude_type}"
        target.image(image, caption=caption, width="stretch")


# --- 6) 실험 기록 -------------------------------------------------------------

def _experiments_tab() -> None:
    st.markdown(
        "설정과 지표를 같은 줄에 남겨 실행을 비교한다. 4단계 모델 레지스트리의 토대가 된다."
    )
    runs = experiments.comparison_frame()
    if runs.empty:
        st.info("아직 기록된 실행이 없습니다. 모델을 학습하면 자동으로 남습니다.", icon="🧪")
        return

    st.dataframe(runs, hide_index=True, width="stretch")

    best = runs.dropna(subset=["recall"])
    if not best.empty:
        top = best.sort_values(["recall", "precision"], ascending=False).iloc[0]
        st.success(
            f"현재 최고 재현율: `{top['run_id']}` ({top['kind']}/{top['model']}) — "
            f"재현율 {top['recall']:.3f}, 정밀도 {top['precision']:.3f}, 미탐 {int(top['fn'])}건",
            icon="🏆",
        )

    picked = st.selectbox("상세 보기", runs["run_id"].tolist(), key="p3_exp_pick")
    detail = experiments.load_run_detail(picked)
    if detail:
        st.json(detail)

    st.download_button(
        "실험 기록 내려받기",
        experiments.load_runs().to_csv(index=False).encode("utf-8-sig"),
        file_name="experiments.csv", mime="text/csv",
    )


# --- 페이지 ------------------------------------------------------------------

def render() -> None:
    st.title("🧠 3단계 · 모델 개발 · 평가")
    st.caption(
        "베이스라인 → 이상탐지 → Claude 2차 판정 순으로 쌓는다. "
        "미탐(결함을 놓침)을 최우선 리스크로 보고 재현율 중심으로 평가한다."
    )

    df = _resolved()
    tabs = st.tabs(
        ["📦 학습 데이터", "🧮 베이스라인", "🔎 이상탐지", "🤖 Claude 2차 판정", "📈 평가 리포트", "🧪 실험 기록"]
    )
    with tabs[0]:
        _data_tab(df)
    with tabs[1]:
        _baseline_tab(df)
    with tabs[2]:
        _anomaly_tab(df)
    with tabs[3]:
        _claude_tab(df)
    with tabs[4]:
        _report_tab(df)
    with tabs[5]:
        _experiments_tab()


render()
