"""2단계: 라벨링 화면."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from vision_ai import config, guide, labeling, storage, ui, viz

# 라벨을 기록하면 큐에서 빠지는 모드 (커서를 그대로 두면 다음 항목이 올라온다)
_DRAINING_MODES = frozenset({"unlabeled", "unspecified", "unmapped"})

QUEUE_MODES = {
    "unspecified": "결함이지만 유형 미지정 (VisA는 대부분 여기)",
    "unlabeled": "라벨 없음",
    "unverified": "폴더 라벨 미확인",
    "unmapped": "표준 유형으로 정규화되지 않은 결함",
    "all": "전체",
}

DEFECT_TYPE_CHOICES = list(config.DEFECT_TYPES)


def _defect_label(key: str) -> str:
    return f"{config.defect_type_label(key)} ({key})"


def _cursor_key(mode: str, category: str) -> str:
    return f"p2_cursor::{mode}::{category}"


# --- 라벨 검수 큐 ----------------------------------------------------------

def _review_tab(resolved: pd.DataFrame) -> None:
    if resolved.empty:
        st.info("수집된 이미지가 없습니다. 1단계에서 먼저 등록하세요.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    col1, col2 = st.columns([2, 1])
    mode = col1.selectbox(
        "검수 대상", list(QUEUE_MODES), format_func=lambda k: QUEUE_MODES[k], key="p2_mode"
    )
    categories = ["(전체)"] + sorted(resolved["category"].dropna().astype(str).unique().tolist())
    category = col2.selectbox("카테고리", categories, key="p2_category")

    queue = labeling.review_queue(
        resolved, mode=mode, category=None if category == "(전체)" else category
    )
    if queue.empty:
        st.success("이 조건에 검수할 이미지가 없습니다.", icon="🎉")
        return

    cursor_key = _cursor_key(mode, category)
    cursor = min(st.session_state.get(cursor_key, 0), len(queue) - 1)
    st.session_state[cursor_key] = cursor
    row = queue.iloc[cursor]

    st.progress((cursor + 1) / len(queue), text=f"{cursor + 1} / {len(queue)}")

    # 컬럼을 먼저 만들고 폼을 먼저 처리한다 — 그래야 선택한 ROI를 이미지에 미리 그려서
    # 저장 전에 눈으로 확인할 수 있다 (Streamlit은 실행 순서가 아니라 컨테이너로 배치한다).
    image_col, form_col = st.columns([3, 2])

    path = storage.resolve_path(row["path"])
    rgb = viz.load_rgb(path)

    with form_col:
        st.markdown(
            f"**현재 라벨**: {config.LABEL_KO.get(str(row['label']), row['label'])}"
            f" / {config.defect_type_label(str(row['defect_type']))}"
        )
        st.caption(
            f"출처 `{row['source']}` · 라벨 근거 `{row['label_source']}` · "
            f"원본 유형 `{row['raw_defect_type']}` · 확인 {'✅' if row['verified'] else '미확인'}"
        )

        label_options = [config.LABEL_NORMAL, config.LABEL_DEFECT, config.LABEL_UNLABELED]
        current_label = str(row["label"])
        new_label = st.radio(
            "판정",
            label_options,
            index=label_options.index(current_label) if current_label in label_options else 2,
            format_func=lambda x: config.LABEL_KO.get(x, x),
            horizontal=True,
            key=f"p2_label::{row['image_id']}",
        )

        # 유형이 아직 정해지지 않은 이미지는 기본 선택을 두지 않는다.
        # 기본값을 두면 그냥 저장을 눌렀을 때 엉뚱한 유형이 붙는다.
        current_type = str(row["defect_type"])
        type_index = (
            DEFECT_TYPE_CHOICES.index(current_type) if current_type in DEFECT_TYPE_CHOICES else None
        )
        new_type = st.selectbox(
            "결함 유형",
            DEFECT_TYPE_CHOICES,
            index=type_index,
            format_func=_defect_label,
            placeholder="결함 유형을 선택하세요",
            disabled=new_label != config.LABEL_DEFECT,
            key=f"p2_type::{row['image_id']}",
        )

        roi = _roi_editor(row, rgb, enabled=new_label == config.LABEL_DEFECT)
        note = st.text_input("메모 (선택)", key=f"p2_note::{row['image_id']}")

        needs_type = new_label == config.LABEL_DEFECT and not new_type
        if needs_type:
            st.caption("⚠️ 결함으로 판정했으면 유형을 선택해야 저장할 수 있다.")

        save, skip = st.columns(2)
        if save.button(
            "💾 저장하고 다음", type="primary", width="stretch", disabled=needs_type
        ):
            is_defect = new_label == config.LABEL_DEFECT
            labeling.record_label(
                str(row["image_id"]),
                label=new_label,
                defect_type=new_type if is_defect else config.DEFECT_TYPE_NONE,
                roi=roi if is_defect else None,
                verified=True,
                note=note,
            )
            if mode not in _DRAINING_MODES:
                st.session_state[cursor_key] = min(cursor + 1, len(queue) - 1)
            st.rerun()
        if skip.button("⏭️ 건너뛰기", width="stretch"):
            st.session_state[cursor_key] = (cursor + 1) % len(queue)
            st.rerun()

    with image_col:
        if rgb is None:
            st.error(f"이미지를 읽을 수 없습니다: {path}")
        elif st.session_state.get(f"p2_roimode::{row['image_id']}") == MODE_DRAG and (
            new_label == config.LABEL_DEFECT
        ):
            _roi_canvas(row, rgb, roi)
        else:
            preview_roi = roi if new_label == config.LABEL_DEFECT else None
            st.image(
                viz.draw_roi(rgb, preview_roi, label=new_type or ""),
                caption=f"{row['category']} · {row['image_id']} · {rgb.shape[1]}×{rgb.shape[0]}",
                width="stretch",
            )
            if preview_roi:
                st.caption(f"표시된 ROI: {preview_roi}")

    prev, nxt = st.columns(2)
    if prev.button("◀ 이전", width="stretch", key="p2_prev"):
        st.session_state[cursor_key] = (cursor - 1) % len(queue)
        st.rerun()
    if nxt.button("다음 ▶", width="stretch", key="p2_next"):
        st.session_state[cursor_key] = (cursor + 1) % len(queue)
        st.rerun()


def _roi_of(row: pd.Series) -> tuple[int, int, int, int] | None:
    """행에서 ROI를 꺼낸다. 값이 없으면 None."""
    values = [row.get(col) for col in ("roi_x", "roi_y", "roi_w", "roi_h")]
    if any(pd.isna(v) or v is None for v in values):
        return None
    return tuple(int(v) for v in values)  # type: ignore[return-value]


MODE_MASK = "마스크에서 자동 추출"
MODE_DRAG = "이미지에서 직접 그리기"
MODE_NONE = "지정 안 함"


def _roi_key(image_id: str) -> str:
    """드래그 선택을 담는 위젯 key. 이미지가 바뀌면 선택도 새로 시작해야 한다."""
    return f"p2_roidrag::{image_id}"


def _roi_editor(row: pd.Series, rgb, *, enabled: bool) -> tuple[int, int, int, int] | None:
    """결함 위치(ROI)를 정한다. **화면은 그리지 않고 값만 정한다.**

    실제 지정은 이미지 위에서 드래그로 한다(`_roi_canvas`). 그런데 저장 버튼은 오른쪽
    칸에 있고 이미지는 왼쪽 칸이라, 코드 실행 순서상 저장 버튼이 먼저다. 그래서 여기서는
    이전 실행이 세션에 남긴 선택을 읽기만 하고, 그리는 일은 이미지 칸에서 한다.
    """
    if not enabled or rgb is None:
        return None

    st.markdown("**결함 위치 (ROI)**")
    auto_roi = labeling.roi_from_image_path(str(row["path"]))

    options = [MODE_DRAG, MODE_NONE]
    if auto_roi:
        options.insert(0, MODE_MASK)
    choice = st.radio(
        "ROI 지정 방식", options, index=0, horizontal=True,
        key=f"p2_roimode::{row['image_id']}", label_visibility="collapsed",
    )

    if choice == MODE_MASK:
        st.caption(f"ground truth 마스크에서 자동 추출: {auto_roi}")
        return auto_roi
    if choice == MODE_NONE:
        return None

    height, width = rgb.shape[:2]
    drawn = ui.roi_box(_roi_key(str(row["image_id"])), width, height)
    if drawn:
        x, y, w, h = drawn
        st.caption(f"지정됨 — x {x} · y {y} · 폭 {w} · 높이 {h}")
        return drawn

    # 아직 안 그렸으면 기존 값이나 마스크 값을 그대로 쓴다. 결함이라고 판정해 놓고
    # 영역을 못 그렸다는 이유로 위치 정보를 잃는 것보다 낫다.
    fallback = _roi_of(row) or auto_roi
    if fallback:
        st.caption(f"기존 값 유지: {fallback} — 새로 그리면 바뀝니다.")
    else:
        st.caption("왼쪽 이미지 위에서 드래그해 결함 위치를 감싸세요.")
    return fallback


def _roi_canvas(row: pd.Series, rgb, roi) -> None:
    """이미지 위에서 드래그로 영역을 지정하는 화면. 왼쪽 칸에 그린다."""
    height, width = rgb.shape[:2]
    ui.roi_picker(
        viz.draw_roi(rgb, roi),
        key=_roi_key(str(row["image_id"])),
        width=width,
        height=height,
    )
    st.caption(
        f"{row['category']} · {row['image_id']} · {width}×{height} — "
        "**드래그해 영역을 지정**하고, 지정한 영역 **안쪽을 끌면 위치를 옮길 수 있습니다.**"
    )


# --- 폴더 라벨 검증 --------------------------------------------------------

def _verify_tab(resolved: pd.DataFrame) -> None:
    st.markdown(
        "오픈 데이터셋은 폴더 구조로 라벨이 이미 주어진다. 그대로 신뢰하지 않고 표본을 눈으로 "
        "확인해 오라벨을 잡아낸다. 확인한 이미지는 `verified`로 기록된다."
    )
    if resolved.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    pending = resolved[
        (resolved["label_source"].astype(str) == "folder")
        & (~resolved["verified"].fillna(False).astype(bool))
    ]
    if pending.empty:
        st.success("폴더 라벨을 모두 확인했습니다.", icon="🎉")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("미확인 폴더 라벨", f"{len(pending):,}")
    sample_size = col2.slider("표본 개수", 4, 24, 8, 4, key="p2_verify_n")
    seed = col3.number_input("표본 시드", 0, 9999, 0, 1, key="p2_verify_seed")

    sample = pending.sample(min(sample_size, len(pending)), random_state=int(seed))

    with st.form("p2_verify_form"):
        st.caption("라벨이 틀린 이미지는 아래에서 값을 바꾼 뒤 한 번에 저장한다.")
        decisions: dict[str, str] = {}
        columns = st.columns(4)
        for position, (_, row) in enumerate(sample.iterrows()):
            target = columns[position % 4]
            with target:
                rgb = viz.load_rgb(storage.resolve_path(row["path"]))
                if rgb is not None:
                    st.image(rgb, width="stretch")
                st.caption(f"{row['category']} · `{row['raw_defect_type']}`")
                current = str(row["label"])
                decisions[str(row["image_id"])] = st.radio(
                    "판정",
                    [config.LABEL_NORMAL, config.LABEL_DEFECT],
                    index=0 if current == config.LABEL_NORMAL else 1,
                    format_func=lambda x: config.LABEL_KO.get(x, x),
                    key=f"p2_verify::{row['image_id']}",
                    label_visibility="collapsed",
                )

        if st.form_submit_button("✅ 표본 확인 처리", type="primary"):
            lookup = sample.set_index(sample["image_id"].astype(str))
            items = []
            changed = 0
            for image_id, label in decisions.items():
                original = lookup.loc[image_id]
                if label != str(original["label"]):
                    changed += 1
                items.append(
                    {
                        "image_id": image_id,
                        "label": label,
                        "defect_type": (
                            str(original["defect_type"])
                            if label == config.LABEL_DEFECT
                            else config.DEFECT_TYPE_NONE
                        ),
                        "verified": True,
                        "note": "folder_label_review",
                    }
                )
            labeling.record_labels(items)
            st.success(f"{len(items)}건 확인 처리 (라벨 수정 {changed}건)", icon="✅")
            st.rerun()


# --- 결함 유형 정규화 ------------------------------------------------------

def _mapping_tab(resolved: pd.DataFrame) -> None:
    st.markdown(
        "데이터셋마다 결함 유형명이 다르다. 여러 데이터셋을 함께 학습·평가하려면 "
        "프로젝트 표준 유형으로 모아야 한다."
    )
    unmapped = labeling.unmapped_defect_types(resolved)
    mapping = labeling.load_type_map()

    if not unmapped:
        st.success("정규화가 필요한 결함 유형이 없습니다.", icon="🎉")
    else:
        st.warning(f"표준 유형으로 매핑되지 않은 유형 {len(unmapped)}종", icon="🔀")
        counts = (
            resolved[resolved["label"] == config.LABEL_DEFECT]["defect_type"]
            .astype(str).value_counts()
        )
        with st.form("p2_mapping_form"):
            choices: dict[str, str] = {}
            for raw in unmapped:
                col1, col2 = st.columns([1, 2])
                col1.markdown(f"`{raw}`")
                col1.caption(f"{int(counts.get(raw, 0)):,}건")
                choices[raw] = col2.selectbox(
                    f"{raw} → 표준 유형",
                    DEFECT_TYPE_CHOICES,
                    index=DEFECT_TYPE_CHOICES.index("other"),
                    format_func=_defect_label,
                    key=f"p2_map::{raw}",
                    label_visibility="collapsed",
                )
            if st.form_submit_button("💾 매핑 저장", type="primary"):
                mapping.update(choices)
                labeling.save_type_map(mapping)
                st.success(f"{len(choices)}종 매핑을 저장했습니다.", icon="✅")
                st.rerun()

    st.divider()
    st.markdown("**현재 매핑 (기본 매핑 포함)**")
    table = pd.DataFrame(
        [{"원본 유형": k, "표준 유형": v, "표준 유형(한글)": config.defect_type_label(v)}
         for k, v in sorted(mapping.items())]
    )
    st.dataframe(table, hide_index=True, width="stretch", height=280)


# --- 데이터 분할 -----------------------------------------------------------

def _split_mode(resolved: pd.DataFrame) -> str:
    """무엇을 하나로 묶어 옮길지 고르게 한다.

    **고르게 하되, 모르고 고르지는 않게 한다.** 영상 프레임이 있는데 무작위를 고르면
    학습에 쓴 것과 거의 같은 장면이 평가에 들어가 성능이 실제보다 높게 나온다. 이건
    화면이 알려주지 않으면 알아채기 어렵다.
    """
    groups = labeling.image_groups(resolved)
    grouped = int((groups != resolved["image_id"].astype(str)).sum())

    mode = st.radio(
        "무엇을 기준으로 나눌까",
        list(labeling.SPLIT_MODES),
        format_func=lambda key: labeling.SPLIT_MODE_LABELS[key],
        horizontal=True, key="p2_split_mode",
    )

    if grouped:
        st.caption(
            f"영상에서 뽑은 프레임 {grouped:,}장이 있습니다 "
            f"(영상 {groups[groups != resolved['image_id'].astype(str)].nunique():,}개)."
        )
    else:
        st.caption("영상 프레임이 없어 이미지 하나가 곧 그룹입니다 — 세 방식의 결과가 거의 같습니다.")

    if mode == labeling.SPLIT_BY_GROUP:
        st.caption("같은 영상의 프레임은 통째로 같은 분할로 갑니다.")
    elif mode == labeling.SPLIT_BY_TIME:
        st.caption(
            "수집 시각 순으로 앞은 학습, 뒤는 평가로 나눕니다. 영상이 하나뿐이라 그룹으로 "
            "나눌 수 없을 때 씁니다 — 경계 부근 프레임은 여전히 비슷해 누수가 조금 남습니다."
        )
    elif grouped:
        st.warning(
            "**영상 프레임이 있는데 무작위로 나눕니다.** 같은 영상의 프레임이 학습과 평가에 "
            "섞여 들어가 **성능이 실제보다 높게 나옵니다.** 벤치마크와 맞추려는 경우가 "
            "아니라면 '영상(그룹) 단위'를 쓰세요.",
            icon="⚠️",
        )
    return mode


def _split_tab(resolved: pd.DataFrame) -> None:
    st.markdown(
        "카테고리 × 라벨로 **층화 분할**한다. 층화하지 않으면 특정 카테고리나 결함 클래스가 "
        "한쪽 분할에만 몰려 평가 결과를 신뢰할 수 없다."
    )
    if resolved.empty:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    col1, col2, col3, col4 = st.columns(4)
    train = col1.slider("train 비율", 0.1, 0.9, 0.6, 0.05, key="p2_train")
    val = col2.slider("val 비율", 0.0, 0.5, 0.2, 0.05, key="p2_val")
    test = round(max(0.0, 1.0 - train - val), 2)
    col3.metric("test 비율", f"{test:.2f}")
    seed = col4.number_input("시드", 0, 9999, 42, 1, key="p2_split_seed")

    labeled_only = st.checkbox(
        "라벨된 이미지만 분할", value=True, key="p2_split_labeled",
        help="미라벨 이미지는 학습에 쓸 수 없으므로 기본적으로 제외한다.",
    )

    mode = _split_mode(resolved)

    if test <= 0:
        st.error("test 비율이 0 이하다. train/val 비율을 줄여야 한다.")
    elif st.button("✂️ 층화 분할 실행", type="primary", key="p2_do_split"):
        mapping = labeling.assign_splits(
            resolved, train=train, val=val, test=test, seed=int(seed),
            labeled_only=labeled_only, mode=mode,
        )
        if not mapping:
            st.warning("분할할 대상이 없습니다.")
        else:
            labeling.save_splits(mapping)
            st.success(f"{len(mapping):,}건에 분할을 배정했습니다.", icon="✅")
            st.rerun()

    st.divider()
    st.markdown("**공식 분할 정의 임포트**")
    st.caption(
        "VisA는 `split_csv/` 아래에 공식 train/test 분할을 제공한다. 벤치마크 수치와 비교하려면 "
        "직접 분할하는 대신 공식 정의를 쓰는 편이 낫다. 이미지 경로 컬럼과 split 컬럼을 자동으로 "
        "찾아 **경로 접미사**로 매칭한다(VisA는 Normal/Anomaly에 같은 파일명을 쓰므로 파일명만으로는 구분되지 않는다)."
    )
    uploaded = st.file_uploader("분할 정의 CSV", type=["csv"], key="p2_split_csv")
    if uploaded is not None and st.button("📥 CSV 분할 적용", key="p2_apply_csv"):
        try:
            mapping, unmatched = labeling.import_split_csv(uploaded, storage.load_manifest())
        except (ValueError, KeyError) as exc:
            st.error(f"임포트 실패: {exc}")
        else:
            if mapping:
                labeling.save_splits(mapping)
                st.success(
                    f"{len(mapping):,}건 적용, 매칭 실패 {unmatched:,}건", icon="✅"
                )
                st.rerun()
            else:
                st.warning(f"매칭된 이미지가 없습니다 (실패 {unmatched:,}건). 컬럼과 파일명을 확인하세요.")

    st.divider()
    current = resolved["split"].astype(str)
    if current.isin([config.SPLIT_TRAIN, config.SPLIT_VAL, config.SPLIT_TEST]).any():
        st.markdown("**현재 분할 결과**")
        crosstab = pd.crosstab(
            [resolved["category"].astype(str), resolved["label"].astype(str)], current
        )
        st.dataframe(crosstab, width="stretch")
        if st.button("🗑️ 분할 초기화", key="p2_clear_split"):
            labeling.clear_splits()
            st.rerun()
    else:
        st.info("아직 분할이 배정되지 않았습니다.", icon="✂️")


# --- 라벨 현황 -------------------------------------------------------------

def _status_tab(resolved: pd.DataFrame) -> None:
    stats = labeling.stats(resolved)
    if stats["total"] == 0:
        st.info("수집된 이미지가 없습니다.", icon="📥")
        st.page_link(guide.PAGE_INGEST, label="1단계 데이터 수집으로 이동", icon="➡️")
        return

    cols = st.columns(5)
    cols[0].metric("전체", f"{stats['total']:,}")
    cols[1].metric("정상", f"{stats['normal']:,}")
    cols[2].metric("결함", f"{stats['defect']:,}")
    cols[3].metric("사람이 라벨", f"{stats['human']:,}")
    cols[4].metric("확인 완료", f"{stats['verified']:,}")

    cols = st.columns(4)
    cols[0].metric("유형 미지정 결함", f"{stats['unspecified_type']:,}")
    cols[1].metric("미매핑 유형 수", f"{stats['unmapped_types']:,}")
    cols[2].metric("ROI 지정", f"{stats['with_roi']:,}")
    cols[3].metric("분할 배정", f"{stats['split_assigned']:,}")

    if stats["unspecified_type"]:
        st.info(
            f"결함 {stats['unspecified_type']:,}건의 유형이 아직 미지정이다. "
            "**라벨 검수** 탭에서 지정한다.",
            icon="🏷️",
        )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**결함 유형 분포 (정규화 후)**")
        defects = resolved[resolved["label"] == config.LABEL_DEFECT]
        if defects.empty:
            st.caption("결함 이미지가 없습니다.")
        else:
            st.bar_chart(defects["defect_type"].astype(str).value_counts(), horizontal=True)
    with right:
        st.markdown("**라벨 근거**")
        st.bar_chart(resolved["label_source"].astype(str).value_counts(), horizontal=True)

    st.divider()
    st.markdown("### 라벨 이력")
    events = labeling.load_events()
    if events.empty:
        st.caption("아직 기록된 라벨 이벤트가 없습니다.")
    else:
        st.caption(f"총 {len(events):,}건 (같은 이미지의 최신 이벤트가 유효 라벨이 된다)")
        st.dataframe(events.tail(200).iloc[::-1], hide_index=True, width="stretch", height=280)

    with st.expander("내려받기"):
        st.download_button(
            "유효 라벨 (resolved.csv)",
            resolved.to_csv(index=False).encode("utf-8-sig"),
            file_name="resolved_labels.csv",
            mime="text/csv",
        )
        if not events.empty:
            st.download_button(
                "라벨 이력 (labels.csv)",
                events.to_csv(index=False).encode("utf-8-sig"),
                file_name="labels.csv",
                mime="text/csv",
            )


def render() -> None:
    st.title("🏷️ 2단계 · 라벨링")
    st.caption(
        "manifest는 그대로 두고 사람의 판정을 별도 이력으로 쌓는다. "
        "유효 라벨 = 폴더 추론 라벨 + 사람 판정 덮어쓰기 + 유형 정규화."
    )

    resolved = labeling.resolve()

    # 2단계에서 반드시 해야 하는 것은 **데이터 분할**이다. 나머지는 데이터에 따라 건너뛸 수
    # 있는데, 분할을 빼먹으면 3단계가 통째로 막힌다. 그래서 그 탭만 상태를 표시한다.
    split_done = (
        not resolved.empty
        and (resolved["split"].astype(str) != config.SPLIT_NONE).any()
    )
    split_mark = "✅" if split_done else "👉"
    tabs = st.tabs(
        ["🔍 라벨 검수", "✅ 폴더 라벨 검증", "🔀 결함 유형 정규화",
         f"{split_mark} ✂️ 데이터 분할", "📊 라벨 현황"]
    )
    if not split_done:
        st.caption(
            "👉 **데이터 분할**은 건너뛸 수 없습니다. 학습용과 평가용을 나눠 두지 않으면 "
            "3단계에서 학습을 시작할 수 없습니다."
        )
    with tabs[0]:
        _review_tab(resolved)
    with tabs[1]:
        _verify_tab(resolved)
    with tabs[2]:
        _mapping_tab(resolved)
    with tabs[3]:
        _split_tab(resolved)
    with tabs[4]:
        _status_tab(resolved)


render()
