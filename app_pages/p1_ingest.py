"""1단계: 데이터 수집(입력) 화면."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from vision_ai import config, datasets, glossary, ingest, quality, storage, ui

FIT_STARS = {5: "★★★★★", 4: "★★★★☆", 3: "★★★☆☆", 2: "★★☆☆☆", 1: "★☆☆☆☆"}


def _catalog_tab() -> None:
    default = datasets.default_dataset()
    st.markdown(
        "사내 데이터를 쓰지 않으므로 공개 데이터셋이 학습 데이터의 출발점이다. "
        "아래는 표면 결함 분야에서 널리 쓰이는 데이터셋을 정리한 것이다."
    )
    st.success(
        f"**기본 예시 데이터셋: {default.name}** — 라이선스가 `{default.license}`로 "
        "상업적 이용이 가능해, 이후 회사 업무로 연장할 때 데이터셋을 갈아치우지 않아도 된다.",
        icon="⭐",
    )
    st.warning(
        "라이선스·URL은 정리 시점 기준 정보다. 특히 **비상업(NC)** 조건이 붙은 데이터셋이 많으므로, "
        "사용 전 원본 배포 페이지에서 조건을 직접 확인할 것.",
        icon="⚠️",
    )

    catalog = datasets.recommended()
    table = pd.DataFrame(
        [
            {
                "데이터셋": ("⭐ " if d.is_default else "") + d.name,
                "일상 적합도": FIT_STARS.get(d.everyday_fit, ""),
                "상업적 이용": d.commercial_use,
                "라이선스": d.license,
                "카테고리 수": len(d.categories),
                "폴더 구조": d.layout,
                "key": d.key,
            }
            for d in catalog
        ]
    )
    # 열이 7개라 좁은 화면에서는 폭에 맞추려다 글자가 뭉개진다. 폭을 지정해 찌그러뜨리는
    # 대신 가로 스크롤이 생기게 하고, 이름 열은 고정해 스크롤해도 어느 줄인지 잃지 않게 한다.
    # ⭐ 표시는 별도 열이었으나 이름 앞에 붙여 열을 하나 줄였다.
    ui.responsive_table(
        table,
        key="catalog",
        title_column="데이터셋",
        hide_index=True,
        width="stretch",
        column_config={
            "데이터셋": st.column_config.TextColumn(width="medium", pinned=True),
            "일상 적합도": st.column_config.TextColumn(
                width="small",
                help="일상 사물 촬영본으로 이 앱을 시험해 볼 때의 적합도 (별 5개가 가장 적합)",
            ),
            "상업적 이용": st.column_config.TextColumn(
                width="medium", help="회사 업무로 연장할 수 있는지. 원본 배포 페이지에서 재확인할 것"
            ),
            "라이선스": st.column_config.TextColumn(width="medium"),
            "카테고리 수": st.column_config.NumberColumn(width="small"),
            "폴더 구조": st.column_config.TextColumn(
                width="small", help="'로컬 폴더 임포트' 탭이 이 구조로 폴더를 읽는다"
            ),
            "key": st.column_config.TextColumn(
                width="small", help="스크립트에서 데이터셋을 가리킬 때 쓰는 식별자"
            ),
        },
    )

    st.markdown("#### 상세 정보")
    selected_name = st.selectbox(
        "데이터셋 선택", [d.name for d in catalog], key="catalog_select"
    )
    dataset = next(d for d in catalog if d.name == selected_name)

    st.markdown(f"**{dataset.name}** — {dataset.summary}")
    left, right = st.columns(2)
    with left:
        st.markdown(
            f"- 배포 페이지: {dataset.url}\n"
            f"- 라이선스: `{dataset.license}` (상업적 이용: {dataset.commercial_use})\n"
            f"- 일상 물건 적합도: {FIT_STARS.get(dataset.everyday_fit, '')}"
        )
        if dataset.tags:
            st.caption("태그: " + ", ".join(dataset.tags))
    with right:
        st.markdown(
            f"- 폴더 구조(`{dataset.layout}`): `{dataset.layout_note}`\n"
            f"- 다운로드: {dataset.download_note}"
        )
    st.info(f"일상 물건 관점: {dataset.everyday_note}", icon="🏠")
    st.caption(f"⚠️ {dataset.license_note}")

    with st.expander("카테고리 목록"):
        st.write(", ".join(dataset.categories) if dataset.categories else "정보 없음")

    st.success(
        "다운로드·압축 해제까지 마쳤다면 **로컬 폴더 임포트** 탭에서 폴더 경로를 지정해 등록한다.",
        icon="➡️",
    )


def _folder_tab() -> None:
    st.markdown(
        "내려받아 압축을 푼 데이터셋 폴더를 지정하면, 폴더 구조에서 "
        "**카테고리 · 분할 · 정상/결함 · 결함 유형**을 자동으로 추론해 등록한다. "
        "원본 파일은 이동·복사하지 않고 경로만 인덱싱한다."
    )

    catalog = datasets.recommended()
    options = [f"{d.name} ({d.key})" for d in catalog] + ["(직접 입력)"]
    choice = st.selectbox(
        "데이터셋", options, key="folder_dataset",
        help="기본 예시 데이터셋인 VisA가 맨 앞에 온다.",
    )

    default_layout, source_default = datasets.default_dataset().layout, ""
    if choice != "(직접 입력)":
        key = choice.rsplit("(", 1)[-1].rstrip(")")
        dataset = datasets.get(key)
        if dataset:
            default_layout, source_default = dataset.layout, dataset.key
            st.caption(f"예상 폴더 구조: `{dataset.layout_note}`")

    col1, col2 = st.columns([3, 1])
    root_input = col1.text_input(
        "데이터셋 루트 폴더 경로",
        key="folder_path",
        placeholder="/home/user/datasets/VisA",
    )
    layout_options = list(datasets.LAYOUT_OPTIONS)
    layout = col2.selectbox(
        "구조 해석 방식",
        layout_options,
        index=layout_options.index(default_layout) if default_layout in layout_options else 0,
        key="folder_layout",
        help=" · ".join(f"{k}: {v}" for k, v in datasets.LAYOUT_HELP.items()),
    )

    col3, col4, col5 = st.columns(3)
    source = col3.text_input("출처 이름 (source)", value=source_default or "local", key="folder_source")
    limit_on = col4.checkbox("건수 제한 (미리보기)", value=False, key="folder_limit_on")
    limit = col4.number_input(
        "최대 건수", min_value=1, max_value=100_000, value=200, step=50,
        key="folder_limit", disabled=not limit_on,
    )
    include_masks = col5.checkbox(
        "ground_truth 마스크도 등록", value=False, key="folder_masks",
        help="기본적으로 결함 마스크 이미지는 학습 입력이 아니라서 제외한다.",
    )

    if not root_input:
        return

    root = Path(root_input).expanduser()
    if not root.is_dir():
        st.error(f"폴더를 찾을 수 없습니다: `{root}`")
        return

    total = ingest.count_image_files(root)
    st.info(f"이미지 파일 {total:,}건을 발견했습니다.", icon="🔎")
    if total == 0:
        return

    # 실제 등록 전에 라벨 추론 결과를 미리 보여준다
    with st.expander("구조 해석 미리보기 (상위 10건)", expanded=True):
        rows = []
        for path in list(ingest.iter_image_files(root))[:10]:
            parsed = datasets.parse_path(path.relative_to(root), layout)
            rows.append(
                {
                    "상대경로": str(path.relative_to(root)),
                    "카테고리": parsed["category"],
                    "분할": parsed["split"],
                    "라벨": config.LABEL_KO.get(parsed["label"], parsed["label"]),
                    "결함유형": parsed["defect_type"],
                    "마스크": "예" if parsed["is_mask"] else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("추론 결과가 의도와 다르면 '구조 해석 방식'을 바꿔 다시 확인한다.")

    if st.button("📥 manifest에 등록", type="primary", key="folder_ingest"):
        bar = st.progress(0.0, text="등록 중...")

        def on_progress(done: int, count: int) -> None:
            bar.progress(done / max(count, 1), text=f"등록 중... {done:,}/{count:,}")

        try:
            result = ingest.ingest_folder(
                root,
                source=source or "local",
                layout=layout,
                include_masks=include_masks,
                limit=int(limit) if limit_on else None,
                progress=on_progress,
            )
        except (OSError, ValueError) as exc:
            bar.empty()
            st.error(f"등록 실패: {exc}")
            return
        bar.empty()
        st.success(result.as_message(), icon="✅")
        if result.failed:
            with st.expander(f"읽기 실패 {len(result.failed)}건"):
                st.write(result.failed[:50])


def _upload_tab() -> None:
    st.markdown(
        "일상에서 직접 촬영한 이미지를 올린다. 오픈 데이터셋으로 학습한 모델이 "
        "실제 생활 이미지에서 어떻게 동작하는지 확인하는 검증셋으로 쓰기 좋다."
    )

    uploaded = st.file_uploader(
        "이미지 선택 (여러 장 가능)",
        type=[ext.lstrip(".") for ext in sorted(config.IMAGE_EXTENSIONS)],
        accept_multiple_files=True,
        key="upload_files",
    )

    col1, col2, col3 = st.columns(3)
    category = col1.text_input(
        "카테고리", value="", key="upload_category",
        placeholder="mug, phone_case, wood_table ...",
        help="같은 종류의 물건끼리 묶는 이름. 비우면 uncategorized로 등록된다.",
    )
    label_choice = col2.selectbox(
        "라벨",
        [config.LABEL_UNLABELED, config.LABEL_NORMAL, config.LABEL_DEFECT],
        format_func=lambda x: config.LABEL_KO.get(x, x),
        key="upload_label",
        help="지금 모르면 미라벨로 두고 2단계 라벨링에서 지정한다.",
    )
    defect_options = [config.DEFECT_TYPE_NONE, *config.DEFECT_TYPES]
    defect_type = col3.selectbox(
        "결함 유형",
        defect_options,
        format_func=config.defect_type_label,
        key="upload_defect",
        disabled=label_choice != config.LABEL_DEFECT,
    )

    if not uploaded:
        return

    st.caption(f"선택된 파일 {len(uploaded)}건")
    preview = st.columns(min(len(uploaded), 5))
    for col, file in zip(preview, uploaded[:5]):
        col.image(file.getvalue(), caption=file.name, width="stretch")

    if st.button("📥 업로드 이미지 등록", type="primary", key="upload_ingest"):
        payload = [(f.name, f.getvalue()) for f in uploaded]
        result = ingest.ingest_uploads(
            payload,
            category=category,
            label=label_choice,
            defect_type=defect_type if label_choice == config.LABEL_DEFECT else config.DEFECT_TYPE_NONE,
        )
        st.success(result.as_message(), icon="✅")
        if result.failed:
            st.warning(f"이미지로 읽을 수 없는 파일: {', '.join(result.failed[:10])}")


def _synthetic_tab() -> None:
    st.markdown(
        "오픈 데이터셋 다운로드는 용량이 크고 약관 동의가 필요한 경우가 많다. "
        "**수집 → 라벨링 → 학습 → 운영** 전 과정을 먼저 돌려보기 위해, "
        "MVTec AD와 동일한 폴더 구조로 가짜 표면 이미지를 생성한다."
    )
    st.caption(
        "합성 데이터는 파이프라인 배선 검증용이다. 모델 성능의 근거로 삼을 수는 없다."
    )

    col1, col2, col3 = st.columns(3)
    categories = col1.multiselect(
        "표면 종류",
        list(ingest.SURFACE_STYLES),
        default=list(ingest.SURFACE_STYLES)[:2],
        key="syn_categories",
    )
    n_normal = col2.number_input("카테고리별 정상 이미지", 5, 500, 40, 5, key="syn_normal")
    n_defect = col3.number_input("카테고리별 결함 이미지", 4, 400, 20, 4, key="syn_defect")

    col4, col5, col6, col7 = st.columns(4)
    size = col4.select_slider("이미지 크기(px)", [128, 192, 256, 320, 512], value=256, key="syn_size")
    seed = col5.number_input("랜덤 시드", 0, 10_000, 42, 1, key="syn_seed")
    syn_layout = col6.selectbox(
        "폴더 구조", ["visa", "mvtec"], key="syn_layout",
        help=(
            "visa: 기본 예시 데이터셋과 같은 구조 (결함 유형 폴더 없음, 마스크 제공) · "
            "mvtec: 결함 유형별 폴더 + ground_truth 마스크"
        ),
    )
    overwrite = col7.checkbox("기존 합성 데이터 삭제 후 재생성", value=True, key="syn_overwrite")

    st.caption(
        f"생성될 결함 유형: {', '.join(ingest.SYNTHETIC_DEFECTS)} · "
        "결함 픽셀 마스크를 함께 만들어 2단계에서 ROI 자동 추출을 시험할 수 있다."
    )
    if syn_layout == "visa":
        st.caption(
            "VisA 구조는 결함 유형을 폴더로 나누지 않으므로, 등록 시 모든 결함이 "
            "`유형 미지정`으로 들어온다 — 실제 VisA와 같은 상황이며 2단계에서 유형을 지정한다."
        )

    if not categories:
        st.info("표면 종류를 1개 이상 선택하세요.")
        return

    if st.button("🧪 합성 데이터 생성 후 등록", type="primary", key="syn_generate"):
        with st.spinner("이미지 생성 중..."):
            out_dir = ingest.generate_synthetic(
                categories=categories,
                n_normal=int(n_normal),
                n_defect=int(n_defect),
                size=int(size),
                seed=int(seed),
                overwrite=overwrite,
                layout=syn_layout,
            )
            if overwrite:
                ingest.remove_source(ingest.SYNTHETIC_SOURCE)
            result = ingest.ingest_folder(
                out_dir, source=ingest.SYNTHETIC_SOURCE, layout=syn_layout
            )
        st.success(f"{out_dir} 생성 완료 — {result.as_message()}", icon="✅")


def _status_tab() -> None:
    df = storage.load_manifest()
    if df.empty:
        st.info("등록된 이미지가 없습니다. 다른 탭에서 데이터를 먼저 수집하세요.", icon="📭")
        return

    stats = storage.summarize(df)
    cols = st.columns(5)
    cols[0].metric("전체", f"{stats['total']:,}")
    cols[1].metric("정상", f"{stats['normal']:,}")
    cols[2].metric("결함", f"{stats['defect']:,}")
    cols[3].metric("미라벨", f"{stats['unlabeled']:,}")
    cols[4].metric("출처 수", f"{stats['sources']:,}")

    # 클래스 불균형은 이후 학습 전략을 좌우하므로 눈에 띄게 알린다
    if stats["normal"] and stats["defect"]:
        ratio = stats["normal"] / stats["defect"]
        if ratio >= 5 or ratio <= 0.2:
            st.warning(
                f"정상:결함 비율이 {ratio:.1f}:1로 치우쳐 있다. "
                "이상탐지 접근이나 클래스 가중치·증강을 고려할 것.",
                icon="⚖️",
            )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("**카테고리별 이미지 수**")
        st.bar_chart(df["category"].value_counts(), horizontal=True)
    with right:
        st.markdown("**결함 유형 분포**")
        st.bar_chart(df["defect_type"].value_counts(), horizontal=True)

    st.markdown("**출처 × 라벨 교차표**")
    crosstab = pd.crosstab(df["source"], df["label"])
    crosstab.columns = [config.LABEL_KO.get(c, c) for c in crosstab.columns]
    st.dataframe(crosstab, width="stretch")

    st.divider()
    st.markdown("### 품질 점검")
    flagged = df[df["note"].astype(str).str.contains("blurry|exposed|low_resolution", na=False)]
    q1, q2, q3 = st.columns(3)
    q1.metric("품질 경고", f"{len(flagged):,}")
    q2.metric("평균 선명도(Laplacian)", f"{df['blur_score'].mean():.1f}",
          help=glossary.detail("blur_score"))
    q3.metric("평균 밝기", f"{df['brightness'].mean():.1f}",
          help=glossary.detail("brightness"))
    st.caption(
        f"{glossary.caption('blur_score')} {glossary.caption('brightness')} "
        f"{glossary.ARBITRARY['quality']}"
    )
    if not flagged.empty:
        with st.expander(f"경고 이미지 {len(flagged):,}건 보기"):
            view = flagged[["image_id", "category", "label", "width", "height", "blur_score", "brightness", "note"]].copy()
            view["점검 결과"] = view["note"].map(quality.describe_flags)
            st.dataframe(view.drop(columns=["note"]), hide_index=True, width="stretch")

    st.divider()
    st.markdown("### 이미지 미리보기")
    fcol1, fcol2, fcol3 = st.columns(3)
    category_filter = fcol1.selectbox(
        "카테고리", ["(전체)"] + sorted(df["category"].dropna().unique().tolist()), key="prev_cat"
    )
    label_filter = fcol2.selectbox(
        "라벨", ["(전체)"] + sorted(df["label"].dropna().unique().tolist()),
        format_func=lambda x: config.LABEL_KO.get(x, x), key="prev_label",
    )
    count = fcol3.slider("표시 개수", 4, 24, 8, 4, key="prev_count")

    subset = df
    if category_filter != "(전체)":
        subset = subset[subset["category"] == category_filter]
    if label_filter != "(전체)":
        subset = subset[subset["label"] == label_filter]

    if subset.empty:
        st.info("조건에 맞는 이미지가 없습니다.")
    else:
        sample = subset.sample(min(count, len(subset)), random_state=0)
        columns = st.columns(4)
        for position, (_, row) in enumerate(sample.iterrows()):
            path = storage.resolve_path(row["path"])
            target = columns[position % 4]
            if path.exists():
                caption = f"{row['category']} · {config.LABEL_KO.get(row['label'], row['label'])}"
                if row["defect_type"] not in (config.DEFECT_TYPE_NONE, None):
                    caption += f" ({row['defect_type']})"
                target.image(str(path), caption=caption, width="stretch")
            else:
                target.error(f"파일 없음: {row['image_id']}")

    st.divider()
    with st.expander("manifest 원본 보기 / 내려받기"):
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button(
            "manifest.csv 내려받기",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name="manifest.csv",
            mime="text/csv",
        )

    with st.expander("⚠️ 출처별 레코드 삭제"):
        st.caption("manifest에서 레코드만 제거하며, 원본 이미지 파일은 지우지 않는다.")
        source_to_drop = st.selectbox(
            "삭제할 출처", sorted(df["source"].dropna().unique().tolist()), key="drop_source"
        )
        if st.button("레코드 삭제", key="drop_button"):
            removed = ingest.remove_source(source_to_drop)
            st.success(f"{source_to_drop} 레코드 {removed:,}건을 제거했습니다.")
            st.rerun()


def render() -> None:
    st.title("📥 1단계 · 데이터 수집 (입력)")
    st.caption(
        "오픈 데이터셋과 직접 촬영 이미지를 하나의 manifest로 모은다. "
        "manifest는 이후 모든 단계의 입력이 된다."
    )

    # 탭이 5개인데 어디부터 눌러야 할지 표시가 없으면 초보자는 첫 탭부터 훑는다.
    # 데이터가 없을 때는 가장 빠른 길(합성 샘플)을 가리키고, 채워지면 완료로 바꾼다.
    has_data = not storage.load_manifest().empty
    mark = "✅" if has_data else "👉"
    tabs = st.tabs(
        ["🗂️ 오픈 데이터셋 카탈로그", "📁 로컬 폴더 임포트", "⬆️ 이미지 업로드",
         f"{mark} 🧪 합성 샘플 생성", f"{'✅ ' if has_data else ''}📊 수집 현황"]
    )
    if not has_data:
        st.caption("👉 표시된 탭이 가장 빠른 시작점입니다. 다운로드 없이 전 과정을 시험할 수 있습니다.")
    with tabs[0]:
        _catalog_tab()
    with tabs[1]:
        _folder_tab()
    with tabs[2]:
        _upload_tab()
    with tabs[3]:
        _synthetic_tab()
    with tabs[4]:
        _status_tab()


render()
