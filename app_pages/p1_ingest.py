"""1단계: 데이터 수집(입력) 화면."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from vision_ai import config, datasets, framing, glossary, ingest, quality, storage, ui, video

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


def _video_tab() -> None:
    """영상에서 학습용 프레임을 뽑는다 (설계: docs/video-frame-extraction-plan.md)."""
    st.markdown(
        "현장에서 데이터를 모으는 방식은 대체로 **영상 촬영**이다. 영상을 지정하면 "
        "일정 간격으로 프레임을 뽑아 manifest에 등록한다. 라벨은 2단계에서 붙인다."
    )
    st.info(
        "같은 영상에서 뽑은 프레임은 서로 매우 비슷합니다. 그래서 **영상 id를 그룹으로 달아** "
        "2단계 분할이 통째로 같은 쪽(학습 또는 평가)에 넣을 수 있게 합니다. "
        "섞이면 성능이 실제보다 높게 나옵니다.",
        icon="🔗",
    )

    with st.expander("영상이 없다면 — 시험용 영상 만들기"):
        st.caption(
            "물체가 3초마다 하나씩 지나가고 일부에 흠집이 있는 10초짜리 영상을 만듭니다. "
            "다운로드 없이 조작을 익혀 볼 수 있습니다. **성능 근거로는 쓸 수 없습니다.**"
        )
        if st.button("🎬 시험용 영상 만들기", key="video_sample"):
            try:
                made = video.make_sample()
            except OSError as exc:
                st.error(str(exc))
            else:
                # 만든 영상이 곧바로 목록에서 골라진 상태가 되게 한다. 만들어 놓고 다시
                # 찾아 고르게 하면 한 단계를 더 시키는 것이다.
                st.session_state["video_source"] = SOURCE_LIST
                st.session_state["video_pick"] = made
                st.success(f"만들었습니다: `{made}`", icon="✅")
                st.rerun()

    path = _video_source()
    if path is None:
        return

    try:
        info = video.probe(path)
    except OSError as exc:
        st.error(str(exc))
        return
    if not info.usable:
        st.error("fps나 프레임 수를 읽을 수 없는 영상입니다.")
        return

    cols = st.columns(4)
    cols[0].metric("길이", f"{info.duration_sec:.1f}초")
    cols[1].metric("fps", f"{info.fps:.0f}")
    cols[2].metric("전체 프레임", f"{info.frame_count:,}")
    cols[3].metric("해상도", f"{info.width}×{info.height}")

    _framing_check(info)

    plan = _extraction_plan(info)
    if plan is None:
        return

    st.caption(plan.reason)
    if not plan.feasible:
        st.warning(
            "**이 촬영으로는 목표 장수를 채울 수 없습니다.** 추출 설정이 아니라 촬영 계획의 "
            "문제이므로, 프레임을 뽑기 전에 라인 속도나 카메라를 조정하는 편이 낫습니다.",
            icon="🎥",
        )

    left, right = st.columns(2)
    left.metric("예상 추출 장수", f"{plan.expected_frames:,}장")
    right.metric("초당 추출", f"{plan.per_second:.1f}장")

    col1, col2, col3 = st.columns(3)
    category = col1.text_input("카테고리", value="video", key="video_category")
    dedupe = col2.checkbox(
        "거의 같은 프레임 버리기", value=True, key="video_dedupe",
        help="정지 구간에서는 같은 그림이 쏟아집니다. 균등 추출만으로는 거를 수 없습니다.",
    )
    check_quality = col3.checkbox(
        "흐린 프레임 버리기", value=False, key="video_quality",
        help=(
            "영상 프레임은 정지 이미지보다 전반적으로 흐립니다. 정지 이미지용 기준을 그대로 "
            "걸면 과하게 버릴 수 있어 기본은 꺼 둡니다."
        ),
    )

    if not st.button("🎞️ 프레임 추출 후 등록", type="primary", key="video_run"):
        return

    progress = ui.Progress(
        "프레임 추출 중",
        note=(
            f"{info.frame_count:,}프레임을 훑으며 {plan.stride}프레임마다 한 장씩 저장합니다. "
            "건너뛰는 프레임은 디코딩하지 않습니다."
        ),
    )
    try:
        result, extracted = ingest.ingest_video(
            path, stride=plan.stride, category=category or "video",
            min_change=video.DEFAULT_MIN_CHANGE if dedupe else None,
            check_quality=check_quality, progress=progress.update,
        )
    except OSError as exc:
        progress.done()
        st.error(str(exc))
        return
    progress.done(extracted.as_message())

    st.success(f"{result.as_message()} · 그룹 `{extracted.video_id}`", icon="✅")
    if result.duplicates:
        st.caption(f"이미 등록된 프레임 {result.duplicates:,}건은 건너뛰었습니다.")
    _warn_if_mostly_dropped(extracted, plan)
    if extracted.saved:
        st.markdown("**추출 표본**")
        for column, sample in zip(st.columns(4), extracted.saved[:4]):
            column.image(str(sample), width="stretch")
    st.info("다음 — **2단계 라벨링**에서 결함 구간과 위치를 지정합니다.", icon="➡️")


# 이 비율을 넘게 버렸으면 사용자에게 알린다. 잘 걸러진 것일 수도, 전멸한 것일 수도 있다.
# (주석으로 둔다 — 화면 파일에서 모듈 수준 문자열은 Streamlit이 그대로 출력해 버린다.)
MOSTLY_DROPPED = 0.8


def _warn_if_mostly_dropped(extracted, plan) -> None:
    """대부분이 버려졌으면 그 사실을 말한다.

    "N장 추출"만 찍고 넘어가면 40장을 기대했는데 3장이 나온 것을 **2단계에 가서야** 안다.
    그때는 이미 추출 설정을 잊은 뒤다. 다만 이것이 꼭 오류는 아니다 — CCTV처럼 빈 장면이
    대부분이면 많이 버려지는 게 정상이다. 그래서 막지 않고 판단 근거만 준다.
    """
    if extracted.scanned < 5 or extracted.drop_rate < MOSTLY_DROPPED:
        return
    st.warning(
        f"후보 {extracted.scanned:,}장 중 **{extracted.drop_rate:.0%}를 버려** "
        f"{extracted.kept:,}장만 남았습니다. 빈 장면이 대부분인 영상이면 정상입니다. "
        "그게 아니라면 **거의 같은 프레임 버리기**를 끄고 다시 뽑아 보세요.",
        icon="🧹",
    )
    if extracted.forced:
        st.caption(
            f"연속으로 너무 오래 버려서 {extracted.forced:,}장을 강제로 남겼습니다 — "
            "이 표시가 보이면 중복 판정이 이 영상에 잘 안 맞는다는 뜻입니다."
        )


def _framing_check(info) -> None:
    """이 촬영으로 결함이 보이기는 하는가 (V0).

    **모델을 고르기 전에 답해야 하는 질문이다.** 결함이 3픽셀로 잡히면 어떤 모델을 써도
    안 되는데, 보통은 프레임을 다 뽑고 라벨링을 하고 학습을 돌린 뒤에야 그 사실이 드러난다.
    그래서 **추출 버튼 위에** 둔다 — 며칠 뒤에 알 일을 지금 알려주자는 것.
    """
    with st.expander("📏 이 촬영으로 결함이 보이는가 (추출 전에 확인)", expanded=False):
        st.caption(
            "결함이 **모델 입력에서 몇 픽셀**이 되는지 계산합니다. 원본에서 100픽셀이어도 "
            "학습할 때 640으로 줄이면 그만큼 작아집니다 — 판정은 줄인 뒤 크기로 해야 합니다."
        )

        left, middle, right = st.columns(3)
        fov_m = left.number_input(
            "카메라가 담는 폭 (m)", 0.05, 50.0, 1.0, 0.05, key="video_fov",
            help="카메라 한 대가 화면 가로에 담는 실제 폭. 컨베이어 한 줄이면 1m 안팎입니다.",
        )
        defect_mm = middle.number_input(
            "잡아야 하는 가장 작은 결함 (mm)", 0.1, 500.0, 10.0, 0.5, key="video_defect_mm",
            help="짧은 변 기준입니다. 가장 작은 것을 넣으세요 — 그것이 기준을 정합니다.",
        )
        names = list(framing.MODEL_INPUT_PX)
        model_name = right.selectbox("어디에 태울 것인가", names, key="video_model_px")

        tiles = st.slider(
            "타일 분할 (가로 등분 수)", 1, 8, 1, key="video_tiles",
            help="프레임을 잘라 조각마다 추론하면 같은 결함이 입력에서 커집니다. "
                 "소프트웨어만 고치면 되지만 추론 횟수가 등분 수의 제곱만큼 늘어납니다.",
        )

        view = framing.Framing(
            defect_mm=float(defect_mm), fov_mm=float(fov_m) * 1000,
            sensor_px=int(info.width), model_px=framing.MODEL_INPUT_PX[model_name],
            tiles=int(tiles),
        )

        stat = st.columns(3)
        stat[0].metric("원본에서", f"{view.native_px:.0f}px")
        stat[1].metric("모델 입력에서", f"{view.model_input_px:.0f}px")
        stat[2].metric("판정", view.verdict)
        st.caption(
            f"이 촬영으로 잡을 수 있는 **가장 작은 결함은 약 {view.smallest_catchable_mm():.0f}mm**입니다. "
            f"(안정 기준 {framing.SAFE_PX:.0f}px · 한계 {framing.FLOOR_PX:.0f}px)"
        )

        note = framing.advice(view)
        if view.verdict == framing.VERDICT_OK:
            st.success(note, icon="✅")
        elif view.verdict == framing.VERDICT_HARD:
            st.warning(note, icon="⚠️")
        else:
            st.error(note, icon="🚫")

        for lever in framing.levers(view):
            if lever.reachable:
                st.caption(f"**{lever.name}** — {lever.change} · {lever.cost}")
            else:
                st.caption(
                    f"~~**{lever.name}** — {lever.change}~~ · "
                    "**원본에 그만한 정보가 없어 소용없습니다** (확대는 없던 것을 만들지 못합니다)"
                )

        _framing_hint_from_boxes(float(fov_m) * 1000)


def _framing_hint_from_boxes(fov_mm: float) -> None:
    """결함 크기를 모를 때, 이미 그려 둔 박스로 어림한다.

    실물 크기를 자로 재 본 사람은 드물지만 라벨링한 박스는 있다. 다만 **비율은 그 촬영을
    어떻게 했느냐의 결과이지 물리 상수가 아니므로**, 그 촬영이 담던 폭을 함께 물어야 한다.
    """
    from vision_ai import boxes as box_store

    frame = box_store.load()
    if frame.empty:
        return
    fraction = framing.fraction_from_boxes(frame, storage.load_manifest())
    if not fraction:
        return

    st.caption(
        f"참고 — 지금 이 프로젝트에 그려 둔 박스의 짧은 변 중앙값은 **화면 폭의 "
        f"{fraction:.2%}** 입니다. 그 이미지를 담던 폭이 "
        f"{framing.TYPICAL_SOURCE_FOV_MM / 1000:.1f}m였다면 결함 실물은 약 "
        f"**{framing.defect_mm_from_fraction(fraction, framing.TYPICAL_SOURCE_FOV_MM):.0f}mm**입니다. "
        "비율은 촬영 방식에 따라 달라지므로 실제로 한 번 재 보는 편이 훨씬 정확합니다."
    )


SOURCE_LIST = "목록에서 고르기"
SOURCE_UPLOAD = "올리기"
SOURCE_PATH = "경로 직접 입력"


def _describe(path: Path) -> str:
    """목록에 보일 이름. 크기를 붙여야 같은 이름의 다른 영상을 구분할 수 있다.

    영상마다 열어서 길이를 재면 목록을 그릴 때마다 파일을 전부 여는 셈이 되므로 크기만
    쓴다. MB로만 적으면 작은 영상이 전부 `0MB`가 되어 구분이 되지 않는다.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return path.name
    if size >= 1_048_576:
        return f"{path.name} · {size / 1_048_576:,.1f}MB"
    return f"{path.name} · {max(size // 1024, 1):,}KB"


def _video_source() -> Path | None:
    """쓸 영상을 정한다.

    예전에는 파일 경로를 통째로 타이핑해야 했다. 오타 하나로 실패하고, 무엇보다 **어떤
    영상이 이미 올라와 있는지 화면에서 알 수가 없었다.** 목록을 기본으로 두고, 목록에
    없는 영상만 올리거나 경로로 지정한다.
    """
    scan_extra = st.session_state.get("video_scan_dir", "").strip()
    found = video.listed(*( [scan_extra] if scan_extra else [] ))

    options = [SOURCE_LIST, SOURCE_UPLOAD, SOURCE_PATH]
    source = st.radio(
        "영상 지정 방식", options,
        index=0 if found else 1,
        horizontal=True, key="video_source",
        help="큰 영상은 올리기 상한(200MB)에 걸립니다. 로컬 실행이면 폴더를 훑는 편이 낫습니다.",
    )

    if source == SOURCE_LIST:
        picked = None
        if not found:
            st.info(
                f"`{video.video_dir()}`에 영상이 없습니다. **올리기**로 넣거나, 아래에 "
                "영상이 있는 폴더를 적어 훑으세요.",
                icon="📂",
            )
        else:
            picked = st.selectbox(
                "영상", found, format_func=_describe, key="video_pick",
                help=f"{video.video_dir()} 및 아래에 적은 폴더를 훑은 결과입니다.",
            )
            st.caption(f"경로: `{picked}`")
        st.text_input(
            "다른 폴더도 훑기 (선택)", key="video_scan_dir",
            placeholder="/mnt/nas/line1-cctv",
            help="하위 폴더까지 내려갑니다. 영상이 프로젝트 밖에 있는 경우가 오히려 보통입니다.",
        )
        if scan_extra and not Path(scan_extra).expanduser().is_dir():
            st.warning(f"폴더를 찾을 수 없습니다: `{scan_extra}`", icon="📁")
        return picked

    if source == SOURCE_UPLOAD:
        uploaded = st.file_uploader(
            "영상 올리기", type=sorted(e.lstrip(".") for e in video.VIDEO_EXTENSIONS),
            key="video_upload",
        )
        if uploaded is None:
            return None
        target = video.video_dir() / uploaded.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded.getvalue())
        st.caption(f"저장 위치: `{target}` — 다음부터는 **목록에서** 고를 수 있습니다.")
        return target

    entered = st.text_input(
        "영상 파일 경로", key="video_path", placeholder="/home/user/videos/line1.mp4"
    )
    if not entered:
        return None
    candidate = Path(entered).expanduser()
    if not candidate.is_file():
        st.error(f"파일을 찾을 수 없습니다: `{candidate}`")
        return None
    if not video.is_video(candidate):
        st.error(f"영상 파일이 아닙니다: `{candidate.suffix}`")
        return None
    return candidate


def _extraction_plan(info: "video.VideoInfo"):
    """간격을 정하는 방법을 고르게 한다.

    CCTV처럼 물체가 불규칙하게 지나가면 라인 속도로 계산할 수 없으므로 초당 장수 지정이
    기본이다. 컨베이어처럼 조건을 아는 경우에만 계산 모드가 의미가 있다.
    """
    mode = st.radio(
        "추출 간격 정하기",
        ["초당 장수로 지정", "공정 값으로 계산 (컨베이어)"],
        horizontal=True, key="video_plan_mode",
        help="CCTV는 물체 속도가 제각각이라 계산이 성립하지 않습니다. 초당 장수를 쓰세요.",
    )
    try:
        if mode.startswith("초당"):
            per_second = st.slider(
                "초당 몇 장", 0.1, min(float(info.fps), 30.0), 2.0, 0.1, key="video_rate"
            )
            return video.plan_from_rate(info, per_second=per_second)

        col1, col2, col3 = st.columns(3)
        field = col1.number_input(
            "카메라 시야 길이 (m)", 0.01, 100.0, 0.30, 0.01, key="video_fov",
            help="화면 안에 들어오는 컨베이어 길이",
        )
        speed = col2.number_input(
            "라인 속도 (m/s)", 0.01, 50.0, 0.5, 0.01, key="video_speed"
        )
        per_object = int(col3.number_input(
            "물체당 확보할 장수", 1, 50, video.DEFAULT_FRAMES_PER_OBJECT, 1, key="video_per_object",
            help="흔들림·가림에 대비한 여유까지 포함한 값",
        ))
        return video.plan_from_process(
            info, field_of_view_m=field, speed_mps=speed, frames_per_object=per_object
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


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
         "🎞️ 영상에서 프레임 추출",
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
        _video_tab()
    with tabs[4]:
        _synthetic_tab()
    with tabs[5]:
        _status_tab()


render()
