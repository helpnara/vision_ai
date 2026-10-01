"""3단계: 모델 개발 · 평가 — 학습 데이터 · 베이스라인 · 이상탐지 · Claude 2차 판정 · 평가 리포트 · 실험 기록.

Streamlit 화면(`app_pages/p3_modeling.py`)을 그대로 옮긴 얇은 층이다. 판단 로직은 코어에
있고, 여기서는 코어를 부르고 JSON으로 바꾼다. 학습·추출·판정처럼 오래 걸리는 것은 잡으로
돌리고, 마지막 학습 결과(점수 배열·모델 객체)는 `state`에 둔다 — 평가 리포트와 Claude
탭이 그것을 읽는다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel

from vision_ai import (
    charts,
    claude_review,
    cnn_features,
    config,
    evaluate,
    experiments,
    feature_cache,
    features,
    glossary,
    guide,
    labeling,
    models,
    monitoring,
    patch_cache,
    registry,
    report,
    settings as user_settings,
    storage,
    viz,
)

from .. import state
from ..common import bad_request, jsonable, not_found, records, table
from ..jobs import runner
from .files import image_response

router = APIRouter(prefix="/modeling", tags=["modeling"])

KEY_LAST_SUMMARY = "p3_last_summary"
"""마지막 학습 잡의 JSON 결과. Streamlit은 결과를 같은 실행 안에서 그렸지만, 여기서는
탭을 옮겼다 돌아오면 화면이 다시 그려지므로 서버가 마지막 요약을 들고 있어야 한다."""

KEY_HEATMAP = "p3_heatmap"
"""히트맵은 이미지 3장(입력·히트맵·오버레이)이 각각 다른 요청으로 오므로, 같은 이미지의
score_map을 세 번 계산하지 않도록 마지막 한 장을 기억한다."""

HEATMAP_CUTOFF = 0.6
"""오버레이에 칠하는 «상위 영역»의 기준 (정규화 점수). 원문 값 그대로."""

EFFORTS = ("low", "medium", "high")
OUTCOMES = ("FN", "FP", "TP", "TN")
OUTCOME_LABELS = {
    "FN": "미탐 (놓친 결함)", "FP": "오탐 (정상을 결함으로)", "TP": "정탐", "TN": "정상 판정",
}


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
    if df.empty:
        return df
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


def _is_synthetic(labeled: pd.DataFrame) -> bool:
    """합성 샘플뿐인가. 결함이 인위적으로 뚜렷해 지표가 실제보다 높게 나온다."""
    if labeled.empty:
        return False
    return set(labeled["source"].astype(str)) <= {"synthetic"}


def _empty_route(df: pd.DataFrame) -> dict:
    """막혔을 때 어디로 가야 하는지. 문구만 띄우면 초보자는 길을 잃는다."""
    if df.empty:
        return {"route": guide.route_of(guide.PAGE_INGEST), "label": "1단계 데이터 수집으로 이동"}
    if _labeled(df).empty:
        return {"route": guide.route_of(guide.PAGE_LABELING), "label": "2단계 라벨 검수로 이동"}
    return {"route": guide.route_of(guide.PAGE_LABELING), "label": "2단계 데이터 분할로 이동"}


def _advice(df: pd.DataFrame, kind: str) -> dict:
    """이 탭의 방식이 지금 데이터로 가능한지, 권장되는지 미리 알려준다.

    선택지만 주고 무엇을 골라야 할지 알려주지 않으면 초보자는 막힌다. 특히 VisA 공식 분할은
    학습 분할에 결함이 없어 지도학습이 시작조차 안 되는데, 화면은 선택지를 똑같이 보여준다.
    """
    advice = guide.model_advice(df)
    blocked = advice.baseline_blocked if kind == guide.KIND_BASELINE else advice.anomaly_blocked
    other = "이상탐지" if kind == guide.KIND_BASELINE else "베이스라인"
    if blocked:
        return {
            "blocked": True,
            "error": "**지금 데이터로는 이 방식을 쓸 수 없습니다.**\n\n" + "\n\n".join(advice.notes),
            "info": f"대신 **{other}** 탭을 사용하세요. {advice.reason}",
            "notes": [],
        }
    if advice.recommended == kind:
        banner = {"kind": "success", "icon": "✅",
                  "text": f"**이 데이터에는 이 방식을 권합니다.** {advice.reason}"}
    else:
        rec = "이상탐지" if advice.recommended == guide.KIND_ANOMALY else "베이스라인"
        banner = {"kind": "info", "icon": "💡",
                  "text": f"쓸 수는 있지만 이 데이터에는 **{rec}** 를 권합니다. {advice.reason}"}
    return {"blocked": False, "banner": banner, "notes": list(advice.notes)}


def _metric_row(metrics: dict) -> list[dict]:
    """지표 카드 묶음. 핵심 지표는 캡션으로 **바로 보이게** 하고, 부가 설명은 `?`에 둔다.

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
    return [
        {"key": key, "label": label, "value": value,
         "help": glossary.detail(key), "caption": glossary.caption(key)}
        for key, label, value in items
    ]


def _delta(new_value, old_value, *, fmt: str) -> str | None:
    """값이 그대로면 화살표를 띄우지 않는다.

    Streamlit은 0도 화살표와 함께 그린다. 빨간 `↑ +0건`은 나빠진 것처럼 읽힌다.
    """
    difference = new_value - old_value
    return None if difference == 0 else format(difference, fmt)


def _show_category_switch(labeled: pd.DataFrame) -> bool:
    """카테고리가 하나뿐이면 스위치를 아예 보이지 않는다. 고를 것이 없는 선택지는 소음이다."""
    return not labeled.empty and labeled["category"].astype(str).nunique() >= 2


def _category_thresholds(ids, labeled, y_eval, scores, target_recall, threshold, *, enabled):
    """카테고리마다 임계값을 따로 골랐을 때의 효과를 계산하고, 켜져 있으면 적용한다.

    **임계값 하나를 전체에 쓰면 가장 어려운 카테고리가 전체를 끌어내린다.** VisA 실측에서
    평균 재현율이 0.925인데 pcb4만 0.840이었다 — pcb4를 잡으려고 임계값을 낮추면 나머지
    셋의 오탐이 함께 늘고, 안 낮추면 pcb4에서 결함을 놓친다.

    표는 **꺼져 있어도 보여준다.** 켜면 무엇이 달라지는지 모르는 채로 고르게 할 수는 없다.
    반환: (화면 블록 또는 None, 적용할 카테고리별 임계값, 전체 기준값)
    """
    lookup = dict(zip(labeled["image_id"].astype(str), labeled["category"].astype(str)))
    categories = [lookup.get(str(image_id), "") for image_id in ids]
    if len(set(categories)) < 2:
        return None, {}, None

    per_category, overall, skipped = evaluate.thresholds_by_category(
        categories, y_eval, scores, target_recall
    )
    fallback = float(threshold if overall is None else overall)
    single = evaluate.summarize(y_eval, scores, threshold)
    mixed = evaluate.summarize_mixed(categories, y_eval, scores, per_category, fallback)
    frame = evaluate.summarize_by_category(categories, y_eval, scores, per_category, fallback)

    block = {
        "enabled": enabled,
        "metrics": [
            {"label": "재현율", "value": f"{mixed['recall']:.1%}",
             "delta": _delta(mixed["recall"], single["recall"], fmt="+.1%"),
             "inverse": False, "help": glossary.caption("recall")},
            {"label": "오탐", "value": f"{mixed['fp']:,}건",
             "delta": _delta(mixed["fp"], single["fp"], fmt="+,"), "inverse": True, "help": None},
            {"label": "미탐", "value": f"{mixed['fn']:,}건",
             "delta": _delta(mixed["fn"], single["fn"], fmt="+,"), "inverse": True, "help": None},
        ],
        "single_caption": (
            f"임계값 하나({threshold:.3f})로는 재현율 {single['recall']:.1%} · "
            f"오탐 {single['fp']:,}건 · 미탐 {single['fn']:,}건입니다."
        ),
        "table": table(frame),
        "skipped": [{"name": name, "reason": reason} for name, reason in skipped.items()],
    }
    return block, (per_category if enabled else {}), (fallback if enabled else None)


def _apply_categories(block, per_category, fallback, *, ids, labeled, y_eval, scores,
                      metrics, threshold, settings):
    """켜져 있으면 카테고리별 임계값을 지표와 설정에 반영한다."""
    if not per_category:
        return metrics, threshold
    lookup = dict(zip(labeled["image_id"].astype(str), labeled["category"].astype(str)))
    metrics = evaluate.summarize_mixed(
        [lookup.get(str(i), "") for i in ids], y_eval, scores, per_category, fallback
    )
    settings["category_thresholds"] = per_category
    return metrics, fallback


def _store_result(**payload) -> None:
    state.put(state.KEY_MODEL_RESULT, payload)


def _importance_bars(importance) -> list[dict]:
    if importance is None or len(importance) != len(features.FEATURE_NAMES):
        return []
    order = np.argsort(-importance)[:15]
    return [{"name": features.FEATURE_NAMES[i], "count": float(importance[i])} for i in order]


def _summary_if(kind: str) -> dict | None:
    """마지막 학습 요약이 이 탭의 것이면 돌려준다."""
    summary = state.get(KEY_LAST_SUMMARY)
    return summary if summary and summary.get("kind") == kind else None


# --- 1) 학습 데이터 -----------------------------------------------------------

@router.get("/data")
def data_tab() -> dict:
    df = _resolved()
    if df.empty:
        return {"empty": True, "link": _empty_route(df)}

    labeled = _labeled(df)
    ready, problems = _readiness(df)
    body: dict = {
        "empty": False,
        "metrics": {
            "labeled": int(len(labeled)),
            "normal": int((labeled["y"] == 0).sum()),
            "defect": int((labeled["y"] == 1).sum()),
            "categories": int(labeled["category"].nunique()),
        },
        "ready": ready,
        "problems": problems,
        "synthetic": _is_synthetic(labeled),
        "crosstab": None,
        "features": {
            "count": len(features.FEATURE_NAMES),
            "image_size": features.IMAGE_SIZE,
            "names": list(features.FEATURE_NAMES),
        },
        "dataset": None,
    }
    if labeled.empty:
        return jsonable(body)

    crosstab = pd.crosstab(
        labeled["split"].astype(str),
        labeled["label"].astype(str).map(lambda x: config.LABEL_KO.get(x, x)),
    )
    crosstab.index.name = "split"
    body["crosstab"] = table(crosstab, index=True)

    cached = state.get(state.KEY_DATASET)
    if cached and cached.get("signature") == _split_signature(labeled):
        dataset = cached["dataset"]
        body["dataset"] = {"rows": int(dataset.X.shape[0]), "cols": int(dataset.X.shape[1])}
    return jsonable(body)


def _extract_note(labeled: pd.DataFrame) -> str:
    """잡 설명. 장수는 라벨 표만 세면 알 수 있다 — 특징 추출은 잡 안에서 한다."""
    total = len(labeled)
    known = feature_cache.load()
    fresh = sum(
        1
        for image_id, path in zip(labeled["image_id"].astype(str), labeled["path_abs"])
        if known.get(image_id, path) is None
    )
    return (
        f"이미지 {total:,}장에서 각각 {len(features.FEATURE_NAMES)}개 값을 계산합니다. "
        + (
            f"이 중 {total - fresh:,}장은 전에 계산해 둔 값을 그대로 씁니다."
            if fresh < total
            else "처음 계산하는 이미지라 시간이 걸립니다. 다음부터는 저장된 값을 씁니다."
        )
    )


def _build_dataset(labeled: pd.DataFrame, job) -> models.Dataset:
    """특징 행렬을 만들고 상태에 캐시한다.

    상태 캐시 아래에 디스크 캐시가 한 겹 더 있다(`feature_cache`). 이미 계산해 둔 이미지는
    다시 뽑지 않는다.
    """
    signature = _split_signature(labeled)
    cached = state.get(state.KEY_DATASET)
    if cached and cached.get("signature") == signature:
        return cached["dataset"]
    dataset = models.build_dataset(
        labeled["path_abs"].tolist(),
        labeled["y"].tolist(),
        labeled["image_id"].astype(str).tolist(),
        progress=lambda d, t: job.progress(d, t, f"특징 추출 중 {d:,}/{t:,}"),
    )
    if len(dataset) == 0:
        raise ValueError("특징을 추출할 수 있는 이미지가 없습니다.")
    state.put(state.KEY_DATASET, {"signature": signature, "dataset": dataset})
    return dataset


def _dataset_messages(dataset: models.Dataset) -> tuple[str, list[str]]:
    done = f"특징 준비 완료 — 새로 계산 {dataset.extracted:,}장 · 저장분 재사용 {dataset.reused:,}장"
    warnings = [f"읽기 실패 {len(dataset.failed)}건은 제외했습니다."] if dataset.failed else []
    return done, warnings


@router.post("/data/extract")
def start_extract() -> dict:
    labeled = _labeled(_resolved())
    if labeled.empty:
        raise bad_request("라벨된 이미지가 없습니다. 1~2단계를 먼저 진행하세요.")

    def run(job):
        state.pop(state.KEY_DATASET)
        dataset = _build_dataset(labeled, job)
        done, warnings = _dataset_messages(dataset)
        return {
            "rows": int(dataset.X.shape[0]), "cols": int(dataset.X.shape[1]),
            "done": done, "warnings": warnings,
        }

    job = runner.submit("특징 추출 중", run, note=_extract_note(labeled))
    return {"job_id": job.id}


# --- 2) 베이스라인 ------------------------------------------------------------

@router.get("/baseline")
def baseline_tab() -> dict:
    df = _resolved()
    body: dict = {
        "empty": df.empty,
        "term": glossary.term("베이스라인"),
        "link": _empty_route(df),
        "last": _summary_if("baseline"),
    }
    if df.empty:
        return jsonable(body)

    body["advice"] = _advice(df, guide.KIND_BASELINE)
    if body["advice"]["blocked"]:
        return jsonable(body)

    ready, problems = _readiness(df)
    body["ready"], body["problems"] = ready, problems
    if not ready:
        return jsonable(body)

    labeled = _labeled(df)
    splits = labeled["split"].astype(str)
    body.update({
        "kinds": [{"key": k, "label": models.BASELINE_KIND_LABELS.get(k, k)} for k in models.BASELINE_KINDS],
        "splits": [config.SPLIT_TEST, config.SPLIT_VAL],
        "target_recall": user_settings.load().target_recall,
        "show_categories": _show_category_switch(labeled),
        # 장수는 라벨 표만 세면 알 수 있다. 특징 추출은 학습을 누른 뒤로 미룬다 —
        # 화면을 열기만 해도 전량 추출하면 4,584장 기준 90초를 기다려야 한다.
        "counts": {
            "train": int((splits == config.SPLIT_TRAIN).sum()),
            config.SPLIT_TEST: int((splits == config.SPLIT_TEST).sum()),
            config.SPLIT_VAL: int((splits == config.SPLIT_VAL).sum()),
        },
    })
    return jsonable(body)


class BaselineBody(BaseModel):
    kind: str = "logreg"
    balanced: bool = True
    eval_split: str = config.SPLIT_TEST
    target_recall: float = 0.95
    use_categories: bool = False


def _split_masks(dataset: models.Dataset, labeled: pd.DataFrame) -> dict[str, np.ndarray]:
    lookup = labeled.set_index(labeled["image_id"].astype(str))["split"].astype(str).to_dict()
    assigned = np.array([lookup.get(i, config.SPLIT_NONE) for i in dataset.image_ids])
    return {name: assigned == name for name in config.SPLITS}


@router.post("/baseline/train")
def start_baseline(body: BaselineBody) -> dict:
    if body.kind not in models.BASELINE_KINDS:
        raise bad_request(f"지원하지 않는 모델 종류: {body.kind}")
    if body.eval_split not in (config.SPLIT_TEST, config.SPLIT_VAL):
        raise bad_request(f"평가 분할은 {config.SPLIT_TEST}/{config.SPLIT_VAL} 중 하나입니다.")
    df = _resolved()
    if df.empty:
        raise bad_request("수집된 이미지가 없습니다.")
    advice = guide.model_advice(df)
    if advice.baseline_blocked:
        raise bad_request("지금 데이터로는 이 방식을 쓸 수 없습니다. " + " ".join(advice.notes))
    ready, problems = _readiness(df)
    if not ready:
        raise bad_request(" ".join(problems))

    labeled = _labeled(df)
    splits = labeled["split"].astype(str)
    n_train = int((splits == config.SPLIT_TRAIN).sum())
    n_eval = int((splits == body.eval_split).sum())
    if n_eval == 0:
        raise bad_request(f"`{body.eval_split}` 분할에 이미지가 없습니다.")
    use_categories = body.use_categories and _show_category_switch(labeled)

    def run(job):
        warnings: list[str] = []
        dataset = _build_dataset(labeled, job)
        done_text, dataset_warnings = _dataset_messages(dataset)
        warnings += dataset_warnings
        masks = _split_masks(dataset, labeled)
        train_mask, eval_mask = masks[config.SPLIT_TRAIN], masks[body.eval_split]
        if not eval_mask.any():
            raise ValueError(f"`{body.eval_split}` 분할에서 특징을 뽑을 수 있는 이미지가 없습니다.")

        job.progress(job.total or 1, job.total or 1, "학습 중...")
        try:
            model = models.BaselineModel(
                models.BaselineConfig(kind=body.kind, balanced=body.balanced)
            ).fit(dataset.X[train_mask], dataset.y[train_mask])
            scores = model.score(dataset.X[eval_mask])
        except (ValueError, RuntimeError) as exc:
            raise ValueError(f"학습 실패: {exc}") from exc

        y_eval = dataset.y[eval_mask]
        threshold = evaluate.threshold_for_target_recall(y_eval, scores, body.target_recall)
        if threshold is None:
            threshold = float(np.median(scores))
            warnings.append(
                f"목표 재현율 {body.target_recall:.0%}를 만족하는 임계값이 없습니다. 중앙값을 임시로 사용합니다."
            )

        metrics = evaluate.summarize(y_eval, scores, threshold)
        ids = [dataset.image_ids[i] for i in np.flatnonzero(eval_mask)]
        settings = {"kind": body.kind, "balanced": body.balanced, "target_recall": body.target_recall}
        block, per_category, fallback = _category_thresholds(
            ids, labeled, y_eval, scores, body.target_recall, threshold, enabled=use_categories,
        )
        metrics, threshold = _apply_categories(
            block, per_category, fallback, ids=ids, labeled=labeled, y_eval=y_eval,
            scores=scores, metrics=metrics, threshold=threshold, settings=settings,
        )

        _store_result(
            kind="baseline", model=body.kind, split=body.eval_split, scores=scores, y=y_eval,
            image_ids=ids, threshold=threshold, metrics=metrics, settings=settings, n_train=n_train,
        )

        importance = model.feature_importance()
        artifact_path = config.model_dir() / f"baseline_{body.kind}.joblib"
        try:
            model.save(artifact_path)
        except Exception as exc:  # noqa: BLE001 — 저장 실패가 평가를 막지는 않게 한다
            warnings.append(f"모델 저장 실패: {exc}")
            artifact_path = None

        run_id = experiments.record_run(
            kind="baseline", model=body.kind, split=body.eval_split, metrics=metrics,
            settings=settings, n_train=n_train, n_eval=n_eval,
            artifacts={"model": str(artifact_path) if artifact_path else None},
        )
        summary = jsonable({
            "kind": "baseline", "run_id": run_id, "metrics": metrics,
            "metric_row": _metric_row(metrics), "importance": _importance_bars(importance),
            "categories": block, "warnings": warnings, "dataset_note": done_text,
        })
        state.put(KEY_LAST_SUMMARY, summary)
        return summary

    job = runner.submit("학습 후 평가", run, note=_extract_note(labeled))
    return {"job_id": job.id}


# --- 3) 이상탐지 --------------------------------------------------------------

def _defect_options(df: pd.DataFrame) -> list[str]:
    defects = _labeled(df)
    if defects.empty:
        return []
    return defects[defects["y"] == 1]["image_id"].astype(str).tolist()


@router.get("/anomaly")
def anomaly_tab() -> dict:
    df = _resolved()
    body: dict = {
        "empty": df.empty,
        "term": glossary.term("이상탐지"),
        "link": _empty_route(df),
        "last": _summary_if("anomaly"),
        "heatmap": {
            "available": state.get(state.KEY_ANOMALY_MODEL) is not None,
            "defects": _defect_options(df),
        },
    }
    if df.empty:
        return jsonable(body)

    body["advice"] = _advice(df, guide.KIND_ANOMALY)
    if body["advice"]["blocked"]:
        return jsonable(body)

    labeled = _labeled(df)
    splits = labeled["split"].astype(str)
    train_normal = labeled[(splits == config.SPLIT_TRAIN) & (labeled["y"] == 0)]
    per_split = {}
    for name in (config.SPLIT_TEST, config.SPLIT_VAL):
        rows = labeled[splits == name]
        per_split[name] = {"total": int(len(rows)), "defect": int((rows["y"] == 1).sum())}
    body.update({
        "splits": [config.SPLIT_TEST, config.SPLIT_VAL],
        "backends": [{"key": k, "label": models.BACKEND_LABELS.get(k, k)} for k in models.BACKENDS],
        "backend_cnn": models.BACKEND_CNN,
        "cnn": {
            "available": cnn_features.available(),
            "describe": cnn_features.describe(),
            "size_mb": cnn_features.MODEL_SIZE_MB,
        },
        "score_modes": [{"key": k, "label": models.score_mode_label(k)} for k in models.IMAGE_SCORE_MODES],
        "patch_feature_dim": features.PATCH_FEATURE_DIM,
        "target_recall": user_settings.load().target_recall,
        "show_categories": _show_category_switch(labeled),
        "counts": {"train_normal": int(len(train_normal)), "eval": per_split},
    })
    return jsonable(body)


@router.post("/anomaly/download-cnn")
def start_cnn_download() -> dict:
    """사전학습 모델 파일(약 45MB). 저장소에 넣기엔 커서 필요할 때 내려받는다."""
    def run(job):
        try:
            cnn_features.download(
                progress=lambda done, total: job.progress(
                    done, total, f"내려받는 중... {done/1024**2:.0f}/{total/1024**2:.0f}MB"
                )
            )
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
        return {"describe": cnn_features.describe()}

    job = runner.submit("모델 내려받는 중", run,
                        note=f"사전학습 모델 파일 약 {cnn_features.MODEL_SIZE_MB}MB를 내려받습니다.")
    return {"job_id": job.id}


class AnomalyBody(BaseModel):
    eval_split: str = config.SPLIT_TEST
    backend: str = models.BACKEND_CLASSIC
    per_position: bool = True
    score_mode: str = "p99"
    target_recall: float = 0.95
    use_categories: bool = False


@router.post("/anomaly/train")
def start_anomaly(body: AnomalyBody) -> dict:
    if body.eval_split not in (config.SPLIT_TEST, config.SPLIT_VAL):
        raise bad_request(f"평가 분할은 {config.SPLIT_TEST}/{config.SPLIT_VAL} 중 하나입니다.")
    if body.backend not in models.BACKENDS:
        raise bad_request(f"지원하지 않는 특징 추출 방식: {body.backend}")
    if body.score_mode not in models.IMAGE_SCORE_MODES:
        raise bad_request(f"지원하지 않는 집계 방식: {body.score_mode}")
    df = _resolved()
    if df.empty:
        raise bad_request("수집된 이미지가 없습니다.")
    advice = guide.model_advice(df)
    if advice.anomaly_blocked:
        raise bad_request("지금 데이터로는 이 방식을 쓸 수 없습니다. " + " ".join(advice.notes))

    labeled = _labeled(df)
    splits = labeled["split"].astype(str)
    train_normal = labeled[(splits == config.SPLIT_TRAIN) & (labeled["y"] == 0)]
    eval_rows = labeled[splits == body.eval_split]
    if train_normal.empty or eval_rows.empty:
        raise bad_request("학습용 정상 이미지와 평가 분할이 모두 필요합니다.")
    # CNN 모델 파일이 없으면 원문처럼 고전 CV로 되돌린다 — 화면도 같은 규칙으로 안내한다.
    backend = body.backend
    if backend == models.BACKEND_CNN and not cnn_features.available():
        backend = models.BACKEND_CLASSIC
    use_categories = body.use_categories and _show_category_switch(labeled)

    def run(job):
        warnings: list[str] = []
        model = models.PatchAnomalyModel(
            models.AnomalyConfig(per_position=body.per_position, image_score=body.score_mode, backend=backend)
        )
        cache = patch_cache.load(model.config)
        reused_before = len(cache)

        # 잡 하나가 두 단계로 되어 있다(정상 학습 50% + 평가 50%). 비율을 직접 환산해 넘긴다.
        job.progress(0, 1000, "정상 분포 학습 중...")
        try:
            model.fit_grids(
                grid for _, grid in patch_cache.grids_for(
                    train_normal, model, cache=cache,
                    progress=lambda done, total: job.progress(
                        int(done / max(total, 1) * 500), 1000, f"정상 학습 {done}/{total}"
                    ),
                )
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            patch_cache.save(cache)   # 학습이 실패해도 뽑아 둔 특징은 남긴다
            raise ValueError(f"학습 실패: {exc}") from exc

        scores, ys, ids = [], [], []
        truth = {str(row["image_id"]): int(row["y"]) for _, row in eval_rows.iterrows()}
        for image_id, grid in patch_cache.grids_for(
            eval_rows, model, cache=cache,
            progress=lambda done, total: job.progress(
                500 + int(done / max(total, 1) * 500), 1000, f"평가 {done}/{total}"
            ),
        ):
            scores.append(model.aggregate(model.score_grid_features(grid)))
            ys.append(truth[image_id])
            ids.append(image_id)

        added = cache.added
        patch_cache.save(cache)
        cache_note = None
        if added:
            cache_note = (
                f"격자 특징 {added:,}장을 새로 뽑아 저장했습니다 (저장해 둔 것 재사용 "
                f"{reused_before:,}장). 다음 실행부터는 이 부분을 건너뜁니다."
            )
        elif len(cache):
            cache_note = f"저장해 둔 격자 특징 {len(cache):,}장을 그대로 썼습니다."

        if not scores:
            raise ValueError("평가할 이미지를 읽지 못했습니다.")

        scores_array, y_array = np.asarray(scores), np.asarray(ys)
        threshold = evaluate.threshold_for_target_recall(y_array, scores_array, body.target_recall)
        if threshold is None:
            threshold = float(np.median(scores_array))
            warnings.append(f"목표 재현율 {body.target_recall:.0%} 달성 임계값이 없어 중앙값을 사용합니다.")

        metrics = evaluate.summarize(y_array, scores_array, threshold)
        settings = {
            "per_position": body.per_position, "image_score": body.score_mode, "backend": backend,
            "patch": model.config.patch, "stride": model.config.stride,
            "target_recall": body.target_recall,
        }
        block, per_category, fallback = _category_thresholds(
            ids, labeled, y_array, scores_array, body.target_recall, threshold, enabled=use_categories,
        )
        metrics, threshold = _apply_categories(
            block, per_category, fallback, ids=ids, labeled=labeled, y_eval=y_array,
            scores=scores_array, metrics=metrics, threshold=threshold, settings=settings,
        )
        state.put(state.KEY_ANOMALY_MODEL, model)
        state.pop(KEY_HEATMAP)
        _store_result(
            kind="anomaly", model="mahalanobis", split=body.eval_split, scores=scores_array,
            y=y_array, image_ids=ids, threshold=threshold, metrics=metrics,
            settings=settings, n_train=model.n_train,
        )

        artifact_path = config.model_dir() / "anomaly_patch.npz"
        try:
            model.save(artifact_path)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"모델 저장 실패: {exc}")
            artifact_path = None

        run_id = experiments.record_run(
            kind="anomaly", model="mahalanobis", split=body.eval_split, metrics=metrics,
            settings=settings, n_train=model.n_train, n_eval=len(ids),
            artifacts={"model": str(artifact_path) if artifact_path else None},
        )
        summary = jsonable({
            "kind": "anomaly", "run_id": run_id, "metrics": metrics,
            "metric_row": _metric_row(metrics), "categories": block,
            "warnings": warnings, "cache_note": cache_note,
        })
        state.put(KEY_LAST_SUMMARY, summary)
        return summary

    job = runner.submit(
        "이상탐지 학습 후 평가", run,
        note=(f"정상 {len(train_normal):,}장으로 정상 분포를 학습한 뒤 평가({body.eval_split}) "
              f"{len(eval_rows):,}장에 점수를 매깁니다. 격자 특징은 한 번 뽑으면 저장해 두고 다시 씁니다."),
    )
    return {"job_id": job.id}


def _heatmap_for(image_id: str) -> dict:
    """학습된 이상탐지 모델로 결함 위치 히트맵을 만든다 (마지막 한 장은 기억해 둔다)."""
    model = state.get(state.KEY_ANOMALY_MODEL)
    if model is None:
        raise not_found("먼저 이상탐지 모델을 학습하세요.")
    cached = state.get(KEY_HEATMAP)
    if cached and cached.get("image_id") == image_id and cached.get("model") is model:
        return cached

    df = _resolved()
    rows = df[df["image_id"].astype(str) == image_id] if not df.empty else df
    if rows.empty:
        raise not_found(f"이미지가 없습니다: {image_id}")
    row = rows.iloc[0]
    image = viz.load_rgb(row["path_abs"])
    if image is None:
        raise not_found("이미지를 읽을 수 없습니다.")

    resized = features.preprocess(image)
    heatmap = model.score_map(resized)
    normalized = heatmap - heatmap.min()
    normalized = normalized / max(float(normalized.max()), 1e-6)
    entry = {
        "image_id": image_id, "model": model, "resized": resized, "heatmap": heatmap,
        "normalized": normalized, "path": row["path"],
    }
    state.put(KEY_HEATMAP, entry)
    return entry


@router.get("/anomaly/heatmap/{image_id}")
def heatmap_info(image_id: str) -> dict:
    """위치 정확도. 정답 마스크가 없으면 계산할 수 없다고 말한다."""
    entry = _heatmap_for(image_id)
    mask_path = labeling.find_mask_path(storage.resolve_path(entry["path"]))
    if mask_path is None:
        return {"image_id": image_id, "localization": None,
                "note": "정답 마스크가 없어 위치 정확도는 계산할 수 없습니다."}
    import cv2

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {"image_id": image_id, "localization": None, "note": None}
    scores = evaluate.localization_metrics(entry["heatmap"], mask)
    return jsonable({"image_id": image_id, "localization": scores, "note": None})


@router.get("/anomaly/heatmap/{image_id}/input")
def heatmap_input(image_id: str) -> Response:
    return image_response(_heatmap_for(image_id)["resized"], width=None)


@router.get("/anomaly/heatmap/{image_id}/heatmap")
def heatmap_image(image_id: str) -> Response:
    normalized = _heatmap_for(image_id)["normalized"]
    gray = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    return image_response(np.stack([gray, gray, gray], axis=-1), width=None)


@router.get("/anomaly/heatmap/{image_id}/overlay")
def heatmap_overlay(image_id: str) -> Response:
    entry = _heatmap_for(image_id)
    mask = (entry["normalized"] > HEATMAP_CUTOFF).astype(np.uint8) * 255
    return image_response(viz.overlay_mask(entry["resized"], mask, alpha=0.45), width=None)


# --- 4) Claude 2차 판정 -------------------------------------------------------

def _candidates(result: dict, *, limit: int, skip_reviewed: bool) -> pd.DataFrame:
    errors = evaluate.error_frame(
        result["image_ids"], result["y"], result["scores"], result["threshold"]
    )
    candidates = claude_review.select_uncertain(errors, threshold=result["threshold"], limit=limit * 3)
    if skip_reviewed:
        done = claude_review.reviewed_ids()
        candidates = candidates[~candidates["image_id"].isin(done)]
    return candidates.head(limit)


def _reviews_block() -> dict:
    reviews = claude_review.load_reviews()
    if reviews.empty:
        return {"empty": True}
    ok = reviews[reviews["status"] == "ok"]
    total_tokens = pd.to_numeric(reviews["input_tokens"], errors="coerce").fillna(0).sum()
    return {
        "empty": False,
        "metrics": {
            "total": int(len(reviews)),
            "ok": int(len(ok)),
            "defect": int(ok["defect"].astype(str).isin(["True", "true"]).sum()),
            "input_tokens": int(total_tokens),
        },
        "table": table(reviews[["image_id", "defect", "defect_type", "severity", "confidence", "reason", "status", "model"]]),
        "csv": reviews.to_csv(index=False),
    }


@router.get("/claude")
def claude_tab(limit: int = 8, skip_reviewed: bool = True) -> dict:
    result = state.get(state.KEY_MODEL_RESULT)
    body: dict = {
        "sdk_installed": claude_review.sdk_installed(),
        "hint": claude_review.credentials_hint(),
        "has_result": result is not None,
        "efforts": list(EFFORTS),
        "reviews": _reviews_block(),
    }
    if result is None:
        return jsonable(body)

    limit = max(1, min(int(limit), 100))
    candidates = _candidates(result, limit=limit, skip_reviewed=skip_reviewed)
    body["threshold"] = float(result["threshold"])
    body["candidates"] = {
        "count": int(len(candidates)),
        "cost": claude_review.estimate_cost(len(candidates)),
        "model": claude_review.DEFAULT_MODEL,
        "table": table(candidates[["image_id", "label", "score", "outcome", "uncertainty"]]),
    }
    return jsonable(body)


class ClaudeBody(BaseModel):
    limit: int = 8
    effort: str = claude_review.DEFAULT_EFFORT
    skip_reviewed: bool = True


@router.post("/claude/run")
def start_claude(body: ClaudeBody) -> dict:
    if body.effort not in EFFORTS:
        raise bad_request(f"추론 강도는 {'/'.join(EFFORTS)} 중 하나입니다.")
    if not claude_review.sdk_installed():
        raise bad_request("`anthropic` SDK가 설치되지 않았습니다. `pip install anthropic` 후 다시 시도하세요.")
    result = state.get(state.KEY_MODEL_RESULT)
    if result is None:
        raise bad_request("먼저 베이스라인 또는 이상탐지 탭에서 1차 모델을 실행하세요.")
    df = _resolved()
    candidates = _candidates(result, limit=max(1, min(body.limit, 100)), skip_reviewed=body.skip_reviewed)
    if candidates.empty:
        raise bad_request("2차 판정이 필요한 이미지가 없습니다.")

    def run(job):
        lookup = df.set_index(df["image_id"].astype(str))
        results = []
        total = len(candidates)
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
                    image, image_id=image_id, roi=roi, category=str(row["category"]),
                    candidate_type=str(row["defect_type"]), model_score=float(candidate["score"]),
                    effort=body.effort,
                )
            )
            job.progress(index, total, f"Claude 판정 중... {index}/{total}")

        if not results:
            raise ValueError("판정할 이미지를 준비하지 못했습니다.")
        claude_review.save_reviews(results)
        statuses = pd.Series([r.status for r in results]).value_counts().to_dict()
        return jsonable({
            "ok": bool(statuses.get("ok")),
            "statuses": {str(k): int(v) for k, v in statuses.items()},
            "first_reason": results[0].reason,
        })

    job = runner.submit("Claude 판정 중", run,
                        note=f"임계값 근처의 {len(candidates)}건을 한 장씩 Claude에 보냅니다.")
    return {"job_id": job.id}


# --- 5) 평가 리포트 -----------------------------------------------------------

def _target_recall(result: dict) -> float:
    return float(result.get("settings", {}).get("target_recall", 0.95))


@router.get("/report")
def report_tab() -> dict:
    result = state.get(state.KEY_MODEL_RESULT)
    if result is None:
        return {"has_result": False}
    scores, y = result["scores"], result["y"]
    low, high = float(np.min(scores)), float(np.max(scores))
    if high <= low:
        high = low + 1e-6
    saved = user_settings.load()
    return jsonable({
        "has_result": True,
        "header": (
            f"**{result['kind']} / {result['model']}** · 평가 분할 `{result['split']}` · "
            f"학습 {result['n_train']:,}장 / 평가 {len(y):,}장"
        ),
        "synthetic": _is_synthetic(_labeled(_resolved())),
        "slider": {"min": low, "max": high, "step": (high - low) / 200,
                   "value": float(np.clip(result["threshold"], low, high))},
        "defaults": {"prevalence": saved.prevalence, "volume": saved.volume},
        "help": {"prevalence": user_settings.HELP["prevalence"], "volume": user_settings.HELP["volume"]},
        "target_recall": _target_recall(result),
        "outcomes": [{"key": k, "label": OUTCOME_LABELS[k]} for k in OUTCOMES],
        "kind": result["kind"], "split": result["split"],
    })


class EvaluateBody(BaseModel):
    threshold: float
    prevalence: float = 0.01
    volume: int = 1000
    outcome: str = "FN"


def _errors_with_reviews(result: dict, threshold: float) -> pd.DataFrame:
    errors = evaluate.error_frame(result["image_ids"], result["y"], result["scores"], threshold)
    reviews = claude_review.load_reviews()
    if not reviews.empty:
        errors = errors.merge(
            reviews[["image_id", "defect_type", "reason", "status"]].rename(
                columns={"defect_type": "claude_type", "reason": "claude_reason"}
            ),
            on="image_id", how="left",
        )
    return errors


def _impact_table(recall: float, fpr: float, volume: int) -> dict:
    frame = evaluate.impact_by_prevalence(recall, fpr, volume=volume)
    display = frame.copy()
    display["불량률"] = display["불량률"].map(lambda v: f"{v:.0%}")
    for column in ("기대 정밀도", "검수 비율", "검수량 절감률"):
        display[column] = display[column].map(lambda v: f"{v:.1%}")
    for column in display.columns:
        if "장당" in column:
            display[column] = display[column].map(lambda v: f"{v:,.0f}")
    return table(display)


def _samples(df: pd.DataFrame, subset: pd.DataFrame) -> list[dict]:
    """오탐·미탐 표본. 숫자만으로는 무엇이 잘못됐는지 알 수 없다 — 이미지를 봐야 한다."""
    lookup = df.set_index(df["image_id"].astype(str)) if not df.empty else df
    items = []
    for _, row in subset.iterrows():
        image_id = str(row["image_id"])
        if df.empty or image_id not in lookup.index:
            items.append({"image_id": image_id, "missing": f"{image_id}: 원본 없음"})
            continue
        source = lookup.loc[image_id]
        if not Path(source["path_abs"]).is_file():
            items.append({"image_id": image_id, "missing": f"{image_id}: 읽기 실패"})
            continue
        caption = f"{source['category']} · 점수 {row['score']:.3f}"
        claude_type = row.get("claude_type")
        if isinstance(claude_type, str) and claude_type:
            caption += f" · Claude: {claude_type}"
        items.append({"image_id": image_id, "caption": caption})
    return items


@router.post("/report/evaluate")
def report_evaluate(body: EvaluateBody) -> dict:
    """임계값·불량률·물량이 바뀔 때마다 리포트 전체를 한 번에 다시 계산한다."""
    result = state.get(state.KEY_MODEL_RESULT)
    if result is None:
        raise bad_request("먼저 베이스라인 또는 이상탐지 탭에서 모델을 실행하세요.")
    if body.outcome not in OUTCOMES:
        raise bad_request(f"표시할 종류는 {'/'.join(OUTCOMES)} 중 하나입니다.")
    prevalence = min(max(float(body.prevalence), 0.001), 0.5)
    volume = int(min(max(int(body.volume), 100), 1_000_000))

    scores, y = result["scores"], result["y"]
    metrics = evaluate.summarize(y, scores, body.threshold)
    impact = evaluate.business_impact(
        float(metrics.get("recall", 0.0)), float(metrics.get("false_alarm_rate", 0.0)),
        prevalence=prevalence, volume=volume,
    )
    verdict = glossary.verdict(metrics, impact, target_recall=_target_recall(result))

    sweep = evaluate.threshold_sweep(y, scores)
    sweep_spec = None
    if not sweep.empty:
        sweep_spec = charts.line_chart(
            records(sweep[["threshold", "recall", "precision"]]), x="threshold",
            ys=["recall", "precision"], x_title="threshold",
        )
    roc = evaluate.roc_curve_frame(y, scores)
    roc_spec = None if roc.empty else charts.line_chart(records(roc[["fpr", "tpr"]]), x="fpr", ys=["tpr"], x_title="fpr")

    errors = _errors_with_reviews(result, body.threshold)
    subset = errors[errors["outcome"] == body.outcome]
    df = _resolved()

    return jsonable({
        "metrics": metrics,
        "metric_row": _metric_row(metrics),
        "impact": impact,
        "impact_captions": {
            "reduction": {"help": glossary.detail("reduction"), "caption": glossary.caption("reduction")},
            "missed": {"help": glossary.detail("missed"), "caption": glossary.caption("missed")},
        },
        "verdict": {"level": verdict.level, "headline": verdict.headline, "actions": list(verdict.actions)},
        "confusion": table(pd.DataFrame(
            [["실제 정상", metrics["tn"], metrics["fp"]], ["실제 결함", metrics["fn"], metrics["tp"]]],
            columns=["", "예측 정상", "예측 결함"],
        )),
        "sweep": sweep_spec,
        "roc": roc_spec,
        "samples": _samples(df, subset.head(8)),
        "subset": table(subset),
        "subset_count": int(len(subset)),
        "impact_by_prevalence": _impact_table(
            float(metrics.get("recall", 0.0)), float(metrics.get("false_alarm_rate", 0.0)), volume
        ),
    })


class ReportFilesBody(BaseModel):
    threshold: float
    prevalence: float = 0.01
    volume: int = 1000


@router.post("/report/files")
def report_files(body: ReportFilesBody) -> dict:
    """보고서(.md)와 판정 결과(errors.csv). 작은 텍스트라 JSON에 문자열로 담는다."""
    result = state.get(state.KEY_MODEL_RESULT)
    if result is None:
        raise bad_request("먼저 베이스라인 또는 이상탐지 탭에서 모델을 실행하세요.")
    df = _resolved()
    metrics = evaluate.summarize(result["y"], result["scores"], body.threshold)
    text = report.build(
        resolved=df, result=result, metrics=metrics,
        prevalence=body.prevalence, volume=body.volume,
        target_recall=_target_recall(result),
        production=registry.production(),
        log=monitoring.load_log(),
    )
    errors = _errors_with_reviews(result, body.threshold)
    return {
        "report": {"filename": report.filename(), "text": text},
        "errors": {"filename": f"eval_{result['kind']}_{result['split']}.csv", "text": errors.to_csv(index=False)},
    }


# --- 6) 실험 기록 -------------------------------------------------------------

@router.get("/experiments")
def experiments_tab() -> dict:
    runs = experiments.comparison_frame()
    if runs.empty:
        return {"empty": True}
    best_text = None
    best = runs.dropna(subset=["recall"])
    if not best.empty:
        top = best.sort_values(["recall", "precision"], ascending=False).iloc[0]
        best_text = (
            f"현재 최고 재현율: `{top['run_id']}` ({top['kind']}/{top['model']}) — "
            f"재현율 {top['recall']:.3f}, 정밀도 {top['precision']:.3f}, 미탐 {int(top['fn'])}건"
        )
    return jsonable({
        "empty": False,
        "table": table(runs),
        "help": {c: glossary.column_help(c) for c in runs.columns if glossary.column_help(c)},
        "best": best_text,
        "run_ids": runs["run_id"].astype(str).tolist(),
        "csv": experiments.load_runs().to_csv(index=False),
    })


@router.get("/experiments/{run_id}")
def experiment_detail(run_id: str) -> dict:
    detail = experiments.load_run_detail(run_id)
    if detail is None:
        raise not_found(f"실험 기록이 없습니다: {run_id}")
    return jsonable(detail)

