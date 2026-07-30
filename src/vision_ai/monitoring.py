"""4단계: 추론 로그 · 드리프트 감시 · 재학습 판단.

모델을 만드는 것보다 **만든 뒤 성능이 떨어지는 것을 알아채는 일**이 어렵다.
성능 저하는 두 경로로 온다.

1. **데이터 드리프트** — 입력 분포가 변한다(조명·렌즈·공정 변경). 모델을 건드리지 않아도
   성능이 떨어지며, 정답 라벨 없이도 감지할 수 있다.
2. **성능 드리프트** — 사후 검수로 확인된 정답과 모델 판정이 벌어진다. 정답이 필요하지만
   가장 직접적인 신호다.

둘을 나눠 보는 이유는 조치가 다르기 때문이다. 전자는 재학습·전처리 조정, 후자는 임계값
재조정이나 모델 교체로 이어진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate, labeling

INFERENCE_CSV = "inference_log.csv"

INFERENCE_COLUMNS: tuple[str, ...] = (
    "logged_at", "version", "image_id", "source", "category",
    "score", "threshold", "decision", "latency_ms", "note",
)

# PSI 해석 기준 (업계 통용값)
PSI_STABLE = 0.10
PSI_SHIFTED = 0.25

# PSI는 표본이 적으면 같은 분포에서도 크게 나온다. 측정해 보면 10구간 기준으로
# n=40일 때 동일 분포의 PSI 중앙값이 0.24(= "변화" 임계값 수준), n=20이면 1.0을 넘는다.
# 그래서 (1) 구간당 최소 표본을 확보하도록 구간을 합치고, (2) 그래도 부족하면 판정을 보류한다.
MIN_SAMPLES_PER_BIN = 30
MIN_DRIFT_SAMPLES = 60

LEVEL_INSUFFICIENT = "표본 부족"
DRIFT_LEVELS = ("안정", "주의", "변화", LEVEL_INSUFFICIENT)


def _log_path() -> Path:
    return config.ARTIFACT_ROOT / INFERENCE_CSV


# --- 추론 로그 --------------------------------------------------------------

def empty_log() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in INFERENCE_COLUMNS})


def load_log() -> pd.DataFrame:
    """추론 로그를 읽는다 (기록 순서 유지)."""
    path = _log_path()
    if not path.exists():
        return empty_log()
    df = pd.read_csv(path, dtype={"image_id": "str", "version": "str"})
    for column in INFERENCE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    for column in ("score", "threshold", "latency_ms"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce", utc=True)
    return df[list(INFERENCE_COLUMNS)]


def log_inference(records: list[dict], *, note: str = "") -> int:
    """추론 결과를 로그에 추가한다."""
    if not records:
        return 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    frame = pd.DataFrame([{**r, "logged_at": r.get("logged_at", now), "note": r.get("note", note)} for r in records])
    for column in INFERENCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[list(INFERENCE_COLUMNS)]

    config.ensure_dirs()
    path = _log_path()
    header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=header, index=False)
    return len(frame)


def clear_log() -> None:
    """추론 로그를 삭제한다."""
    _log_path().unlink(missing_ok=True)


def log_summary(log: pd.DataFrame) -> dict:
    """추론 로그 요약."""
    if log.empty:
        return {"total": 0, "versions": 0, "defect": 0, "defect_rate": 0.0, "latency_p95": 0.0}
    decisions = log["decision"].astype(str)
    defect = int((decisions == config.LABEL_DEFECT).sum())
    return {
        "total": len(log),
        "versions": int(log["version"].nunique()),
        "defect": defect,
        "defect_rate": defect / len(log),
        "latency_p95": float(np.nanpercentile(log["latency_ms"].astype(float), 95)),
    }


# --- 데이터 드리프트 --------------------------------------------------------

def psi_from_edges(
    edges: list[float],
    current: np.ndarray,
    *,
    expected: np.ndarray | None = None,
    epsilon: float = 1e-4,
) -> float:
    """기준선 구간 경계로 현재 분포의 PSI를 계산한다.

    `expected`를 생략하면 경계가 기준선의 **등분위**라고 보고 각 구간 비율을 1/구간수로 둔다.
    구간을 불균등하게 병합했다면 반드시 실제 기준 비율을 넘겨야 한다 — 균등으로 가정하면
    분포가 그대로인데도 PSI가 부풀려진다.

    반환값 해석: <0.10 안정, 0.10~0.25 주의, >0.25 유의미한 변화.
    """
    current = np.asarray(current, dtype=float)
    current = current[np.isfinite(current)]
    if current.size == 0:
        return float("nan")

    unique_edges = np.unique(np.asarray(edges, dtype=float))
    if unique_edges.size < 3:
        # 기준선에서 사실상 상수인 특징 — 분포 비교가 의미 없다
        return float("nan")

    # 양 끝을 열어 두어 범위를 벗어난 값도 첫/마지막 구간에 들어가게 한다
    bounded = unique_edges.copy()
    bounded[0], bounded[-1] = -np.inf, np.inf
    counts, _ = np.histogram(current, bins=bounded)
    actual = counts / counts.sum()

    if expected is None:
        expected = np.full(len(actual), 1.0 / len(actual))
    else:
        expected = np.asarray(expected, dtype=float)
        if expected.shape != actual.shape:
            raise ValueError(
                f"기준 비율 개수({expected.shape})가 구간 수({actual.shape})와 다릅니다."
            )

    actual = np.clip(actual, epsilon, None)
    expected = np.clip(expected, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def coarsen_edges(
    edges: list[float],
    n: int,
    *,
    props: list[float] | np.ndarray | None = None,
    min_per_bin: int = MIN_SAMPLES_PER_BIN,
) -> tuple[list[float], np.ndarray]:
    """표본 수에 맞춰 구간을 합치고, 병합된 구간의 **기준 비율**을 함께 반환한다.

    구간당 표본이 적으면 무작위 변동만으로 PSI가 커진다. 구간 수를 줄이면 그 과대추정이
    눌린다. 병합하면 기준 비율도 합쳐지므로 함께 계산해 돌려준다.

    Args:
        props: 기준선의 구간별 실제 비율. 생략하면 등분위로 가정한다 — 값에 동점이 많아
            분위 경계가 겹친 특징에서는 이 가정이 틀리므로 반드시 저장된 비율을 넘길 것.

    Returns:
        (병합된 경계, 각 구간의 기준 비율)
    """
    unique = list(np.unique(np.asarray(edges, dtype=float)))
    bins = len(unique) - 1
    if bins < 2:
        return unique, np.array([1.0])

    if props is not None and len(props) == bins:
        base_props = np.asarray(props, dtype=float)
    else:
        base_props = np.full(bins, 1.0 / bins)

    target = max(2, min(bins, n // max(min_per_bin, 1)))
    if target >= bins:
        return unique, base_props

    picks = sorted(set(np.linspace(0, bins, target + 1).round().astype(int).tolist()))
    merged = np.array([base_props[a:b].sum() for a, b in zip(picks[:-1], picks[1:])])
    return [unique[i] for i in picks], merged


def drift_level(psi: float) -> str:
    """PSI를 단계 라벨로 바꾼다. 계산 불가(NaN)는 '안정'이 아니라 판정 보류로 본다."""
    if not np.isfinite(psi):
        return LEVEL_INSUFFICIENT
    if psi > PSI_SHIFTED:
        return "변화"
    if psi > PSI_STABLE:
        return "주의"
    return "안정"


def feature_drift(baseline: dict, current: np.ndarray, feature_names: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """특징별 분포 이동을 표로 만든다.

    표본이 `MIN_DRIFT_SAMPLES`보다 적으면 PSI를 계산하지 않고 '표본 부족'으로 표시한다.
    소표본에서 나온 큰 PSI를 드리프트로 오해하면 불필요한 재학습을 유발한다.
    """
    current = np.asarray(current, dtype=float)
    edges = baseline.get("edges", {})
    all_props = baseline.get("props", {})
    stats = baseline.get("stats", {})
    n_current = current.shape[0]
    enough = n_current >= MIN_DRIFT_SAMPLES

    rows = []
    for index, name in enumerate(feature_names):
        if name not in edges or index >= current.shape[1]:
            continue
        column = current[:, index]
        if enough:
            merged_edges, proportions = coarsen_edges(
                edges[name], n_current, props=all_props.get(name) or None
            )
            psi = psi_from_edges(merged_edges, column, expected=proportions)
        else:
            psi = float("nan")
        base = stats.get(name, {})
        base_mean, base_std = base.get("mean", float("nan")), base.get("std", float("nan"))
        shift = (
            (float(column.mean()) - base_mean) / base_std
            if base_std and np.isfinite(base_std) and base_std > 0 else float("nan")
        )
        rows.append(
            {
                "feature": name,
                "psi": psi,
                "level": drift_level(psi),
                "baseline_mean": base_mean,
                "current_mean": float(column.mean()),
                "shift_sigma": shift,
                "n_current": n_current,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("psi", ascending=False, na_position="last").reset_index(drop=True)


def drift_summary(drift: pd.DataFrame) -> dict:
    """드리프트 표를 한 줄 요약으로 압축한다."""
    empty = {
        "checked": 0, "mean_psi": float("nan"), "shifted": 0, "watch": 0,
        "insufficient": 0, "n_current": 0, "level": LEVEL_INSUFFICIENT,
    }
    if drift.empty:
        return empty

    valid = drift["psi"].dropna()
    shifted = int((drift["level"] == "변화").sum())
    watch = int((drift["level"] == "주의").sum())
    insufficient = int((drift["level"] == LEVEL_INSUFFICIENT).sum())

    if insufficient == len(drift):
        level = LEVEL_INSUFFICIENT
    elif shifted:
        level = "변화"
    elif watch:
        level = "주의"
    else:
        level = "안정"
    return {
        "checked": int(len(drift)),
        "mean_psi": float(valid.mean()) if len(valid) else float("nan"),
        "shifted": shifted,
        "watch": watch,
        "insufficient": insufficient,
        "n_current": int(drift["n_current"].iloc[0]) if "n_current" in drift.columns else 0,
        "level": level,
    }


def score_drift(baseline: dict, scores: np.ndarray) -> dict:
    """결함 점수 분포가 기준선 대비 얼마나 이동했는지.

    입력 특징은 그대로인데 점수 분포만 이동하면 모델 판정 성향이 바뀐 것으로,
    임계값 재조정이 먼저 필요하다는 신호다.
    """
    reference = baseline.get("score")
    scores = np.asarray(scores, dtype=float)
    if not reference or scores.size == 0:
        return {"available": False}
    base_mean, base_std = reference.get("mean", float("nan")), reference.get("std", float("nan"))
    return {
        "available": True,
        "baseline_mean": base_mean,
        "current_mean": float(scores.mean()),
        "shift_sigma": (
            (float(scores.mean()) - base_mean) / base_std
            if base_std and np.isfinite(base_std) and base_std > 0 else float("nan")
        ),
    }


# --- 성능 드리프트 ----------------------------------------------------------

def feedback_frame(log: pd.DataFrame, resolved: pd.DataFrame | None = None) -> pd.DataFrame:
    """추론 로그에 사후 검수 정답을 붙인다.

    정답의 출처는 2단계 라벨 이력이다. 사람이 확인한(`verified`) 라벨만 정답으로 쓴다 —
    폴더 구조에서 추론한 라벨을 정답으로 쓰면 자기 채점이 된다.
    """
    if log.empty:
        return pd.DataFrame()
    resolved = labeling.resolve() if resolved is None else resolved
    if resolved.empty:
        return pd.DataFrame()

    truth = resolved[
        (resolved["label_source"].astype(str) == "human")
        & (resolved["verified"].fillna(False).astype(bool))
        & (resolved["label"].astype(str).isin([config.LABEL_NORMAL, config.LABEL_DEFECT]))
    ]
    if truth.empty:
        return pd.DataFrame()

    merged = log.merge(
        truth[["image_id", "label", "category"]].rename(columns={"category": "truth_category"}),
        on="image_id", how="inner",
    )
    if merged.empty:
        return merged

    merged["y_true"] = (merged["label"].astype(str) == config.LABEL_DEFECT).astype(int)
    merged["y_pred"] = (merged["decision"].astype(str) == config.LABEL_DEFECT).astype(int)
    merged["outcome"] = np.where(
        (merged["y_pred"] == 1) & (merged["y_true"] == 1), "TP",
        np.where((merged["y_pred"] == 1) & (merged["y_true"] == 0), "FP",
                 np.where((merged["y_pred"] == 0) & (merged["y_true"] == 1), "FN", "TN")),
    )
    return merged


def performance_metrics(feedback: pd.DataFrame) -> dict | None:
    """사후 검수로 측정한 실제 성능."""
    if feedback.empty:
        return None
    tp = int(((feedback["y_pred"] == 1) & (feedback["y_true"] == 1)).sum())
    fp = int(((feedback["y_pred"] == 1) & (feedback["y_true"] == 0)).sum())
    fn = int(((feedback["y_pred"] == 0) & (feedback["y_true"] == 1)).sum())
    tn = int(((feedback["y_pred"] == 0) & (feedback["y_true"] == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    result = {
        "n": len(feedback), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision,
    }
    if feedback["y_true"].nunique() == 2:
        result["auroc"] = evaluate.auroc(feedback["y_true"], feedback["score"].astype(float))
    return result


def performance_by_version(feedback: pd.DataFrame) -> pd.DataFrame:
    """버전별 실측 성능."""
    if feedback.empty:
        return pd.DataFrame()
    rows = []
    for version, group in feedback.groupby("version"):
        metrics = performance_metrics(group)
        if metrics:
            rows.append({"version": version, **metrics})
    return pd.DataFrame(rows)


def performance_trend(feedback: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """기간별 재현율·정밀도 추이."""
    if feedback.empty or feedback["logged_at"].isna().all():
        return pd.DataFrame()
    frame = feedback.dropna(subset=["logged_at"]).copy()
    frame["bucket"] = frame["logged_at"].dt.floor(freq)
    rows = []
    for bucket, group in frame.groupby("bucket"):
        metrics = performance_metrics(group)
        if metrics:
            rows.append({"bucket": bucket, **metrics})
    return pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)


# --- 재학습 판단 ------------------------------------------------------------

LEVEL_OK, LEVEL_WARN, LEVEL_ALERT = "ok", "warn", "alert"

LEVEL_ICON = {LEVEL_OK: "✅", LEVEL_WARN: "⚠️", LEVEL_ALERT: "🚨"}


@dataclass
class Signal:
    """재학습 판단 근거 하나."""

    key: str
    level: str
    title: str
    detail: str

    @property
    def icon(self) -> str:
        return LEVEL_ICON.get(self.level, "•")


@dataclass
class RetrainingDecision:
    """재학습 제안 여부와 근거."""

    recommended: bool
    signals: list[Signal]

    @property
    def alerts(self) -> list[Signal]:
        return [s for s in self.signals if s.level == LEVEL_ALERT]

    @property
    def warnings(self) -> list[Signal]:
        return [s for s in self.signals if s.level == LEVEL_WARN]


def retraining_signals(
    *,
    production=None,
    log: pd.DataFrame | None = None,
    feedback: pd.DataFrame | None = None,
    drift: pd.DataFrame | None = None,
    new_labels: int = 0,
    new_label_threshold: int = 50,
    recall_margin: float = 0.05,
) -> RetrainingDecision:
    """운영 신호를 모아 재학습 필요 여부를 판단한다.

    자동 배포가 아니라 **사람이 승인하는 제안**을 만드는 것이 목적이다.
    """
    signals: list[Signal] = []

    if production is None:
        signals.append(Signal(
            "no_production", LEVEL_ALERT, "서비스 중 모델 없음",
            "레지스트리에서 버전을 하나 승격해야 운영 감시가 시작된다.",
        ))
        return RetrainingDecision(recommended=False, signals=signals)

    signals.append(Signal(
        "production", LEVEL_OK, f"서비스 중: {production['version']}",
        f"{production.get('kind', '')} / {production.get('model', '')} · "
        f"승격 {production.get('promoted_at', '—')}",
    ))

    # 1) 추론 로그 축적
    log_count = 0 if log is None or log.empty else len(log)
    if log_count == 0:
        signals.append(Signal(
            "no_log", LEVEL_WARN, "추론 로그 없음",
            "배치 추론을 한 번도 돌리지 않았다. 로그가 없으면 성능 저하를 감지할 근거가 없다.",
        ))
    else:
        signals.append(Signal("log", LEVEL_OK, f"추론 로그 {log_count:,}건", "감시 근거가 축적되고 있다."))

    # 2) 성능 드리프트
    measured = performance_metrics(feedback) if feedback is not None and not feedback.empty else None
    if measured is None:
        signals.append(Signal(
            "no_feedback", LEVEL_WARN, "사후 검수 정답 없음",
            "2단계에서 사람이 확인한 라벨이 있어야 실제 성능을 측정할 수 있다.",
        ))
    else:
        baseline_recall = production.get("recall")
        baseline_recall = float(baseline_recall) if pd.notna(baseline_recall) else None
        actual = measured["recall"]
        if baseline_recall is not None and np.isfinite(actual):
            gap = baseline_recall - actual
            if gap > recall_margin:
                signals.append(Signal(
                    "recall_drop", LEVEL_ALERT, "재현율 저하",
                    f"등록 시 {baseline_recall:.3f} → 실측 {actual:.3f} "
                    f"(허용 낙폭 {recall_margin:.2f} 초과). 임계값 재조정 또는 재학습이 필요하다.",
                ))
            else:
                signals.append(Signal(
                    "recall_ok", LEVEL_OK, "재현율 유지",
                    f"등록 시 {baseline_recall:.3f} → 실측 {actual:.3f} (표본 {measured['n']}건)",
                ))
        if measured["fn"] > 0:
            signals.append(Signal(
                "miss", LEVEL_ALERT, f"미탐 {measured['fn']}건",
                "결함을 놓친 사례가 있다. 개별 리뷰가 필요하다 — 이 프로젝트에서 가장 큰 리스크다.",
            ))

    # 3) 데이터 드리프트
    if drift is None or drift.empty:
        signals.append(Signal(
            "no_drift", LEVEL_WARN, "드리프트 기준선 없음",
            "승격 시 기준선을 저장하지 않았거나 추론 특징이 없어 입력 분포를 비교할 수 없다.",
        ))
    else:
        summary = drift_summary(drift)
        if summary["level"] == LEVEL_INSUFFICIENT:
            signals.append(Signal(
                "drift_insufficient", LEVEL_WARN, "드리프트 판정 보류 (표본 부족)",
                f"추론 표본이 {summary['n_current']}건으로 {MIN_DRIFT_SAMPLES}건 미만이다. "
                "소표본에서는 같은 분포에서도 PSI가 크게 나와 오경보가 된다. 추론을 더 돌린 뒤 판단한다.",
            ))
        elif summary["level"] == "변화":
            top = drift.iloc[0]
            signals.append(Signal(
                "data_drift", LEVEL_ALERT, f"입력 분포 변화 (특징 {summary['shifted']}개)",
                f"PSI가 {PSI_SHIFTED} 초과인 특징이 있다. 최상위: `{top['feature']}` "
                f"(PSI {top['psi']:.3f}). 촬영 환경이나 공정이 바뀌었는지 먼저 확인한다.",
            ))
        elif summary["level"] == "주의":
            signals.append(Signal(
                "data_drift_watch", LEVEL_WARN, f"입력 분포 주의 (특징 {summary['watch']}개)",
                f"PSI {PSI_STABLE}~{PSI_SHIFTED} 구간. 아직 조치할 정도는 아니지만 추이를 본다.",
            ))
        else:
            signals.append(Signal(
                "data_drift_ok", LEVEL_OK, "입력 분포 안정",
                f"검사한 특징 {summary['checked']}개 모두 PSI {PSI_STABLE} 이하.",
            ))

    # 4) 신규 라벨 축적
    if new_labels >= new_label_threshold:
        signals.append(Signal(
            "new_labels", LEVEL_WARN, f"신규 라벨 {new_labels:,}건",
            f"승격 이후 {new_label_threshold}건 이상 쌓였다. 재학습하면 성능이 오를 여지가 있다.",
        ))
    elif new_labels:
        signals.append(Signal(
            "new_labels_ok", LEVEL_OK, f"신규 라벨 {new_labels:,}건",
            f"재학습 기준({new_label_threshold}건)에는 아직 못 미친다.",
        ))

    recommended = any(s.level == LEVEL_ALERT for s in signals) or any(
        s.key == "new_labels" for s in signals
    )
    return RetrainingDecision(recommended=recommended, signals=signals)


def new_labels_since(promoted_at, events: pd.DataFrame | None = None) -> int:
    """승격 시점 이후 사람이 남긴 라벨 이벤트 수.

    타임스탬프가 초 단위라 승격과 **같은 초**에 기록된 라벨이 생길 수 있다. 경계를
    포함(`>=`)해 세는 이유는, 빠뜨리면 재학습 시점이 늦어지기 때문이다 — 이 프로젝트에서는
    한 건을 더 세는 쪽이 한 건을 놓치는 쪽보다 안전하다.
    """
    events = labeling.load_events() if events is None else events
    if events.empty or promoted_at is None or pd.isna(promoted_at):
        return 0
    timestamps = pd.to_datetime(events["labeled_at"], errors="coerce", utc=True)
    cutoff = pd.to_datetime(promoted_at, errors="coerce", utc=True)
    if pd.isna(cutoff):
        return 0
    return int((timestamps >= cutoff).sum())


# --- 판정 이력 조회 ---------------------------------------------------------

def trace_image(image_id: str) -> dict:
    """이미지 하나의 전 과정을 모아 온다.

    "이 이미지는 왜 그렇게 판정됐나"를 되짚기 위한 뷰. 1~3단계 산출물이 모두 image_id로
    연결되어 있어 이렇게 합칠 수 있다.
    """
    from . import claude_review, storage

    image_id = str(image_id)
    manifest = storage.load_manifest()
    record = manifest[manifest["image_id"].astype(str) == image_id]

    events = labeling.load_events()
    events = events[events["image_id"].astype(str) == image_id] if not events.empty else events

    log = load_log()
    log = log[log["image_id"].astype(str) == image_id] if not log.empty else log

    reviews = claude_review.load_reviews()
    reviews = reviews[reviews["image_id"].astype(str) == image_id] if not reviews.empty else reviews

    resolved = labeling.resolve()
    effective = resolved[resolved["image_id"].astype(str) == image_id] if not resolved.empty else resolved

    return {
        "image_id": image_id,
        "manifest": None if record.empty else record.iloc[0].to_dict(),
        "effective": None if effective.empty else effective.iloc[0].to_dict(),
        "label_events": events,
        "inferences": log,
        "claude_reviews": reviews,
    }
