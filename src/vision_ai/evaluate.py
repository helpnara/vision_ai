"""3단계 평가 지표.

이 프로젝트는 **미탐(결함을 놓침)** 을 가장 큰 리스크로 본다. 따라서 단일 정확도가 아니라
재현율 중심으로 보고, 임계값을 바꿔가며 미탐/오탐 트레이드오프를 직접 보게 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"라벨과 점수의 길이가 다릅니다: {y.shape} vs {s.shape}")
    return y, s


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    """임계값 하나에서의 혼동행렬과 지표를 계산한다 (1=결함)."""
    y, s = _as_arrays(y_true, scores)
    predicted = (s >= threshold).astype(int)

    tp = int(((predicted == 1) & (y == 1)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total = tp + fp + fn + tn

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "accuracy": (tp + tn) / total if total else 0.0,
        "miss_rate": fn / (tp + fn) if (tp + fn) else 0.0,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) else 0.0,
    }


def auroc(y_true, scores) -> float:
    """ROC 곡선 아래 면적. 한 클래스만 있으면 NaN."""
    y, s = _as_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def average_precision(y_true, scores) -> float:
    """PR 곡선 기반 평균 정밀도. 결함이 드물 때 AUROC보다 민감하다."""
    y, s = _as_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def roc_curve_frame(y_true, scores) -> pd.DataFrame:
    """ROC 곡선을 DataFrame으로 반환한다."""
    y, s = _as_arrays(y_true, scores)
    if len(np.unique(y)) < 2:
        return pd.DataFrame(columns=["fpr", "tpr", "threshold"])
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y, s)
    return pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})


def threshold_sweep(y_true, scores, steps: int = 60) -> pd.DataFrame:
    """임계값을 훑어 지표 변화를 표로 만든다."""
    y, s = _as_arrays(y_true, scores)
    if s.size == 0:
        return pd.DataFrame()
    candidates = np.unique(np.quantile(s, np.linspace(0.0, 1.0, min(steps, max(s.size, 2)))))
    rows = [metrics_at_threshold(y, s, float(t)) for t in candidates]
    return pd.DataFrame(rows)


def threshold_for_target_recall(y_true, scores, target_recall: float = 0.95) -> float | None:
    """목표 재현율을 만족하는 임계값 중 오탐이 가장 적은 값을 찾는다.

    미탐 우선 정책이므로, 재현율 하한을 먼저 만족시키고 그 안에서 오탐을 줄이는 순서다.
    """
    sweep = threshold_sweep(y_true, scores)
    if sweep.empty:
        return None
    feasible = sweep[sweep["recall"] >= target_recall]
    if feasible.empty:
        return None
    best = feasible.sort_values(["fp", "threshold"], ascending=[True, False]).iloc[0]
    return float(best["threshold"])


def summarize(y_true, scores, threshold: float) -> dict:
    """임계값 지표 + 임계값과 무관한 지표를 함께 요약한다."""
    result = metrics_at_threshold(y_true, scores, threshold)
    result["auroc"] = auroc(y_true, scores)
    result["average_precision"] = average_precision(y_true, scores)
    return result


def error_frame(
    image_ids, y_true, scores, threshold: float, extra: pd.DataFrame | None = None
) -> pd.DataFrame:
    """이미지별 판정 결과를 오류 종류(TP/FP/FN/TN)와 함께 반환한다."""
    y, s = _as_arrays(y_true, scores)
    predicted = (s >= threshold).astype(int)
    kinds = np.where(
        (predicted == 1) & (y == 1), "TP",
        np.where((predicted == 1) & (y == 0), "FP",
                 np.where((predicted == 0) & (y == 1), "FN", "TN")),
    )
    frame = pd.DataFrame(
        {
            "image_id": [str(i) for i in image_ids],
            "label": y,
            "score": s,
            "predicted": predicted,
            "outcome": kinds,
        }
    )
    if extra is not None and not extra.empty:
        frame = frame.merge(extra, on="image_id", how="left")
    return frame


# --- 위치 추정 평가 --------------------------------------------------------

def localization_metrics(score_map: np.ndarray, mask: np.ndarray, quantile: float = 0.99) -> dict:
    """히트맵이 실제 결함 위치를 얼마나 맞혔는지 평가한다.

    - hit   : 히트맵 최고점이 정답 마스크 안에 있는가 (가장 해석하기 쉬운 지표)
    - iou   : 상위 분위 영역과 정답 마스크의 IoU
    - pixel_auroc : 픽셀 단위 AUROC
    """
    import cv2

    score_map = np.asarray(score_map, dtype=np.float32)
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    if mask.shape != score_map.shape:
        mask = cv2.resize(
            mask.astype(np.uint8), (score_map.shape[1], score_map.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    truth = mask > 0
    if not truth.any():
        return {"hit": False, "iou": float("nan"), "pixel_auroc": float("nan")}

    peak = np.unravel_index(int(np.argmax(score_map)), score_map.shape)
    hit = bool(truth[peak])

    cutoff = float(np.quantile(score_map, quantile))
    predicted = score_map >= cutoff
    union = int((predicted | truth).sum())
    iou = float((predicted & truth).sum() / union) if union else float("nan")

    flat_truth = truth.ravel().astype(int)
    pixel_auroc = float("nan")
    if len(np.unique(flat_truth)) == 2:
        from sklearn.metrics import roc_auc_score

        pixel_auroc = float(roc_auc_score(flat_truth, score_map.ravel()))

    return {"hit": hit, "iou": iou, "pixel_auroc": pixel_auroc}
