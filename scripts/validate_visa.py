"""VisA 실데이터 검증 — 공식 1-class 분할로 이상탐지 성능을 측정한다.

지금까지의 모든 수치는 합성 데이터 기준이라 성능 근거가 되지 못했다. 이 스크립트는
실제 VisA PCB 데이터로 같은 파이프라인(`vision_ai.features` → `vision_ai.models` →
`vision_ai.evaluate`)을 돌려 그 수치를 대체한다.

## 평가 설계

VisA 공식 분할(`split_csv/1cls.csv`)은 train/test 두 가지뿐이고, train에는 정상만 있다.
1-class 이상탐지 프로토콜이므로 우리 `PatchAnomalyModel`(정상만으로 학습)과 그대로 맞는다.

문제는 **임계값을 고를 데이터가 없다**는 것이다. test로 임계값을 고르고 test로 평가하면
같은 데이터를 두 번 쓰는 것이라 성능이 부풀려진다. 그래서 공식 test를 라벨 비율을 유지한 채
절반으로 나눠, 앞쪽 절반에서 임계값을 고르고 뒤쪽 절반에서만 임계값 기반 지표를 보고한다.

- **AUROC / AP** — 공식 test 전체 기준. 임계값과 무관하므로 벤치마크와 직접 비교할 수 있다.
- **재현율 / 정밀도 / F1** — 임계값이 필요하므로 위에서 나눈 홀드아웃 절반에서만 측정한다.

## 실행

    PYTHONPATH=src python scripts/validate_visa.py
    PYTHONPATH=src python scripts/validate_visa.py --categories pcb1 --limit-train 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_ai import evaluate, models, viz  # noqa: E402

TARGET_RECALL = 0.95


def load_official_split(csv_path: Path, categories: list[str]) -> pd.DataFrame:
    """공식 분할 CSV에서 대상 카테고리만 뽑는다."""
    df = pd.read_csv(csv_path)
    df = df[df["object"].astype(str).isin(categories)].copy()
    df["y"] = (df["label"].astype(str).str.lower() == "anomaly").astype(int)
    return df


def score_images(model: models.PatchAnomalyModel, paths: list[Path]) -> tuple[np.ndarray, list[int]]:
    """이미지별 이상 점수. 읽기 실패한 항목의 인덱스는 제외한다."""
    scores: list[float] = []
    kept: list[int] = []
    for index, path in enumerate(paths):
        image = viz.load_rgb(str(path))
        if image is None:
            continue
        scores.append(model.image_score(image))
        kept.append(index)
    return np.asarray(scores, dtype=float), kept


def holdout_halves(y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """라벨 비율을 유지하며 절반으로 나눈다 (임계값 선택용 / 최종 보고용)."""
    rng = np.random.default_rng(seed)
    pick = np.zeros(len(y), dtype=bool)
    for value in np.unique(y):
        index = np.flatnonzero(y == value)
        rng.shuffle(index)
        pick[index[: len(index) // 2]] = True
    return pick, ~pick


def localization(model: models.PatchAnomalyModel, root: Path, rows: pd.DataFrame, limit: int) -> dict:
    """마스크가 있는 결함 이미지에서 히트맵이 실제 위치를 맞히는지 측정한다."""
    import cv2

    hits, ious, pixel_aurocs, areas = [], [], [], []
    subset = rows[rows["mask"].notna()].head(limit)
    for _, row in subset.iterrows():
        image = viz.load_rgb(str(root / row["image"]))
        mask = cv2.imread(str(root / row["mask"]), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue
        result = evaluate.localization_metrics(model.score_map(image), mask)
        hits.append(bool(result["hit"]))
        areas.append(float((mask > 0).mean()))
        if not np.isnan(result["iou"]):
            ious.append(float(result["iou"]))
        if not np.isnan(result["pixel_auroc"]):
            pixel_aurocs.append(float(result["pixel_auroc"]))
    if not hits:
        return {}
    # 적중률만 보면 낮아 보이지만, VisA PCB 결함은 이미지의 1% 미만인 미세 결함이다.
    # 아무 데나 찍었을 때의 적중 확률(= 결함 면적 비율)과 견줘야 의미가 있다.
    baseline = float(np.mean(areas)) if areas else float("nan")
    hit_rate = float(np.mean(hits))
    return {
        "n": len(hits),
        "hit_rate": hit_rate,
        "mask_area_fraction": baseline,
        "hit_rate_vs_random": (hit_rate / baseline) if baseline else float("nan"),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "pixel_auroc_mean": float(np.mean(pixel_aurocs)) if pixel_aurocs else float("nan"),
    }


def run_category(root: Path, split: pd.DataFrame, category: str, args) -> dict:
    rows = split[split["object"] == category]
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"] == "test"].reset_index(drop=True)

    if args.limit_train:
        train_rows = train_rows.head(args.limit_train)

    started = time.time()
    train_images = []
    for path in train_rows["image"]:
        image = viz.load_rgb(str(root / path))
        if image is not None:
            train_images.append(image)

    config = models.AnomalyConfig(per_position=args.per_position, backend=args.backend)
    model = models.PatchAnomalyModel(config).fit(train_images)
    fit_seconds = time.time() - started

    test_paths = [root / p for p in test_rows["image"]]
    scores, kept = score_images(model, test_paths)
    y = test_rows["y"].to_numpy()[kept]

    auroc = evaluate.auroc(y, scores)
    ap = evaluate.average_precision(y, scores)

    # 임계값은 절반에서만 고르고, 나머지 절반에서 보고한다 (같은 데이터 재사용 방지)
    pick_val, pick_test = holdout_halves(y)
    threshold = evaluate.threshold_for_target_recall(y[pick_val], scores[pick_val], TARGET_RECALL)
    if threshold is None:
        held_out = {"note": f"검증 절반에서 재현율 {TARGET_RECALL:.0%}를 만족하는 임계값이 없습니다."}
    else:
        held_out = evaluate.metrics_at_threshold(y[pick_test], scores[pick_test], threshold)
        held_out["threshold"] = float(threshold)

    return {
        "category": category,
        "n_train": len(train_images),
        "n_test": int(len(y)),
        "n_test_defect": int(y.sum()),
        "fit_seconds": round(fit_seconds, 1),
        "auroc": auroc,
        "average_precision": ap,
        "held_out": held_out,
        "localization": localization(model, root, test_rows, args.localization_limit),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="VisA 실데이터 검증")
    parser.add_argument("--root", default="data/visa", help="VisA 압축 해제 루트")
    parser.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    parser.add_argument("--limit-train", type=int, default=0, help="카테고리별 학습 이미지 상한 (0=전량)")
    parser.add_argument("--localization-limit", type=int, default=40)
    parser.add_argument("--per-position", action="store_true", default=True)
    parser.add_argument("--pooled", dest="per_position", action="store_false")
    parser.add_argument(
        "--backend", default=models.BACKEND_CLASSIC, choices=list(models.BACKENDS),
        help="특징 추출 방식. cnn은 사전학습 모델 파일이 있어야 한다.",
    )
    parser.add_argument("--out", default="artifacts/reports/visa_validation.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    split = load_official_split(root / "split_csv" / "1cls.csv", args.categories)
    if split.empty:
        print(f"대상 카테고리를 찾을 수 없습니다: {args.categories}", file=sys.stderr)
        return 1

    results = []
    for category in args.categories:
        print(f"\n=== {category} ===", flush=True)
        result = run_category(root, split, category, args)
        results.append(result)
        held = result["held_out"]
        print(
            f"  AUROC {result['auroc']:.4f} / AP {result['average_precision']:.4f}"
            f" / 학습 {result['n_train']}장 {result['fit_seconds']}s",
            flush=True,
        )
        if "recall" in held:
            print(
                f"  홀드아웃 임계값 {held['threshold']:.3f} → "
                f"재현율 {held['recall']:.3f} 정밀도 {held['precision']:.3f} F1 {held['f1']:.3f}",
                flush=True,
            )
        if result["localization"]:
            loc = result["localization"]
            print(
                f"  위치 적중률 {loc['hit_rate']:.3f} (무작위 대비 {loc['hit_rate_vs_random']:.0f}배)"
                f" / 픽셀 AUROC {loc['pixel_auroc_mean']:.4f} (n={loc['n']})",
                flush=True,
            )

    summary = {
        "dataset": "VisA (CC BY 4.0)",
        "protocol": "공식 1cls 분할 — train은 정상만, test는 정상+결함",
        "threshold_policy": f"검증 절반에서 목표 재현율 {TARGET_RECALL:.0%}, 나머지 절반에서 보고",
        "config": {"per_position": args.per_position, "limit_train": args.limit_train,
                   "backend": args.backend},
        "categories": results,
        "mean_auroc": float(np.mean([r["auroc"] for r in results])),
        "mean_average_precision": float(np.mean([r["average_precision"] for r in results])),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n평균 AUROC {summary['mean_auroc']:.4f} / 평균 AP {summary['mean_average_precision']:.4f}")
    print(f"저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
