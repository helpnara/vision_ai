"""카테고리별 성능 차이의 원인을 진단한다.

`validate_visa.py`는 "얼마나 잘 맞히는가"를 알려주지만 "왜 이 카테고리만 낮은가"는
알려주지 않는다. 이 스크립트는 그 차이를 만드는 두 가지를 재 본다.

1. **결함 크기** — 결함이 이미지에서 차지하는 면적. 작을수록 어렵다.
2. **점수 분포** — 정상/결함 각각의 이상 점수 평균과 산포, 그리고 둘의 분리도(Cohen's d).
   AUROC가 낮을 때 원인이 "정상을 잘 못 맞춰서"인지 "결함이 제각각이라서"인지 구분된다.

실행:
    PYTHONPATH=src python scripts/diagnose_visa.py
    PYTHONPATH=src python scripts/diagnose_visa.py --categories pcb3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_ai import models, viz  # noqa: E402


def mask_areas(root: Path, rows: pd.DataFrame) -> np.ndarray:
    """결함 마스크가 이미지에서 차지하는 면적 비율."""
    import cv2

    values = []
    for path in rows[rows["mask"].notna()]["mask"]:
        mask = cv2.imread(str(root / path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            values.append(float((mask > 0).mean()))
    return np.asarray(values)


def score_split(root: Path, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """학습(정상)으로 모델을 만들고 test 점수와 정답을 돌려준다."""
    train = [viz.load_rgb(str(root / p)) for p in rows[rows["split"] == "train"]["image"]]
    train = [image for image in train if image is not None]
    model = models.PatchAnomalyModel(models.AnomalyConfig(per_position=True)).fit(train)

    scores, labels = [], []
    for _, row in rows[rows["split"] == "test"].iterrows():
        image = viz.load_rgb(str(root / row["image"]))
        if image is None:
            continue
        scores.append(model.image_score(image))
        labels.append(1 if str(row["label"]).lower() == "anomaly" else 0)
    return np.asarray(scores), np.asarray(labels)


def main() -> int:
    parser = argparse.ArgumentParser(description="VisA 카테고리별 원인 진단")
    parser.add_argument("--root", default="data/visa")
    parser.add_argument("--categories", nargs="+", default=["pcb1", "pcb2", "pcb3", "pcb4"])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    split = pd.read_csv(root / "split_csv" / "1cls.csv")

    print("## 결함 크기 (이미지 대비 %)\n")
    print(f"{'카테고리':<10}{'중앙값':>10}{'평균':>10}{'최소':>10}")
    for category in args.categories:
        area = mask_areas(root, split[split["object"] == category]) * 100
        if area.size:
            print(f"{category:<10}{np.median(area):>10.3f}{area.mean():>10.3f}{area.min():>10.3f}")

    print("\n## 이상 점수 분포\n")
    print(f"{'카테고리':<10}{'정상 평균':>10}{'정상 std':>10}{'결함 평균':>10}{'결함 std':>10}{'분리도':>8}")
    for category in args.categories:
        scores, labels = score_split(root, split[split["object"] == category])
        normal, defect = scores[labels == 0], scores[labels == 1]
        if not normal.size or not defect.size:
            continue
        pooled = np.sqrt((normal.var() + defect.var()) / 2)
        cohen_d = (defect.mean() - normal.mean()) / pooled if pooled else float("nan")
        print(
            f"{category:<10}{normal.mean():>10.3f}{normal.std():>10.3f}"
            f"{defect.mean():>10.3f}{defect.std():>10.3f}{cohen_d:>8.3f}"
        )

    print(
        "\n분리도(Cohen's d)가 낮은데 **결함 std가 크다면**, 모델이 못 잡는 결함이 섞여 있다는 뜻이다.\n"
        "정상 평균/std가 높다면 정상 자체를 느슨하게 학습한 것이므로 학습 데이터를 먼저 본다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
