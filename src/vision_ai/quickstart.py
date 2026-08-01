"""빠른 시작: 한 번에 데모 상태까지 만든다.

## 왜 필요한가

이 앱을 처음 연 사람이 운영 화면까지 보려면 1단계 안쪽 탭 → 2단계 분할 → 3단계 학습 →
4단계 등록·승격을 순서대로 찾아 들어가야 한다. 배포본을 시연용으로 여는 사람에게는
그 자체가 장벽이다.

여기서는 **이미 있는 기능들을 순서대로 호출할 뿐** 새로운 로직을 만들지 않는다.
합성 샘플 생성 → 등록 → 분할 → 이상탐지 학습 → 실행 기록 → 레지스트리 승격까지 한 번에
밟아, 4단계 '운영 시나리오 시연'을 곧바로 누를 수 있는 상태로 만든다.

## 왜 이상탐지인가

합성 데이터는 정상이 결함보다 많고, 정상만으로 학습하는 방식이 이 파이프라인의 주력이다.
지도학습 베이스라인은 결함 예시가 충분해야 하므로 빠른 시작에는 맞지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from . import (
    config,
    experiments,
    features,
    ingest,
    labeling,
    models,
    registry,
    storage,
    viz,
)

ProgressCallback = Callable[[int, int, str], None]

# 이상탐지가 위치별 분포를 학습하려면 정상이 특징 차원보다 넉넉해야 한다.
# 카테고리 6종 × 이 값이 실제 정상 장수가 된다.
DEFAULT_NORMAL = 30
DEFAULT_DEFECT = 10

STEPS = ("합성 샘플 생성", "manifest 등록", "데이터 분할", "모델 학습", "레지스트리 등록·승격")


@dataclass
class QuickstartResult:
    version: str | None = None
    n_images: int = 0
    n_train: int = 0
    threshold: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.version is not None


def run(
    *,
    n_normal: int = DEFAULT_NORMAL,
    n_defect: int = DEFAULT_DEFECT,
    seed: int = 42,
    progress: ProgressCallback | None = None,
) -> QuickstartResult:
    """데모 한 바퀴를 만든다. 실패해도 예외 대신 경고로 돌려준다.

    시연 중에 화면이 멈추는 것보다 **어디까지 됐고 무엇이 부족한지** 보이는 편이 낫다.
    """
    result = QuickstartResult()
    total = len(STEPS)

    def step(index: int) -> None:
        if progress is not None:
            progress(index, total, STEPS[index - 1])

    config.ensure_dirs()

    step(1)
    folder = ingest.generate_synthetic(
        n_normal=n_normal, n_defect=n_defect, layout="visa", seed=seed
    )

    step(2)
    ingest.ingest_folder(Path(folder), source="synthetic", layout="visa")
    manifest = storage.load_manifest()
    result.n_images = len(manifest)
    if manifest.empty:
        result.warnings.append("합성 샘플을 만들지 못했습니다.")
        return result

    step(3)
    mapping = labeling.assign_splits(labeling.resolve(), train=0.6, val=0.2, test=0.2, seed=seed)
    if not mapping:
        result.warnings.append("분할할 대상이 없습니다.")
        return result
    labeling.save_splits(mapping)

    step(4)
    resolved = labeling.resolve()
    train = resolved[
        (resolved["split"].astype(str) == config.SPLIT_TRAIN)
        & (resolved["label"].astype(str) == config.LABEL_NORMAL)
    ]
    images = [viz.load_rgb(str(storage.resolve_path(p))) for p in train["path"]]
    images = [image for image in images if image is not None]
    if len(images) <= features.PATCH_FEATURE_DIM:
        result.warnings.append(
            f"학습용 정상 이미지가 {len(images)}장뿐이라 위치별 분포를 학습할 수 없습니다. "
            "생성 장수를 늘려 다시 시도하세요."
        )
        return result

    model = models.PatchAnomalyModel(models.AnomalyConfig(per_position=True)).fit(images)
    artifact = config.MODEL_DIR / "anomaly_patch.npz"
    model.save(artifact)

    # 평가 분할로 임계값을 정한다. 학습 데이터로 정하면 실제보다 관대해진다.
    threshold, metrics = _evaluate(model, resolved)
    result.threshold = threshold
    result.n_train = len(images)

    run_id = experiments.record_run(
        kind="anomaly", model="mahalanobis", metrics=metrics,
        settings={"per_position": True, "n_train": len(images), "target_recall": 0.95},
        n_train=len(images), note="빠른 시작", artifacts={"model": str(artifact)},
    )

    step(5)
    matrix = np.stack([features.image_features(image) for image in images])
    baseline = registry.make_baseline(matrix, features.FEATURE_NAMES)
    registered = registry.register(run_id, baseline=baseline, note="빠른 시작", promote_now=True)
    result.version = registered.version
    result.warnings.extend(registered.warnings)
    return result


def _evaluate(model: models.PatchAnomalyModel, resolved) -> tuple[float, dict]:
    """평가 분할에서 점수를 내고 목표 재현율 기준으로 임계값을 고른다."""
    from . import evaluate as evaluate_module

    evaluation = resolved[resolved["split"].astype(str) == config.SPLIT_TEST]
    scores, truth = [], []
    for _, row in evaluation.iterrows():
        image = viz.load_rgb(str(storage.resolve_path(row["path"])))
        if image is None:
            continue
        scores.append(model.image_score(image))
        truth.append(1 if str(row["label"]) == config.LABEL_DEFECT else 0)

    if not scores or len(set(truth)) < 2:
        return float("nan"), {}

    scores_array = np.asarray(scores, dtype=float)
    truth_array = np.asarray(truth, dtype=int)
    threshold = evaluate_module.threshold_for_target_recall(truth_array, scores_array, 0.95)
    if threshold is None:
        threshold = float(np.median(scores_array))
    metrics = evaluate_module.summarize(truth_array, scores_array, threshold)
    return float(threshold), metrics
