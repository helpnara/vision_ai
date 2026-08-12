"""4단계: 등록된 모델로 배치 추론.

3단계 학습 코드와 분리한 이유는 운영에서 쓰는 경로가 다르기 때문이다. 운영에서는
"레지스트리에 등록된 버전"만 불러 쓰고, 학습 설정을 다시 고르지 않는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config, features, models, registry, viz

ProgressCallback = Callable[[int, int], None]


@dataclass
class LoadedModel:
    """추론에 쓸 수 있게 준비된 모델."""

    version: str
    kind: str
    threshold: float
    thresholds: dict[str, float] = field(default_factory=dict)
    baseline: models.BaselineModel | None = None
    anomaly: models.PatchAnomalyModel | None = None

    def threshold_for(self, category) -> float:
        """이 카테고리에 쓸 임계값.

        카테고리별 값이 없으면 전체 기준값을 쓴다. **운영 중에 새 제품이 들어오면** 그
        카테고리는 학습할 때 없던 것이므로, 판정을 거부하는 대신 전체 기준으로 처리한다.
        """
        return float(self.thresholds.get(str(category), self.threshold))

    def score_image(self, rgb: np.ndarray) -> float:
        """이미지 1장의 결함 점수."""
        if self.kind == "baseline":
            if self.baseline is None:
                raise RuntimeError("베이스라인 모델이 로드되지 않았습니다.")
            return float(self.baseline.score(features.image_features(rgb)[None, :])[0])
        if self.anomaly is None:
            raise RuntimeError("이상탐지 모델이 로드되지 않았습니다.")
        return float(self.anomaly.image_score(rgb))

    def score_map(self, rgb: np.ndarray) -> np.ndarray | None:
        """결함 위치 히트맵. 이상탐지 모델만 지원한다."""
        if self.kind != "anomaly" or self.anomaly is None:
            return None
        return self.anomaly.score_map(features.preprocess(rgb))


def _parse_thresholds(value) -> dict[str, float]:
    """레지스트리에 JSON으로 적힌 카테고리별 임계값을 읽는다.

    깨져 있으면 **빈 값으로 본다** — 판정을 멈추느니 전체 기준값으로 돌아가는 편이 낫다.
    """
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    found = {}
    for name, number in parsed.items():
        try:
            found[str(name)] = float(number)
        except (TypeError, ValueError):
            continue
    return found


def load_version(version: str) -> LoadedModel:
    """레지스트리 버전을 불러온다."""
    row = registry.get(version)
    if row is None:
        raise ValueError(f"등록되지 않은 버전입니다: {version}")

    artifact = registry.artifact_path(row.get("artifact"))
    if artifact is None or not artifact.exists():
        raise FileNotFoundError(
            f"{version}의 모델 파일이 없습니다. 지표만 등록된 버전은 추론에 쓸 수 없습니다."
        )

    threshold = row.get("threshold")
    threshold = float(threshold) if pd.notna(threshold) else 0.5
    thresholds = _parse_thresholds(row.get("thresholds"))
    kind = str(row.get("kind", ""))

    if kind == "baseline":
        return LoadedModel(
            version=version, kind=kind, threshold=threshold, thresholds=thresholds,
            baseline=models.BaselineModel.load(Path(artifact)),
        )
    if kind == "anomaly":
        return LoadedModel(
            version=version, kind=kind, threshold=threshold, thresholds=thresholds,
            anomaly=models.PatchAnomalyModel.load(Path(artifact)),
        )
    raise ValueError(f"추론을 지원하지 않는 종류입니다: {kind}")


@dataclass
class BatchResult:
    """배치 추론 결과."""

    records: list[dict]
    features: np.ndarray
    image_ids: list[str]
    failed: list[str]

    def __len__(self) -> int:
        return len(self.records)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def run_batch(
    model: LoadedModel,
    rows: pd.DataFrame,
    *,
    threshold: float | None = None,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    progress: ProgressCallback | None = None,
) -> BatchResult:
    """이미지 묶음에 추론을 돌리고 로그 레코드를 만든다.

    특징 행렬도 함께 반환한다 — 드리프트 감시가 같은 특징을 다시 계산하지 않도록.

    `transform`은 이미지를 읽은 뒤 특징을 뽑기 전에 끼워 넣는다. 운영 시나리오 시뮬레이터가
    조명·초점 변화를 재현할 때 쓴다. 실제 배치 추론에서는 쓰지 않는다.
    """
    # threshold를 직접 주면 그 값 하나로 전체를 판정한다(시뮬레이터·비교용).
    # 안 주면 모델이 등록될 때 정해진 카테고리별 값을 쓴다.
    fixed = None if threshold is None else float(threshold)
    records: list[dict] = []
    vectors: list[np.ndarray] = []
    image_ids: list[str] = []
    failed: list[str] = []

    total = len(rows)
    for index, (_, row) in enumerate(rows.iterrows(), start=1):
        path = row.get("path_abs") or row.get("path")
        image = viz.load_rgb(str(path))
        if image is None:
            failed.append(str(row.get("image_id", path)))
        else:
            if transform is not None:
                image = transform(image)
            started = time.perf_counter()
            vector = features.image_features(image)
            score = (
                float(model.baseline.score(vector[None, :])[0])
                if model.kind == "baseline" and model.baseline is not None
                else model.score_image(image)
            )
            latency_ms = (time.perf_counter() - started) * 1000.0

            category = str(row.get("category", ""))
            used = fixed if fixed is not None else model.threshold_for(category)
            records.append(
                {
                    "version": model.version,
                    "image_id": str(row.get("image_id", "")),
                    "source": str(row.get("source", "")),
                    "category": category,
                    "score": score,
                    "threshold": used,
                    "decision": config.LABEL_DEFECT if score >= used else config.LABEL_NORMAL,
                    "latency_ms": round(latency_ms, 2),
                }
            )
            vectors.append(vector)
            image_ids.append(str(row.get("image_id", "")))
        if progress is not None:
            progress(index, total)

    matrix = (
        np.stack(vectors) if vectors
        else np.empty((0, len(features.FEATURE_NAMES)), dtype=np.float32)
    )
    return BatchResult(records=records, features=matrix, image_ids=image_ids, failed=failed)


def selectable_versions(registry_frame: pd.DataFrame | None = None) -> Sequence[str]:
    """추론에 쓸 수 있는(모델 파일이 존재하는) 버전 목록."""
    frame = registry.load_registry() if registry_frame is None else registry_frame
    if frame.empty:
        return []
    usable = []
    for _, row in frame.iterrows():
        artifact = registry.artifact_path(row.get("artifact"))
        if artifact is not None and artifact.exists():
            usable.append(str(row["version"]))
    return usable
