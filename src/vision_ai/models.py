"""3단계 모델: 베이스라인 분류기와 패치 기반 이상탐지.

두 접근을 나란히 둔 이유:
- 베이스라인(지도학습)은 결함 라벨이 있을 때 성능 하한선을 빠르게 확보한다.
- 이상탐지는 정상만 학습하므로 결함 샘플이 적은 상황(VisA의 기본 전제)에 맞고,
  결함 **위치**까지 히트맵으로 낸다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import features


# --- 베이스라인 분류기 ------------------------------------------------------

BASELINE_KINDS: tuple[str, ...] = ("logreg", "random_forest", "gradient_boosting")

BASELINE_KIND_LABELS = {
    "logreg": "로지스틱 회귀 (선형, 빠름)",
    "random_forest": "랜덤 포레스트 (비선형, 안정적)",
    "gradient_boosting": "그래디언트 부스팅 (비선형, 느림)",
}


@dataclass
class BaselineConfig:
    """베이스라인 학습 설정."""

    kind: str = "logreg"
    balanced: bool = True   # 클래스 불균형 보정
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class BaselineModel:
    """이미지 단위 특징 → 결함 확률."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self._pipeline = None

    def _build(self):
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        kind = self.config.kind
        weight = "balanced" if self.config.balanced else None
        if kind == "logreg":
            estimator = LogisticRegression(
                max_iter=2000, class_weight=weight, random_state=self.config.seed
            )
        elif kind == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=300, class_weight=weight, random_state=self.config.seed, n_jobs=-1
            )
        elif kind == "gradient_boosting":
            # GradientBoosting은 class_weight를 지원하지 않는다 — fit에서 sample_weight로 보정한다
            estimator = GradientBoostingClassifier(random_state=self.config.seed)
        else:
            raise ValueError(f"지원하지 않는 모델 종류: {kind} (가능: {BASELINE_KINDS})")

        return Pipeline([("scale", StandardScaler()), ("clf", estimator)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        """특징 행렬과 라벨(1=결함)로 학습한다."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("학습에는 정상·결함 두 클래스가 모두 필요합니다.")

        self._pipeline = self._build()
        if self.config.kind == "gradient_boosting" and self.config.balanced:
            counts = np.bincount(y, minlength=2).astype(float)
            weights = np.where(y == 1, counts.sum() / (2 * counts[1]), counts.sum() / (2 * counts[0]))
            self._pipeline.fit(X, y, clf__sample_weight=weights)
        else:
            self._pipeline.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """결함 확률(0~1)을 반환한다."""
        if self._pipeline is None:
            raise RuntimeError("학습되지 않은 모델입니다.")
        X = np.asarray(X, dtype=np.float64)
        return self._pipeline.predict_proba(X)[:, 1]

    def feature_importance(self) -> np.ndarray | None:
        """특징 중요도(가능한 모델만). 없으면 None."""
        if self._pipeline is None:
            return None
        estimator = self._pipeline.named_steps["clf"]
        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_, dtype=float)
        if hasattr(estimator, "coef_"):
            return np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
        return None

    def save(self, path: Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": self.config.to_dict(), "pipeline": self._pipeline}, path)

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        import joblib

        payload = joblib.load(Path(path))
        model = cls(BaselineConfig(**payload["config"]))
        model._pipeline = payload["pipeline"]
        return model


# --- 패치 기반 이상탐지 ----------------------------------------------------

IMAGE_SCORE_MODES: tuple[str, ...] = ("max", "p999", "p99", "p95", "mean")

_SCORE_MODE_LABELS = {
    "max": "최대값 (가장 민감)",
    "p999": "99.9 분위",
    "p99": "99 분위 (권장)",
    "p95": "95 분위",
    "mean": "평균 (가장 둔감)",
}


BACKEND_CLASSIC = "classic"
BACKEND_CNN = "cnn"
BACKENDS = (BACKEND_CLASSIC, BACKEND_CNN)
BACKEND_LABELS = {
    BACKEND_CLASSIC: "고전 CV 특징 (설치 불필요)",
    BACKEND_CNN: "사전학습 CNN 특징 (모델 45MB 필요)",
}


@dataclass
class AnomalyConfig:
    """이상탐지 설정."""

    patch: int = features.PATCH_SIZE
    stride: int = features.PATCH_STRIDE
    per_position: bool = True   # 격자 위치마다 별도 분포를 학습 (정렬된 이미지에 유리)
    shrinkage: float = 0.05     # 공분산 정규화 — 표본이 적을 때 역행렬 안정화
    image_score: str = "p99"
    size: int = features.IMAGE_SIZE
    # "classic" = 고전 CV 패치 특징(기본, 의존성 없음)
    # "cnn"     = 사전학습 ResNet18 특징 (모델 파일 필요, 정확도가 크게 높다)
    backend: str = BACKEND_CLASSIC

    def to_dict(self) -> dict:
        return asdict(self)


def score_mode_label(mode: str) -> str:
    return _SCORE_MODE_LABELS.get(mode, mode)


class PatchAnomalyModel:
    """정상 이미지만으로 격자별 정상 분포를 학습하고, 이탈 정도를 결함 점수로 쓴다.

    각 격자 위치에서 정상 특징의 평균·공분산을 구해 마할라노비스 거리를 계산한다.
    거리 지도가 곧 결함 위치 히트맵이 되므로, 별도 위치 예측기가 필요 없다.
    """

    def __init__(self, config: AnomalyConfig | None = None) -> None:
        self.config = config or AnomalyConfig()
        self._mean: np.ndarray | None = None       # (P, D) 또는 (1, D)
        self._precision: np.ndarray | None = None  # (P, D, D) 또는 (1, D, D)
        self._grid: tuple[int, int] | None = None
        self.n_train = 0
        self._extractor = None   # CNN 백엔드에서만 쓴다 (모델을 한 번만 읽도록)

    def _grid_features(self, image: np.ndarray) -> np.ndarray:
        """이미지 한 장 → (높이, 너비, 차원) 격자 특징.

        학습과 추론이 **반드시 같은 방식**을 써야 하므로 추출을 여기 한 곳에 둔다.
        """
        if self.config.backend != BACKEND_CNN:
            return features.patch_features(image, self.config.patch, self.config.stride)

        if self._extractor is None:
            from . import cnn_features

            self._extractor = cnn_features.load()
        flat = self._extractor.patch_grid(image)           # (위치, 차원)
        side = self._extractor.grid
        return flat.reshape(side, side, -1)

    # -- 학습 --
    def fit(self, images: Iterable[np.ndarray]) -> "PatchAnomalyModel":
        """정상 이미지들로 학습한다."""
        stacks: list[np.ndarray] = []
        for image in images:
            grid = self._grid_features(image)
            stacks.append(grid.reshape(-1, grid.shape[-1]))
            if self._grid is None:
                self._grid = grid.shape[:2]

        if not stacks:
            raise ValueError("학습할 정상 이미지가 없습니다.")

        data = np.stack(stacks)  # (N, P, D)
        self.n_train = data.shape[0]
        dim = data.shape[-1]

        if self.config.per_position:
            if self.n_train <= dim:
                raise ValueError(
                    f"위치별 학습에는 정상 이미지가 특징 차원({dim})보다 많아야 합니다 "
                    f"(현재 {self.n_train}장). 정상 이미지를 늘리거나 '위치별 분포'를 끄세요."
                )
            self._mean = data.mean(axis=0)                       # (P, D)
            centered = data - self._mean
            cov = np.einsum("npd,npe->pde", centered, centered) / (self.n_train - 1)
        else:
            pooled = data.reshape(-1, dim)
            if pooled.shape[0] <= dim:
                raise ValueError("학습 표본이 특징 차원보다 적습니다.")
            self._mean = pooled.mean(axis=0)[None, :]            # (1, D)
            centered = pooled - self._mean
            cov = (centered.T @ centered / (pooled.shape[0] - 1))[None, :, :]

        # 대각 성분에 shrinkage를 더해 특이 행렬을 피한다
        trace = np.trace(cov, axis1=-2, axis2=-1) / dim
        eye = np.eye(dim, dtype=cov.dtype)[None, :, :]
        cov = cov + self.config.shrinkage * trace[:, None, None] * eye
        self._precision = np.linalg.inv(cov)
        return self

    @property
    def is_fitted(self) -> bool:
        return self._mean is not None and self._precision is not None

    # -- 추론 --
    def score_grid(self, rgb: np.ndarray) -> np.ndarray:
        """격자별 마할라노비스 거리를 반환한다."""
        if not self.is_fitted:
            raise RuntimeError("학습되지 않은 모델입니다.")
        grid = self._grid_features(rgb)
        rows, cols, dim = grid.shape
        flat = grid.reshape(-1, dim)

        centered = flat - self._mean  # 브로드캐스트: (P,D) 또는 (1,D)
        if self._precision.shape[0] == 1:
            squared = np.einsum("pd,de,pe->p", centered, self._precision[0], centered)
        else:
            squared = np.einsum("pd,pde,pe->p", centered, self._precision, centered)
        return np.sqrt(np.maximum(squared, 0.0)).reshape(rows, cols)

    def score_map(self, rgb: np.ndarray, smooth: float = 4.0) -> np.ndarray:
        """이미지 크기의 결함 점수 히트맵을 반환한다."""
        return features.upsample_grid(self.score_grid(rgb), self.config.size, smooth)

    def image_score(self, rgb: np.ndarray) -> float:
        """이미지 1장의 대표 결함 점수."""
        return self.aggregate(self.score_grid(rgb))

    def aggregate(self, grid: np.ndarray) -> float:
        """격자 점수를 이미지 점수로 집계한다."""
        mode = self.config.image_score
        flat = np.asarray(grid, dtype=float).ravel()
        if mode == "max":
            return float(flat.max())
        if mode == "mean":
            return float(flat.mean())
        if mode == "p999":
            return float(np.percentile(flat, 99.9))
        if mode == "p99":
            return float(np.percentile(flat, 99))
        if mode == "p95":
            return float(np.percentile(flat, 95))
        raise ValueError(f"지원하지 않는 집계 방식: {mode} (가능: {IMAGE_SCORE_MODES})")

    # -- 저장/불러오기 --
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self._mean,
            precision=self._precision,
            grid=np.asarray(self._grid or (0, 0)),
            n_train=self.n_train,
            config=json.dumps(self.config.to_dict()),
        )

    @classmethod
    def load(cls, path: Path) -> "PatchAnomalyModel":
        payload = np.load(Path(path), allow_pickle=False)
        model = cls(AnomalyConfig(**json.loads(str(payload["config"]))))
        model._mean = payload["mean"]
        model._precision = payload["precision"]
        model._grid = tuple(int(v) for v in payload["grid"])
        model.n_train = int(payload["n_train"])
        return model


# --- 학습 데이터 준비 ------------------------------------------------------

@dataclass
class Dataset:
    """학습/평가에 쓰는 특징 행렬 묶음."""

    X: np.ndarray
    y: np.ndarray
    image_ids: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    reused: int = 0       # 캐시에서 그대로 가져온 장수
    extracted: int = 0    # 이번에 새로 계산한 장수

    def __len__(self) -> int:
        return len(self.y)


def build_dataset(
    paths: Sequence[str],
    labels: Sequence[int],
    image_ids: Sequence[str] | None = None,
    *,
    loader=None,
    progress=None,
    use_cache: bool = True,
) -> Dataset:
    """이미지 경로 목록에서 특징 행렬을 만든다. 읽기 실패한 이미지는 제외한다.

    한 번 계산한 특징은 프로젝트별 캐시에 남겨 다음 실행에서 다시 쓴다. 이미지 단위로
    남기므로, 데이터가 조금 늘어나면 **늘어난 만큼만** 계산한다. 캐시가 잘못 남을 위험보다
    다시 계산하는 비용이 크기 때문에, 조금이라도 미심쩍으면(파일이 바뀌었거나 특징 정의가
    달라졌으면) 캐시를 버리고 다시 뽑는다.

    ``progress(done, total)``은 이미지 하나를 처리할 때마다 불린다. 캐시에서 가져온 것도
    포함해 세므로, 캐시가 채워져 있으면 막대가 훨씬 빠르게 찬다.
    """
    from . import feature_cache, viz

    load = loader or viz.load_rgb
    rows: list[np.ndarray] = []
    kept_labels: list[int] = []
    kept_ids: list[str] = []
    failed: list[str] = []
    ids = list(image_ids) if image_ids is not None else [str(i) for i in range(len(paths))]

    cache = feature_cache.load() if use_cache else None
    reused = extracted = 0

    total = len(paths)
    for index, (path, label, image_id) in enumerate(zip(paths, labels, ids), start=1):
        vector = cache.get(str(image_id), path) if cache is not None else None
        if vector is None:
            image = load(path)
            if image is None:
                failed.append(str(path))
                if progress is not None:
                    progress(index, total)
                continue
            vector = features.image_features(image)
            extracted += 1
            if cache is not None:
                cache.put(str(image_id), path, vector)
        else:
            reused += 1

        rows.append(vector)
        kept_labels.append(int(label))
        kept_ids.append(str(image_id))
        if progress is not None:
            progress(index, total)

    if cache is not None and extracted:
        feature_cache.save(cache)

    X = np.stack(rows) if rows else np.empty((0, len(features.FEATURE_NAMES)), dtype=np.float32)
    return Dataset(
        X=X,
        y=np.asarray(kept_labels, dtype=int),
        image_ids=kept_ids,
        failed=failed,
        reused=reused,
        extracted=extracted,
    )
