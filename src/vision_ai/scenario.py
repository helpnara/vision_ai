"""4단계: 운영 시나리오 시뮬레이터.

## 왜 필요한가

드리프트 감시·성능 추이·재학습 판단은 **시간이 흐르고 데이터가 쌓여야** 의미가 생긴다.
기능은 다 구현되어 있어도, 방금 배포한 앱을 열면 화면이 전부 비어 있어서
"운영관리가 된다"를 보여줄 수가 없다. 배치 추론을 한 번 돌려도 점 하나만 찍힌다.

이 모듈은 **새 기능을 만들지 않는다.** 이미 있는 배치 추론·드리프트 감시·성능 측정·
재학습 판단을 그대로 쓰되, 몇 달치 운영을 몇 초 만에 재생해서 그 화면들이 실제로
작동하는 모습을 보이게 한다.

## 어떻게 재현하는가

- **시간**: 추론 로그의 `logged_at`을 과거 날짜로 채워 넣는다. 로그는 원래 이 값을
  그대로 받으므로 별도 장치가 필요 없다.
- **환경 변화**: 조명이 밝아지거나 초점이 흐려지는 상황을 이미지 변환으로 재현한다.
  `serving.run_batch(transform=...)`에 끼워 넣으므로 추론 경로 자체는 운영과 동일하다.
- **사후 검수**: 사람이 나중에 확인해 주는 정답을 시뮬레이터가 대신 기록한다.
  정답의 출처는 manifest의 라벨이며 모델 점수와 무관하므로 자기 채점이 아니다.

## 이 시나리오에서 실제로 관찰되는 것

환경이 변하면 **미탐이 늘기 전에 오탐이 먼저 폭증한다.** 이상탐지 모델은 "평소와 다름"을
점수로 매기는데, 조명이나 초점이 변하면 정상품도 "평소와 다르게" 보이기 때문이다.
측정해 보면 결함 판정 비율이 6% → 34% → 83%로 오르는 반면 재현율은 크게 흔들리지 않는다.

즉 이 상황의 실제 피해는 "결함을 놓치는 것"이 아니라 **사람이 볼 물량이 감당 불가로 늘어나는
것**이다. 스크리닝 도구로서의 가치가 사라지는 것이 곧 성능 저하다.

## 주의

시뮬레이션 결과는 **화면 시연용이지 성능 근거가 아니다.** 여기서 나온 재현율·PSI를
모델 성능으로 인용하면 안 된다. 기록되는 라벨은 `labeled_by="시나리오"`로 남아
사람이 실제로 검수한 라벨과 구분된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config, labeling, monitoring, registry, serving

ProgressCallback = Callable[[int, int, str], None]

LABELED_BY = "시나리오"
NOTE = "운영 시나리오 시뮬레이션"


@dataclass(frozen=True)
class Environment:
    """촬영 환경 변화. 0이면 변화 없음."""

    brightness: float = 0.0   # 밝기 가산값 (조명 변화)
    blur: float = 0.0         # 가우시안 시그마 (초점 저하)

    @property
    def changed(self) -> bool:
        return bool(self.brightness or self.blur)

    def describe(self) -> str:
        if not self.changed:
            return "변화 없음"
        parts = []
        if self.brightness:
            parts.append(f"밝기 {self.brightness:+.0f}")
        if self.blur:
            parts.append(f"흐림 σ={self.blur:.1f}")
        return " · ".join(parts)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        if not self.changed:
            return rgb
        import cv2

        out = rgb
        if self.brightness:
            out = np.clip(out.astype(np.int16) + self.brightness, 0, 255).astype(np.uint8)
        if self.blur:
            # 시그마에서 커널 크기를 정한다 (홀수, 최소 3)
            size = max(3, int(self.blur * 4) | 1)
            out = cv2.GaussianBlur(out, (size, size), self.blur)
        return out


@dataclass(frozen=True)
class Phase:
    """운영 한 구간."""

    key: str
    title: str
    narration: str       # 이 구간에 무슨 일이 일어났는가
    watch: str           # 그래서 어느 화면을 봐야 하는가
    days: int = 30
    environment: Environment = Environment()
    # 드리프트 판정에는 구간당 최소 표본이 필요하고(`monitoring.MIN_DRIFT_SAMPLES`),
    # 재현율은 검수된 **결함** 건수가 적으면 크게 흔들린다. 둘 다 만족하도록 잡은 값이다.
    n_images: int = 200
    verify_ratio: float = 0.4    # 사후 검수를 받는 비율
    # 실제 라인에는 정상이 압도적으로 많다. 데이터셋 비율(정상:결함 ≈ 1:1)을 그대로 흘려보내면
    # 오탐이 과소평가되고, 기준선과 분포가 달라 안정 구간마저 드리프트로 잡힌다.
    defect_rate: float = 0.08


# 드리프트 판정에는 구간당 최소 표본이 필요하다(`monitoring.MIN_DRIFT_SAMPLES`).
# 기본 구간 크기를 그보다 넉넉히 잡아 "표본 부족"으로 판정이 보류되지 않게 한다.
DEFAULT_TIMELINE: tuple[Phase, ...] = (
    Phase(
        key="stable",
        title="1개월차 · 안정 운영",
        narration="모델을 승격하고 정상적으로 검사를 돌린 구간이다. 환경 변화가 없다.",
        watch="드리프트 감시에서 대부분의 특징이 '안정'으로 나오는지 확인한다. 이게 기준선이다.",
        environment=Environment(),
    ),
    Phase(
        key="lighting",
        title="2개월차 · 조명이 밝아짐",
        narration=(
            "설비 점검 후 조명이 밝게 바뀐 상황이다. 사람 눈에는 큰 문제가 없어 보이지만 "
            "모델이 학습한 정상 분포와는 달라진다."
        ),
        watch=(
            "드리프트 감시에서 **밝기 계열 특징**의 PSI가 올라간다. 모델은 그대로인데 입력이 먼저 변했다. "
            "동시에 결함 판정 비율이 뛴다 — 정상품을 결함으로 잘못 거르기 시작한 것이다."
        ),
        environment=Environment(brightness=35),
    ),
    Phase(
        key="focus",
        title="3개월차 · 초점까지 흐려짐",
        narration="카메라 초점이 틀어져 이미지가 흐려진 상황이다. 조명 변화도 그대로다.",
        watch=(
            "**선명도 특징(lap_p99)** 의 PSI가 가장 크게 튄다. 결함 판정 비율이 더 올라가 "
            "사람이 볼 물량이 감당이 안 되는 수준이 된다. 재학습 판단에 신호가 뜬다. "
            "여기서 3단계로 돌아가 다시 학습하고 새 버전을 승격하면 한 바퀴가 끝난다."
        ),
        environment=Environment(brightness=35, blur=1.6),
    ),
)


@dataclass
class PhaseResult:
    phase: Phase
    logged: int = 0
    verified: int = 0
    # 검수된 것 중 실제 결함 건수. 재현율은 이 값이 작으면 크게 흔들리므로 함께 보여준다.
    verified_defects: int = 0
    mean_score: float = float("nan")
    defect_rate: float = float("nan")
    recall: float = float("nan")
    precision: float = float("nan")
    drift_level: str = ""
    drift_changed: int = 0
    top_drift: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class ScenarioResult:
    version: str
    phases: list[PhaseResult]
    total_logged: int
    total_verified: int
    warnings: list[str] = field(default_factory=list)


def _timestamps(start: datetime, days: int, count: int) -> list[str]:
    """구간 기간에 걸쳐 시각을 고르게 뿌린다."""
    if count <= 0:
        return []
    span = max(days, 1) * 24 * 60 * 60
    steps = np.linspace(0, span, count, endpoint=False)
    return [
        (start + timedelta(seconds=float(offset))).isoformat(timespec="seconds")
        for offset in steps
    ]


def _sample(rows: pd.DataFrame, n: int, defect_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    """정상/결함 비율을 현장에 가깝게 맞춰 표본을 뽑는다."""
    label = rows["label"].astype(str)
    normal = rows[label == config.LABEL_NORMAL]
    defect = rows[label == config.LABEL_DEFECT]
    if normal.empty or defect.empty:
        # 한쪽이 없으면 비율을 맞출 수 없다. 있는 것만 그대로 쓴다.
        pool = rows if normal.empty and defect.empty else (normal if defect.empty else defect)
        return _take(pool, n, rng)

    want_defect = min(len(defect), max(1, int(round(n * defect_rate))))
    want_normal = min(len(normal), n - want_defect)
    picked = pd.concat([_take(normal, want_normal, rng), _take(defect, want_defect, rng)])
    return picked.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))


def _take(rows: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0:
        return rows.iloc[:0]
    if len(rows) <= n:
        return rows
    index = rng.choice(len(rows), size=n, replace=False)
    return rows.iloc[np.sort(index)]


def run(
    version: str,
    rows: pd.DataFrame,
    *,
    timeline: Sequence[Phase] = DEFAULT_TIMELINE,
    end: datetime | None = None,
    seed: int = 0,
    progress: ProgressCallback | None = None,
) -> ScenarioResult:
    """시나리오를 실행해 추론 로그와 사후 검수 라벨을 쌓는다.

    Args:
        version: 레지스트리에 등록된 모델 버전
        rows: manifest 행 (`path_abs`, `image_id`, `label` 필요)
        timeline: 재생할 구간들
        end: 마지막 구간이 끝나는 시각 (기본: 지금)
        seed: 표본 추출 재현용

    Returns:
        구간별 결과. 실패해도 예외 대신 `warnings`에 담아 돌려준다 — 시연 중에 화면이
        멈추는 것보다 무엇이 부족한지 보여주는 편이 낫다.
    """
    warnings: list[str] = []
    if rows.empty:
        return ScenarioResult(version=version, phases=[], total_logged=0, total_verified=0,
                              warnings=["이미지가 없습니다. 1단계에서 데이터를 먼저 등록하세요."])

    model = serving.load_version(version)
    baseline = registry.load_baseline(version)
    if baseline is None:
        warnings.append("드리프트 기준선이 없는 버전이라 PSI를 계산할 수 없습니다.")

    end = end or datetime.now(timezone.utc)
    total_days = sum(phase.days for phase in timeline)
    cursor = end - timedelta(days=total_days)

    rng = np.random.default_rng(seed)
    results: list[PhaseResult] = []
    total_logged = total_verified = 0
    steps = len(timeline)

    for index, phase in enumerate(timeline, start=1):
        if progress is not None:
            progress(index, steps, phase.title)

        subset = _sample(rows, phase.n_images, phase.defect_rate, rng)
        batch = serving.run_batch(
            model, subset, transform=phase.environment.apply if phase.environment.changed else None
        )
        result = PhaseResult(phase=phase)

        if not batch.records:
            warnings.append(f"{phase.title}: 읽을 수 있는 이미지가 없어 건너뛰었습니다.")
            results.append(result)
            cursor += timedelta(days=phase.days)
            continue

        stamps = _timestamps(cursor, phase.days, len(batch.records))
        for record, stamp in zip(batch.records, stamps):
            record["logged_at"] = stamp
            record["note"] = NOTE
        result.logged = monitoring.log_inference(batch.records, note=NOTE)
        total_logged += result.logged

        scores = np.array([r["score"] for r in batch.records], dtype=float)
        decisions = [r["decision"] for r in batch.records]
        result.mean_score = float(scores.mean())
        result.defect_rate = float(np.mean([d == config.LABEL_DEFECT for d in decisions]))

        # 사후 검수: 사람이 확인해 주는 정답을 대신 기록한다.
        verified_ids = _record_verification(subset, batch.image_ids, phase.verify_ratio, rng)
        result.verified = len(verified_ids)
        total_verified += result.verified

        # 이번 구간만의 성능 (전체 누적이 아니라 구간별로 봐야 하락이 보인다)
        result.recall, result.precision, result.verified_defects = _phase_performance(
            subset, batch, verified_ids
        )

        if baseline is not None:
            drift = monitoring.feature_drift(baseline, batch.features, list(baseline["feature_names"]))
            summary = monitoring.drift_summary(drift)
            result.drift_level = str(summary.get("level", ""))
            result.drift_changed = int(summary.get("shifted", 0))
            top = drift.sort_values("psi", ascending=False).head(3)
            result.top_drift = [(str(r["feature"]), float(r["psi"])) for _, r in top.iterrows()]

        results.append(result)
        cursor += timedelta(days=phase.days)

    return ScenarioResult(
        version=version, phases=results,
        total_logged=total_logged, total_verified=total_verified, warnings=warnings,
    )


def _record_verification(
    subset: pd.DataFrame, image_ids: list[str], ratio: float, rng: np.random.Generator
) -> set[str]:
    """사후 검수 라벨을 기록한다.

    정답은 manifest 라벨에서 가져온다. 모델 점수와 무관한 값이므로 자기 채점이 아니다.
    다만 사람이 실제로 확인한 라벨과 섞이지 않도록 `labeled_by`를 따로 남긴다.
    """
    if ratio <= 0 or not image_ids:
        return set()

    truth = subset.set_index(subset["image_id"].astype(str))["label"].astype(str)
    candidates = [i for i in image_ids if i in truth.index and truth[i] in
                  (config.LABEL_NORMAL, config.LABEL_DEFECT)]
    if not candidates:
        return set()

    count = max(1, int(len(candidates) * ratio))
    picked = rng.choice(candidates, size=min(count, len(candidates)), replace=False)

    items = [
        {
            "image_id": str(image_id),
            "label": truth[image_id],
            "defect_type": (
                config.DEFECT_TYPE_UNSPECIFIED
                if truth[image_id] == config.LABEL_DEFECT
                else config.DEFECT_TYPE_NONE
            ),
            "roi": None,
            "verified": True,
            "labeled_by": LABELED_BY,
            "note": NOTE,
        }
        for image_id in picked
    ]
    labeling.record_labels(items)
    return {str(i) for i in picked}


def _phase_performance(
    subset: pd.DataFrame, batch: serving.BatchResult, verified_ids: set[str]
) -> tuple[float, float, int]:
    """이 구간에서 검수된 건만으로 재현율·정밀도를 계산한다.

    검수 대상은 무작위로 고른다. 모델이 결함으로 판정한 것만 검수하면 미탐이 표본에
    거의 안 잡혀 재현율이 실제보다 높게 나온다.
    """
    if not verified_ids:
        return float("nan"), float("nan"), 0

    truth = subset.set_index(subset["image_id"].astype(str))["label"].astype(str)
    tp = fp = fn = 0
    for record in batch.records:
        image_id = str(record["image_id"])
        if image_id not in verified_ids or image_id not in truth.index:
            continue
        actual = truth[image_id] == config.LABEL_DEFECT
        predicted = record["decision"] == config.LABEL_DEFECT
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return recall, precision, tp + fn


def clear() -> int:
    """시나리오가 만든 추론 로그를 지운다.

    라벨 이력은 지우지 않는다 — 추가 전용으로 설계된 파일이라 중간을 도려내면
    "누가 언제 무엇을 판정했는가"의 연속성이 깨진다. 시나리오가 남긴 라벨은
    `labeled_by="시나리오"`로 구분할 수 있다.
    """
    log = monitoring.load_log()
    if log.empty:
        return 0
    keep = log[log["note"].astype(str) != NOTE]
    removed = len(log) - len(keep)
    monitoring.clear_log()
    if not keep.empty:
        monitoring.log_inference(keep.to_dict("records"))
    return removed
