"""초보자 안내: 지금 어디까지 왔고 다음에 무엇을 해야 하는가.

## 왜 필요한가

이 앱은 4단계를 순서대로 밟아야 하는데, 그 순서를 화면이 알려주지 않는다. 처음 들어온
사람은 페이지 5개와 탭 20여 개를 마주하고 어디부터 눌러야 할지 알 수 없다. 게다가 순서를
건너뛰면 "라벨된 이미지가 없습니다" 같은 문구만 나올 뿐, 그래서 어디로 가야 하는지는
말해주지 않는다.

이 모듈은 현재 상태를 보고 **다음 한 걸음**을 정한다. 화면은 그것을 표시만 한다.
판단 로직을 UI에서 분리해야 테스트할 수 있고, 여러 화면이 같은 답을 내놓는다.

## 모델 선택 안내

선택지를 주면서 무엇을 골라야 할지 알려주지 않으면 초보자는 막힌다. 특히 VisA 공식 분할은
train에 결함이 한 장도 없어서 **지도학습 베이스라인은 아예 학습이 불가능**하다. 이걸 모르고
베이스라인을 누르면 원인을 알 수 없는 오류를 만난다. `model_advice()`가 학습 데이터를 보고
가능한 것과 권장하는 것을 미리 알려준다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config, features

# 화면 경로 — `st.page_link`에 그대로 넘긴다.
PAGE_INGEST = "app_pages/p1_ingest.py"
PAGE_LABELING = "app_pages/p2_labeling.py"
PAGE_MODELING = "app_pages/p3_modeling.py"
PAGE_OPERATIONS = "app_pages/p4_operations.py"

KIND_ANOMALY = "anomaly"
KIND_BASELINE = "baseline"

# 이상탐지가 정상 분포를 위치별로 학습하려면 정상 이미지가 특징 차원보다 넉넉해야 한다.
MIN_NORMAL_FOR_ANOMALY = features.PATCH_FEATURE_DIM + 1
# 지도학습이 결함을 배우려면 최소한 이 정도는 있어야 한다 (경험적 하한).
MIN_DEFECT_FOR_BASELINE = 10

# `production=None`은 "승격된 모델이 없다"는 뜻이라 "인자를 안 넘겼다"와 구분해야 한다.
# None을 미제공으로 취급하면, 승격 안 된 상태를 넘겨도 레지스트리를 다시 읽어버린다.
_UNSET = object()


@dataclass(frozen=True)
class Step:
    """파이프라인 한 걸음."""

    key: str
    stage: int
    title: str
    done: bool
    detail: str       # 현재 상태 (완료면 무엇이 됐는지, 아니면 왜 아직인지)
    action: str       # 다음에 할 일 (완료면 빈 문자열)
    page: str
    tab: str = ""

    @property
    def where(self) -> str:
        return f"{self.stage}단계 · {self.tab}" if self.tab else f"{self.stage}단계"


def pipeline_steps(
    *,
    manifest: pd.DataFrame | None = None,
    resolved: pd.DataFrame | None = None,
    runs: pd.DataFrame | None = None,
    production=_UNSET,
    log: pd.DataFrame | None = None,
) -> list[Step]:
    """현재 상태를 걸음 목록으로 만든다.

    인자를 넘기지 않으면 직접 읽는다. 화면이 이미 읽어 둔 것이 있으면 넘겨서 중복 I/O를 피한다.
    """
    from . import experiments, labeling, monitoring, registry, storage

    manifest = storage.load_manifest() if manifest is None else manifest
    resolved = labeling.resolve(manifest) if resolved is None else resolved
    runs = experiments.load_runs() if runs is None else runs
    log = monitoring.load_log() if log is None else log
    if production is _UNSET:
        production = registry.production()

    n_images = len(manifest)
    labeled = _labeled(resolved)
    n_labeled = len(labeled)
    n_split = _assigned_splits(resolved)
    n_runs = len(runs)
    n_log = len(log)

    return [
        Step(
            key="collect", stage=1, title="이미지 모으기",
            done=n_images > 0,
            detail=f"{n_images:,}장 등록됨" if n_images else "등록된 이미지가 없습니다",
            action="" if n_images else (
                "내려받은 데이터셋이 없다면 **합성 샘플 생성**으로 시작하는 것이 가장 빠릅니다. "
                "다운로드 없이 전 과정을 시험할 수 있습니다."
            ),
            page=PAGE_INGEST, tab="합성 샘플 생성",
        ),
        Step(
            key="label", stage=2, title="정상/결함 라벨 확인",
            done=n_labeled > 0,
            detail=f"{n_labeled:,}장 라벨됨" if n_labeled else "라벨된 이미지가 없습니다",
            action="" if n_labeled else (
                "폴더 구조에서 추론한 라벨이 맞는지 확인합니다. "
                "공개 데이터셋을 폴더로 넣었다면 대부분 이미 라벨이 붙어 있습니다."
            ),
            page=PAGE_LABELING, tab="라벨 검수",
        ),
        Step(
            key="split", stage=2, title="학습/평가 데이터 나누기",
            done=n_split > 0,
            detail=f"{n_split:,}장에 분할 배정됨" if n_split else "분할이 배정되지 않았습니다",
            action="" if n_split else (
                "학습에 쓴 이미지로 평가하면 성능이 실제보다 높게 나옵니다. "
                "**데이터 분할**에서 나눠 두어야 평가를 믿을 수 있습니다."
            ),
            page=PAGE_LABELING, tab="데이터 분할",
        ),
        Step(
            key="train", stage=3, title="모델 학습하고 평가하기",
            done=n_runs > 0,
            detail=f"실행 {n_runs:,}건 기록됨" if n_runs else "학습한 모델이 없습니다",
            action="" if n_runs else (
                "어느 방식을 골라야 할지는 화면이 데이터를 보고 알려줍니다."
            ),
            page=PAGE_MODELING, tab="이상탐지",
        ),
        Step(
            key="promote", stage=4, title="쓸 모델 정하기 (승격)",
            done=production is not None,
            detail=(
                f"서비스 중: {production['version']}" if production is not None
                else "서비스 중인 모델이 없습니다"
            ),
            action="" if production is not None else (
                "학습만으로는 운영이 시작되지 않습니다. 쓸 버전을 하나 정해서 승격해야 "
                "감시와 롤백의 기준이 생깁니다."
            ),
            page=PAGE_OPERATIONS, tab="모델 레지스트리",
        ),
        Step(
            key="operate", stage=4, title="운영이 돌아가는지 보기",
            done=n_log > 0,
            detail=f"판정 로그 {n_log:,}건" if n_log else "판정 로그가 없습니다",
            action="" if n_log else (
                "**운영 시나리오 시연**을 누르면 운영 3개월치를 몇 초 만에 재생해 "
                "드리프트 감시·성능 추이 화면이 실제로 채워집니다."
            ),
            page=PAGE_OPERATIONS, tab="운영 시나리오 시연",
        ),
    ]


def next_step(steps: list[Step] | None = None, **kwargs) -> Step | None:
    """아직 안 끝난 첫 걸음. 전부 끝났으면 None."""
    steps = pipeline_steps(**kwargs) if steps is None else steps
    return next((step for step in steps if not step.done), None)


def progress(steps: list[Step]) -> tuple[int, int]:
    """(끝난 걸음 수, 전체 걸음 수)."""
    return sum(1 for step in steps if step.done), len(steps)


# --- 모델 선택 안내 ---------------------------------------------------------

@dataclass(frozen=True)
class ModelAdvice:
    recommended: str          # KIND_ANOMALY | KIND_BASELINE
    reason: str
    baseline_blocked: bool    # 지도학습이 아예 불가능한가
    anomaly_blocked: bool
    notes: list[str]


def model_advice(resolved: pd.DataFrame) -> ModelAdvice:
    """학습 데이터를 보고 어느 방식이 가능하고 무엇을 권하는지 판단한다.

    가장 중요한 것은 **불가능한 선택지를 미리 알려주는 것**이다. VisA 공식 분할처럼
    train에 결함이 없으면 지도학습은 시작조차 못 하는데, 화면은 선택지를 똑같이 보여준다.
    """
    labeled = _labeled(resolved)
    if labeled.empty or "split" not in labeled.columns:
        n_normal = n_defect = 0
    else:
        train = labeled[labeled["split"].astype(str) == config.SPLIT_TRAIN]
        n_normal = int((train["label"].astype(str) == config.LABEL_NORMAL).sum())
        n_defect = int((train["label"].astype(str) == config.LABEL_DEFECT).sum())

    notes: list[str] = []
    baseline_blocked = n_defect == 0
    anomaly_blocked = n_normal == 0

    if baseline_blocked:
        notes.append(
            "학습 분할에 결함이 **한 장도 없습니다.** 지도학습(베이스라인)은 결함을 배울 수 없어 "
            "학습이 불가능합니다. VisA 공식 분할이 이런 형태입니다 — 정상만으로 학습하고 "
            "test에서만 결함을 봅니다."
        )
    elif n_defect < MIN_DEFECT_FOR_BASELINE:
        notes.append(
            f"학습 분할의 결함이 {n_defect}장뿐입니다. 지도학습은 결함 예시가 적으면 "
            "제대로 배우지 못하므로, 정상만 학습하는 이상탐지가 유리합니다."
        )

    if anomaly_blocked:
        notes.append("학습 분할에 정상 이미지가 없습니다. 이상탐지는 정상만으로 학습합니다.")
    elif 0 < n_normal < MIN_NORMAL_FOR_ANOMALY:
        notes.append(
            f"정상 이미지가 {n_normal}장으로 적습니다. '위치별 분포 학습'을 끄거나 "
            f"정상을 {MIN_NORMAL_FOR_ANOMALY}장 이상으로 늘리세요."
        )

    if anomaly_blocked and not baseline_blocked:
        recommended, reason = KIND_BASELINE, "정상 이미지가 없어 이상탐지를 쓸 수 없습니다."
    elif baseline_blocked or n_defect < MIN_DEFECT_FOR_BASELINE:
        recommended, reason = KIND_ANOMALY, (
            "정상 이미지만으로 '평소와 다른 것'을 찾는 방식이라 결함 예시가 적어도 됩니다."
        )
    else:
        recommended, reason = KIND_BASELINE, (
            "결함 예시가 충분합니다. 먼저 베이스라인으로 하한선을 잡고, 이상탐지가 그보다 "
            "나은지 비교하는 순서가 좋습니다."
        )

    return ModelAdvice(
        recommended=recommended, reason=reason,
        baseline_blocked=baseline_blocked, anomaly_blocked=anomaly_blocked, notes=notes,
    )


# --- 내부 ------------------------------------------------------------------

def _labeled(resolved: pd.DataFrame) -> pd.DataFrame:
    if resolved.empty or "label" not in resolved.columns:
        return resolved
    return resolved[
        resolved["label"].astype(str).isin([config.LABEL_NORMAL, config.LABEL_DEFECT])
    ]


def _assigned_splits(resolved: pd.DataFrame) -> int:
    if resolved.empty or "split" not in resolved.columns:
        return 0
    return int((resolved["split"].astype(str) != config.SPLIT_NONE).sum())
