"""사용자 설정.

## 왜 필요한가

지금까지 목표 재현율 0.95, 검수량 절감률 50%, 재학습 기준 50건 같은 값이 **코드에 박혀
있었다.** 그런데 이 값들은 물리 상수가 아니라 업무 판단이다. 미탐 1건의 비용이 큰 라인이면
재현율을 0.98로 올려야 하고, 검사 물량이 적으면 절감률 기준이 달라진다.

값이 화면마다 흩어져 있어도 안 된다. 3단계에서 목표 재현율을 0.98로 바꿔 임계값을 골라
놓고, 4단계 승격 점검은 0.95로 판단하면 **같은 모델을 두 화면이 다르게 평가한다.**
여기 한 곳에 두고 모든 화면이 같은 값을 읽는다.

## 저장 위치

`data/settings.json`에 둔다. 데이터와 함께 사라져도 기본값으로 되돌아갈 뿐이라 안전하고,
저장소에 커밋되지 않아 사람마다 다른 기준을 쓸 수 있다.

## 범위를 강제하는 이유

재현율 목표를 1.0으로 두면 "모든 것을 결함으로 판정"이 유일한 해가 되어 도구가 무의미해진다.
반대로 0.5 아래면 절반을 놓치겠다는 뜻이라 검사 도구로서 의미가 없다. 입력을 허용 범위로
자르되, 잘렸다는 사실은 화면에 알린다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from . import config

SETTINGS_FILE = "settings.json"


@dataclass(frozen=True)
class Settings:
    """판정 기준. 전부 업무 판단이므로 사용자가 바꿀 수 있어야 한다."""

    # D5 — 모델을 서비스에 올려도 되는지의 기준
    target_recall: float = 0.95
    min_reduction: float = 0.50

    # 업무 효과 환산에 쓰는 현장 가정
    prevalence: float = 0.01
    volume: int = 1000

    # 재학습 제안 기준
    new_label_threshold: int = 50
    recall_margin: float = 0.05


DEFAULTS = Settings()

# (최소, 최대). 범위를 벗어난 값은 잘라서 쓴다.
BOUNDS: dict[str, tuple[float, float]] = {
    "target_recall": (0.50, 0.999),
    "min_reduction": (0.0, 0.95),
    "prevalence": (0.0001, 0.50),
    "volume": (10, 1_000_000),
    "new_label_threshold": (1, 100_000),
    "recall_margin": (0.005, 0.50),
}

LABELS: dict[str, str] = {
    "target_recall": "목표 재현율",
    "min_reduction": "최소 검수량 절감률",
    "prevalence": "현장 불량률 (가정)",
    "volume": "검사 물량 (환산 기준)",
    "new_label_threshold": "재학습 제안 — 신규 라벨 수",
    "recall_margin": "재학습 경보 — 허용 재현율 낙폭",
}

HELP: dict[str, str] = {
    "target_recall": (
        "실제 결함 중 이 비율 이상을 잡도록 임계값을 자동으로 고릅니다. "
        "**올리면 결함을 덜 놓치지만 오탐이 늘어 검수량이 함께 늘어납니다.** "
        "미탐 1건의 비용이 크면 0.98 이상을 검토하세요."
    ),
    "min_reduction": (
        "이만큼 검수량이 줄지 않으면 승격 시 경고합니다. 도입해도 사람 일이 안 줄면 "
        "쓸 이유가 없기 때문입니다."
    ),
    "prevalence": (
        "실제 라인에서 100개 중 몇 개가 불량인지. 정밀도와 검수량 절감 환산에 쓰입니다. "
        "모르면 1%로 두세요 — 평가 데이터(1:1)를 그대로 믿는 것보다 훨씬 현실에 가깝습니다."
    ),
    "volume": "건수 환산의 기준 물량입니다. 판정 자체에는 영향을 주지 않습니다.",
    "new_label_threshold": "승격 이후 이만큼 라벨이 쌓이면 재학습을 제안합니다.",
    "recall_margin": "등록 시 재현율 대비 이보다 더 떨어지면 경보를 냅니다.",
}


def _path() -> Path:
    return config.DATA_ROOT / SETTINGS_FILE


def clamp(name: str, value):
    """허용 범위로 자른다. 범위가 없는 항목은 그대로 둔다."""
    if name not in BOUNDS:
        return value
    low, high = BOUNDS[name]
    caster = int if isinstance(getattr(DEFAULTS, name), int) else float
    try:
        number = caster(value)
    except (TypeError, ValueError):
        return getattr(DEFAULTS, name)
    return caster(min(max(number, low), high))


def load() -> Settings:
    """저장된 설정. 파일이 없거나 망가졌으면 기본값으로 돌아간다.

    설정 파일이 깨졌다고 앱이 멈추면 안 된다 — 기준값은 없어도 기본값으로 돌아가면 그만이다.
    """
    path = _path()
    if not path.exists():
        return DEFAULTS
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULTS
    if not isinstance(raw, dict):
        return DEFAULTS

    values = {}
    for entry in fields(Settings):
        values[entry.name] = (
            clamp(entry.name, raw[entry.name]) if entry.name in raw
            else getattr(DEFAULTS, entry.name)
        )
    return Settings(**values)


def save(settings: Settings) -> Settings:
    """설정을 저장한다. 범위를 벗어난 값은 잘라서 저장한다."""
    cleaned = Settings(**{f.name: clamp(f.name, getattr(settings, f.name)) for f in fields(Settings)})
    config.ensure_dirs()
    _path().write_text(
        json.dumps(asdict(cleaned), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cleaned


def reset() -> Settings:
    """기본값으로 되돌린다."""
    _path().unlink(missing_ok=True)
    return DEFAULTS


def changed(settings: Settings | None = None) -> dict[str, tuple]:
    """기본값과 다른 항목만 (현재값, 기본값)으로 돌려준다."""
    settings = load() if settings is None else settings
    return {
        entry.name: (getattr(settings, entry.name), getattr(DEFAULTS, entry.name))
        for entry in fields(Settings)
        if getattr(settings, entry.name) != getattr(DEFAULTS, entry.name)
    }
