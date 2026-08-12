"""이 촬영으로 결함이 보이기는 하는가 (V0).

## 왜 모델보다 이것이 먼저인가

**결함이 3픽셀이면 어떤 모델도 못 잡는다.** 사람도 못 본다. 그런데 이 사실은 영상을 다 뽑고
라벨링을 하고 학습을 돌린 **뒤에야** 드러나는 것이 보통이다 — 그때는 이미 며칠이 지나 있다.
`video.plan_from_process()`가 "이 촬영으로는 목표 장수를 못 채운다"를 미리 알려주는 것과
같은 성격의 점검이다. 촬영 계획의 문제를 촬영 계획 단계에서 알아채자는 것.

## 무엇을 기준으로 재는가

**원본 픽셀이 아니라 «모델이 실제로 보는 픽셀»** 이다. 원본에서 100픽셀이어도 학습할 때
640으로 줄이면 그 비율만큼 작아진다. 지금 이 앱의 이상탐지는 256으로, 검출(YOLO)은 640으로
줄여 본다 — 같은 촬영이라도 어느 쪽에 태우느냐로 결과가 갈린다.

이 프로젝트에 쌓인 **실제 결함 박스 1,334개**를 재 보니 짧은 변 중앙값이 화면 폭의
**1.78%** 였다. 표면 결함은 원래 이 정도로 작다.

## 한계선 두 개

| | 모델 입력에서의 크기 | 뜻 |
|---|---|---|
| `SAFE_PX` | 20px | 안정적으로 잡힌다 |
| `FLOOR_PX` | 10px | 이 아래는 사실상 불가능 |

YOLO 계열의 가장 촘촘한 격자 간격이 8픽셀이라, 물체가 격자 한 칸을 못 채우면 학습 신호가
거의 생기지 않는다. **경험칙이지 이 프로젝트에서 측정한 값은 아니다** — 다만 방향은 분명해서
"3픽셀은 안 된다"를 말하는 데는 충분하다.

## 안 되면 무엇을 바꾸는가

이 모듈의 값어치는 판정보다 **지렛대를 알려주는 데** 있다. 네 가지가 있고 비용이 다르다.

1. **타일 분할** — 프레임을 격자로 잘라 조각마다 추론한다. **소프트웨어만 고치면 되므로
   가장 싸다.** 대신 추론 횟수가 타일 수만큼 늘어난다.
2. **모델 입력 크기** — 640 → 1280. 역시 설정이지만 학습 시간과 메모리가 제곱으로 는다.
3. **카메라 해상도** — 장비를 바꿔야 한다.
4. **화각을 좁힌다** — 카메라를 옮기거나 더 달아야 한다. 한 대가 보던 폭을 나눠 맡는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

SAFE_PX = 20.0
"""모델 입력에서 이만큼이면 안정적으로 잡힌다 (경험칙)."""

FLOOR_PX = 10.0
"""모델 입력에서 이 아래는 사실상 불가능 (경험칙). YOLO 격자 간격이 8px다."""

VERDICT_OK = "가능"
VERDICT_HARD = "어려움"
VERDICT_NO = "불가"

TYPICAL_DEFECT_FRACTION = 0.0178
"""표면 결함의 짧은 변이 화면 폭에서 차지하는 비율 (VisA 실제 박스 1,334개 중앙값).

**이 값은 물리 상수가 아니다.** 결함이 원래 그만큼 크다는 뜻이 아니라, VisA를 그 정도
화각으로 찍었다는 뜻이다. 다른 화각으로 찍으면 같은 결함이 다른 비율이 된다.

그래서 이 값만으로는 판정할 수 없다. **결함의 실물 크기(mm)** 가 있어야 하고, 그것을
모를 때만 «그 촬영이 담던 폭이 얼마였는가»를 함께 가정해 되짚는다
(`defect_mm_from_fraction`). 가정을 숨기면 없는 근거를 만든 것이 된다.
"""

TYPICAL_SOURCE_FOV_MM = 300.0
"""위 비율을 잰 촬영이 담았을 폭(mm) — VisA는 물체 하나를 채우는 근접 촬영이다.

정확한 값이 공개되어 있지 않아 **어림한 값**이다. 결함 실물 크기를 모를 때의 출발점으로만
쓰고, 실제 결함을 한 번 재 보면 그 값으로 바꾸는 편이 훨씬 낫다.
"""

MODEL_INPUT_PX = {
    "검출 (YOLO 640)": 640,
    "검출 (YOLO 1280)": 1280,
    "이상탐지 (256)": 256,
}
"""쓸 수 있는 모델 입력 크기. 같은 촬영이라도 어디에 태우느냐로 결과가 갈린다."""


@dataclass(frozen=True)
class Framing:
    """이 촬영이 결함을 몇 픽셀로 담는가.

    Args:
        defect_mm: 잡아야 하는 **가장 작은** 결함의 짧은 변 (mm).
        fov_mm: 카메라 한 대가 담는 가로 폭 (mm).
        sensor_px: 카메라 가로 해상도 (px).
        model_px: 모델이 보는 입력 가로 크기 (px).
        tiles: 프레임을 가로로 몇 등분해 따로 추론할지. 1이면 통째로.
    """

    defect_mm: float
    fov_mm: float
    sensor_px: int
    model_px: int = 640
    tiles: int = 1

    @property
    def native_px(self) -> float:
        """원본 영상에서 결함이 차지하는 픽셀. 카메라가 애초에 담아내는 한계다."""
        if self.fov_mm <= 0:
            return 0.0
        return self.sensor_px * self.defect_mm / self.fov_mm

    @property
    def model_input_px(self) -> float:
        """**모델이 실제로 보는** 픽셀. 판정은 이 값으로 한다.

        타일로 나누면 한 조각이 담는 폭이 1/tiles로 줄고, 그 조각을 다시 `model_px`로
        키워 보므로 결함은 그만큼 커진다. 다만 **원본에 없는 정보가 생기지는 않으므로**
        원본 픽셀 수를 넘을 수는 없다.
        """
        enlarged = self.model_px * self.tiles * self.defect_mm / max(self.fov_mm, 1e-9)
        return min(enlarged, self.native_px)

    @property
    def verdict(self) -> str:
        size = self.model_input_px
        if size >= SAFE_PX:
            return VERDICT_OK
        if size >= FLOOR_PX:
            return VERDICT_HARD
        return VERDICT_NO

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_OK

    def smallest_catchable_mm(self, *, target: float = SAFE_PX) -> float:
        """**이 촬영으로 잡을 수 있는 가장 작은 결함**(mm).

        판정보다 이 숫자가 쓸모 있을 때가 많다. "우리 결함은 20mm쯤 된다"는 알아도
        "그래서 되는 건가"는 모르는데, 이 값과 비교하면 바로 답이 나온다.
        """
        # 모델 입력에서 target px가 되려면: model_px * tiles * d / fov = target
        by_model = target * self.fov_mm / max(self.model_px * self.tiles, 1)
        # 원본에서도 target px는 담겨야 한다 — 없는 정보는 확대해도 안 생긴다
        by_sensor = target * self.fov_mm / max(self.sensor_px, 1)
        return max(by_model, by_sensor)

    def summary(self) -> str:
        return (
            f"원본에서 {self.native_px:.0f}px → 모델 입력에서 **{self.model_input_px:.0f}px** "
            f"({self.verdict})"
        )


def fraction_from_boxes(frame, images) -> float | None:
    """쌓인 박스에서 «결함이 화면 폭의 몇 %인가»를 잰다 (짧은 변 중앙값).

    결함 실물 크기(mm)를 재 본 사람은 드물지만, 이미 라벨링한 박스는 있다. 그것으로
    비율을 알면 실물 크기를 몰라도 화각 판정을 할 수 있다.
    """
    import numpy as np
    import pandas as pd

    if frame is None or frame.empty or images is None or images.empty:
        return None

    sizes = {
        str(row["image_id"]): (row.get("width"), row.get("height"))
        for _, row in images.iterrows()
    }
    found = []
    for _, row in frame.iterrows():
        width, height = sizes.get(str(row["image_id"]), (None, None))
        if not width or not height or pd.isna(width) or pd.isna(height):
            continue
        found.append(min(float(row["w"]) / float(width), float(row["h"]) / float(height)))
    return float(np.median(found)) if found else None


def defect_mm_from_fraction(fraction: float, source_fov_mm: float) -> float:
    """«화면 폭의 몇 %»와 **그 촬영이 담던 폭**으로 결함 실물 크기를 되짚는다.

    `source_fov_mm`는 비율을 잰 **원본 촬영**의 화각이지 앞으로 찍을 화각이 아니다.
    이 둘을 헷갈리면 "지금 화각에서도 결함이 그 비율만큼 보인다"고 가정하는 셈이 되어,
    화각을 넓힐수록 결함도 같이 커지는 말이 안 되는 결론이 나온다.
    """
    return max(fraction, 0.0) * max(source_fov_mm, 0.0)


# --- 안 될 때 무엇을 바꾸는가 -----------------------------------------------

@dataclass(frozen=True)
class Lever:
    """바꿀 수 있는 것 하나와, 그렇게 하면 얼마가 되는지."""

    name: str
    change: str
    cost: str
    reachable: bool


def _needed_tiles(framing: Framing, target: float) -> int:
    """목표 픽셀을 채우려면 가로로 몇 등분해야 하는가."""
    now = framing.model_input_px
    if now >= target or now <= 0:
        return framing.tiles
    return int(ceil(framing.tiles * target / now))


def levers(framing: Framing, *, target: float = SAFE_PX) -> list[Lever]:
    """목표를 채우는 방법들을 **비용이 싼 순서로** 돌려준다.

    판정만 하고 끝내면 "그래서 어쩌라고"가 된다. 화각과 장비는 현장을 바꿔야 하지만
    타일 분할과 입력 크기는 설정이라, 오늘 당장 해 볼 수 있는 것부터 알려준다.
    """
    found: list[Lever] = []

    # 1) 타일 분할 — 소프트웨어만 고치면 된다
    tiles = _needed_tiles(framing, target)
    if tiles > framing.tiles:
        # 원본 픽셀이 목표에 못 미치면 아무리 잘라도 안 된다 — 없는 정보는 안 생긴다
        reachable = framing.native_px >= target
        found.append(
            Lever(
                "타일 분할",
                f"프레임을 가로 {tiles}등분해 조각마다 추론",
                f"추론 횟수 {tiles ** 2}배 (소프트웨어만 고치면 된다)",
                reachable,
            )
        )

    # 2) 모델 입력 크기 — 설정이지만 학습 비용이 제곱으로 는다
    bigger = framing.model_px * target / max(framing.model_input_px, 1e-9)
    if bigger > framing.model_px:
        rounded = int(ceil(bigger / 32) * 32)      # YOLO는 32의 배수를 쓴다
        found.append(
            Lever(
                "모델 입력 크기",
                f"{framing.model_px}px → {rounded}px",
                f"학습 시간·메모리 약 {(rounded / framing.model_px) ** 2:.1f}배",
                framing.native_px >= target,
            )
        )

    # 3) 카메라 해상도 — 장비를 바꿔야 한다
    need_sensor = framing.sensor_px * target / max(framing.model_input_px, 1e-9)
    if need_sensor > framing.sensor_px:
        found.append(
            Lever(
                "카메라 해상도",
                f"가로 {framing.sensor_px:,}px → {int(ceil(need_sensor / 160) * 160):,}px",
                "장비 교체",
                True,
            )
        )

    # 4) 화각을 좁힌다 — 카메라를 옮기거나 더 단다
    need_fov = framing.fov_mm * framing.model_input_px / max(target, 1e-9)
    if need_fov < framing.fov_mm:
        cameras = int(ceil(framing.fov_mm / max(need_fov, 1e-9)))
        found.append(
            Lever(
                "화각을 좁힌다",
                f"한 대가 담는 폭 {framing.fov_mm / 1000:.2f}m → {need_fov / 1000:.2f}m",
                f"같은 폭을 덮으려면 카메라 {cameras}대",
                True,
            )
        )

    return found


def advice(framing: Framing, *, target: float = SAFE_PX) -> str:
    """한 줄 조언. 판정에 따라 말이 달라야 한다."""
    if framing.verdict == VERDICT_OK:
        return "이 촬영이면 결함이 충분히 크게 잡힙니다. 그대로 진행하세요."
    if framing.verdict == VERDICT_HARD:
        return (
            "잡히기는 하지만 놓치는 것이 많을 크기입니다. 아래 중 하나를 바꾸면 안정권에 "
            "듭니다 — **타일 분할이 가장 쌉니다.**"
        )
    return (
        "**이 촬영으로는 결함이 보이지 않습니다.** 모델을 바꿔도 소용이 없고, 라벨링을 "
        "시작하기 전에 촬영 계획을 고쳐야 합니다."
    )
