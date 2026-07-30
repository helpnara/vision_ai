"""이미지 품질 점검.

수집 단계에서 "학습에 쓸 수 있는 이미지인가"를 걸러내기 위한 최소한의 지표를 계산한다.
일상에서 직접 촬영한 이미지는 흐림·노출 편차가 크기 때문에 이 점검이 특히 중요하다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

# 임계값 (경험적 기본값 — 데이터 특성에 맞춰 조정 필요)
BLUR_THRESHOLD = 100.0      # Laplacian 분산이 이보다 낮으면 흐림 의심
DARK_THRESHOLD = 40.0       # 평균 밝기
BRIGHT_THRESHOLD = 215.0
MIN_SIDE = 64               # 짧은 변 최소 픽셀


@dataclass(frozen=True)
class ImageQuality:
    """단일 이미지의 품질 측정 결과."""

    width: int
    height: int
    channels: int
    blur_score: float
    brightness: float
    flags: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        """치명적 문제(해상도 부족)가 없으면 True."""
        return "low_resolution" not in self.flags

    def to_dict(self) -> dict:
        data = asdict(self)
        data["flags"] = ",".join(self.flags)
        return data


def decode(data: bytes) -> np.ndarray | None:
    """바이트열을 BGR 이미지로 디코드한다. 실패 시 None."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image


def read(path: Path) -> np.ndarray | None:
    """파일을 BGR 이미지로 읽는다. 한글 경로에서도 동작하도록 바이트로 읽는다."""
    try:
        return decode(Path(path).read_bytes())
    except OSError:
        return None


def assess_array(image: np.ndarray) -> ImageQuality:
    """BGR 이미지 배열의 품질 지표를 계산한다."""
    height, width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    flags: list[str] = []
    if min(width, height) < MIN_SIDE:
        flags.append("low_resolution")
    if blur_score < BLUR_THRESHOLD:
        flags.append("blurry")
    if brightness < DARK_THRESHOLD:
        flags.append("underexposed")
    elif brightness > BRIGHT_THRESHOLD:
        flags.append("overexposed")

    return ImageQuality(
        width=width,
        height=height,
        channels=channels,
        blur_score=round(blur_score, 2),
        brightness=round(brightness, 2),
        flags=tuple(flags),
    )


def assess_file(path: Path) -> ImageQuality | None:
    """파일 경로로 품질을 측정한다. 디코드 실패 시 None."""
    image = read(path)
    if image is None:
        return None
    return assess_array(image)


FLAG_KO = {
    "low_resolution": "해상도 부족",
    "blurry": "흐림",
    "underexposed": "노출 부족(어두움)",
    "overexposed": "노출 과다(밝음)",
}


def describe_flags(flags: str | tuple[str, ...]) -> str:
    """플래그를 한글 설명 문자열로 변환한다."""
    items = flags.split(",") if isinstance(flags, str) else list(flags)
    items = [f for f in items if f]
    if not items:
        return "양호"
    return ", ".join(FLAG_KO.get(f, f) for f in items)
