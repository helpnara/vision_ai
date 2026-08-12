"""이미지 시각화 보조 함수.

Streamlit은 RGB 배열을 기대하지만 OpenCV는 BGR을 쓴다. 이 변환과 오버레이 그리기를
한 곳에 모아 두어 이후 단계(결함 점수 히트맵 등)에서도 재사용한다.
"""

from __future__ import annotations

import functools
from pathlib import Path

import cv2
import numpy as np

ROI_COLOR = (255, 64, 64)  # RGB — 결함 위치 표시색

# 그림 위에 한글을 얹으려면 글꼴 파일이 필요하다. OpenCV가 들고 있는 Hershey 글꼴에는
# 한글 자모가 없어서 `putText`로는 네모만 찍힌다. 아래 후보를 순서대로 찾아보고 하나도
# 없으면 한글을 포기하고 영문 대체 문구를 찍는다 — 글꼴이 없다고 그림을 못 만들면 안 된다.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",            # macOS
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",                            # Windows
)

# 파일이 있다고 한글이 나오는 것은 아니다. **CJK라는 이름에 속으면 안 된다** — 이 환경에
# 깔린 일본어 고딕은 한자는 그리지만 한글 자리에는 네모(두부 글자)를 찍는다. 그림에 네모가
# 줄줄이 찍힌 뒤에야 알게 되므로, 쓰기 전에 «없는 글자»와 같은 모양인지 대 본다.
_MISSING_CODEPOINT = "\U000FFFFD"  # 사용자 정의 영역 — 어떤 글꼴에도 없다
_PROBE_CHAR = "가"


def to_rgb(image: np.ndarray) -> np.ndarray:
    """BGR(또는 흑백) 이미지를 RGB로 변환한다."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_rgb(path: Path) -> np.ndarray | None:
    """파일을 RGB 배열로 읽는다. 실패 시 None."""
    try:
        data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if image is None else to_rgb(image)


def draw_roi(
    rgb: np.ndarray,
    roi: tuple[int, int, int, int] | None,
    *,
    color: tuple[int, int, int] = ROI_COLOR,
    label: str = "",
) -> np.ndarray:
    """RGB 이미지에 ROI 사각형을 그려 새 배열로 반환한다."""
    if roi is None:
        return rgb
    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        return rgb

    out = rgb.copy()
    thickness = max(1, round(min(out.shape[:2]) / 200))
    cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    if label:
        scale = max(0.4, min(out.shape[:2]) / 500)
        cv2.putText(
            out, label, (x, max(int(scale * 24), y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA,
        )
    return out


def overlay_mask(
    rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int] = ROI_COLOR, alpha: float = 0.4
) -> np.ndarray:
    """결함 마스크를 반투명하게 덮어 표시한다."""
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    weight = (mask > 0).astype(np.float32)[:, :, None] * alpha
    tint = np.full_like(rgb, color, dtype=np.float32)
    blended = rgb.astype(np.float32) * (1 - weight) + tint * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def korean_font_path() -> str | None:
    """한글을 그릴 수 있는 글꼴 파일 경로. 없으면 None.

    글꼴을 찾는 일은 파일 존재 확인이라 싸지만, 프레임마다 하면 수백 번이 된다. 결과가
    바뀔 일이 없으므로 한 번만 찾는다.
    """
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists() and draws_hangul(candidate):
            return candidate
    return None


def draws_hangul(font_path: str) -> bool:
    """이 글꼴이 한글을 실제로 그리는가. 두부 글자면 False."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    try:
        font = ImageFont.truetype(font_path, 24)
    except OSError:
        return False

    def ink(char: str) -> np.ndarray:
        canvas = Image.new("L", (48, 48), 0)
        ImageDraw.Draw(canvas).text((2, 2), char, font=font, fill=255)
        return np.asarray(canvas)

    return not np.array_equal(ink(_PROBE_CHAR), ink(_MISSING_CODEPOINT))


def put_text(
    rgb: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = ROI_COLOR,
    size: int = 20,
    ascii_fallback: str = "",
) -> np.ndarray:
    """이미지 위에 글자를 얹는다. 한글이면 글꼴을 찾아 쓰고, 없으면 영문으로 대체한다.

    `origin`은 글자 상자의 **왼쪽 위** 모서리다 — OpenCV의 `putText`가 쓰는 기준선(baseline)이
    아니다. 판정 띠처럼 «여기부터 아래로» 그리는 자리가 대부분이라 이 편이 계산이 쉽다.
    """
    if not text:
        return rgb
    font_path = korean_font_path()
    if font_path is None:
        legible = ascii_fallback or text.encode("ascii", "replace").decode("ascii")
        out = rgb.copy()
        scale = size / 30.0
        cv2.putText(
            out, legible, (origin[0], origin[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, max(1, round(scale * 2)), cv2.LINE_AA,
        )
        return out

    from PIL import Image, ImageDraw  # 지연 임포트 — 글꼴이 있을 때만 필요하다

    image = Image.fromarray(rgb)
    ImageDraw.Draw(image).text(origin, text, font=_font(font_path, size), fill=color)
    return np.asarray(image)


@functools.lru_cache(maxsize=8)
def _font(path: str, size: int):
    """글꼴 파일을 크기별로 한 번만 연다 — 프레임마다 다시 열면 그 자체가 비용이다."""
    from PIL import ImageFont

    return ImageFont.truetype(path, size)


def crop(rgb: np.ndarray, roi: tuple[int, int, int, int], *, margin: int = 0) -> np.ndarray:
    """ROI 영역을 잘라낸다 (3단계 Claude 2차 판정용 크롭에 사용)."""
    height, width = rgb.shape[:2]
    x, y, w, h = (int(v) for v in roi)
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(width, x + w + margin)
    y1 = min(height, y + h + margin)
    if x1 <= x0 or y1 <= y0:
        return rgb
    return rgb[y0:y1, x0:x1]
