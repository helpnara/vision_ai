"""이미지 시각화 보조 함수.

Streamlit은 RGB 배열을 기대하지만 OpenCV는 BGR을 쓴다. 이 변환과 오버레이 그리기를
한 곳에 모아 두어 이후 단계(결함 점수 히트맵 등)에서도 재사용한다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROI_COLOR = (255, 64, 64)  # RGB — 결함 위치 표시색


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
