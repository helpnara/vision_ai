"""3단계 전처리와 특징 추출.

딥러닝 스택 없이도 동작하는 것을 우선했다. 표면 결함은 국소적인 밝기·엣지·텍스처
이상으로 나타나므로, 고전 CV 통계량만으로도 쓸 만한 하한선을 만들 수 있다.

- `image_features`  : 이미지 1장 → 고정 길이 벡터 (베이스라인 분류기 입력)
- `patch_features`  : 이미지 1장 → 격자별 특징 (이상탐지 · 위치 추정 입력)
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# 모든 이미지를 같은 크기로 맞춘다. 격자 정렬이 어긋나면 위치별 이상탐지가 성립하지 않는다.
IMAGE_SIZE = 256
PATCH_SIZE = 16
PATCH_STRIDE = 8

_HIST_BINS = 16
_LBP_BINS = 16


def preprocess(rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB 이미지를 정사각형으로 리사이즈한다 (종횡비는 무시하고 격자 정렬을 우선)."""
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    if rgb.shape[0] == size and rgb.shape[1] == size:
        return rgb
    interp = cv2.INTER_AREA if rgb.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(rgb, (size, size), interpolation=interp)


# --- 이미지 단위 특징 ------------------------------------------------------

def _lbp_codes(gray: np.ndarray) -> np.ndarray:
    """8-이웃 LBP 코드 (텍스처 기술자). skimage 없이 numpy로 계산한다."""
    center = gray[1:-1, 1:-1]
    code = np.zeros(center.shape, dtype=np.uint8)
    offsets = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
    height, width = gray.shape
    for bit, (dy, dx) in enumerate(offsets):
        neighbor = gray[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
        code |= ((neighbor >= center).astype(np.uint8) << bit)
    return code


def _percentiles(values: np.ndarray, qs: tuple[float, ...]) -> list[float]:
    return [float(v) for v in np.percentile(values, qs)]


def image_features(rgb: np.ndarray) -> np.ndarray:
    """이미지 1장에서 고정 길이 특징 벡터를 만든다."""
    rgb = preprocess(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    grayf = gray.astype(np.float32)

    values: list[float] = []

    # 밝기 분포
    values.append(float(grayf.mean()))
    values.append(float(grayf.std()))
    values.extend(_percentiles(grayf, (1, 5, 25, 50, 75, 95, 99)))

    # 선명도 / 엣지
    laplacian = cv2.Laplacian(grayf, cv2.CV_32F)
    values.append(float(laplacian.var()))
    values.append(float(np.abs(laplacian).mean()))
    values.append(float(np.percentile(np.abs(laplacian), 99)))

    sobel = cv2.magnitude(
        cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3),
    )
    values.append(float(sobel.mean()))
    values.append(float(sobel.std()))
    values.append(float(np.percentile(sobel, 95)))

    edges = cv2.Canny(gray, 50, 150)
    values.append(float((edges > 0).mean()))

    # 색상
    for channel in range(3):
        plane = rgb[:, :, channel].astype(np.float32)
        values.append(float(plane.mean()))
        values.append(float(plane.std()))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    for channel in range(3):
        values.append(float(hsv[:, :, channel].mean()))
        values.append(float(hsv[:, :, channel].std()))

    # 고주파 성분 (블러와의 차이) — 미세 스크래치가 여기에 잡힌다
    residual = np.abs(grayf - cv2.GaussianBlur(grayf, (0, 0), 2.0))
    values.append(float(residual.mean()))
    values.append(float(residual.std()))
    values.append(float(np.percentile(residual, 99)))

    # 국소 이상 비율 — 중앙값에서 크게 벗어난 픽셀 비중
    median = float(np.median(grayf))
    deviation = np.abs(grayf - median)
    values.append(float((deviation > 25).mean()))
    values.append(float((deviation > 50).mean()))

    # 구조성 (행/열 평균 프로파일의 변동) — PCB처럼 규칙적인 패턴에서 유용
    values.append(float(grayf.mean(axis=0).std()))
    values.append(float(grayf.mean(axis=1).std()))

    # 밝기 히스토그램
    hist = cv2.calcHist([gray], [0], None, [_HIST_BINS], [0, 256]).ravel()
    values.extend((hist / max(hist.sum(), 1.0)).tolist())

    # LBP 텍스처 히스토그램
    codes = _lbp_codes(gray)
    lbp_hist = np.bincount(codes.ravel() // (256 // _LBP_BINS), minlength=_LBP_BINS)
    values.extend((lbp_hist / max(lbp_hist.sum(), 1)).astype(float).tolist())

    return np.asarray(values, dtype=np.float32)


def _build_feature_names() -> tuple[str, ...]:
    names = [
        "gray_mean", "gray_std",
        "gray_p01", "gray_p05", "gray_p25", "gray_p50", "gray_p75", "gray_p95", "gray_p99",
        "lap_var", "lap_absmean", "lap_p99",
        "sobel_mean", "sobel_std", "sobel_p95",
        "canny_density",
    ]
    names += [f"{c}_{s}" for c in "rgb" for s in ("mean", "std")]
    names += [f"{c}_{s}" for c in ("h", "s", "v") for s in ("mean", "std")]
    names += ["hf_mean", "hf_std", "hf_p99", "dev_gt25", "dev_gt50", "col_profile_std", "row_profile_std"]
    names += [f"hist_{i:02d}" for i in range(_HIST_BINS)]
    names += [f"lbp_{i:02d}" for i in range(_LBP_BINS)]
    return tuple(names)


FEATURE_NAMES: tuple[str, ...] = _build_feature_names()


# --- 패치 단위 특징 (이상탐지 · 위치 추정) --------------------------------

PATCH_PLANES: tuple[str, ...] = ("gray", "laplacian", "sobel", "hue", "r", "g", "b")
PATCH_FEATURE_DIM = len(PATCH_PLANES) * 2  # 평면별 평균 + 표준편차


def patch_features(
    rgb: np.ndarray, patch: int = PATCH_SIZE, stride: int = PATCH_STRIDE
) -> np.ndarray:
    """격자별 특징을 (격자높이, 격자너비, PATCH_FEATURE_DIM) 배열로 반환한다."""
    rgb = preprocess(rgb)
    grayf = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    laplacian = np.abs(cv2.Laplacian(grayf, cv2.CV_32F))
    sobel = cv2.magnitude(
        cv2.Sobel(grayf, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(grayf, cv2.CV_32F, 0, 1, ksize=3),
    )
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[:, :, 0].astype(np.float32)
    planes = (
        grayf, laplacian, sobel, hue,
        rgb[:, :, 0].astype(np.float32),
        rgb[:, :, 1].astype(np.float32),
        rgb[:, :, 2].astype(np.float32),
    )

    stacked = []
    for plane in planes:
        windows = sliding_window_view(plane, (patch, patch))[::stride, ::stride]
        stacked.append(windows.mean(axis=(-1, -2)))
        stacked.append(windows.std(axis=(-1, -2)))
    return np.stack(stacked, axis=-1).astype(np.float32)


def patch_grid_shape(
    size: int = IMAGE_SIZE, patch: int = PATCH_SIZE, stride: int = PATCH_STRIDE
) -> tuple[int, int]:
    """격자 크기를 계산한다."""
    side = (size - patch) // stride + 1
    return side, side


def patch_centers(
    size: int = IMAGE_SIZE, patch: int = PATCH_SIZE, stride: int = PATCH_STRIDE
) -> tuple[np.ndarray, np.ndarray]:
    """각 격자 칸의 중심 좌표(y, x)를 반환한다 — 히트맵을 원본 좌표로 되돌릴 때 쓴다."""
    rows, cols = patch_grid_shape(size, patch, stride)
    ys = np.arange(rows) * stride + patch // 2
    xs = np.arange(cols) * stride + patch // 2
    return ys, xs


def upsample_grid(grid: np.ndarray, size: int = IMAGE_SIZE, smooth: float = 4.0) -> np.ndarray:
    """격자 점수를 이미지 크기 히트맵으로 확대한다."""
    heatmap = cv2.resize(grid.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    if smooth > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), smooth)
    return heatmap
