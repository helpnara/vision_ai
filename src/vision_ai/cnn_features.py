"""사전학습 CNN 특징 추출 (선택 사항).

## 왜 필요한가

고전 CV 특징으로 VisA PCB 평균 AUROC 0.830이 나왔는데, 공개 벤치마크는 사전학습 특징 기반이
0.95 내외다. pcb3 진단에서도 "고전 특징이 결함의 상당수를 표현하지 못한다"가 원인으로 나왔다.

## 왜 PyTorch를 쓰지 않는가

PyTorch는 PyPI 기본 설치가 약 1.7GB라 무료 배포 환경에 부담이 크다. 여기서는 이미 설치된
**OpenCV의 `cv2.dnn`으로 ONNX 모델을 읽는다.** 새 의존성이 없다.

## OpenCV 5의 함정 (직접 확인한 것)

`net.forward("레이어이름")`이 **그 이름을 무시한다.** 초기 레이어(`resnetv15_relu0_fwd`,
정답 (1,64,64,64))를 요청해도 마지막 특징맵 (1,512,8,8)을 돌려준다. 이름이 존재하는지만
확인하고 실제로는 항상 마지막 특징을 낸다.

- 이름을 **안 주면** 최종 분류 출력 (1,1000)이 나온다.
- 유효한 중간 이름을 **주면** 마지막 특징맵 (1,512,8,8)이 나온다.

그래서 원하는 층을 골라 쓸 수 없고, 쓸 수 있는 것은 마지막 특징맵뿐이다. 이 동작에 기대는
것이 위태로우므로 **로드할 때 출력 모양을 검사**하고, 기대와 다르면 즉시 알린다.

## 차원 축소가 필요한 이유

특징이 512차원인데 위치별 공분산을 추정하려면 표본이 차원보다 충분히 많아야 한다. 학습
이미지가 보통 수백 장이라 그대로는 특이행렬이 된다. PaDiM이 쓰는 방식대로 **차원을 무작위로
골라 줄인다.** 시드를 고정해 학습과 추론이 같은 차원을 쓴다.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import config

# ONNX Model Zoo (Apache-2.0). Git LFS 실체 파일 경로다.
MODEL_URL = (
    "https://media.githubusercontent.com/media/onnx/models/main/validated/"
    "vision/classification/resnet/model/resnet18-v1-7.onnx"
)
MODEL_NAME = "resnet18-v1-7.onnx"
MODEL_SIZE_MB = 45

# ImageNet 전처리 (ResNet v1 계열 기준)
INPUT_SIZE = 256
MEAN = (123.68, 116.78, 103.94)
SCALE = 1 / 58.0

# 유효한 중간 텐서 이름. 위 함정 때문에 **어느 이름을 줘도 마지막 특징맵이 나오지만**,
# 이름을 주지 않으면 분류 출력이 나오므로 반드시 하나는 넘겨야 한다.
SPATIAL_LAYER = "resnetv15_stage2_activation1"
EXPECTED_CHANNELS = 512
EXPECTED_GRID = 8

# 위치별 공분산을 추정할 수 있도록 줄일 차원 수 (PaDiM 방식).
REDUCED_DIM = 100
REDUCTION_SEED = 0

ProgressCallback = Callable[[int, int], None]


def model_path() -> Path:
    return config.MODEL_DIR / "onnx" / MODEL_NAME


def available() -> bool:
    path = model_path()
    return path.exists() and path.stat().st_size > 1_000_000


def download(progress: ProgressCallback | None = None) -> Path:
    """모델 파일을 내려받는다 (약 45MB).

    저장소에 넣기엔 크고 재생성 가능한 파일이라 `artifacts/`에 캐시한다.
    """
    path = model_path()
    if available():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")

    def hook(block: int, block_size: int, total: int) -> None:
        if progress is not None and total > 0:
            progress(min(block * block_size, total), total)

    try:
        urllib.request.urlretrieve(MODEL_URL, temporary, reporthook=hook)
    except (urllib.error.URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"모델을 내려받지 못했습니다: {exc}. 인터넷 연결이 막혀 있으면 "
            f"{MODEL_URL} 를 직접 받아 {path} 에 두어도 됩니다."
        ) from exc
    temporary.replace(path)
    return path


@dataclass
class Extractor:
    """ONNX 모델로 이미지 한 장에서 위치별 특징 격자를 뽑는다."""

    net: object
    dims: np.ndarray
    grid: int
    channels: int

    @property
    def n_positions(self) -> int:
        return self.grid * self.grid

    @property
    def dim(self) -> int:
        return len(self.dims)

    def patch_grid(self, rgb: np.ndarray) -> np.ndarray:
        """(위치 수, 축소 차원) 배열. `features.patch_features`와 같은 모양이다."""
        import cv2

        blob = cv2.dnn.blobFromImage(rgb, SCALE, (INPUT_SIZE, INPUT_SIZE), MEAN, swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward(SPATIAL_LAYER)
        if out.ndim != 4:
            raise RuntimeError(
                f"특징맵이 아니라 {out.shape} 이 나왔습니다. OpenCV 동작이 바뀌었을 수 있습니다."
            )
        # (1, C, H, W) → (H*W, C) → 차원 축소
        flat = out[0].reshape(out.shape[1], -1).T
        return flat[:, self.dims]


def load(path: Path | str | None = None) -> Extractor:
    """모델을 읽고 **출력 모양이 기대와 맞는지 확인한다.**

    OpenCV의 중간 레이어 처리에 기대고 있으므로, 조용히 다른 것이 나오면 성능이 이상해질 뿐
    원인을 알 수 없다. 로드할 때 한 번 확인해 두면 그때 바로 드러난다.
    """
    import cv2

    path = Path(path) if path is not None else model_path()
    if not Path(path).exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {path}. 먼저 내려받으세요.")

    net = cv2.dnn.readNetFromONNX(str(path))
    probe = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(probe, SCALE, (INPUT_SIZE, INPUT_SIZE), MEAN, swapRB=True)
    net.setInput(blob)
    out = net.forward(SPATIAL_LAYER)

    if out.ndim != 4 or out.shape[1] != EXPECTED_CHANNELS:
        raise RuntimeError(
            f"기대한 특징맵 (1,{EXPECTED_CHANNELS},{EXPECTED_GRID},{EXPECTED_GRID}) 대신 "
            f"{out.shape} 이 나왔습니다. 사전학습 특징을 쓸 수 없습니다."
        )

    channels, grid = int(out.shape[1]), int(out.shape[2])
    rng = np.random.default_rng(REDUCTION_SEED)
    dims = np.sort(rng.choice(channels, size=min(REDUCED_DIM, channels), replace=False))
    return Extractor(net=net, dims=dims, grid=grid, channels=channels)


def describe() -> str:
    """화면에 띄울 현재 상태 한 줄."""
    if available():
        size = model_path().stat().st_size / 1024**2
        return f"사전학습 모델 준비됨 (ResNet18, {size:.0f}MB)"
    return f"사전학습 모델이 없습니다 (약 {MODEL_SIZE_MB}MB 내려받기 필요)"
