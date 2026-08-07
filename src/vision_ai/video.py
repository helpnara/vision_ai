"""영상에서 학습용 프레임을 뽑는다.

설계: `docs/video-frame-extraction-plan.md`

## 이 모듈이 답하려는 질문

"몇 장을 뽑아야 하는가." 영상은 프레임이 넘쳐나서(30fps 10분 = 18,000장) 전부 뽑는 것도,
감으로 "10프레임마다" 정하는 것도 답이 아니다. 두 방향에서 동시에 틀릴 수 있다 —
적게 뽑으면 결함 물체가 한 장도 안 찍히고, 많이 뽑으면 라벨링 비용만 늘고 정보량은
그대로다(게다가 인접 프레임이 학습·평가에 나뉘어 성능이 부풀려진다).

그래서 뽑는 간격을 정하는 방법을 세 가지 둔다.

* **공정 값으로 계산** — 컨베이어처럼 라인 속도를 아는 경우. `plan_from_process()`
* **초당 장수 지정** — 조건을 모르거나 감이 있는 경우. `plan_from_rate()`
* **장면이 바뀔 때만** — CCTV처럼 물체가 멈춰 있기도 하고 속도도 제각각인 경우.
  `extract(..., similarity=...)`

CCTV가 주 입력이면 세 번째가 주가 된다. 고정 카메라 앞을 사람·차량이 불규칙하게 지나므로
"물체가 화면을 지나는 시간"을 정할 수가 없다.

## 왜 grab()과 retrieve()를 나눠 쓰는가

건너뛸 프레임까지 디코딩하면 그만큼 그냥 버린다. `grab()`은 디코딩 없이 다음 프레임으로
넘어가고, `retrieve()`가 실제로 디코딩한다. 720p 600프레임 실측에서 전부 디코딩 0.49초,
건너뛰며 뽑기 0.20초였다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import ceil, floor
from pathlib import Path
from typing import Callable

import numpy as np

from . import config, quality, storage

VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg"})

DEFAULT_FRAMES_PER_OBJECT = 3
"""물체 하나당 확보할 장수. 흔들림·가림에 대비한 여유까지 포함한 값."""

DEFAULT_SIMILARITY = 0.02
"""직전 채택 프레임과의 평균 절대 차이(0~1). 이보다 비슷하면 버린다.

컨베이어가 멈춘 구간, CCTV의 빈 장면에서는 같은 그림이 쏟아진다. 균등 추출만으로는
그것을 거를 수 없다.
"""

THUMB = 32
"""비슷한지 볼 때 쓰는 축소 크기. 버릴 프레임에 67차원 특징을 쓰는 것은 과하다."""

JPEG_QUALITY = 92

ProgressCallback = Callable[[int, int], None]


# --- 영상 정보 --------------------------------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    @property
    def usable(self) -> bool:
        return self.fps > 0 and self.frame_count > 0


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def probe(path) -> VideoInfo:
    """영상 메타데이터를 읽는다. 열 수 없으면 OSError."""
    import cv2

    path = Path(path).expanduser()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"영상을 열 수 없습니다: {path}")
    try:
        return VideoInfo(
            path=path,
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        capture.release()


# --- 몇 장을 뽑을 것인가 -----------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """추출 계획과 그 근거."""

    stride: int
    expected_frames: int
    per_second: float
    frames_per_object: float = 0.0   # 이 설정으로 물체 하나가 몇 장 찍히는가
    required_fps: float = 0.0        # 목표를 채우려면 필요한 최소 fps
    feasible: bool = True            # 지금 카메라로 목표를 채울 수 있는가
    reason: str = ""


def plan_from_rate(info: VideoInfo, per_second: float) -> Plan:
    """"초당 n장"으로 간격을 정한다. 조건을 모를 때 쓰는 가장 단순한 방법."""
    if not info.usable:
        raise ValueError("프레임 수나 fps를 읽을 수 없는 영상입니다.")
    if per_second <= 0:
        raise ValueError("초당 장수는 0보다 커야 합니다.")

    stride = max(1, round(info.fps / per_second))
    return Plan(
        stride=stride,
        expected_frames=ceil(info.frame_count / stride),
        per_second=info.fps / stride,
        reason=f"초당 약 {info.fps / stride:.1f}장 ({stride}프레임마다 1장)",
    )


def plan_from_process(
    info: VideoInfo,
    *,
    field_of_view_m: float,
    speed_mps: float,
    frames_per_object: int = DEFAULT_FRAMES_PER_OBJECT,
) -> Plan:
    """컨베이어 조건에서 간격을 계산한다.

    물체 하나가 화면을 지나는 시간 `t = 시야 길이 / 라인 속도` 동안 `fps × t` 프레임이
    찍힌다. 그중 `frames_per_object`장만 남기면 되므로 간격은 `fps × t / 장수`다.

    **이 계산의 값어치는 숫자보다 "이 촬영이 애초에 가능한가"를 알려주는 데 있다.**
    같은 식을 뒤집으면 최소 요구 fps가 나온다. 라인이 빠르면 지금 카메라로는 물체당
    한두 장밖에 못 건지는데, 그건 추출 설정이 아니라 촬영 계획의 문제이고 프레임을 뽑기
    전에 알아야 비용을 아낀다.
    """
    if not info.usable:
        raise ValueError("프레임 수나 fps를 읽을 수 없는 영상입니다.")
    if field_of_view_m <= 0 or speed_mps <= 0:
        raise ValueError("시야 길이와 라인 속도는 0보다 커야 합니다.")
    if frames_per_object < 1:
        raise ValueError("물체당 장수는 1 이상이어야 합니다.")

    transit = field_of_view_m / speed_mps            # 물체가 화면에 머무는 시간(초)
    captured = info.fps * transit                    # 그동안 찍히는 프레임 수
    required_fps = frames_per_object / transit
    stride = max(1, floor(captured / frames_per_object))
    feasible = captured >= frames_per_object

    if feasible:
        reason = (
            f"물체 하나가 {transit:.2f}초 동안 {captured:.1f}프레임 찍힙니다. "
            f"그중 {frames_per_object}장을 남기려면 {stride}프레임마다 1장입니다."
        )
    else:
        reason = (
            f"지금 카메라로는 물체당 {captured:.1f}장뿐입니다 — 목표 {frames_per_object}장을 "
            f"채우려면 **{required_fps:.0f}fps 이상**이 필요합니다. "
            "라인을 늦추거나 더 빠른 카메라가 필요합니다."
        )

    return Plan(
        stride=stride,
        expected_frames=ceil(info.frame_count / stride),
        per_second=info.fps / stride,
        frames_per_object=captured,
        required_fps=required_fps,
        feasible=feasible,
        reason=reason,
    )


# --- 추출 -------------------------------------------------------------------

@dataclass
class ExtractResult:
    """무엇을 뽑았고 무엇을 왜 버렸는지.

    버린 것을 조용히 넘기면 "3,000장이라더니 왜 2,700장이지?"가 된다.
    """

    video_id: str = ""
    saved: list[Path] = field(default_factory=list)
    scanned: int = 0
    dropped_similar: int = 0
    dropped_quality: int = 0

    @property
    def kept(self) -> int:
        return len(self.saved)

    def as_message(self) -> str:
        parts = [f"{self.kept:,}장 추출"]
        if self.dropped_similar:
            parts.append(f"직전과 거의 같아 {self.dropped_similar:,}장 제외")
        if self.dropped_quality:
            parts.append(f"흐리거나 노출이 나빠 {self.dropped_quality:,}장 제외")
        return " · ".join(parts)


def video_id(path) -> str:
    """영상 하나를 가리키는 짧은 이름. 프레임 폴더 이름이자 분할 그룹 열쇠가 된다."""
    stem = re.sub(r"[^0-9A-Za-z가-힣_-]+", "-", Path(path).stem).strip("-")
    digest = storage.sha1_of_file(Path(path))[:8]
    return f"{stem[:40]}-{digest}" if stem else digest


def _thumb(frame) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def extract(
    path,
    *,
    stride: int,
    out_dir: Path | None = None,
    similarity: float | None = DEFAULT_SIMILARITY,
    check_quality: bool = False,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> ExtractResult:
    """영상에서 프레임을 뽑아 이미지 파일로 저장한다.

    Args:
        stride: 몇 프레임마다 한 장을 남길지.
        out_dir: 저장 폴더. 기본은 `raw/video/<video_id>/`.
        similarity: 직전 채택 프레임과의 차이가 이보다 작으면 버린다. None이면 끈다.
        check_quality: 흐림·노출 기준으로 거를지. **기본은 끈다** — 영상 프레임은 정지
            이미지보다 전반적으로 흐려서, 정지 이미지용 기준을 그대로 걸면 과하게 버린다.
            먼저 분포를 보고 사용자가 정하는 편이 낫다.
        limit: 최대 저장 장수 (미리보기용).

    메모리에 모아 두지 않고 **한 장씩 바로 저장한다.** 1080p 프레임 하나가 비압축 6MB라
    3,000장을 들고 있으면 18GB다.
    """
    import cv2

    path = Path(path).expanduser()
    stride = max(1, int(stride))
    identifier = video_id(path)
    out_dir = Path(out_dir) if out_dir else config.raw_dir() / "video" / identifier
    out_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"영상을 열 수 없습니다: {path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    result = ExtractResult(video_id=identifier)
    previous: np.ndarray | None = None
    index = 0

    try:
        while True:
            if not capture.grab():        # 디코딩 없이 다음 프레임으로
                break
            if index % stride == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    result.scanned += 1
                    keep, previous = _decide(frame, previous, similarity, check_quality, result)
                    if keep:
                        target = out_dir / f"{index:08d}.jpg"
                        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        result.saved.append(target)
                        if limit is not None and result.kept >= limit:
                            break
            index += 1
            if progress is not None and total:
                progress(min(index, total), total)
    finally:
        capture.release()

    if progress is not None and total:
        progress(total, total)
    return result


def _decide(frame, previous, similarity, check_quality, result) -> tuple[bool, np.ndarray | None]:
    """이 프레임을 남길지 정하고, 다음 비교에 쓸 축소본을 돌려준다."""
    thumb = _thumb(frame)
    if similarity is not None and previous is not None:
        if float(np.abs(thumb - previous).mean()) < similarity:
            result.dropped_similar += 1
            return False, previous          # 버린 프레임은 기준으로 삼지 않는다

    if check_quality:
        import cv2

        metrics = quality.assess_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if metrics.flags:
            result.dropped_quality += 1
            return False, previous

    return True, thumb


# --- 시험용 영상 만들기 ------------------------------------------------------

def make_sample(
    path: Path | None = None,
    *,
    seconds: int = 10,
    fps: int = 30,
    size: tuple[int, int] = (640, 480),
    with_defect: bool = True,
    seed: int = 7,
) -> Path:
    """물체가 지나가는 시험용 영상을 만든다.

    영상 파일이 없으면 이 기능을 시험조차 할 수 없다. 배포본은 컨테이너가 일회성이라
    올려 둔 영상도 남지 않는다. 1단계의 **합성 샘플 생성**과 같은 이유로, 다운로드 없이
    바로 돌려볼 수 있는 입력을 만들어 둔다.

    **성능 근거로 쓸 수 없다.** 배선과 조작을 확인하는 용도다.
    """
    import cv2

    path = Path(path) if path else config.interim_dir() / "video" / "sample.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise OSError("영상을 만들 수 없습니다 (코덱을 쓸 수 없음).")

    rng = np.random.default_rng(seed)
    period = fps * 3                      # 3초에 하나씩 지나간다
    try:
        for index in range(fps * seconds):
            frame = np.full((height, width, 3), 45, np.uint8)
            cv2.rectangle(frame, (0, height * 5 // 8), (width, height * 5 // 8 + 40), (70, 70, 70), -1)
            x = int(((index % period) / period) * (width + 120)) - 120
            top, bottom = height * 3 // 8, height * 5 // 8
            cv2.rectangle(frame, (x, top), (x + 120, bottom), (150, 170, 190), -1)
            if with_defect and (index // period) % 2 == 1:
                cv2.line(frame, (x + 30, top + 30), (x + 90, bottom - 30), (40, 40, 40), 3)
            writer.write(cv2.add(frame, rng.integers(0, 8, frame.shape, dtype=np.uint8)))
    finally:
        writer.release()
    return path
