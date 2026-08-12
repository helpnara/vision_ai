"""판정 결과를 영상으로 되돌려 재생한다 (V7).

«재현율 0.82»는 다음에 무엇을 할지 알려주지 않는다. **어디서 놓치고 어디서 헛짚는지**가
알려준다. 그래서 테스트 영상을 프레임마다 판정하고, 그 판정을 영상 위에 그려 넣은
**판정본**을 따로 만들어 그 자리에서 재생한다.

원본은 건드리지 않는다 — 판정본은 별도 폴더에 새 파일로 쓴다. 판정 기준을 바꿔 다시
돌리는 일이 잦은데, 원본을 덮으면 되돌릴 방법이 없다.

**이상탐지 모델로도 만들 수 있다.** 열지도를 판정 기준선에서 자르고 연결 요소를 박스로
바꾸면 된다(마스크→박스에서 쓴 것과 같은 기법). 검출 모델을 학습하기 *전에도* 확인용
영상을 볼 수 있다는 뜻이다. 열지도와 박스는 **나란히** 붙여 보여 준다 — 박스만 보면
«왜 저기냐»를 알 수 없고, 열지도만 보면 «잡았다/놓쳤다»가 안 보인다.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import pandas as pd

from . import config, viz

ProgressCallback = Callable[[int, int], None]

PLAYBACK_DIR = "playback"
VIDEO_STEM = "judged"
FRAMES_FILE = "frames.csv"
META_FILE = "meta.json"
STILL_DIR = "stills"

# 브라우저가 재생할 수 있는 코덱만 쓸모가 있다. 여기 OpenCV 빌드에는 H.264 인코더가
# 없고(라이선스 문제로 보통 빠져 있다), 대신 있는 mp4v(MPEG-4 Part 2)는 크롬·파이어폭스가
# 재생하지 못한다. WebM/VP8은 인코더도 있고 브라우저도 전부 재생한다. mp4v는 그마저도
# 안 될 때를 위한 대비책이고, 그때는 «내려받아 보라»고 말해 줘야 한다.
CODECS = (
    ("VP80", ".webm", "video/webm", True),
    ("mp4v", ".mp4", "video/mp4", False),
)

DEFAULT_LIMIT = 150         # 기본으로 판정할 프레임 수 — 화면에서 늘릴 수 있다
MIN_BOX_FRACTION = 0.0002   # 프레임 넓이의 0.02% 미만 덩어리는 잡음으로 본다
MAX_BOXES = 8               # 열지도가 통째로 붉으면 박스 200개가 아니라 «전부»가 답이다
STILL_LIMIT = 4             # 아래에 붙일 대표 판정 결과 장수
FALLBACK_PERCENTILE = 99.5  # 기준선을 넘겼는데 열지도에는 넘긴 화소가 없을 때 쓰는 컷

# 열지도 색을 **프레임마다 다시 정규화하면 안 된다.** 정상 프레임의 미세한 얼룩이 결함과
# 똑같이 새빨갛게 나와서, 재생하면 화면이 계속 요동친다. 판정 기준선을 색 눈금의 한가운데에
# 고정하면 «붉으면 기준선 위»가 영상 내내 같은 뜻이 된다.
HEAT_LOW = 0.55             # 기준선의 55% → 색 눈금의 바닥
HEAT_HIGH = 1.45            # 기준선의 145% → 색 눈금의 꼭대기
HEAT_ALPHA = 0.7

DEFECT_COLOR = (255, 64, 64)     # RGB
NORMAL_COLOR = (72, 196, 128)
DIVIDER_COLOR = (24, 24, 24)
DIVIDER_PX = 4


# --- 프레임 하나의 판정 ------------------------------------------------------

@dataclass(frozen=True)
class Judged:
    """프레임 한 장에 내린 판정."""

    index: int                                   # 원본 영상에서의 프레임 번호
    seconds: float
    score: float
    threshold: float
    decision: str
    boxes: tuple[tuple[int, int, int, int], ...] = ()

    @property
    def defect(self) -> bool:
        return self.decision == config.LABEL_DEFECT

    @property
    def margin(self) -> float:
        """기준선을 얼마나 넘겼는가. 대표 장면을 고를 때의 순위 기준이다.

        점수 자체로 줄을 세우면 카테고리마다 기준선이 다를 때 비교가 어긋난다.
        """
        return self.score - self.threshold

    def caption(self) -> str:
        return (
            f"{self.seconds:.1f}초 · {config.LABEL_KO.get(self.decision, self.decision)} "
            f"(점수 {self.score:.3f} / 기준 {self.threshold:.3f})"
        )


@dataclass(frozen=True)
class Event:
    """결함으로 판정된 **연속 구간**.

    프레임 단위로 세면 «3장 잡음»이지만 사람이 보기에는 «한 군데서 잡음»이다. 대표 장면을
    고를 때 이 차이가 중요하다 — 안 그러면 같은 순간의 이웃 프레임 4장이 대표랍시고 나온다.
    """

    frames: tuple[Judged, ...]

    @property
    def peak(self) -> Judged:
        return max(self.frames, key=lambda f: f.margin)

    @property
    def start(self) -> float:
        return self.frames[0].seconds

    @property
    def end(self) -> float:
        return self.frames[-1].seconds

    def span(self) -> str:
        if len(self.frames) == 1:
            return f"{self.start:.1f}초"
        return f"{self.start:.1f}~{self.end:.1f}초"


# --- 만들어진 판정본 ---------------------------------------------------------

@dataclass
class Playback:
    """만들어 둔 판정본 한 벌 — 영상 파일 + 프레임별 판정 + 대표 장면."""

    directory: Path
    video: Path
    mime: str
    playable: bool                 # 브라우저가 바로 재생할 수 있는 코덱인가
    frames: list[Judged] = field(default_factory=list)
    stills: list[Path] = field(default_factory=list)
    fps: float = 10.0
    scanned: int = 0               # 원본에서 훑은 프레임 수
    version: str = ""
    source: str = ""                # 원본 영상의 전체 경로
    side_by_side: bool = False

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def source_name(self) -> str:
        return Path(self.source).name

    @property
    def defects(self) -> list[Judged]:
        return [f for f in self.frames if f.defect]

    @property
    def defect_rate(self) -> float:
        return len(self.defects) / len(self.frames) if self.frames else 0.0

    def events(self) -> list[Event]:
        """결함 판정이 이어진 구간들을 시간 순으로 묶는다."""
        found: list[Event] = []
        run: list[Judged] = []
        for judged in self.frames:
            if judged.defect:
                run.append(judged)
            elif run:
                found.append(Event(tuple(run)))
                run = []
        if run:
            found.append(Event(tuple(run)))
        return found

    def highlights(self, limit: int = STILL_LIMIT) -> list[Judged]:
        """아래에 붙일 **대표 판정 결과**.

        결함 구간이 있으면 구간마다 정점 한 장씩(세게 잡은 구간 순서로). 결함이 하나도
        없으면 «기준선에 가장 가까웠던» 프레임을 보여 준다 — 아무것도 안 잡혔다는 화면보다
        «이만큼까지 갔는데 못 넘었다»가 다음에 무엇을 할지 알려준다.
        """
        events = self.events()
        if events:
            peaks = sorted(events, key=lambda e: e.peak.margin, reverse=True)
            picked = [event.peak for event in peaks[:limit]]
            return sorted(picked, key=lambda f: f.index)
        near = sorted(self.frames, key=lambda f: f.margin, reverse=True)[:limit]
        return sorted(near, key=lambda f: f.index)

    def summary(self) -> str:
        if not self.frames:
            return "판정한 프레임이 없습니다."
        events = self.events()
        return (
            f"{len(self.frames)}장을 판정해 {len(self.defects)}장이 결함"
            f"({self.defect_rate:.0%}), 결함 구간 {len(events)}곳."
        )

    def note(self) -> str:
        """화면에 같이 띄울 주의 문구. 없으면 빈 문자열."""
        if not self.playable:
            return (
                "이 환경에는 브라우저가 재생할 수 있는 코덱이 없어 파일로만 만들었습니다. "
                "내려받아 재생해 주세요."
            )
        return ""

    def frame(self) -> pd.DataFrame:
        """프레임별 판정을 표로. 저장·비교·타임라인(V6)이 같은 표를 쓴다."""
        return pd.DataFrame(
            [
                {
                    "index": f.index,
                    "seconds": round(f.seconds, 3),
                    "score": f.score,
                    "threshold": f.threshold,
                    "decision": f.decision,
                    "boxes": len(f.boxes),
                }
                for f in self.frames
            ]
        )

    def save(self) -> None:
        """다시 만들지 않고도 되살릴 수 있게 판정 결과를 폴더에 남긴다.

        Streamlit은 위젯을 건드릴 때마다 스크립트를 처음부터 다시 돌린다. 판정본을
        메모리에만 들고 있으면 체크박스 하나 눌렀다고 몇 분짜리 작업이 사라진다.
        """
        self.frame().to_csv(self.directory / FRAMES_FILE, index=False)
        (self.directory / META_FILE).write_text(
            json.dumps(
                {
                    "video": self.video.name,
                    "mime": self.mime,
                    "playable": self.playable,
                    "fps": self.fps,
                    "scanned": self.scanned,
                    "version": self.version,
                    "source": self.source,
                    "side_by_side": self.side_by_side,
                    "stills": [p.name for p in self.stills],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def load(directory) -> Playback | None:
    """폴더에 남겨 둔 판정본을 되살린다. 온전하지 않으면 None — 반쯤 살아난 것보다 낫다."""
    directory = Path(directory)
    meta_path = directory / META_FILE
    frames_path = directory / FRAMES_FILE
    if not meta_path.exists() or not frames_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        table = pd.read_csv(frames_path)
    except (OSError, ValueError):
        return None

    video = directory / str(meta.get("video", ""))
    if not video.exists():
        return None

    frames = [
        Judged(
            index=int(row["index"]),
            seconds=float(row["seconds"]),
            score=float(row["score"]),
            threshold=float(row["threshold"]),
            decision=str(row["decision"]),
        )
        for _, row in table.iterrows()
    ]
    stills = [directory / STILL_DIR / name for name in meta.get("stills", [])]
    return Playback(
        directory=directory,
        video=video,
        mime=str(meta.get("mime", "video/webm")),
        playable=bool(meta.get("playable", True)),
        frames=frames,
        stills=[p for p in stills if p.exists()],
        fps=float(meta.get("fps", 10.0)),
        scanned=int(meta.get("scanned", len(frames))),
        version=str(meta.get("version", "")),
        source=str(meta.get("source", "")),
        side_by_side=bool(meta.get("side_by_side", False)),
    )


def find(source, version: str) -> Playback | None:
    """이 영상 × 이 버전으로 **이미 만들어 둔** 판정본을 찾는다. 없으면 None.

    화면을 새로 열 때마다 몇 분짜리 작업을 다시 시키면 아무도 두 번 쓰지 않는다. 판정본은
    폴더에 그대로 남아 있으므로 되찾아 오면 된다.

    폴더 이름으로 찾지 않고 **안에 적힌 것을 읽어서** 맞춘다. 폴더 이름은 영상 내용의 해시로
    짓는데, 그 해시를 내려면 매번 영상 전체를 읽어야 한다 — 위젯 하나 건드릴 때마다 수백
    MB를 다시 읽게 된다.
    """
    root = config.interim_dir() / PLAYBACK_DIR
    if not root.is_dir():
        return None
    wanted = str(source)
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        made = load(candidate)
        if made is not None and made.source == wanted and made.version == version:
            return made
    return None


# --- 열지도 → 박스 -----------------------------------------------------------

def cut_level(score_map: np.ndarray, threshold: float, *, defect: bool) -> float:
    """열지도를 어디서 자를지 정한다.

    기본은 **판정 기준선 그대로**다. 그래야 영상에 박스가 뜨는 순간과 «결함» 판정이 나는
    순간이 어긋나지 않는다 — 어긋나면 보는 사람이 둘 중 무엇을 믿어야 할지 모르게 된다.

    다만 이미지 점수를 평균이나 백분위로 집계하는 모델은 «기준선은 넘겼는데 기준선을 넘긴
    화소는 한 개도 없는» 상태가 될 수 있다. 그때는 가장 뜨거운 쪽을 잘라서 **어디를 보고
    그렇게 판정했는지**는 보여 준다. 박스 없는 결함 프레임은 아무것도 알려주지 않는다.
    """
    hottest = float(np.max(score_map)) if score_map.size else 0.0
    if defect and hottest < threshold:
        return float(np.percentile(score_map, FALLBACK_PERCENTILE))
    return float(threshold)


def boxes_from_score_map(
    score_map: np.ndarray,
    level: float,
    *,
    shape: tuple[int, int],
    min_area: int | None = None,
    max_boxes: int = MAX_BOXES,
) -> list[tuple[int, int, int, int]]:
    """열지도에서 기준선을 넘은 덩어리마다 박스를 하나씩 뽑는다.

    열지도는 모델이 보는 크기(보통 256×256)라 프레임 크기와 다르다. **먼저 프레임 크기로
    늘린 뒤** 자른다 — 작은 격자에서 자르고 좌표만 곱하면 박스가 격자 눈금에 딱 붙어서,
    실제 결함보다 크게도 작게도 나온다.
    """
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0 or score_map.size == 0:
        return []
    if min_area is None:
        min_area = max(9, int(height * width * MIN_BOX_FRACTION))

    full = cv2.resize(
        np.asarray(score_map, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    hot = (full >= level).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    found = [
        (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]),
        )
        for i in range(1, count)                       # 0번은 배경
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area
    ]
    found.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in found[:max_boxes]]


def heat_overlay(rgb: np.ndarray, score_map: np.ndarray, threshold: float) -> np.ndarray:
    """열지도를 색으로 바꿔 원본 위에 반투명하게 덮는다.

    색 눈금은 **판정 기준선에 고정**한다(HEAT_LOW/HIGH). 프레임마다 최대·최소로 정규화하면
    아무 일 없는 프레임도 새빨개져서, 재생하는 내내 «지금 뜨거운 건가 아닌가»를 알 수 없다.
    """
    height, width = rgb.shape[:2]
    full = cv2.resize(
        np.asarray(score_map, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    low, high = threshold * HEAT_LOW, threshold * HEAT_HIGH
    if high <= low:
        low, high = float(full.min()), float(full.max()) or 1.0
    scaled = np.clip((full - low) / max(high - low, 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    # 색을 **뜨거운 만큼만** 덮는다. 전면을 같은 세기로 덮으면 아무 일 없는 바닥까지 파랗게
    # 물들어, 왼쪽 원본과 대 보며 «저기가 뭐길래»를 따질 수가 없다.
    weight = (scaled[:, :, None] * HEAT_ALPHA).astype(np.float32)
    blended = rgb.astype(np.float32) * (1 - weight) + colored.astype(np.float32) * weight
    return np.clip(blended, 0, 255).astype(np.uint8)


# --- 프레임 그리기 -----------------------------------------------------------

def _banner(rgb: np.ndarray, judged: Judged) -> np.ndarray:
    """왼쪽 위에 판정 한 줄. 재생 중에 «지금 이 프레임은 무엇인가»를 읽을 수 있어야 한다."""
    color = DEFECT_COLOR if judged.defect else NORMAL_COLOR
    size = max(14, round(min(rgb.shape[:2]) / 22))
    out = rgb.copy()
    # 어둡게 까는 것은 **띠 부분만**이다. 화면 전체를 섞으면 그림도 박스도 같이 흐려져서,
    # 판정을 읽기 쉽게 하려던 것이 정작 판정 대상을 안 보이게 만든다.
    strip = slice(0, min(size * 2, out.shape[0]))
    out[strip] = (out[strip].astype(np.float32) * 0.35 + 16 * 0.65).astype(np.uint8)
    text = (
        f"{judged.seconds:5.1f}초  "
        f"{config.LABEL_KO.get(judged.decision, judged.decision)}  "
        f"{judged.score:.3f} / {judged.threshold:.3f}"
    )
    ascii_text = (
        f"{judged.seconds:5.1f}s  {'DEFECT' if judged.defect else 'OK'}  "
        f"{judged.score:.3f} / {judged.threshold:.3f}"
    )
    return viz.put_text(
        out, text, (size // 2, size // 2), color=color, size=size, ascii_fallback=ascii_text
    )


def draw_judgement(rgb: np.ndarray, judged: Judged) -> np.ndarray:
    """원본 프레임에 박스와 판정 띠를 그린다."""
    color = DEFECT_COLOR if judged.defect else NORMAL_COLOR
    out = rgb
    for box in judged.boxes:
        out = viz.draw_roi(out, box, color=color)
    return _banner(out, judged)


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """두 장을 나란히 붙인다. 사이에 가는 선을 넣어 경계를 분명히 한다."""
    if left.shape[0] != right.shape[0]:
        right = cv2.resize(right, (right.shape[1], left.shape[0]))
    divider = np.full((left.shape[0], DIVIDER_PX, 3), DIVIDER_COLOR, dtype=np.uint8)
    return np.hstack([left, divider, right])


def pane_label(rgb: np.ndarray, text: str, ascii_fallback: str) -> np.ndarray:
    """칸 아래쪽에 «이 칸이 무엇인지»를 적는다.

    두 칸이 같은 그림에서 나왔기 때문에 설명이 없으면 왼쪽·오른쪽이 뒤집혀도 아무도 모른다.
    """
    size = max(12, round(min(rgb.shape[:2]) / 30))
    return viz.put_text(
        rgb, text, (size // 2, rgb.shape[0] - size * 2), color=(235, 235, 235), size=size,
        ascii_fallback=ascii_fallback,
    )


def compose(rgb: np.ndarray, judged: Judged, score_map: np.ndarray | None) -> np.ndarray:
    """재생할 한 프레임을 완성한다.

    이상탐지 모델이면 **왼쪽에 박스, 오른쪽에 열지도**를 나란히 둔다. 박스만 보면 «왜
    저기를 잡았나»를 알 수 없고, 열지도만 보면 «잡았다/놓쳤다»가 안 보인다. 둘을 같이 봐야
    판정이 납득이 되거나, 납득이 안 되는 이유가 보인다.
    """
    boxed = draw_judgement(rgb, judged)
    if score_map is None:
        return boxed
    heat = heat_overlay(rgb, score_map, judged.threshold)
    return side_by_side(
        pane_label(boxed, "판정 박스", "BOXES"),
        pane_label(heat, "열지도 (붉을수록 기준선 위)", "HEATMAP (red = above threshold)"),
    )


# --- 만들기 ------------------------------------------------------------------

def _open_writer(directory: Path, size: tuple[int, int], fps: float):
    """재생 가능한 코덱부터 차례로 시도한다. 되는 것이 나오면 그것으로 쓴다."""
    for fourcc, suffix, mime, playable in CODECS:
        path = directory / f"{VIDEO_STEM}{suffix}"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer, path, mime, playable
        writer.release()
        path.unlink(missing_ok=True)
    raise RuntimeError("이 환경의 OpenCV로는 영상을 만들 수 없습니다.")


def _sampled(capture, stride: int, limit: int) -> Iterator[tuple[int, np.ndarray]]:
    """`stride`장마다 한 장씩, `limit`장까지 읽는다.

    건너뛰는 프레임은 `grab()`으로 넘긴다 — 디코딩(`retrieve`)까지 하면 버리는 프레임에
    같은 비용을 낸다.
    """
    index = 0
    taken = 0
    while taken < limit:
        if not capture.grab():
            return
        if index % stride == 0:
            ok, frame = capture.retrieve()
            if not ok:
                return
            yield index, frame
            taken += 1
        index += 1


def plan_stride(total_frames: int, limit: int) -> int:
    """영상 전체를 **고르게** 훑도록 간격을 정한다.

    앞에서부터 `limit`장을 자르면 10분짜리의 앞 1분만 본 것이 된다. 조명이 바뀌거나 물건이
    달라지는 뒷부분을 통째로 놓치는데, 화면에는 그 사실이 드러나지 않는다.
    """
    if total_frames <= 0 or limit <= 0:
        return 1
    return max(1, int(np.ceil(total_frames / limit)))


def output_dir(source, version: str) -> Path:
    """판정본을 둘 폴더. 영상 × 모델 버전마다 따로 둔다.

    같은 영상을 다른 모델로 돌린 결과가 서로를 덮으면 «어느 모델이 잡은 것인지» 비교가
    불가능해진다. 비교하려고 만드는 기능이므로 이것부터 갈라 둔다.
    """
    from . import video as video_mod

    stem = video_mod.video_id(source) if Path(source).exists() else Path(source).stem
    return config.interim_dir() / PLAYBACK_DIR / f"{stem}__{version or 'model'}"


def render(
    source,
    model,
    *,
    limit: int = DEFAULT_LIMIT,
    threshold: float | None = None,
    stills: int = STILL_LIMIT,
    directory: Path | None = None,
    progress: ProgressCallback | None = None,
) -> Playback:
    """영상을 프레임마다 판정하고, 판정을 그려 넣은 재생본과 대표 장면을 만든다.

    `model`은 4단계에서 쓰는 `serving.LoadedModel`이다. 이상탐지 모델이면 열지도까지
    나오므로 박스와 열지도를 나란히 붙이고, 분류 모델이면 판정 띠만 얹는다 — 분류 모델은
    «어디»를 모르기 때문에 박스를 그리면 없는 정보를 지어내는 것이 된다.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        source_fps = source_fps if 1.0 <= source_fps <= 240.0 else 30.0
        stride = plan_stride(total, limit)

        target = directory or output_dir(source, getattr(model, "version", ""))
        target = Path(target)
        if target.exists():
            shutil.rmtree(target)                 # 지난 판정본이 섞이면 무엇을 보는지 모르게 된다
        (target / STILL_DIR).mkdir(parents=True, exist_ok=True)

        writer = None
        video_path = target / f"{VIDEO_STEM}.webm"
        mime, playable = "video/webm", True
        judged_frames: list[Judged] = []
        # 대표 장면은 **다 훑어 본 뒤에야** 고를 수 있다. 그렇다고 완성된 프레임을 통째로
        # 들고 있으면 150장에 300MB가 넘는다. JPEG로 눌러 두면 20MB 남짓이고, 고른 장면은
        # 그 바이트를 그대로 파일에 쓰면 되니 다시 풀 일도 없다.
        encoded: list[bytes] = []
        expected = plan_count(total, stride, limit)

        for index, bgr in _sampled(capture, stride, limit):
            rgb = viz.to_rgb(bgr)
            score = float(model.score_image(rgb))
            used = float(threshold) if threshold is not None else float(model.threshold)
            decision = config.LABEL_DEFECT if score >= used else config.LABEL_NORMAL

            score_map = model.score_map(rgb)
            boxes: tuple[tuple[int, int, int, int], ...] = ()
            if score_map is not None:
                level = cut_level(score_map, used, defect=decision == config.LABEL_DEFECT)
                boxes = tuple(boxes_from_score_map(score_map, level, shape=rgb.shape[:2]))

            judged = Judged(
                index=index,
                seconds=index / source_fps,
                score=score,
                threshold=used,
                decision=decision,
                boxes=boxes,
            )
            judged_frames.append(judged)

            picture = compose(rgb, judged, score_map)
            if writer is None:
                writer, video_path, mime, playable = _open_writer(
                    target, (picture.shape[1], picture.shape[0]), max(source_fps / stride, 1.0)
                )
            bgr_picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
            writer.write(bgr_picture)
            ok, buffer = cv2.imencode(".jpg", bgr_picture, [cv2.IMWRITE_JPEG_QUALITY, 90])
            encoded.append(buffer.tobytes() if ok else b"")
            if progress is not None:
                progress(len(judged_frames), max(expected, len(judged_frames)))

        if writer is not None:
            writer.release()
    finally:
        capture.release()

    playback = Playback(
        directory=target,
        video=video_path,
        mime=mime,
        playable=playable,
        frames=judged_frames,
        fps=max(source_fps / stride, 1.0),
        scanned=total,
        version=str(getattr(model, "version", "")),
        source=str(source),
        side_by_side=_has_map(model),
    )
    playback.stills = _write_stills(playback, encoded, limit=stills)
    playback.save()
    return playback


def plan_count(total: int, stride: int, limit: int) -> int:
    """이번에 판정할 프레임 수 예상치. 진행 표시가 끝까지 차게 하려고 미리 센다."""
    if total <= 0:
        return limit
    return max(1, min(limit, int(np.ceil(total / max(stride, 1)))))


def _has_map(model) -> bool:
    return getattr(model, "kind", "") == "anomaly"


def _write_stills(playback: Playback, encoded: list[bytes], *, limit: int) -> list[Path]:
    """대표 장면을 파일로 남긴다 — 영상 아래에 붙여 **멈춰 놓고** 볼 수 있게."""
    if not encoded:
        return []
    by_index = {judged.index: order for order, judged in enumerate(playback.frames)}
    written: list[Path] = []
    for judged in playback.highlights(limit):
        order = by_index.get(judged.index)
        if order is None or order >= len(encoded) or not encoded[order]:
            continue
        path = playback.directory / STILL_DIR / f"{judged.index:08d}.jpg"
        path.write_bytes(encoded[order])
        written.append(path)
    return written
