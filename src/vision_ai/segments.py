"""구간 단위로 잡았는지 센다 (V3).

지금까지의 지표는 전부 **프레임 단위**다. 그런데 현장이 묻는 것은 «이 영상에 있던 결함을
잡았느냐»이지 «프레임 몇 %를 맞혔느냐»가 아니다.

결함이 100프레임에 걸쳐 지나가고 그중 3장만 잡았다고 하자. 사람이 보기에는 **잡은 것**이다
— 알람이 울렸고 그 물건은 걸러진다. 그런데 프레임 재현율은 3%로 찍힌다. 이 숫자를 그대로
보고하면 «쓸 수 없는 모델»이라는 결론이 나온다. **실제보다 나쁘게 보고하는 방향으로
틀리기 때문에** 도입 판단 자체를 그르친다.

반대 방향도 있다. 정상 구간에서 한 프레임씩 띄엄띄엄 헛짚으면 프레임 정밀도는 별로 안
떨어지는데, 현장에서는 **알람이 계속 울리는** 것으로 나타난다. 그래서 오경보는 건수가
아니라 **분당 몇 번**으로 센다 — 그것이 검사원이 실제로 겪는 값이다.

두 숫자를 같이 내고, 서로 얼마나 다른지 말해 준다. 하나만 보면 어느 쪽으로든 틀린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEFAULT_MIN_HITS = 1
DEFAULT_GAP = 0
SECONDS_PER_MINUTE = 60.0

# 정답 구간에 바로 붙은 **짧은** 알람은 헛알람으로 세지 않는다. 결함이 화면에서 빠져나가는
# 한두 장은 모델이 조금 더 붙잡기 마련이고, 그것까지 벌점을 주면 잘 잡았는데 점수가 깎인다.
# 다만 «붙어 있다»만으로 봐주면 안 된다 — 전 구간에 알람을 켜 놓은 모델은 모든 정상 구간이
# 정답 구간에 붙어 있으므로 헛알람 0건이 되어 **가장 나쁜 모델이 만점을 받는다.**
BOUNDARY_SLACK = 2

# **정상인 시간 중** 알람이 이만큼 켜져 있으면 «계속 울리고 있다»고 본다.
# 영상 전체를 기준으로 재면 안 된다 — 결함이 원래 많은 영상에서는 잘 잡을수록 비율이 높아져,
# 좋은 모델이 «계속 울린다»로 찍힌다. 문제는 **결함이 없는데 울리는 것**이다.
ALARM_FLOOD = 0.5

# 프레임 재현율과 구간 재현율이 이만큼 벌어지면 «프레임 숫자만 보면 오해한다»고 말해 준다.
NOTABLE_GAP = 0.15

# 타임라인의 두 줄과 네 가지 띠 (V6). 이름이 곧 범례에 찍히므로 여기서 정한다.
LANE_TRUTH = "정답 구간"
LANE_ALARM = "모델 알람"
KIND_CAUGHT = "잡음"
KIND_MISSED = "놓침"
KIND_ON_TARGET = "결함 위 알람"
KIND_FALSE = "헛알람"


@dataclass(frozen=True)
class Segment:
    """이어진 구간 하나. `start`/`end`는 **표본 안에서의 위치**이고 양 끝을 포함한다.

    프레임 번호가 아니라 위치인 이유는, 판정 대상이 영상 전체에서 **고르게 뽑은 표본**이라
    번호가 3씩·10씩 건너뛰기 때문이다. 번호로 «이어짐»을 따지면 표본 간격을 결함이 끊긴
    것으로 잘못 읽는다.
    """

    start: int
    end: int
    start_sec: float
    end_sec: float

    @property
    def length(self) -> int:
        """이 구간에 든 표본 프레임 수."""
        return self.end - self.start + 1

    @property
    def duration_sec(self) -> float:
        return max(self.end_sec - self.start_sec, 0.0)

    def positions(self) -> range:
        return range(self.start, self.end + 1)

    def overlaps(self, other: "Segment") -> bool:
        return self.start <= other.end and other.start <= self.end

    def span_text(self) -> str:
        if self.start == self.end:
            return f"{self.start_sec:.1f}초"
        return f"{self.start_sec:.1f}~{self.end_sec:.1f}초"


def runs(flags: Sequence[bool], seconds: Sequence[float], *, gap: int = DEFAULT_GAP) -> list[Segment]:
    """참인 곳이 이어진 구간들을 찾는다.

    `gap`은 **몇 장까지 끊긴 것으로 안 볼지**다. 결함이 지나가는 중에 한 프레임 흔들려서
    놓치면 사람 눈에는 한 구간인데 기계는 두 구간으로 센다. 구간 수가 부풀면 오경보율이
    실제보다 크게 나온다. 기본은 0(끊기면 끊긴 것) — 정답 구간은 «몇 초~몇 초»를 끌어서
    만들기 때문에 원래 이어져 있다.
    """
    if len(flags) != len(seconds):
        raise ValueError("판정과 시각의 길이가 다릅니다.")

    found: list[Segment] = []
    start: int | None = None
    last_true: int | None = None
    for index, flag in enumerate(flags):
        if flag:
            if start is None:
                start = index
            elif last_true is not None and index - last_true - 1 > max(gap, 0):
                found.append(Segment(start, last_true, seconds[start], seconds[last_true]))
                start = index
            last_true = index
    if start is not None and last_true is not None:
        found.append(Segment(start, last_true, seconds[start], seconds[last_true]))
    return found


@dataclass(frozen=True)
class SegmentReport:
    """구간 단위 성적표. 프레임 단위 숫자도 같이 들고 있어야 둘을 나란히 보여줄 수 있다."""

    truth: tuple[Segment, ...]
    predicted: tuple[Segment, ...]
    caught: tuple[Segment, ...]
    missed: tuple[Segment, ...]
    false_alarms: tuple[Segment, ...]
    frame_recall: float
    frame_precision: float
    duration_sec: float
    on_target: tuple[Segment, ...] = ()      # 정답 위에서 울린 알람 구간
    tick: float = 0.0                        # 표본 사이의 대표 간격(초)
    false_alarm_time_ratio: float = 0.0     # 정상인 시간 중 알람이 켜져 있던 비율
    min_hits: int = DEFAULT_MIN_HITS

    @property
    def flooding(self) -> bool:
        """정상인데도 알람이 켜져 있는 시간이 대부분인가.

        구간 지표만으로는 안 드러나는 실패 방식이다 — 다 결함이라고 하면 구간 재현율은
        만점이 된다.
        """
        return self.false_alarm_time_ratio >= ALARM_FLOOD

    @property
    def segment_recall(self) -> float:
        """정답 구간 중 몇 곳을 잡았는가. 정답 구간이 없으면 잴 것이 없다."""
        if not self.truth:
            return float("nan")
        return len(self.caught) / len(self.truth)

    @property
    def false_alarms_per_min(self) -> float:
        """분당 헛알람 횟수. **건수가 아니라 빈도**여야 현장이 겪는 값이 된다.

        30초짜리에서 3번과 10분짜리에서 3번은 전혀 다른 얘기인데, 건수로만 적으면 같아 보인다.
        """
        if self.duration_sec <= 0:
            return float("nan")
        return len(self.false_alarms) * SECONDS_PER_MINUTE / self.duration_sec

    def summary(self) -> str:
        if not self.truth:
            return (
                f"정답 구간이 없어 검출률을 잴 수 없습니다. 모델은 구간 "
                f"{len(self.predicted)}곳에서 알람을 냈습니다."
            )
        text = (
            f"정답 구간 {len(self.truth)}곳 중 **{len(self.caught)}곳**을 잡았습니다 "
            f"(구간 재현율 {self.segment_recall:.2f}). 헛알람은 {len(self.false_alarms)}건, "
            f"분당 {self.false_alarms_per_min:.1f}회입니다."
        )
        if self.flooding:
            text += (
                f" **다만 정상인 시간의 {self.false_alarm_time_ratio:.0%}에서도 알람이 켜져 "
                "있습니다** — 구간 재현율이 높은 것은 «다 결함이라고 했기 때문»일 수 있습니다."
            )
        return text

    def contrast(self) -> str:
        """프레임 숫자와 구간 숫자가 얼마나 다른지 말해 준다. 다르지 않으면 빈 문자열."""
        if not self.truth or self.frame_recall != self.frame_recall:  # NaN
            return ""
        difference = self.segment_recall - self.frame_recall
        if difference > NOTABLE_GAP:
            return (
                f"프레임 재현율은 {self.frame_recall:.2f}인데 구간 재현율은 "
                f"{self.segment_recall:.2f}입니다. 결함이 지나가는 **여러 장 중 일부만** 잡고 "
                "있다는 뜻이며, 알람을 울리는 것이 목적이라면 **프레임 숫자는 실제보다 나쁘게 "
                "보고합니다.**"
            )
        if difference < -NOTABLE_GAP:
            return (
                f"프레임 재현율은 {self.frame_recall:.2f}인데 구간 재현율은 "
                f"{self.segment_recall:.2f}입니다. 긴 구간은 잘 잡고 **짧게 지나가는 결함을 "
                "통째로 놓치고** 있습니다 — 프레임 숫자만 보면 이 사실이 가려집니다."
            )
        return ""


def bands(report: "SegmentReport", *, tick: float | None = None) -> list[dict]:
    """타임라인에 그릴 띠 목록 (V6).

    미탐이 **몇 건**인지보다 **영상의 어디서** 놓쳤는지가 다음에 무엇을 더 라벨링할지
    정해 준다. 정답 줄과 모델 줄을 위아래로 놓으면 어긋난 자리가 눈에 바로 들어온다.

    모델 줄에는 «예측 구간»을 그대로 그리지 않는다. 전 구간에 알람을 켠 모델은 예측 구간이
    하나뿐이라 화면상 «정답을 다 덮은 훌륭한 모델»로 보인다. 대신 알람이 켜진 시간을 **정답
    위 / 정답 밖**으로 갈라 그리면, 그 경우 주황 띠가 화면을 통째로 덮어 한눈에 드러난다.

    `tick`은 한 장짜리 구간에 줄 최소 폭이다. 시작과 끝이 같으면 폭 0이라 아예 안 보인다.
    """
    tick = max(float(report.tick if tick is None else tick), 0.0)

    def row(segment: Segment, lane: str, kind: str) -> dict:
        return {
            "lane": lane,
            "kind": kind,
            "start": float(segment.start_sec),
            "end": float(segment.end_sec) + tick,
            "span": segment.span_text(),
        }

    rows = [row(s, LANE_TRUTH, KIND_CAUGHT) for s in report.caught]
    rows += [row(s, LANE_TRUTH, KIND_MISSED) for s in report.missed]
    rows += [row(s, LANE_ALARM, KIND_ON_TARGET) for s in report.on_target]
    rows += [row(s, LANE_ALARM, KIND_FALSE) for s in report.false_alarms]
    return sorted(rows, key=lambda r: (r["lane"], r["start"]))


def sampling_tick(seconds: Sequence[float]) -> float:
    """표본 사이의 대표 간격. 한 장짜리 구간에 줄 폭으로 쓴다."""
    if len(seconds) < 2:
        return 0.0
    gaps = [float(b) - float(a) for a, b in zip(seconds[:-1], seconds[1:]) if b > a]
    if not gaps:
        return 0.0
    gaps.sort()
    return gaps[len(gaps) // 2]


def evaluate(
    truth: Sequence[bool],
    predicted: Sequence[bool],
    seconds: Sequence[float],
    *,
    min_hits: int = DEFAULT_MIN_HITS,
    gap: int = DEFAULT_GAP,
) -> SegmentReport:
    """정답 구간과 모델 판정을 구간 단위로 맞춰 본다.

    `min_hits`는 **몇 장을 잡아야 그 구간을 잡았다고 볼지**다. 기본은 1장 — 알람이 목적이면
    한 번 울리는 것으로 충분하기 때문이다. 사람이 확인하러 가는 비용이 큰 현장이라면 이 값을
    올려서 «한 장은 튄 것일 수 있다»를 반영한다.

    헛알람은 **정답이 아닌 곳에서 울린 알람이 이어진 구간**이다. 정답 구간을 한두 장 넘겨
    잡은 것은 봐준다(`BOUNDARY_SLACK`) — 잘 잡았는데 벌점을 받으면 안 된다.

    «정답 구간과 겹치지 않은 예측 구간»으로 세면 안 된다. 전 구간에 알람을 켜 놓은 모델은
    예측 구간이 하나뿐이고 그것이 모든 정답 구간과 겹치므로 **헛알람 0건에 구간 재현율
    만점**이 된다. 실측에서 실제로 그렇게 나왔다.
    """
    if not (len(truth) == len(predicted) == len(seconds)):
        raise ValueError("정답·판정·시각의 길이가 서로 다릅니다.")

    truth_runs = runs(truth, seconds, gap=gap)

    hit_at = set()
    caught: list[Segment] = []
    for segment in truth_runs:
        hits = sum(1 for position in segment.positions() if predicted[position])
        if hits >= max(min_hits, 1):
            caught.append(segment)
            hit_at.add((segment.start, segment.end))

    missed = [s for s in truth_runs if (s.start, s.end) not in hit_at]
    wrong = [bool(p) and not bool(t) for t, p in zip(truth, predicted)]
    right = [bool(p) and bool(t) for t, p in zip(truth, predicted)]
    false_alarms = [
        alarm for alarm in runs(wrong, seconds, gap=gap)
        if not _is_boundary_slop(alarm, truth_runs)
    ]
    normal_frames = sum(1 for t in truth if not t)
    wrongly_flagged = sum(1 for flag in wrong if flag)

    return SegmentReport(
        truth=tuple(truth_runs),
        predicted=tuple(runs(predicted, seconds, gap=gap)),
        caught=tuple(caught),
        missed=tuple(missed),
        false_alarms=tuple(false_alarms),
        on_target=tuple(runs(right, seconds, gap=gap)),
        tick=sampling_tick(seconds),
        frame_recall=_ratio(
            sum(1 for t, p in zip(truth, predicted) if t and p), sum(1 for t in truth if t)
        ),
        frame_precision=_ratio(
            sum(1 for t, p in zip(truth, predicted) if t and p), sum(1 for p in predicted if p)
        ),
        duration_sec=_duration(seconds),
        false_alarm_time_ratio=wrongly_flagged / normal_frames if normal_frames else 0.0,
        min_hits=max(min_hits, 1),
    )


def _is_boundary_slop(alarm: Segment, truth_runs: list[Segment]) -> bool:
    """정답 구간에 바로 붙은 **짧은** 알람인가.

    길이 조건이 없으면 안 된다. 정상 구간은 정의상 정답 구간 사이에 있으므로, «붙어 있다»만
    보면 아무리 긴 알람도 전부 봐주게 된다.
    """
    if alarm.length > BOUNDARY_SLACK:
        return False
    return any(
        alarm.start - segment.end - 1 <= BOUNDARY_SLACK
        and segment.start - alarm.end - 1 <= BOUNDARY_SLACK
        for segment in truth_runs
    )


def _ratio(hit: int, total: int) -> float:
    return hit / total if total else float("nan")


def _duration(seconds: Sequence[float]) -> float:
    """표본이 덮은 시간. 표본이 한 장뿐이면 길이를 알 수 없다."""
    if len(seconds) < 2:
        return 0.0
    return max(float(seconds[-1]) - float(seconds[0]), 0.0)


# --- 판정 로그에서 구간 성적표 만들기 ----------------------------------------
#
# 여기부터는 표(`DataFrame`)를 다룬다. 화면 코드가 직접 하면 테스트할 수 없어서, 판단은
# 전부 이쪽에 두고 화면은 부르기만 한다.

MIN_FRAMES_FOR_SEGMENTS = 2


def videos_in(feedback) -> list[str]:
    """정답 대조가 가능한 영상 이름들. 낱장 사진(`group`이 빈 값)은 구간이 없으므로 뺀다."""
    if feedback is None or getattr(feedback, "empty", True) or "group" not in feedback:
        return []
    names = feedback["group"].astype(str).str.strip()
    return sorted(name for name in names.unique() if name)


def from_feedback(
    feedback,
    frames,
    group: str,
    *,
    fps: float = 30.0,
    min_hits: int = DEFAULT_MIN_HITS,
    gap: int = DEFAULT_GAP,
) -> SegmentReport | None:
    """영상 하나의 판정 로그와 정답을 구간 성적표로 만든다. 잴 수 없으면 None.

    **영상 순서로 세우는 것이 전부다.** 로그는 판정한 순서대로 쌓이는데 그 순서는 영상
    순서가 아니다(고르게 뽑으면 더욱 아니다). 순서가 어긋난 채로 «이어짐»을 따지면 한
    구간이 여러 조각으로 부서져서 오경보율이 실제보다 크게 나온다.
    """
    import pandas as pd

    if feedback is None or feedback.empty or "group" not in feedback:
        return None
    rows = feedback[feedback["group"].astype(str) == str(group)]
    if rows.empty:
        return None

    if frames is not None and not frames.empty and "frame_index" in frames:
        order = frames[["image_id", "frame_index"]].drop_duplicates("image_id")
        rows = rows.merge(order, on="image_id", how="left")
    else:
        rows = rows.assign(frame_index=pd.NA)

    rows = rows.dropna(subset=["frame_index"])
    if len(rows) < MIN_FRAMES_FOR_SEGMENTS:
        return None
    rows = rows.sort_values("frame_index")

    step = float(fps) if fps and fps > 0 else 30.0
    return evaluate(
        rows["y_true"].astype(int).eq(1).tolist(),
        rows["y_pred"].astype(int).eq(1).tolist(),
        (rows["frame_index"].astype(float) / step).tolist(),
        min_hits=min_hits,
        gap=gap,
    )
