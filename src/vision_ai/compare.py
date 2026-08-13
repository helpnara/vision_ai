"""두 영상을 나란히 놓고 «모델 탓인가 촬영 탓인가»에 답한다 (V4).

영상 A로 만든 모델을 영상 B에 걸었더니 성능이 떨어졌다. 이때 할 수 있는 일은 두 가지로
갈리고, **둘은 서로 배타적이다**:

* 촬영이 달라진 것이라면 → 화각·조명을 맞추거나, B로 다시 학습한다.
* 입력은 그대로인데 못 잡는 것이라면 → 재촬영해 봐야 소용없다. 모델이나 결함 종류의 문제다.

지금까지 이 판단의 근거는 **성능 추이**와 **드리프트 감시**에 흩어져 있었다. 둘을 각각
읽고 머릿속에서 맞춰 봐야 했는데, 그 맞춰 보는 일이 정확히 초보자가 못 하는 일이다.
여기서는 두 영상의 지표와 입력 분포 이동을 한 화면에 놓고 **결론 문장까지 붙인다.**

**인과를 증명하지는 않는다.** 성능이 떨어진 것과 입력이 변한 것이 같이 일어났다는 사실을
말할 뿐이다. 그래서 문장도 «~탓이다»가 아니라 «~일 가능성이 크다»로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import monitoring, registry

# 구간 재현율이 이만큼 떨어지면 «성능이 떨어졌다»고 본다. 표본이 작으면 이 정도는 우연으로도
# 흔들리므로, 구간 수가 적을 때는 아예 판단을 보류한다(MIN_SEGMENTS).
RECALL_DROP = 0.10

# 정답 구간이 이보다 적으면 재현율을 추세로 읽지 않는다. 구간이 1곳이면 재현율은 0 아니면
# 1이라 «떨어졌다»가 늘 100% 낙폭으로 보인다.
MIN_SEGMENTS = 3

# 크게 이동한 특징을 몇 개까지 이름으로 짚어 줄지. 14개를 다 늘어놓으면 아무것도 안 읽힌다.
NAMED_FEATURES = 3

CAUSE_NONE = "이상 없음"
CAUSE_FRAMING = "촬영"
CAUSE_MODEL = "모델"
CAUSE_UNKNOWN = "판단 보류"


@dataclass(frozen=True)
class Side:
    """비교 한쪽 — 영상 하나의 성적."""

    name: str
    frames: int
    truth_segments: int
    segment_recall: float
    frame_recall: float
    false_alarms_per_min: float
    false_alarm_time_ratio: float = 0.0

    @property
    def measurable(self) -> bool:
        """이 영상의 재현율을 추세로 읽어도 되는가."""
        return self.truth_segments >= MIN_SEGMENTS and np.isfinite(self.segment_recall)

    @property
    def flooding(self) -> bool:
        from . import segments

        return self.false_alarm_time_ratio >= segments.ALARM_FLOOD


def side_from(report, name: str, *, frames: int) -> Side:
    """`segments.SegmentReport`를 비교용 한쪽으로 옮긴다."""
    return Side(
        name=name,
        frames=int(frames),
        truth_segments=len(report.truth),
        segment_recall=report.segment_recall,
        frame_recall=report.frame_recall,
        false_alarms_per_min=report.false_alarms_per_min,
        false_alarm_time_ratio=report.false_alarm_time_ratio,
    )


@dataclass(frozen=True)
class Drift:
    """두 영상의 입력 분포가 얼마나 다른가."""

    psi_mean: float
    level: str
    moved: tuple[tuple[str, float], ...]
    n_samples: int

    @property
    def changed(self) -> bool:
        return self.level in ("주의", "변화")

    @property
    def measurable(self) -> bool:
        return self.level != monitoring.LEVEL_INSUFFICIENT and np.isfinite(self.psi_mean)


def drift_between(reference, candidate, feature_names) -> Drift:
    """영상 A의 분포를 기준선 삼아 영상 B가 얼마나 벗어났는지 잰다.

    **레지스트리에 저장된 기준선을 쓰지 않는다.** 그것은 «학습 분할 전체»의 분포여서, 두
    영상을 서로 견주는 질문에는 답하지 못한다. 여기서는 A 자체를 기준선으로 세운다.
    """
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.size == 0 or candidate.size == 0:
        return Drift(float("nan"), monitoring.LEVEL_INSUFFICIENT, (), int(candidate.shape[0]))

    baseline = registry.make_baseline(reference, feature_names)
    table = monitoring.feature_drift(baseline, candidate, feature_names)
    summary = monitoring.drift_summary(table)

    moved: list[tuple[str, float]] = []
    if not table.empty:
        top = table.dropna(subset=["psi"]).head(NAMED_FEATURES)
        moved = [
            (str(row["feature"]), float(row["psi"]))
            for _, row in top.iterrows()
            if float(row["psi"]) > monitoring.PSI_STABLE
        ]
    return Drift(
        psi_mean=float(summary["mean_psi"]),
        level=str(summary["level"]),
        moved=tuple(moved),
        n_samples=int(candidate.shape[0]),
    )


@dataclass(frozen=True)
class Comparison:
    """«영상 A로 만든 모델이 영상 B에서 어떤가»에 대한 한 장짜리 답."""

    reference: Side
    candidate: Side
    drift: Drift

    @property
    def recall_drop(self) -> float:
        """구간 재현율이 얼마나 떨어졌는가. 오르면 음수."""
        if not (self.reference.measurable and self.candidate.measurable):
            return float("nan")
        return self.reference.segment_recall - self.candidate.segment_recall

    @property
    def started_flooding(self) -> bool:
        """새 영상에서 알람이 계속 켜져 있게 됐는가.

        이때 구간 재현율은 오히려 **올라간다** — 다 결함이라고 했으니 다 잡은 것이 된다.
        재현율만 보면 «좋아졌다»가 되므로 따로 잡아내지 않으면 놓친다.
        """
        return self.candidate.flooding and not self.reference.flooding

    @property
    def performance_held(self) -> bool:
        drop = self.recall_drop
        return np.isfinite(drop) and drop <= RECALL_DROP and not self.started_flooding

    def cause(self) -> str:
        """무엇을 고쳐야 하는가. 네 갈래뿐이고, 갈래마다 다음 행동이 다르다."""
        drop = self.recall_drop
        if not np.isfinite(drop):
            return CAUSE_UNKNOWN
        if self.performance_held:
            return CAUSE_NONE
        if not self.drift.measurable:
            return CAUSE_UNKNOWN
        return CAUSE_FRAMING if self.drift.changed else CAUSE_MODEL

    def verdict(self) -> str:
        """결론 한 문장."""
        cause = self.cause()
        left, right = self.reference.name, self.candidate.name
        if cause == CAUSE_UNKNOWN:
            return f"**아직 판단할 수 없습니다.** {self._blocker()}"
        if cause == CAUSE_NONE:
            base = (
                f"**`{right}`에서도 성능이 유지됩니다** "
                f"(구간 재현율 {self.reference.segment_recall:.2f} → "
                f"{self.candidate.segment_recall:.2f})."
            )
            if self.drift.changed:
                return base + " 촬영은 달라졌는데도 버텼습니다 — 다만 더 벌어지면 무너질 수 있습니다."
            return base + " 입력 분포도 안정적입니다."
        drop_text = (
            f"`{right}`에서는 정상인 시간의 {self.candidate.false_alarm_time_ratio:.0%}에도 "
            "알람이 켜져 있습니다(구간 재현율은 오히려 올라갔지만 «다 결함이라고 했기 "
            "때문»입니다)"
            if self.started_flooding
            else (
                f"구간 재현율이 {self.reference.segment_recall:.2f} → "
                f"{self.candidate.segment_recall:.2f}로 떨어졌습니다"
            )
        )
        if cause == CAUSE_FRAMING:
            return (
                f"**촬영이 달라진 탓일 가능성이 큽니다.** {drop_text}. 동시에 `{left}`와 "
                f"`{right}`의 입력 분포도 벌어졌습니다(PSI {self.drift.psi_mean:.3f}, "
                f"{self.drift.level})."
            )
        return (
            f"**모델 쪽 문제일 가능성이 큽니다.** {drop_text}. 그런데 입력 분포는 거의 "
            f"그대로입니다(PSI {self.drift.psi_mean:.3f}, {self.drift.level}) — 같은 그림을 "
            "보고도 못 잡고 있다는 뜻입니다."
        )

    def advice(self) -> str:
        """다음에 무엇을 할지. 갈래마다 **해도 소용없는 일**이 다르다."""
        cause = self.cause()
        if cause == CAUSE_UNKNOWN:
            return (
                "먼저 표본을 채우세요. 2단계 **영상 구간 라벨링**으로 정답 구간을 늘리고, "
                "배치 추론에서 판정 장수를 늘리면 됩니다."
            )
        if cause == CAUSE_NONE:
            return (
                "이 영상은 그대로 쓸 수 있습니다. 다른 촬영본으로도 한 번 더 확인하면 "
                "«이 라인 전반에 통한다»고 말할 근거가 됩니다."
            )
        if cause == CAUSE_FRAMING:
            named = self._named_features()
            return (
                f"**촬영 조건부터 맞춰 보세요**{named}. 그래도 안 되면 `{self.candidate.name}`의 "
                "프레임을 학습에 넣어 다시 학습합니다. 모델 구조를 바꾸는 것은 그다음입니다."
            )
        return (
            "**재촬영으로는 낫지 않습니다.** 입력이 같은데 못 잡는 것이므로, "
            f"`{self.candidate.name}`에 **다른 종류의 결함**이 있는지 먼저 눈으로 확인하고"
            "(판정 영상 탭), 그 결함을 라벨링해 학습에 넣으세요. 임계값 조정으로 회복되는지도 "
            "함께 봅니다."
        )

    def _blocker(self) -> str:
        for side in (self.reference, self.candidate):
            if side.truth_segments < MIN_SEGMENTS:
                return (
                    f"`{side.name}`의 정답 구간이 {side.truth_segments}곳뿐이라 재현율을 "
                    f"추세로 읽을 수 없습니다(최소 {MIN_SEGMENTS}곳)."
                )
        if not self.drift.measurable:
            return (
                f"입력 분포를 비교할 표본이 부족합니다"
                f"(현재 {self.drift.n_samples}장, 최소 {monitoring.MIN_DRIFT_SAMPLES}장). "
                "성능은 떨어졌지만 원인은 아직 가릴 수 없습니다."
            )
        return "비교에 필요한 값이 모자랍니다."

    def _named_features(self) -> str:
        if not self.drift.moved:
            return ""
        names = ", ".join(f"`{name}`" for name, _ in self.drift.moved)
        return f" — 특히 {names}가(이) 크게 이동했습니다"

    def table(self):
        """화면에 나란히 놓을 표. 같은 줄에 두 영상을 두어야 눈으로 대 볼 수 있다."""
        import pandas as pd

        def column(side: Side) -> dict:
            return {
                "판정 프레임": f"{side.frames:,}장",
                "정답 구간": f"{side.truth_segments}곳",
                "구간 재현율": _figure(side.segment_recall),
                "프레임 재현율": _figure(side.frame_recall),
                "헛알람(분당)": _figure(side.false_alarms_per_min),
                # 구간 재현율이 만점인데 이 값이 높으면 «다 결함이라고 한» 것이다.
                "정상인데 알람": f"{side.false_alarm_time_ratio:.0%}",
            }

        left, right = column(self.reference), column(self.candidate)
        return pd.DataFrame(
            [{"항목": key, self.reference.name: left[key], self.candidate.name: right[key]}
             for key in left]
        )


def _figure(value: float) -> str:
    return f"{value:.2f}" if np.isfinite(value) else "—"
