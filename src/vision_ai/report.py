"""결과 보고서 생성.

## 왜 필요한가

품질 개선 과제의 결과는 결국 **남에게 보여줘야** 한다. 그런데 지금까지는 화면에만 있고
꺼낼 방법이 없어서, 회의에 들고 가려면 화면을 캡처해 옮겨 적어야 했다.

## 무엇을 담는가

숫자만 나열하면 받는 사람이 해석을 못 한다. 화면에서 하던 해석(업무 효과 환산, 판정,
주의사항)을 **그대로 문서에 담는다.** 특히 정밀도가 평가 구성에 좌우된다는 점은
보고서에서 빠지면 곧바로 오해로 이어지므로 반드시 넣는다.

합성 데이터로 낸 수치라면 그 사실을 **맨 위에** 적는다. 보고서는 화면과 달리 맥락 없이
전달되므로, 경고가 아래쪽에 있으면 못 보고 인용된다.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from . import config, evaluate, glossary

TITLE = "표면 결함 탐지 결과 보고서"


def _fmt(value, digits: int = 3, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if number != number:   # NaN
        return "—"
    return f"{number:.1%}" if percent else f"{number:.{digits}f}"


def _data_section(resolved: pd.DataFrame) -> list[str]:
    lines = ["## 1. 데이터", ""]
    if resolved.empty:
        lines += ["등록된 이미지가 없습니다.", ""]
        return lines

    labels = resolved["label"].astype(str)
    normal = int((labels == config.LABEL_NORMAL).sum())
    defect = int((labels == config.LABEL_DEFECT).sum())
    sources = sorted(set(resolved["source"].astype(str)))

    lines += [
        f"- 전체 이미지: **{len(resolved):,}장** (정상 {normal:,} / 결함 {defect:,})",
        f"- 카테고리: {resolved['category'].nunique()}종",
        f"- 출처: {', '.join(sources)}",
    ]
    if "split" in resolved.columns:
        counts = resolved["split"].astype(str).value_counts()
        parts = [f"{name} {int(counts.get(name, 0)):,}" for name in config.SPLITS if counts.get(name, 0)]
        lines.append(f"- 분할: {' / '.join(parts)}")
    lines.append("")
    return lines


def _model_section(result: dict) -> list[str]:
    settings = result.get("settings") or {}
    lines = [
        "## 2. 모델",
        "",
        f"- 방식: **{result.get('kind', '—')} / {result.get('model', '—')}**",
        f"- 학습 이미지: {result.get('n_train', 0):,}장",
        f"- 평가 분할: `{result.get('split', '—')}`",
    ]
    if settings:
        detail = ", ".join(f"{k}={v}" for k, v in settings.items())
        lines.append(f"- 설정: {detail}")
    lines.append("")
    return lines


def _performance_section(metrics: dict) -> list[str]:
    return [
        "## 3. 성능",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 재현율 (실제 결함 중 잡아낸 비율) | {_fmt(metrics.get('recall'), percent=True)} |",
        f"| 정밀도 (결함이라 한 것 중 실제 비율) | {_fmt(metrics.get('precision'), percent=True)} |",
        f"| AUROC | {_fmt(metrics.get('auroc'))} |",
        f"| AP (평균 정밀도) | {_fmt(metrics.get('average_precision'))} |",
        f"| 미탐 / 오탐 | {metrics.get('fn', '—')} / {metrics.get('fp', '—')} |",
        f"| 판정 임계값 | {_fmt(metrics.get('threshold'), digits=4)} |",
        "",
        "> 위 정밀도는 **평가 데이터 구성 기준**입니다. 실제 라인은 정상이 훨씬 많으므로 "
        "이 값을 현장 기대치로 읽으면 크게 과대평가하게 됩니다. 아래 4장을 함께 보십시오.",
        "",
    ]


def _impact_section(metrics: dict, prevalence: float, volume: int) -> list[str]:
    impact = evaluate.business_impact(
        float(metrics.get("recall", 0.0)),
        float(metrics.get("false_alarm_rate", 0.0)),
        prevalence=prevalence,
        volume=volume,
    )
    return [
        "## 4. 도입하면 무엇이 좋아지는가",
        "",
        f"가정: 불량률 **{prevalence:.1%}**, 검사 물량 **{volume:,}장**",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 검수량 절감 | **{impact['reduction_ratio']:.0%}** "
        f"(사람이 볼 물량 {volume:,}장 → {impact['reviewed']:.0f}장) |",
        f"| 놓치는 결함 | **{impact['missed']:.0f}건** (전체 결함 {impact['defects']:.0f}건 중) |",
        f"| 걸러내야 할 오탐 | {impact['false_alarms']:.0f}건 |",
        f"| 현장 기대 정밀도 | {impact['precision']:.1%} |",
        "",
        "> 검수량 절감은 **결함을 놓치는 것을 대가로** 얻는 값입니다. 두 값을 함께 보고 "
        "감당 가능한 수준인지 업무 기준으로 판단해야 합니다.",
        "",
    ], impact


def _verdict_section(metrics: dict, impact: dict, target_recall: float) -> list[str]:
    result = glossary.verdict(metrics, impact, target_recall=target_recall)
    lines = ["## 5. 판정", "", f"**{result.headline}**", ""]
    if result.actions:
        lines.append("다음에 할 일:")
        lines += [f"- {action}" for action in result.actions]
        lines.append("")
    return lines


def _operations_section(production, log: pd.DataFrame, drift_summary: dict | None) -> list[str]:
    lines = ["## 6. 운영 현황", ""]
    if production is None:
        lines += ["서비스 중인 모델이 없습니다. 아직 운영 감시가 시작되지 않았습니다.", ""]
        return lines

    lines.append(f"- 서비스 중: **{production['version']}** (승격 {production.get('promoted_at', '—')})")
    lines.append(f"- 누적 판정 로그: {len(log):,}건")
    if drift_summary:
        level = drift_summary.get("level", "—")
        lines.append(
            f"- 입력 분포 상태: **{level}** "
            f"(변화 특징 {drift_summary.get('shifted', 0)}개 / 검사 {drift_summary.get('checked', 0)}개)"
        )
    lines.append("")
    return lines


def build(
    *,
    resolved: pd.DataFrame,
    result: dict,
    metrics: dict,
    prevalence: float = evaluate.DEFAULT_PREVALENCE,
    volume: int = 1000,
    target_recall: float = 0.95,
    production=None,
    log: pd.DataFrame | None = None,
    drift_summary: dict | None = None,
) -> str:
    """현재 상태를 마크다운 보고서 한 장으로 만든다."""
    log = pd.DataFrame() if log is None else log
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M")

    lines = [f"# {TITLE}", "", f"생성 시각: {now}", ""]

    # 합성 데이터 경고는 맨 위에 둔다. 보고서는 맥락 없이 전달되므로 아래에 있으면 못 보고 인용된다.
    sources = set(resolved["source"].astype(str)) if not resolved.empty else set()
    if sources and sources <= {"synthetic"}:
        lines += [
            "> ⚠️ **이 보고서의 수치는 합성 샘플로 낸 것입니다.** 결함이 인위적으로 뚜렷해 "
            "실제보다 높게 나옵니다. 파이프라인이 도는지 확인한 결과일 뿐 성능 근거가 아닙니다.",
            "",
        ]

    lines += _data_section(resolved)
    lines += _model_section(result)
    lines += _performance_section(metrics)

    impact_lines, impact = _impact_section(metrics, prevalence, volume)
    lines += impact_lines
    lines += _verdict_section(metrics, impact, target_recall)
    lines += _operations_section(production, log, drift_summary)

    lines += [
        "## 7. 읽을 때 주의할 점",
        "",
        f"- {glossary.ARBITRARY['psi']}",
        f"- {glossary.ARBITRARY['new_labels']}",
        "- 성능 기준(재현율 · 검수량 절감률)은 아직 승인 전 제안값입니다.",
        "",
    ]
    return "\n".join(lines)


def filename(prefix: str = "defect-report") -> str:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M")
    return f"{prefix}-{stamp}.md"
