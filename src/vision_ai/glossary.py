"""지표 해석 도움말과 결과 판정.

## 왜 코어에 두는가

설명을 화면마다 직접 쓰면 같은 지표를 두 곳에서 다르게 설명하게 된다. 실제로 정밀도는
3단계·홈·문서에서 각각 다른 문장으로 설명되고 있었다. 여기 한 곳에 두고 화면이 가져다 쓴다.

## 표시 방식 (혼합안)

초보자 대상이므로 **핵심 지표는 눌러야 보이면 안 된다.** 하지만 전부 펼치면 화면이 길어진다.

- `short` — 캡션으로 **바로 보이게**. 핵심 지표(`primary=True`)만.
- `detail` — `?` 아이콘(`help=`)에. 더 깊은 설명과 주의점.

## 임의값을 임의값이라고 밝히기

PSI 0.10/0.25, 재학습 기준(신규 라벨 50건·재현율 낙폭 0.05), 품질 기준(선명도 100 등)은
업계 통용값이거나 이 프로젝트가 정한 값이다. 초보자가 이를 물리 상수처럼 받아들이면
자기 라인에 맞게 조정할 생각을 못 한다. `ARBITRARY` 항목에 근거를 함께 적는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricHelp:
    key: str
    label: str
    short: str            # 캡션 (바로 보임)
    detail: str = ""      # ? 아이콘
    primary: bool = False


def _entry(*args, **kwargs) -> MetricHelp:
    return MetricHelp(*args, **kwargs)


METRICS: dict[str, MetricHelp] = {
    m.key: m
    for m in [
        _entry(
            "recall", "재현율",
            "실제 결함 100개 중 몇 개를 잡았는가. **놓치면 불량이 나가므로 가장 중요합니다.**",
            "임계값을 낮추면 올라가지만 오탐도 함께 늘어납니다. 이 프로젝트는 목표 재현율을 "
            "먼저 정하고, 그것을 만족하는 임계값 중 오탐이 가장 적은 값을 자동으로 고릅니다.",
            primary=True,
        ),
        _entry(
            "precision", "정밀도",
            "결함이라고 올린 것 중 실제로 결함인 비율. **평가 데이터 기준이라 현장과 다릅니다.**",
            "평가 데이터는 정상:결함이 1:1에 가깝지만 실제 라인은 정상이 압도적으로 많습니다. "
            "결함이 드물수록 같은 모델이라도 정밀도가 크게 떨어집니다. 아래 '이 모델을 쓰면 "
            "무엇이 좋아지는가'에서 실제 불량률로 환산한 값을 보세요.",
            primary=True,
        ),
        _entry(
            "f1", "F1",
            "",
            "재현율과 정밀도의 조화평균입니다. 둘을 하나로 합친 값이라 편하지만, "
            "미탐과 오탐의 비용이 다른 검사 업무에서는 둘을 따로 보는 편이 낫습니다.",
        ),
        _entry(
            "auroc", "AUROC",
            "임계값과 무관한 종합 분리도. 0.5는 찍기, 1.0은 완벽입니다.",
            "정상과 결함의 점수 분포가 얼마나 잘 나뉘는지를 봅니다. 임계값을 어디에 두든 "
            "달라지지 않아 모델끼리 비교하기 좋습니다. 다만 **결함이 드물면 낙관적으로 "
            "보이므로**, 그럴 때는 AP(평균 정밀도)를 함께 봅니다.",
            primary=True,
        ),
        _entry(
            "average_precision", "AP (평균 정밀도)",
            "",
            "결함이 드문 상황에서 AUROC보다 민감합니다. 정상이 압도적으로 많으면 AUROC는 "
            "높게 나와도 실제로는 오탐이 많을 수 있는데, AP는 그것을 더 잘 드러냅니다.",
        ),
        _entry(
            "miss_fp", "미탐 / 오탐",
            "미탐은 놓친 결함, 오탐은 정상을 결함이라 한 것. **미탐이 더 위험합니다.**",
            "미탐(FN)은 불량이 그대로 나가는 것이고, 오탐(FP)은 사람이 다시 보면 걸러집니다. "
            "그래서 이 프로젝트는 미탐을 줄이는 쪽을 우선합니다.",
            primary=True,
        ),
        _entry(
            "reduction", "검수량 절감",
            "전수 검사 대비 사람이 안 봐도 되는 비율. **이 도구의 실제 효용입니다.**",
            "= 1 − (재현율×불량률 + 오탐률×(1−불량률)). 모델이 결함이라 올린 것만 사람이 보면 "
            "되므로, 나머지는 건너뛸 수 있습니다.",
            primary=True,
        ),
        _entry(
            "missed", "놓치는 결함",
            "검수량을 줄이는 **대가**입니다. 이 값을 받아들일 수 있는지가 도입 판단의 핵심입니다.",
            "절감률만 보면 안 됩니다. 사람이 덜 보는 만큼 놓치는 결함이 생깁니다. "
            "이 건수가 감당 가능한 수준인지 업무 기준으로 판단해야 합니다.",
            primary=True,
        ),
        _entry(
            "psi", "PSI (분포 변화)",
            "학습할 때의 입력 분포와 지금 들어오는 입력이 얼마나 달라졌는가.",
            "0.10 미만은 안정, 0.10~0.25는 주의, 0.25 이상은 변화로 봅니다(업계 통용값). "
            "정답 라벨 없이 계산되므로 사후 검수보다 먼저 경고를 줍니다. "
            "**표본이 적으면 같은 분포에서도 값이 커지므로** 구간을 병합하고, "
            "60건 미만이면 판정을 보류합니다.",
            primary=True,
        ),
        _entry(
            "hit_rate", "최고점 명중률",
            "히트맵에서 가장 의심스러운 점이 실제 결함 안에 있었던 비율.",
            "낮아 보여도 **무작위 대비로 봐야 합니다.** VisA PCB 결함은 이미지의 0.57%에 불과한 "
            "미세 결함이라, 아무 데나 찍었을 때 맞을 확률이 0.57%입니다. 실측 19.4%는 "
            "무작위의 약 34배입니다.",
        ),
        _entry(
            "pixel_auroc", "픽셀 AUROC",
            "",
            "픽셀 하나하나를 결함/정상으로 보고 계산한 AUROC입니다. 최고점 명중률과 달리 "
            "히트맵 전체의 모양이 정답과 얼마나 맞는지를 봅니다.",
        ),
        _entry(
            "blur_score", "선명도 (Laplacian 분산)",
            "값이 낮을수록 흐린 이미지입니다.",
            "이미지의 2차 미분(Laplacian) 분산으로 잽니다. 초점이 맞으면 경계가 뚜렷해 값이 커집니다. "
            "이 앱은 100 미만을 흐림 의심으로 봅니다 — 널리 쓰이는 기준이지만 "
            "촬영 조건에 따라 다르므로 자기 데이터에 맞게 조정해야 합니다.",
        ),
        _entry(
            "brightness", "평균 밝기",
            "0(검정)~255(흰색). 너무 어둡거나 밝으면 결함이 묻힙니다.",
            "이 앱은 40 미만을 어두움, 215 초과를 과노출로 봅니다. 조명이 바뀌면 이 값이 먼저 "
            "움직이므로, 운영 중 드리프트의 조기 신호이기도 합니다.",
        ),
    ]
}


# 화면에 숫자로 노출되지만 **근거가 관습이거나 이 프로젝트가 임의로 정한** 값들.
# 물리 상수처럼 보이면 자기 라인에 맞게 조정할 생각을 못 하므로 출처를 함께 밝힌다.
ARBITRARY: dict[str, str] = {
    "psi": "PSI 0.10 / 0.25는 신용평가 분야에서 굳어진 관습값입니다. 이 데이터로 검증한 값이 아닙니다.",
    "drift_samples": "드리프트 판정 최소 표본 60건, 구간당 30건은 **측정으로 정했습니다** — "
                     "10구간 기준 n=40에서 같은 분포의 PSI 중앙값이 0.24까지 나왔습니다.",
    "new_labels": "재학습 제안 기준인 신규 라벨 50건은 임의값입니다. 운영 이력이 쌓이면 조정하세요.",
    "recall_margin": "재현율 낙폭 0.05도 임의값입니다. 미탐 비용이 크면 더 작게 잡아야 합니다.",
    "quality": "선명도 100, 밝기 40/215는 일반적인 기준이며 이 데이터로 검증하지 않았습니다.",
}


def caption(key: str) -> str:
    """바로 보이는 한 줄 설명. 핵심 지표가 아니면 빈 문자열."""
    entry = METRICS.get(key)
    return entry.short if entry else ""


def detail(key: str) -> str:
    """`?` 아이콘에 넣을 설명. 캡션이 없는 지표는 캡션 내용까지 합쳐서 준다."""
    entry = METRICS.get(key)
    if entry is None:
        return ""
    if entry.short and entry.detail:
        return f"{entry.short}\n\n{entry.detail}"
    return entry.detail or entry.short


# --- 표의 열 이름 설명 (G8) -------------------------------------------------
#
# 지표를 지표 카드에서는 캡션으로 설명해 놓고 표에서는 아무 설명 없이 내보내면, 같은 값을
# 두 화면에서 다르게 만나게 된다. 표의 열 도움말도 이 사전 한 곳에서 나오게 한다.
#
# 여기 없는 열은 도움말을 붙이지 않는다. 이름만으로 뜻이 분명한 열(version, created_at
# 같은 것)에 굳이 설명을 다는 것은 소음이다.

_COLUMN_TO_METRIC: dict[str, str] = {
    "recall": "recall",
    "재현율": "recall",
    "precision": "precision",
    "정밀도": "precision",
    "기대 정밀도": "precision",
    "f1": "f1",
    "auroc": "auroc",
    "AUROC": "auroc",
    "psi": "psi",
    "PSI": "psi",
    "검수량 절감률": "reduction",
}

_COLUMN_EXTRA: dict[str, str] = {
    # 혼동행렬 성분 — 약어라 처음 보면 무엇의 줄임인지 알 수 없다.
    "tp": "진짜 결함을 결함으로 맞게 판정한 건수.",
    "fp": "정상품을 결함으로 잘못 판정한 건수 (오탐). 사람이 헛되이 보게 되는 물량입니다.",
    "fn": "결함을 정상으로 놓친 건수 (미탐). **이 프로젝트가 가장 피하려는 오류입니다.**",
    "tn": "정상품을 정상으로 맞게 판정한 건수.",
    "n": "이 줄을 계산하는 데 쓴 표본 수. 적으면 지표가 흔들립니다.",
    # 실험 기록
    "run_id": "실행 하나를 가리키는 식별자. 4단계 모델 레지스트리가 이 값으로 모델을 찾습니다.",
    "kind": "베이스라인(지도학습)인지 이상탐지인지.",
    "threshold": "이 값보다 점수가 높으면 결함으로 판정합니다. 낮출수록 재현율은 오르고 오탐도 늡니다.",
    "n_train": "학습에 쓴 이미지 수.",
    "n_eval": "평가에 쓴 이미지 수.",
    # 불량률 환산표
    "불량률": "전체 물량 중 실제 결함의 비율. 현장 값으로 바꾸면 아래 숫자가 전부 달라집니다.",
    "검수 비율": "모델이 결함으로 지목해 사람이 봐야 하는 물량의 비율.",
    # 드리프트
    "feature": "이미지에서 뽑은 특징의 내부 이름. 뜻은 옆 열에 있습니다.",
    "level": "PSI 관습 구간 — 안정 / 주의 / 이동.",
    "baseline_mean": "기준선(승격 시점) 데이터에서의 평균값.",
    "current_mean": "지금 들어오는 데이터에서의 평균값.",
    "shift_sigma": "기준선 표준편차 몇 개만큼 옮겨갔는지. 부호는 방향입니다.",
}


def column_help(name: str) -> str:
    """표의 열 이름에 붙일 한 줄 설명. 설명할 것이 없으면 빈 문자열."""
    key = _COLUMN_TO_METRIC.get(name)
    if key:
        return caption(key) or detail(key)
    return _COLUMN_EXTRA.get(name, "")


# --- 결과 판정 --------------------------------------------------------------

LEVEL_GOOD = "good"
LEVEL_USABLE = "usable"
LEVEL_WEAK = "weak"

# AUROC 해석 구간. 공개 벤치마크(사전학습 특징 기반 0.95 내외)를 참고해 잡은 값이며
# 절대 기준은 아니다.
AUROC_GOOD = 0.90
AUROC_USABLE = 0.75


@dataclass(frozen=True)
class Verdict:
    level: str
    headline: str
    actions: list[str] = field(default_factory=list)


def verdict(metrics: dict, impact: dict, *, target_recall: float = 0.95) -> Verdict:
    """지표를 보고 "쓸 만한지, 아니면 무엇을 해야 하는지"를 문장으로 만든다.

    숫자만 보여주고 끝내면 초보자는 다음에 무엇을 할지 모른다. 판단과 조치를 함께 낸다.
    """
    auroc = float(metrics.get("auroc") or float("nan"))
    recall = float(metrics.get("recall", 0.0))
    reduction = float(impact.get("reduction_ratio", 0.0))
    precision = float(impact.get("precision", 0.0))

    actions: list[str] = []

    if auroc != auroc:   # NaN
        level = LEVEL_WEAK
        headline = "AUROC를 계산할 수 없습니다 — 평가 분할에 한 종류의 라벨만 있습니다."
        actions.append("2단계 **데이터 분할**에서 평가 분할에 정상과 결함이 모두 들어가게 나누세요.")
        return Verdict(level, headline, actions)

    if auroc >= AUROC_GOOD:
        level = LEVEL_GOOD
        headline = f"분리 성능이 좋습니다 (AUROC {auroc:.3f}). 정상과 결함이 잘 나뉩니다."
    elif auroc >= AUROC_USABLE:
        level = LEVEL_USABLE
        headline = (
            f"쓸 만하지만 완벽하지 않습니다 (AUROC {auroc:.3f}). "
            "자동 판정보다 **사람 검수를 줄이는 1차 스크리닝**에 맞습니다."
        )
    else:
        level = LEVEL_WEAK
        headline = (
            f"분리 성능이 약합니다 (AUROC {auroc:.3f}). "
            "지금 특징으로는 이 결함을 충분히 구분하지 못합니다."
        )

    if recall < target_recall:
        actions.append(
            f"재현율 {recall:.1%}가 목표 {target_recall:.0%}에 못 미칩니다. "
            "임계값을 낮추면 올라가지만 오탐이 늘어납니다 — 아래 슬라이더로 직접 확인하세요."
        )
    if reduction <= 0:
        actions.append(
            "**검수량이 전혀 줄지 않습니다.** 거의 모든 이미지를 사람이 다시 봐야 하므로 "
            "지금 상태로는 도입 효과가 없습니다."
        )
    elif precision < 0.5:
        actions.append(
            f"실제 불량률에서 정밀도가 {precision:.1%}입니다. 자동 판정에는 쓸 수 없고, "
            "모델이 올린 것을 사람이 다시 확인하는 방식이어야 합니다."
        )
    if level == LEVEL_WEAK:
        actions.append(
            "개선 방향: ① 라벨을 더 모으거나 ② 결함이 잘 드러나도록 촬영 조건을 고치거나 "
            "③ 더 표현력 있는 특징(사전학습 CNN)으로 바꿉니다."
        )
    if not actions:
        actions.append("4단계에서 이 모델을 등록·승격하면 운영 감시가 시작됩니다.")
    return Verdict(level, headline, actions)


# --- 드리프트 원인을 현장 언어로 (A3) --------------------------------------
#
# 화면에는 `gray_mean`, `lap_p99` 같은 내부 특징명이 그대로 나온다. 이걸 보고
# "조명이 바뀐 것 같다"까지 연결하려면 특징이 무엇을 재는지 알아야 하는데, 그건
# 이 파이프라인을 만든 사람만 안다. 접두사로 묶어 현장에서 확인할 것으로 옮긴다.

@dataclass(frozen=True)
class FeatureFamily:
    meaning: str      # 이 특징이 무엇을 재는가
    cause: str        # 이 값이 변했다면 현장에서 무엇을 의심할까


# 접두사가 긴 것부터 검사한다 (`lap_var`가 `lap`보다 먼저 잡히도록).
FEATURE_FAMILIES: tuple[tuple[str, FeatureFamily], ...] = (
    ("gray", FeatureFamily("이미지 밝기", "조명 밝기나 노출 설정이 바뀌었을 가능성")),
    # 실데이터 측정에서 밝기를 올렸을 때 이 계열이 가장 크게 튀었다. 밝기가 변하면 히스토그램
    # 구간이 통째로 밀리기 때문이다. 그래서 "조명 방향"보다 밝기를 먼저 짚는다.
    ("hist", FeatureFamily("밝기 분포 모양", "조명 밝기나 방향이 바뀌었을 가능성 (노출 설정 포함)")),
    ("lap", FeatureFamily("경계의 뚜렷함(선명도)", "카메라 초점이 틀어졌거나 진동·흔들림이 있을 가능성")),
    ("hf", FeatureFamily("미세한 무늬 성분", "초점 저하 또는 이미지 압축·해상도 변경 가능성")),
    ("sobel", FeatureFamily("윤곽선의 세기", "제품 모양이나 놓인 각도가 달라졌을 가능성")),
    ("canny", FeatureFamily("윤곽선의 양", "제품 종류가 바뀌었거나 배경이 달라졌을 가능성")),
    ("lbp", FeatureFamily("표면 질감 패턴", "소재나 표면 처리(도장·코팅)가 바뀌었을 가능성")),
    ("h_", FeatureFamily("색상(색조)", "조명 색온도가 바뀌었거나 소재 색이 달라졌을 가능성")),
    ("s_", FeatureFamily("색의 진하기", "조명 색온도 변화 또는 소재 변경 가능성")),
    ("v_", FeatureFamily("색 밝기", "조명 밝기가 바뀌었을 가능성")),
    ("r_", FeatureFamily("빨강 성분", "조명 색온도가 바뀌었을 가능성")),
    ("g_", FeatureFamily("초록 성분", "조명 색온도가 바뀌었을 가능성")),
    ("b_", FeatureFamily("파랑 성분", "조명 색온도가 바뀌었을 가능성")),
    ("dev_gt", FeatureFamily("평균에서 크게 벗어난 픽셀의 양", "얼룩·이물 증가 또는 조명 불균일 가능성")),
    ("col_profile", FeatureFamily("좌우 방향 밝기 흐름", "조명 위치가 치우쳤거나 제품 정렬이 틀어졌을 가능성")),
    ("row_profile", FeatureFamily("위아래 방향 밝기 흐름", "조명 위치가 치우쳤거나 제품 정렬이 틀어졌을 가능성")),
)


def feature_family(name: str) -> FeatureFamily | None:
    """특징명을 그 특징이 속한 묶음으로 옮긴다. 모르는 이름이면 None."""
    text = str(name)
    for prefix, family in FEATURE_FAMILIES:
        if text.startswith(prefix):
            return family
    return None


def feature_meaning(name: str) -> str:
    """특징이 무엇을 재는지 한 줄로. 모르면 이름을 그대로 돌려준다."""
    family = feature_family(name)
    return family.meaning if family else str(name)


def drift_causes(feature_names) -> list[str]:
    """분포가 변한 특징들로부터 현장에서 확인할 것을 추린다.

    같은 원인을 가리키는 특징이 여러 개 뜨는 것이 보통이므로(조명이 바뀌면 밝기 계열이
    한꺼번에 움직인다) **중복을 없애고 순서를 유지한다.** 원인을 나열하는 것이 목적이지
    몇 개가 떴는지를 세는 것이 목적이 아니다.
    """
    causes: list[str] = []
    for name in feature_names:
        family = feature_family(name)
        if family and family.cause not in causes:
            causes.append(family.cause)
    return causes


# --- 승격 전 점검 (A2) ------------------------------------------------------
#
# 지금은 성능이 어떻든 경고 없이 '서비스 중'으로 올릴 수 있다. 초보자가 하기 쉬운 실수이고,
# 한 번 올리면 그 뒤의 감시·재학습 판단이 전부 그 모델을 기준으로 돌아간다.

# 기본값은 D5 제안값(설계 문서 4.4). 실제로 쓰이는 값은 **설정 화면에서 바꿀 수 있고**,
# `promotion_check`가 저장된 설정을 읽는다. 화면마다 다른 값을 쓰면 같은 모델을 두 화면이
# 다르게 평가하게 된다.
PROMOTION_MIN_RECALL = 0.95
PROMOTION_MIN_REDUCTION = 0.50


@dataclass(frozen=True)
class PromotionCheck:
    passed: bool
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def promotion_check(
    metrics: dict,
    impact: dict | None = None,
    *,
    min_recall: float | None = None,
    min_reduction: float | None = None,
) -> PromotionCheck:
    """이 모델을 서비스에 올려도 되는지 기준 대비 점검한다.

    기준을 넘기지 않으면 **사용자 설정**을 읽는다. 설정 화면에서 목표 재현율을 올렸는데
    승격 점검이 옛 값으로 판단하면 안 되기 때문이다.

    **막지는 않는다.** 시연이나 비교 목적으로 일부러 낮은 모델을 올릴 수도 있다.
    대신 무엇이 미달인지 알리고 확인을 받는다.
    """
    if min_recall is None or min_reduction is None:
        from . import settings as settings_module

        current = settings_module.load()
        min_recall = current.target_recall if min_recall is None else min_recall
        min_reduction = current.min_reduction if min_reduction is None else min_reduction

    problems: list[str] = []
    notes: list[str] = []

    recall = metrics.get("recall")
    if recall is None or recall != recall:
        notes.append("재현율 기록이 없어 성능을 점검할 수 없습니다.")
    elif float(recall) < min_recall:
        problems.append(
            f"재현율 {float(recall):.1%}가 기준 {min_recall:.0%}에 못 미칩니다 — "
            "결함을 그만큼 놓친 채로 운영이 시작됩니다."
        )

    if impact is not None:
        reduction = float(impact.get("reduction_ratio", 0.0))
        if reduction < min_reduction:
            problems.append(
                f"검수량 절감률 {reduction:.0%}가 기준 {min_reduction:.0%}에 "
                "못 미칩니다 — 사람이 볼 물량이 충분히 줄지 않아 도입 효과가 작습니다."
            )
    else:
        notes.append("오탐률 기록이 없어 검수량 절감률을 계산하지 못했습니다.")

    notes.append(
        f"적용 기준: 재현율 {min_recall:.0%} · 검수량 절감률 {min_reduction:.0%}. "
        "**설정 화면에서 바꿀 수 있습니다** — 미탐 비용이 크면 재현율 목표를 더 높게 잡으세요."
    )
    return PromotionCheck(passed=not problems, problems=problems, notes=notes)


# --- 용어 평문화 (B4) -------------------------------------------------------
#
# 화면의 1차 이름 자체가 전문 용어다. 도움말을 다는 것과 별개로, 이름 옆에 **무엇을 하는
# 곳인지** 한 줄을 붙여야 초보자가 탭을 열어볼지 말지 판단할 수 있다.

TERMS: dict[str, str] = {
    "베이스라인": "이미지 통계로 정상/결함을 배우는 기본 방식입니다. 결함 예시가 있어야 합니다.",
    "이상탐지": "정상 제품만 학습시켜 '평소와 다른 것'을 찾습니다. 결함 예시가 적어도 됩니다.",
    "위치별 분포 학습": "이미지의 자리마다 '평소 모습'을 따로 기억합니다. PCB처럼 늘 같은 자리에 "
                  "같은 것이 오는 제품에 유리합니다.",
    "층화 분할": "정상/결함 비율을 유지한 채 학습용과 평가용으로 나눕니다. 한쪽에 몰리면 "
             "평가를 믿을 수 없습니다.",
    "manifest": "수집한 이미지 목록입니다. 어떤 파일을 어디서 가져왔는지만 담습니다.",
    "레지스트리": "지금 무엇을 쓰고 있는지의 기록입니다. 버전을 남겨 두어야 문제가 생겼을 때 "
              "이전 것으로 되돌릴 수 있습니다.",
    "드리프트": "현장 조건이 변해 들어오는 이미지가 학습 때와 달라지는 것입니다. "
            "모델은 그대로인데 성능이 떨어집니다.",
    "임계값": "몇 점부터 결함으로 볼지 정하는 값입니다. 낮추면 많이 잡고, 높이면 적게 잡습니다.",
    "ROI": "이미지에서 결함이 있는 자리를 네모로 표시한 것입니다.",
    "사후 검수": "모델이 판정한 뒤 사람이 다시 확인하는 것입니다. 이게 있어야 실제 성능을 잽니다.",
}


def term(name: str) -> str:
    """용어를 한 줄 설명으로. 등록되지 않은 말이면 빈 문자열."""
    return TERMS.get(name, "")
