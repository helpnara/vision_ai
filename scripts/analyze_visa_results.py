"""VisA 검증 결과를 D5(정량 합격 기준) 판단에 쓸 수 있는 형태로 해석한다.

## 왜 별도 해석이 필요한가

`validate_visa.py`가 보고하는 정밀도는 **공식 test 구성(정상 100 : 결함 100)** 기준이다.
실제 생산 라인의 불량률은 그보다 훨씬 낮으므로, 이 정밀도를 그대로 현장 기대치로
쓰면 **크게 과대평가**된다. 같은 모델이라도 불량률이 낮아지면 정상품이 압도적으로 많아져
오탐 건수가 진탐 건수를 쉽게 넘어선다.

재현율(TPR)과 오탐률(FPR)은 불량률과 무관한 모델 고유 특성이므로, 이 둘로부터
임의의 불량률 p에서의 정밀도를 환산할 수 있다.

    정밀도 = TPR·p / (TPR·p + FPR·(1-p))

실행:
    PYTHONPATH=src python scripts/analyze_visa_results.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from vision_ai import evaluate  # noqa: E402

PREVALENCES = (0.50, 0.10, 0.05, 0.01)

# 환산 로직은 `vision_ai.evaluate`에 한 곳으로 모아 두었다 — 앱 화면과 같은 계산을 써야
# 문서의 숫자와 화면의 숫자가 어긋나지 않는다.
precision_at = evaluate.precision_at_prevalence


def main() -> int:
    parser = argparse.ArgumentParser(description="VisA 검증 결과 해석")
    parser.add_argument("--input", default="artifacts/reports/visa_validation.json")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        print(f"결과 파일이 없습니다: {path}\n먼저 scripts/validate_visa.py를 실행하세요.")
        return 1

    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["categories"]

    print("## 1. 카테고리별 성능 (공식 1cls 분할)\n")
    print("| 카테고리 | AUROC | AP | 재현율 | 정밀도 | 오탐률 | 위치 적중률 | 픽셀 AUROC |")
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        held = row.get("held_out", {})
        loc = row.get("localization") or {}
        print(
            f"| {row['category']} | {row['auroc']:.3f} | {row['average_precision']:.3f} "
            f"| {held.get('recall', float('nan')):.3f} | {held.get('precision', float('nan')):.3f} "
            f"| {held.get('false_alarm_rate', float('nan')):.3f} "
            f"| {loc.get('hit_rate', float('nan')):.3f} | {loc.get('pixel_auroc_mean', float('nan')):.3f} |"
        )
    print(
        f"\n평균 AUROC **{report['mean_auroc']:.3f}** / 평균 AP **{report['mean_average_precision']:.3f}**"
    )

    locs = [r["localization"] for r in rows if r.get("localization")]
    if locs and "mask_area_fraction" in locs[0]:
        hit = sum(loc["hit_rate"] for loc in locs) / len(locs)
        area = sum(loc["mask_area_fraction"] for loc in locs) / len(locs)
        print(
            f"\n위치 적중률 평균 {hit:.1%}는 낮아 보이지만, VisA PCB 결함은 이미지의 "
            f"{area:.2%}에 불과한 미세 결함이다. 아무 데나 찍었을 때의 적중 확률이 {area:.2%}이므로 "
            f"**무작위 대비 약 {hit / area:.0f}배**다."
        )

    print("\n## 2. 불량률별 기대 정밀도\n")
    print(
        "위 정밀도는 정상:결함 = 1:1인 시험 구성에서 나온 값이다. "
        "실제 불량률에서는 아래와 같이 떨어진다.\n"
    )
    header = " | ".join(f"불량률 {p:.0%}" for p in PREVALENCES)
    print(f"| 카테고리 | 재현율 | 오탐률 | {header} |")
    print("|---|---|---|" + "---|" * len(PREVALENCES))

    usable = [r for r in rows if "recall" in r.get("held_out", {})]
    for row in usable:
        held = row["held_out"]
        tpr, fpr = held["recall"], held["false_alarm_rate"]
        cells = " | ".join(f"{precision_at(tpr, fpr, p):.3f}" for p in PREVALENCES)
        print(f"| {row['category']} | {tpr:.3f} | {fpr:.3f} | {cells} |")

    if usable:
        mean_tpr = sum(r["held_out"]["recall"] for r in usable) / len(usable)
        mean_fpr = sum(r["held_out"]["false_alarm_rate"] for r in usable) / len(usable)
        print("\n### 평균 기준 해석\n")
        print(f"- 재현율 {mean_tpr:.1%}, 오탐률 {mean_fpr:.1%}")
        for p in PREVALENCES:
            impact = evaluate.business_impact(mean_tpr, mean_fpr, prevalence=p, volume=1000)
            print(
                f"- 불량률 {p:.0%}: 정밀도 {impact['precision']:.1%} · "
                f"검수량 절감 {impact['reduction_ratio']:.0%} — 1,000장 검사 시 "
                f"오탐 {impact['false_alarms']:.0f}건을 걸러내고 결함 {impact['missed']:.0f}건을 놓친다"
            )

    print("\n## 3. 판정\n")
    if usable:
        low = precision_at(mean_tpr, mean_fpr, 0.01)
        print(
            f"재현율 {mean_tpr:.0%}는 확보되나, 불량률 1%에서 정밀도가 {low:.1%}까지 떨어진다. "
            "즉 **자동 판정용으로는 부족하고, 사람 검수 부하를 줄이는 1차 스크리닝용**으로 봐야 한다.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
