"""지도학습 검출 모델(YOLO) 학습 — **로컬호스트에서 돌린다.**

## 왜 앱 안에서 돌리지 않는가

검출 학습은 GPU가 사실상 필수다. 이 앱은 브라우저에서 도는 PoC이고 배포본 컨테이너는
일회성이라, 학습을 앱에서 시작하면 몇 시간 뒤에 결과째로 잃는다. 앱은 **학습 폴더까지**
만들고(`vision_ai.detection.export_yolo`), 학습은 GPU가 있는 기계에서 이 스크립트로 한다.

## 왜 ultralytics를 의존성에 넣지 않는가

torch까지 딸려 와 설치가 몇 GB 늘어난다. 이 앱의 나머지 기능은 그것 없이 전부 돌고,
검출까지 갈 사람만 필요하다. 그래서 **선택 의존성**으로 두고, 없으면 설치 명령을 알려준다.

## 실행

    # 1. 앱의 2단계에서 「검출 학습 폴더 내보내기」를 누르거나
    PYTHONPATH=src python -c "from vision_ai import detection; print(detection.export_yolo().as_message())"

    # 2. GPU가 있는 기계에서
    pip install ultralytics
    PYTHONPATH=src python scripts/train_detector.py --data artifacts/기본/detection/data.yaml

    # 이어서 학습하거나 모델을 바꾸려면
    PYTHONPATH=src python scripts/train_detector.py --model yolo11s.pt --epochs 200
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_ai import detection  # noqa: E402

INSTALL_HINT = (
    "ultralytics가 설치되어 있지 않습니다.\n\n"
    "    pip install ultralytics\n\n"
    "GPU를 쓰려면 CUDA 버전에 맞는 torch를 먼저 설치하세요 — https://pytorch.org/get-started/"
)


def _check_gpu() -> str:
    """GPU가 있는지 본다. **없으면 막지 않고 알린다** — 느릴 뿐 안 되는 것은 아니다."""
    try:
        import torch
    except ImportError:
        return "torch를 찾을 수 없습니다."
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return (
        "GPU를 찾을 수 없습니다 — CPU로도 돌지만 몇십 배 느립니다. "
        "수백 장 규모의 시험이 아니라면 GPU가 있는 기계를 쓰세요."
    )


def _check_data(data: Path) -> int:
    """학습 폴더가 실제로 쓸 만한지 본다. 시작하고 몇 분 뒤에 아는 것보다 싸다."""
    if not data.is_file():
        raise SystemExit(
            f"학습 설정을 찾을 수 없습니다: {data}\n"
            "앱의 2단계 **라벨 현황** 탭에서 「검출 학습 폴더 내보내기」를 먼저 누르세요."
        )
    root = data.parent
    counts = {}
    for split in detection.SPLITS:
        folder = root / detection.IMAGES_DIR / split
        counts[split] = len(list(folder.glob("*"))) if folder.is_dir() else 0
    print("학습 폴더: " + " · ".join(f"{k} {v:,}장" for k, v in counts.items()))

    if counts.get("train", 0) == 0:
        raise SystemExit("train에 이미지가 없습니다. 2단계에서 데이터 분할을 먼저 실행하세요.")
    if counts.get("val", 0) == 0:
        raise SystemExit(
            "val에 이미지가 없습니다. 검증 세트 없이 학습하면 언제 멈춰야 할지 알 수 없습니다."
        )

    # 빈 txt만 있으면 "결함은 없다"만 배운다. 학습은 멀쩡히 돌고 지표까지 좋게 나온다 —
    # 아무것도 못 찾아도 틀린 적이 없으니까. 몇 시간 뒤에 알 일이 아니다.
    positives = sum(
        1 for txt in (root / detection.LABELS_DIR / "train").glob("*.txt")
        if txt.stat().st_size > 0
    )
    print(f"결함이 그려진 학습 이미지: {positives:,}장")
    if positives == 0:
        raise SystemExit(
            "결함 박스가 있는 학습 이미지가 한 장도 없습니다. 이대로 돌리면 모델은\n"
            "'결함은 없다'만 배웁니다. 2단계에서 박스를 그리고 데이터 분할을 다시 실행하세요."
        )
    return counts["train"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=detection.default_root() / detection.DATA_FILE,
        help="export_yolo()가 만든 data.yaml",
    )
    parser.add_argument("--model", default="yolo11n.pt", help="시작 가중치 (n<s<m<l<x 순으로 큼)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="학습 입력 크기")
    parser.add_argument("--batch", type=int, default=16, help="한 번에 볼 장수. GPU 메모리가 좌우한다")
    parser.add_argument("--name", default="defect", help="결과 폴더 이름")
    parser.add_argument("--dry-run", action="store_true", help="점검만 하고 학습은 시작하지 않는다")
    args = parser.parse_args()

    print(_check_gpu())
    train_count = _check_data(args.data)

    if args.dry_run:
        print("\n--dry-run 이므로 여기서 멈춥니다. 위 점검이 통과하면 그대로 돌리면 됩니다.")
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(INSTALL_HINT) from None

    if train_count < 200:
        print(
            f"\n⚠️ 학습 이미지가 {train_count:,}장뿐입니다. 검출 모델은 이 정도로는 잘 학습되지\n"
            "   않습니다 — 결과가 나빠도 모델이 아니라 데이터 문제일 가능성이 큽니다."
        )

    project = args.data.parent / "runs"
    print(f"\n학습 시작 — 결과는 {project / args.name} 에 쌓입니다.")
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project),
        name=args.name,
        exist_ok=True,
    )

    best = project / args.name / "weights" / "best.pt"
    if best.is_file():
        keep = args.data.parent / "best.pt"
        shutil.copy2(best, keep)
        print(f"\n가장 좋은 가중치를 복사했습니다: {keep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
