"""영상 프레임 추출 비용을 실제 영상으로 잰다 (설계안 §7의 숫자를 갱신하기 위한 것).

## 왜 다시 재는가

설계안 §7의 숫자는 **단색 합성 영상 기준이라 낙관적이다.** 단색 면은 압축이 거의 공짜라
디코딩 비용이 실제 촬영본보다 훨씬 싸게 나온다. 중복 제거도 마찬가지다 — 배경이 완전히
같으면 "거의 같은 프레임"이 실제보다 훨씬 잘 잡힌다.

이 스크립트는 **아무 영상에나** 같은 측정을 돌린다. 합성 영상, 공개 데이터셋, 직접 촬영본
어느 것이든 넣으면 같은 표가 나오므로 서로 비교할 수 있다.

## 무엇을 재는가

1. **디코딩 비용** — `grab()`만 / `read()` 전부 / `stride` 추출. 구현이 고른
   "건너뛸 프레임은 `grab()`, 채택할 프레임만 `retrieve()`" 전략이 실제로 싼지 확인한다.
2. **중복 제거** — 임계값별로 몇 장이 걸러지는지. 정지 구간이 많은 CCTV에서 이 값이
   추출 장수를 좌우한다.
3. **처리량** — 영상 1초를 처리하는 데 걸리는 시간. 10분 영상이 얼마나 걸릴지 여기서 나온다.

## 실행

    PYTHONPATH=src python scripts/measure_video.py <영상경로> [<영상경로> ...]
    PYTHONPATH=src python scripts/measure_video.py --synthetic        # 합성 3종 비교
    PYTHONPATH=src python scripts/measure_video.py cam.mp4 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vision_ai import video  # noqa: E402


REPEATS = 3
"""같은 측정을 몇 번 돌릴지. **가장 빠른 값**을 쓴다.

한 번만 재면 잡음이 결론을 뒤집는다. 실제로 0.13초와 0.33초가 같은 코드에서 나와
"grab이 오히려 손해"라는 잘못된 표가 찍힌 적이 있다. 느린 쪽은 다른 프로세스나 캐시
미스가 섞인 것이므로, 최소값이 이 코드의 실제 비용에 가장 가깝다.
"""


def _time(function, *, repeats: int = REPEATS) -> tuple[float, object]:
    best = float("inf")
    value = None
    for _ in range(max(repeats, 1)):
        start = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - start)
    return best, value


def _grab_only(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.grab():
        count += 1
    capture.release()
    return count


def _read_all(path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count


def _stride_decoding_everything(path: Path, stride: int) -> int:
    """`stride` 간격으로 뽑되 **모든 프레임을 디코딩한다** — grab 최적화의 대조군.

    `extract()`와 견주려면 이쪽도 같은 일(디코딩 + 저장)을 해야 한다. 앞서 `read` 전부와
    `extract`를 견주었다가 낭패를 봤다 — `extract`는 JPEG 인코딩과 파일 쓰기까지 하므로
    디코딩만 하는 쪽과 비교하면 grab의 효과가 아니라 저장 비용이 섞여 나온다.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    kept = 0
    index = 0
    while True:
        ok, frame = capture.read()          # 건너뛸 프레임도 전부 디코딩한다
        if not ok:
            break
        if index % stride == 0 and frame is not None:
            cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, video.JPEG_QUALITY])
            kept += 1
        index += 1
    capture.release()
    return kept


def measure(path: Path, *, per_second: float = 2.0, repeats: int = REPEATS) -> dict:
    """영상 하나를 재서 결과를 딕셔너리로 돌려준다."""
    info = video.probe(path)
    if not info.usable:
        raise OSError(f"fps나 프레임 수를 읽을 수 없습니다: {path}")

    plan = video.plan_from_rate(info, per_second)
    size_mb = path.stat().st_size / 1_048_576

    grab_seconds, grabbed = _time(lambda: _grab_only(path), repeats=repeats)
    read_seconds, decoded = _time(lambda: _read_all(path), repeats=repeats)

    naive_seconds, _ = _time(lambda: _stride_decoding_everything(path, plan.stride), repeats=repeats)

    dedupe: dict[str, dict] = {}
    extract_seconds = 0.0
    with tempfile.TemporaryDirectory() as work:
        # 중복 제거 없음 = 추출 비용의 기준선
        extract_seconds, plain = _time(
            lambda: video.extract(
                path, stride=plan.stride, out_dir=Path(work) / "plain", min_change=None
            ),
            repeats=repeats,
        )
        for threshold in (0.002, 0.005, 0.02):
            seconds, result = _time(
                lambda t=threshold: video.extract(
                    path, stride=plan.stride,
                    out_dir=Path(work) / f"d{t}", min_change=t,
                )
            )
            dedupe[f"{threshold}"] = {
                "saved": len(result.saved),
                "dropped": result.dropped_similar,
                "seconds": round(seconds, 3),
            }

    return {
        "name": path.name,
        "size_mb": round(size_mb, 1),
        "resolution": f"{info.width}×{info.height}",
        "fps": round(info.fps, 1),
        "frames": info.frame_count,
        "duration_sec": round(info.duration_sec, 1),
        "stride": plan.stride,
        "grab_seconds": round(grab_seconds, 3),
        "read_seconds": round(read_seconds, 3),
        "extract_seconds": round(extract_seconds, 3),
        "naive_seconds": round(naive_seconds, 3),
        "grabbed": grabbed,
        "decoded": decoded,
        "extracted": len(plain.saved),
        "grab_saving": round(naive_seconds / extract_seconds, 2) if extract_seconds else None,
        "realtime_factor": round(info.duration_sec / extract_seconds, 1) if extract_seconds else None,
        "dedupe": dedupe,
    }


def _synthetic(work: Path) -> list[Path]:
    """비교용 합성 영상 3종. 무늬가 있고 없고가 비용을 얼마나 가르는지 보기 위한 것."""
    made = []
    for name, kwargs in [
        ("flat_480p", {"size": (640, 480), "textured": False}),
        ("textured_480p", {"size": (640, 480), "textured": True}),
        ("textured_720p", {"size": (1280, 720), "textured": True}),
        # CCTV는 대부분의 시간이 빈 장면이다. 중복 제거의 값어치가 여기서 나온다.
        ("cctv_720p_idle", {"size": (1280, 720), "textured": True, "idle_seconds": 7}),
    ]:
        made.append(video.make_sample(work / f"{name}.mp4", seconds=20, fps=30, **kwargs))
    return made


def _report(rows: list[dict]) -> None:
    print("\n## 영상")
    print("| 영상 | 해상도 | fps | 길이 | 크기 | 간격 |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['name']} | {row['resolution']} | {row['fps']:.0f} | "
            f"{row['duration_sec']:.0f}초 | {row['size_mb']:.1f}MB | {row['stride']}프레임마다 |"
        )

    print("\n## 디코딩 비용 (같은 일을 하는 두 방식 비교)")
    print("| 영상 | grab만 | read 전부 | 전부 디코딩+저장 | grab 건너뛰기+저장 | 아낀 배수 | 실시간 대비 |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['name']} | {row['grab_seconds']:.2f}초 ({row['grabbed']}프레임) | "
            f"{row['read_seconds']:.2f}초 | {row['naive_seconds']:.2f}초 | "
            f"{row['extract_seconds']:.2f}초 ({row['extracted']}장) | "
            f"{row['grab_saving']}× | {row['realtime_factor']}× |"
        )

    print("\n## 중복 제거 (임계값별)")
    print("| 영상 | 0.002 | 0.005 (기본) | 0.02 |")
    print("|---|---|---|---|")
    for row in rows:
        cells = []
        for threshold in ("0.002", "0.005", "0.02"):
            item = row["dedupe"][threshold]
            cells.append(f"{item['saved']}장 (버림 {item['dropped']})")
        print(f"| {row['name']} | " + " | ".join(cells) + " |")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="*", type=Path, help="측정할 영상 파일")
    parser.add_argument("--synthetic", action="store_true", help="비교용 합성 영상 3종을 만들어 측정")
    parser.add_argument("--per-second", type=float, default=2.0, help="초당 몇 장을 뽑을 것인가")
    parser.add_argument("--json", type=Path, help="결과를 이 경로에 JSON으로 저장")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="측정 반복 횟수 (최소값을 쓴다)")
    args = parser.parse_args()

    if not args.videos and not args.synthetic:
        parser.error("영상 경로를 주거나 --synthetic 을 쓰세요.")

    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as work:
        targets = list(args.videos)
        if args.synthetic:
            print("합성 영상 3종을 만드는 중...", flush=True)
            targets = _synthetic(Path(work)) + targets

        for path in targets:
            path = Path(path).expanduser()
            if not path.is_file():
                print(f"건너뜀 — 파일이 없습니다: {path}", file=sys.stderr)
                continue
            print(f"측정 중: {path.name}", flush=True)
            try:
                rows.append(measure(path, per_second=args.per_second, repeats=args.repeats))
            except OSError as exc:
                print(f"건너뜀 — {exc}", file=sys.stderr)

    if not rows:
        print("측정한 영상이 없습니다.", file=sys.stderr)
        return 1

    _report(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장: {args.json}")
    print(
        "\n> 합성 영상 수치는 참고치다. 압축 방식과 장면 복잡도가 실제 촬영본과 다르므로,\n"
        "> 자기 현장 영상으로 다시 재는 것이 맞다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
