"""검증 단계 체크리스트 1~4를 한 번에 돌린다 (docs/process.md §3).

    python scripts/verify.py          # 테스트 · 린트 · 앱 기동
    python scripts/verify.py --web    # + 화면 빌드와 web/dist 차이

왜 스크립트인가: 이관하면서 «기준 숫자와 같은가»를 손으로 대 보다가 ruff 판올림(9 → 115건)을 코드 회귀로
오해할 뻔했다. 기준은 사람이 기억하지 말고 CLAUDE.md에서 읽어 와 기계가 대 보게 한다.

* pytest 기준: CLAUDE.md의 «기준: N passed · M skipped» — passed가 그보다 **적으면** 실패(늘어나는 건 정상).
* ruff 기준: 9건 (CLAUDE.md «명령»). 규칙 묶음은 pyproject.toml에 고정돼 있다.
* 스크린샷(5)·함정 회귀 판단(6)·사용자 확인(7)은 사람이 본다 — 여기서 대신하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 경로는 저장소 루트 기준 절대경로 (CLAUDE.md «의존성 판올림 주의»)
RUFF_BASELINE = 9


def baseline_passed() -> int:
    text = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    m = re.search(r"기준:\s*(\d+)\s*passed", text)
    if not m:
        raise SystemExit("CLAUDE.md에서 «기준: N passed»를 찾지 못했습니다")
    return int(m.group(1))


def run(cmd: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def check_pytest() -> tuple[bool, str]:
    p = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
    last = (p.stdout.strip().splitlines() or [""])[-1]
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", last)) else 0
    failed = re.search(r"(\d+) (failed|error)", last)
    base = baseline_passed()
    ok = p.returncode == 0 and not failed and passed >= base
    return ok, f"{last}  (기준 {base} passed)"


def check_ruff() -> tuple[bool, str]:
    p = run([sys.executable, "-m", "ruff", "check", ".", "--output-format", "concise"])
    if "No module named ruff" in p.stderr:
        return False, "ruff가 설치돼 있지 않습니다 (pip install ruff)"
    count = 0 if "All checks passed" in p.stdout else int(m.group(1)) if (
        m := re.search(r"Found (\d+) error", p.stdout)) else -1
    return count == RUFF_BASELINE, f"{count}건 (기준 {RUFF_BASELINE}건)"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def check_serve() -> tuple[bool, str]:
    port = free_port()
    proc = subprocess.Popen([sys.executable, "serve.py", "--no-browser", "--port", str(port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # 프록시 환경에서도 로컬에 닿게 프록시를 끈 opener를 쓴다 (curl --noproxy '*' 와 같은 이유)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for _ in range(60):
            try:
                health = json.loads(opener.open(f"http://127.0.0.1:{port}/api/health", timeout=2).read())
                root = opener.open(f"http://127.0.0.1:{port}/", timeout=5).status
                return bool(health.get("ok")) and root == 200, f"health={health} · / {root}"
            except OSError:
                time.sleep(0.5)
        return False, "30초 안에 뜨지 않았습니다"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def check_web() -> tuple[bool, str]:
    web = ROOT / "web"
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    if not (web / "node_modules").exists():
        p = run([npm, "ci"], cwd=web)
        if p.returncode:
            return False, "npm ci 실패\n" + p.stderr[-800:]
    p = run([npm, "run", "build"], cwd=web)
    if p.returncode:
        return False, "빌드 실패\n" + (p.stdout + p.stderr)[-800:]
    diff = run(["git", "status", "--porcelain", "--", "web/dist"])
    changed = diff.stdout.strip()
    return not changed, "dist 차이 없음" if not changed else "빌드 결과가 커밋된 dist와 다릅니다 — 같은 커밋에 넣을 것\n" + changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--web", action="store_true", help="화면 빌드와 web/dist 차이까지 본다")
    args = ap.parse_args()

    checks = [("1 pytest", check_pytest), ("2 ruff", check_ruff), ("3 앱 기동", check_serve)]
    if args.web:
        checks.append(("4 화면 빌드", check_web))

    all_ok = True
    for name, fn in checks:
        print(f"… {name}", flush=True)
        ok, note = fn()
        all_ok &= ok
        print(f"{'✅' if ok else '❌'} {name}: {note}", flush=True)
    if not args.web:
        print("· 4 화면 빌드: 건너뜀 (화면을 고쳤으면 --web)")
    print("남은 것(사람이 본다): 5 스크린샷 · 6 함정 회귀 판단 · 7 사용자 확인 — docs/process.md §3")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
