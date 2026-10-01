"""오래 걸리는 작업을 백그라운드에서 돌리고 진행률을 알린다.

Streamlit에서는 학습·추출·추론이 요청 안에서 동기로 돌며 `st.progress`를 갱신했다.
HTTP API에서는 그럴 수 없다 — 요청이 몇 분 동안 열려 있으면 브라우저가 끊고, 끊기면
작업도 함께 죽는다. 그래서 작업을 스레드에 넘기고 **잡 id를 바로 돌려준 뒤** 화면이
`GET /api/jobs/{id}`로 진행률을 묻는다.

워커는 **하나**다. 코어가 CSV 파일을 읽고 쓰는데 잠금이 없으므로, 학습과 추론이 동시에
돌면 manifest가 반쯤 쓰인 상태로 읽힐 수 있다. 단일 사용자 로컬 도구라 동시에 두 개를
돌릴 이유도 없다.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

MIN_SAMPLES_FOR_ETA = 12
"""남은 시간을 말하기 전에 재어 볼 최소 건수. 처음 몇 건은 들쭉날쭉해 값을 못 믿는다."""

KEEP_FINISHED = 50
"""끝난 잡을 메모리에 남겨 두는 수. 화면이 결과를 가져가기 전에 사라지면 안 되지만
무한히 쌓을 이유도 없다."""


def duration_text(seconds: float) -> str:
    """사람이 읽는 시간. '92.3초'보다 '1분 32초'가 기다릴지 말지 판단하기 쉽다."""
    seconds = max(int(round(seconds)), 0)
    if seconds < 60:
        return f"{seconds}초"
    minutes, rest = divmod(seconds, 60)
    return f"{minutes}분 {rest}초" if rest else f"{minutes}분"


def eta_seconds(done: int, total: int, elapsed: float) -> float | None:
    """지금까지의 평균 속도로 본 남은 시간(초). 아직 말할 수 없으면 None."""
    if done < MIN_SAMPLES_FOR_ETA or done >= total or elapsed <= 0:
        return None
    return elapsed / done * (total - done)


@dataclass
class Job:
    id: str
    label: str
    note: str = ""
    status: str = STATUS_QUEUED
    done: int = 0
    total: int = 0
    text: str = ""
    result: Any = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def progress(self, done: int, total: int, text: str | None = None) -> None:
        """코어의 progress 콜백이 부른다. `(done, total)` 모양이면 그대로 넘길 수 있다."""
        with self.lock:
            self.done, self.total = int(done), int(total)
            if text is not None:
                self.text = str(text)

    def progress3(self, index: int, total: int, name: str) -> None:
        """`(index, total, name)` 모양 콜백(quickstart·scenario)용 어댑터."""
        self.progress(index, total, f"{name}... ({index}/{total})")

    def as_dict(self) -> dict:
        with self.lock:
            elapsed = (
                (self.finished or time.monotonic()) - self.started if self.started else 0.0
            )
            remaining = (
                eta_seconds(self.done, self.total, elapsed)
                if self.status == STATUS_RUNNING else None
            )
            return {
                "id": self.id,
                "label": self.label,
                "note": self.note,
                "status": self.status,
                "done": self.done,
                "total": self.total,
                "fraction": (min(self.done / self.total, 1.0) if self.total else None),
                "text": self.text,
                "elapsed": elapsed,
                "elapsed_text": duration_text(elapsed),
                "eta_text": duration_text(remaining) if remaining is not None else None,
                "result": self.result if self.status == STATUS_DONE else None,
                "error": self.error,
            }


class JobRunner:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    def submit(self, label: str, fn: Callable[[Job], Any], *, note: str = "") -> Job:
        """`fn(job)`을 백그라운드에서 돌린다. 반환값이 `job.result`가 된다.

        `fn`은 JSON으로 바꿀 수 있는 값을 돌려줘야 한다 (dict·list·숫자·문자열).
        """
        job = Job(id=uuid.uuid4().hex[:12], label=label, note=note)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[Job], Any]) -> None:
        with job.lock:
            job.status = STATUS_RUNNING
            job.started = time.monotonic()
        try:
            result = fn(job)
        except Exception as exc:  # noqa: BLE001 — 어떤 실패든 화면에 보여야 한다
            with job.lock:
                job.status = STATUS_FAILED
                job.error = f"{exc}" or exc.__class__.__name__
                job.finished = time.monotonic()
            job.traceback = traceback.format_exc()  # type: ignore[attr-defined]
            return
        with job.lock:
            job.result = result
            job.status = STATUS_DONE
            job.finished = time.monotonic()
            if job.total:
                job.done = job.total

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _trim(self) -> None:
        finished = [
            jid for jid in self._order
            if self._jobs[jid].status in (STATUS_DONE, STATUS_FAILED)
        ]
        for jid in finished[:-KEEP_FINISHED] if len(finished) > KEEP_FINISHED else []:
            self._jobs.pop(jid, None)
            self._order.remove(jid)


runner = JobRunner()
