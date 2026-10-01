import type { JobState } from "../api";
import { Alert } from "./ui";

/**
 * 오래 걸리는 작업의 진행을 알린다 (예전 ui.Progress).
 *
 * 막대만 채우면 «얼마나 더 기다려야 하는지»를 알 수 없다. 특징 추출은 4,584장 기준 70초가
 * 넘고, 그동안 화면이 멈춘 것처럼 보이면 사람은 새로고침을 누른다. 그래서 **왜 오래 걸리는지**
 * (note)를 먼저 말하고, 진행하면서 **남은 시간**(서버가 실측으로 계산)을 갱신한다.
 */
export function JobProgress({ job, error, label }: { job: JobState | null; error?: string | null; label?: string }) {
  if (error) return <Alert kind="error">{error}</Alert>;
  if (!job || job.status === "done") return null;
  const fraction = job.fraction;
  const head = label ?? job.label ?? "작업 중";
  const count = job.total ? ` ${job.done.toLocaleString()}/${job.total.toLocaleString()}` : "";
  const eta = job.eta_text ? ` · 남은 시간 약 ${job.eta_text}` : "";
  return (
    <div className="progress" aria-live="polite">
      <div className="bar">
        <div className={fraction === null ? "indeterminate" : ""} style={fraction === null ? undefined : { width: `${fraction * 100}%` }} />
      </div>
      <div className="text">
        <span className="spinner" />{job.text || `${head}${count}`}{eta}
      </div>
      {job.note ? <div className="caption">{job.note}</div> : null}
    </div>
  );
}
