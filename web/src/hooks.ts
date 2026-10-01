import { useCallback, useEffect, useRef, useState } from "react";
import { get, getJob, type JobState } from "./api";

/** GET 한 번 + 다시 읽기. 화면마다 같은 모양의 로딩·오류 처리를 반복하지 않기 위한 것. */
export function useFetch<T>(url: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(!!url);
  const version = useRef(0);

  const reload = useCallback(async () => {
    if (!url) return;
    const mine = ++version.current;
    setLoading(true);
    try {
      const got = await get<T>(url);
      if (mine === version.current) {
        setData(got);
        setError(null);
      }
    } catch (e) {
      if (mine === version.current) setError((e as Error).message);
    } finally {
      if (mine === version.current) setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reload, ...deps]);

  return { data, error, loading, reload, setData };
}

/**
 * 백그라운드 잡을 시작하고 끝날 때까지 진행률을 폴링한다.
 *
 * `start(fn)`에 «잡 id를 돌려주는 요청»을 넘긴다. 끝나면 `onDone(result)`가 불린다.
 * 폴링 간격은 500ms — 진행률 막대가 끊겨 보이지 않으면서 서버를 괴롭히지 않는 값.
 */
export function useJob<R = unknown>(onDone?: (result: R) => void) {
  const [job, setJob] = useState<JobState<R> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  const stop = () => {
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = null;
  };

  const poll = useCallback(async (id: string) => {
    try {
      const state = await getJob<R>(id);
      setJob(state);
      if (state.status === "done") {
        stop();
        onDoneRef.current?.(state.result as R);
      } else if (state.status === "failed") {
        stop();
        setError(state.error ?? "작업이 실패했습니다.");
      } else {
        timer.current = window.setTimeout(() => void poll(id), 500);
      }
    } catch (e) {
      stop();
      setError((e as Error).message);
    }
  }, []);

  const start = useCallback(
    async (begin: () => Promise<{ job_id: string }>) => {
      setError(null);
      setJob({
        id: "", label: "", note: "", status: "queued", done: 0, total: 0, fraction: null,
        text: "", elapsed: 0, elapsed_text: "0초", eta_text: null, result: null, error: null,
      });
      try {
        const { job_id } = await begin();
        await poll(job_id);
      } catch (e) {
        setJob(null);
        setError((e as Error).message);
      }
    },
    [poll],
  );

  const reset = useCallback(() => {
    stop();
    setJob(null);
    setError(null);
  }, []);

  useEffect(() => stop, []);

  const running = !!job && (job.status === "queued" || job.status === "running");
  return { job, error, running, start, reset };
}
