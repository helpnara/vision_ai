import { useEffect, useState } from "react";

/**
 * 입력이 멈춘 뒤에야 값을 내보낸다. 폴더 경로·화각 숫자처럼 글자마다 서버를 부르면 안 되는
 * 입력에 쓴다. (공유 hooks.ts에 넣으면 좋을 것 — 이식 중에는 공유 파일을 고치지 않는다.)
 *
 * **객체를 넘길 때는 useMemo로 참조를 고정할 것.** 매 렌더마다 새 객체를 넣으면 디바운스가
 * 끝날 때마다 다시 렌더되어 새 객체가 들어오고, 250ms마다 서버를 부르는 고리가 된다 —
 * 영상 탭의 화각 점검·추출 계획에서 실제로 그렇게 돌았다 (서버 로그로 발견).
 */
export function useDebounced<T>(value: T, ms = 300): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const t = window.setTimeout(() => setSettled(value), ms);
    return () => window.clearTimeout(t);
  }, [value, ms]);
  return settled;
}

export type Notice = { kind: "success" | "info" | "warning" | "error"; text: string; icon?: string };
