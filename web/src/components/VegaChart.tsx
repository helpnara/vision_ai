import { useEffect, useRef } from "react";

/**
 * Vega-Lite 차트. 명세(spec)는 서버(`vision_ai.charts`)가 만든다 — 실측으로 잡은 높이·줄 순서
 * 같은 «데인 것»들이 거기 주석과 테스트로 잠겨 있기 때문이다. 화면은 그리기만 한다.
 *
 * vega 묶음은 크므로 이 컴포넌트를 처음 쓸 때만 내려받는다.
 */
export function VegaChart({ spec, className }: { spec: object | null | undefined; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!spec || !ref.current) return;
    let view: { finalize: () => void } | null = null;
    let cancelled = false;
    void import("vega-embed").then(({ default: embed }) => {
      if (cancelled || !ref.current) return;
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      return embed(ref.current, spec as never, {
        actions: false,
        renderer: "svg",
        theme: dark ? "dark" : undefined,
      }).then((r) => { view = r.view; });
    });
    return () => { cancelled = true; view?.finalize(); };
  }, [spec]);
  return <div ref={ref} className={`vega-wrap ${className ?? ""}`} />;
}
