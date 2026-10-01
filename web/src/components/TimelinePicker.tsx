import { useRef, useState } from "react";

export interface Tick { t: number; mark: string }

const MARK_COLORS: Record<string, string> = { "정상": "#2e9e5b", "결함": "#d64545", "미라벨": "#9aa3ae" };

/**
 * 영상 타임라인 위에서 시간 구간을 끌어 고른다 (예전 ui.timeline_picker).
 * 프레임마다 눈금을 라벨 색으로 찍고, 드래그한 범위를 초 단위로 돌려준다.
 */
export function TimelinePicker({ ticks, duration, span, onChange }: {
  ticks: Tick[]; duration: number; span: [number, number] | null; onChange: (span: [number, number] | null) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const total = Math.max(duration, 0.001);
  const toSec = (cx: number) => {
    const rect = wrap.current!.getBoundingClientRect();
    return Math.min(Math.max((cx - rect.left) / rect.width, 0), 1) * total;
  };
  const onDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const s = toSec(e.clientX);
    setDrag({ a: s, b: s });
  };
  const onMove = (e: React.PointerEvent) => { if (drag) setDrag({ ...drag, b: toSec(e.clientX) }); };
  const onUp = () => {
    if (!drag) return;
    const start = Math.min(drag.a, drag.b), end = Math.max(drag.a, drag.b);
    setDrag(null);
    onChange(end > start ? [start, end] : null);
  };
  const live = drag ? [Math.min(drag.a, drag.b), Math.max(drag.a, drag.b)] as [number, number] : span;
  const axis = [0, 0.25, 0.5, 0.75, 1].map((f) => f * total);
  const kinds = Array.from(new Set(ticks.map((t) => t.mark)));
  return (
    <div>
      <div ref={wrap} className="timeline" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={() => setDrag(null)} style={{ cursor: "col-resize" }}>
        {ticks.map((tick, i) => (
          <div key={i} className="tick" style={{ left: `${(tick.t / total) * 100}%`, background: MARK_COLORS[tick.mark] ?? "#3b7dd8" }} title={`${tick.t.toFixed(1)}초 · ${tick.mark}`} />
        ))}
        {live ? <div className="span" style={{ left: `${(live[0] / total) * 100}%`, width: `${((live[1] - live[0]) / total) * 100}%` }} /> : null}
        {axis.map((s) => <span key={s} className="axis" style={{ left: `${(s / total) * 100}%` }}>{s.toFixed(1)}s</span>)}
      </div>
      <div className="legend">
        {kinds.map((k) => <span key={k}><span className="swatch" style={{ background: MARK_COLORS[k] ?? "#3b7dd8" }} />{k}</span>)}
        <span>영상 시각 (초)</span>
      </div>
    </div>
  );
}
