import { useRef, useState } from "react";

export interface Box { x: number; y: number; w: number; h: number; label?: string }

/**
 * 이미지 위에서 드래그해 결함 영역을 지정한다 (예전 ui.roi_picker).
 *
 * 좌표는 **원본 픽셀 기준**으로 돌려준다. 화면에 그릴 때는 `view`(원본에서 지금 보는 범위)를
 * 잘라 보여주고, 드래그한 화면 좌표를 그 범위로 되돌린다. 확대(zoom)는 `view`를 좁히는 것일 뿐
 * 좌표 계산은 같다 — 되짚는 계산이 따로 필요 없다.
 *
 * `saved`는 이미 그려 둔 박스 (초록), `pending`은 지금 끌어 놓은 영역 (빨강).
 */
export function RoiPicker({ src, width, height, view, pending, saved, onChange, displayWidth = 640 }: {
  src: string; width: number; height: number;
  view: Box;                       // 원본에서 지금 보여주는 범위 (확대 전이면 전체)
  pending: Box | null; saved: Box[];
  onChange: (box: Box | null) => void;
  displayWidth?: number;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const displayHeight = Math.max(1, Math.round(displayWidth * view.h / Math.max(view.w, 1)));

  // 화면 px → 원본 px
  const toImage = (cx: number, cy: number) => {
    const rect = wrap.current!.getBoundingClientRect();
    const fx = Math.min(Math.max((cx - rect.left) / rect.width, 0), 1);
    const fy = Math.min(Math.max((cy - rect.top) / rect.height, 0), 1);
    return { x: view.x + fx * view.w, y: view.y + fy * view.h };
  };
  // 원본 px → 화면 px (CSS)
  const toScreen = (b: Box) => ({
    left: `${((b.x - view.x) / view.w) * 100}%`,
    top: `${((b.y - view.y) / view.h) * 100}%`,
    width: `${(b.w / view.w) * 100}%`,
    height: `${(b.h / view.h) * 100}%`,
  });

  const onDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const p = toImage(e.clientX, e.clientY);
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y });
  };
  const onMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const p = toImage(e.clientX, e.clientY);
    setDrag({ ...drag, x1: p.x, y1: p.y });
  };
  const onUp = () => {
    if (!drag) return;
    const x0 = Math.max(0, Math.min(drag.x0, drag.x1)), x1 = Math.min(width, Math.max(drag.x0, drag.x1));
    const y0 = Math.max(0, Math.min(drag.y0, drag.y1)), y1 = Math.min(height, Math.max(drag.y0, drag.y1));
    const w = Math.round(x1 - x0), h = Math.round(y1 - y0);
    setDrag(null);
    // 클릭(끌지 않음)은 박스가 아니다 — 지정을 지우는 뜻으로 본다
    onChange(w >= 1 && h >= 1 ? { x: Math.round(x0), y: Math.round(y0), w, h } : null);
  };

  const live: Box | null = drag
    ? { x: Math.min(drag.x0, drag.x1), y: Math.min(drag.y0, drag.y1), w: Math.abs(drag.x1 - drag.x0), h: Math.abs(drag.y1 - drag.y0) }
    : pending;

  // 보는 범위만 잘라 보이도록 이미지를 CSS로 확대·이동한다
  const scale = width / view.w;
  return (
    <div ref={wrap} className="picker" style={{ width: displayWidth, height: displayHeight, overflow: "hidden", cursor: "crosshair" }}
      onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={() => setDrag(null)}>
      <img src={src} alt="" draggable={false} style={{
        position: "absolute", left: 0, top: 0,
        width: displayWidth * scale, height: displayHeight * (height / view.h), maxWidth: "none",
        transform: `translate(${-(view.x / view.w) * displayWidth}px, ${-(view.y / view.h) * displayHeight}px)`,
      }} />
      {saved.map((b, i) => (
        <div key={i} className="box saved" style={toScreen(b)}><span className="tag">{i + 1}. {b.label ?? ""}</span></div>
      ))}
      {live && live.w > 0 && live.h > 0 ? <div className="box" style={toScreen(live)} /> : null}
    </div>
  );
}

export const ZOOM_MIN_PX = 48;
/** 더 이상 좁힐 수 없는 확대 범위(원본 픽셀). 이보다 좁히면 손 떨림이 곧 좌표 오차가 된다. */
export const ZOOM_MARGIN = 0.6;
/** 확대할 때 지정한 영역 둘레에 남기는 여유. 딱 맞게 자르면 결함 가장자리가 화면 끝에 붙어 경계가 안 보인다. */

/** 지정한 영역 둘레로 확대한 범위. 화면 비율은 원본과 같게 유지한다 (예전 ui.zoom_to). */
export function zoomTo(box: Box, width: number, height: number, margin = ZOOM_MARGIN): Box {
  const padX = box.w * margin, padY = box.h * margin;
  let x0 = box.x - padX, y0 = box.y - padY, x1 = box.x + box.w + padX, y1 = box.y + box.h + padY;
  let vw = x1 - x0, vh = y1 - y0;
  if (vw / vh < width / height) vw = vh * width / height; else vh = vw * height / width;
  vw = Math.max(ZOOM_MIN_PX, Math.min(vw, width));
  vh = Math.max(ZOOM_MIN_PX * height / Math.max(width, 1), Math.min(vh, height));
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const left = Math.max(0, Math.min(Math.round(cx - vw / 2), width - Math.round(vw)));
  const top = Math.max(0, Math.min(Math.round(cy - vh / 2), height - Math.round(vh)));
  return { x: left, y: top, w: Math.round(vw), h: Math.round(vh) };
}

/** «화면 1px이 원본 몇 px인가» — 곧 그릴 수 있는 가장 작은 눈금. */
export function zoomNote(view: Box, displayWidth = 640): string {
  const perPixel = view.w / Math.max(displayWidth, 1);
  const magnification = displayWidth / Math.max(view.w, 1);
  return `${magnification.toFixed(1)}× · 화면 1px = 원본 ${perPixel.toFixed(2)}px`;
}
