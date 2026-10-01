/**
 * 공용 화면 조각. Streamlit의 st.metric / st.success / st.tabs / st.dataframe … 에 해당한다.
 * 모든 화면이 이것만 쓰면 모양이 어긋나지 않는다.
 */
import { useId, useState, type ReactNode } from "react";
import type { Table } from "../api";

// --- 글 ---------------------------------------------------------------------

/** 아주 작은 마크다운: **굵게**, `코드`, 줄바꿈, "- " 목록, "1. " 목록. 외부 의존성 없이 충분하다. */
export function md(text: string): ReactNode {
  const lines = text.split("\n");
  const blocks: ReactNode[] = [];
  let list: { kind: "ul" | "ol"; items: ReactNode[] } | null = null;
  const flush = () => {
    if (!list) return;
    const Tag = list.kind;
    blocks.push(<Tag key={blocks.length}>{list.items.map((item, i) => <li key={i}>{item}</li>)}</Tag>);
    list = null;
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const ul = /^\s*[-*]\s+(.*)$/.exec(line);
    const ol = /^\s*\d+\.\s+(.*)$/.exec(line);
    if (ul || ol) {
      const kind = ul ? "ul" : "ol";
      if (!list || list.kind !== kind) { flush(); list = { kind, items: [] }; }
      list.items.push(inline((ul ?? ol)![1]));
      continue;
    }
    flush();
    if (!line.trim()) continue;
    blocks.push(<p key={blocks.length}>{inline(line)}</p>);
  }
  flush();
  return <div className="md">{blocks}</div>;
}

function inline(text: string): ReactNode {
  // **굵게** 와 `코드` 와 ~~취소~~ 만 다룬다.
  const parts: ReactNode[] = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|~~[^~]+~~)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(text))) {
    if (m.index > last) parts.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) parts.push(<strong key={k++}>{tok.slice(2, -2)}</strong>);
    else if (tok.startsWith("`")) parts.push(<code key={k++}>{tok.slice(1, -1)}</code>);
    else parts.push(<s key={k++}>{tok.slice(2, -2)}</s>);
    last = m.index + tok.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function Caption({ children }: { children: ReactNode }) {
  return <div className="caption">{typeof children === "string" ? inline(children) : children}</div>;
}

export function Help({ text }: { text?: string | null }) {
  if (!text) return null;
  return <span className="help" title={text}>?</span>;
}

// --- 알림 -------------------------------------------------------------------

const ICONS = { success: "✅", info: "ℹ️", warning: "⚠️", error: "🚫" };

export function Alert({ kind, icon, children }: { kind: keyof typeof ICONS; icon?: string; children: ReactNode }) {
  return (
    <div className={`alert ${kind}`} role={kind === "error" ? "alert" : undefined}>
      <span className="icon">{icon ?? ICONS[kind]}</span>
      <div>{typeof children === "string" ? inline(children) : children}</div>
    </div>
  );
}

export function ErrorBox({ error }: { error: string | null | undefined }) {
  return error ? <Alert kind="error">{error}</Alert> : null;
}

// --- 숫자 카드 ----------------------------------------------------------------

export function Metric({ label, value, help, caption, delta, deltaInverse }: {
  label: string; value: ReactNode; help?: string | null; caption?: string; delta?: string | null; deltaInverse?: boolean;
}) {
  let deltaClass = "";
  if (delta) {
    const negative = delta.trim().startsWith("-");
    deltaClass = negative !== !!deltaInverse ? "down" : "up";
  }
  return (
    <div className="metric">
      <div className="label">{label}<Help text={help} /></div>
      <div className="value">{value}</div>
      {delta ? <div className={`delta ${deltaClass}`}>{deltaClass === "up" ? "↑" : "↓"} {delta}</div> : null}
      {caption ? <div className="caption">{inline(caption)}</div> : null}
    </div>
  );
}

export function Cols({ n, children }: { n: 2 | 3 | 4 | 5; children: ReactNode }) {
  return <div className={`cols cols-${n}`}>{children}</div>;
}

export function Card({ children }: { children: ReactNode }) {
  return <div className="card">{children}</div>;
}

export function Divider() {
  return <hr />;
}

// --- 버튼 -------------------------------------------------------------------

export function Button({ primary, tertiary, small, block, busy, children, ...rest }: {
  primary?: boolean; tertiary?: boolean; small?: boolean; block?: boolean; busy?: boolean; children: ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const cls = ["btn", primary && "primary", tertiary && "tertiary", small && "small", block && "block"].filter(Boolean).join(" ");
  return (
    <button type="button" className={cls} {...rest} disabled={rest.disabled || busy}>
      {busy ? <span className="spinner" /> : null}{children}
    </button>
  );
}

// --- 탭 ---------------------------------------------------------------------

export function Tabs({ tabs, active, onChange }: { tabs: string[]; active: number; onChange: (i: number) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map((name, i) => (
        <button key={name} role="tab" aria-selected={i === active} className={i === active ? "active" : ""} onClick={() => onChange(i)}>
          {name}
        </button>
      ))}
    </div>
  );
}

/** 탭 상태를 URL 해시(#tab=2)에 두어 새로고침해도 같은 탭이 열린다. */
export function useTab(count: number, initial = 0): [number, (i: number) => void] {
  const fromHash = () => {
    const m = /tab=(\d+)/.exec(window.location.hash);
    const n = m ? Number(m[1]) : initial;
    return n >= 0 && n < count ? n : initial;
  };
  const [active, setActive] = useState(fromHash);
  const change = (i: number) => {
    setActive(i);
    window.history.replaceState(null, "", `#tab=${i}`);
  };
  return [active, change];
}

export function Expander({ title, open, children }: { title: string; open?: boolean; children: ReactNode }) {
  return (
    <details className="expander" open={open}>
      <summary>{title}</summary>
      <div className="body">{children}</div>
    </details>
  );
}

// --- 입력 -------------------------------------------------------------------

type FieldProps = { label: string; help?: string | null; disabled?: boolean };

export function TextInput({ label, help, value, onChange, placeholder, disabled }: FieldProps & {
  value: string; onChange: (v: string) => void; placeholder?: string;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}<Help text={help} /></label>
      <input id={id} type="text" value={value} placeholder={placeholder} disabled={disabled} onChange={(e) => onChange(e.target.value)} />
    </div>
  );
}

export function NumberInput({ label, help, value, onChange, min, max, step, disabled }: FieldProps & {
  value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}<Help text={help} /></label>
      <input id={id} type="number" value={Number.isFinite(value) ? value : ""} min={min} max={max} step={step} disabled={disabled}
        onChange={(e) => { const n = Number(e.target.value); if (Number.isFinite(n)) onChange(n); }} />
    </div>
  );
}

export function Slider({ label, help, value, onChange, min, max, step, format, disabled }: FieldProps & {
  value: number; onChange: (v: number) => void; min: number; max: number; step: number; format?: (v: number) => string;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}<Help text={help} /><span className="range-value">{format ? format(value) : value}</span></label>
      <input id={id} type="range" value={value} min={min} max={max} step={step} disabled={disabled} onChange={(e) => onChange(Number(e.target.value))} />
    </div>
  );
}

export function Select<T extends string>({ label, help, value, onChange, options, labels, disabled, placeholder }: FieldProps & {
  value: T | null; onChange: (v: T) => void; options: readonly T[]; labels?: (v: T) => string; placeholder?: string;
}) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}<Help text={help} /></label>
      <select id={id} value={value ?? ""} disabled={disabled} onChange={(e) => onChange(e.target.value as T)}>
        {value === null || placeholder ? <option value="" disabled>{placeholder ?? "선택하세요"}</option> : null}
        {options.map((o) => <option key={o} value={o}>{labels ? labels(o) : o}</option>)}
      </select>
    </div>
  );
}

export function Checkbox({ label, help, checked, onChange, disabled }: FieldProps & { checked: boolean; onChange: (v: boolean) => void }) {
  const id = useId();
  return (
    <div className="field inline">
      <input id={id} type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <label htmlFor={id}>{label}<Help text={help} /></label>
    </div>
  );
}

export function Radio<T extends string>({ label, help, value, onChange, options, labels, disabled }: FieldProps & {
  value: T; onChange: (v: T) => void; options: readonly T[]; labels?: (v: T) => string;
}) {
  const name = useId();
  return (
    <div className="field">
      {label ? <label>{label}<Help text={help} /></label> : null}
      <div className="radio-row">
        {options.map((o) => (
          <label key={o}>
            <input type="radio" name={name} value={o} checked={o === value} disabled={disabled} onChange={() => onChange(o)} />
            {labels ? labels(o) : o}
          </label>
        ))}
      </div>
    </div>
  );
}

// --- 표 ---------------------------------------------------------------------

function cell(value: unknown): ReactNode {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  if (typeof value === "boolean") return value ? "예" : "아니오";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** 서버의 table() 응답을 그린다. `help`는 열 이름 → 설명 (용어 사전). */
export function DataTable({ table, help, scroll, limit, columns }: {
  table: Table | null | undefined; help?: Record<string, string>; scroll?: boolean; limit?: number; columns?: string[];
}) {
  if (!table || table.rows.length === 0) return <Caption>표시할 행이 없습니다.</Caption>;
  const cols = columns ?? table.columns;
  const rows = limit ? table.rows.slice(0, limit) : table.rows;
  return (
    <div className={`table-wrap${scroll ? " scroll" : ""}`}>
      <table className="data">
        <thead>
          <tr>{cols.map((c) => <th key={c} title={help?.[c]}>{c}{help?.[c] ? " ⓘ" : ""}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{cols.map((c) => <td key={c} className={typeof row[c] === "number" ? "num" : ""}>{cell(row[c])}</td>)}</tr>
          ))}
        </tbody>
      </table>
      {table.total > rows.length ? <div className="table-foot">{table.total.toLocaleString()}행 중 {rows.length.toLocaleString()}행 표시</div> : null}
    </div>
  );
}

/** 가로 막대 (value_counts 그림). vega 없이 그린다 — 이 정도에 차트 라이브러리는 과하다. */
export function Bars({ items, title }: { items: { name: string; count: number }[]; title?: string }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div>
      {title ? <div><strong>{title}</strong></div> : null}
      {items.length === 0 ? <Caption>데이터가 없습니다.</Caption> : (
        <div className="bars">
          {items.map((it) => (
            <>
              <span className="name" key={`${it.name}-n`} title={it.name}>{it.name}</span>
              <div className="track" key={`${it.name}-t`}><div style={{ width: `${(it.count / max) * 100}%` }} /></div>
              <span className="count" key={`${it.name}-c`}>{it.count.toLocaleString()}</span>
            </>
          ))}
        </div>
      )}
    </div>
  );
}

export function Json({ value }: { value: unknown }) {
  return <pre className="json">{JSON.stringify(value, null, 2)}</pre>;
}

export function Spinner({ text }: { text?: string }) {
  return <div className="caption"><span className="spinner" />{text ?? "불러오는 중..."}</div>;
}

// --- 숫자 표기 ----------------------------------------------------------------

export const fmt = {
  int: (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Math.round(v).toLocaleString()),
  num: (v: number | null | undefined, digits = 3) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits)),
  pct: (v: number | null | undefined, digits = 0) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(digits)}%`),
};
