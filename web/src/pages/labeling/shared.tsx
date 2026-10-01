import { Link } from "react-router-dom";
import { Alert } from "../../components/ui";

/** `/api/labeling/overview` — 탭 표식과 공통 상수(결함 유형·라벨 한글명). */
export interface Overview {
  total: number;
  split_done: boolean;
  defect_types: { key: string; label: string }[];
  labels_ko: Record<string, string>;
  queue_modes: { key: string; label: string }[];
  draining_modes: string[];
}

export const LABEL_NORMAL = "normal";
export const LABEL_DEFECT = "defect";
export const LABEL_UNLABELED = "unlabeled";
export const DEFECT_TYPE_UNSPECIFIED = "unspecified";
export const DEFECT_TYPE_NONE = "none";

export const SPECIAL_DEFECT_LABELS: Record<string, string> = { [DEFECT_TYPE_NONE]: "해당 없음", [DEFECT_TYPE_UNSPECIFIED]: "유형 미지정" };

/** `config.defect_type_label(key)` — 표준 유형이면 한글명, 아니면 키 그대로. */
export function defectTypeLabel(overview: Overview, key: string): string {
  if (key in SPECIAL_DEFECT_LABELS) return SPECIAL_DEFECT_LABELS[key];
  return overview.defect_types.find((t) => t.key === key)?.label ?? key;
}

/** 예전 `_defect_label(key)` — «스크래치 / 긁힘 (scratch)». */
export const defectChoiceLabel = (overview: Overview) => (key: string) => `${defectTypeLabel(overview, key)} (${key})`;

export const labelKo = (overview: Overview, label: string) => overview.labels_ko[label] ?? label;

/** 수집된 이미지가 없을 때 — 1단계로 보내는 안내 (st.info + st.page_link). */
export function NoImages({ text = "수집된 이미지가 없습니다." }: { text?: string }) {
  return (
    <>
      <Alert kind="info" icon="📥">{text}</Alert>
      <Link to="/ingest">➡️ 1단계 데이터 수집으로 이동</Link>
    </>
  );
}

export type Notice = { kind: "success" | "info" | "warning" | "error"; text: string; icon?: string } | null;

/** 코드 블록 (st.code). */
export function CodeBlock({ text }: { text: string }) {
  return <pre><code>{text}</code></pre>;
}
