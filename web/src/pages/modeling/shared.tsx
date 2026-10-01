/**
 * 3단계 탭들이 함께 쓰는 조각. Streamlit 화면의 `_metric_row` · `_category_thresholds` ·
 * `_advice_banner` · `_empty_state_links` · `_synthetic_warning` 자리다.
 */
import { Link } from "react-router-dom";
import type { Table } from "../../api";
import { Alert, Caption, Cols, DataTable, Divider, Metric, md } from "../../components/ui";

export interface MetricItem { key: string; label: string; value: string; help: string; caption: string }

export interface CategoryBlock {
  enabled: boolean;
  metrics: { label: string; value: string; delta: string | null; inverse: boolean; help: string | null }[];
  single_caption: string;
  table: Table;
  skipped: { name: string; reason: string }[];
}

export interface Advice {
  blocked: boolean;
  error?: string;
  info?: string;
  banner?: { kind: "success" | "info"; icon: string; text: string };
  notes: string[];
}

export interface EmptyLink { route: string; label: string }

/** 학습 잡의 JSON 결과. 베이스라인·이상탐지가 같은 모양을 돌려준다. */
export interface TrainSummary {
  kind: "baseline" | "anomaly";
  run_id: string;
  metrics: Record<string, number | null>;
  metric_row: MetricItem[];
  importance?: { name: string; count: number }[];
  categories: CategoryBlock | null;
  warnings: string[];
  dataset_note?: string;
  cache_note?: string | null;
}

/**
 * 지표 카드. 핵심 지표는 캡션으로 **바로 보이게** 하고, 부가 설명은 `?` 아이콘에 둔다.
 * 눌러야 보이는 설명은 초보자가 그냥 지나치기 때문이다. 문구는 서버(glossary)가 내려준다.
 */
export function MetricRow({ items }: { items: MetricItem[] }) {
  return (
    <Cols n={5}>
      {items.map((m) => <Metric key={m.key} label={m.label} value={m.value} help={m.help} caption={m.caption || undefined} />)}
    </Cols>
  );
}

/**
 * 카테고리마다 임계값을 따로 골랐을 때의 효과. **꺼져 있어도 보여준다** — 켜면 무엇이
 * 달라지는지 모르는 채로 고르게 할 수는 없다. 값이 그대로면 화살표를 띄우지 않는다
 * (서버 `_delta`): 빨간 `↑ +0건`은 나빠진 것처럼 읽힌다.
 */
export function CategoryThresholds({ block }: { block: CategoryBlock | null }) {
  if (!block) return null;
  return (
    <>
      <Divider />
      <div><strong>카테고리별 임계값</strong></div>
      <Caption>카테고리는 서로 다른 제품이라 정상 분포부터 다릅니다. 임계값 하나를 전체에 쓰면 **가장 어려운 카테고리가 전체를 끌어내립니다** — 그것을 잡으려 낮추면 나머지의 오탐이 늘고, 안 낮추면 그 카테고리에서 결함을 놓칩니다. 현장에서도 라인마다 기준을 따로 둡니다.</Caption>
      <Cols n={3}>
        {block.metrics.map((m) => <Metric key={m.label} label={m.label} value={m.value} delta={m.delta} deltaInverse={m.inverse} help={m.help} />)}
      </Cols>
      <Caption>{block.single_caption}</Caption>
      <DataTable table={block.table} />
      {block.skipped.map((s) => <Caption key={s.name}>{`\`${s.name}\` — ${s.reason}`}</Caption>)}
      {!block.enabled ? (
        <Caption>지금은 **임계값 하나**를 쓰고 있습니다. 위의 «카테고리별 임계값»을 켜고 다시 학습하면 이 표의 값이 적용되어 모델과 함께 등록됩니다.</Caption>
      ) : null}
    </>
  );
}

/**
 * 이 탭의 방식이 지금 데이터로 가능한지, 권장되는지 미리 알려준다.
 * 선택지만 주고 무엇을 골라야 할지 알려주지 않으면 초보자는 막힌다. 특히 VisA 공식 분할은
 * 학습 분할에 결함이 없어 지도학습이 시작조차 안 되는데, 화면은 선택지를 똑같이 보여준다.
 */
export function AdviceBanner({ advice }: { advice: Advice }) {
  if (advice.blocked) {
    return (
      <>
        <Alert kind="error" icon="🚧">{md(advice.error ?? "")}</Alert>
        <Alert kind="info" icon="👉">{advice.info ?? ""}</Alert>
      </>
    );
  }
  return (
    <>
      {advice.banner ? <Alert kind={advice.banner.kind} icon={advice.banner.icon}>{advice.banner.text}</Alert> : null}
      {advice.notes.map((n) => <Alert key={n} kind="warning">{n}</Alert>)}
    </>
  );
}

/** 막혔을 때 어디로 가야 하는지 링크로 알려준다. 문구만 띄우면 초보자는 길을 잃는다. */
export function EmptyStateLink({ link }: { link: EmptyLink }) {
  return <p><Link to={link.route}>➡️ {link.label}</Link></p>;
}

export function NoImages({ link }: { link: EmptyLink }) {
  return (
    <>
      <Alert kind="info" icon="📥">수집된 이미지가 없습니다.</Alert>
      <EmptyStateLink link={link} />
    </>
  );
}

export function SyntheticWarning({ show }: { show: boolean }) {
  if (!show) return null;
  return <Alert kind="warning" icon="🧪">지금 데이터는 **합성 샘플뿐**입니다. 결함이 인위적으로 뚜렷해 지표가 실제보다 높게 나옵니다. 파이프라인 배선 확인용으로만 보고, 성능 근거로 삼지 마세요.</Alert>;
}

/** 학습 잡이 끝난 뒤 공통으로 보여주는 것: 경고 → 완료 문구 → 지표 → 카테고리 표. */
export function TrainOutcome({ summary, children }: { summary: TrainSummary; children?: React.ReactNode }) {
  return (
    <>
      {summary.dataset_note ? <Caption>{summary.dataset_note}</Caption> : null}
      {summary.cache_note ? <Caption>{summary.cache_note}</Caption> : null}
      {summary.warnings.map((w) => <Alert key={w} kind="warning">{w}</Alert>)}
      <CategoryThresholds block={summary.categories} />
      <Alert kind="success">{`학습·평가 완료 — 실험 기록 \`${summary.run_id}\``}</Alert>
      <MetricRow items={summary.metric_row} />
      {children}
    </>
  );
}
