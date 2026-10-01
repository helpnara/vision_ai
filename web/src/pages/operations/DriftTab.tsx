import { useFetch } from "../../hooks";
import { Alert, Bars, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, Spinner, fmt, md } from "../../components/ui";
import type { Table } from "../../api";
import { ProductionBanner, type ProductionRow } from "./shared";

interface Summary { checked: number; mean_psi: number | null; shifted: number; watch: number; insufficient: number; n_current: number; level: string }
interface Payload {
  production: ProductionRow | null; runs_empty: boolean;
  baseline: { n_samples: number; features: number; bins: number } | null;
  has_batch: boolean; summary: Summary | null; causes?: string[]; table?: Table; help?: Record<string, string>;
  bars?: { name: string; psi: number }[];
  score?: { available: boolean; baseline_mean?: number; current_mean?: number; shift_sigma?: number | null };
  thresholds: { stable: number; shifted: number; min_samples: number; insufficient: string };
  glossary: { psi_caption: string; psi_detail: string; arbitrary_psi: string; arbitrary_samples: string };
}

export function DriftTab() {
  const { data, error } = useFetch<Payload>("/api/operations/drift");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const intro = md("입력 분포가 변하면 모델을 건드리지 않아도 성능이 떨어진다. **정답 라벨 없이** 감지할 수 있는 것이 장점이라, 사후 검수보다 먼저 경고를 준다.");
  const banner = <ProductionBanner production={data.production} runsEmpty={data.runs_empty} />;
  if (!data.production) return <>{intro}{banner}</>;
  if (!data.baseline) {
    return <>{intro}{banner}<Alert kind="error" icon="📉">{`${data.production.version}에 드리프트 기준선이 없습니다. 등록 시 학습 분할이 비어 있었을 수 있습니다. 다시 등록하면 기준선이 저장됩니다.`}</Alert></>;
  }
  const baselineLine = <Caption>{`기준선: ${fmt.int(data.baseline.n_samples)}개 표본 · 특징 ${data.baseline.features}개 · 구간 ${data.baseline.bins}개`}</Caption>;
  if (!data.has_batch || !data.summary) {
    return <>{intro}{banner}{baselineLine}<Alert kind="info" icon="▶️">**배치 추론** 탭에서 추론을 한 번 실행하면 그 입력으로 드리프트를 계산합니다.</Alert></>;
  }

  const s = data.summary;
  const t = data.thresholds;
  const score = data.score;
  return (
    <>
      {intro}{banner}{baselineLine}
      <Cols n={4}>
        <Metric label="검사 표본" value={fmt.int(s.n_current)} />
        <Metric label="판정" value={s.level} />
        {/* «판정: 주의» 옆에 «변화 특징 0»만 있으면 카드가 근거 없이 주의라고 말하는 것으로 읽힌다.
            주의 단계에서 판정을 끌어올린 것은 «주의 특징»이므로 둘을 같이 센다. */}
        <Metric label="변화 / 주의 특징" value={`${fmt.int(s.shifted)} / ${fmt.int(s.watch)}`} help={data.glossary.psi_detail} />
        <Metric label="평균 PSI" value={fmt.num(s.mean_psi, 3)} />
      </Cols>
      {s.level === t.insufficient ? (
        <Alert kind="warning" icon="📏">{`표본이 ${s.n_current}건으로 최소 ${t.min_samples}건에 못 미칩니다. 소표본에서는 같은 분포에서도 PSI가 커져 오경보가 되므로 판정을 보류합니다.`}</Alert>
      ) : s.level === "변화" ? (
        <Alert kind="error" icon="🚨">{`입력 분포가 유의미하게 변했습니다 (PSI > ${t.shifted}인 특징 ${s.shifted}개). 촬영 환경·공정 변경을 먼저 확인하세요.`}</Alert>
      ) : s.level === "주의" ? (
        <Alert kind="warning">{`일부 특징이 주의 구간입니다 (${s.watch}개). 추이를 지켜보세요.`}</Alert>
      ) : (
        <Alert kind="success">입력 분포가 안정적입니다.</Alert>
      )}
      <Caption>{data.glossary.psi_caption}</Caption>
      <Caption>{`PSI 해석: ${t.stable} 이하 안정 · ${t.stable}~${t.shifted} 주의 · ${t.shifted} 초과 변화`}</Caption>
      <Expander title="이 기준은 어디서 왔나">
        {md(`- ${data.glossary.arbitrary_psi}\n- ${data.glossary.arbitrary_samples}`)}
      </Expander>

      <Divider />
      {data.causes && data.causes.length ? (
        <>
          {/* 분포가 변한 특징을 현장에서 확인할 것으로 옮겨 준다. 이름만 보고 «조명이 바뀌었나?»까지
              연결하려면 각 특징이 무엇을 재는지 알아야 하는데, 그건 이 파이프라인을 만든 사람만 안다. */}
          {md("**무엇을 확인해야 하나**")}
          <Caption>분포가 변한 특징으로부터 추정한 것입니다. 모델을 다시 학습하기 전에 **현장 조건이 바뀌지 않았는지 먼저 확인**하는 편이 빠릅니다.</Caption>
          {md(data.causes.map((c) => `- ${c}`).join("\n"))}
          <Divider />
        </>
      ) : null}

      {md("**특징별 분포 이동 (PSI 상위)**")}
      <DataTable table={data.table} help={data.help} />
      {data.bars && data.bars.length ? (
        <Bars items={data.bars.map((b) => ({ name: b.name, count: Math.round(b.psi * 1000) / 1000 }))} title="PSI 상위 15" />
      ) : null}

      <Divider />
      {md("**결함 점수 분포 이동**")}
      {!score || !score.available ? <Caption>기준선에 점수 분포가 없어 비교할 수 없습니다.</Caption> : (
        <>
          <Cols n={3}>
            <Metric label="기준 평균" value={fmt.num(score.baseline_mean, 4)} />
            <Metric label="현재 평균" value={fmt.num(score.current_mean, 4)} />
            <Metric label="이동 (σ)" value={fmt.num(score.shift_sigma, 2)} />
          </Cols>
          <Caption>입력 특징은 그대로인데 점수 분포만 이동했다면, 재학습보다 **임계값 재조정**이 먼저다.</Caption>
        </>
      )}
    </>
  );
}
