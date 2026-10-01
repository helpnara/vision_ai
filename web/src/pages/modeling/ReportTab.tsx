import { useEffect, useState } from "react";
import { downloadText, post, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { Gallery } from "../../components/Gallery";
import { VegaChart } from "../../components/VegaChart";
import { Alert, Button, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, NumberInput, Radio, Slider, Spinner, fmt, md } from "../../components/ui";
import { MetricRow, SyntheticWarning, type MetricItem } from "./shared";

interface ReportPayload {
  has_result: boolean;
  header?: string;
  synthetic?: boolean;
  slider?: { min: number; max: number; step: number; value: number };
  defaults?: { prevalence: number; volume: number };
  help?: { prevalence: string; volume: string };
  target_recall?: number;
  outcomes?: { key: string; label: string }[];
  kind?: string;
  split?: string;
}

interface Impact {
  prevalence: number; volume: number; reduction_ratio: number; precision: number; defects: number;
  caught: number; missed: number; reviewed: number; false_alarms: number;
}

interface Evaluated {
  metrics: Record<string, number | null>;
  metric_row: MetricItem[];
  impact: Impact;
  impact_captions: Record<"reduction" | "missed", { help: string; caption: string }>;
  verdict: { level: "good" | "usable" | "weak"; headline: string; actions: string[] };
  confusion: Table;
  sweep: object | null;
  roc: object | null;
  samples: { image_id: string; caption?: string; missing?: string }[];
  subset: Table;
  subset_count: number;
  impact_by_prevalence: Table;
}

interface Files { report: { filename: string; text: string }; errors: { filename: string; text: string } }

const VERDICT: Record<Evaluated["verdict"]["level"], { kind: "success" | "info" | "warning"; icon: string }> = {
  good: { kind: "success", icon: "✅" }, usable: { kind: "info", icon: "💡" }, weak: { kind: "warning", icon: "⚠️" },
};

export function ReportTab() {
  const { data, error } = useFetch<ReportPayload>("/api/modeling/report");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  if (!data.has_result) {
    return <Alert kind="info" icon="🧠">먼저 **베이스라인** 또는 **이상탐지** 탭에서 모델을 실행하세요. 어느 쪽을 골라야 할지는 각 탭이 데이터를 보고 알려줍니다.</Alert>;
  }
  return <ReportBody data={data} />;
}

function ReportBody({ data }: { data: ReportPayload }) {
  const slider = data.slider!;
  const [threshold, setThreshold] = useState(slider.value);
  const [prevalence, setPrevalence] = useState(data.defaults!.prevalence);
  const [volume, setVolume] = useState(data.defaults!.volume);
  const [outcome, setOutcome] = useState("FN");
  const [evaluated, setEvaluated] = useState<Evaluated | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  // 슬라이더를 움직이면 지표·효과·샘플이 따라 움직여야 판단할 수 있다 (300ms 디바운스).
  useEffect(() => {
    const t = window.setTimeout(() => {
      post<Evaluated>("/api/modeling/report/evaluate", { threshold, prevalence, volume, outcome })
        .then((got) => { setEvaluated(got); setEvalError(null); })
        .catch((e) => setEvalError((e as Error).message));
    }, 300);
    return () => window.clearTimeout(t);
  }, [threshold, prevalence, volume, outcome]);

  const files = async () => post<Files>("/api/modeling/report/files", { threshold, prevalence, volume });
  const e = evaluated;
  const outcomeLabel = (k: string) => data.outcomes!.find((o) => o.key === k)?.label ?? k;

  return (
    <>
      {md(data.header!)}
      <SyntheticWarning show={!!data.synthetic} />

      <Divider />
      <h3>임계값 조정</h3>
      <Caption>임계값은 **어느 점수부터 결함으로 볼지** 정하는 값입니다. 낮추면 결함을 더 많이 잡지만(재현율 ↑) 정상품도 함께 걸립니다(오탐 ↑). 한쪽만 좋게 만들 수는 없습니다 — 어디서 타협할지를 고르는 것입니다.</Caption>
      <Slider label="판정 임계값" value={threshold} min={slider.min} max={slider.max} step={slider.step} format={(v) => v.toFixed(4)} onChange={setThreshold} />
      <ErrorBox error={evalError} />
      {!e ? <Spinner text="계산 중..." /> : (
        <>
          <MetricRow items={e.metric_row} />

          {/*
            성능 지표를 "사람이 볼 물량이 얼마나 줄어드는가"로 바꿔 보여준다. AUROC나 정밀도만
            보면 이 도구를 도입할지 판단할 수 없다. 게다가 여기 평가 데이터는 정상:결함 비율이
            실제 라인과 달라서, 화면의 정밀도를 그대로 믿으면 크게 과대평가한다.
          */}
          <Divider />
          <h3>이 모델을 쓰면 무엇이 좋아지는가</h3>
          <div className="row">
            <NumberInput label="가정 불량률 (%)" value={Math.round(prevalence * 10000) / 100} min={0.1} max={50} step={0.1} help={data.help!.prevalence} onChange={(v) => setPrevalence(Math.min(50, Math.max(0.1, v)) / 100)} />
            <NumberInput label="검사 물량 (장)" value={volume} min={100} max={1000000} step={100} help={data.help!.volume} onChange={(v) => setVolume(Math.round(Math.min(1000000, Math.max(100, v))))} />
            <div style={{ flex: "2 1 0" }}><Caption>기본값은 **설정** 화면에서 바꿀 수 있습니다.</Caption></div>
          </div>
          <Cols n={3}>
            <Metric label="검수량 절감" value={fmt.pct(e.impact.reduction_ratio)} help={e.impact_captions.reduction.help} caption={e.impact_captions.reduction.caption} />
            <Metric label="놓치는 결함" value={`${fmt.int(e.impact.missed)}건`} help={e.impact_captions.missed.help} caption={e.impact_captions.missed.caption} />
            <Metric label="현장 기대 정밀도" value={fmt.pct(e.impact.precision, 1)} help="위 표의 정밀도는 평가 데이터 구성 기준이라 현장과 다릅니다." caption="위에서 본 정밀도를 실제 불량률로 환산한 값입니다." />
          </Cols>
          <Alert kind="info" icon="💡">{`**${e.impact.volume.toLocaleString()}장을 검사하면** — 결함 ${fmt.int(e.impact.defects)}건 중 **${fmt.int(e.impact.caught)}건을 잡고 ${fmt.int(e.impact.missed)}건을 놓친다.** 사람이 볼 물량은 ${e.impact.volume.toLocaleString()}장에서 **${fmt.int(e.impact.reviewed)}장으로 줄어든다** (그중 오탐 ${fmt.int(e.impact.false_alarms)}건을 걸러내야 한다).`}</Alert>
          {e.impact.precision < 0.5 ? (
            <Alert kind="warning">{`불량률 ${fmt.pct(e.impact.prevalence, 1)}에서 기대 정밀도가 **${fmt.pct(e.impact.precision, 1)}** 다. 즉 **자동 판정용으로는 쓸 수 없고**, 사람 검수 부하를 줄이는 **1차 스크리닝용**으로 봐야 한다. 모델이 올린 것은 사람이 다시 확인해야 한다.`}</Alert>
          ) : null}
          <Expander title="불량률이 달라지면 (정밀도가 왜 이렇게 떨어지는가)">
            <Caption>재현율과 오탐률은 불량률과 무관한 모델 고유 특성이다. 반면 정밀도는 불량률에 따라 크게 달라진다. 결함이 드물수록 정상품이 압도적으로 많아져, 같은 오탐률이라도 오탐 건수가 진짜 결함 건수를 쉽게 넘어서기 때문이다.</Caption>
            <DataTable table={e.impact_by_prevalence} />
          </Expander>

          {/* 숫자를 보여주고 끝내지 않고, 쓸 만한지와 다음에 무엇을 할지 문장으로 말해준다. */}
          <Alert kind={VERDICT[e.verdict.level].kind} icon={VERDICT[e.verdict.level].icon}>{`**${e.verdict.headline}**`}</Alert>
          {e.verdict.actions.length ? (
            <>
              <div><strong>다음에 할 일</strong></div>
              {md(e.verdict.actions.map((a) => `- ${a}`).join("\n"))}
            </>
          ) : null}

          <Cols n={2}>
            <div>
              <div><strong>혼동행렬</strong></div>
              <DataTable table={e.confusion} />
              <Caption>{`미탐률 ${fmt.pct(e.metrics.miss_rate, 1)} · 오경보율 ${fmt.pct(e.metrics.false_alarm_rate, 1)}`}</Caption>
            </div>
            <div>
              <div><strong>임계값별 재현율 / 정밀도</strong></div>
              {e.sweep ? <VegaChart spec={e.sweep} /> : null}
            </div>
          </Cols>
          <Expander title="ROC 곡선">
            {e.roc ? <VegaChart spec={e.roc} /> : <Caption>한 클래스만 있어 ROC를 그릴 수 없습니다.</Caption>}
          </Expander>

          <Divider />
          <h3>오탐 · 미탐 샘플</h3>
          <Caption>숫자만으로는 무엇이 잘못됐는지 알 수 없다. 실제 이미지를 봐야 다음 조치가 정해진다.</Caption>
          <Radio label="표시할 종류" value={outcome} options={data.outcomes!.map((o) => o.key)} labels={outcomeLabel} onChange={setOutcome} />
          <Caption>{`${outcome} ${e.subset_count}건`}</Caption>
          {e.subset_count === 0 ? <Alert kind="success" icon="🎉">{`${outcome} 사례가 없습니다.`}</Alert> : (
            <>
              <Gallery items={e.samples.filter((s) => !s.missing).map((s) => ({ imageId: s.image_id, caption: s.caption }))} />
              {e.samples.filter((s) => s.missing).map((s) => <Alert key={s.image_id} kind="warning">{s.missing}</Alert>)}
              <Expander title="표로 보기"><DataTable table={e.subset} scroll /></Expander>
            </>
          )}

          <Divider />
          <h3>결과 보고서</h3>
          <Caption>성능·업무 효과·판정·주의사항을 한 장으로 묶어 내려받습니다. 화면을 캡처해 옮겨 적지 않아도 됩니다.</Caption>
          <Button primary onClick={() => void files().then((f) => downloadText(f.report.filename, f.report.text, "text/markdown"))}>📄 보고서 내려받기 (.md)</Button>

          <Divider />
          <Button onClick={() => void files().then((f) => downloadText(f.errors.filename, f.errors.text))}>판정 결과 내려받기 (errors.csv)</Button>
        </>
      )}
    </>
  );
}
