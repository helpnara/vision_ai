import { useEffect, useState } from "react";
import { get, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { Gallery } from "../../components/Gallery";
import { VegaChart } from "../../components/VegaChart";
import { Alert, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, NumberInput, Select, Spinner, fmt, md } from "../../components/ui";

interface Metrics { n: number; tp: number; fp: number; fn: number; tn: number; recall: number | null; precision: number | null; auroc?: number | null }
interface Payload {
  log_empty: boolean; feedback_empty: boolean; metrics?: Metrics;
  gap?: { registered: number; actual: number; gap: number } | null;
  misses?: { imageId: string; caption: string }[]; videos?: string[];
  segment_defaults?: { fps: number; min_hits: number };
  by_version?: Table; by_version_help?: Record<string, string>; trend?: object | null; detail?: Table;
}
interface Segments {
  available: boolean; metrics?: { segment_recall: string; frame_recall: string; missed: number; false_alarms_per_min: string };
  summary?: string; contrast?: string; timeline?: object | null; missed_text?: string; false_text?: string;
}

export function PerformanceTab() {
  const { data, error } = useFetch<Payload>("/api/operations/performance");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const intro = md("사후 검수로 확인된 정답과 모델 판정을 비교한다. 정답의 출처는 2단계에서 **사람이 확인한** 라벨뿐이다 — 폴더 구조에서 추론한 라벨을 정답으로 쓰면 자기 채점이 된다.");
  if (data.log_empty) return <>{intro}<Alert kind="info" icon="▶️">추론 로그가 없습니다. **배치 추론**을 먼저 실행하세요.</Alert></>;
  if (data.feedback_empty || !data.metrics) {
    return <>{intro}<Alert kind="warning" icon="🏷️">사후 검수 정답이 없습니다. **2. 라벨링 → 라벨 검수**에서 사람이 확인한 라벨이 쌓여야 실제 성능을 측정할 수 있습니다.</Alert></>;
  }
  const m = data.metrics;
  const gap = data.gap;
  return (
    <>
      {intro}
      <Cols n={5}>
        <Metric label="검수 대조 건수" value={fmt.int(m.n)} />
        <Metric label="재현율" value={fmt.num(m.recall, 3)} />
        <Metric label="정밀도" value={fmt.num(m.precision, 3)} />
        <Metric label="미탐" value={fmt.int(m.fn)} />
        <Metric label="오탐" value={fmt.int(m.fp)} />
      </Cols>
      {gap ? (
        gap.gap > 0.05 ? (
          <Alert kind="error" icon="🚨">{`등록 시 재현율 ${gap.registered.toFixed(3)} → 실측 ${gap.actual.toFixed(3)} (낙폭 ${gap.gap.toFixed(3)}). 임계값 재조정 또는 재학습이 필요합니다.`}</Alert>
        ) : (
          <Alert kind="success">{`등록 시 ${gap.registered.toFixed(3)} → 실측 ${gap.actual.toFixed(3)}. 유지되고 있습니다.`}</Alert>
        )
      ) : null}
      {m.fn ? (
        <>
          <Alert kind="error" icon="🚨">{`미탐 ${m.fn}건 — 결함을 놓친 사례입니다. 이 프로젝트에서 가장 큰 리스크이므로 개별 리뷰가 필요합니다.`}</Alert>
          <Gallery items={(data.misses ?? []).map((x) => ({ imageId: x.imageId, caption: x.caption }))} />
        </>
      ) : null}

      <Divider />
      <SegmentSection videos={data.videos ?? []} defaults={data.segment_defaults ?? { fps: 30, min_hits: 1 }} />

      <Divider />
      {md("**버전별 실측 성능**")}
      {data.by_version ? <DataTable table={data.by_version} help={data.by_version_help} /> : null}

      {md("**기간별 추이**")}
      {data.trend ? <VegaChart spec={data.trend} /> : <Caption>추이를 그리려면 서로 다른 날짜의 추론 로그가 필요합니다.</Caption>}

      <Expander title="검수 대조 상세">
        <DataTable table={data.detail} scroll />
      </Expander>
    </>
  );
}

/**
 * 구간 단위 검출률 (V3).
 *
 * 프레임 단위 숫자는 결함이 여러 장에 걸칠 때 **실제보다 나쁘게** 나온다. 100프레임짜리
 * 결함에서 3장만 잡아도 알람은 울렸는데 재현율은 3%로 찍힌다. 두 숫자를 나란히 놓고
 * 서로 다르면 그 사실을 말해 준다.
 */
function SegmentSection({ videos, defaults }: { videos: string[]; defaults: { fps: number; min_hits: number } }) {
  const [picked, setPicked] = useState<string | null>(videos[0] ?? null);
  const [fps, setFps] = useState(defaults.fps);
  const [minHits, setMinHits] = useState(defaults.min_hits);
  const [report, setReport] = useState<Segments | null>(null);
  useEffect(() => { if (picked === null && videos.length) setPicked(videos[0]); }, [videos, picked]);
  useEffect(() => {
    if (!picked) return;
    get<Segments>(`/api/operations/performance/segments?video=${encodeURIComponent(picked)}&fps=${fps}&min_hits=${minHits}`).then(setReport).catch(() => setReport(null));
  }, [picked, fps, minHits]);

  const head = (
    <>
      <h3>영상·구간 단위 검출률</h3>
      <Caption>현장이 묻는 것은 «이 영상에 있던 결함을 잡았느냐»이지 «프레임 몇 %를 맞혔느냐»가 아닙니다. 구간 하나를 한 건으로 세고, 오경보는 **분당 횟수**로 냅니다.</Caption>
    </>
  );
  if (!videos.length) return <>{head}<Alert kind="info" icon="🎞️">영상 프레임의 판정 로그가 없습니다. **배치 추론 → 대상: 영상 하나**를 먼저 실행하세요.</Alert></>;
  return (
    <>
      {head}
      <Cols n={3}>
        <Select label="영상" value={picked} options={videos} onChange={setPicked} />
        <NumberInput label="영상 fps" value={fps} min={1} max={240} step={1} onChange={(v) => setFps(Math.max(1, Math.min(240, v)))} help="구간의 «몇 초»를 계산할 때만 씁니다. 몰라도 구간 수와 검출률은 같습니다." />
        <NumberInput label="잡았다고 볼 최소 프레임" value={minHits} min={1} max={20} step={1} onChange={(v) => setMinHits(Math.max(1, Math.min(20, Math.round(v))))} help="기본은 1장입니다 — 알람이 목적이면 한 번 울리는 것으로 충분합니다. 사람이 확인하러 가는 비용이 크면 올려서 «한 장은 튄 것일 수 있다»를 반영하세요." />
      </Cols>
      {!report ? <Spinner /> : !report.available || !report.metrics ? <Caption>이 영상은 정답과 대조할 프레임이 부족합니다.</Caption> : (
        <>
          <Cols n={4}>
            <Metric label="구간 재현율" value={report.metrics.segment_recall} />
            <Metric label="프레임 재현율" value={report.metrics.frame_recall} />
            <Metric label="놓친 구간" value={`${report.metrics.missed}곳`} />
            <Metric label="헛알람" value={`분당 ${report.metrics.false_alarms_per_min}회`} />
          </Cols>
          {md(report.summary ?? "")}
          {report.contrast ? <Alert kind="warning" icon="📐">{report.contrast}</Alert> : null}
          {/* V6 — 미탐이 몇 건인지보다 **어디서** 놓쳤는지가 다음에 무엇을 라벨링할지 정해 준다.
              명세는 서버(charts.segment_timeline)가 만든다 — 높이·줄 순서가 실측으로 잠겨 있다. */}
          {md("**정답 구간과 모델 알람**")}
          {report.timeline ? <VegaChart spec={report.timeline} /> : <Caption>그릴 구간이 없습니다.</Caption>}
          <Caption>위가 정답, 아래가 모델이 알람을 켠 시간입니다. **빨간 띠가 다음에 볼 곳**이고, 주황 띠가 길게 이어지면 «다 결함이라고 하는 중»입니다.</Caption>
          {report.missed_text ? <Alert kind="error" icon="🚨">{`놓친 구간 — ${report.missed_text}`}</Alert> : null}
          {report.false_text ? <Caption>{`헛알람 구간 — ${report.false_text}`}</Caption> : null}
        </>
      )}
    </>
  );
}
