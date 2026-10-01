import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { downloadText, get, post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, DataTable, Divider, ErrorBox, Expander, Metric, NumberInput, Select, Spinner, fmt, md } from "../../components/ui";
import { NoticeBox, type Notice } from "./shared";

interface Last { version: string; n: number; defect: number; normal: number; latency_mean: number | null; head: Table; failed: number }
interface Payload {
  empty: boolean; usable: string[]; default_version: string | null; scopes: string[]; categories: string[];
  videos: { name: string; count: number }[]; min_drift_samples: number; threshold_default: number;
  last: Last | null; log: { summary: { total: number; versions: number; defect_rate: number; latency_p95: number | null }; table: Table | null };
}
interface Target { count: number; notes: string[]; blocked: string | null; threshold_default: number }

const SCOPE_CATEGORY = "특정 카테고리";
const SCOPE_VIDEO = "영상 하나";

export function InferenceTab() {
  const [version, setVersion] = useState<string | null>(null);
  const { data, error, reload } = useFetch<Payload>(`/api/operations/inference${version ? `?version=${encodeURIComponent(version)}` : ""}`, [version]);
  const [scope, setScope] = useState<string>("전체");
  const [limit, setLimit] = useState(200);
  const [category, setCategory] = useState<string | null>(null);
  const [videoName, setVideoName] = useState<string | null>(null);
  const [override, setOverride] = useState(false);
  const [threshold, setThreshold] = useState<number | null>(null);
  const [target, setTarget] = useState<Target | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const job = useJob<{ last: Last; failed: number; warning: string | null }>((r) => {
    setNotice(r.warning ? { kind: "warning", text: r.warning } : null);
    void reload();
  });

  useEffect(() => {
    if (!data) return;
    if (version === null && data.default_version) setVersion(data.default_version);
    if (category === null && data.categories.length) setCategory(data.categories[0]);
    if (videoName === null && data.videos.length) setVideoName(data.videos[0].name);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  // 대상 건수는 실행 전에 보여야 한다 — 입력이 바뀔 때마다 서버에 센다.
  useEffect(() => {
    if (!version) return;
    const params = new URLSearchParams({ version, scope, limit: String(limit) });
    if (category) params.set("category", category);
    if (videoName) params.set("video", videoName);
    get<Target>(`/api/operations/inference/target?${params}`).then(setTarget).catch(() => setTarget(null));
  }, [version, scope, limit, category, videoName, data]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const intro = md("등록된 버전으로 이미지를 판정하고 로그를 남긴다. **로그가 없으면 성능 저하를 감지할 근거 자체가 없다** — 운영 감시의 출발점이다.");
  if (data.empty) {
    return <>{intro}<Alert kind="info" icon="📥">수집된 이미지가 없습니다.</Alert><Link to="/ingest">➡️ 1단계 데이터 수집으로 이동</Link></>;
  }
  if (!data.usable.length) {
    return <>{intro}<Alert kind="warning" icon="📚">추론에 쓸 수 있는 버전이 없습니다. 모델 파일이 함께 등록된 버전이 필요합니다.</Alert></>;
  }

  const thresholdDefault = target?.threshold_default ?? data.threshold_default;
  const used = override && threshold !== null ? threshold : thresholdDefault;
  const run = () => void job.start(() => post("/api/operations/inference/run", {
    version, scope, limit, category: scope === SCOPE_CATEGORY ? category : null,
    video: scope === SCOPE_VIDEO ? videoName : null, threshold: override ? used : null,
  }));
  const clearLog = async () => {
    await post("/api/operations/inference/clear_log");
    setNotice({ kind: "success", text: "누적 로그를 지웠습니다." });
    void reload();
  };
  const downloadLog = async () => {
    const { csv } = await get<{ csv: string }>("/api/operations/inference/log/csv");
    downloadText("inference_log.csv", csv);
  };

  const last = data.last;
  const summary = data.log.summary;
  return (
    <>
      {intro}
      <Cols n={3}>
        <Select label="버전" value={version} options={data.usable} onChange={setVersion} />
        <Select label="대상" value={scope} options={data.scopes} onChange={setScope} />
        <NumberInput label="최대 건수" value={limit} min={1} max={5000} step={10} onChange={(v) => setLimit(Math.max(1, Math.min(5000, Math.round(v))))} />
      </Cols>
      {scope === SCOPE_CATEGORY ? <Select label="카테고리" value={category} options={data.categories} onChange={setCategory} /> : null}
      {scope === SCOPE_VIDEO ? (
        target?.blocked ? (
          <>
            <Alert kind="info" icon="🎞️">{target.blocked}</Alert>
            <Link to="/ingest">➡️ 1단계 데이터 수집으로 이동</Link>
          </>
        ) : (
          <Select label="영상" value={videoName} options={data.videos.map((v) => v.name)} labels={(n) => { const v = data.videos.find((x) => x.name === n); return v ? `${n} (${fmt.int(v.count)}장)` : n; }} onChange={setVideoName} />
        )
      ) : null}
      {target?.notes.map((n) => <Caption key={n}>{n}</Caption>)}

      <Checkbox label="임계값 직접 지정" checked={override} onChange={setOverride} />
      <NumberInput label="임계값" value={override && threshold !== null ? threshold : thresholdDefault} step={0.01} disabled={!override} onChange={setThreshold} />

      {target && !target.blocked ? (
        <>
          <Caption>{`대상 ${fmt.int(target.count)}건 · 임계값 ${used.toFixed(4)}`}</Caption>
          {target.count < data.min_drift_samples ? (
            <Alert kind="info" icon="📏">{`드리프트 판정에는 최소 ${data.min_drift_samples}건이 필요합니다. 표본이 적으면 같은 분포에서도 지표가 커져 오경보가 됩니다.`}</Alert>
          ) : null}
          {target.count === 0 ? <Alert kind="warning">대상 이미지가 없습니다.</Alert> : (
            <Button primary busy={job.running} onClick={run}>▶️ 배치 추론 실행</Button>
          )}
        </>
      ) : null}
      <JobProgress job={job.job} error={job.error} />
      <NoticeBox notice={notice} />

      {last ? (
        <>
          <Alert kind="success">{`최근 실행: ${fmt.int(last.n)}건`}</Alert>
          <Cols n={3}>
            <Metric label="결함 판정" value={fmt.int(last.defect)} />
            <Metric label="정상 판정" value={fmt.int(last.normal)} />
            <Metric label="평균 처리시간" value={`${fmt.num(last.latency_mean, 1)} ms`} />
          </Cols>
          <DataTable table={last.head} scroll />
        </>
      ) : null}

      <Divider />
      <h3>누적 추론 로그</h3>
      <Cols n={4}>
        <Metric label="누적 건수" value={fmt.int(summary.total)} />
        <Metric label="버전 수" value={fmt.int(summary.versions)} />
        <Metric label="결함 판정 비율" value={fmt.pct(summary.defect_rate, 1)} />
        <Metric label="처리시간 p95" value={`${fmt.num(summary.latency_p95, 1)} ms`} />
      </Cols>
      {data.log.table ? (
        <>
          <DataTable table={data.log.table} scroll />
          <Button onClick={() => void downloadLog()}>추론 로그 내려받기</Button>
          <Expander title="⚠️ 로그 초기화">
            <Caption>누적 로그를 모두 지웁니다. 드리프트·성능 추이 근거가 사라집니다.</Caption>
            <Button onClick={() => void clearLog()}>로그 삭제</Button>
          </Expander>
        </>
      ) : null}
    </>
  );
}
