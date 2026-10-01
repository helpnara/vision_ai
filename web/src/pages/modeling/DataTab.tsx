import { post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, Spinner, fmt } from "../../components/ui";
import { EmptyStateLink, SyntheticWarning, type EmptyLink } from "./shared";

interface DataPayload {
  empty: boolean;
  link: EmptyLink;
  metrics?: { labeled: number; normal: number; defect: number; categories: number };
  ready?: boolean;
  problems?: string[];
  synthetic?: boolean;
  crosstab?: Table | null;
  features?: { count: number; image_size: number; names: string[] };
  dataset?: { rows: number; cols: number } | null;
}

interface ExtractResult { rows: number; cols: number; done: string; warnings: string[] }

export function DataTab() {
  const { data, error, reload } = useFetch<DataPayload>("/api/modeling/data");
  const extract = useJob<ExtractResult>(() => void reload());

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  if (data.empty) {
    return (
      <>
        <Alert kind="info" icon="📥">수집된 이미지가 없습니다. 1단계에서 시작하세요.</Alert>
        <EmptyStateLink link={data.link} />
      </>
    );
  }
  const m = data.metrics!;
  const f = data.features!;
  const result = extract.job?.result ?? null;

  return (
    <>
      <Cols n={4}>
        <Metric label="라벨된 이미지" value={fmt.int(m.labeled)} />
        <Metric label="정상" value={fmt.int(m.normal)} />
        <Metric label="결함" value={fmt.int(m.defect)} />
        <Metric label="카테고리" value={fmt.int(m.categories)} />
      </Cols>
      {data.problems?.map((p) => <Alert key={p} kind="error" icon="🚧">{p}</Alert>)}
      {data.ready ? <Alert kind="success">학습을 시작할 수 있습니다.</Alert> : null}
      <SyntheticWarning show={!!data.synthetic} />
      {m.labeled === 0 ? null : (
        <>
          <Divider />
          <div><strong>분할 × 라벨</strong></div>
          <DataTable table={data.crosstab} />

          <Divider />
          <h3>특징 추출</h3>
          <Caption>{`이미지 1장 → ${f.count}차원 벡터 (밝기·엣지·색상·고주파·텍스처 통계). 이미지는 ${f.image_size}×${f.image_size}로 맞춘 뒤 계산합니다.`}</Caption>
          <Button primary busy={extract.running} onClick={() => void extract.start(() => post("/api/modeling/data/extract"))}>🧮 특징 추출 / 갱신</Button>
          <JobProgress job={extract.job} error={extract.error} />
          {result ? (
            <>
              <Caption>{result.done}</Caption>
              {result.warnings.map((w) => <Alert key={w} kind="warning">{w}</Alert>)}
            </>
          ) : null}
          {data.dataset ? (
            <>
              <Alert kind="success">{`특징 행렬 준비됨: ${data.dataset.rows.toLocaleString()}행 × ${data.dataset.cols}열`}</Alert>
              <Expander title="특징 목록">{f.names.join(", ")}</Expander>
            </>
          ) : (
            <Alert kind="info" icon="🧮">아직 특징을 추출하지 않았습니다.</Alert>
          )}
        </>
      )}
    </>
  );
}
