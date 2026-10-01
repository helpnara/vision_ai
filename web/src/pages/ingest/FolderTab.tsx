import { useEffect, useState } from "react";
import { post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, DataTable, ErrorBox, Expander, NumberInput, Select, Spinner, TextInput, md } from "../../components/ui";
import type { IngestOptions } from "./types";
import { useDebounced } from "./util";

interface CatalogLite { datasets: { key: string; name: string; layout: string; layout_note: string }[] }
interface Preview { root: string; total: number; preview: Table }
interface FolderResult { message: string; added: number; duplicates: number; skipped_masks: number; failed: string[]; n_failed: number }

const MANUAL = "(직접 입력)";

export function FolderTab({ options, onDataChanged }: { options: IngestOptions; onDataChanged: () => void }) {
  const { data: catalog, error: catalogError } = useFetch<CatalogLite>("/api/ingest/catalog");
  const [choice, setChoice] = useState<string | null>(null);
  const [root, setRoot] = useState("");
  const [layout, setLayout] = useState(options.default_dataset.layout);
  const [source, setSource] = useState("local");
  const [limitOn, setLimitOn] = useState(false);
  const [limit, setLimit] = useState(200);
  const [includeMasks, setIncludeMasks] = useState(false);
  const job = useJob<FolderResult>(() => onDataChanged());

  const names = catalog ? [...catalog.datasets.map((d) => `${d.name} (${d.key})`), MANUAL] : [MANUAL];
  const current = choice ?? names[0];
  const dataset = catalog?.datasets.find((d) => `${d.name} (${d.key})` === current);
  // 데이터셋을 고르면 구조 해석 방식과 출처 이름이 그 데이터셋의 기본값으로 따라온다.
  useEffect(() => {
    if (!catalog) return;
    if (dataset) { setLayout(dataset.layout); setSource(dataset.key); }
    else { setLayout(options.default_dataset.layout); setSource("local"); }
  }, [current, catalog]);   // eslint-disable-line react-hooks/exhaustive-deps

  const settledRoot = useDebounced(root.trim(), 400);
  const previewUrl = settledRoot ? `/api/ingest/folder/preview?root=${encodeURIComponent(settledRoot)}&layout=${encodeURIComponent(layout)}` : null;
  const { data: preview, error: previewError, loading } = useFetch<Preview>(previewUrl, [previewUrl]);

  const start = () => void job.start(() => post("/api/ingest/folder/ingest", {
    root: settledRoot, layout, source: source || "local", include_masks: includeMasks, limit: limitOn ? Math.round(limit) : null,
  }));
  const result = job.job?.result ?? null;

  return (
    <>
      {md("내려받아 압축을 푼 데이터셋 폴더를 지정하면, 폴더 구조에서 **카테고리 · 분할 · 정상/결함 · 결함 유형**을 자동으로 추론해 등록한다. 원본 파일은 이동·복사하지 않고 경로만 인덱싱한다.")}
      {catalogError ? <ErrorBox error={catalogError} /> : null}
      <Select label="데이터셋" value={current} options={names} onChange={setChoice} help="기본 예시 데이터셋인 VisA가 맨 앞에 온다." />
      {dataset ? <Caption>{`예상 폴더 구조: \`${dataset.layout_note}\``}</Caption> : null}

      <div className="row">
        <div style={{ flex: "3 1 0" }}>
          <TextInput label="데이터셋 루트 폴더 경로" value={root} onChange={setRoot} placeholder="/home/user/datasets/VisA" />
        </div>
        <div style={{ flex: "1 1 0" }}>
          <Select label="구조 해석 방식" value={layout} options={options.layouts} onChange={setLayout}
            help={options.layouts.map((k) => `${k}: ${options.layout_help[k]}`).join(" · ")} />
        </div>
      </div>
      <Cols n={3}>
        <TextInput label="출처 이름 (source)" value={source} onChange={setSource} />
        <div>
          <Checkbox label="건수 제한 (미리보기)" checked={limitOn} onChange={setLimitOn} />
          <NumberInput label="최대 건수" value={limit} min={1} max={100000} step={50} disabled={!limitOn} onChange={setLimit} />
        </div>
        <Checkbox label="ground_truth 마스크도 등록" checked={includeMasks} onChange={setIncludeMasks}
          help="기본적으로 결함 마스크 이미지는 학습 입력이 아니라서 제외한다." />
      </Cols>

      {!settledRoot ? null : previewError ? <Alert kind="error">{previewError}</Alert> : !preview || loading ? <Spinner /> : (
        <>
          <Alert kind="info" icon="🔎">{`이미지 파일 ${preview.total.toLocaleString()}건을 발견했습니다.`}</Alert>
          {preview.total > 0 ? (
            <>
              {/* 실제 등록 전에 라벨 추론 결과를 미리 보여준다 */}
              <Expander title="구조 해석 미리보기 (상위 10건)" open>
                <DataTable table={preview.preview} />
                <Caption>추론 결과가 의도와 다르면 '구조 해석 방식'을 바꿔 다시 확인한다.</Caption>
              </Expander>
              <Button primary busy={job.running} onClick={start}>📥 manifest에 등록</Button>
              <JobProgress job={job.job} error={job.error} />
              {result ? (
                <>
                  <Alert kind="success" icon="✅">{result.message}</Alert>
                  {result.n_failed ? (
                    <Expander title={`읽기 실패 ${result.n_failed}건`}>{md(result.failed.map((f) => `- \`${f}\``).join("\n"))}</Expander>
                  ) : null}
                </>
              ) : null}
            </>
          ) : null}
        </>
      )}
    </>
  );
}
