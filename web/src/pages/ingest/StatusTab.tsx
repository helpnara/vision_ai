import { useEffect, useState } from "react";
import { del, downloadText, type Count, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { Gallery } from "../../components/Gallery";
import { Alert, Bars, Button, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, Select, Slider, Spinner, fmt } from "../../components/ui";

interface Status {
  empty: boolean;
  stats: { total: number; normal: number; defect: number; unlabeled: number; sources: number; categories: number };
  ratio: number | null;
  by_category: Count[];
  by_defect_type: Count[];
  crosstab: Table;
  quality: { flagged: number; mean_blur: number | null; mean_brightness: number | null; help_blur: string; help_brightness: string; caption: string; flagged_table: Table | null };
  categories: string[];
  labels: { key: string; label: string }[];
  sources: string[];
}
interface Preview { items: { image_id: string; caption: string; exists: boolean }[]; total: number }
interface Manifest { table: Table; csv: string }

const ALL = "(전체)";

export function StatusTab({ onDataChanged }: { onDataChanged: () => void }) {
  const { data, error, reload } = useFetch<Status>("/api/ingest/status");
  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  if (data.empty) return <Alert kind="info" icon="📭">등록된 이미지가 없습니다. 다른 탭에서 데이터를 먼저 수집하세요.</Alert>;
  const { stats, quality } = data;

  return (
    <>
      <Cols n={5}>
        <Metric label="전체" value={fmt.int(stats.total)} />
        <Metric label="정상" value={fmt.int(stats.normal)} />
        <Metric label="결함" value={fmt.int(stats.defect)} />
        <Metric label="미라벨" value={fmt.int(stats.unlabeled)} />
        <Metric label="출처 수" value={fmt.int(stats.sources)} />
      </Cols>
      {/* 클래스 불균형은 이후 학습 전략을 좌우하므로 눈에 띄게 알린다 */}
      {data.ratio !== null ? (
        <Alert kind="warning" icon="⚖️">{`정상:결함 비율이 ${data.ratio.toFixed(1)}:1로 치우쳐 있다. 이상탐지 접근이나 클래스 가중치·증강을 고려할 것.`}</Alert>
      ) : null}

      <Divider />
      <Cols n={2}>
        <Bars items={data.by_category} title="카테고리별 이미지 수" />
        <Bars items={data.by_defect_type} title="결함 유형 분포" />
      </Cols>
      <div><strong>출처 × 라벨 교차표</strong></div>
      <DataTable table={data.crosstab} />

      <Divider />
      <h3>품질 점검</h3>
      <Cols n={3}>
        <Metric label="품질 경고" value={fmt.int(quality.flagged)} />
        <Metric label="평균 선명도(Laplacian)" value={fmt.num(quality.mean_blur, 1)} help={quality.help_blur} />
        <Metric label="평균 밝기" value={fmt.num(quality.mean_brightness, 1)} help={quality.help_brightness} />
      </Cols>
      <Caption>{quality.caption}</Caption>
      {quality.flagged_table ? (
        <Expander title={`경고 이미지 ${quality.flagged.toLocaleString()}건 보기`}>
          <DataTable table={quality.flagged_table} scroll />
        </Expander>
      ) : null}

      <Divider />
      <h3>이미지 미리보기</h3>
      <PreviewSection categories={data.categories} labels={data.labels} />

      <Divider />
      <ManifestSection />

      <RemoveSection sources={data.sources} onRemoved={() => { void reload(); onDataChanged(); }} />
    </>
  );
}

function PreviewSection({ categories, labels }: { categories: string[]; labels: { key: string; label: string }[] }) {
  const [category, setCategory] = useState(ALL);
  const [label, setLabel] = useState(ALL);
  const [count, setCount] = useState(8);
  const url = `/api/ingest/preview?category=${encodeURIComponent(category === ALL ? "" : category)}&label=${encodeURIComponent(label === ALL ? "" : label)}&count=${count}`;
  const { data, error } = useFetch<Preview>(url, [url]);
  return (
    <>
      <Cols n={3}>
        <Select label="카테고리" value={category} options={[ALL, ...categories]} onChange={setCategory} />
        <Select label="라벨" value={label} options={[ALL, ...labels.map((l) => l.key)]} labels={(k) => labels.find((l) => l.key === k)?.label ?? k} onChange={setLabel} />
        <Slider label="표시 개수" value={count} min={4} max={24} step={4} onChange={setCount} />
      </Cols>
      {error ? <ErrorBox error={error} /> : !data ? <Spinner /> : data.items.length === 0 ? (
        <Alert kind="info">조건에 맞는 이미지가 없습니다.</Alert>
      ) : (
        <>
          <Gallery items={data.items.filter((i) => i.exists).map((i) => ({ imageId: i.image_id, caption: i.caption }))} />
          {data.items.filter((i) => !i.exists).map((i) => <Alert key={i.image_id} kind="error">{`파일 없음: ${i.image_id}`}</Alert>)}
        </>
      )}
    </>
  );
}

function ManifestSection() {
  const { data, error } = useFetch<Manifest>("/api/ingest/manifest");
  return (
    <Expander title="manifest 원본 보기 / 내려받기">
      {error ? <ErrorBox error={error} /> : !data ? <Spinner /> : (
        <>
          <DataTable table={data.table} scroll />
          <Button onClick={() => downloadText("manifest.csv", data.csv)}>manifest.csv 내려받기</Button>
        </>
      )}
    </Expander>
  );
}

function RemoveSection({ sources, onRemoved }: { sources: string[]; onRemoved: () => void }) {
  const [target, setTarget] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ kind: "success" | "error"; text: string } | null>(null);
  useEffect(() => { if (target && !sources.includes(target)) setTarget(null); }, [sources, target]);
  const chosen = target ?? sources[0];
  const remove = async () => {
    try {
      const got = await del<{ removed: number; message: string }>(`/api/ingest/sources?name=${encodeURIComponent(chosen)}`);
      setNotice({ kind: "success", text: got.message });
      onRemoved();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };
  return (
    <Expander title="⚠️ 출처별 레코드 삭제">
      <Caption>manifest에서 레코드만 제거하며, 원본 이미지 파일은 지우지 않는다.</Caption>
      <Select label="삭제할 출처" value={chosen} options={sources} onChange={setTarget} />
      <Button onClick={() => void remove()} disabled={!chosen}>레코드 삭제</Button>
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
    </Expander>
  );
}
