import { useState } from "react";
import { post, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { useProject } from "../../components/Layout";
import { Alert, Button, Caption, Checkbox, Cols, DataTable, Divider, ErrorBox, Metric, NumberInput, Radio, Slider, Spinner, md } from "../../components/ui";
import { NoImages, type Notice } from "./shared";

interface Payload {
  total_images: number; grouped: number; videos: number;
  split_modes: { key: string; label: string }[];
  gap: { total: number; assigned: number; unassigned: number; labeled_unassigned: number; boxed_unassigned: number; complete: boolean; message: string; advice: string };
  has_split: boolean; crosstab: Table | null;
}

const SPLIT_BY_GROUP = "group";
const SPLIT_BY_TIME = "time";

/** 카테고리 × 라벨로 층화 분할한다. 층화하지 않으면 특정 카테고리나 결함 클래스가 한쪽 분할에만 몰려 평가 결과를 신뢰할 수 없다. */
export function SplitTab({ onChanged }: { onChanged: () => void }) {
  const project = useProject();
  const { data, error, reload } = useFetch<Payload>("/api/labeling/split", [project.version]);
  const [train, setTrain] = useState(0.6);
  const [val, setVal] = useState(0.2);
  const [seed, setSeed] = useState(42);
  const [labeledOnly, setLabeledOnly] = useState(true);
  const [mode, setMode] = useState<string>(SPLIT_BY_GROUP);
  const [file, setFile] = useState<File | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const intro = md("카테고리 × 라벨로 **층화 분할**한다. 층화하지 않으면 특정 카테고리나 결함 클래스가 한쪽 분할에만 몰려 평가 결과를 신뢰할 수 없다.");
  if (error) return <>{intro}<ErrorBox error={error} /></>;
  if (!data) return <>{intro}<Spinner /></>;
  if (data.total_images === 0) return <>{intro}<NoImages /></>;

  const test = Math.round(Math.max(0, 1 - train - val) * 100) / 100;
  const changed = async () => { await reload(); onChanged(); };

  const run = async (fn: () => Promise<Notice>) => {
    setBusy(true);
    try { setNotice(await fn()); await changed(); }
    catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); }
  };
  const doSplit = () => run(async () => {
    const out = await post<{ count: number }>("/api/labeling/split/run", { train, val, test, seed, labeled_only: labeledOnly, mode });
    return out.count ? { kind: "success", text: `${out.count.toLocaleString()}건에 분할을 배정했습니다.` } : { kind: "warning", text: "분할할 대상이 없습니다." };
  });
  const applyCsv = () => run(async () => {
    if (!file) return null;
    const form = new FormData();
    form.append("file", file);
    const out = await post<{ count: number; unmatched: number }>("/api/labeling/split/import", form);
    return out.count
      ? { kind: "success", text: `${out.count.toLocaleString()}건 적용, 매칭 실패 ${out.unmatched.toLocaleString()}건` }
      : { kind: "warning", text: `매칭된 이미지가 없습니다 (실패 ${out.unmatched.toLocaleString()}건). 컬럼과 파일명을 확인하세요.` };
  });
  const clear = () => run(async () => { await post("/api/labeling/split/clear"); return null; });

  const gap = data.gap;
  return (
    <>
      {intro}
      <Cols n={4}>
        <Slider label="train 비율" value={train} min={0.1} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={setTrain} />
        <Slider label="val 비율" value={val} min={0} max={0.5} step={0.05} format={(v) => v.toFixed(2)} onChange={setVal} />
        <Metric label="test 비율" value={test.toFixed(2)} />
        <NumberInput label="시드" value={seed} min={0} max={9999} step={1} onChange={(v) => setSeed(Math.round(v))} />
      </Cols>
      <Checkbox label="라벨된 이미지만 분할" checked={labeledOnly} onChange={setLabeledOnly} help="미라벨 이미지는 학습에 쓸 수 없으므로 기본적으로 제외한다." />

      {/* 무엇을 하나로 묶어 옮길지 고르게 한다. 고르게 하되, 모르고 고르지는 않게 한다. 영상 프레임이 있는데
          무작위를 고르면 학습에 쓴 것과 거의 같은 장면이 평가에 들어가 성능이 실제보다 높게 나온다.
          이건 화면이 알려주지 않으면 알아채기 어렵다. */}
      <Radio label="무엇을 기준으로 나눌까" value={mode} options={data.split_modes.map((m) => m.key)} labels={(k) => data.split_modes.find((m) => m.key === k)?.label ?? k} onChange={setMode} />
      {data.grouped
        ? <Caption>{`영상에서 뽑은 프레임 ${data.grouped.toLocaleString()}장이 있습니다 (영상 ${data.videos.toLocaleString()}개).`}</Caption>
        : <Caption>영상 프레임이 없어 이미지 하나가 곧 그룹입니다 — 세 방식의 결과가 거의 같습니다.</Caption>}
      {mode === SPLIT_BY_GROUP ? <Caption>같은 영상의 프레임은 통째로 같은 분할로 갑니다.</Caption>
        : mode === SPLIT_BY_TIME ? <Caption>수집 시각 순으로 앞은 학습, 뒤는 평가로 나눕니다. 영상이 하나뿐이라 그룹으로 나눌 수 없을 때 씁니다 — 경계 부근 프레임은 여전히 비슷해 누수가 조금 남습니다.</Caption>
        : data.grouped ? <Alert kind="warning">**영상 프레임이 있는데 무작위로 나눕니다.** 같은 영상의 프레임이 학습과 평가에 섞여 들어가 **성능이 실제보다 높게 나옵니다.** 벤치마크와 맞추려는 경우가 아니라면 '영상(그룹) 단위'를 쓰세요.</Alert>
        : null}

      {test <= 0 ? <Alert kind="error">test 비율이 0 이하다. train/val 비율을 줄여야 한다.</Alert>
        : <Button primary busy={busy} onClick={() => void doSplit()}>✂️ 층화 분할 실행</Button>}
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}

      <Divider />
      {md("**공식 분할 정의 임포트**")}
      <Caption>VisA는 `split_csv/` 아래에 공식 train/test 분할을 제공한다. 벤치마크 수치와 비교하려면 직접 분할하는 대신 공식 정의를 쓰는 편이 낫다. 이미지 경로 컬럼과 split 컬럼을 자동으로 찾아 **경로 접미사**로 매칭한다(VisA는 Normal/Anomaly에 같은 파일명을 쓰므로 파일명만으로는 구분되지 않는다).</Caption>
      <div className="file-drop">
        분할 정의 CSV
        <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </div>
      {file ? <Button busy={busy} onClick={() => void applyCsv()}>📥 CSV 분할 적용</Button> : null}

      <Divider />
      {/* 분할이 배정되지 않은 장수를 밝힌다 (S4). 분할을 한 번 돌리고 나서 데이터를 더 모으면 새로 들어온 것은
          배정 없이 남는다. 그 상태로 검출 학습 폴더를 내보내면 그만큼이 통째로 빠지는데, 이 화면은 그 사실을
          말해 주지 않았다 — 내보내기 메시지를 읽고서야 알게 됐다. */}
      <Cols n={3}>
        <Metric label="전체" value={`${gap.total.toLocaleString()}장`} />
        <Metric label="분할 배정" value={`${gap.assigned.toLocaleString()}장`} />
        <Metric label="미배정" value={`${gap.unassigned.toLocaleString()}장`} help="분할이 없는 이미지는 학습에도 평가에도 들어가지 않습니다." />
      </Cols>
      {gap.complete ? <Alert kind="success">{gap.message}</Alert>
        : gap.boxed_unassigned ? <Alert kind="error" icon="🚨">{`${gap.message} ${gap.advice}`}</Alert>
        : <Alert kind="warning" icon="✂️">{`${gap.message} ${gap.advice}`}</Alert>}

      {data.has_split && data.crosstab ? (
        <>
          {md("**현재 분할 결과**")}
          <DataTable table={data.crosstab} />
          <Button busy={busy} onClick={() => void clear()}>🗑️ 분할 초기화</Button>
        </>
      ) : <Alert kind="info" icon="✂️">아직 분할이 배정되지 않았습니다.</Alert>}
    </>
  );
}
