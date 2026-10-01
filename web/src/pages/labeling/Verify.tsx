import { useEffect, useState } from "react";
import { imageUrl, post } from "../../api";
import { useFetch } from "../../hooks";
import { useProject } from "../../components/Layout";
import { Alert, Button, Caption, Cols, ErrorBox, Metric, NumberInput, Radio, Slider, Spinner, md } from "../../components/ui";
import { LABEL_DEFECT, LABEL_NORMAL, NoImages, labelKo, type Notice, type Overview } from "./shared";

interface Item { image_id: string; category: string; raw_defect_type: string; label: string; defect_type: string }
interface Payload { total_images: number; pending: number; items: Item[] }

/** 오픈 데이터셋은 폴더 구조로 라벨이 이미 주어진다. 그대로 신뢰하지 않고 표본을 눈으로 확인해 오라벨을 잡아낸다. */
export function VerifyTab({ overview }: { overview: Overview }) {
  const project = useProject();
  const [sampleSize, setSampleSize] = useState(8);
  const [seed, setSeed] = useState(0);
  const { data, error, reload } = useFetch<Payload>(`/api/labeling/verify?size=${sampleSize}&seed=${seed}`, [project.version]);
  const [decisions, setDecisions] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  // 표본이 바뀌면 판정도 표본의 현재 라벨로 다시 시작한다 (st.form의 기본값).
  useEffect(() => {
    if (data) setDecisions(Object.fromEntries(data.items.map((it) => [it.image_id, it.label === LABEL_NORMAL ? LABEL_NORMAL : LABEL_DEFECT])));
  }, [data]);

  const intro = md("오픈 데이터셋은 폴더 구조로 라벨이 이미 주어진다. 그대로 신뢰하지 않고 표본을 눈으로 확인해 오라벨을 잡아낸다. 확인한 이미지는 `verified`로 기록된다.");
  if (error) return <>{intro}<ErrorBox error={error} /></>;
  if (!data) return <>{intro}<Spinner /></>;
  if (data.total_images === 0) return <>{intro}<NoImages /></>;
  if (data.pending === 0) return <>{intro}{notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}<Alert kind="success" icon="🎉">폴더 라벨을 모두 확인했습니다.</Alert></>;

  const submit = async () => {
    setBusy(true);
    try {
      const out = await post<{ count: number; changed: number }>("/api/labeling/verify", {
        decisions: data.items.map((it) => ({ image_id: it.image_id, label: decisions[it.image_id] ?? it.label })),
      });
      setNotice({ kind: "success", text: `${out.count}건 확인 처리 (라벨 수정 ${out.changed}건)` });
      await reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); }
  };

  return (
    <>
      {intro}
      <Cols n={3}>
        <Metric label="미확인 폴더 라벨" value={data.pending.toLocaleString()} />
        <Slider label="표본 개수" value={sampleSize} min={4} max={24} step={4} onChange={setSampleSize} />
        <NumberInput label="표본 시드" value={seed} min={0} max={9999} step={1} onChange={(v) => setSeed(Math.round(v))} />
      </Cols>
      <Caption>라벨이 틀린 이미지는 아래에서 값을 바꾼 뒤 한 번에 저장한다.</Caption>
      <div className="gallery cols-4">
        {data.items.map((it) => (
          <figure key={it.image_id}>
            <img src={imageUrl(it.image_id)} alt={it.image_id} loading="lazy" />
            <figcaption>{it.category} · <code>{it.raw_defect_type}</code></figcaption>
            <Radio label="" value={decisions[it.image_id] ?? (it.label === LABEL_NORMAL ? LABEL_NORMAL : LABEL_DEFECT)}
              options={[LABEL_NORMAL, LABEL_DEFECT]} labels={(v) => labelKo(overview, v)}
              onChange={(v) => setDecisions({ ...decisions, [it.image_id]: v })} />
          </figure>
        ))}
      </div>
      <Button primary busy={busy} onClick={() => void submit()}>✅ 표본 확인 처리</Button>
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
    </>
  );
}
