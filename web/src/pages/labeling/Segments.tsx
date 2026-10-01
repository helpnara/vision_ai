import { useState } from "react";
import { post } from "../../api";
import { useFetch } from "../../hooks";
import { useProject } from "../../components/Layout";
import { Gallery } from "../../components/Gallery";
import { TimelinePicker } from "../../components/TimelinePicker";
import { Alert, Button, Caption, Cols, ErrorBox, Expander, Metric, NumberInput, Radio, Select, Spinner, md } from "../../components/ui";
import { LABEL_DEFECT, LABEL_NORMAL, LABEL_UNLABELED, NoImages, defectChoiceLabel, labelKo, type Notice, type Overview } from "./shared";

interface Frame { image_id: string; frame_index: number; label: string; label_ko: string }
interface Payload { videos: { name: string; frames: Frame[] }[] }

/**
 * 영상 구간 라벨링 (H4). 검사원은 프레임을 한 장씩 보지 않는다. 영상을 돌려 보며
 * "여기부터 여기까지 불량"이라고 짚는다. 타임라인에서 구간을 끌면 그 구간의 프레임에 라벨이 한 번에 붙는다.
 */
export function SegmentsTab({ overview }: { overview: Overview }) {
  const project = useProject();
  const { data, error, reload } = useFetch<Payload>("/api/labeling/video-frames", [project.version]);
  const [picked, setPicked] = useState<string | null>(null);
  const [fps, setFps] = useState(30);
  // 구간은 영상마다 따로 기억한다 (예전 key=f"p2_segspan::{picked}").
  const [spans, setSpans] = useState<Record<string, [number, number] | null>>({});
  const [label, setLabel] = useState<string>(LABEL_DEFECT);
  const [defectType, setDefectType] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  const intro = md("검사원은 프레임을 한 장씩 보지 않는다. 영상을 돌려 보며 **\"여기부터 여기까지 불량\"** 이라고 짚는다. 타임라인에서 구간을 끌면 그 구간의 프레임에 라벨이 한 번에 붙는다.");
  if (error) return <>{intro}<ErrorBox error={error} /></>;
  if (!data) return <>{intro}<Spinner /></>;
  if (data.videos.length === 0) {
    return <>{intro}<NoImages text="영상에서 뽑은 프레임이 없습니다. 1단계에서 영상을 먼저 등록하세요." /></>;
  }

  const names = data.videos.map((v) => v.name);
  const name = picked && names.includes(picked) ? picked : names[0];
  const subset = data.videos.find((v) => v.name === name)!.frames;
  const safeFps = fps > 0 ? fps : 30;
  const seconds = subset.map((f) => f.frame_index / safeFps);
  const duration = seconds.length ? Math.max(...seconds) : 1.0;
  const labeled = subset.filter((f) => f.label !== LABEL_UNLABELED).length;
  const span = spans[name] ?? null;
  const inside = span ? subset.filter((_, i) => seconds[i] >= span[0] && seconds[i] <= span[1]) : [];
  const isDefect = label === LABEL_DEFECT;
  const needsType = isDefect && !defectType;

  const apply = async () => {
    if (!span) return;
    setBusy(true);
    try {
      const out = await post<{ count: number }>("/api/labeling/segments/apply", {
        video: name, start: span[0], end: span[1], image_ids: inside.map((f) => f.image_id),
        label, defect_type: isDefect ? defectType : null,
      });
      setNotice({ kind: "success", text: `${out.count.toLocaleString()}장에 라벨을 붙였습니다.` });
      await reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); }
  };

  return (
    <>
      {intro}
      <Select label="영상" value={name} options={names} onChange={setPicked} />
      <NumberInput label="영상 fps" value={fps} min={1} max={240} step={1} onChange={setFps}
        help="타임라인의 초를 계산할 때만 씁니다. 원본 fps를 모르면 그대로 두어도 구간 지정은 됩니다." />
      <Cols n={3}>
        <Metric label="이 영상의 프레임" value={`${subset.length.toLocaleString()}장`} />
        <Metric label="길이" value={`${duration.toFixed(1)}초`} />
        <Metric label="라벨된 프레임" value={`${labeled.toLocaleString()}장`} />
      </Cols>
      <TimelinePicker ticks={subset.map((f, i) => ({ t: seconds[i], mark: f.label_ko }))} duration={duration}
        span={span} onChange={(s) => setSpans({ ...spans, [name]: s })} />
      <Caption>타임라인 위를 **드래그**해 구간을 고르세요. 고른 구간 안쪽을 끌면 위치가 옮겨집니다.</Caption>
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
      {notice?.kind === "success" ? (
        <Caption>결함 **위치(박스)** 는 구간으로 정할 수 없습니다 — 프레임마다 다르기 때문입니다. **라벨 검수** 탭에서 한 장씩 그리세요.</Caption>
      ) : null}
      {!span ? (
        <Alert kind="info" icon="👆">아직 구간을 고르지 않았습니다.</Alert>
      ) : (
        <>
          <Alert kind="success" icon="🎯">{`**${span[0].toFixed(1)}초 ~ ${span[1].toFixed(1)}초** — 프레임 ${inside.length.toLocaleString()}장이 들어갑니다.`}</Alert>
          {inside.length ? (
            <>
              <Expander title="이 구간의 프레임 미리보기 (앞 8장)" open>
                <Gallery items={inside.slice(0, 8).map((f) => ({ imageId: f.image_id }))} />
              </Expander>
              <Cols n={2}>
                <Radio label="이 구간의 판정" value={label} options={[LABEL_DEFECT, LABEL_NORMAL]} labels={(v) => labelKo(overview, v)} onChange={setLabel} />
                <Select label="결함 유형" value={defectType} options={overview.defect_types.map((t) => t.key)} labels={defectChoiceLabel(overview)}
                  placeholder="결함 유형을 선택하세요" disabled={!isDefect} onChange={setDefectType} />
              </Cols>
              {needsType ? <Caption>⚠️ 결함으로 판정했으면 유형을 선택해야 저장할 수 있다.</Caption> : null}
              <Button primary busy={busy} disabled={needsType} onClick={() => void apply()}>{`🏷️ ${inside.length.toLocaleString()}장에 한 번에 라벨`}</Button>
            </>
          ) : null}
        </>
      )}
    </>
  );
}
