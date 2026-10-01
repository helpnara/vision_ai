import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { downloadText, downloadUrl, get, post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { Gallery } from "../../components/Gallery";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Cols, DataTable, Divider, ErrorBox, Expander, NumberInput, Select, Spinner, fmt, md } from "../../components/ui";
import { VideoSource } from "./VideoSource";

interface Result {
  source_name: string; version: string; note: string; video_path: string; video_name: string; mime: string;
  summary: string; scanned: number; fps: number; has_defects: boolean;
  highlights: { path: string; caption: string }[]; frames: Table; csv: string; directory: string;
}
interface Payload { usable: string[]; default_version: string | null; kind: string; default_limit: number }

/** 판정 영상 (V7) — 테스트 영상을 걸어 프레임마다 판정을 그려 넣은 영상을 만든다. */
export function PlaybackTab() {
  const [version, setVersion] = useState<string | null>(null);
  const { data, error } = useFetch<Payload>(`/api/operations/playback${version ? `?version=${encodeURIComponent(version)}` : ""}`, [version]);
  const [source, setSource] = useState<string | null>(null);
  const [limit, setLimit] = useState<number | null>(null);
  const [found, setFound] = useState<Result | null>(null);
  const [searching, setSearching] = useState(false);
  const job = useJob<Result>((r) => setFound(r));

  useEffect(() => {
    if (!data) return;
    if (version === null && data.default_version) setVersion(data.default_version);
    if (limit === null) setLimit(data.default_limit);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  // 이미 만들어 둔 판정본이 있으면 되찾아 온다. 화면을 새로 열 때마다 몇 분짜리 작업을
  // 다시 시키면 아무도 두 번 쓰지 않는다.
  useEffect(() => {
    if (!source || !version) { setFound(null); return; }
    setSearching(true);
    get<{ result: Result | null }>(`/api/operations/playback/find?source=${encodeURIComponent(source)}&version=${encodeURIComponent(version)}`)
      .then((got) => setFound(got.result)).catch(() => setFound(null)).finally(() => setSearching(false));
  }, [source, version]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const intro = (
    <>
      {md("테스트 영상을 걸어 **프레임마다 판정을 그려 넣은 영상**을 만든다. 재현율 0.82는 다음에 무엇을 할지 알려주지 않지만, **어디서 놓치고 어디서 헛짚는지**는 알려준다.")}
      <Caption>원본 영상은 그대로 둡니다. 판정본은 별도 폴더에 새 파일로 만듭니다.</Caption>
    </>
  );
  if (!data.usable.length) {
    return (
      <>
        {intro}
        <Alert kind="warning" icon="📚">추론에 쓸 수 있는 버전이 없습니다. 모델 파일이 함께 등록된 버전이 필요합니다.</Alert>
        <Link to="/modeling">➡️ 3단계 모델링으로 이동</Link>
      </>
    );
  }

  const render = () => void job.start(() => post("/api/operations/playback/render", { source, version, limit }));
  return (
    <>
      {intro}
      <VideoSource onChange={setSource} />
      <Cols n={2}>
        <Select label="버전" value={version} options={data.usable} onChange={setVersion} />
        <NumberInput label="판정할 프레임 수" value={limit ?? data.default_limit} min={10} max={600} step={10} onChange={(v) => setLimit(Math.max(10, Math.min(600, Math.round(v))))} help="영상 전체에서 고르게 뽑습니다. 앞에서부터 자르면 뒷부분을 통째로 놓칩니다." />
      </Cols>
      {data.kind === "anomaly" ? (
        <Caption>이상탐지 모델이므로 **왼쪽에 판정 박스, 오른쪽에 열지도**를 나란히 그립니다. 박스만 보면 «왜 저기냐»를 알 수 없고, 열지도만 보면 «잡았다/놓쳤다»가 안 보입니다.</Caption>
      ) : (
        <Caption>분류 모델은 결함의 **위치를 모릅니다.** 박스 대신 프레임마다 판정만 얹습니다. 위치까지 보려면 이상탐지 또는 검출 모델이 필요합니다.</Caption>
      )}
      {source ? (
        <>
          <Button primary busy={job.running || searching} onClick={render}>{found ? "🎥 다시 만들기" : "🎥 판정 영상 만들기"}</Button>
          <JobProgress job={job.job} error={job.error} />
          {found && !job.running ? <><Divider /><PlaybackResult result={found} /></> : null}
        </>
      ) : null}
    </>
  );
}

/** 판정 영상이 위, 대표 판정 결과가 아래. 움직이는 것을 먼저 보고 멈춰서 확인한다. */
function PlaybackResult({ result }: { result: Result }) {
  return (
    <>
      <h3>▶️ 판정 영상 — <code>{result.source_name}</code> / <code>{result.version}</code></h3>
      {result.note ? <Alert kind="warning" icon="🎞️">{result.note}</Alert> : (
        <video className="player" src={`/api/operations/playback/video?path=${encodeURIComponent(result.video_path)}`} controls />
      )}
      <Button onClick={() => void downloadUrl(`/api/operations/playback/download?path=${encodeURIComponent(result.video_path)}`, result.video_name)}>판정 영상 내려받기</Button>
      <Caption>{`${result.summary} 원본 ${fmt.int(result.scanned)}프레임에서 고르게 뽑았고, 재생 속도는 ${result.fps.toFixed(1)}fps입니다.`}</Caption>

      <h3>대표 판정 결과</h3>
      {result.highlights.length === 0 ? <Alert kind="info">보여 줄 장면이 없습니다.</Alert> : null}
      {result.highlights.length > 0 && !result.has_defects ? (
        <Alert kind="info" icon="🔍">결함으로 판정된 프레임이 없습니다. 아래는 **기준선에 가장 가까웠던** 장면입니다 — 여기까지 갔는데 못 넘었다는 뜻이므로 임계값을 조정할지 판단할 근거가 됩니다.</Alert>
      ) : null}
      <Gallery cols={2} items={result.highlights.map((h) => ({ path: h.path, caption: h.caption, width: 1200 }))} />

      <Expander title="프레임별 판정 표">
        <DataTable table={result.frames} scroll />
        <Button onClick={() => downloadText(`playback_${result.version || "model"}.csv`, result.csv)}>판정 결과 내려받기</Button>
      </Expander>
      <Caption>{`판정본 위치: \`${result.directory}\``}</Caption>
    </>
  );
}
