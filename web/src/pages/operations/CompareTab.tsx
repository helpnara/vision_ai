import { useEffect, useState } from "react";
import { get, post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Cols, DataTable, ErrorBox, NumberInput, Select, Spinner, md } from "../../components/ui";

interface Payload { empty: boolean; names: string[] }
interface Result { table: Table; cause: string; verdict: string; advice: string; drift_text: string; causes: { none: string; unknown: string } }

/** «모델 탓인가 촬영 탓인가» 한 화면 (V4). */
export function CompareTab() {
  const { data, error } = useFetch<Payload>("/api/operations/compare");
  const [left, setLeft] = useState<string | null>(null);
  const [right, setRight] = useState<string | null>(null);
  const [fps, setFps] = useState(30);
  const [ready, setReady] = useState<boolean | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const job = useJob<Result>((r) => setResult(r));

  useEffect(() => {
    if (!data || data.names.length < 2) return;
    if (left === null) setLeft(data.names[0]);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps
  const others = data ? data.names.filter((n) => n !== left) : [];
  useEffect(() => {
    if (left !== null && (right === null || right === left) && others.length) setRight(others[0]);
  }, [left, others, right]);
  // 두 영상 모두 정답과 대조할 프레임이 있어야 비교할 수 있다 — 버튼을 보이기 전에 묻는다.
  useEffect(() => {
    if (!left || !right) return;
    setResult(null);
    get<{ ok: boolean }>(`/api/operations/compare/check?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}&fps=${fps}`)
      .then((got) => setReady(got.ok)).catch(() => setReady(false));
  }, [left, right, fps]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const intro = md("영상 A로 만든 모델을 영상 B에 걸었더니 성능이 떨어졌다. 그때 할 일은 **촬영을 맞추는 것**과 **모델을 손보는 것**으로 갈리는데, 둘은 서로 배타적이다 — 틀린 쪽을 고르면 시간만 버린다. 두 영상의 지표와 입력 분포를 한 화면에 놓고 결론을 낸다.");
  if (data.empty) return <>{intro}<Alert kind="info" icon="📥">수집된 이미지가 없습니다.</Alert></>;
  if (data.names.length < 2) {
    return (
      <>
        {intro}
        <Alert kind="warning" icon="🎞️">{`비교하려면 **정답이 붙은 영상이 두 편** 필요합니다. 지금은 ${data.names.length}편입니다. 배치 추론(대상: 영상 하나)을 각 영상에 돌리고, 2단계 **영상 구간 라벨링**으로 정답을 붙이세요.`}</Alert>
      </>
    );
  }

  return (
    <>
      {intro}
      <Cols n={3}>
        <Select label="기준 영상 (학습에 쓴 쪽)" value={left} options={data.names} onChange={setLeft} />
        <Select label="비교할 영상 (새 촬영본)" value={right} options={others} onChange={setRight} />
        <NumberInput label="영상 fps" value={fps} min={1} max={240} step={1} onChange={(v) => setFps(Math.max(1, Math.min(240, v)))} />
      </Cols>
      {ready === false ? <Caption>두 영상 모두 정답과 대조할 프레임이 있어야 비교할 수 있습니다.</Caption> : null}
      {ready ? (
        <>
          <Button primary busy={job.running} onClick={() => void job.start(() => post("/api/operations/compare/run", { left, right, fps }))}>🆚 두 영상 비교</Button>
          {!result && !job.running ? <Caption>입력 분포까지 비교하려면 특징을 다시 읽어야 해서 버튼으로 시작합니다.</Caption> : null}
          <JobProgress job={job.job} error={job.error} />
        </>
      ) : null}
      {result ? (
        <>
          <DataTable table={result.table} />
          {result.cause === result.causes.none ? <Alert kind="success">{result.verdict}</Alert>
            : result.cause === result.causes.unknown ? <Alert kind="warning" icon="📏">{result.verdict}</Alert>
            : <Alert kind="error" icon="🚨">{result.verdict}</Alert>}
          {md(`**다음에 할 일** — ${result.advice}`)}
          <Caption>{result.drift_text}</Caption>
          <Caption>성능이 떨어진 것과 입력이 변한 것이 **같이 일어났다**는 사실을 말할 뿐, 인과를 증명하지는 않습니다.</Caption>
        </>
      ) : null}
    </>
  );
}
