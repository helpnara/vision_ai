import { useEffect, useState } from "react";
import { downloadText, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { Alert, Button, DataTable, ErrorBox, Json, Select, Spinner, md } from "../../components/ui";

interface ExperimentsPayload {
  empty: boolean;
  table?: Table;
  help?: Record<string, string>;
  best?: string | null;
  run_ids?: string[];
  csv?: string;
}

export function ExperimentsTab() {
  const { data, error } = useFetch<ExperimentsPayload>("/api/modeling/experiments");
  const [picked, setPicked] = useState<string | null>(null);
  const runIds = data?.run_ids ?? [];
  useEffect(() => { if (runIds.length && (!picked || !runIds.includes(picked))) setPicked(runIds[0]); }, [runIds.join("|")]);
  const { data: detail } = useFetch<unknown>(picked ? `/api/modeling/experiments/${encodeURIComponent(picked)}` : null, [picked]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  return (
    <>
      {md("설정과 지표를 같은 줄에 남겨 실행을 비교한다. 4단계 모델 레지스트리의 토대가 된다.")}
      {data.empty ? <Alert kind="info" icon="🧪">아직 기록된 실행이 없습니다. 모델을 학습하면 자동으로 남습니다.</Alert> : (
        <>
          <DataTable table={data.table} help={data.help} scroll />
          {data.best ? <Alert kind="success" icon="🏆">{data.best}</Alert> : null}
          <Select label="상세 보기" value={picked} options={runIds} onChange={setPicked} />
          {detail ? <Json value={detail} /> : null}
          <Button onClick={() => downloadText("experiments.csv", data.csv ?? "")}>실험 기록 내려받기</Button>
        </>
      )}
    </>
  );
}
