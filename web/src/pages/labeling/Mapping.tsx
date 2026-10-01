import { useEffect, useState } from "react";
import { post, type Table } from "../../api";
import { useFetch } from "../../hooks";
import { useProject } from "../../components/Layout";
import { Alert, Button, Caption, DataTable, Divider, ErrorBox, Select, Spinner, md } from "../../components/ui";
import { defectChoiceLabel, type Notice, type Overview } from "./shared";

interface Payload { unmapped: { raw: string; count: number }[]; mapping: Table }

/** 데이터셋마다 결함 유형명이 다르다. 여러 데이터셋을 함께 학습·평가하려면 프로젝트 표준 유형으로 모아야 한다. */
export function MappingTab({ overview }: { overview: Overview }) {
  const project = useProject();
  const { data, error, reload } = useFetch<Payload>("/api/labeling/mapping", [project.version]);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [notice, setNotice] = useState<Notice>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (data) setChoices(Object.fromEntries(data.unmapped.map((u) => [u.raw, "other"])));
  }, [data]);

  const intro = md("데이터셋마다 결함 유형명이 다르다. 여러 데이터셋을 함께 학습·평가하려면 프로젝트 표준 유형으로 모아야 한다.");
  if (error) return <>{intro}<ErrorBox error={error} /></>;
  if (!data) return <>{intro}<Spinner /></>;

  const save = async () => {
    setBusy(true);
    try {
      const out = await post<{ count: number }>("/api/labeling/mapping", { choices });
      setNotice({ kind: "success", text: `${out.count}종 매핑을 저장했습니다.` });
      await reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
    finally { setBusy(false); }
  };

  return (
    <>
      {intro}
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
      {data.unmapped.length === 0 ? (
        <Alert kind="success" icon="🎉">정규화가 필요한 결함 유형이 없습니다.</Alert>
      ) : (
        <>
          <Alert kind="warning" icon="🔀">{`표준 유형으로 매핑되지 않은 유형 ${data.unmapped.length}종`}</Alert>
          {data.unmapped.map((u) => (
            <div key={u.raw} style={{ display: "grid", gridTemplateColumns: "1fr 2fr", gap: "0.75rem", alignItems: "center" }}>
              <div><code>{u.raw}</code><Caption>{`${u.count.toLocaleString()}건`}</Caption></div>
              <Select label="" value={choices[u.raw] ?? "other"} options={overview.defect_types.map((t) => t.key)} labels={defectChoiceLabel(overview)}
                onChange={(v) => setChoices({ ...choices, [u.raw]: v })} />
            </div>
          ))}
          <Button primary busy={busy} onClick={() => void save()}>💾 매핑 저장</Button>
        </>
      )}
      <Divider />
      {md("**현재 매핑 (기본 매핑 포함)**")}
      <DataTable table={data.mapping} scroll />
    </>
  );
}
