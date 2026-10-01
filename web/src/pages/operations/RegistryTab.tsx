import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { get, post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, DataTable, Divider, ErrorBox, Expander, Json, Select, Spinner, TextInput, md } from "../../components/ui";
import { NoticeBox, type Notice } from "./shared";

interface Payload {
  table: Table | null; versions: string[]; archivable: string[]; runs_empty: boolean;
  options: { run_id: string; label: string }[]; default_promote_now: boolean;
  rollback_target: { version: string; recall: number | null; promoted_at: string | null } | null;
}
interface Gate { passed: boolean; problems: string[]; notes: string[] }

export function RegistryTab() {
  const { data, error, reload } = useFetch<Payload>("/api/operations/registry");
  const [notice, setNotice] = useState<Notice>(null);
  const [run, setRun] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [promoteNow, setPromoteNow] = useState<boolean | null>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [toArchive, setToArchive] = useState<string | null>(null);
  const [detail, setDetail] = useState<string | null>(null);

  useEffect(() => {
    if (!data) return;
    if (promoteNow === null) setPromoteNow(data.default_promote_now);
    if (run === null && data.options.length) setRun(data.options[0].run_id);
    if (target === null && data.versions.length) setTarget(data.versions[0]);
    if (toArchive === null && data.archivable.length) setToArchive(data.archivable[0]);
    if (detail === null && data.versions.length) setDetail(data.versions[0]);
  }, [data]); // eslint-disable-line react-hooks/exhaustive-deps

  const register = useJob<{ version: string; warnings: string[] }>((r) => {
    setNotice({ kind: "success", text: `${r.version} 등록 완료.` + (r.warnings.length ? "\n" + r.warnings.map((w) => `⚠️ ${w}`).join("\n") : "") });
    setRun(null);
    void reload();
  });

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const act = async (url: string, body?: unknown, icon = "✅") => {
    try {
      const got = await post<{ message: string }>(url, body);
      setNotice({ kind: "success", text: got.message, icon });
      void reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };

  return (
    <>
      {md("3단계 실험 기록은 *무엇을 시도했는지*, 레지스트리는 *무엇을 쓰고 있는지*의 기록이다. 승격할 때 모델 파일을 **버전 폴더로 복사**하므로, 이후 학습이 덮어써도 롤백할 수 있다.")}
      <NoticeBox notice={notice} />
      {data.table ? <DataTable table={data.table} /> : <Alert kind="info" icon="📚">등록된 버전이 없습니다. 아래에서 3단계 실행을 등록하세요.</Alert>}

      <Divider />
      <h3>3단계 실행 등록</h3>
      {data.runs_empty ? (
        <>
          <Alert kind="info" icon="🧪">등록할 실행이 없습니다. 3단계에서 모델을 먼저 학습하세요.</Alert>
          <Link to="/modeling">➡️ 3단계 모델 개발·평가로 이동</Link>
        </>
      ) : data.options.length === 0 ? (
        <Caption>모든 실행이 이미 등록되어 있습니다.</Caption>
      ) : (
        <>
          <Cols n={2}>
            <Select label="실행" value={run} options={data.options.map((o) => o.run_id)} labels={(id) => data.options.find((o) => o.run_id === id)?.label ?? id} onChange={setRun} />
            <TextInput label="메모" value={note} onChange={setNote} />
          </Cols>
          <Checkbox label="등록 후 바로 서비스 중으로 승격" checked={!!promoteNow} onChange={setPromoteNow} />
          <Caption>승격 시 현재 학습 분할로 **드리프트 기준선**(특징 분위 경계)을 함께 저장합니다. 이 기준선이 없으면 입력 분포 변화를 감지할 수 없습니다.</Caption>
          <Button primary busy={register.running} disabled={!run} onClick={() => void register.start(() => post("/api/operations/registry/register", { run_id: run, note, promote_now: !!promoteNow }))}>📚 레지스트리에 등록</Button>
          <JobProgress job={register.job} error={register.error} />
        </>
      )}

      {data.versions.length ? (
        <>
          <Divider />
          <h3>버전 전환</h3>
          <Cols n={2}>
            <div>
              <Select label="승격할 버전" value={target} options={data.versions} onChange={setTarget} />
              {target ? <PromotionGate version={target} onPromote={() => void act("/api/operations/registry/promote", { version: target })} /> : null}
            </div>
            <div>
              <RollbackControl target={data.rollback_target} onRollback={() => void act("/api/operations/registry/rollback", undefined, "↩️")} />
              {data.archivable.length ? (
                <>
                  <Select label="보관할 버전" value={toArchive} options={data.archivable} onChange={setToArchive} />
                  <Button disabled={!toArchive} onClick={() => void act("/api/operations/registry/archive", { version: toArchive })}>🗄️ 보관 처리</Button>
                </>
              ) : null}
            </div>
          </Cols>

          <Expander title="버전 상세">
            <Select label="버전" value={detail} options={data.versions} onChange={setDetail} />
            {detail ? <VersionDetail version={detail} /> : null}
          </Expander>
        </>
      ) : null}
    </>
  );
}

/**
 * 승격 전에 D5 기준으로 점검한다. 미달이면 확인을 받는다.
 *
 * **막지는 않는다.** 기준은 아직 승인 전 제안값이고, 비교나 시연 목적으로 일부러 낮은
 * 모델을 올릴 수도 있다. 다만 초보자가 성능 미달 모델을 아무 신호 없이 올리면, 그 뒤의
 * 드리프트 감시와 재학습 판단이 전부 그 모델을 기준으로 돌아간다.
 */
function PromotionGate({ version, onPromote }: { version: string; onPromote: () => void }) {
  const { data } = useFetch<Gate>(`/api/operations/registry/gate?version=${encodeURIComponent(version)}`, [version]);
  const [ack, setAck] = useState(false);
  useEffect(() => setAck(false), [version]);
  if (!data) return <Spinner />;
  const blocked = !data.passed && !ack;
  return (
    <>
      {data.passed ? <Caption>✅ 성능 기준을 만족합니다.</Caption> : (
        <>
          <Alert kind="warning">{md("**이 버전은 권장 기준에 못 미칩니다.**\n" + data.problems.map((p) => `- ${p}`).join("\n"))}</Alert>
          {data.notes.map((n) => <Caption key={n}>{n}</Caption>)}
          <Checkbox label="기준 미달을 확인했고 그래도 승격합니다" checked={ack} onChange={setAck} />
        </>
      )}
      <Button disabled={blocked} onClick={onPromote}>🟢 서비스 중으로 승격</Button>
    </>
  );
}

/**
 * 직전 서비스 버전으로 되돌린다.
 *
 * 버전을 골라 승격하는 것과 같은 동작이지만, **어느 버전으로 가야 하는지 고르는 일**이
 * 사람에게는 어렵다. 새 모델이 더 나빴을 때 급히 되돌려야 하는 상황이면 더 그렇다.
 * 직전에 쓰던 버전을 찾아 한 번에 되돌린다.
 */
function RollbackControl({ target, onRollback }: { target: Payload["rollback_target"]; onRollback: () => void }) {
  if (!target) return <Caption>되돌릴 이전 서비스 버전이 없습니다. (한 번도 교체하지 않았습니다)</Caption>;
  const detail = target.recall !== null && target.recall !== undefined ? ` · 재현율 ${target.recall.toFixed(3)}` : "";
  return (
    <>
      <Caption>{`직전 서비스 버전: **${target.version}**${detail} (승격 ${target.promoted_at ?? "—"})`}</Caption>
      <Button onClick={onRollback}>↩️ {target.version}(으)로 되돌리기</Button>
    </>
  );
}

function VersionDetail({ version }: { version: string }) {
  const [detail, setDetail] = useState<{ run: unknown; baseline_samples: number | null } | null>(null);
  useEffect(() => {
    setDetail(null);
    get<{ run: unknown; baseline_samples: number | null }>(`/api/operations/registry/detail?version=${encodeURIComponent(version)}`).then(setDetail).catch(() => setDetail({ run: null, baseline_samples: null }));
  }, [version]);
  if (!detail) return <Spinner />;
  return (
    <>
      <Caption>{`드리프트 기준선: ${detail.baseline_samples !== null ? `있음 (${detail.baseline_samples}개 표본)` : "없음"}`}</Caption>
      {detail.run ? <Json value={detail.run} /> : null}
    </>
  );
}
