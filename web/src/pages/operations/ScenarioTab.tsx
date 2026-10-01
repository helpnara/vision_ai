import { Link } from "react-router-dom";
import { post } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Cols, DataTable, Divider, ErrorBox, Expander, Metric, Spinner, fmt, md } from "../../components/ui";
import type { Table } from "../../api";
import { NoticeBox, type Notice, type ProductionRow } from "./shared";
import { useState } from "react";

interface Phase { title: string; environment: string; narration: string; watch: string }
interface ScenarioResult {
  version: string; warnings: string[]; total_logged: number; total_verified: number; days: number;
  table: Table | null; top_drift: { title: string; features: { name: string; psi: number | null }[] }[];
}
interface Payload {
  production: ProductionRow | null; n_images: number; timeline: Phase[]; n_per_phase: number;
  verify_ratio: number; labeled_by: string; result: ScenarioResult | null;
}

/** 운영 몇 달치를 재생해 나머지 탭이 작동하는 모습을 보이게 한다. */
export function ScenarioTab() {
  const { data, error, reload } = useFetch<Payload>("/api/operations/scenario");
  const [notice, setNotice] = useState<Notice>(null);
  const job = useJob<ScenarioResult>(() => { setNotice(null); void reload(); });

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;

  const clear = async () => {
    try {
      const { removed } = await post<{ removed: number }>("/api/operations/scenario/clear");
      setNotice({ kind: "success", text: `시나리오가 만든 추론 로그 ${fmt.int(removed)}건을 지웠습니다.` });
      void reload();
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };

  const result = data.result;
  return (
    <>
      {md("드리프트 감시 · 성능 추이 · 재학습 판단은 **시간이 흐르고 데이터가 쌓여야** 의미가 생긴다. 방금 만든 앱에서는 이 화면들이 전부 비어 있어 운영이 되는지 확인할 수가 없다. 여기서 **운영 3개월치를 몇 초 만에 재생**하면 나머지 탭에 실제로 값이 채워진다.")}
      <Alert kind="info" icon="ℹ️">새 기능을 흉내 내는 것이 아닙니다. 배치 추론 · 드리프트 감시 · 성능 측정은 다른 탭에서 쓰는 것과 **같은 코드**를 그대로 실행합니다. 다른 것은 추론 시각을 과거로 채우고, 조명·초점 변화를 이미지에 입힌다는 점뿐입니다.</Alert>

      {!data.production ? (
        <>
          <Alert kind="warning">서비스 중인 모델이 없습니다. **모델 레지스트리** 탭에서 3단계 실행을 등록하고 승격해야 시나리오를 돌릴 수 있습니다.</Alert>
          <Link to="/modeling">➡️ 3단계에서 모델 학습하기</Link>
        </>
      ) : data.n_images === 0 ? (
        <Alert kind="warning">등록된 이미지가 없습니다. 1단계에서 데이터를 먼저 등록하세요.</Alert>
      ) : (
        <>
          <Caption>{`대상 모델: **${data.production.version}** · 이미지 ${fmt.int(data.n_images)}건`}</Caption>
          <Expander title="재생할 시나리오" open>
            {data.timeline.map((phase, i) => (
              <div key={phase.title} style={{ marginBottom: "0.6rem" }}>
                {md(`**${i + 1}. ${phase.title}** — 환경: \`${phase.environment}\`\n${phase.narration}\n➡️ ${phase.watch}`)}
              </div>
            ))}
          </Expander>
          <Caption>{`구간마다 ${data.n_per_phase}장을 검사하고 그중 ${Math.round(data.verify_ratio * 100)}%를 사후 검수합니다. 검수 정답은 manifest 라벨에서 오고 모델 점수와 무관하므로 자기 채점이 아닙니다. 기록되는 라벨은 \`${data.labeled_by}\` 출처로 남아 사람이 검수한 라벨과 구분됩니다.`}</Caption>

          <div className="row tight">
            <Button primary busy={job.running} onClick={() => void job.start(() => post("/api/operations/scenario/run"))}>🎬 운영 3개월치 재생</Button>
            <Button disabled={job.running} onClick={() => void clear()}>🧹 시나리오 로그 지우기</Button>
          </div>
          <JobProgress job={job.job} error={job.error} />
          <NoticeBox notice={notice} />

          {result ? (
            <>
              {result.warnings.map((w) => <Alert key={w} kind="warning">{w}</Alert>)}
              {result.table ? (
                <>
                  <Divider />
                  <h3>재생 결과</h3>
                  <Cols n={3}>
                    <Metric label="쌓인 추론 로그" value={`${fmt.int(result.total_logged)}건`} />
                    <Metric label="사후 검수" value={`${fmt.int(result.total_verified)}건`} />
                    <Metric label="재생 기간" value={`${result.days}일`} />
                  </Cols>
                  <DataTable table={result.table} />
                  <Caption>재현율은 **검수된 결함 건수**가 적으면 크게 흔들린다. 옆 칸의 건수를 함께 보고 판단하세요.</Caption>
                  {result.top_drift.map((p) => (
                    <div key={p.title}>{md(`**${p.title}** — PSI 상위: ` + p.features.map((f) => `\`${f.name}\` ${fmt.num(f.psi, 2)}`).join(", "))}</div>
                  ))}
                  <Alert kind="success">이제 **드리프트 감시 · 성능 추이 · 재학습 판단 · 판정 이력** 탭에 값이 채워져 있습니다. 차례로 열어 확인하세요.</Alert>
                  <Alert kind="info" icon="🔁">**그 다음은 이렇게 이어집니다.** 3단계로 돌아가 다시 학습하고 새 버전을 승격합니다. 새 모델이 오히려 더 나쁘면 **모델 레지스트리** 탭의 `↩️ 되돌리기`로 직전 버전으로 즉시 복구할 수 있습니다 — 승격할 때 모델 파일을 버전 폴더에 복사해 두기 때문입니다.</Alert>
                  <Alert kind="warning">여기서 나온 수치는 **화면 시연용이지 모델 성능 근거가 아닙니다.** 환경 변화를 인위적으로 넣은 결과이므로 재현율·PSI를 성능으로 인용하면 안 됩니다.</Alert>
                </>
              ) : null}
            </>
          ) : null}
        </>
      )}
    </>
  );
}
