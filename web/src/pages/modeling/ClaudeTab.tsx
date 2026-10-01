import { useState } from "react";
import { downloadText, post, type Table } from "../../api";
import { useFetch, useJob } from "../../hooks";
import { JobProgress } from "../../components/JobProgress";
import { Alert, Button, Caption, Checkbox, Cols, DataTable, Divider, ErrorBox, Metric, NumberInput, Select, Spinner, fmt, md } from "../../components/ui";

interface ClaudePayload {
  sdk_installed: boolean;
  hint: string | null;
  has_result: boolean;
  efforts: string[];
  reviews: { empty: true } | { empty: false; metrics: { total: number; ok: number; defect: number; input_tokens: number }; table: Table; csv: string };
  threshold?: number;
  candidates?: { count: number; cost: number; model: string; table: Table };
}

interface RunResult { ok: boolean; statuses: Record<string, number>; first_reason: string }

export function ClaudeTab() {
  const [limit, setLimit] = useState(8);
  const [effort, setEffort] = useState("low");
  const [skipReviewed, setSkipReviewed] = useState(true);
  const { data, error, reload } = useFetch<ClaudePayload>(`/api/modeling/claude?limit=${limit}&skip_reviewed=${skipReviewed}`, [limit, skipReviewed]);
  const run = useJob<RunResult>(() => void reload());

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const result = run.job?.result ?? null;

  return (
    <>
      {md("1차 모델이 **애매하다고 본 이미지만** Claude에 넘겨 결함 유형과 판단 근거를 받는다. 전량을 보내면 비용이 선형으로 늘기 때문에, 임계값 근처만 골라 보내는 것이 핵심이다.")}
      {!data.sdk_installed ? <Alert kind="error" icon="📦">`anthropic` SDK가 설치되지 않았습니다. `pip install anthropic` 후 다시 시도하세요.</Alert> : null}
      {data.hint ? <Alert kind="warning" icon="🔑">{data.hint}</Alert> : null}
      {!data.has_result ? (
        <Alert kind="info" icon="🧠">먼저 **베이스라인** 또는 **이상탐지** 탭에서 1차 모델을 실행하세요. 그 결과에서 애매한 이미지를 골라옵니다.</Alert>
      ) : (
        <>
          <Cols n={3}>
            <NumberInput label="최대 판정 건수" value={limit} min={1} max={100} step={1} onChange={(v) => setLimit(Math.max(1, Math.min(100, Math.round(v))))} />
            <Select label="추론 강도(effort)" value={effort} options={data.efforts} onChange={setEffort}
              help="단일 이미지 판정은 짧은 과업이라 low로도 충분한 편이다. 비용·지연에 직접 영향을 준다." />
            <Checkbox label="이미 판정한 이미지 제외" checked={skipReviewed} onChange={setSkipReviewed} help="같은 이미지를 두 번 청구하지 않는다." />
          </Cols>
          {data.candidates!.count === 0 ? <Alert kind="success" icon="🎉">2차 판정이 필요한 이미지가 없습니다.</Alert> : (
            <>
              <Caption>{`1차 모델 임계값 ${data.threshold!.toFixed(4)} 근처의 ${data.candidates!.count}건을 선정했습니다. 예상 비용 약 $${data.candidates!.cost.toFixed(3)} (모델 ${data.candidates!.model}, 이미지 해상도에 따라 달라짐)`}</Caption>
              <DataTable table={data.candidates!.table} />
              <Button primary disabled={!data.sdk_installed} busy={run.running} onClick={() => void run.start(() => post("/api/modeling/claude/run", { limit, effort, skip_reviewed: skipReviewed }))}>🤖 Claude 2차 판정 실행</Button>
              <JobProgress job={run.job} error={run.error} />
            </>
          )}
          {result ? (
            result.ok ? <Alert kind="success">{`판정 완료: ${JSON.stringify(result.statuses)}`}</Alert> : (
              <>
                <Alert kind="error">{`모든 판정이 실패했습니다: ${JSON.stringify(result.statuses)}`}</Alert>
                <Caption>{`첫 실패 사유: ${result.first_reason}`}</Caption>
              </>
            )
          ) : null}
        </>
      )}

      <Divider />
      <h3>판정 이력</h3>
      {data.reviews.empty ? <Caption>아직 판정 기록이 없습니다.</Caption> : (
        <>
          <Cols n={4}>
            <Metric label="전체 판정" value={fmt.int(data.reviews.metrics.total)} />
            <Metric label="성공" value={fmt.int(data.reviews.metrics.ok)} />
            <Metric label="결함 판정" value={fmt.int(data.reviews.metrics.defect)} />
            <Metric label="입력 토큰 합" value={fmt.int(data.reviews.metrics.input_tokens)} />
          </Cols>
          <DataTable table={data.reviews.table} scroll />
          <Button onClick={() => downloadText("claude_reviews.csv", (data.reviews as { csv: string }).csv)}>판정 결과 내려받기</Button>
        </>
      )}
    </>
  );
}
