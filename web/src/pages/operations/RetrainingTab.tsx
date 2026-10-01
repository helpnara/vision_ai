import { useEffect, useState } from "react";
import { post } from "../../api";
import { useFetch } from "../../hooks";
import { Alert, Caption, Card, Cols, Divider, ErrorBox, NumberInput, Slider, Spinner, md } from "../../components/ui";

interface Signal { key: string; level: string; icon: string; title: string; detail: string }
interface Decision { recommended: boolean; signals: Signal[] }
interface Payload {
  defaults: { new_label_threshold: number; recall_margin: number };
  help: { new_label_threshold: string; recall_margin: string }; caption: string; decision: Decision;
}

export function RetrainingTab() {
  const { data, error } = useFetch<Payload>("/api/operations/retraining");
  const [threshold, setThreshold] = useState<number | null>(null);
  const [margin, setMargin] = useState<number | null>(null);
  const [decision, setDecision] = useState<Decision | null>(null);

  useEffect(() => {
    if (!data) return;
    setThreshold(data.defaults.new_label_threshold);
    setMargin(data.defaults.recall_margin);
    setDecision(data.decision);
  }, [data]);
  // 입력이 바뀌면 다시 판단한다 — Streamlit이 rerun마다 계산하던 것 (300ms 디바운스).
  useEffect(() => {
    if (!data || threshold === null || margin === null) return;
    if (threshold === data.defaults.new_label_threshold && margin === data.defaults.recall_margin) return;
    const t = window.setTimeout(() => {
      post<Decision>("/api/operations/retraining", { new_label_threshold: threshold, recall_margin: margin }).then(setDecision).catch(() => undefined);
    }, 300);
    return () => window.clearTimeout(t);
  }, [threshold, margin, data]);

  if (error) return <ErrorBox error={error} />;
  if (!data || threshold === null || margin === null || !decision) return <Spinner />;

  return (
    <>
      {md("운영 신호를 모아 재학습이 필요한지 판단한다. **자동 배포가 아니라 사람이 승인하는 제안**을 만드는 것이 목적이다.")}
      <Cols n={2}>
        <NumberInput label="재학습 기준 신규 라벨 수" value={threshold} min={10} max={5000} step={10} help={data.help.new_label_threshold} onChange={(v) => setThreshold(Math.max(10, Math.min(5000, Math.round(v))))} />
        <Slider label="허용 재현율 낙폭" value={margin} min={0.01} max={0.3} step={0.01} format={(v) => v.toFixed(2)} help={data.help.recall_margin} onChange={setMargin} />
      </Cols>
      <Caption>{data.caption}</Caption>

      {decision.recommended ? (
        <Alert kind="error" icon="🔁">**재학습을 제안합니다.** 아래 근거를 확인하고 사람이 승인하세요.</Alert>
      ) : (
        <Alert kind="success">지금은 재학습이 필요해 보이지 않습니다.</Alert>
      )}

      <Divider />
      <h3>판단 근거</h3>
      {decision.signals.map((s) => (
        <Card key={s.key}>
          {md(`${s.icon} **${s.title}**`)}
          <Caption>{s.detail}</Caption>
        </Card>
      ))}

      {decision.recommended ? (
        <>
          <Divider />
          <h3>재학습 절차</h3>
          {md("1. **2. 라벨링** — 미탐 사례를 검수하고 신규 라벨을 정리한다.\n2. **3. 모델 개발·평가** — 같은 설정으로 다시 학습하고 지표를 비교한다.\n3. **4. 모델 레지스트리** — 새 실행을 등록하고, 지표가 나아졌을 때만 승격한다.\n4. 승격 후 **배치 추론**을 다시 돌려 드리프트 기준선 대비 상태를 확인한다.")}
          <Caption>승격은 되돌릴 수 있다 — 이전 버전이 보관 상태로 남아 있어 다시 승격하면 롤백된다.</Caption>
        </>
      ) : null}
    </>
  );
}
