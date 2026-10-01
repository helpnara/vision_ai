import { Link } from "react-router-dom";
import { Alert, Caption } from "../../components/ui";

/** 레지스트리 행 (registry.REGISTRY_COLUMNS). 값이 없으면 null. */
export interface ProductionRow {
  version: string; kind?: string | null; model?: string | null; threshold?: number | null;
  recall?: number | null; promoted_at?: string | null;
}

/** `_production_banner`의 자리. 서비스 중 모델이 없으면 경고, 있으면 초록 배너. */
export function ProductionBanner({ production, runsEmpty }: { production: ProductionRow | null; runsEmpty: boolean }) {
  if (!production) {
    return (
      <>
        <Alert kind="warning" icon="📚">서비스 중인 모델이 없습니다. **모델 레지스트리** 탭에서 3단계 실행을 등록하고 승격하세요.</Alert>
        {runsEmpty ? (
          <>
            <Caption>등록할 실행이 아직 없습니다. 3단계에서 모델을 먼저 학습하세요.</Caption>
            <Link to="/modeling">➡️ 3단계 모델 개발·평가로 이동</Link>
          </>
        ) : null}
      </>
    );
  }
  const threshold = production.threshold;
  const text = threshold !== null && threshold !== undefined
    ? `서비스 중: **${production.version}** (${production.kind ?? ""}/${production.model ?? ""}) · 임계값 ${threshold.toFixed(4)} · 승격 ${production.promoted_at ?? "—"}`
    : `서비스 중: **${production.version}**`;
  return <Alert kind="success" icon="🟢">{text}</Alert>;
}

export type Notice = { kind: "success" | "info" | "warning" | "error"; text: string; icon?: string } | null;

export function NoticeBox({ notice }: { notice: Notice }) {
  return notice ? <Alert kind={notice.kind} icon={notice.icon}>{notice.text}</Alert> : null;
}
