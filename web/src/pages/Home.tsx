import { Link } from "react-router-dom";
import { post, type Table } from "../api";
import { useFetch, useJob } from "../hooks";
import { PageTitle } from "../components/Layout";
import { JobProgress } from "../components/JobProgress";
import { Alert, Button, Caption, Card, Cols, DataTable, Divider, Expander, ErrorBox, Metric, Spinner, fmt, md } from "../components/ui";

interface Overview {
  stats: { total: number; normal: number; defect: number; unlabeled: number; categories: number; sources: number };
  label_stats: { human: number; unspecified_type: number; with_roi: number; split_assigned: number } | null;
  default_dataset: { name: string; license: string };
  progress: { done: number; total: number };
  next_step: { key: string; stage: number; title: string; detail: string; action: string; where: string; route: string } | null;
  steps: { key: string; stage: number; title: string; done: boolean; detail: string; where: string; route: string }[];
  stages: { no: number; name: string; icon: string; detail: string }[];
  validation: {
    available: boolean; mean_auroc?: number; mean_ap?: number; mean_recall?: number; source?: string; assumed_prevalence?: number;
    impact?: { reduction_ratio: number; missed: number; precision: number };
    categories?: { category: string; auroc: number; ap: number; recall: number | null; precision: number | null; false_alarm_rate: number | null }[];
  };
  data_root: string;
  glossary: { term: string; text: string }[];
  defect_types: { key: string; label: string }[];
  quickstart_available: boolean;
}

interface QuickstartResult { ok: boolean; version: string | null; n_images: number; n_train: number; warnings: string[] }

export function Home() {
  const { data, error, reload } = useFetch<Overview>("/api/home/overview");
  const quick = useJob<QuickstartResult>(() => void reload());

  if (error) return <ErrorBox error={error} />;
  if (!data) return <Spinner />;
  const { stats, label_stats: ls, validation: v } = data;
  const result = quick.job?.result ?? null;
  const upcoming = data.next_step;

  return (
    <>
      <PageTitle title="🔍 비전 기반 표면 결함 탐지 파이프라인" caption={md("**제조업 기준 PoC** — 물건 표면의 결함을 탐지하는 모델을 오픈 데이터셋 기반으로 개발한다. 수집부터 운영까지 4단계를 이 앱에서 순차적으로 다룬다.")} />

      <h2>데이터 현황</h2>
      <Cols n={5}>
        <Metric label="전체 이미지" value={fmt.int(stats.total)} />
        <Metric label="정상" value={fmt.int(stats.normal)} />
        <Metric label="결함" value={fmt.int(stats.defect)} />
        <Metric label="미라벨" value={fmt.int(stats.unlabeled)} />
        <Metric label="카테고리" value={fmt.int(stats.categories)} />
      </Cols>
      {stats.total === 0 ? (
        <Caption>{`기본 예시 데이터셋은 **${data.default_dataset.name}**(${data.default_dataset.license})입니다. 내려받지 않았어도 합성 샘플로 전 과정을 시험할 수 있습니다.`}</Caption>
      ) : ls ? (
        <>
          <h2>라벨링 현황</h2>
          <Cols n={4}>
            <Metric label="사람이 라벨/확인" value={fmt.int(ls.human)} />
            <Metric label="유형 미지정 결함" value={fmt.int(ls.unspecified_type)} />
            <Metric label="ROI 지정" value={fmt.int(ls.with_roi)} />
            <Metric label="분할 배정" value={fmt.int(ls.split_assigned)} />
          </Cols>
        </>
      ) : null}

      <Divider />
      {data.quickstart_available ? (
        <Card>
          <h4>⚡ 처음이라면 — 한 번에 시작하기</h4>
          <Caption>합성 샘플 생성 → 분할 → 학습 → 승격까지 한 번에 밟아, **4단계 운영 시연을 곧바로 누를 수 있는 상태**로 만듭니다. 다운로드도 설정도 필요 없습니다.</Caption>
          <Button primary busy={quick.running} onClick={() => void quick.start(() => post("/api/home/quickstart"))}>⚡ 데모 한 바퀴 만들기</Button>
          <JobProgress job={quick.job} error={quick.error} />
          <Caption>여기서 만든 것은 **합성 샘플**입니다. 결함이 인위적으로 뚜렷해 지표가 실제보다 높게 나오므로 성능 근거로 쓰면 안 됩니다.</Caption>
        </Card>
      ) : null}
      {result ? (
        <>
          {result.warnings.map((w) => <Alert key={w} kind="warning">{w}</Alert>)}
          {result.ok ? (
            <Alert kind="success">{`준비 완료 — **${result.version}** 을 서비스 중으로 올렸습니다 (이미지 ${result.n_images.toLocaleString()}장 / 학습 ${result.n_train.toLocaleString()}장). 이제 **4단계 → 운영 시나리오 시연**을 눌러 보세요.`}</Alert>
          ) : null}
        </>
      ) : null}

      <h2>진행 순서</h2>
      <div className="progress"><div className="bar"><div style={{ width: `${(data.progress.done / data.progress.total) * 100}%` }} /></div>
        <div className="text">{data.progress.done}/{data.progress.total} 완료</div></div>
      {upcoming === null ? (
        <Alert kind="success" icon="🎉">전 과정을 한 바퀴 돌았습니다. 이제 **4단계**에서 드리프트와 성능 추이를 살펴보거나, **3단계**로 돌아가 모델을 개선할 수 있습니다.</Alert>
      ) : (
        <Card>
          <h4>👉 다음: {upcoming.title}</h4>
          <Caption>{`${upcoming.where} · 현재 상태: ${upcoming.detail}`}</Caption>
          {upcoming.action ? md(upcoming.action) : null}
          <Link to={upcoming.route}>➡️ {upcoming.stage}단계로 이동</Link>
        </Card>
      )}
      <Expander title="전체 순서 보기" open={upcoming === null}>
        {data.steps.map((s) => (
          <div key={s.key} style={{ margin: "0.3rem 0" }}>
            {s.done ? "✅" : s.key === upcoming?.key ? "👉" : "⬜"} <strong>{s.title}</strong> — {s.detail}
            <div className="caption" style={{ marginLeft: "1.6rem" }}><Link to={s.route}>{s.where}</Link></div>
          </div>
        ))}
      </Expander>

      <Divider />
      <h2>기능별 구현 현황</h2>
      <Caption>아래는 앱에 구현된 기능 목록이다. 내가 어디까지 했는지는 위의 진행 순서를 본다.</Caption>
      {data.stages.map((s) => (
        <Card key={s.no}>
          <div className="row"><div><strong>{s.icon} {s.no}단계 · {s.name}</strong></div><div style={{ flex: "0 0 auto" }}>✅ 구현 완료</div></div>
          <Caption>{s.detail}</Caption>
        </Card>
      ))}

      <Divider />
      <h2>실측 성능 (VisA PCB 4종)</h2>
      {!v.available ? (
        <Caption>아직 실데이터 측정 결과가 없습니다. `PYTHONPATH=src python scripts/validate_visa.py`로 생성합니다.</Caption>
      ) : (
        <>
          <Cols n={3}>
            <Metric label="평균 AUROC" value={fmt.num(v.mean_auroc)} />
            <Metric label="평균 AP" value={fmt.num(v.mean_ap)} />
            <Metric label="평균 재현율" value={fmt.num(v.mean_recall)} />
          </Cols>
          <DataTable table={validationTable(v.categories ?? [])} />
          {v.impact ? (
            <>
              <Cols n={2}>
                <Metric label="검수량 절감" value={fmt.pct(v.impact.reduction_ratio)} help={`불량률 ${fmt.pct(v.assumed_prevalence)} 가정. 전수 검수 대비 사람이 안 봐도 되는 비율입니다.`} />
                <Metric label="1,000장당 놓치는 결함" value={`${fmt.int(v.impact.missed)}건`} help="절감의 대가입니다. 이 값을 받아들일 수 있는지가 도입 판단의 핵심입니다." />
              </Cols>
              <Alert kind="warning">{`위 표의 정밀도는 **정상:결함 = 1:1인 시험 구성** 기준이라 현장 기대치가 아닙니다. 불량률을 ${fmt.pct(v.assumed_prevalence)}로 가정하면 기대 정밀도는 **${fmt.pct(v.impact.precision, 1)}** 로 떨어집니다. 따라서 이 모델은 자동 판정용이 아니라 **1차 스크리닝용**입니다 — 사람이 볼 물량을 줄여 주는 것이 실제 효용입니다.`}</Alert>
            </>
          ) : null}
          <Caption>{`출처: ${v.source ?? ""}`}</Caption>
        </>
      )}

      <Divider />
      <Caption>{`데이터 저장 위치: \`${data.data_root}\` — git 추적 대상이 아닙니다. 로컬 PC에서 실행하면 이 폴더에 그대로 남습니다.`}</Caption>

      <Expander title="데이터 사용 원칙">
        {md(`- **사내(회사) 결함 데이터는 사용하지 않는다.** 공개 데이터셋과 직접 촬영 이미지만 사용한다.\n- 오픈 데이터셋은 라이선스가 각기 다르다. 비상업 조건(CC BY-NC-SA 등)이 붙은 것이 많으므로 사용 전 원본 배포 페이지에서 조건을 직접 확인한다.\n- 데이터 루트: \`${data.data_root}\` (git 추적 대상 아님)`)}
      </Expander>
      <Expander title="용어 사전 — 화면에 나오는 말이 낯설다면">
        <Caption>이 앱에 나오는 전문 용어를 한 줄씩 풀어 썼습니다.</Caption>
        {md(data.glossary.map((g) => `- **${g.term}** — ${g.text}`).join("\n"))}
      </Expander>
      <Expander title="결함 유형 분류 체계 (표준 10종)">
        {md(data.defect_types.map((d) => `- \`${d.key}\` — ${d.label}`).join("\n"))}
      </Expander>
    </>
  );
}

function validationTable(rows: NonNullable<Overview["validation"]["categories"]>): Table {
  const r3 = (x: number | null | undefined) => (x === null || x === undefined ? null : Math.round(x * 1000) / 1000);
  return {
    columns: ["카테고리", "AUROC", "AP", "재현율", "정밀도", "오탐률"],
    rows: rows.map((r) => ({ "카테고리": r.category, AUROC: r3(r.auroc), AP: r3(r.ap), "재현율": r3(r.recall), "정밀도": r3(r.precision), "오탐률": r3(r.false_alarm_rate) })),
    total: rows.length,
  };
}
