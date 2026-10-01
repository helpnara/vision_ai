import { useEffect, useState } from "react";
import { del, post, put } from "../api";
import { useFetch } from "../hooks";
import { PageTitle, useProject } from "../components/Layout";
import { Alert, Button, Caption, Cols, Divider, ErrorBox, Expander, Metric, NumberInput, Select, Slider, Spinner, TextInput, fmt, md } from "../components/ui";

type Values = { target_recall: number; min_reduction: number; prevalence: number; volume: number; new_label_threshold: number; recall_margin: number };
interface Preview { measured_recall: number; measured_fpr: number; impact: { reduction_ratio: number; missed: number; precision: number }; passes: boolean; missing: string[] }
interface SettingsPayload {
  values: Values; defaults: Values; bounds: Record<keyof Values, [number, number]>; labels: Record<keyof Values, string>; help: Record<keyof Values, string>;
  changed: Record<string, { value: number; default: number }>; path: string; preview: Preview;
  cache: { features: { count: number; size_mb: number; path: string }; patches: { count: number; size_mb: number; variants: number } };
}

export function Settings() {
  const { data, error, reload, setData } = useFetch<SettingsPayload>("/api/settings");
  const project = useProject();
  const [values, setValues] = useState<Values | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [notice, setNotice] = useState<{ kind: "success" | "warning" | "error"; text: string } | null>(null);
  const [cacheNotice, setCacheNotice] = useState<{ kind: "success" | "error"; text: string } | null>(null);

  useEffect(() => { if (data) { setValues(data.values); setPreview(data.preview); } }, [data]);
  // 슬라이더를 움직이면 미리보기가 따라 움직여야 판단할 수 있다 (300ms 디바운스).
  useEffect(() => {
    if (!values) return;
    const t = window.setTimeout(() => { post<Preview>("/api/settings/preview", values).then(setPreview).catch(() => undefined); }, 300);
    return () => window.clearTimeout(t);
  }, [values]);

  if (error) return <ErrorBox error={error} />;
  if (!data || !values) return <Spinner />;
  const set = (k: keyof Values, v: number) => setValues({ ...values, [k]: v });
  const field = (k: keyof Values) => ({ label: data.labels[k], help: data.help[k] });
  const b = data.bounds;

  const save = async () => {
    try {
      const got = await put<SettingsPayload & { clamped: boolean }>("/api/settings", values);
      setData(got);
      setNotice(got.clamped ? { kind: "warning", text: "일부 값이 허용 범위를 벗어나 잘렸습니다." } : { kind: "success", text: "저장했습니다. 3·4단계가 이 기준으로 판단합니다." });
    } catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };
  // 캐시 비우기는 실패할 수 있다 — Windows에서 학습 중이면 파일이 잠겨 있다. 실패를 삼키면
  // «비웠습니다»처럼 보이는데 숫자는 그대로라 사용자가 무엇이 잘못됐는지 알 수 없다.
  const clearCache = async (url: string) => {
    try {
      setData(await post<SettingsPayload>(url));
      setCacheNotice({ kind: "success", text: "비웠습니다. 다음 학습에서 다시 계산합니다." });
    } catch (e) { setCacheNotice({ kind: "error", text: (e as Error).message }); }
  };
  const reset = async () => { setData(await post<SettingsPayload>("/api/settings/reset")); setNotice({ kind: "success", text: "기본값으로 되돌렸습니다." }); };

  return (
    <>
      <PageTitle title="⚙️ 설정" caption={md("여기 값들은 **업무 판단**이지 물리 상수가 아니다. 미탐 1건의 비용이 큰 라인이면 재현율 목표를 올려야 하고, 라인마다 불량률도 다르다. 라인에 맞게 바꿔 쓴다.")} />
      <Alert kind="info">바꾼 값은 **3단계 평가와 4단계 승격 점검이 함께** 사용합니다. 한 곳에서만 바꾸면 같은 모델을 두 화면이 다르게 평가하게 되므로 여기에 모아 두었습니다.</Alert>

      <ProjectSection />

      <Divider />
      {Object.keys(data.changed).length ? (
        <Caption>{`기본값과 다른 항목 ${Object.keys(data.changed).length}개: ` + Object.entries(data.changed).map(([k, v]) => `${data.labels[k as keyof Values]} ${v.value} (기본 ${v.default})`).join(", ")}</Caption>
      ) : <Caption>현재 전부 기본값입니다.</Caption>}

      <Divider />
      <h3>판정 기준 (D5)</h3>
      <Caption>모델을 서비스에 올려도 되는지 판단하는 기준입니다.</Caption>
      <Cols n={2}>
        <Slider {...field("target_recall")} value={values.target_recall} min={b.target_recall[0]} max={b.target_recall[1]} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set("target_recall", v)} />
        <Slider {...field("min_reduction")} value={values.min_reduction} min={b.min_reduction[0]} max={b.min_reduction[1]} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set("min_reduction", v)} />
      </Cols>

      <Divider />
      <h3>현장 가정</h3>
      <Caption>성능 지표를 업무 언어(검수량·놓치는 결함)로 환산할 때 쓰는 값입니다.</Caption>
      <Cols n={2}>
        <NumberInput label={`${data.labels.prevalence} (%)`} help={data.help.prevalence} value={Math.round(values.prevalence * 10000) / 100} min={b.prevalence[0] * 100} max={b.prevalence[1] * 100} step={0.1} onChange={(v) => set("prevalence", v / 100)} />
        <NumberInput {...field("volume")} value={values.volume} min={b.volume[0]} max={b.volume[1]} step={100} onChange={(v) => set("volume", Math.round(v))} />
      </Cols>

      <Divider />
      <h3>재학습 판단</h3>
      <Cols n={2}>
        <NumberInput {...field("new_label_threshold")} value={values.new_label_threshold} min={b.new_label_threshold[0]} max={b.new_label_threshold[1]} step={10} onChange={(v) => set("new_label_threshold", Math.round(v))} />
        <Slider {...field("recall_margin")} value={values.recall_margin} min={b.recall_margin[0]} max={b.recall_margin[1]} step={0.005} format={(v) => v.toFixed(3)} onChange={(v) => set("recall_margin", v)} />
      </Cols>

      <Divider />
      <h3>이 기준이면 어떻게 되는가</h3>
      <Caption>VisA 실측 성능(재현율 92.5% · 오탐률 45.5%)에 지금 설정을 대입한 결과입니다. 설정을 바꾸면 이 숫자가 함께 움직입니다.</Caption>
      {preview ? (
        <>
          <Cols n={3}>
            <Metric label="검수량 절감" value={fmt.pct(preview.impact.reduction_ratio)} />
            <Metric label="놓치는 결함" value={`${fmt.int(preview.impact.missed)}건`} />
            <Metric label="현장 기대 정밀도" value={fmt.pct(preview.impact.precision, 1)} />
          </Cols>
          {preview.passes ? <Alert kind="success">실측 모델이 이 기준을 **통과**합니다.</Alert>
            : <Alert kind="warning">{`실측 모델은 이 기준에 **미달**합니다 — ${preview.missing.join(" · ")}. 승격할 때 확인을 요구받게 됩니다.`}</Alert>}
        </>
      ) : null}

      <Divider />
      <div className="row tight">
        <Button primary onClick={() => void save()}>💾 저장</Button>
        <Button onClick={() => void reset()}>↩️ 기본값으로 되돌리기</Button>
      </div>
      {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
      <Caption>{`저장 위치: \`${data.path}\` (git 추적 대상 아님)`}</Caption>

      <Divider />
      <h3>특징 캐시</h3>
      <Caption>한 번 계산한 이미지 특징을 저장해 두고 다시 씁니다. 이미지 단위로 저장하므로 데이터가 늘어나면 늘어난 만큼만 계산합니다. 지워도 다음 학습 때 다시 만들어집니다.</Caption>
      <div><strong>분류용 (이미지 한 장 = 숫자 14개)</strong></div>
      <Cols n={3}>
        <Metric label="저장된 이미지" value={`${fmt.int(data.cache.features.count)}장`} />
        <Metric label="파일 크기" value={`${data.cache.features.size_mb.toFixed(1)} MB`} />
        <div style={{ paddingTop: "0.8rem" }}><Button disabled={data.cache.features.count === 0} onClick={() => void clearCache("/api/settings/cache/clear")}>🗑️ 캐시 비우기</Button></div>
      </Cols>
      <Caption>{`저장 위치: \`${data.cache.features.path}\``}</Caption>
      <div><strong>이상탐지용 (격자 31×31마다 숫자 14개)</strong></div>
      <Caption>장당 53KB로 분류용의 약 천 배입니다. 그래서 압축하지 않고 float16으로 저장하고, 쓸 때 필요한 줄만 읽습니다. 설정(백엔드·패치 크기)이 다르면 따로 쌓입니다.</Caption>
      <Cols n={3}>
        <Metric label="저장된 이미지" value={`${fmt.int(data.cache.patches.count)}장`} />
        <Metric label="파일 크기" value={`${data.cache.patches.size_mb.toFixed(0)} MB`} help={`설정 ${data.cache.patches.variants}종`} />
        <div style={{ paddingTop: "0.8rem" }}><Button disabled={data.cache.patches.count === 0} onClick={() => void clearCache("/api/settings/cache/patches/clear")}>🗑️ 격자 캐시 비우기</Button></div>
      </Cols>
      {cacheNotice ? <Alert kind={cacheNotice.kind}>{cacheNotice.text}</Alert> : null}
      <span style={{ display: "none" }}>{project.version}{String(reload)}</span>
    </>
  );
}

function ProjectSection() {
  const { listing, refresh } = useProject();
  const [name, setName] = useState("");
  const [renamed, setRenamed] = useState("");
  const [target, setTarget] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ kind: "success" | "error"; text: string } | null>(null);
  const { data: summary, reload: reloadSummary } = useFetch<{ images: number }>(listing ? `/api/projects/${listing.active}/summary` : null, [listing?.active]);
  const current = listing?.projects.find((p) => p.slug === listing.active);
  useEffect(() => { if (current) setRenamed(current.name); }, [current?.slug, current?.name]);
  if (!listing || !current) return null;
  const others = listing.projects.filter((p) => p.slug !== current.slug);

  const run = async (fn: () => Promise<unknown>, ok: string) => {
    try { await fn(); await refresh(); await reloadSummary(); setNotice({ kind: "success", text: ok }); }
    catch (e) { setNotice({ kind: "error", text: (e as Error).message }); }
  };
  return (
    <>
      <h3>프로젝트 (작업공간)</h3>
      <Caption>현장·라인마다 데이터·라벨·모델·판정 기준을 따로 관리합니다. 한 manifest에 섞으면 정상 분포가 넓어져 결함을 놓치고, 성능 지표도 여러 현장의 평균이 되어 어디가 문제인지 알 수 없게 됩니다.</Caption>
      <Cols n={3}>
        <Metric label="현재 프로젝트" value={current.name} />
        <Metric label="등록된 이미지" value={`${fmt.int(summary?.images ?? 0)}장`} />
        <Metric label="전체 프로젝트" value={`${listing.projects.length}개`} />
      </Cols>
      <Caption>{`데이터 위치: \`${listing.data_root}\``}</Caption>
      <Expander title="프로젝트 만들기 · 이름 바꾸기">
        <Cols n={2}>
          <div>
            <strong>새 프로젝트</strong>
            <TextInput label="이름" value={name} placeholder="예: 2공장 도장라인" help="만들면 빈 작업공간으로 전환됩니다. 기존 데이터는 그대로 남습니다." onChange={setName} />
            <Button onClick={() => void run(() => post("/api/projects", { name }), `'${name}' 프로젝트로 전환했습니다.`)}>➕ 만들고 전환</Button>
          </div>
          <div>
            <strong>이름 바꾸기</strong>
            <TextInput label="새 이름" value={renamed} help="폴더 이름은 그대로 둡니다 — 바꾸면 쌓인 데이터를 통째로 옮겨야 합니다." onChange={setRenamed} />
            <Button onClick={() => void run(() => post(`/api/projects/${current.slug}/rename`, { name: renamed }), "이름을 바꿨습니다.")}>✏️ 이름 저장</Button>
          </div>
        </Cols>
        {others.length ? (
          <>
            <Divider />
            <Select label="목록에서 뺄 프로젝트" value={target ?? others[0].slug} options={others.map((p) => p.slug)} labels={(s) => others.find((p) => p.slug === s)?.name ?? s} onChange={setTarget} />
            <Caption>목록에서만 뺍니다. **파일은 지우지 않으므로** 같은 이름으로 다시 만들면 돌아옵니다.</Caption>
            <Button onClick={() => { const slug = target ?? others[0].slug; void run(() => del(`/api/projects/${slug}`), "목록에서 뺐습니다."); }}>🗂️ 목록에서 빼기</Button>
          </>
        ) : null}
        {notice ? <Alert kind={notice.kind}>{notice.text}</Alert> : null}
      </Expander>
    </>
  );
}
