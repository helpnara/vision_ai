import { useEffect, useState } from "react";
import { get, post } from "../../api";
import { Alert, Caption, Radio, Select, TextInput } from "../../components/ui";

/**
 * 쓸 영상을 정한다 (`ui.video_source`의 자리). 고른 경로를 `onChange`로 알린다.
 *
 * 예전에는 파일 경로를 통째로 타이핑해야 했다. 오타 하나로 실패하고, 무엇보다 **어떤
 * 영상이 이미 올라와 있는지 화면에서 알 수가 없었다.** 목록을 기본으로 두고, 목록에
 * 없는 영상만 올리거나 경로로 지정한다.
 *
 * 1단계 쪽과 같은 일을 하지만 엔드포인트는 이 화면의 라우터(`/api/operations/videos*`)에
 * 둔다 — 두 화면이 동시에 만들어지는 중이라, 나중에 하나로 합친다.
 */
const SOURCE_LIST = "목록에서 고르기";
const SOURCE_UPLOAD = "올리기";
const SOURCE_PATH = "경로 직접 입력";
const SOURCES = [SOURCE_LIST, SOURCE_UPLOAD, SOURCE_PATH] as const;
type Source = (typeof SOURCES)[number];

interface Listing { videos: { path: string; label: string }[]; video_dir: string; scan_missing: boolean; extensions: string[] }

export function VideoSource({ onChange }: { onChange: (path: string | null) => void }) {
  const [scan, setScan] = useState("");
  const [listing, setListing] = useState<Listing | null>(null);
  const [source, setSource] = useState<Source | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<{ path: string; message: string } | null>(null);
  const [entered, setEntered] = useState("");
  const [check, setCheck] = useState<{ ok: boolean; error: string | null } | null>(null);

  useEffect(() => {
    const t = window.setTimeout(() => {
      get<Listing>(`/api/operations/videos?scan=${encodeURIComponent(scan)}`).then((got) => {
        setListing(got);
        if (source === null) setSource(got.videos.length ? SOURCE_LIST : SOURCE_UPLOAD);
        if (got.videos.length && !got.videos.some((v) => v.path === picked)) setPicked(got.videos[0].path);
      }).catch(() => setListing(null));
    }, 300);
    return () => window.clearTimeout(t);
  }, [scan]); // eslint-disable-line react-hooks/exhaustive-deps

  // 경로 직접 입력은 서버가 «파일이 있는가 · 영상인가»를 확인한다.
  useEffect(() => {
    if (source !== SOURCE_PATH || !entered) { setCheck(null); return; }
    const t = window.setTimeout(() => {
      get<{ ok: boolean; error: string | null }>(`/api/operations/videos/check?path=${encodeURIComponent(entered)}`).then(setCheck).catch(() => setCheck(null));
    }, 300);
    return () => window.clearTimeout(t);
  }, [source, entered]);

  useEffect(() => {
    if (source === SOURCE_LIST) onChange(listing?.videos.length ? picked : null);
    else if (source === SOURCE_UPLOAD) onChange(uploaded?.path ?? null);
    else if (source === SOURCE_PATH) onChange(check?.ok ? entered : null);
    else onChange(null);
  }, [source, picked, uploaded, check, entered, listing]); // eslint-disable-line react-hooks/exhaustive-deps

  const upload = async (file: File | undefined) => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try { setUploaded(await post<{ path: string; message: string }>("/api/operations/videos/upload", form)); }
    catch (e) { setUploaded(null); setCheck({ ok: false, error: (e as Error).message }); }
  };

  if (!listing || source === null) return <Caption>영상 목록을 읽는 중...</Caption>;
  return (
    <>
      <Radio label="영상 지정 방식" value={source} options={SOURCES} onChange={setSource} help="큰 영상은 올리기 상한(200MB)에 걸립니다. 로컬 실행이면 폴더를 훑는 편이 낫습니다." />
      {source === SOURCE_LIST ? (
        <>
          {listing.videos.length === 0 ? (
            <Alert kind="info" icon="📂">{`\`${listing.video_dir}\`에 영상이 없습니다. **올리기**로 넣거나, 아래에 영상이 있는 폴더를 적어 훑으세요.`}</Alert>
          ) : (
            <>
              <Select label="영상" value={picked} options={listing.videos.map((v) => v.path)} labels={(p) => listing.videos.find((v) => v.path === p)?.label ?? p} onChange={setPicked} help={`${listing.video_dir} 및 아래에 적은 폴더를 훑은 결과입니다.`} />
              {picked ? <Caption>{`경로: \`${picked}\``}</Caption> : null}
            </>
          )}
          <TextInput label="다른 폴더도 훑기 (선택)" value={scan} onChange={setScan} placeholder="/mnt/nas/line1-cctv" help="하위 폴더까지 내려갑니다. 영상이 프로젝트 밖에 있는 경우가 오히려 보통입니다." />
          {listing.scan_missing ? <Alert kind="warning" icon="📁">{`폴더를 찾을 수 없습니다: \`${scan}\``}</Alert> : null}
        </>
      ) : null}
      {source === SOURCE_UPLOAD ? (
        <div className="file-drop">
          영상 올리기
          <input type="file" accept={listing.extensions.join(",")} onChange={(e) => void upload(e.target.files?.[0])} />
          {uploaded ? <Caption>{uploaded.message}</Caption> : null}
          {check && !check.ok && check.error ? <Alert kind="error">{check.error}</Alert> : null}
        </div>
      ) : null}
      {source === SOURCE_PATH ? (
        <>
          <TextInput label="영상 파일 경로" value={entered} onChange={setEntered} placeholder="/home/user/videos/line1.mp4" />
          {entered && check && !check.ok && check.error ? <Alert kind="error">{check.error}</Alert> : null}
        </>
      ) : null}
    </>
  );
}
