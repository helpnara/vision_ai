import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { get, post } from "../../api";
import { Alert, Caption, Radio, Select, Spinner, TextInput } from "../../components/ui";

/**
 * 쓸 영상을 정한다 — 목록에서 고르기 · 올리기 · 경로 직접 입력.
 *
 * 1단계(추출)와 4단계(판정 영상)가 같은 일을 한다 — «어떤 영상을 쓸 것인가». 화면마다 따로
 * 만들면 한쪽만 고쳐지고 다른 쪽은 옛날 방식으로 남는다. 그래서 하나로 두고 양쪽이 쓴다
 * (예전 `ui.video_source`). `prefix`는 한 화면에 선택기가 둘 있을 때 상태를 가르는 용도다.
 *
 * 예전에는 파일 경로를 통째로 타이핑해야 했다. 오타 하나로 실패하고, 무엇보다 **어떤
 * 영상이 이미 올라와 있는지 화면에서 알 수가 없었다.** 목록을 기본으로 두고, 목록에
 * 없는 영상만 올리거나 경로로 지정한다.
 *
 * API는 1단계 라우터에 있다: GET /api/ingest/videos?scan= · POST /api/ingest/videos/upload ·
 * GET /api/ingest/videos/check?path=
 */

export const SOURCE_LIST = "목록에서 고르기";
export const SOURCE_UPLOAD = "올리기";
export const SOURCE_PATH = "경로 직접 입력";
export const SOURCES = [SOURCE_LIST, SOURCE_UPLOAD, SOURCE_PATH] as const;
export type Source = (typeof SOURCES)[number];

export interface VideoEntry { path: string; name: string; label: string }
interface Listing { videos: VideoEntry[]; video_dir: string; scan_dir: string; scan_dir_missing: boolean }

export interface VideoSourceHandle {
  /** 방금 만든(또는 올린) 영상을 목록 모드에서 바로 골라진 상태로 만든다. */
  pick: (path: string) => void;
}

export const VideoSource = forwardRef<VideoSourceHandle, {
  prefix: string;
  value: string | null;
  onChange: (path: string | null) => void;
  extensions: string[];
}>(function VideoSource({ prefix, value, onChange, extensions }, ref) {
  const [source, setSource] = useState<Source | null>(null);   // null = 목록 결과를 보고 정한다
  const [scanDir, setScanDir] = useState("");
  const [scanApplied, setScanApplied] = useState("");
  const [listing, setListing] = useState<Listing | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [entered, setEntered] = useState("");
  const [pathError, setPathError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState<VideoEntry | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const version = useRef(0);

  const reload = async (scan: string) => {
    const mine = ++version.current;
    try {
      const got = await get<Listing>(`/api/ingest/videos?scan=${encodeURIComponent(scan)}`);
      if (mine !== version.current) return;
      setListing(got);
      setListError(null);
      // 처음 한 번: 영상이 있으면 목록, 없으면 올리기 (예전 index=0 if found else 1)
      setSource((s) => s ?? (got.videos.length ? SOURCE_LIST : SOURCE_UPLOAD));
    } catch (e) {
      if (mine === version.current) setListError((e as Error).message);
    }
  };
  useEffect(() => { void reload(scanApplied); }, [scanApplied]);   // eslint-disable-line react-hooks/exhaustive-deps
  // 폴더 입력이 멈춘 뒤에 훑는다 — 글자마다 NAS를 rglob 하면 안 된다.
  useEffect(() => {
    const t = window.setTimeout(() => setScanApplied(scanDir.trim()), 500);
    return () => window.clearTimeout(t);
  }, [scanDir]);

  useImperativeHandle(ref, () => ({
    pick: (path: string) => {
      setSource(SOURCE_LIST);
      onChange(path);
      void reload(scanApplied);
    },
  }));

  const found = listing?.videos ?? [];
  // 목록 모드에서 아직 고른 것이 없으면 첫 영상을 고른 것으로 친다 (selectbox의 기본 선택).
  useEffect(() => {
    if (source !== SOURCE_LIST) return;
    if (found.length === 0) { if (value !== null) onChange(null); return; }
    if (!value || !found.some((v) => v.path === value)) onChange(found[0].path);
  }, [source, listing]);   // eslint-disable-line react-hooks/exhaustive-deps

  const changeSource = (s: Source) => {
    setSource(s);
    if (s === SOURCE_UPLOAD) onChange(uploaded?.path ?? null);
    else if (s === SOURCE_PATH) onChange(pathError || !entered ? null : entered);
    // 목록 모드는 위 effect가 첫 영상을 고른다
  };

  const onUpload = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setUploadError(null);
    try {
      const form = new FormData();
      form.append("file", file);
      const got = await post<VideoEntry>("/api/ingest/videos/upload", form);
      setUploaded(got);
      onChange(got.path);
      void reload(scanApplied);
    } catch (e) {
      setUploadError((e as Error).message);
      onChange(null);
    } finally { setBusy(false); }
  };

  // 경로 직접 입력: 멈춘 뒤에 서버에 «파일이 있는가 · 영상인가»를 묻는다.
  useEffect(() => {
    if (source !== SOURCE_PATH) return;
    if (!entered.trim()) { setPathError(null); onChange(null); return; }
    const t = window.setTimeout(async () => {
      try {
        const got = await get<VideoEntry>(`/api/ingest/videos/check?path=${encodeURIComponent(entered.trim())}`);
        setPathError(null);
        onChange(got.path);
      } catch (e) {
        setPathError((e as Error).message);
        onChange(null);
      }
    }, 400);
    return () => window.clearTimeout(t);
  }, [entered, source]);   // eslint-disable-line react-hooks/exhaustive-deps

  if (listError) return <Alert kind="error">{listError}</Alert>;
  if (!listing || source === null) return <Spinner />;

  return (
    <div data-video-source={prefix}>
      <Radio<Source> label="영상 지정 방식" value={source} options={SOURCES} onChange={changeSource}
        help="큰 영상은 올리기 상한(200MB)에 걸립니다. 로컬 실행이면 폴더를 훑는 편이 낫습니다." />

      {source === SOURCE_LIST ? (
        <>
          {found.length === 0 ? (
            <Alert kind="info" icon="📂">{`\`${listing.video_dir}\`에 영상이 없습니다. **올리기**로 넣거나, 아래에 영상이 있는 폴더를 적어 훑으세요.`}</Alert>
          ) : (
            <>
              <Select label="영상" value={value && found.some((v) => v.path === value) ? value : found[0].path}
                options={found.map((v) => v.path)} labels={(p) => found.find((v) => v.path === p)?.label ?? p}
                onChange={(p) => onChange(p)}
                help={`${listing.video_dir} 및 아래에 적은 폴더를 훑은 결과입니다.`} />
              {value ? <Caption>{`경로: \`${value}\``}</Caption> : null}
            </>
          )}
          <TextInput label="다른 폴더도 훑기 (선택)" value={scanDir} onChange={setScanDir} placeholder="/mnt/nas/line1-cctv"
            help="하위 폴더까지 내려갑니다. 영상이 프로젝트 밖에 있는 경우가 오히려 보통입니다." />
          {listing.scan_dir_missing ? <Alert kind="warning" icon="📁">{`폴더를 찾을 수 없습니다: \`${listing.scan_dir}\``}</Alert> : null}
        </>
      ) : source === SOURCE_UPLOAD ? (
        <>
          <div className="file-drop">
            <label>영상 올리기{busy ? <span className="spinner" style={{ marginLeft: "0.5rem" }} /> : null}
              <input type="file" accept={extensions.join(",")} disabled={busy} onChange={(e) => void onUpload(e.target.files?.[0])} />
            </label>
          </div>
          {uploadError ? <Alert kind="error">{uploadError}</Alert> : null}
          {uploaded ? <Caption>{`저장 위치: \`${uploaded.path}\` — 다음부터는 **목록에서** 고를 수 있습니다.`}</Caption> : null}
        </>
      ) : (
        <>
          <TextInput label="영상 파일 경로" value={entered} onChange={setEntered} placeholder="/home/user/videos/line1.mp4" />
          {pathError ? <Alert kind="error">{pathError}</Alert> : null}
        </>
      )}
    </div>
  );
});
