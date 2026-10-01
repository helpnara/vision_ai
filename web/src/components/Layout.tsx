import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";
import { get, post } from "../api";
import { Button } from "./ui";

export const PAGES = [
  { path: "/", title: "홈 / 진행 현황", icon: "🏠" },
  { path: "/ingest", title: "1. 데이터 수집", icon: "📥" },
  { path: "/labeling", title: "2. 라벨링", icon: "🏷️" },
  { path: "/modeling", title: "3. 모델 개발 · 평가", icon: "🧠" },
  { path: "/operations", title: "4. 운영 관리 (MLOps)", icon: "⚙️" },
  { path: "/settings", title: "설정", icon: "🛠️" },
];

const CAPTIONS = ["비전 기반 표면 결함 탐지 (제조업 PoC)", "데이터: 오픈 데이터셋 + 직접 촬영 (사내 데이터 미사용)"];

export interface ProjectInfo { slug: string; name: string; note: string; created_at: string }
export interface ProjectListing { active: string; projects: ProjectInfo[]; data_root: string; artifact_root: string }

/**
 * 활성 프로젝트는 서버 전역이다. 바뀌면 **모든 화면의 데이터가 바뀌므로** 화면 전체를 다시
 * 그린다 (`version`이 오르면 각 화면의 useFetch가 다시 읽는다).
 */
const ProjectContext = createContext<{ listing: ProjectListing | null; version: number; refresh: () => Promise<void> }>({
  listing: null, version: 0, refresh: async () => {},
});
export const useProject = () => useContext(ProjectContext);

const RAIL_KEY = "nav_rail";

export function Layout() {
  const [rail, setRail] = useState<boolean>(() => {
    try { return localStorage.getItem(RAIL_KEY) === "1"; } catch { return false; }
  });
  const [listing, setListing] = useState<ProjectListing | null>(null);
  const [version, setVersion] = useState(0);
  const navigate = useNavigate();

  const refresh = useCallback(async () => {
    const got = await get<ProjectListing>("/api/projects");
    setListing((prev) => {
      if (prev && prev.active !== got.active) setVersion((v) => v + 1);
      return got;
    });
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);

  const toggle = () => {
    setRail((r) => { try { localStorage.setItem(RAIL_KEY, r ? "0" : "1"); } catch { /* 비공개 창 등 */ } return !r; });
  };
  const switchProject = async (slug: string) => {
    await post("/api/projects/use", { slug });
    await refresh();
    navigate("/");
  };

  const current = listing?.projects.find((p) => p.slug === listing.active);
  return (
    <ProjectContext.Provider value={{ listing, version, refresh }}>
      <div className="app">
        <aside className={`sidebar${rail ? " rail" : ""}`}>
          <Button tertiary small className="btn tertiary small toggle" onClick={toggle} title={rail ? "메뉴를 펼칩니다." : "메뉴를 아이콘만 남기고 접습니다."}>
            {rail ? "»" : "«"}
          </Button>
          <nav className="nav">
            {PAGES.map((p) => (
              <NavLink key={p.path} to={p.path} end={p.path === "/"} title={rail ? p.title : undefined}>
                <span className="icon">{p.icon}</span>{rail ? null : <span>{p.title}</span>}
              </NavLink>
            ))}
          </nav>
          {rail ? (
            current ? <div className="initials" title={`프로젝트: ${current.name}`}>{current.name.slice(0, 2)}</div> : null
          ) : (
            <>
              <div className="project">
                <label title="현장·라인마다 데이터와 라벨을 섞지 않으려면 프로젝트를 나눕니다.">프로젝트</label>
                <select value={listing?.active ?? ""} onChange={(e) => void switchProject(e.target.value)} style={{ width: "100%", font: "inherit", padding: "0.3rem" }}>
                  {listing?.projects.map((p) => <option key={p.slug} value={p.slug}>{p.name}</option>)}
                </select>
              </div>
              <div className="captions">{CAPTIONS.map((c) => <div key={c}>{c}</div>)}</div>
            </>
          )}
        </aside>
        <main className="main" key={version}>
          <Outlet />
        </main>
      </div>
    </ProjectContext.Provider>
  );
}

export function PageTitle({ title, caption }: { title: string; caption: ReactNode }) {
  return (
    <>
      <h1>{title}</h1>
      <div className="page-caption">{caption}</div>
    </>
  );
}
