/**
 * API 클라이언트. 모든 화면이 이 파일을 통해서만 서버를 부른다.
 *
 * 오류는 서버가 `{detail: "..."}`로 보내므로(FastAPI 관례) 그 문장을 그대로 Error로 던진다.
 * 화면은 `err.message`를 그대로 보여주면 된다 — 코어가 ValueError로 말한 문장이 그것이다.
 */

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : typeof body === "string" && body
          ? body
          : `${res.status} ${res.statusText}`;
    throw new ApiError(res.status, detail);
  }
  return body as T;
}

export async function get<T>(url: string): Promise<T> {
  return parse<T>(await fetch(url, { headers: { Accept: "application/json" } }));
}

export async function send<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method, headers: { Accept: "application/json" } };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers = { ...init.headers, "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  return parse<T>(await fetch(url, init));
}

export const post = <T,>(url: string, body?: unknown) => send<T>("POST", url, body);
export const put = <T,>(url: string, body?: unknown) => send<T>("PUT", url, body);
export const del = <T,>(url: string) => send<T>("DELETE", url);

/** 백그라운드 잡의 상태 (server/jobs.py의 Job.as_dict와 같은 모양). */
export interface JobState<R = unknown> {
  id: string;
  label: string;
  note: string;
  status: "queued" | "running" | "done" | "failed";
  done: number;
  total: number;
  fraction: number | null;
  text: string;
  elapsed: number;
  elapsed_text: string;
  eta_text: string | null;
  result: R | null;
  error: string | null;
}

export const getJob = <R,>(id: string) => get<JobState<R>>(`/api/jobs/${id}`);

/** 파일 서빙 URL. `w=0`이면 상한 안에서 원본 크기. */
export const imageUrl = (imageId: string, width = 480) =>
  `/api/files/image/${encodeURIComponent(imageId)}?w=${width}`;
export const pathUrl = (path: string, width = 480) =>
  `/api/files/path?path=${encodeURIComponent(path)}&w=${width}`;
export const videoUrl = (path: string) => `/api/files/video?path=${encodeURIComponent(path)}`;

/** 표 응답 (server/common.py의 table()). */
export interface Table {
  columns: string[];
  rows: Record<string, unknown>[];
  total: number;
}

export interface Count {
  name: string;
  count: number;
}

/** CSV 등 텍스트를 브라우저 내려받기로 넘긴다. */
export function downloadText(filename: string, text: string, mime = "text/csv") {
  const blob = new Blob(["﻿", text], { type: `${mime};charset=utf-8` });
  downloadBlob(filename, blob);
}

export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** 서버가 만든 파일을 그대로 내려받는다 (잡 결과의 download 경로 등). */
export async function downloadUrl(url: string, filename: string) {
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  downloadBlob(filename, await res.blob());
}
