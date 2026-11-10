export function fmtDuration(sec?: number | null): string {
  if (sec === undefined || sec === null || !isFinite(sec) || sec <= 0) return "0s";
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function fmtBytes(n: number): string {
  if (!n || !Number.isFinite(n) || n < 0) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function pct(v?: unknown): number {
  const n = num(v, NaN);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n));
}

export function shortEndpoint(ep?: string | null): string {
  if (!ep) return "—";
  return ep.length > 22 ? `${ep.slice(0, 10)}…${ep.slice(-8)}` : ep;
}

/** 远端遥测 JSON 防御：仅接受有限数值，否则返回默认值（防止 null/NaN/字符串导致 .toFixed 崩溃） */
export function num(v: unknown, d = 0): number {
  if (typeof v === "number") return Number.isFinite(v) ? v : d;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isFinite(n) ? n : d;
  }
  return d;
}

/** 有限数值或 null（用于 TTFT 等可缺省指标） */
export function numOrNull(v: unknown): number | null {
  const n = num(v, NaN);
  return Number.isFinite(n) ? n : null;
}

/** 远端遥测 JSON 防御：仅接受字符串（数字会被转为字符串），否则返回默认值 */
export function str(v: unknown, d = ""): string {
  if (typeof v === "string") return v;
  if (typeof v === "number" && Number.isFinite(v)) return String(v);
  return d;
}

export function timeAgo(tsSec?: number | null): string {
  if (!tsSec) return "—";
  return fmtDuration(Date.now() / 1000 - tsSec) + " 前";
}
