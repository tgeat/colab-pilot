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
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / Math.pow(1024, i)).toFixed(i ? 1 : 0)} ${u[i]}`;
}

export function pct(v?: number | null): number {
  if (v === undefined || v === null || !isFinite(v)) return 0;
  return Math.max(0, Math.min(100, v));
}

export function shortEndpoint(ep?: string | null): string {
  if (!ep) return "—";
  return ep.length > 22 ? `${ep.slice(0, 10)}…${ep.slice(-8)}` : ep;
}

export function timeAgo(tsSec?: number | null): string {
  if (!tsSec) return "—";
  return fmtDuration(Date.now() / 1000 - tsSec) + " 前";
}
