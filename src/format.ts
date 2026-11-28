import type { Ccu } from "./types";

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
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  if (n < 1) return "< 1 B";
  const u = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.max(0, Math.min(u.length - 1, Math.floor(Math.log(n) / Math.log(1024))));
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

/**
 * 格式化计费速率 (#56)：
 * 当 ccu 缺失或 burn_rate_hourly 非有效有限数值时，统一标识为“未知”（中性或告警样式），绝不伪装为绿色 0.00 CU/h。
 */
export function formatBurnRate(
  ccu: Ccu | null | undefined,
  hasActiveInstances = false,
): { text: string; cls: "ok" | "warn" | "dim"; known: boolean } {
  if (!ccu || typeof ccu !== "object") {
    return {
      text: "未知",
      cls: hasActiveInstances ? "warn" : "dim",
      known: false,
    };
  }
  const raw = (ccu as { burn_rate_hourly?: unknown }).burn_rate_hourly;
  if (raw === null || raw === undefined || typeof raw === "boolean") {
    return {
      text: "未知",
      cls: hasActiveInstances ? "warn" : "dim",
      known: false,
    };
  }
  const rate = numOrNull(raw);
  if (rate === null || rate < 0) {
    return {
      text: "未知",
      cls: hasActiveInstances ? "warn" : "dim",
      known: false,
    };
  }
  return {
    text: `${rate.toFixed(2)} CU/h`,
    cls: rate > 0 || hasActiveInstances ? "warn" : "ok",
    known: true,
  };
}

/** 格式化 Unix 秒级快照时间戳（#57 缓存快照时间展示） */
export function fmtSnapshotTime(tsSec?: number | null): string {
  const n = numOrNull(tsSec);
  if (n === null || n <= 0) return "未知时间";
  const d = new Date(n * 1000);
  if (Number.isNaN(d.getTime())) return "未知时间";
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}

/**
 * 严格区分实时输出速率与上次请求速率 (#65)：
 * 空闲时 instant_tps === 0 保持显示 0.0 tok/s（空闲），绝不用 last_request_tps 冒充实时输出速率。
 */
export function formatRealtimeTps(opts: {
  instantTps?: unknown;
  lastRequestTps?: unknown;
  peakTps?: unknown;
  activeStreams?: unknown;
  liveStreamTps?: unknown;
}): {
  realtimeTps: number;
  realtimeText: string;
  isIdle: boolean;
  lastRequestTps: number | null;
  lastRequestText: string | null;
  peakTps: number | null;
  activeStreams: number;
} {
  const live = numOrNull(opts.liveStreamTps);
  const inst = live !== null ? Math.max(0, live) : Math.max(0, num(opts.instantTps, 0));
  const streams = Math.max(0, Math.floor(num(opts.activeStreams, 0)));
  const rawLast = numOrNull(opts.lastRequestTps);
  const lastTps = rawLast !== null && rawLast > 0 ? rawLast : null;
  const rawPeak = numOrNull(opts.peakTps);
  const peak = rawPeak !== null && rawPeak > 0 ? rawPeak : null;
  const isIdle = inst <= 0 && streams === 0 && live === null;

  return {
    realtimeTps: inst,
    realtimeText: `${inst.toFixed(1)} tok/s`,
    isIdle,
    lastRequestTps: lastTps,
    lastRequestText: lastTps !== null ? `上次请求: ${lastTps.toFixed(1)} tok/s` : null,
    peakTps: peak,
    activeStreams: streams,
  };
}
