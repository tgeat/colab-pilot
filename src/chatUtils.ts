import type { ChatMessage, LlmBlueGreenState, VmView } from "./types";
import { str } from "./format";

export interface ConversationTurnLike {
  id: string;
  role: "user" | "assistant";
  content: string;
  error?: string | null;
  cancelled?: boolean;
}

/**
 * Google Colab 仅提供固定单一内存规格（拒绝 `&shape=hm`，传 `--high-mem` 会触发 `/tun/m/assign` HTTP 400）的加速器集合。
 * 仅 T4、A100、CPU 支持 Standard / High-RAM 双规格切换。
 */
export const SINGLE_SHAPE_ACCELERATORS: ReadonlySet<string> = new Set([
  "G4",
  "H100",
  "L4",
  "V5E1",
  "V6E1",
]);

export function isSingleShapeAccelerator(accelerator: string): boolean {
  const norm = String(accelerator ?? "").trim().toUpperCase();
  return SINGLE_SHAPE_ACCELERATORS.has(norm);
}

/**
 * 计算发往后端 / `colab new` 的有效 `high_mem` 布尔值：
 * 单规格型号（G4 / H100 / L4 / V5E1 / V6E1）必须强制为 `false`，杜绝附加 `shape=hm` 导致 HTTP 400。
 */
export function resolveEffectiveHighMem(accelerator: string, highMem: boolean): boolean {
  if (isSingleShapeAccelerator(accelerator)) return false;
  return Boolean(highMem);
}

/** 判断某实例是否属于大模型工作负载实例（含正在部署或已分配专属 API Key 的实例） */
export function isLlmVm(vm: VmView): boolean {
  if (!vm.on_server) return false;
  if (vm.telemetry?.llm) return true;
  if (vm.telemetry?.threads?.llm_process) return true;
  if (vm.llm_api) return true;
  if (vm.busy === "deploying_llm") return true;
  if (vm.name.startsWith("llm-")) return true;
  return false;
}

/**
 * 判断某大模型实例是否已真正就绪可发起对话请求：
 * - 必须在线 (`on_server === true`) 且不在开机/后台部署/销毁阶段
 * - 若已有 `telemetry.llm`，需 `online` 或 `backend_healthy` 为真
 * - 若无边车遥测（如手动通过 SSH 启动推理引擎），需检测到活跃 `llm_process` 且无 `deploy_error`
 */
export function isLlmVmReady(vm: VmView | null | undefined): boolean {
  if (!vm || !vm.on_server) return false;
  if (vm.busy === "creating" || vm.busy === "deploying_llm" || vm.busy === "deleting") {
    return false;
  }
  const llm = vm.telemetry?.llm;
  if (llm) {
    return Boolean(llm.online || llm.backend_healthy);
  }
  if (vm.deploy_error) return false;
  return Boolean(vm.telemetry?.threads?.llm_process);
}

/** 是否存在至少一台在线且健康就绪的大模型实例（用于顶栏与全局对话入口控权） */
export function hasAnyReadyLlmVm(vms: VmView[]): boolean {
  return Array.isArray(vms) && vms.some((v) => isLlmVmReady(v));
}

/**
 * 仅放行显式安全协议（https: / http: / mailto:），彻底拦截 javascript: / data: / vbscript: / file: 等危险协议（H-1）。
 */
export function isSafeMarkdownUrl(rawUrl: string): boolean {
  const trimmed = rawUrl.trim();
  if (!trimmed || trimmed.startsWith("//")) return false;
  try {
    const parsed = new URL(trimmed);
    return (
      parsed.protocol === "https:" ||
      parsed.protocol === "http:" ||
      parsed.protocol === "mailto:"
    );
  } catch {
    return false;
  }
}

/**
 * 提取 <think>...</think> 思考链，并抑制流式输出首几个字符 `<th...` 尚未构成完整标签时的正文闪烁（L-5）。
 */
export function extractThinkParts(
  content: string,
  streaming = false,
): { thinkText: string | null; mainText: string } {
  const closedBlocks: string[] = [];
  let rest = content.replace(/<think>([\s\S]*?)<\/think>\s*/gi, (_m, inner: string) => {
    closedBlocks.push(inner.trim());
    return "";
  });

  const openMatch = rest.match(/<think>([\s\S]*)$/i);
  if (openMatch && openMatch.index !== undefined) {
    closedBlocks.push(openMatch[1].trim());
    rest = rest.slice(0, openMatch.index);
  }

  if (closedBlocks.length > 0) {
    return {
      thinkText: closedBlocks.filter(Boolean).join("\n\n"),
      mainText: rest,
    };
  }

  if (streaming) {
    const trimmedStart = content.trimStart().toLowerCase();
    if (trimmedStart.length > 0 && trimmedStart.length < 7 && "<think>".startsWith(trimmedStart)) {
      return { thinkText: "", mainText: "" };
    }
  }
  return { thinkText: null, mainText: content };
}

export function stripThinkBlocks(content: string): string {
  return content.replace(/<think>[\s\S]*?(?:<\/think>|$)/gi, "").trim();
}

/**
 * 拆分 Markdown 围栏代码块，兼容 `c#`、`c++`、`f#`、`ts title="a.ts"` 等任意首行语言/元信息标记（M-1）。
 */
export function parseMarkdownSegments(
  mainText: string,
): { type: "text" | "code"; lang?: string; text: string }[] {
  const segments: { type: "text" | "code"; lang?: string; text: string }[] = [];
  const fenceRegex = /```([^\n`]*)\n?([\s\S]*?)(?:```|$)/g;
  let lastIdx = 0;
  let m: RegExpExecArray | null;

  while ((m = fenceRegex.exec(mainText)) !== null) {
    if (m.index > lastIdx) {
      segments.push({ type: "text", text: mainText.slice(lastIdx, m.index) });
    }
    const rawInfo = (m[1] ?? "").trim();
    const langToken = rawInfo ? rawInfo.split(/\s+/)[0] : "";
    segments.push({
      type: "code",
      lang: langToken || "text",
      text: m[2].replace(/\n$/, ""),
    });
    lastIdx = fenceRegex.lastIndex;
  }
  if (lastIdx < mainText.length) {
    segments.push({ type: "text", text: mainText.slice(lastIdx) });
  }
  return segments;
}

/**
 * 严格成对构造 `user -> assistant` 历史消息（H-9）：
 * 任何失败、中断无输出、孤立的 user/assistant 轮次均整对丢弃，杜绝连续两个 `role: "user"` 导致 vLLM/SGLang 400 锁死对话。
 */
export function buildHistoryMessages(
  turns: ConversationTurnLike[],
  systemPrompt: string,
  newUserContent: string,
  maxPairs = 10,
): ChatMessage[] {
  const msgs: ChatMessage[] = [];
  if (systemPrompt.trim()) {
    msgs.push({ role: "system", content: systemPrompt.trim() });
  }

  const pairs: [ChatMessage, ChatMessage][] = [];
  let pendingUser: string | null = null;

  for (const t of turns) {
    if (t.role === "user") {
      if (!t.error && !t.cancelled && t.content.trim()) {
        pendingUser = t.content.trim();
      } else {
        pendingUser = null;
      }
    } else if (t.role === "assistant") {
      if (pendingUser !== null && !t.error && !t.cancelled) {
        const cleaned = stripThinkBlocks(t.content);
        if (cleaned) {
          pairs.push([
            { role: "user", content: pendingUser },
            { role: "assistant", content: cleaned },
          ]);
        }
      }
      pendingUser = null;
    }
  }

  for (const [u, a] of pairs.slice(-Math.max(0, maxPairs))) {
    msgs.push(u, a);
  }
  msgs.push({ role: "user", content: newUserContent.trim() });
  return msgs;
}

export function resolveBlueGreenStatus(
  bg: LlmBlueGreenState | null | undefined,
  vms: VmView[],
): {
  prefix: string;
  blueVm: VmView | undefined;
  greenVm: VmView | undefined;
  activeSlot: "blue" | "green";
  activeVm: VmView | undefined;
  clusterState: "online" | "starting" | "offline";
  anyOnlineVm: boolean;
} {
  const prefix = (str(bg?.session_prefix, "llm") || "llm").trim();
  const findSlotVm = (slot: "blue" | "green") =>
    vms.find((v) => v.name === `${prefix}-${slot}`) ??
    vms.find(
      (v) =>
        !bg && isLlmVm(v) && str(v.telemetry?.llm?.slot).toLowerCase() === slot,
    );

  const blueVm = findSlotVm("blue");
  const greenVm = findSlotVm("green");

  const defaultSlot =
    str(greenVm?.telemetry?.llm?.role).toLowerCase() === "active"
      ? "green"
      : "blue";
  const rawActive = str(bg?.active_slot, defaultSlot).toLowerCase();
  const activeSlot: "blue" | "green" = rawActive === "green" ? "green" : "blue";
  const activeVm = activeSlot === "green" ? greenVm : blueVm;

  const isSlotReady = (vm: VmView | undefined) => isLlmVmReady(vm);

  const clusterState: "online" | "starting" | "offline" =
    isSlotReady(activeVm) || isSlotReady(blueVm) || isSlotReady(greenVm)
      ? "online"
      : activeVm?.on_server || blueVm?.on_server || greenVm?.on_server
        ? "starting"
        : "offline";

  const anyOnlineVm = hasAnyReadyLlmVm(vms);

  return {
    prefix,
    blueVm,
    greenVm,
    activeSlot,
    activeVm,
    clusterState,
    anyOnlineVm,
  };
}
