import type {
  ChatMessage,
  CreateVmLlmConfig,
  EnvStatus,
  LlmBlueGreenState,
  Overview,
  RecentSessionView,
  RelayConfig,
  RelayStatus,
  VmView,
} from "./types";
import { num, str } from "./format";

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

/** 支持 NVIDIA CUDA 的 GPU 型号集合 (#68) */
export const NVIDIA_CUDA_ACCELERATORS: ReadonlySet<string> = new Set([
  "T4",
  "L4",
  "G4",
  "A100",
  "H100",
]);

/** 强依赖 NVIDIA CUDA 环境的推理引擎集合 (#68) */
export const CUDA_ONLY_LLM_ENGINES: ReadonlySet<string> = new Set([
  "llama.cpp",
  "vllm",
  "transformers",
]);

export const MIN_RELAY_SESSION_HOURS = 0.1;
export const MAX_RELAY_SESSION_HOURS = 23.5;

export function isSingleShapeAccelerator(accelerator: string): boolean {
  const norm = String(accelerator ?? "").trim().toUpperCase();
  return SINGLE_SHAPE_ACCELERATORS.has(norm);
}

export function isNvidiaCudaAccelerator(accelerator: string): boolean {
  const norm = String(accelerator ?? "").trim().toUpperCase();
  return NVIDIA_CUDA_ACCELERATORS.has(norm);
}

export function isLlmEngineCompatibleWithAccelerator(
  engine: string,
  accelerator: string,
): boolean {
  const normEng = String(engine ?? "").trim().toLowerCase();
  if (!CUDA_ONLY_LLM_ENGINES.has(normEng)) return true;
  return isNvidiaCudaAccelerator(accelerator);
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
 * 判断某大模型实例是否已真正就绪可发起对话请求 (#64)：
 * - 必须在线 (`on_server === true`) 且不在开机/后台部署/销毁阶段
 * - 若 `role === "warming"`，或 `online === false` / `backend_healthy === false`，一律视为预热/未就绪
 * - 若无边车遥测（如手动通过 SSH 启动推理引擎），需检测到活跃 `llm_process` 且无 `deploy_error`
 */
export function isLlmVmReady(vm: VmView | null | undefined): boolean {
  if (!vm || !vm.on_server) return false;
  if (vm.busy === "creating" || vm.busy === "deploying_llm" || vm.busy === "deleting") {
    return false;
  }
  const llm = vm.telemetry?.llm;
  if (llm) {
    if (str(llm.role).trim().toLowerCase() === "warming") {
      return false;
    }
    if (llm.online === false || llm.backend_healthy === false) {
      return false;
    }
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

function getExplicitSlotSessionName(
  bg: LlmBlueGreenState | null | undefined,
  slot: "blue" | "green",
): string {
  if (!bg) return "";
  const rawSlotEntry = bg.slots?.[slot] as unknown;
  if (typeof rawSlotEntry === "string" && rawSlotEntry.trim()) {
    return rawSlotEntry.trim();
  }
  if (rawSlotEntry && typeof rawSlotEntry === "object") {
    const obj = rawSlotEntry as { session_name?: unknown; name?: unknown };
    const fromSlots = str(obj.session_name).trim() || str(obj.name).trim();
    if (fromSlots) return fromSlots;
  }
  const direct = slot === "blue" ? str(bg.blue_session).trim() : str(bg.green_session).trim();
  if (direct) return direct;
  const activeSlotNorm = str(bg.active_slot).trim().toLowerCase();
  if (activeSlotNorm === slot && str(bg.active_session).trim()) {
    return str(bg.active_session).trim();
  }
  const standbySlotNorm = str(bg.standby_slot).trim().toLowerCase();
  if (standbySlotNorm === slot && str(bg.standby_session).trim()) {
    return str(bg.standby_session).trim();
  }
  return "";
}

function belongsToOtherPrefixSlotConvention(vmName: string, prefix: string): boolean {
  const m = vmName.match(/^(.+)-(blue|green)$/i);
  if (!m) return false;
  return m[1].toLowerCase() !== prefix.toLowerCase();
}

/**
 * 解析实例所属蓝/绿槽位（#48）：优先匹配 `llm_bluegreen` 状态记录的真实会话名，其次匹配遥测/API 槽位，最后回退后缀。
 */
export function resolveVmSlot(
  vm: VmView | null | undefined,
  bg?: LlmBlueGreenState | null,
): "blue" | "green" | null {
  if (!vm) return null;
  const blueSess = getExplicitSlotSessionName(bg, "blue");
  if (blueSess && vm.name === blueSess) return "blue";
  const greenSess = getExplicitSlotSessionName(bg, "green");
  if (greenSess && vm.name === greenSess) return "green";

  const activeSess = str(bg?.active_session).trim();
  const activeSlot = str(bg?.active_slot).trim().toLowerCase();
  if (activeSess && vm.name === activeSess && (activeSlot === "blue" || activeSlot === "green")) {
    return activeSlot;
  }

  const telSlot = str(vm.telemetry?.llm?.slot).trim().toLowerCase();
  if (telSlot === "blue" || telSlot === "green") return telSlot;

  const apiSlot = str(vm.llm_api?.slot).trim().toLowerCase();
  if (apiSlot === "blue" || apiSlot === "green") return apiSlot;

  const prefix = (str(bg?.session_prefix, "llm") || "llm").trim();
  if (vm.name === `${prefix}-blue` || vm.name.toLowerCase().endsWith("-blue")) return "blue";
  if (vm.name === `${prefix}-green` || vm.name.toLowerCase().endsWith("-green")) return "green";
  return null;
}

/**
 * 解析蓝绿集群状态 (#48)：
 * 1. 优先匹配 `bg.slots[slot].session_name` / `blue_session` / `green_session` / `active_session` 真实会话名（兼容自动命名与任意自定义会话名）
 * 2. 其次匹配 `vm.telemetry?.llm?.slot`（同时隔离不同 `<otherPrefix>-blue/green` 集群，保持 H-4 隔离性）
 * 3. 最后回退 `${prefix}-${slot}` 或（无 bg 时）`-${slot}` 后缀
 */
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
  const onlineOrKnownVms = Array.isArray(vms) ? vms : [];

  const findSlotVm = (slot: "blue" | "green"): VmView | undefined => {
    const explicitSession = getExplicitSlotSessionName(bg, slot);
    if (explicitSession) {
      const exact = onlineOrKnownVms.find((v) => v.name === explicitSession);
      if (exact) return exact;
      const slotEndpoint = str(bg?.slots?.[slot]?.endpoint).trim();
      if (slotEndpoint) {
        const byEp = onlineOrKnownVms.find((v) => v.endpoint && v.endpoint === slotEndpoint);
        if (byEp) return byEp;
      }
      // 显式指定了该槽位真实会话名但当前不在线时，不串槽劫持无关实例
      return onlineOrKnownVms.find((v) => v.name === `${prefix}-${slot}`);
    }

    const slotEndpoint = str(bg?.slots?.[slot]?.endpoint).trim();
    if (slotEndpoint) {
      const byEp = onlineOrKnownVms.find((v) => v.endpoint && v.endpoint === slotEndpoint);
      if (byEp) return byEp;
    }

    // 其次匹配遥测报告的真实 slot（若存在 bg，排除明确属于其他 <otherPrefix>-blue/green 的跨集群实例）
    const byTelemetrySlot = onlineOrKnownVms.find((v) => {
      if (!isLlmVm(v)) return false;
      const vSlot = str(v.telemetry?.llm?.slot).trim().toLowerCase();
      if (vSlot !== slot) return false;
      if (bg && belongsToOtherPrefixSlotConvention(v.name, prefix)) return false;
      return true;
    });
    if (byTelemetrySlot) return byTelemetrySlot;

    // 最后回退 `${prefix}-${slot}` 或（无 bg 时）`-${slot}` 后缀
    const byPrefixConvention = onlineOrKnownVms.find((v) => v.name === `${prefix}-${slot}`);
    if (byPrefixConvention) return byPrefixConvention;

    if (!bg) {
      return onlineOrKnownVms.find(
        (v) => isLlmVm(v) && v.name.toLowerCase().endsWith(`-${slot}`),
      );
    }
    return undefined;
  };

  let blueVm = findSlotVm("blue");
  let greenVm = findSlotVm("green");

  const defaultSlot =
    str(greenVm?.telemetry?.llm?.role).toLowerCase() === "active"
      ? "green"
      : "blue";
  const rawActive = str(bg?.active_slot, defaultSlot).toLowerCase();
  const activeSlot: "blue" | "green" = rawActive === "green" ? "green" : "blue";

  const explicitActiveSess = str(bg?.active_session).trim();
  const explicitActiveVm = explicitActiveSess
    ? onlineOrKnownVms.find((v) => v.name === explicitActiveSess)
    : undefined;

  if (explicitActiveVm) {
    if (activeSlot === "blue" && !blueVm) blueVm = explicitActiveVm;
    if (activeSlot === "green" && !greenVm) greenVm = explicitActiveVm;
  }

  const activeVm = explicitActiveVm ?? (activeSlot === "green" ? greenVm : blueVm);
  const isSlotReady = (vm: VmView | undefined) => isLlmVmReady(vm);

  const clusterState: "online" | "starting" | "offline" =
    isSlotReady(activeVm) || isSlotReady(blueVm) || isSlotReady(greenVm)
      ? "online"
      : activeVm?.on_server || blueVm?.on_server || greenVm?.on_server
        ? "starting"
        : "offline";

  const anyOnlineVm = hasAnyReadyLlmVm(onlineOrKnownVms);

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

/**
 * 格式化对话窗口下拉框中的实例选项文案 (#64)：
 * 对预热中 / 后端未就绪的实例明确标注“（预热中，暂不可对话）”。
 */
export function formatChatOptionLabel(
  vm: VmView,
  bg?: LlmBlueGreenState | null,
): string {
  const sl = resolveVmSlot(vm, bg) ?? str(vm.telemetry?.llm?.slot);
  const role = str(vm.telemetry?.llm?.role);
  const base = `🖥️ ${vm.name} (${vm.accelerator}${sl ? ` · ${sl.toUpperCase()}` : ""}${role ? `/${role}` : ""})`;
  if (vm.deploy_error) {
    return `${base}（部署失败，暂不可对话）`;
  }
  if (!isLlmVmReady(vm)) {
    return `${base}（预热中，暂不可对话）`;
  }
  return base;
}

/**
 * 解析对话弹窗的目标节点状态、可用性与实际提交参数 (#48, #64)：
 * - 当原选定的具体 `selectedTarget` 从在线列表中消失时，标记 `targetDisappeared = true` 并同步切回 `"auto"`
 * - 拒绝向预热中 / 未就绪的实例发送请求（`canSend = false`）
 * - 自动路由或直连时确保 `submitTargetVm` 与 `submitModel` 精确对应目标真实会话
 */
export function resolveChatTargetState(
  selectedTarget: string,
  bg: LlmBlueGreenState | null | undefined,
  vms: VmView[],
): {
  llmVms: VmView[];
  effectiveSelectValue: string;
  targetDisappeared: boolean;
  disappearedVmName: string | null;
  resolvedVm: VmView | undefined;
  resolvedSlot: string | undefined;
  submitTargetVm: string | null;
  submitModel: string | null;
  canSend: boolean;
  disabledReason: string | null;
} {
  const allVms = Array.isArray(vms) ? vms : [];
  const matchedLlm = allVms.filter(isLlmVm);
  const llmVms = matchedLlm.length > 0 ? matchedLlm : allVms.filter((v) => v.on_server);

  const wanted = (selectedTarget || "auto").trim();
  const isAutoRequested = wanted === "" || wanted.toLowerCase() === "auto";
  const directExists = !isAutoRequested && llmVms.some((v) => v.name === wanted);
  const targetDisappeared = !isAutoRequested && !directExists;
  const disappearedVmName = targetDisappeared ? wanted : null;
  const effectiveSelectValue = isAutoRequested || targetDisappeared ? "auto" : wanted;

  const bgStatus = resolveBlueGreenStatus(bg, allVms);

  let resolvedVm: VmView | undefined;
  if (effectiveSelectValue === "auto") {
    // 优先选择蓝绿活跃槽位且已就绪的真实会话实例，绝不把流量路由到 warming 预热节点
    const candidates: (VmView | undefined)[] = [
      bgStatus.activeVm,
      bgStatus.activeSlot === "blue" ? bgStatus.greenVm : bgStatus.blueVm,
      llmVms.find(
        (v) =>
          isLlmVmReady(v) &&
          str(v.telemetry?.llm?.role, "active").toLowerCase() === "active",
      ),
      llmVms.find((v) => isLlmVmReady(v)),
    ];
    resolvedVm =
      candidates.find((v) => v && isLlmVmReady(v)) ??
      bgStatus.activeVm ??
      bgStatus.blueVm ??
      bgStatus.greenVm ??
      llmVms[0];
  } else {
    resolvedVm = llmVms.find((v) => v.name === effectiveSelectValue);
  }

  const slotFromVm = resolveVmSlot(resolvedVm, bg);
  const resolvedSlot =
    (slotFromVm || str(resolvedVm?.telemetry?.llm?.slot).toLowerCase() || "") || undefined;

  const slotMetaModel = resolvedSlot
    ? str(bg?.slots?.[resolvedSlot]?.model_name).trim()
    : "";
  const submitModel =
    str(resolvedVm?.telemetry?.llm?.model_name).trim() ||
    str(resolvedVm?.llm_api?.model_name).trim() ||
    slotMetaModel ||
    str(bg?.model_name).trim() ||
    null;

  const ready = isLlmVmReady(resolvedVm);
  if (!resolvedVm) {
    return {
      llmVms,
      effectiveSelectValue,
      targetDisappeared,
      disappearedVmName,
      resolvedVm: undefined,
      resolvedSlot: undefined,
      submitTargetVm: null,
      submitModel,
      canSend: false,
      disabledReason: "当前没有在线的大模型实例，暂不可发送消息",
    };
  }

  if (!ready) {
    const reason = resolvedVm.deploy_error
      ? `节点 [${resolvedVm.name}] 部署失败，暂不可对话`
      : effectiveSelectValue === "auto"
        ? `当前尚无就绪节点（[${resolvedVm.name}] 预热中，暂不可对话）`
        : `所选节点 [${resolvedVm.name}] 正在预热中，暂不可对话`;
    return {
      llmVms,
      effectiveSelectValue,
      targetDisappeared,
      disappearedVmName,
      resolvedVm,
      resolvedSlot,
      submitTargetVm: null,
      submitModel,
      canSend: false,
      disabledReason: reason,
    };
  }

  return {
    llmVms,
    effectiveSelectValue,
    targetDisappeared,
    disappearedVmName,
    resolvedVm,
    resolvedSlot,
    submitTargetVm: resolvedVm.name,
    submitModel,
    canSend: true,
    disabledReason: null,
  };
}

/** 归一化实例名作为云盘接力隔离目录段 (#44) */
export function sanitizeRelayVmSegment(name: string): string {
  const cleaned = String(name ?? "")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || "default";
}

export function normalizeGdriveDir(rawDir: string): string {
  return String(rawDir ?? "").trim().replace(/\/+$/, "");
}

export function isDefaultBareGdriveDir(rawDir: string): boolean {
  return normalizeGdriveDir(rawDir) === "gdrive:colab_relay";
}

/**
 * 计算接力任务的实际云盘隔离路径 (#44)：
 * 默认裸路径 `gdrive:colab_relay` 自动按实例名隔离为 `gdrive:colab_relay/<sanitized_vm_name>`。
 */
export function resolveEffectiveRelayGdriveDir(
  rawDir: string,
  vmName?: string | null,
): string {
  const norm = normalizeGdriveDir(rawDir);
  if (norm === "gdrive:colab_relay") {
    const trimmedName = String(vmName ?? "").trim();
    if (trimmedName) {
      return `gdrive:colab_relay/${sanitizeRelayVmSegment(trimmedName)}`;
    }
    return "gdrive:colab_relay";
  }
  return norm;
}

/**
 * 检查目标云盘目录是否正被另一台运行中的接力实例占用 (#44)。
 */
export function findConflictingRelayVm(
  rawDir: string,
  currentVmName: string | null | undefined,
  vms: VmView[],
): VmView | undefined {
  if (!Array.isArray(vms) || vms.length === 0) return undefined;
  const selfName = String(currentVmName ?? "").trim();
  const targetEffective = resolveEffectiveRelayGdriveDir(rawDir, selfName || null);
  if (!targetEffective || targetEffective === "gdrive:colab_relay") {
    // 若当前尚未填写实例名且使用默认裸路径，后端将按最终自动生成的唯一实例名隔离，不会与其它隔离目录冲突
    return undefined;
  }
  return vms.find((v) => {
    if (!v?.relay?.running) return false;
    if (selfName && v.name === selfName) return false;
    const otherEffective = resolveEffectiveRelayGdriveDir(v.relay.config?.gdrive_dir ?? "", v.name);
    const otherTelemetryDir = normalizeGdriveDir(v.telemetry?.training?.gdrive_dir ?? "");
    return otherEffective === targetEffective || (Boolean(otherTelemetryDir) && otherTelemetryDir === targetEffective);
  });
}

/** 校验接力单棒时长是否在合法的 `0.1 ~ 23.5` 小时范围内 (#54) */
export function isValidRelaySessionHours(hours: unknown): boolean {
  if (hours === null || hours === undefined || typeof hours === "boolean") return false;
  if (typeof hours === "string" && hours.trim() === "") return false;
  const n = typeof hours === "number" ? hours : Number(hours);
  return Number.isFinite(n) && n >= MIN_RELAY_SESSION_HOURS && n <= MAX_RELAY_SESSION_HOURS;
}

/**
 * 严格区分“云端训练已完成”与“最终权重已成功下载到本地” (#53)。
 */
function sanitizeRelayStateLabelWhenBillingUnknown(label: string, fallback: string): string {
  const clean = str(label).trim();
  if (!clean) return fallback;
  if (/(停费核验通过|销毁云端实例|已释放|已停止计费|不再产生计费)/.test(clean)) {
    return "本机接力已停止（云端计费状态未知）";
  }
  return clean;
}

export function resolveRelayCompletionView(
  relay: RelayStatus | null | undefined,
  billingUnknown = false,
): {
  isDownloadFailed: boolean;
  isCompletedAndDownloaded: boolean;
  isError: boolean;
  stateLabel: string;
  sub: string;
  detailError: string | null;
  canRetryDownload: boolean;
} {
  if (!relay) {
    return {
      isDownloadFailed: false,
      isCompletedAndDownloaded: false,
      isError: false,
      stateLabel: "未开启接力",
      sub: "开启后自动换棒、定时热备、到期销毁旧实例",
      detailError: null,
      canRetryDownload: false,
    };
  }

  if (relay.running) {
    return {
      isDownloadFailed: false,
      isCompletedAndDownloaded: false,
      isError: false,
      stateLabel: relay.phase_label || "接力运行中",
      sub: `第 ${relay.round || 1} 棒运行中`,
      detailError: null,
      canRetryDownload: false,
    };
  }

  const dlErr = str(relay.download_error).trim();
  const lastErr = str(relay.last_error).trim();
  const isDownloadFailed =
    relay.phase === "download_failed" ||
    Boolean(dlErr) ||
    (relay.training_completed === true && relay.weights_downloaded === false) ||
    (relay.phase === "completed" && relay.weights_downloaded === false);

  if (isDownloadFailed) {
    const reason = dlErr || lastErr || "本地目标文件缺失或下载校验未通过";
    return {
      isDownloadFailed: true,
      isCompletedAndDownloaded: false,
      isError: true,
      stateLabel: "云端训练已完成，但最终权重下载失败",
      sub: `云端训练已完成，但最终权重下载到本地失败：${reason}`,
      detailError: reason,
      canRetryDownload: true,
    };
  }

  const isCompletedAndDownloaded =
    relay.weights_downloaded === true ||
    (relay.phase === "completed" && relay.weights_downloaded !== false && !dlErr);

  if (isCompletedAndDownloaded) {
    const savedAt = str(relay.local_weights_path).trim();
    const rawLabel = relay.phase_label || "训练与回收已完成";
    return {
      isDownloadFailed: false,
      isCompletedAndDownloaded: true,
      isError: false,
      stateLabel: billingUnknown
        ? sanitizeRelayStateLabelWhenBillingUnknown(rawLabel, "训练与回收已完成")
        : rawLabel,
      sub: savedAt
        ? `已从 Google Drive 回收最终 checkpoint（${savedAt}）`
        : "已从 Google Drive 回收最终 checkpoint",
      detailError: null,
      canRetryDownload: false,
    };
  }

  if (relay.phase === "error" || Boolean(lastErr)) {
    const rawLabel = relay.phase_label || "接力异常停止";
    return {
      isDownloadFailed: false,
      isCompletedAndDownloaded: false,
      isError: true,
      stateLabel: billingUnknown
        ? sanitizeRelayStateLabelWhenBillingUnknown(rawLabel, "接力异常停止")
        : rawLabel,
      sub: lastErr || "请查看下方「接力日志」排查原因",
      detailError: lastErr || null,
      canRetryDownload: false,
    };
  }

  const rawIdleLabel = relay.phase_label || "未开启接力";
  return {
    isDownloadFailed: false,
    isCompletedAndDownloaded: false,
    isError: false,
    stateLabel: billingUnknown
      ? sanitizeRelayStateLabelWhenBillingUnknown(rawIdleLabel, "本机接力已停止")
      : rawIdleLabel,
    sub: "开启后自动换棒、定时热备、到期销毁旧实例",
    detailError: null,
    canRetryDownload: false,
  };
}

/**
 * 创建实例时的凭证安全守卫 (#43, #75)：
 * - 引导命令统一使用有效的 `colab auth` 与 `colab status`
 * - 当 `!env.token_ok` 且选择 `adc` 时，阻止盲目分配无法监控和无法核验停费的计费实例
 */
export function checkCreateVmAuthGuard(
  authMode: "oauth2" | "adc",
  env?: Pick<EnvStatus, "token_ok" | "adc_ok"> | null,
): {
  blocked: boolean;
  level: "none" | "info" | "warn" | "danger";
  message: string | null;
} {
  if (!env) {
    if (authMode === "adc") {
      return {
        blocked: false,
        level: "warn",
        message:
          "提示：仪表盘实时监控与关机停费核验依赖 Colab OAuth 授权（请确保已执行 colab auth 并可通过 colab status 验证）。",
      };
    }
    return { blocked: false, level: "none", message: null };
  }

  if (!env.token_ok && authMode === "adc") {
    return {
      blocked: true,
      level: "danger",
      message:
        "已阻止分配：本机尚未完成 Colab OAuth 授权，仅使用 ADC 凭证创建的实例将无法通过仪表盘监控，也无法调用服务端列表核验停费！请先在终端执行 `colab auth` 完成授权（可用 `colab status` 检查）后再创建。",
    };
  }

  if (!env.token_ok) {
    return {
      blocked: false,
      level: "warn",
      message:
        "检测到本机尚未完成 Colab OAuth 登录，请先在终端运行 `colab auth` 完成授权（可通过 `colab status` 查看登录状态）。",
    };
  }

  if (authMode === "adc") {
    const adcMissing = env.adc_ok === false;
    return {
      blocked: false,
      level: "warn",
      message: adcMissing
        ? "未检测到本机 Google Cloud ADC 凭证文件，且仪表盘监控与关机停费核验仍依赖 Colab OAuth 授权（`colab auth`）。建议改用 OAuth2 账号模式。"
        : "注意：实例将以 ADC 凭证分配，但仪表盘实时监控与关机停费核验仍依赖本机 Colab OAuth 授权（`colab auth` / `colab status`）。",
    };
  }

  return { blocked: false, level: "none", message: null };
}

/**
 * 判断开机配置是否真正依赖 Google Drive / `rclone.conf` (#67)。
 */
export function requiresGdriveConfig(opts: {
  isDeployExisting?: boolean;
  bootMode: "general" | "llm" | "relay";
  syncRclone: boolean;
  mountGdrive: boolean;
  gdriveModelPath?: string | null;
}): boolean {
  const hasModelGdrive = Boolean(String(opts.gdriveModelPath ?? "").trim());
  if (opts.isDeployExisting) {
    return hasModelGdrive;
  }
  if (opts.bootMode === "relay") return true;
  if (opts.syncRclone || opts.mountGdrive) return true;
  if (opts.bootMode === "llm" && hasModelGdrive) return true;
  return false;
}

export interface RestoredSessionPreset {
  name: string;
  accelerator: string;
  high_mem: boolean;
  boot_mode: "general" | "llm" | "relay";
  workspace_path?: string;
  llm_slot?: "blue" | "green" | "standalone";
  llm_model?: string;
  llm_engine?: string;
  llm_config?: CreateVmLlmConfig | null;
  relay_config?: RelayConfig | null;
  unverified_history_profile?: boolean;
}

/**
 * 从历史会话构造重新开机预设 (#74)：
 * - `profile_saved === true` 时精确还原原 `boot_mode`、`llm_config`、`relay_config`
 * - `!profile_saved` 时绝不凭 `llm-` 名称前缀猜测模式，保持 `"general"` 并附带确认提示标记
 */
export function buildPresetFromRecentSession(r: RecentSessionView): RestoredSessionPreset {
  const acc = (r.accelerator || "T4").trim().toUpperCase() || "T4";
  const highMem = resolveEffectiveHighMem(acc, r.machine_shape === "HIGH_RAM");

  if (r.profile_saved) {
    const mode: "general" | "llm" | "relay" =
      r.boot_mode === "llm" || r.boot_mode === "relay" ? r.boot_mode : "general";
    const rawSlot = str(r.llm_config?.slot).trim().toLowerCase();
    const llmSlot: "blue" | "green" | "standalone" | undefined =
      rawSlot === "blue" || rawSlot === "green" || rawSlot === "standalone"
        ? rawSlot
        : undefined;
    return {
      name: r.session_name,
      accelerator: acc,
      high_mem:
        mode === "relay" && r.relay_config
          ? resolveEffectiveHighMem(acc, r.relay_config.high_mem)
          : highMem,
      boot_mode: mode,
      workspace_path: r.workspace_path ?? undefined,
      llm_slot: llmSlot,
      llm_model: r.llm_config?.model_name || undefined,
      llm_engine: r.llm_config?.engine || undefined,
      llm_config: r.llm_config ?? null,
      relay_config: r.relay_config ?? null,
      unverified_history_profile: false,
    };
  }

  return {
    name: r.session_name,
    accelerator: acc,
    high_mem: highMem,
    boot_mode: "general",
    workspace_path: r.workspace_path ?? undefined,
    unverified_history_profile: true,
  };
}

/**
 * 无障碍焦点锁定循环索引计算 (#70)：严格在 `[0, total - 1]` 内循环，绝不逃逸到弹窗外背景元素。
 */
export function computeNextFocusTrapIndex(
  total: number,
  currentIndex: number,
  shiftKey: boolean,
): number {
  if (total <= 0) return -1;
  if (shiftKey) {
    if (currentIndex <= 0 || currentIndex >= total) return total - 1;
    return currentIndex - 1;
  }
  if (currentIndex < 0 || currentIndex >= total - 1) return 0;
  return currentIndex + 1;
}

/**
 * 判断总览快照是否因鉴权缺失或超出缓存窗口的列表查询失败而处于“云端计费状态未知”(#78)。
 */
export function isOverviewServerListUnknown(ov: Overview | null | undefined): boolean {
  if (!ov) return true;
  if (ov.server_list_known === false) return true;
  if (ov.server_list_known === undefined && Boolean(ov.error) && !ov.is_stale) {
    return true;
  }
  return false;
}

/**
 * 合并总览快照 (#82)：
 * - 严格拒绝时间更早或代次 `seq` 更旧的迟到快照（包括 `is_stale=true` 的旧快照）回退覆盖已收到的更新实例数据；
 * - 若新快照生成时间更晚 (`next.timestamp > prev.timestamp`) 但因查询失败携带较旧的 `last_ok_timestamp`，
 *   仅将查询错误作为独立告警状态附加到当前最新数据上，绝不清空或回退上一帧在线实例列表。
 */
export function mergeOverviewSnapshot(
  prev: Overview | null,
  next: Overview | null | undefined,
): Overview | null {
  if (!next) return prev;
  if (!prev) return next;

  const prevSeq = typeof prev.seq === "number" && prev.seq > 0 ? prev.seq : 0;
  const nextSeq = typeof next.seq === "number" && next.seq > 0 ? next.seq : 0;
  const prevTs = typeof prev.timestamp === "number" ? prev.timestamp : 0;
  const nextTs = typeof next.timestamp === "number" ? next.timestamp : 0;
  const prevOk = prev.last_ok_timestamp ?? prevTs;
  const nextOk = next.last_ok_timestamp ?? nextTs;

  // 当后端单调代次 seq 均存在时，严格以 seq 判定快照先后顺序（避免 60s 缓存回退将 timestamp 置为 prev_ok_ts 后被误丢弃）
  if (prevSeq > 0 && nextSeq > 0) {
    if (nextSeq < prevSeq) {
      return prev;
    }
    if (next.server_list_known === false) {
      return next;
    }
    if (nextOk < prevOk) {
      if (next.is_stale || next.stale_reason || next.error) {
        return {
          ...prev,
          seq: nextSeq,
          is_stale: true,
          stale_reason: next.stale_reason || next.error || "最新列表查询失败",
        };
      }
      return prev;
    }
    return next;
  }

  if (nextTs < prevTs) {
    return prev;
  }

  if (next.server_list_known === false) {
    return next;
  }

  if (nextOk < prevOk) {
    if (nextTs > prevTs && (next.is_stale || next.stale_reason || next.error)) {
      return {
        ...prev,
        timestamp: nextTs,
        seq: nextSeq || prevSeq || undefined,
        is_stale: true,
        stale_reason: next.stale_reason || next.error || "最新列表查询失败",
      };
    }
    return prev;
  }

  return next;
}

/**
 * 计算实例卡片顶部状态徽章文案与色调 (#65, #73, #76, #78)：
 * - `vm.relay?.running === true` 时必须反映接力活跃阶段，绝不显示“空闲”
 * - `llm` 空闲时只使用 `instant_tps`（0.0 tok/s 空闲），绝不把 `last_request_tps` 标为实时速率
 * - `!vm.on_server && vm.relay` 离线接力记录卡片：若服务端列表查询失败则保留“计费状态未知”，仅在确认释放后显示已释放/已下线
 */
export function resolveVmStatusPillSpec(
  vm: VmView,
  hasActiveProc: boolean,
  serverStateUnknown = false,
): { cls: string; pulse: boolean; text: string; title?: string } {
  const t = vm.telemetry;
  const llm = t?.llm;
  const tr = t?.training;
  const billingUnknown = Boolean(vm.server_state_unknown || serverStateUnknown);

  if (vm.busy === "creating") {
    return { cls: "pill-amber", pulse: true, text: "开机分配中…" };
  }
  if (vm.busy === "deploying_llm") {
    return { cls: "pill-amber", pulse: true, text: "部署大模型中…" };
  }
  if (vm.deploy_error) {
    return { cls: "pill-red", pulse: false, text: "部署失败", title: vm.deploy_error };
  }
  if (vm.busy === "probing") {
    return { cls: "pill-blue", pulse: true, text: "连接中" };
  }
  if (llm && (llm.online || llm.backend_healthy) && vm.on_server) {
    const role = str(llm.role).toLowerCase();
    const instTps = Math.max(0, num(llm.instant_tps, 0));
    if (role === "warming") {
      return { cls: "pill-amber", pulse: true, text: "LLM · 预热中" };
    }
    const cls = role === "draining" ? "pill-gray" : "pill-green";
    if (instTps > 0 || num(llm.active_streams, 0) > 0) {
      return { cls, pulse: true, text: `LLM · ${instTps.toFixed(1)} tok/s` };
    }
    return { cls, pulse: false, text: "LLM · 0.0 tok/s (空闲)" };
  }
  if (tr && tr.status && vm.on_server) {
    const isDone = tr.status === "completed" && !vm.relay?.running;
    if (isDone) {
      return { cls: "pill-green", pulse: false, text: "训练完成" };
    }
    if (tr.status !== "completed") {
      return {
        cls: "pill-blue",
        pulse: true,
        text: `训练中 · Epoch ${tr.current_epoch ?? 0}/${tr.total_epochs ?? "?"}`,
      };
    }
  }
  if (vm.relay?.running) {
    return {
      cls: "pill-blue",
      pulse: true,
      text: vm.relay.phase_label || "接力运行中",
    };
  }
  if (!vm.on_server && vm.relay) {
    const view = resolveRelayCompletionView(vm.relay, billingUnknown);
    if (billingUnknown) {
      if (view.isError) {
        return {
          cls: "pill-amber",
          pulse: false,
          text: "接力失败 · 计费状态未知",
          title: "服务端实例列表查询失败，暂无法确认云端实例是否已停费",
        };
      }
      if (view.isCompletedAndDownloaded || view.isDownloadFailed) {
        return {
          cls: "pill-amber",
          pulse: false,
          text: "接力已结束 · 计费状态未知",
          title: "服务端实例列表查询失败，暂无法确认云端实例是否已停费",
        };
      }
      return {
        cls: "pill-amber",
        pulse: false,
        text: "已停止记录 · 计费状态未知",
        title: "服务端实例列表查询失败，暂无法确认云端实例是否已停费",
      };
    }
    if (view.isDownloadFailed) {
      return { cls: "pill-amber", pulse: false, text: "权重待补下载 · 实例已下线" };
    }
    if (view.isCompletedAndDownloaded) {
      return { cls: "pill-green", pulse: false, text: "接力已完成 · 实例已释放" };
    }
    if (view.isError) {
      return { cls: "pill-red", pulse: false, text: "接力失败 · 实例已下线" };
    }
    return { cls: "pill-gray", pulse: false, text: "已结束记录 · 实例已下线" };
  }
  if (vm.running_task || hasActiveProc) {
    return { cls: "pill-blue", pulse: true, text: "运行中" };
  }
  if (vm.on_server) {
    return { cls: "pill-green", pulse: false, text: "空闲" };
  }
  return { cls: "pill-gray", pulse: false, text: "离线" };
}
