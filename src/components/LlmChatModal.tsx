import { type ReactNode, memo, useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  Check,
  Copy,
  Cpu,
  Gauge,
  GitCompare,
  Loader2,
  Send,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  User,
  X,
  Zap,
} from "lucide-react";
import { api, errText } from "../api";
import type { LlmChatStreamEvent, Overview, VmView } from "../types";
import { formatRealtimeTps, num, numOrNull, str } from "../format";
import {
  buildHistoryMessages,
  extractThinkParts,
  formatChatOptionLabel,
  hasAnyReadyLlmVm,
  isLlmVm,
  isLlmVmReady,
  isSafeMarkdownUrl,
  parseMarkdownSegments,
  resolveChatTargetState,
} from "../chatUtils";
import { ColabIcon, useModalFocusTrap, useToast } from "./ui";

export { isLlmVm };

export interface ConversationTurn {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
  vmName?: string;
  slot?: string | null;
  model?: string;
  ttftMs?: number | null;
  tps?: number | null;
  tokens?: number | null;
  elapsedMs?: number | null;
  transport?: string | null;
  streaming?: boolean;
  cancelled?: boolean;
  error?: string | null;
}

const DEFAULT_SYSTEM_PROMPT =
  "你是由 Colab Pilot 托管在 Google Colab 云端 GPU 上的大语言模型助手，回答准确、精炼、专业。请默认使用中文回答。";

const QUICK_PROMPTS = [
  "👋 请简要介绍一下你自己，并说明当前推理节点的状态。",
  "🐍 用 Python 实现一个带过期时间与线程安全锁的 LRU Cache。",
  "🚀 请详细解析大语言模型蓝绿热切换 (Make-Before-Break) 的核心原理。",
  "⚡ 写一段 PyTorch 混合精度训练 (AMP) 与断点续训的标准模板代码。",
];

// ───────────────────────────── 轻量级流式 Markdown 渲染器 ─────────────────────────────

function renderInlineMarkdown(text: string, keyPrefix = "in"): ReactNode[] {
  const nodes: ReactNode[] = [];
  // 匹配行内代码 `code`、链接 [text](url)、粗斜体 ***text***、粗体 **text**、删除线 ~~text~~、斜体 *text*
  const tokenRegex =
    /(`[^`\n]+`|\[[^\]]+\]\([^)\s]+\)|\*\*\*[^*\n]+\*\*\*|\*\*[^*\n]+\*\*|__[^_\n]+__|~~[^~\n]+~~|\*[^*\s][^*\n]*\*)/g;

  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let idx = 0;

  while ((match = tokenRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    const raw = match[0];
    const k = `${keyPrefix}-${idx++}`;

    if (raw.startsWith("`") && raw.endsWith("`")) {
      nodes.push(
        <code key={k} className="md-inline-code">
          {raw.slice(1, -1)}
        </code>,
      );
    } else if (raw.startsWith("[") && raw.includes("](") && raw.endsWith(")")) {
      const splitPos = raw.indexOf("](");
      const label = raw.slice(1, splitPos);
      const url = raw.slice(splitPos + 2, -1);
      if (isSafeMarkdownUrl(url)) {
        nodes.push(
          <a key={k} href={url.trim()} target="_blank" rel="noopener noreferrer" className="md-link">
            {renderInlineMarkdown(label, k)}
          </a>,
        );
      } else {
        nodes.push(
          <span key={k} className="dim" title="已拦截非 http/https/mailto 链接">
            {renderInlineMarkdown(label, k)}
          </span>,
        );
      }
    } else if (raw.startsWith("***") && raw.endsWith("***")) {
      nodes.push(
        <strong key={k}>
          <em>{renderInlineMarkdown(raw.slice(3, -3), k)}</em>
        </strong>,
      );
    } else if (
      (raw.startsWith("**") && raw.endsWith("**")) ||
      (raw.startsWith("__") && raw.endsWith("__"))
    ) {
      nodes.push(<strong key={k}>{renderInlineMarkdown(raw.slice(2, -2), k)}</strong>);
    } else if (raw.startsWith("~~") && raw.endsWith("~~")) {
      nodes.push(<del key={k}>{renderInlineMarkdown(raw.slice(2, -2), k)}</del>);
    } else if (raw.startsWith("*") && raw.endsWith("*")) {
      nodes.push(<em key={k}>{renderInlineMarkdown(raw.slice(1, -1), k)}</em>);
    } else {
      nodes.push(raw);
    }
    lastIndex = tokenRegex.lastIndex;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

function CodeBlockCard({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const onCopy = () => {
    navigator.clipboard
      ?.writeText(code)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => {});
  };

  return (
    <div className="md-code-card">
      <div className="md-code-head">
        <span className="md-code-lang">{lang || "code"}</span>
        <button type="button" className="md-code-copy" onClick={onCopy} title="复制代码">
          {copied ? (
            <>
              <Check size={12} color="var(--success)" /> 已复制
            </>
          ) : (
            <>
              <Copy size={12} /> 复制代码
            </>
          )}
        </button>
      </div>
      <pre className="md-pre">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function renderTextBlocks(markdown: string, keyPrefix: string): ReactNode[] {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let i = 0;
  let bIdx = 0;

  const isTableSep = (line: string) =>
    /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*(?::?-{3,}:?\s*)?\|?\s*$/.test(line);

  const splitTableCells = (line: string) => {
    let trimmed = line.trim();
    if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
    return trimmed.split("|").map((c) => c.trim());
  };

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i++;
      continue;
    }

    // 1. 分割线 --- / ***
    if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
      blocks.push(<hr key={`${keyPrefix}-hr-${bIdx++}`} className="md-hr" />);
      i++;
      continue;
    }

    // 2. 标题 # ~ ######（兼容流式中间态 "## "：标题文字尚未到达时也必须被消费，否则会死循环）
    const headingMatch = line.match(/^\s*(#{1,6})(?:\s+(.*))?$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const content = renderInlineMarkdown((headingMatch[2] ?? "").trim(), `${keyPrefix}-h-${bIdx}`);
      const k = `${keyPrefix}-h-${bIdx++}`;
      if (level === 1) blocks.push(<h1 key={k} className="md-h1">{content}</h1>);
      else if (level === 2) blocks.push(<h2 key={k} className="md-h2">{content}</h2>);
      else if (level === 3) blocks.push(<h3 key={k} className="md-h3">{content}</h3>);
      else blocks.push(<h4 key={k} className="md-h4">{content}</h4>);
      i++;
      continue;
    }

    // 3. Markdown 表格
    if (line.includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const headers = splitTableCells(lines[i]);
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) {
        rows.push(splitTableCells(lines[i]));
        i++;
      }
      const k = `${keyPrefix}-tbl-${bIdx++}`;
      blocks.push(
        <div key={k} className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>
                {headers.map((h, hIdx) => (
                  <th key={hIdx}>{renderInlineMarkdown(h, `${k}-th-${hIdx}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, rIdx) => (
                <tr key={rIdx}>
                  {headers.map((_, cIdx) => (
                    <td key={cIdx}>
                      {renderInlineMarkdown(r[cIdx] ?? "", `${k}-td-${rIdx}-${cIdx}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      );
      continue;
    }

    // 4. 引用块 >
    if (/^\s*>\s?/.test(line)) {
      const quoteLines: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      const k = `${keyPrefix}-q-${bIdx++}`;
      blocks.push(
        <blockquote key={k} className="md-quote">
          {renderInlineMarkdown(quoteLines.join("\n"), k)}
        </blockquote>,
      );
      continue;
    }

    // 5. 无序列表 - / * / +
    if (/^\s*[-*+]\s+/.test(line)) {
      const items: { checked?: boolean; text: string }[] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        const rawItem = lines[i].replace(/^\s*[-*+]\s+/, "");
        const taskMatch = rawItem.match(/^\[([ xX])\]\s+(.*)$/);
        if (taskMatch) {
          items.push({
            checked: taskMatch[1].toLowerCase() === "x",
            text: taskMatch[2],
          });
        } else {
          items.push({ text: rawItem });
        }
        i++;
      }
      const k = `${keyPrefix}-ul-${bIdx++}`;
      blocks.push(
        <ul key={k} className="md-list">
          {items.map((it, idx) => (
            <li key={idx} className={it.checked !== undefined ? "md-task-item" : ""}>
              {it.checked !== undefined && (
                <input type="checkbox" checked={it.checked} readOnly className="md-checkbox" />
              )}
              <span>{renderInlineMarkdown(it.text, `${k}-li-${idx}`)}</span>
            </li>
          ))}
        </ul>,
      );
      continue;
    }

    // 6. 有序列表 1. / 2.
    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      const k = `${keyPrefix}-ol-${bIdx++}`;
      blocks.push(
        <ol key={k} className="md-list md-ol">
          {items.map((it, idx) => (
            <li key={idx}>{renderInlineMarkdown(it, `${k}-oli-${idx}`)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    // 7. 普通段落（合并连续普通文本行）
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^\s*(#{1,6}\s+|[-*+]\s+|\d+\.\s+|>\s?|(?:-{3,}|\*{3,}|_{3,})$)/.test(lines[i]) &&
      !(lines[i].includes("|") && i + 1 < lines.length && isTableSep(lines[i + 1]))
    ) {
      paraLines.push(lines[i]);
      i++;
    }
    // 兜底：任何未被上述分支消费的行都按普通文本处理，保证循环必然前进（杜绝死循环导致 WebView Out of Memory）
    if (paraLines.length === 0) {
      paraLines.push(lines[i]);
      i++;
    }
    const k = `${keyPrefix}-p-${bIdx++}`;
    blocks.push(
      <p key={k} className="md-p">
        {paraLines.map((pl, pIdx) => (
          <span key={pIdx}>
            {pIdx > 0 && <br />}
            {renderInlineMarkdown(pl, `${k}-l-${pIdx}`)}
          </span>
        ))}
      </p>,
    );
  }

  return blocks;
}

// 使用 React.memo：父组件每 1.5s 的概览刷新不会导致已完成的对话被重复解析
const MarkdownContent = memo(function MarkdownContent({
  content,
  streaming,
}: {
  content: string;
  streaming?: boolean;
}) {
  const { thinkText, mainText } = extractThinkParts(content, !!streaming);
  const segments = parseMarkdownSegments(mainText);

  return (
    <div className="md-body">
      {thinkText !== null && (
        <details className="md-think" open={!!streaming && !mainText}>
          <summary>💭 深度思考过程</summary>
          <div className="md-think-body">{thinkText}</div>
        </details>
      )}
      {segments.map((seg, idx) =>
        seg.type === "code" ? (
          <CodeBlockCard key={`code-${idx}`} lang={seg.lang ?? "code"} code={seg.text} />
        ) : (
          renderTextBlocks(seg.text, `seg-${idx}`)
        ),
      )}
      {streaming && <span className="chat-cursor" />}
    </div>
  );
});

// ───────────────────────────── 大模型交互对话弹窗 ─────────────────────────────

const CHAT_STORAGE_KEY = "colab_pilot_chat_turns_v1";
/** 发送给模型的历史问答对上限（System Prompt 始终保留） */
const MAX_HISTORY_PAIRS = 10;

function loadPersistedTurns(): ConversationTurn[] {
  try {
    const raw = window.localStorage.getItem(CHAT_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(
        (t): t is ConversationTurn =>
          !!t && typeof t === "object" && (t.role === "user" || t.role === "assistant"),
      )
      .map((t, idx) => ({
        ...t,
        id: str(t.id) || `restored-${idx}-${Math.random().toString(36).slice(2, 7)}`,
        content: str(t.content),
        createdAt: str(t.createdAt),
        streaming: false,
      }));
  } catch {
    return [];
  }
}

export function LlmChatModal(props: {
  open: boolean;
  initialTarget: string; // "auto" or vm.name
  ov: Overview | null;
  onClose: () => void;
}) {
  const { open, initialTarget, ov, onClose } = props;
  const toast = useToast();

  const [targetVm, setTargetVm] = useState<string>(initialTarget || "auto");
  const [turns, setTurns] = useState<ConversationTurn[]>(() => loadPersistedTurns());
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [maxTokens, setMaxTokens] = useState(2048);
  const [temperature, setTemperature] = useState(0.7);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [liveStreamMetrics, setLiveStreamMetrics] = useState<{
    tps: number | null;
    ttft: number | null;
    tokens: number | null;
    slot: string | null;
    vmName: string | null;
  } | null>(null);

  const activeReqIdRef = useRef<string | null>(null);
  const cancelledReqIdRef = useRef<string | null>(null);
  const stickToBottomRef = useRef<boolean>(true);
  const chatBodyRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const modalRef = useRef<HTMLDivElement | null>(null);
  const titleId = useId();
  const [lostTargetNotice, setLostTargetNotice] = useState<string | null>(null);

  useModalFocusTrap(open, modalRef, inputRef);

  const handleStop = useCallback(() => {
    const reqId = activeReqIdRef.current;
    if (!reqId) return;
    cancelledReqIdRef.current = reqId;
    api.cancelLlmChat(reqId).catch(() => {});
  }, []);

  const handleCloseModal = useCallback(() => {
    if (activeReqIdRef.current) {
      cancelledReqIdRef.current = activeReqIdRef.current;
      api.cancelLlmChat(activeReqIdRef.current).catch(() => {});
    }
    onClose();
  }, [onClose]);

  // 仅在非流式生成期间持久化（避免每个 token 增量都序列化整段历史写入 localStorage）
  useEffect(() => {
    if (sending) return;
    try {
      window.localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(turns.slice(-60)));
    } catch {
      // ignore storage quota errors
    }
  }, [turns, sending]);

  useEffect(() => {
    if (open && initialTarget) {
      setTargetVm(initialTarget);
      setLostTargetNotice(null);
      stickToBottomRef.current = true;
      setTimeout(() => inputRef.current?.focus(), 60);
    }
  }, [open, initialTarget]);

  // 支持 Esc 快捷键：正在生成时停止生成，空闲时关闭窗口
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        if (sending) handleStop();
        else handleCloseModal();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, sending, handleStop, handleCloseModal]);

  const vms = ov?.vms ?? [];
  const bg = ov?.llm_bluegreen ?? null;
  const targetState = useMemo(
    () => resolveChatTargetState(targetVm, bg, vms),
    [targetVm, bg, vms],
  );
  const llmVms = targetState.llmVms;
  const resolvedVm: VmView | undefined = targetState.resolvedVm;
  const anyReadyNode = hasAnyReadyLlmVm(vms);

  // #64：当选定的具体目标实例从在线列表中消失时，同步切回 "auto" 并提示用户，确保下拉框展示值与 IPC 提交一致
  useEffect(() => {
    if (!open) return;
    if (targetState.targetDisappeared && targetState.disappearedVmName) {
      setLostTargetNotice(
        `原选定节点 [${targetState.disappearedVmName}] 已离线或被释放，已自动切回「自动路由活跃节点」`,
      );
      setTargetVm("auto");
    }
  }, [open, targetState.targetDisappeared, targetState.disappearedVmName]);

  const llmTel = resolvedVm?.telemetry?.llm ?? null;
  const displayModel =
    targetState.submitModel ||
    str(resolvedVm?.workspace?.description) ||
    "Colab Cloud LLM";
  const displayEngine =
    str(llmTel?.engine) ||
    str(resolvedVm?.llm_api?.engine) ||
    str(bg?.engine) ||
    "OpenAI-Compatible";
  const displaySlot =
    str(liveStreamMetrics?.slot ?? targetState.resolvedSlot ?? null).toLowerCase() ||
    undefined;

  // #65：严格区分实时输出频率与上次请求速率，空闲时保持 0.0 tok/s (空闲)，绝不把 last_request_tps 冒充为实时速率
  const tpsView = formatRealtimeTps({
    instantTps: llmTel?.instant_tps,
    lastRequestTps: llmTel?.last_request_tps ?? bg?.warmup_tps,
    peakTps: llmTel?.peak_tps,
    activeStreams: llmTel?.active_streams,
    liveStreamTps: sending ? liveStreamMetrics?.tps : null,
  });
  const displayAvgTps = num(llmTel?.avg_tps_60s);
  const displayTtft =
    numOrNull(liveStreamMetrics?.ttft) ??
    numOrNull(llmTel?.ttft_ms) ??
    numOrNull(llmTel?.avg_ttft_ms);
  const displayTotalTokens = num(llmTel?.total_completion_tokens);

  // 智能贴底滚动：仅当用户位于底部附近（80px 内）或窗口刚打开时自动跟随滚动，不打断向上翻阅长回复
  useEffect(() => {
    const el = chatBodyRef.current;
    if (el && stickToBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    }
  }, [turns, open]);

  // 监听后端 Tauri 推送的流式 Token 事件（带卸载竞态保护与异常捕获）
  useEffect(() => {
    let disposed = false;
    const unlistenPromise = listen<LlmChatStreamEvent>("llm-chat-event", (event) => {
      if (disposed) return;
      const ev = event.payload;
      if (!ev || ev.request_id !== activeReqIdRef.current) return;

      if (ev.kind === "start") {
        setLiveStreamMetrics({
          tps: null,
          ttft: null,
          tokens: 0,
          slot: ev.slot,
          vmName: ev.vm_name,
        });
        setTurns((prev) =>
          prev.map((t) =>
            t.id === ev.request_id
              ? {
                  ...t,
                  content: "",
                  vmName: ev.vm_name,
                  slot: ev.slot,
                  model: ev.model,
                  streaming: true,
                }
              : t,
          ),
        );
      } else if (ev.kind === "delta" && ev.delta) {
        setLiveStreamMetrics({
          tps: ev.instant_tps ?? null,
          ttft: ev.ttft_ms ?? null,
          tokens: ev.completion_tokens ?? null,
          slot: ev.slot,
          vmName: ev.vm_name,
        });
        setTurns((prev) =>
          prev.map((t) =>
            t.id === ev.request_id
              ? {
                  ...t,
                  content: t.content + ev.delta!,
                  vmName: ev.vm_name,
                  slot: ev.slot,
                  model: ev.model,
                  ttftMs: ev.ttft_ms ?? t.ttftMs,
                  tps: ev.instant_tps ?? t.tps,
                  tokens: ev.completion_tokens ?? t.tokens,
                  elapsedMs: ev.elapsed_ms ?? t.elapsedMs,
                  transport: ev.transport ?? t.transport,
                  streaming: true,
                }
              : t,
          ),
        );
      } else if (ev.kind === "done") {
        setLiveStreamMetrics(null);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === ev.request_id
              ? {
                  ...t,
                  vmName: ev.vm_name,
                  slot: ev.slot,
                  model: ev.model,
                  ttftMs: ev.ttft_ms ?? t.ttftMs,
                  tps: ev.avg_tps ?? ev.instant_tps ?? t.tps,
                  tokens: ev.completion_tokens ?? t.tokens,
                  elapsedMs: ev.elapsed_ms ?? t.elapsedMs,
                  transport: ev.transport ?? t.transport,
                  streaming: false,
                }
              : t,
          ),
        );
      } else if (ev.kind === "error") {
        setLiveStreamMetrics(null);
        const wasCancelled =
          cancelledReqIdRef.current === ev.request_id ||
          (ev.error ?? "").includes("对话已手动停止");
        setTurns((prev) =>
          prev.map((t) =>
            t.id === ev.request_id
              ? {
                  ...t,
                  streaming: false,
                  cancelled: wasCancelled || t.cancelled,
                  error: wasCancelled ? null : (ev.error ?? "推理请求失败"),
                }
              : t,
          ),
        );
      }
    })
      .then((unlisten) => {
        if (disposed) {
          unlisten();
          return null;
        }
        return unlisten;
      })
      .catch(() => null);

    return () => {
      disposed = true;
      unlistenPromise.then((fn) => fn?.());
    };
  }, []);

  if (!open) return null;

  const handleSend = async (customPrompt?: string) => {
    const text = (customPrompt ?? input).trim();
    if (!text || sending) return;
    if (!targetState.canSend || !targetState.submitTargetVm) {
      toast(targetState.disabledReason ?? "当前目标节点尚未就绪，暂不可发送消息", "warn");
      return;
    }

    const submitTarget = targetState.submitTargetVm;
    const submitModel = targetState.submitModel;
    setLostTargetNotice(null);

    const nowStr = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    const reqId = `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    activeReqIdRef.current = reqId;
    cancelledReqIdRef.current = null;
    stickToBottomRef.current = true;

    const userTurn: ConversationTurn = {
      id: `user-${reqId}`,
      role: "user",
      content: text,
      createdAt: nowStr,
    };

    const assistantTurn: ConversationTurn = {
      id: reqId,
      role: "assistant",
      content: "",
      createdAt: nowStr,
      vmName: submitTarget,
      slot: displaySlot ?? null,
      model: displayModel,
      streaming: true,
    };

    const historyMessages = buildHistoryMessages(
      turns,
      systemPrompt,
      text,
      MAX_HISTORY_PAIRS,
    );

    if (!customPrompt) {
      setInput("");
    }
    setTurns((prev) => [...prev, userTurn, assistantTurn]);
    setSending(true);

    try {
      const resp = await api.streamLlmChat({
        request_id: reqId,
        target_vm: submitTarget,
        model: submitModel,
        messages: historyMessages,
        max_tokens: maxTokens,
        temperature,
      });

      const wasCancelled = cancelledReqIdRef.current === reqId;
      setTurns((prev) =>
        prev.map((t) =>
          t.id === reqId
            ? {
                ...t,
                content: resp.reply || t.content,
                vmName: resp.vm_name,
                slot: resp.slot,
                model: resp.model,
                ttftMs: resp.ttft_ms,
                tps: resp.avg_tps,
                tokens: resp.completion_tokens,
                elapsedMs: resp.elapsed_ms,
                transport: resp.transport,
                cancelled: wasCancelled || t.cancelled,
                streaming: false,
              }
            : t,
        ),
      );
    } catch (e) {
      const msg = errText(e);
      const wasCancelled =
        cancelledReqIdRef.current === reqId || msg.includes("对话已手动停止");
      setTurns((prev) =>
        prev.map((t) =>
          t.id === reqId
            ? {
                ...t,
                streaming: false,
                cancelled: wasCancelled || t.cancelled,
                error: wasCancelled ? null : msg,
              }
            : t,
        ),
      );
      if (!wasCancelled) {
        toast(`大模型对话请求失败: ${msg}`, "error");
      }
    } finally {
      if (activeReqIdRef.current === reqId) {
        activeReqIdRef.current = null;
      }
      setSending(false);
      setLiveStreamMetrics(null);
      setTimeout(() => inputRef.current?.focus(), 40);
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard
      ?.writeText(text)
      .then(() => {
        setCopiedId(id);
        setTimeout(() => setCopiedId((cur) => (cur === id ? null : cur)), 1500);
      })
      .catch(() => {});
  };

  return (
    <div
      className="overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget && !sending) handleCloseModal();
      }}
    >
      <div
        ref={modalRef}
        className="modal chat-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onClick={(e) => e.stopPropagation()}
      >
        {/* 顶栏：模型信息、目标节点切换器与实时 tok/s 监控 */}
        <div className="chat-modal-head">
          <div className="chat-head-left">
            <div className="chat-brand-icon">
              <ColabIcon size={26} />
            </div>
            <div>
              <div className="chat-title-row">
                <h3 id={titleId}>大模型交互对话窗口</h3>
                <code className="chat-model-badge">{displayModel}</code>
                {displaySlot && (
                  <span className={`slot-badge slot-badge-${displaySlot}`}>
                    {displaySlot.toUpperCase()}
                  </span>
                )}
                <span className="chip chip-blue">
                  <Cpu size={11} />
                  {resolvedVm ? `${resolvedVm.name} · ${resolvedVm.accelerator}` : "未连接"}
                </span>
              </div>
              <div className="chat-sub-row">
                <span>引擎: {displayEngine}</span>
                <span>·</span>
                <span>
                  {targetState.effectiveSelectValue === "auto"
                    ? `蓝绿零中断自动路由 (${targetState.submitTargetVm ?? "等待节点就绪"})`
                    : `直连指定节点 (${targetState.effectiveSelectValue})`}
                </span>
              </div>
            </div>
          </div>

          <div className="chat-head-right">
            <div className="chat-target-select-wrap" title="选择对话路由目标">
              <GitCompare size={13} color="var(--primary)" />
              <select
                className="chat-target-select"
                value={targetState.effectiveSelectValue}
                onChange={(e) => {
                  setLostTargetNotice(null);
                  setTargetVm(e.target.value);
                }}
                disabled={sending}
              >
                <option value="auto">
                  🔄 自动路由活跃节点
                  {targetState.effectiveSelectValue === "auto" && resolvedVm
                    ? ` (${resolvedVm.name}${displaySlot ? ` · ${displaySlot.toUpperCase()}` : ""})`
                    : str(bg?.active_slot)
                      ? ` (${str(bg?.active_slot).toUpperCase()})`
                      : ""}
                  {!anyReadyNode ? "（暂无就绪节点）" : ""}
                </option>
                {llmVms.map((v) => {
                  const ready = isLlmVmReady(v);
                  return (
                    <option key={v.name} value={v.name} disabled={!ready}>
                      {formatChatOptionLabel(v, bg)}
                    </option>
                  );
                })}
              </select>
            </div>

            <button
              type="button"
              className={`icon-btn ${showSettings ? "active-icon-btn" : ""}`}
              onClick={() => setShowSettings((s) => !s)}
              title="生成参数与 System Prompt 设置"
            >
              <SlidersHorizontal size={15} />
            </button>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setTurns([])}
              disabled={sending || turns.length === 0}
              title="清空对话历史"
            >
              <Trash2 size={15} />
            </button>
            <button type="button" className="icon-btn" onClick={handleCloseModal} title="关闭窗口 (Esc)">
              <X size={16} />
            </button>
          </div>
        </div>

        {/* 实时推理遥测指标条 */}
        <div className="chat-telemetry-strip">
          <div className="ct-item">
            <span className="ct-label">
              <Gauge size={12} color="var(--primary)" /> 实时输出频率
            </span>
            <span className="ct-val mono" style={{ textAlign: "right" }}>
              <b>{tpsView.realtimeTps.toFixed(1)}</b>{" "}
              <small>tok/s{tpsView.isIdle ? " (空闲)" : ""}</small>
              {tpsView.isIdle && tpsView.lastRequestText && (
                <small className="dim" style={{ display: "block", fontSize: 10.5 }}>
                  {tpsView.lastRequestText}
                </small>
              )}
            </span>
          </div>
          <div className="ct-item">
            <span className="ct-label">60s 平均速率</span>
            <span className="ct-val mono">
              <b>{displayAvgTps.toFixed(1)}</b> <small>tok/s</small>
            </span>
          </div>
          <div className="ct-item">
            <span className="ct-label">首字延迟 (TTFT)</span>
            <span className="ct-val mono">
              <b>{displayTtft !== null ? `${displayTtft.toFixed(0)} ms` : "—"}</b>
            </span>
          </div>
          <div className="ct-item">
            <span className="ct-label">节点累计生成</span>
            <span className="ct-val mono">
              <b>{displayTotalTokens}</b> <small>tokens</small>
            </span>
          </div>
        </div>

        {/* 可折叠推理参数配置栏 */}
        {showSettings && (
          <div className="chat-settings-bar">
            <div className="chat-setting-field wide">
              <label>System Prompt (系统提示词)</label>
              <input
                type="text"
                value={systemPrompt}
                onChange={(e) => setSystemPrompt(e.target.value)}
                placeholder="设置系统角色提示词…"
              />
            </div>
            <div className="chat-setting-field">
              <label>单次最大生成 (Max Tokens): {maxTokens}</label>
              <input
                type="range"
                min={256}
                max={8192}
                step={128}
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
              />
            </div>
            <div className="chat-setting-field">
              <label>采样温度 (Temperature): {temperature.toFixed(2)}</label>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.05}
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
              />
            </div>
          </div>
        )}

        {/* #64：目标消失或节点预热未就绪提示条 */}
        {(lostTargetNotice || !targetState.canSend) && (
          <div
            className="banner banner-warn"
            role="status"
            style={{ margin: 0, borderRadius: 0, borderLeft: "none", borderRight: "none", flexShrink: 0 }}
          >
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ flex: 1, fontSize: 12.5 }}>
              {lostTargetNotice && <div>{lostTargetNotice}</div>}
              {!targetState.canSend && targetState.disabledReason && (
                <div>
                  <b>{targetState.disabledReason}</b>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 对话消息流 */}
        <div
          className="chat-body"
          ref={chatBodyRef}
          onScroll={(e) => {
            const el = e.currentTarget;
            stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
          }}
        >
          {turns.length === 0 ? (
            <div className="chat-empty">
              <div className="chat-empty-icon">
                <ColabIcon size={40} />
              </div>
              <h4>已连接至 Colab 云端大模型：{displayModel}</h4>
              <p className="dim">
                每次提问均实时流式返回，并自动触发云端计量边车更新仪表盘上的 Token 输出频率 (tok/s) 与首字延迟 (TTFT)。
              </p>
              <div className="chat-quick-grid">
                {QUICK_PROMPTS.map((q, idx) => (
                  <button
                    type="button"
                    key={idx}
                    className="chat-quick-btn"
                    onClick={() => handleSend(q)}
                    disabled={sending || !targetState.canSend}
                  >
                    <Sparkles size={13} color="var(--primary)" />
                    <span>{q}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            turns.map((turn) => (
              <div key={turn.id} className={`chat-row ${turn.role}`}>
                <div className="chat-avatar">
                  {turn.role === "user" ? (
                    <User size={16} color="#fff" />
                  ) : (
                    <ColabIcon size={20} />
                  )}
                </div>

                <div className="chat-bubble-wrap">
                  <div className="chat-meta-top">
                    <span className="chat-sender">
                      {turn.role === "user" ? "你" : str(turn.model) || displayModel}
                    </span>
                    {turn.role === "assistant" && str(turn.slot) && (
                      <span className={`slot-badge slot-badge-${str(turn.slot).toLowerCase()}`}>
                        {str(turn.slot).toUpperCase()}
                      </span>
                    )}
                    {turn.role === "assistant" && str(turn.vmName) && (
                      <span className="chat-vm-tag">{str(turn.vmName)}</span>
                    )}
                    {turn.cancelled && <span className="chat-vm-tag">已停止</span>}
                    <span className="chat-time">{turn.createdAt}</span>
                  </div>

                  <div className={`chat-bubble ${turn.role} ${turn.error ? "is-error" : ""}`}>
                    {turn.error ? (
                      <div className="chat-err-text">请求异常：{turn.error}</div>
                    ) : turn.content ? (
                      turn.role === "assistant" ? (
                        <MarkdownContent content={turn.content} streaming={turn.streaming} />
                      ) : (
                        <div className="chat-content">{turn.content}</div>
                      )
                    ) : turn.streaming ? (
                      <div className="chat-thinking">
                        <Loader2 size={14} className="spin" />
                        <span>正在等待云端 GPU 生成首字响应…</span>
                      </div>
                    ) : (
                      <span className="dim">
                        {turn.cancelled ? "（已手动停止生成）" : "（无输出内容）"}
                      </span>
                    )}
                  </div>

                  {turn.role === "assistant" && (!!turn.content || !!num(turn.tps)) && (
                    <div className="chat-meta-foot">
                      <div className="chat-stats-pills">
                        {numOrNull(turn.tps) !== null && (
                          <span className="chat-stat-pill highlight">
                            <Zap size={11} />
                            {num(turn.tps).toFixed(1)} tok/s
                          </span>
                        )}
                        {numOrNull(turn.ttftMs) !== null && (
                          <span className="chat-stat-pill">
                            TTFT {num(turn.ttftMs).toFixed(0)} ms
                          </span>
                        )}
                        {numOrNull(turn.tokens) !== null && (
                          <span className="chat-stat-pill">{num(turn.tokens)} tokens</span>
                        )}
                        {numOrNull(turn.elapsedMs) !== null && (
                          <span className="chat-stat-pill">
                            {(num(turn.elapsedMs) / 1000).toFixed(1)}s
                          </span>
                        )}
                        {str(turn.transport) && (
                          <span className="chat-stat-pill dim">{str(turn.transport)}</span>
                        )}
                      </div>

                      {turn.content && !turn.streaming && (
                        <button
                          className="chat-copy-btn"
                          onClick={() => handleCopy(turn.id, turn.content)}
                          title="复制回复原文"
                        >
                          {copiedId === turn.id ? (
                            <>
                              <Check size={12} color="var(--success)" /> 已复制
                            </>
                          ) : (
                            <>
                              <Copy size={12} /> 复制
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ))
          )}
        </div>

        {/* 底部输入栏 */}
        <div className="chat-input-bar">
          <textarea
            ref={inputRef}
            className="chat-textarea"
            rows={2}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                handleSend();
              }
            }}
            placeholder={
              targetState.canSend
                ? `向 ${displayModel} 提问（Enter 发送，Shift + Enter 换行，Esc 停止/关闭）…`
                : (targetState.disabledReason ?? "等待云端推理节点预热就绪后即可对话…")
            }
            disabled={sending}
          />
          {sending ? (
            <button
              type="button"
              className="btn btn-danger chat-send-btn"
              onClick={handleStop}
              title="立即中断本次流式生成 (Esc)"
            >
              <Square size={14} />
              停止生成
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary chat-send-btn"
              onClick={() => handleSend()}
              disabled={!input.trim() || !targetState.canSend}
              title={!targetState.canSend ? (targetState.disabledReason ?? "目标节点未就绪") : "发送消息"}
            >
              <Send size={15} />
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
