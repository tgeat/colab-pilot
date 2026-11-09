import { useEffect, useMemo, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  Check,
  Copy,
  Cpu,
  Gauge,
  GitCompare,
  Loader2,
  Send,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  User,
  X,
  Zap,
} from "lucide-react";
import { api, errText } from "../api";
import type { ChatMessage, LlmChatStreamEvent, Overview, VmView } from "../types";
import { ColabIcon, useToast } from "./ui";

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

export function isLlmVm(vm: VmView): boolean {
  if (!vm.on_server) return false;
  if (vm.telemetry?.llm) return true;
  if (vm.telemetry?.threads?.llm_process) return true;
  if (vm.name.startsWith("llm-")) return true;
  return false;
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
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [maxTokens, setMaxTokens] = useState(512);
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
  const chatBodyRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (open && initialTarget) {
      setTargetVm(initialTarget);
      setTimeout(() => inputRef.current?.focus(), 60);
    }
  }, [open, initialTarget]);

  const vms = ov?.vms ?? [];
  const bg = ov?.llm_bluegreen ?? null;
  const llmVms = useMemo(() => {
    const matched = vms.filter(isLlmVm);
    return matched.length > 0 ? matched : vms.filter((v) => v.on_server);
  }, [vms]);

  // 解析当前选中的目标实例详情
  const resolvedVm: VmView | undefined = useMemo(() => {
    if (targetVm === "auto") {
      const prefix = bg?.session_prefix ?? "llm";
      const activeSlot = (bg?.active_slot ?? "blue").toLowerCase();
      return (
        llmVms.find(
          (v) =>
            v.telemetry?.llm?.slot?.toLowerCase() === activeSlot ||
            v.name === `${prefix}-${activeSlot}` ||
            v.name.endsWith(`-${activeSlot}`),
        ) ??
        llmVms.find((v) => v.telemetry?.llm?.online && v.telemetry?.llm?.role === "active") ??
        llmVms[0]
      );
    }
    return vms.find((v) => v.name === targetVm) ?? llmVms[0];
  }, [targetVm, bg, llmVms, vms]);

  const llmTel = resolvedVm?.telemetry?.llm ?? null;
  const displayModel =
    llmTel?.model_name ?? bg?.model_name ?? resolvedVm?.workspace.description ?? "Colab Cloud LLM";
  const displayEngine = llmTel?.engine ?? bg?.engine ?? "OpenAI-Compatible";
  const displaySlot = (
    liveStreamMetrics?.slot ??
    llmTel?.slot ??
    (resolvedVm?.name.endsWith("-green")
      ? "green"
      : resolvedVm?.name.endsWith("-blue")
        ? "blue"
        : bg?.active_slot ?? null)
  )?.toLowerCase();

  const displayTps =
    liveStreamMetrics?.tps ??
    (llmTel ? llmTel.instant_tps || llmTel.last_request_tps || 0 : bg?.warmup_tps ?? 0);
  const displayAvgTps = llmTel?.avg_tps_60s ?? 0;
  const displayTtft = liveStreamMetrics?.ttft ?? llmTel?.ttft_ms ?? llmTel?.avg_ttft_ms ?? null;
  const displayTotalTokens = llmTel?.total_completion_tokens ?? 0;

  // 自动滚动到底部
  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight;
    }
  }, [turns, open]);

  // 监听后端 Tauri 推送的流式 Token 事件
  useEffect(() => {
    const unlistenPromise = listen<LlmChatStreamEvent>("llm-chat-event", (event) => {
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
        setTurns((prev) =>
          prev.map((t) =>
            t.id === ev.request_id
              ? {
                  ...t,
                  streaming: false,
                  error: ev.error ?? "推理请求失败",
                }
              : t,
          ),
        );
      }
    });

    return () => {
      unlistenPromise.then((unlisten) => unlisten());
    };
  }, []);

  if (!open) return null;

  const handleSend = async (customPrompt?: string) => {
    const text = (customPrompt ?? input).trim();
    if (!text || sending) return;

    const nowStr = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    const reqId = `chat-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    activeReqIdRef.current = reqId;

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
      vmName: resolvedVm?.name,
      slot: displaySlot ?? null,
      model: displayModel,
      streaming: true,
    };

    const historyMessages: ChatMessage[] = [];
    if (systemPrompt.trim()) {
      historyMessages.push({ role: "system", content: systemPrompt.trim() });
    }
    for (const t of turns) {
      if (!t.error && t.content.trim()) {
        historyMessages.push({ role: t.role, content: t.content });
      }
    }
    historyMessages.push({ role: "user", content: text });

    if (!customPrompt) {
      setInput("");
    }
    setTurns((prev) => [...prev, userTurn, assistantTurn]);
    setSending(true);

    try {
      const resp = await api.streamLlmChat({
        request_id: reqId,
        target_vm: targetVm,
        model: llmTel?.model_name ?? bg?.model_name ?? null,
        messages: historyMessages,
        max_tokens: maxTokens,
        temperature,
      });

      setTurns((prev) =>
        prev.map((t) =>
          t.id === reqId
            ? {
                ...t,
                content: t.content || resp.reply,
                vmName: resp.vm_name,
                slot: resp.slot,
                model: resp.model,
                ttftMs: resp.ttft_ms,
                tps: resp.avg_tps,
                tokens: resp.completion_tokens,
                elapsedMs: resp.elapsed_ms,
                transport: resp.transport,
                streaming: false,
              }
            : t,
        ),
      );
    } catch (e) {
      const msg = errText(e);
      setTurns((prev) =>
        prev.map((t) =>
          t.id === reqId
            ? {
                ...t,
                streaming: false,
                error: msg,
              }
            : t,
        ),
      );
      toast(`大模型对话请求失败: ${msg}`, "error");
    } finally {
      setSending(false);
      setLiveStreamMetrics(null);
      setTimeout(() => inputRef.current?.focus(), 40);
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId((cur) => (cur === id ? null : cur)), 1500);
    });
  };

  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal chat-modal" onClick={(e) => e.stopPropagation()}>
        {/* 顶栏：模型信息、目标节点切换器与实时 tok/s 监控 */}
        <div className="chat-modal-head">
          <div className="chat-head-left">
            <div className="chat-brand-icon">
              <ColabIcon size={26} />
            </div>
            <div>
              <div className="chat-title-row">
                <h3>大模型交互对话窗口</h3>
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
                  {targetVm === "auto"
                    ? "蓝绿零中断自动路由 (Make-Before-Break)"
                    : `直连指定节点 (${targetVm})`}
                </span>
              </div>
            </div>
          </div>

          <div className="chat-head-right">
            <div className="chat-target-select-wrap" title="选择对话路由目标">
              <GitCompare size={13} color="var(--primary)" />
              <select
                className="chat-target-select"
                value={targetVm}
                onChange={(e) => setTargetVm(e.target.value)}
                disabled={sending}
              >
                <option value="auto">
                  🔄 自动路由活跃节点
                  {bg?.active_slot ? ` (${bg.active_slot.toUpperCase()})` : ""}
                </option>
                {llmVms.map((v) => {
                  const sl = v.telemetry?.llm?.slot;
                  const role = v.telemetry?.llm?.role;
                  return (
                    <option key={v.name} value={v.name}>
                      🖥️ {v.name} ({v.accelerator}
                      {sl ? ` · ${sl.toUpperCase()}` : ""}
                      {role ? `/${role}` : ""})
                    </option>
                  );
                })}
              </select>
            </div>

            <button
              className={`icon-btn ${showSettings ? "active-icon-btn" : ""}`}
              onClick={() => setShowSettings((s) => !s)}
              title="生成参数与 System Prompt 设置"
            >
              <SlidersHorizontal size={15} />
            </button>
            <button
              className="icon-btn"
              onClick={() => setTurns([])}
              disabled={sending || turns.length === 0}
              title="清空对话历史"
            >
              <Trash2 size={15} />
            </button>
            <button className="icon-btn" onClick={onClose} title="关闭窗口 (Esc)">
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
            <span className="ct-val mono">
              <b>{displayTps.toFixed(1)}</b> <small>tok/s</small>
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
                min={64}
                max={2048}
                step={64}
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

        {/* 对话消息流 */}
        <div className="chat-body" ref={chatBodyRef}>
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
                    key={idx}
                    className="chat-quick-btn"
                    onClick={() => handleSend(q)}
                    disabled={sending}
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
                      {turn.role === "user" ? "你" : turn.model || displayModel}
                    </span>
                    {turn.role === "assistant" && turn.slot && (
                      <span className={`slot-badge slot-badge-${turn.slot.toLowerCase()}`}>
                        {turn.slot.toUpperCase()}
                      </span>
                    )}
                    {turn.role === "assistant" && turn.vmName && (
                      <span className="chat-vm-tag">{turn.vmName}</span>
                    )}
                    <span className="chat-time">{turn.createdAt}</span>
                  </div>

                  <div className={`chat-bubble ${turn.role} ${turn.error ? "is-error" : ""}`}>
                    {turn.error ? (
                      <div className="chat-err-text">请求异常：{turn.error}</div>
                    ) : turn.content ? (
                      <div className="chat-content">
                        {turn.content}
                        {turn.streaming && <span className="chat-cursor" />}
                      </div>
                    ) : turn.streaming ? (
                      <div className="chat-thinking">
                        <Loader2 size={14} className="spin" />
                        <span>正在等待云端 GPU 生成首字响应…</span>
                      </div>
                    ) : (
                      <span className="dim">（无输出内容）</span>
                    )}
                  </div>

                  {turn.role === "assistant" && (turn.content || turn.tps) && (
                    <div className="chat-meta-foot">
                      <div className="chat-stats-pills">
                        {turn.tps !== undefined && turn.tps !== null && (
                          <span className="chat-stat-pill highlight">
                            <Zap size={11} />
                            {turn.tps.toFixed(1)} tok/s
                          </span>
                        )}
                        {turn.ttftMs !== undefined && turn.ttftMs !== null && (
                          <span className="chat-stat-pill">
                            TTFT {turn.ttftMs.toFixed(0)} ms
                          </span>
                        )}
                        {turn.tokens !== undefined && turn.tokens !== null && (
                          <span className="chat-stat-pill">{turn.tokens} tokens</span>
                        )}
                        {turn.elapsedMs !== undefined && turn.elapsedMs !== null && (
                          <span className="chat-stat-pill">
                            {(turn.elapsedMs / 1000).toFixed(1)}s
                          </span>
                        )}
                        {turn.transport && (
                          <span className="chat-stat-pill dim">{turn.transport}</span>
                        )}
                      </div>

                      {turn.content && !turn.streaming && (
                        <button
                          className="chat-copy-btn"
                          onClick={() => handleCopy(turn.id, turn.content)}
                          title="复制回复内容"
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
            placeholder={`向 ${displayModel} 提问（Enter 发送，Shift + Enter 换行）…`}
            disabled={sending}
          />
          <button
            className="btn btn-primary chat-send-btn"
            onClick={() => handleSend()}
            disabled={sending || !input.trim()}
          >
            {sending ? (
              <>
                <Loader2 size={15} className="spin" />
                生成中…
              </>
            ) : (
              <>
                <Send size={15} />
                发送
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
