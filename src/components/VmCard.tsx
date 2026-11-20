import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Cloud,
  Code2,
  Copy,
  Cpu,
  Database,
  Eye,
  EyeOff,
  FileCode2,
  FolderGit2,
  Gauge,
  GitCompare,
  KeyRound,
  Link2,
  Loader2,
  MessageSquare,
  PowerOff,
  RefreshCw,
  Repeat2,
  RotateCw,
  ShieldCheck,
  Sparkles,
  Terminal,
  Trash2,
} from "lucide-react";
import type {
  LlmApiAccessInfo,
  LlmTelemetry,
  ProcInfo,
  RelayConfig,
  TrainingTelemetry,
  VmSshInfo,
  VmView,
} from "../types";
import { fmtDuration, formatRealtimeTps, num, numOrNull, pct, shortEndpoint, str } from "../format";
import {
  isLlmVm,
  isLlmVmReady,
  resolveRelayCompletionView,
  resolveVmStatusPillSpec,
} from "../chatUtils";
import { RelayPanel } from "./RelayPanel";
import { KV } from "./ui";

function CopyBtn({
  text,
  label,
  title,
  compact = false,
}: {
  text: string;
  label?: string;
  title?: string;
  compact?: boolean;
}) {
  const [copied, setCopied] = useState(false);
  const doCopy = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // ignore
    }
  };
  return (
    <button
      type="button"
      className={`copy-chip-btn ${copied ? "copied" : ""} ${compact ? "compact" : ""}`}
      onClick={doCopy}
      title={title ?? `点击复制: ${text}`}
    >
      {copied ? <Check size={11} /> : <Copy size={11} />}
      {label && <span>{copied ? "已复制" : label}</span>}
    </button>
  );
}

/** 云端真正在运行的任务进程（排除僵尸进程 Z 与 Kernel 本身） */
function liveActiveProcess(vm: VmView): ProcInfo | null {
  const ap = vm.telemetry?.threads?.active_process ?? null;
  if (!ap) return null;
  if (str(ap.state).startsWith("Z") || str(ap.command).includes("<defunct>")) return null;
  if (ap.role === "kernel") return null;
  return ap;
}

function MiniMeter({ name, value, display }: { name: string; value: number; display: string }) {
  const v = pct(value);
  const cls = v >= 88 ? "fill high" : v >= 70 ? "fill mid" : "fill";
  return (
    <div className="mini-meter">
      <div className="mini-meter-head">
        <span className="mini-meter-name">{name}</span>
        <span className="mini-meter-num">{display}</span>
      </div>
      <div className="track">
        <div className={cls} style={{ width: `${v}%` }} />
      </div>
    </div>
  );
}

function statusPill(vm: VmView, activeProc: ProcInfo | null, serverStateUnknown = false) {
  const spec = resolveVmStatusPillSpec(vm, Boolean(activeProc), serverStateUnknown);
  return (
    <span className={`pill ${spec.cls}`} title={spec.title}>
      <span className={`dot ${spec.pulse ? "pulse" : ""}`} />
      {spec.text}
    </span>
  );
}

function fmtIsoTime(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleTimeString("zh-CN", { hour12: false });
}

function eventBadgeCls(et: string, hasError: boolean) {
  if (hasError) return "ev-badge ev-err";
  if (et === "execution") return "ev-badge ev-exec";
  if (et === "file_operation") return "ev-badge ev-file";
  if (et === "session_created") return "ev-badge ev-new";
  if (et === "automation" || et === "automation_result") return "ev-badge ev-auto";
  return "ev-badge";
}

/** 每台虚拟机标配的本机免密 SSH 访问栏（全部采用 colab ssh -s <name>，Windows 不使用 /dev/null） */
function SshAccessBar({
  vmName,
  ssh,
  orphan,
  onServer,
  onOpenSshTerminal,
}: {
  vmName: string;
  ssh?: VmSshInfo | null;
  orphan?: boolean;
  onServer: boolean;
  onOpenSshTerminal?: (name: string) => void;
}) {
  const hostAliasCmd = ssh?.ssh_command || `ssh colab-${vmName}`;
  const colabSshCmd = ssh?.colab_ssh_command || `colab ssh -s ${vmName}`;
  const portCmd =
    ssh?.ssh_port_command && !ssh.ssh_port_command.includes("/dev/null")
      ? ssh.ssh_port_command
      : colabSshCmd;
  const localPort = ssh?.local_port;

  return (
    <div className="ssh-access-bar">
      <div className="ssh-bar-left">
        <span className="ssh-badge">
          <Terminal size={12} />
          <span>SSH</span>
        </span>

        <div className="ssh-cmd-pill" title="已自动写入 ~/.ssh/config，可在任意终端或 VS Code Remote-SSH 中直接使用">
          <code className="mono">{hostAliasCmd}</code>
          <CopyBtn text={hostAliasCmd} compact title={`复制 SSH 命令: ${hostAliasCmd}`} />
        </div>

        {localPort ? (
          <div className="ssh-cmd-pill" title={`本机按需 SSH 桥接端口 127.0.0.1:${localPort} -> 云端 /colab/ssh WebSocket`}>
            <span className="dim" style={{ fontSize: 10.5 }}>端口</span>
            <code className="mono">127.0.0.1:{localPort}</code>
            <CopyBtn text={portCmd} label="复制直连" compact title={`复制带密钥与端口的完整命令:\n${portCmd}`} />
          </div>
        ) : null}

        <div className="ssh-cmd-pill" title="colab-cli v0.8+ 原生 SSH 会话命令">
          <code className="mono">{colabSshCmd}</code>
          <CopyBtn text={colabSshCmd} compact title={`复制: ${colabSshCmd}`} />
        </div>

        {ssh?.ready && (
          <span className="ssh-key-tag" title={`免密私钥: ${ssh.private_key_path}\n指纹: ${ssh.public_key_fingerprint}`}>
            <ShieldCheck size={11} />
            Ed25519 免密就绪
          </span>
        )}
      </div>

      {onOpenSshTerminal && onServer && (
        <button
          type="button"
          className="btn btn-sm ssh-open-btn"
          onClick={() => onOpenSshTerminal(vmName)}
          title={
            orphan
              ? `一键弹出交互式远程终端窗口（浏览器创建的外部实例若未开放 SSH 将自动回退 Web TTY）`
              : `一键弹出本机交互式 SSH 终端窗口并进入 ${vmName}:/content`
          }
        >
          <Terminal size={12} />
          <span>打开 SSH 终端</span>
        </button>
      )}
    </div>
  );
}

/** 部署大模型的虚拟机专属的 OpenAI API Endpoint 与 API Key 面板（精确区分预热中 / 部署失败 / 后端未就绪 / 在线） */
export function LlmApiAccessBox({
  apiInfo,
  busy,
  llm,
  deployError,
  hasLlmProc,
  onRotateKey,
  onRetryDeploy,
}: {
  apiInfo: LlmApiAccessInfo;
  busy?: string | null;
  llm?: LlmTelemetry | null;
  deployError?: string | null;
  hasLlmProc?: boolean;
  onRotateKey?: (vmName: string, customKey?: string | null) => Promise<void>;
  onRetryDeploy?: () => void;
}) {
  const [showKey, setShowKey] = useState(false);
  const [showSnippet, setShowSnippet] = useState<"none" | "curl" | "python">("none");
  const [rotating, setRotating] = useState(false);

  const isDeploying =
    busy === "deploying_llm" ||
    busy === "creating" ||
    Boolean(llm && !llm.online && !llm.backend_healthy && !deployError);
  const isOnline =
    !isDeploying && !deployError && Boolean(llm?.online || llm?.backend_healthy || hasLlmProc);

  let statusBadge: JSX.Element;
  if (isDeploying) {
    statusBadge = (
      <span className="pill pill-amber">
        <span className="dot pulse" />
        部署预热中…
      </span>
    );
  } else if (deployError) {
    statusBadge = (
      <span className="pill pill-red" title={deployError}>
        <span className="dot" />
        部署失败
      </span>
    );
  } else if (isOnline) {
    statusBadge = (
      <span className="pill pill-green">
        <span className="dot pulse" />
        在线 · 可直接调用
      </span>
    );
  } else {
    statusBadge = (
      <span className="pill pill-gray">
        <span className="dot" />
        后端未就绪
      </span>
    );
  }

  const maskedKey =
    apiInfo.api_key.length > 16
      ? `${apiInfo.api_key.slice(0, 12)}••••••••${apiInfo.api_key.slice(-6)}`
      : apiInfo.api_key;

  const handleRotate = async () => {
    if (!onRotateKey || rotating) return;
    setRotating(true);
    try {
      await onRotateKey(apiInfo.vm_name, null);
    } finally {
      setRotating(false);
    }
  };

  return (
    <div className="llm-api-box">
      <div className="llm-api-head">
        <span className="llm-api-title">
          <KeyRound size={13} />
          <span>模型 OpenAI 兼容 API 与专属 Key</span>
          <span className="llm-api-port-tag mono">:{apiInfo.local_port}</span>
          {statusBadge}
        </span>

        <div className="llm-api-actions">
          {!isOnline && !isDeploying && onRetryDeploy && (
            <button
              type="button"
              className="copy-chip-btn"
              onClick={onRetryDeploy}
              title="重新配置并在该实例上启动大模型推理后端"
            >
              <Sparkles size={11} />
              <span>{deployError ? "重新部署大模型" : "启动模型后端"}</span>
            </button>
          )}
          <CopyBtn text={apiInfo.curl_example} label="复制 cURL" title="复制完整流式 cURL 调用命令" />
          <CopyBtn text={apiInfo.python_example} label="复制 Python SDK" title="复制完整 Python OpenAI SDK 调用代码" />
          <button
            type="button"
            className={`copy-chip-btn ${showSnippet !== "none" ? "copied" : ""}`}
            onClick={() => setShowSnippet((s) => (s === "none" ? "python" : "none"))}
            title="展开/收起代码示例"
          >
            <Code2 size={11} />
            <span>{showSnippet === "none" ? "调用示例" : "收起示例"}</span>
          </button>
        </div>
      </div>

      <div className="llm-api-grid">
        <div className="llm-api-row">
          <span className="llm-api-k">
            <Link2 size={11} /> 专属 Base URL
          </span>
          <code className="llm-api-v mono">{apiInfo.api_base_url}</code>
          <CopyBtn text={apiInfo.api_base_url} label="复制" compact />
        </div>

        <div className="llm-api-row">
          <span className="llm-api-k">
            <GitCompare size={11} /> 蓝绿网关 URL
          </span>
          <code className="llm-api-v mono">{apiInfo.gateway_base_url}</code>
          <CopyBtn text={apiInfo.gateway_base_url} label="复制" compact />
        </div>

        <div className="llm-api-row">
          <span className="llm-api-k">
            <KeyRound size={11} /> API Key
          </span>
          <code className="llm-api-v mono key-val">{showKey ? apiInfo.api_key : maskedKey}</code>
          <div style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <button
              type="button"
              className="copy-chip-btn compact"
              onClick={() => setShowKey((v) => !v)}
              title={showKey ? "隐藏 API Key" : "显示完整 API Key"}
            >
              {showKey ? <EyeOff size={11} /> : <Eye size={11} />}
            </button>
            <CopyBtn text={apiInfo.api_key} label="复制 Key" compact />
            {onRotateKey && (
              <button
                type="button"
                className="copy-chip-btn compact"
                onClick={handleRotate}
                disabled={rotating}
                title="重新生成此虚拟机的 API Key"
              >
                <RotateCw size={11} className={rotating ? "spin" : ""} />
              </button>
            )}
          </div>
        </div>

        <div className="llm-api-row">
          <span className="llm-api-k">
            <Sparkles size={11} /> Model ID
          </span>
          <code className="llm-api-v mono" title={apiInfo.model_name}>
            {apiInfo.model_name}
          </code>
          <CopyBtn text={apiInfo.model_name} label="复制" compact />
        </div>
      </div>

      {showSnippet !== "none" && (
        <div className="llm-snippet-drawer">
          <div className="llm-snippet-tabs">
            <button
              type="button"
              className={showSnippet === "python" ? "active" : ""}
              onClick={() => setShowSnippet("python")}
            >
              Python (openai SDK)
            </button>
            <button
              type="button"
              className={showSnippet === "curl" ? "active" : ""}
              onClick={() => setShowSnippet("curl")}
            >
              cURL 流式请求
            </button>
          </div>
          <pre className="llm-snippet-pre mono">
            {showSnippet === "python" ? apiInfo.python_example : apiInfo.curl_example}
          </pre>
        </div>
      )}
    </div>
  );
}

function TrainingProgressBox({ tr, hasCkpt, ckptMb }: { tr: TrainingTelemetry; hasCkpt: boolean; ckptMb: number }) {
  const cur = num(tr.current_epoch);
  const tot = num(tr.total_epochs, 100);
  const progress = tot > 0 ? Math.min(100, Math.max(0, (cur / tot) * 100)) : 0;
  const loss = numOrNull(tr.loss);
  const modelName = str(tr.model_name);
  const syncStatus = str(tr.sync_status);
  const datasetStatus = str(tr.dataset_status);
  const statusMap: Record<string, string> = {
    pulling_dataset: "拉取数据集",
    pulling_checkpoint: "拉取断点",
    training: "训练中",
    syncing_gdrive: "同步网盘",
    handover: "换棒交接",
    handover_sync: "换棒前同步",
    handover_exit: "本棒已退出",
    completed: "已完成",
  };
  const statusLabel = statusMap[str(tr.status)] ?? (str(tr.status) || "运行中");

  return (
    <div className="workload-box dl-box">
      <div className="workload-head">
        <span className="workload-title">
          <Sparkles size={14} color="var(--primary)" />
          训练
          {modelName && <code>{modelName}</code>}
          <span className="dim">· {statusLabel}</span>
        </span>
        <span className="chip chip-blue">Round #{num(tr.round, 1)}</span>
      </div>

      <div className="dl-progress-head">
        <span className="mono">
          Epoch <b>{cur}</b> / {tot}
        </span>
        <span className="mono">{progress.toFixed(1)}%</span>
      </div>
      <div className="track" style={{ height: 8, margin: "6px 0 8px" }}>
        <div className="fill" style={{ width: `${progress}%` }} />
      </div>

      <div className="workload-grid">
        <div className="wg-item">
          <span className="wg-k">Loss</span>
          <span className="wg-v mono">{loss !== null ? loss.toFixed(5) : "—"}</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">ETA</span>
          <span className="wg-v mono">{num(tr.eta_seconds) ? fmtDuration(num(tr.eta_seconds)) : "—"}</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">换棒倒计时</span>
          <span className="wg-v mono">{fmtDuration(num(tr.handover_in_seconds))}</span>
        </div>
        <div className="wg-item" title={str(tr.gdrive_dir)}>
          <span className="wg-k">
            <Cloud size={10} /> 网盘
          </span>
          <span className="wg-v mono">{syncStatus || (hasCkpt ? `${ckptMb} MB` : "就绪")}</span>
        </div>
      </div>
      {datasetStatus && (
        <div className="workload-foot">
          <span>
            <Database size={12} /> 数据集: <b>{datasetStatus}</b>
          </span>
          <span>
            <Clock size={12} /> 下次热备: <b>{fmtDuration(num(tr.next_backup_in_seconds))}</b>
          </span>
        </div>
      )}
    </div>
  );
}

function LlmSupervisionBox({
  llm,
  vmName,
  vmBusy,
  deployError,
  llmApi,
  onOpenChat,
  onRotateKey,
  onRetryDeploy,
}: {
  llm: LlmTelemetry;
  vmName: string;
  vmBusy?: string | null;
  deployError?: string | null;
  llmApi?: LlmApiAccessInfo | null;
  onOpenChat?: (name: string) => void;
  onRotateKey?: (vmName: string, customKey?: string | null) => Promise<void>;
  onRetryDeploy?: () => void;
}) {
  const slot = str(llm.slot, "blue").toLowerCase() || "blue";
  const role = str(llm.role, "active").toLowerCase() || "active";
  const modelName = str(llm.model_name) || llmApi?.model_name || "LLM";
  // #65：严格区分实时输出速率与上次请求速率，空闲时保持 0.0 tok/s，绝不把 last_request_tps 冒充为实时速率
  const tpsView = formatRealtimeTps({
    instantTps: llm.instant_tps,
    lastRequestTps: llm.last_request_tps,
    peakTps: llm.peak_tps,
    activeStreams: llm.active_streams,
  });
  const avgTps = num(llm.avg_tps_60s);
  const peakTps = num(llm.peak_tps);
  const ttft = numOrNull(llm.ttft_ms) ?? numOrNull(llm.avg_ttft_ms);
  const activeStreams = tpsView.activeStreams;
  const rawHistory = Array.isArray(llm.tps_history) ? llm.tps_history.map((v) => num(v)) : [];
  const history = rawHistory.length > 0 ? rawHistory : [0, 0, 0, 0, 0, 0, 0, 0];
  const maxHist = Math.max(50, ...history, peakTps);
  const chatReady =
    Boolean(llm.online || llm.backend_healthy) &&
    role !== "warming" &&
    llm.online !== false &&
    llm.backend_healthy !== false &&
    vmBusy !== "deploying_llm" &&
    vmBusy !== "creating";

  return (
    <div className={`workload-box llm-box slot-${slot}`}>
      <div className="workload-head">
        <span className="workload-title">
          <GitCompare size={14} color={slot === "green" ? "var(--success)" : "var(--primary)"} />
          <code>{modelName}</code>
          <span className={`slot-badge slot-badge-${slot}`}>{slot.toUpperCase()}</span>
          <span className="dim">· {str(llm.engine) || llmApi?.engine || "llama.cpp"} · {role}</span>
        </span>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
          <span className={`online-dot ${chatReady ? "is-online" : "is-offline"}`}>
            <span className="dot pulse" />
            {chatReady ? "在线" : "启动中"}
          </span>
          {onOpenChat && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => onOpenChat(vmName)}
              disabled={!chatReady}
              title={
                chatReady
                  ? `打开与 ${vmName} (${modelName}) 的交互对话窗口`
                  : `${vmName} 云端推理引擎正在预热启动中，就绪后自动开放对话`
              }
            >
              <MessageSquare size={13} />
              <span>打开对话窗口</span>
            </button>
          )}
        </div>
      </div>

      <div className="tps-hero">
        <div className="tps-main">
          <div className="tps-label">
            <Gauge size={13} /> 实时速率
          </div>
          <div className="tps-value">
            {tpsView.realtimeTps.toFixed(1)}{" "}
            <small>tok/s{tpsView.isIdle ? " (空闲)" : ""}</small>
          </div>
          <div className="tps-sub">
            {activeStreams > 0
              ? `${activeStreams} 个流式请求进行中`
              : tpsView.lastRequestText
                ? `${tpsView.lastRequestText}${peakTps > 0 ? ` · 峰值 ${peakTps.toFixed(1)} tok/s` : ""}`
                : "空闲 · 等待请求"}
          </div>
        </div>

        <div className="tps-sparkline" title="最近采样的 Token 速率 (tok/s)">
          <div className="spark-bars">
            {history.slice(-20).map((val, i) => {
              const hPct = Math.max(8, Math.min(100, (val / maxHist) * 100));
              return (
                <div
                  key={i}
                  className={`spark-bar ${val > 0 ? "active" : ""}`}
                  style={{ height: `${hPct}%` }}
                  title={`${val.toFixed(1)} tok/s`}
                />
              );
            })}
          </div>
        </div>
      </div>

      <div className="workload-grid">
        <div className="wg-item">
          <span className="wg-k">60s 均值</span>
          <span className="wg-v mono">{avgTps.toFixed(1)} tok/s</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">TTFT</span>
          <span className="wg-v mono">{ttft !== null ? `${ttft.toFixed(0)} ms` : "—"}</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">累计 Token</span>
          <span className="wg-v mono">
            {num(llm.total_completion_tokens)} ({num(llm.total_requests)} 次)
          </span>
        </div>
        <div className="wg-item">
          <span className="wg-k">换棒倒计时</span>
          <span className="wg-v mono">{fmtDuration(num(llm.handover_in_seconds))}</span>
        </div>
      </div>

      {llmApi && (
        <LlmApiAccessBox
          apiInfo={llmApi}
          busy={vmBusy}
          llm={llm}
          deployError={deployError}
          hasLlmProc={true}
          onRotateKey={onRotateKey}
          onRetryDeploy={onRetryDeploy}
        />
      )}
    </div>
  );
}

export function VmCard({
  vm,
  allVms,
  serverStateUnknown = false,
  onOpenLlmChat,
  onShutdown,
  onOpenSshTerminal,
  onRotateLlmKey,
  onRetryProbe,
  onDeployLlm,
  onStartRelay,
  onRequestStopRelay,
  onDismissRelay,
  onOpenScriptDialog,
}: {
  vm: VmView;
  allVms?: VmView[];
  serverStateUnknown?: boolean;
  onOpenLlmChat?: (name: string) => void;
  onShutdown?: (vm: VmView) => void;
  onOpenSshTerminal?: (name: string) => void;
  onRotateLlmKey?: (vmName: string, customKey?: string | null) => Promise<void>;
  onRetryProbe?: (name: string) => Promise<void>;
  onDeployLlm?: (vm: VmView) => void;
  onStartRelay?: (
    name: string,
    config: RelayConfig,
    workspacePath?: string | null,
    legacyGdriveAction?: "migrate" | "ignore" | null,
  ) => Promise<void>;
  onRequestStopRelay?: (vm: VmView) => void;
  onDismissRelay?: (name: string) => Promise<void>;
  onOpenScriptDialog?: (vm: VmView) => void;
}) {
  const [probingLocal, setProbingLocal] = useState(false);
  const [relayPending, setRelayPending] = useState(false);
  const [dismissingRelay, setDismissingRelay] = useState(false);
  // #76 & #73：当接力运行中或为离线接力记录时初始展开一次，但允许用户随时手动点击折叠/展开
  const [relayExpanded, setRelayExpanded] = useState<boolean>(
    () => Boolean(vm.relay?.running) || (!vm.on_server && Boolean(vm.relay)),
  );
  const prevRelayRunningRef = useRef<boolean>(Boolean(vm.relay?.running));
  const prevOfflineRelayRef = useRef<boolean>(!vm.on_server && Boolean(vm.relay));

  useEffect(() => {
    const nowRunning = Boolean(vm.relay?.running);
    if (nowRunning && !prevRelayRunningRef.current) {
      setRelayExpanded(true);
    }
    prevRelayRunningRef.current = nowRunning;
  }, [vm.relay?.running]);

  useEffect(() => {
    const nowOffline = !vm.on_server && Boolean(vm.relay);
    if (nowOffline && !prevOfflineRelayRef.current) {
      setRelayExpanded(true);
    }
    prevOfflineRelayRef.current = nowOffline;
  }, [vm.on_server, vm.relay]);

  const t = vm.telemetry;
  const tr = t?.training ?? null;
  const llm = t?.llm ?? null;
  const hasLlm = isLlmVm(vm);
  const llmReady = isLlmVmReady(vm);
  const hasLlmProc = Boolean(t?.threads?.llm_process);
  const stale = vm.telemetry_age !== null && vm.telemetry_age > 25;
  const isDeleting = vm.busy === "deleting";
  const isCreating = vm.busy === "creating";
  const isDeployingLlm = vm.busy === "deploying_llm";
  const isProbing = probingLocal || vm.busy === "probing";
  const isOfflineRelayRecord = !vm.on_server && vm.relay !== null;
  const isBillingUnknown = Boolean(vm.server_state_unknown || serverStateUnknown);
  const offlineRelayView = isOfflineRelayRecord ? resolveRelayCompletionView(vm.relay) : null;

  const gpu = t?.gpu;
  const rawProcs = t?.threads?.processes;
  const procs = (
    Array.isArray(rawProcs)
      ? rawProcs.filter((p): p is ProcInfo => !!p && typeof p === "object")
      : []
  ).sort((a, b) => num(b?.cpu_percent) - num(a?.cpu_percent));
  const activeProc = liveActiveProcess(vm);
  const kernel = procs.find((p) => str(p.role) === "kernel") ?? null;
  const rawFiles = t?.files?.entries;
  const files = Array.isArray(rawFiles)
    ? rawFiles.filter((f) => !!f && typeof f === "object" && typeof f.name === "string")
    : [];
  const busy = !!(vm.running_task || activeProc || llm?.online || vm.relay?.running);

  // #76：直接使用 relayExpanded，确保点击折叠箭头可真正切换展开与收起状态
  const showRelayDrawer = relayExpanded;
  const canUseWorkloadTools = vm.on_server && !isDeleting && !isCreating;

  const handleRetryProbeClick = async () => {
    if (!onRetryProbe || isProbing) return;
    setProbingLocal(true);
    try {
      await onRetryProbe(vm.name);
    } finally {
      setProbingLocal(false);
    }
  };

  const handleStartRelayClick = async (
    cfg: RelayConfig,
    legacyGdriveAction?: "migrate" | "ignore" | null,
  ) => {
    if (!onStartRelay || relayPending) return;
    setRelayPending(true);
    try {
      // #42 & #80：始终携带当前实例绑定的工作区路径与可选旧目录迁移确认动作
      await onStartRelay(vm.name, cfg, vm.workspace?.path ?? null, legacyGdriveAction ?? null);
    } finally {
      setRelayPending(false);
    }
  };

  const handleDismissRelayClick = async () => {
    if (!onDismissRelay || dismissingRelay) return;
    setDismissingRelay(true);
    try {
      await onDismissRelay(vm.name);
    } finally {
      setDismissingRelay(false);
    }
  };

  return (
    <section
      className={`vm-card ${busy ? "relay-on" : ""} ${isOfflineRelayRecord ? "offline-relay-card" : ""} ${isDeleting ? "deleting" : ""}`}
    >
      {/* 头部：会话名 · 硬件 · 状态 · 工作区与运行时长 · 部署/对话/关机按钮 */}
      <div className="vm-head">
        <span className="vm-name">{vm.name}</span>
        <span className="chip chip-blue">
          {vm.accelerator}
          {vm.high_ram ? " · High-RAM" : ""}
        </span>
        {isOfflineRelayRecord && (
          <span
            className={`chip ${isBillingUnknown ? "chip-red" : ""}`.trim()}
            title={
              isBillingUnknown
                ? "服务端实例列表查询失败，暂无法确认云端实例是否已停费释放"
                : "云端实例已释放下线，保留最近一次接力任务结果与日志"
            }
          >
            {isBillingUnknown ? "云端计费状态未知 · 接力记录" : "云端实例已下线 · 接力记录"}
          </span>
        )}
        {vm.orphan && (
          <span className="chip chip-red" title="浏览器或外部会话创建的云端实例（已自动接管监控与远程终端）">
            外部会话
          </span>
        )}
        {statusPill(vm, activeProc, isBillingUnknown)}
        <span className="vm-meta">
          <span title={vm.workspace.path ?? ""}>
            <FolderGit2 size={12} />
            {vm.workspace.name}
          </span>
          {vm.on_server && (
            <span>
              <Clock size={12} />
              {fmtDuration(t?.uptime_seconds)}
            </span>
          )}
          {vm.on_server && stale && (
            <span className="warn">遥测延迟 {vm.telemetry_age?.toFixed(0)}s</span>
          )}
        </span>

        {canUseWorkloadTools && !llmReady && onDeployLlm && (
          <button
            type="button"
            className="btn btn-sm"
            onClick={() => onDeployLlm(vm)}
            disabled={isDeployingLlm}
            title={
              isDeployingLlm
                ? `${vm.name} 正在后台拉取并部署大模型…`
                : vm.deploy_error
                  ? `上次部署失败，点击重新配置并在 ${vm.name} 上部署大模型`
                  : `在当前运行实例 (${vm.name}) 上直接部署大模型并开启专属 OpenAI API 与 Key`
            }
          >
            <Sparkles size={13} color="var(--primary)" />
            <span>{vm.deploy_error || hasLlm ? "重新部署大模型" : "部署大模型"}</span>
          </button>
        )}

        {hasLlm && !llm && onOpenLlmChat && (
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => onOpenLlmChat(vm.name)}
            disabled={!llmReady}
            title={
              llmReady
                ? `打开与 ${vm.name} 的大模型对话窗口`
                : isDeployingLlm
                  ? `${vm.name} 正在后台拉取权重并预热大模型服务，就绪后自动开放对话`
                  : vm.deploy_error
                    ? `大模型部署异常：${vm.deploy_error}`
                    : `${vm.name} 的大模型推理后端尚未就绪`
            }
          >
            <MessageSquare size={13} />
            <span>{isDeployingLlm ? "部署预热中…" : llmReady ? "大模型对话" : "等待模型就绪"}</span>
          </button>
        )}

        {vm.on_server && onShutdown && (
          <button
            type="button"
            className="btn btn-danger-ghost btn-sm"
            onClick={() => onShutdown(vm)}
            disabled={isDeleting}
            title={`关机并停止计费 (${vm.name})`}
          >
            {isDeleting ? (
              <>
                <Loader2 size={13} className="spin" />
                <span>关机中…</span>
              </>
            ) : (
              <>
                <PowerOff size={13} />
                <span>关机</span>
              </>
            )}
          </button>
        )}

        {isOfflineRelayRecord && !vm.relay?.running && onDismissRelay && (
          <button
            type="button"
            className="btn btn-sm"
            onClick={handleDismissRelayClick}
            disabled={dismissingRelay}
            title="清除此已结束/已释放的离线接力任务记录"
          >
            {dismissingRelay ? <Loader2 size={13} className="spin" /> : <Trash2 size={13} />}
            <span>清除记录</span>
          </button>
        )}
      </div>

      {/* 每台在线虚拟机均提供免密 SSH 连接栏 */}
      {vm.on_server && (
        <SshAccessBar
          vmName={vm.name}
          ssh={vm.ssh}
          orphan={vm.orphan}
          onServer={vm.on_server}
          onOpenSshTerminal={onOpenSshTerminal}
        />
      )}

      <div className="vm-main">
        {/* 部署异常横幅（可见、可复制、可一键重新部署） */}
        {vm.deploy_error && (
          <div className="vm-alert-box vm-alert-danger" role="alert">
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <div className="vm-alert-content">
              <div className="vm-alert-title">云端任务 / 大模型部署异常</div>
              <div className="vm-alert-desc mono">{vm.deploy_error}</div>
            </div>
            <div className="vm-alert-actions">
              <CopyBtn text={vm.deploy_error} label="复制报错" compact />
              {onDeployLlm && vm.on_server && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => onDeployLlm(vm)}
                  disabled={isDeployingLlm || isDeleting}
                >
                  <Sparkles size={12} color="var(--primary)" />
                  <span>重新部署大模型</span>
                </button>
              )}
            </div>
          </div>
        )}

        {llm && (
          <LlmSupervisionBox
            llm={llm}
            vmName={vm.name}
            vmBusy={vm.busy}
            deployError={vm.deploy_error}
            llmApi={vm.llm_api}
            onOpenChat={onOpenLlmChat}
            onRotateKey={onRotateLlmKey}
            onRetryDeploy={onDeployLlm ? () => onDeployLlm(vm) : undefined}
          />
        )}
        {!llm && vm.llm_api && (
          <LlmApiAccessBox
            apiInfo={vm.llm_api}
            busy={vm.busy}
            llm={llm}
            deployError={vm.deploy_error}
            hasLlmProc={hasLlmProc}
            onRotateKey={onRotateLlmKey}
            onRetryDeploy={onDeployLlm ? () => onDeployLlm(vm) : undefined}
          />
        )}
        {tr && (
          <TrainingProgressBox
            tr={tr}
            hasCkpt={!!t?.files?.has_checkpoint}
            ckptMb={t?.files?.checkpoint_size_mb ?? 0}
          />
        )}

        {t ? (
          <>
            {stale && vm.telemetry_error && (
              <div className="vm-alert-box vm-alert-warn" role="status">
                <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                <div className="vm-alert-content">
                  <span className="vm-alert-title">遥测同步延迟（展示上次快照）：</span>
                  <span className="mono">{vm.telemetry_error}</span>
                </div>
                {onRetryProbe && vm.on_server && (
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={handleRetryProbeClick}
                    disabled={isProbing || isDeleting}
                  >
                    {isProbing ? <Loader2 size={12} className="spin" /> : <RefreshCw size={12} />}
                    <span>{isProbing ? "注入探针中…" : "重试连接探针"}</span>
                  </button>
                )}
              </div>
            )}
            <div className="mini-meters">
              <MiniMeter
                name="CPU"
                value={num(t.cpu?.percent)}
                display={`${num(t.cpu?.percent).toFixed(0)}% · ${num(t.cpu?.cores)} 核`}
              />
              <MiniMeter
                name="内存"
                value={num(t.memory?.percent)}
                display={`${num(t.memory?.used_gb)} / ${num(t.memory?.total_gb)} GB`}
              />
              {gpu?.available && (
                <MiniMeter
                  name="GPU"
                  value={num(gpu.gpu_util_percent)}
                  display={`${num(gpu.gpu_util_percent).toFixed(0)}%`}
                />
              )}
              {gpu?.available && num(gpu.vram_total_mb) > 0 ? (
                <MiniMeter
                  name="显存"
                  value={num(gpu.vram_percent)}
                  display={`${(num(gpu.vram_used_mb) / 1024).toFixed(1)} / ${(num(gpu.vram_total_mb) / 1024).toFixed(0)} GB`}
                />
              ) : null}
              <MiniMeter
                name="磁盘"
                value={num(t.disk?.percent)}
                display={`${num(t.disk?.used_gb)} / ${num(t.disk?.total_gb)} GB`}
              />
            </div>
          </>
        ) : vm.telemetry_error && !isCreating ? (
          <div className="vm-alert-box vm-alert-warn" role="alert">
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <div className="vm-alert-content">
              <div className="vm-alert-title">遥测探针连接异常</div>
              <div className="vm-alert-desc mono">{vm.telemetry_error}</div>
            </div>
            <div className="vm-alert-actions">
              <CopyBtn text={vm.telemetry_error} label="复制" compact />
              {onRetryProbe && vm.on_server && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={handleRetryProbeClick}
                  disabled={isProbing || isDeleting}
                  title="重新向云端实例静默注入遥测守护探针并拉取硬件指标"
                >
                  {isProbing ? <Loader2 size={12} className="spin" /> : <RefreshCw size={12} />}
                  <span>{isProbing ? "注入探针中…" : "重试连接探针"}</span>
                </button>
              )}
            </div>
          </div>
        ) : isOfflineRelayRecord ? (
          <div
            className={`vm-alert-box ${
              offlineRelayView?.isError || offlineRelayView?.isDownloadFailed
                ? "vm-alert-danger"
                : "vm-alert-warn"
            }`}
            role="status"
          >
            <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
            <div className="vm-alert-content">
              <div className="vm-alert-title">
                {vm.relay?.running
                  ? isBillingUnknown
                    ? `接力处理中 · 云端计费状态未知（${vm.relay.phase_label || "正在推进流程"}）`
                    : `接力换棒处理中（${vm.relay.phase_label || "正在分配下一棒实例"}）`
                  : isBillingUnknown
                    ? `本地接力记录 · 云端计费状态未知（${offlineRelayView?.stateLabel ?? "已停止"}）`
                    : `已结束/已释放接力任务记录 · ${offlineRelayView?.stateLabel ?? "实例已下线"}`}
              </div>
              <div className="vm-alert-desc">
                {vm.relay?.running
                  ? isBillingUnknown
                    ? "当前服务端计费列表查询失败，暂无法确认云端实例状态，本机监工正在推进流程。"
                    : "上一棒云端实例已释放，本机监工正在推进下一棒流程。"
                  : isBillingUnknown
                    ? `服务端计费列表查询失败，当前云端实例计费状态未知（待服务端查询恢复并确认释放后，再显示已停费）。${offlineRelayView?.sub ?? ""}`
                    : `云端 Colab 实例已下线释放（不再产生计费）。${offlineRelayView?.sub ?? ""}`}
              </div>
            </div>
            {!vm.relay?.running && onDismissRelay && (
              <div className="vm-alert-actions">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={handleDismissRelayClick}
                  disabled={dismissingRelay}
                >
                  <Trash2 size={12} />
                  <span>清除记录</span>
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="probe-line">
            <Loader2 size={14} className="spin" />
            <span>
              {isCreating
                ? "正在 Google Colab 云端分配算力实例并配置免密 SSH…"
                : isDeployingLlm
                  ? "正在云端拉取大模型权重并启动推理服务（首次预热约需 1–5 分钟）…"
                  : "正在连接遥测…"}
            </span>
            {!isCreating && onRetryProbe && vm.on_server && (
              <button
                type="button"
                className="copy-chip-btn"
                style={{ marginLeft: "auto" }}
                onClick={handleRetryProbeClick}
                disabled={isProbing || isDeleting}
                title="立即重新触发远程遥测守护探针注入"
              >
                {isProbing ? <Loader2 size={11} className="spin" /> : <RefreshCw size={11} />}
                <span>{isProbing ? "注入中…" : "重试连接探针"}</span>
              </button>
            )}
          </div>
        )}

        {(vm.running_task || activeProc) && (
          <div className="task-line" title={activeProc?.command ?? ""}>
            <Terminal size={13} />
            <span className="task-cmd">{vm.running_task ? `colab ${vm.running_task}` : activeProc?.command}</span>
            {activeProc && (
              <span className="task-meta">
                PID {activeProc.pid} · {activeProc.elapsed}
              </span>
            )}
          </div>
        )}

        {/* 新建/通用实例的任务控制面板（集成 ScriptDialog 与 RelayPanel，支持配置脚本与跑训练） */}
        {onStartRelay && (!llmReady || Boolean(vm.relay)) && !isCreating && (
          <div className="vm-tools-drawer">
            <div className="vm-tools-head">
              <button
                type="button"
                className="vm-tools-toggle"
                aria-expanded={showRelayDrawer}
                onClick={() => setRelayExpanded((v) => !v)}
              >
                <Repeat2 size={13} color="var(--primary)" />
                <span>接力训练 / 脚本配置</span>
                {vm.relay?.running ? (
                  <span className="chip chip-blue">运行中 · 第 {vm.relay.round || 1} 棒</span>
                ) : isOfflineRelayRecord ? (
                  <span className="chip">
                    {offlineRelayView?.stateLabel ?? "已结束接力记录"}
                  </span>
                ) : (
                  <span className="dim" style={{ fontWeight: 400, fontSize: 11.5 }}>
                    配置 `train.py` / `requirements.txt` 或开启 24h+ Google Drive 跨实例自动换棒训练
                  </span>
                )}
                {showRelayDrawer ? (
                  <ChevronUp size={14} style={{ marginLeft: "auto" }} />
                ) : (
                  <ChevronDown size={14} style={{ marginLeft: "auto" }} />
                )}
              </button>

              {onOpenScriptDialog && (
                <button
                  type="button"
                  className="copy-chip-btn"
                  onClick={() => onOpenScriptDialog(vm)}
                  title={`打开编辑器配置工作区 (${vm.workspace.path ?? vm.workspace.name}) 的 train.py 与 requirements.txt`}
                >
                  <FileCode2 size={12} />
                  <span>编辑训练脚本</span>
                </button>
              )}
            </div>

            {showRelayDrawer && (
              <div className="vm-tools-body">
                <RelayPanel
                  vm={vm}
                  allVms={allVms}
                  pending={relayPending}
                  onStart={handleStartRelayClick}
                  onStopRequest={() => onRequestStopRelay?.(vm)}
                />
                {vm.relay?.logs && vm.relay.logs.length > 0 && (
                  <details className="relay-logs-details" open={isOfflineRelayRecord || undefined}>
                    <summary>接力监工日志 ({vm.relay.logs.length})</summary>
                    <pre className="llm-snippet-pre mono" style={{ maxHeight: 220, overflowY: "auto" }}>
                      {vm.relay.logs.slice(-200).join("\n")}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* 详情：默认折叠 */}
      <details className="vm-details">
        <summary>
          详情
          <span className="dim">
            · {procs.length} 进程 · {vm.cli_events.length} 条 CLI 记录 · {files.length} 个文件
          </span>
        </summary>

        <div className="details-body">
          <div className="kv">
            <KV wide label="本地工作区" value={vm.workspace.path ?? "—"} mono title={vm.workspace.path ?? ""} />
            <KV label="节点" value={shortEndpoint(vm.endpoint)} mono title={vm.endpoint ?? ""} />
            <KV label="云端目录" value={vm.workspace.remote_cwd || "/content"} mono />
            {vm.ssh && (
              <KV
                label="SSH 桥接"
                value={`127.0.0.1:${vm.ssh.local_port} (${vm.ssh.host_alias})`}
                mono
                title={vm.ssh.colab_ssh_command}
              />
            )}
            <KV
              label="Kernel"
              value={kernel ? `PID ${kernel.pid}` : vm.kernel_id ? vm.kernel_id.slice(0, 10) : "未连接"}
              mono
            />
            <KV
              label="断点 / 结果"
              value={
                t?.files?.has_checkpoint
                  ? `ckpt ${t.files.checkpoint_size_mb} MB${t.files.has_results_dir ? " + results/" : ""}`
                  : t?.files?.has_results_dir
                    ? "results/"
                    : "无"
              }
              mono
            />
            {t && (
              <KV
                label="CPU 型号"
                value={
                  <>
                    <Cpu size={11} /> {str(t.cpu?.model) || "—"}
                  </>
                }
                title={str(t.cpu?.model)}
              />
            )}
          </div>

          {vm.cli_events.length > 0 && (
            <div className="details-block">
              <div className="details-label">CLI 记录</div>
              <div className="cli-feed">
                {vm.cli_events.map((ev, idx) => (
                  <div className={`cli-event ${ev.has_error ? "err" : ""}`} key={idx}>
                    <div className="cli-event-top">
                      <span className={eventBadgeCls(ev.event_type, ev.has_error)}>{ev.label}</span>
                      <span className="cli-event-sum" title={ev.detail ?? ev.summary}>
                        {ev.summary}
                      </span>
                      <span className="cli-event-time">{fmtIsoTime(ev.timestamp)}</span>
                    </div>
                    {ev.output_preview && (
                      <div className="cli-event-out" title={ev.output_preview}>
                        {ev.output_preview}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {files.length > 0 && (
            <div className="details-block">
              <div className="details-label">/content 文件</div>
              <div className="file-pills">
                {files.map((f) => (
                  <span key={f.name} className="file-pill">
                    {f.is_dir ? `${f.name}/` : f.name}
                    {!f.is_dir && f.size_mb > 0 ? <em>{f.size_mb} MB</em> : null}
                  </span>
                ))}
              </div>
            </div>
          )}

          {procs.length > 0 && (
            <div className="details-block">
              <div className="details-label">
                进程 <span className="dim">· {t?.threads?.total_threads ?? 0} 线程</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>PID</th>
                      <th>角色</th>
                      <th>CPU%</th>
                      <th>内存</th>
                      <th>耗时</th>
                      <th>命令</th>
                    </tr>
                  </thead>
                  <tbody>
                    {procs.map((p) => (
                      <tr key={p.pid} className={activeProc?.pid === p.pid ? "row-active" : ""}>
                        <td>{p.pid}</td>
                        <td>
                          <span className={`role role-${p.role}`}>{p.role}</span>
                        </td>
                        <td>{p.cpu_percent}%</td>
                        <td>{p.rss_mb} MB</td>
                        <td>{p.elapsed}</td>
                        <td className="cmd" title={p.command}>
                          {p.command}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </details>
    </section>
  );
}
