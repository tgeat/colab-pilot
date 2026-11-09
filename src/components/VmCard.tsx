import { Clock, Cloud, Cpu, Database, FolderGit2, Gauge, GitCompare, Loader2, MessageSquare, Sparkles, Terminal } from "lucide-react";
import type { LlmTelemetry, ProcInfo, TrainingTelemetry, VmView } from "../types";
import { fmtDuration, pct, shortEndpoint } from "../format";
import { isLlmVm } from "./LlmChatModal";
import { KV } from "./ui";

/** 云端真正在运行的任务进程（排除僵尸进程 Z 与 Kernel 本身） */
function liveActiveProcess(vm: VmView): ProcInfo | null {
  const ap = vm.telemetry?.threads?.active_process ?? null;
  if (!ap) return null;
  if ((ap.state ?? "").startsWith("Z") || ap.command.includes("<defunct>")) return null;
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

function statusPill(vm: VmView, activeProc: ProcInfo | null) {
  const t = vm.telemetry;
  const llm = t?.llm;
  const tr = t?.training;

  if (vm.busy === "probing") {
    return (
      <span className="pill pill-blue">
        <span className="dot pulse" />
        连接中
      </span>
    );
  }
  if (llm && llm.online) {
    const tps = llm.instant_tps || llm.last_request_tps || 0;
    const cls = llm.role === "warming" ? "pill-amber" : llm.role === "draining" ? "pill-gray" : "pill-green";
    return (
      <span className={`pill ${cls}`}>
        <span className="dot pulse" />
        LLM · {tps.toFixed(1)} tok/s
      </span>
    );
  }
  if (tr && tr.status) {
    const isDone = tr.status === "completed";
    return (
      <span className={`pill ${isDone ? "pill-green" : "pill-blue"}`}>
        <span className={`dot ${isDone ? "" : "pulse"}`} />
        {isDone ? "训练完成" : `训练中 · Epoch ${tr.current_epoch ?? 0}/${tr.total_epochs ?? "?"}`}
      </span>
    );
  }
  if (vm.running_task || activeProc) {
    return (
      <span className="pill pill-blue">
        <span className="dot pulse" />
        运行中
      </span>
    );
  }
  if (vm.on_server) {
    return (
      <span className="pill pill-green">
        <span className="dot" />
        空闲
      </span>
    );
  }
  return (
    <span className="pill pill-gray">
      <span className="dot" />
      离线
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

function TrainingProgressBox({ tr, hasCkpt, ckptMb }: { tr: TrainingTelemetry; hasCkpt: boolean; ckptMb: number }) {
  const cur = tr.current_epoch ?? 0;
  const tot = tr.total_epochs ?? 100;
  const progress = tot > 0 ? Math.min(100, Math.max(0, (cur / tot) * 100)) : 0;
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
  const statusLabel = statusMap[tr.status ?? ""] ?? (tr.status || "运行中");

  return (
    <div className="workload-box dl-box">
      <div className="workload-head">
        <span className="workload-title">
          <Sparkles size={14} color="var(--primary)" />
          训练
          {tr.model_name && <code>{tr.model_name}</code>}
          <span className="dim">· {statusLabel}</span>
        </span>
        <span className="chip chip-blue">Round #{tr.round ?? 1}</span>
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
          <span className="wg-v mono">{tr.loss !== undefined && tr.loss !== null ? tr.loss.toFixed(5) : "—"}</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">ETA</span>
          <span className="wg-v mono">{tr.eta_seconds ? fmtDuration(tr.eta_seconds) : "—"}</span>
        </div>
        <div className="wg-item">
          <span className="wg-k">换棒倒计时</span>
          <span className="wg-v mono">{fmtDuration(tr.handover_in_seconds)}</span>
        </div>
        <div className="wg-item" title={tr.gdrive_dir ?? ""}>
          <span className="wg-k">
            <Cloud size={10} /> 网盘
          </span>
          <span className="wg-v mono">{tr.sync_status ?? (hasCkpt ? `${ckptMb} MB` : "就绪")}</span>
        </div>
      </div>
      {tr.dataset_status && (
        <div className="workload-foot">
          <span>
            <Database size={12} /> 数据集: <b>{tr.dataset_status}</b>
          </span>
          <span>
            <Clock size={12} /> 下次热备: <b>{fmtDuration(tr.next_backup_in_seconds)}</b>
          </span>
        </div>
      )}
    </div>
  );
}

function LlmSupervisionBox({
  llm,
  vmName,
  onOpenChat,
}: {
  llm: LlmTelemetry;
  vmName: string;
  onOpenChat?: (name: string) => void;
}) {
  const slot = (llm.slot ?? "blue").toLowerCase();
  const role = (llm.role ?? "active").toLowerCase();
  const instantTps = llm.instant_tps ?? 0;
  const lastTps = llm.last_request_tps ?? 0;
  const displayTps = instantTps > 0 ? instantTps : lastTps;
  const avgTps = llm.avg_tps_60s ?? 0;
  const peakTps = llm.peak_tps ?? 0;
  const ttft = llm.ttft_ms ?? llm.avg_ttft_ms ?? null;
  const history = llm.tps_history && llm.tps_history.length > 0 ? llm.tps_history : [0, 0, 0, 0, 0, 0, 0, 0];
  const maxHist = Math.max(50, ...history, peakTps);

  return (
    <div className={`workload-box llm-box slot-${slot}`}>
      <div className="workload-head">
        <span className="workload-title">
          <GitCompare size={14} color={slot === "green" ? "var(--success)" : "var(--primary)"} />
          <code>{llm.model_name ?? "LLM"}</code>
          <span className={`slot-badge slot-badge-${slot}`}>{slot.toUpperCase()}</span>
          <span className="dim">· {llm.engine ?? "vLLM"} · {role}</span>
        </span>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 10 }}>
          <span className={`online-dot ${llm.online ? "is-online" : "is-offline"}`}>
            <span className="dot pulse" />
            {llm.online ? "在线" : "启动中"}
          </span>
          {onOpenChat && (
            <button
              className="btn btn-primary btn-sm"
              onClick={() => onOpenChat(vmName)}
              title={`打开与 ${vmName} (${llm.model_name ?? "LLM"}) 的交互对话窗口`}
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
            {displayTps.toFixed(1)} <small>tok/s</small>
          </div>
          <div className="tps-sub">
            {llm.active_streams && llm.active_streams > 0
              ? `${llm.active_streams} 个流式请求进行中`
              : lastTps > 0
                ? `峰值 ${peakTps.toFixed(1)} tok/s`
                : "等待请求"}
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
            {llm.total_completion_tokens ?? 0} ({llm.total_requests ?? 0} 次)
          </span>
        </div>
        <div className="wg-item">
          <span className="wg-k">换棒倒计时</span>
          <span className="wg-v mono">{fmtDuration(llm.handover_in_seconds)}</span>
        </div>
      </div>
    </div>
  );
}

export function VmCard({
  vm,
  onOpenLlmChat,
}: {
  vm: VmView;
  onOpenLlmChat?: (name: string) => void;
}) {
  const t = vm.telemetry;
  const tr = t?.training ?? null;
  const llm = t?.llm ?? null;
  const hasLlm = isLlmVm(vm);
  const stale = vm.telemetry_age !== null && vm.telemetry_age > 25;

  const gpu = t?.gpu;
  const procs = [...(t?.threads?.processes ?? [])].sort((a, b) => b.cpu_percent - a.cpu_percent);
  const activeProc = liveActiveProcess(vm);
  const kernel = procs.find((p) => p.role === "kernel") ?? null;
  const files = t?.files?.entries ?? [];
  const busy = !!(vm.running_task || activeProc || llm?.online);

  return (
    <section className={`vm-card ${busy ? "relay-on" : ""}`}>
      {/* 头部：会话名 · 硬件 · 状态 · 工作区与运行时长 */}
      <div className="vm-head">
        <span className="vm-name">{vm.name}</span>
        <span className="chip chip-blue">
          {vm.accelerator}
          {vm.high_ram ? " · High-RAM" : ""}
        </span>
        {vm.orphan && (
          <span className="chip chip-red" title="服务端存在但本地 sessions.json 中没有的实例">
            外部会话
          </span>
        )}
        {statusPill(vm, activeProc)}
        <span className="vm-meta">
          <span title={vm.workspace.path ?? ""}>
            <FolderGit2 size={12} />
            {vm.workspace.name}
          </span>
          <span>
            <Clock size={12} />
            {fmtDuration(t?.uptime_seconds)}
          </span>
          {stale && <span className="warn">遥测延迟 {vm.telemetry_age?.toFixed(0)}s</span>}
        </span>
        {hasLlm && !llm && onOpenLlmChat && (
          <button
            className="btn btn-primary btn-sm"
            onClick={() => onOpenLlmChat(vm.name)}
            title={`打开与 ${vm.name} 的大模型对话窗口`}
          >
            <MessageSquare size={13} />
            <span>大模型对话</span>
          </button>
        )}
      </div>

      <div className="vm-main">
        {llm && (
          <LlmSupervisionBox
            llm={llm}
            vmName={vm.name}
            onOpenChat={onOpenLlmChat}
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
          <div className="mini-meters">
            <MiniMeter name="CPU" value={t.cpu.percent} display={`${t.cpu.percent.toFixed(0)}% · ${t.cpu.cores} 核`} />
            <MiniMeter name="内存" value={t.memory.percent} display={`${t.memory.used_gb} / ${t.memory.total_gb} GB`} />
            {gpu?.available && (
              <MiniMeter
                name="GPU"
                value={gpu.gpu_util_percent ?? 0}
                display={`${(gpu.gpu_util_percent ?? 0).toFixed(0)}%`}
              />
            )}
            {gpu?.available && gpu.vram_total_mb ? (
              <MiniMeter
                name="显存"
                value={gpu.vram_percent ?? 0}
                display={`${((gpu.vram_used_mb ?? 0) / 1024).toFixed(1)} / ${(gpu.vram_total_mb / 1024).toFixed(0)} GB`}
              />
            ) : null}
            <MiniMeter name="磁盘" value={t.disk.percent} display={`${t.disk.used_gb} / ${t.disk.total_gb} GB`} />
          </div>
        ) : (
          <div className="probe-line">
            <Loader2 size={14} className="spin" />
            正在连接遥测…
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
                    <Cpu size={11} /> {t.cpu.model}
                  </>
                }
                title={t.cpu.model}
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
