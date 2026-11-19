import { useCallback, useEffect, useRef, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import {
  AlertTriangle,
  ArrowRightLeft,
  Check,
  Copy,
  FileCode2,
  GitCompare,
  History,
  KeyRound,
  Link2,
  MessageSquare,
  Power,
  PowerOff,
} from "lucide-react";
import { api, errText } from "./api";
import type {
  CreateVmLlmConfig,
  CreateVmRequest,
  LlmBlueGreenState,
  Overview,
  RelayConfig,
  VmView,
} from "./types";
import { num, numOrNull, str } from "./format";
import { isLlmVmReady, resolveBlueGreenStatus, resolveEffectiveHighMem } from "./chatUtils";
import {
  CreateVmDialog,
  DEFAULT_LLM_ENGINE,
  DEFAULT_LLM_MODEL,
  type CreateVmPreset,
} from "./components/CreateVmDialog";
import { DeleteVmDialog } from "./components/DeleteVmDialog";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Header } from "./components/Header";
import { LlmChatModal } from "./components/LlmChatModal";
import { ScriptDialog } from "./components/ScriptDialog";
import { StopRelayDialog } from "./components/StopRelayDialog";
import { VmCard } from "./components/VmCard";
import { ColabIcon, ToastProvider, useToast } from "./components/ui";

function fmtLocalTime(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", { hour12: false });
}

function InlineCopy({ text, label }: { text: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className={`copy-chip-btn compact ${copied ? "copied" : ""}`}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // ignore
        }
      }}
      title={`复制: ${text}`}
    >
      {copied ? <Check size={11} /> : <Copy size={11} />}
      {label && <span>{copied ? "已复制" : label}</span>}
    </button>
  );
}

function LlmBlueGreenBanner({
  bg,
  vms,
  onOpenChat,
  onShutdown,
  onBootSlot,
}: {
  bg: LlmBlueGreenState | null | undefined;
  vms: VmView[];
  onOpenChat: (target: string) => void;
  onShutdown: (vm: VmView) => void;
  onBootSlot: (slot: "blue" | "green") => void;
}) {
  const { blueVm, greenVm, activeSlot, activeVm, clusterState, anyOnlineVm } =
    resolveBlueGreenStatus(bg, vms);

  if (
    !bg &&
    !blueVm?.telemetry?.llm &&
    !greenVm?.telemetry?.llm &&
    !blueVm?.llm_api &&
    !greenVm?.llm_api
  ) {
    return null;
  }

  const fallbackActiveVm = activeVm ?? blueVm ?? greenVm;
  const gatewayUrl =
    bg?.gateway_url || fallbackActiveVm?.llm_api?.gateway_base_url || "http://127.0.0.1:18080/v1";
  const activeKey = fallbackActiveVm?.llm_api?.api_key || bg?.api_key || "";

  const renderSlotCard = (slot: "blue" | "green", vm: VmView | undefined) => {
    const llm = vm?.telemetry?.llm;
    const isOnline = isLlmVmReady(vm);
    const defaultRole = isOnline
      ? activeSlot === slot
        ? "active"
        : "warming"
      : vm?.on_server
        ? "warming"
        : "standby";
    const role = str(llm?.role, defaultRole).toLowerCase();
    const tps = llm ? num(llm.instant_tps) || num(llm.last_request_tps) : 0;
    const avgTps = num(llm?.avg_tps_60s);
    const ttft = numOrNull(llm?.ttft_ms) ?? numOrNull(llm?.avg_ttft_ms);

    return (
      <div className={`bg-slot-card slot-${slot} ${isOnline && role === "active" ? "is-active" : ""}`}>
        <div className="bg-slot-top">
          <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
            <span className={`slot-badge slot-badge-${slot}`}>{slot.toUpperCase()}</span>
            <span className={`slot-role slot-role-${isOnline ? role : vm?.on_server ? "warming" : "standby"}`}>
              {!vm?.on_server
                ? "未启用"
                : vm.deploy_error
                  ? "部署失败"
                  : !isOnline
                    ? "预热中"
                    : role === "active"
                      ? "承载流量"
                      : role === "warming"
                        ? "预热中"
                        : role === "draining"
                          ? "排空中"
                          : "已就绪"}
            </span>
          </div>
          {vm && vm.on_server ? (
            <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() => onOpenChat(vm.name)}
                disabled={!isOnline}
                title={
                  isOnline
                    ? `直接与 ${vm.name} (${slot.toUpperCase()}) 槽位进行对话测试`
                    : `${vm.name} 推理引擎正在预热或未就绪`
                }
              >
                <MessageSquare size={13} color="var(--primary)" />
                <span>对话此槽位</span>
              </button>
              <button
                type="button"
                className="btn btn-danger-ghost btn-sm"
                onClick={() => onShutdown(vm)}
                disabled={vm.busy === "deleting"}
                title={`关机并停止计费 (${vm.name})`}
              >
                <PowerOff size={13} />
                <span>关机</span>
              </button>
            </div>
          ) : (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => onBootSlot(slot)}
              title={`开机并部署 ${slot.toUpperCase()} 槽位大模型实例`}
            >
              <Power size={12} color="var(--primary)" />
              <span>开机此槽位</span>
            </button>
          )}
        </div>

        {vm ? (
          <>
            <div className="bg-slot-metrics">
              <div>
                <div className="bg-m-label">实时</div>
                <div className="bg-m-val">
                  {tps.toFixed(1)} <small>tok/s</small>
                </div>
              </div>
              <div>
                <div className="bg-m-label">60s 均值</div>
                <div className="bg-m-val">
                  {avgTps.toFixed(1)} <small>tok/s</small>
                </div>
              </div>
              <div>
                <div className="bg-m-label">TTFT</div>
                <div className="bg-m-val">{ttft !== null ? `${ttft.toFixed(0)} ms` : "—"}</div>
              </div>
              <div>
                <div className="bg-m-label">累计 Token</div>
                <div className="bg-m-val">{num(llm?.total_completion_tokens)}</div>
              </div>
            </div>
            <div className="bg-slot-meta">
              <span>
                <b>{vm.name}</b> · {vm.accelerator}
                {vm.llm_api ? ` · :${vm.llm_api.local_port}` : ""}
              </span>
              <code>{str(llm?.model_name) || vm.llm_api?.model_name || str(bg?.model_name) || "LLM"}</code>
            </div>
          </>
        ) : (
          <div className="bg-slot-offline">未分配实例（点击右上角“开机此槽位”一键拉起）</div>
        )}
      </div>
    );
  };

  const pillCls =
    clusterState === "online"
      ? "pill pill-green"
      : clusterState === "starting"
        ? "pill pill-amber"
        : "pill pill-gray";
  const pillLabel =
    clusterState === "online"
      ? `${activeSlot.toUpperCase()} 在线`
      : clusterState === "starting"
        ? `${activeSlot.toUpperCase()} 启动中`
        : "集群离线";

  return (
    <section className="matrix-card bg-cluster-card">
      <div className="matrix-head">
        <div className="matrix-title">
          <GitCompare size={16} color="var(--primary)" />
          <span>LLM 蓝绿部署</span>
          <span className={pillCls}>
            <span className={`dot ${clusterState !== "offline" ? "pulse" : ""}`} />
            {pillLabel}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <div className="ssh-cmd-pill" title="蓝绿零中断统一 OpenAI 网关端点">
            <Link2 size={11} />
            <code className="mono">{gatewayUrl}</code>
            <InlineCopy text={gatewayUrl} label="复制网关" />
          </div>
          {activeKey && (
            <div className="ssh-cmd-pill" title="当前活跃大模型 API Key">
              <KeyRound size={11} />
              <code className="mono">
                {activeKey.length > 18 ? `${activeKey.slice(0, 11)}••••${activeKey.slice(-5)}` : activeKey}
              </code>
              <InlineCopy text={activeKey} label="复制 Key" />
            </div>
          )}
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => onOpenChat("auto")}
            disabled={!anyOnlineVm}
            title={
              anyOnlineVm
                ? "打开大模型对话窗口（蓝绿热切换时自动无感路由至当前 ACTIVE 槽位）"
                : "当前没有在线就绪的大模型实例"
            }
          >
            <MessageSquare size={14} />
            <span>打开大模型对话窗口 ({activeSlot.toUpperCase()})</span>
          </button>
        </div>
      </div>

      {bg?.last_error && clusterState !== "online" && (
        <div className="vm-alert-box vm-alert-danger" style={{ marginBottom: 10 }}>
          <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
          <div className="vm-alert-content">
            <span className="vm-alert-title">蓝绿部署异常：</span>
            <span className="mono">{bg.last_error}</span>
          </div>
          <InlineCopy text={bg.last_error} label="复制报错" />
        </div>
      )}

      <div className="bg-cluster-grid">
        {renderSlotCard("blue", blueVm)}
        <div className="bg-switch-arrow">
          <ArrowRightLeft size={18} color="var(--muted)" />
        </div>
        {renderSlotCard("green", greenVm)}
      </div>
    </section>
  );
}

function Dashboard() {
  const toast = useToast();
  const [ov, setOv] = useState<Overview | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [chatTarget, setChatTarget] = useState<string | null>(null);
  const [shutdownVm, setShutdownVm] = useState<VmView | null>(null);
  const [stopRelayVm, setStopRelayVm] = useState<VmView | null>(null);
  const [scriptOpen, setScriptOpen] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [createPreset, setCreatePreset] = useState<CreateVmPreset | null>(null);
  const didInitRef = useRef(false);

  const applyOverview = useCallback((next: Overview) => {
    if (!next) return;
    setOv((prev) => (!prev || next.timestamp >= prev.timestamp ? next : prev));
  }, []);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      applyOverview(await api.refreshNow());
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setRefreshing(false);
    }
  }, [applyOverview, toast]);

  useEffect(() => {
    let disposed = false;
    api
      .getOverview()
      .then((o) => {
        if (!disposed && o.timestamp) applyOverview(o);
      })
      .catch(() => {});

    if (!didInitRef.current) {
      didInitRef.current = true;
      refresh();
    }

    const unPromise = listen<Overview>("overview", (e) => {
      if (!disposed && e.payload) applyOverview(e.payload);
    })
      .then((fn) => {
        if (disposed) {
          fn();
          return null;
        }
        return fn;
      })
      .catch(() => null);

    return () => {
      disposed = true;
      unPromise.then((fn) => fn?.());
    };
  }, [applyOverview, refresh]);

  const openCreateModal = (preset?: CreateVmPreset | null) => {
    setCreatePreset(preset ?? null);
    setCreateOpen(true);
  };

  const handleCreateVm = async (req: CreateVmRequest) => {
    try {
      const createdName = await api.createVm(req);
      if (req.boot_mode === "llm") {
        toast(`实例 [${createdName}] 已开机，正在后台部署大模型…`, "info");
      } else if (req.boot_mode === "relay" || req.auto_relay) {
        toast(`实例 [${createdName}] 已开机，正在启动自动接力训练…`, "info");
      } else {
        toast(`实例 [${createdName}] 已开机分配并自动配置免密 SSH`, "success");
      }
      await refresh();
    } catch (e) {
      toast(errText(e), "error");
      throw e;
    }
  };

  const handleDeployLlmOnVm = async (name: string, config: CreateVmLlmConfig) => {
    try {
      await api.deployLlmOnVm(name, config);
      toast(`已在实例 [${name}] 启动后台大模型部署，预热完成后自动开放对话与 API`, "info");
      await refresh();
    } catch (e) {
      toast(errText(e), "error");
      throw e;
    }
  };

  const handleOpenSshTerminal = async (name: string) => {
    try {
      const cmd = await api.openVmSshTerminal(name);
      toast(`已打开 [${name}] 的交互式远程终端 (${cmd})`, "success");
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const handleRotateLlmKey = async (vmName: string, customKey?: string | null) => {
    try {
      const updated = await api.rotateLlmApiKey(vmName, customKey);
      toast(`已更新 [${vmName}] 的专属 API Key: ${updated.api_key.slice(0, 14)}…`, "success");
      await refresh();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const handleRetryProbe = async (name: string) => {
    try {
      await api.deployProbe(name);
      toast(`已重新触发 [${name}] 的遥测探针注入`, "info");
      await refresh();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const handleStartRelay = async (name: string, config: RelayConfig) => {
    try {
      await api.startRelay(name, config);
      toast(`已在实例 [${name}] 启动自动接力训练监工`, "success");
      await refresh();
    } catch (e) {
      toast(errText(e), "error");
    }
  };

  const handleStopRelay = async (vm: VmView, destroy: boolean) => {
    const receipt = await api.stopRelay(vm.name, destroy);
    toast(destroy ? `已关闭 [${vm.name}] 接力并销毁云端实例` : `已停止 [${vm.name}] 接力监工（保留实例）`, "info");
    await refresh();
    return receipt;
  };

  const env = ov?.env;
  const vms = ov?.vms ?? [];
  const recents = ov?.recent_sessions ?? [];
  const liveShutdownVm = shutdownVm
    ? vms.find((x) => x.name === shutdownVm.name) ?? shutdownVm
    : null;
  const liveStopRelayVm = stopRelayVm
    ? vms.find((x) => x.name === stopRelayVm.name) ?? stopRelayVm
    : null;

  return (
    <>
      <Header
        ov={ov}
        refreshing={refreshing}
        onRefresh={refresh}
        onOpenLlmChat={(target) => setChatTarget(target)}
        onOpenCreateVm={() => openCreateModal(null)}
      />

      <main className="main">
        {ov?.error && (
          <div className="banner banner-danger">
            <AlertTriangle size={16} />
            {ov.error}
          </div>
        )}
        {env && (!env.colab_ok || !env.rclone_ok || !env.rclone_conf_ok) && (
          <div className="banner banner-warn">
            <AlertTriangle size={16} />
            <div>
              {!env.colab_ok && <div>未找到 colab-cli：{env.colab_bin}</div>}
              {!env.rclone_ok && <div>未找到 rclone：{env.rclone_bin}</div>}
              {!env.rclone_conf_ok && <div>未检测到 rclone 授权文件：{env.rclone_conf}</div>}
            </div>
          </div>
        )}

        {/* LLM 蓝绿槽位（仅在存在 LLM 部署时显示） */}
        <ErrorBoundary label="llm-bluegreen">
          <LlmBlueGreenBanner
            bg={ov?.llm_bluegreen}
            vms={vms}
            onOpenChat={(target) => setChatTarget(target)}
            onShutdown={(vm) => setShutdownVm(vm)}
            onBootSlot={(slot) => {
              const slotGpu = ov?.llm_bluegreen?.gpu ?? "T4";
              openCreateModal({
                name: `${ov?.llm_bluegreen?.session_prefix ?? "llm"}-${slot}`,
                accelerator: slotGpu,
                high_mem: resolveEffectiveHighMem(slotGpu, ov?.llm_bluegreen?.high_mem ?? true),
                boot_mode: "llm",
                llm_slot: slot,
                llm_model: ov?.llm_bluegreen?.model_name || DEFAULT_LLM_MODEL,
                llm_engine: ov?.llm_bluegreen?.engine || DEFAULT_LLM_ENGINE,
              });
            }}
          />
        </ErrorBoundary>

        {vms.length === 0 ? (
          <div className="empty">
            <ColabIcon size={38} />
            <h3>{ov ? "当前没有运行中的 Colab 虚拟机" : "正在同步云端实例…"}</h3>
            <p className="dim">
              点击下方按钮选择 GPU / TPU / CPU 型号与 Colab 运行时配置一键开机，每台虚拟机自动提供免密 SSH，大模型虚拟机自动提供 OpenAI API 与 Key。
            </p>
            <div style={{ marginTop: 14, display: "inline-flex", gap: 10, flexWrap: "wrap", justifyContent: "center" }}>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => openCreateModal({ boot_mode: "general", accelerator: "T4" })}
              >
                <Power size={15} />
                <span>开机 / 新建通用 GPU 实例</span>
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => openCreateModal({ boot_mode: "llm", accelerator: "T4", llm_slot: "blue" })}
              >
                <KeyRound size={15} color="var(--primary)" />
                <span>开机并部署大模型 (含 API & Key)</span>
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => setScriptOpen(true)}
                title="编辑工作区 train.py 与 requirements.txt 接力训练脚本"
              >
                <FileCode2 size={15} color="var(--text-2)" />
                <span>配置训练脚本</span>
              </button>
            </div>
          </div>
        ) : (
          <div className="vm-list">
            {vms.map((vm) => (
              <ErrorBoundary key={vm.name} label={`vm:${vm.name}`}>
                <VmCard
                  vm={vm}
                  onOpenLlmChat={(name) => setChatTarget(name)}
                  onShutdown={(v) => setShutdownVm(v)}
                  onOpenSshTerminal={handleOpenSshTerminal}
                  onRotateLlmKey={handleRotateLlmKey}
                  onRetryProbe={handleRetryProbe}
                  onDeployLlm={(v) =>
                    openCreateModal({
                      target_existing_vm: v.name,
                      boot_mode: "llm",
                      llm_slot: v.name.endsWith("-blue")
                        ? "blue"
                        : v.name.endsWith("-green")
                          ? "green"
                          : "standalone",
                      llm_model: v.llm_api?.model_name || DEFAULT_LLM_MODEL,
                      llm_engine: v.llm_api?.engine || DEFAULT_LLM_ENGINE,
                    })
                  }
                  onStartRelay={handleStartRelay}
                  onRequestStopRelay={(v) => setStopRelayVm(v)}
                  onOpenScriptDialog={() => setScriptOpen(true)}
                />
              </ErrorBoundary>
            ))}
          </div>
        )}

        {/* 历史会话：默认折叠，支持按原规格一键重新开机 */}
        {recents.length > 0 && (
          <details className="history">
            <summary>
              <History size={14} />
              历史会话 <span className="dim">({recents.length})</span>
            </summary>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>会话</th>
                    <th>规格</th>
                    <th>最后操作</th>
                    <th>时间</th>
                    <th style={{ width: 96, textAlign: "right" }}>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {recents.map((r) => (
                    <tr key={r.session_name}>
                      <td>
                        <b>{r.session_name}</b> <span className="dim">{r.workspace_name}</span>
                      </td>
                      <td>
                        {r.accelerator}
                        {r.machine_shape === "HIGH_RAM" ? " · High-RAM" : ""}
                      </td>
                      <td className="cmd" title={r.last_summary}>
                        {r.last_summary}
                      </td>
                      <td className="dim">{fmtLocalTime(r.last_event_at)}</td>
                      <td style={{ textAlign: "right" }}>
                        <button
                          type="button"
                          className="btn btn-sm"
                          onClick={() => {
                            const acc = r.accelerator || "T4";
                            openCreateModal({
                              name: r.session_name,
                              accelerator: acc,
                              high_mem: resolveEffectiveHighMem(acc, r.machine_shape === "HIGH_RAM"),
                              boot_mode: r.session_name.startsWith("llm-") ? "llm" : "general",
                              workspace_path: r.workspace_path ?? undefined,
                            });
                          }}
                          title={`使用原规格 (${r.accelerator}) 重新开机启动 ${r.session_name}`}
                        >
                          <Power size={12} color="var(--primary)" />
                          <span>重新开机</span>
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </main>

      <ErrorBoundary label="create-vm">
        <CreateVmDialog
          open={createOpen}
          defaultWorkspace={env?.workspace}
          preset={createPreset}
          onClose={() => setCreateOpen(false)}
          onCreate={handleCreateVm}
          onDeployLlmOnVm={handleDeployLlmOnVm}
        />
      </ErrorBoundary>

      <ErrorBoundary label="script-dialog">
        <ScriptDialog open={scriptOpen} onClose={() => setScriptOpen(false)} />
      </ErrorBoundary>

      <ErrorBoundary label="stop-relay">
        <StopRelayDialog
          vm={liveStopRelayVm}
          onClose={() => setStopRelayVm(null)}
          onStop={handleStopRelay}
        />
      </ErrorBoundary>

      <ErrorBoundary label="llm-chat">
        <LlmChatModal
          open={chatTarget !== null}
          initialTarget={chatTarget ?? "auto"}
          ov={ov}
          onClose={() => setChatTarget(null)}
        />
      </ErrorBoundary>

      <ErrorBoundary label="shutdown-vm">
        <DeleteVmDialog
          vm={liveShutdownVm}
          onClose={() => setShutdownVm(null)}
          onDelete={(v) => api.deleteVm(v.name, v.endpoint)}
        />
      </ErrorBoundary>
    </>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <Dashboard />
    </ToastProvider>
  );
}
