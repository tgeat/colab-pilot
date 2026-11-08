import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import { AlertTriangle, ArrowRightLeft, GitCompare, History } from "lucide-react";
import { api, errText } from "./api";
import type { LlmBlueGreenState, Overview, VmView } from "./types";
import { Header } from "./components/Header";
import { VmCard } from "./components/VmCard";
import { ColabIcon, ToastProvider, useToast } from "./components/ui";

function fmtLocalTime(iso: string | null) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", { hour12: false });
}

function LlmBlueGreenBanner({
  bg,
  vms,
}: {
  bg: LlmBlueGreenState | null | undefined;
  vms: VmView[];
}) {
  const prefix = bg?.session_prefix ?? "llm";
  const blueVm = vms.find(
    (v) => v.telemetry?.llm?.slot === "blue" || v.name === `${prefix}-blue` || v.name.endsWith("-blue"),
  );
  const greenVm = vms.find(
    (v) => v.telemetry?.llm?.slot === "green" || v.name === `${prefix}-green` || v.name.endsWith("-green"),
  );

  if (!bg && !blueVm?.telemetry?.llm && !greenVm?.telemetry?.llm) {
    return null;
  }

  const activeSlot = (
    bg?.active_slot ??
    (greenVm?.telemetry?.llm?.role === "active" ? "green" : "blue")
  ).toLowerCase();

  const renderSlotCard = (slot: "blue" | "green", vm: VmView | undefined) => {
    const llm = vm?.telemetry?.llm;
    const isOnline = !!vm?.on_server && (llm?.online ?? true);
    const role = (
      llm?.role ??
      (isOnline ? (activeSlot === slot ? "active" : "warming") : "standby")
    ).toLowerCase();
    const tps = llm ? (llm.instant_tps || llm.last_request_tps || 0) : 0;
    const avgTps = llm?.avg_tps_60s ?? 0;
    const ttft = llm?.ttft_ms ?? llm?.avg_ttft_ms ?? null;

    return (
      <div className={`bg-slot-card slot-${slot} ${role === "active" ? "is-active" : ""}`}>
        <div className="bg-slot-top">
          <span className={`slot-badge slot-badge-${slot}`}>{slot.toUpperCase()}</span>
          <span className={`slot-role slot-role-${role}`}>
            {role === "active"
              ? "承载流量"
              : role === "warming"
                ? "预热中"
                : role === "draining"
                  ? "排空中"
                  : "未启用"}
          </span>
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
                <div className="bg-m-val">{llm?.total_completion_tokens ?? 0}</div>
              </div>
            </div>
            <div className="bg-slot-meta">
              <span>
                <b>{vm.name}</b> · {vm.accelerator}
              </span>
              <code>{llm?.model_name ?? bg?.model_name ?? "LLM"}</code>
            </div>
          </>
        ) : (
          <div className="bg-slot-offline">未分配实例</div>
        )}
      </div>
    );
  };

  return (
    <section className="matrix-card bg-cluster-card">
      <div className="matrix-head">
        <div className="matrix-title">
          <GitCompare size={16} color="var(--primary)" />
          <span>LLM 蓝绿部署</span>
          <span className="pill pill-green">
            <span className="dot pulse" />
            {activeSlot.toUpperCase()} 在线
          </span>
        </div>
      </div>

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

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      setOv(await api.refreshNow());
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setRefreshing(false);
    }
  }, [toast]);

  useEffect(() => {
    api.getOverview().then((o) => o.timestamp && setOv(o)).catch(() => {});
    refresh();
    const un = listen<Overview>("overview", (e) => setOv(e.payload));
    return () => {
      un.then((f) => f());
    };
  }, [refresh]);

  const env = ov?.env;
  const vms = ov?.vms ?? [];
  const recents = ov?.recent_sessions ?? [];

  return (
    <>
      <Header ov={ov} refreshing={refreshing} onRefresh={refresh} />

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
        <LlmBlueGreenBanner bg={ov?.llm_bluegreen} vms={vms} />

        {vms.length === 0 ? (
          <div className="empty">
            <ColabIcon size={36} />
            <h3>{ov ? "没有运行中的实例" : "正在同步…"}</h3>
            <p className="dim">
              用 <code>colab new -s &lt;名称&gt;</code> 或 Antigravity Skill 启动实例后，这里会自动显示。
            </p>
          </div>
        ) : (
          <div className="vm-list">
            {vms.map((vm) => (
              <VmCard key={vm.name} vm={vm} />
            ))}
          </div>
        )}

        {/* 历史会话：默认折叠 */}
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
                  </tr>
                </thead>
                <tbody>
                  {recents.map((r) => (
                    <tr key={r.session_name}>
                      <td>
                        <b>{r.session_name}</b> <span className="dim">{r.workspace_name}</span>
                      </td>
                      <td>{r.accelerator}</td>
                      <td className="cmd" title={r.last_summary}>
                        {r.last_summary}
                      </td>
                      <td className="dim">{fmtLocalTime(r.last_event_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        )}
      </main>
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
