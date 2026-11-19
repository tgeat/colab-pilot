import { useEffect, useState } from "react";
import { AlertTriangle, Loader2, PowerOff, ShieldAlert, Square, Trash2 } from "lucide-react";
import { errText, evaluateStopRelayOutcome } from "../api";
import { formatBurnRate, num } from "../format";
import { ColabIcon, KV, Modal, useToast } from "./ui";
import type { StopRelayResult, TerminateReceipt, VmView } from "../types";

export function StopRelayDialog(props: {
  vm: VmView | null;
  onClose: () => void;
  onStop: (
    vm: VmView,
    destroy: boolean,
  ) => Promise<StopRelayResult | TerminateReceipt | null | void>;
}) {
  const toast = useToast();
  const [busy, setBusy] = useState<null | "keep" | "destroy">(null);
  const [failedOutcome, setFailedOutcome] = useState<{
    destroy: boolean;
    message: string;
    result: StopRelayResult;
  } | null>(null);

  useEffect(() => {
    setBusy(null);
    setFailedOutcome(null);
  }, [props.vm?.name]);

  if (!props.vm) return null;
  const vm = props.vm;

  const go = async (destroy: boolean) => {
    setBusy(destroy ? "destroy" : "keep");
    setFailedOutcome(null);
    try {
      const raw = await props.onStop(vm, destroy);
      const outcome = evaluateStopRelayOutcome(
        vm.name,
        destroy,
        raw as StopRelayResult | TerminateReceipt | null | undefined,
      );
      toast(outcome.message, outcome.toastKind);
      if (!outcome.keepDialogOpen) {
        props.onClose();
        return;
      }
      setFailedOutcome({
        destroy,
        message: outcome.message,
        result: outcome.result,
      });
    } catch (e) {
      toast(errText(e), "error");
    } finally {
      setBusy(null);
    }
  };

  const receipt = failedOutcome?.result.terminate_receipt ?? null;
  const receiptBurn = receipt
    ? formatBurnRate(
        receipt.ccu,
        !receipt.verified_unassigned ||
          receipt.remaining_endpoints.length > 0 ||
          (receipt.ccu?.assignments_count ?? 0) > 0,
      )
    : null;

  return (
    <Modal
      open
      onClose={props.onClose}
      closable={!busy}
      icon={<ColabIcon size={24} />}
      iconBg={failedOutcome ? "var(--danger-50)" : "var(--warning-50)"}
      title={
        failedOutcome
          ? failedOutcome.destroy
            ? `停费核验未通过 · ${vm.name}`
            : `未能确认中断云端训练 · ${vm.name}`
          : `关闭 ${vm.name} 的接力训练`
      }
      sub="已同步到 Google Drive 的断点会保留，下次开启自动续训"
      footer={
        failedOutcome ? (
          <>
            {failedOutcome.destroy ? (
              <button
                type="button"
                className="btn btn-danger"
                disabled={!!busy}
                onClick={() => go(true)}
              >
                {busy === "destroy" ? <Loader2 size={14} className="spin" /> : <PowerOff size={14} />}
                <span>重试销毁并核验停费</span>
              </button>
            ) : (
              <button
                type="button"
                className="btn btn-danger-ghost"
                disabled={!!busy}
                onClick={() => go(true)}
              >
                <Trash2 size={14} />
                <span>直接销毁实例并停费</span>
              </button>
            )}
            <button type="button" className="btn btn-primary" disabled={!!busy} onClick={props.onClose}>
              关闭窗口
            </button>
          </>
        ) : undefined
      }
    >
      {failedOutcome ? (
        <>
          <div className="receipt bad" style={{ display: "flex", gap: 8, alignItems: "flex-start" }}>
            <ShieldAlert size={16} style={{ flexShrink: 0, marginTop: 2 }} />
            <div>{failedOutcome.message}</div>
          </div>

          {failedOutcome.destroy && receipt && (
            <>
              {receipt.remaining_endpoints.length > 0 && (
                <div className="banner banner-warn" style={{ margin: 0 }}>
                  <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <b>未释放计费端点：</b>
                    <span className="mono">{receipt.remaining_endpoints.join(", ")}</span>
                  </div>
                </div>
              )}
              <div className="receipt-grid">
                <KV
                  label="剩余计费实例"
                  value={`${receipt.ccu?.assignments_count ?? receipt.remaining_endpoints.length} 台`}
                  mono
                />
                <KV
                  label="当前计费速率"
                  value={receiptBurn?.text ?? "未知"}
                  mono
                  cls={receiptBurn?.cls ?? "warn"}
                />
                <KV
                  label="账户余额"
                  value={receipt.ccu ? `${num(receipt.ccu.balance).toFixed(2)} CU` : "—"}
                  mono
                />
              </div>
              {receipt.steps.length > 0 && (
                <div className="step-log">
                  {receipt.steps.map((s, idx) => (
                    <div key={idx}>
                      {idx + 1}. {s}
                    </div>
                  ))}
                </div>
              )}
            </>
          )}

          {!failedOutcome.destroy && failedOutcome.result.kernel_interrupt_error && (
            <div className="step-log">
              <div>Kernel 中断失败详情：{failedOutcome.result.kernel_interrupt_error}</div>
              <div>建议通过卡片上的 SSH 终端检查云端训练进程，或直接执行关机销毁停费。</div>
            </div>
          )}
        </>
      ) : (
        <div className="choice-list">
          <button type="button" className="choice" disabled={!!busy} onClick={() => go(false)}>
            {busy === "keep" ? <Loader2 size={18} className="spin" /> : <Square size={18} color="var(--warning)" />}
            <div>
              <div style={{ fontWeight: 600 }}>仅关闭接力，保留实例</div>
              <div className="dim" style={{ fontSize: 12 }}>
                停止监工并尝试重启远程 Kernel 中断训练；实例仍在线，<b className="warn">会继续计费</b>。
              </div>
            </div>
          </button>
          <button type="button" className="choice danger" disabled={!!busy} onClick={() => go(true)}>
            {busy === "destroy" ? <Loader2 size={18} className="spin" /> : <Trash2 size={18} color="var(--danger)" />}
            <div>
              <div style={{ fontWeight: 600 }}>关闭接力并彻底销毁实例</div>
              <div className="dim" style={{ fontSize: 12 }}>
                unassign 并二次核验服务端计费列表确认释放，<b className="ok">立即停止计费</b>。
              </div>
            </div>
          </button>
        </div>
      )}
    </Modal>
  );
}

/**
 * 桌面窗口关闭时的活跃接力保护弹窗 (#72)：
 * 明确告知活跃接力实例列表与继续计费影响，提供「取消关闭」「仅退出保留实例」「关机停费后退出」三项明确操作。
 */
export function CloseAppRelayDialog(props: {
  open: boolean;
  activeVms: string[];
  onCancel: () => void;
  onConfirmClose: (mode: "keep_vms" | "shutdown_vms") => Promise<TerminateReceipt[]>;
}) {
  const toast = useToast();
  const [busyMode, setBusyMode] = useState<null | "keep_vms" | "shutdown_vms">(null);
  const [failedReceipts, setFailedReceipts] = useState<TerminateReceipt[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (props.open) {
      setBusyMode(null);
      setFailedReceipts(null);
      setErr(null);
    }
  }, [props.open]);

  if (!props.open) return null;

  const handleAction = async (mode: "keep_vms" | "shutdown_vms") => {
    setBusyMode(mode);
    setErr(null);
    setFailedReceipts(null);
    try {
      const receipts = await props.onConfirmClose(mode);
      if (mode === "shutdown_vms" && Array.isArray(receipts)) {
        const unverified = receipts.filter((r) => !r.verified_unassigned);
        if (unverified.length > 0) {
          setFailedReceipts(unverified);
          toast(
            `有 ${unverified.length} 台云端实例未通过停费核验，已阻止直接退出以免产生隐性扣费`,
            "error",
          );
          return;
        }
      }
    } catch (e) {
      const msg = errText(e);
      setErr(msg);
      toast(msg, "error");
    } finally {
      setBusyMode(null);
    }
  };

  return (
    <Modal
      open={props.open}
      onClose={() => !busyMode && props.onCancel()}
      closable={!busyMode}
      icon={<ShieldAlert size={22} color="var(--warning)" />}
      iconBg="var(--warning-50)"
      title="检测到正在运行的自动接力训练任务"
      sub="请选择关闭窗口前如何处理本机接力监工与云端计费实例"
      footer={
        <button type="button" className="btn" disabled={!!busyMode} onClick={props.onCancel}>
          取消关闭，继续运行接力
        </button>
      }
    >
      <div className="banner banner-warn" style={{ margin: 0 }}>
        <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <div>
            <b>当前活跃接力实例：</b>
            <span className="mono">
              {props.activeVms.length > 0 ? props.activeVms.join(", ") : "运行中接力任务"}
            </span>
          </div>
          <div style={{ marginTop: 4, fontSize: 12.5 }}>
            关闭 Colab Pilot 窗口将退出本机自动换棒监工进程；若不销毁云端实例，Google Colab 云端虚拟机仍会继续运行并持续扣除计算单元（CU）。
          </div>
        </div>
      </div>

      {err && <div className="receipt bad">{err}</div>}

      {failedReceipts && failedReceipts.length > 0 && (
        <div className="receipt bad" style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ fontWeight: 700 }}>
            ⚠️ 以下实例关机未通过服务端停费核验（可能仍在计费）：
          </div>
          {failedReceipts.map((r) => {
            const br = formatBurnRate(r.ccu, true);
            return (
              <div
                key={r.name}
                style={{
                  background: "var(--surface)",
                  border: "1px solid #f6c4c6",
                  borderRadius: 8,
                  padding: "8px 10px",
                  color: "var(--text)",
                }}
              >
                <div>
                  <b className="mono">{r.name}</b> · {r.error || "服务端仍存在活跃分配"} · 计费速率:{" "}
                  <span className={`mono ${br.cls}`}>{br.text}</span>
                </div>
                {r.remaining_endpoints.length > 0 && (
                  <div className="dim mono" style={{ fontSize: 11.5, marginTop: 2 }}>
                    未释放端点: {r.remaining_endpoints.join(", ")}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="choice-list">
        <button
          type="button"
          className="choice danger"
          disabled={!!busyMode}
          onClick={() => handleAction("shutdown_vms")}
        >
          {busyMode === "shutdown_vms" ? (
            <Loader2 size={18} className="spin" />
          ) : (
            <PowerOff size={18} color="var(--danger)" />
          )}
          <div>
            <div style={{ fontWeight: 600 }}>停止接力并关机停费后退出（推荐）</div>
            <div className="dim" style={{ fontSize: 12 }}>
              对所有活跃接力实例执行四步关机与服务端停费核验，确认零遗留扣费后安全退出软件。
            </div>
          </div>
        </button>

        <button
          type="button"
          className="choice"
          disabled={!!busyMode}
          onClick={() => handleAction("keep_vms")}
        >
          {busyMode === "keep_vms" ? (
            <Loader2 size={18} className="spin" />
          ) : (
            <Square size={18} color="var(--warning)" />
          )}
          <div>
            <div style={{ fontWeight: 600 }}>仅退出软件（停止自动换棒，保留云端实例继续计费）</div>
            <div className="dim" style={{ fontSize: 12 }}>
              仅结束桌面端接力监工；云端实例不会销毁，也不会在超时前自动换棒，<b className="warn">将继续产生计费</b>。
            </div>
          </div>
        </button>
      </div>
    </Modal>
  );
}
