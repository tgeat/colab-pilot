import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, ShieldAlert, ShieldCheck, Trash2 } from "lucide-react";
import { ColabIcon, Modal, KV } from "./ui";
import type { TerminateReceipt, VmView } from "../types";

type Stage = "confirm" | "running" | "done";

export function DeleteVmDialog(props: {
  vm: VmView | null;
  onClose: () => void;
  onDelete: (vm: VmView) => Promise<TerminateReceipt>;
}) {
  const [stage, setStage] = useState<Stage>("confirm");
  const [receipt, setReceipt] = useState<TerminateReceipt | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (props.vm) {
      setStage("confirm");
      setReceipt(null);
      setErr(null);
    }
  }, [props.vm?.name]);

  if (!props.vm) return null;
  const vm = props.vm;

  const run = async () => {
    setStage("running");
    try {
      setReceipt(await props.onDelete(vm));
    } catch (e) {
      setErr(String(e));
    }
    setStage("done");
  };

  const ok = receipt?.verified_unassigned;

  return (
    <Modal
      open
      onClose={props.onClose}
      closable={stage !== "running"}
      icon={stage === "done" ? (ok ? <ShieldCheck size={20} color="var(--success)" /> : <ShieldAlert size={20} color="var(--danger)" />) : <ColabIcon size={24} />}
      iconBg={stage === "done" && ok ? "var(--success-50)" : "var(--danger-50)"}
      title={stage === "done" ? (ok ? "已彻底销毁，计费已停止" : "销毁未通过核验") : `删除虚拟机 ${vm.name}`}
      sub={vm.endpoint ?? undefined}
      footer={
        stage === "confirm" ? (
          <>
            <button className="btn" onClick={props.onClose}>取消</button>
            <button className="btn btn-danger" onClick={run}><Trash2 size={15} />彻底删除并停止计费</button>
          </>
        ) : stage === "running" ? (
          <button className="btn" disabled><Loader2 size={15} className="spin" />正在销毁并核验…</button>
        ) : (
          <>
            {!ok && <button className="btn btn-danger" onClick={run}>重试销毁</button>}
            <button className="btn btn-primary" onClick={props.onClose}>完成</button>
          </>
        )
      }
    >
      {stage !== "done" && (
        <>
          <div style={{ fontSize: 13, color: "var(--text-2)" }}>
            将执行以下步骤；<b>只有 Google Colab 服务端计费列表确认该实例已消失</b>，才会判定删除成功，杜绝“界面已删、云端仍在扣费”的假性删除。
          </div>
          <ul className="verify-steps">
            {[
              vm.relay?.running ? "停止本机接力监工任务，并结束 colab exec 进程树" : "检查并停止本机关联的接力任务与子进程",
              "colab stop：关闭远程 Jupyter Kernel 并释放运行时",
              "直接调用服务端 /tun/m/unassign/{endpoint} 兜底释放",
              "轮询 /tun/m/assignments，确认 endpoint 已从计费列表中移除",
            ].map((s, i) => (
              <li key={i}>
                <span className="n">{stage === "running" ? <Loader2 size={12} className="spin" /> : i + 1}</span>
                {s}
              </li>
            ))}
          </ul>
          {vm.relay?.running && (
            <div className="banner banner-warn" style={{ margin: 0 }}>
              该实例正在接力训练。最近一次网盘热备之后的训练进度将会丢失。
            </div>
          )}
        </>
      )}

      {stage === "done" && (
        <>
          {err && <div className="receipt bad">{err}</div>}
          {receipt && (
            <>
              <div className={`receipt ${ok ? "ok" : "bad"}`} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                {ok ? <CheckCircle2 size={16} /> : <ShieldAlert size={16} />}
                {ok
                  ? "服务端二次核验通过：该实例已不在 Google Colab 计费列表中。"
                  : receipt.error ?? "服务端仍存在该实例"}
              </div>
              <div className="receipt-grid">
                <KV label="剩余计费实例" value={`${receipt.ccu?.assignments_count ?? receipt.remaining_endpoints.length} 台`} mono />
                <KV
                  label="当前计费速率"
                  value={`${(receipt.ccu?.burn_rate_hourly ?? 0).toFixed(2)} CU/h`}
                  mono
                  cls={(receipt.ccu?.burn_rate_hourly ?? 0) > 0 ? "warn" : "ok"}
                />
                <KV label="账户余额" value={receipt.ccu ? `${receipt.ccu.balance.toFixed(2)} CU` : "—"} mono />
              </div>
              <div className="step-log">
                {receipt.steps.map((s, i) => <div key={i}>{i + 1}. {s}</div>)}
              </div>
            </>
          )}
        </>
      )}
    </Modal>
  );
}
