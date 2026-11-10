import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, PowerOff, ShieldAlert, ShieldCheck } from "lucide-react";
import { ColabIcon, Modal, KV } from "./ui";
import type { TerminateReceipt, VmView } from "../types";
import { num } from "../format";

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
  const isLlm = !!vm.telemetry?.llm || vm.name.startsWith("llm-");
  const isTraining = !!vm.telemetry?.training || !!vm.relay?.running;

  const run = async () => {
    setStage("running");
    setErr(null);
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
      icon={
        stage === "done" ? (
          ok ? (
            <ShieldCheck size={20} color="var(--success)" />
          ) : (
            <ShieldAlert size={20} color="var(--danger)" />
          )
        ) : (
          <ColabIcon size={24} />
        )
      }
      iconBg={stage === "done" && ok ? "var(--success-50)" : "var(--danger-50)"}
      title={
        stage === "done"
          ? ok
            ? "已彻底关机，服务端计费已停止"
            : "关机未通过服务端核验"
          : `关机并停止计费 · ${vm.name}`
      }
      sub={`${vm.accelerator}${vm.high_ram ? " · High-RAM" : ""}${vm.endpoint ? ` · ${vm.endpoint}` : ""}`}
      footer={
        stage === "confirm" ? (
          <>
            <button className="btn" onClick={props.onClose}>
              取消
            </button>
            <button className="btn btn-danger" onClick={run}>
              <PowerOff size={15} />
              立即关机并停止计费
            </button>
          </>
        ) : stage === "running" ? (
          <button className="btn" disabled>
            <Loader2 size={15} className="spin" />
            正在关机并核验计费状态…
          </button>
        ) : (
          <>
            {!ok && (
              <button className="btn btn-danger" onClick={run}>
                重试关机
              </button>
            )}
            <button className="btn btn-primary" onClick={props.onClose}>
              完成
            </button>
          </>
        )
      }
    >
      {stage !== "done" && (
        <>
          <div style={{ fontSize: 13, color: "var(--text-2)" }}>
            所有 Colab 虚拟机（大模型推理、接力训练、普通会话或外部孤儿实例）均采用<b>统一四步关机核验流程</b>；
            <b>只有 Google Colab 服务端计费列表确认该实例已消失</b>，才会判定关机成功，彻底杜绝“假性关机仍在扣费”。
          </div>
          <ul className="verify-steps">
            {[
              isTraining
                ? "停止本机接力监工任务，并结束关联的 colab exec 进程树"
                : isLlm
                  ? "清理或切换本机 LLM 蓝绿槽位状态与本地网关隧道"
                  : "检查并停止本机关联的任务与子进程",
              "执行 colab stop：关闭远程 Jupyter Kernel 并释放运行时",
              "直接调用 Google Colab 服务端 /tun/m/unassign/{endpoint} 强制释放",
              "轮询 /tun/m/assignments 二次核验，确认实例已从服务端计费列表中移除",
            ].map((s, i) => (
              <li key={i}>
                <span className="n">
                  {stage === "running" ? <Loader2 size={12} className="spin" /> : i + 1}
                </span>
                {s}
              </li>
            ))}
          </ul>
          {isTraining && (
            <div className="banner banner-warn" style={{ margin: 0 }}>
              该实例正在运行深度学习训练。关机后，最近一次 Google Drive 热备之后的未同步进度将会丢失。
            </div>
          )}
          {isLlm && (
            <div className="banner banner-warn" style={{ margin: 0 }}>
              该实例承载着云端大模型推理服务（{vm.telemetry?.llm?.model_name ?? vm.name}），关机后将立即释放 GPU 显存并停止计费。
            </div>
          )}
        </>
      )}

      {stage === "done" && (
        <>
          {err && <div className="receipt bad">{err}</div>}
          {receipt && (
            <>
              <div
                className={`receipt ${ok ? "ok" : "bad"}`}
                style={{ display: "flex", gap: 8, alignItems: "center" }}
              >
                {ok ? <CheckCircle2 size={16} /> : <ShieldAlert size={16} />}
                {ok
                  ? "服务端二次核验通过：该实例已从 Google Colab 计费列表中彻底移除。"
                  : receipt.error ?? "服务端仍存在该实例，请点击重试关机"}
              </div>
              <div className="receipt-grid">
                <KV
                  label="剩余计费实例"
                  value={`${receipt.ccu?.assignments_count ?? receipt.remaining_endpoints.length} 台`}
                  mono
                />
                <KV
                  label="当前计费速率"
                  value={`${num(receipt.ccu?.burn_rate_hourly).toFixed(2)} CU/h`}
                  mono
                  cls={(receipt.ccu?.burn_rate_hourly ?? 0) > 0 ? "warn" : "ok"}
                />
                <KV
                  label="账户余额"
                  value={receipt.ccu ? `${num(receipt.ccu.balance).toFixed(2)} CU` : "—"}
                  mono
                />
              </div>
              <div className="step-log">
                {receipt.steps.map((s, i) => (
                  <div key={i}>
                    {i + 1}. {s}
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </Modal>
  );
}
