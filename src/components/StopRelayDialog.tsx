import { useState } from "react";
import { Loader2, Square, Trash2 } from "lucide-react";
import { ColabIcon, Modal } from "./ui";
import type { VmView } from "../types";

export function StopRelayDialog(props: {
  vm: VmView | null;
  onClose: () => void;
  onStop: (vm: VmView, destroy: boolean) => Promise<void>;
}) {
  const [busy, setBusy] = useState<null | "keep" | "destroy">(null);
  if (!props.vm) return null;
  const vm = props.vm;

  const go = async (destroy: boolean) => {
    setBusy(destroy ? "destroy" : "keep");
    try {
      await props.onStop(vm, destroy);
      props.onClose();
    } finally {
      setBusy(null);
    }
  };

  return (
    <Modal
      open
      onClose={props.onClose}
      closable={!busy}
      icon={<ColabIcon size={24} />}
      iconBg="var(--warning-50)"
      title={`关闭 ${vm.name} 的接力训练`}
      sub="已同步到 Google Drive 的断点会保留，下次开启自动续训"
    >
      <div className="choice-list">
        <button className="choice" disabled={!!busy} onClick={() => go(false)}>
          {busy === "keep" ? <Loader2 size={18} className="spin" /> : <Square size={18} color="var(--warning)" />}
          <div>
            <div style={{ fontWeight: 600 }}>仅关闭接力，保留实例</div>
            <div className="dim" style={{ fontSize: 12 }}>停止监工并重启远程 Kernel 中断训练；实例仍在线，<b className="warn">会继续计费</b>。</div>
          </div>
        </button>
        <button className="choice danger" disabled={!!busy} onClick={() => go(true)}>
          {busy === "destroy" ? <Loader2 size={18} className="spin" /> : <Trash2 size={18} color="var(--danger)" />}
          <div>
            <div style={{ fontWeight: 600 }}>关闭接力并彻底销毁实例</div>
            <div className="dim" style={{ fontSize: 12 }}>unassign 并核验服务端计费列表，<b className="ok">立即停止计费</b>。</div>
          </div>
        </button>
      </div>
    </Modal>
  );
}
