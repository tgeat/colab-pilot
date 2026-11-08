import { useState } from "react";
import { Loader2, Repeat2 } from "lucide-react";
import { ColabIcon, Modal } from "./ui";
import { DEFAULT_RELAY } from "../types";

const HW = [
  { v: "T4", d: "16GB · 性价比" },
  { v: "L4", d: "24GB · 均衡" },
  { v: "A100", d: "40/80GB · 训练" },
  { v: "H100", d: "80GB · 旗舰" },
  { v: "G4", d: "GPU · G4" },
  { v: "V5E1", d: "TPU v5e-1" },
  { v: "V6E1", d: "TPU v6e-1" },
  { v: "CPU", d: "标准 CPU" },
];

export function CreateVmDialog(props: {
  open: boolean;
  onClose: () => void;
  onCreate: (p: { name: string | null; accelerator: string; high_mem: boolean; auto_relay: boolean }) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [acc, setAcc] = useState("T4");
  const [highMem, setHighMem] = useState(true);
  const [autoRelay, setAutoRelay] = useState(false);
  const [busy, setBusy] = useState(false);
  const nameOk = name === "" || /^[A-Za-z0-9_-]{1,40}$/.test(name);

  const submit = async () => {
    setBusy(true);
    try {
      await props.onCreate({ name: name.trim() || null, accelerator: acc, high_mem: highMem, auto_relay: autoRelay });
      setName("");
      props.onClose();
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={props.open}
      onClose={() => !busy && props.onClose()}
      closable={!busy}
      icon={<ColabIcon size={24} />}
      iconBg="#fff7e6"
      title="新增 Colab 虚拟机"
      sub="通过 colab-cli 在你的 Google 账号下分配一台独立的云端运行时"
      footer={
        <>
          <button className="btn" onClick={props.onClose} disabled={busy}>取消</button>
          <button className="btn btn-primary" onClick={submit} disabled={busy || !nameOk}>
            {busy ? <><Loader2 size={15} className="spin" />正在分配（约 10–60 秒）…</> : "创建虚拟机"}
          </button>
        </>
      }
    >
      <div className="field">
        <label>名称（留空自动生成）</label>
        <input
          value={name}
          placeholder="例如 relay-a100-01"
          onChange={(e) => setName(e.target.value)}
          style={!nameOk ? { borderColor: "var(--danger)" } : undefined}
        />
        {!nameOk && <div className="bad" style={{ fontSize: 11.5, marginTop: 4 }}>仅支持字母、数字、- 和 _</div>}
      </div>

      <div>
        <div className="field"><label>硬件加速器</label></div>
        <div className="hw-grid">
          {HW.map((h) => (
            <button key={h.v} className={`hw-opt ${acc === h.v ? "sel" : ""}`} onClick={() => setAcc(h.v)}>
              <span className="t" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                <ColabIcon size={14} />{h.v}
              </span>
              <span className="d">{h.d}</span>
            </button>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>
          <div style={{ fontWeight: 600, fontSize: 13.5 }}>内存规格</div>
          <div className="dim" style={{ fontSize: 12 }}>High-RAM 需 Colab Pro / Pro+；L4 / TPU 仅单一规格</div>
        </div>
        <div className="seg">
          <button className={!highMem ? "sel" : ""} onClick={() => setHighMem(false)}>Standard</button>
          <button className={highMem ? "sel" : ""} onClick={() => setHighMem(true)}>High-RAM</button>
        </div>
      </div>

      <label className="option-card" style={{ cursor: "pointer" }}>
        <input type="checkbox" checked={autoRelay} onChange={(e) => setAutoRelay(e.target.checked)} style={{ marginTop: 3, accentColor: "var(--primary)" }} />
        <div>
          <div className="t" style={{ display: "flex", alignItems: "center", gap: 6 }}><Repeat2 size={15} />创建后立即开启 24h+ 接力训练</div>
          <div className="d">
            使用默认参数（单棒 {DEFAULT_RELAY.max_session_hours}h · 每 {DEFAULT_RELAY.backup_interval_minutes} 分钟热备 · {DEFAULT_RELAY.gdrive_dir}），可随时在卡片中关闭。
          </div>
        </div>
      </label>
    </Modal>
  );
}
