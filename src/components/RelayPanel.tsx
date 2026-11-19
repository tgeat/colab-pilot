import { useEffect, useState } from "react";
import { Repeat2 } from "lucide-react";
import type { RelayConfig, VmView } from "../types";
import { DEFAULT_RELAY } from "../types";
import { fmtDuration, pct } from "../format";
import { isSingleShapeAccelerator, resolveEffectiveHighMem } from "../chatUtils";

const PHASES = [
  ["preparing_gdrive", "准备网盘"],
  ["provisioning", "分配实例"],
  ["uploading_auth", "上传密钥"],
  ["installing_env", "安装环境"],
  ["training", "训练热备"],
  ["destroying", "销毁换棒"],
] as const;

const GPUS = ["T4", "L4", "A100", "H100", "G4", "V5E1", "V6E1", "CPU"];

export function RelayPanel(props: {
  vm: VmView;
  onStart: (cfg: RelayConfig) => void;
  onStopRequest: () => void;
  pending: boolean;
}) {
  const { vm } = props;
  const relay = vm.relay;
  const running = !!relay?.running;
  const tr = vm.telemetry?.training ?? null;

  const [cfg, setCfg] = useState<RelayConfig>(() => {
    const initGpu =
      vm.accelerator && vm.accelerator !== "CPU" && vm.accelerator !== "…"
        ? vm.accelerator
        : "T4";
    return {
      ...DEFAULT_RELAY,
      gpu_type: initGpu,
      high_mem: resolveEffectiveHighMem(initGpu, vm.high_ram || DEFAULT_RELAY.high_mem),
      ...(relay?.config ?? {}),
    };
  });

  // 运行中以后端配置为准
  useEffect(() => {
    if (running && relay?.config) setCfg(relay.config);
  }, [running, relay?.config]);

  const rawIdx = PHASES.findIndex(([k]) => k === relay?.phase);
  const phaseIdx = relay?.phase === "cooldown" ? PHASES.length - 1 : rawIdx;
  const curEpoch = tr?.current_epoch ?? 0;
  const totEpoch = tr?.total_epochs ?? cfg.total_epochs;
  const epochPct = totEpoch > 0 ? (curEpoch / totEpoch) * 100 : 0;
  const elapsed = tr?.elapsed_seconds ?? 0;
  const sessionSec = Math.max(180, (Number.isFinite(cfg.max_session_hours) ? cfg.max_session_hours : 22.5) * 3600);
  const handoverPct = (elapsed / sessionSec) * 100;

  const singleShape = isSingleShapeAccelerator(cfg.gpu_type);
  const effectiveHighMem = resolveEffectiveHighMem(cfg.gpu_type, cfg.high_mem);
  const set = <K extends keyof RelayConfig>(k: K, v: RelayConfig[K]) => setCfg((c) => ({ ...c, [k]: v }));
  const gdriveValid =
    /^[A-Za-z0-9_-]{2,}:[^\\\s]+$/.test(cfg.gdrive_dir.trim()) &&
    !cfg.gdrive_dir.trim().startsWith("-") &&
    !cfg.gdrive_dir.includes("..");
  const disabled = running || props.pending || !!vm.busy;

  let stateLabel = relay?.phase_label ?? "未开启接力";
  let sub = "开启后自动换棒、定时热备、到期销毁旧实例";
  if (running) sub = `第 ${relay?.round || 1} 棒 · 已运行 ${fmtDuration(relay?.started_at ? Date.now() / 1000 - relay.started_at : 0)}`;
  else if (relay?.phase === "completed") sub = "已从 Google Drive 回收最终 checkpoint";
  else if (relay?.phase === "error") sub = "请查看下方「接力日志」";

  return (
    <div>
      <div className="relay-top">
        <div className="relay-state">
          <div className="label" title={stateLabel}>
            {running ? <span className="ok">● </span> : null}
            {stateLabel}
          </div>
          <div className="sub">{sub}</div>
        </div>
        <button
          type="button"
          className={`switch ${running ? "on" : ""}`}
          disabled={props.pending || !!vm.busy || (!running && !gdriveValid)}
          title={
            running
              ? "关闭接力训练"
              : gdriveValid
                ? "开启接力训练"
                : "Google Drive 目录格式必须为 remote:path（如 gdrive:colab_relay）"
          }
          onClick={() =>
            running
              ? props.onStopRequest()
              : props.onStart({
                  ...cfg,
                  high_mem: effectiveHighMem,
                })
          }
        />
      </div>

      <div className="steps" title={PHASES.map((p) => p[1]).join(" → ")}>
        {PHASES.map(([k, label], i) => (
          <div
            key={k}
            title={label}
            className={`step ${running && i < phaseIdx ? "done" : ""} ${running && i === phaseIdx ? "active" : ""}`}
          />
        ))}
      </div>

      <div className="form-grid">
        <div className="field">
          <label>换棒硬件</label>
          <select value={cfg.gpu_type} disabled={disabled} onChange={(e) => set("gpu_type", e.target.value)}>
            {GPUS.map((g) => <option key={g} value={g}>{g}</option>)}
          </select>
        </div>
        <div className="field">
          <label>单棒时长 (h)</label>
          <input
            type="number"
            step={0.5}
            min={0.1}
            max={23.5}
            value={cfg.max_session_hours}
            disabled={disabled}
            onChange={(e) => {
              const v = Number(e.target.value);
              set("max_session_hours", Number.isFinite(v) ? Math.min(48, Math.max(0.05, v)) : 22.5);
            }}
          />
        </div>
        <div className="field">
          <label>热备间隔 (min)</label>
          <input
            type="number"
            step={5}
            min={1}
            max={1440}
            value={cfg.backup_interval_minutes}
            disabled={disabled}
            onChange={(e) => {
              const v = Number(e.target.value);
              set("backup_interval_minutes", Number.isFinite(v) ? Math.min(1440, Math.max(0.5, v)) : 30);
            }}
          />
        </div>
        <div className="field">
          <label>总轮数 Epochs</label>
          <input
            type="number"
            step={10}
            min={1}
            max={1000000}
            value={cfg.total_epochs}
            disabled={disabled}
            onChange={(e) => {
              const v = Number(e.target.value);
              set("total_epochs", Number.isFinite(v) ? Math.min(1000000, Math.max(1, Math.round(v))) : 1);
            }}
          />
        </div>
      </div>
      <div className="field" style={{ marginTop: 8 }}>
        <label>Google Drive 接力目录</label>
        <input value={cfg.gdrive_dir} disabled={disabled} onChange={(e) => set("gdrive_dir", e.target.value)} />
      </div>
      <div className="checks">
        <label title={singleShape ? `${cfg.gpu_type} 采用固定内存规格，无需开启 High-RAM` : undefined}>
          <input
            type="checkbox"
            checked={effectiveHighMem}
            disabled={disabled || singleShape}
            onChange={(e) => set("high_mem", e.target.checked)}
          />
          High-RAM {singleShape ? <span className="dim">(该型号固定内存规格)</span> : null}
        </label>
        <label>
          <input type="checkbox" checked={cfg.reset_done_flag} disabled={disabled}
            onChange={(e) => set("reset_done_flag", e.target.checked)} />
          重置完成标记（重新训练）
        </label>
      </div>

      <div className="progress-row">
        <Repeat2 size={14} />
        <span style={{ width: 92 }}>Epoch {curEpoch}/{totEpoch}</span>
        <div className="track"><div className="fill" style={{ width: `${pct(epochPct)}%` }} /></div>
        <span className="mono" style={{ width: 40, textAlign: "right" }}>{pct(epochPct).toFixed(0)}%</span>
      </div>
      <div className="progress-row" style={{ marginTop: 6 }}>
        <span style={{ width: 106 }}>换棒倒计时</span>
        <div className="track"><div className="fill mid" style={{ width: `${pct(handoverPct)}%` }} /></div>
        <span className="mono" style={{ width: 70, textAlign: "right" }}>
          {tr?.handover_in_seconds !== undefined ? fmtDuration(tr.handover_in_seconds) : "—"}
        </span>
      </div>
    </div>
  );
}
