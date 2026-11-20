import { useEffect, useState } from "react";
import { AlertTriangle, Download, Repeat2 } from "lucide-react";
import type { RelayConfig, VmView } from "../types";
import { DEFAULT_RELAY } from "../types";
import { fmtDuration, pct } from "../format";
import {
  MAX_RELAY_SESSION_HOURS,
  MIN_RELAY_SESSION_HOURS,
  findConflictingRelayVm,
  isDefaultBareGdriveDir,
  isSingleShapeAccelerator,
  isValidRelaySessionHours,
  resolveEffectiveHighMem,
  resolveEffectiveRelayGdriveDir,
  resolveRelayCompletionView,
  sanitizeRelayVmSegment,
} from "../chatUtils";

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
  allVms?: VmView[];
  onStart: (cfg: RelayConfig, legacyGdriveAction?: "migrate" | "ignore" | null) => void;
  onStopRequest: () => void;
  pending: boolean;
}) {
  const { vm } = props;
  const relay = vm.relay;
  const running = !!relay?.running;
  const tr = vm.telemetry?.training ?? null;

  const [cfg, setCfg] = useState<RelayConfig>(() => {
    const normAcc = (vm.accelerator ?? "").trim().toUpperCase();
    // #52：保留 CPU 等全部受支持加速器类型，绝不将 CPU 静默篡改为 T4
    const initGpu = GPUS.includes(normAcc) ? normAcc : "T4";
    return {
      ...DEFAULT_RELAY,
      gpu_type: initGpu,
      high_mem: resolveEffectiveHighMem(initGpu, vm.high_ram || DEFAULT_RELAY.high_mem),
      ...(relay?.config ?? {}),
    };
  });
  const [confirmReuseGdrive, setConfirmReuseGdrive] = useState(false);

  // 运行中或后端已有配置时同步
  useEffect(() => {
    if (relay?.config) {
      setCfg(relay.config);
    }
  }, [running, relay?.config]);

  const rawIdx = PHASES.findIndex(([k]) => k === relay?.phase);
  const phaseIdx = relay?.phase === "cooldown" ? PHASES.length - 1 : rawIdx;
  const curEpoch = tr?.current_epoch ?? 0;
  const totEpoch = tr?.total_epochs ?? cfg.total_epochs;
  const epochPct = totEpoch > 0 ? (curEpoch / totEpoch) * 100 : 0;
  const elapsed = tr?.elapsed_seconds ?? 0;
  const sessionSec = Math.max(
    180,
    (isValidRelaySessionHours(cfg.max_session_hours) ? cfg.max_session_hours : 22.5) * 3600,
  );
  const handoverPct = (elapsed / sessionSec) * 100;

  const singleShape = isSingleShapeAccelerator(cfg.gpu_type);
  const effectiveHighMem = resolveEffectiveHighMem(cfg.gpu_type, cfg.high_mem);
  const set = <K extends keyof RelayConfig>(k: K, v: RelayConfig[K]) => {
    setCfg((c) => ({ ...c, [k]: v }));
    if (k === "gdrive_dir") setConfirmReuseGdrive(false);
  };

  const gdriveValid =
    /^[A-Za-z0-9_-]{2,}:[^\\\s]+$/.test(cfg.gdrive_dir.trim()) &&
    !cfg.gdrive_dir.trim().startsWith("-") &&
    !cfg.gdrive_dir.includes("..");
  const hoursValid = isValidRelaySessionHours(cfg.max_session_hours);
  const effectiveGdriveDir = resolveEffectiveRelayGdriveDir(cfg.gdrive_dir, vm.name);
  const conflictingVm = !running
    ? findConflictingRelayVm(cfg.gdrive_dir, vm.name, props.allVms ?? [])
    : undefined;
  const gdriveConflictBlocked = Boolean(conflictingVm && !confirmReuseGdrive);

  const disabled = running || props.pending || !!vm.busy;
  const canStart = gdriveValid && hoursValid && !gdriveConflictBlocked;

  const completionView = resolveRelayCompletionView(relay);
  const stateLabel = running ? (relay?.phase_label || "接力运行中") : completionView.stateLabel;
  const sub = running
    ? `第 ${relay?.round || 1} 棒 · 已运行 ${fmtDuration(relay?.started_at ? Date.now() / 1000 - relay.started_at : 0)}`
    : completionView.sub;

  const isLegacyMigrationPrompt = Boolean(
    !running && relay?.last_error && relay.last_error.includes("旧版默认目录"),
  );

  const triggerStart = (
    overrideResetDoneFlag?: boolean,
    legacyGdriveAction?: "migrate" | "ignore" | null,
    overrideGdriveDir?: string,
  ) => {
    if (!canStart || props.pending || !!vm.busy) return;
    props.onStart(
      {
        ...cfg,
        gdrive_dir: overrideGdriveDir ?? cfg.gdrive_dir,
        high_mem: effectiveHighMem,
        reset_done_flag: overrideResetDoneFlag ?? cfg.reset_done_flag,
      },
      legacyGdriveAction ?? null,
    );
  };

  return (
    <div>
      <div className="relay-top">
        <div className="relay-state">
          <div className="label" title={stateLabel}>
            {running ? (
              <span className="ok">● </span>
            ) : completionView.isCompletedAndDownloaded ? (
              <span className="ok">✓ </span>
            ) : completionView.isDownloadFailed ? (
              <span className="bad">⚠ </span>
            ) : null}
            {stateLabel}
          </div>
          <div className={`sub ${completionView.isDownloadFailed ? "bad" : completionView.isCompletedAndDownloaded ? "ok" : ""}`}>
            {sub}
          </div>
        </div>
        <button
          type="button"
          className={`switch ${running ? "on" : ""}`}
          disabled={props.pending || !!vm.busy || (!running && !canStart)}
          title={
            running
              ? "关闭接力训练"
              : !hoursValid
                ? `单棒时长须在 ${MIN_RELAY_SESSION_HOURS} ~ ${MAX_RELAY_SESSION_HOURS} 小时之间`
                : !gdriveValid
                  ? "Google Drive 目录格式必须为 remote:path（如 gdrive:colab_relay）"
                  : gdriveConflictBlocked
                    ? `目录正被实例 [${conflictingVm?.name}] 占用，请修改目录或勾选下方显式确认续训`
                    : "开启接力训练"
          }
          onClick={() => (running ? props.onStopRequest() : triggerStart())}
        />
      </div>

      {/* #80：检测到旧版根目录 gdrive:colab_relay 下存在历史断点时，显式提示目录迁移并要求用户确认 */}
      {isLegacyMigrationPrompt && (
        <div className="vm-alert-box vm-alert-warn" role="alert" style={{ marginBottom: 10 }}>
          <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
          <div className="vm-alert-content">
            <div className="vm-alert-title">检测到旧版根目录存在历史断点，请确认迁移策略</div>
            <div className="vm-alert-desc">{relay?.last_error}</div>
            <div style={{ display: "inline-flex", gap: 8, flexWrap: "wrap", marginTop: 8 }}>
              <button
                type="button"
                className="btn btn-primary btn-sm"
                disabled={props.pending || !!vm.busy}
                onClick={() => triggerStart(false, "migrate", DEFAULT_RELAY.gdrive_dir)}
              >
                <span>迁移旧断点并续训</span>
              </button>
              <button
                type="button"
                className="btn btn-sm"
                disabled={props.pending || !!vm.busy}
                onClick={() => triggerStart(cfg.reset_done_flag, "ignore", DEFAULT_RELAY.gdrive_dir)}
              >
                <span>忽略旧断点全新开始</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* #53：最终权重下载失败时展示醒目告警与一键补下载按钮 */}
      {!running && completionView.isDownloadFailed && (
        <div className="vm-alert-box vm-alert-danger" role="alert" style={{ marginBottom: 10 }}>
          <AlertTriangle size={15} style={{ flexShrink: 0, marginTop: 1 }} />
          <div className="vm-alert-content">
            <div className="vm-alert-title">云端训练已完成，但最终权重下载到本地失败</div>
            <div className="vm-alert-desc mono">{completionView.detailError}</div>
          </div>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            disabled={props.pending || !!vm.busy || !canStart}
            onClick={() => triggerStart(false)}
            title="保留云盘完成标记并立即重跑补下载最终权重到本地工作区"
          >
            <Download size={13} />
            <span>重试补下载权重</span>
          </button>
        </div>
      )}

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
            {GPUS.map((g) => (
              <option key={g} value={g}>
                {g}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <label>单棒时长 (0.1~23.5h)</label>
          <input
            type="number"
            step={0.5}
            min={MIN_RELAY_SESSION_HOURS}
            max={MAX_RELAY_SESSION_HOURS}
            value={Number.isFinite(cfg.max_session_hours) ? cfg.max_session_hours : ""}
            disabled={disabled}
            style={!hoursValid ? { borderColor: "var(--danger)" } : undefined}
            onChange={(e) => {
              const raw = e.target.value;
              const v = raw === "" ? Number.NaN : Number(raw);
              set("max_session_hours", v);
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
      {!hoursValid && (
        <div className="field-err">
          受限于 Colab 单次最长运行周期，单棒时长必须在 {MIN_RELAY_SESSION_HOURS} ~ {MAX_RELAY_SESSION_HOURS} 小时之间。
        </div>
      )}

      <div className="field" style={{ marginTop: 8 }}>
        <label>Google Drive 接力目录</label>
        <input
          value={cfg.gdrive_dir}
          disabled={disabled}
          onChange={(e) => set("gdrive_dir", e.target.value)}
        />
        {isDefaultBareGdriveDir(cfg.gdrive_dir) && (
          <div className="field-hint">
            默认目录将自动按实例子目录隔离为：<code>gdrive:colab_relay/{sanitizeRelayVmSegment(vm.name)}</code>
          </div>
        )}
      </div>

      {conflictingVm && !running && (
        <div className="vm-alert-box vm-alert-warn" role="alert" style={{ marginTop: 8 }}>
          <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
          <div className="vm-alert-content">
            <div className="vm-alert-title">
              目录 <code>{effectiveGdriveDir}</code> 正被运行中的接力任务 [{conflictingVm.name}] 占用
            </div>
            <div className="vm-alert-desc">
              多个运行中任务共用同一云盘目录会互相覆盖 checkpoint 与完成标记，建议改用独立子目录。
            </div>
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 5, cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={confirmReuseGdrive}
                onChange={(e) => setConfirmReuseGdrive(e.target.checked)}
              />
              <span>我已知晓风险，显式确认复用该目录续训</span>
            </label>
          </div>
        </div>
      )}

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
          <input
            type="checkbox"
            checked={cfg.reset_done_flag}
            disabled={disabled}
            onChange={(e) => set("reset_done_flag", e.target.checked)}
          />
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
