import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronDown,
  ChevronUp,
  Copy,
  Cpu,
  FolderGit2,
  HardDrive,
  History,
  KeyRound,
  Loader2,
  Power,
  Repeat2,
  Server,
  ShieldCheck,
  Sparkles,
  Terminal,
  Wrench,
} from "lucide-react";
import { errText } from "../api";
import {
  MAX_RELAY_SESSION_HOURS,
  MIN_RELAY_SESSION_HOURS,
  checkCreateVmAuthGuard,
  findConflictingRelayVm,
  isDefaultBareGdriveDir,
  isLlmEngineCompatibleWithAccelerator,
  isNvidiaCudaAccelerator,
  isSingleShapeAccelerator,
  isValidRelaySessionHours,
  requiresGdriveConfig,
  resolveEffectiveHighMem,
  resolveEffectiveRelayGdriveDir,
  sanitizeRelayVmSegment,
} from "../chatUtils";
import { ColabIcon, Modal } from "./ui";
import {
  DEFAULT_RELAY,
  type CreateVmLlmConfig,
  type CreateVmRequest,
  type EnvStatus,
  type RelayConfig,
  type VmView,
} from "../types";

export const DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct-GGUF";
export const DEFAULT_LLM_ENGINE = "llama.cpp";

export interface CreateVmPreset {
  name?: string;
  accelerator?: string;
  high_mem?: boolean;
  boot_mode?: "general" | "llm" | "relay";
  llm_slot?: "blue" | "green" | "standalone";
  llm_model?: string;
  llm_engine?: string;
  llm_config?: CreateVmLlmConfig | null;
  relay_config?: RelayConfig | null;
  workspace_path?: string;
  /** 历史记录无持久化配置时提示用户确认工作模式与参数 (#74) */
  unverified_history_profile?: boolean;
  /** 若指定已有运行中实例名，则切换为“在该实例上直接部署大模型”模式（调用 deploy_llm_on_vm） */
  target_existing_vm?: string;
}

export const ACCELERATORS = [
  { v: "T4", tag: "GPU · 16GB", d: "NVIDIA T4 · 高性价比推理/微调", variant: "GPU", singleShape: false },
  { v: "L4", tag: "GPU · 24GB", d: "NVIDIA L4 · Ada 架构中大显存", variant: "GPU", singleShape: true },
  { v: "G4", tag: "GPU · 96GB", d: "RTX PRO 6000 Blackwell · 96GB 超大显存", variant: "GPU", singleShape: true },
  { v: "A100", tag: "GPU · 40/80GB", d: "NVIDIA A100 · 大模型与深度学习", variant: "GPU", singleShape: false },
  { v: "H100", tag: "GPU · 80GB", d: "NVIDIA H100 · 旗舰级 SXM/PCIe", variant: "GPU", singleShape: true },
  { v: "V5E1", tag: "TPU · v5e-1", d: "Google Cloud TPU v5e-1", variant: "TPU", singleShape: true },
  { v: "V6E1", tag: "TPU · v6e-1", d: "Google Trillium TPU v6e-1", variant: "TPU", singleShape: true },
  { v: "CPU", tag: "CPU · 标准", d: "标准多核 CPU · 脚本与轻量任务", variant: "CPU", singleShape: false },
];

export const LLM_ENGINES = [
  { v: "llama.cpp", label: "llama.cpp (CUDA GGUF)", desc: "极速拉取 GGUF 量化权重，原生 CUDA 推理", requiresCuda: true },
  { v: "vllm", label: "vLLM (PagedAttention)", desc: "高并发吞吐 OpenAI 兼容服务端", requiresCuda: true },
  { v: "ollama", label: "Ollama", desc: "一键拉取 Ollama 模型仓库并暴露接口", requiresCuda: false },
  { v: "transformers", label: "HF Transformers", desc: "原生 PyTorch 流式生成服务", requiresCuda: false },
  { v: "builtin", label: "Builtin 轻量边车", desc: "秒级启动云端边车用于链路冒烟与调试", requiresCuda: false },
];

export const MODEL_PRESETS = [
  { label: "Qwen2.5-7B-Instruct GGUF (推荐)", value: "Qwen/Qwen2.5-7B-Instruct-GGUF", engine: "llama.cpp" },
  { label: "DeepSeek-R1-Distill-Qwen-7B GGUF", value: "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", engine: "llama.cpp" },
  { label: "Qwen3.8-Flash GGUF", value: "SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF", engine: "llama.cpp" },
  { label: "Qwen2.5-Coder-7B-Instruct (vLLM/HF)", value: "Qwen/Qwen2.5-Coder-7B-Instruct", engine: "vllm" },
  { label: "qwen2.5:7b (Ollama)", value: "qwen2.5:7b", engine: "ollama" },
];

export function CreateVmDialog(props: {
  open: boolean;
  defaultWorkspace?: string;
  env?: EnvStatus | null;
  vms?: VmView[];
  preset?: CreateVmPreset | null;
  onClose: () => void;
  onCreate: (req: CreateVmRequest) => Promise<void>;
  onDeployLlmOnVm?: (name: string, config: CreateVmLlmConfig) => Promise<void>;
}) {
  const [name, setName] = useState("");
  const [acc, setAcc] = useState("T4");
  const [highMem, setHighMem] = useState(true);
  const [authMode, setAuthMode] = useState<"oauth2" | "adc">("oauth2");
  const [workspacePath, setWorkspacePath] = useState(props.defaultWorkspace ?? "");
  const [description, setDescription] = useState("");
  const [bootMode, setBootMode] = useState<"general" | "llm" | "relay">("general");
  const [historyUnverifiedNotice, setHistoryUnverifiedNotice] = useState(false);

  // LLM 部署配置
  const [llmEngine, setLlmEngine] = useState(DEFAULT_LLM_ENGINE);
  const [llmModel, setLlmModel] = useState(DEFAULT_LLM_MODEL);
  const [llmSlot, setLlmSlot] = useState<"blue" | "green" | "standalone">("blue");
  const [gdriveModelPath, setGdriveModelPath] = useState("");
  const [customApiKey, setCustomApiKey] = useState("");

  // 24h+ 接力训练配置
  const [relayCfg, setRelayCfg] = useState<RelayConfig>({ ...DEFAULT_RELAY });
  const [confirmReuseRelayGdrive, setConfirmReuseRelayGdrive] = useState(false);

  // Colab 高级运行时环境配置（#67：通用模式下 syncRclone 默认 false）
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [syncRclone, setSyncRclone] = useState(false);
  const [mountGdrive, setMountGdrive] = useState(false);
  const [installReqTxt, setInstallReqTxt] = useState(false);
  const [pipPackages, setPipPackages] = useState("");
  const [envVars, setEnvVars] = useState("");
  const [startupScript, setStartupScript] = useState("");

  const [busy, setBusy] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [copiedError, setCopiedError] = useState(false);
  const prevOpenRef = useRef(false);
  const userToggledSyncRcloneRef = useRef(false);

  const targetExistingVm = props.preset?.target_existing_vm?.trim() || null;
  const isDeployExisting = Boolean(targetExistingVm);

  // 每次弹窗打开（含 preset === null）时完整重置全部表单状态
  useEffect(() => {
    const justOpened = props.open && !prevOpenRef.current;
    prevOpenRef.current = props.open;
    if (!justOpened) return;

    const p = props.preset ?? null;
    const initAcc = (p?.accelerator ?? "T4").toUpperCase();
    const validAcc = ACCELERATORS.some((a) => a.v === initAcc) ? initAcc : "T4";
    const initSingle = isSingleShapeAccelerator(validAcc);
    const existingTarget = p?.target_existing_vm?.trim() || "";
    const initBootMode: "general" | "llm" | "relay" = existingTarget
      ? "llm"
      : (p?.boot_mode ?? "general");

    setBusy(false);
    setSubmitError(null);
    setCopiedError(false);
    userToggledSyncRcloneRef.current = false;

    setName(p?.name ?? "");
    setAcc(validAcc);
    setHighMem(initSingle ? false : (p?.high_mem ?? true));
    setAuthMode("oauth2");
    setWorkspacePath(p?.workspace_path ?? props.defaultWorkspace ?? "");
    setDescription("");
    setBootMode(initBootMode);
    setHistoryUnverifiedNotice(Boolean(p?.unverified_history_profile));

    const savedLlm = p?.llm_config ?? null;
    const initGdriveModel = savedLlm?.gdrive_model_path ?? "";
    const rawSavedSlot = (savedLlm?.slot ?? "").toLowerCase();
    const resolvedSavedSlot: "blue" | "green" | "standalone" | undefined =
      rawSavedSlot === "blue" || rawSavedSlot === "green" || rawSavedSlot === "standalone"
        ? rawSavedSlot
        : undefined;

    const defaultEngineForAcc = isNvidiaCudaAccelerator(validAcc)
      ? DEFAULT_LLM_ENGINE
      : "ollama";
    setLlmEngine(savedLlm?.engine ?? p?.llm_engine ?? defaultEngineForAcc);
    setLlmModel(
      savedLlm?.model_name ??
        p?.llm_model ??
        (isNvidiaCudaAccelerator(validAcc) ? DEFAULT_LLM_MODEL : "qwen2.5:7b"),
    );
    setLlmSlot(
      p?.llm_slot ??
        resolvedSavedSlot ??
        (existingTarget
          ? existingTarget.endsWith("-blue")
            ? "blue"
            : existingTarget.endsWith("-green")
              ? "green"
              : "standalone"
          : "blue"),
    );
    setGdriveModelPath(initGdriveModel);
    setCustomApiKey(savedLlm?.custom_api_key ?? "");

    setRelayCfg(
      p?.relay_config
        ? { ...DEFAULT_RELAY, ...p.relay_config }
        : { ...DEFAULT_RELAY, gpu_type: validAcc },
    );
    setConfirmReuseRelayGdrive(false);

    setShowAdvanced(false);
    // #67：通用算力模式下默认不要求/不同步 rclone.conf；仅接力或显式填写 Google Drive 权重路径时默认开启
    setSyncRclone(initBootMode === "relay" || Boolean(initGdriveModel.trim()));
    setMountGdrive(false);
    setInstallReqTxt(false);
    setPipPackages("");
    setEnvVars("");
    setStartupScript("");
  }, [props.open, props.preset, props.defaultWorkspace]);

  const handleSelectBootMode = (nextMode: "general" | "llm" | "relay") => {
    setBootMode(nextMode);
    if (!userToggledSyncRcloneRef.current) {
      if (nextMode === "relay") setSyncRclone(true);
      else if (nextMode === "general") setSyncRclone(false);
      else setSyncRclone(Boolean(gdriveModelPath.trim()));
    }
  };

  const selectedAcc = ACCELERATORS.find((a) => a.v === acc) ?? ACCELERATORS[0];
  const effectiveHighMem = selectedAcc.singleShape
    ? false
    : resolveEffectiveHighMem(selectedAcc.v, highMem);
  const nameOk = isDeployExisting || name === "" || /^[A-Za-z0-9_-]{1,40}$/.test(name.trim());
  const isLlmFlow = isDeployExisting || bootMode === "llm";
  const llmModelOk = !isLlmFlow || llmEngine === "builtin" || llmModel.trim().length > 0;

  // #68：CUDA 推理预设与非 NVIDIA 硬件（CPU / TPU）兼容性检查
  const llmHwCompatible = !isLlmFlow || isLlmEngineCompatibleWithAccelerator(llmEngine, acc);
  const selectedEngineMeta = LLM_ENGINES.find((e) => e.v === llmEngine);

  // #43：ADC / OAuth 凭证监控与停费核验守卫
  const authGuard = !isDeployExisting
    ? checkCreateVmAuthGuard(authMode, props.env)
    : { blocked: false, level: "none" as const, message: null };

  // #54：接力单棒时长上限校验（0.1 ~ 23.5h）
  const isRelayFlow = !isDeployExisting && bootMode === "relay";
  const relayHoursOk = !isRelayFlow || isValidRelaySessionHours(relayCfg.max_session_hours);
  const relayGdriveFormatOk =
    !isRelayFlow ||
    (/^[A-Za-z0-9_-]{2,}:[^\\\s]+$/.test(relayCfg.gdrive_dir.trim()) &&
      !relayCfg.gdrive_dir.trim().startsWith("-") &&
      !relayCfg.gdrive_dir.includes(".."));

  // #44：默认云盘目录隔离展示与运行中目录冲突拦截
  const effectiveRelayGdrive = isRelayFlow
    ? resolveEffectiveRelayGdriveDir(relayCfg.gdrive_dir, name.trim() || null)
    : "";
  const relayConflictVm = isRelayFlow
    ? findConflictingRelayVm(relayCfg.gdrive_dir, name.trim() || null, props.vms ?? [])
    : undefined;
  const relayConflictBlocked = Boolean(relayConflictVm && !confirmReuseRelayGdrive);

  // #67：当所选配置依赖 Google Drive / rclone.conf 且本机缺失授权时，在底部上方醒目提示（即使高级配置折叠）
  const needsGdrive = requiresGdriveConfig({
    isDeployExisting,
    bootMode,
    syncRclone,
    mountGdrive,
    gdriveModelPath,
  });
  const missingGdriveEnv = Boolean(
    needsGdrive && props.env && (!props.env.rclone_ok || !props.env.rclone_conf_ok),
  );

  const canSubmit =
    !busy &&
    nameOk &&
    llmModelOk &&
    llmHwCompatible &&
    !authGuard.blocked &&
    relayHoursOk &&
    relayGdriveFormatOk &&
    !relayConflictBlocked;

  const copyError = async () => {
    if (!submitError) return;
    try {
      await navigator.clipboard.writeText(submitError);
      setCopiedError(true);
      setTimeout(() => setCopiedError(false), 1600);
    } catch {
      // ignore
    }
  };

  const submit = async () => {
    if (busy) return;
    if (authGuard.blocked && authGuard.message) {
      setSubmitError(authGuard.message);
      return;
    }
    if (!llmHwCompatible) {
      setSubmitError(
        `推理引擎「${selectedEngineMeta?.label ?? llmEngine}」依赖 NVIDIA CUDA，无法在 ${acc} 上运行。请切换为 NVIDIA GPU 或改用兼容引擎（如 Ollama）。`,
      );
      return;
    }
    if (!relayHoursOk) {
      setSubmitError(
        `接力单棒时长必须在 ${MIN_RELAY_SESSION_HOURS} ~ ${MAX_RELAY_SESSION_HOURS} 小时之间（受限于 Colab 单次最长运行周期）。`,
      );
      return;
    }
    if (relayConflictBlocked && relayConflictVm) {
      setSubmitError(
        `云盘目录 ${effectiveRelayGdrive} 正被运行中的接力任务 [${relayConflictVm.name}] 占用，请更换目录或勾选显式确认续训。`,
      );
      return;
    }
    if (!nameOk || !llmModelOk || !relayGdriveFormatOk) return;

    setSubmitError(null);
    setCopiedError(false);
    setBusy(true);
    try {
      if (isDeployExisting && targetExistingVm) {
        if (!props.onDeployLlmOnVm) {
          throw new Error("未绑定实例大模型部署处理函数");
        }
        await props.onDeployLlmOnVm(targetExistingVm, {
          engine: llmEngine,
          model_name: llmModel.trim(),
          slot: llmSlot,
          gdrive_model_path: gdriveModelPath.trim() || null,
          custom_api_key: customApiKey.trim() || null,
        });
        props.onClose();
        return;
      }

      const req: CreateVmRequest = {
        name: name.trim() || null,
        accelerator: acc,
        high_mem: effectiveHighMem,
        auth_mode: authMode,
        workspace_path: workspacePath.trim() || null,
        description: description.trim() || null,
        boot_mode: bootMode,
        auto_relay: bootMode === "relay",
        relay_config:
          bootMode === "relay"
            ? {
                ...relayCfg,
                // #52：严格保留用户选择的硬件规格（包括 "CPU"），绝不静默篡改为 "T4"
                gpu_type: acc,
                high_mem: effectiveHighMem,
              }
            : null,
        llm_config:
          bootMode === "llm"
            ? {
                engine: llmEngine,
                model_name: llmModel.trim(),
                slot: llmSlot,
                gdrive_model_path: gdriveModelPath.trim() || null,
                custom_api_key: customApiKey.trim() || null,
              }
            : null,
        env_config: {
          sync_rclone: syncRclone,
          mount_gdrive: mountGdrive,
          pip_packages: pipPackages.trim() || null,
          install_workspace_requirements: installReqTxt,
          startup_script: startupScript.trim() || null,
          env_vars: envVars.trim() || null,
        },
      };
      await props.onCreate(req);
      setName("");
      props.onClose();
    } catch (e) {
      setSubmitError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={props.open}
      onClose={() => !busy && props.onClose()}
      closable={!busy}
      wide
      icon={<ColabIcon size={24} />}
      iconBg="#fff7e6"
      title={
        isDeployExisting
          ? `在运行实例 [${targetExistingVm}] 上部署大模型`
          : "开机 / 新建 Colab 云端虚拟机"
      }
      sub={
        isDeployExisting
          ? `复用当前已运行的云端算力实例 (${acc}) 拉取模型权重并启动推理引擎，自动生成专属 OpenAI API 与 Key`
          : "自由选择 GPU / TPU / CPU 与全量 Colab 运行时配置 · 自动为每台实例提供免密 SSH，并为大模型实例生成专属 OpenAI API 与 Key"
      }
      footer={
        <>
          <div
            style={{ marginRight: "auto", display: "flex", alignItems: "center", gap: 8, fontSize: 11.5 }}
            className="dim"
          >
            {isDeployExisting ? (
              <>
                <KeyRound size={13} />
                <span>后台自动拉取权重并预热服务，生成专属 OpenAI API 端点与 Key</span>
              </>
            ) : (
              <>
                <Terminal size={13} />
                <span>开机后自动分配 SSH 端口与免密密钥</span>
                {bootMode === "llm" && (
                  <>
                    <span>·</span>
                    <KeyRound size={13} />
                    <span>自动生成 OpenAI API 端点与 Key</span>
                  </>
                )}
              </>
            )}
          </div>
          <button type="button" className="btn" onClick={props.onClose} disabled={busy}>
            取消
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={submit}
            disabled={!canSubmit}
          >
            {busy ? (
              <>
                <Loader2 size={15} className="spin" />
                {isDeployExisting
                  ? "正在提交云端大模型部署任务…"
                  : "正在云端分配并初始化（约 15–60 秒）…"}
              </>
            ) : isDeployExisting ? (
              <>
                <Sparkles size={15} />
                立即部署大模型 ({targetExistingVm})
              </>
            ) : (
              <>
                <Power size={15} />
                {bootMode === "llm"
                  ? `开机并部署大模型 (${acc})`
                  : bootMode === "relay"
                    ? `开机并启动接力训练 (${acc})`
                    : `立即开机 (${acc})`}
              </>
            )}
          </button>
        </>
      }
    >
      {/* #74：无持久化工作模式档案的历史会话提示 */}
      {historyUnverifiedNotice && !isDeployExisting && (
        <div className="banner banner-warn" role="status" style={{ margin: 0 }}>
          <History size={16} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <b>该历史记录仅复用了会话名称与硬件规格，请确认所需的工作模式与参数</b>
          </div>
        </div>
      )}

      {submitError && (
        <div className="banner banner-danger modal-inline-error" role="alert">
          <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
          <div className="modal-inline-error-body">
            <div className="modal-inline-error-title">
              {isDeployExisting ? "大模型部署任务提交失败" : "新建实例开机失败（已保留当前表单配置，可直接修改后重试）"}
            </div>
            <pre className="modal-inline-error-text mono">{submitError}</pre>
          </div>
          <button
            type="button"
            className={`copy-chip-btn ${copiedError ? "copied" : ""}`}
            onClick={copyError}
            title="复制完整报错文本"
          >
            {copiedError ? <Check size={11} /> : <Copy size={11} />}
            <span>{copiedError ? "已复制" : "复制报错"}</span>
          </button>
        </div>
      )}

      {!isDeployExisting && (
        <>
          {/* 1. 开机工作负载模式选择 */}
          <div>
            <div className="field">
              <label>开机工作模式</label>
            </div>
            <div className="boot-mode-tabs">
              <button
                type="button"
                className={`boot-mode-tab ${bootMode === "general" ? "sel" : ""}`}
                onClick={() => handleSelectBootMode("general")}
              >
                <div className="boot-mode-tab-head">
                  <Terminal size={15} />
                  <span>通用算力实例 + SSH</span>
                </div>
                <div className="boot-mode-tab-desc">
                  分配纯净 GPU/TPU/CPU 实例，自动配置本机免密 SSH 桥接与实时监控探针，开机后仍可随时跑训练或部署模型
                </div>
              </button>

              <button
                type="button"
                className={`boot-mode-tab ${bootMode === "llm" ? "sel" : ""}`}
                onClick={() => handleSelectBootMode("llm")}
              >
                <div className="boot-mode-tab-head">
                  <Sparkles size={15} />
                  <span>部署大模型 + API / Key</span>
                </div>
                <div className="boot-mode-tab-desc">
                  自动拉取大模型权重、启动推理引擎，并提供专属 OpenAI API URL、API Key 与 SSH
                </div>
              </button>

              <button
                type="button"
                className={`boot-mode-tab ${bootMode === "relay" ? "sel" : ""}`}
                onClick={() => handleSelectBootMode("relay")}
              >
                <div className="boot-mode-tab-head">
                  <Repeat2 size={15} />
                  <span>24h+ 自动接力训练</span>
                </div>
                <div className="boot-mode-tab-desc">
                  挂载 Google Drive 断点热备，超时前自动换棒新实例连续训练，同时提供 SSH
                </div>
              </button>
            </div>
          </div>

          {/* 2. 硬件加速器类型（GPU / TPU / CPU 全量支持） */}
          <div>
            <div className="field" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <Cpu size={13} />
                硬件加速器类型（GPU / TPU / CPU）
              </label>
              <span className="dim mono" style={{ fontSize: 11 }}>
                当前选择: {selectedAcc.v} ({selectedAcc.tag}) ·{" "}
                {selectedAcc.singleShape ? "该型号采用固定内存规格" : effectiveHighMem ? "High-RAM" : "Standard RAM"}
              </span>
            </div>
            <div className="hw-grid hw-grid-4">
              {ACCELERATORS.map((h) => (
                <button
                  type="button"
                  key={h.v}
                  className={`hw-opt ${acc === h.v ? "sel" : ""}`}
                  onClick={() => setAcc(h.v)}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 4 }}>
                    <span className="t" style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                      <ColabIcon size={14} />
                      {h.v}
                    </span>
                    <span className="hw-tag">{h.tag}</span>
                  </div>
                  <span className="d">{h.d}</span>
                </button>
              ))}
            </div>
          </div>

          {/* 3. 实例名称 + 内存规格 + 认证方式 */}
          <div className="form-grid-3">
            <div className="field">
              <label>实例会话名称（留空自动命名）</label>
              <input
                value={name}
                placeholder={
                  bootMode === "llm"
                    ? `例如 llm-${llmSlot === "standalone" ? "qwen" : llmSlot}`
                    : "例如 gpu-dev-01"
                }
                onChange={(e) => setName(e.target.value)}
                style={!nameOk ? { borderColor: "var(--danger)" } : undefined}
              />
              {!nameOk && (
                <div className="bad" style={{ fontSize: 11, marginTop: 3 }}>
                  仅支持字母、数字、- 和 _（≤40 字符）
                </div>
              )}
            </div>

            <div className="field">
              <label>机器内存规格 (Machine Shape)</label>
              {selectedAcc.singleShape ? (
                <div
                  className="single-shape-note"
                  title={`${acc} 在 Google Colab 云端采用单一固定内存规格，不附加 --high-mem 参数以避免 HTTP 400`}
                >
                  该型号采用固定内存规格
                </div>
              ) : (
                <div className="seg" style={{ width: "100%", display: "flex" }}>
                  <button
                    type="button"
                    style={{ flex: 1 }}
                    className={!effectiveHighMem ? "sel" : ""}
                    onClick={() => setHighMem(false)}
                    title="标准内存规格 (Standard)"
                  >
                    Standard
                  </button>
                  <button
                    type="button"
                    style={{ flex: 1 }}
                    className={effectiveHighMem ? "sel" : ""}
                    onClick={() => setHighMem(true)}
                    title="高内存规格 (High-RAM)"
                  >
                    High-RAM
                  </button>
                </div>
              )}
            </div>

            <div className="field">
              <label>Colab 认证策略 (--auth)</label>
              <div className="seg" style={{ width: "100%", display: "flex" }}>
                <button
                  type="button"
                  style={{ flex: 1 }}
                  className={authMode === "oauth2" ? "sel" : ""}
                  onClick={() => setAuthMode("oauth2")}
                  title="使用 Google OAuth2 账号令牌（通过 colab auth 授权）"
                >
                  OAuth2 账号
                </button>
                <button
                  type="button"
                  style={{ flex: 1 }}
                  className={authMode === "adc" ? "sel" : ""}
                  onClick={() => setAuthMode("adc")}
                  title="使用 Google Cloud Application Default Credentials（监控与停费核验仍需 colab auth）"
                >
                  ADC 凭证
                </button>
              </div>
            </div>
          </div>

          {/* #43 & #75：ADC / 未登录 Colab OAuth 限制与引导警告 */}
          {authGuard.message && (
            <div
              className={`banner ${authGuard.blocked ? "banner-danger" : "banner-warn"}`}
              role="alert"
              style={{ margin: 0 }}
            >
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
              <div style={{ flex: 1 }}>
                <div>{authGuard.message}</div>
              </div>
            </div>
          )}
        </>
      )}

      {/* 4. 模式专属配置区：大模型 API/Key 配置 或 24h+ 接力训练配置 */}
      {isLlmFlow && (
        <div className="cfg-panel cfg-panel-llm">
          <div className="cfg-panel-head">
            <span className="cfg-panel-title">
              <Sparkles size={14} />
              大模型部署与专属 API / Key 配置
            </span>
            <span className="cfg-panel-badge">OpenAI 兼容 `/v1/chat/completions`</span>
          </div>

          {/* #68：CUDA 推理预设与 CPU/TPU 不兼容拦截提示 */}
          {!llmHwCompatible && (
            <div className="banner banner-danger" role="alert" style={{ margin: 0, alignItems: "center" }}>
              <AlertTriangle size={16} style={{ flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <div style={{ fontWeight: 700 }}>
                  硬件不兼容：推理引擎「{selectedEngineMeta?.label ?? llmEngine}」依赖 NVIDIA CUDA
                </div>
                <div style={{ fontSize: 12, marginTop: 2 }}>
                  当前所选硬件为 <b>{acc}</b>（非 NVIDIA GPU），无法运行 CUDA 推理预设。请切换为 NVIDIA GPU（如 T4 / L4 / A100）或改用兼容引擎（如 Ollama）。
                </div>
              </div>
              <div style={{ display: "inline-flex", gap: 6, flexWrap: "wrap" }}>
                {!isDeployExisting && (
                  <button type="button" className="btn btn-sm" onClick={() => setAcc("T4")}>
                    切换硬件至 T4
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => {
                    setLlmEngine("ollama");
                    if (llmModel.toLowerCase().includes("gguf")) {
                      setLlmModel("qwen2.5:7b");
                    }
                  }}
                >
                  切换至 Ollama
                </button>
              </div>
            </div>
          )}

          <div className="form-grid-2">
            <div className="field">
              <label>推理引擎 (Engine)</label>
              <select
                value={llmEngine}
                onChange={(e) => setLlmEngine(e.target.value)}
                style={!llmHwCompatible ? { borderColor: "var(--danger)" } : undefined}
              >
                {LLM_ENGINES.map((eng) => {
                  const compatible = isLlmEngineCompatibleWithAccelerator(eng.v, acc);
                  return (
                    <option key={eng.v} value={eng.v} disabled={!compatible}>
                      {eng.label}
                      {!compatible ? `（需 NVIDIA GPU，不支持 ${acc}）` : ` — ${eng.desc}`}
                    </option>
                  );
                })}
              </select>
            </div>

            <div className="field">
              <label>部署槽位 / 网关路由角色</label>
              <div className="seg" style={{ display: "flex", width: "100%" }}>
                <button
                  type="button"
                  style={{ flex: 1 }}
                  className={llmSlot === "blue" ? "sel" : ""}
                  onClick={() => setLlmSlot("blue")}
                >
                  BLUE 蓝槽 (:18081)
                </button>
                <button
                  type="button"
                  style={{ flex: 1 }}
                  className={llmSlot === "green" ? "sel" : ""}
                  onClick={() => setLlmSlot("green")}
                >
                  GREEN 绿槽 (:18082)
                </button>
                <button
                  type="button"
                  style={{ flex: 1 }}
                  className={llmSlot === "standalone" ? "sel" : ""}
                  onClick={() => setLlmSlot("standalone")}
                >
                  独立实例 (:18083+)
                </button>
              </div>
            </div>
          </div>

          <div className="field">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <label>模型 ID（HuggingFace 仓库 / GGUF 路径 / Ollama 模型名）</label>
              <div className="preset-pills">
                {MODEL_PRESETS.map((p) => {
                  const presetOk = isLlmEngineCompatibleWithAccelerator(p.engine, acc);
                  return (
                    <button
                      type="button"
                      key={p.label}
                      disabled={!presetOk}
                      className={`preset-pill ${llmModel === p.value ? "active" : ""}`}
                      title={
                        presetOk
                          ? `选用预设模型 ${p.value} (${p.engine})`
                          : `该预设使用 ${p.engine}（需 NVIDIA CUDA GPU），不支持当前硬件 ${acc}`
                      }
                      onClick={() => {
                        if (!presetOk) return;
                        setLlmModel(p.value);
                        setLlmEngine(p.engine);
                      }}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
            </div>
            <input
              value={llmModel}
              placeholder="例如 Qwen/Qwen2.5-7B-Instruct-GGUF 或自定义 HuggingFace 仓库路径"
              onChange={(e) => setLlmModel(e.target.value)}
            />
          </div>

          <div className="form-grid-2">
            <div className="field">
              <label>专属 API Key（留空自动生成 `sk-colab-...`）</label>
              <input
                value={customApiKey}
                placeholder="留空自动生成持久化专属密钥 sk-colab-..."
                onChange={(e) => setCustomApiKey(e.target.value)}
              />
            </div>

            <div className="field">
              <label>Google Drive 模型权重路径（可选，rclone 拉取）</label>
              <input
                value={gdriveModelPath}
                placeholder="例如 gdrive:models/qwen2.5-7b-gguf（留空直接从云端拉取）"
                onChange={(e) => {
                  const v = e.target.value;
                  setGdriveModelPath(v);
                  if (!userToggledSyncRcloneRef.current) {
                    setSyncRclone(Boolean(v.trim()));
                  }
                }}
              />
            </div>
          </div>
        </div>
      )}

      {isRelayFlow && (
        <div className="cfg-panel cfg-panel-relay">
          <div className="cfg-panel-head">
            <span className="cfg-panel-title">
              <Repeat2 size={14} />
              24h+ 跨实例自动接力训练配置
            </span>
            <span className="cfg-panel-badge">换棒硬件: {acc} · 自动断点热备</span>
          </div>

          <div className="form-grid-4">
            <div className="field">
              <label>总训练轮数 (Epochs)</label>
              <input
                type="number"
                min={1}
                value={relayCfg.total_epochs}
                onChange={(e) =>
                  setRelayCfg({ ...relayCfg, total_epochs: Math.max(1, Number(e.target.value) || 1) })
                }
              />
            </div>
            <div className="field">
              <label>单棒上限时长 (0.1~23.5h)</label>
              <input
                type="number"
                step={0.5}
                min={MIN_RELAY_SESSION_HOURS}
                max={MAX_RELAY_SESSION_HOURS}
                value={Number.isFinite(relayCfg.max_session_hours) ? relayCfg.max_session_hours : ""}
                style={!relayHoursOk ? { borderColor: "var(--danger)" } : undefined}
                onChange={(e) => {
                  const raw = e.target.value;
                  const v = raw === "" ? Number.NaN : Number(raw);
                  setRelayCfg({ ...relayCfg, max_session_hours: v });
                }}
              />
            </div>
            <div className="field">
              <label>云盘热备间隔 (分钟)</label>
              <input
                type="number"
                min={5}
                value={relayCfg.backup_interval_minutes}
                onChange={(e) =>
                  setRelayCfg({ ...relayCfg, backup_interval_minutes: Math.max(5, Number(e.target.value) || 30) })
                }
              />
            </div>
            <div className="field">
              <label>Google Drive 断点目录</label>
              <input
                value={relayCfg.gdrive_dir}
                style={!relayGdriveFormatOk ? { borderColor: "var(--danger)" } : undefined}
                onChange={(e) => {
                  setRelayCfg({ ...relayCfg, gdrive_dir: e.target.value });
                  setConfirmReuseRelayGdrive(false);
                }}
              />
            </div>
          </div>

          {!relayHoursOk && (
            <div className="field-err">
              受限于 Colab 单次最长运行周期，单棒上限时长必须在 {MIN_RELAY_SESSION_HOURS} ~ {MAX_RELAY_SESSION_HOURS} 小时之间。
            </div>
          )}

          {isDefaultBareGdriveDir(relayCfg.gdrive_dir) && (
            <div className="field-hint">
              默认目录将自动按实例子目录隔离为：
              <code>
                gdrive:colab_relay/{name.trim() ? sanitizeRelayVmSegment(name) : "<实例名>"}
              </code>
              （防止多个默认接力任务共用 checkpoint 与完成标记）
            </div>
          )}

          {relayConflictVm && (
            <div className="vm-alert-box vm-alert-warn" role="alert">
              <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: 2 }} />
              <div className="vm-alert-content">
                <div className="vm-alert-title">
                  云盘目录 <code>{effectiveRelayGdrive}</code> 正被运行中的接力实例 [{relayConflictVm.name}] 占用
                </div>
                <div className="vm-alert-desc">
                  多任务共用同一目录会互相覆盖断点与完成标记，建议修改为独立目录。
                </div>
                <label style={{ display: "inline-flex", alignItems: "center", gap: 6, marginTop: 5, cursor: "pointer" }}>
                  <input
                    type="checkbox"
                    checked={confirmReuseRelayGdrive}
                    onChange={(e) => setConfirmReuseRelayGdrive(e.target.checked)}
                  />
                  <span>我已知晓风险，显式确认复用该云盘目录续训</span>
                </label>
              </div>
            </div>
          )}

          <label style={{ display: "inline-flex", alignItems: "center", gap: 7, fontSize: 12, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={relayCfg.reset_done_flag}
              onChange={(e) => setRelayCfg({ ...relayCfg, reset_done_flag: e.target.checked })}
              style={{ accentColor: "var(--primary)" }}
            />
            <span>启动前清除云盘上的完成标记（从头训练）</span>
          </label>
        </div>
      )}

      {!isDeployExisting && (
        /* 5. Colab 高级环境与自动化配置折叠面板 */
        <div className="adv-config-box">
          <button
            type="button"
            className="adv-config-toggle"
            onClick={() => setShowAdvanced((v) => !v)}
          >
            <span style={{ display: "inline-flex", alignItems: "center", gap: 7, fontWeight: 600 }}>
              <Wrench size={14} />
              更多 Colab 运行时与环境配置（工作区绑定、SSH 免密、Google Drive 挂载、预装依赖、启动脚本）
            </span>
            {showAdvanced ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
          </button>

          {showAdvanced && (
            <div className="adv-config-body">
              <div className="form-grid-2">
                <div className="field">
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <FolderGit2 size={12} />
                    绑定本地工作区目录 (Workspace Path)
                  </label>
                  <input
                    value={workspacePath}
                    placeholder="默认绑定当前项目工作区目录"
                    onChange={(e) => setWorkspacePath(e.target.value)}
                  />
                </div>

                <div className="field">
                  <label style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <Server size={12} />
                    实例用途说明 / 备注
                  </label>
                  <input
                    value={description}
                    placeholder="例如：Qwen2.5 推理网关 / YOLOv11 接力训练"
                    onChange={(e) => setDescription(e.target.value)}
                  />
                </div>
              </div>

              <div className="adv-checks-grid">
                <label className="adv-check-item">
                  <input type="checkbox" checked readOnly style={{ accentColor: "var(--primary)" }} />
                  <div>
                    <div className="t">
                      <ShieldCheck size={13} /> 自动配置 Ed25519 免密 SSH 与本机端口转发
                    </div>
                    <div className="d">自动写入 `~/.ssh/config`，支持 `ssh colab-&lt;name&gt;`、`colab ssh -s &lt;name&gt;` 与一键打开终端</div>
                  </div>
                </label>

                <label className="adv-check-item">
                  <input
                    type="checkbox"
                    checked={syncRclone}
                    onChange={(e) => {
                      userToggledSyncRcloneRef.current = true;
                      setSyncRclone(e.target.checked);
                    }}
                    style={{ accentColor: "var(--primary)" }}
                  />
                  <div>
                    <div className="t">
                      <HardDrive size={13} /> 自动同步本机 `rclone.conf` 到云端 `/content/rclone.conf`
                    </div>
                    <div className="d">使云端实例无需重新授权即可高速读写 Google Drive（通用模式下默认关闭）</div>
                  </div>
                </label>

                <label className="adv-check-item">
                  <input
                    type="checkbox"
                    checked={mountGdrive}
                    onChange={(e) => setMountGdrive(e.target.checked)}
                    style={{ accentColor: "var(--primary)" }}
                  />
                  <div>
                    <div className="t">自动挂载 Google Drive 到 `/content/drive`</div>
                    <div className="d">开机后后台通过 rclone mount 挂载网盘文件系统</div>
                  </div>
                </label>

                <label className="adv-check-item">
                  <input
                    type="checkbox"
                    checked={installReqTxt}
                    onChange={(e) => setInstallReqTxt(e.target.checked)}
                    style={{ accentColor: "var(--primary)" }}
                  />
                  <div>
                    <div className="t">自动上传并安装工作区 `requirements.txt`</div>
                    <div className="d">开机后使用 `uv pip install --system -r` 高速安装依赖</div>
                  </div>
                </label>
              </div>

              <div className="form-grid-2">
                <div className="field">
                  <label>开机预装 Python 包（空格或逗号分隔，调用 `colab install`）</label>
                  <input
                    value={pipPackages}
                    placeholder="例如 transformers accelerate bitsandbytes wandb"
                    onChange={(e) => setPipPackages(e.target.value)}
                  />
                </div>

                {/* #69：多行环境变量输入框，与每行一个 KEY=VALUE 提示及后端按行解析严格对齐 */}
                <div className="field">
                  <label>注入环境变量（每行一个 `KEY=VALUE`）</label>
                  <textarea
                    rows={3}
                    value={envVars}
                    placeholder={"例如：\nHF_TOKEN=hf_xxx\nWANDB_API_KEY=xxx"}
                    onChange={(e) => setEnvVars(e.target.value)}
                  />
                </div>
              </div>

              <div className="field">
                <label>开机初始化 Python 脚本（可选，实例分配就绪后自动在 `/content` 执行）</label>
                <textarea
                  rows={3}
                  value={startupScript}
                  placeholder={"# 例如：\nimport subprocess\nsubprocess.run('nvidia-smi', shell=True)"}
                  onChange={(e) => setStartupScript(e.target.value)}
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* #67：无论高级配置是否折叠，只要当前配置依赖 Google Drive 而本机缺少 rclone.conf，即在提交按钮上方醒目提示 */}
      {missingGdriveEnv && (
        <div className="banner banner-warn" role="alert" style={{ margin: 0, alignItems: "center" }}>
          <AlertTriangle size={16} style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700 }}>当前配置依赖 Google Drive（rclone.conf）授权</div>
            <div style={{ fontSize: 12, marginTop: 2 }}>
              {!props.env?.rclone_ok
                ? `未检测到 rclone 可执行文件（${props.env?.rclone_bin ?? "rclone"}）。`
                : `未检测到本机 rclone 授权配置文件（${props.env?.rclone_conf ?? "rclone.conf"}）。请先在终端运行 rclone config 完成 Google Drive 授权，或关闭网盘同步选项。`}
            </div>
          </div>
          {!isDeployExisting && bootMode !== "relay" && (syncRclone || mountGdrive) && (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => {
                userToggledSyncRcloneRef.current = true;
                setSyncRclone(false);
                setMountGdrive(false);
              }}
            >
              关闭网盘同步选项
            </button>
          )}
        </div>
      )}
    </Modal>
  );
}
