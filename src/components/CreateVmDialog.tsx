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
  HeartPulse,
  History,
  KeyRound,
  Loader2,
  Power,
  PowerOff,
  Repeat2,
  Server,
  ShieldCheck,
  Sparkles,
  Terminal,
  Wrench,
  Zap,
} from "lucide-react";
import { errText } from "../api";
import {
  DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS,
  DEFAULT_RELAY_GRAB_RETRY_SECONDS,
  DEFAULT_VM_GRAB_RETRY_SECONDS,
  KEEP_ALIVE_PRESET_INTERVALS,
  MAX_GRAB_RETRY_SECONDS,
  MAX_KEEP_ALIVE_INTERVAL_SECONDS,
  MAX_RELAY_SESSION_HOURS,
  MIN_GRAB_RETRY_SECONDS,
  MIN_KEEP_ALIVE_INTERVAL_SECONDS,
  MIN_RELAY_SESSION_HOURS,
  RECOMMENDED_MAX_RELAY_SESSION_HOURS,
  checkCreateVmAuthGuard,
  describeGrabStatus,
  findConflictingRelayVm,
  isDefaultBareGdriveDir,
  isLlmEngineCompatibleWithAccelerator,
  isNvidiaCudaAccelerator,
  isSingleShapeAccelerator,
  isValidGrabRetrySeconds,
  isValidKeepAliveInterval,
  isValidRelayGrabTimeoutMinutes,
  isValidRelaySessionHours,
  normalizeKeepAliveConfig,
  parseFallbackGpuInput,
  requiresGdriveConfig,
  resolveEffectiveHighMem,
  resolveEffectiveRelayGdriveDir,
  sanitizeRelayVmSegment,
} from "../chatUtils";
import { ColabIcon, Modal } from "./ui";
import {
  DEFAULT_GRAB_CONFIG,
  DEFAULT_KEEP_ALIVE,
  DEFAULT_RELAY,
  type CreateVmLlmConfig,
  type CreateVmRequest,
  type EnvStatus,
  type GrabConfig,
  type KeepAliveConfig,
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
  keep_alive?: KeepAliveConfig | null;
  grab_config?: GrabConfig | null;
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
  { v: "transformers", label: "HF Transformers (CUDA)", desc: "原生 PyTorch CUDA bfloat16 流式生成服务", requiresCuda: true },
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
  onCancelCreate?: (name: string) => Promise<void>;
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
  const [relayFallbacksRaw, setRelayFallbacksRaw] = useState("");
  const [confirmReuseRelayGdrive, setConfirmReuseRelayGdrive] = useState(false);

  // 开机抢卡策略（支持快速探卡 / 标准重试 / 一直抢卡）
  const [grabCfg, setGrabCfg] = useState<GrabConfig>({ ...DEFAULT_GRAB_CONFIG });
  const [grabFallbacksRaw, setGrabFallbacksRaw] = useState("");

  // 新建实例默认每 90 秒检查运行时连接并在 Kernel 空闲时发送脉冲。
  const [keepAliveCfg, setKeepAliveCfg] = useState<KeepAliveConfig>({ ...DEFAULT_KEEP_ALIVE });

  // Colab 高级运行时环境配置（#67：通用模式下 syncRclone 默认 false）
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [syncRclone, setSyncRclone] = useState(false);
  const [mountGdrive, setMountGdrive] = useState(false);
  const [installReqTxt, setInstallReqTxt] = useState(false);
  const [pipPackages, setPipPackages] = useState("");
  const [envVars, setEnvVars] = useState("");
  const [startupScript, setStartupScript] = useState("");

  const [busy, setBusy] = useState(false);
  const [activeCreatingName, setActiveCreatingName] = useState<string | null>(null);
  const [cancellingCreate, setCancellingCreate] = useState(false);
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000));
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [copiedError, setCopiedError] = useState(false);
  const prevOpenRef = useRef(false);
  const openRef = useRef(props.open);
  openRef.current = props.open;
  const cancelledByUserRef = useRef(false);
  const userToggledSyncRcloneRef = useRef(false);

  const targetExistingVm = props.preset?.target_existing_vm?.trim() || null;
  const isDeployExisting = Boolean(targetExistingVm);

  useEffect(() => {
    if (!busy || !activeCreatingName) return;
    setNowSec(Math.floor(Date.now() / 1000));
    const timer = window.setInterval(() => {
      setNowSec(Math.floor(Date.now() / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [busy, activeCreatingName]);

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
    setActiveCreatingName(null);
    setCancellingCreate(false);
    cancelledByUserRef.current = false;
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

    const initRelay = p?.relay_config
      ? { ...DEFAULT_RELAY, ...p.relay_config }
      : { ...DEFAULT_RELAY, gpu_type: validAcc };
    setRelayCfg(initRelay);
    setRelayFallbacksRaw((initRelay.fallback_gpus ?? []).join(", "));
    setConfirmReuseRelayGdrive(false);

    const initGrab = p?.grab_config
      ? { ...DEFAULT_GRAB_CONFIG, ...p.grab_config }
      : { ...DEFAULT_GRAB_CONFIG };
    setGrabCfg(initGrab);
    setGrabFallbacksRaw((initGrab.fallback_gpus ?? []).join(", "));

    setKeepAliveCfg(
      p?.keep_alive
        ? normalizeKeepAliveConfig({ ...DEFAULT_KEEP_ALIVE, ...p.keep_alive })
        : { ...DEFAULT_KEEP_ALIVE },
    );

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

  const keepAliveIntervalOk =
    isDeployExisting ||
    !keepAliveCfg.enabled ||
    isValidKeepAliveInterval(keepAliveCfg.interval_seconds);

  const parsedVmFallbacks = parseFallbackGpuInput(grabFallbacksRaw, acc);
  const parsedRelayFallbacks = parseFallbackGpuInput(relayFallbacksRaw, acc);

  const vmGrabRetryOk =
    isDeployExisting ||
    isRelayFlow ||
    grabCfg.mode === "fast" ||
    isValidGrabRetrySeconds(grabCfg.retry_seconds);
  const vmGrabFallbacksOk =
    isDeployExisting || isRelayFlow || parsedVmFallbacks.invalid.length === 0;

  const relayGrabMode = relayCfg.grab_mode ?? "budget";
  const relayGrabBudgetOk =
    !isRelayFlow ||
    relayGrabMode !== "budget" ||
    isValidRelayGrabTimeoutMinutes(relayCfg.grab_timeout_minutes ?? 60);
  const relayGrabRetryOk =
    !isRelayFlow ||
    relayGrabMode === "fast" ||
    isValidGrabRetrySeconds(relayCfg.grab_retry_seconds ?? DEFAULT_RELAY_GRAB_RETRY_SECONDS);
  const relayGrabFallbacksOk = !isRelayFlow || parsedRelayFallbacks.invalid.length === 0;

  const liveCreatingVm =
    busy && activeCreatingName
      ? (props.vms ?? []).find((v) => v.name === activeCreatingName)
      : undefined;
  const liveGrabSummary = describeGrabStatus(liveCreatingVm?.grab_status, nowSec);

  const canSubmit =
    !busy &&
    nameOk &&
    llmModelOk &&
    llmHwCompatible &&
    !authGuard.blocked &&
    relayHoursOk &&
    relayGdriveFormatOk &&
    !relayConflictBlocked &&
    keepAliveIntervalOk &&
    vmGrabRetryOk &&
    vmGrabFallbacksOk &&
    relayGrabBudgetOk &&
    relayGrabRetryOk &&
    relayGrabFallbacksOk;

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

  const handleCancelGrab = async () => {
    if (!activeCreatingName || !props.onCancelCreate || cancellingCreate) return;
    setCancellingCreate(true);
    cancelledByUserRef.current = true;
    try {
      await props.onCancelCreate(activeCreatingName);
      setSubmitError(null);
    } catch (e) {
      setSubmitError(errText(e));
    } finally {
      setCancellingCreate(false);
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
    if (!keepAliveIntervalOk) {
      setSubmitError(
        `连接检查间隔必须在 ${MIN_KEEP_ALIVE_INTERVAL_SECONDS} ~ ${MAX_KEEP_ALIVE_INTERVAL_SECONDS} 秒之间。`,
      );
      return;
    }
    if (!vmGrabRetryOk || !relayGrabRetryOk) {
      setSubmitError(
        `抢卡重试间隔必须在 ${MIN_GRAB_RETRY_SECONDS} ~ ${MAX_GRAB_RETRY_SECONDS} 秒之间。`,
      );
      return;
    }
    if (!relayGrabBudgetOk) {
      setSubmitError("接力抢卡等待上限必须在 1 ~ 1440 分钟（24 小时）之间。");
      return;
    }
    if (!vmGrabFallbacksOk || !relayGrabFallbacksOk) {
      const badList = isRelayFlow ? parsedRelayFallbacks.invalid : parsedVmFallbacks.invalid;
      setSubmitError(
        `不支持的备选硬件型号：${badList.join(", ")}（可选值：T4, L4, G4, A100, H100, V5E1, V6E1, CPU）。`,
      );
      return;
    }
    if (!nameOk || !llmModelOk || !relayGdriveFormatOk) return;

    setSubmitError(null);
    setCopiedError(false);
    cancelledByUserRef.current = false;
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
        if (openRef.current) props.onClose();
        return;
      }

      const concreteName =
        name.trim() || `vm-${Math.random().toString(36).slice(2, 8).padEnd(6, "0")}`;
      setActiveCreatingName(concreteName);

      const req: CreateVmRequest = {
        name: concreteName,
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
                grab_mode: relayGrabMode,
                grab_timeout_minutes: Number.isFinite(relayCfg.grab_timeout_minutes)
                  ? Math.round(relayCfg.grab_timeout_minutes ?? 60)
                  : 60,
                grab_retry_seconds: Number.isFinite(relayCfg.grab_retry_seconds)
                  ? Math.round(relayCfg.grab_retry_seconds ?? DEFAULT_RELAY_GRAB_RETRY_SECONDS)
                  : DEFAULT_RELAY_GRAB_RETRY_SECONDS,
                fallback_gpus: parsedRelayFallbacks.gpus,
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
        keep_alive: normalizeKeepAliveConfig(keepAliveCfg),
        grab_config: {
          mode: grabCfg.mode,
          retry_seconds: Number.isFinite(grabCfg.retry_seconds)
            ? Math.round(grabCfg.retry_seconds)
            : DEFAULT_VM_GRAB_RETRY_SECONDS,
          fallback_gpus: parsedVmFallbacks.gpus,
        },
      };
      await props.onCreate(req);
      if (openRef.current) {
        setName("");
        props.onClose();
      }
    } catch (e) {
      const msg = errText(e);
      if (cancelledByUserRef.current || msg.includes("放弃抢卡")) {
        if (openRef.current) setSubmitError(null);
      } else if (openRef.current) {
        setSubmitError(msg);
      }
    } finally {
      setBusy(false);
      setActiveCreatingName(null);
      setCancellingCreate(false);
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
                <span>分配成功后即刻关闭弹窗并在后台注入监控与 SSH</span>
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
          {busy && !isDeployExisting ? (
            <>
              <button
                type="button"
                className="btn"
                onClick={props.onClose}
                title="关闭此弹窗，继续在仪表盘实例卡片上后台抢卡（卡片上同样可随时放弃抢卡）"
              >
                转入后台抢卡
              </button>
              {props.onCancelCreate && activeCreatingName && (
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleCancelGrab}
                  disabled={cancellingCreate}
                  title="立即中止当前抢卡进程并核验释放残留实例"
                >
                  {cancellingCreate ? (
                    <>
                      <Loader2 size={14} className="spin" />
                      <span>正在停止抢卡…</span>
                    </>
                  ) : (
                    <>
                      <PowerOff size={14} />
                      <span>放弃抢卡</span>
                    </>
                  )}
                </button>
              )}
            </>
          ) : (
            <button type="button" className="btn" onClick={props.onClose} disabled={busy}>
              取消
            </button>
          )}
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
                  : liveGrabSummary?.countdownSec !== null && liveGrabSummary?.countdownSec !== undefined
                    ? `${liveGrabSummary.countdownSec}s 后自动重试抢卡…`
                    : `正在抢卡分配 (${activeCreatingName ?? acc})…`}
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
                    : grabCfg.mode === "persistent"
                      ? `开始一直抢卡 (${acc})`
                      : `立即开机 (${acc})`}
              </>
            )}
          </button>
        </>
      }
    >
      {/* 实时抢卡状态面板：在弹窗等待期间直观展示抢卡轮次、倒计时与一键放弃/转后台按钮 */}
      {busy && !isDeployExisting && (
        <div className="banner banner-info" role="status" style={{ margin: 0, alignItems: "center" }}>
          <Loader2 size={16} className="spin" style={{ flexShrink: 0 }} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 700 }}>
              {liveGrabSummary
                ? liveGrabSummary.headline
                : `正在向 Google Colab 请求分配 [${acc}] 实例 (${activeCreatingName ?? ""})…`}
            </div>
            <div style={{ fontSize: 12, marginTop: 2, opacity: 0.9 }}>
              {liveGrabSummary
                ? liveGrabSummary.subline
                : "单次请求无库存时约 25 秒内立即反馈；获得实例后将自动关闭弹窗并在后台完成初始化。"}
            </div>
          </div>
        </div>
      )}
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

          {relayHoursOk && relayCfg.max_session_hours > RECOMMENDED_MAX_RELAY_SESSION_HOURS && (
            <div className="vm-alert-box vm-alert-warn" role="status" style={{ alignItems: "center" }}>
              <AlertTriangle size={14} style={{ flexShrink: 0 }} />
              <div className="vm-alert-content">
                <div className="vm-alert-title">
                  建议预留 ≥1.5h 换棒安全缓冲窗口（当前距 24h 强制断连仅剩 {(24 - relayCfg.max_session_hours).toFixed(1)}h）
                </div>
                <div className="vm-alert-desc">
                  预留充足缓冲可确保大模型权重完整落盘并上传至 Google Drive，防止在最后一刻被硬性回收。
                </div>
              </div>
              <button
                type="button"
                className="btn btn-sm"
                onClick={() =>
                  setRelayCfg({ ...relayCfg, max_session_hours: RECOMMENDED_MAX_RELAY_SESSION_HOURS })
                }
              >
                设为推荐 22.5h
              </button>
            </div>
          )}

          {/* 接力换棒抢卡时间与准备策略 */}
          <div className="grab-policy-subbox">
            <div className="field" style={{ marginBottom: 6 }}>
              <label style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                <span>换棒抢卡策略与时间预算（换棒排队期间上一棒已停机释放，0 CU 计费）</span>
                <span className="dim" style={{ fontWeight: 500 }}>
                  停机前自动核验并抢救同步网盘断点
                </span>
              </label>
              <div className="seg" style={{ marginTop: 4 }}>
                <button
                  type="button"
                  className={relayGrabMode === "budget" ? "sel" : ""}
                  onClick={() => setRelayCfg({ ...relayCfg, grab_mode: "budget" })}
                >
                  ⏱️ 限时蹲守（默认 {relayCfg.grab_timeout_minutes ?? 60}m）
                </button>
                <button
                  type="button"
                  className={relayGrabMode === "persistent" ? "sel" : ""}
                  onClick={() => setRelayCfg({ ...relayCfg, grab_mode: "persistent" })}
                >
                  🎯 一直抢卡（直到有卡）
                </button>
                <button
                  type="button"
                  className={relayGrabMode === "fast" ? "sel" : ""}
                  onClick={() => setRelayCfg({ ...relayCfg, grab_mode: "fast" })}
                >
                  ⚡ 单次探卡（无卡立停）
                </button>
              </div>
            </div>

            <div className="form-grid-3">
              {relayGrabMode === "budget" && (
                <div className="field">
                  <label>每棒抢卡等待上限 (分钟)</label>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                    <input
                      type="number"
                      min={1}
                      max={1440}
                      value={
                        Number.isFinite(relayCfg.grab_timeout_minutes)
                          ? relayCfg.grab_timeout_minutes
                          : ""
                      }
                      style={
                        !relayGrabBudgetOk
                          ? { borderColor: "var(--danger)", width: 82 }
                          : { width: 82 }
                      }
                      onChange={(e) => {
                        const raw = e.target.value;
                        setRelayCfg({
                          ...relayCfg,
                          grab_timeout_minutes: raw === "" ? Number.NaN : Number(raw),
                        });
                      }}
                    />
                    <div className="preset-pills">
                      {[30, 60, 180, 360].map((m) => (
                        <button
                          type="button"
                          key={m}
                          className={`preset-pill ${(relayCfg.grab_timeout_minutes ?? 60) === m ? "active" : ""}`}
                          onClick={() => setRelayCfg({ ...relayCfg, grab_timeout_minutes: m })}
                        >
                          {m >= 60 ? `${m / 60}h` : `${m}m`}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {relayGrabMode !== "fast" && (
                <div className="field">
                  <label>抢卡重试间隔 ({MIN_GRAB_RETRY_SECONDS}~{MAX_GRAB_RETRY_SECONDS}s)</label>
                  <input
                    type="number"
                    min={MIN_GRAB_RETRY_SECONDS}
                    max={MAX_GRAB_RETRY_SECONDS}
                    value={
                      Number.isFinite(relayCfg.grab_retry_seconds)
                        ? relayCfg.grab_retry_seconds
                        : ""
                    }
                    style={!relayGrabRetryOk ? { borderColor: "var(--danger)" } : undefined}
                    onChange={(e) => {
                      const raw = e.target.value;
                      setRelayCfg({
                        ...relayCfg,
                        grab_retry_seconds: raw === "" ? Number.NaN : Number(raw),
                      });
                    }}
                  />
                </div>
              )}

              <div className="field" style={{ gridColumn: relayGrabMode === "fast" ? "1 / -1" : undefined }}>
                <label>换棒备选卡型（可选，逗号分隔如 G4, L4；留空严格只抢 {acc}）</label>
                <input
                  value={relayFallbacksRaw}
                  placeholder={`留空严格只抢 [${acc}]；如填 G4, L4 则在主卡售罄时轮询备选卡`}
                  style={!relayGrabFallbacksOk ? { borderColor: "var(--danger)" } : undefined}
                  onChange={(e) => setRelayFallbacksRaw(e.target.value)}
                />
              </div>
            </div>

            {!relayGrabFallbacksOk && (
              <div className="field-err">
                存在不支持的备选硬件：{parsedRelayFallbacks.invalid.join(", ")}
              </div>
            )}
          </div>

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

      {!isDeployExisting && !isRelayFlow && (
        /* 4.2 开机抢卡策略（支持快速探卡 / 标准重试 / 一直抢卡 + 随时放弃抢卡） */
        <div className="cfg-panel cfg-panel-grab">
          <div className="cfg-panel-head">
            <span className="cfg-panel-title">
              <Zap size={14} color="var(--primary)" />
              开机抢卡策略（无卡快速反馈 / 一直抢卡蹲守 / 随时可放弃）
            </span>
            <span className="cfg-panel-badge">
              {grabCfg.mode === "persistent"
                ? "一直抢卡直到成功"
                : grabCfg.mode === "standard"
                  ? "标准重试 3 轮"
                  : "快速探卡·没卡立报"}
            </span>
          </div>

          <div className="seg">
            <button
              type="button"
              className={grabCfg.mode === "fast" ? "sel" : ""}
              onClick={() => setGrabCfg((prev) => ({ ...prev, mode: "fast" }))}
            >
              ⚡ 快速探卡（没卡约 25s 立报）
            </button>
            <button
              type="button"
              className={grabCfg.mode === "standard" ? "sel" : ""}
              onClick={() => setGrabCfg((prev) => ({ ...prev, mode: "standard" }))}
            >
              🔁 标准重试（自动尝试 3 轮）
            </button>
            <button
              type="button"
              className={grabCfg.mode === "persistent" ? "sel" : ""}
              onClick={() => setGrabCfg((prev) => ({ ...prev, mode: "persistent" }))}
            >
              🎯 一直抢卡（持续蹲守直到有卡）
            </button>
          </div>

          <div className="form-grid-2" style={{ marginTop: 4 }}>
            {grabCfg.mode !== "fast" && (
              <div className="field">
                <label>抢卡重试间隔 ({MIN_GRAB_RETRY_SECONDS}~{MAX_GRAB_RETRY_SECONDS}s)</label>
                <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                  <input
                    type="number"
                    min={MIN_GRAB_RETRY_SECONDS}
                    max={MAX_GRAB_RETRY_SECONDS}
                    value={Number.isFinite(grabCfg.retry_seconds) ? grabCfg.retry_seconds : ""}
                    style={!vmGrabRetryOk ? { borderColor: "var(--danger)", width: 84 } : { width: 84 }}
                    onChange={(e) => {
                      const raw = e.target.value;
                      setGrabCfg((prev) => ({
                        ...prev,
                        retry_seconds: raw === "" ? Number.NaN : Number(raw),
                      }));
                    }}
                  />
                  <div className="preset-pills">
                    {[10, 15, 30, 60].map((sec) => (
                      <button
                        type="button"
                        key={sec}
                        className={`preset-pill ${grabCfg.retry_seconds === sec ? "active" : ""}`}
                        onClick={() => setGrabCfg((prev) => ({ ...prev, retry_seconds: sec }))}
                      >
                        {sec}s
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div className="field" style={{ gridColumn: grabCfg.mode === "fast" ? "1 / -1" : undefined }}>
              <label>备选卡型（可选，逗号分隔如 G4, L4；留空严格只抢 {acc}）</label>
              <input
                value={grabFallbacksRaw}
                placeholder={`留空严格只抢 [${acc}]；如填 G4, L4 则在 [${acc}] 售罄时自动轮询`}
                style={!vmGrabFallbacksOk ? { borderColor: "var(--danger)" } : undefined}
                onChange={(e) => setGrabFallbacksRaw(e.target.value)}
              />
            </div>
          </div>

          {!vmGrabFallbacksOk && (
            <div className="field-err">
              存在不支持的备选硬件：{parsedVmFallbacks.invalid.join(", ")}
            </div>
          )}

          <div className="field-hint">
            {grabCfg.mode === "persistent"
              ? `将每 ${Number.isFinite(grabCfg.retry_seconds) ? Math.round(grabCfg.retry_seconds) : DEFAULT_VM_GRAB_RETRY_SECONDS}s 循环请求 [${acc}]，期间可点击「转入后台抢卡」收起弹窗，或在弹窗/卡片上随时点击「放弃抢卡」立即停止。`
              : grabCfg.mode === "standard"
                ? `若遇到库存紧张或拥堵，将自动重试最多 3 轮（间隔 ${Number.isFinite(grabCfg.retry_seconds) ? Math.round(grabCfg.retry_seconds) : DEFAULT_VM_GRAB_RETRY_SECONDS}s），期间随时可点击「放弃抢卡」。`
                : `已移除内部 3 分 15 秒静默退避重试：有卡时即刻分配并秒关弹窗；无卡 (503) 时约 25 秒内立即报错，方便快速切换卡型。`}
          </div>
        </div>
      )}

      {!isDeployExisting && (
        /* 4.5 实例保活策略（只要创建实例，统一配置保活策略） */
        <div className={`cfg-panel cfg-panel-keepalive ${keepAliveCfg.enabled ? "is-on" : "is-off"}`}>
          <div className="cfg-panel-head">
            <span className="cfg-panel-title">
              <HeartPulse size={14} color={keepAliveCfg.enabled ? "var(--success)" : "var(--muted)"} />
              实例保活策略
            </span>
            <label className="ka-inline-switch">
              <input
                type="checkbox"
                checked={keepAliveCfg.enabled}
                onChange={(e) =>
                  setKeepAliveCfg((prev) =>
                    normalizeKeepAliveConfig({ ...prev, enabled: e.target.checked }),
                  )
                }
                style={{ accentColor: "var(--success)" }}
              />
              <span>{keepAliveCfg.enabled ? "开机后自动检查运行时连接" : "暂不开启（手动控制）"}</span>
            </label>
          </div>

          <div className="ka-create-row">
            <div className="field" style={{ minWidth: 165 }}>
              <label>连接检查间隔 ({MIN_KEEP_ALIVE_INTERVAL_SECONDS}~{MAX_KEEP_ALIVE_INTERVAL_SECONDS}s)</label>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <input
                  type="number"
                  min={MIN_KEEP_ALIVE_INTERVAL_SECONDS}
                  max={MAX_KEEP_ALIVE_INTERVAL_SECONDS}
                  step={10}
                  disabled={!keepAliveCfg.enabled}
                  value={Number.isFinite(keepAliveCfg.interval_seconds) ? keepAliveCfg.interval_seconds : ""}
                  style={!keepAliveIntervalOk ? { borderColor: "var(--danger)", width: 88 } : { width: 88 }}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const v = raw === "" ? Number.NaN : Number(raw);
                    setKeepAliveCfg((prev) => ({ ...prev, interval_seconds: v }));
                  }}
                />
                <div className="preset-pills">
                  {KEEP_ALIVE_PRESET_INTERVALS.map((sec) => (
                    <button
                      type="button"
                      key={sec}
                      disabled={!keepAliveCfg.enabled}
                      className={`preset-pill ${keepAliveCfg.interval_seconds === sec ? "active" : ""}`}
                      onClick={() =>
                        setKeepAliveCfg((prev) => ({ ...prev, interval_seconds: sec }))
                      }
                    >
                      {sec}s
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <label
              className="ka-kernel-opt"
              style={{ opacity: keepAliveCfg.enabled ? 1 : 0.55 }}
            >
              <input
                type="checkbox"
                disabled={!keepAliveCfg.enabled}
                checked={keepAliveCfg.kernel_ping}
                onChange={(e) =>
                  setKeepAliveCfg((prev) => ({ ...prev, kernel_ping: e.target.checked }))
                }
                style={{ accentColor: "var(--primary)" }}
              />
              <div>
                <div className="t">启用轻量 Jupyter Kernel 脉冲心跳（<code>pass</code>）</div>
                <div className="d">
                  仅在 Kernel 空闲时发送脉冲，任务运行时自动跳过。
                </div>
              </div>
            </label>
          </div>

          {!keepAliveIntervalOk && (
            <div className="field-err">
              连接检查间隔必须在 {MIN_KEEP_ALIVE_INTERVAL_SECONDS} ~ {MAX_KEEP_ALIVE_INTERVAL_SECONDS} 秒之间。
            </div>
          )}

          <div className="field-hint" style={{ marginTop: 2 }}>
            {keepAliveCfg.enabled
              ? `开机后每 ${Number.isFinite(keepAliveCfg.interval_seconds) ? Math.round(keepAliveCfg.interval_seconds) : DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS} 秒检查运行时连接${keepAliveCfg.kernel_ping ? "，并在 Kernel 空闲时发送脉冲" : ""}。`
              : "可在实例卡片上开启运行时连接检查。"}
          </div>
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
