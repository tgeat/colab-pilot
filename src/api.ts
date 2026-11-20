import { invoke } from "@tauri-apps/api/core";
import type {
  CreateVmLlmConfig,
  CreateVmRequest,
  LlmApiAccessInfo,
  LlmChatRequest,
  LlmChatResponse,
  Overview,
  RelayConfig,
  RelayStatus,
  StopRelayResult,
  TerminateReceipt,
  WorkspaceFiles,
} from "./types";

export const api = {
  getOverview: () => invoke<Overview>("get_overview"),
  refreshNow: () => invoke<Overview>("refresh_now"),
  createVm: (req: CreateVmRequest) => invoke<string>("create_vm", { req }),
  deleteVm: (name: string, endpoint: string | null) =>
    invoke<TerminateReceipt>("delete_vm", { name, endpoint }),
  deployProbe: (name: string) => invoke<void>("deploy_probe", { name }),
  startRelay: (
    name: string,
    config: RelayConfig,
    workspacePath?: string | null,
    legacyGdriveAction?: "migrate" | "ignore" | null,
  ) =>
    invoke<RelayStatus>("start_relay", {
      name,
      config,
      workspacePath: workspacePath ?? null,
      ...(legacyGdriveAction ? { legacyGdriveAction } : {}),
    }),
  stopRelay: (name: string, destroyVm: boolean) =>
    invoke<StopRelayResult>("stop_relay", { name, destroyVm }),
  readWorkspaceFiles: (workspacePath?: string | null, vmName?: string | null) =>
    invoke<WorkspaceFiles>("read_workspace_files", {
      workspacePath: workspacePath ?? null,
      vmName: vmName ?? null,
    }),
  saveWorkspaceFiles: (
    trainPy: string,
    requirementsTxt: string,
    workspacePath?: string | null,
    vmName?: string | null,
  ) =>
    invoke<void>("save_workspace_files", {
      trainPy,
      requirementsTxt,
      workspacePath: workspacePath ?? null,
      vmName: vmName ?? null,
    }),
  setScriptDirty: (dirty: boolean) => invoke<void>("set_script_dirty", { dirty }),
  streamLlmChat: (req: LlmChatRequest) => invoke<LlmChatResponse>("stream_llm_chat", { req }),
  cancelLlmChat: (requestId: string) => invoke<void>("cancel_llm_chat", { requestId }),
  openVmSshTerminal: (name: string) => invoke<string>("open_vm_ssh_terminal", { name }),
  rotateLlmApiKey: (vmName: string, customKey?: string | null) =>
    invoke<LlmApiAccessInfo>("rotate_llm_api_key", {
      vmName,
      customKey: customKey ?? null,
    }),
  deployLlmOnVm: (name: string, config: CreateVmLlmConfig) =>
    invoke<LlmApiAccessInfo>("deploy_llm_on_vm", { name, config }),
  confirmAppClose: (mode: "keep_vms" | "shutdown_vms" | "force_exit") =>
    invoke<TerminateReceipt[]>("confirm_app_close", { mode }),
  dismissRelayStatus: (name: string) => invoke<void>("dismiss_relay_status", { name }),
};

/**
 * 兼容解析新结构 `StopRelayResult` 与旧版测试模拟 `TerminateReceipt | null`（#40, #41）。
 */
export function normalizeStopRelayResult(
  raw: StopRelayResult | TerminateReceipt | null | undefined,
  destroyVm: boolean,
): StopRelayResult {
  if (raw && typeof raw === "object" && "supervisor_stopped" in raw) {
    const r = raw as StopRelayResult;
    return {
      supervisor_stopped: Boolean(r.supervisor_stopped),
      destroy_vm: typeof r.destroy_vm === "boolean" ? r.destroy_vm : destroyVm,
      kernel_interrupted: Boolean(r.kernel_interrupted),
      kernel_interrupt_error: r.kernel_interrupt_error ?? null,
      terminate_receipt: r.terminate_receipt ?? null,
    };
  }
  if (raw && typeof raw === "object" && "verified_unassigned" in raw) {
    const receipt = raw as TerminateReceipt;
    return {
      supervisor_stopped: true,
      destroy_vm: destroyVm,
      kernel_interrupted: Boolean(receipt.verified_unassigned),
      kernel_interrupt_error: receipt.error ?? null,
      terminate_receipt: receipt,
    };
  }
  return {
    supervisor_stopped: true,
    destroy_vm: destroyVm,
    kernel_interrupted: !destroyVm,
    kernel_interrupt_error: destroyVm ? "未收到服务端停费核验回执" : null,
    terminate_receipt: null,
  };
}

/**
 * 根据 `StopRelayResult` 计算准确的停止接力提示信息与状态（#40, #41）：
 * - destroy === true：仅当 `terminate_receipt?.verified_unassigned === true` 才判定已销毁成功！
 * - destroy === false：仅当 `kernel_interrupted === true` 才提示云端训练已中断；否则明确告警原训练可能仍在运行。
 */
export function evaluateStopRelayOutcome(
  vmName: string,
  destroyVm: boolean,
  raw: StopRelayResult | TerminateReceipt | null | undefined,
): {
  result: StopRelayResult;
  ok: boolean;
  toastKind: "success" | "info" | "warn" | "error";
  message: string;
  keepDialogOpen: boolean;
} {
  const res = normalizeStopRelayResult(raw, destroyVm);
  if (destroyVm) {
    const receipt = res.terminate_receipt;
    if (receipt?.verified_unassigned === true) {
      return {
        result: res,
        ok: true,
        toastKind: "success",
        message: `已关闭 [${vmName}] 接力并销毁云端实例`,
        keepDialogOpen: false,
      };
    }
    const reason =
      receipt?.error ||
      res.kernel_interrupt_error ||
      (receipt?.remaining_endpoints?.length
        ? `剩余未释放端点: ${receipt.remaining_endpoints.join(", ")}`
        : "未通过服务端计费列表二次核验");
    return {
      result: res,
      ok: false,
      toastKind: "error",
      message: `本机接力已停止，但云端实例 [${vmName}] 停费核验未通过（可能仍在计费）：${reason}`,
      keepDialogOpen: true,
    };
  }

  if (res.kernel_interrupted) {
    return {
      result: res,
      ok: true,
      toastKind: "info",
      message: `本机接力监工已停止，且 [${vmName}] 云端训练已中断（实例保留，仍在计费）`,
      keepDialogOpen: false,
    };
  }

  const errDetail = res.kernel_interrupt_error || "restart-kernel 未确认成功";
  return {
    result: res,
    ok: false,
    toastKind: "warn",
    message: `本机接力监工已停止，但未能确认中断 [${vmName}] 云端 Kernel（${errDetail}），原训练可能仍在实例中继续运行`,
    keepDialogOpen: true,
  };
}

export function errText(e: unknown): string {
  if (e === null || e === undefined) return "未知错误";
  if (typeof e === "string") return e || "未知错误";
  if (typeof e === "object" && "message" in e && (e as { message?: unknown }).message !== undefined) {
    return String((e as { message?: unknown }).message);
  }
  try {
    const s = JSON.stringify(e);
    return s ?? String(e);
  } catch {
    return String(e);
  }
}
