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
  startRelay: (name: string, config: RelayConfig) =>
    invoke<RelayStatus>("start_relay", { name, config }),
  stopRelay: (name: string, destroyVm: boolean) =>
    invoke<TerminateReceipt | null>("stop_relay", { name, destroyVm }),
  readWorkspaceFiles: () => invoke<WorkspaceFiles>("read_workspace_files"),
  saveWorkspaceFiles: (trainPy: string, requirementsTxt: string) =>
    invoke<void>("save_workspace_files", { trainPy, requirementsTxt }),
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
};

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
