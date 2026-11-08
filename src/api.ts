import { invoke } from "@tauri-apps/api/core";
import type { Overview, RelayConfig, RelayStatus, TerminateReceipt, WorkspaceFiles } from "./types";

export const api = {
  getOverview: () => invoke<Overview>("get_overview"),
  refreshNow: () => invoke<Overview>("refresh_now"),
  createVm: (req: {
    name: string | null;
    accelerator: string;
    high_mem: boolean;
    auto_relay: boolean;
    relay_config: RelayConfig | null;
  }) => invoke<string>("create_vm", { req }),
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
};

export function errText(e: unknown): string {
  if (typeof e === "string") return e;
  if (e && typeof e === "object" && "message" in e) return String((e as Error).message);
  return JSON.stringify(e);
}
