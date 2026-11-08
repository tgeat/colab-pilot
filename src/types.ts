export interface Ccu {
  balance: number;
  burn_rate_hourly: number;
  assignments_count: number;
}

export interface EnvStatus {
  colab_bin: string;
  colab_ok: boolean;
  rclone_bin: string;
  rclone_ok: boolean;
  rclone_conf: string;
  rclone_conf_ok: boolean;
  token_ok: boolean;
  workspace: string;
}

export interface GdriveFile {
  name: string;
  size: number;
  mod_time: string;
  is_dir: boolean;
}

export interface GdriveStatus {
  dir: string;
  ok: boolean;
  error: string | null;
  has_checkpoint: boolean;
  checkpoint_size: number;
  checkpoint_mtime: string | null;
  has_done_flag: boolean;
  files: GdriveFile[];
}

export interface RelayConfig {
  gpu_type: string;
  high_mem: boolean;
  gdrive_dir: string;
  max_session_hours: number;
  backup_interval_minutes: number;
  total_epochs: number;
  reset_done_flag: boolean;
}

export interface RelayStatus {
  vm_name: string;
  running: boolean;
  round: number;
  phase: string;
  phase_label: string;
  started_at: number | null;
  round_started_at: number | null;
  config: RelayConfig;
  logs: string[];
  child_pid: number | null;
}

export interface ProcInfo {
  pid: number;
  threads: number;
  cpu_percent: number;
  mem_percent: number;
  rss_mb: number;
  state: string;
  elapsed: string;
  role: string;
  command: string;
}

export interface Telemetry {
  timestamp: number;
  uptime_seconds: number;
  cpu: { model: string; cores: number; percent: number; load_avg: number[] };
  memory: { total_gb: number; used_gb: number; avail_gb: number; percent: number };
  disk: { total_gb: number; used_gb: number; free_gb: number; percent: number };
  gpu: {
    available: boolean;
    name: string;
    gpu_util_percent?: number;
    mem_util_percent?: number;
    vram_used_mb?: number;
    vram_total_mb?: number;
    vram_percent?: number;
    temperature_c?: number;
    power_draw_w?: number;
    power_limit_w?: number;
  };
  threads: {
    total_processes: number;
    total_threads: number;
    train_process: ProcInfo | null;
    rclone_process: ProcInfo | null;
    processes: ProcInfo[];
  };
  files: {
    has_rclone_conf: boolean;
    has_checkpoint: boolean;
    checkpoint_size_mb: number;
    has_done_flag: boolean;
  };
  training: {
    status?: string;
    current_epoch?: number;
    total_epochs?: number;
    loss?: number;
    elapsed_seconds?: number;
    next_backup_in_seconds?: number;
    handover_in_seconds?: number;
    last_backup_epoch?: number;
    last_backup_time?: number;
  } | null;
}

export interface VmView {
  name: string;
  endpoint: string | null;
  accelerator: string;
  variant: string;
  high_ram: boolean;
  on_server: boolean;
  managed: boolean;
  orphan: boolean;
  running_task: string | null;
  last_execution: [string, string | null, string] | null;
  telemetry: Telemetry | null;
  telemetry_age: number | null;
  telemetry_error: string | null;
  relay: RelayStatus | null;
  busy: string | null;
}

export interface Overview {
  timestamp: number;
  auth_ok: boolean;
  error: string | null;
  ccu: Ccu | null;
  server_count: number;
  gdrive: GdriveStatus | null;
  env: EnvStatus;
  vms: VmView[];
}

export interface TerminateReceipt {
  name: string;
  endpoint: string | null;
  verified_unassigned: boolean;
  remaining_endpoints: string[];
  steps: string[];
  ccu: Ccu | null;
  error: string | null;
}

export interface WorkspaceFiles {
  workspace: string;
  train_py: string;
  requirements_txt: string;
}

export const DEFAULT_RELAY: RelayConfig = {
  gpu_type: "T4",
  high_mem: true,
  gdrive_dir: "gdrive:colab_relay",
  max_session_hours: 22.5,
  backup_interval_minutes: 30,
  total_epochs: 500,
  reset_done_flag: false,
};
