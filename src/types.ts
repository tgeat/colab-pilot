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
  sessions_path: string;
  history_dir: string;
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

export interface WorkspaceInfo {
  name: string;
  path: string | null;
  remote_cwd: string;
  source: string;
  active_script: string | null;
  description: string | null;
}

export interface CliEventView {
  timestamp: string;
  event_type: string;
  label: string;
  summary: string;
  detail: string | null;
  output_preview: string | null;
  has_error: boolean;
}

export interface RecentSessionView {
  session_name: string;
  workspace_name: string;
  workspace_path: string | null;
  endpoint: string | null;
  accelerator: string;
  machine_shape: string;
  created_at: string | null;
  terminated_at: string | null;
  last_event_at: string;
  exec_count: number;
  file_op_count: number;
  last_summary: string;
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
  cwd?: string;
  command: string;
}

export interface ContentEntry {
  name: string;
  is_dir: boolean;
  size_mb: number;
  mtime: number;
}

export interface TrainingTelemetry {
  mode?: string;
  framework?: string;
  model_name?: string;
  status?: string;
  round?: number;
  current_epoch?: number;
  total_epochs?: number;
  current_step?: number;
  total_steps?: number;
  loss?: number;
  best_metric?: number;
  metrics?: Record<string, number>;
  lr?: number;
  gdrive_dir?: string;
  dataset_dir?: string;
  results_dir?: string;
  dataset_status?: string;
  sync_status?: string;
  elapsed_seconds?: number;
  eta_seconds?: number;
  next_backup_in_seconds?: number;
  handover_in_seconds?: number;
  last_backup_epoch?: number;
  last_backup_time?: number;
}

export interface LlmTelemetry {
  mode?: string;
  status?: string;
  online?: boolean;
  backend_healthy?: boolean;
  slot?: "blue" | "green" | string;
  role?: "active" | "warming" | "draining" | "standby" | string;
  engine?: string;
  model_name?: string;
  gdrive_model_path?: string;
  port?: number;
  backend_url?: string;
  instant_tps?: number;
  last_request_tps?: number;
  avg_tps_60s?: number;
  peak_tps?: number;
  ttft_ms?: number | null;
  avg_ttft_ms?: number | null;
  active_streams?: number;
  total_requests?: number;
  total_prompt_tokens?: number;
  total_completion_tokens?: number;
  error_count?: number;
  tps_history?: number[];
  last_request_at?: number | null;
  elapsed_seconds?: number;
  handover_in_seconds?: number;
}

export interface LlmBlueGreenState {
  session_prefix?: string;
  active_slot?: string;
  standby_slot?: string | null;
  phase?: string;
  engine?: string;
  model_name?: string;
  gdrive_model_path?: string;
  gpu?: string;
  high_mem?: boolean;
  gateway_url?: string;
  last_switchover_at?: number;
  warmup_tps?: number;
  updated_at?: number;
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
    active_process?: ProcInfo | null;
    train_process: ProcInfo | null;
    llm_process?: ProcInfo | null;
    rclone_process: ProcInfo | null;
    processes: ProcInfo[];
  };
  files: {
    has_rclone_conf: boolean;
    has_checkpoint: boolean;
    checkpoint_size_mb: number;
    has_done_flag: boolean;
    has_results_dir?: boolean;
    entries?: ContentEntry[];
  };
  training: TrainingTelemetry | null;
  llm?: LlmTelemetry | null;
}

export interface VmView {
  name: string;
  workspace: WorkspaceInfo;
  endpoint: string | null;
  accelerator: string;
  variant: string;
  high_ram: boolean;
  on_server: boolean;
  managed: boolean;
  orphan: boolean;
  kernel_id: string | null;
  session_id: string | null;
  running_task: string | null;
  last_execution: [string, string | null, string] | null;
  cli_events: CliEventView[];
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
  recent_sessions: RecentSessionView[];
  llm_bluegreen?: LlmBlueGreenState | null;
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
