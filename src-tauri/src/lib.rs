//! Colab Pilot — Tauri 2 Rust 后端（工作区与 Colab 虚拟机实时监控仪表盘）
//!
//! - 后台轮询：Colab 服务端计费实例、账户 CU、云端遥测快照、colab-cli 会话与历史、Google Drive 状态
//! - 自动探针注入：自动感知 Antigravity 通过 colab-cli 创建的实例，通过 Contents API + /colab/tty 静默注入遥测守护进程（不阻塞 Kernel，不污染 CLI 会话状态）
//! - 工作区映射：自动将每台 Colab 虚拟机关联到对应的本地工作区（Workspace），汇总资源分配与运行状态

mod cli;
mod colab;
mod relay;

use anyhow::Context;
use cli::{s, CliEventView, LlmApiAccessInfo, RecentSessionView, VmSshInfo, WorkspaceInfo};
use colab::{Assignment, Ccu, ColabClient};
use futures::future::join_all;
use relay::{GrabConfig, GrabLoopOptions, GrabStatus, RelayConfig, RelayHandle, RelayStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};

/// 容忍 Mutex 中毒的加锁：任一线程持锁 panic 后，其余线程仍可继续读写（绝不因 unwrap 连锁崩溃）
pub(crate) trait LockExt<T> {
    fn l(&self) -> MutexGuard<'_, T>;
}

impl<T> LockExt<T> for Mutex<T> {
    fn l(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(|e| e.into_inner())
    }
}

const TELEMETRY_DAEMON_PY: &str = include_str!("../remote/telemetry_daemon.py");
const DEFAULT_TRAIN_PY: &str = include_str!("../../train.py");
const LLM_LOCAL_GATEWAY_PY: &str =
    include_str!("../../.agents/skills/colab-llm-bluegreen/scripts/llm_local_gateway.py");
const LLM_BLUEGREEN_MANAGER_PY: &str =
    include_str!("../../.agents/skills/colab-llm-bluegreen/scripts/llm_bluegreen_manager.py");
const LLM_METRICS_SIDECAR_PY: &str =
    include_str!("../../.agents/skills/colab-llm-bluegreen/scripts/llm_metrics_sidecar.py");
const DL_RELAY_RUNNER_PY: &str =
    include_str!("../../.agents/skills/colab-dl-relay/scripts/dl_relay_runner.py");
const COLAB_RELAY_ADAPTER_PY: &str =
    include_str!("../../.agents/skills/colab-dl-relay/scripts/colab_relay_adapter.py");
const DEFAULT_GDRIVE_DIR: &str = "gdrive:colab_relay";

// ───────────────────────────── 数据结构 ─────────────────────────────

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct EnvStatus {
    pub colab_bin: String,
    pub colab_ok: bool,
    pub rclone_bin: String,
    pub rclone_ok: bool,
    pub rclone_conf: String,
    pub rclone_conf_ok: bool,
    pub token_ok: bool,
    #[serde(default)]
    pub adc_ok: bool,
    pub workspace: String,
    pub sessions_path: String,
    pub history_dir: String,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct GdriveFile {
    pub name: String,
    pub size: i64,
    pub mod_time: String,
    pub is_dir: bool,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct GdriveStatus {
    pub dir: String,
    pub ok: bool,
    pub error: Option<String>,
    pub has_checkpoint: bool,
    pub checkpoint_size: i64,
    pub checkpoint_mtime: Option<String>,
    pub has_done_flag: bool,
    pub files: Vec<GdriveFile>,
}

pub const DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS: u64 = 90;
pub const MIN_KEEP_ALIVE_INTERVAL_SECONDS: u64 = 30;
pub const MAX_KEEP_ALIVE_INTERVAL_SECONDS: u64 = 600;

fn default_keep_alive_true() -> bool {
    true
}

fn default_keep_alive_interval() -> u64 {
    DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct KeepAliveConfig {
    /// 是否启用实例保活（创建任何实例默认开启）
    #[serde(default = "default_keep_alive_true")]
    pub enabled: bool,
    /// 保活心跳间隔（秒），范围 30..=600，默认 90 秒
    #[serde(default = "default_keep_alive_interval")]
    pub interval_seconds: u64,
    /// 空闲时是否发送轻量 Jupyter Kernel `pass` 脉冲心跳（忙时自动避让，不阻塞正在执行的任务）
    #[serde(default = "default_keep_alive_true")]
    pub kernel_ping: bool,
}

impl Default for KeepAliveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_seconds: DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS,
            kernel_ping: true,
        }
    }
}

impl KeepAliveConfig {
    pub fn validate(&self) -> Result<Self, String> {
        if !self.enabled {
            let mut normalized = self.clone();
            if normalized.interval_seconds < MIN_KEEP_ALIVE_INTERVAL_SECONDS
                || normalized.interval_seconds > MAX_KEEP_ALIVE_INTERVAL_SECONDS
            {
                normalized.interval_seconds = DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS;
            }
            return Ok(normalized);
        }
        if self.interval_seconds < MIN_KEEP_ALIVE_INTERVAL_SECONDS
            || self.interval_seconds > MAX_KEEP_ALIVE_INTERVAL_SECONDS
        {
            return Err(format!(
                "实例保活心跳间隔必须在 {} ~ {} 秒之间（当前 {} 秒）",
                MIN_KEEP_ALIVE_INTERVAL_SECONDS,
                MAX_KEEP_ALIVE_INTERVAL_SECONDS,
                self.interval_seconds
            ));
        }
        Ok(self.clone())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct KeepAliveStatus {
    pub enabled: bool,
    pub interval_seconds: u64,
    pub kernel_ping: bool,
    /// "active" | "paused" | "error" | "idle"
    pub state: String,
    pub last_ok_at: Option<i64>,
    pub last_attempt_at: Option<i64>,
    /// Result of the authenticated runtime connection check.
    pub tunnel_ok: bool,
    /// "ok" | "skipped_busy" | "skipped_no_kernel" | "disabled" | "error" | "pending"
    pub kernel_ping_state: String,
    pub success_count: u64,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
    #[serde(default = "default_keep_alive_true")]
    pub profile_saved: bool,
    #[serde(default)]
    pub persistence_error: Option<String>,
}

impl Default for KeepAliveStatus {
    fn default() -> Self {
        Self {
            enabled: false,
            interval_seconds: 0,
            kernel_ping: false,
            state: String::new(),
            last_ok_at: None,
            last_attempt_at: None,
            tunnel_ok: false,
            kernel_ping_state: String::new(),
            success_count: 0,
            consecutive_failures: 0,
            last_error: None,
            profile_saved: true,
            persistence_error: None,
        }
    }
}

#[derive(Clone, Debug)]
struct VmKeepAliveRuntime {
    config: KeepAliveConfig,
    endpoint: Option<String>,
    epoch: u64,
    policy_version: u64,
    last_verified_endpoint: Option<String>,
    last_attempt_instant: Option<Instant>,
    last_ok_at: Option<i64>,
    last_attempt_at: Option<i64>,
    tunnel_ok: bool,
    kernel_ping_state: String,
    success_count: u64,
    consecutive_failures: u32,
    last_error: Option<String>,
    in_flight: bool,
    in_flight_tick_id: Option<u64>,
    active_cancel: Option<tokio_util::sync::CancellationToken>,
    profile_saved: bool,
    persistence_error: Option<String>,
}

impl VmKeepAliveRuntime {
    fn from_config(config: KeepAliveConfig, endpoint: Option<String>) -> Self {
        let k_state = if !config.enabled || !config.kernel_ping {
            "disabled".to_string()
        } else {
            "pending".to_string()
        };
        Self {
            config,
            endpoint,
            epoch: 0,
            policy_version: 1,
            last_verified_endpoint: None,
            last_attempt_instant: None,
            last_ok_at: None,
            last_attempt_at: None,
            tunnel_ok: false,
            kernel_ping_state: k_state,
            success_count: 0,
            consecutive_failures: 0,
            last_error: None,
            in_flight: false,
            in_flight_tick_id: None,
            active_cancel: None,
            profile_saved: true,
            persistence_error: None,
        }
    }

    fn reset_verification(&mut self, endpoint: Option<String>) {
        if let Some(tok) = self.active_cancel.take() {
            tok.cancel();
        }
        self.endpoint = endpoint;
        self.last_verified_endpoint = None;
        self.last_attempt_instant = None;
        self.last_ok_at = None;
        self.last_attempt_at = None;
        self.tunnel_ok = false;
        self.kernel_ping_state = if !self.config.enabled || !self.config.kernel_ping {
            "disabled".to_string()
        } else {
            "pending".to_string()
        };
        self.success_count = 0;
        self.consecutive_failures = 0;
        self.last_error = None;
        self.in_flight = false;
        self.in_flight_tick_id = None;
    }

    fn to_status(&self) -> KeepAliveStatus {
        let state = if self.consecutive_failures > 0 {
            "error"
        } else if !self.config.enabled {
            "paused"
        } else if self.last_ok_at.is_some() {
            "active"
        } else {
            "idle"
        };
        let last_error = self.last_error.clone();
        KeepAliveStatus {
            enabled: self.config.enabled,
            interval_seconds: self.config.interval_seconds,
            kernel_ping: self.config.kernel_ping,
            state: state.to_string(),
            last_ok_at: self.last_ok_at,
            last_attempt_at: self.last_attempt_at,
            tunnel_ok: self.tunnel_ok,
            kernel_ping_state: if !self.config.enabled || !self.config.kernel_ping {
                "disabled".to_string()
            } else {
                self.kernel_ping_state.clone()
            },
            success_count: self.success_count,
            consecutive_failures: self.consecutive_failures,
            last_error,
            profile_saved: self.profile_saved,
            persistence_error: self.persistence_error.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct VmView {
    pub name: String,
    pub workspace: WorkspaceInfo,
    pub endpoint: Option<String>,
    pub accelerator: String,
    pub variant: String,
    pub high_ram: bool,
    pub on_server: bool,
    pub managed: bool,
    pub orphan: bool,
    pub kernel_id: Option<String>,
    pub session_id: Option<String>,
    pub running_task: Option<String>,
    pub last_execution: Value,
    pub cli_events: Vec<CliEventView>,
    pub telemetry: Option<Value>,
    pub telemetry_age: Option<f64>,
    pub telemetry_error: Option<String>,
    pub relay: Option<RelayStatus>,
    pub busy: Option<String>,
    pub ssh: Option<VmSshInfo>,
    pub llm_api: Option<LlmApiAccessInfo>,
    #[serde(default)]
    pub keep_alive: Option<KeepAliveStatus>,
    #[serde(default)]
    pub grab_status: Option<GrabStatus>,
    #[serde(default)]
    pub deploy_error: Option<String>,
    #[serde(default)]
    pub server_state_unknown: bool,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct LlmBlueGreenState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_slot: Option<String>,
    #[serde(default)]
    pub standby_slot: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standby_session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blue_session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub green_session: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slots: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gdrive_model_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub high_mem: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gateway_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_switchover_at: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warmup_tps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_at: Option<f64>,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct Overview {
    pub timestamp: i64,
    pub auth_ok: bool,
    pub error: Option<String>,
    pub ccu: Option<Ccu>,
    pub server_count: usize,
    pub gdrive: Option<GdriveStatus>,
    pub env: EnvStatus,
    pub vms: Vec<VmView>,
    pub recent_sessions: Vec<RecentSessionView>,
    pub llm_bluegreen: Option<LlmBlueGreenState>,
    #[serde(default)]
    pub last_ok_timestamp: Option<u64>,
    #[serde(default)]
    pub is_stale: bool,
    #[serde(default)]
    pub stale_reason: Option<String>,
    #[serde(default)]
    pub server_list_known: bool,
    #[serde(default)]
    pub seq: u64,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct TerminateReceipt {
    pub name: String,
    pub endpoint: Option<String>,
    pub verified_unassigned: bool,
    pub remaining_endpoints: Vec<String>,
    pub steps: Vec<String>,
    pub ccu: Option<Ccu>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StopRelayResult {
    pub supervisor_stopped: bool,
    pub destroy_vm: bool,
    pub kernel_interrupted: bool,
    pub kernel_interrupt_error: Option<String>,
    pub terminate_receipt: Option<TerminateReceipt>,
}

#[derive(Deserialize, Serialize, Clone, Default, Debug)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize, Clone, Debug)]
pub struct LlmChatRequest {
    pub request_id: String,
    pub target_vm: String,
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Serialize, Clone, Debug)]
pub struct LlmChatStreamEvent {
    pub request_id: String,
    /// "start" | "delta" | "done" | "error"
    pub kind: String,
    pub vm_name: String,
    pub slot: Option<String>,
    pub model: String,
    pub delta: Option<String>,
    pub ttft_ms: Option<f64>,
    pub instant_tps: Option<f64>,
    pub avg_tps: Option<f64>,
    pub completion_tokens: Option<u32>,
    pub elapsed_ms: Option<f64>,
    pub transport: Option<String>,
    pub error: Option<String>,
}

#[derive(Serialize, Clone, Debug)]
pub struct LlmChatResponse {
    pub request_id: String,
    pub vm_name: String,
    pub slot: Option<String>,
    pub model: String,
    pub reply: String,
    pub ttft_ms: f64,
    pub avg_tps: f64,
    pub completion_tokens: u32,
    pub elapsed_ms: f64,
    pub transport: String,
}

struct TelemetryEntry {
    data: Option<Value>,
    error: Option<String>,
}

#[derive(Default)]
struct GdriveCache {
    status: Option<GdriveStatus>,
    updated_at: Option<Instant>,
    in_flight: bool,
}

#[derive(Clone, Debug)]
struct CreatingVmMeta {
    accelerator: String,
    high_mem: bool,
    workspace_name: String,
    workspace_path: Option<String>,
    description: Option<String>,
    had_ws_entry: bool,
    cancel: tokio_util::sync::CancellationToken,
    child_pid: Arc<Mutex<Option<u32>>>,
    pre_provision_endpoints: Arc<Mutex<std::collections::HashSet<String>>>,
    pre_provision_snapshot_invalid: Arc<std::sync::atomic::AtomicBool>,
    grab_status: Arc<Mutex<Option<GrabStatus>>>,
}

struct SshBridgeTarget {
    endpoint: String,
    proxy_url: String,
    proxy_token: String,
}

struct LocalSshBridge {
    local_port: u16,
    target: Mutex<SshBridgeTarget>,
    cancel: tokio_util::sync::CancellationToken,
}

impl LocalSshBridge {
    fn shutdown(&self) {
        self.cancel.cancel();
    }
}

#[derive(Clone, Debug)]
struct ActiveChatEntry {
    cancel: tokio_util::sync::CancellationToken,
    vm_name: Option<String>,
}

pub struct AppState {
    pub client: ColabClient,
    pub workspace: PathBuf,
    overview: tokio::sync::RwLock<Overview>,
    telemetry: Mutex<HashMap<String, TelemetryEntry>>,
    last_assignments: Mutex<Option<(Vec<Assignment>, Instant, u64)>>,
    last_ccu: Mutex<Option<Ccu>>,
    relays: Mutex<HashMap<String, Arc<RelayHandle>>>,
    creating_vms: Mutex<HashMap<String, CreatingVmMeta>>,
    ssh_bridges: Mutex<HashMap<String, Arc<LocalSshBridge>>>,
    deploy_errors: Mutex<HashMap<String, String>>,
    active_chats: Mutex<HashMap<String, ActiveChatEntry>>,
    busy: Mutex<HashMap<String, String>>,
    probe_cooldown: Mutex<HashMap<String, Instant>>,
    keep_alive: Mutex<HashMap<String, VmKeepAliveRuntime>>,
    keep_alive_epochs: Mutex<HashMap<String, u64>>,
    keep_alive_seq: std::sync::atomic::AtomicU64,
    gdrive: tokio::sync::Mutex<GdriveCache>,
    refresh_lock: tokio::sync::Mutex<()>,
    overview_seq: std::sync::atomic::AtomicU64,
    force_close_confirmed: std::sync::atomic::AtomicBool,
    script_dirty: std::sync::atomic::AtomicBool,
    pending_close_vms: Mutex<std::collections::HashSet<String>>,
}

pub(crate) fn legacy_orphan_name(endpoint: &str) -> String {
    let tail: String = endpoint
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let tail = if tail.is_empty() {
        "vm".to_string()
    } else {
        tail.to_lowercase()
    };
    format!("orphan-{tail}")
}

pub(crate) fn orphan_name(endpoint: &str) -> String {
    let mut hash: u32 = 0x811c_9dc5;
    for &b in endpoint.as_bytes() {
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    let base = legacy_orphan_name(endpoint);
    format!("{base}-{:04x}", hash & 0xffff)
}

pub(crate) fn matches_orphan_name(endpoint: &str, candidate: &str) -> bool {
    orphan_name(endpoint) == candidate || legacy_orphan_name(endpoint) == candidate
}

fn session_endpoint(v: &Value) -> Option<String> {
    v.get("endpoint").and_then(|e| e.as_str()).map(|s| s.to_string())
}

impl AppState {
    pub fn new_with(client: ColabClient, workspace: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            client,
            workspace,
            overview: tokio::sync::RwLock::new(Overview::default()),
            telemetry: Mutex::new(HashMap::new()),
            last_assignments: Mutex::new(None),
            last_ccu: Mutex::new(None),
            relays: Mutex::new(HashMap::new()),
            creating_vms: Mutex::new(HashMap::new()),
            ssh_bridges: Mutex::new(HashMap::new()),
            deploy_errors: Mutex::new(HashMap::new()),
            active_chats: Mutex::new(HashMap::new()),
            busy: Mutex::new(HashMap::new()),
            probe_cooldown: Mutex::new(HashMap::new()),
            keep_alive: Mutex::new(HashMap::new()),
            keep_alive_epochs: Mutex::new(HashMap::new()),
            keep_alive_seq: std::sync::atomic::AtomicU64::new(1),
            gdrive: tokio::sync::Mutex::new(GdriveCache::default()),
            refresh_lock: tokio::sync::Mutex::new(()),
            overview_seq: std::sync::atomic::AtomicU64::new(0),
            force_close_confirmed: std::sync::atomic::AtomicBool::new(false),
            script_dirty: std::sync::atomic::AtomicBool::new(false),
            pending_close_vms: Mutex::new(std::collections::HashSet::new()),
        })
    }

    pub(crate) fn current_keep_alive_epoch(&self, vm_name: &str) -> u64 {
        *self.keep_alive_epochs.l().get(vm_name).unwrap_or(&0)
    }

    pub(crate) fn bump_keep_alive_epoch(&self, vm_name: &str) -> u64 {
        let next = self
            .keep_alive_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .saturating_add(1);
        self.keep_alive_epochs.l().insert(vm_name.to_string(), next);
        next
    }

    pub(crate) fn invalidate_keep_alive(&self, vm_name: &str, remove_runtime: bool) -> u64 {
        let next = self.bump_keep_alive_epoch(vm_name);
        let mut map = self.keep_alive.l();
        if remove_runtime {
            if let Some(old) = map.remove(vm_name) {
                if let Some(tok) = old.active_cancel {
                    tok.cancel();
                }
            }
        } else if let Some(rt) = map.get_mut(vm_name) {
            rt.epoch = next;
            if let Some(tok) = rt.active_cancel.take() {
                tok.cancel();
            }
            rt.in_flight = false;
            rt.in_flight_tick_id = None;
        }
        next
    }

    fn next_keep_alive_tick_id(&self) -> u64 {
        self.keep_alive_seq
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .saturating_add(1)
    }

    fn relay(&self, name: &str) -> Option<Arc<RelayHandle>> {
        self.relays.l().get(name).cloned()
    }

    pub(crate) fn is_deleting(&self, name: &str) -> bool {
        self.busy
            .l()
            .get(name)
            .map(|v| v == "deleting")
            .unwrap_or(false)
    }

    fn active_relay_gdrive_dir(&self) -> String {
        self.relays
            .l()
            .values()
            .find(|h| h.is_running())
            .map(|h| h.status.l().config.gdrive_dir.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_GDRIVE_DIR.to_string())
    }

    fn set_busy(&self, name: &str, what: Option<&str>) {
        let mut b = self.busy.l();
        match what {
            Some(w) => {
                b.insert(name.into(), w.into());
            }
            None => {
                b.remove(name);
            }
        }
    }

    fn set_deploy_error(&self, name: &str, err: Option<String>) {
        let mut map = self.deploy_errors.l();
        match err.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
            Some(msg) => {
                map.insert(name.to_string(), msg);
            }
            None => {
                map.remove(name);
            }
        }
    }

    /// 仅当该实例当前空闲时才占用忙碌标记（绝不覆盖 deleting / creating 等更高优先级状态）
    fn try_claim_busy(&self, name: &str, what: &str) -> bool {
        let mut b = self.busy.l();
        if b.contains_key(name) {
            return false;
        }
        b.insert(name.into(), what.into());
        true
    }

    /// 仅当忙碌标记仍为自己占用的值时才释放（避免误清其他操作设置的状态）
    fn release_busy(&self, name: &str, what: &str) {
        let mut b = self.busy.l();
        if b.get(name).map(|v| v == what).unwrap_or(false) {
            b.remove(name);
        }
    }

    /// 根据会话名（或新旧格式 orphan-*）在服务端计费列表中查找实例
    pub async fn find_assignment_by_name(&self, name: &str) -> Option<Assignment> {
        let sessions = cli::read_sessions();
        let ep = sessions.get(name).and_then(session_endpoint);
        let list = self.client.list_assignments().await.ok()?;
        if let Some(ref target_ep) = ep {
            if let Some(found) = list.iter().find(|a| &a.endpoint == target_ep) {
                return Some(found.clone());
            }
        }
        list.into_iter()
            .find(|a| matches_orphan_name(&a.endpoint, name))
    }

    fn env_status(&self) -> EnvStatus {
        let colab = cli::colab_bin();
        let rclone = cli::rclone_bin();
        let conf = cli::rclone_conf_path();
        EnvStatus {
            colab_ok: colab.exists(),
            colab_bin: colab.display().to_string(),
            rclone_ok: rclone.exists(),
            rclone_bin: rclone.display().to_string(),
            rclone_conf_ok: std::fs::metadata(&conf).map(|m| m.len() > 30).unwrap_or(false),
            rclone_conf: conf.display().to_string(),
            token_ok: self.client.is_client_authorized(),
            adc_ok: cli::is_adc_configured(),
            workspace: self.workspace.display().to_string(),
            sessions_path: cli::sessions_path().display().to_string(),
            history_dir: cli::history_dir().display().to_string(),
        }
    }
}

// ───────────────────────────── Google Drive 状态 ─────────────────────────────

async fn fetch_gdrive_status(dir: &str) -> GdriveStatus {
    let mut st = GdriveStatus {
        dir: dir.into(),
        ..Default::default()
    };
    let conf = cli::rclone_conf_path();
    let mut args = Vec::new();
    if conf.exists() {
        args.push("--config".into());
        args.push(conf.to_string_lossy().to_string());
    }
    args.extend(s(&["lsjson", "--retries=5", "--low-level-retries=20", "--"]));
    args.push(dir.to_string());
    let out = cli::run_simple(&cli::rclone_bin(), &args, Duration::from_secs(90)).await;
    match out {
        Ok(o) if o.code == 0 => {
            let trimmed = o.stdout.trim();
            let json_slice = trimmed
                .find('[')
                .map(|idx| &trimmed[idx..])
                .unwrap_or("[]");
            match serde_json::from_str::<Vec<Value>>(json_slice) {
                Ok(arr) => {
                    st.ok = true;
                    for it in arr {
                        let f = GdriveFile {
                            name: it["Name"].as_str().unwrap_or("").into(),
                            size: it["Size"].as_i64().unwrap_or(0),
                            mod_time: it["ModTime"].as_str().unwrap_or("").into(),
                            is_dir: it["IsDir"].as_bool().unwrap_or(false),
                        };
                        if f.name == "checkpoint.pt" {
                            st.has_checkpoint = true;
                            st.checkpoint_size = f.size;
                            st.checkpoint_mtime = Some(f.mod_time.clone());
                        }
                        if relay::is_done_flag_filename(&f.name) {
                            st.has_done_flag = true;
                        }
                        st.files.push(f);
                    }
                }
                Err(e) => {
                    st.error = Some(format!("解析 rclone lsjson 输出失败: {e}"));
                }
            }
        }
        Ok(o) if o.output.contains("directory not found") => {
            st.ok = true;
        }
        Ok(o) if o.timed_out => {
            st.error = Some("rclone 查询超时（共享 client_id 触发 Google Drive 限流退避中）".into());
        }
        Ok(o) => {
            st.error = Some(
                o.output
                    .lines()
                    .filter(|l| !l.contains("NOTICE"))
                    .last()
                    .unwrap_or("rclone 执行失败")
                    .to_string(),
            )
        }
        Err(e) => st.error = Some(e.to_string()),
    }
    st
}

// ───────────────────────────── 静默遥测探针部署（自动感知新实例） ─────────────────────────────

fn build_probe_bootstrap_code(deploy_nonce: &str) -> String {
    let nonce_lit = serde_json::to_string(deploy_nonce).unwrap_or_else(|_| "\"probe\"".into());
    format!(
        r#"
import subprocess, sys
res = subprocess.run(
    [sys.executable, "/content/.colab_pilot_daemon.py", "--bootstrap", "--nonce", {nonce_lit}],
    capture_output=True,
    text=True,
    timeout=20,
)
if res.stdout:
    print(res.stdout.strip(), flush=True)
if res.returncode != 0:
    raise RuntimeError(f"telemetry probe bootstrap failed (rc={{res.returncode}}): {{(res.stderr or res.stdout or '').strip()}}")
"#
    )
}

/// 使用 colab-cli 虚拟环境直接通过 /colab/tty（或 ColabRuntime）静默启动遥测守护进程，
/// 注入本次部署专属 `deploy_nonce` 并严格核验 `/proc/<pid>` 存活守护进程与新鲜遥测样本，绝不因残留旧文件误报成功。
pub(crate) const SILENT_PROBE_LAUNCHER_PY: &str = r#"
import base64, json, shlex, sys, time, urllib.request
from urllib.parse import quote, urlparse

url = sys.argv[1]
token = sys.argv[2]
existing_kid = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3].strip() else None
deploy_nonce = sys.argv[4].strip() if len(sys.argv) > 4 and sys.argv[4].strip() else f"nonce-{int(time.time() * 1000)}"
safe_tok = quote(token, safe="")

def fetch_contents_json(rel_path):
    chk_url = f"{url.rstrip('/')}/api/contents/{rel_path}?authuser=0&colab-runtime-proxy-token={safe_tok}&content=1"
    req = urllib.request.Request(
        chk_url,
        headers={
            "X-Colab-Runtime-Proxy-Token": token,
            "X-Colab-Client-Agent": "colab-cli",
        },
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        if not (200 <= r.status < 300):
            return None
        outer = json.loads(r.read().decode("utf-8", "replace"))
    content = outer.get("content")
    fmt = outer.get("format") or "text"
    if isinstance(content, str):
        raw_txt = base64.b64decode(content).decode("utf-8", "replace") if fmt == "base64" else content
        return json.loads(raw_txt)
    if isinstance(content, dict):
        return content
    return None

def check_probe_fresh_for_nonce():
    try:
        ack = fetch_contents_json("content/.colab_pilot_probe_ack.json")
        tel = fetch_contents_json("content/.colab_pilot_telemetry.json")
        if not isinstance(ack, dict) or not isinstance(tel, dict):
            return False
        if not ack.get("ok") or ack.get("nonce") != deploy_nonce:
            return False
        if tel.get("probe_nonce") != deploy_nonce:
            return False
        ack_pid = int(ack.get("pid") or 0)
        tel_pid = int(tel.get("daemon_pid") or 0)
        launch_ts = float(ack.get("launch_ts") or 0.0)
        tel_ts = float(tel.get("timestamp") or 0.0)
        return ack_pid > 0 and ack_pid == tel_pid and tel_ts >= launch_ts
    except Exception:
        return False

started = False
try:
    import websocket
    parsed = urlparse(url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/colab/tty?colab-runtime-proxy-token={safe_tok}"
    ws = websocket.create_connection(
        ws_url,
        header=[
            f"X-Colab-Runtime-Proxy-Token: {token}",
            "X-Colab-Client-Agent: colab-cli",
        ],
        timeout=10,
    )
    ws.send(json.dumps({"cols": 120, "rows": 30}))
    time.sleep(0.25)
    q_nonce = shlex.quote(deploy_nonce)
    ws.send(json.dumps({"data": f"python3 /content/.colab_pilot_daemon.py --bootstrap --nonce {q_nonce} >/dev/null 2>&1 </dev/null &\n"}))
    for _ in range(12):
        time.sleep(0.5)
        if check_probe_fresh_for_nonce():
            started = True
            print(f"PROBE_OK via_tty nonce={deploy_nonce}", flush=True)
            break
    ws.close()
except Exception:
    pass

if not started:
    from colab_cli.runtime import ColabRuntime
    rt = ColabRuntime(url, token, kernel_id=existing_kid, session_id=None)
    try:
        if not existing_kid and getattr(rt, "kernel_id", None):
            print(f"KERNEL_ID:{rt.kernel_id}", flush=True)
        out_chunks = []
        def _hook(msg):
            if isinstance(msg, dict):
                if msg.get("output_type") == "stream" and msg.get("text"):
                    out_chunks.append(str(msg["text"]))
                elif msg.get("output_type") == "error":
                    out_chunks.append("ERROR:" + str(msg.get("evalue") or msg.get("ename") or "kernel_error"))
        boot_code = (
            "import subprocess, sys\n"
            f"res = subprocess.run([sys.executable, '/content/.colab_pilot_daemon.py', '--bootstrap', '--nonce', {json.dumps(deploy_nonce)}], capture_output=True, text=True, timeout=20)\n"
            "print(res.stdout.strip(), flush=True)\n"
            "if res.returncode != 0:\n"
            "    raise RuntimeError(res.stderr.strip() or res.stdout.strip() or 'bootstrap failed')\n"
        )
        rt.execute_code(boot_code, output_hook=_hook, timeout=35)
        joined_out = "\n".join(out_chunks)
        if (f"PROBE_OK started" in joined_out and f"nonce={deploy_nonce}" in joined_out) or check_probe_fresh_for_nonce():
            print(f"PROBE_OK via_kernel nonce={deploy_nonce}", flush=True)
        else:
            print(f"PROBE_FAIL via_kernel: {joined_out.strip()}", flush=True)
            sys.exit(1)
    finally:
        rt.stop()
"#;

pub async fn deploy_probe_for_assignment(app: &Arc<AppState>, a: &Assignment) -> anyhow::Result<()> {
    if let Err(e) = app
        .client
        .contents_upload(a, "content/.colab_pilot_daemon.py", TELEMETRY_DAEMON_PY.as_bytes())
        .await
    {
        if colab::is_proxy_auth_error(&e) {
            *app.last_assignments.l() = None;
        }
        return Err(e);
    }

    let existing_kid = cli::read_sessions()
        .values()
        .find(|v| session_endpoint(v).as_deref() == Some(a.endpoint.as_str()))
        .and_then(|v| v.get("kernel_id").and_then(|x| x.as_str()))
        .unwrap_or("")
        .to_string();

    let deploy_nonce = format!("probe-{}", uuid::Uuid::new_v4().simple());
    let py = cli::colab_python_bin();
    let args = vec![
        "-u".to_string(),
        "-c".to_string(),
        SILENT_PROBE_LAUNCHER_PY.to_string(),
        a.runtime_proxy_info.url.clone(),
        a.runtime_proxy_info.token.clone(),
        existing_kid,
        deploy_nonce.clone(),
    ];
    let out = cli::run_simple(&py, &args, Duration::from_secs(65)).await?;
    for line in out.stdout.lines() {
        if let Some(kid) = line.trim().strip_prefix("KERNEL_ID:") {
            cli::update_session_kernel_id(&a.endpoint, kid);
        }
    }
    if out.code == 0
        && out.output.contains("PROBE_OK")
        && out.output.contains(&format!("nonce={deploy_nonce}"))
    {
        Ok(())
    } else {
        let cleaned = cli::strip_colab_cli_banners(&out.output);
        if cleaned.contains("401") || cleaned.contains("403") {
            *app.last_assignments.l() = None;
        }
        anyhow::bail!(
            "探针自动注入失败：{}",
            cli::last_meaningful_cli_line(&cleaned, "无输出")
        )
    }
}

pub async fn deploy_probe_inner(app: &Arc<AppState>, name: &str) -> anyhow::Result<()> {
    let a = app
        .find_assignment_by_name(name)
        .await
        .ok_or_else(|| anyhow::anyhow!("服务端未找到实例 [{name}]"))?;
    let silent_err = match deploy_probe_for_assignment(app, &a).await {
        Ok(()) => return Ok(()),
        Err(e) => e,
    };
    if !cli::read_sessions().contains_key(name) || name.starts_with("orphan-") {
        return Err(silent_err);
    }
    let deploy_nonce = format!("probe-{}", uuid::Uuid::new_v4().simple());
    let exec_args = cli::colab_args_for_session(name, &s(&["exec", "-s", name, "--timeout", "60"]));
    let out = cli::run_streaming(
        &cli::colab_bin(),
        &exec_args,
        Some(build_probe_bootstrap_code(&deploy_nonce)),
        Duration::from_secs(150),
        None,
        None,
        |_| {},
    )
    .await?;
    if out.code == 0
        && out.output.contains("PROBE_OK")
        && out.output.contains(&format!("nonce={deploy_nonce}"))
    {
        Ok(())
    } else {
        let cleaned = cli::strip_colab_cli_banners(&out.output);
        anyhow::bail!(
            "探针启动失败：{}",
            cli::last_meaningful_cli_line(&cleaned, "无输出")
        )
    }
}

// ───────────────────── 本机按需 SSH WebSocket 桥接与交互式 /colab/tty 回退终端 ─────────────────────

/// 使用 google-colab-cli 内置的 `websocket-client`（`import websocket`）桥接 `/colab/ssh` 二进制流，
/// 彻底修复缺失 `websockets` 模块导致的 SSH 握手失败，并兼容 `google-colab-cli v0.8+` 请求头规范。
pub(crate) const WS_SSH_STDIO_BRIDGE_PY: &str = r#"
import os, sys, threading
from urllib.parse import quote, urlparse
import websocket

url = sys.argv[1]
token = sys.argv[2]
pubkey = sys.argv[3].strip()

if os.name == "nt":
    import msvcrt
    msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
    msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

parsed = urlparse(url)
ws_scheme = "wss" if parsed.scheme == "https" else "ws"
safe_tok = quote(token, safe="")
ws_url = (
    f"{ws_scheme}://{parsed.netloc}/colab/ssh"
    f"?colab-runtime-proxy-token={safe_tok}"
    f"&colab-client-agent=colab-cli&colab-client-agent=ssh-mode-proxy"
)
headers = ["X-Colab-Client-Agent: colab-cli"]
if pubkey:
    headers.append(f"X-Colab-Ssh-Pubkey: {pubkey}")

ws = websocket.create_connection(ws_url, header=headers, timeout=15)
if getattr(ws, "sock", None) is not None:
    ws.sock.settimeout(None)

def stdin_to_ws():
    try:
        while True:
            chunk = os.read(0, 16384)
            if not chunk:
                try:
                    ws.close()
                except Exception:
                    pass
                break
            ws.send_binary(chunk)
    except Exception:
        try:
            ws.close()
        except Exception:
            pass

t = threading.Thread(target=stdin_to_ws, daemon=True)
t.start()

try:
    while True:
        opcode, data = ws.recv_data(control_frame=True)
        if opcode == websocket.ABNF.OPCODE_CLOSE:
            break
        if opcode in (websocket.ABNF.OPCODE_BINARY, websocket.ABNF.OPCODE_TEXT) and data:
            payload = data.encode("utf-8") if isinstance(data, str) else data
            view = memoryview(payload)
            written = 0
            while written < len(view):
                n = os.write(1, view[written:])
                if n <= 0:
                    break
                written += n
except Exception:
    pass
finally:
    try:
        ws.close()
    except Exception:
        pass
"#;

/// 针对 `google-colab-cli v0.8.0` 移除 `colab console` 及浏览器/新版实例不支持 `/colab/ssh` 的内置双协议交互式终端客户端：
/// 1. 优先连接新版 Colab Tornado Jupyter Server 的 `/api/terminals` + `/terminals/websocket/<name>`（携带 `_xsrf` Cookie，退出自动 DELETE 清理）；
/// 2. 若 `/api/terminals` 不可用，自动无缝回退至传统 `/colab/tty` WebSocket。
pub(crate) const INTERACTIVE_TTY_CONSOLE_PY: &str = r#"
import http.cookiejar, json, os, shutil, sys, threading, urllib.request
from urllib.parse import quote, urlparse
import websocket

if os.name == "nt":
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        for std_id, flag in ((-11, 0x0004), (-12, 0x0004)):
            h = k32.GetStdHandle(std_id)
            mode = ctypes.c_uint32()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(h, mode.value | flag)
    except Exception:
        pass

url = (os.environ.get("COLAB_PILOT_PROXY_URL") or (sys.argv[1] if len(sys.argv) > 1 else "")).rstrip("/")
token = os.environ.get("COLAB_PILOT_PROXY_TOKEN") or (sys.argv[2] if len(sys.argv) > 2 else "")
vm_label = os.environ.get("COLAB_PILOT_VM_NAME") or (sys.argv[3] if len(sys.argv) > 3 else "colab-vm")

parsed = urlparse(url)
ws_scheme = "wss" if parsed.scheme == "https" else "ws"
safe_tok = quote(token, safe="")

ws = None
proto = "jupyter"
term_name = None
opener = None
post_hdrs = {}

# 1. 优先尝试 Jupyter Server /api/terminals + /terminals/websocket/<name>
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
base_hdrs = {
    "X-Colab-Runtime-Proxy-Token": token,
    "X-Colab-Client-Agent": "colab-cli",
}
post_hdrs = dict(base_hdrs)
try:
    sys.stdout.write(f"\x1b[36m[Colab Pilot]\x1b[0m 正在连接云端 Jupyter 交互终端 ({vm_label} · /api/terminals) ...\r\n")
    sys.stdout.flush()
    req_get = urllib.request.Request(
        f"{url}/api/terminals?authuser=0&colab-runtime-proxy-token={safe_tok}",
        headers=base_hdrs,
    )
    with opener.open(req_get, timeout=12) as _:
        pass
    xsrf = next((c.value for c in cj if c.name == "_xsrf"), "")
    if xsrf:
        post_hdrs["X-XSRFToken"] = xsrf
    post_hdrs["Content-Type"] = "application/json"
    req_post = urllib.request.Request(
        f"{url}/api/terminals?authuser=0&colab-runtime-proxy-token={safe_tok}",
        data=b"{}",
        headers=post_hdrs,
        method="POST",
    )
    with opener.open(req_post, timeout=12) as resp:
        term_info = json.loads(resp.read().decode("utf-8", "replace"))
        term_name = term_info.get("name")
    if term_name:
        ws_headers = [
            f"X-Colab-Runtime-Proxy-Token: {token}",
            "X-Colab-Client-Agent: colab-cli",
        ]
        cookie_header = "; ".join(f"{c.name}={c.value}" for c in cj)
        if cookie_header:
            ws_headers.append(f"Cookie: {cookie_header}")
        ws_url = f"{ws_scheme}://{parsed.netloc}/terminals/websocket/{quote(str(term_name), safe='')}?authuser=0&colab-runtime-proxy-token={safe_tok}"
        ws = websocket.create_connection(ws_url, header=ws_headers, timeout=15)
        if getattr(ws, "sock", None) is not None:
            ws.sock.settimeout(None)
except Exception as e:
    sys.stdout.write(f"\x1b[33m[Colab Pilot] /api/terminals 暂不可用 ({e})，回退至 /colab/tty ...\x1b[0m\r\n")
    sys.stdout.flush()
    ws = None
    term_name = None

# 2. 若 /api/terminals 不可用，无缝回退至 /colab/tty
if ws is None:
    proto = "colab_tty"
    ws_url = f"{ws_scheme}://{parsed.netloc}/colab/tty?colab-runtime-proxy-token={safe_tok}"
    try:
        ws = websocket.create_connection(
            ws_url,
            header=[
                f"X-Colab-Runtime-Proxy-Token: {token}",
                "X-Colab-Client-Agent: colab-cli",
            ],
            timeout=15,
        )
        if getattr(ws, "sock", None) is not None:
            ws.sock.settimeout(None)
    except Exception as e:
        sys.stdout.write(f"\x1b[31m[Colab Pilot] 无法连接远程交互终端 (/api/terminals 与 /colab/tty 均失败): {e}\x1b[0m\r\n")
        sys.stdout.flush()
        sys.exit(1)

cols, rows = shutil.get_terminal_size((120, 32))
try:
    if proto == "jupyter":
        ws.send(json.dumps(["set_size", rows, cols, 0, 0]))
        ws.send(json.dumps(["stdin", "cd /content 2>/dev/null && clear\r"]))
    else:
        ws.send(json.dumps({"cols": cols, "rows": rows}))
        ws.send(json.dumps({"data": "cd /content 2>/dev/null; clear\r"}))
except Exception:
    pass

closed = threading.Event()

def send_input(seq):
    if proto == "jupyter":
        ws.send(json.dumps(["stdin", seq]))
    else:
        ws.send(json.dumps({"data": seq}))

def recv_loop():
    try:
        while not closed.is_set():
            msg = ws.recv()
            if msg is None:
                break
            txt = msg.decode("utf-8", "replace") if isinstance(msg, bytes) else msg
            if proto == "jupyter":
                try:
                    arr = json.loads(txt)
                    if isinstance(arr, list) and len(arr) >= 2:
                        kind = arr[0]
                        if kind == "stdout":
                            sys.stdout.write(str(arr[1]))
                            sys.stdout.flush()
                        elif kind == "disconnect":
                            break
                        continue
                except Exception:
                    pass
            elif txt.startswith("{"):
                try:
                    obj = json.loads(txt)
                    if isinstance(obj, dict) and "data" in obj:
                        txt = str(obj["data"])
                except Exception:
                    pass
            if txt:
                sys.stdout.write(txt)
                sys.stdout.flush()
    except Exception:
        pass
    finally:
        closed.set()

threading.Thread(target=recv_loop, daemon=True).start()

WIN_SPECIAL_KEYS = {
    "H": "\x1b[A",
    "P": "\x1b[B",
    "M": "\x1b[C",
    "K": "\x1b[D",
    "G": "\x1b[H",
    "O": "\x1b[F",
    "S": "\x1b[3~",
    "I": "\x1b[5~",
    "Q": "\x1b[6~",
}

try:
    if os.name == "nt":
        import msvcrt, time
        while not closed.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.015)
                continue
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                k = msvcrt.getwch()
                seq = WIN_SPECIAL_KEYS.get(k, "")
            else:
                seq = ch
            if seq:
                send_input(seq)
    else:
        import select, termios, tty
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while not closed.is_set():
                r, _, _ = select.select([fd], [], [], 0.1)
                if r:
                    data = os.read(fd, 1024)
                    if not data:
                        break
                    send_input(data.decode("utf-8", "replace"))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
except KeyboardInterrupt:
    try:
        send_input("\x03")
    except Exception:
        pass
finally:
    closed.set()
    try:
        ws.close()
    except Exception:
        pass
    if proto == "jupyter" and term_name and opener is not None:
        try:
            req_del = urllib.request.Request(
                f"{url}/api/terminals/{quote(str(term_name), safe='')}?authuser=0&colab-runtime-proxy-token={safe_tok}",
                headers=post_hdrs,
                method="DELETE",
            )
            opener.open(req_del, timeout=5).close()
        except Exception:
            pass
    sys.stdout.write("\r\n\x1b[33m[Colab Pilot] 远程终端会话已断开。\x1b[0m\r\n")
    sys.stdout.flush()
"#;

async fn ensure_vm_ssh_bridge(app: &AppState, vm_name: &str, a: &Assignment) -> Option<u16> {
    if !cli::is_safe_vm_alias(vm_name) {
        return None;
    }
    if let Some(existing) = app.ssh_bridges.l().get(vm_name).cloned() {
        let mut t = existing.target.l();
        t.endpoint = a.endpoint.clone();
        t.proxy_url = a.runtime_proxy_info.url.clone();
        t.proxy_token = a.runtime_proxy_info.token.clone();
        return Some(existing.local_port);
    }

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.ok()?;
    let local_port = listener.local_addr().ok()?.port();
    let cancel = tokio_util::sync::CancellationToken::new();
    let bridge = Arc::new(LocalSshBridge {
        local_port,
        target: Mutex::new(SshBridgeTarget {
            endpoint: a.endpoint.clone(),
            proxy_url: a.runtime_proxy_info.url.clone(),
            proxy_token: a.runtime_proxy_info.token.clone(),
        }),
        cancel: cancel.clone(),
    });
    app.ssh_bridges
        .l()
        .insert(vm_name.to_string(), bridge.clone());

    let bridge_loop = bridge.clone();
    tokio::spawn(async move {
        loop {
            let accepted = tokio::select! {
                _ = cancel.cancelled() => break,
                r = listener.accept() => r,
            };
            let Ok((mut tcp_stream, _)) = accepted else {
                break;
            };
            let (proxy_url, proxy_token) = {
                let t = bridge_loop.target.l();
                (t.proxy_url.clone(), t.proxy_token.clone())
            };
            let conn_cancel = cancel.clone();
            tokio::spawn(async move {
                let pubkey = tokio::task::spawn_blocking(|| {
                    cli::ensure_ssh_keypair().ok().map(|(_, pub_s)| pub_s)
                })
                .await
                .ok()
                .flatten()
                .unwrap_or_default();

                let py = cli::colab_python_bin();
                let mut cmd = tokio::process::Command::new(py);
                cmd.args([
                    "-u",
                    "-c",
                    WS_SSH_STDIO_BRIDGE_PY,
                    &proxy_url,
                    &proxy_token,
                    &pubkey,
                ])
                .env("PYTHONUTF8", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::null())
                .kill_on_drop(true);
                #[cfg(windows)]
                cmd.creation_flags(0x0800_0000);

                let Ok(mut child) = cmd.spawn() else {
                    return;
                };
                let (Some(mut py_stdin), Some(mut py_stdout)) =
                    (child.stdin.take(), child.stdout.take())
                else {
                    cli::kill_tree(&mut child).await;
                    return;
                };
                let (mut tcp_r, mut tcp_w) = tcp_stream.split();
                tokio::select! {
                    _ = conn_cancel.cancelled() => {},
                    _ = tokio::io::copy(&mut tcp_r, &mut py_stdin) => {},
                    _ = tokio::io::copy(&mut py_stdout, &mut tcp_w) => {},
                }
                cli::kill_tree(&mut child).await;
            });
        }
    });

    Some(local_port)
}

fn build_vm_ssh_info(vm_name: &str, a: &Assignment, local_port: Option<u16>) -> VmSshInfo {
    let alias = format!("colab-{vm_name}");
    let (priv_path_str, fingerprint) = match cli::ensure_ssh_keypair() {
        Ok((priv_p, pub_raw)) => {
            let fp = pub_raw
                .split_whitespace()
                .nth(1)
                .map(|b64| {
                    let tail: String = b64.chars().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect();
                    format!("ed25519:…{tail}")
                })
                .unwrap_or_else(|| "ed25519".into());
            (priv_p.to_string_lossy().replace('\\', "/"), fp)
        }
        Err(_) => (
            cli::pilot_ssh_private_key_path()
                .to_string_lossy()
                .replace('\\', "/"),
            "ed25519".into(),
        ),
    };
    let port = local_port.unwrap_or(0);
    let null_hosts = cli::known_hosts_null_device();
    let ssh_d = cli::ssh_dir();
    let has_default_ssh_key =
        ssh_d.join("id_ed25519.pub").exists() || ssh_d.join("id_ecdsa.pub").exists();
    let auth_prefix = if matches!(cli::get_session_auth_mode(vm_name).as_deref(), Some("adc")) {
        "colab --auth adc"
    } else {
        "colab"
    };
    let colab_ssh_command = if has_default_ssh_key {
        format!("{auth_prefix} ssh -s {vm_name}")
    } else {
        format!("{auth_prefix} ssh -i \"{priv_path_str}\" -s {vm_name}")
    };
    let ssh_port_command = if port > 0 {
        format!(
            "ssh -p {port} -i \"{priv_path_str}\" -o StrictHostKeyChecking=no -o UserKnownHostsFile={null_hosts} root@127.0.0.1"
        )
    } else {
        colab_ssh_command.clone()
    };
    // 绝不在前端可见结构体中暴露 colab-runtime-proxy-token 查询参数
    let remote_ws_url = a
        .runtime_proxy_info
        .url
        .trim_end_matches('/')
        .replacen("https://", "wss://", 1)
        .replacen("http://", "ws://", 1)
        + "/colab/ssh";

    VmSshInfo {
        ready: port > 0 && cli::is_safe_vm_alias(vm_name),
        host_alias: alias.clone(),
        local_port: port,
        ssh_command: format!("ssh {alias}"),
        ssh_port_command,
        colab_ssh_command,
        private_key_path: priv_path_str,
        public_key_fingerprint: fingerprint,
        remote_ws_url,
        user: "root".into(),
    }
}

// ───────────────────────────── 总览刷新 ─────────────────────────────

/// 本机蓝绿网关计量快照专用的复用 HTTP 客户端（直连回环、超短超时，避免每次轮询重建连接池）
fn local_metrics_client() -> Option<&'static reqwest::Client> {
    static CLIENT: std::sync::OnceLock<Option<reqwest::Client>> = std::sync::OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .no_proxy()
                .connect_timeout(Duration::from_millis(180))
                .timeout(Duration::from_millis(250))
                .build()
                .ok()
        })
        .as_ref()
}

pub(crate) fn parse_llm_bluegreen_state(raw: Option<Value>) -> Option<LlmBlueGreenState> {
    let val = raw?;
    if !val.is_object() {
        return None;
    }
    serde_json::from_value::<LlmBlueGreenState>(val).ok()
}

pub(crate) fn bg_session_prefix(bg: &LlmBlueGreenState) -> &str {
    bg.session_prefix
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("llm")
}

/// 从 `LlmBlueGreenState` 中提取指定槽位（`"blue"` / `"green"`）显式记录的真实会话名（不回退 `${prefix}-${slot}`）
pub(crate) fn bg_explicit_slot_session(bg: &LlmBlueGreenState, slot: &str) -> Option<String> {
    let slot_low = slot.trim().to_ascii_lowercase();
    if let Some(ref slots_val) = bg.slots {
        if let Some(slot_entry) = slots_val.get(&slot_low).or_else(|| slots_val.get(slot)) {
            if let Some(s) = slot_entry.as_str().map(str::trim).filter(|s| !s.is_empty()) {
                return Some(s.to_string());
            }
            if let Some(s) = slot_entry
                .get("session_name")
                .or_else(|| slot_entry.get("vm_name"))
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                return Some(s.to_string());
            }
        }
    }
    let direct = if slot_low == "blue" {
        bg.blue_session.as_deref()
    } else if slot_low == "green" {
        bg.green_session.as_deref()
    } else {
        None
    };
    if let Some(s) = direct.map(str::trim).filter(|s| !s.is_empty()) {
        return Some(s.to_string());
    }
    if bg
        .active_slot
        .as_deref()
        .map(|s| s.trim().eq_ignore_ascii_case(&slot_low))
        .unwrap_or(false)
    {
        if let Some(s) = bg.active_session.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    if bg
        .standby_slot
        .as_deref()
        .map(|s| s.trim().eq_ignore_ascii_case(&slot_low))
        .unwrap_or(false)
    {
        if let Some(s) = bg.standby_session.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            return Some(s.to_string());
        }
    }
    None
}

/// 解析指定槽位的目标会话名：优先使用蓝绿状态中记录的真实会话名，无记录时才回退 `${prefix}-${slot}`
pub(crate) fn bg_slot_session_name(bg: &LlmBlueGreenState, slot: &str) -> String {
    let slot_low = slot.trim().to_ascii_lowercase();
    bg_explicit_slot_session(bg, &slot_low)
        .unwrap_or_else(|| format!("{}-{slot_low}", bg_session_prefix(bg)))
}

/// 匹配指定实例所属的蓝绿槽位：
/// 1. 优先匹配 `LlmBlueGreenState` 中记录的真实会话名（`slots[slot].session_name` / `blue_session` / `green_session` / `active_session` / `standby_session`）；
/// 2. 其次检查 `telemetry.llm.slot`（不与已登记的其他真实槽位会话冲突时）；
/// 3. 最后回退 `${prefix}-${slot}`。
pub(crate) fn match_bluegreen_slot_for_vm(
    vm_name: &str,
    bg: Option<&LlmBlueGreenState>,
    llm_tel: Option<&Value>,
) -> Option<&'static str> {
    if let Some(b) = bg {
        let exp_blue = bg_explicit_slot_session(b, "blue");
        let exp_green = bg_explicit_slot_session(b, "green");
        if exp_blue.as_deref() == Some(vm_name) {
            return Some("blue");
        }
        if exp_green.as_deref() == Some(vm_name) {
            return Some("green");
        }
        if b.active_session.as_deref().map(str::trim) == Some(vm_name) {
            return if b
                .active_slot
                .as_deref()
                .map(|s| s.trim().eq_ignore_ascii_case("green"))
                .unwrap_or(false)
            {
                Some("green")
            } else {
                Some("blue")
            };
        }
        if b.standby_session.as_deref().map(str::trim) == Some(vm_name) {
            return if b
                .standby_slot
                .as_deref()
                .map(|s| s.trim().eq_ignore_ascii_case("green"))
                .unwrap_or(false)
            {
                Some("green")
            } else {
                Some("blue")
            };
        }
        if let Some(tel_slot) = llm_tel
            .and_then(|l| l.get("slot"))
            .and_then(|x| x.as_str())
            .map(|s| s.trim().to_ascii_lowercase())
        {
            let pfx = bg_session_prefix(b);
            let same_cluster = pfx == "llm" || vm_name == pfx || vm_name.starts_with(&format!("{pfx}-"));
            if tel_slot == "blue" && exp_blue.is_none() && same_cluster {
                return Some("blue");
            }
            if tel_slot == "green" && exp_green.is_none() && same_cluster {
                return Some("green");
            }
        }
        let pfx = bg_session_prefix(b);
        if exp_blue.is_none() && vm_name == format!("{pfx}-blue") {
            return Some("blue");
        }
        if exp_green.is_none() && vm_name == format!("{pfx}-green") {
            return Some("green");
        }
        return None;
    }

    if let Some(tel_slot) = llm_tel
        .and_then(|l| l.get("slot"))
        .and_then(|x| x.as_str())
        .map(|s| s.trim().to_ascii_lowercase())
    {
        if tel_slot == "blue" {
            return Some("blue");
        }
        if tel_slot == "green" {
            return Some("green");
        }
    }
    None
}

/// 严格判定指定 VM 是否属于当前蓝绿集群固定槽位（blue -> 18081，green -> 18082）：
/// 优先匹配状态中记录的真实会话名，其次匹配遥测槽位，最后回退 `{session_prefix}-{slot}`，
/// 杜绝其他无关同名后缀实例（如 `other-blue`）与本地蓝绿网关串流或混入指标（H-4 + #48）。
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn bluegreen_tunnel_port_for_vm(
    vm_name: &str,
    bg: Option<&LlmBlueGreenState>,
) -> Option<u16> {
    let b = bg?;
    match match_bluegreen_slot_for_vm(vm_name, Some(b), None) {
        Some("blue") => Some(18081),
        Some("green") => Some(18082),
        _ => None,
    }
}

/// 过滤陈旧的 `llm_bluegreen_state.json`：
/// - 若处于 `switching` / `initializing` / `warming_up` / `deploying` / `warming_*` 且更新时间在 30 分钟内，保留展示；
/// - 若处于 `failed` 且关联的 `failed_session` 在线/创建中，或错误发生在最近 10 分钟内，保留展示以便界面显示错误诊断；
/// - 其余状态（如 `serving`）检查状态中记录的全部真实会话名（`slots[slot].session_name` / `blue_session` / `green_session` / `active_session` / `standby_session`）及回退 `${prefix}-${slot}` 是否在线/创建中/部署中。
pub(crate) fn filter_live_bluegreen_state(
    bg: Option<LlmBlueGreenState>,
    online_names: &[String],
    busy: &HashMap<String, String>,
    creating_names: &[String],
    now_ts: f64,
) -> Option<LlmBlueGreenState> {
    let bg = bg?;
    let phase = bg
        .phase
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    let updated_at = bg.updated_at.unwrap_or(0.0);
    let age_secs = if updated_at > 0.0 {
        (now_ts - updated_at).max(0.0)
    } else {
        f64::INFINITY
    };

    let is_live_vm = |cand: &str| -> bool {
        let c = cand.trim();
        !c.is_empty()
            && (online_names.iter().any(|n| n == c)
                || creating_names.iter().any(|n| n == c)
                || busy
                    .get(c)
                    .map(|b| b == "creating" || b == "deploying_llm")
                    .unwrap_or(false))
    };

    if (matches!(
        phase.as_str(),
        "switching" | "initializing" | "warming_up" | "deploying"
    ) || phase.starts_with("warming_"))
        && age_secs <= 1800.0
    {
        return Some(bg);
    }

    if phase == "failed" {
        let failed_sess = bg
            .extra
            .get("failed_session")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .trim();
        if is_live_vm(failed_sess) || age_secs <= 600.0 {
            return Some(bg);
        }
        return None;
    }

    let blue_cand = bg_slot_session_name(&bg, "blue");
    let green_cand = bg_slot_session_name(&bg, "green");
    let active_sess = bg.active_session.as_deref().unwrap_or("");
    let standby_sess = bg.standby_session.as_deref().unwrap_or("");

    if is_live_vm(&blue_cand)
        || is_live_vm(&green_cand)
        || is_live_vm(active_sess)
        || is_live_vm(standby_sess)
    {
        Some(bg)
    } else {
        None
    }
}

async fn fetch_telemetry(
    app: &AppState,
    a: &Assignment,
    vm_name: Option<&str>,
    bg: Option<&LlmBlueGreenState>,
) -> TelemetryEntry {
    match app
        .client
        .contents_read_text(a, "content/.colab_pilot_telemetry.json")
        .await
    {
        Ok(Some(text)) => match serde_json::from_str::<Value>(&text) {
            Ok(mut v) => {
                let llm_tel = v.get("llm").filter(|x| x.is_object());
                // 匹配蓝绿槽位固定端口（优先真实会话名）或独立 LLM 实例在 llm_api_keys.json 中登记的专属本地端口，合并实时指标
                let local_port = vm_name.and_then(|n| {
                    bg.and_then(|b| match match_bluegreen_slot_for_vm(n, Some(b), llm_tel) {
                        Some("blue") => Some(18081),
                        Some("green") => Some(18082),
                        _ => None,
                    })
                    .or_else(|| cli::lookup_vm_api_key_meta(n).and_then(|(_, p)| p))
                });
                if let Some(port) = local_port {
                    if let Some(client) = local_metrics_client() {
                        let mut req_builder =
                            client.get(format!("http://127.0.0.1:{port}/pilot/metrics"));
                        if let Some(n) = vm_name {
                            req_builder = req_builder.header("X-Colab-Vm", n);
                        }
                        if let Ok(resp) = req_builder.send().await {
                            if resp.status().is_success() {
                                if let Ok(live_llm) = resp.json::<Value>().await {
                                    let ep_matches = live_llm
                                        .get("endpoint")
                                        .and_then(|x| x.as_str())
                                        .filter(|s| !s.is_empty())
                                        .map(|ep| ep == a.endpoint)
                                        .unwrap_or(true);
                                    let live_ts = live_llm.get("timestamp").and_then(|x| x.as_f64());
                                    let cloud_ts = v.pointer("/llm/timestamp").and_then(|x| x.as_f64());
                                    let not_older = match (live_ts, cloud_ts) {
                                        (Some(l), Some(c)) => l >= c,
                                        _ => true,
                                    };
                                    if ep_matches
                                        && not_older
                                        && live_llm.get("total_completion_tokens").is_some()
                                        && live_llm.get("tps_history").is_some()
                                    {
                                        if let Some(live_obj) = live_llm.as_object() {
                                            if let Some(root_obj) = v.as_object_mut() {
                                                match root_obj.get_mut("llm").and_then(|x| x.as_object_mut()) {
                                                    Some(existing_llm) => {
                                                        for (k, val) in live_obj {
                                                            existing_llm.insert(k.clone(), val.clone());
                                                        }
                                                    }
                                                    None => {
                                                        root_obj.insert("llm".into(), live_llm);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                TelemetryEntry { data: Some(v), error: None }
            }
            Err(e) => TelemetryEntry { data: None, error: Some(format!("遥测解析失败: {e}")) },
        },
        Ok(None) => TelemetryEntry { data: None, error: Some("正在自动注入遥测探针…".into()) },
        Err(e) => {
            if colab::is_proxy_auth_error(&e) {
                *app.last_assignments.l() = None;
            }
            TelemetryEntry { data: None, error: Some(e.to_string()) }
        }
    }
}

/// 刷新总览：若已有刷新在进行中，直接返回最近一次缓存的总览（不排队等待，避免命令与轮询互相阻塞）。
/// `force_gdrive = true`（用户手动刷新 / 无界面诊断）时会等待正在进行的刷新结束后再完整刷新一次。
pub async fn refresh_overview(app: &Arc<AppState>, force_gdrive: bool) -> Overview {
    refresh_overview_opts(app, force_gdrive, force_gdrive).await
}

async fn refresh_overview_opts(app: &Arc<AppState>, force_gdrive: bool, wait: bool) -> Overview {
    let _g = if wait {
        app.refresh_lock.lock().await
    } else {
        match app.refresh_lock.try_lock() {
            Ok(g) => g,
            Err(_) => return app.overview.read().await.clone(),
        }
    };
    let env = app.env_status();
    let seq = app
        .overview_seq
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        + 1;
    let mut ov = Overview {
        timestamp: chrono::Utc::now().timestamp(),
        seq,
        env: env.clone(),
        ..Default::default()
    };

    // Google Drive 状态刷新（带 in_flight 防并发风暴；首次手动/无头诊断时直接等待结果，确保 --print-overview 不返回 null）
    let target_gdrive_dir = app.active_relay_gdrive_dir();
    if env.rclone_ok && env.rclone_conf_ok {
        let sync_wait_needed = {
            let gd = app.gdrive.lock().await;
            force_gdrive && !gd.in_flight
        };
        if sync_wait_needed {
            {
                let mut gd = app.gdrive.lock().await;
                gd.in_flight = true;
            }
            let status = fetch_gdrive_status(&target_gdrive_dir).await;
            let mut gd = app.gdrive.lock().await;
            gd.status = Some(status.clone());
            gd.updated_at = Some(Instant::now());
            gd.in_flight = false;
            ov.gdrive = Some(status);
        } else {
            let mut gd = app.gdrive.lock().await;
            let dir_changed = gd
                .status
                .as_ref()
                .map(|s| s.dir != target_gdrive_dir)
                .unwrap_or(false);
            let stale = gd
                .updated_at
                .map(|t| t.elapsed() > Duration::from_secs(30))
                .unwrap_or(true);
            if !gd.in_flight && (stale || force_gdrive || dir_changed) {
                gd.in_flight = true;
                let app_bg = app.clone();
                let dir_bg = target_gdrive_dir.clone();
                tokio::spawn(async move {
                    let status = fetch_gdrive_status(&dir_bg).await;
                    let mut g = app_bg.gdrive.lock().await;
                    g.status = Some(status);
                    g.updated_at = Some(Instant::now());
                    g.in_flight = false;
                });
            }
            ov.gdrive = gd.status.clone();
        }
    } else {
        ov.gdrive = app.gdrive.lock().await.status.clone();
    }

    // 本地会话表与蓝绿状态读取涉及磁盘 I/O，放入阻塞线程池执行
    let (sessions, bg_snapshot) = tokio::task::spawn_blocking(|| {
        (
            cli::read_sessions(),
            parse_llm_bluegreen_state(cli::read_llm_bluegreen_state()),
        )
    })
    .await
    .unwrap_or_default();
    let mut assignments: Vec<Assignment> = vec![];

    if !env.token_ok {
        ov.last_ok_timestamp = app
            .last_assignments
            .l()
            .as_ref()
            .map(|(_, _, ok_ts)| *ok_ts);
        ov.error = Some("未检测到 Colab 登录凭证，请在终端运行 `colab auth` 或 `colab status` 完成 Google 授权".into());
    } else {
        // 高频遥测优化：8 秒内直接复用已缓存的计费实例列表与 CU 信息，省去跨洋 /tun/m/ 串行等待，直奔实例遥测快照
        let cached_fresh = if !force_gdrive {
            app.last_assignments
                .l()
                .as_ref()
                .filter(|(_, ts, _)| ts.elapsed() < Duration::from_secs(8))
                .map(|(l, _, ok_ts)| (l.clone(), *ok_ts))
        } else {
            None
        };

        if let Some((cached_list, prev_ok_ts)) = cached_fresh {
            ov.auth_ok = true;
            ov.server_list_known = true;
            assignments = cached_list;
            ov.ccu = app.last_ccu.l().clone();
            ov.timestamp = prev_ok_ts as i64;
            ov.last_ok_timestamp = Some(prev_ok_ts);
            ov.is_stale = false;
            ov.stale_reason = None;
        } else {
            let (ccu, list) = tokio::join!(app.client.ccu_info(), app.client.list_assignments());
            match list {
                Ok(l) => {
                    let now_unix = chrono::Utc::now().timestamp().max(0) as u64;
                    ov.timestamp = now_unix as i64;
                    ov.last_ok_timestamp = Some(now_unix);
                    ov.is_stale = false;
                    ov.stale_reason = None;
                    ov.auth_ok = true;
                    ov.server_list_known = true;
                    *app.last_assignments.l() = Some((l.clone(), Instant::now(), now_unix));
                    assignments = l;
                }
                Err(e) => {
                    let err_msg = format!("{e:#}");
                    let fallback = app
                        .last_assignments
                        .l()
                        .as_ref()
                        .filter(|(_, ts, _)| ts.elapsed() < Duration::from_secs(60))
                        .map(|(l, _, ok_ts)| (l.clone(), *ok_ts));
                    if let Some((cached, prev_ok_ts)) = fallback {
                        ov.auth_ok = true;
                        ov.server_list_known = true;
                        assignments = cached;
                        ov.timestamp = prev_ok_ts as i64;
                        ov.last_ok_timestamp = Some(prev_ok_ts);
                        ov.is_stale = true;
                        ov.stale_reason = Some(err_msg);
                    } else {
                        ov.server_list_known = false;
                        ov.last_ok_timestamp = app
                            .last_assignments
                            .l()
                            .as_ref()
                            .map(|(_, _, ok_ts)| *ok_ts);
                        ov.error = Some(err_msg);
                    }
                }
            }
            match ccu {
                Ok(c) => {
                    *app.last_ccu.l() = Some(c.clone());
                    ov.ccu = Some(c);
                }
                Err(_) => {
                    ov.ccu = if ov.server_list_known {
                        app.last_ccu.l().clone()
                    } else {
                        None
                    };
                }
            }
        }
    }
    ov.server_count = assignments.len();

    // 优先绑定正式受管会话名（避免历史残留 orphan-* 遮蔽新注册的命名会话）
    let mut ep_to_name: HashMap<String, String> = HashMap::new();
    for (n, v) in &sessions {
        if let Some(ep) = session_endpoint(v) {
            let replace = match ep_to_name.get(&ep) {
                None => true,
                Some(prev) => prev.starts_with("orphan-") && !n.starts_with("orphan-"),
            };
            if replace {
                ep_to_name.insert(ep, n.clone());
            }
        }
    }

    let busy = app.busy.l().clone();
    let creating_map = app.creating_vms.l().clone();
    let creating_names: Vec<String> = creating_map.keys().cloned().collect();
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

    let prelim_online_names: Vec<String> = assignments
        .iter()
        .map(|a| {
            ep_to_name
                .get(&a.endpoint)
                .cloned()
                .unwrap_or_else(|| orphan_name(&a.endpoint))
        })
        .collect();
    let live_bg_prelim = filter_live_bluegreen_state(
        bg_snapshot,
        &prelim_online_names,
        &busy,
        &creating_names,
        now,
    );

    // 并发拉取每台在线虚拟机的遥测快照（Contents API，不占用 Kernel）
    // 采用 Stale-While-Revalidate：单次网络抖动失败时保留上一份有效遥测快照，避免界面闪断或误触发重连探针
    let fetched = join_all(assignments.iter().map(|a| {
        fetch_telemetry(
            app,
            a,
            ep_to_name.get(&a.endpoint).map(|s| s.as_str()),
            live_bg_prelim.as_ref(),
        )
    }))
    .await;
    if ov.server_list_known {
        let mut t = app.telemetry.l();
        t.retain(|ep, _| assignments.iter().any(|a| &a.endpoint == ep));
        for (a, entry) in assignments.iter().zip(fetched) {
            match entry.data {
                Some(_) => {
                    t.insert(a.endpoint.clone(), entry);
                }
                None => {
                    if let Some(prev) = t.get_mut(&a.endpoint) {
                        prev.error = entry.error;
                    } else {
                        t.insert(a.endpoint.clone(), entry);
                    }
                }
            }
        }
    }

    struct VmDraft {
        a: Assignment,
        name: String,
        managed: bool,
        sess: Option<Value>,
        tel: Option<Value>,
        terr: Option<String>,
        age: Option<f64>,
    }
    let mut drafts: Vec<VmDraft> = vec![];

    for a in &assignments {
        let (name, managed) = match ep_to_name.get(&a.endpoint) {
            Some(n) if !n.starts_with("orphan-") => (n.clone(), true),
            Some(n) => (n.clone(), false),
            None => (orphan_name(&a.endpoint), false),
        };
        let sess = sessions.get(&name).cloned();
        let (tel, terr) = {
            let t = app.telemetry.l();
            t.get(&a.endpoint)
                .map(|e| (e.data.clone(), e.error.clone()))
                .unwrap_or((None, None))
        };
        let age = tel
            .as_ref()
            .and_then(|v| v.get("timestamp"))
            .and_then(|x| x.as_f64())
            .map(|ts| (now - ts).max(0.0));

        // 自动感知新实例或探针失联实例（超过 45 秒未更新），后台静默注入遥测守护进程
        let needs_probe = tel.is_none() || age.map(|x| x > 45.0).unwrap_or(true);
        if needs_probe && !busy.contains_key(&name) {
            let should_trigger = {
                let mut cd = app.probe_cooldown.l();
                match cd.get(&a.endpoint) {
                    Some(last) if last.elapsed() < Duration::from_secs(40) => false,
                    _ => {
                        cd.insert(a.endpoint.clone(), Instant::now());
                        true
                    }
                }
            };
            // 仅在实例空闲时占用 probing 标记，探针结束只释放自己的标记，绝不覆盖/清除 deleting、creating
            if should_trigger && app.try_claim_busy(&name, "probing") {
                let app_bg = app.clone();
                let a_clone = a.clone();
                let name_clone = name.clone();
                tokio::spawn(async move {
                    let _ = deploy_probe_for_assignment(&app_bg, &a_clone).await;
                    app_bg.release_busy(&name_clone, "probing");
                });
            }
        }

        drafts.push(VmDraft {
            a: a.clone(),
            name,
            managed,
            sess,
            tel,
            terr,
            age,
        });
    }

    // 维护每台在线受管实例的本机按需 SSH WebSocket 桥接监听，并自动回收已下线实例的桥接器
    let active_draft_names: std::collections::HashSet<String> =
        drafts.iter().map(|d| d.name.clone()).collect();
    if ov.server_list_known {
        let mut bridges = app.ssh_bridges.l();
        let stale_keys: Vec<String> = bridges
            .keys()
            .filter(|k| !active_draft_names.contains(*k))
            .cloned()
            .collect();
        for k in stale_keys {
            if let Some(b) = bridges.remove(&k) {
                b.shutdown();
            }
        }
    }
    let mut bridge_ports: HashMap<String, Option<u16>> = HashMap::new();
    let mut ssh_entries: Vec<(String, u16)> = Vec::new();
    for d in &drafts {
        let port = ensure_vm_ssh_bridge(app, &d.name, &d.a).await;
        if let Some(p) = port.filter(|&p| p > 0) {
            ssh_entries.push((d.name.clone(), p));
        }
        bridge_ports.insert(d.name.clone(), port);
    }

    let mut seen: Vec<String> = drafts.iter().map(|d| d.name.clone()).collect();

    // 接力换棒间隙、已完成/已失败离线接力记录（#73）与正在创建中的虚拟机均保留卡片
    let relay_names: Vec<(String, Arc<RelayHandle>)> = app
        .relays
        .l()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let gaps: Vec<(String, Arc<RelayHandle>)> = relay_names
        .into_iter()
        .filter(|(n, _h)| !seen.contains(n))
        .collect();
    seen.extend(gaps.iter().map(|(n, _)| n.clone()));

    let creating_pending: Vec<(String, CreatingVmMeta)> = creating_map
        .into_iter()
        .filter(|(n, _)| !seen.contains(n))
        .collect();
    seen.extend(creating_pending.iter().map(|(n, _)| n.clone()));

    let sync_items: Vec<(String, String, String, String, i64, String, String, bool)> = drafts
        .iter()
        .map(|d| {
            (
                d.name.clone(),
                d.a.endpoint.clone(),
                d.a.runtime_proxy_info.url.clone(),
                d.a.runtime_proxy_info.token.clone(),
                d.a.runtime_proxy_info.token_expires_in_seconds,
                d.a.accelerator.clone(),
                d.a.variant_label(),
                d.a.is_high_ram(),
            )
        })
        .collect();

    // 会话同步、SSH Config 写入、历史事件解析、工作区推断、API Key 解析均为同步磁盘 I/O，统一放入阻塞线程池批量执行
    let vm_jobs: Vec<(
        String,
        Assignment,
        Option<u16>,
        Option<Value>,
        Option<Value>,
    )> = drafts
        .iter()
        .map(|d| {
            (
                d.name.clone(),
                d.a.clone(),
                bridge_ports.get(&d.name).copied().flatten(),
                d.sess.clone(),
                d.tel.clone(),
            )
        })
        .collect();
    let online_names_bg: Vec<String> = drafts.iter().map(|d| d.name.clone()).collect();
    let gap_jobs: Vec<String> = gaps.iter().map(|(n, _)| n.clone()).collect();
    let creating_jobs: Vec<(String, CreatingVmMeta)> = creating_pending.clone();
    let seen_bg = seen.clone();
    let busy_bg = busy.clone();
    let creating_names_bg = creating_names.clone();
    let sync_sessions_allowed = ov.auth_ok && ov.error.is_none();

    let (vm_extra, gap_extra, creating_extra, recent, live_bg, saved_profiles) =
        tokio::task::spawn_blocking(move || {
            if sync_sessions_allowed {
                cli::sync_sessions_with_assignments(&sync_items, !creating_names_bg.is_empty());
                cli::sync_ssh_config_hosts(&ssh_entries);
            }
            let saved_profiles = cli::read_session_profiles();
            let bg_raw = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state());
            let live_bg = filter_live_bluegreen_state(
                bg_raw,
                &online_names_bg,
                &busy_bg,
                &creating_names_bg,
                now,
            );

            let vm_extra: Vec<(
                Vec<CliEventView>,
                WorkspaceInfo,
                VmSshInfo,
                Option<LlmApiAccessInfo>,
            )> = vm_jobs
                .iter()
                .map(|(name, a, bridge_port, sess, tel)| {
                    let (cli_events, history_paths) = cli::read_session_events(name, 12);
                    let workspace = cli::resolve_workspace_for_vm(
                        name,
                        Some(&a.endpoint),
                        sess.as_ref(),
                        tel.as_ref(),
                        &history_paths,
                    );
                    let ssh_info = build_vm_ssh_info(name, a, *bridge_port);

                    let llm_tel = tel.as_ref().and_then(|t| t.get("llm")).filter(|v| v.is_object());
                    let has_llm_proc = tel
                        .as_ref()
                        .map(|t| t["threads"]["llm_process"].is_object())
                        .unwrap_or(false);
                    let matched_bg_slot =
                        match_bluegreen_slot_for_vm(name, live_bg.as_ref(), llm_tel);
                    let bg_port = match matched_bg_slot {
                        Some("blue") => Some(18081),
                        Some("green") => Some(18082),
                        _ => None,
                    };
                    let is_bg_active_session = live_bg
                        .as_ref()
                        .and_then(|v| v.active_session.as_deref())
                        .map(|s| s.trim() == name)
                        .unwrap_or(false);
                    let is_deploying_llm = busy_bg
                        .get(name)
                        .map(|b| b == "deploying_llm")
                        .unwrap_or(false);
                    let has_reg_key =
                        !name.starts_with("orphan-") && cli::has_registered_llm_api_key(name);

                    let llm_api = if llm_tel.is_some()
                        || has_llm_proc
                        || bg_port.is_some()
                        || is_bg_active_session
                        || is_deploying_llm
                        || has_reg_key
                    {
                        let slot = matched_bg_slot
                            .map(|s| s.to_string())
                            .or_else(|| {
                                llm_tel
                                    .and_then(|l| l.get("slot"))
                                    .and_then(|x| x.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_else(|| "standalone".into());
                        let slot_meta_model = matched_bg_slot.and_then(|sl| {
                            live_bg
                                .as_ref()
                                .and_then(|b| b.slots.as_ref())
                                .and_then(|sv| sv.get(sl))
                                .and_then(|e| e.get("model_name"))
                                .and_then(|x| x.as_str())
                                .filter(|s| !s.trim().is_empty())
                        });
                        let engine = llm_tel
                            .and_then(|l| l.get("engine"))
                            .and_then(|x| x.as_str())
                            .or_else(|| {
                                live_bg
                                    .as_ref()
                                    .filter(|_| bg_port.is_some() || is_bg_active_session)
                                    .and_then(|b| b.engine.as_deref())
                            })
                            .unwrap_or("vllm");
                        let model_name = slot_meta_model
                            .or_else(|| {
                                llm_tel
                                    .and_then(|l| l.get("model_name"))
                                    .and_then(|x| x.as_str())
                            })
                            .or_else(|| {
                                live_bg
                                    .as_ref()
                                    .filter(|_| bg_port.is_some() || is_bg_active_session)
                                    .and_then(|b| b.model_name.as_deref())
                            })
                            .unwrap_or("default");
                        let remote_port = llm_tel
                            .and_then(|l| l.get("port"))
                            .and_then(|x| x.as_u64())
                            .map(|p| p as u16)
                            .unwrap_or(8000);
                        Some(cli::resolve_or_create_llm_api_info(
                            name,
                            &slot,
                            engine,
                            bg_port.unwrap_or(0),
                            remote_port,
                            model_name,
                        ))
                    } else {
                        None
                    };

                    (cli_events, workspace, ssh_info, llm_api)
                })
                .collect();

            let gap_extra: Vec<(Vec<CliEventView>, WorkspaceInfo)> = gap_jobs
                .iter()
                .map(|name| {
                    let (cli_events, history_paths) = cli::read_session_events(name, 12);
                    let workspace =
                        cli::resolve_workspace_for_vm(name, None, None, None, &history_paths);
                    (cli_events, workspace)
                })
                .collect();

            let creating_extra: Vec<(Vec<CliEventView>, WorkspaceInfo)> = creating_jobs
                .iter()
                .map(|(name, meta)| {
                    let (cli_events, history_paths) = cli::read_session_events(name, 12);
                    let mut workspace =
                        cli::resolve_workspace_for_vm(name, None, None, None, &history_paths);
                    if !meta.workspace_name.is_empty() {
                        workspace.name = meta.workspace_name.clone();
                    }
                    if meta.workspace_path.is_some() {
                        workspace.path = meta.workspace_path.clone();
                    }
                    if meta.description.is_some() && workspace.description.is_none() {
                        workspace.description = meta.description.clone();
                    }
                    (cli_events, workspace)
                })
                .collect();

            let recent = cli::read_recent_sessions(&seen_bg, 8);
            (vm_extra, gap_extra, creating_extra, recent, live_bg, saved_profiles)
        })
        .await
        .unwrap_or_default();

    let fallback_extra = |name: &str| {
        (
            Vec::<CliEventView>::new(),
            WorkspaceInfo {
                name: name.to_string(),
                ..Default::default()
            },
        )
    };

    let deploy_errs = app.deploy_errors.l().clone();
    let mut vm_extra = vm_extra.into_iter();
    for d in drafts {
        let (cli_events, workspace, ssh_info, llm_api) = match vm_extra.next() {
            Some(x) => x,
            None => {
                let (ev, ws) = fallback_extra(&d.name);
                let ssh = build_vm_ssh_info(
                    &d.name,
                    &d.a,
                    bridge_ports.get(&d.name).copied().flatten(),
                );
                (ev, ws, ssh, None)
            }
        };
        let cur_busy = app.busy.l().get(&d.name).cloned();
        let sess = d.sess.as_ref();
        let a = &d.a;
        let ka_status = snapshot_keep_alive_for_vm(app, &d.name, Some(&a.endpoint), &saved_profiles);

        ov.vms.push(VmView {
            name: d.name.clone(),
            workspace,
            endpoint: Some(a.endpoint.clone()),
            accelerator: a.hardware_label(),
            variant: a.variant_label(),
            high_ram: a.is_high_ram(),
            on_server: true,
            managed: d.managed,
            orphan: !d.managed,
            kernel_id: sess
                .and_then(|v| v.get("kernel_id"))
                .and_then(|x| x.as_str())
                .map(|x| x.to_string()),
            session_id: sess
                .and_then(|v| v.get("session_id"))
                .and_then(|x| x.as_str())
                .map(|x| x.to_string()),
            running_task: sess
                .and_then(|v| v.get("running"))
                .and_then(|x| x.as_str())
                .map(|x| x.to_string()),
            last_execution: sess
                .and_then(|v| v.get("last_execution"))
                .cloned()
                .unwrap_or(Value::Null),
            cli_events,
            telemetry: d.tel,
            telemetry_age: d.age,
            telemetry_error: d.terr,
            relay: app.relay(&d.name).map(|h| h.snapshot()),
            busy: cur_busy,
            ssh: Some(ssh_info),
            llm_api,
            keep_alive: Some(ka_status),
            grab_status: None,
            deploy_error: deploy_errs.get(&d.name).cloned(),
            server_state_unknown: false,
        });
    }

    let mut gap_extra = gap_extra.into_iter();
    for (name, h) in gaps {
        let (cli_events, mut workspace) = gap_extra.next().unwrap_or_else(|| fallback_extra(&name));
        if let Some(ref bound_ws) = *h.workspace_dir.l() {
            if workspace.path.is_none() {
                workspace.path = Some(bound_ws.display().to_string());
            }
            if workspace.name.is_empty() || workspace.name == name {
                if let Some(dn) = bound_ws
                    .file_name()
                    .and_then(|s| s.to_str())
                    .filter(|s| !s.is_empty())
                {
                    workspace.name = dn.to_string();
                }
            }
        }
        let snap = h.snapshot();
        let cfg = snap.config.clone();
        let acc_upper = cfg.gpu_type.trim().to_uppercase();
        let variant = if matches!(acc_upper.as_str(), "V5E1" | "V6E1") {
            "TPU".to_string()
        } else if acc_upper.is_empty() || acc_upper == "CPU" || acc_upper == "NONE" {
            "DEFAULT".to_string()
        } else {
            "GPU".to_string()
        };
        let cur_busy = app.busy.l().get(&name).cloned();
        if ov.server_list_known && !ov.is_stale && !snap.running {
            h.cloud_unassign_verified
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let card_state_unknown = !ov.server_list_known
            || (ov.is_stale
                && !h
                    .cloud_unassign_verified
                    .load(std::sync::atomic::Ordering::SeqCst));
        let relay_grab = snap.grab_status.clone();
        ov.vms.push(VmView {
            name: name.clone(),
            workspace,
            accelerator: if acc_upper.is_empty() || acc_upper == "NONE" {
                "CPU".into()
            } else {
                acc_upper
            },
            variant,
            high_ram: cfg.high_mem,
            on_server: false,
            managed: true,
            cli_events,
            relay: Some(snap),
            busy: cur_busy,
            grab_status: relay_grab,
            deploy_error: deploy_errs.get(&name).cloned(),
            server_state_unknown: card_state_unknown,
            ..Default::default()
        });
    }

    let mut creating_extra = creating_extra.into_iter();
    for (name, meta) in creating_pending {
        let (cli_events, workspace) =
            creating_extra.next().unwrap_or_else(|| fallback_extra(&name));
        let live_grab = meta.grab_status.l().clone();
        let active_acc = live_grab
            .as_ref()
            .filter(|g| g.active && !g.target_gpu.is_empty())
            .map(|g| g.target_gpu.clone())
            .unwrap_or_else(|| meta.accelerator.clone());
        let acc_upper = active_acc.trim().to_uppercase();
        let variant = if matches!(acc_upper.as_str(), "V5E1" | "V6E1") {
            "TPU".to_string()
        } else if acc_upper.is_empty() || acc_upper == "CPU" || acc_upper == "NONE" {
            "DEFAULT".to_string()
        } else {
            "GPU".to_string()
        };
        let cur_busy = app
            .busy
            .l()
            .get(&name)
            .cloned()
            .or_else(|| Some("creating".into()));
        let ka_creating = app.keep_alive.l().get(&name).map(|rt| rt.to_status());
        ov.vms.push(VmView {
            name: name.clone(),
            workspace,
            accelerator: if acc_upper.is_empty() || acc_upper == "NONE" {
                "CPU".into()
            } else {
                acc_upper
            },
            variant,
            high_ram: meta.high_mem,
            managed: true,
            cli_events,
            busy: cur_busy,
            keep_alive: ka_creating,
            grab_status: live_grab,
            deploy_error: deploy_errs.get(&name).cloned(),
            ..Default::default()
        });
    }

    ov.vms
        .sort_by(|a, b| a.workspace.name.cmp(&b.workspace.name).then(a.name.cmp(&b.name)));
    ov.recent_sessions = recent;
    ov.llm_bluegreen = live_bg;

    *app.overview.write().await = ov.clone();
    ov
}

// ───────────────────────────── 彻底销毁（保留底层能力供 CLI 测试使用） ─────────────────────────────

#[cfg(windows)]
fn query_process_image_path(pid: u32) -> Option<PathBuf> {
    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> isize;
        fn QueryFullProcessImageNameW(
            process: isize,
            flags: u32,
            exe_name: *mut u16,
            size: *mut u32,
        ) -> i32;
        fn CloseHandle(object: isize) -> i32;
    }
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    if pid == 0 {
        return None;
    }
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
        if h == 0 {
            return None;
        }
        let mut buf = vec![0u16; 1024];
        let mut size = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(h, 0, buf.as_mut_ptr(), &mut size);
        let _ = CloseHandle(h);
        if ok == 0 || size == 0 {
            return None;
        }
        Some(PathBuf::from(String::from_utf16_lossy(&buf[..size as usize])))
    }
}

#[cfg(target_os = "linux")]
fn query_process_image_path(pid: u32) -> Option<PathBuf> {
    if pid == 0 {
        return None;
    }
    std::fs::read_link(format!("/proc/{pid}/exe")).ok()
}

#[cfg(not(any(windows, target_os = "linux")))]
fn query_process_image_path(_pid: u32) -> Option<PathBuf> {
    None
}

fn is_same_executable_pid(pid: u32) -> bool {
    let Some(target) = query_process_image_path(pid) else {
        return false;
    };
    let Ok(me) = std::env::current_exe() else {
        return false;
    };
    if let (Ok(c1), Ok(c2)) = (std::fs::canonicalize(&target), std::fs::canonicalize(&me)) {
        return c1 == c2;
    }
    #[cfg(windows)]
    {
        target
            .to_string_lossy()
            .eq_ignore_ascii_case(&me.to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        target == me
    }
}

fn is_python_process_pid(pid: u32) -> bool {
    let Some(img) = query_process_image_path(pid) else {
        return false;
    };
    img.file_stem()
        .and_then(|s| s.to_str())
        .map(|stem| {
            let lower = stem.to_ascii_lowercase();
            lower.starts_with("python") || lower == "uv"
        })
        .unwrap_or(false)
}

fn cleanup_llm_bluegreen_after_terminate(
    terminated_name: &str,
    remaining_endpoints: &[String],
    steps: &mut Vec<String>,
) {
    let state_path = cli::llm_bluegreen_state_path();
    let Some(mut bg_raw) = cli::read_llm_bluegreen_state() else {
        return;
    };
    let Some(bg_typed) = parse_llm_bluegreen_state(Some(bg_raw.clone())) else {
        return;
    };
    let active_session = bg_typed.active_session.as_deref().unwrap_or("").trim().to_string();
    let standby_session = bg_typed.standby_session.as_deref().unwrap_or("").trim().to_string();
    let blue_name = bg_slot_session_name(&bg_typed, "blue");
    let green_name = bg_slot_session_name(&bg_typed, "green");
    let relevant = terminated_name == blue_name
        || terminated_name == green_name
        || (!active_session.is_empty() && terminated_name == active_session)
        || (!standby_session.is_empty() && terminated_name == standby_session)
        || terminated_name.starts_with("llm-")
        || terminated_name.starts_with("orphan-")
        || remaining_endpoints.is_empty();
    if !relevant {
        return;
    }

    let sessions = cli::read_sessions();
    let slot_still_on_server = |slot_key: &str, slot_vm: &str| -> bool {
        if slot_vm.is_empty() || slot_vm == terminated_name {
            return false;
        }
        let ep_opt = sessions
            .get(slot_vm)
            .and_then(session_endpoint)
            .or_else(|| {
                bg_typed
                    .slots
                    .as_ref()
                    .and_then(|sv| sv.get(slot_key))
                    .and_then(|e| e.get("endpoint"))
                    .and_then(|x| x.as_str())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            });
        ep_opt
            .map(|ep| remaining_endpoints.contains(&ep))
            .unwrap_or(false)
    };

    let blue_alive = slot_still_on_server("blue", &blue_name);
    let green_alive = slot_still_on_server("green", &green_name);

    if !blue_alive && !green_alive {
        let _ = std::fs::remove_file(&state_path);
        let pid_path = cli::colab_config_dir().join("llm_local_gateway.pid");
        if let Ok(raw_pid) = std::fs::read_to_string(&pid_path) {
            if let Ok(pid) = raw_pid.trim().parse::<u32>() {
                if pid > 0 && pid != std::process::id() && is_python_process_pid(pid) {
                    kill_process_tree(pid);
                }
            }
            let _ = std::fs::remove_file(&pid_path);
        }
        steps.push("已同步清理本机 LLM 蓝绿状态与本地网关隧道进程".into());
    } else {
        let surviving_slot = if blue_alive { "blue" } else { "green" };
        let surviving_sess = if blue_alive {
            blue_name.clone()
        } else {
            green_name.clone()
        };
        let terminated_slot = if surviving_slot == "blue" {
            "green"
        } else {
            "blue"
        };
        if let Some(obj) = bg_raw.as_object_mut() {
            obj.insert("active_slot".into(), Value::String(surviving_slot.into()));
            obj.insert("active_session".into(), Value::String(surviving_sess.clone()));
            obj.insert("standby_slot".into(), Value::Null);
            obj.insert("standby_session".into(), Value::Null);
            if terminated_slot == "blue" && obj.contains_key("blue_session") {
                obj.insert("blue_session".into(), Value::Null);
            }
            if terminated_slot == "green" && obj.contains_key("green_session") {
                obj.insert("green_session".into(), Value::Null);
            }
            if let Some(slots_obj) = obj.get_mut("slots").and_then(|s| s.as_object_mut()) {
                slots_obj.remove(terminated_slot);
            }
            obj.insert("phase".into(), Value::String("serving".into()));
        }
        if let Ok(raw) = serde_json::to_string_pretty(&bg_raw) {
            let tmp = state_path.with_extension("json.tmp");
            if std::fs::write(&tmp, &raw).is_ok() {
                let _ = std::fs::rename(&tmp, &state_path);
            } else {
                let _ = std::fs::write(&state_path, raw);
            }
        }
        steps.push(format!(
            "已更新蓝绿状态：当前由剩余在线槽位 [{} · {}] 继续提供服务",
            surviving_slot.to_uppercase(),
            surviving_sess
        ));
    }
}

fn cleanup_local_vm_artifacts(app: &AppState, name: &str) {
    app.set_deploy_error(name, None);
    app.creating_vms.l().remove(name);
    app.invalidate_keep_alive(name, true);
    let remaining_ssh: Option<Vec<(String, u16)>> = {
        let mut bridges = app.ssh_bridges.l();
        if let Some(b) = bridges.remove(name) {
            b.shutdown();
            Some(
                bridges
                    .iter()
                    .map(|(k, v)| (k.clone(), v.local_port))
                    .collect(),
            )
        } else {
            None
        }
    };
    if let Some(entries) = remaining_ssh {
        cli::sync_ssh_config_hosts(&entries);
    }
    if name.starts_with("orphan-") {
        cli::remove_llm_api_key_entry(name);
    }
}

fn push_unique_endpoint(
    targets: &mut Vec<String>,
    other_session_endpoints: &std::collections::HashSet<String>,
    cand: Option<String>,
) {
    if let Some(ep) = cand.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
        if !other_session_endpoints.contains(&ep) && !targets.contains(&ep) {
            targets.push(ep);
        }
    }
}

pub async fn hard_terminate(
    app: &Arc<AppState>,
    name: &str,
    endpoint_hint: Option<String>,
    stop_relay: bool,
) -> TerminateReceipt {
    let mut r = TerminateReceipt {
        name: name.into(),
        ..Default::default()
    };
    app.invalidate_keep_alive(name, true);

    let mut stopped_local_relay = false;
    let mut stopped_creating_vm = false;
    let mut stopped_during_provisioning = false;
    let mut pre_prov_snapshot_invalid = true;
    let mut pre_prov_endpoints: std::collections::HashSet<String> =
        std::collections::HashSet::new();
    let removed_creating = app.creating_vms.l().remove(name);
    if let Some(cm) = removed_creating {
        cm.cancel.cancel();
        let pid = *cm.child_pid.l();
        if let Some(pid) = pid {
            cli::kill_pid_tree(pid).await;
        }
        stopped_creating_vm = true;
        stopped_during_provisioning = true;
        pre_prov_endpoints = cm.pre_provision_endpoints.l().clone();
        pre_prov_snapshot_invalid = cm
            .pre_provision_snapshot_invalid
            .swap(true, std::sync::atomic::Ordering::SeqCst);
        if !cm.had_ws_entry && !cli::read_sessions().contains_key(name) {
            cli::remove_workspace_entry(name);
        }
        r.steps
            .push("已取消本机抢卡创建任务并结束 colab 子进程树".into());
    }
    if let Some(h) = app.relay(name) {
        if stop_relay {
            h.keep_vm_on_stop
                .store(false, std::sync::atomic::Ordering::SeqCst);
            if h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst) {
                h.stop().await;
                stopped_local_relay = true;
                stopped_during_provisioning = h
                    .stopped_during_provisioning
                    .swap(false, std::sync::atomic::Ordering::SeqCst);
                r.steps
                    .push("已停止本机接力监工任务并结束 colab 子进程树".into());
            } else {
                h.stopped_during_provisioning
                    .store(false, std::sync::atomic::Ordering::SeqCst);
            }
        } else {
            stopped_during_provisioning = h
                .stopped_during_provisioning
                .swap(false, std::sync::atomic::Ordering::SeqCst);
        }
        pre_prov_endpoints = h.pre_provision_endpoints.l().clone();
        pre_prov_snapshot_invalid = h
            .pre_provision_snapshot_invalid
            .swap(true, std::sync::atomic::Ordering::SeqCst);
    }

    if stopped_local_relay || stopped_creating_vm {
        tokio::time::sleep(Duration::from_millis(if cfg!(test) { 10 } else { 300 })).await;
    }

    let sessions = cli::read_sessions();
    let managed = sessions.contains_key(name);
    let other_session_endpoints: std::collections::HashSet<String> = sessions
        .iter()
        .filter(|(k, _)| k.as_str() != name && !k.starts_with("orphan-"))
        .filter_map(|(_, v)| session_endpoint(v))
        .collect();

    let mut initial_assignments = app.client.list_assignments().await;

    // Issue #39：同时收集该会话当前绑定的真实端点（sessions.json / SSH 桥接 / 总览快照）以及界面传入的 endpoint_hint，
    // 防止接力换棒后界面传入旧端点 A 而忽略服务端仍在运行的当前端点 B！
    let mut target_endpoints: Vec<String> = Vec::new();
    let current_session_ep = sessions.get(name).and_then(session_endpoint);
    push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, current_session_ep.clone());
    push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, endpoint_hint.clone());
    let bridge_ep = app
        .ssh_bridges
        .l()
        .get(name)
        .map(|b| b.target.l().endpoint.clone());
    push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, bridge_ep);
    let ov_ep = app
        .overview
        .read()
        .await
        .vms
        .iter()
        .find(|v| v.name == name)
        .and_then(|v| v.endpoint.clone());
    push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, ov_ep);

    if let Ok(ref list) = initial_assignments {
        let orphan_ep = list
            .iter()
            .find(|a| matches_orphan_name(&a.endpoint, name))
            .map(|a| a.endpoint.clone());
        push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, orphan_ep);
    }

    // Issue #77：仅在明确处于 provisioning（colab new）阶段被取消、尚未确认任何会话归属端点、且持有本次分配前有效快照时，
    // 才探测取消分配期间刚诞生的残留孤儿；运行态或已锁定目标端点时严禁将其他项目的未登记端点判作残留误关！
    if stopped_during_provisioning && target_endpoints.is_empty() && !pre_prov_snapshot_invalid {
        for probe in 0..3 {
            let refreshed_sess_ep = cli::read_sessions().get(name).and_then(session_endpoint);
            push_unique_endpoint(
                &mut target_endpoints,
                &other_session_endpoints,
                refreshed_sess_ep,
            );
            if let Ok(ref list) = initial_assignments {
                if let Some(ep_found) = list
                    .iter()
                    .find(|a| matches_orphan_name(&a.endpoint, name))
                    .map(|a| a.endpoint.clone())
                {
                    push_unique_endpoint(
                        &mut target_endpoints,
                        &other_session_endpoints,
                        Some(ep_found),
                    );
                }
            }
            if !target_endpoints.is_empty() {
                break;
            }
            if let Ok(ref list) = initial_assignments {
                let newly_leaked: Vec<String> = list
                    .iter()
                    .map(|a| a.endpoint.clone())
                    .filter(|e| {
                        !other_session_endpoints.contains(e)
                            && !pre_prov_endpoints.contains(e)
                            && !target_endpoints.contains(e)
                    })
                    .collect();
                if newly_leaked.len() == 1 {
                    let leaked_ep = newly_leaked[0].clone();
                    r.steps.push(format!(
                        "捕获到取消分配期间新诞生的残留实例 {leaked_ep}，正在自动释放"
                    ));
                    push_unique_endpoint(
                        &mut target_endpoints,
                        &other_session_endpoints,
                        Some(leaked_ep),
                    );
                    break;
                }
            }
            if probe + 1 < 3 {
                tokio::time::sleep(Duration::from_millis(if cfg!(test) { 15 } else { 700 })).await;
                initial_assignments = app.client.list_assignments().await;
            }
        }
    }

    r.endpoint = target_endpoints.first().cloned();

    if target_endpoints.is_empty() {
        match initial_assignments {
            Ok(list) => {
                r.remaining_endpoints = list.iter().map(|a| a.endpoint.clone()).collect();
                let unaccounted: Vec<String> = r
                    .remaining_endpoints
                    .iter()
                    .filter(|e| {
                        !other_session_endpoints.contains(*e)
                            && !(stopped_during_provisioning
                                && !pre_prov_snapshot_invalid
                                && pre_prov_endpoints.contains(*e))
                    })
                    .cloned()
                    .collect();
                let can_verify_absent = list.is_empty()
                    || ((stopped_local_relay || stopped_creating_vm || managed)
                        && unaccounted.is_empty());
                if can_verify_absent {
                    r.verified_unassigned = true;
                    if stopped_creating_vm {
                        r.steps.push(
                            "服务端核验通过：已放弃抢卡且未产生残留计费实例".into(),
                        );
                    } else if stopped_local_relay {
                        r.steps.push(
                            "服务端核验通过：未产生未受管残留计费实例，本机接力任务已终止".into(),
                        );
                    } else {
                        r.steps
                            .push("服务端计费列表中已不存在该实例（无需重复释放）".into());
                    }
                    if managed && cli::colab_bin().exists() {
                        let sess_args = cli::colab_args_for_session(name, &s(&["sessions"]));
                        let _ = cli::run_simple(
                            &cli::colab_bin(),
                            &sess_args,
                            Duration::from_secs(60),
                        )
                        .await;
                    }
                    cleanup_local_vm_artifacts(app, name);
                    cleanup_llm_bluegreen_after_terminate(
                        name,
                        &r.remaining_endpoints,
                        &mut r.steps,
                    );
                } else if !unaccounted.is_empty() {
                    r.verified_unassigned = false;
                    r.error = Some(format!(
                        "无法将 [{name}] 映射到会话表，但服务端仍存在 {} 台未关联计费实例（{}），请刷新总览后销毁残留孤儿实例",
                        unaccounted.len(),
                        unaccounted.join(", ")
                    ));
                } else {
                    r.verified_unassigned = false;
                    r.error = Some(format!(
                        "未找到名为 [{name}] 的本地会话或匹配实例（当前服务端仍有 {} 台其他受管实例在线），未执行任何释放操作",
                        r.remaining_endpoints.len()
                    ));
                }
            }
            Err(e) => {
                r.verified_unassigned = false;
                r.error = Some(format!(
                    "无法确定 [{name}] 的 endpoint，且查询服务端计费列表失败：{e}"
                ));
            }
        }
        if let Some(h) = app.relay(name) {
            h.cloud_unassign_verified
                .store(r.verified_unassigned, std::sync::atomic::Ordering::SeqCst);
        }
        if let Ok(c) = app.client.ccu_info().await {
            r.ccu = Some(c);
        }
        return r;
    }

    if managed && !name.starts_with("orphan-") && cli::colab_bin().exists() {
        let stop_args = cli::colab_args_for_session(name, &s(&["stop", "-s", name]));
        match cli::run_simple(
            &cli::colab_bin(),
            &stop_args,
            Duration::from_secs(120),
        )
        .await
        {
            Ok(o) => {
                let cleaned = cli::strip_colab_cli_banners(&o.output);
                let last = cli::last_meaningful_cli_line(&cleaned, "已执行");
                r.steps.push(format!("colab stop -s {name} (exit={})：{last}", o.code));
            }
            Err(e) => r.steps.push(format!("colab stop 执行失败：{e}")),
        }
    }

    for attempt in 1..=6 {
        // 每轮复查本地 sessions.json，防止并发换棒写入了新的当前端点而漏释放
        let latest_sess_ep = cli::read_sessions().get(name).and_then(session_endpoint);
        push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, latest_sess_ep);

        match app.client.list_assignments().await {
            Ok(list) => {
                r.remaining_endpoints = list.iter().map(|a| a.endpoint.clone()).collect();
                let still_active: Vec<String> = target_endpoints
                    .iter()
                    .filter(|ep| r.remaining_endpoints.contains(*ep))
                    .cloned()
                    .collect();
                if still_active.is_empty() {
                    r.verified_unassigned = true;
                    r.steps.push(format!(
                        "服务端核验通过：/tun/m/assignments 中已不存在 {}（剩余计费实例 {} 台）",
                        target_endpoints.join(", "),
                        r.remaining_endpoints.len()
                    ));
                    break;
                }
                r.endpoint = still_active.first().cloned().or(r.endpoint);
                for active_ep in &still_active {
                    match app.client.unassign(active_ep).await {
                        Ok(_) => r.steps.push(format!(
                            "第 {attempt} 次直接调用 /tun/m/unassign/{active_ep} 成功"
                        )),
                        Err(e) => r
                            .steps
                            .push(format!("第 {attempt} 次 unassign({active_ep}) 返回：{e}")),
                    }
                }
            }
            Err(e) => r.steps.push(format!("第 {attempt} 次核验请求失败：{e}")),
        }
        tokio::time::sleep(Duration::from_millis(if cfg!(test) { 10 } else { 1500 })).await;
    }

    if !r.verified_unassigned {
        let latest_sess_ep = cli::read_sessions().get(name).and_then(session_endpoint);
        push_unique_endpoint(&mut target_endpoints, &other_session_endpoints, latest_sess_ep);
        if let Ok(list) = app.client.list_assignments().await {
            r.remaining_endpoints = list.iter().map(|a| a.endpoint.clone()).collect();
            let still_active: Vec<String> = target_endpoints
                .iter()
                .filter(|ep| r.remaining_endpoints.contains(*ep))
                .cloned()
                .collect();
            if still_active.is_empty() {
                r.verified_unassigned = true;
                r.steps.push(format!(
                    "最终核验通过：/tun/m/assignments 中已不存在 {}（剩余计费实例 {} 台）",
                    target_endpoints.join(", "),
                    r.remaining_endpoints.len()
                ));
            } else {
                r.endpoint = still_active.first().cloned().or(r.endpoint);
            }
        }
    }

    if r.verified_unassigned {
        {
            let mut tel_map = app.telemetry.l();
            for ep in &target_endpoints {
                tel_map.remove(ep);
            }
        }
        if let Some((list, _, _)) = app.last_assignments.l().as_mut() {
            list.retain(|a| !target_endpoints.contains(&a.endpoint));
        }
        if cli::read_sessions().contains_key(name) && cli::colab_bin().exists() {
            let sess_args = cli::colab_args_for_session(name, &s(&["sessions"]));
            let _ = cli::run_simple(&cli::colab_bin(), &sess_args, Duration::from_secs(60)).await;
            r.steps.push("已同步清理 colab-cli 本地会话表".into());
        }
        cleanup_local_vm_artifacts(app, name);
        cleanup_llm_bluegreen_after_terminate(name, &r.remaining_endpoints, &mut r.steps);
    } else {
        let active_str: Vec<String> = target_endpoints
            .iter()
            .filter(|ep| r.remaining_endpoints.contains(*ep))
            .cloned()
            .collect();
        if !active_str.is_empty() {
            r.error = Some(format!(
                "服务端计费列表中仍存在该会话的活跃端点（{}），未通过停费核验，请重试",
                active_str.join(", ")
            ));
        } else {
            r.error =
                Some("服务端计费列表中仍存在该实例或核验请求失败，未执行本地移除（避免假性删除），请重试".into());
        }
    }

    if let Some(h) = app.relay(name) {
        h.cloud_unassign_verified
            .store(r.verified_unassigned, std::sync::atomic::Ordering::SeqCst);
    }
    if let Ok(c) = app.client.ccu_info().await {
        r.ccu = Some(c);
    }
    r
}

// ───────────────────────────── Tauri Commands ─────────────────────────────

type AppStateRef<'a> = State<'a, Arc<AppState>>;

fn emit_overview(handle: &AppHandle, ov: &Overview) {
    let _ = handle.emit("overview", ov);
}

/// 后台刷新一次总览并推送前端（命令立即返回，不等待跨洋网络请求；排队等待正在进行的刷新以确保状态变更被反映）
fn spawn_refresh(app: &Arc<AppState>, handle: &AppHandle) {
    let app = app.clone();
    let handle = handle.clone();
    tauri::async_runtime::spawn(async move {
        let ov = refresh_overview_opts(&app, false, true).await;
        emit_overview(&handle, &ov);
    });
}

#[tauri::command]
async fn get_overview(state: AppStateRef<'_>) -> Result<Overview, String> {
    Ok(state.overview.read().await.clone())
}

#[tauri::command]
async fn refresh_now(handle: AppHandle, state: AppStateRef<'_>) -> Result<Overview, String> {
    let app = state.inner().clone();
    let ov = refresh_overview(&app, true).await;
    emit_overview(&handle, &ov);
    Ok(ov)
}

#[derive(Deserialize, Serialize, Clone, Default, Debug)]
pub struct CreateVmLlmConfig {
    pub engine: String,
    #[serde(alias = "model")]
    pub model_name: String,
    #[serde(default)]
    pub slot: Option<String>,
    #[serde(default)]
    pub gdrive_model_path: Option<String>,
    #[serde(default)]
    pub custom_api_key: Option<String>,
}

#[derive(Deserialize, Serialize, Clone, Default, Debug)]
pub struct CreateVmEnvConfig {
    #[serde(default)]
    pub sync_rclone: bool,
    #[serde(default)]
    pub mount_gdrive: bool,
    #[serde(default)]
    pub pip_packages: Option<String>,
    #[serde(default)]
    pub install_workspace_requirements: bool,
    #[serde(default)]
    pub startup_script: Option<String>,
    #[serde(default)]
    pub env_vars: Option<String>,
}

// ───────────────────────────── 通用每实例保活引擎 ─────────────────────────────
// 覆盖所有在线实例（通用算力 / 终端 SSH / 大模型 / 接力训练）：
// 1. 通过 Jupyter /api/kernels 检查运行时连接。
// 2. 代理令牌实时保鲜：始终从最新 Assignment 获取有效 1h runtimeProxyInfo.token，401/403 自动刷新重试
// 3. 忙时避让的 Jupyter Kernel 脉冲心跳：先查本地忙态与 GET /api/kernels，若存在 busy/starting 内核则主动跳过（不抢占、不阻塞）；
//    仅当空闲时通过独立 ZMQ session_id=None 执行毫秒级 `pass` 刷新 last_activity，复用已有空闲 kernel_id 不产生多余内核。

const REMOTE_KERNEL_KEEPALIVE_PY: &str = r#"
import queue
import sys
import time
import jupyter_kernel_client.manager as _jkm
from colab_cli.runtime import ColabRuntime

url, tok, kid = sys.argv[1], sys.argv[2], (sys.argv[3] or "").strip()
if not kid:
    print("KA_NO_KERNEL", flush=True)
    sys.exit(0)

def _forbid_start_kernel(*args, **kwargs):
    raise RuntimeError("KA_KERNEL_GONE: start_kernel forbidden during keep-alive")

_jkm.KernelHttpManager.start_kernel = _forbid_start_kernel

rt = ColabRuntime(url, tok, kernel_id=kid, session_id=None)
try:
    wsclient = rt.kernel_client._manager.client
    msg_id = wsclient.execute("pass", silent=True, store_history=False, allow_stdin=False)

    def _next_msg(get_msg, waiting_for):
        deadline = time.monotonic() + 15.0
        while True:
            if not wsclient.connection_ready.is_set():
                raise RuntimeError("Connection was lost.")
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"Timeout waiting for {waiting_for}")
            try:
                msg = get_msg(timeout=min(0.25, left))
            except queue.Empty:
                continue
            state = msg.get("content", {}).get("execution_state")
            if msg.get("header", {}).get("msg_type") == "status" and state in ("dead", "restarting"):
                raise RuntimeError(f"Kernel {state} before keep-alive finished.")
            if msg.get("parent_header", {}).get("msg_id") == msg_id:
                return msg

    while True:
        iopub = _next_msg(wsclient.get_iopub_msg, "iopub")
        if (
            iopub.get("header", {}).get("msg_type") == "status"
            and iopub.get("content", {}).get("execution_state") == "idle"
        ):
            break

    reply = _next_msg(wsclient.get_shell_msg, "execute_reply")
    content = reply.get("content", {})
    if content.get("status") != "ok":
        ename = content.get("ename") or content.get("status") or "error"
        evalue = content.get("evalue") or ""
        print(f"KA_KERNEL_ERR:{ename}: {evalue}".strip(), file=sys.stderr, flush=True)
        sys.exit(2)

    print(f"KA_KERNEL_OK:{rt.kernel_id or kid}", flush=True)
except Exception as exc:
    if "KA_KERNEL_GONE" in str(exc):
        print("KA_NO_KERNEL", flush=True)
        sys.exit(0)
    print(f"KA_KERNEL_ERR:{exc}", file=sys.stderr, flush=True)
    sys.exit(2)
finally:
    rt.stop()
"#;

fn snapshot_keep_alive_for_vm(
    app: &AppState,
    vm_name: &str,
    endpoint: Option<&str>,
    saved_profiles: &HashMap<String, cli::SessionProfile>,
) -> KeepAliveStatus {
    if app.is_deleting(vm_name) {
        return KeepAliveStatus {
            enabled: false,
            interval_seconds: DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS,
            kernel_ping: true,
            state: "paused".into(),
            kernel_ping_state: "disabled".into(),
            ..Default::default()
        };
    }
    let mut map = app.keep_alive.l();
    if let Some(rt) = map.get_mut(vm_name) {
        if let Some(ep) = endpoint {
            let endpoint_handover = rt.endpoint.as_deref().is_some_and(|cur| cur != ep)
                || rt
                    .last_verified_endpoint
                    .as_deref()
                    .is_some_and(|v| v != ep);
            if endpoint_handover {
                rt.reset_verification(Some(ep.to_string()));
                rt.epoch = app.bump_keep_alive_epoch(vm_name);
            } else if rt.endpoint.is_none() {
                rt.endpoint = Some(ep.to_string());
            }
        }
        return rt.to_status();
    }
    if let Some(cfg) = saved_profiles
        .get(vm_name)
        .and_then(|p| p.keep_alive.clone())
    {
        let mut rt = VmKeepAliveRuntime::from_config(cfg, endpoint.map(|s| s.to_string()));
        rt.epoch = app.current_keep_alive_epoch(vm_name);
        let status = rt.to_status();
        map.insert(vm_name.to_string(), rt);
        return status;
    }
    // 外部/历史未配置过保活的在线实例：默认展示为“未开启”，用户可在卡片上一键开启
    KeepAliveStatus {
        enabled: false,
        interval_seconds: DEFAULT_KEEP_ALIVE_INTERVAL_SECONDS,
        kernel_ping: true,
        state: "paused".into(),
        kernel_ping_state: "disabled".into(),
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KernelPingOutcome {
    Ok,
    NoKernel,
}

async fn execute_remote_kernel_ping(
    a: &Assignment,
    kernel_id: &str,
    cancel: Option<tokio_util::sync::CancellationToken>,
) -> Result<KernelPingOutcome, String> {
    let py = cli::colab_python_bin();
    let args = vec![
        "-u".to_string(),
        "-c".to_string(),
        REMOTE_KERNEL_KEEPALIVE_PY.to_string(),
        a.runtime_proxy_info.url.clone(),
        a.runtime_proxy_info.token.clone(),
        kernel_id.to_string(),
    ];
    let out = cli::run_streaming(
        &py,
        &args,
        None,
        Duration::from_secs(20),
        cancel,
        None,
        |_| {},
    )
    .await
    .map_err(|e| format!("启动 Kernel 心跳子进程失败: {e:#}"))?;
    if out.cancelled {
        return Err("保活心跳已取消".into());
    }
    if out.timed_out {
        return Err("Jupyter Kernel `pass` 心跳超时 (20s)".into());
    }
    if out.code == 0 && out.stdout.contains("KA_NO_KERNEL") {
        return Ok(KernelPingOutcome::NoKernel);
    }
    if out.code != 0
        || !out.stdout.contains("KA_KERNEL_OK:")
        || out.output.contains("KA_KERNEL_ERR:")
    {
        let detail = out
            .output
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("未知异常")
            .trim();
        return Err(format!("Kernel 心跳失败 (code={}): {detail}", out.code));
    }
    Ok(KernelPingOutcome::Ok)
}

struct KeepAliveInFlightGuard {
    app: Arc<AppState>,
    vm_name: String,
    tick_id: u64,
}

impl Drop for KeepAliveInFlightGuard {
    fn drop(&mut self) {
        let mut map = self.app.keep_alive.l();
        if let Some(rt) = map.get_mut(&self.vm_name) {
            if rt.in_flight_tick_id == Some(self.tick_id) {
                rt.in_flight = false;
                rt.in_flight_tick_id = None;
                rt.active_cancel = None;
            }
        }
    }
}

pub(crate) async fn run_keep_alive_tick_for_vm(
    app: &Arc<AppState>,
    vm_name: &str,
    a: &Assignment,
    kernel_hint: Option<String>,
    force: bool,
    require_enabled: bool,
    expected_epoch: Option<u64>,
) -> Option<KeepAliveStatus> {
    run_keep_alive_tick_inner(
        app,
        vm_name,
        a,
        kernel_hint,
        force,
        require_enabled,
        expected_epoch,
        None,
    )
    .await
}

async fn run_keep_alive_tick_inner(
    app: &Arc<AppState>,
    vm_name: &str,
    a: &Assignment,
    kernel_hint: Option<String>,
    force: bool,
    require_enabled: bool,
    expected_epoch: Option<u64>,
    pre_claimed: Option<(
        u64,
        tokio_util::sync::CancellationToken,
        KeepAliveConfig,
        u64,
        u64,
        String,
        KeepAliveInFlightGuard,
    )>,
) -> Option<KeepAliveStatus> {
    if app.is_deleting(vm_name) {
        return None;
    }

    let (
        tick_id,
        tick_cancel,
        cfg,
        start_epoch,
        start_policy_version,
        start_endpoint,
        _flight_guard,
    ) = if let Some(claimed) = pre_claimed {
        if let Some(exp) = expected_epoch {
            if exp != app.current_keep_alive_epoch(vm_name) || exp != claimed.3 {
                return None;
            }
        }
        claimed
    } else {
        let tick_id = app.next_keep_alive_tick_id();
        let tick_cancel = tokio_util::sync::CancellationToken::new();
        let (cfg, start_epoch, start_policy_version, start_endpoint) = {
            let cur_epoch = app.current_keep_alive_epoch(vm_name);
            if let Some(exp) = expected_epoch {
                if exp != cur_epoch {
                    return None;
                }
            }
            let mut map = app.keep_alive.l();
            if app.is_deleting(vm_name) {
                return None;
            }
            let rt = map.entry(vm_name.to_string()).or_insert_with(|| {
                let persisted = cli::get_session_profile(vm_name)
                    .and_then(|p| p.keep_alive)
                    .unwrap_or(KeepAliveConfig {
                        enabled: false,
                        ..KeepAliveConfig::default()
                    });
                let mut r = VmKeepAliveRuntime::from_config(persisted, Some(a.endpoint.clone()));
                r.epoch = cur_epoch;
                r
            });
            if let Some(exp) = expected_epoch {
                if exp != rt.epoch {
                    return None;
                }
            }
            let endpoint_handover = rt
                .endpoint
                .as_deref()
                .is_some_and(|cur| cur != a.endpoint.as_str())
                || rt
                    .last_verified_endpoint
                    .as_deref()
                    .is_some_and(|v| v != a.endpoint.as_str());
            if endpoint_handover {
                rt.reset_verification(Some(a.endpoint.clone()));
                rt.epoch = app.bump_keep_alive_epoch(vm_name);
                if expected_epoch.is_some() {
                    return None;
                }
            } else if rt.endpoint.is_none() {
                rt.endpoint = Some(a.endpoint.clone());
            }
            if require_enabled && !rt.config.enabled {
                return None;
            }
            if rt.in_flight {
                return None;
            }
            if !force {
                if let Some(prev) = rt.last_attempt_instant {
                    if prev.elapsed() < Duration::from_secs(rt.config.interval_seconds) {
                        return None;
                    }
                }
            }
            rt.in_flight = true;
            rt.in_flight_tick_id = Some(tick_id);
            rt.active_cancel = Some(tick_cancel.clone());
            rt.last_attempt_instant = Some(Instant::now());
            (
                rt.config.clone(),
                rt.epoch,
                rt.policy_version,
                a.endpoint.clone(),
            )
        };
        let guard = KeepAliveInFlightGuard {
            app: app.clone(),
            vm_name: vm_name.to_string(),
            tick_id,
        };
        (
            tick_id,
            tick_cancel,
            cfg,
            start_epoch,
            start_policy_version,
            start_endpoint,
            guard,
        )
    };

    let is_tick_still_valid = |app: &AppState| -> bool {
        if app.is_deleting(vm_name) {
            return false;
        }
        if app.current_keep_alive_epoch(vm_name) != start_epoch {
            return false;
        }
        let map = app.keep_alive.l();
        let Some(rt) = map.get(vm_name) else {
            return false;
        };
        rt.in_flight_tick_id == Some(tick_id)
            && rt.epoch == start_epoch
            && rt.policy_version == start_policy_version
            && rt.endpoint.as_deref() == Some(start_endpoint.as_str())
            && (!require_enabled || rt.config.enabled)
    };

    if tick_cancel.is_cancelled() || !is_tick_still_valid(app) {
        return None;
    }

    let now_ts = chrono::Utc::now().timestamp();

    // Stage 1: 运行时连接检查（也提供内核忙闲信息；若被暂停/删除/换棒则立即取消并中止）
    let mut cur_assignment = a.clone();
    let kernels_fut = async {
        let mut k = app.client.list_kernels(&cur_assignment).await;
        if k.as_ref().is_err_and(crate::colab::is_proxy_auth_error) {
            if let Ok(assignments) = app.client.list_assignments().await {
                if let Some(fresh) = assignments
                    .into_iter()
                    .find(|item| item.endpoint == cur_assignment.endpoint)
                {
                    cur_assignment = fresh;
                    k = app.client.list_kernels(&cur_assignment).await;
                }
            }
        }
        (cur_assignment, k)
    };

    let (cur_assignment, kernels) = tokio::select! {
        _ = tick_cancel.cancelled() => return None,
        res = kernels_fut => res,
    };

    if tick_cancel.is_cancelled() || !is_tick_still_valid(app) {
        return None;
    }

    let tunnel_ok = kernels.is_ok();
    let mut err_parts = Vec::new();
    let local_busy = cfg.kernel_ping
        && (app.relay(vm_name).map(|h| h.is_running()).unwrap_or(false)
            || app.busy.l().contains_key(vm_name)
            || app
                .active_chats
                .l()
                .values()
                .any(|e| e.vm_name.as_deref() == Some(vm_name)));
    let kernel_ping_state = match kernels {
        Err(err) => {
            err_parts.push(format!("运行时连接检查失败: {err:#}"));
            if cfg.kernel_ping {
                "error"
            } else {
                "disabled"
            }
        }
        Ok(_) if !cfg.kernel_ping => "disabled",
        Ok(ref kernels)
            if local_busy
                || kernels.iter().any(|k| {
                    matches!(
                        k.execution_state.trim().to_ascii_lowercase().as_str(),
                        "busy" | "starting"
                    )
                }) =>
        {
            "skipped_busy"
        }
        Ok(ref kernels) => {
            let is_live_idle_kernel = |k: &crate::colab::JupyterKernelInfo| -> bool {
                !k.id.trim().is_empty()
                    && !matches!(
                        k.execution_state.trim().to_ascii_lowercase().as_str(),
                        "busy" | "starting" | "dead" | "terminating" | "restarting"
                    )
            };
            let hint_live = kernel_hint
                .as_deref()
                .map(str::trim)
                .filter(|h| !h.is_empty())
                .and_then(|h| kernels.iter().find(|k| is_live_idle_kernel(k) && k.id.trim() == h))
                .map(|k| k.id.clone());
            let idle_kid = hint_live.or_else(|| {
                kernels
                    .iter()
                    .find(|k| is_live_idle_kernel(k))
                    .map(|k| k.id.clone())
            });
            match idle_kid {
                None => "skipped_no_kernel",
                Some(kid) => {
                    match execute_remote_kernel_ping(
                        &cur_assignment,
                        &kid,
                        Some(tick_cancel.clone()),
                    )
                    .await
                    {
                        Ok(KernelPingOutcome::Ok) => "ok",
                        Ok(KernelPingOutcome::NoKernel) => "skipped_no_kernel",
                        Err(err) => {
                            if tick_cancel.is_cancelled() {
                                return None;
                            }
                            err_parts.push(err);
                            "error"
                        }
                    }
                }
            }
        }
    }
    .to_string();

    if tick_cancel.is_cancelled() || !is_tick_still_valid(app) {
        return None;
    }

    let overall_ok = tunnel_ok && kernel_ping_state != "error";
    let status = {
        if app.is_deleting(vm_name) || app.current_keep_alive_epoch(vm_name) != start_epoch {
            return None;
        }
        let mut map = app.keep_alive.l();
        let rt = map.get_mut(vm_name)?;
        if rt.in_flight_tick_id != Some(tick_id)
            || rt.epoch != start_epoch
            || rt.policy_version != start_policy_version
            || rt.endpoint.as_deref() != Some(start_endpoint.as_str())
            || (require_enabled && !rt.config.enabled)
        {
            return None;
        }
        rt.in_flight = false;
        rt.in_flight_tick_id = None;
        rt.active_cancel = None;
        rt.last_attempt_at = Some(now_ts);
        rt.tunnel_ok = tunnel_ok;
        rt.kernel_ping_state = kernel_ping_state.clone();
        if overall_ok {
            rt.last_ok_at = Some(now_ts);
            rt.last_verified_endpoint = Some(start_endpoint);
            rt.success_count = rt.success_count.saturating_add(1);
            rt.consecutive_failures = 0;
            rt.last_error = None;
        } else {
            rt.consecutive_failures = rt.consecutive_failures.saturating_add(1);
            rt.last_error = Some(err_parts.join("；"));
        }

        if !require_enabled && !rt.config.enabled {
            let mut probe_status = rt.to_status();
            probe_status.enabled = false;
            probe_status.kernel_ping_state = if !rt.config.kernel_ping {
                "disabled".to_string()
            } else {
                kernel_ping_state
            };
            probe_status.last_error = if overall_ok {
                None
            } else {
                rt.last_error.clone()
            };
            probe_status.state = if !overall_ok {
                "error".to_string()
            } else {
                "paused".to_string()
            };
            probe_status
        } else {
            rt.to_status()
        }
    };
    Some(status)
}

#[cfg(test)]
pub(crate) async fn dispatch_keep_alive_scheduler_pass(app: &Arc<AppState>) -> usize {
    dispatch_keep_alive_scheduler_pass_with(app, |_| async {}).await
}

async fn dispatch_keep_alive_scheduler_pass_with<F, Fut>(
    app: &Arc<AppState>,
    on_tick_done: F,
) -> usize
where
    F: Fn(Arc<AppState>) -> Fut + Clone + Send + Sync + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let ov_vms = {
        let ov = app.overview.read().await;
        ov.vms
            .iter()
            .filter(|v| v.on_server && v.endpoint.is_some())
            .map(|v| (v.name.clone(), v.endpoint.clone().unwrap(), v.kernel_id.clone()))
            .collect::<Vec<_>>()
    };
    if ov_vms.is_empty() {
        return 0;
    }

    let cached_assignments = app
        .last_assignments
        .l()
        .as_ref()
        .map(|(list, _, _)| list.clone())
        .unwrap_or_default();
    let assign_map: HashMap<String, Assignment> = cached_assignments
        .into_iter()
        .map(|a| (a.endpoint.clone(), a))
        .collect();
    if assign_map.is_empty() {
        return 0;
    }

    let mut dispatched = 0usize;
    for (vm_name, ep, kid) in ov_vms {
        if app.is_deleting(&vm_name) {
            continue;
        }
        let Some(a) = assign_map.get(&ep).cloned() else {
            continue;
        };
        let claimed = {
            let cur_epoch = app.current_keep_alive_epoch(&vm_name);
            let mut map = app.keep_alive.l();
            match map.get_mut(&vm_name) {
                Some(rt) => {
                    let handover = rt.endpoint.as_deref().is_some_and(|c| c != ep.as_str())
                        || rt
                            .last_verified_endpoint
                            .as_deref()
                            .is_some_and(|v| v != ep.as_str());
                    if handover {
                        rt.reset_verification(Some(ep.clone()));
                        rt.epoch = app.bump_keep_alive_epoch(&vm_name);
                    } else if rt.endpoint.is_none() {
                        rt.endpoint = Some(ep.clone());
                    }
                    let due = rt.config.enabled
                        && !rt.in_flight
                        && rt
                            .last_attempt_instant
                            .map(|t| t.elapsed() >= Duration::from_secs(rt.config.interval_seconds))
                            .unwrap_or(true);
                    if (due && rt.epoch == cur_epoch) || (due && handover) {
                        let tick_id = app.next_keep_alive_tick_id();
                        let tick_cancel = tokio_util::sync::CancellationToken::new();
                        rt.in_flight = true;
                        rt.in_flight_tick_id = Some(tick_id);
                        rt.active_cancel = Some(tick_cancel.clone());
                        rt.last_attempt_instant = Some(Instant::now());
                        Some((
                            tick_id,
                            tick_cancel,
                            rt.config.clone(),
                            rt.epoch,
                            rt.policy_version,
                            ep.clone(),
                        ))
                    } else {
                        None
                    }
                }
                None => None,
            }
        };

        if let Some((tick_id, tick_cancel, cfg, start_epoch, start_pv, start_ep)) = claimed {
            dispatched += 1;
            let guard = KeepAliveInFlightGuard {
                app: app.clone(),
                vm_name: vm_name.clone(),
                tick_id,
            };
            let app_c = app.clone();
            let on_done_c = on_tick_done.clone();
            tokio::spawn(async move {
                let res = run_keep_alive_tick_inner(
                    &app_c,
                    &vm_name,
                    &a,
                    kid,
                    false,
                    true,
                    Some(start_epoch),
                    Some((
                        tick_id,
                        tick_cancel,
                        cfg,
                        start_epoch,
                        start_pv,
                        start_ep,
                        guard,
                    )),
                )
                .await;
                if res.is_some() {
                    on_done_c(app_c).await;
                }
            });
        }
    }
    dispatched
}

fn record_profile_persistence_result(app: &AppState, name: &str, res: anyhow::Result<()>) {
    let mut map = app.keep_alive.l();
    if let Some(rt) = map.get_mut(name) {
        match res {
            Ok(()) => {
                rt.profile_saved = true;
                rt.persistence_error = None;
            }
            Err(e) => {
                rt.profile_saved = false;
                rt.persistence_error = Some(format!(
                    "保活/会话档案持久化失败（重启后可能失效）: {e:#}"
                ));
            }
        }
    }
}

fn persist_relay_profile_before_alloc(
    app: &AppState,
    name: &str,
    had_ws_entry: bool,
    session_profile: &cli::SessionProfile,
) -> Result<(), String> {
    if let Err(e) = cli::save_session_profile(name, session_profile) {
        if !had_ws_entry && !cli::read_sessions().contains_key(name) {
            cli::remove_workspace_entry(name);
        }
        app.invalidate_keep_alive(name, true);
        app.release_busy(name, "creating");
        return Err(format!(
            "保存接力会话档案失败，已取消创建（未申请云端实例）: {e:#}"
        ));
    }
    Ok(())
}

pub(crate) async fn set_vm_keep_alive_inner(
    app: &Arc<AppState>,
    name: &str,
    config: KeepAliveConfig,
) -> Result<KeepAliveStatus, String> {
    let clean_name = name.trim();
    if clean_name.is_empty() {
        return Err("实例名称不能为空".into());
    }
    let validated = config.validate()?;
    cli::save_session_keep_alive(clean_name, &validated)
        .map_err(|e| format!("保存实例 [{clean_name}] 保活配置失败: {e:#}"))?;

    let status = {
        let mut map = app.keep_alive.l();
        let entry = map
            .entry(clean_name.to_string())
            .or_insert_with(|| VmKeepAliveRuntime::from_config(validated.clone(), None));
        let was_enabled = entry.config.enabled;
        let policy_changed = entry.config != validated;
        entry.config = validated.clone();
        entry.profile_saved = true;
        entry.persistence_error = None;
        if policy_changed {
            entry.policy_version = entry.policy_version.saturating_add(1);
        }
        if !validated.enabled {
            entry.epoch = app.bump_keep_alive_epoch(clean_name);
            if let Some(tok) = entry.active_cancel.take() {
                tok.cancel();
            }
            entry.in_flight = false;
            entry.in_flight_tick_id = None;
            entry.consecutive_failures = 0;
            entry.last_error = None;
            entry.kernel_ping_state = "disabled".into();
        } else if policy_changed || !was_enabled {
            let cur_ep = entry.endpoint.clone();
            entry.reset_verification(cur_ep);
            entry.epoch = app.bump_keep_alive_epoch(clean_name);
        } else if entry.epoch == 0 {
            entry.epoch = app.current_keep_alive_epoch(clean_name);
        }
        entry.to_status()
    };
    Ok(status)
}

#[tauri::command]
async fn set_vm_keep_alive(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    config: KeepAliveConfig,
) -> Result<KeepAliveStatus, String> {
    let app = state.inner().clone();
    let status = set_vm_keep_alive_inner(&app, &name, config).await?;
    if status.enabled {
        let expected_ep = app.current_keep_alive_epoch(name.trim());
        if let Some(a) = app.find_assignment_by_name(&name).await {
            let kid = app
                .overview
                .read()
                .await
                .vms
                .iter()
                .find(|v| v.name == name)
                .and_then(|v| v.kernel_id.clone());
            let app_bg = app.clone();
            let handle_bg = handle.clone();
            let name_bg = name.clone();
            tauri::async_runtime::spawn(async move {
                let _ = run_keep_alive_tick_for_vm(
                    &app_bg,
                    &name_bg,
                    &a,
                    kid,
                    true,
                    true,
                    Some(expected_ep),
                )
                .await;
                let ov = refresh_overview_opts(&app_bg, false, true).await;
                emit_overview(&handle_bg, &ov);
            });
        }
    }
    spawn_refresh(&app, &handle);
    Ok(status)
}

pub(crate) async fn ping_vm_keep_alive_inner(
    app: &Arc<AppState>,
    name: &str,
) -> Result<KeepAliveStatus, String> {
    let a = app
        .find_assignment_by_name(name)
        .await
        .ok_or_else(|| format!("服务端未找到在线实例 [{name}]"))?;
    let kid = app
        .overview
        .read()
        .await
        .vms
        .iter()
        .find(|v| v.name == name)
        .and_then(|v| v.kernel_id.clone());
    let expected_ep = app.current_keep_alive_epoch(name);
    run_keep_alive_tick_for_vm(app, name, &a, kid, true, false, Some(expected_ep))
        .await
        .ok_or_else(|| format!("实例 [{name}] 正在执行保活心跳或状态已变更，请稍候"))
}

#[tauri::command]
async fn ping_vm_keep_alive(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
) -> Result<KeepAliveStatus, String> {
    let app = state.inner().clone();
    let status = ping_vm_keep_alive_inner(&app, &name).await?;
    spawn_refresh(&app, &handle);
    Ok(status)
}

#[derive(Deserialize, Clone, Default, Debug)]
pub struct CreateVmRequest {
    pub name: Option<String>,
    pub accelerator: String,
    #[serde(default)]
    pub high_mem: bool,
    #[serde(default)]
    pub auth_mode: Option<String>,
    #[serde(default)]
    pub workspace_path: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub boot_mode: Option<String>,
    #[serde(default)]
    pub auto_relay: bool,
    #[serde(default)]
    pub relay_config: Option<RelayConfig>,
    #[serde(default)]
    pub llm_config: Option<CreateVmLlmConfig>,
    #[serde(default)]
    pub env_config: Option<CreateVmEnvConfig>,
    #[serde(default)]
    pub keep_alive: Option<KeepAliveConfig>,
    #[serde(default)]
    pub grab_config: Option<GrabConfig>,
}

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 40
        && !n.starts_with('-')
        && n.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

pub(crate) fn normalize_llm_engine(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "vllm" => "vllm".into(),
        "ollama" => "ollama".into(),
        "llamacpp" | "llama_cpp" | "llama.cpp" | "gguf" | "llamacpp-cuda" | "llama.cpp-cuda"
        | "llamacpp-cpu" | "llama_cpp_cpu" | "llama.cpp-cpu" => "llama.cpp".into(),
        "transformers" | "hf" => "transformers".into(),
        "builtin" => "builtin".into(),
        other => other.into(),
    }
}

pub(crate) fn is_nvidia_cuda_accelerator(acc: &str) -> bool {
    matches!(
        acc.trim().to_ascii_uppercase().as_str(),
        "T4" | "L4" | "G4" | "A100" | "H100" | "V100" | "P100"
    )
}

pub(crate) fn engine_requires_nvidia_cuda(engine_raw: &str) -> bool {
    let low = engine_raw.trim().to_ascii_lowercase();
    let norm = normalize_llm_engine(&low);
    match norm.as_str() {
        "vllm" | "transformers" => true,
        "llama.cpp" => !low.contains("cpu"),
        _ => low.contains("cuda"),
    }
}

pub(crate) fn validate_llm_hardware_compatibility(
    accelerator: &str,
    engine_raw: &str,
) -> Result<(), String> {
    let acc_norm = {
        let u = accelerator.trim().to_ascii_uppercase();
        if u.is_empty() || u == "NONE" {
            "CPU".to_string()
        } else {
            u
        }
    };
    let low_engine = engine_raw.trim().to_ascii_lowercase();
    let norm_engine = normalize_llm_engine(&low_engine);
    // 显式 CPU 模式版 llama.cpp 仅允许运行在 CPU 或 NVIDIA 实例上，不能误发往 TPU
    if norm_engine == "llama.cpp" && low_engine.contains("cpu") && matches!(acc_norm.as_str(), "V5E1" | "V6E1" | "TPU") {
        return Err(format!(
            "硬件与推理引擎不兼容：当前选择的加速器为 `{acc_norm}`（TPU），不支持 `llama.cpp` 预设。请改选 CPU 或 NVIDIA GPU（如 T4 / L4 / G4 / A100）"
        ));
    }
    if engine_requires_nvidia_cuda(engine_raw) && !is_nvidia_cuda_accelerator(&acc_norm) {
        return Err(format!(
            "硬件与推理引擎不兼容：当前选择的加速器为 `{acc_norm}`（非 NVIDIA CUDA GPU），无法运行依赖 CUDA 的 `{norm_engine}` 预设。请改选 NVIDIA GPU（如 T4 / L4 / G4 / A100），或切换为支持非 CUDA 环境的推理引擎（如 ollama / builtin）"
        ));
    }
    Ok(())
}

fn normalize_llm_slot(slot_opt: Option<&str>, vm_name: &str) -> String {
    let raw = slot_opt.unwrap_or("").trim().to_ascii_lowercase();
    match raw.as_str() {
        "blue" | "green" | "standalone" => raw,
        _ if vm_name.ends_with("-blue") => "blue".into(),
        _ if vm_name.ends_with("-green") => "green".into(),
        _ => "standalone".into(),
    }
}

pub(crate) fn validate_pip_packages(raw: &str) -> Result<Vec<String>, String> {
    let mut pkgs = Vec::new();
    for tok in raw.split_whitespace() {
        if tok.starts_with('-') {
            return Err(format!("pip 包名不允许携带命令行选项 `{tok}`"));
        }
        if tok.len() > 160
            || !tok.chars().all(|c| {
                c.is_ascii_alphanumeric()
                    || matches!(
                        c,
                        '_' | '.' | '-' | '[' | ']' | '=' | '<' | '>' | '!' | '~' | '+' | ','
                    )
            })
        {
            return Err(format!("非法的 pip 包依赖格式 `{tok}`"));
        }
        pkgs.push(tok.to_string());
    }
    Ok(pkgs)
}

fn parse_env_vars(raw: &str) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let body = trimmed
            .strip_prefix("export ")
            .map(str::trim)
            .unwrap_or(trimmed);
        let Some((k, v)) = body.split_once('=') else {
            return Err(format!("环境变量格式应为 KEY=VALUE：`{trimmed}`"));
        };
        let key = k.trim();
        let mut chars = key.chars();
        let valid_key = match chars.next() {
            Some(first) if first.is_ascii_alphabetic() || first == '_' => {
                key.len() <= 64 && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
            }
            _ => false,
        };
        if !valid_key {
            return Err(format!("非法的环境变量名 `{key}`"));
        }
        let upper = key.to_ascii_uppercase();
        if matches!(upper.as_str(), "LD_PRELOAD" | "LD_LIBRARY_PATH" | "PROMPT_COMMAND") {
            return Err(format!("出于安全限制，不允许覆盖系统敏感变量 `{key}`"));
        }
        let mut val = v.trim().to_string();
        if val.len() >= 2
            && ((val.starts_with('"') && val.ends_with('"'))
                || (val.starts_with('\'') && val.ends_with('\'')))
        {
            val = val[1..val.len() - 1].to_string();
        }
        pairs.push((key.to_string(), val));
    }
    Ok(pairs)
}

fn write_if_changed(path: &Path, content: &str) -> bool {
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            return false;
        }
    }
    let unchanged = std::fs::read_to_string(path)
        .map(|cur| cur == content)
        .unwrap_or(false);
    if unchanged {
        return true;
    }
    std::fs::write(path, content).is_ok()
}

/// 确保 5 个内嵌 Python 技能脚本在 `~/.config/colab-cli/skills` 下始终与当前二进制版本一致并落盘：
/// - 正式构建（`not(debug_assertions)`）及任何外部用户工作区一律返回释放自 `include_str!` 的最新内嵌脚本路径，
///   严禁盲目信任用户工作区（如旧版项目目录 `.agents/skills`）中未同步升级的旧脚本；
/// - 仅在 `debug_assertions` 开发态且 `workspace` 恰好为当前编译期源码仓库根目录时，才直接复用源码目录脚本。
fn ensure_skill_scripts(workspace: &Path) -> Option<PathBuf> {
    let root = cli::colab_config_dir().join("skills");
    let bg_dir = root.join("colab-llm-bluegreen").join("scripts");
    let relay_dir = root.join("colab-dl-relay").join("scripts");

    let mgr_path = bg_dir.join("llm_bluegreen_manager.py");
    let gw_path = bg_dir.join("llm_local_gateway.py");
    let sidecar_path = bg_dir.join("llm_metrics_sidecar.py");
    let relay_path = relay_dir.join("dl_relay_runner.py");
    let adapter_path = relay_dir.join("colab_relay_adapter.py");

    let embedded_ok = write_if_changed(&mgr_path, LLM_BLUEGREEN_MANAGER_PY)
        && write_if_changed(&gw_path, LLM_LOCAL_GATEWAY_PY)
        && write_if_changed(&sidecar_path, LLM_METRICS_SIDECAR_PY);
    let _ = write_if_changed(&relay_path, DL_RELAY_RUNNER_PY);
    let _ = write_if_changed(&adapter_path, COLAB_RELAY_ADAPTER_PY);

    #[cfg(debug_assertions)]
    {
        let is_current_dev_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|dev_root| dev_root == workspace)
            .unwrap_or(false);
        if is_current_dev_repo {
            let ws_bg_dir = workspace
                .join(".agents")
                .join("skills")
                .join("colab-llm-bluegreen")
                .join("scripts");
            let ws_mgr = ws_bg_dir.join("llm_bluegreen_manager.py");
            let ws_gw = ws_bg_dir.join("llm_local_gateway.py");
            let ws_sidecar = ws_bg_dir.join("llm_metrics_sidecar.py");
            if ws_mgr.exists() && ws_gw.exists() && ws_sidecar.exists() {
                return Some(ws_mgr);
            }
        }
    }
    #[cfg(not(debug_assertions))]
    let _ = workspace;

    if embedded_ok {
        Some(mgr_path)
    } else {
        None
    }
}

fn ensure_gateway_script_path(workspace: &Path) -> Option<PathBuf> {
    let mgr = ensure_skill_scripts(workspace)?;
    let gw = mgr.with_file_name("llm_local_gateway.py");
    if gw.exists() {
        Some(gw)
    } else {
        None
    }
}

fn has_enabled_env_steps(cfg: &CreateVmEnvConfig) -> bool {
    cfg.sync_rclone
        || cfg.mount_gdrive
        || cfg.install_workspace_requirements
        || cfg
            .pip_packages
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
        || cfg
            .startup_script
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
        || cfg
            .env_vars
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
}

async fn apply_post_boot_env_config(
    app: &Arc<AppState>,
    vm_name: &str,
    workspace_dir: &Path,
    cfg: &CreateVmEnvConfig,
) -> anyhow::Result<()> {
    let pip_list = match cfg.pip_packages.as_deref() {
        Some(raw) if !raw.trim().is_empty() => {
            validate_pip_packages(raw).map_err(|e| anyhow::anyhow!(e))?
        }
        _ => Vec::new(),
    };
    let env_pairs = match cfg.env_vars.as_deref() {
        Some(raw) if !raw.trim().is_empty() => parse_env_vars(raw).map_err(|e| anyhow::anyhow!(e))?,
        _ => Vec::new(),
    };
    let startup_sh = cfg
        .startup_script
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let a = app
        .find_assignment_by_name(vm_name)
        .await
        .ok_or_else(|| anyhow::anyhow!("未找到刚创建的云端实例 [{vm_name}]"))?;

    let need_rclone = cfg.sync_rclone || cfg.mount_gdrive;
    if need_rclone {
        let conf_path = cli::rclone_conf_path();
        let bytes = std::fs::read(&conf_path).with_context(|| {
            format!(
                "未读取到本机 rclone.conf（{}），请先配置 Google Drive",
                conf_path.display()
            )
        })?;
        app.client
            .contents_upload(&a, "content/rclone.conf", &bytes)
            .await
            .context("上传 rclone.conf 到云端失败")?;
    }

    let mut uploaded_reqs = false;
    if cfg.install_workspace_requirements {
        let req_path = workspace_dir.join("requirements.txt");
        if let Ok(raw) = std::fs::read_to_string(&req_path) {
            let has_pkg = raw
                .lines()
                .map(str::trim)
                .any(|l| !l.is_empty() && !l.starts_with('#'));
            if has_pkg {
                app.client
                    .contents_upload(&a, "content/requirements.txt", raw.as_bytes())
                    .await
                    .context("上传 requirements.txt 到云端失败")?;
                uploaded_reqs = true;
            }
        }
    }

    if let Some(ref script_body) = startup_sh {
        let normalized = script_body.replace("\r\n", "\n");
        app.client
            .contents_upload(
                &a,
                "content/.colab_pilot_startup.sh",
                format!("#!/usr/bin/env bash\nset -e\n{normalized}\n").as_bytes(),
            )
            .await
            .context("上传自定义启动脚本失败")?;
    }

    let payload = serde_json::json!({
        "need_rclone": need_rclone,
        "mount_gdrive": cfg.mount_gdrive,
        "uploaded_reqs": uploaded_reqs,
        "pip_list": pip_list,
        "env_pairs": env_pairs,
        "has_startup": startup_sh.is_some(),
    });
    let payload_json = serde_json::to_string(&payload)?;
    let remote_py = format!(
        r##"
import json, os, shlex, shutil, subprocess

cfg = json.loads({payload_literal})
os.chdir("/content")

env_map = dict(cfg.get("env_pairs") or [])
if env_map:
    with open("/content/.colab_pilot_env.json", "w", encoding="utf-8") as f:
        json.dump(env_map, f, ensure_ascii=False, indent=2)
    bashrc = os.path.expanduser("~/.bashrc")
    lines = []
    for k, v in env_map.items():
        os.environ[k] = v
        lines.append(f"export {{k}}={{shlex.quote(v)}}")
    block = "\n# >>> colab-pilot env >>>\n" + "\n".join(lines) + "\n# <<< colab-pilot env <<<\n"
    cur = open(bashrc, "r", encoding="utf-8", errors="ignore").read() if os.path.exists(bashrc) else ""
    if "# >>> colab-pilot env >>>" not in cur:
        with open(bashrc, "a", encoding="utf-8") as f:
            f.write(block)

if cfg.get("need_rclone"):
    if os.path.exists("/content/rclone.conf"):
        os.chmod("/content/rclone.conf", 0o600)
        rc_dir = os.path.expanduser("~/.config/rclone")
        os.makedirs(rc_dir, exist_ok=True)
        shutil.copy2("/content/rclone.conf", os.path.join(rc_dir, "rclone.conf"))
    if not shutil.which("rclone"):
        subprocess.run("curl -fsSL https://rclone.org/install.sh | sudo bash", shell=True, check=False)

if cfg.get("mount_gdrive") and shutil.which("rclone") and os.path.exists("/content/rclone.conf"):
    os.makedirs("/content/drive", exist_ok=True)
    subprocess.Popen(
        [
            "rclone", "--config", "/content/rclone.conf",
            "mount", "gdrive:", "/content/drive",
            "--vfs-cache-mode", "writes", "--allow-non-empty",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

use_uv = shutil.which("uv") is not None
if cfg.get("uploaded_reqs") and os.path.exists("/content/requirements.txt"):
    cmd = ["uv", "pip", "install", "--system", "-r", "/content/requirements.txt"] if use_uv else ["pip", "install", "-q", "-r", "/content/requirements.txt"]
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"安装 requirements.txt 依赖失败 (exit={{r.returncode}})")

pkgs = cfg.get("pip_list") or []
if pkgs:
    cmd = (["uv", "pip", "install", "--system"] + pkgs) if use_uv else (["pip", "install", "-q"] + pkgs)
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"安装自定义 pip 包失败 (exit={{r.returncode}})")

if cfg.get("has_startup") and os.path.exists("/content/.colab_pilot_startup.sh"):
    os.chmod("/content/.colab_pilot_startup.sh", 0o700)
    r = subprocess.run(["bash", "/content/.colab_pilot_startup.sh"], env=os.environ.copy(), check=False)
    if r.returncode != 0:
        raise RuntimeError(f"执行启动脚本失败 (exit={{r.returncode}})")

print("ENV_INIT_OK", flush=True)
"##,
        payload_literal = serde_json::to_string(&payload_json)?
    );

    let exec_args = cli::colab_args_for_session(vm_name, &s(&["exec", "-s", vm_name, "--timeout", "600"]));
    let out = cli::run_streaming(
        &cli::colab_bin(),
        &exec_args,
        Some(remote_py),
        Duration::from_secs(630),
        None,
        None,
        |_| {},
    )
    .await?;

    if out.code != 0 || !out.output.contains("ENV_INIT_OK") {
        let cleaned = cli::strip_colab_cli_banners(&out.output);
        anyhow::bail!(
            "初始化环境配置失败：{}",
            cli::last_meaningful_cli_line(&cleaned, "执行返回非零状态")
        );
    }
    Ok(())
}

fn spawn_llm_deployment_task(
    app: Arc<AppState>,
    handle: AppHandle,
    vm_name: String,
    accelerator: String,
    high_mem: bool,
    workspace_dir: PathBuf,
    config: CreateVmLlmConfig,
) -> Result<LlmApiAccessInfo, String> {
    let norm_engine = normalize_llm_engine(&config.engine);
    let slot = normalize_llm_slot(config.slot.as_deref(), &vm_name);
    let model_name = {
        let m = config.model_name.trim();
        if m.is_empty() {
            "Qwen/Qwen2.5-7B-Instruct".to_string()
        } else {
            m.to_string()
        }
    };
    let custom_key = config
        .custom_api_key
        .as_deref()
        .map(str::trim)
        .filter(|s| s.len() >= 8)
        .map(|s| s.to_string());
    let gdrive_model = config
        .gdrive_model_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let preferred_port = match slot.as_str() {
        "blue" => 18081,
        "green" => 18082,
        _ => 0,
    };
    let api_info = if custom_key.is_some() {
        cli::rotate_llm_api_key_entry(
            &vm_name,
            &slot,
            &norm_engine,
            custom_key.as_deref(),
            preferred_port,
            8000,
            &model_name,
        )
        .map_err(|e| format!("持久化写入自定义 API Key 失败：{e:#}"))?
    } else {
        cli::resolve_or_create_llm_api_info(
            &vm_name,
            &slot,
            &norm_engine,
            preferred_port,
            8000,
            &model_name,
        )
    };

    app.set_deploy_error(&vm_name, None);
    app.set_busy(&vm_name, Some("deploying_llm"));
    spawn_refresh(&app, &handle);

    let effective_hm = high_mem && relay::supports_high_mem(&accelerator);
    tokio::spawn(async move {
        let Some(mgr_script) = ensure_skill_scripts(&workspace_dir) else {
            app.release_busy(&vm_name, "deploying_llm");
            app.set_deploy_error(
                &vm_name,
                Some("无法释放内置大模型蓝绿管理脚本 llm_bluegreen_manager.py".into()),
            );
            spawn_refresh(&app, &handle);
            return;
        };

        let acc_arg = {
            let u = accelerator.trim().to_uppercase();
            if u.is_empty() || u == "NONE" {
                "CPU".to_string()
            } else {
                u
            }
        };
        let mut args = vec![
            "-u".to_string(),
            mgr_script.to_string_lossy().to_string(),
            "deploy".to_string(),
            "--session-name".to_string(),
            vm_name.clone(),
            "--slot".to_string(),
            slot.clone(),
            "--workspace".to_string(),
            workspace_dir.to_string_lossy().to_string(),
            "--gpu".to_string(),
            acc_arg,
            if effective_hm {
                "--high-mem".to_string()
            } else {
                "--no-high-mem".to_string()
            },
            "--engine".to_string(),
            norm_engine,
            "--model-name".to_string(),
            model_name,
            "--keep-vm-on-fail".to_string(),
            "--gateway-port".to_string(),
            "18080".to_string(),
        ];
        if let Some(ref gp) = gdrive_model {
            args.push("--gdrive-model-path".into());
            args.push(gp.clone());
        }
        if let Some(ref ck) = custom_key {
            args.push("--api-key".into());
            args.push(ck.clone());
        }

        let py = cli::colab_python_bin();
        let res = cli::run_streaming(
            &py,
            &args,
            None,
            Duration::from_secs(2700),
            None,
            None,
            |_| {},
        )
        .await;

        app.release_busy(&vm_name, "deploying_llm");
        match res {
            Ok(o) if o.code == 0 => {
                app.set_deploy_error(&vm_name, None);
                spawn_local_gateway_async(workspace_dir).await;
                if let Some(a) = app.find_assignment_by_name(&vm_name).await {
                    let bg = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state());
                    let entry = fetch_telemetry(&app, &a, Some(&vm_name), bg.as_ref()).await;
                    if entry.data.is_some() {
                        app.telemetry.l().insert(a.endpoint.clone(), entry);
                    }
                }
            }
            Ok(o) => {
                let cleaned = cli::strip_colab_cli_banners(&o.output);
                let msg = if o.timed_out {
                    "云端大模型拉取与预热超时（45 分钟），虚拟机已保留，请通过 SSH 查看 /content/engine.log".to_string()
                } else {
                    cli::last_meaningful_cli_line(&cleaned, "大模型引擎启动或预热失败").to_string()
                };
                app.set_deploy_error(&vm_name, Some(msg));
            }
            Err(e) => {
                let cleaned = cli::strip_colab_cli_banners(&format!("{e:#}"));
                app.set_deploy_error(&vm_name, Some(cleaned));
            }
        }
        let ov = refresh_overview_opts(&app, false, true).await;
        emit_overview(&handle, &ov);
    });

    Ok(api_info)
}

/// 归一化接力配置（保留 CPU 规格、自动按实例名隔离裸 `gdrive:colab_relay` 目录、校验单棒时长 0.1~23.5h，并拦截运行中目录冲突）
/// 返回 `Ok(was_auto_isolated)`，供接力监工执行旧版根目录断点检测与迁移提示（#80）。
pub(crate) fn prepare_relay_config_for_vm(
    app: &AppState,
    vm_name: &str,
    cfg: &mut RelayConfig,
    override_gpu: Option<&str>,
    override_high_mem: Option<bool>,
) -> Result<bool, String> {
    let raw_gpu = override_gpu.unwrap_or(&cfg.gpu_type).trim();
    let gpu_upper = raw_gpu.to_ascii_uppercase();
    cfg.gpu_type = if gpu_upper.is_empty() || gpu_upper == "NONE" {
        "CPU".into()
    } else {
        gpu_upper
    };
    cfg.high_mem = override_high_mem.unwrap_or(cfg.high_mem) && relay::supports_high_mem(&cfg.gpu_type);
    let was_auto_isolated =
        relay::normalize_gdrive_dir(&cfg.gdrive_dir) == relay::DEFAULT_BARE_GDRIVE_DIR;
    cfg.gdrive_dir = relay::resolve_isolated_gdrive_dir(&cfg.gdrive_dir, vm_name);
    relay::validate_relay_config(cfg).map_err(|e| e.to_string())?;

    let target_gd = relay::normalize_gdrive_dir(&cfg.gdrive_dir);
    let relays_guard = app.relays.l();
    for (other_name, other_h) in relays_guard.iter() {
        if other_name != vm_name && other_h.is_running() {
            let other_gd = relay::normalize_gdrive_dir(&other_h.status.l().config.gdrive_dir);
            if other_gd == target_gd {
                return Err(format!(
                    "云盘接力目录 `{target_gd}` 正被另一运行中的接力任务 [{other_name}] 占用，请修改云盘目录或先停止 [{other_name}]，避免断点与完成标记互相覆盖"
                ));
            }
        }
    }
    Ok(was_auto_isolated)
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedCreateVm {
    pub name: String,
    pub boot_mode: String,
    pub auth_mode: Option<String>,
    pub effective_high_mem: bool,
    pub effective_ws_dir: PathBuf,
    pub ws_display_name: String,
    pub desc_clean: Option<String>,
    pub validated_relay_cfg: Option<RelayConfig>,
    pub relay_auto_isolated: bool,
    pub validated_llm_cfg: Option<CreateVmLlmConfig>,
    pub validated_keep_alive: KeepAliveConfig,
    pub validated_grab_cfg: GrabConfig,
}

pub(crate) fn validate_and_prepare_create_vm(
    app: &AppState,
    req: &CreateVmRequest,
) -> Result<PreparedCreateVm, String> {
    let name = req
        .name
        .as_deref()
        .map(str::trim)
        .filter(|n| !n.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| format!("vm-{}", &uuid::Uuid::new_v4().simple().to_string()[..6]));
    if !valid_name(&name) {
        return Err("名称只能包含字母、数字、- 和 _，且不能以 - 开头（不超过 40 个字符）".into());
    }
    if name.starts_with("orphan-") {
        return Err("实例名称不能以保留前缀 `orphan-` 开头".into());
    }

    let auth_mode = cli::normalize_auth_mode(req.auth_mode.as_deref());
    let env = app.env_status();
    if !env.token_ok {
        if auth_mode.as_deref() == Some("adc") || env.adc_ok {
            return Err(
                "检测到本机仅配置了 Google ADC 凭证但缺少有效的 Colab OAuth 登录令牌：仪表盘监控与 `/tun/m/assignments` 停费核验依赖 OAuth 授权，为防止创建出持续计费却无法监控和核验停费的实例，请先在终端执行 `colab auth` 完成 OAuth 授权后再创建实例".into(),
            );
        }
        return Err(
            "未检测到有效的 Colab OAuth 登录凭证，请先在终端执行 `colab auth`（可通过 `colab status` 核对状态）完成 Google 授权".into(),
        );
    }
    if auth_mode.as_deref() == Some("adc") && !env.adc_ok {
        return Err(
            "选择了 ADC 凭证模式，但未检测到 Google Application Default Credentials 文件（application_default_credentials.json），请先配置 ADC 或切换为 OAuth 2.0 模式".into(),
        );
    }

    let boot_mode = req
        .boot_mode
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            if req.auto_relay {
                "relay".into()
            } else if req.llm_config.is_some() {
                "llm".into()
            } else {
                "general".into()
            }
        });

    let effective_high_mem = req.high_mem && relay::supports_high_mem(&req.accelerator);

    // 预先校验附加环境与模式参数，避免分配计费实例后因本地配置错误中断（#67：通用模式未勾选 rclone 时绝不强制要求 rclone.conf）
    if let Some(ref env_cfg) = req.env_config {
        if let Some(ref raw_pkgs) = env_cfg.pip_packages {
            if !raw_pkgs.trim().is_empty() {
                validate_pip_packages(raw_pkgs)?;
            }
        }
        if let Some(ref raw_env) = env_cfg.env_vars {
            if !raw_env.trim().is_empty() {
                parse_env_vars(raw_env)?;
            }
        }
        if (env_cfg.sync_rclone || env_cfg.mount_gdrive) && !cli::rclone_conf_path().exists() {
            return Err(
                "未检测到本机 rclone.conf，请先完成 Google Drive 授权后再勾选 rclone 同步/挂载"
                    .into(),
            );
        }
    }

    let (validated_relay_cfg, relay_auto_isolated) = if boot_mode == "relay" {
        if !cli::rclone_conf_path().exists() {
            return Err("未检测到 rclone.conf，请先完成 Google Drive 授权后再开启自动接力".into());
        }
        let mut cfg = req.relay_config.clone().unwrap_or_default();
        let was_iso = prepare_relay_config_for_vm(
            app,
            &name,
            &mut cfg,
            Some(&req.accelerator),
            Some(effective_high_mem),
        )?;
        (Some(cfg), was_iso)
    } else {
        (None, false)
    };

    let validated_llm_cfg = if boot_mode == "llm" {
        let cfg = req
            .llm_config
            .clone()
            .ok_or_else(|| "大模型部署模式缺少模型配置参数".to_string())?;
        validate_llm_hardware_compatibility(&req.accelerator, &cfg.engine)?;
        let eng = normalize_llm_engine(&cfg.engine);
        if eng != "builtin" && cfg.model_name.trim().is_empty() {
            return Err("请填写要部署的大语言模型名称（如 Qwen/Qwen2.5-7B-Instruct）".into());
        }
        if let Some(ref ck) = cfg.custom_api_key {
            let trimmed = ck.trim();
            if !trimmed.is_empty() {
                if trimmed.len() < 8 {
                    return Err("自定义 API Key 长度至少需要 8 个字符".into());
                }
                if trimmed.chars().any(|c| c.is_control() || c.is_whitespace()) {
                    return Err("自定义 API Key 不能包含空格或控制字符".into());
                }
            }
        }
        if cfg
            .gdrive_model_path
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false)
            && !cli::rclone_conf_path().exists()
        {
            return Err("指定了 Google Drive 模型路径，但本机未检测到 rclone.conf".into());
        }
        Some(cfg)
    } else {
        None
    };

    let custom_ws_path = req
        .workspace_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(PathBuf::from);
    if let Some(ref p) = custom_ws_path {
        if !p.is_dir() {
            return Err(format!("指定的工作区目录不存在或不是文件夹：{}", p.display()));
        }
    }
    let effective_ws_dir = custom_ws_path
        .clone()
        .unwrap_or_else(|| app.workspace.clone());
    let ws_display_name = effective_ws_dir
        .file_name()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("default")
        .to_string();
    let desc_clean = req
        .description
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    let relay_active = app
        .relays
        .l()
        .get(&name)
        .map(|h| h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst))
        .unwrap_or(false);
    if cli::read_sessions().contains_key(&name)
        || app.busy.l().contains_key(&name)
        || app.creating_vms.l().contains_key(&name)
        || relay_active
    {
        return Err(format!("名称 [{name}] 已被占用"));
    }
    if !cli::colab_bin().exists() {
        return Err("未找到 colab-cli，可执行文件应位于 ~/.local/bin/colab".into());
    }

    // 只要创建实例，统一校验并配置实例保活策略（缺省默认开启：每 90 秒运行时连接检查 + 忙时避让 Kernel 心跳）
    let validated_keep_alive = req
        .keep_alive
        .clone()
        .unwrap_or_default()
        .validate()?;

    let validated_grab_cfg = relay::validate_grab_config(
        &req.grab_config.clone().unwrap_or_default(),
    )
    .map_err(|e| e.to_string())?;

    Ok(PreparedCreateVm {
        name,
        boot_mode,
        auth_mode,
        effective_high_mem,
        effective_ws_dir,
        ws_display_name,
        desc_clean,
        validated_relay_cfg,
        relay_auto_isolated,
        validated_llm_cfg,
        validated_keep_alive,
        validated_grab_cfg,
    })
}

pub(crate) async fn cancel_create_vm_inner(
    app: &Arc<AppState>,
    name: &str,
) -> Result<TerminateReceipt, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() || !valid_name(trimmed) {
        return Err("非法的实例名称".into());
    }
    let has_creating = app.creating_vms.l().contains_key(trimmed);
    let has_relay = app
        .relay(trimmed)
        .map(|h| h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst))
        .unwrap_or(false);
    let busy_creating = matches!(
        app.busy.l().get(trimmed).map(String::as_str),
        Some("creating" | "probing" | "initializing_env")
    );
    if !has_creating && !has_relay && !busy_creating {
        return Ok(TerminateReceipt {
            name: trimmed.into(),
            verified_unassigned: true,
            steps: vec!["抢卡任务已结束或不存在".into()],
            ..Default::default()
        });
    }
    let receipt = hard_terminate(app, trimmed, None, true).await;
    app.release_busy(trimmed, "creating");
    app.release_busy(trimmed, "probing");
    app.release_busy(trimmed, "initializing_env");
    if receipt.verified_unassigned && !cli::read_sessions().contains_key(trimmed) {
        if let Some(h) = app.relay(trimmed) {
            if !h.is_running() && h.status.l().round <= 1 && !h.status.l().training_completed {
                app.relays.l().remove(trimmed);
            }
        }
    }
    Ok(receipt)
}

#[tauri::command]
async fn cancel_create_vm(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
) -> Result<TerminateReceipt, String> {
    let app = state.inner().clone();
    let receipt = cancel_create_vm_inner(&app, &name).await?;
    spawn_refresh(&app, &handle);
    Ok(receipt)
}

#[tauri::command]
async fn create_vm(
    handle: AppHandle,
    state: AppStateRef<'_>,
    req: CreateVmRequest,
) -> Result<String, String> {
    let app = state.inner().clone();
    let prep = validate_and_prepare_create_vm(&app, &req)?;
    let name = prep.name.clone();
    let effective_high_mem = prep.effective_high_mem;
    let effective_ws_dir = prep.effective_ws_dir.clone();
    let ws_display_name = prep.ws_display_name.clone();
    let desc_clean = prep.desc_clean.clone();

    if !app.try_claim_busy(&name, "creating") {
        return Err(format!("名称 [{name}] 已被占用"));
    }
    app.set_deploy_error(&name, None);
    let ep_init = app.bump_keep_alive_epoch(&name);
    {
        let mut map = app.keep_alive.l();
        if let Some(old) = map.remove(&name) {
            if let Some(tok) = old.active_cancel {
                tok.cancel();
            }
        }
        let mut rt = VmKeepAliveRuntime::from_config(prep.validated_keep_alive.clone(), None);
        rt.epoch = ep_init;
        map.insert(name.clone(), rt);
    }
    let had_ws_entry = cli::has_workspace_entry(&name);
    cli::register_workspace_entry(
        &name,
        &ws_display_name,
        Some(&effective_ws_dir.to_string_lossy()),
        desc_clean.as_deref(),
    );

    let norm_acc = {
        let u = req.accelerator.trim().to_ascii_uppercase();
        if u.is_empty() || u == "NONE" {
            "CPU".to_string()
        } else {
            u
        }
    };

    // 接力训练模式：直接移交后台接力监工执行网盘预检与第 1 棒受控抢卡，弹窗秒关并在卡片/接力面板实时展示抢卡进度
    if let Some(cfg) = prep.validated_relay_cfg {
        let session_profile = cli::SessionProfile {
            boot_mode: prep.boot_mode.clone(),
            accelerator: norm_acc,
            high_mem: effective_high_mem,
            workspace_path: Some(effective_ws_dir.to_string_lossy().to_string()),
            auth_mode: prep.auth_mode.clone(),
            llm_config: None,
            relay_config: Some(cfg.clone()),
            keep_alive: Some(prep.validated_keep_alive.clone()),
            grab_config: Some(prep.validated_grab_cfg.clone()),
            updated_at: None,
        };
        if let Err(err_msg) =
            persist_relay_profile_before_alloc(&app, &name, had_ws_entry, &session_profile)
        {
            spawn_refresh(&app, &handle);
            return Err(err_msg);
        }
        app.release_busy(&name, "creating");
        start_relay_inner(
            &app,
            &name,
            cfg,
            effective_ws_dir,
            prep.auth_mode.clone(),
            prep.relay_auto_isolated,
            None,
        );
        spawn_refresh(&app, &handle);
        return Ok(name);
    }

    let cancel = tokio_util::sync::CancellationToken::new();
    let child_pid = Arc::new(Mutex::new(None));
    let pre_prov_eps = Arc::new(Mutex::new(std::collections::HashSet::new()));
    let pre_prov_invalid = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let grab_slot = Arc::new(Mutex::new(None));

    app.creating_vms.l().insert(
        name.clone(),
        CreatingVmMeta {
            accelerator: req.accelerator.clone(),
            high_mem: effective_high_mem,
            workspace_name: ws_display_name.clone(),
            workspace_path: Some(effective_ws_dir.to_string_lossy().to_string()),
            description: desc_clean.clone(),
            had_ws_entry,
            cancel: cancel.clone(),
            child_pid: child_pid.clone(),
            pre_provision_endpoints: pre_prov_eps.clone(),
            pre_provision_snapshot_invalid: pre_prov_invalid.clone(),
            grab_status: grab_slot.clone(),
        },
    );
    spawn_refresh(&app, &handle);

    let grab_opts = GrabLoopOptions {
        name: name.clone(),
        primary_gpu: req.accelerator.clone(),
        fallback_gpus: prep.validated_grab_cfg.fallback_gpus.clone(),
        high_mem: effective_high_mem,
        auth_mode: prep.auth_mode.clone(),
        mode: prep.validated_grab_cfg.mode.clone(),
        budget_minutes: None,
        retry_seconds: prep.validated_grab_cfg.retry_seconds,
        cancel: Some(cancel.clone()),
    };

    let app_for_status = app.clone();
    let handle_for_status = handle.clone();
    let grab_slot_cb = grab_slot.clone();
    let res = relay::grab_colab_vm_loop(
        &grab_opts,
        Some((&app, &pre_prov_eps, &pre_prov_invalid)),
        Some(&child_pid),
        &move |gs: &GrabStatus| {
            *grab_slot_cb.l() = Some(gs.clone());
            spawn_refresh(&app_for_status, &handle_for_status);
        },
        &|_l: &str| {},
    )
    .await;

    app.creating_vms.l().remove(&name);
    *app.last_assignments.l() = None;

    let allocated_acc = match res {
        Ok(acc) => acc,
        Err(e) => {
            if !had_ws_entry && !cli::read_sessions().contains_key(&name) {
                cli::remove_workspace_entry(&name);
            }
            app.invalidate_keep_alive(&name, true);
            app.release_busy(&name, "creating");
            spawn_refresh(&app, &handle);
            return Err(format!("{e:#}"));
        }
    };

    // 持久化会话工作模式、认证方式、保活策略、抢卡策略与实际分配硬件（#74 + #43）
    let final_acc = if allocated_acc.trim().is_empty() {
        norm_acc
    } else {
        allocated_acc
    };
    let session_profile = cli::SessionProfile {
        boot_mode: prep.boot_mode.clone(),
        accelerator: final_acc.clone(),
        high_mem: effective_high_mem,
        workspace_path: Some(effective_ws_dir.to_string_lossy().to_string()),
        auth_mode: prep.auth_mode.clone(),
        llm_config: prep.validated_llm_cfg.clone(),
        relay_config: None,
        keep_alive: Some(prep.validated_keep_alive.clone()),
        grab_config: Some(prep.validated_grab_cfg.clone()),
        updated_at: None,
    };
    record_profile_persistence_result(
        &app,
        &name,
        cli::save_session_profile(&name, &session_profile),
    );

    {
        let mut map = app.relays.l();
        if map.get(&name).map(|h| !h.is_running()).unwrap_or(false) {
            map.remove(&name);
        }
    }

    // Session READY 后立即返回释放弹窗，遥测探针、环境依赖、LLM 部署与首轮保活全部转入后台异步推进
    app.set_busy(&name, Some("probing"));
    spawn_refresh(&app, &handle);

    let app_bg = app.clone();
    let handle_bg = handle.clone();
    let name_bg = name.clone();
    let env_cfg_bg = req.env_config.clone();
    let ws_bg = effective_ws_dir.clone();
    let llm_cfg_bg = prep.validated_llm_cfg.clone();
    let ka_enabled = prep.validated_keep_alive.enabled;
    tauri::async_runtime::spawn(async move {
        if app_bg.is_deleting(&name_bg) {
            return;
        }
        let _ = deploy_probe_inner(&app_bg, &name_bg).await;

        if !app_bg.is_deleting(&name_bg) {
            if let Some(ref env_cfg) = env_cfg_bg {
                if has_enabled_env_steps(env_cfg) {
                    app_bg.set_busy(&name_bg, Some("initializing_env"));
                    spawn_refresh(&app_bg, &handle_bg);
                    if let Err(e) =
                        apply_post_boot_env_config(&app_bg, &name_bg, &ws_bg, env_cfg).await
                    {
                        app_bg.set_deploy_error(&name_bg, Some(format!("{e:#}")));
                    }
                }
            }
        }

        {
            let should_clear = matches!(
                app_bg.busy.l().get(&name_bg).map(String::as_str),
                Some("probing" | "initializing_env" | "creating")
            );
            if should_clear {
                app_bg.set_busy(&name_bg, None);
            }
        }

        if !app_bg.is_deleting(&name_bg) {
            if let Some(llm_cfg) = llm_cfg_bg {
                if let Err(e) = spawn_llm_deployment_task(
                    app_bg.clone(),
                    handle_bg.clone(),
                    name_bg.clone(),
                    final_acc,
                    effective_high_mem,
                    ws_bg,
                    llm_cfg,
                ) {
                    app_bg.set_deploy_error(&name_bg, Some(e));
                }
            }
        }

        if ka_enabled && !app_bg.is_deleting(&name_bg) {
            if let Some(a) = app_bg.find_assignment_by_name(&name_bg).await {
                let _ = run_keep_alive_tick_for_vm(
                    &app_bg,
                    &name_bg,
                    &a,
                    None,
                    true,
                    true,
                    Some(ep_init),
                )
                .await;
            }
        }

        let ov = refresh_overview_opts(&app_bg, false, true).await;
        emit_overview(&handle_bg, &ov);
    });

    Ok(name)
}

#[tauri::command]
async fn delete_vm(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    endpoint: Option<String>,
) -> Result<TerminateReceipt, String> {
    let app = state.inner().clone();
    if app.is_deleting(&name) {
        return Err("该虚拟机正在销毁中".into());
    }
    app.set_busy(&name, Some("deleting"));
    app.invalidate_keep_alive(&name, true);
    spawn_refresh(&app, &handle);
    let r = hard_terminate(&app, &name, endpoint, true).await;
    app.release_busy(&name, "deleting");
    if r.verified_unassigned {
        app.relays.l().remove(&name);
        app.pending_close_vms.l().remove(&name);
    }
    spawn_refresh(&app, &handle);
    Ok(r)
}

#[tauri::command]
async fn deploy_probe(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
) -> Result<(), String> {
    let app = state.inner().clone();
    // 仅在实例空闲时占用 probing 标记；若后台自动探针正在注入则复用其标记，绝不覆盖 deleting / creating
    let claimed = app.try_claim_busy(&name, "probing");
    if !claimed {
        let cur = app.busy.l().get(&name).cloned().unwrap_or_default();
        if cur != "probing" {
            return Err(format!("该虚拟机正忙（{cur}），请稍后再试"));
        }
    }
    let res = deploy_probe_inner(&app, &name).await;
    if claimed {
        app.release_busy(&name, "probing");
    }
    spawn_refresh(&app, &handle);
    res.map_err(|e| format!("{e:#}"))
}

/// 按约定优先级解析命令对应的工作区路径（#42）：
/// 1. 显式传入的非空 `workspace_path`
/// 2. 指定 `vm_name` 已绑定的工作区路径（`overview.vms` -> `relays.workspace_dir` -> `creating_vms` -> 会话元数据 `workspaces.json` / `pilot_session_profiles.json` / `sessions.json`）
/// 3. 默认 `app.workspace`
pub(crate) async fn resolve_workspace_dir_for_command(
    app: &AppState,
    workspace_path: Option<&str>,
    vm_name: Option<&str>,
) -> PathBuf {
    if let Some(explicit) = workspace_path.map(str::trim).filter(|s| !s.is_empty()) {
        return PathBuf::from(explicit);
    }
    if let Some(vm) = vm_name.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(p) = app
            .overview
            .read()
            .await
            .vms
            .iter()
            .find(|v| v.name == vm)
            .and_then(|v| v.workspace.path.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
        {
            return p;
        }
        if let Some(p) = app
            .relays
            .l()
            .get(vm)
            .and_then(|h| h.workspace_dir.l().clone())
        {
            return p;
        }
        if let Some(p) = app
            .creating_vms
            .l()
            .get(vm)
            .and_then(|m| m.workspace_path.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
        {
            return p;
        }
        if let Some(p) = cli::lookup_bound_workspace_path(vm) {
            return p;
        }
    }
    app.workspace.clone()
}

fn start_relay_inner(
    app: &Arc<AppState>,
    name: &str,
    cfg: RelayConfig,
    ws_dir: PathBuf,
    auth_mode: Option<String>,
    auto_isolated_gdrive: bool,
    legacy_gdrive_action: Option<String>,
) -> RelayStatus {
    let h = {
        let mut map = app.relays.l();
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(RelayHandle::new(name)))
            .clone()
    };
    if h.is_running() {
        return h.snapshot();
    }
    // 上一轮接力任务（已手动停止）仍在清理退出中：拒绝重复启动，避免两个监工任务并发操作同一实例
    if h.task_alive.load(std::sync::atomic::Ordering::SeqCst) {
        h.log("上一轮接力任务仍在退出中，请稍候");
        return h.snapshot();
    }
    *h.workspace_dir.l() = Some(ws_dir);
    *h.auth_mode.l() = auth_mode;
    h.keep_vm_on_stop
        .store(false, std::sync::atomic::Ordering::SeqCst);
    h.stopped_during_provisioning
        .store(false, std::sync::atomic::Ordering::SeqCst);
    h.pre_provision_snapshot_invalid
        .store(true, std::sync::atomic::Ordering::SeqCst);
    h.pre_provision_endpoints.l().clear();
    h.auto_isolated_gdrive
        .store(auto_isolated_gdrive, std::sync::atomic::Ordering::SeqCst);
    h.cloud_unassign_verified
        .store(false, std::sync::atomic::Ordering::SeqCst);
    *h.legacy_gdrive_action.l() = legacy_gdrive_action;
    h.task_alive
        .store(true, std::sync::atomic::Ordering::SeqCst);
    h.gen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    *h.cancel.l() = tokio_util::sync::CancellationToken::new();
    {
        let mut st = h.status.l();
        st.running = true;
        st.round = 0;
        st.config = cfg;
        st.started_at = Some(chrono::Utc::now().timestamp());
        st.phase = "starting".into();
        st.phase_label = "正在初始化接力监工…".into();
        st.training_completed = false;
        st.weights_downloaded = false;
        st.download_error = None;
        st.local_weights_path = None;
        st.last_error = None;
    }
    if cli::get_session_profile(name)
        .and_then(|p| p.keep_alive)
        .is_none()
    {
        let default_ka = KeepAliveConfig::default();
        let res = cli::save_session_keep_alive(name, &default_ka);
        {
            let mut map = app.keep_alive.l();
            let entry = map
                .entry(name.to_string())
                .or_insert_with(|| VmKeepAliveRuntime::from_config(default_ka, None));
            if entry.epoch == 0 {
                entry.epoch = app.current_keep_alive_epoch(name);
            }
        }
        record_profile_persistence_result(app, name, res);
    }
    h.log("▶ 接力训练已开启");
    tauri::async_runtime::spawn(relay::run_relay(app.clone(), h.clone()));
    h.snapshot()
}

#[tauri::command]
async fn start_relay(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    mut config: RelayConfig,
    workspace_path: Option<String>,
    legacy_gdrive_action: Option<String>,
) -> Result<RelayStatus, String> {
    let app = state.inner().clone();
    if !valid_name(&name) {
        return Err("非法的实例名称".into());
    }
    if app.is_deleting(&name) {
        return Err("该实例正在销毁中，请稍候".into());
    }
    if let Some(existing) = app.relay(&name) {
        if !existing.is_running() && existing.task_alive.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("上一轮接力任务仍在退出清理中，请稍候再试".into());
        }
    }

    let ws_dir = resolve_workspace_dir_for_command(&app, workspace_path.as_deref(), Some(&name)).await;
    let ws_meta = std::fs::metadata(&ws_dir)
        .map_err(|e| format!("无法访问接力工作区目录 `{}`：{e}", ws_dir.display()))?;
    if !ws_meta.is_dir() {
        return Err(format!("接力工作区路径不是有效文件夹：`{}`", ws_dir.display()));
    }

    let mut was_auto_isolated = prepare_relay_config_for_vm(&app, &name, &mut config, None, None)?;
    if !was_auto_isolated {
        let iso_default = relay::resolve_isolated_gdrive_dir(relay::DEFAULT_BARE_GDRIVE_DIR, &name);
        if relay::normalize_gdrive_dir(&config.gdrive_dir) == iso_default {
            if let Some(existing_h) = app.relay(&name) {
                if existing_h
                    .auto_isolated_gdrive
                    .load(std::sync::atomic::Ordering::SeqCst)
                {
                    was_auto_isolated = true;
                }
            }
        }
    }
    if !cli::rclone_conf_path().exists() {
        return Err("未检测到 rclone.conf，请先完成 Google Drive 授权".into());
    }

    if workspace_path
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .is_some()
    {
        let ws_name = ws_dir
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("default")
            .to_string();
        cli::register_workspace_entry(
            &name,
            &ws_name,
            Some(&ws_dir.to_string_lossy()),
            None,
        );
    }

    let auth_mode = cli::get_session_auth_mode(&name);
    let st = start_relay_inner(
        &app,
        &name,
        config,
        ws_dir,
        auth_mode,
        was_auto_isolated || legacy_gdrive_action.is_some(),
        legacy_gdrive_action,
    );
    spawn_refresh(&app, &handle);
    Ok(st)
}

pub(crate) async fn stop_relay_inner<F>(
    app: &Arc<AppState>,
    name: &str,
    destroy_vm: bool,
    on_progress: F,
) -> Result<StopRelayResult, String>
where
    F: Fn(),
{
    if app.is_deleting(name) {
        return Err("该虚拟机正在销毁中".into());
    }
    let busy_tag = if destroy_vm { "deleting" } else { "stopping" };
    app.set_busy(name, Some(busy_tag));
    on_progress();

    let relay_handle = app.relay(name);
    if let Some(ref h) = relay_handle {
        h.keep_vm_on_stop
            .store(!destroy_vm, std::sync::atomic::Ordering::SeqCst);
        h.stop().await;
        {
            let mut st = h.status.l();
            st.running = false;
            st.phase = "stopped".into();
            st.phase_label = if destroy_vm {
                "本机接力监工已停止，正在销毁云端实例…".into()
            } else {
                "本机接力监工已停止，正在中断云端 Kernel…".into()
            };
        }
        h.log("⏹ 已关闭本机接力监工");
    }

    let result = if destroy_vm {
        if let Some(ref h) = relay_handle {
            h.log("🧹 正在销毁实例并核验停止计费…");
        }
        let r = hard_terminate(app, name, None, false).await;
        let verified = r.verified_unassigned;
        if verified {
            app.pending_close_vms.l().remove(name);
        }
        if let Some(ref h) = relay_handle {
            let mut st = h.status.l();
            if verified {
                st.phase = "stopped".into();
                st.phase_label = "已关闭接力并销毁云端实例（停费核验通过）".into();
            } else {
                let err_msg = r.error.clone().unwrap_or_else(|| {
                    "云端实例停费核验未通过，可能仍在计费".to_string()
                });
                st.phase = "stopped".into();
                st.phase_label = format!("本机接力已停止，但停费核验未通过：{err_msg}");
                st.last_error = Some(err_msg);
            }
            drop(st);
            h.log(if verified {
                "✅ 实例已彻底销毁，计费已停止"
            } else {
                "⚠️ 本机接力已停止，但云端实例销毁未通过核验，请重试删除"
            });
        }
        let interrupt_err = if verified {
            None
        } else {
            Some(r.error.clone().unwrap_or_else(|| {
                "云端实例停费核验未通过，原实例及训练进程可能仍在运行并计费".to_string()
            }))
        };
        StopRelayResult {
            supervisor_stopped: true,
            destroy_vm: true,
            kernel_interrupted: verified,
            kernel_interrupt_error: interrupt_err,
            terminate_receipt: Some(r),
        }
    } else {
        let (interrupted, interrupt_err) = if name.starts_with("orphan-")
            || !cli::read_sessions().contains_key(name)
        {
            (
                false,
                Some(format!(
                    "未在本地会话表中找到受管会话 [{name}]，无法执行 `colab restart-kernel`，云端原训练进程可能仍在实例中继续运行"
                )),
            )
        } else if !cli::colab_bin().exists() {
            (
                false,
                Some(
                    "未找到本机 colab CLI 可执行文件，无法重启云端 Kernel，云端原训练进程可能仍在实例中继续运行"
                        .to_string(),
                ),
            )
        } else {
            let args = cli::colab_args_for_session(name, &s(&["restart-kernel", "-s", name]));
            match cli::run_simple(&cli::colab_bin(), &args, Duration::from_secs(90)).await {
                Ok(o) if !o.timed_out && o.code == 0 => (true, None),
                Ok(o) if o.timed_out => (
                    false,
                    Some(
                        "执行 `colab restart-kernel` 超时（90s），未能确认中断云端 Kernel，云端原训练进程可能仍在实例中继续运行"
                            .to_string(),
                    ),
                ),
                Ok(o) => {
                    let cleaned = cli::strip_colab_cli_banners(&o.output);
                    let detail = cli::last_meaningful_cli_line(&cleaned, "CLI 返回非零退出码");
                    (
                        false,
                        Some(format!(
                            "执行 `colab restart-kernel` 失败（exit={}：{detail}），云端原训练进程可能仍在实例中继续运行",
                            o.code
                        )),
                    )
                }
                Err(e) => (
                    false,
                    Some(format!(
                        "调用 `colab restart-kernel` 异常（{e:#}），云端原训练进程可能仍在实例中继续运行"
                    )),
                ),
            }
        };

        if let Some(ref h) = relay_handle {
            let mut st = h.status.l();
            st.phase = "stopped".into();
            if interrupted {
                st.phase_label = "接力已停止，云端 Kernel 已重启中断（实例保留计费中）".into();
            } else if let Some(ref err_msg) = interrupt_err {
                st.phase_label = format!("本机接力已停止，但云端 Kernel 未确认中断：{err_msg}");
                st.last_error = Some(err_msg.clone());
            }
            drop(st);
            if interrupted {
                h.log("🔁 已重启远程 Kernel，云端训练进程已中断（实例保留，仍在计费）");
            } else if let Some(ref err_msg) = interrupt_err {
                h.log(format!("⚠️ {err_msg}"));
            }
        }

        StopRelayResult {
            supervisor_stopped: true,
            destroy_vm: false,
            kernel_interrupted: interrupted,
            kernel_interrupt_error: interrupt_err,
            terminate_receipt: None,
        }
    };

    app.release_busy(name, busy_tag);
    on_progress();
    Ok(result)
}

#[tauri::command]
async fn stop_relay(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    destroy_vm: bool,
) -> Result<StopRelayResult, String> {
    let app = state.inner().clone();
    let app_cb = app.clone();
    let handle_cb = handle.clone();
    stop_relay_inner(&app, &name, destroy_vm, move || {
        spawn_refresh(&app_cb, &handle_cb);
    })
    .await
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorkspaceFiles {
    pub workspace: String,
    pub train_py: String,
    pub requirements_txt: String,
}

fn read_optional_utf8_workspace_file(ws: &Path, fname: &str) -> Result<String, String> {
    let path = ws.join(fname);
    match std::fs::symlink_metadata(&path) {
        Ok(_sym_meta) => {
            let real_meta = std::fs::metadata(&path)
                .map_err(|e| format!("无法读取工作区文件 `{fname}`（{}）：{e}", path.display()))?;
            if !real_meta.is_file() {
                return Err(format!(
                    "无法读取工作区文件 `{fname}`（{}）：目标路径不是普通文件",
                    path.display()
                ));
            }
            let bytes = std::fs::read(&path)
                .map_err(|e| format!("读取工作区文件 `{fname}`（{}）失败：{e}", path.display()))?;
            String::from_utf8(bytes).map_err(|e| {
                format!(
                    "工作区文件 `{fname}`（{}）不是有效的 UTF-8 文本编码：{e}",
                    path.display()
                )
            })
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(e) => Err(format!(
            "无法检查工作区文件 `{fname}`（{}）：{e}",
            path.display()
        )),
    }
}

pub(crate) async fn read_workspace_files_inner(
    app: &AppState,
    workspace_path: Option<&str>,
    vm_name: Option<&str>,
) -> Result<WorkspaceFiles, String> {
    let ws = resolve_workspace_dir_for_command(app, workspace_path, vm_name).await;
    let ws_meta = std::fs::metadata(&ws)
        .map_err(|e| format!("无法访问工作区目录 `{}`：{e}", ws.display()))?;
    if !ws_meta.is_dir() {
        return Err(format!("工作区路径不是有效文件夹：`{}`", ws.display()));
    }
    std::fs::read_dir(&ws).map_err(|e| format!("无法读取工作区目录 `{}`：{e}", ws.display()))?;

    let train_py = read_optional_utf8_workspace_file(&ws, "train.py")?;
    let requirements_txt = read_optional_utf8_workspace_file(&ws, "requirements.txt")?;

    Ok(WorkspaceFiles {
        workspace: ws.display().to_string(),
        train_py,
        requirements_txt,
    })
}

#[tauri::command]
async fn read_workspace_files(
    state: AppStateRef<'_>,
    workspace_path: Option<String>,
    vm_name: Option<String>,
) -> Result<WorkspaceFiles, String> {
    read_workspace_files_inner(
        state.inner(),
        workspace_path.as_deref(),
        vm_name.as_deref(),
    )
    .await
}

fn write_staged_temp_file(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(bytes)?;
    f.sync_all()
}

fn replace_target_from_staged(staged: &std::path::Path, target: &std::path::Path) -> std::io::Result<()> {
    match std::fs::rename(staged, target) {
        Ok(()) => Ok(()),
        Err(rename_err) => {
            // Windows 上若目标被只读或独占句柄占用，rename 与 write 均会失败并返回底层错误
            let bytes = std::fs::read(staged)?;
            match std::fs::write(target, &bytes) {
                Ok(()) => {
                    let _ = std::fs::remove_file(staged);
                    Ok(())
                }
                Err(_) => {
                    let _ = std::fs::remove_file(staged);
                    Err(rename_err)
                }
            }
        }
    }
}

pub(crate) async fn save_workspace_files_inner(
    app: &AppState,
    train_py: String,
    requirements_txt: String,
    workspace_path: Option<&str>,
    vm_name: Option<&str>,
) -> Result<(), String> {
    const MAX_BYTES: usize = 1024 * 1024;
    if train_py.len() > MAX_BYTES || requirements_txt.len() > MAX_BYTES {
        return Err("脚本文本大小超出 1 MiB 限制".into());
    }
    let ws = resolve_workspace_dir_for_command(app, workspace_path, vm_name).await;
    let ws_meta = std::fs::metadata(&ws)
        .map_err(|e| format!("无法访问工作区目录 `{}`：{e}", ws.display()))?;
    if !ws_meta.is_dir() {
        return Err(format!("工作区路径不是有效文件夹：`{}`", ws.display()));
    }

    // Issue #84：两阶段预检 + 临时文件暂存 + 失败原子回滚，确保两份文件要么全部生效，要么保留一致的旧版本
    let train_target = ws.join("train.py");
    let req_target = ws.join("requirements.txt");
    let mut orig_snapshots: [Option<Vec<u8>>; 2] = [None, None];

    for (idx, (fname, target)) in [("train.py", &train_target), ("requirements.txt", &req_target)]
        .into_iter()
        .enumerate()
    {
        match std::fs::symlink_metadata(target) {
            Ok(meta) => {
                if meta.file_type().is_symlink() {
                    return Err(format!("拒绝写入符号链接文件: {fname}"));
                }
                if meta.is_dir() {
                    return Err(format!("无法写入 `{fname}`：目标路径是一个文件夹"));
                }
                if meta.permissions().readonly() {
                    return Err(format!("无法写入 `{fname}`：文件处于只读状态，未修改任何工作区文件"));
                }
                // 在不截断旧文件的前提下探测可写句柄，提前拦截 ACL 拒绝或独占锁
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(target)
                    .map_err(|e| format!("无法写入 `{fname}`（未修改任何工作区文件）：{e}"))?;
                let prev = std::fs::read(target)
                    .map_err(|e| format!("备份 `{fname}` 原始内容失败（未修改任何工作区文件）：{e}"))?;
                orig_snapshots[idx] = Some(prev);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(format!("无法访问 `{fname}` 元数据（未修改任何工作区文件）：{e}"));
            }
        }
    }

    let nonce = uuid::Uuid::new_v4().simple().to_string();
    let staged_train = ws.join(format!(".train.py.{nonce}.saving.tmp"));
    let staged_req = ws.join(format!(".requirements.txt.{nonce}.saving.tmp"));

    if let Err(e) = write_staged_temp_file(&staged_train, train_py.as_bytes()) {
        let _ = std::fs::remove_file(&staged_train);
        return Err(format!("暂存 `train.py` 失败（未修改任何工作区文件）：{e}"));
    }
    if let Err(e) = write_staged_temp_file(&staged_req, requirements_txt.as_bytes()) {
        let _ = std::fs::remove_file(&staged_train);
        let _ = std::fs::remove_file(&staged_req);
        return Err(format!("暂存 `requirements.txt` 失败（未修改任何工作区文件）：{e}"));
    }

    if let Err(e) = replace_target_from_staged(&staged_train, &train_target) {
        let _ = std::fs::remove_file(&staged_train);
        let _ = std::fs::remove_file(&staged_req);
        return Err(format!("写入 `train.py` 失败（未修改任何工作区文件）：{e}"));
    }

    if let Err(e) = replace_target_from_staged(&staged_req, &req_target) {
        let _ = std::fs::remove_file(&staged_req);
        let rollback_res = match &orig_snapshots[0] {
            Some(prev_bytes) => std::fs::write(&train_target, prev_bytes),
            None => match std::fs::remove_file(&train_target) {
                Ok(()) => Ok(()),
                Err(rm_err) if rm_err.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(rm_err) => Err(rm_err),
            },
        };
        return match rollback_res {
            Ok(()) => Err(format!(
                "写入 `requirements.txt` 失败：{e}（已自动回滚 `train.py`，保留一致的旧版本）"
            )),
            Err(rb_err) => Err(format!(
                "写入 `requirements.txt` 失败：{e}（注意：`train.py` 新内容已生效，但回滚旧版本失败：{rb_err}）"
            )),
        };
    }

    Ok(())
}

#[tauri::command]
async fn save_workspace_files(
    state: AppStateRef<'_>,
    train_py: String,
    requirements_txt: String,
    workspace_path: Option<String>,
    vm_name: Option<String>,
) -> Result<(), String> {
    save_workspace_files_inner(
        state.inner(),
        train_py,
        requirements_txt,
        workspace_path.as_deref(),
        vm_name.as_deref(),
    )
    .await
}

#[tauri::command]
async fn open_vm_ssh_terminal(state: AppStateRef<'_>, name: String) -> Result<String, String> {
    let vm_name = name.trim().to_string();
    if !cli::is_safe_vm_alias(&vm_name) {
        return Err("非法的实例名称".into());
    }
    let app = state.inner().clone();
    let a = app
        .find_assignment_by_name(&vm_name)
        .await
        .ok_or_else(|| format!("服务端未找到在线实例 [{vm_name}]，请先刷新状态"))?;

    let tty_script = cli::colab_config_dir().join("pilot_tty_console.py");
    if !write_if_changed(&tty_script, INTERACTIVE_TTY_CONSOLE_PY) {
        return Err("无法写入交互式远程终端桥接脚本 pilot_tty_console.py".into());
    }

    let is_orphan = vm_name.starts_with("orphan-") || !cli::read_sessions().contains_key(&vm_name);
    let auth_mode = cli::get_session_auth_mode(&vm_name).unwrap_or_default();
    let colab_bin = cli::colab_bin();
    let py_bin = cli::colab_python_bin();
    let ssh_key_path = tokio::task::spawn_blocking(|| {
        let ssh_dir = cli::ssh_dir();
        let cfg_path = cli::ssh_config_path();
        cli::ensure_ssh_dir_and_config_acl(
            &ssh_dir,
            cfg_path.exists().then_some(cfg_path.as_path()),
        );
        cli::ensure_ssh_keypair()
    })
    .await
    .ok()
    .and_then(|r| r.ok())
    .map(|(p, _)| p)
    .unwrap_or_else(cli::pilot_ssh_private_key_path);
    let has_ssh_key = ssh_key_path.exists() && cli::pilot_ssh_public_key_path().exists();

    #[cfg(windows)]
    {
        use base64::Engine as _;
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;

        // 零拼接静态 PowerShell 脚本：所有动态路径与凭据均通过子进程私有环境变量传递，不在命令行暴露代理令牌
        const PS_LAUNCH_SCRIPT: &str = r#"
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$vm = $env:COLAB_PILOT_VM_NAME
$host.UI.RawUI.WindowTitle = "Colab Pilot Terminal - $vm"
if ($env:COLAB_PILOT_IS_ORPHAN -eq "1" -or -not (Test-Path -LiteralPath $env:COLAB_PILOT_BIN)) {
    Write-Host "[Colab Pilot] 正在通过内置 Jupyter /api/terminals 交互终端直连云端容器 ($vm) ..." -ForegroundColor Cyan
    & $env:COLAB_PILOT_PY_BIN -u $env:COLAB_PILOT_TTY_SCRIPT
} else {
    Write-Host "[Colab Pilot] 正在通过 colab ssh 连接云端实例 ($vm) ..." -ForegroundColor Cyan
    $authArgs = @()
    if ($env:COLAB_PILOT_AUTH_MODE -eq "adc") {
        $authArgs = @("--auth", "adc")
    }
    if ($env:COLAB_PILOT_HAS_SSH_KEY -eq "1" -and (Test-Path -LiteralPath $env:COLAB_PILOT_SSH_KEY)) {
        & $env:COLAB_PILOT_BIN @authArgs ssh -i $env:COLAB_PILOT_SSH_KEY -s $vm
    } else {
        & $env:COLAB_PILOT_BIN @authArgs ssh -s $vm
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "[Colab Pilot] colab ssh 返回退出码 $LASTEXITCODE，正在无缝回退至 Jupyter /api/terminals 交互终端 ..." -ForegroundColor Yellow
        & $env:COLAB_PILOT_PY_BIN -u $env:COLAB_PILOT_TTY_SCRIPT
    }
}
Remove-Item Env:COLAB_PILOT_PROXY_TOKEN -ErrorAction SilentlyContinue
"#;
        let utf16_bytes: Vec<u8> = PS_LAUNCH_SCRIPT
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let encoded = base64::engine::general_purpose::STANDARD.encode(&utf16_bytes);

        let mut cmd = std::process::Command::new("powershell.exe");
        cmd.args([
            "-NoLogo",
            "-NoExit",
            "-ExecutionPolicy",
            "Bypass",
            "-EncodedCommand",
            &encoded,
        ])
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("COLAB_PILOT_VM_NAME", &vm_name)
        .env("COLAB_PILOT_AUTH_MODE", &auth_mode)
        .env("COLAB_PILOT_IS_ORPHAN", if is_orphan { "1" } else { "0" })
        .env("COLAB_PILOT_BIN", &colab_bin)
        .env("COLAB_PILOT_PY_BIN", &py_bin)
        .env("COLAB_PILOT_TTY_SCRIPT", &tty_script)
        .env("COLAB_PILOT_SSH_KEY", &ssh_key_path)
        .env("COLAB_PILOT_HAS_SSH_KEY", if has_ssh_key { "1" } else { "0" })
        .env("COLAB_PILOT_PROXY_URL", &a.runtime_proxy_info.url)
        .env("COLAB_PILOT_PROXY_TOKEN", &a.runtime_proxy_info.token)
        .creation_flags(CREATE_NEW_CONSOLE);

        cmd.spawn()
            .map_err(|e| format!("无法拉起 PowerShell 交互终端窗口：{e}"))?;
    }

    #[cfg(not(windows))]
    {
        let shell_cmd = r#"if [ "$COLAB_PILOT_IS_ORPHAN" = "1" ] || [ ! -x "$COLAB_PILOT_BIN" ]; then "$COLAB_PILOT_PY_BIN" -u "$COLAB_PILOT_TTY_SCRIPT"; elif [ "$COLAB_PILOT_AUTH_MODE" = "adc" ]; then if [ "$COLAB_PILOT_HAS_SSH_KEY" = "1" ] && [ -f "$COLAB_PILOT_SSH_KEY" ]; then "$COLAB_PILOT_BIN" --auth adc ssh -i "$COLAB_PILOT_SSH_KEY" -s "$COLAB_PILOT_VM_NAME" || "$COLAB_PILOT_PY_BIN" -u "$COLAB_PILOT_TTY_SCRIPT"; else "$COLAB_PILOT_BIN" --auth adc ssh -s "$COLAB_PILOT_VM_NAME" || "$COLAB_PILOT_PY_BIN" -u "$COLAB_PILOT_TTY_SCRIPT"; fi; elif [ "$COLAB_PILOT_HAS_SSH_KEY" = "1" ] && [ -f "$COLAB_PILOT_SSH_KEY" ]; then "$COLAB_PILOT_BIN" ssh -i "$COLAB_PILOT_SSH_KEY" -s "$COLAB_PILOT_VM_NAME" || "$COLAB_PILOT_PY_BIN" -u "$COLAB_PILOT_TTY_SCRIPT"; else "$COLAB_PILOT_BIN" ssh -s "$COLAB_PILOT_VM_NAME" || "$COLAB_PILOT_PY_BIN" -u "$COLAB_PILOT_TTY_SCRIPT"; fi; exec "${SHELL:-bash}""#;
        let mut launched = false;
        for (term, flag) in [
            ("x-terminal-emulator", "-e"),
            ("gnome-terminal", "--"),
            ("konsole", "-e"),
            ("xfce4-terminal", "-x"),
            ("xterm", "-e"),
        ] {
            let mut cmd = std::process::Command::new(term);
            cmd.args([flag, "sh", "-c", shell_cmd])
                .env("PYTHONUTF8", "1")
                .env("COLAB_PILOT_VM_NAME", &vm_name)
                .env("COLAB_PILOT_AUTH_MODE", &auth_mode)
                .env("COLAB_PILOT_IS_ORPHAN", if is_orphan { "1" } else { "0" })
                .env("COLAB_PILOT_BIN", &colab_bin)
                .env("COLAB_PILOT_PY_BIN", &py_bin)
                .env("COLAB_PILOT_TTY_SCRIPT", &tty_script)
                .env("COLAB_PILOT_SSH_KEY", &ssh_key_path)
                .env("COLAB_PILOT_HAS_SSH_KEY", if has_ssh_key { "1" } else { "0" })
                .env("COLAB_PILOT_PROXY_URL", &a.runtime_proxy_info.url)
                .env("COLAB_PILOT_PROXY_TOKEN", &a.runtime_proxy_info.token);
            if cmd.spawn().is_ok() {
                launched = true;
                break;
            }
        }
        if !launched {
            return Err("未找到系统图形终端模拟器，请复制卡片上的 SSH 命令在终端中执行".into());
        }
    }

    Ok(if is_orphan {
        format!("已打开 [{vm_name}] 的 /colab/tty 云端交互终端窗口")
    } else {
        format!("已打开 [{vm_name}] 的 SSH 交互终端窗口")
    })
}

#[tauri::command]
async fn rotate_llm_api_key(
    handle: AppHandle,
    state: AppStateRef<'_>,
    vm_name: String,
    custom_key: Option<String>,
) -> Result<LlmApiAccessInfo, String> {
    let target = vm_name.trim().to_string();
    if !cli::is_safe_vm_alias(&target) {
        return Err("非法的实例名称".into());
    }
    if let Some(ref ck) = custom_key {
        let trimmed = ck.trim();
        if !trimmed.is_empty() && trimmed.len() < 8 {
            return Err("自定义 API Key 长度至少需要 8 个字符".into());
        }
        if trimmed.chars().any(|c| c.is_control() || c.is_whitespace()) {
            return Err("自定义 API Key 不能包含空格或控制字符".into());
        }
    }

    let app = state.inner().clone();
    let existing_vm = app
        .overview
        .read()
        .await
        .vms
        .iter()
        .find(|v| v.name == target)
        .cloned();

    let (slot, engine, local_port, remote_port, model_name) = match existing_vm
        .as_ref()
        .and_then(|v| v.llm_api.as_ref())
    {
        Some(cur) => (
            cur.slot.clone(),
            cur.engine.clone(),
            cur.local_port,
            cur.remote_port,
            cur.model_name.clone(),
        ),
        None => {
            let bg = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state());
            let llm_tel = existing_vm
                .as_ref()
                .and_then(|v| v.telemetry.as_ref())
                .and_then(|t| t.get("llm"));
            let matched_slot = match_bluegreen_slot_for_vm(&target, bg.as_ref(), llm_tel);
            let slot = matched_slot
                .map(str::to_string)
                .unwrap_or_else(|| normalize_llm_slot(None, &target));
            let lp = match matched_slot {
                Some("blue") => 18081,
                Some("green") => 18082,
                _ => match slot.as_str() {
                    "blue" if target.ends_with("-blue") => 18081,
                    "green" if target.ends_with("-green") => 18082,
                    _ => 0,
                },
            };
            (slot, "vllm".into(), lp, 8000, "default".into())
        }
    };

    let ck_clean = custom_key
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());
    let workspace_dir = app.workspace.clone();

    let updated = tokio::task::spawn_blocking(move || {
        cli::rotate_llm_api_key_entry(
            &target,
            &slot,
            &engine,
            ck_clean.as_deref(),
            local_port,
            remote_port,
            &model_name,
        )
    })
    .await
    .map_err(|e| format!("更新 API Key 任务异常: {e}"))?
    .map_err(|e| format!("持久化写入 API Key 失败: {e:#}"))?;

    spawn_local_gateway_async(workspace_dir).await;
    spawn_refresh(&app, &handle);
    Ok(updated)
}

#[tauri::command]
async fn deploy_llm_on_vm(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    config: CreateVmLlmConfig,
) -> Result<LlmApiAccessInfo, String> {
    let vm_name = name.trim().to_string();
    if !valid_name(&vm_name) {
        return Err("非法的实例名称".into());
    }
    if vm_name.starts_with("orphan-") {
        return Err("浏览器未纳管孤儿实例不支持直接部署蓝绿推理栈，请通过界面新建受管实例".into());
    }
    let eng = normalize_llm_engine(&config.engine);
    if eng != "builtin" && config.model_name.trim().is_empty() {
        return Err("请填写要部署的大语言模型名称（如 Qwen/Qwen2.5-7B-Instruct）".into());
    }
    if let Some(ref ck) = config.custom_api_key {
        let trimmed = ck.trim();
        if !trimmed.is_empty() {
            if trimmed.len() < 8 {
                return Err("自定义 API Key 长度至少需要 8 个字符".into());
            }
            if trimmed.chars().any(|c| c.is_control() || c.is_whitespace()) {
                return Err("自定义 API Key 不能包含空格或控制字符".into());
            }
        }
    }

    let app = state.inner().clone();
    if let Some(cur_busy) = app.busy.l().get(&vm_name).cloned() {
        if matches!(
            cur_busy.as_str(),
            "deleting" | "creating" | "deploying_llm" | "initializing_env"
        ) {
            return Err(format!("实例 [{vm_name}] 当前正处于 `{cur_busy}` 状态，请稍候重试"));
        }
    }

    let a = app
        .find_assignment_by_name(&vm_name)
        .await
        .ok_or_else(|| format!("服务端未找到在线实例 [{vm_name}]，请确认虚拟机仍在运行"))?;

    validate_llm_hardware_compatibility(&a.hardware_label(), &config.engine)?;

    let ws_dir = resolve_workspace_dir_for_command(&app, None, Some(&vm_name)).await;
    let ws_dir = if ws_dir.is_dir() {
        ws_dir
    } else {
        app.workspace.clone()
    };

    let info = spawn_llm_deployment_task(
        app,
        handle,
        vm_name,
        a.hardware_label(),
        a.is_high_ram(),
        ws_dir,
        config,
    )?;
    Ok(info)
}

// ───────────────────────────── 大语言模型 (LLM) 流式对话网关与云端桥接 ─────────────────────────────

pub(crate) const REMOTE_LLM_CHAT_STREAM_PY: &str = r#"
import base64, json, sys

url = sys.argv[1]
token = sys.argv[2]
pref_port = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8000
kernel_id = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None

req_raw = sys.stdin.buffer.read()
req_b64 = base64.b64encode(req_raw).decode("ascii")

remote_code = f'''
import base64, http.client, json, sys, time

raw_body = base64.b64decode({json.dumps(req_b64)})
ports = []
for p in ({pref_port}, 8000, 8001, 11434):
    if p and p not in ports:
        ports.append(p)

conn = None
resp = None
http_err = None
last_err = "未能连接到云端推理端口 (8000 / 8001 / 11434)"
for port in ports:
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        c.request("POST", "/v1/chat/completions", body=raw_body, headers={{"Content-Type": "application/json"}})
        r = c.getresponse()
        if r.status < 400:
            conn = c
            resp = r
            break
        err_txt = r.read().decode("utf-8", "ignore")[:280]
        http_err = f"端口 {{port}} 返回 HTTP {{r.status}}: {{err_txt}}"
        last_err = http_err
        c.close()
        # 若该端口服务已响应明确业务/引擎错误（非 404 路径不存在），立即停止跨端口重放，防止二次执行或被后续连接拒绝覆盖真实报错
        if r.status != 404:
            break
    except Exception as e:
        if http_err is None:
            last_err = f"端口 {{port}} 连接异常: {{e}}"

if resp is None:
    print("PILOT_ERR:" + json.dumps({{"error": http_err or last_err}}, ensure_ascii=False), flush=True)
else:
    ctype = (resp.getheader("Content-Type") or "").lower()
    if "text/event-stream" in ctype:
        while True:
            line_b = resp.fp.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str or data_str == "[DONE]":
                if data_str == "[DONE]":
                    break
                continue
            try:
                obj = json.loads(data_str)
                if isinstance(obj, dict) and obj.get("error"):
                    err_v = obj["error"]
                    err_m = err_v.get("message") if isinstance(err_v, dict) else str(err_v)
                    print("PILOT_ERR:" + json.dumps({{"error": f"云端流式推理异常: {{err_m or err_v}}"}}, ensure_ascii=False), flush=True)
                    break
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {{}}
                    piece = delta.get("content") or delta.get("reasoning_content") or choices[0].get("text") or ""
                    if piece:
                        print("PILOT_DELTA:" + json.dumps({{"d": piece}}, ensure_ascii=False), flush=True)
            except Exception:
                pass
    else:
        raw_resp = resp.read().decode("utf-8", "ignore")
        if raw_resp.lstrip().startswith("data:"):
            for ln in raw_resp.splitlines():
                ln = ln.strip()
                if ln.startswith("data:") and ln[5:].strip() != "[DONE]":
                    try:
                        obj = json.loads(ln[5:].strip())
                        if isinstance(obj, dict) and obj.get("error"):
                            err_v = obj["error"]
                            err_m = err_v.get("message") if isinstance(err_v, dict) else str(err_v)
                            print("PILOT_ERR:" + json.dumps({{"error": f"云端流式推理异常: {{err_m or err_v}}"}}, ensure_ascii=False), flush=True)
                            break
                        delta = (obj.get("choices") or [{{}}])[0].get("delta") or {{}}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            print("PILOT_DELTA:" + json.dumps({{"d": piece}}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
        else:
            try:
                obj = json.loads(raw_resp)
                if isinstance(obj, dict) and obj.get("error"):
                    err_v = obj["error"]
                    err_m = err_v.get("message") if isinstance(err_v, dict) else str(err_v)
                    print("PILOT_ERR:" + json.dumps({{"error": f"云端推理异常: {{err_m or err_v}}"}}, ensure_ascii=False), flush=True)
                else:
                    choices = obj.get("choices") or []
                    if choices:
                        msg = choices[0].get("message") or {{}}
                        piece = msg.get("content") or choices[0].get("text") or ""
                        if piece:
                            print("PILOT_DELTA:" + json.dumps({{"d": piece}}, ensure_ascii=False), flush=True)
            except Exception as e:
                print("PILOT_ERR:" + json.dumps({{"error": f"解析云端响应失败: {{e}} ({{raw_resp[:200]}})"}}), flush=True)
    try:
        conn.close()
    except Exception:
        pass
'''

from colab_cli.runtime import ColabRuntime

# 使用独立 ZMQ session_id=None，避免与后台保活或并发通道互相抢占/关闭 WebSocket
rt = ColabRuntime(url, token, kernel_id=kernel_id, session_id=None)
try:
    def on_out(out):
        otype = out.get("output_type", "")
        if otype == "stream":
            txt = out.get("text", "")
            if txt:
                sys.stdout.write(txt)
                sys.stdout.flush()
        elif otype == "error":
            ename = out.get("ename", "Error")
            evalue = out.get("evalue", "")
            sys.stdout.write("PILOT_ERR:" + json.dumps({"error": f"{ename}: {evalue}"}, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    rt.execute_code(remote_code, output_hook=on_out, timeout=300)
    if getattr(rt, "kernel_id", None):
        sys.stdout.write(f"PILOT_KID:{rt.kernel_id}\n")
        sys.stdout.flush()
finally:
    rt.stop()
"#;

fn estimate_tokens(text: &str) -> u32 {
    if text.is_empty() {
        return 0;
    }
    let mut cjk = 0u32;
    let mut ascii_chars = 0u32;
    for ch in text.chars() {
        if ('\u{4e00}'..='\u{9fff}').contains(&ch) {
            cjk += 1;
        } else {
            ascii_chars += 1;
        }
    }
    let non_cjk = if ascii_chars > 0 {
        (ascii_chars / 4).max(1)
    } else {
        0
    };
    (cjk + non_cjk).max(1)
}

/// 若本机存在蓝绿网关脚本且尚未启动（或需同步新监听端口），静默在后台拉起以预热网关通道
/// （同步阻塞实现，须通过 `spawn_local_gateway_async` 在阻塞线程池中调用；15 秒内最多拉起一次）
fn maybe_spawn_local_gateway(workspace: &Path) {
    static LAST_SPAWN: Mutex<Option<Instant>> = Mutex::new(None);
    let addr: std::net::SocketAddr = match "127.0.0.1:18080".parse() {
        Ok(a) => a,
        Err(_) => return,
    };
    if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok() {
        return;
    }
    {
        let mut guard = LAST_SPAWN.l();
        if guard
            .map(|t| t.elapsed() < Duration::from_secs(15))
            .unwrap_or(false)
        {
            return;
        }
        let Some(gw_script) = ensure_gateway_script_path(workspace) else {
            return;
        };
        let py = cli::colab_python_bin();
        let mut cmd = std::process::Command::new(py);
        cmd.arg(gw_script)
            .arg("--port")
            .arg("18080")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let Ok(mut child) = cmd.spawn() else {
            return;
        };
        *guard = Some(Instant::now());
        // 回收子进程退出状态，杜绝 Unix 僵尸进程累积
        std::thread::spawn(move || {
            let _ = child.wait();
        });
    }
    // 等待本地网关监听端口就绪（最多约 1.5 秒），使首轮本地通道请求即可命中
    for _ in 0..15 {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(80)).is_ok() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

async fn spawn_local_gateway_async(workspace: PathBuf) {
    let _ = tokio::task::spawn_blocking(move || maybe_spawn_local_gateway(&workspace)).await;
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedLlmTarget {
    pub vm_name: String,
    pub slot: Option<String>,
    pub model: String,
    pub remote_port: u16,
    pub local_tunnel_port: Option<u16>,
    pub api_key: Option<String>,
    pub kernel_id: Option<String>,
}

pub(crate) async fn resolve_llm_target(
    app: &Arc<AppState>,
    req: &LlmChatRequest,
) -> anyhow::Result<ResolvedLlmTarget> {
    let mut ov = app.overview.read().await.clone();
    if ov.vms.is_empty() {
        ov = refresh_overview(app, false).await;
    }
    let online_names: Vec<String> = ov
        .vms
        .iter()
        .filter(|v| v.on_server)
        .map(|v| v.name.clone())
        .collect();
    let busy_snapshot = app.busy.l().clone();
    let creating_names: Vec<String> = app.creating_vms.l().keys().cloned().collect();
    let now_ts = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
    let bg = filter_live_bluegreen_state(
        parse_llm_bluegreen_state(cli::read_llm_bluegreen_state()),
        &online_names,
        &busy_snapshot,
        &creating_names,
        now_ts,
    );

    let prefix = bg.as_ref().map(bg_session_prefix).unwrap_or("llm");
    let bg_active_session = bg
        .as_ref()
        .and_then(|v| v.active_session.as_deref())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let bg_active_slot = bg
        .as_ref()
        .and_then(|v| v.active_slot.as_deref())
        .map(|s| s.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty());
    let bg_explicit_for_active_slot = bg.as_ref().and_then(|b| {
        bg_active_slot
            .as_deref()
            .and_then(|sl| bg_explicit_slot_session(b, sl))
    });

    let wanted = req.target_vm.trim();
    let vm_opt = if wanted.is_empty() || wanted.eq_ignore_ascii_case("auto") {
        // 1. 优先严格匹配蓝绿状态记录的真实会话名（slots[slot].session_name -> blue_session/green_session -> active_session），
        //    其次匹配遥测中 slot 一致的就绪节点，最后才回退 `{prefix}-{slot}`（#48）
        let by_bg = bg_explicit_for_active_slot
            .as_ref()
            .and_then(|sess| ov.vms.iter().find(|v| v.on_server && &v.name == sess))
            .or_else(|| {
                bg_active_session
                    .as_ref()
                    .and_then(|sess| ov.vms.iter().find(|v| v.on_server && &v.name == sess))
            })
            .or_else(|| {
                bg_active_slot.as_ref().and_then(|slot| {
                    ov.vms.iter().find(|v| {
                        v.on_server
                            && v.telemetry
                                .as_ref()
                                .map(|t| {
                                    t["llm"]["online"].as_bool() == Some(true)
                                        && t["llm"]["slot"]
                                            .as_str()
                                            .map(|s| s.eq_ignore_ascii_case(slot))
                                            .unwrap_or(false)
                                        && !t["llm"]["role"]
                                            .as_str()
                                            .unwrap_or("")
                                            .eq_ignore_ascii_case("warming")
                                })
                                .unwrap_or(false)
                    })
                })
            })
            .or_else(|| {
                if bg_explicit_for_active_slot.is_none() && bg_active_session.is_none() {
                    bg_active_slot.as_ref().and_then(|slot| {
                        let expected = format!("{prefix}-{slot}");
                        ov.vms.iter().find(|v| v.on_server && v.name == expected)
                    })
                } else {
                    None
                }
            });
        // 2. 匹配任意 telemetry.llm.online == true 且 role == active 的实例
        let by_active_llm = ov.vms.iter().find(|v| {
            v.on_server
                && v.telemetry
                    .as_ref()
                    .map(|t| {
                        t["llm"]["online"].as_bool() == Some(true)
                            && t["llm"]["role"]
                                .as_str()
                                .unwrap_or("active")
                                .eq_ignore_ascii_case("active")
                    })
                    .unwrap_or(false)
        });
        // 3. 匹配任意非 warming 的就绪 LLM 实例，最后回退任意含 LLM 元数据的在线实例（绝不劫持普通通用/训练实例！）
        let is_llm_vm = |v: &&VmView| -> bool {
            v.on_server
                && (v.llm_api.is_some()
                    || v.telemetry
                        .as_ref()
                        .map(|t| t["llm"].is_object() || t["threads"]["llm_process"].is_object())
                        .unwrap_or(false))
        };
        let by_ready_llm = ov.vms.iter().find(|v| {
            is_llm_vm(v)
                && !v
                    .telemetry
                    .as_ref()
                    .and_then(|t| t["llm"]["role"].as_str())
                    .map(|r| r.eq_ignore_ascii_case("warming"))
                    .unwrap_or(false)
        });
        let by_any_llm = ov.vms.iter().find(is_llm_vm);
        by_bg.or(by_active_llm).or(by_ready_llm).or(by_any_llm)
    } else {
        ov.vms.iter().find(|v| v.on_server && v.name == wanted)
    };

    let Some(vm) = vm_opt else {
        if !wanted.is_empty() && !wanted.eq_ignore_ascii_case("auto") {
            anyhow::bail!("指定的实例 [{wanted}] 当前不在线或不存在");
        }
        if ov.vms.iter().any(|v| v.on_server) {
            anyhow::bail!(
                "当前在线虚拟机尚未部署大模型推理引擎，请点击实例卡片上的「一键部署 LLM」或新建大模型模式实例"
            );
        }
        anyhow::bail!(
            "当前没有在线的大模型 Colab 实例，请先点击「新建虚拟机」选择大模型部署模式"
        );
    };

    let llm_tel = vm.telemetry.as_ref().and_then(|t| t.get("llm"));
    let matched_bg_slot = match_bluegreen_slot_for_vm(&vm.name, bg.as_ref(), llm_tel);
    let key_meta = cli::lookup_vm_api_key_meta(&vm.name);
    let local_tunnel_port = matched_bg_slot
        .and_then(|sl| match sl {
            "blue" => Some(18081),
            "green" => Some(18082),
            _ => None,
        })
        .or_else(|| vm.llm_api.as_ref().map(|x| x.local_port).filter(|&p| p > 0))
        .or_else(|| key_meta.as_ref().and_then(|(_, p)| *p));
    let api_key = vm
        .llm_api
        .as_ref()
        .map(|x| x.api_key.clone())
        .filter(|s| !s.is_empty())
        .or_else(|| key_meta.map(|(k, _)| k));

    let slot = matched_bg_slot
        .map(str::to_string)
        .or_else(|| {
            llm_tel
                .and_then(|l| l.get("slot"))
                .and_then(|x| x.as_str())
                .map(|s| s.to_lowercase())
        })
        .or_else(|| vm.llm_api.as_ref().map(|x| x.slot.clone()))
        .or_else(|| match local_tunnel_port {
            Some(18081) => Some("blue".into()),
            Some(18082) => Some("green".into()),
            _ => None,
        });

    let bg_slot_model = matched_bg_slot.and_then(|sl| {
        bg.as_ref()
            .and_then(|b| b.slots.as_ref())
            .and_then(|sv| sv.get(sl))
            .and_then(|e| e.get("model_name"))
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    });
    let bg_model = bg_slot_model.or_else(|| {
        bg.as_ref()
            .and_then(|v| v.model_name.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    });

    let model = req
        .model
        .clone()
        .filter(|m| !m.trim().is_empty())
        .or_else(|| {
            llm_tel
                .and_then(|l| l.get("model_name"))
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        })
        .or_else(|| {
            vm.llm_api
                .as_ref()
                .map(|x| x.model_name.clone())
                .filter(|m| !m.is_empty() && m != "default")
        })
        .or(bg_model)
        .unwrap_or_else(|| "default".into());

    let remote_port = llm_tel
        .and_then(|l| l.get("port"))
        .and_then(|x| x.as_u64())
        .map(|p| p as u16)
        .or_else(|| vm.llm_api.as_ref().map(|x| x.remote_port).filter(|&p| p > 0))
        .unwrap_or(8000);

    Ok(ResolvedLlmTarget {
        vm_name: vm.name.clone(),
        slot,
        model,
        remote_port,
        local_tunnel_port,
        api_key,
        kernel_id: vm.kernel_id.clone(),
    })
}

pub(crate) fn parse_sse_frame(line_bytes: &[u8]) -> Result<Option<String>, String> {
    let line_owned = String::from_utf8_lossy(line_bytes);
    let line = line_owned.trim();
    let Some(raw_data) = line.strip_prefix("data:") else {
        return Ok(None);
    };
    let data_str = raw_data.trim();
    if data_str.is_empty() || data_str == "[DONE]" {
        return Ok(None);
    }
    let Ok(v) = serde_json::from_str::<Value>(data_str) else {
        return Ok(None);
    };
    if let Some(err) = v.get("error") {
        if !err.is_null() {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .map(|s| s.to_string())
                .or_else(|| err.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| err.to_string());
            return Err(format!("云端流式推理异常: {msg}"));
        }
    }
    let piece = v["choices"]
        .get(0)
        .and_then(|c| {
            c["delta"]["content"]
                .as_str()
                .or_else(|| c["delta"]["reasoning_content"].as_str())
                .or_else(|| c["text"].as_str())
        })
        .unwrap_or("");
    if piece.is_empty() {
        Ok(None)
    } else {
        Ok(Some(piece.to_string()))
    }
}

#[cfg(test)]
pub(crate) fn extract_sse_piece(line_bytes: &[u8]) -> Option<String> {
    parse_sse_frame(line_bytes).ok().flatten()
}

#[allow(clippy::too_many_arguments)]
async fn try_stream_via_local_tunnel(
    handle: &AppHandle,
    req: &LlmChatRequest,
    target: &ResolvedLlmTarget,
    local_port: u16,
    body_val: &Value,
    emitted: &mut bool,
    gateway_handled: &mut bool,
    chat_cancel: &tokio_util::sync::CancellationToken,
) -> anyhow::Result<LlmChatResponse> {
    if chat_cancel.is_cancelled() {
        anyhow::bail!("对话已手动停止");
    }
    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_millis(650))
        .timeout(Duration::from_secs(300))
        .build()?;

    let health_url = format!("http://127.0.0.1:{local_port}/pilot/health");
    let mut h_req = client
        .get(&health_url)
        .header("X-Colab-Vm", &target.vm_name)
        .timeout(Duration::from_millis(650));
    if let Some(ref k) = target.api_key {
        h_req = h_req.header("Authorization", format!("Bearer {k}"));
    }
    let h_resp = tokio::select! {
        _ = chat_cancel.cancelled() => anyhow::bail!("对话已手动停止"),
        r = h_req.send() => r?,
    };
    if !h_resp.status().is_success() {
        anyhow::bail!("本地网关端口 {local_port} 尚未就绪");
    }

    let chat_url = format!("http://127.0.0.1:{local_port}/v1/chat/completions");
    let mut c_req = client
        .post(&chat_url)
        .header("Content-Type", "application/json")
        .header("X-Colab-Vm", &target.vm_name);
    if let Some(ref k) = target.api_key {
        c_req = c_req.header("Authorization", format!("Bearer {k}"));
    }
    let mut resp = tokio::select! {
        _ = chat_cancel.cancelled() => anyhow::bail!("对话已手动停止"),
        r = c_req.json(body_val).send() => r?,
    };
    *gateway_handled = true;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("本地网关 HTTP {status}: {err_txt}");
    }

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut reply = String::new();
    let mut completion_tokens = 0u32;
    // 以字节缓冲按 b'\n' 切分完整行后再解码，绝不对半截 UTF-8 多字节字符（中文等）做有损解码
    let mut buf: Vec<u8> = Vec::new();
    let mut cancelled = false;

    loop {
        let next = tokio::select! {
            _ = chat_cancel.cancelled() => {
                cancelled = true;
                break;
            }
            c = resp.chunk() => c,
        };
        let chunk = match next {
            Ok(Some(c)) => c,
            Ok(None) => break,
            // 已输出部分内容后中断：保留已生成的部分回复，不再报错回退（避免重复回答）
            Err(_) if !reply.is_empty() => break,
            Err(e) => return Err(e.into()),
        };
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            match parse_sse_frame(&line_bytes) {
                Ok(Some(piece)) => {
                    let first = *t_first.get_or_insert_with(Instant::now);
                    let ttft_ms = first.duration_since(t_start).as_secs_f64() * 1000.0;
                    completion_tokens += estimate_tokens(&piece);
                    reply.push_str(&piece);
                    let gen_s = first.elapsed().as_secs_f64().max(0.025);
                    let inst_tps = (completion_tokens as f64) / gen_s;
                    *emitted = true;
                    let _ = handle.emit(
                        "llm-chat-event",
                        LlmChatStreamEvent {
                            request_id: req.request_id.clone(),
                            kind: "delta".into(),
                            vm_name: target.vm_name.clone(),
                            slot: target.slot.clone(),
                            model: target.model.clone(),
                            delta: Some(piece),
                            ttft_ms: Some((ttft_ms * 10.0).round() / 10.0),
                            instant_tps: Some((inst_tps * 10.0).round() / 10.0),
                            avg_tps: Some((inst_tps * 10.0).round() / 10.0),
                            completion_tokens: Some(completion_tokens),
                            elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                            transport: Some(format!("本地蓝绿网关 (:{local_port})")),
                            error: None,
                        },
                    );
                }
                Ok(None) => {}
                Err(err_msg) => {
                    if reply.is_empty() {
                        anyhow::bail!("{err_msg}");
                    }
                    let tail_note = format!("\n\n⚠️ [{err_msg}]");
                    reply.push_str(&tail_note);
                    *emitted = true;
                    let _ = handle.emit(
                        "llm-chat-event",
                        LlmChatStreamEvent {
                            request_id: req.request_id.clone(),
                            kind: "delta".into(),
                            vm_name: target.vm_name.clone(),
                            slot: target.slot.clone(),
                            model: target.model.clone(),
                            delta: Some(tail_note),
                            ttft_ms: None,
                            instant_tps: None,
                            avg_tps: None,
                            completion_tokens: Some(completion_tokens),
                            elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                            transport: Some(format!("本地蓝绿网关 (:{local_port})")),
                            error: None,
                        },
                    );
                    break;
                }
            }
        }
    }
    if !cancelled && !buf.is_empty() {
        match parse_sse_frame(&buf) {
            Ok(Some(piece)) => {
                let first = *t_first.get_or_insert_with(Instant::now);
                let ttft_ms = first.duration_since(t_start).as_secs_f64() * 1000.0;
                completion_tokens += estimate_tokens(&piece);
                reply.push_str(&piece);
                let gen_s = first.elapsed().as_secs_f64().max(0.025);
                let inst_tps = (completion_tokens as f64) / gen_s;
                *emitted = true;
                let _ = handle.emit(
                    "llm-chat-event",
                    LlmChatStreamEvent {
                        request_id: req.request_id.clone(),
                        kind: "delta".into(),
                        vm_name: target.vm_name.clone(),
                        slot: target.slot.clone(),
                        model: target.model.clone(),
                        delta: Some(piece),
                        ttft_ms: Some((ttft_ms * 10.0).round() / 10.0),
                        instant_tps: Some((inst_tps * 10.0).round() / 10.0),
                        avg_tps: Some((inst_tps * 10.0).round() / 10.0),
                        completion_tokens: Some(completion_tokens),
                        elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                        transport: Some(format!("本地蓝绿网关 (:{local_port})")),
                        error: None,
                    },
                );
            }
            Ok(None) => {}
            Err(err_msg) if reply.is_empty() => anyhow::bail!("{err_msg}"),
            Err(err_msg) => {
                let tail_note = format!("\n\n⚠️ [{err_msg}]");
                reply.push_str(&tail_note);
                *emitted = true;
            }
        }
        buf.clear();
    }
    if cancelled && reply.is_empty() {
        anyhow::bail!("对话已手动停止");
    }

    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = t_first
        .map(|f| f.duration_since(t_start).as_secs_f64() * 1000.0)
        .unwrap_or(elapsed_ms);
    let gen_s = t_first
        .map(|f| f.elapsed().as_secs_f64().max(0.025))
        .unwrap_or_else(|| (elapsed_ms / 1000.0).max(0.025));
    let avg_tps = ((completion_tokens as f64) / gen_s * 10.0).round() / 10.0;

    Ok(LlmChatResponse {
        request_id: req.request_id.clone(),
        vm_name: target.vm_name.clone(),
        slot: target.slot.clone(),
        model: target.model.clone(),
        reply,
        ttft_ms: (ttft_ms * 10.0).round() / 10.0,
        avg_tps,
        completion_tokens,
        elapsed_ms: elapsed_ms.round(),
        transport: format!("本地蓝绿网关 (127.0.0.1:{local_port})"),
    })
}

async fn stream_via_colab_runtime(
    handle: &AppHandle,
    a: &Assignment,
    req: &LlmChatRequest,
    target: &ResolvedLlmTarget,
    body_val: &Value,
    chat_cancel: &tokio_util::sync::CancellationToken,
) -> anyhow::Result<LlmChatResponse> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

    if chat_cancel.is_cancelled() {
        anyhow::bail!("对话已手动停止");
    }

    let py = cli::colab_python_bin();
    let mut cmd = Command::new(&py);
    cmd.args([
        "-u",
        "-c",
        REMOTE_LLM_CHAT_STREAM_PY,
        &a.runtime_proxy_info.url,
        &a.runtime_proxy_info.token,
        &target.remote_port.to_string(),
        target.kernel_id.as_deref().unwrap_or(""),
    ])
    .env("PYTHONUTF8", "1")
    .env("PYTHONIOENCODING", "utf-8")
    .env("NO_COLOR", "1")
    .stdin(std::process::Stdio::piped())
    .stdout(std::process::Stdio::piped())
    .stderr(std::process::Stdio::piped())
    .kill_on_drop(true);

    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000);

    let payload_bytes = serde_json::to_vec(body_val)?;
    let mut child = cmd.spawn().context("启动 ColabRuntime 流式桥接进程失败")?;
    let stdin_task = child.stdin.take().map(|mut stdin| {
        tokio::spawn(async move {
            let _ = stdin.write_all(&payload_bytes).await;
            let _ = stdin.shutdown().await;
            drop(stdin);
        })
    });

    // 持续排空 stderr（防止管道写满导致子进程阻塞），仅保留末尾约 2KB 用于错误提示
    let stderr_task = child.stderr.take().map(|mut se| {
        tokio::spawn(async move {
            use tokio::io::AsyncReadExt;
            const KEEP: usize = 2048;
            let mut tail: Vec<u8> = Vec::new();
            let mut chunk = [0u8; 4096];
            loop {
                match se.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        tail.extend_from_slice(&chunk[..n]);
                        if tail.len() > KEEP {
                            let cut = tail.len() - KEEP;
                            tail.drain(..cut);
                        }
                    }
                }
            }
            String::from_utf8_lossy(&tail).trim().to_string()
        })
    });

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("无法读取流式桥接输出"))?;
    let mut lines = BufReader::new(stdout).lines();

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut reply = String::new();
    let mut completion_tokens = 0u32;
    let mut remote_err: Option<String> = None;
    let mut discovered_kid: Option<String> = None;

    // 整体 330 秒截止（云端推理 300 秒超时 + 桥接握手余量），超时或手动取消立即结束桥接进程树
    let deadline = tokio::time::sleep(Duration::from_secs(330));
    tokio::pin!(deadline);
    let mut timed_out = false;
    let mut cancelled = false;

    loop {
        let line = tokio::select! {
            _ = chat_cancel.cancelled() => {
                cancelled = true;
                break;
            }
            l = lines.next_line() => match l {
                Ok(Some(l)) => l,
                _ => break,
            },
            _ = &mut deadline => {
                timed_out = true;
                break;
            }
        };
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("PILOT_DELTA:") {
            if let Ok(v) = serde_json::from_str::<Value>(rest) {
                if let Some(piece) = v.get("d").and_then(|x| x.as_str()) {
                    if !piece.is_empty() {
                        let first = *t_first.get_or_insert_with(Instant::now);
                        let ttft_ms = first.duration_since(t_start).as_secs_f64() * 1000.0;
                        completion_tokens += estimate_tokens(piece);
                        reply.push_str(piece);
                        let gen_s = first.elapsed().as_secs_f64().max(0.025);
                        let inst_tps = (completion_tokens as f64) / gen_s;
                        let _ = handle.emit(
                            "llm-chat-event",
                            LlmChatStreamEvent {
                                request_id: req.request_id.clone(),
                                kind: "delta".into(),
                                vm_name: target.vm_name.clone(),
                                slot: target.slot.clone(),
                                model: target.model.clone(),
                                delta: Some(piece.to_string()),
                                ttft_ms: Some((ttft_ms * 10.0).round() / 10.0),
                                instant_tps: Some((inst_tps * 10.0).round() / 10.0),
                                avg_tps: Some((inst_tps * 10.0).round() / 10.0),
                                completion_tokens: Some(completion_tokens),
                                elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                                transport: Some(format!(
                                    "Colab 直连 (:{} 边车)",
                                    target.remote_port
                                )),
                                error: None,
                            },
                        );
                    }
                }
            }
        } else if let Some(rest) = trimmed.strip_prefix("PILOT_ERR:") {
            if let Ok(v) = serde_json::from_str::<Value>(rest) {
                remote_err = v.get("error").and_then(|x| x.as_str()).map(|s| s.to_string());
            } else {
                remote_err = Some(rest.to_string());
            }
        } else if let Some(kid) = trimmed.strip_prefix("PILOT_KID:") {
            if !kid.trim().is_empty() {
                discovered_kid = Some(kid.trim().to_string());
            }
        }
    }

    if let Some(kid) = discovered_kid {
        let ep = a.endpoint.clone();
        let _ = tokio::task::spawn_blocking(move || cli::update_session_kernel_id(&ep, &kid)).await;
    }

    if let Some(t) = stdin_task {
        t.abort();
    }

    if cancelled {
        cli::kill_tree(&mut child).await;
        if let Some(t) = stderr_task {
            t.abort();
        }
        if reply.is_empty() {
            anyhow::bail!("对话已手动停止");
        }
    } else if timed_out {
        cli::kill_tree(&mut child).await;
        if let Some(t) = stderr_task {
            t.abort();
        }
        anyhow::bail!("Colab 直连流式超时 (330s)");
    } else {
        let status = child.wait().await.ok();
        let stderr_tail = match stderr_task {
            Some(t) => tokio::time::timeout(Duration::from_secs(3), t)
                .await
                .ok()
                .and_then(|r| r.ok())
                .unwrap_or_default(),
            None => String::new(),
        };
        if let Some(err) = remote_err {
            if reply.is_empty() {
                anyhow::bail!("{err}");
            }
            let tail_note = format!("\n\n⚠️ [{err}]");
            reply.push_str(&tail_note);
            let _ = handle.emit(
                "llm-chat-event",
                LlmChatStreamEvent {
                    request_id: req.request_id.clone(),
                    kind: "delta".into(),
                    vm_name: target.vm_name.clone(),
                    slot: target.slot.clone(),
                    model: target.model.clone(),
                    delta: Some(tail_note),
                    ttft_ms: None,
                    instant_tps: None,
                    avg_tps: None,
                    completion_tokens: Some(completion_tokens),
                    elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                    transport: Some(format!("Colab 直连 (:{} 边车)", target.remote_port)),
                    error: None,
                },
            );
        }
        if reply.is_empty()
            && !status.map(|s| s.success()).unwrap_or(false)
            && !stderr_tail.is_empty()
        {
            let cleaned = cli::strip_colab_cli_banners(&stderr_tail);
            let last = cli::last_meaningful_cli_line(&cleaned, "未知异常");
            anyhow::bail!("Colab 直连流式桥接异常退出：{last}");
        }
    }

    let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
    let ttft_ms = t_first
        .map(|f| f.duration_since(t_start).as_secs_f64() * 1000.0)
        .unwrap_or(elapsed_ms);
    let gen_s = t_first
        .map(|f| f.elapsed().as_secs_f64().max(0.025))
        .unwrap_or_else(|| (elapsed_ms / 1000.0).max(0.025));
    let avg_tps = ((completion_tokens as f64) / gen_s * 10.0).round() / 10.0;

    Ok(LlmChatResponse {
        request_id: req.request_id.clone(),
        vm_name: target.vm_name.clone(),
        slot: target.slot.clone(),
        model: target.model.clone(),
        reply,
        ttft_ms: (ttft_ms * 10.0).round() / 10.0,
        avg_tps,
        completion_tokens,
        elapsed_ms: elapsed_ms.round(),
        transport: format!("Colab 直连 (:{} 边车)", target.remote_port),
    })
}

fn emit_chat_start(handle: &AppHandle, req: &LlmChatRequest, target: &ResolvedLlmTarget) {
    let _ = handle.emit(
        "llm-chat-event",
        LlmChatStreamEvent {
            request_id: req.request_id.clone(),
            kind: "start".into(),
            vm_name: target.vm_name.clone(),
            slot: target.slot.clone(),
            model: target.model.clone(),
            delta: None,
            ttft_ms: None,
            instant_tps: None,
            avg_tps: None,
            completion_tokens: Some(0),
            elapsed_ms: Some(0.0),
            transport: None,
            error: None,
        },
    );
}

struct ActiveChatGuard {
    app: Arc<AppState>,
    request_id: String,
}

impl Drop for ActiveChatGuard {
    fn drop(&mut self) {
        self.app.active_chats.l().remove(&self.request_id);
    }
}

#[tauri::command]
async fn cancel_llm_chat(state: AppStateRef<'_>, request_id: String) -> Result<(), String> {
    let tok = state
        .active_chats
        .l()
        .get(&request_id)
        .map(|e| e.cancel.clone());
    if let Some(t) = tok {
        t.cancel();
    }
    Ok(())
}

#[tauri::command]
async fn stream_llm_chat(
    handle: AppHandle,
    state: AppStateRef<'_>,
    req: LlmChatRequest,
) -> Result<LlmChatResponse, String> {
    let app = state.inner().clone();
    let chat_cancel = tokio_util::sync::CancellationToken::new();
    app.active_chats.l().insert(
        req.request_id.clone(),
        ActiveChatEntry {
            cancel: chat_cancel.clone(),
            vm_name: None,
        },
    );
    let _guard = ActiveChatGuard {
        app: app.clone(),
        request_id: req.request_id.clone(),
    };

    let target = resolve_llm_target(&app, &req)
        .await
        .map_err(|e| e.to_string())?;
    if let Some(entry) = app.active_chats.l().get_mut(&req.request_id) {
        entry.vm_name = Some(target.vm_name.clone());
    }

    let a = app
        .find_assignment_by_name(&target.vm_name)
        .await
        .ok_or_else(|| format!("服务端未找到在线实例 [{}]", target.vm_name))?;

    if target.local_tunnel_port.is_some() && !chat_cancel.is_cancelled() {
        spawn_local_gateway_async(app.workspace.clone()).await;
    }

    emit_chat_start(&handle, &req, &target);

    let body_val = serde_json::json!({
        "model": target.model,
        "messages": req.messages,
        "max_tokens": req.max_tokens.unwrap_or(2048),
        "temperature": req.temperature.unwrap_or(0.7),
        "stream": true,
    });

    // 优先使用已就绪的本机蓝绿网关，仅在本地网关未就绪/不可达时无缝回退至 ColabRuntime 直连流式通道；
    // 若本地网关已受理请求并返回业务错误（如 HTTP 400/500/502）、已被用户手动停止或已输出部分内容，则绝不回退重放提示词
    let mut res = if let Some(lport) = target.local_tunnel_port {
        let mut tunnel_emitted = false;
        let mut gateway_handled = false;
        match try_stream_via_local_tunnel(
            &handle,
            &req,
            &target,
            lport,
            &body_val,
            &mut tunnel_emitted,
            &mut gateway_handled,
            &chat_cancel,
        )
        .await
        {
            Ok(r) if !r.reply.is_empty() || tunnel_emitted || gateway_handled => Ok(r),
            Err(e) if tunnel_emitted || gateway_handled || chat_cancel.is_cancelled() => Err(e),
            _ => {
                emit_chat_start(&handle, &req, &target);
                stream_via_colab_runtime(&handle, &a, &req, &target, &body_val, &chat_cancel).await
            }
        }
    } else {
        stream_via_colab_runtime(&handle, &a, &req, &target, &body_val, &chat_cancel).await
    };

    // 对话正常完成后立即拉取一次该实例的最新遥测快照（同步云端边车实测 tok/s、TTFT 与曲线）并广播给仪表盘
    if !chat_cancel.is_cancelled() {
        if let Ok(ref mut ok_resp) = res {
            let bg = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state());
            let entry =
                fetch_telemetry(&app, &a, Some(&target.vm_name), bg.as_ref()).await;
            if let Some(ref tel_val) = entry.data {
                if let Some(sidecar_tps) = tel_val["llm"]["last_request_tps"]
                    .as_f64()
                    .filter(|&x| x > 0.0)
                    .or_else(|| tel_val["llm"]["instant_tps"].as_f64().filter(|&x| x > 0.0))
                {
                    ok_resp.avg_tps = (sidecar_tps * 10.0).round() / 10.0;
                }
                if let Some(sidecar_ttft) = tel_val["llm"]["ttft_ms"].as_f64().filter(|&x| x > 0.0)
                {
                    ok_resp.ttft_ms = (sidecar_ttft * 10.0).round() / 10.0;
                }
                app.telemetry.l().insert(a.endpoint.clone(), entry);
            }
            let app_bg = app.clone();
            let handle_bg = handle.clone();
            tokio::spawn(async move {
                let ov = refresh_overview(&app_bg, false).await;
                emit_overview(&handle_bg, &ov);
            });
        }
    }

    match res {
        Ok(resp) => {
            let _ = handle.emit(
                "llm-chat-event",
                LlmChatStreamEvent {
                    request_id: req.request_id.clone(),
                    kind: "done".into(),
                    vm_name: resp.vm_name.clone(),
                    slot: resp.slot.clone(),
                    model: resp.model.clone(),
                    delta: None,
                    ttft_ms: Some(resp.ttft_ms),
                    instant_tps: Some(resp.avg_tps),
                    avg_tps: Some(resp.avg_tps),
                    completion_tokens: Some(resp.completion_tokens),
                    elapsed_ms: Some(resp.elapsed_ms),
                    transport: Some(resp.transport.clone()),
                    error: None,
                },
            );
            Ok(resp)
        }
        Err(e) => {
            let err_str = format!("{e:#}");
            let _ = handle.emit(
                "llm-chat-event",
                LlmChatStreamEvent {
                    request_id: req.request_id.clone(),
                    kind: "error".into(),
                    vm_name: target.vm_name,
                    slot: target.slot,
                    model: target.model,
                    delta: None,
                    ttft_ms: None,
                    instant_tps: None,
                    avg_tps: None,
                    completion_tokens: None,
                    elapsed_ms: None,
                    transport: None,
                    error: Some(err_str.clone()),
                },
            );
            Err(err_str)
        }
    }
}

// ───────────────────────────── 启动 ─────────────────────────────

fn resolve_workspace(app: &tauri::App) -> PathBuf {
    if let Ok(p) = std::env::var("COLAB_PILOT_WORKSPACE") {
        let p = PathBuf::from(p);
        if p.is_dir() {
            return p;
        }
    }
    #[cfg(debug_assertions)]
    if let Some(dev) = Path::new(env!("CARGO_MANIFEST_DIR")).parent() {
        if dev.join("train.py").exists() {
            return dev.to_path_buf();
        }
    }
    let dir = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir())
        .join("workspace");
    let _ = std::fs::create_dir_all(&dir);
    if !dir.join("train.py").exists() {
        let _ = std::fs::write(dir.join("train.py"), DEFAULT_TRAIN_PY);
    }
    if !dir.join("requirements.txt").exists() {
        let _ = std::fs::write(
            dir.join("requirements.txt"),
            "# 每行一个云端需额外安装的 Python 包\n",
        );
    }
    dir
}

/// 无界面诊断模式：`colab-pilot.exe --print-overview` 输出一次后端总览 JSON 后退出
fn print_overview_headless() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let workspace = std::env::var("COLAB_PILOT_WORKSPACE")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
            .or_else(|| {
                #[cfg(debug_assertions)]
                {
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|p| p.to_path_buf())
                        .filter(|p| p.is_dir())
                }
                #[cfg(not(debug_assertions))]
                {
                    None
                }
            })
            .unwrap_or_else(|| cli::colab_config_dir().join("workspace"));
        let state = AppState::new_with(ColabClient::new(), workspace);
        let ov = refresh_overview(&state, true).await;
        println!("{}", serde_json::to_string_pretty(&ov).unwrap_or_default());
    });
}

#[cfg(debug_assertions)]
fn selftest_lifecycle() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let state = AppState::new_with(
            ColabClient::new(),
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .to_path_buf(),
        );
        let name = "selftest-cpu";
        println!("[1] creating {name} …");
        if let Err(e) = relay::colab_new(
            name,
            "CPU",
            false,
            None,
            None,
            &|l: &str| println!("    | {l}"),
        )
        .await
        {
            println!("CREATE FAILED: {e}");
            return;
        }
        println!("[2] waiting for auto-probe trigger …");
        let _ = refresh_overview(&state, false).await;
        tokio::time::sleep(Duration::from_secs(7)).await;
        let ov = refresh_overview(&state, false).await;
        for vm in &ov.vms {
            let t = vm.telemetry.as_ref();
            println!(
                "[3] vm={} ws={} on_server={} cpu%={:?} mem%={:?} threads={:?} age={:?}",
                vm.name,
                vm.workspace.name,
                vm.on_server,
                t.and_then(|t| t["cpu"]["percent"].as_f64()),
                t.and_then(|t| t["memory"]["percent"].as_f64()),
                t.and_then(|t| t["threads"]["total_threads"].as_i64()),
                vm.telemetry_age,
            );
        }
        println!("[4] hard terminate …");
        let r = hard_terminate(&state, name, None, true).await;
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
    });
}

fn append_crash_log(text: &str) {
    let path = cli::colab_config_dir().join("colab-pilot-crash.log");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        use std::io::Write;
        let _ = f.write_all(text.as_bytes());
    }
}

/// 全局 panic 钩子：任何线程 panic 都会追加写入 `~/.config/colab-cli/colab-pilot-crash.log`，便于定位“闪退”
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>").to_string();
        append_crash_log(&format!(
            "[{}] thread '{}' panicked: {}\nbacktrace:\n{}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
            name,
            info,
            std::backtrace::Backtrace::force_capture()
        ));
        default_hook(info);
    }));
}

// ───────────────────────────── 单实例守护 ─────────────────────────────
// 修复“双击 exe 闪退”并加固单实例 IPC 安全（H-3）：
// - 使用每用户私有密钥文件 `~/.config/colab-cli/.pilot_instance_token` 双向鉴权；
// - 限制每个连接最多读取 512 字节，杜绝无换行长报文导致内存耗尽；
// - 若旧实例正在执行后台接力训练任务，永不返回 NOWIN 自毁；
// - 新实例在终止无窗口僵尸旧实例前，强制校验目标 PID 的可执行文件路径与当前进程一致。

const SINGLE_INSTANCE_PORT: u16 = 18079;
const INTERACTIVE_DESKTOP: &str = "WinSta0\\Default";

fn single_instance_token_path() -> PathBuf {
    cli::colab_config_dir().join(".pilot_instance_token")
}

fn read_single_instance_token() -> Option<String> {
    let raw = std::fs::read_to_string(single_instance_token_path()).ok()?;
    let trimmed = raw.trim().to_string();
    if trimmed.len() >= 16 && trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
        Some(trimmed)
    } else {
        None
    }
}

fn ensure_single_instance_token() -> String {
    if let Some(existing) = read_single_instance_token() {
        return existing;
    }
    let tok = uuid::Uuid::new_v4().simple().to_string();
    let path = single_instance_token_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
        {
            use std::io::Write;
            let _ = f.write_all(tok.as_bytes());
            return tok;
        }
    }
    let _ = std::fs::write(&path, &tok);
    tok
}

/// 当前线程所在的 “窗口站\桌面” 名称（例如用户交互桌面 `WinSta0\Default`）
#[cfg(windows)]
fn current_desktop_name() -> String {
    #[link(name = "user32")]
    extern "system" {
        fn GetThreadDesktop(thread_id: u32) -> isize;
        fn GetProcessWindowStation() -> isize;
        fn GetUserObjectInformationW(
            h: isize,
            index: i32,
            pv: *mut u16,
            len: u32,
            needed: *mut u32,
        ) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> u32;
    }
    const UOI_NAME: i32 = 2;
    unsafe fn obj_name(h: isize) -> String {
        if h == 0 {
            return String::new();
        }
        let mut buf = [0u16; 256];
        let mut need = 0u32;
        if GetUserObjectInformationW(
            h,
            UOI_NAME,
            buf.as_mut_ptr(),
            (buf.len() * 2) as u32,
            &mut need,
        ) == 0
        {
            return String::new();
        }
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        String::from_utf16_lossy(&buf[..len])
    }
    unsafe {
        format!(
            "{}\\{}",
            obj_name(GetProcessWindowStation()),
            obj_name(GetThreadDesktop(GetCurrentThreadId()))
        )
    }
}

#[cfg(not(windows))]
fn current_desktop_name() -> String {
    String::new()
}

enum InstanceRole {
    Primary(std::net::TcpListener, String),
    AlreadyRunning,
    NoLock,
}

fn kill_process_tree(pid: u32) {
    if pid == 0 || pid == std::process::id() {
        return;
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let _ = std::process::Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .creation_flags(0x0800_0000)
            .status();
    }
    #[cfg(not(windows))]
    {
        let pid_str = pid.to_string();
        let _ = std::process::Command::new("pkill")
            .args(["-TERM", "-P", &pid_str])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let _ = std::process::Command::new("kill")
            .args(["-TERM", &pid_str])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        std::thread::sleep(Duration::from_millis(120));
        let _ = std::process::Command::new("kill")
            .args(["-KILL", &pid_str])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

fn acquire_single_instance() -> InstanceRole {
    use std::io::{BufRead, BufReader, Read, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], SINGLE_INSTANCE_PORT));
    let my_desktop = current_desktop_name();
    for _ in 0..3 {
        if let Ok(l) = std::net::TcpListener::bind(addr) {
            let tok = ensure_single_instance_token();
            return InstanceRole::Primary(l, tok);
        }
        let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(800))
        else {
            std::thread::sleep(Duration::from_millis(400));
            continue;
        };
        let tok = read_single_instance_token().unwrap_or_default();
        let _ = s.set_read_timeout(Some(Duration::from_millis(2500)));
        let _ = s.write_all(format!("SHOW {tok}\t{my_desktop}\n").as_bytes());
        let mut line = String::new();
        let _ = BufReader::new((&s).take(512)).read_line(&mut line);
        let line = line.trim();
        if line.starts_with("OK") {
            return InstanceRole::AlreadyRunning;
        }
        if let Some(rest) = line.strip_prefix("NOWIN ") {
            let mut parts = rest.split_whitespace();
            let resp_tok = parts.next().unwrap_or("");
            let pid_opt = parts.next().and_then(|p| p.parse::<u32>().ok());
            if !tok.is_empty() && resp_tok == tok {
                if let Some(pid) = pid_opt {
                    if pid != std::process::id() && is_same_executable_pid(pid) {
                        append_crash_log(&format!(
                            "[{}] 检测到已验明身份的无窗口僵尸实例 pid={pid}，已结束并接管启动\n",
                            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                        ));
                        kill_process_tree(pid);
                        std::thread::sleep(Duration::from_millis(900));
                        continue;
                    }
                }
            }
        }
        // 端口被非 Colab Pilot 程序占用、令牌不匹配或对方无响应：放弃单实例锁，照常启动
        return InstanceRole::NoLock;
    }
    InstanceRole::NoLock
}

fn serve_single_instance(
    listener: std::net::TcpListener,
    handle: AppHandle,
    state: Arc<AppState>,
    expected_token: String,
) {
    use std::io::{BufRead, BufReader, Read, Write};
    let my_desktop = current_desktop_name();
    let i_am_interactive = my_desktop.eq_ignore_ascii_case(INTERACTIVE_DESKTOP);
    std::thread::Builder::new()
        .name("single-instance".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let _ = s.set_read_timeout(Some(Duration::from_millis(1500)));
                let mut line = String::new();
                let _ = BufReader::new((&s).take(512)).read_line(&mut line);
                let line = line.trim();
                let Some(rest) = line
                    .get(..4)
                    .filter(|p| p.eq_ignore_ascii_case("SHOW"))
                    .map(|_| line[4..].trim())
                else {
                    continue;
                };
                let (req_tok, requester_desktop) = match rest.split_once('\t') {
                    Some((t, d)) => (t.trim(), d.trim()),
                    None => (rest.trim(), ""),
                };
                if expected_token.is_empty() || req_tok != expected_token {
                    let _ = s.write_all(b"DENIED\n");
                    continue;
                }

                let has_active_relay = state.relays.l().values().any(|h| {
                    h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst)
                });
                let same_desktop = requester_desktop.is_empty()
                    || requester_desktop.eq_ignore_ascii_case(&my_desktop);

                let reply = if has_active_relay {
                    // 后台接力监工运行中：尝试恢复窗口，绝不返回 NOWIN 杀死自己
                    if let Some(w) = handle.get_webview_window("main") {
                        let _ = w.unminimize();
                        let _ = w.show();
                        let _ = w.set_focus();
                    }
                    "OK\n".to_string()
                } else if !same_desktop && !i_am_interactive {
                    format!("NOWIN {expected_token} {}\n", std::process::id())
                } else if !same_desktop {
                    let _ = handle.get_webview_window("main").map(|w| w.set_focus());
                    "OK\n".to_string()
                } else {
                    let visible = match handle.get_webview_window("main") {
                        Some(w) => {
                            let _ = w.unminimize();
                            let _ = w.show();
                            let _ = w.set_focus();
                            w.is_visible().unwrap_or(false)
                        }
                        None => false,
                    };
                    if visible {
                        "OK\n".to_string()
                    } else {
                        format!("NOWIN {expected_token} {}\n", std::process::id())
                    }
                };
                let _ = s.write_all(reply.as_bytes());
            }
        })
        .ok();
}

// ───────────────────── 窗口尺寸与位置历史记忆 ─────────────────────
// 默认初始化保持最小尺寸（780 × 520 逻辑像素），当用户拖拽改变窗口大小、移动位置或最大化后，
// 自动持久化到 `~/.config/colab-cli/window_state.json`，下次启动自动恢复并做显示器边界安全钳制。

const MIN_WINDOW_WIDTH: f64 = 780.0;
const MIN_WINDOW_HEIGHT: f64 = 520.0;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WindowStatePrefs {
    width: f64,
    height: f64,
    #[serde(default)]
    x: Option<i32>,
    #[serde(default)]
    y: Option<i32>,
    #[serde(default)]
    maximized: bool,
}

impl Default for WindowStatePrefs {
    fn default() -> Self {
        Self {
            width: MIN_WINDOW_WIDTH,
            height: MIN_WINDOW_HEIGHT,
            x: None,
            y: None,
            maximized: false,
        }
    }
}

struct WindowStateTracker {
    prefs: WindowStatePrefs,
    dirty: bool,
    scheduled: bool,
    last_saved: Instant,
    ignore_until: Instant,
}

fn window_state_path() -> PathBuf {
    cli::colab_config_dir().join("window_state.json")
}

fn load_window_state() -> Option<WindowStatePrefs> {
    let raw = std::fs::read_to_string(window_state_path()).ok()?;
    let prefs: WindowStatePrefs = serde_json::from_str(&raw).ok()?;
    if !prefs.width.is_finite() || !prefs.height.is_finite() {
        return None;
    }
    Some(prefs)
}

fn save_window_state(prefs: &WindowStatePrefs) {
    let path = window_state_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(prefs) {
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, &json).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        } else {
            let _ = std::fs::write(&path, json);
        }
    }
}

fn monitor_logical_bounds(m: &tauri::Monitor) -> (f64, f64) {
    let wa = m.work_area();
    let s = wa.size.to_logical::<f64>(m.scale_factor());
    (
        (s.width - 16.0).max(MIN_WINDOW_WIDTH),
        (s.height - 16.0).max(MIN_WINDOW_HEIGHT),
    )
}

fn restore_window_state(window: &tauri::WebviewWindow) -> Arc<Mutex<WindowStateTracker>> {
    let prefs = if let Some(mut saved) = load_window_state() {
        let monitors = window.available_monitors().unwrap_or_default();
        let target_monitor = match (saved.x, saved.y) {
            (Some(px), Some(py)) => monitors.into_iter().find(|m| {
                let wa = m.work_area();
                let wp = wa.position;
                let ws = wa.size;
                let check_x = px.saturating_add(80);
                let check_y = py.saturating_add(20);
                py >= wp.y.saturating_sub(16)
                    && check_x >= wp.x
                    && check_x < wp.x.saturating_add(ws.width as i32)
                    && check_y >= wp.y
                    && check_y < wp.y.saturating_add(ws.height as i32)
            }),
            _ => None,
        };

        if let (Some(ref m), Some(px), Some(py)) = (&target_monitor, saved.x, saved.y) {
            // 按目标显示器工作区（扣除 Windows 任务栏）钳制窗口尺寸与坐标，避免标题栏或底边被任务栏遮挡
            let (max_w, max_h) = monitor_logical_bounds(m);
            saved.width = saved.width.clamp(MIN_WINDOW_WIDTH, max_w).round();
            saved.height = saved.height.clamp(MIN_WINDOW_HEIGHT, max_h).round();
            let wa = m.work_area();
            let scale = m.scale_factor();
            let phys_w = (saved.width * scale).round() as i32;
            let phys_h = (saved.height * scale).round() as i32;
            let max_x = wa
                .position
                .x
                .saturating_add(wa.size.width as i32)
                .saturating_sub(phys_w)
                .max(wa.position.x);
            let max_y = wa
                .position
                .y
                .saturating_add(wa.size.height as i32)
                .saturating_sub(phys_h)
                .max(wa.position.y);
            let clamped_x = px.clamp(wa.position.x, max_x);
            let clamped_y = py.clamp(wa.position.y, max_y);
            saved.x = Some(clamped_x);
            saved.y = Some(clamped_y);
            let _ = window.set_position(tauri::Position::Physical(
                tauri::PhysicalPosition::new(clamped_x, clamped_y),
            ));
            let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize::new(
                saved.width,
                saved.height,
            )));
        } else {
            let (max_w, max_h) = window
                .current_monitor()
                .ok()
                .flatten()
                .map(|m| monitor_logical_bounds(&m))
                .unwrap_or((1920.0, 1080.0));
            saved.width = saved.width.clamp(MIN_WINDOW_WIDTH, max_w).round();
            saved.height = saved.height.clamp(MIN_WINDOW_HEIGHT, max_h).round();
            let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize::new(
                saved.width,
                saved.height,
            )));
            let _ = window.center();
        }
        if saved.maximized {
            let _ = window.maximize();
        }
        saved
    } else {
        // 首次启动无历史记录：严格保持最小窗口尺寸并居中
        let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize::new(
            MIN_WINDOW_WIDTH,
            MIN_WINDOW_HEIGHT,
        )));
        let _ = window.center();
        WindowStatePrefs::default()
    };

    Arc::new(Mutex::new(WindowStateTracker {
        prefs,
        dirty: false,
        scheduled: false,
        last_saved: Instant::now(),
        ignore_until: Instant::now() + Duration::from_millis(550),
    }))
}

fn record_window_state(
    window: &tauri::WebviewWindow,
    tracker: &Arc<Mutex<WindowStateTracker>>,
    force_flush: bool,
) {
    if window.is_minimized().unwrap_or(false) || !window.is_visible().unwrap_or(false) {
        if force_flush {
            let to_save = {
                let mut g = tracker.l();
                if g.dirty {
                    g.dirty = false;
                    Some(g.prefs.clone())
                } else {
                    None
                }
            };
            if let Some(p) = to_save {
                save_window_state(&p);
            }
        }
        return;
    }

    let now = Instant::now();
    let (immediate_save, schedule_delayed) = {
        let mut g = tracker.l();
        if !force_flush && now < g.ignore_until {
            return;
        }
        let maximized = window.is_maximized().unwrap_or(false);
        g.prefs.maximized = maximized;
        if !maximized {
            let scale = window.scale_factor().unwrap_or(1.0);
            if let Ok(phys) = window.inner_size() {
                let logical = phys.to_logical::<f64>(scale);
                if logical.width >= MIN_WINDOW_WIDTH - 4.0
                    && logical.height >= MIN_WINDOW_HEIGHT - 4.0
                {
                    g.prefs.width = logical.width.max(MIN_WINDOW_WIDTH).round();
                    g.prefs.height = logical.height.max(MIN_WINDOW_HEIGHT).round();
                }
            }
            if let Ok(pos) = window.outer_position() {
                if pos.x > -16000 && pos.y > -16000 {
                    g.prefs.x = Some(pos.x);
                    g.prefs.y = Some(pos.y);
                }
            }
        }
        g.dirty = true;
        if force_flush || g.last_saved.elapsed() >= Duration::from_millis(280) {
            g.dirty = false;
            g.last_saved = now;
            (Some(g.prefs.clone()), false)
        } else if !g.scheduled {
            g.scheduled = true;
            (None, true)
        } else {
            (None, false)
        }
    };

    if let Some(p) = immediate_save {
        if force_flush {
            save_window_state(&p);
        } else {
            let _ = tauri::async_runtime::spawn_blocking(move || save_window_state(&p));
        }
    } else if schedule_delayed {
        let tracker_bg = tracker.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_millis(320)).await;
            let to_save = {
                let mut g = tracker_bg.l();
                g.scheduled = false;
                if g.dirty {
                    g.dirty = false;
                    g.last_saved = Instant::now();
                    Some(g.prefs.clone())
                } else {
                    None
                }
            };
            if let Some(p) = to_save {
                let _ = tokio::task::spawn_blocking(move || save_window_state(&p)).await;
            }
        });
    }
}

fn shutdown_child_processes(app: &AppState) {
    for entry in app.active_chats.l().values() {
        entry.cancel.cancel();
    }
    for bridge in app.ssh_bridges.l().values() {
        bridge.shutdown();
    }
    let handles: Vec<Arc<RelayHandle>> = app.relays.l().values().cloned().collect();
    for h in handles {
        h.keep_vm_on_stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        h.cancel.l().cancel();
        let pid = *h.child_pid.l();
        if let Some(p) = pid {
            kill_process_tree(p);
        }
    }
}

pub(crate) fn active_relay_vm_names(app: &AppState) -> Vec<String> {
    let mut set: std::collections::BTreeSet<String> = app
        .relays
        .l()
        .iter()
        .filter(|(_, h)| h.is_running())
        .map(|(k, _)| k.clone())
        .collect();
    for pending in app.pending_close_vms.l().iter() {
        set.insert(pending.clone());
    }
    set.into_iter().collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowCloseInterception {
    pub active_vms: Vec<String>,
    pub unsaved_script: bool,
}

pub(crate) fn evaluate_window_close_interception(
    app: &AppState,
) -> Option<WindowCloseInterception> {
    if app
        .force_close_confirmed
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return None;
    }
    let active_vms = should_intercept_window_close(app).unwrap_or_default();
    let unsaved_script = app.script_dirty.load(std::sync::atomic::Ordering::SeqCst);
    if active_vms.is_empty() && !unsaved_script {
        None
    } else {
        Some(WindowCloseInterception {
            active_vms,
            unsaved_script,
        })
    }
}

pub(crate) fn should_intercept_window_close(app: &AppState) -> Option<Vec<String>> {
    if app
        .force_close_confirmed
        .load(std::sync::atomic::Ordering::SeqCst)
    {
        return None;
    }
    let active = active_relay_vm_names(app);
    if active.is_empty() {
        None
    } else {
        Some(active)
    }
}

#[tauri::command]
fn set_script_dirty(state: AppStateRef<'_>, dirty: bool) {
    state
        .inner()
        .script_dirty
        .store(dirty, std::sync::atomic::Ordering::SeqCst);
}

pub(crate) async fn confirm_app_close_inner(
    app: &Arc<AppState>,
    mode: &str,
) -> Result<(Vec<TerminateReceipt>, bool), String> {
    let m = mode.trim().to_ascii_lowercase();
    if m == "force_exit" {
        app.script_dirty
            .store(false, std::sync::atomic::Ordering::SeqCst);
        if should_intercept_window_close(app).is_some() {
            return Ok((Vec::new(), false));
        }
        app.force_close_confirmed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok((Vec::new(), true))
    } else if m == "keep_vms" {
        let handles: Vec<Arc<RelayHandle>> = app.relays.l().values().cloned().collect();
        for h in handles {
            if h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst) {
                h.keep_vm_on_stop
                    .store(true, std::sync::atomic::Ordering::SeqCst);
                h.stop().await;
                {
                    let mut st = h.status.l();
                    st.running = false;
                    st.phase = "stopped".into();
                    st.phase_label = "软件退出：已停止本机接力监工（云端实例保留运行中）".into();
                }
                h.log("⏹ 软件退出：已停止本机接力监工，保留云端实例继续运行");
            }
        }
        app.pending_close_vms.l().clear();
        app.script_dirty
            .store(false, std::sync::atomic::Ordering::SeqCst);
        app.force_close_confirmed
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok((Vec::new(), true))
    } else if m == "shutdown_vms" {
        let targets = active_relay_vm_names(app);
        if targets.is_empty() {
            app.pending_close_vms.l().clear();
            app.script_dirty
                .store(false, std::sync::atomic::Ordering::SeqCst);
            app.force_close_confirmed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            return Ok((Vec::new(), true));
        }
        let mut receipts = Vec::with_capacity(targets.len());
        for vm_name in targets {
            app.set_busy(&vm_name, Some("deleting"));
            if let Some(h) = app.relay(&vm_name) {
                h.keep_vm_on_stop
                    .store(false, std::sync::atomic::Ordering::SeqCst);
                h.stop().await;
                h.log("🧹 退出前正在销毁云端实例并核验停止计费…");
            }
            let r = hard_terminate(app, &vm_name, None, false).await;
            app.release_busy(&vm_name, "deleting");
            if r.verified_unassigned {
                app.relays.l().remove(&vm_name);
            }
            receipts.push(r);
        }

        let unverified: Vec<String> = receipts
            .iter()
            .filter(|r| !r.verified_unassigned)
            .map(|r| r.name.clone())
            .collect();
        {
            let mut pending = app.pending_close_vms.l();
            pending.clear();
            for n in &unverified {
                pending.insert(n.clone());
            }
        }
        if unverified.is_empty() {
            app.script_dirty
                .store(false, std::sync::atomic::Ordering::SeqCst);
            app.force_close_confirmed
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok((receipts, true))
        } else {
            Ok((receipts, false))
        }
    } else {
        Err(format!(
            "不支持的窗口关闭确认模式 `{mode}`（仅支持 keep_vms、shutdown_vms 或 force_exit）"
        ))
    }
}

#[tauri::command]
async fn confirm_app_close(
    handle: AppHandle,
    state: AppStateRef<'_>,
    mode: String,
) -> Result<Vec<TerminateReceipt>, String> {
    let app = state.inner().clone();
    let (receipts, should_exit) = confirm_app_close_inner(&app, &mode).await?;
    if should_exit {
        shutdown_child_processes(&app);
        if let Some(w) = handle.get_webview_window("main") {
            let _ = w.close();
        }
        handle.exit(0);
    } else {
        if mode.trim().eq_ignore_ascii_case("force_exit") {
            if let Some(active_vms) = should_intercept_window_close(&app) {
                let _ = handle.emit(
                    "close-requested-with-relay",
                    serde_json::json!({ "active_vms": active_vms }),
                );
            }
        }
        let ov = refresh_overview(&app, true).await;
        emit_overview(&handle, &ov);
    }
    Ok(receipts)
}

pub(crate) fn dismiss_relay_status_inner(app: &AppState, name: &str) -> Result<(), String> {
    let target = name.trim();
    if target.is_empty() {
        return Err("实例名称不能为空".into());
    }
    let mut map = app.relays.l();
    if let Some(h) = map.get(target) {
        if h.is_running() || h.task_alive.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(format!(
                "接力任务 [{target}] 仍在运行中，请先停止接力后再清除记录"
            ));
        }
        map.remove(target);
    }
    app.pending_close_vms.l().remove(target);
    Ok(())
}

#[tauri::command]
async fn dismiss_relay_status(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
) -> Result<(), String> {
    let app = state.inner().clone();
    dismiss_relay_status_inner(&app, &name)?;
    let ov = refresh_overview(&app, false).await;
    emit_overview(&handle, &ov);
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    install_panic_hook();
    if std::env::args().any(|a| a == "--print-overview") {
        print_overview_headless();
        return;
    }
    #[cfg(debug_assertions)]
    if std::env::args().any(|a| a == "--selftest-lifecycle") {
        selftest_lifecycle();
        return;
    }
    // 确保开启系统代理（如 Clash 127.0.0.1:7890）时，WebView2 内部协议 http://tauri.localhost 永不走代理
    if std::env::var_os("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS").is_none() {
        std::env::set_var(
            "WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS",
            "--proxy-bypass-list=<-loopback>;*.localhost;tauri.localhost",
        );
    }
    let instance_listener = match acquire_single_instance() {
        InstanceRole::AlreadyRunning => return,
        InstanceRole::Primary(l, tok) => Some((l, tok)),
        InstanceRole::NoLock => None,
    };
    let instance_listener = Mutex::new(instance_listener);
    let win_tracker_slot: Arc<Mutex<Option<Arc<Mutex<WindowStateTracker>>>>> =
        Arc::new(Mutex::new(None));
    let win_tracker_setup = win_tracker_slot.clone();

    let built = tauri::Builder::default()
        .setup(move |app| {
            if let Some(w) = app.get_webview_window("main") {
                let tracker = restore_window_state(&w);
                *win_tracker_setup.l() = Some(tracker);
            }
            let state = AppState::new_with(ColabClient::new(), resolve_workspace(app));
            let ws_skills = state.workspace.clone();
            let _ = tauri::async_runtime::spawn_blocking(move || {
                let _ = ensure_skill_scripts(&ws_skills);
            });
            app.manage(state.clone());

            if let Some((l, tok)) = instance_listener.l().take() {
                serve_single_instance(l, app.handle().clone(), state.clone(), tok);
            }

            // 后台高频轮询：每 1.5 秒同步云端资源遥测与实时 tok/s（计费列表每 8 秒刷新），并推送前端
            // 每轮在独立任务中执行：单轮 panic 不会终止整个轮询；出错时指数退避（1.5s → 30s），成功即恢复
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                const BASE_MS: u64 = 1500;
                const MAX_MS: u64 = 30_000;
                let mut delay_ms = BASE_MS;
                loop {
                    let st = state.clone();
                    let h = handle.clone();
                    let res = tauri::async_runtime::spawn(async move {
                        let ov = refresh_overview(&st, false).await;
                        emit_overview(&h, &ov);
                        ov.error.is_some() || ov.is_stale
                    })
                    .await;
                    delay_ms = match res {
                        Ok(false) => BASE_MS,
                        Ok(true) => (delay_ms * 2).min(MAX_MS),
                        Err(e) => {
                            append_crash_log(&format!(
                                "[{}] 后台总览刷新任务异常（已自动恢复轮询）: {e}\n",
                                chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                            ));
                            (delay_ms * 2).min(MAX_MS)
                        }
                    };
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
            });

            // 后台每实例保活调度器：每 10 秒检查一次各在线实例是否到达保活周期（默认 90s），
            // 并发独立派发，不等待慢实例，避免推迟其他实例下一轮调度（#100）。
            let ka_state = app.state::<Arc<AppState>>().inner().clone();
            let ka_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let mut ticker = tokio::time::interval(Duration::from_secs(10));
                ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                ticker.tick().await;
                loop {
                    ticker.tick().await;
                    let st = ka_state.clone();
                    let h = ka_handle.clone();
                    let res = tauri::async_runtime::spawn(async move {
                        dispatch_keep_alive_scheduler_pass_with(&st, move |app_done| {
                            let h_done = h.clone();
                            async move {
                                let ov = refresh_overview_opts(&app_done, false, false).await;
                                emit_overview(&h_done, &ov);
                            }
                        })
                        .await;
                    })
                    .await;
                    if let Err(e) = res {
                        append_crash_log(&format!(
                            "[{}] 后台实例保活调度任务异常（已自动恢复）: {e}\n",
                            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                        ));
                    }
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_overview,
            refresh_now,
            create_vm,
            cancel_create_vm,
            delete_vm,
            deploy_probe,
            start_relay,
            stop_relay,
            read_workspace_files,
            save_workspace_files,
            set_script_dirty,
            stream_llm_chat,
            cancel_llm_chat,
            open_vm_ssh_terminal,
            rotate_llm_api_key,
            deploy_llm_on_vm,
            confirm_app_close,
            dismiss_relay_status,
            set_vm_keep_alive,
            ping_vm_keep_alive
        ])
        .build(tauri::generate_context!());
    match built {
        Ok(app) => app.run(move |handle, event| {
            if let tauri::RunEvent::WindowEvent { label, event, .. } = &event {
                if label == "main" {
                    match event {
                        tauri::WindowEvent::Resized(_) | tauri::WindowEvent::Moved(_) => {
                            if let (Some(w), Some(tr)) = (
                                handle.get_webview_window("main"),
                                win_tracker_slot.l().clone(),
                            ) {
                                record_window_state(&w, &tr, false);
                            }
                        }
                        tauri::WindowEvent::CloseRequested { api, .. } => {
                            if let (Some(w), Some(tr)) = (
                                handle.get_webview_window("main"),
                                win_tracker_slot.l().clone(),
                            ) {
                                record_window_state(&w, &tr, true);
                            }
                            if let Some(st) = handle.try_state::<Arc<AppState>>() {
                                if let Some(intercept) =
                                    evaluate_window_close_interception(st.inner())
                                {
                                    api.prevent_close();
                                    if intercept.unsaved_script {
                                        let _ = handle.emit(
                                            "close-requested-unsaved-script",
                                            serde_json::json!({
                                                "active_vms": intercept.active_vms,
                                                "unsaved_script": true,
                                            }),
                                        );
                                    } else {
                                        let _ = handle.emit(
                                            "close-requested-with-relay",
                                            serde_json::json!({
                                                "active_vms": intercept.active_vms,
                                            }),
                                        );
                                    }
                                }
                            }
                        }
                        // 主窗口被销毁时立即刷盘、结束所有本机派生子进程树并退出进程，绝不残留僵尸进程
                        tauri::WindowEvent::Destroyed => {
                            if let Some(tr) = win_tracker_slot.l().clone() {
                                let to_save = {
                                    let mut g = tr.l();
                                    if g.dirty {
                                        g.dirty = false;
                                        Some(g.prefs.clone())
                                    } else {
                                        None
                                    }
                                };
                                if let Some(p) = to_save {
                                    save_window_state(&p);
                                }
                            }
                            if let Some(st) = handle.try_state::<Arc<AppState>>() {
                                shutdown_child_processes(st.inner());
                            }
                            handle.exit(0);
                        }
                        _ => {}
                    }
                }
            }
        }),
        Err(e) => {
            append_crash_log(&format!(
                "[{}] Colab Pilot 启动失败: {e:?}\n",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn test_bluegreen_tunnel_port_for_vm_strict_prefix_isolation() {
        let bg = parse_llm_bluegreen_state(Some(serde_json::json!({ "session_prefix": "qwen" })))
            .unwrap();
        assert_eq!(bluegreen_tunnel_port_for_vm("qwen-blue", Some(&bg)), Some(18081));
        assert_eq!(bluegreen_tunnel_port_for_vm("qwen-green", Some(&bg)), Some(18082));
        assert_eq!(bluegreen_tunnel_port_for_vm("llm-blue", Some(&bg)), None);
        assert_eq!(bluegreen_tunnel_port_for_vm("other-qwen-blue", Some(&bg)), None);
        assert_eq!(bluegreen_tunnel_port_for_vm("qwen-blue", None), None);
    }

    #[test]
    fn test_orphan_name_avoids_suffix_collision_and_matches_legacy() {
        let ep1 = "m-s-alpha-123456";
        let ep2 = "m-s-bravo-123456";
        let a = orphan_name(ep1);
        let b = orphan_name(ep2);
        assert!(a.starts_with("orphan-123456-"));
        assert!(b.starts_with("orphan-123456-"));
        assert_ne!(a, b);
        assert!(matches_orphan_name(ep1, &a));
        assert!(matches_orphan_name(ep1, "orphan-123456"));
        assert!(!matches_orphan_name(ep1, &b));
    }

    #[test]
    fn test_validate_pip_packages_and_engine_normalization() {
        assert_eq!(normalize_llm_engine("llamacpp"), "llama.cpp");
        assert_eq!(normalize_llm_engine("llama.cpp"), "llama.cpp");
        assert_eq!(normalize_llm_engine("vLLM"), "vllm");
        assert!(validate_pip_packages("vllm transformers[torch]>=4.46.0").is_ok());
        assert!(validate_pip_packages("--extra-index-url https://evil.example").is_err());
        assert!(validate_pip_packages("-e /tmp/pkg").is_err());
        assert!(parse_env_vars("HF_HUB_ENABLE_HF_TRANSFER=1\nexport WANDB_DISABLED=\"true\"").is_ok());
        assert!(parse_env_vars("LD_PRELOAD=/tmp/bad.so").is_err());
    }

    #[test]
    fn test_extract_sse_piece_variants() {
        assert_eq!(
            extract_sse_piece(r#"data:{"choices":[{"delta":{"reasoning_content":"思考"}}]}"#.as_bytes()),
            Some("思考".to_string())
        );
        assert_eq!(
            extract_sse_piece(r#"data: {"choices":[{"delta":{"content":"你好"}}]}"#.as_bytes()),
            Some("你好".to_string())
        );
        assert_eq!(extract_sse_piece(b"data: [DONE]\n"), None);
        assert_eq!(extract_sse_piece(b": keep-alive\n"), None);
        let err_frame = parse_sse_frame(br#"data: {"error":{"message":"CUDA out of memory"}}"#);
        assert!(err_frame.is_err());
        assert!(err_frame.unwrap_err().contains("CUDA out of memory"));
    }

    #[test]
    fn test_embedded_python_bridges_syntax_and_websocket_client_available() {
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let check_code = r#"
import sys, websocket
for idx, src in enumerate(sys.argv[1:]):
    compile(src, f"<embedded_{idx}>", "exec")
print("SYNTAX_AND_WS_OK")
"#;
        let out = std::process::Command::new(&py)
            .args([
                "-B",
                "-c",
                check_code,
                WS_SSH_STDIO_BRIDGE_PY,
                INTERACTIVE_TTY_CONSOLE_PY,
                SILENT_PROBE_LAUNCHER_PY,
                REMOTE_LLM_CHAT_STREAM_PY,
            ])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .expect("run colab python syntax check");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success() && stdout.contains("SYNTAX_AND_WS_OK"),
            "embedded python bridge check failed: stdout={stdout} stderr={stderr}"
        );
    }

    #[tokio::test]
    async fn test_resolve_llm_target_refuses_bare_vm_and_filters_stale_bluegreen() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_llm_route_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("nonexistent_colab"));
        *cli::TEST_RCLONE_CONF.l() = Some(tmp.join("nonexistent_rclone.conf"));

        // 写入一份数天前残留的陈旧 serving 状态（对应 llm-blue 早已下线）
        std::fs::write(
            cfg_dir.join("llm_bluegreen_state.json"),
            serde_json::json!({
                "session_prefix": "llm",
                "active_slot": "blue",
                "active_session": "llm-blue",
                "phase": "serving",
                "model_name": "stale-model",
                "updated_at": 1700000000.0
            })
            .to_string(),
        )
        .unwrap();

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            serde_json::json!({
                "token": "valid-token",
                "expiry": "2099-01-01T00:00:00Z"
            })
            .to_string(),
        )
        .unwrap();
        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let app = AppState::new_with(client, ws_dir);

        // 1. 陈旧蓝绿状态必须被过滤为 None
        let filtered = filter_live_bluegreen_state(
            parse_llm_bluegreen_state(cli::read_llm_bluegreen_state()),
            &["general-dev".to_string()],
            &HashMap::new(),
            &[],
            1794736000.0,
        );
        assert!(filtered.is_none(), "stale serving bluegreen state must be filtered out");

        // 2. 仅有一台普通通用实例在线时，target_vm = "auto" 绝不能劫持该通用实例
        {
            let mut ov = app.overview.write().await;
            ov.vms = vec![VmView {
                name: "general-dev".into(),
                on_server: true,
                managed: true,
                accelerator: "G4".into(),
                ..Default::default()
            }];
        }
        let auto_req = LlmChatRequest {
            request_id: "req-1".into(),
            target_vm: "auto".into(),
            model: None,
            messages: vec![],
            max_tokens: None,
            temperature: None,
        };
        let err = resolve_llm_target(&app, &auto_req)
            .await
            .expect_err("auto target must refuse bare non-LLM VM");
        assert!(
            err.to_string().contains("尚未部署大模型推理引擎"),
            "unexpected error: {err}"
        );

        // 3. 当在线实例包含 LLM API 信息时，target_vm = "auto" 正常命中并携带专属端口与 API Key
        let api_info = cli::resolve_or_create_llm_api_info(
            "qwen-srv",
            "standalone",
            "vllm",
            0,
            8000,
            "Qwen/Qwen2.5-7B-Instruct",
        );
        {
            let mut ov = app.overview.write().await;
            ov.vms.push(VmView {
                name: "qwen-srv".into(),
                on_server: true,
                managed: true,
                accelerator: "G4".into(),
                llm_api: Some(api_info.clone()),
                ..Default::default()
            });
        }
        let resolved = resolve_llm_target(&app, &auto_req)
            .await
            .expect("should resolve online LLM VM");
        assert_eq!(resolved.vm_name, "qwen-srv");
        assert_eq!(resolved.local_tunnel_port, Some(api_info.local_port));
        assert_eq!(resolved.api_key.as_deref(), Some(api_info.api_key.as_str()));
        assert_eq!(resolved.model, "Qwen/Qwen2.5-7B-Instruct");

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_hard_terminate_c1_endpoint_none_and_m3_sixth_retry() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_lib_test_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        std::fs::write(cfg_dir.join("sessions.json"), "{}").unwrap();

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("nonexistent_colab_bin"));

        let has_rogue = Arc::new(AtomicBool::new(true));
        let post_unassign_count = Arc::new(AtomicUsize::new(0));
        let six_alive = Arc::new(AtomicBool::new(false));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let has_rogue_srv = has_rogue.clone();
        let post_cnt_srv = post_unassign_count.clone();
        let six_alive_srv = six_alive.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let hr = has_rogue_srv.clone();
                let pc = post_cnt_srv.clone();
                let sa = six_alive_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let body = if first.starts_with("GET /tun/m/assignments") {
                        let mut items = Vec::new();
                        if hr.load(Ordering::SeqCst) {
                            items.push(serde_json::json!({
                                "endpoint": "m-s-rogue999",
                                "accelerator": "T4",
                                "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1" }
                            }));
                        }
                        if sa.load(Ordering::SeqCst) {
                            items.push(serde_json::json!({
                                "endpoint": "m-s-six006",
                                "accelerator": "A100",
                                "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1" }
                            }));
                        }
                        format!(")]}}'\n{}", serde_json::json!({ "assignments": items }))
                    } else if first.starts_with("GET /tun/m/unassign/") {
                        r#"{"token":"xsrf-tok"}"#.to_string()
                    } else if first.starts_with("POST /tun/m/unassign/m-s-six006") {
                        let c = pc.fetch_add(1, Ordering::SeqCst) + 1;
                        if c >= 6 {
                            sa.store(false, Ordering::SeqCst);
                        }
                        "{}".to_string()
                    } else if first.starts_with("GET /tun/m/ccu-info") {
                        r#"{"currentBalance":99.5}"#.to_string()
                    } else {
                        "{}".to_string()
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            serde_json::json!({
                "token": "valid-token",
                "expiry": "2099-01-01T00:00:00Z",
                "refresh_token": "rt",
                "client_id": "cid",
                "client_secret": "csec"
            })
            .to_string(),
        )
        .unwrap();

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir);

        // 1. C-1: endpoint is None while an unaccounted VM is billing on server -> MUST NOT claim verified_unassigned!
        let r1 = hard_terminate(&app, "provisioning-vm", None, true).await;
        assert!(!r1.verified_unassigned, "must refuse false verified_unassigned when unaccounted VM exists");
        assert_eq!(r1.remaining_endpoints, vec!["m-s-rogue999".to_string()]);
        assert!(r1.error.is_some());

        // 2. C-1: once server truly has 0 VMs -> verified_unassigned == true
        has_rogue.store(false, Ordering::SeqCst);
        let r2 = hard_terminate(&app, "provisioning-vm", None, true).await;
        assert!(r2.verified_unassigned);
        assert!(r2.error.is_none());

        // 3. M-3: unassign succeeds only on the 6th attempt -> post-loop final check marks verified_unassigned == true
        six_alive.store(true, Ordering::SeqCst);
        let r3 = hard_terminate(&app, "six-vm", Some("m-s-six006".into()), false).await;
        assert_eq!(post_unassign_count.load(Ordering::SeqCst), 6);
        assert!(r3.verified_unassigned, "6th unassign should be verified by final list_assignments check");

        // 4. C-1: unknown VM name while another valid session is active on server -> MUST refuse verified_unassigned
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({
                "other-active": { "endpoint": "m-s-rogue999" }
            })
            .to_string(),
        )
        .unwrap();
        has_rogue.store(true, Ordering::SeqCst);
        let r4 = hard_terminate(&app, "typo-unknown-vm", None, false).await;
        assert!(
            !r4.verified_unassigned,
            "unknown VM must not report verified_unassigned while other server VMs are billing"
        );

        // 5. C-1: cancelling a relay mid-provisioning catches and unassigns newly leaked server VM
        six_alive.store(true, Ordering::SeqCst);
        post_unassign_count.store(5, Ordering::SeqCst); // next unassign on m-s-six006 will drop it
        {
            let h = Arc::new(RelayHandle::new("prov-relay"));
            h.pre_provision_endpoints
                .l()
                .insert("m-s-rogue999".to_string());
            h.status.l().running = true;
            h.status.l().phase = "provisioning".into();
            h.task_alive.store(true, Ordering::SeqCst);
            app.relays.l().insert("prov-relay".to_string(), h);
        }
        let r5 = hard_terminate(&app, "prov-relay", None, true).await;
        assert!(r5.verified_unassigned, "newly leaked provisioning VM should be caught and unassigned");
        assert_eq!(r5.endpoint.as_deref(), Some("m-s-six006"));
        assert!(!six_alive.load(Ordering::SeqCst));
        assert!(has_rogue.load(Ordering::SeqCst), "pre-existing other session VM must remain untouched");

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_orphan_ssh_bridge_identity_and_embedded_relay_adapter() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_ssh_skills_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let empty_ws = tmp.join("packaged_ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&empty_ws).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());

        // 1. External user workspace even with stale .agents/skills must NOT override embedded scripts;
        //    all 5 embedded scripts including colab_relay_adapter.py must be extracted into colab_config_dir()/skills/
        let stale_bg = empty_ws
            .join(".agents")
            .join("skills")
            .join("colab-llm-bluegreen")
            .join("scripts");
        std::fs::create_dir_all(&stale_bg).unwrap();
        for fname in [
            "llm_bluegreen_manager.py",
            "llm_local_gateway.py",
            "llm_metrics_sidecar.py",
        ] {
            std::fs::write(stale_bg.join(fname), "# STALE_2026_11_10_SCRIPT\n").unwrap();
        }
        let mgr = ensure_skill_scripts(&empty_ws).expect("ensure_skill_scripts must succeed");
        assert!(
            mgr.starts_with(&cfg_dir),
            "must resolve to embedded skills under colab_config_dir(), not stale user workspace: {}",
            mgr.display()
        );
        let mgr_body = std::fs::read_to_string(&mgr).unwrap();
        assert!(
            !mgr_body.contains("STALE_2026_11_10_SCRIPT"),
            "extracted manager script must match current embedded source"
        );
        assert!(mgr.with_file_name("llm_local_gateway.py").exists());
        assert!(mgr.with_file_name("llm_metrics_sidecar.py").exists());
        let relay_scripts = cfg_dir.join("skills").join("colab-dl-relay").join("scripts");
        assert!(relay_scripts.join("dl_relay_runner.py").exists());
        assert!(
            relay_scripts.join("colab_relay_adapter.py").exists(),
            "colab_relay_adapter.py must be extracted alongside dl_relay_runner.py"
        );

        // 1b. Workspace registry pre-write & rollback helper check
        assert!(!cli::has_workspace_entry("vm-failed-temp"));
        cli::register_workspace_entry("vm-failed-temp", "ws", Some("/tmp/ws"), None);
        assert!(cli::has_workspace_entry("vm-failed-temp"));
        cli::remove_workspace_entry("vm-failed-temp");
        assert!(!cli::has_workspace_entry("vm-failed-temp"));

        // 2. Valid orphan-* VM gets SSH bridge port, Host colab-orphan-* in ~/.ssh/config, and -i <pilot_key> when id_ed25519.pub is absent
        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let app = AppState::new_with(client, empty_ws);
        let orphan_assign: Assignment = serde_json::from_value(serde_json::json!({
            "endpoint": "m-s-orphan01",
            "accelerator": "G4",
            "variant": 1,
            "machineShape": 0,
            "runtimeProxyInfo": {
                "url": "https://8080-m-s-orphan01.prod.colab.dev",
                "token": "secret-tok",
                "tokenExpiresInSeconds": 3600
            }
        }))
        .unwrap();

        let port = ensure_vm_ssh_bridge(&app, "orphan-orphan01-1234", &orphan_assign)
            .await
            .expect("orphan-* with safe alias must receive local SSH bridge port");
        assert!(port > 0);
        cli::sync_ssh_config_hosts(&[("orphan-orphan01-1234".into(), port)]);
        let ssh_cfg = std::fs::read_to_string(cli::ssh_config_path()).unwrap();
        assert!(
            ssh_cfg.contains("Host colab-orphan-orphan01-1234"),
            "~/.ssh/config must contain Host colab-orphan-*"
        );

        let info_no_default = build_vm_ssh_info("orphan-orphan01-1234", &orphan_assign, Some(port));
        assert!(info_no_default.ready, "orphan-* SSH info must be ready when bridge port > 0");
        assert!(
            info_no_default.colab_ssh_command.contains(" -i \""),
            "colab_ssh_command must pass -i <pilot_key> when ~/.ssh/id_ed25519.pub is absent: {}",
            info_no_default.colab_ssh_command
        );
        assert!(!info_no_default.remote_ws_url.contains("secret-tok"));

        // When default ~/.ssh/id_ed25519.pub exists, colab_ssh_command stays concise
        std::fs::write(cli::ssh_dir().join("id_ed25519.pub"), "ssh-ed25519 AAAATEST test").unwrap();
        let info_with_default = build_vm_ssh_info("orphan-orphan01-1234", &orphan_assign, Some(port));
        assert_eq!(
            info_with_default.colab_ssh_command,
            "colab ssh -s orphan-orphan01-1234"
        );

        shutdown_child_processes(&app);
        *cli::TEST_CONFIG_DIR.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_39_hard_terminate_handover_stale_endpoint_hint_verifies_current_endpoint() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_issue39_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("nonexistent_colab_bin"));

        // Session `relay-job` handed over from `m-s-oldA` to `m-s-newB`, while UI passes stale hint `m-s-oldA`
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({
                "relay-job": { "endpoint": "m-s-newB" },
                "other-vm": { "endpoint": "m-s-other9" }
            })
            .to_string(),
        )
        .unwrap();

        let allow_unassign_b = Arc::new(AtomicBool::new(false));
        let b_alive = Arc::new(AtomicBool::new(true));
        let unassign_b_attempts = Arc::new(AtomicUsize::new(0));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let allow_srv = allow_unassign_b.clone();
        let b_alive_srv = b_alive.clone();
        let attempts_srv = unassign_b_attempts.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let allow = allow_srv.clone();
                let alive = b_alive_srv.clone();
                let att = attempts_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let (status, body) = if first.starts_with("GET /tun/m/assignments") {
                        let mut items = vec![serde_json::json!({
                            "endpoint": "m-s-other9",
                            "accelerator": "T4",
                            "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1" }
                        })];
                        if alive.load(Ordering::SeqCst) {
                            items.push(serde_json::json!({
                                "endpoint": "m-s-newB",
                                "accelerator": "L4",
                                "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1" }
                            }));
                        }
                        (
                            "200 OK",
                            format!(")]}}'\n{}", serde_json::json!({ "assignments": items })),
                        )
                    } else if first.starts_with("GET /tun/m/unassign/") {
                        ("200 OK", r#"{"token":"xsrf-tok"}"#.to_string())
                    } else if first.starts_with("POST /tun/m/unassign/m-s-newB") {
                        att.fetch_add(1, Ordering::SeqCst);
                        if allow.load(Ordering::SeqCst) {
                            alive.store(false, Ordering::SeqCst);
                            ("200 OK", "{}".to_string())
                        } else {
                            ("500 Internal Server Error", r#"{"error":"busy"}"#.to_string())
                        }
                    } else {
                        ("200 OK", "{}".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir);

        // 1. Stale endpoint `m-s-oldA` is already absent from server, while current `m-s-newB` refuses unassign:
        //    hard_terminate MUST try unassigning `m-s-newB` AND return verified_unassigned == false!
        let r_fail = hard_terminate(&app, "relay-job", Some("m-s-oldA".into()), false).await;
        assert!(
            unassign_b_attempts.load(Ordering::SeqCst) >= 1,
            "must attempt to unassign current bound endpoint m-s-newB"
        );
        assert!(
            !r_fail.verified_unassigned,
            "must NOT falsely claim verified_unassigned while current endpoint m-s-newB is still active"
        );
        assert!(r_fail.remaining_endpoints.contains(&"m-s-newB".to_string()));

        // 2. Once server accepts unassign on `m-s-newB`, retry with stale `m-s-oldA` hint unassigns `m-s-newB` cleanly
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({
                "relay-job": { "endpoint": "m-s-newB" },
                "other-vm": { "endpoint": "m-s-other9" }
            })
            .to_string(),
        )
        .unwrap();
        allow_unassign_b.store(true, Ordering::SeqCst);
        let r_ok = hard_terminate(&app, "relay-job", Some("m-s-oldA".into()), false).await;
        assert!(r_ok.verified_unassigned);
        assert!(!b_alive.load(Ordering::SeqCst));

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_40_and_41_stop_relay_result_destroy_and_keep_vm_kernel_interrupt() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pilot_stop_relay_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        let args_log = tmp.join("colab_args.txt");
        let mock_colab_py = tmp.join("mock_colab.py");
        std::fs::write(
            &mock_colab_py,
            r#"import json, os, sys
from pathlib import Path
Path(os.environ["MOCK_COLAB_LOG"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
rc = int(os.environ.get("MOCK_COLAB_RC", "0"))
if rc != 0:
    sys.stderr.write("Error: kernel restart rejected by runtime\n")
sys.exit(rc)
"#,
        )
        .unwrap();

        #[cfg(windows)]
        let mock_colab = {
            let p = tmp.join("mock_colab.cmd");
            std::fs::write(
                &p,
                format!("@echo off\r\n\"{}\" \"{}\" %*\r\n", py.display(), mock_colab_py.display()),
            )
            .unwrap();
            p
        };
        #[cfg(not(windows))]
        let mock_colab = {
            use std::os::unix::fs::PermissionsExt;
            let p = tmp.join("mock_colab.sh");
            std::fs::write(
                &p,
                format!("#!/bin/sh\nexec \"{}\" \"{}\" \"$@\"\n", py.display(), mock_colab_py.display()),
            )
            .unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        };

        std::env::set_var("MOCK_COLAB_LOG", &args_log);
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(mock_colab);

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        // Unreachable server so destroy_vm=true fails unassign verification (#40)
        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let app = AppState::new_with(client, ws_dir);

        // 1. #41: destroy_vm = false when session is missing -> kernel_interrupted == false with reason
        let res_missing = stop_relay_inner(&app, "missing-sess", false, || {})
            .await
            .unwrap();
        assert!(res_missing.supervisor_stopped);
        assert!(!res_missing.destroy_vm);
        assert!(!res_missing.kernel_interrupted);
        assert!(res_missing
            .kernel_interrupt_error
            .as_deref()
            .unwrap_or("")
            .contains("可能仍在实例中继续运行"));

        // Register `adc-train` session + ADC profile
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({ "adc-train": { "endpoint": "m-s-adc01" } }).to_string(),
        )
        .unwrap();
        cli::save_session_profile(
            "adc-train",
            &cli::SessionProfile {
                boot_mode: "relay".into(),
                accelerator: "T4".into(),
                high_mem: true,
                workspace_path: None,
                auth_mode: Some("adc".into()),
                llm_config: None,
                relay_config: None,
                keep_alive: None,
                grab_config: None,
                updated_at: None,
            },
        )
        .unwrap();

        // 2. #41: destroy_vm = false when `colab restart-kernel` returns non-zero -> kernel_interrupted == false
        std::env::set_var("MOCK_COLAB_RC", "2");
        let res_rc2 = stop_relay_inner(&app, "adc-train", false, || {})
            .await
            .unwrap();
        assert!(!res_rc2.kernel_interrupted);
        assert!(res_rc2
            .kernel_interrupt_error
            .as_deref()
            .unwrap_or("")
            .contains("exit=2"));

        // 3. #41 + #43: destroy_vm = false when `colab restart-kernel` returns 0 -> kernel_interrupted == true & passes `--auth adc`
        std::env::set_var("MOCK_COLAB_RC", "0");
        let res_ok = stop_relay_inner(&app, "adc-train", false, || {})
            .await
            .unwrap();
        assert!(res_ok.kernel_interrupted);
        assert!(res_ok.kernel_interrupt_error.is_none());
        let logged_argv: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(&args_log).unwrap()).unwrap();
        assert_eq!(
            logged_argv,
            vec!["--auth", "adc", "restart-kernel", "-s", "adc-train"]
        );

        // 4. #40: destroy_vm = true when server unassign verification fails -> kernel_interrupted == false & receipt.verified_unassigned == false
        std::env::set_var("MOCK_COLAB_RC", "1");
        let h = Arc::new(RelayHandle::new("adc-train"));
        h.log("round 1 log preserved");
        app.relays.l().insert("adc-train".into(), h);
        let res_destroy_unverified = stop_relay_inner(&app, "adc-train", true, || {})
            .await
            .unwrap();
        assert!(res_destroy_unverified.destroy_vm);
        assert!(!res_destroy_unverified.kernel_interrupted);
        assert!(
            !res_destroy_unverified
                .terminate_receipt
                .as_ref()
                .unwrap()
                .verified_unassigned
        );
        assert!(app.relays.l().contains_key("adc-train"), "offline relay logs must be retained");

        std::env::remove_var("MOCK_COLAB_LOG");
        std::env::remove_var("MOCK_COLAB_RC");
        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_42_and_58_workspace_binding_priority_and_strict_read_errors() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_ws_bind_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_default = tmp.join("ws_default");
        let ws_bound = tmp.join("ws_bound");
        let ws_explicit = tmp.join("ws_explicit");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_default).unwrap();
        std::fs::create_dir_all(&ws_bound).unwrap();
        std::fs::create_dir_all(&ws_explicit).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        std::fs::write(ws_default.join("train.py"), "# default\n").unwrap();
        std::fs::write(ws_bound.join("train.py"), "# bound\n").unwrap();
        std::fs::write(ws_explicit.join("train.py"), "# explicit\n").unwrap();

        cli::register_workspace_entry(
            "vm-custom",
            "ws_bound",
            Some(&ws_bound.to_string_lossy()),
            None,
        );

        let tok_file = tmp.join("token.json");
        std::fs::write(&tok_file, "{}").unwrap();
        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let app = AppState::new_with(client, ws_default.clone());

        // 1. Priority: explicit workspace_path > vm_name bound workspace > default app.workspace
        let f_default = read_workspace_files_inner(&app, None, None).await.unwrap();
        assert_eq!(f_default.train_py, "# default\n");
        assert_eq!(f_default.requirements_txt, "");

        let f_bound = read_workspace_files_inner(&app, None, Some("vm-custom"))
            .await
            .unwrap();
        assert_eq!(f_bound.train_py, "# bound\n");
        assert_eq!(PathBuf::from(&f_bound.workspace), ws_bound);

        let f_explicit = read_workspace_files_inner(
            &app,
            Some(&ws_explicit.to_string_lossy()),
            Some("vm-custom"),
        )
        .await
        .unwrap();
        assert_eq!(f_explicit.train_py, "# explicit\n");

        save_workspace_files_inner(
            &app,
            "print('saved_to_bound')\n".into(),
            "torch>=2.4\n".into(),
            None,
            Some("vm-custom"),
        )
        .await
        .unwrap();
        assert_eq!(
            std::fs::read_to_string(ws_bound.join("train.py")).unwrap(),
            "print('saved_to_bound')\n"
        );
        assert_eq!(
            std::fs::read_to_string(ws_default.join("train.py")).unwrap(),
            "# default\n",
            "saving bound VM workspace must NOT overwrite default workspace"
        );

        // 2. #58: Missing workspace dir must return Err, never empty strings
        let missing_err = read_workspace_files_inner(
            &app,
            Some(&tmp.join("does_not_exist").to_string_lossy()),
            None,
        )
        .await
        .expect_err("missing workspace directory must return Err");
        assert!(missing_err.contains("无法访问工作区目录"));

        // 3. #58: Invalid UTF-8 in train.py or requirements.txt must return Err naming the file
        std::fs::write(ws_bound.join("train.py"), [0xffu8, 0xfe, 0xd5, 0x00]).unwrap();
        let utf8_err = read_workspace_files_inner(&app, None, Some("vm-custom"))
            .await
            .expect_err("invalid UTF-8 train.py must return Err instead of empty string");
        assert!(utf8_err.contains("train.py") && utf8_err.contains("UTF-8"));

        std::fs::write(ws_bound.join("train.py"), "# valid\n").unwrap();
        std::fs::remove_file(ws_bound.join("requirements.txt")).ok();
        std::fs::create_dir_all(ws_bound.join("requirements.txt")).unwrap();
        let dir_err = read_workspace_files_inner(&app, None, Some("vm-custom"))
            .await
            .expect_err("directory in place of requirements.txt must return Err");
        assert!(dir_err.contains("requirements.txt"));

        *cli::TEST_CONFIG_DIR.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_43_44_52_54_67_68_75_create_vm_validations_and_no_auth_login_prompt() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_create_val_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        let fake_colab = tmp.join("fake_colab");
        std::fs::write(&fake_colab, "stub").unwrap();
        let fake_adc = tmp.join("adc.json");
        std::fs::write(&fake_adc, r#"{"type":"authorized_user"}"#).unwrap();
        let missing_rclone_conf = tmp.join("missing_rclone.conf");

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(fake_colab);
        *cli::TEST_ADC_PATH.l() = Some(fake_adc);
        *cli::TEST_RCLONE_CONF.l() = Some(missing_rclone_conf.clone());

        // 1. #43 & #75: ADC present (`adc_ok == true`) without Colab OAuth token (`!token_ok`) must block create_vm
        let missing_tok = tmp.join("no_token.json");
        let unauth_client = ColabClient::new_for_test("http://127.0.0.1:9".into(), missing_tok);
        let unauth_app = AppState::new_with(unauth_client, ws_dir.clone());
        let env_unauth = unauth_app.env_status();
        assert!(!env_unauth.token_ok);
        assert!(env_unauth.adc_ok);

        let adc_only_req = CreateVmRequest {
            name: Some("adc-only-vm".into()),
            accelerator: "T4".into(),
            auth_mode: Some("adc".into()),
            boot_mode: Some("general".into()),
            ..Default::default()
        };
        let adc_err = validate_and_prepare_create_vm(&unauth_app, &adc_only_req)
            .expect_err("ADC-only without OAuth token must be rejected before VM allocation");
        assert!(adc_err.contains("colab auth") && !adc_err.contains("colab auth login"));

        let ov_unauth = refresh_overview(&unauth_app, false).await;
        let ov_err = ov_unauth.error.unwrap_or_default();
        assert!(ov_err.contains("colab auth") && !ov_err.contains("colab auth login"));

        // Now authorize OAuth token (still NO rclone.conf on disk!)
        let valid_tok = tmp.join("valid_token.json");
        std::fs::write(
            &valid_tok,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let auth_client = ColabClient::new_for_test("http://127.0.0.1:9".into(), valid_tok);
        let app = AppState::new_with(auth_client, ws_dir.clone());

        // 2. #67: General mode without sync_rclone/mount_gdrive succeeds WITHOUT rclone.conf!
        let gen_ok = validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("gen-no-drive".into()),
                accelerator: "T4".into(),
                boot_mode: Some("general".into()),
                auto_relay: true, // stale auto_relay flag must be ignored when boot_mode == "general"
                env_config: Some(CreateVmEnvConfig {
                    sync_rclone: false,
                    mount_gdrive: false,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("general VM without rclone options must not require rclone.conf");
        assert_eq!(gen_ok.boot_mode, "general");
        assert!(gen_ok.validated_relay_cfg.is_none());

        // Enabling sync_rclone without rclone.conf rejects cleanly
        assert!(validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("gen-need-drive".into()),
                accelerator: "T4".into(),
                boot_mode: Some("general".into()),
                env_config: Some(CreateVmEnvConfig {
                    sync_rclone: true,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .is_err());

        // 3. #68: CUDA LLM presets (vllm / llama.cpp / transformers) rejected on CPU & TPU before VM allocation
        for bad_hw in ["CPU", "V5E1", "V6E1"] {
            for cuda_eng in ["vllm", "llama.cpp", "transformers"] {
                let res = validate_and_prepare_create_vm(
                    &app,
                    &CreateVmRequest {
                        name: Some("llm-bad".into()),
                        accelerator: bad_hw.into(),
                        boot_mode: Some("llm".into()),
                        llm_config: Some(CreateVmLlmConfig {
                            engine: cuda_eng.into(),
                            model_name: "Qwen/Qwen2.5-7B-Instruct".into(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                );
                assert!(
                    res.is_err(),
                    "CUDA engine {cuda_eng} on non-NVIDIA {bad_hw} must be rejected"
                );
            }
        }
        // NVIDIA GPU + vllm AND CPU + ollama succeed
        assert!(validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("llm-g4-ok".into()),
                accelerator: "G4".into(),
                boot_mode: Some("llm".into()),
                llm_config: Some(CreateVmLlmConfig {
                    engine: "vllm".into(),
                    model_name: "Qwen/Qwen2.5-7B-Instruct".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .is_ok());
        assert!(validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("llm-cpu-ollama".into()),
                accelerator: "CPU".into(),
                boot_mode: Some("llm".into()),
                llm_config: Some(CreateVmLlmConfig {
                    engine: "ollama".into(),
                    model_name: "qwen2.5:3b".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .is_ok());

        // 4. #44, #52, #54: Create rclone.conf and test CPU relay preservation, bare gdrive_dir isolation, 23.5h cap & conflict rejection
        std::fs::write(&missing_rclone_conf, "[gdrive]\ntype = drive\n").unwrap();
        let cpu_relay_prep = validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("cpu-relay-1".into()),
                accelerator: "CPU".into(),
                high_mem: true,
                boot_mode: Some("relay".into()),
                relay_config: Some(RelayConfig {
                    gpu_type: "T4".into(),
                    high_mem: true,
                    gdrive_dir: "gdrive:colab_relay".into(),
                    max_session_hours: 23.5,
                    backup_interval_minutes: 20.0,
                    total_epochs: 50,
                    reset_done_flag: false,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect("CPU relay with 23.5h must pass validation");
        let rcfg = cpu_relay_prep.validated_relay_cfg.unwrap();
        assert_eq!(rcfg.gpu_type, "CPU", "Issue #52: CPU must never be rewritten to T4");
        assert_eq!(
            rcfg.gdrive_dir, "gdrive:colab_relay/cpu-relay-1",
            "Issue #44: bare gdrive:colab_relay must isolate by VM name"
        );

        // Reject >23.5h in create_vm (#54)
        assert!(validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("relay-30h".into()),
                accelerator: "T4".into(),
                boot_mode: Some("relay".into()),
                relay_config: Some(RelayConfig {
                    max_session_hours: 30.0,
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .is_err());

        // Simulate `cpu-relay-1` actively running on `gdrive:colab_relay/shared_exp` and verify another VM is blocked from using it (#44)
        let running_h = Arc::new(RelayHandle::new("cpu-relay-1"));
        running_h.status.l().running = true;
        running_h.status.l().config.gdrive_dir = "gdrive:colab_relay/shared_exp/".into();
        app.relays.l().insert("cpu-relay-1".into(), running_h);

        let conflict_err = validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("cpu-relay-2".into()),
                accelerator: "CPU".into(),
                boot_mode: Some("relay".into()),
                relay_config: Some(RelayConfig {
                    gdrive_dir: "gdrive:colab_relay/shared_exp".into(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )
        .expect_err("duplicate active gdrive_dir must be rejected");
        assert!(conflict_err.contains("cpu-relay-1"));

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_ADC_PATH.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_48_bluegreen_custom_session_names_routing_and_cleanup() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_bg48_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());

        let bg_json = serde_json::json!({
            "session_prefix": "prod",
            "active_slot": "blue",
            "standby_slot": "green",
            "active_session": "custom-node-alpha",
            "standby_session": "vm-auto99",
            "blue_session": "custom-node-alpha",
            "green_session": "vm-auto99",
            "slots": {
                "blue": { "session_name": "custom-node-alpha", "role": "active", "model_name": "Qwen/Qwen2.5-32B-Instruct", "endpoint": "m-s-alpha01" },
                "green": { "session_name": "vm-auto99", "role": "warming", "model_name": "Qwen/Qwen2.5-72B-Instruct", "endpoint": "m-s-auto99" }
            },
            "phase": "serving",
            "model_name": "Qwen/Qwen2.5-32B-Instruct",
            "updated_at": 1794736000.0
        });
        std::fs::write(cfg_dir.join("llm_bluegreen_state.json"), bg_json.to_string()).unwrap();

        let bg_parsed = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state()).unwrap();
        assert_eq!(
            bluegreen_tunnel_port_for_vm("custom-node-alpha", Some(&bg_parsed)),
            Some(18081)
        );
        assert_eq!(
            bluegreen_tunnel_port_for_vm("vm-auto99", Some(&bg_parsed)),
            Some(18082)
        );
        // Fallback `${prefix}-blue` must NOT hijack port 18081 when `blue` explicitly binds `custom-node-alpha`
        assert_eq!(
            bluegreen_tunnel_port_for_vm("prod-blue", Some(&bg_parsed)),
            None
        );

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let app = AppState::new_with(client, ws_dir);
        {
            let mut ov = app.overview.write().await;
            ov.vms = vec![
                VmView {
                    name: "vm-auto99".into(),
                    on_server: true,
                    managed: true,
                    accelerator: "A100".into(),
                    telemetry: Some(serde_json::json!({
                        "llm": { "online": false, "role": "warming", "slot": "green" }
                    })),
                    ..Default::default()
                },
                VmView {
                    name: "custom-node-alpha".into(),
                    on_server: true,
                    managed: true,
                    accelerator: "A100".into(),
                    telemetry: Some(serde_json::json!({
                        "llm": { "online": true, "role": "active", "slot": "blue", "port": 8000 }
                    })),
                    ..Default::default()
                },
            ];
        }

        let resolved = resolve_llm_target(
            &app,
            &LlmChatRequest {
                request_id: "r48".into(),
                target_vm: "auto".into(),
                model: None,
                messages: vec![],
                max_tokens: None,
                temperature: None,
            },
        )
        .await
        .expect("auto target should resolve custom-named active bluegreen VM");
        assert_eq!(resolved.vm_name, "custom-node-alpha");
        assert_eq!(resolved.slot.as_deref(), Some("blue"));
        assert_eq!(resolved.local_tunnel_port, Some(18081));
        assert_eq!(resolved.model, "Qwen/Qwen2.5-32B-Instruct");

        // Terminating custom-node-alpha while `vm-auto99` (`m-s-auto99`) remains online promotes `green` (`vm-auto99`)
        let mut steps = Vec::new();
        cleanup_llm_bluegreen_after_terminate(
            "custom-node-alpha",
            &["m-s-auto99".to_string()],
            &mut steps,
        );
        let after_one = parse_llm_bluegreen_state(cli::read_llm_bluegreen_state())
            .expect("state should remain and promote surviving green session");
        assert_eq!(after_one.active_slot.as_deref(), Some("green"));
        assert_eq!(after_one.active_session.as_deref(), Some("vm-auto99"));

        // Terminating the last remaining session removes state file
        cleanup_llm_bluegreen_after_terminate("vm-auto99", &[], &mut steps);
        assert!(cli::read_llm_bluegreen_state().is_none());

        *cli::TEST_CONFIG_DIR.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_55_deploy_probe_rejects_stale_telemetry_json() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pilot_probe55_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        // Mock colab exec that pretends exit 0 ("PROBE_OK") without updating remote telemetry JSON
        let mock_colab_py = tmp.join("mock_colab_probe.py");
        std::fs::write(
            &mock_colab_py,
            "import sys; sys.stdin.read(); print('PROBE_OK'); sys.exit(0)\n",
        )
        .unwrap();
        #[cfg(windows)]
        let mock_colab = {
            let p = tmp.join("mock_colab_probe.cmd");
            std::fs::write(
                &p,
                format!("@echo off\r\n\"{}\" \"{}\" %*\r\n", py.display(), mock_colab_py.display()),
            )
            .unwrap();
            p
        };
        #[cfg(not(windows))]
        let mock_colab = {
            use std::os::unix::fs::PermissionsExt;
            let p = tmp.join("mock_colab_probe.sh");
            std::fs::write(
                &p,
                format!("#!/bin/sh\nexec \"{}\" \"{}\" \"$@\"\n", py.display(), mock_colab_py.display()),
            )
            .unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
            p
        };

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(mock_colab);
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({ "probe-vm": { "endpoint": "m-s-probe55" } }).to_string(),
        )
        .unwrap();

        // Mock proxy Contents server returning a STALE `.colab_pilot_telemetry.json` from an old PID / wrong nonce
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 65536];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let body = if first.starts_with("GET /api/contents/content/.colab_pilot_telemetry.json") {
                        serde_json::json!({
                            "type": "file",
                            "format": "text",
                            "content": serde_json::json!({
                                "timestamp": 1700000000.0,
                                "probe_nonce": "stale_old_nonce",
                                "daemon_pid": 99999
                            }).to_string()
                        })
                        .to_string()
                    } else {
                        r#"{"type":"file"}"#.to_string()
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir);
        let a: Assignment = serde_json::from_value(serde_json::json!({
            "endpoint": "m-s-probe55",
            "accelerator": "T4",
            "runtimeProxyInfo": {
                "url": format!("http://{addr}"),
                "token": "ptok",
                "tokenExpiresInSeconds": 3600
            }
        }))
        .unwrap();

        let err = deploy_probe_for_assignment(&app, &a)
            .await
            .expect_err("stale telemetry JSON with mismatched nonce/timestamp MUST fail probe verification");
        assert!(
            err.to_string().contains("未能产出匹配本次部署") || err.to_string().contains("探针"),
            "unexpected error: {err:#}"
        );

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_57_72_73_overview_stale_cache_offline_relay_retention_and_window_close_guard() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_ov_close_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("nonexistent_colab"));
        *cli::TEST_RCLONE_CONF.l() = Some(tmp.join("nonexistent_rclone.conf"));

        let fail_assignments = Arc::new(AtomicBool::new(false));
        let vm_on_server = Arc::new(AtomicBool::new(true));
        let allow_unassign = Arc::new(AtomicBool::new(false));

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let fail_srv = fail_assignments.clone();
        let on_srv = vm_on_server.clone();
        let unassign_srv = allow_unassign.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let f = fail_srv.clone();
                let o = on_srv.clone();
                let u = unassign_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let (status, body) = if first.starts_with("GET /tun/m/assignments") {
                        if f.load(Ordering::SeqCst) {
                            ("502 Bad Gateway", r#"{"error":"upstream timeout"}"#.to_string())
                        } else {
                            let mut items = Vec::new();
                            if o.load(Ordering::SeqCst) {
                                items.push(serde_json::json!({
                                    "endpoint": "m-s-active01",
                                    "accelerator": "T4",
                                    "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1:9" }
                                }));
                            }
                            (
                                "200 OK",
                                format!(")]}}'\n{}", serde_json::json!({ "assignments": items })),
                            )
                        }
                    } else if first.starts_with("GET /tun/m/unassign/") {
                        ("200 OK", r#"{"token":"xsrf-tok"}"#.to_string())
                    } else if first.starts_with("POST /tun/m/unassign/m-s-active01") {
                        if u.load(Ordering::SeqCst) {
                            o.store(false, Ordering::SeqCst);
                            ("200 OK", "{}".to_string())
                        } else {
                            ("500 Internal Server Error", r#"{"error":"unassign failed"}"#.to_string())
                        }
                    } else {
                        ("200 OK", r#"{"currentBalance":88.0}"#.to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({ "active-relay": { "endpoint": "m-s-active01" } }).to_string(),
        )
        .unwrap();
        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir);

        // Add an offline finished/failed relay (#73) AND a running relay (#72)
        let finished_h = Arc::new(RelayHandle::new("finished-offline-relay"));
        finished_h.status.l().running = false;
        finished_h.status.l().phase = "error".into();
        finished_h.status.l().last_error = Some("OOM on epoch 4".into());
        finished_h.log("❌ OOM traceback line");
        app.relays
            .l()
            .insert("finished-offline-relay".into(), finished_h);

        let active_h = Arc::new(RelayHandle::new("active-relay"));
        active_h.status.l().running = true;
        active_h.status.l().phase = "training".into();
        app.relays.l().insert("active-relay".into(), active_h);

        // 1. First refresh succeeds; offline finished relay MUST be retained in ov.vms (#73)
        let ov1 = refresh_overview(&app, true).await;
        assert!(!ov1.is_stale);
        assert!(ov1.stale_reason.is_none());
        let ok_ts = ov1.last_ok_timestamp.expect("last_ok_timestamp set on success");
        assert_eq!(ov1.timestamp, ok_ts as i64);
        let offline_card = ov1
            .vms
            .iter()
            .find(|v| v.name == "finished-offline-relay")
            .expect("offline finished/failed relay card must remain visible (#73)");
        assert!(!offline_card.on_server);
        assert_eq!(
            offline_card
                .relay
                .as_ref()
                .and_then(|r| r.last_error.as_deref()),
            Some("OOM on epoch 4")
        );

        // Force cached snapshot timestamp back to a distinct known past value (`1700001234`) within 60s window (#57)
        {
            let mut guard = app.last_assignments.l();
            if let Some((list, inst, _)) = guard.clone() {
                *guard = Some((list, inst, 1700001234));
            }
        }
        fail_assignments.store(true, Ordering::SeqCst);
        let ov_stale = refresh_overview(&app, true).await;
        assert!(ov_stale.is_stale, "60s fallback on error must set is_stale=true");
        assert!(ov_stale.stale_reason.is_some());
        assert_eq!(ov_stale.last_ok_timestamp, Some(1700001234));
        assert_eq!(
            ov_stale.timestamp, 1700001234,
            "Overview.timestamp must preserve previous snapshot timestamp on stale fallback"
        );
        fail_assignments.store(false, Ordering::SeqCst);

        // 2. #73: dismiss_relay_status refuses running relay, removes finished offline relay
        assert!(dismiss_relay_status_inner(&app, "active-relay").is_err());
        dismiss_relay_status_inner(&app, "finished-offline-relay")
            .expect("dismissing finished relay should succeed");
        assert!(!app.relays.l().contains_key("finished-offline-relay"));

        // 3. #72: Window close interception & confirm_app_close_inner("shutdown_vms") unverified -> retry verified
        assert_eq!(
            should_intercept_window_close(&app),
            Some(vec!["active-relay".to_string()])
        );
        let (receipts_unverified, should_exit_1) = confirm_app_close_inner(&app, "shutdown_vms")
            .await
            .unwrap();
        assert!(
            !should_exit_1,
            "must NOT exit app when shutdown_vms fails unassign verification"
        );
        assert!(!receipts_unverified[0].verified_unassigned);
        assert_eq!(
            should_intercept_window_close(&app),
            Some(vec!["active-relay".to_string()]),
            "unverified pending close VM must still intercept window close and allow retry"
        );

        // Retry shutdown_vms once server allows unassign -> verified & exits!
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({ "active-relay": { "endpoint": "m-s-active01" } }).to_string(),
        )
        .unwrap();
        allow_unassign.store(true, Ordering::SeqCst);
        let (receipts_ok, should_exit_2) = confirm_app_close_inner(&app, "shutdown_vms")
            .await
            .unwrap();
        assert!(should_exit_2);
        assert!(receipts_ok[0].verified_unassigned);
        assert!(should_intercept_window_close(&app).is_none());

        shutdown_child_processes(&app);
        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_77_hard_terminate_running_relay_never_unassigns_unrelated_endpoint() {
        use std::sync::atomic::Ordering;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_issue77_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("no_colab"));

        let relay_alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let other_alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let unassigned_eps: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let r_alive = relay_alive.clone();
        let o_alive = other_alive.clone();
        let u_log = unassigned_eps.clone();
        let srv = tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let ra = r_alive.clone();
                let oa = o_alive.clone();
                let ul = u_log.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let (status, body) = if first.starts_with("GET /tun/m/assignments") {
                        let mut items = Vec::new();
                        if ra.load(Ordering::SeqCst) {
                            items.push(serde_json::json!({
                                "endpoint": "m-s-relay001",
                                "accelerator": "T4",
                                "runtimeProxyInfo": { "token": "t", "url": "http://127.0.0.1:9" }
                            }));
                        }
                        if oa.load(Ordering::SeqCst) {
                            items.push(serde_json::json!({
                                "endpoint": "m-s-other-project",
                                "accelerator": "A100",
                                "runtimeProxyInfo": { "token": "t", "url": "http://127.0.0.1:9" }
                            }));
                        }
                        (
                            "200 OK",
                            format!(")]}}'\n{}", serde_json::json!({ "assignments": items })),
                        )
                    } else if first.starts_with("GET /tun/m/unassign/") {
                        ("200 OK", r#"{"token":"xsrf"}"#.to_string())
                    } else if let Some(rest) = first.strip_prefix("POST /tun/m/unassign/") {
                        let ep = rest
                            .split(['?', ' '])
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        ul.l().push(ep.clone());
                        if ep == "m-s-relay001" {
                            ra.store(false, Ordering::SeqCst);
                        } else if ep == "m-s-other-project" {
                            oa.store(false, Ordering::SeqCst);
                        }
                        ("200 OK", "{}".to_string())
                    } else {
                        ("200 OK", r#"{"currentBalance":50.0}"#.to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        // Register ONLY `managed-relay` in sessions.json; `m-s-other-project` is an unregistered endpoint from another project
        std::fs::write(
            cfg_dir.join("sessions.json"),
            serde_json::json!({ "managed-relay": { "endpoint": "m-s-relay001" } }).to_string(),
        )
        .unwrap();
        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir);

        let h = Arc::new(RelayHandle::new("managed-relay"));
        h.status.l().running = true;
        h.status.l().phase = "training".into();
        h.task_alive.store(true, Ordering::SeqCst);
        app.relays.l().insert("managed-relay".into(), h);

        let receipt = hard_terminate(&app, "managed-relay", None, true).await;
        assert!(receipt.verified_unassigned);
        assert_eq!(receipt.endpoint.as_deref(), Some("m-s-relay001"));
        assert!(!relay_alive.load(Ordering::SeqCst));
        assert!(
            other_alive.load(Ordering::SeqCst),
            "unregistered endpoint from another project must NOT be unassigned (#77)"
        );
        assert_eq!(
            *unassigned_eps.l(),
            vec!["m-s-relay001".to_string()],
            "unassign must ONLY be called on the confirmed session endpoint"
        );

        // Subcase B (#77): consumed/repeated provisioning snapshot or 2-endpoint concurrent race must NEVER guess & kill m-s-other-project
        std::fs::write(cfg_dir.join("sessions.json"), "{}").unwrap();
        relay_alive.store(true, Ordering::SeqCst);
        let h_race = Arc::new(RelayHandle::new("race-relay"));
        h_race.status.l().running = true;
        h_race.status.l().phase = "provisioning".into();
        h_race.task_alive.store(true, Ordering::SeqCst);
        h_race.pre_provision_snapshot_invalid.store(false, Ordering::SeqCst);
        app.relays.l().insert("race-relay".into(), h_race.clone());

        // Both `m-s-relay001` and `m-s-other-project` appear as newly_leaked (len == 2) -> must refuse guessing!
        let r_race = hard_terminate(&app, "race-relay", None, true).await;
        assert!(!r_race.verified_unassigned, "ambiguous 2-endpoint race must refuse guessing");
        assert!(other_alive.load(Ordering::SeqCst), "must leave unrelated endpoint untouched");
        assert!(relay_alive.load(Ordering::SeqCst), "must leave both untouched when ambiguous");
        assert_eq!(*unassigned_eps.l(), vec!["m-s-relay001".to_string()]);

        // Now suppose `m-s-relay001` disappeared externally leaving ONLY `m-s-other-project`, and a second hard_terminate runs:
        // because the provisioning snapshot was already consumed, `m-s-other-project` MUST still NOT be unassigned!
        relay_alive.store(false, Ordering::SeqCst);
        let r_repeat = hard_terminate(&app, "race-relay", None, false).await;
        assert!(!r_repeat.verified_unassigned);
        assert!(other_alive.load(Ordering::SeqCst), "consumed snapshot must never kill remaining other-project endpoint");
        assert_eq!(*unassigned_eps.l(), vec!["m-s-relay001".to_string()]);

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_78_83_84_unknown_billing_on_list_failure_unsaved_close_and_atomic_save_rollback() {
        use std::sync::atomic::Ordering;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_78_83_84_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("no_colab"));
        *cli::TEST_RCLONE_CONF.l() = Some(tmp.join("no_rclone.conf"));

        let fail_server = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let fs_flag = fail_server.clone();
        let srv = tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let f = fs_flag.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 2048];
                    let _ = sock.read(&mut buf).await;
                    let (status, body) = if f.load(Ordering::SeqCst) {
                        ("503 Service Unavailable", r#"{"error":"down"}"#.to_string())
                    } else {
                        ("200 OK", ")]}'\n{\"assignments\":[]}".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir.clone());

        // #78: Expired cache (>60s) + server list failure with local stopped relay record
        *app.last_assignments.l() = Some((
            vec![],
            Instant::now() - Duration::from_secs(120),
            1700000100,
        ));
        *app.last_ccu.l() = Some(Ccu {
            balance: 99.0,
            burn_rate_hourly: 0.0,
            assignments_count: 0,
        });
        let stopped_h = Arc::new(RelayHandle::new("offline-relay"));
        stopped_h.status.l().running = false;
        stopped_h.status.l().phase = "error".into();
        stopped_h.status.l().last_error = Some("failed".into());
        app.relays.l().insert("offline-relay".into(), stopped_h);

        let ov_unknown = refresh_overview(&app, true).await;
        assert!(ov_unknown.error.is_some());
        assert!(!ov_unknown.is_stale);
        assert!(!ov_unknown.server_list_known, "expired cache query failure must set server_list_known=false");
        assert!(ov_unknown.ccu.is_none(), "must not expose stale 0.0 CU/h when server_list_known is false");
        let card_unknown = ov_unknown
            .vms
            .iter()
            .find(|v| v.name == "offline-relay")
            .expect("offline relay record retained");
        assert!(card_unknown.server_state_unknown, "offline relay card must mark server_state_unknown=true");

        // Server recovers and confirms 0 assignments -> server_list_known=true & server_state_unknown=false
        fail_server.store(false, Ordering::SeqCst);
        let ov_confirmed = refresh_overview(&app, true).await;
        assert!(ov_confirmed.error.is_none());
        assert!(ov_confirmed.server_list_known);
        let card_confirmed = ov_confirmed
            .vms
            .iter()
            .find(|v| v.name == "offline-relay")
            .unwrap();
        assert!(!card_confirmed.server_state_unknown);

        // #78 subcase: unverified offline relay during <60s stale cache fallback MUST ALSO mark server_state_unknown=true
        let unverified_h = Arc::new(RelayHandle::new("unverified-stop-relay"));
        unverified_h.status.l().running = false;
        unverified_h.status.l().phase = "stopped".into();
        unverified_h.cloud_unassign_verified.store(false, Ordering::SeqCst);
        app.relays.l().insert("unverified-stop-relay".into(), unverified_h);
        fail_server.store(true, Ordering::SeqCst);
        let ov_stale_unverified = refresh_overview(&app, true).await;
        assert!(ov_stale_unverified.is_stale);
        let card_stale_unverified = ov_stale_unverified
            .vms
            .iter()
            .find(|v| v.name == "unverified-stop-relay")
            .expect("unverified offline relay retained");
        assert!(
            card_stale_unverified.server_state_unknown,
            "unverified offline relay must NOT claim stopped billing on a stale cache fallback (#78)"
        );
        let card_prev_verified = ov_stale_unverified
            .vms
            .iter()
            .find(|v| v.name == "offline-relay")
            .unwrap();
        assert!(
            !card_prev_verified.server_state_unknown,
            "previously verified offline relay stays known within 60s cache window"
        );
        fail_server.store(false, Ordering::SeqCst);

        // #84: Atomic workspace script save — read-only requirements.txt MUST preserve original train.py
        let train_p = ws_dir.join("train.py");
        let req_p = ws_dir.join("requirements.txt");
        std::fs::write(&train_p, "# original train v1\n").unwrap();
        std::fs::write(&req_p, "numpy==1.26.4\n").unwrap();
        let mut perms = std::fs::metadata(&req_p).unwrap().permissions();
        perms.set_readonly(true);
        std::fs::set_permissions(&req_p, perms).unwrap();

        let save_err = save_workspace_files_inner(
            &app,
            "# modified train v2\n".into(),
            "torch==2.5.1\n".into(),
            Some(&ws_dir.to_string_lossy()),
            None,
        )
        .await
        .expect_err("saving when requirements.txt is read-only must fail");
        assert!(save_err.contains("requirements.txt"));
        assert_eq!(
            std::fs::read_to_string(&train_p).unwrap(),
            "# original train v1\n",
            "train.py must NOT be overwritten when requirements.txt write fails (#84)"
        );
        assert_eq!(std::fs::read_to_string(&req_p).unwrap(), "numpy==1.26.4\n");

        // Restore write permission and verify both files update atomically
        let mut perms_rw = std::fs::metadata(&req_p).unwrap().permissions();
        perms_rw.set_readonly(false);
        std::fs::set_permissions(&req_p, perms_rw).unwrap();
        save_workspace_files_inner(
            &app,
            "# modified train v2\n".into(),
            "torch==2.5.1\n".into(),
            Some(&ws_dir.to_string_lossy()),
            None,
        )
        .await
        .expect("saving writable files must succeed");
        assert_eq!(std::fs::read_to_string(&train_p).unwrap(), "# modified train v2\n");
        assert_eq!(std::fs::read_to_string(&req_p).unwrap(), "torch==2.5.1\n");

        // #83: System main window close intercepts unsaved script edits even with 0 active relays
        assert!(evaluate_window_close_interception(&app).is_none());
        app.script_dirty.store(true, Ordering::SeqCst);
        assert_eq!(
            evaluate_window_close_interception(&app),
            Some(WindowCloseInterception {
                active_vms: vec![],
                unsaved_script: true,
            })
        );
        let (_, exit_ok) = confirm_app_close_inner(&app, "force_exit").await.unwrap();
        assert!(exit_ok);
        assert!(!app.script_dirty.load(Ordering::SeqCst));
        assert!(evaluate_window_close_interception(&app).is_none());

        // #83 + #72: `force_exit` MUST NOT bypass active relay protection if a relay is active
        app.force_close_confirmed.store(false, Ordering::SeqCst);
        app.script_dirty.store(true, Ordering::SeqCst);
        let live_h = Arc::new(RelayHandle::new("live-relay"));
        live_h.status.l().running = true;
        live_h.status.l().phase = "training".into();
        live_h.task_alive.store(true, Ordering::SeqCst);
        app.relays.l().insert("live-relay".into(), live_h.clone());
        let (_, exit_with_live) = confirm_app_close_inner(&app, "force_exit").await.unwrap();
        assert!(
            !exit_with_live,
            "force_exit must NOT exit or kill active relays when a relay is running (#72 + #83)"
        );
        assert!(live_h.is_running(), "live relay must still be running until CloseAppRelayDialog decision");
        assert!(!app.script_dirty.load(Ordering::SeqCst));
        assert_eq!(
            should_intercept_window_close(&app),
            Some(vec!["live-relay".to_string()])
        );

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn keep_alive_engine_checks_runtime_and_skips_busy_kernel() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let ka_hits = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let ka_hits_srv = ka_hits.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let hits = ka_hits_srv.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let req_low = req.to_ascii_lowercase();
                    let first = req.lines().next().unwrap_or("");
                    let (status, body) = if first.starts_with("GET /api/kernels?")
                        && req_low.contains("x-colab-runtime-proxy-token: proxy-tok")
                    {
                        hits.fetch_add(1, Ordering::SeqCst);
                        ("200 OK", r#"[{"id":"k-busy-1","name":"python3","execution_state":"busy","connections":1}]"#)
                    } else {
                        ("404 Not Found", "{}")
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka_engine_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let fake_colab = tmp.join("fake_colab.cmd");
        std::fs::write(&fake_colab, "@echo off\r\nexit /b 0\r\n").unwrap();

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(fake_colab);
        *cli::TEST_RCLONE_CONF.l() = Some(tmp.join("missing_rclone.conf"));

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        // 1. Any CreateVmRequest with keep_alive: None defaults to enabled=true, 90s, kernel_ping=true
        let req_default = CreateVmRequest {
            name: Some("ka-general-vm".into()),
            accelerator: "A100".into(),
            high_mem: true,
            boot_mode: Some("general".into()),
            ..Default::default()
        };
        let prep = validate_and_prepare_create_vm(&app, &req_default)
            .expect("general VM creation should default keep_alive to enabled");
        assert!(prep.validated_keep_alive.enabled);
        assert_eq!(prep.validated_keep_alive.interval_seconds, 90);
        assert!(prep.validated_keep_alive.kernel_ping);

        // 2. Out-of-range keep_alive interval is rejected before allocation
        let req_bad_ka = CreateVmRequest {
            name: Some("ka-bad-vm".into()),
            accelerator: "T4".into(),
            boot_mode: Some("general".into()),
            keep_alive: Some(KeepAliveConfig {
                enabled: true,
                interval_seconds: 10,
                kernel_ping: true,
            }),
            ..Default::default()
        };
        let err = validate_and_prepare_create_vm(&app, &req_bad_ka)
            .expect_err("interval < 30s must be rejected");
        assert!(err.contains("30 ~ 600"));

        // 3. Persist keep-alive config and run a tick while Kernel is busy -> tunnel ok + skipped_busy (zero blocking!)
        let cfg_on = KeepAliveConfig {
            enabled: true,
            interval_seconds: 60,
            kernel_ping: true,
        };
        let st_init = set_vm_keep_alive_inner(&app, "ka-vm1", cfg_on.clone())
            .await
            .expect("set_vm_keep_alive_inner should persist");
        assert!(st_init.enabled);
        assert_eq!(st_init.interval_seconds, 60);
        assert_eq!(
            cli::get_session_profile("ka-vm1").and_then(|p| p.keep_alive),
            Some(cfg_on)
        );

        let a = Assignment {
            accelerator: "A100".into(),
            endpoint: "m-s-ka-vm1".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("HIGH_RAM".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "proxy-tok".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };
        let st_tick = run_keep_alive_tick_for_vm(&app, "ka-vm1", &a, None, true, true, None)
            .await
            .expect("forced keep-alive tick should run");
        assert_eq!(ka_hits.load(Ordering::SeqCst), 1);
        assert!(st_tick.tunnel_ok);
        assert_eq!(st_tick.kernel_ping_state, "skipped_busy");
        assert_eq!(st_tick.state, "active");
        assert_eq!(st_tick.success_count, 1);
        assert!(st_tick.last_ok_at.is_some());

        let mut unreachable = a.clone();
        unreachable.runtime_proxy_info.url.push_str("/unreachable");
        let st_failed = run_keep_alive_tick_for_vm(&app, "ka-vm1", &unreachable, None, true, true, None)
            .await
            .expect("failed connection check should report keep-alive status");
        assert!(st_failed.enabled);
        assert!(!st_failed.tunnel_ok);
        assert_eq!(st_failed.state, "error");
        assert_eq!(st_failed.success_count, 1);

        // 4. Disabling keep-alive persists enabled=false and transitions to paused
        let st_off = set_vm_keep_alive_inner(
            &app,
            "ka-vm1",
            KeepAliveConfig {
                enabled: false,
                interval_seconds: 60,
                kernel_ping: true,
            },
        )
        .await
        .expect("disabling keep-alive should succeed");
        assert!(!st_off.enabled);
        assert_eq!(st_off.state, "paused");
        assert!(
            run_keep_alive_tick_for_vm(&app, "ka-vm1", &a, None, false, true, None)
                .await
                .is_none(),
            "scheduled tick must skip paused keep-alive VM"
        );

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_grab_modes_fallback_env_and_cancel_create_vm() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pilot_grab_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        let tok_file = cfg_dir.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"grab-test-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(cfg_dir.join("sessions.json"), "{}").unwrap();
        std::fs::write(cfg_dir.join("rclone.conf"), "[gdrive]\ntype = drive\n").unwrap();

        // 1. Verify differentiated request timeouts for GPU/CPU (60s) vs TPU cold boot (300s)
        assert_eq!(
            relay::colab_assign_env_for_gpu("A100"),
            ("60", Duration::from_secs(150))
        );
        assert_eq!(
            relay::colab_assign_env_for_gpu("V6E1"),
            ("300", Duration::from_secs(360))
        );

        // 2. Mock Colab CLI script recording COLAB_CLI_ASSIGN_RETRIES & COLAB_CLI_REQUEST_TIMEOUT
        let env_log = tmp.join("colab_env_log.jsonl");
        let colab_py = tmp.join("mock_colab_grab.py");
        std::fs::write(
            &colab_py,
            r#"import json, os, sys
log_path = os.environ.get("MOCK_COLAB_ENV_LOG", "")
if log_path:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "argv": sys.argv[1:],
            "retries": os.environ.get("COLAB_CLI_ASSIGN_RETRIES"),
            "timeout": os.environ.get("COLAB_CLI_REQUEST_TIMEOUT"),
        }) + "\n")
if "new" in sys.argv[1:]:
    if "--gpu" in sys.argv and sys.argv[sys.argv.index("--gpu") + 1] == "A100":
        sys.stderr.write("Error: HTTP 503 Service Unavailable\n")
        sys.exit(1)
    sys.stdout.write("Session READY\n")
    sys.exit(0)
sys.exit(0)
"#,
        )
        .unwrap();

        #[cfg(windows)]
        let mock_colab = {
            let cmd_path = tmp.join("mock_colab_grab.cmd");
            std::fs::write(
                &cmd_path,
                format!("@echo off\r\n\"{}\" \"{}\" %*\r\n", py.display(), colab_py.display()),
            )
            .unwrap();
            cmd_path
        };
        #[cfg(not(windows))]
        let mock_colab = {
            use std::os::unix::fs::PermissionsExt;
            let sh_path = tmp.join("mock_colab_grab.sh");
            std::fs::write(
                &sh_path,
                format!("#!/bin/sh\nexec \"{}\" \"{}\" \"$@\"\n", py.display(), colab_py.display()),
            )
            .unwrap();
            std::fs::set_permissions(&sh_path, std::fs::Permissions::from_mode(0o755)).unwrap();
            sh_path
        };

        // 3. Mock Colab backend with 1 pre-existing VM (`m-s-existing`) and 1 newly leaked VM (`m-s-leaked`)
        let existing_alive = Arc::new(AtomicBool::new(true));
        let leaked_alive = Arc::new(AtomicBool::new(true));
        let unassigned = Arc::new(Mutex::new(Vec::<String>::new()));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let ex_srv = existing_alive.clone();
        let lk_srv = leaked_alive.clone();
        let un_srv = unassigned.clone();
        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let ex = ex_srv.clone();
                let lk = lk_srv.clone();
                let un = un_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first = req.lines().next().unwrap_or("");
                    let body = if first.starts_with("GET /tun/m/unassign/") {
                        r#"{"token":"xsrf-tok"}"#.to_string()
                    } else if first.starts_with("POST /tun/m/unassign/") {
                        let ep = first
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("")
                            .trim_start_matches("/tun/m/unassign/")
                            .split('?')
                            .next()
                            .unwrap_or("")
                            .to_string();
                        un.l().push(ep.clone());
                        if ep == "m-s-existing" {
                            ex.store(false, Ordering::SeqCst);
                        } else if ep == "m-s-leaked" {
                            lk.store(false, Ordering::SeqCst);
                        }
                        "{}".to_string()
                    } else {
                        let mut arr = Vec::new();
                        if ex.load(Ordering::SeqCst) {
                            arr.push(serde_json::json!({
                                "endpoint": "m-s-existing",
                                "accelerator": "T4",
                                "variant": "GPU",
                                "machineShape": "STANDARD",
                                "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1:9" }
                            }));
                        }
                        if lk.load(Ordering::SeqCst) {
                            arr.push(serde_json::json!({
                                "endpoint": "m-s-leaked",
                                "accelerator": "A100",
                                "variant": "GPU",
                                "machineShape": "HIGH_RAM",
                                "runtimeProxyInfo": { "token": "ptok", "url": "http://127.0.0.1:9" }
                            }));
                        }
                        format!(")]}}'\n{}", serde_json::json!({ "assignments": arr }))
                    };
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        std::env::set_var("MOCK_COLAB_ENV_LOG", &env_log);
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_COLAB_BIN.l() = Some(mock_colab);
        *cli::TEST_RCLONE_CONF.l() = Some(cfg_dir.join("rclone.conf"));

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let app = AppState::new_with(client, ws_dir.clone());

        // Validate grab_config & relay grab bounds in CreateVmRequest
        let prep_ok = validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("grab-ok".into()),
                accelerator: "A100".into(),
                boot_mode: Some("general".into()),
                grab_config: Some(GrabConfig {
                    mode: "persistent".into(),
                    retry_seconds: 15,
                    fallback_gpus: vec!["g4".into(), "L4".into(), "g4".into()],
                }),
                ..Default::default()
            },
        )
        .expect("persistent grab config with deduplicated fallbacks must pass");
        assert_eq!(prep_ok.validated_grab_cfg.mode, "persistent");
        assert_eq!(prep_ok.validated_grab_cfg.fallback_gpus, vec!["G4", "L4"]);

        assert!(validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("grab-bad".into()),
                accelerator: "T4".into(),
                grab_config: Some(GrabConfig {
                    mode: "persistent".into(),
                    retry_seconds: 2,
                    fallback_gpus: vec![],
                }),
                ..Default::default()
            },
        )
        .is_err());

        // Run grab_colab_vm_loop in fast mode with fallback G4: A100 503 -> immediate G4 success with COLAB_CLI_ASSIGN_RETRIES=0
        let grab_opts = GrabLoopOptions {
            name: "fast-fb-vm".into(),
            primary_gpu: "A100".into(),
            fallback_gpus: vec!["G4".into()],
            high_mem: true,
            auth_mode: None,
            mode: "fast".into(),
            budget_minutes: None,
            retry_seconds: 15,
            cancel: None,
        };
        let got_gpu = relay::grab_colab_vm_loop(&grab_opts, None, None, &|_| {}, &|_| {})
            .await
            .expect("fallback G4 should allocate");
        assert_eq!(got_gpu, "G4");

        let logged = std::fs::read_to_string(&env_log).unwrap();
        let entries: Vec<Value> = logged
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["retries"], "0");
        assert_eq!(entries[0]["timeout"], "60");
        assert_eq!(entries[1]["retries"], "0");

        // Test cancel_create_vm_inner: cancels active grab, unassigns newly leaked m-s-leaked, leaves pre-existing m-s-existing untouched!
        let cancel_tok = tokio_util::sync::CancellationToken::new();
        let pre_eps = Arc::new(Mutex::new(
            ["m-s-existing".to_string()].into_iter().collect(),
        ));
        let pre_invalid = Arc::new(std::sync::atomic::AtomicBool::new(false));
        app.try_claim_busy("cancel-me", "creating");
        cli::register_workspace_entry("cancel-me", "ws", Some(&ws_dir.to_string_lossy()), None);
        app.creating_vms.l().insert(
            "cancel-me".into(),
            CreatingVmMeta {
                accelerator: "A100".into(),
                high_mem: true,
                workspace_name: "ws".into(),
                workspace_path: Some(ws_dir.to_string_lossy().to_string()),
                description: None,
                had_ws_entry: false,
                cancel: cancel_tok.clone(),
                child_pid: Arc::new(Mutex::new(None)),
                pre_provision_endpoints: pre_eps,
                pre_provision_snapshot_invalid: pre_invalid,
                grab_status: Arc::new(Mutex::new(None)),
            },
        );

        let receipt = cancel_create_vm_inner(&app, "cancel-me")
            .await
            .expect("cancel_create_vm_inner should succeed");
        assert!(cancel_tok.is_cancelled());
        assert!(receipt.verified_unassigned);
        assert_eq!(receipt.endpoint.as_deref(), Some("m-s-leaked"));
        assert!(!leaked_alive.load(Ordering::SeqCst));
        assert!(
            existing_alive.load(Ordering::SeqCst),
            "pre-existing endpoint must NEVER be unassigned"
        );
        assert_eq!(*unassigned.l(), vec!["m-s-leaked".to_string()]);
        assert!(!app.creating_vms.l().contains_key("cancel-me"));
        assert!(!cli::has_workspace_entry("cancel-me"));

        std::env::remove_var("MOCK_COLAB_ENV_LOG");
        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn sha1_digest(data: &[u8]) -> [u8; 20] {
        let mut msg = data.to_vec();
        let bit_len = (data.len() as u64) * 8;
        msg.push(0x80);
        while (msg.len() % 64) != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_be_bytes());
        let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
        for chunk in msg.chunks_exact(64) {
            let mut w = [0u32; 80];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    chunk[4 * i],
                    chunk[4 * i + 1],
                    chunk[4 * i + 2],
                    chunk[4 * i + 3],
                ]);
            }
            for i in 16..80 {
                w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
            }
            let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
            for (i, &wi) in w.iter().enumerate() {
                let (f, k) = match i {
                    0..=19 => ((b & c) | ((!b) & d), 0x5A827999),
                    20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                    40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                    _ => (b ^ c ^ d, 0xCA62C1D6),
                };
                let temp = a
                    .rotate_left(5)
                    .wrapping_add(f)
                    .wrapping_add(e)
                    .wrapping_add(k)
                    .wrapping_add(wi);
                e = d;
                d = c;
                c = b.rotate_left(30);
                b = a;
                a = temp;
            }
            h[0] = h[0].wrapping_add(a);
            h[1] = h[1].wrapping_add(b);
            h[2] = h[2].wrapping_add(c);
            h[3] = h[3].wrapping_add(d);
            h[4] = h[4].wrapping_add(e);
        }
        let mut out = [0u8; 20];
        for (i, val) in h.iter().enumerate() {
            out[4 * i..4 * i + 4].copy_from_slice(&val.to_be_bytes());
        }
        out
    }

    async fn read_ws_frame(sock: &mut tokio::net::TcpStream) -> Option<(u8, Vec<u8>)> {
        use tokio::io::AsyncReadExt;
        let mut hdr = [0u8; 2];
        sock.read_exact(&mut hdr).await.ok()?;
        let opcode = hdr[0] & 0x0f;
        let masked = (hdr[1] & 0x80) != 0;
        let mut len = (hdr[1] & 0x7f) as usize;
        if len == 126 {
            let mut ext = [0u8; 2];
            sock.read_exact(&mut ext).await.ok()?;
            len = u16::from_be_bytes(ext) as usize;
        } else if len == 127 {
            let mut ext = [0u8; 8];
            sock.read_exact(&mut ext).await.ok()?;
            len = u64::from_be_bytes(ext) as usize;
        }
        let mut mask = [0u8; 4];
        if masked {
            sock.read_exact(&mut mask).await.ok()?;
        }
        let mut payload = vec![0u8; len];
        sock.read_exact(&mut payload).await.ok()?;
        if masked {
            for (i, b) in payload.iter_mut().enumerate() {
                *b ^= mask[i % 4];
            }
        }
        Some((opcode, payload))
    }

    async fn write_ws_frame(sock: &mut tokio::net::TcpStream, opcode: u8, payload: &[u8]) {
        use tokio::io::AsyncWriteExt;
        let mut frame = Vec::with_capacity(payload.len() + 10);
        frame.push(0x80 | (opcode & 0x0f));
        let len = payload.len();
        if len < 126 {
            frame.push(len as u8);
        } else if len <= 65535 {
            frame.push(126);
            frame.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            frame.push(127);
            frame.extend_from_slice(&(len as u64).to_be_bytes());
        }
        frame.extend_from_slice(payload);
        let _ = sock.write_all(&frame).await;
    }

    #[tokio::test]
    async fn test_issue_89_pause_and_delete_invalidate_inflight_and_queued_keep_alive() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let hits = Arc::new(AtomicUsize::new(0));
        let block_next = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let arrived = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let hits_s = hits.clone();
        let block_s = block_next.clone();
        let arrived_s = arrived.clone();
        let release_s = release.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let hits_c = hits_s.clone();
                let block_c = block_s.clone();
                let arrived_c = arrived_s.clone();
                let release_c = release_s.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf).await;
                    hits_c.fetch_add(1, Ordering::SeqCst);
                    if block_c.load(Ordering::SeqCst) {
                        arrived_c.notify_waiters();
                        release_c.notified().await;
                    }
                    let body = r#"[{"id":"k-busy","name":"python3","execution_state":"busy","connections":1}]"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka89_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        let a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-89".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "proxy-89".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        set_vm_keep_alive_inner(&app, "vm-89", KeepAliveConfig::default())
            .await
            .unwrap();
        let ep1 = app.current_keep_alive_epoch("vm-89");

        // 1. Pause while tick #1 is blocked in Stage 1 (/api/kernels) -> must cancel & return None
        let app_t1 = app.clone();
        let a_t1 = a.clone();
        let t1 = tokio::spawn(async move {
            run_keep_alive_tick_for_vm(&app_t1, "vm-89", &a_t1, None, true, true, Some(ep1)).await
        });
        arrived.notified().await;
        set_vm_keep_alive_inner(
            &app,
            "vm-89",
            KeepAliveConfig {
                enabled: false,
                interval_seconds: 90,
                kernel_ping: true,
            },
        )
        .await
        .unwrap();
        release.notify_waiters();
        assert!(t1.await.unwrap().is_none(), "paused tick must abort and return None");
        assert!(!app.keep_alive.l().get("vm-89").unwrap().in_flight);

        // 2. Queued force=true tick with stale epoch from before pause must do 0 HTTP calls
        let hits_before = hits.load(Ordering::SeqCst);
        let stale_res = run_keep_alive_tick_for_vm(&app, "vm-89", &a, None, true, true, Some(ep1)).await;
        assert!(stale_res.is_none());
        assert_eq!(hits.load(Ordering::SeqCst), hits_before);

        // 3. Re-enable and start tick #2, delete VM via cleanup_local_vm_artifacts while in flight -> must NOT re-insert!
        set_vm_keep_alive_inner(&app, "vm-89", KeepAliveConfig::default())
            .await
            .unwrap();
        let ep2 = app.current_keep_alive_epoch("vm-89");
        let app_t2 = app.clone();
        let a_t2 = a.clone();
        let t2 = tokio::spawn(async move {
            run_keep_alive_tick_for_vm(&app_t2, "vm-89", &a_t2, None, true, true, Some(ep2)).await
        });
        arrived.notified().await;
        app.set_busy("vm-89", Some("deleting"));
        cleanup_local_vm_artifacts(&app, "vm-89");
        release.notify_waiters();
        assert!(t2.await.unwrap().is_none(), "deleted VM tick must return None");
        assert!(
            !app.keep_alive.l().contains_key("vm-89"),
            "late tick must NEVER re-insert deleted VM into app.keep_alive"
        );
        // snapshot_keep_alive_for_vm while deleting must also skip inserting
        let snap_del = snapshot_keep_alive_for_vm(&app, "vm-89", Some("m-s-89"), &cli::read_session_profiles());
        assert_eq!(snap_del.state, "paused");
        assert!(!app.keep_alive.l().contains_key("vm-89"));

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_90_handover_and_reenable_reset_and_reject_stale_endpoint() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let block_ep_a = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ep_a_arrived = Arc::new(tokio::sync::Notify::new());
        let ep_a_release = Arc::new(tokio::sync::Notify::new());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let block_s = block_ep_a.clone();
        let arrived_s = ep_a_arrived.clone();
        let release_s = ep_a_release.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let block_c = block_s.clone();
                let arrived_c = arrived_s.clone();
                let release_c = release_s.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.contains("tok-a") && block_c.load(Ordering::SeqCst) {
                        arrived_c.notify_waiters();
                        release_c.notified().await;
                    }
                    let body = r#"[{"id":"k-busy","name":"python3","execution_state":"busy","connections":1}]"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka90_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        let assign_a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-A".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-a".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };
        let assign_b = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-B".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-b".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        set_vm_keep_alive_inner(&app, "relay-90", KeepAliveConfig::default())
            .await
            .unwrap();
        let st_a = run_keep_alive_tick_for_vm(&app, "relay-90", &assign_a, None, true, true, None)
            .await
            .unwrap();
        assert_eq!(st_a.success_count, 1);
        assert!(st_a.tunnel_ok);
        assert!(st_a.last_ok_at.is_some());

        // Start hanging tick on endpoint A, then switch VM to endpoint B via snapshot_keep_alive_for_vm
        block_ep_a.store(true, Ordering::SeqCst);
        let app_bg = app.clone();
        let assign_a_bg = assign_a.clone();
        let late_tick = tokio::spawn(async move {
            run_keep_alive_tick_for_vm(&app_bg, "relay-90", &assign_a_bg, None, true, true, None)
                .await
        });
        ep_a_arrived.notified().await;

        let profiles = cli::read_session_profiles();
        let snap_b = snapshot_keep_alive_for_vm(&app, "relay-90", Some("m-s-B"), &profiles);
        assert_eq!(snap_b.state, "idle");
        assert!(!snap_b.tunnel_ok);
        assert!(snap_b.last_ok_at.is_none());
        assert_eq!(snap_b.success_count, 0);

        // Release old endpoint A response -> late tick must be discarded and NOT credit endpoint B
        ep_a_release.notify_waiters();
        assert!(late_tick.await.unwrap().is_none());
        let snap_b_after = snapshot_keep_alive_for_vm(&app, "relay-90", Some("m-s-B"), &profiles);
        assert_eq!(snap_b_after.state, "idle");
        assert!(!snap_b_after.tunnel_ok);
        assert!(snap_b_after.last_ok_at.is_none());
        assert_eq!(snap_b_after.success_count, 0);

        // Endpoint B's own tick activates it
        let st_b = run_keep_alive_tick_for_vm(&app, "relay-90", &assign_b, None, true, true, None)
            .await
            .unwrap();
        assert_eq!(st_b.state, "active");
        assert_eq!(st_b.success_count, 1);
        assert!(st_b.last_ok_at.is_some());

        // Pause and re-enable must reset prior verification until re-verified
        set_vm_keep_alive_inner(
            &app,
            "relay-90",
            KeepAliveConfig {
                enabled: false,
                interval_seconds: 90,
                kernel_ping: true,
            },
        )
        .await
        .unwrap();
        let st_re = set_vm_keep_alive_inner(&app, "relay-90", KeepAliveConfig::default())
            .await
            .unwrap();
        assert_eq!(st_re.state, "idle");
        assert!(!st_re.tunnel_ok);
        assert!(st_re.last_ok_at.is_none());
        assert_eq!(st_re.success_count, 0);

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_91_ping_preserves_paused_policy_and_reports_failure() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let mode = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mode_s = mode.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let cur_mode = mode_s.load(Ordering::SeqCst);
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let first = req.lines().next().unwrap_or("");
                    let (status, body) = if first.starts_with("GET /tun/m/assignments") {
                        (
                            "200 OK".to_string(),
                            format!(
                                r#"{{"assignments":[{{"accelerator":"T4","endpoint":"m-s-91","variant":"GPU","machineShape":"STANDARD","runtimeProxyInfo":{{"token":"proxy-91","tokenExpiresInSeconds":3600,"url":"http://{addr}"}}}}]}}"#
                            ),
                        )
                    } else if first.starts_with("GET /api/kernels") {
                        if cur_mode == 0 {
                            ("503 Service Unavailable".to_string(), "upstream 503".to_string())
                        } else {
                            (
                                "200 OK".to_string(),
                                r#"[{"id":"k-busy-91","name":"python3","execution_state":"busy","connections":1}]"#.to_string(),
                            )
                        }
                    } else {
                        ("404 Not Found".to_string(), "{}".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka91_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            cfg_dir.join("sessions.json"),
            r#"{"vm-91":{"endpoint":"m-s-91"}}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        let paused_cfg = KeepAliveConfig {
            enabled: false,
            interval_seconds: 120,
            kernel_ping: true,
        };
        set_vm_keep_alive_inner(&app, "vm-91", paused_cfg.clone())
            .await
            .unwrap();

        // 1. Ping against 503 -> preserves enabled=false AND reports real error
        mode.store(0, Ordering::SeqCst);
        let st_err = ping_vm_keep_alive_inner(&app, "vm-91")
            .await
            .expect("ping_vm_keep_alive_inner should return KeepAliveStatus");
        assert!(!st_err.enabled, "manual ping must NEVER enable paused policy");
        assert_eq!(st_err.state, "error");
        assert!(!st_err.tunnel_ok);
        assert_eq!(st_err.kernel_ping_state, "error");
        assert!(st_err.last_error.is_some());
        assert_eq!(
            cli::get_session_profile("vm-91").and_then(|p| p.keep_alive),
            Some(paused_cfg.clone())
        );

        // 2. Ping against 200 busy kernel -> preserves enabled=false & reports skipped_busy
        mode.store(1, Ordering::SeqCst);
        let st_ok = ping_vm_keep_alive_inner(&app, "vm-91")
            .await
            .expect("ping_vm_keep_alive_inner should return KeepAliveStatus");
        assert!(!st_ok.enabled);
        assert_eq!(st_ok.state, "paused");
        assert!(st_ok.tunnel_ok);
        assert_eq!(st_ok.kernel_ping_state, "skipped_busy");
        assert!(st_ok.last_error.is_none());
        assert_eq!(
            cli::get_session_profile("vm-91").and_then(|p| p.keep_alive),
            Some(paused_cfg)
        );

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_92_93_99_remote_kernel_ping_ws_protocol_and_no_kernel_spawn() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }

        // mode: 0 = empty /api/kernels (#93), 1 = idle kernel ok (#99), 2 = idle kernel execute_reply error (#92)
        let mode = Arc::new(AtomicUsize::new(0));
        let post_kernel_spawns = Arc::new(AtomicUsize::new(0));
        let recorded_executes = Arc::new(Mutex::new(Vec::<Value>::new()));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mode_s = mode.clone();
        let spawns_s = post_kernel_spawns.clone();
        let execs_s = recorded_executes.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let mode_c = mode_s.clone();
                let spawns_c = spawns_s.clone();
                let execs_c = execs_s.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        let n = sock.read(&mut tmp).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let req = String::from_utf8_lossy(&buf).to_string();
                    let first = req.lines().next().unwrap_or("");
                    let cur_mode = mode_c.load(Ordering::SeqCst);

                    if req.to_ascii_lowercase().contains("upgrade: websocket") {
                        let ws_key = req
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                if k.trim().eq_ignore_ascii_case("sec-websocket-key") {
                                    Some(v.trim().to_string())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        let accept_raw = format!("{ws_key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
                        use base64::Engine as _;
                        let accept = base64::engine::general_purpose::STANDARD
                            .encode(sha1_digest(accept_raw.as_bytes()));
                        let hs = format!(
                            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                        );
                        let _ = sock.write_all(hs.as_bytes()).await;

                        while let Some((opcode, payload)) = read_ws_frame(&mut sock).await {
                            if opcode == 0x8 {
                                write_ws_frame(&mut sock, 0x8, &[]).await;
                                break;
                            }
                            if opcode == 0x9 {
                                write_ws_frame(&mut sock, 0xA, &payload).await;
                                continue;
                            }
                            let json_slice: &[u8] = if payload.first() == Some(&b'{') {
                                &payload
                            } else if payload.len() >= 12 {
                                let off0 = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
                                let off1 = u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
                                if off0 <= off1 && off1 <= payload.len() {
                                    &payload[off0..off1]
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            };
                            let Ok(msg) = serde_json::from_slice::<Value>(json_slice) else {
                                continue;
                            };
                            let msg_type = msg["header"]["msg_type"].as_str().unwrap_or("");
                            let parent_hdr = msg["header"].clone();
                            if msg_type == "execute_request" {
                                execs_c.l().push(msg["content"].clone());
                                let busy_msg = serde_json::json!({
                                    "channel": "iopub",
                                    "msg_id": "m-busy",
                                    "msg_type": "status",
                                    "header": {"msg_id": "m-busy", "msg_type": "status"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": {"execution_state": "busy"}
                                });
                                write_ws_frame(&mut sock, 0x1, busy_msg.to_string().as_bytes()).await;

                                let idle_msg = serde_json::json!({
                                    "channel": "iopub",
                                    "msg_id": "m-idle",
                                    "msg_type": "status",
                                    "header": {"msg_id": "m-idle", "msg_type": "status"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": {"execution_state": "idle"}
                                });
                                write_ws_frame(&mut sock, 0x1, idle_msg.to_string().as_bytes()).await;

                                let reply_content = if cur_mode == 2 {
                                    serde_json::json!({
                                        "status": "error",
                                        "ename": "PolicyError",
                                        "evalue": "execution rejected"
                                    })
                                } else {
                                    serde_json::json!({
                                        "status": "ok",
                                        "execution_count": 1
                                    })
                                };
                                let reply_msg = serde_json::json!({
                                    "channel": "shell",
                                    "msg_id": "m-reply",
                                    "msg_type": "execute_reply",
                                    "header": {"msg_id": "m-reply", "msg_type": "execute_reply"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": reply_content
                                });
                                write_ws_frame(&mut sock, 0x1, reply_msg.to_string().as_bytes()).await;
                            }
                        }
                        return;
                    }

                    let (status, body) = if first.starts_with("POST /api/kernels") {
                        spawns_c.fetch_add(1, Ordering::SeqCst);
                        ("500 Internal Server Error", r#"{"error":"spawn forbidden"}"#)
                    } else if first.starts_with("GET /api/kernels/k-idle-1") {
                        if cur_mode == 0 {
                            ("404 Not Found", r#"{"message":"not found"}"#)
                        } else {
                            ("200 OK", r#"{"id":"k-idle-1","name":"python3","execution_state":"idle","connections":0}"#)
                        }
                    } else if first.starts_with("GET /api/kernels") {
                        if cur_mode == 0 {
                            ("200 OK", r#"[]"#)
                        } else {
                            ("200 OK", r#"[{"id":"k-idle-1","name":"python3","execution_state":"idle","connections":0}]"#)
                        }
                    } else {
                        ("404 Not Found", "{}")
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka92_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);
        set_vm_keep_alive_inner(&app, "vm-92", KeepAliveConfig::default())
            .await
            .unwrap();

        let a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-92".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "proxy-92".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        // 1. #93: Empty /api/kernels + stale kernel_hint -> skipped_no_kernel, 0 POST /api/kernels
        mode.store(0, Ordering::SeqCst);
        let st_empty = run_keep_alive_tick_for_vm(
            &app,
            "vm-92",
            &a,
            Some("retired-kid".into()),
            true,
            true,
            None,
        )
        .await
        .unwrap();
        assert!(st_empty.tunnel_ok);
        assert_eq!(st_empty.kernel_ping_state, "skipped_no_kernel");
        assert_eq!(st_empty.state, "active");
        assert_eq!(post_kernel_spawns.load(Ordering::SeqCst), 0);

        // Also test retired-kid direct call to execute_remote_kernel_ping returns NoKernel with 0 POST
        let direct_retired = execute_remote_kernel_ping(&a, "k-idle-1", None)
            .await
            .expect("retired kernel must return Ok(NoKernel)");
        assert_eq!(direct_retired, KernelPingOutcome::NoKernel);
        assert_eq!(post_kernel_spawns.load(Ordering::SeqCst), 0);

        // 2. #99: Idle kernel sends silent=True, store_history=False, allow_stdin=False and succeeds
        mode.store(1, Ordering::SeqCst);
        let st_ok = run_keep_alive_tick_for_vm(&app, "vm-92", &a, None, true, true, None)
            .await
            .unwrap();
        assert!(st_ok.tunnel_ok);
        assert_eq!(st_ok.kernel_ping_state, "ok");
        assert_eq!(st_ok.state, "active");
        {
            let execs = recorded_executes.l();
            assert_eq!(execs.len(), 1);
            assert_eq!(execs[0]["code"], "pass");
            assert_eq!(execs[0]["silent"], true);
            assert_eq!(execs[0]["store_history"], false);
            assert_eq!(execs[0]["allow_stdin"], false);
        }

        // 3. #92: execute_reply status="error" -> execute_remote_kernel_ping errors & state="error"
        mode.store(2, Ordering::SeqCst);
        let st_err = run_keep_alive_tick_for_vm(&app, "vm-92", &a, None, true, true, None)
            .await
            .unwrap();
        assert!(st_err.tunnel_ok);
        assert_eq!(st_err.kernel_ping_state, "error");
        assert_eq!(st_err.state, "error");
        assert!(st_err.last_error.as_deref().unwrap_or("").contains("PolicyError"));
        assert_eq!(post_kernel_spawns.load(Ordering::SeqCst), 0);

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_93_dead_terminating_restarting_kernels_skipped_without_error() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let post_kernel_spawns = Arc::new(AtomicUsize::new(0));
        let ws_attempts = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let spawns_s = post_kernel_spawns.clone();
        let ws_s = ws_attempts.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let spawns_c = spawns_s.clone();
                let ws_c = ws_s.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    let first = req.lines().next().unwrap_or("");
                    if req.to_ascii_lowercase().contains("upgrade: websocket") {
                        ws_c.fetch_add(1, Ordering::SeqCst);
                    }
                    let (status, body) = if first.starts_with("POST /api/kernels") {
                        spawns_c.fetch_add(1, Ordering::SeqCst);
                        ("500 Internal Server Error", r#"{"error":"spawn forbidden"}"#)
                    } else if first.starts_with("GET /api/kernels") {
                        (
                            "200 OK",
                            r#"[{"id":"k1","name":"python3","execution_state":"dead","connections":0},{"id":"k2","name":"python3","execution_state":"terminating","connections":0},{"id":"k3","name":"python3","execution_state":"restarting","connections":0}]"#,
                        )
                    } else {
                        ("404 Not Found", "{}")
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka93_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);
        set_vm_keep_alive_inner(&app, "vm-93", KeepAliveConfig::default())
            .await
            .unwrap();

        let a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-93".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "proxy-93".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        let st = run_keep_alive_tick_for_vm(
            &app,
            "vm-93",
            &a,
            Some("k1".into()),
            true,
            true,
            None,
        )
        .await
        .expect("keep-alive tick should return status");
        assert!(st.tunnel_ok);
        assert_eq!(st.kernel_ping_state, "skipped_no_kernel");
        assert_eq!(st.state, "active");
        assert!(st.last_error.is_none());
        assert_eq!(post_kernel_spawns.load(Ordering::SeqCst), 0);
        assert_eq!(ws_attempts.load(Ordering::SeqCst), 0);

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_95_active_chat_only_skips_target_vm() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let _ = sock.read(&mut buf).await;
                    // Return empty kernels so vm-b completes without needing Python WS,
                    // whereas vm-a skips via local_busy ("skipped_busy") before kernel selection!
                    let body = r#"[]"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka95_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        set_vm_keep_alive_inner(&app, "vm-a", KeepAliveConfig::default())
            .await
            .unwrap();
        set_vm_keep_alive_inner(&app, "vm-b", KeepAliveConfig::default())
            .await
            .unwrap();

        // Active chat ONLY on vm-a
        app.active_chats.l().insert(
            "req-1".into(),
            ActiveChatEntry {
                cancel: tokio_util::sync::CancellationToken::new(),
                vm_name: Some("vm-a".into()),
            },
        );

        let assign_a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-a".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-a".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };
        let assign_b = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-b".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-b".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        let st_a = run_keep_alive_tick_for_vm(&app, "vm-a", &assign_a, None, true, true, None)
            .await
            .unwrap();
        let st_b = run_keep_alive_tick_for_vm(&app, "vm-b", &assign_b, None, true, true, None)
            .await
            .unwrap();

        assert_eq!(st_a.kernel_ping_state, "skipped_busy", "vm-a must skip due to active chat on vm-a");
        assert_ne!(st_b.kernel_ping_state, "skipped_busy", "vm-b must NOT be suppressed by chat on vm-a");
        assert_eq!(st_b.kernel_ping_state, "skipped_no_kernel");

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_96_disabled_keep_alive_normalizes_bad_interval() {
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_ka96_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let fake_colab = tmp.join("fake_colab.cmd");
        std::fs::write(&fake_colab, "@echo off\r\nexit /b 0\r\n").unwrap();

        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);
        *cli::TEST_COLAB_BIN.l() = Some(fake_colab);

        let client = ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_path);
        let app = AppState::new_with(client, ws_dir);

        // Disabled keep-alive with out-of-range intervals normalizes to 90s
        let v_low = KeepAliveConfig {
            enabled: false,
            interval_seconds: 29,
            kernel_ping: true,
        }
        .validate()
        .expect("disabled keep-alive with 29s must normalize");
        assert_eq!(v_low.interval_seconds, 90);

        let v_high = KeepAliveConfig {
            enabled: false,
            interval_seconds: 601,
            kernel_ping: false,
        }
        .validate()
        .expect("disabled keep-alive with 601s must normalize");
        assert_eq!(v_high.interval_seconds, 90);

        // Disabled keep-alive with valid interval preserves it
        let v_ok = KeepAliveConfig {
            enabled: false,
            interval_seconds: 180,
            kernel_ping: true,
        }
        .validate()
        .expect("disabled keep-alive with valid interval must pass");
        assert_eq!(v_ok.interval_seconds, 180);

        // Enabled keep-alive with out-of-range intervals still errors
        assert!(KeepAliveConfig {
            enabled: true,
            interval_seconds: 29,
            kernel_ping: true,
        }
        .validate()
        .is_err());
        assert!(KeepAliveConfig {
            enabled: true,
            interval_seconds: 601,
            kernel_ping: true,
        }
        .validate()
        .is_err());

        // validate_and_prepare_create_vm succeeds with disabled stale interval=10
        let prep = validate_and_prepare_create_vm(
            &app,
            &CreateVmRequest {
                name: Some("vm-96".into()),
                accelerator: "T4".into(),
                boot_mode: Some("general".into()),
                keep_alive: Some(KeepAliveConfig {
                    enabled: false,
                    interval_seconds: 10,
                    kernel_ping: true,
                }),
                ..Default::default()
            },
        )
        .expect("create_vm validation must not block disabled keep-alive");
        assert!(!prep.validated_keep_alive.enabled);
        assert_eq!(prep.validated_keep_alive.interval_seconds, 90);

        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_97_profile_persistence_failure_surfaces_and_recovers() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;
        let mode = Arc::new(AtomicUsize::new(0));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mode_s = mode.clone();
        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let mode_c = mode_s.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut tmp = [0u8; 4096];
                    loop {
                        let n = sock.read(&mut tmp).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&tmp[..n]);
                        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                            break;
                        }
                    }
                    let req = String::from_utf8_lossy(&buf).to_string();
                    let first = req.lines().next().unwrap_or("");
                    let cur_mode = mode_c.load(Ordering::SeqCst);

                    if req.to_ascii_lowercase().contains("upgrade: websocket") {
                        let ws_key = req
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                if k.trim().eq_ignore_ascii_case("sec-websocket-key") {
                                    Some(v.trim().to_string())
                                } else {
                                    None
                                }
                            })
                            .unwrap_or_default();
                        let accept_raw = format!("{ws_key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
                        use base64::Engine as _;
                        let accept = base64::engine::general_purpose::STANDARD
                            .encode(sha1_digest(accept_raw.as_bytes()));
                        let hs = format!(
                            "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n"
                        );
                        let _ = sock.write_all(hs.as_bytes()).await;

                        while let Some((opcode, payload)) = read_ws_frame(&mut sock).await {
                            if opcode == 0x8 {
                                write_ws_frame(&mut sock, 0x8, &[]).await;
                                break;
                            }
                            if opcode == 0x9 {
                                write_ws_frame(&mut sock, 0xA, &payload).await;
                                continue;
                            }
                            let json_slice: &[u8] = if payload.first() == Some(&b'{') {
                                &payload
                            } else if payload.len() >= 12 {
                                let off0 = u32::from_be_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
                                let off1 = u32::from_be_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;
                                if off0 <= off1 && off1 <= payload.len() {
                                    &payload[off0..off1]
                                } else {
                                    continue;
                                }
                            } else {
                                continue;
                            };
                            let Ok(msg) = serde_json::from_slice::<Value>(json_slice) else {
                                continue;
                            };
                            let msg_type = msg["header"]["msg_type"].as_str().unwrap_or("");
                            let parent_hdr = msg["header"].clone();
                            if msg_type == "execute_request" {
                                let busy_msg = serde_json::json!({
                                    "channel": "iopub",
                                    "msg_id": "m-busy-97",
                                    "msg_type": "status",
                                    "header": {"msg_id": "m-busy-97", "msg_type": "status"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": {"execution_state": "busy"}
                                });
                                write_ws_frame(&mut sock, 0x1, busy_msg.to_string().as_bytes()).await;
                                let idle_msg = serde_json::json!({
                                    "channel": "iopub",
                                    "msg_id": "m-idle-97",
                                    "msg_type": "status",
                                    "header": {"msg_id": "m-idle-97", "msg_type": "status"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": {"execution_state": "idle"}
                                });
                                write_ws_frame(&mut sock, 0x1, idle_msg.to_string().as_bytes()).await;
                                let reply_msg = serde_json::json!({
                                    "channel": "shell",
                                    "msg_id": "m-reply-97",
                                    "msg_type": "execute_reply",
                                    "header": {"msg_id": "m-reply-97", "msg_type": "execute_reply"},
                                    "parent_header": parent_hdr,
                                    "metadata": {},
                                    "content": {"status": "ok", "execution_count": 1}
                                });
                                write_ws_frame(&mut sock, 0x1, reply_msg.to_string().as_bytes()).await;
                            }
                        }
                        return;
                    }

                    let (status, body) = if first.starts_with("GET /tun/m/assignments") {
                        (
                            "200 OK".to_string(),
                            format!(
                                r#"{{"assignments":[{{"accelerator":"T4","endpoint":"m-s-97","variant":"GPU","machineShape":"STANDARD","runtimeProxyInfo":{{"token":"proxy-97","tokenExpiresInSeconds":3600,"url":"http://{addr}"}}}}]}}"#
                            ),
                        )
                    } else if first.starts_with("GET /api/kernels/k-idle-97") {
                        ("200 OK".to_string(), r#"{"id":"k-idle-97","name":"python3","execution_state":"idle","connections":0}"#.to_string())
                    } else if first.starts_with("GET /api/kernels") {
                        if cur_mode == 1 {
                            ("503 Service Unavailable".to_string(), "upstream 503".to_string())
                        } else {
                            (
                                "200 OK".to_string(),
                                r#"[{"id":"k-idle-97","name":"python3","execution_state":"idle","connections":0}]"#.to_string(),
                            )
                        }
                    } else {
                        ("404 Not Found".to_string(), "{}".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka97_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        std::fs::write(
            cfg_dir.join("sessions.json"),
            r#"{"vm-97":{"endpoint":"m-s-97"}}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        // Replace pilot_session_profiles.json with a directory so saving fails
        let prof_path = cli::session_profiles_path();
        std::fs::create_dir_all(&prof_path).unwrap();

        // 1. Relay pre-alloc failure returns Err AND rolls back workspace / keep_alive / busy
        app.try_claim_busy("relay-97", "creating");
        app.keep_alive.l().insert(
            "relay-97".into(),
            VmKeepAliveRuntime::from_config(KeepAliveConfig::default(), None),
        );
        cli::register_workspace_entry("relay-97", "ws", None, None);
        let err = persist_relay_profile_before_alloc(
            &app,
            "relay-97",
            false,
            &cli::SessionProfile::default(),
        )
        .expect_err("unwritable profiles path must fail pre-alloc relay creation");
        assert!(err.contains("档案"));
        assert!(!cli::has_workspace_entry("relay-97"));
        assert!(!app.keep_alive.l().contains_key("relay-97"));
        assert!(!app.busy.l().contains_key("relay-97"));

        // 2. Post-alloc persistence failure surfaces independently without corrupting runtime state or last_error
        app.keep_alive.l().insert(
            "vm-97".into(),
            VmKeepAliveRuntime::from_config(
                KeepAliveConfig {
                    enabled: false,
                    interval_seconds: 90,
                    kernel_ping: true,
                },
                None,
            ),
        );
        let res = cli::save_session_profile("vm-97", &cli::SessionProfile::default());
        record_profile_persistence_result(&app, "vm-97", res);
        let st_fail = app.keep_alive.l().get("vm-97").unwrap().to_status();
        assert!(!st_fail.profile_saved);
        assert!(st_fail.persistence_error.is_some());
        assert_eq!(st_fail.state, "paused", "persistence failure must keep paused state");
        assert!(st_fail.last_error.is_none(), "persistence_error must not leak into last_error");

        // 2b. Manual ping succeeding on paused VM with profile_saved=false keeps state="paused", last_error=None
        if cli::colab_python_bin().exists() {
            mode.store(0, Ordering::SeqCst);
            let st_ping_ok = ping_vm_keep_alive_inner(&app, "vm-97")
                .await
                .expect("ping_vm_keep_alive_inner should succeed");
            assert!(!st_ping_ok.enabled);
            assert!(st_ping_ok.tunnel_ok);
            assert_eq!(st_ping_ok.kernel_ping_state, "ok");
            assert_eq!(st_ping_ok.state, "paused");
            assert!(st_ping_ok.last_error.is_none());
            assert!(!st_ping_ok.profile_saved);
            assert!(st_ping_ok.persistence_error.is_some());
        }

        // 2c. Manual ping failing with 503 sets state="error" & last_error=Some(...) alongside persistence_error
        mode.store(1, Ordering::SeqCst);
        let st_ping_err = ping_vm_keep_alive_inner(&app, "vm-97")
            .await
            .expect("ping_vm_keep_alive_inner should return KeepAliveStatus on 503");
        assert!(!st_ping_err.tunnel_ok);
        assert_eq!(st_ping_err.state, "error");
        assert!(st_ping_err.last_error.is_some());
        assert!(!st_ping_err.profile_saved);
        assert!(st_ping_err.persistence_error.is_some());

        // 3. Restore writable file and call set_vm_keep_alive_inner -> clears persistence_error
        std::fs::remove_dir_all(&prof_path).unwrap();
        let st_ok = set_vm_keep_alive_inner(&app, "vm-97", KeepAliveConfig::default())
            .await
            .expect("retry set_vm_keep_alive_inner should succeed");
        assert!(st_ok.profile_saved);
        assert!(st_ok.persistence_error.is_none());
        assert_eq!(st_ok.state, "idle");
        assert!(st_ok.last_error.is_none());

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_100_slow_vm_does_not_delay_fast_vm_next_round() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let _env_guard = cli::TEST_ENV_LOCK.lock().await;

        let fast_hits = Arc::new(AtomicUsize::new(0));
        let slow_hits = Arc::new(AtomicUsize::new(0));
        let slow_arrived = Arc::new(tokio::sync::Notify::new());
        let slow_release = Arc::new(tokio::sync::Notify::new());

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let fh = fast_hits.clone();
        let sh = slow_hits.clone();
        let sa = slow_arrived.clone();
        let sr = slow_release.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let fh_c = fh.clone();
                let sh_c = sh.clone();
                let sa_c = sa.clone();
                let sr_c = sr.clone();
                tokio::spawn(async move {
                    let mut buf = [0u8; 4096];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let req = String::from_utf8_lossy(&buf[..n]).to_string();
                    if req.contains("tok-slow") {
                        sh_c.fetch_add(1, Ordering::SeqCst);
                        sa_c.notify_waiters();
                        sr_c.notified().await;
                    } else {
                        fh_c.fetch_add(1, Ordering::SeqCst);
                    }
                    let body = r#"[{"id":"k-busy","name":"python3","execution_state":"busy","connections":1}]"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp = std::env::temp_dir().join(format!("pilot_ka100_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("cfg");
        let ws_dir = tmp.join("ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();
        let tok_path = cfg_dir.join("token.json");
        std::fs::write(
            &tok_path,
            r#"{"token":"ka-oauth","refresh_token":"rt","client_id":"c","client_secret":"s","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir);

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_path);
        let app = AppState::new_with(client, ws_dir);

        let fast_a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-fast".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-fast".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };
        let slow_a = Assignment {
            accelerator: "T4".into(),
            endpoint: "m-s-slow".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: crate::colab::RuntimeProxyInfo {
                token: "tok-slow".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        let cfg30 = KeepAliveConfig {
            enabled: true,
            interval_seconds: 30,
            kernel_ping: true,
        };
        set_vm_keep_alive_inner(&app, "fast-vm", cfg30.clone())
            .await
            .unwrap();
        set_vm_keep_alive_inner(&app, "slow-vm", cfg30)
            .await
            .unwrap();

        *app.last_assignments.l() = Some((vec![fast_a, slow_a], Instant::now(), 1));
        {
            let mut ov = app.overview.write().await;
            ov.vms = vec![
                VmView {
                    name: "fast-vm".into(),
                    endpoint: Some("m-s-fast".into()),
                    on_server: true,
                    ..Default::default()
                },
                VmView {
                    name: "slow-vm".into(),
                    endpoint: Some("m-s-slow".into()),
                    on_server: true,
                    ..Default::default()
                },
            ];
        }

        // Round 1: dispatch must return immediately (< 100ms) while slow-vm stays blocked
        let t0 = Instant::now();
        let n1 = dispatch_keep_alive_scheduler_pass(&app).await;
        assert_eq!(n1, 2);
        assert!(t0.elapsed() < Duration::from_millis(100));
        slow_arrived.notified().await;

        // Wait briefly for fast-vm's round 1 task to finish
        for _ in 0..50 {
            if !app.keep_alive.l().get("fast-vm").unwrap().in_flight {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(fast_hits.load(Ordering::SeqCst), 1);
        assert_eq!(slow_hits.load(Ordering::SeqCst), 1);
        assert!(app.keep_alive.l().get("slow-vm").unwrap().in_flight);

        // Simulate next interval elapsed for fast-vm while slow-vm is still blocked
        app.keep_alive.l().get_mut("fast-vm").unwrap().last_attempt_instant =
            Some(Instant::now() - Duration::from_secs(31));

        let n2 = dispatch_keep_alive_scheduler_pass(&app).await;
        assert_eq!(n2, 1, "only fast-vm should be dispatched while slow-vm is in_flight");
        for _ in 0..50 {
            if fast_hits.load(Ordering::SeqCst) >= 2
                && !app.keep_alive.l().get("fast-vm").unwrap().in_flight
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(fast_hits.load(Ordering::SeqCst), 2);
        assert_eq!(slow_hits.load(Ordering::SeqCst), 1, "slow-vm must not be double-dispatched");

        slow_release.notify_waiters();
        for _ in 0..50 {
            if !app.keep_alive.l().get("slow-vm").unwrap().in_flight {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert!(!app.keep_alive.l().get("slow-vm").unwrap().in_flight);

        *cli::TEST_CONFIG_DIR.l() = None;
        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
