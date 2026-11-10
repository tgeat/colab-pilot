//! Colab Pilot — Tauri 2 Rust 后端（工作区与 Colab 虚拟机实时监控仪表盘）
//!
//! - 后台轮询：Colab 服务端计费实例、账户 CU、云端遥测快照、colab-cli 会话与历史、Google Drive 状态
//! - 自动探针注入：自动感知 Antigravity 通过 colab-cli 创建的实例，通过 Contents API + /colab/tty 静默注入遥测守护进程（不阻塞 Kernel，不污染 CLI 会话状态）
//! - 工作区映射：自动将每台 Colab 虚拟机关联到对应的本地工作区（Workspace），汇总资源分配与运行状态

mod cli;
mod colab;
mod relay;

use anyhow::Context;
use cli::{s, CliEventView, RecentSessionView, WorkspaceInfo};
use colab::{Assignment, Ccu, ColabClient};
use futures::future::join_all;
use relay::{RelayConfig, RelayHandle, RelayStatus};
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
const DEFAULT_GDRIVE_DIR: &str = "gdrive:colab_relay";

// ───────────────────────────── 数据结构 ─────────────────────────────

#[derive(Serialize, Clone, Default, Debug)]
pub struct EnvStatus {
    pub colab_bin: String,
    pub colab_ok: bool,
    pub rclone_bin: String,
    pub rclone_ok: bool,
    pub rclone_conf: String,
    pub rclone_conf_ok: bool,
    pub token_ok: bool,
    pub workspace: String,
    pub sessions_path: String,
    pub history_dir: String,
}

#[derive(Serialize, Clone, Default, Debug)]
pub struct GdriveFile {
    pub name: String,
    pub size: i64,
    pub mod_time: String,
    pub is_dir: bool,
}

#[derive(Serialize, Clone, Default, Debug)]
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

#[derive(Serialize, Clone, Default, Debug)]
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
}

#[derive(Serialize, Clone, Default, Debug)]
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
    pub llm_bluegreen: Option<Value>,
}

#[derive(Serialize, Clone, Default, Debug)]
pub struct TerminateReceipt {
    pub name: String,
    pub endpoint: Option<String>,
    pub verified_unassigned: bool,
    pub remaining_endpoints: Vec<String>,
    pub steps: Vec<String>,
    pub ccu: Option<Ccu>,
    pub error: Option<String>,
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

pub struct AppState {
    pub client: ColabClient,
    pub workspace: PathBuf,
    overview: tokio::sync::RwLock<Overview>,
    telemetry: Mutex<HashMap<String, TelemetryEntry>>,
    last_assignments: Mutex<Option<(Vec<Assignment>, Instant)>>,
    last_ccu: Mutex<Option<Ccu>>,
    relays: Mutex<HashMap<String, Arc<RelayHandle>>>,
    busy: Mutex<HashMap<String, String>>,
    probe_cooldown: Mutex<HashMap<String, Instant>>,
    gdrive: tokio::sync::Mutex<(Option<GdriveStatus>, Option<Instant>)>,
    refresh_lock: tokio::sync::Mutex<()>,
}

fn orphan_name(endpoint: &str) -> String {
    let tail: String = endpoint
        .chars()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("orphan-{}", tail.to_lowercase())
}

fn session_endpoint(v: &Value) -> Option<String> {
    v.get("endpoint").and_then(|e| e.as_str()).map(|s| s.to_string())
}

impl AppState {
    fn relay(&self, name: &str) -> Option<Arc<RelayHandle>> {
        self.relays.l().get(name).cloned()
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

    /// 根据会话名（或 orphan-xxxxxx）在服务端计费列表中查找实例
    pub async fn find_assignment_by_name(&self, name: &str) -> Option<Assignment> {
        let sessions = cli::read_sessions();
        let ep = sessions.get(name).and_then(session_endpoint);
        let list = self.client.list_assignments().await.ok()?;
        match ep {
            Some(ep) => list.into_iter().find(|a| a.endpoint == ep),
            None => list.into_iter().find(|a| orphan_name(&a.endpoint) == name),
        }
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
            token_ok: ColabClient::is_authorized(),
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
    let out = cli::run_simple(
        &cli::rclone_bin(),
        &[
            s(&["lsjson", "--retries=5", "--low-level-retries=20"]),
            vec![dir.to_string()],
        ]
        .concat(),
        Duration::from_secs(90),
    )
    .await;
    match out {
        Ok(o) if o.code == 0 => {
            let json_start = o.output.find('[').unwrap_or(0);
            let arr: Vec<Value> = serde_json::from_str(&o.output[json_start..]).unwrap_or_default();
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
                if f.name == "training_done.flag" {
                    st.has_done_flag = true;
                }
                st.files.push(f);
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

const PROBE_BOOTSTRAP: &str = r#"
import os, subprocess, sys, time
pid_file = "/content/.colab_pilot_daemon.pid"
alive = False
if os.path.exists(pid_file):
    try:
        os.kill(int(open(pid_file).read().strip()), 0)
        alive = True
    except Exception:
        alive = False
if not alive:
    subprocess.Popen([sys.executable, "/content/.colab_pilot_daemon.py"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1.0)
print("PROBE_OK", "reused" if alive else "started")
"#;

/// 使用 colab-cli 虚拟环境直接通过 /colab/tty（或 ColabRuntime）静默启动遥测守护进程，
/// 不改写 ~/.config/colab-cli/sessions.json 与 history/*.jsonl，也不中断正在运行的训练任务。
const SILENT_PROBE_LAUNCHER_PY: &str = r#"
import json, sys, time
from urllib.parse import urlparse

url = sys.argv[1]
token = sys.argv[2]

started = False
try:
    import websocket
    parsed = urlparse(url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/colab/tty?colab-runtime-proxy-token={token}"
    ws = websocket.create_connection(ws_url, timeout=8)
    cmd = (
        "python3 -c 'import os,subprocess,sys;"
        "pf=\"/content/.colab_pilot_daemon.pid\";"
        "alive=False\n"
        "try:\n"
        " alive=os.path.exists(pf) and (os.kill(int(open(pf).read().strip()),0) or True)\n"
        "except Exception:\n"
        " alive=False\n"
        "if not alive:\n"
        " subprocess.Popen([sys.executable,\"/content/.colab_pilot_daemon.py\"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)'\n"
    )
    ws.send(json.dumps({"data": cmd + "\n"}))
    time.sleep(1.2)
    ws.close()
    started = True
    print("PROBE_OK via_tty")
except Exception as e:
    pass

if not started:
    from colab_cli.runtime import ColabRuntime
    rt = ColabRuntime(url, token)
    try:
        rt.execute_code('''
import os, subprocess, sys, time
pid_file = "/content/.colab_pilot_daemon.pid"
alive = False
if os.path.exists(pid_file):
    try:
        os.kill(int(open(pid_file).read().strip()), 0)
        alive = True
    except Exception:
        alive = False
if not alive:
    subprocess.Popen([sys.executable, "/content/.colab_pilot_daemon.py"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    time.sleep(0.8)
''', timeout=25)
        print("PROBE_OK via_kernel")
    finally:
        rt.stop()
"#;

pub async fn deploy_probe_for_assignment(app: &Arc<AppState>, a: &Assignment) -> anyhow::Result<()> {
    app.client
        .contents_upload(a, "content/.colab_pilot_daemon.py", TELEMETRY_DAEMON_PY.as_bytes())
        .await?;

    let py = cli::colab_python_bin();
    let args = vec![
        "-c".to_string(),
        SILENT_PROBE_LAUNCHER_PY.to_string(),
        a.runtime_proxy_info.url.clone(),
        a.runtime_proxy_info.token.clone(),
    ];
    let out = cli::run_simple(&py, &args, Duration::from_secs(40)).await?;
    if out.output.contains("PROBE_OK") {
        Ok(())
    } else {
        anyhow::bail!(
            "探针自动注入失败：{}",
            out.output.trim().lines().last().unwrap_or("无输出")
        )
    }
}

pub async fn deploy_probe_inner(app: &Arc<AppState>, name: &str) -> anyhow::Result<()> {
    let a = app
        .find_assignment_by_name(name)
        .await
        .ok_or_else(|| anyhow::anyhow!("服务端未找到实例 [{name}]"))?;
    if let Ok(()) = deploy_probe_for_assignment(app, &a).await {
        return Ok(());
    }
    if !cli::read_sessions().contains_key(name) {
        anyhow::bail!("该实例不在 colab-cli 会话表中，且静默注入未响应");
    }
    let out = cli::run_streaming(
        &cli::colab_bin(),
        &s(&["exec", "-s", name, "--timeout", "60"]),
        Some(PROBE_BOOTSTRAP.to_string()),
        Duration::from_secs(150),
        None,
        None,
        |_| {},
    )
    .await?;
    if out.output.contains("PROBE_OK") {
        Ok(())
    } else {
        anyhow::bail!(
            "探针启动失败：{}",
            out.output.trim().lines().last().unwrap_or("无输出")
        )
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

async fn fetch_telemetry(app: &AppState, a: &Assignment) -> TelemetryEntry {
    match app
        .client
        .contents_read_text(a, "content/.colab_pilot_telemetry.json")
        .await
    {
        Ok(Some(text)) => match serde_json::from_str::<Value>(&text) {
            Ok(mut v) => {
                // 若该实例为蓝绿 LLM 节点且本地 SSH 隧道已打通，直接合并本地隧道 <50ms 的实时计量快照
                let slot = v
                    .pointer("/llm/slot")
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_lowercase());
                let local_port = match slot.as_deref() {
                    Some("blue") => Some(18081u16),
                    Some("green") => Some(18082u16),
                    _ => None,
                };
                if let Some(port) = local_port {
                    if let Some(client) = local_metrics_client() {
                        if let Ok(resp) = client
                            .get(format!("http://127.0.0.1:{port}/pilot/metrics"))
                            .send()
                            .await
                        {
                            if resp.status().is_success() {
                                if let Ok(live_llm) = resp.json::<Value>().await {
                                    // 仅当本机网关返回完整的计量结构时才按字段合并，绝不用残缺 fallback 覆盖云端完整指标
                                    // 且仅当本机快照不比云端快照更旧时才合并（避免旧数据回滚曲线）
                                    let live_ts = live_llm.get("timestamp").and_then(|x| x.as_f64());
                                    let cloud_ts = v.pointer("/llm/timestamp").and_then(|x| x.as_f64());
                                    let not_older = match (live_ts, cloud_ts) {
                                        (Some(l), Some(c)) => l >= c,
                                        _ => true,
                                    };
                                    if not_older
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
        Err(e) => TelemetryEntry { data: None, error: Some(e.to_string()) },
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
    let mut ov = Overview {
        timestamp: chrono::Utc::now().timestamp(),
        env: env.clone(),
        ..Default::default()
    };

    // Google Drive 状态始终在后台异步刷新（绝不阻塞 Colab 计费与秒级资源遥测轮询）
    {
        let mut gd = app.gdrive.lock().await;
        let stale = gd.1.map(|t| t.elapsed() > Duration::from_secs(30)).unwrap_or(true);
        if (stale || force_gdrive) && env.rclone_ok && env.rclone_conf_ok {
            gd.1 = Some(Instant::now());
            let app_bg = app.clone();
            tokio::spawn(async move {
                let status = fetch_gdrive_status(DEFAULT_GDRIVE_DIR).await;
                let mut g = app_bg.gdrive.lock().await;
                g.0 = Some(status);
            });
        }
        ov.gdrive = gd.0.clone();
    }

    // 本地会话表读取涉及磁盘 I/O，放入阻塞线程池执行，绝不阻塞异步运行时
    let sessions = tokio::task::spawn_blocking(cli::read_sessions)
        .await
        .unwrap_or_default();
    let mut assignments: Vec<Assignment> = vec![];

    if !env.token_ok {
        ov.error = Some("未检测到 Colab 登录凭证，请在终端运行 `colab status` 完成 Google 授权".into());
    } else {
        // 高频遥测优化：8 秒内直接复用已缓存的计费实例列表与 CU 信息，省去跨洋 /tun/m/ 串行等待，直奔实例遥测快照
        let cached_fresh = if !force_gdrive {
            app.last_assignments
                .l()
                .as_ref()
                .filter(|(_, ts)| ts.elapsed() < Duration::from_secs(8))
                .map(|(l, _)| l.clone())
        } else {
            None
        };

        if let Some(cached_list) = cached_fresh {
            ov.auth_ok = true;
            assignments = cached_list;
            ov.ccu = app.last_ccu.l().clone();
        } else {
            let (ccu, list) = tokio::join!(app.client.ccu_info(), app.client.list_assignments());
            match list {
                Ok(l) => {
                    ov.auth_ok = true;
                    *app.last_assignments.l() = Some((l.clone(), Instant::now()));
                    assignments = l;
                }
                Err(e) => {
                    let fallback = app
                        .last_assignments
                        .l()
                        .as_ref()
                        .filter(|(_, ts)| ts.elapsed() < Duration::from_secs(60))
                        .map(|(l, _)| l.clone());
                    if let Some(cached) = fallback {
                        ov.auth_ok = true;
                        assignments = cached;
                    } else {
                        ov.error = Some(format!("{e:#}"));
                    }
                }
            }
            match ccu {
                Ok(c) => {
                    *app.last_ccu.l() = Some(c.clone());
                    ov.ccu = Some(c);
                }
                Err(_) => {
                    ov.ccu = app.last_ccu.l().clone();
                }
            }
        }
    }
    ov.server_count = assignments.len();

    // 并发拉取每台在线虚拟机的遥测快照（Contents API，不占用 Kernel）
    // 采用 Stale-While-Revalidate：单次网络抖动失败时保留上一份有效遥测快照，避免界面闪断或误触发重连探针
    let fetched = join_all(assignments.iter().map(|a| fetch_telemetry(app, a))).await;
    {
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

    let ep_to_name: HashMap<String, String> = sessions
        .iter()
        .filter_map(|(n, v)| session_endpoint(v).map(|ep| (ep, n.clone())))
        .collect();
    let busy = app.busy.l().clone();
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;

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
            Some(n) => (n.clone(), true),
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

    let mut seen: Vec<String> = drafts.iter().map(|d| d.name.clone()).collect();

    // 接力换棒间隙（旧实例已销毁、新实例未分配）与正在创建中的虚拟机也保留卡片
    let relay_names: Vec<(String, Arc<RelayHandle>)> = app
        .relays
        .l()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let gaps: Vec<(String, Arc<RelayHandle>)> = relay_names
        .into_iter()
        .filter(|(n, h)| !seen.contains(n) && h.is_running())
        .collect();
    seen.extend(gaps.iter().map(|(n, _)| n.clone()));

    // 历史事件解析、工作区推断、近期会话与蓝绿状态读取均为同步磁盘 I/O，统一放入阻塞线程池批量执行
    let vm_jobs: Vec<(String, String, Option<Value>, Option<Value>)> = drafts
        .iter()
        .map(|d| (d.name.clone(), d.a.endpoint.clone(), d.sess.clone(), d.tel.clone()))
        .collect();
    let gap_jobs: Vec<String> = gaps.iter().map(|(n, _)| n.clone()).collect();
    let seen_bg = seen.clone();
    let (vm_extra, gap_extra, recent, bg) = tokio::task::spawn_blocking(move || {
        let vm_extra: Vec<(Vec<CliEventView>, WorkspaceInfo)> = vm_jobs
            .iter()
            .map(|(name, ep, sess, tel)| {
                let (cli_events, history_paths) = cli::read_session_events(name, 12);
                let workspace =
                    cli::resolve_workspace_for_vm(name, Some(ep), sess.as_ref(), tel.as_ref(), &history_paths);
                (cli_events, workspace)
            })
            .collect();
        let gap_extra: Vec<(Vec<CliEventView>, WorkspaceInfo)> = gap_jobs
            .iter()
            .map(|name| {
                let (cli_events, history_paths) = cli::read_session_events(name, 12);
                let workspace = cli::resolve_workspace_for_vm(name, None, None, None, &history_paths);
                (cli_events, workspace)
            })
            .collect();
        let recent = cli::read_recent_sessions(&seen_bg, 8);
        let bg = cli::read_llm_bluegreen_state();
        (vm_extra, gap_extra, recent, bg)
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

    let mut vm_extra = vm_extra.into_iter();
    for d in drafts {
        let (cli_events, workspace) = vm_extra.next().unwrap_or_else(|| fallback_extra(&d.name));
        let cur_busy = app.busy.l().get(&d.name).cloned();
        let sess = d.sess.as_ref();
        let a = &d.a;

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
        });
    }

    let mut gap_extra = gap_extra.into_iter();
    for (name, h) in gaps {
        let (cli_events, workspace) = gap_extra.next().unwrap_or_else(|| fallback_extra(&name));
        let snap = h.snapshot();
        let cfg = snap.config.clone();
        let cur_busy = app.busy.l().get(&name).cloned();
        ov.vms.push(VmView {
            name: name.clone(),
            workspace,
            accelerator: cfg.gpu_type.clone(),
            variant: "GPU".into(),
            high_ram: cfg.high_mem,
            managed: true,
            cli_events,
            relay: Some(snap),
            busy: cur_busy,
            ..Default::default()
        });
    }
    ov.vms.sort_by(|a, b| a.workspace.name.cmp(&b.workspace.name).then(a.name.cmp(&b.name)));
    ov.recent_sessions = recent;
    ov.llm_bluegreen = bg;

    *app.overview.write().await = ov.clone();
    ov
}

// ───────────────────────────── 彻底销毁（保留底层能力供 CLI 测试使用） ─────────────────────────────

fn cleanup_llm_bluegreen_after_terminate(terminated_name: &str, remaining_endpoints: &[String], steps: &mut Vec<String>) {
    let state_path = cli::llm_bluegreen_state_path();
    let Some(mut bg) = cli::read_llm_bluegreen_state() else {
        return;
    };
    let prefix = bg
        .get("session_prefix")
        .and_then(|x| x.as_str())
        .unwrap_or("llm")
        .to_string();
    let blue_name = format!("{prefix}-blue");
    let green_name = format!("{prefix}-green");
    if terminated_name != blue_name && terminated_name != green_name && !terminated_name.starts_with("llm-") {
        return;
    }

    let sessions = cli::read_sessions();
    let slot_still_on_server = |slot_vm: &str| -> bool {
        if slot_vm == terminated_name {
            return false;
        }
        sessions
            .get(slot_vm)
            .and_then(session_endpoint)
            .map(|ep| remaining_endpoints.contains(&ep))
            .unwrap_or(false)
    };

    let blue_alive = slot_still_on_server(&blue_name);
    let green_alive = slot_still_on_server(&green_name);

    if !blue_alive && !green_alive {
        let _ = std::fs::remove_file(&state_path);
        let pid_path = cli::colab_config_dir().join("llm_local_gateway.pid");
        if let Ok(raw_pid) = std::fs::read_to_string(&pid_path) {
            if let Ok(pid) = raw_pid.trim().parse::<u32>() {
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    let _ = std::process::Command::new("taskkill")
                        .args(["/F", "/T", "/PID", &pid.to_string()])
                        .stdout(std::process::Stdio::null())
                        .stderr(std::process::Stdio::null())
                        .creation_flags(0x0800_0000)
                        .status();
                }
            }
            let _ = std::fs::remove_file(&pid_path);
        }
        steps.push("已同步清理本机 LLM 蓝绿状态与本地网关隧道进程".into());
    } else {
        let surviving_slot = if blue_alive { "blue" } else { "green" };
        if let Some(obj) = bg.as_object_mut() {
            obj.insert("active_slot".into(), Value::String(surviving_slot.into()));
            obj.insert("standby_slot".into(), Value::Null);
            obj.insert("phase".into(), Value::String("serving".into()));
        }
        if let Ok(raw) = serde_json::to_string_pretty(&bg) {
            let _ = std::fs::write(&state_path, raw);
        }
        steps.push(format!(
            "已更新蓝绿状态：当前由剩余在线槽位 [{}] 继续提供服务",
            surviving_slot.to_uppercase()
        ));
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

    let mut stopped_local_relay = false;
    if stop_relay {
        if let Some(h) = app.relay(name) {
            if h.is_running() {
                h.stop().await;
                stopped_local_relay = true;
                r.steps.push("已停止本机接力监工任务并结束 colab exec 进程树".into());
            }
        }
    }

    let sessions = cli::read_sessions();
    let managed = sessions.contains_key(name);
    let mut endpoint = endpoint_hint
        .filter(|e| !e.is_empty())
        .or_else(|| sessions.get(name).and_then(session_endpoint));
    if endpoint.is_none() {
        if let Ok(list) = app.client.list_assignments().await {
            endpoint = list
                .iter()
                .find(|a| orphan_name(&a.endpoint) == name)
                .map(|a| a.endpoint.clone());
        }
    }
    r.endpoint = endpoint.clone();

    let Some(ep) = endpoint else {
        if stopped_local_relay {
            r.verified_unassigned = true;
            r.steps.push("当前处于换棒间隙（云端无计费实例），本机接力任务已终止".into());
        } else if let Ok(list) = app.client.list_assignments().await {
            r.remaining_endpoints = list.iter().map(|a| a.endpoint.clone()).collect();
            r.verified_unassigned = true;
            r.steps.push("服务端计费列表中已不存在该实例（无需重复释放）".into());
            cleanup_llm_bluegreen_after_terminate(name, &r.remaining_endpoints, &mut r.steps);
        } else {
            r.error = Some("无法确定该虚拟机的 endpoint，服务端计费列表中也未找到".into());
        }
        if let Ok(c) = app.client.ccu_info().await {
            r.ccu = Some(c);
        }
        return r;
    };

    if managed && cli::colab_bin().exists() {
        match cli::run_simple(&cli::colab_bin(), &s(&["stop", "-s", name]), Duration::from_secs(120)).await {
            Ok(o) => {
                let last = o.output.trim().lines().last().unwrap_or("").to_string();
                r.steps.push(format!("colab stop -s {name}：{}", if last.is_empty() { "已执行".into() } else { last }));
            }
            Err(e) => r.steps.push(format!("colab stop 执行失败：{e}")),
        }
    }

    for attempt in 1..=6 {
        match app.client.list_assignments().await {
            Ok(list) => {
                r.remaining_endpoints = list.iter().map(|a| a.endpoint.clone()).collect();
                if !r.remaining_endpoints.contains(&ep) {
                    r.verified_unassigned = true;
                    r.steps.push(format!(
                        "服务端核验通过：/tun/m/assignments 中已不存在 {ep}（剩余计费实例 {} 台）",
                        r.remaining_endpoints.len()
                    ));
                    break;
                }
                match app.client.unassign(&ep).await {
                    Ok(_) => r.steps.push(format!("第 {attempt} 次直接调用 /tun/m/unassign/{ep} 成功")),
                    Err(e) => r.steps.push(format!("第 {attempt} 次 unassign 返回：{e}")),
                }
            }
            Err(e) => r.steps.push(format!("第 {attempt} 次核验请求失败：{e}")),
        }
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }

    if r.verified_unassigned {
        app.telemetry.l().remove(&ep);
        if let Some((list, _)) = app.last_assignments.l().as_mut() {
            list.retain(|a| a.endpoint != ep);
        }
        if cli::read_sessions().contains_key(name) && cli::colab_bin().exists() {
            let _ = cli::run_simple(&cli::colab_bin(), &s(&["sessions"]), Duration::from_secs(60)).await;
            r.steps.push("已同步清理 colab-cli 本地会话表".into());
        }
        cleanup_llm_bluegreen_after_terminate(name, &r.remaining_endpoints, &mut r.steps);
    } else {
        r.error = Some("服务端计费列表中仍存在该实例，未执行本地移除（避免假性删除），请重试".into());
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

#[derive(Deserialize)]
struct CreateVmRequest {
    name: Option<String>,
    accelerator: String,
    high_mem: bool,
    auto_relay: bool,
    relay_config: Option<RelayConfig>,
}

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 40
        && n.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

#[tauri::command]
async fn create_vm(handle: AppHandle, state: AppStateRef<'_>, req: CreateVmRequest) -> Result<String, String> {
    let app = state.inner().clone();
    let name = req
        .name
        .map(|n| n.trim().to_string())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| format!("vm-{}", &uuid::Uuid::new_v4().simple().to_string()[..6]));
    if !valid_name(&name) {
        return Err("名称只能包含字母、数字、- 和 _（不超过 40 个字符）".into());
    }
    if cli::read_sessions().contains_key(&name) || app.busy.l().contains_key(&name) {
        return Err(format!("名称 [{name}] 已被占用"));
    }
    if !cli::colab_bin().exists() {
        return Err("未找到 colab-cli，可执行文件应位于 ~/.local/bin/colab.exe".into());
    }

    if !app.try_claim_busy(&name, "creating") {
        return Err(format!("名称 [{name}] 已被占用"));
    }
    spawn_refresh(&app, &handle);

    let res = relay::colab_new(&name, &req.accelerator, req.high_mem, &|_l: &str| {}).await;
    if let Err(e) = res {
        app.set_busy(&name, None);
        spawn_refresh(&app, &handle);
        return Err(e.to_string());
    }

    app.set_busy(&name, Some("probing"));
    spawn_refresh(&app, &handle);
    let _ = deploy_probe_inner(&app, &name).await;
    app.set_busy(&name, None);

    if req.auto_relay {
        let mut cfg = req.relay_config.unwrap_or_default();
        cfg.gpu_type = if req.accelerator == "CPU" { "T4".into() } else { req.accelerator.clone() };
        cfg.high_mem = req.high_mem;
        start_relay_inner(&app, &name, cfg);
    }
    spawn_refresh(&app, &handle);
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
    if app.busy.l().get(&name).map(|b| b == "deleting").unwrap_or(false) {
        return Err("该虚拟机正在销毁中".into());
    }
    app.set_busy(&name, Some("deleting"));
    spawn_refresh(&app, &handle);
    let r = hard_terminate(&app, &name, endpoint, true).await;
    app.release_busy(&name, "deleting");
    if r.verified_unassigned {
        app.relays.l().remove(&name);
    }
    spawn_refresh(&app, &handle);
    Ok(r)
}

#[tauri::command]
async fn deploy_probe(handle: AppHandle, state: AppStateRef<'_>, name: String) -> Result<(), String> {
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
    res.map_err(|e| e.to_string())
}

fn start_relay_inner(app: &Arc<AppState>, name: &str, cfg: RelayConfig) -> RelayStatus {
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
    h.task_alive.store(true, std::sync::atomic::Ordering::SeqCst);
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
    config: RelayConfig,
) -> Result<RelayStatus, String> {
    let app = state.inner().clone();
    if !cli::rclone_conf_path().exists() {
        return Err("未检测到 rclone.conf，请先完成 Google Drive 授权".into());
    }
    let st = start_relay_inner(&app, &name, config);
    spawn_refresh(&app, &handle);
    Ok(st)
}

#[tauri::command]
async fn stop_relay(
    handle: AppHandle,
    state: AppStateRef<'_>,
    name: String,
    destroy_vm: bool,
) -> Result<Option<TerminateReceipt>, String> {
    let app = state.inner().clone();
    let mut receipt = None;
    if let Some(h) = app.relay(&name) {
        h.stop().await;
        h.log("⏹ 已关闭接力训练");
        if destroy_vm {
            h.log("🧹 正在销毁实例并核验停止计费…");
            let r = hard_terminate(&app, &name, None, false).await;
            h.log(if r.verified_unassigned { "✅ 实例已彻底销毁，计费已停止" } else { "⚠️ 销毁未通过核验，请重试删除" });
            receipt = Some(r);
        } else if cli::read_sessions().contains_key(&name) {
            let _ = cli::run_simple(
                &cli::colab_bin(),
                &s(&["restart-kernel", "-s", &name]),
                Duration::from_secs(90),
            )
            .await;
            h.log("🔁 已重启远程 Kernel，云端训练进程已中断（实例保留，仍在计费）");
        }
    }
    spawn_refresh(&app, &handle);
    Ok(receipt)
}

#[derive(Serialize)]
struct WorkspaceFiles {
    workspace: String,
    train_py: String,
    requirements_txt: String,
}

#[tauri::command]
fn read_workspace_files(state: AppStateRef<'_>) -> WorkspaceFiles {
    let ws = &state.workspace;
    WorkspaceFiles {
        workspace: ws.display().to_string(),
        train_py: std::fs::read_to_string(ws.join("train.py")).unwrap_or_default(),
        requirements_txt: std::fs::read_to_string(ws.join("requirements.txt")).unwrap_or_default(),
    }
}

#[tauri::command]
fn save_workspace_files(state: AppStateRef<'_>, train_py: String, requirements_txt: String) -> Result<(), String> {
    let ws = &state.workspace;
    std::fs::write(ws.join("train.py"), train_py).map_err(|e| e.to_string())?;
    std::fs::write(ws.join("requirements.txt"), requirements_txt).map_err(|e| e.to_string())?;
    Ok(())
}

// ───────────────────────────── 大语言模型 (LLM) 流式对话网关与云端桥接 ─────────────────────────────

const REMOTE_LLM_CHAT_STREAM_PY: &str = r#"
import base64, json, sys

url = sys.argv[1]
token = sys.argv[2]
pref_port = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 8000
kernel_id = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] else None
session_id = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None

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
        else:
            err_txt = r.read().decode("utf-8", "ignore")[:280]
            last_err = f"端口 {{port}} 返回 HTTP {{r.status}}: {{err_txt}}"
            c.close()
    except Exception as e:
        last_err = f"端口 {{port}} 连接异常: {{e}}"

if resp is None:
    print("PILOT_ERR:" + json.dumps({{"error": last_err}}, ensure_ascii=False), flush=True)
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
                        delta = (obj.get("choices") or [{{}}])[0].get("delta") or {{}}
                        piece = delta.get("content") or delta.get("reasoning_content") or ""
                        if piece:
                            print("PILOT_DELTA:" + json.dumps({{"d": piece}}, ensure_ascii=False), flush=True)
                    except Exception:
                        pass
        else:
            try:
                obj = json.loads(raw_resp)
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
    let non_cjk = if ascii_chars > 0 { (ascii_chars / 4).max(1) } else { 0 };
    (cjk + non_cjk).max(1)
}

/// 若本机存在蓝绿网关脚本且尚未启动，静默在后台拉起以预热 SSH 隧道
/// （同步阻塞实现，须通过 `spawn_local_gateway_async` 在阻塞线程池中调用；15 秒内最多拉起一次）
fn maybe_spawn_local_gateway(workspace: &Path) {
    static LAST_SPAWN: Mutex<Option<Instant>> = Mutex::new(None);
    if LAST_SPAWN
        .l()
        .map(|t| t.elapsed() < Duration::from_secs(15))
        .unwrap_or(false)
    {
        return;
    }
    if let Ok(addr) = "127.0.0.1:18080".parse() {
        if std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok() {
            return;
        }
    }
    let gw_script = workspace
        .join(".agents")
        .join("skills")
        .join("colab-llm-bluegreen")
        .join("scripts")
        .join("llm_local_gateway.py");
    if !gw_script.exists() {
        return;
    }
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
    if cmd.spawn().is_ok() {
        *LAST_SPAWN.l() = Some(Instant::now());
    }
}

async fn spawn_local_gateway_async(workspace: PathBuf) {
    let _ = tokio::task::spawn_blocking(move || maybe_spawn_local_gateway(&workspace)).await;
}

struct ResolvedLlmTarget {
    vm_name: String,
    slot: Option<String>,
    model: String,
    remote_port: u16,
    local_tunnel_port: Option<u16>,
    kernel_id: Option<String>,
    session_id: Option<String>,
}

async fn resolve_llm_target(app: &Arc<AppState>, req: &LlmChatRequest) -> anyhow::Result<ResolvedLlmTarget> {
    let mut ov = app.overview.read().await.clone();
    if ov.vms.is_empty() {
        ov = refresh_overview(app, false).await;
    }
    let bg = cli::read_llm_bluegreen_state();
    let prefix = bg
        .as_ref()
        .and_then(|v| v.get("session_prefix"))
        .and_then(|x| x.as_str())
        .unwrap_or("llm");
    let bg_active_slot = bg
        .as_ref()
        .and_then(|v| v.get("active_slot"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_lowercase());
    let bg_model = bg
        .as_ref()
        .and_then(|v| v.get("model_name"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_string());

    let wanted = req.target_vm.trim();
    let vm_opt = if wanted.is_empty() || wanted.eq_ignore_ascii_case("auto") {
        // 1. 优先匹配蓝绿 ACTIVE 槽位在线实例
        let by_bg = bg_active_slot.as_ref().and_then(|slot| {
            let expected = format!("{prefix}-{slot}");
            ov.vms.iter().find(|v| {
                v.on_server
                    && (v.name == expected
                        || v.name.ends_with(&format!("-{slot}"))
                        || v.telemetry
                            .as_ref()
                            .and_then(|t| t["llm"]["slot"].as_str())
                            .map(|s| s.eq_ignore_ascii_case(slot))
                            .unwrap_or(false))
            })
        });
        // 2. 匹配任意 telemetry.llm.online == true 且 role == active 的实例
        let by_active_llm = ov.vms.iter().find(|v| {
            v.on_server
                && v.telemetry
                    .as_ref()
                    .map(|t| {
                        t["llm"]["online"].as_bool() == Some(true)
                            && t["llm"]["role"].as_str().unwrap_or("active").eq_ignore_ascii_case("active")
                    })
                    .unwrap_or(false)
        });
        // 3. 匹配任意包含 LLM 遥测或 LLM 进程的在线实例
        let by_any_llm = ov.vms.iter().find(|v| {
            v.on_server
                && v.telemetry
                    .as_ref()
                    .map(|t| t["llm"].is_object() || t["threads"]["llm_process"].is_object())
                    .unwrap_or(false)
        });
        by_bg.or(by_active_llm).or(by_any_llm).or_else(|| ov.vms.iter().find(|v| v.on_server))
    } else {
        ov.vms.iter().find(|v| v.name == wanted)
    };

    let Some(vm) = vm_opt else {
        anyhow::bail!("当前没有在线的大模型 Colab 实例，请先通过 colab-llm-bluegreen 或 colab new 部署实例");
    };

    let llm_tel = vm.telemetry.as_ref().and_then(|t| t.get("llm"));
    let slot = llm_tel
        .and_then(|l| l.get("slot"))
        .and_then(|x| x.as_str())
        .map(|s| s.to_lowercase())
        .or_else(|| {
            if vm.name.ends_with("-blue") {
                Some("blue".into())
            } else if vm.name.ends_with("-green") {
                Some("green".into())
            } else {
                None
            }
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
        .or(bg_model)
        .unwrap_or_else(|| "default".into());

    let remote_port = llm_tel
        .and_then(|l| l.get("port"))
        .and_then(|x| x.as_u64())
        .map(|p| p as u16)
        .unwrap_or(8000);

    let local_tunnel_port = match slot.as_deref() {
        Some("blue") => Some(18081),
        Some("green") => Some(18082),
        _ => None,
    };

    Ok(ResolvedLlmTarget {
        vm_name: vm.name.clone(),
        slot,
        model,
        remote_port,
        local_tunnel_port,
        kernel_id: vm.kernel_id.clone(),
        session_id: vm.session_id.clone(),
    })
}

async fn try_stream_via_local_tunnel(
    handle: &AppHandle,
    req: &LlmChatRequest,
    target: &ResolvedLlmTarget,
    local_port: u16,
    body_val: &Value,
    emitted: &mut bool,
) -> anyhow::Result<LlmChatResponse> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .connect_timeout(Duration::from_millis(650))
        .timeout(Duration::from_secs(300))
        .build()?;

    let health_url = format!("http://127.0.0.1:{local_port}/pilot/health");
    let h_resp = client
        .get(&health_url)
        .timeout(Duration::from_millis(650))
        .send()
        .await?;
    if !h_resp.status().is_success() {
        anyhow::bail!("本地 SSH 隧道端口 {local_port} 尚未就绪");
    }

    let chat_url = format!("http://127.0.0.1:{local_port}/v1/chat/completions");
    let mut resp = client
        .post(&chat_url)
        .header("Content-Type", "application/json")
        .json(body_val)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let err_txt = resp.text().await.unwrap_or_default();
        anyhow::bail!("本地隧道 HTTP {status}: {err_txt}");
    }

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut reply = String::new();
    let mut completion_tokens = 0u32;
    // 以字节缓冲按 b'\n' 切分完整行后再解码，绝不对半截 UTF-8 多字节字符（中文等）做有损解码
    let mut buf: Vec<u8> = Vec::new();

    loop {
        let chunk = match resp.chunk().await {
            Ok(Some(c)) => c,
            Ok(None) => break,
            // 已输出部分内容后隧道中断：保留已生成的部分回复，不再报错回退（避免重复回答）
            Err(_) if !reply.is_empty() => break,
            Err(e) => return Err(e.into()),
        };
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
            let line_owned = String::from_utf8_lossy(&line_bytes);
            let line = line_owned.trim();
            let Some(data_str) = line.strip_prefix("data:") else {
                continue;
            };
            let data_str = data_str.trim();
            if data_str.is_empty() || data_str == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data_str) {
                let piece = v["choices"]
                    .get(0)
                    .and_then(|c| {
                        c["delta"]["content"]
                            .as_str()
                            .or_else(|| c["delta"]["reasoning_content"].as_str())
                            .or_else(|| c["text"].as_str())
                    })
                    .unwrap_or("");
                if !piece.is_empty() {
                    let first = *t_first.get_or_insert_with(Instant::now);
                    let ttft_ms = first.duration_since(t_start).as_secs_f64() * 1000.0;
                    completion_tokens += estimate_tokens(piece);
                    reply.push_str(piece);
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
                            delta: Some(piece.to_string()),
                            ttft_ms: Some((ttft_ms * 10.0).round() / 10.0),
                            instant_tps: Some((inst_tps * 10.0).round() / 10.0),
                            avg_tps: Some((inst_tps * 10.0).round() / 10.0),
                            completion_tokens: Some(completion_tokens),
                            elapsed_ms: Some(t_start.elapsed().as_secs_f64() * 1000.0),
                            transport: Some(format!("SSH 隧道 (:{local_port})")),
                            error: None,
                        },
                    );
                }
            }
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
        transport: format!("SSH 隧道 (127.0.0.1:{local_port})"),
    })
}

async fn stream_via_colab_runtime(
    handle: &AppHandle,
    a: &Assignment,
    req: &LlmChatRequest,
    target: &ResolvedLlmTarget,
    body_val: &Value,
) -> anyhow::Result<LlmChatResponse> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::process::Command;

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
        target.session_id.as_deref().unwrap_or(""),
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

    let mut child = cmd.spawn().context("启动 ColabRuntime 流式桥接进程失败")?;
    if let Some(mut stdin) = child.stdin.take() {
        let payload_bytes = serde_json::to_vec(body_val)?;
        let _ = stdin.write_all(&payload_bytes).await;
        let _ = stdin.shutdown().await;
    }

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

    let stdout = child.stdout.take().ok_or_else(|| anyhow::anyhow!("无法读取流式桥接输出"))?;
    let mut lines = BufReader::new(stdout).lines();

    let t_start = Instant::now();
    let mut t_first: Option<Instant> = None;
    let mut reply = String::new();
    let mut completion_tokens = 0u32;
    let mut remote_err: Option<String> = None;

    // 整体 330 秒截止（云端推理 300 秒超时 + 桥接握手余量），超时立即结束桥接进程树
    let deadline = tokio::time::sleep(Duration::from_secs(330));
    tokio::pin!(deadline);
    let mut timed_out = false;

    loop {
        let line = tokio::select! {
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
                                transport: Some(format!("Colab 直连 (:{} 边车)", target.remote_port)),
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
        }
    }

    if timed_out {
        cli::kill_tree(&mut child).await;
        if let Some(t) = stderr_task {
            t.abort();
        }
        anyhow::bail!("Colab 直连流式超时 (330s)");
    }

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
    }
    if reply.is_empty() && !status.map(|s| s.success()).unwrap_or(false) && !stderr_tail.is_empty() {
        let last = stderr_tail
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim()
            .to_string();
        anyhow::bail!("Colab 直连流式桥接异常退出：{last}");
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

#[tauri::command]
async fn stream_llm_chat(
    handle: AppHandle,
    state: AppStateRef<'_>,
    req: LlmChatRequest,
) -> Result<LlmChatResponse, String> {
    let app = state.inner().clone();
    let target = resolve_llm_target(&app, &req).await.map_err(|e| e.to_string())?;

    let a = app
        .find_assignment_by_name(&target.vm_name)
        .await
        .ok_or_else(|| format!("服务端未找到在线实例 [{}]", target.vm_name))?;

    if target.local_tunnel_port.is_some() {
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

    // 优先使用已就绪的本机蓝绿 SSH 隧道（毫秒级直连），若隧道未就绪则无缝回退至 ColabRuntime 直连流式通道
    // 若隧道已输出部分内容后才中断，则不再回退（避免同一问题被回答两遍），直接返回部分结果或错误
    let mut res = if let Some(lport) = target.local_tunnel_port {
        let mut tunnel_emitted = false;
        match try_stream_via_local_tunnel(&handle, &req, &target, lport, &body_val, &mut tunnel_emitted).await {
            Ok(r) if !r.reply.is_empty() || tunnel_emitted => Ok(r),
            Err(e) if tunnel_emitted => Err(e),
            _ => {
                // 回退前重新广播 start，前端据此重置已显示内容
                emit_chat_start(&handle, &req, &target);
                stream_via_colab_runtime(&handle, &a, &req, &target, &body_val).await
            }
        }
    } else {
        stream_via_colab_runtime(&handle, &a, &req, &target, &body_val).await
    };

    // 对话完成后立即拉取一次该实例的最新遥测快照（同步云端边车实测 tok/s、TTFT 与曲线）并广播给仪表盘
    if let Ok(ref mut ok_resp) = res {
        let entry = fetch_telemetry(&app, &a).await;
        if let Some(ref tel_val) = entry.data {
            if let Some(sidecar_tps) = tel_val["llm"]["last_request_tps"]
                .as_f64()
                .filter(|&x| x > 0.0)
                .or_else(|| tel_val["llm"]["instant_tps"].as_f64().filter(|&x| x > 0.0))
            {
                ok_resp.avg_tps = (sidecar_tps * 10.0).round() / 10.0;
            }
            if let Some(sidecar_ttft) = tel_val["llm"]["ttft_ms"].as_f64().filter(|&x| x > 0.0) {
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
        let _ = std::fs::write(dir.join("requirements.txt"), "# 每行一个云端需额外安装的 Python 包\n");
    }
    dir
}

/// 无界面诊断模式：`colab-pilot.exe --print-overview` 输出一次后端总览 JSON 后退出
fn print_overview_headless() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default();
        let state = Arc::new(AppState {
            client: ColabClient::new(),
            workspace,
            overview: tokio::sync::RwLock::new(Overview::default()),
            telemetry: Mutex::new(HashMap::new()),
            last_assignments: Mutex::new(None),
            last_ccu: Mutex::new(None),
            relays: Mutex::new(HashMap::new()),
            busy: Mutex::new(HashMap::new()),
            probe_cooldown: Mutex::new(HashMap::new()),
            gdrive: tokio::sync::Mutex::new((None, None)),
            refresh_lock: tokio::sync::Mutex::new(()),
        });
        let ov = refresh_overview(&state, true).await;
        println!("{}", serde_json::to_string_pretty(&ov).unwrap_or_default());
    });
}

#[cfg(debug_assertions)]
fn selftest_lifecycle() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let state = Arc::new(AppState {
            client: ColabClient::new(),
            workspace: Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
            overview: tokio::sync::RwLock::new(Overview::default()),
            telemetry: Mutex::new(HashMap::new()),
            last_assignments: Mutex::new(None),
            last_ccu: Mutex::new(None),
            relays: Mutex::new(HashMap::new()),
            busy: Mutex::new(HashMap::new()),
            probe_cooldown: Mutex::new(HashMap::new()),
            gdrive: tokio::sync::Mutex::new((None, None)),
            refresh_lock: tokio::sync::Mutex::new(()),
        });
        let name = "selftest-cpu";
        println!("[1] creating {name} …");
        if let Err(e) = relay::colab_new(name, "CPU", false, &|l: &str| println!("    | {l}")).await {
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
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
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
// 修复“双击 exe 闪退”：此前残留的无窗口僵尸实例会占用同一 WebView2 用户数据目录，
// 导致新实例 WebView2 环境创建失败而立即退出。现在通过本机回环端口实现单实例：
// - 若已有健康实例：唤醒并聚焦其窗口，新进程直接退出；
// - 若已有实例但没有可见窗口（僵尸）：强制结束僵尸进程树后由新进程接管启动。
// - 若已有实例窗口位于非交互桌面（如被自动化工具/沙箱从隔离桌面拉起，桌面回收后窗口不可见但句柄仍在），
//   同样视为僵尸并由交互桌面 (WinSta0\Default) 上的新实例接管。

const SINGLE_INSTANCE_PORT: u16 = 18079;
const INTERACTIVE_DESKTOP: &str = "WinSta0\\Default";

/// 当前线程所在的 “窗口站\桌面” 名称（例如用户交互桌面 `WinSta0\Default`）
#[cfg(windows)]
fn current_desktop_name() -> String {
    #[link(name = "user32")]
    extern "system" {
        fn GetThreadDesktop(thread_id: u32) -> isize;
        fn GetProcessWindowStation() -> isize;
        fn GetUserObjectInformationW(h: isize, index: i32, pv: *mut u16, len: u32, needed: *mut u32) -> i32;
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
        if GetUserObjectInformationW(h, UOI_NAME, buf.as_mut_ptr(), (buf.len() * 2) as u32, &mut need) == 0 {
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
    Primary(std::net::TcpListener),
    AlreadyRunning,
    NoLock,
}

fn kill_process_tree(pid: u32) {
    let mut cmd = std::process::Command::new("taskkill");
    cmd.args(["/PID", &pid.to_string(), "/T", "/F"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let _ = cmd.status();
}

fn acquire_single_instance() -> InstanceRole {
    use std::io::{BufRead, BufReader, Write};
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], SINGLE_INSTANCE_PORT));
    let my_desktop = current_desktop_name();
    for _ in 0..3 {
        if let Ok(l) = std::net::TcpListener::bind(addr) {
            return InstanceRole::Primary(l);
        }
        let Ok(mut s) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(800)) else {
            std::thread::sleep(Duration::from_millis(400));
            continue;
        };
        let _ = s.set_read_timeout(Some(Duration::from_millis(2500)));
        let _ = s.write_all(format!("SHOW {my_desktop}\n").as_bytes());
        let mut line = String::new();
        let _ = BufReader::new(&s).read_line(&mut line);
        let line = line.trim();
        if line.starts_with("OK") {
            return InstanceRole::AlreadyRunning;
        }
        if let Some(pid) = line.strip_prefix("NOWIN ").and_then(|p| p.trim().parse::<u32>().ok()) {
            if pid != std::process::id() {
                append_crash_log(&format!(
                    "[{}] 检测到无窗口僵尸实例 pid={pid}，已强制结束并接管启动\n",
                    chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
                ));
                kill_process_tree(pid);
                std::thread::sleep(Duration::from_millis(900));
            }
            continue;
        }
        // 端口被非 Colab Pilot 程序占用或对方无响应：放弃单实例锁，照常启动
        return InstanceRole::NoLock;
    }
    InstanceRole::NoLock
}

fn serve_single_instance(listener: std::net::TcpListener, handle: AppHandle) {
    use std::io::{BufRead, BufReader, Write};
    // 在主线程（窗口所在线程）记录本实例所处桌面
    let my_desktop = current_desktop_name();
    let i_am_interactive = my_desktop.eq_ignore_ascii_case(INTERACTIVE_DESKTOP);
    std::thread::Builder::new()
        .name("single-instance".into())
        .spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let _ = s.set_read_timeout(Some(Duration::from_millis(1500)));
                let mut line = String::new();
                let _ = BufReader::new(&s).read_line(&mut line);
                let line = line.trim();
                let Some(rest) = line
                    .get(..4)
                    .filter(|p| p.eq_ignore_ascii_case("SHOW"))
                    .map(|_| line[4..].trim())
                else {
                    continue;
                };
                let requester_desktop = rest.to_string();
                let same_desktop =
                    requester_desktop.is_empty() || requester_desktop.eq_ignore_ascii_case(&my_desktop);
                let reply = if !same_desktop && !i_am_interactive {
                    // 本实例位于已失效/隔离的非交互桌面：让新实例接管
                    format!("NOWIN {}\n", std::process::id())
                } else if !same_desktop {
                    // 本实例在用户交互桌面上健康运行，请求方来自隔离桌面：保持现状，让请求方退出
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
                        format!("NOWIN {}\n", std::process::id())
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
        let _ = std::fs::write(path, json);
    }
}

fn restore_window_state(window: &tauri::WebviewWindow) -> Arc<Mutex<WindowStateTracker>> {
    let (max_w, max_h) = window
        .current_monitor()
        .ok()
        .flatten()
        .map(|m| {
            let s = m.size().to_logical::<f64>(m.scale_factor());
            (
                (s.width - 24.0).max(MIN_WINDOW_WIDTH),
                (s.height - 48.0).max(MIN_WINDOW_HEIGHT),
            )
        })
        .unwrap_or((1920.0, 1080.0));

    let prefs = if let Some(mut saved) = load_window_state() {
        saved.width = saved.width.clamp(MIN_WINDOW_WIDTH, max_w).round();
        saved.height = saved.height.clamp(MIN_WINDOW_HEIGHT, max_h).round();
        let _ = window.set_size(tauri::Size::Logical(tauri::LogicalSize::new(
            saved.width,
            saved.height,
        )));

        let pos_valid = match (saved.x, saved.y) {
            (Some(px), Some(py)) => window
                .available_monitors()
                .unwrap_or_default()
                .iter()
                .any(|m| {
                    let mp = m.position();
                    let ms = m.size();
                    let check_x = px.saturating_add(80);
                    let check_y = py.saturating_add(20);
                    check_x >= mp.x
                        && check_x < mp.x.saturating_add(ms.width as i32)
                        && check_y >= mp.y
                        && check_y < mp.y.saturating_add(ms.height as i32)
                }),
            _ => false,
        };
        if pos_valid {
            if let (Some(px), Some(py)) = (saved.x, saved.y) {
                let _ = window.set_position(tauri::Position::Physical(
                    tauri::PhysicalPosition::new(px, py),
                ));
            }
        } else {
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
                if logical.width >= MIN_WINDOW_WIDTH - 4.0 && logical.height >= MIN_WINDOW_HEIGHT - 4.0 {
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
        } else {
            (None, true)
        }
    };

    if let Some(p) = immediate_save {
        save_window_state(&p);
    } else if schedule_delayed {
        let tracker_bg = tracker.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(Duration::from_millis(320)).await;
            let to_save = {
                let mut g = tracker_bg.l();
                if g.dirty {
                    g.dirty = false;
                    g.last_saved = Instant::now();
                    Some(g.prefs.clone())
                } else {
                    None
                }
            };
            if let Some(p) = to_save {
                save_window_state(&p);
            }
        });
    }
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
        InstanceRole::Primary(l) => Some(l),
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
            if let Some(l) = instance_listener.lock().unwrap_or_else(|e| e.into_inner()).take() {
                serve_single_instance(l, app.handle().clone());
            }
            let state = Arc::new(AppState {
                client: ColabClient::new(),
                workspace: resolve_workspace(app),
                overview: tokio::sync::RwLock::new(Overview::default()),
                telemetry: Mutex::new(HashMap::new()),
                last_assignments: Mutex::new(None),
                last_ccu: Mutex::new(None),
                relays: Mutex::new(HashMap::new()),
                busy: Mutex::new(HashMap::new()),
                probe_cooldown: Mutex::new(HashMap::new()),
                gdrive: tokio::sync::Mutex::new((None, None)),
                refresh_lock: tokio::sync::Mutex::new(()),
            });
            app.manage(state.clone());

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
                        ov.error.is_some()
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_overview,
            refresh_now,
            create_vm,
            delete_vm,
            deploy_probe,
            start_relay,
            stop_relay,
            read_workspace_files,
            save_workspace_files,
            stream_llm_chat
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
                        tauri::WindowEvent::CloseRequested { .. } => {
                            if let (Some(w), Some(tr)) = (
                                handle.get_webview_window("main"),
                                win_tracker_slot.l().clone(),
                            ) {
                                record_window_state(&w, &tr, true);
                            }
                        }
                        // 主窗口被销毁（含桌面会话被回收等异常情况）时立即刷盘并退出进程，绝不残留无窗口僵尸实例
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
