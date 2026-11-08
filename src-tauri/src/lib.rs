//! Colab Pilot — Tauri 2 Rust 后端（工作区与 Colab 虚拟机实时监控仪表盘）
//!
//! - 后台轮询：Colab 服务端计费实例、账户 CU、云端遥测快照、colab-cli 会话与历史、Google Drive 状态
//! - 自动探针注入：自动感知 Antigravity 通过 colab-cli 创建的实例，通过 Contents API + /colab/tty 静默注入遥测守护进程（不阻塞 Kernel，不污染 CLI 会话状态）
//! - 工作区映射：自动将每台 Colab 虚拟机关联到对应的本地工作区（Workspace），汇总资源分配与运行状态

mod cli;
mod colab;
mod relay;

use cli::{s, CliEventView, RecentSessionView, WorkspaceInfo};
use colab::{Assignment, Ccu, ColabClient};
use futures::future::join_all;
use relay::{RelayConfig, RelayHandle, RelayStatus};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Manager, State};

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

struct TelemetryEntry {
    data: Option<Value>,
    error: Option<String>,
}

pub struct AppState {
    pub client: ColabClient,
    pub workspace: PathBuf,
    overview: tokio::sync::RwLock<Overview>,
    telemetry: Mutex<HashMap<String, TelemetryEntry>>,
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
        self.relays.lock().unwrap().get(name).cloned()
    }

    fn set_busy(&self, name: &str, what: Option<&str>) {
        let mut b = self.busy.lock().unwrap();
        match what {
            Some(w) => {
                b.insert(name.into(), w.into());
            }
            None => {
                b.remove(name);
            }
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

async fn fetch_telemetry(app: &AppState, a: &Assignment) -> TelemetryEntry {
    match app
        .client
        .contents_read_text(a, "content/.colab_pilot_telemetry.json")
        .await
    {
        Ok(Some(text)) => match serde_json::from_str::<Value>(&text) {
            Ok(v) => TelemetryEntry { data: Some(v), error: None },
            Err(e) => TelemetryEntry { data: None, error: Some(format!("遥测解析失败: {e}")) },
        },
        Ok(None) => TelemetryEntry { data: None, error: Some("正在自动注入遥测探针…".into()) },
        Err(e) => TelemetryEntry { data: None, error: Some(e.to_string()) },
    }
}

pub async fn refresh_overview(app: &Arc<AppState>, force_gdrive: bool) -> Overview {
    let _g = app.refresh_lock.lock().await;
    let env = app.env_status();
    let mut ov = Overview {
        timestamp: chrono::Utc::now().timestamp(),
        env: env.clone(),
        ..Default::default()
    };

    // Google Drive 状态每 30 秒后台刷新一次（不阻塞 Colab 计费与遥测轮询）
    {
        let mut gd = app.gdrive.lock().await;
        let stale = gd.1.map(|t| t.elapsed() > Duration::from_secs(30)).unwrap_or(true);
        if (stale || force_gdrive) && env.rclone_ok && env.rclone_conf_ok {
            gd.1 = Some(Instant::now());
            if force_gdrive && gd.0.is_none() {
                gd.0 = Some(fetch_gdrive_status(DEFAULT_GDRIVE_DIR).await);
            } else {
                let app_bg = app.clone();
                tokio::spawn(async move {
                    let status = fetch_gdrive_status(DEFAULT_GDRIVE_DIR).await;
                    let mut g = app_bg.gdrive.lock().await;
                    g.0 = Some(status);
                });
            }
        }
        ov.gdrive = gd.0.clone();
    }

    let sessions = cli::read_sessions();
    let mut assignments: Vec<Assignment> = vec![];

    if !env.token_ok {
        ov.error = Some("未检测到 Colab 登录凭证，请在终端运行 `colab status` 完成 Google 授权".into());
    } else {
        let (ccu, list) = tokio::join!(app.client.ccu_info(), app.client.list_assignments());
        match list {
            Ok(l) => {
                ov.auth_ok = true;
                assignments = l;
            }
            Err(e) => ov.error = Some(format!("{e:#}")),
        }
        if let Ok(c) = ccu {
            ov.ccu = Some(c);
        }
    }
    ov.server_count = assignments.len();

    // 并发拉取每台在线虚拟机的遥测快照（Contents API，不占用 Kernel）
    let fetched = join_all(assignments.iter().map(|a| fetch_telemetry(app, a))).await;
    {
        let mut t = app.telemetry.lock().unwrap();
        t.retain(|ep, _| assignments.iter().any(|a| &a.endpoint == ep));
        for (a, entry) in assignments.iter().zip(fetched) {
            t.insert(a.endpoint.clone(), entry);
        }
    }

    let ep_to_name: HashMap<String, String> = sessions
        .iter()
        .filter_map(|(n, v)| session_endpoint(v).map(|ep| (ep, n.clone())))
        .collect();
    let busy = app.busy.lock().unwrap().clone();
    let now = chrono::Utc::now().timestamp_millis() as f64 / 1000.0;
    let mut seen: Vec<String> = vec![];

    for a in &assignments {
        let (name, managed) = match ep_to_name.get(&a.endpoint) {
            Some(n) => (n.clone(), true),
            None => (orphan_name(&a.endpoint), false),
        };
        let sess = sessions.get(&name);
        let (tel, terr) = {
            let t = app.telemetry.lock().unwrap();
            t.get(&a.endpoint)
                .map(|e| (e.data.clone(), e.error.clone()))
                .unwrap_or((None, None))
        };
        let age = tel
            .as_ref()
            .and_then(|v| v.get("timestamp"))
            .and_then(|x| x.as_f64())
            .map(|ts| (now - ts).max(0.0));

        // 自动感知新实例或探针失联实例，后台静默注入遥测守护进程
        let needs_probe = tel.is_none() || age.map(|x| x > 25.0).unwrap_or(true);
        if needs_probe && !busy.contains_key(&name) {
            let should_trigger = {
                let mut cd = app.probe_cooldown.lock().unwrap();
                match cd.get(&a.endpoint) {
                    Some(last) if last.elapsed() < Duration::from_secs(25) => false,
                    _ => {
                        cd.insert(a.endpoint.clone(), Instant::now());
                        true
                    }
                }
            };
            if should_trigger {
                let app_bg = app.clone();
                let a_clone = a.clone();
                let name_clone = name.clone();
                app.set_busy(&name_clone, Some("probing"));
                tokio::spawn(async move {
                    let _ = deploy_probe_for_assignment(&app_bg, &a_clone).await;
                    app_bg.set_busy(&name_clone, None);
                });
            }
        }

        let (cli_events, history_paths) = cli::read_session_events(&name, 12);
        let workspace = cli::resolve_workspace_for_vm(
            &name,
            Some(&a.endpoint),
            sess,
            tel.as_ref(),
            &history_paths,
        );

        let cur_busy = app.busy.lock().unwrap().get(&name).cloned();

        ov.vms.push(VmView {
            name: name.clone(),
            workspace,
            endpoint: Some(a.endpoint.clone()),
            accelerator: a.hardware_label(),
            variant: a.variant_label(),
            high_ram: a.is_high_ram(),
            on_server: true,
            managed,
            orphan: !managed,
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
            telemetry: tel,
            telemetry_age: age,
            telemetry_error: terr,
            relay: app.relay(&name).map(|h| h.snapshot()),
            busy: cur_busy,
        });
        seen.push(name);
    }

    // 接力换棒间隙（旧实例已销毁、新实例未分配）与正在创建中的虚拟机也保留卡片
    let relay_names: Vec<(String, Arc<RelayHandle>)> = app
        .relays
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (name, h) in relay_names {
        if !seen.contains(&name) && h.is_running() {
            let cfg = h.snapshot().config;
            let (cli_events, history_paths) = cli::read_session_events(&name, 12);
            let workspace = cli::resolve_workspace_for_vm(&name, None, None, None, &history_paths);
            ov.vms.push(VmView {
                name: name.clone(),
                workspace,
                accelerator: cfg.gpu_type.clone(),
                variant: "GPU".into(),
                high_ram: cfg.high_mem,
                managed: true,
                cli_events,
                relay: Some(h.snapshot()),
                busy: busy.get(&name).cloned(),
                ..Default::default()
            });
            seen.push(name);
        }
    }
    ov.vms.sort_by(|a, b| a.workspace.name.cmp(&b.workspace.name).then(a.name.cmp(&b.name)));
    ov.recent_sessions = cli::read_recent_sessions(&seen, 8);
    ov.llm_bluegreen = cli::read_llm_bluegreen_state();

    *app.overview.write().await = ov.clone();
    ov
}

// ───────────────────────────── 彻底销毁（保留底层能力供 CLI 测试使用） ─────────────────────────────

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

    if stop_relay {
        if let Some(h) = app.relay(name) {
            if h.is_running() {
                h.stop().await;
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
        r.error = Some("无法确定该虚拟机的 endpoint，服务端计费列表中也未找到".into());
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
        app.telemetry.lock().unwrap().remove(&ep);
        if cli::read_sessions().contains_key(name) && cli::colab_bin().exists() {
            let _ = cli::run_simple(&cli::colab_bin(), &s(&["sessions"]), Duration::from_secs(60)).await;
            r.steps.push("已同步清理 colab-cli 本地会话表".into());
        }
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
    if cli::read_sessions().contains_key(&name) || app.busy.lock().unwrap().contains_key(&name) {
        return Err(format!("名称 [{name}] 已被占用"));
    }
    if !cli::colab_bin().exists() {
        return Err("未找到 colab-cli，可执行文件应位于 ~/.local/bin/colab.exe".into());
    }

    app.set_busy(&name, Some("creating"));
    emit_overview(&handle, &refresh_overview(&app, false).await);

    let res = relay::colab_new(&name, &req.accelerator, req.high_mem, &|_l: &str| {}).await;
    if let Err(e) = res {
        app.set_busy(&name, None);
        emit_overview(&handle, &refresh_overview(&app, false).await);
        return Err(e.to_string());
    }

    app.set_busy(&name, Some("probing"));
    emit_overview(&handle, &refresh_overview(&app, false).await);
    let _ = deploy_probe_inner(&app, &name).await;
    app.set_busy(&name, None);

    if req.auto_relay {
        let mut cfg = req.relay_config.unwrap_or_default();
        cfg.gpu_type = if req.accelerator == "CPU" { "T4".into() } else { req.accelerator.clone() };
        cfg.high_mem = req.high_mem;
        start_relay_inner(&app, &name, cfg);
    }
    emit_overview(&handle, &refresh_overview(&app, false).await);
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
    if app.busy.lock().unwrap().get(&name).map(|b| b == "deleting").unwrap_or(false) {
        return Err("该虚拟机正在销毁中".into());
    }
    app.set_busy(&name, Some("deleting"));
    emit_overview(&handle, &refresh_overview(&app, false).await);
    let r = hard_terminate(&app, &name, endpoint, true).await;
    app.set_busy(&name, None);
    if r.verified_unassigned {
        app.relays.lock().unwrap().remove(&name);
    }
    emit_overview(&handle, &refresh_overview(&app, false).await);
    Ok(r)
}

#[tauri::command]
async fn deploy_probe(handle: AppHandle, state: AppStateRef<'_>, name: String) -> Result<(), String> {
    let app = state.inner().clone();
    app.set_busy(&name, Some("probing"));
    let res = deploy_probe_inner(&app, &name).await;
    app.set_busy(&name, None);
    emit_overview(&handle, &refresh_overview(&app, false).await);
    res.map_err(|e| e.to_string())
}

fn start_relay_inner(app: &Arc<AppState>, name: &str, cfg: RelayConfig) -> RelayStatus {
    let h = {
        let mut map = app.relays.lock().unwrap();
        map.entry(name.to_string())
            .or_insert_with(|| Arc::new(RelayHandle::new(name)))
            .clone()
    };
    if h.is_running() {
        return h.snapshot();
    }
    *h.cancel.lock().unwrap() = tokio_util::sync::CancellationToken::new();
    {
        let mut st = h.status.lock().unwrap();
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
    emit_overview(&handle, &refresh_overview(&app, false).await);
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
    emit_overview(&handle, &refresh_overview(&app, false).await);
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
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
    tauri::Builder::default()
        .setup(|app| {
            let state = Arc::new(AppState {
                client: ColabClient::new(),
                workspace: resolve_workspace(app),
                overview: tokio::sync::RwLock::new(Overview::default()),
                telemetry: Mutex::new(HashMap::new()),
                relays: Mutex::new(HashMap::new()),
                busy: Mutex::new(HashMap::new()),
                probe_cooldown: Mutex::new(HashMap::new()),
                gdrive: tokio::sync::Mutex::new((None, None)),
                refresh_lock: tokio::sync::Mutex::new(()),
            });
            app.manage(state.clone());

            // 后台轮询：每 4 秒同步服务端计费实例、CU、遥测与 CLI 历史并推送前端
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    let ov = refresh_overview(&state, false).await;
                    emit_overview(&handle, &ov);
                    tokio::time::sleep(Duration::from_secs(4)).await;
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
            save_workspace_files
        ])
        .run(tauri::generate_context!())
        .expect("error while running Colab Pilot");
}
