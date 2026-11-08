//! Colab Pilot — Tauri 2 Rust 后端
//!
//! - 后台轮询：Colab 服务端计费实例、账户 CU、云端遥测快照、Google Drive 接力目录
//! - 虚拟机生命周期：创建（colab-cli）/ 彻底销毁（colab stop + 服务端 unassign + 计费列表二次核验）
//! - 每台虚拟机独立的 24h+ 接力训练监工任务

mod cli;
mod colab;
mod relay;

use cli::s;
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
    pub endpoint: Option<String>,
    pub accelerator: String,
    pub variant: String,
    pub high_ram: bool,
    pub on_server: bool,
    pub managed: bool,
    pub orphan: bool,
    pub running_task: Option<String>,
    pub last_execution: Value,
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
        Ok(None) => TelemetryEntry { data: None, error: Some("未部署遥测探针".into()) },
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
        ov.vms.push(VmView {
            name: name.clone(),
            endpoint: Some(a.endpoint.clone()),
            accelerator: a.hardware_label(),
            variant: a.variant_label(),
            high_ram: a.is_high_ram(),
            on_server: true,
            managed,
            orphan: !managed,
            running_task: sess
                .and_then(|v| v.get("running"))
                .and_then(|x| x.as_str())
                .map(|x| x.to_string()),
            last_execution: sess
                .and_then(|v| v.get("last_execution"))
                .cloned()
                .unwrap_or(Value::Null),
            telemetry: tel,
            telemetry_age: age,
            telemetry_error: terr,
            relay: app.relay(&name).map(|h| h.snapshot()),
            busy: busy.get(&name).cloned(),
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
            ov.vms.push(VmView {
                name: name.clone(),
                accelerator: cfg.gpu_type.clone(),
                variant: "GPU".into(),
                high_ram: cfg.high_mem,
                managed: true,
                relay: Some(h.snapshot()),
                busy: busy.get(&name).cloned(),
                ..Default::default()
            });
            seen.push(name);
        }
    }
    for (name, what) in &busy {
        if !seen.contains(name) && what == "creating" {
            ov.vms.push(VmView {
                name: name.clone(),
                accelerator: "…".into(),
                busy: Some(what.clone()),
                managed: true,
                ..Default::default()
            });
        }
    }
    ov.vms.sort_by(|a, b| a.name.cmp(&b.name));

    *app.overview.write().await = ov.clone();
    ov
}

// ───────────────────────────── 彻底销毁（防假性删除） ─────────────────────────────

/// 彻底销毁虚拟机并停止计费：
/// 1. 停止本机接力监工与 colab exec 进程树；
/// 2. `colab stop`（关闭远程 Kernel + unassign + 清理 CLI 会话）；
/// 3. 轮询服务端 `/tun/m/assignments`，若实例仍在则直接调用 `/tun/m/unassign` 重试；
/// 4. 仅当服务端计费列表中确认不存在该 endpoint 时才判定成功。
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

    // 2. 官方 CLI 停止（同时关闭远程 Kernel）
    if managed && cli::colab_bin().exists() {
        match cli::run_simple(&cli::colab_bin(), &s(&["stop", "-s", name]), Duration::from_secs(120)).await {
            Ok(o) => {
                let last = o.output.trim().lines().last().unwrap_or("").to_string();
                r.steps.push(format!("colab stop -s {name}：{}", if last.is_empty() { "已执行".into() } else { last }));
            }
            Err(e) => r.steps.push(format!("colab stop 执行失败：{e}")),
        }
    }

    // 3. 服务端计费列表核验 + 直接 unassign 重试
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
        // 清理 colab-cli 本地会话表中的残留记录（官方 sync 逻辑会自动修剪已不在服务端的会话）
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

// ───────────────────────────── 遥测探针部署 ─────────────────────────────

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

pub async fn deploy_probe_inner(app: &Arc<AppState>, name: &str) -> anyhow::Result<()> {
    let a = app
        .find_assignment_by_name(name)
        .await
        .ok_or_else(|| anyhow::anyhow!("服务端未找到实例 [{name}]"))?;
    if !cli::read_sessions().contains_key(name) {
        anyhow::bail!("该实例不在 colab-cli 会话表中（外部创建的孤儿实例），无法执行内核命令");
    }
    app.client
        .contents_upload(&a, "content/.colab_pilot_daemon.py", TELEMETRY_DAEMON_PY.as_bytes())
        .await?;
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
    if let Some(h) = app.relay(&name) {
        let st = h.snapshot();
        if st.running && st.phase == "training" {
            return Err("训练进行中，Kernel 被占用；探针已在本棒训练开始前自动部署".into());
        }
    }
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
            // 保留实例时中断远程 Kernel 中仍在执行的 train.py
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
            gdrive: tokio::sync::Mutex::new((None, None)),
            refresh_lock: tokio::sync::Mutex::new(()),
        });
        let ov = refresh_overview(&state, true).await;
        println!("{}", serde_json::to_string_pretty(&ov).unwrap_or_default());
    });
}

/// 调试构建专用：真实创建 CPU 实例 → 部署探针 → 读取遥测 → 彻底销毁并核验
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
            gdrive: tokio::sync::Mutex::new((None, None)),
            refresh_lock: tokio::sync::Mutex::new(()),
        });
        let name = "selftest-cpu";
        println!("[1] creating {name} …");
        if let Err(e) = relay::colab_new(name, "CPU", false, &|l: &str| println!("    | {l}")).await {
            println!("CREATE FAILED: {e}");
            return;
        }
        println!("[2] deploying probe …");
        match deploy_probe_inner(&state, name).await {
            Ok(_) => println!("    probe ok"),
            Err(e) => println!("    probe failed: {e}"),
        }
        tokio::time::sleep(Duration::from_secs(6)).await;
        let ov = refresh_overview(&state, false).await;
        for vm in &ov.vms {
            let t = vm.telemetry.as_ref();
            println!(
                "[3] vm={} on_server={} managed={} cpu%={:?} mem%={:?} threads={:?} age={:?} err={:?}",
                vm.name,
                vm.on_server,
                vm.managed,
                t.and_then(|t| t["cpu"]["percent"].as_f64()),
                t.and_then(|t| t["memory"]["percent"].as_f64()),
                t.and_then(|t| t["threads"]["total_threads"].as_i64()),
                vm.telemetry_age,
                vm.telemetry_error
            );
        }
        println!("[3b] burn rate while running: {:?}", ov.ccu.as_ref().map(|c| c.burn_rate_hourly));
        println!("[4] hard terminate …");
        let r = hard_terminate(&state, name, None, true).await;
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        let after = refresh_overview(&state, false).await;
        println!(
            "[5] after: server_count={} vms={} sessions_has_name={}",
            after.server_count,
            after.vms.len(),
            cli::read_sessions().contains_key(name)
        );
    });
}

/// 调试构建专用：真实执行 2 棒跨实例接力训练端到端测试（第 1 棒训练 ~3 轮后触发限时交接并销毁 → 第 2 棒新建实例从网盘恢复断点续训至 6 轮完成 → 销毁并回收最终权重）
#[cfg(debug_assertions)]
fn selftest_relay() {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    rt.block_on(async {
        let state = Arc::new(AppState {
            client: ColabClient::new(),
            workspace: Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf(),
            overview: tokio::sync::RwLock::new(Overview::default()),
            telemetry: Mutex::new(HashMap::new()),
            relays: Mutex::new(HashMap::new()),
            busy: Mutex::new(HashMap::new()),
            gdrive: tokio::sync::Mutex::new((None, None)),
            refresh_lock: tokio::sync::Mutex::new(()),
        });
        let name = "selftest-relay";
        let cfg = RelayConfig {
            gpu_type: "CPU".into(),
            high_mem: false,
            gdrive_dir: DEFAULT_GDRIVE_DIR.into(),
            max_session_hours: 0.005,        // ~18 秒后触发换棒退出（约 2 棒跑完 6 轮）
            backup_interval_minutes: 0.05,   // ~3 秒触发一次中途网盘热备
            total_epochs: 6,                 // 共 6 轮
            reset_done_flag: true,
        };
        let handle = Arc::new(RelayHandle::new(name));
        {
            let mut st = handle.status.lock().unwrap();
            st.running = true;
            st.config = cfg;
            st.started_at = Some(chrono::Utc::now().timestamp());
            st.phase = "starting".into();
            st.phase_label = "正在初始化接力监工…".into();
        }
        state.relays.lock().unwrap().insert(name.into(), handle.clone());

        let h_log = handle.clone();
        let printer = tokio::spawn(async move {
            let mut printed = 0usize;
            loop {
                let st = h_log.status.lock().unwrap().clone();
                while printed < st.logs.len() {
                    println!("{}", st.logs[printed]);
                    printed += 1;
                }
                if !st.running {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
        });

        relay::run_relay(state.clone(), handle.clone()).await;
        let _ = printer.await;

        let final_st = handle.snapshot();
        let after = refresh_overview(&state, true).await;
        let local_ckpt = state.workspace.join(format!("checkpoint_{name}.pt"));
        println!("====================================================");
        println!(
            "[RESULT] phase={} round={} server_count={} local_ckpt_exists={} local_ckpt_bytes={}",
            final_st.phase,
            final_st.round,
            after.server_count,
            local_ckpt.exists(),
            std::fs::metadata(&local_ckpt).map(|m| m.len()).unwrap_or(0)
        );
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
    #[cfg(debug_assertions)]
    if std::env::args().any(|a| a == "--selftest-relay") {
        selftest_relay();
        return;
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
                gdrive: tokio::sync::Mutex::new((None, None)),
                refresh_lock: tokio::sync::Mutex::new(()),
            });
            app.manage(state.clone());

            // 后台轮询：每 4 秒同步服务端计费实例、CU 与遥测并推送前端
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
