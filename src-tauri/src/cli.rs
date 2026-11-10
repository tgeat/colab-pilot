//! 本机 colab / rclone 命令行与状态读取封装（无控制台窗口、UTF-8、工作区与历史日志解析）

use crate::LockExt;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio_util::sync::CancellationToken;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        vec![".exe".into(), ".cmd".into(), ".bat".into(), String::new()]
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let p = dir.join(format!("{name}{ext}"));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn local_bin(name: &str) -> PathBuf {
    let file = if cfg!(windows) { format!("{name}.exe") } else { name.to_string() };
    dirs::home_dir().unwrap_or_default().join(".local").join("bin").join(file)
}

pub fn colab_bin() -> PathBuf {
    let p = local_bin("colab");
    if p.exists() {
        return p;
    }
    which("colab").unwrap_or(p)
}

/// google-colab-cli 独立虚拟环境中的 Python 解释器（内置 websocket-client 与 colab_cli）
pub fn colab_python_bin() -> PathBuf {
    if cfg!(windows) {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            let p = PathBuf::from(appdata)
                .join("uv")
                .join("tools")
                .join("google-colab-cli")
                .join("Scripts")
                .join("python.exe");
            if p.exists() {
                return p;
            }
        }
    } else {
        let p = dirs::home_dir()
            .unwrap_or_default()
            .join(".local")
            .join("share")
            .join("uv")
            .join("tools")
            .join("google-colab-cli")
            .join("bin")
            .join("python");
        if p.exists() {
            return p;
        }
    }
    which("python").unwrap_or_else(|| PathBuf::from("python"))
}

pub fn rclone_bin() -> PathBuf {
    let p = local_bin("rclone");
    if p.exists() {
        return p;
    }
    which("rclone").unwrap_or(p)
}

pub fn rclone_conf_path() -> PathBuf {
    if let Ok(p) = std::env::var("RCLONE_CONFIG") {
        return PathBuf::from(p);
    }
    if cfg!(windows) {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            return PathBuf::from(appdata).join("rclone").join("rclone.conf");
        }
    }
    dirs::config_dir().unwrap_or_default().join("rclone").join("rclone.conf")
}

pub fn colab_config_dir() -> PathBuf {
    dirs::home_dir().unwrap_or_default().join(".config").join("colab-cli")
}

pub fn sessions_path() -> PathBuf {
    colab_config_dir().join("sessions.json")
}

pub fn history_dir() -> PathBuf {
    colab_config_dir().join("history")
}

pub fn workspaces_registry_path() -> PathBuf {
    colab_config_dir().join("workspaces.json")
}

pub fn llm_bluegreen_state_path() -> PathBuf {
    colab_config_dir().join("llm_bluegreen_state.json")
}

pub fn read_llm_bluegreen_state() -> Option<Value> {
    std::fs::read_to_string(llm_bluegreen_state_path())
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
}

/// 读取 colab-cli 本地会话表（合并默认 sessions.json 与工作区隔离配置）
pub fn read_sessions() -> HashMap<String, Value> {
    let mut merged: HashMap<String, Value> = std::fs::read_to_string(sessions_path())
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, Value>>(&s).ok())
        .unwrap_or_default();

    // 扫描 Desktop 下各工作区可能通过 `--config` 使用的 .colab-sessions.json（文件位置 30 秒缓存，内容每次实时读取）
    for (p, f) in desktop_session_files() {
        if let Ok(raw) = std::fs::read_to_string(&f) {
            if let Ok(map) = serde_json::from_str::<HashMap<String, Value>>(&raw) {
                for (k, mut v) in map {
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert("_workspace_path".into(), Value::String(p.display().to_string()));
                    }
                    merged.entry(k).or_insert(v);
                }
            }
        }
    }
    merged
}

/// 桌面扫描与历史摘要缓存有效期（避免每 1.5 秒轮询都全量扫描磁盘）
const FS_CACHE_TTL: Duration = Duration::from_secs(30);

/// Desktop 下一级子目录列表（30 秒缓存）
fn desktop_subdirs() -> Vec<PathBuf> {
    static CACHE: std::sync::Mutex<Option<(Instant, Vec<PathBuf>)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = CACHE.l().as_ref() {
        if t.elapsed() < FS_CACHE_TTL {
            return v.clone();
        }
    }
    let list: Vec<PathBuf> = dirs::desktop_dir()
        .and_then(|d| std::fs::read_dir(d).ok())
        .map(|entries| entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    *CACHE.l() = Some((Instant::now(), list.clone()));
    list
}

/// 桌面各工作区中存在的会话配置文件：(工作区目录, 配置文件路径)（30 秒缓存）
fn desktop_session_files() -> Vec<(PathBuf, PathBuf)> {
    static CACHE: std::sync::Mutex<Option<(Instant, Vec<(PathBuf, PathBuf)>)>> = std::sync::Mutex::new(None);
    if let Some((t, v)) = CACHE.l().as_ref() {
        if t.elapsed() < FS_CACHE_TTL {
            return v.clone();
        }
    }
    let mut list = Vec::new();
    for p in desktop_subdirs() {
        for cand in [".colab-sessions.json", ".colab/sessions.json"] {
            let f = p.join(cand);
            if f.is_file() {
                list.push((p.clone(), f));
            }
        }
    }
    *CACHE.l() = Some((Instant::now(), list.clone()));
    list
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct WorkspaceInfo {
    pub name: String,
    pub path: Option<String>,
    pub remote_cwd: String,
    pub source: String,
    pub active_script: Option<String>,
    pub description: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct CliEventView {
    pub timestamp: String,
    pub event_type: String,
    pub label: String,
    pub summary: String,
    pub detail: Option<String>,
    pub output_preview: Option<String>,
    pub has_error: bool,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct RecentSessionView {
    pub session_name: String,
    pub workspace_name: String,
    pub workspace_path: Option<String>,
    pub endpoint: Option<String>,
    pub accelerator: String,
    pub machine_shape: String,
    pub created_at: Option<String>,
    pub terminated_at: Option<String>,
    pub last_event_at: String,
    pub exec_count: usize,
    pub file_op_count: usize,
    pub last_summary: String,
}

/// 从任意本地文件路径向上推导所属工作区根目录
pub fn infer_workspace_root_from_path(raw_path: &str) -> Option<(String, String)> {
    let trimmed = raw_path.trim().trim_matches('"').trim_matches('\'');
    if trimmed.is_empty() || trimmed == "stdin" || trimmed.starts_with("automation:") {
        return None;
    }
    let p = Path::new(trimmed);
    let abs = if p.is_absolute() {
        p.to_path_buf()
    } else {
        return None;
    };

    let start_dir = if abs.is_dir() {
        abs.clone()
    } else {
        abs.parent()?.to_path_buf()
    };

    // 向上查找包含项目标志文件的目录
    let markers = [".git", "Cargo.toml", "package.json", "pyproject.toml", "requirements.txt", "README.md"];
    let mut cur: Option<&Path> = Some(&start_dir);
    while let Some(dir) = cur {
        for m in &markers {
            if dir.join(m).exists() {
                let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("workspace").to_string();
                return Some((name, dir.display().to_string()));
            }
        }
        cur = dir.parent();
    }

    let name = start_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("workspace")
        .to_string();
    Some((name, start_dir.display().to_string()))
}

fn summarize_outputs(outputs: Option<&Value>) -> (Option<String>, bool) {
    let Some(arr) = outputs.and_then(|v| v.as_array()) else {
        return (None, false);
    };
    let mut lines: Vec<String> = Vec::new();
    let mut has_err = false;
    for out in arr {
        let otype = out.get("output_type").and_then(|x| x.as_str()).unwrap_or("");
        if otype == "error" {
            has_err = true;
            let ename = out.get("ename").and_then(|x| x.as_str()).unwrap_or("Error");
            let evalue = out.get("evalue").and_then(|x| x.as_str()).unwrap_or("");
            lines.push(format!("{ename}: {evalue}"));
        } else if let Some(text) = out.get("text").and_then(|x| x.as_str()) {
            for l in text.lines() {
                let t = l.trim();
                if !t.is_empty() && !t.starts_with("PROBE_OK") {
                    lines.push(t.to_string());
                }
            }
        }
    }
    if lines.is_empty() {
        return (None, has_err);
    }
    let tail: Vec<String> = lines.into_iter().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect();
    let joined = tail.join(" | ");
    let preview = if joined.chars().count() > 180 {
        format!("{}…", joined.chars().take(180).collect::<String>())
    } else {
        joined
    };
    (Some(preview), has_err)
}

/// 读取指定会话的 colab-cli 结构化历史事件（过滤内部探针启动事件）
pub fn read_session_events(session_name: &str, max_events: usize) -> (Vec<CliEventView>, Vec<String>) {
    let path = history_dir().join(format!("{session_name}.jsonl"));
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return (vec![], vec![]);
    };
    let mut events = Vec::new();
    let mut discovered_paths = Vec::new();

    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let ts = v
            .get("timestamp")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let et = v
            .get("event_type")
            .and_then(|x| x.as_str())
            .unwrap_or("event")
            .to_string();

        // 过滤探针心跳注入事件，避免干扰用户观察 Antigravity 的真实 CLI 操作
        if et == "execution" {
            let code = v.get("code").and_then(|x| x.as_str()).unwrap_or("");
            if code.contains(".colab_pilot_daemon") || code.contains("PROBE_OK") {
                continue;
            }
        }

        let (label, summary, detail, output_preview, has_error) = match et.as_str() {
            "session_created" => {
                let acc = v.get("accelerator").and_then(|x| x.as_str()).unwrap_or("NONE");
                let hw = if acc == "NONE" || acc.is_empty() { "CPU" } else { acc };
                let shape = v.get("machine_shape").and_then(|x| x.as_str()).unwrap_or("STANDARD");
                let ep = v.get("endpoint").and_then(|x| x.as_str()).unwrap_or("");
                (
                    "创建会话".to_string(),
                    format!("colab new -s {session_name} ({hw} · {shape})"),
                    if ep.is_empty() { None } else { Some(ep.to_string()) },
                    None,
                    false,
                )
            }
            "session_terminated" => {
                let reason = v.get("reason").and_then(|x| x.as_str()).unwrap_or("stopped");
                (
                    "销毁会话".to_string(),
                    format!("colab stop -s {session_name}"),
                    Some(format!("原因: {reason}")),
                    None,
                    false,
                )
            }
            "file_operation" => {
                let op = v.get("op").and_then(|x| x.as_str()).unwrap_or("file");
                let local = v.get("local").and_then(|x| x.as_str());
                let remote = v.get("remote").or_else(|| v.get("path")).and_then(|x| x.as_str());
                if let Some(lp) = local {
                    discovered_paths.push(lp.to_string());
                }
                let sum = match (op, local, remote) {
                    ("upload", Some(l), Some(r)) => {
                        let base = Path::new(l).file_name().and_then(|n| n.to_str()).unwrap_or(l);
                        format!("colab upload {base} → {r}")
                    }
                    ("download", Some(l), Some(r)) => {
                        let base = Path::new(l).file_name().and_then(|n| n.to_str()).unwrap_or(l);
                        format!("colab download {r} → {base}")
                    }
                    (_, _, Some(r)) => format!("colab {op} {r}"),
                    _ => format!("colab {op}"),
                };
                (
                    format!("文件 {}", op.to_uppercase()),
                    sum,
                    local.map(|s| s.to_string()),
                    None,
                    false,
                )
            }
            "automation" => {
                let op = v.get("op").and_then(|x| x.as_str()).unwrap_or("auto");
                (
                    "环境自动化".to_string(),
                    format!("colab {op} -s {session_name}"),
                    None,
                    None,
                    false,
                )
            }
            "automation_result" => {
                let op = v.get("op").and_then(|x| x.as_str()).unwrap_or("auto");
                let (prev, err) = summarize_outputs(v.get("outputs"));
                (
                    "自动化结果".to_string(),
                    format!("colab {op} 完成"),
                    None,
                    prev,
                    err,
                )
            }
            "execution" => {
                let code = v.get("code").and_then(|x| x.as_str()).unwrap_or("").trim();
                let first_line = code
                    .lines()
                    .find(|l| {
                        let t = l.trim();
                        !t.is_empty() && !t.starts_with("import os") && !t.starts_with("os.environ[")
                    })
                    .unwrap_or("python code")
                    .trim();
                let short_code = if first_line.chars().count() > 72 {
                    format!("{}…", first_line.chars().take(72).collect::<String>())
                } else {
                    first_line.to_string()
                };
                let (prev, err) = summarize_outputs(v.get("outputs"));
                (
                    "代码执行".to_string(),
                    format!("colab exec · {short_code}"),
                    Some(
                        code.chars()
                            .take(240)
                            .collect::<String>(),
                    ),
                    prev,
                    err,
                )
            }
            "console_started" => (
                "终端连接".to_string(),
                format!("colab console -s {session_name}"),
                None,
                None,
                false,
            ),
            "repl_started" => (
                "交互 REPL".to_string(),
                format!("colab repl -s {session_name}"),
                None,
                None,
                false,
            ),
            other => (
                other.to_string(),
                format!("colab {other}"),
                None,
                None,
                false,
            ),
        };

        events.push(CliEventView {
            timestamp: ts,
            event_type: et,
            label,
            summary,
            detail,
            output_preview,
            has_error,
        });
    }

    let len = events.len();
    let tail = if len > max_events {
        events.into_iter().skip(len - max_events).rev().collect()
    } else {
        events.into_iter().rev().collect()
    };
    (tail, discovered_paths)
}

/// 综合推断某台 Colab 虚拟机对应的本地工作区（Workspace）信息
pub fn resolve_workspace_for_vm(
    session_name: &str,
    endpoint: Option<&str>,
    session_val: Option<&Value>,
    telemetry: Option<&Value>,
    history_paths: &[String],
) -> WorkspaceInfo {
    let mut remote_cwd = "/content".to_string();
    let mut active_script: Option<String> = None;

    if let Some(tel) = telemetry {
        if let Some(ap) = tel.pointer("/threads/active_process") {
            if let Some(cwd) = ap.get("cwd").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
                remote_cwd = cwd.to_string();
            }
            if let Some(cmd) = ap.get("command").and_then(|x| x.as_str()) {
                active_script = Some(cmd.to_string());
            }
        }
    }

    if let Some(sv) = session_val {
        if active_script.is_none() {
            if let Some(le) = sv.get("last_execution").and_then(|x| x.as_array()) {
                if let Some(f) = le.first().and_then(|x| x.as_str()) {
                    if f != "stdin" {
                        active_script = Some(f.to_string());
                    }
                }
            }
        }
    }

    // 1. 优先检查 ~/.config/colab-cli/workspaces.json 显式映射
    if let Ok(raw) = std::fs::read_to_string(workspaces_registry_path()) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, Value>>(&raw) {
            let entry = map
                .get(session_name)
                .or_else(|| endpoint.and_then(|ep| map.get(ep)));
            if let Some(e) = entry {
                let name = e
                    .get("workspace_name")
                    .or_else(|| e.get("name"))
                    .and_then(|x| x.as_str())
                    .unwrap_or(session_name)
                    .to_string();
                let path = e
                    .get("workspace_path")
                    .or_else(|| e.get("path"))
                    .and_then(|x| x.as_str())
                    .map(|s| s.to_string());
                let desc = e.get("description").and_then(|x| x.as_str()).map(|s| s.to_string());
                return WorkspaceInfo {
                    name,
                    path,
                    remote_cwd,
                    source: "工作区注册表 (workspaces.json)".into(),
                    active_script,
                    description: desc,
                };
            }
        }
    }

    // 2. 检查云端 /content/.colab_workspace.json 或进程环境变量 COLAB_WORKSPACE
    if let Some(tel) = telemetry {
        if let Some(wm) = tel.get("workspace_meta").filter(|v| v.is_object()) {
            let name = wm
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or(session_name)
                .to_string();
            let path = wm.get("path").and_then(|x| x.as_str()).map(|s| s.to_string());
            let desc = wm.get("description").and_then(|x| x.as_str()).map(|s| s.to_string());
            return WorkspaceInfo {
                name,
                path,
                remote_cwd,
                source: "云端标记 (.colab_workspace.json)".into(),
                active_script,
                description: desc,
            };
        }
        if let Some(env_ws) = tel.get("detected_workspace_env").and_then(|x| x.as_str()).filter(|s| !s.is_empty()) {
            if let Some((name, path)) = infer_workspace_root_from_path(env_ws) {
                return WorkspaceInfo {
                    name,
                    path: Some(path),
                    remote_cwd,
                    source: "环境变量 (COLAB_WORKSPACE)".into(),
                    active_script,
                    description: None,
                };
            }
            return WorkspaceInfo {
                name: env_ws.to_string(),
                path: None,
                remote_cwd,
                source: "环境变量 (COLAB_WORKSPACE)".into(),
                active_script,
                description: None,
            };
        }
    }

    // 3. 检查会话配置文件所属的工作区目录 (_workspace_path) 或 last_execution / history 中的本地文件路径
    if let Some(sv) = session_val {
        if let Some(wp) = sv.get("_workspace_path").and_then(|x| x.as_str()) {
            if let Some((name, path)) = infer_workspace_root_from_path(wp) {
                return WorkspaceInfo {
                    name,
                    path: Some(path),
                    remote_cwd,
                    source: "工作区隔离配置 (.colab-sessions.json)".into(),
                    active_script,
                    description: None,
                };
            }
        }
        if let Some(le) = sv.get("last_execution").and_then(|x| x.as_array()) {
            if let Some(file_str) = le.first().and_then(|x| x.as_str()) {
                if let Some((name, path)) = infer_workspace_root_from_path(file_str) {
                    return WorkspaceInfo {
                        name,
                        path: Some(path),
                        remote_cwd,
                        source: "CLI 执行路径自动推导".into(),
                        active_script,
                        description: None,
                    };
                }
            }
        }
    }

    for hp in history_paths.iter().rev() {
        if let Some((name, path)) = infer_workspace_root_from_path(hp) {
            return WorkspaceInfo {
                name,
                path: Some(path),
                remote_cwd,
                source: "CLI 传输历史自动推导".into(),
                active_script,
                description: None,
            };
        }
    }

    // 4. 匹配 Desktop 下同名或前缀匹配的工作区文件夹
    if let Some(desktop) = dirs::desktop_dir() {
        let exact = desktop.join(session_name);
        if exact.is_dir() {
            return WorkspaceInfo {
                name: session_name.to_string(),
                path: Some(exact.display().to_string()),
                remote_cwd,
                source: "桌面工作区同名匹配".into(),
                active_script,
                description: None,
            };
        }
        for p in desktop_subdirs() {
            if let Some(dn) = p.file_name().and_then(|n| n.to_str()) {
                let dn_low = dn.to_lowercase();
                let sn_low = session_name.to_lowercase();
                if sn_low.starts_with(&dn_low) || dn_low.starts_with(&sn_low) {
                    return WorkspaceInfo {
                        name: dn.to_string(),
                        path: Some(p.display().to_string()),
                        remote_cwd,
                        source: "桌面工作区名称匹配".into(),
                        active_script,
                        description: None,
                    };
                }
            }
        }
    }

    // 5. 默认将 colab-cli 的 session 名称作为逻辑工作区名
    WorkspaceInfo {
        name: session_name.to_string(),
        path: None,
        remote_cwd,
        source: "Colab CLI 会话命名".into(),
        active_script,
        description: None,
    }
}

/// 读取最近结束的历史会话摘要（用于仪表盘展示近期工作区执行记录）
/// 结果按 (活跃会话集合, limit) 缓存 30 秒；活跃实例增减时立即重新计算
pub fn read_recent_sessions(active_names: &[String], limit: usize) -> Vec<RecentSessionView> {
    type RecentCache = Option<(Instant, Vec<String>, usize, Vec<RecentSessionView>)>;
    static CACHE: std::sync::Mutex<RecentCache> = std::sync::Mutex::new(None);
    let mut key: Vec<String> = active_names.to_vec();
    key.sort();
    if let Some((t, k, l, v)) = CACHE.l().as_ref() {
        if t.elapsed() < FS_CACHE_TTL && *k == key && *l == limit {
            return v.clone();
        }
    }
    let list = read_recent_sessions_uncached(active_names, limit);
    *CACHE.l() = Some((Instant::now(), key, limit, list.clone()));
    list
}

fn read_recent_sessions_uncached(active_names: &[String], limit: usize) -> Vec<RecentSessionView> {
    let dir = history_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let mut list: Vec<RecentSessionView> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if active_names.iter().any(|n| n == stem) {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut ep = None;
        let mut acc = "CPU".to_string();
        let mut shape = "STANDARD".to_string();
        let mut created_at = None;
        let mut terminated_at = None;
        let mut last_event_at = String::new();
        let mut exec_count = 0usize;
        let mut file_op_count = 0usize;
        let mut discovered_paths = Vec::new();

        for line in raw.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line.trim()) else {
                continue;
            };
            let ts = v.get("timestamp").and_then(|x| x.as_str()).unwrap_or("").to_string();
            if !ts.is_empty() {
                last_event_at = ts.clone();
            }
            match v.get("event_type").and_then(|x| x.as_str()).unwrap_or("") {
                "session_created" => {
                    created_at = Some(ts);
                    ep = v.get("endpoint").and_then(|x| x.as_str()).map(|s| s.to_string());
                    let a = v.get("accelerator").and_then(|x| x.as_str()).unwrap_or("NONE");
                    acc = if a == "NONE" || a.is_empty() { "CPU".into() } else { a.into() };
                    shape = v
                        .get("machine_shape")
                        .and_then(|x| x.as_str())
                        .unwrap_or("STANDARD")
                        .into();
                }
                "session_terminated" => {
                    terminated_at = Some(ts);
                }
                "execution" => {
                    let code = v.get("code").and_then(|x| x.as_str()).unwrap_or("");
                    if !code.contains(".colab_pilot_daemon") && !code.contains("PROBE_OK") {
                        exec_count += 1;
                    }
                }
                "file_operation" => {
                    file_op_count += 1;
                    if let Some(lp) = v.get("local").and_then(|x| x.as_str()) {
                        discovered_paths.push(lp.to_string());
                    }
                }
                _ => {}
            }
        }

        let (events, _) = read_session_events(stem, 1);
        let last_summary = events
            .first()
            .map(|e| e.summary.clone())
            .unwrap_or_else(|| "会话已结束".into());
        let ws = resolve_workspace_for_vm(stem, ep.as_deref(), None, None, &discovered_paths);
        list.push(RecentSessionView {
            session_name: stem.to_string(),
            workspace_name: ws.name,
            workspace_path: ws.path,
            endpoint: ep,
            accelerator: acc,
            machine_shape: shape,
            created_at,
            terminated_at,
            last_event_at,
            exec_count,
            file_op_count,
            last_summary,
        });
    }
    list.sort_by(|a, b| b.last_event_at.cmp(&a.last_event_at));
    list.truncate(limit);
    list
}

pub fn build_cmd(bin: &PathBuf, args: &[String]) -> Command {
    let mut cmd = Command::new(bin);
    cmd.args(args)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("NO_COLOR", "1")
        .env("COLUMNS", "200")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);
    cmd
}

/// 结束整个进程树（uv trampoline -> python）
pub async fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        #[cfg(windows)]
        {
            let mut k = Command::new("taskkill");
            k.args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            k.creation_flags(CREATE_NO_WINDOW);
            let _ = k.status().await;
        }
        #[cfg(not(windows))]
        {
            let _ = pid;
        }
    }
    let _ = child.kill().await;
}

pub struct RunOutput {
    pub code: i32,
    pub output: String,
    pub cancelled: bool,
    #[allow(dead_code)]
    pub timed_out: bool,
}

/// 运行命令：可选 stdin、超时、取消，逐行回调（stdout 与 stderr 合并）
pub async fn run_streaming<F>(
    bin: &PathBuf,
    args: &[String],
    stdin_data: Option<String>,
    timeout: Duration,
    cancel: Option<CancellationToken>,
    pid_slot: Option<&std::sync::Mutex<Option<u32>>>,
    mut on_line: F,
) -> Result<RunOutput>
where
    F: FnMut(&str),
{
    let mut cmd = build_cmd(bin, args);
    if stdin_data.is_some() {
        cmd.stdin(Stdio::piped());
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("无法启动 {}", bin.display()))?;
    if let Some(slot) = pid_slot {
        *slot.l() = child.id();
    }

    if let Some(data) = stdin_data {
        if let Some(mut si) = child.stdin.take() {
            si.write_all(data.as_bytes()).await.ok();
            si.shutdown().await.ok();
            drop(si);
        }
    }

    let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    let mut out_lines = BufReader::new(stdout).lines();
    let mut err_lines = BufReader::new(stderr).lines();
    let mut collected = String::new();
    let mut out_done = false;
    let mut err_done = false;
    let cancel = cancel.unwrap_or_default();
    let deadline = tokio::time::sleep(timeout);
    tokio::pin!(deadline);

    let mut cancelled = false;
    let mut timed_out = false;

    while !(out_done && err_done) {
        tokio::select! {
            line = out_lines.next_line(), if !out_done => match line {
                Ok(Some(l)) => { on_line(&l); collected.push_str(&l); collected.push('\n'); }
                _ => out_done = true,
            },
            line = err_lines.next_line(), if !err_done => match line {
                Ok(Some(l)) => { on_line(&l); collected.push_str(&l); collected.push('\n'); }
                _ => err_done = true,
            },
            _ = cancel.cancelled() => { cancelled = true; break; }
            _ = &mut deadline => { timed_out = true; break; }
        }
    }

    if cancelled || timed_out {
        kill_tree(&mut child).await;
    }
    let status = child.wait().await.ok();
    if let Some(slot) = pid_slot {
        *slot.l() = None;
    }
    Ok(RunOutput {
        code: status.and_then(|s| s.code()).unwrap_or(-1),
        output: collected,
        cancelled,
        timed_out,
    })
}

pub async fn run_simple(bin: &PathBuf, args: &[String], timeout: Duration) -> Result<RunOutput> {
    run_streaming(bin, args, None, timeout, None, None, |_| {}).await
}

pub fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}
