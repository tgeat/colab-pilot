//! 本机 colab / rclone 命令行子进程封装（无控制台窗口、UTF-8、可取消、流式输出）

use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
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

pub fn sessions_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".config")
        .join("colab-cli")
        .join("sessions.json")
}

/// 读取 colab-cli 本地会话表（name -> SessionState JSON）
pub fn read_sessions() -> HashMap<String, Value> {
    std::fs::read_to_string(sessions_path())
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, Value>>(&s).ok())
        .unwrap_or_default()
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
        *slot.lock().unwrap() = child.id();
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
        *slot.lock().unwrap() = None;
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
