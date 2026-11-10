//! 单台虚拟机的 24h+ 全自动跨实例接力训练监工（对应方案 relay_rclone.sh + train.py）

use crate::cli::{self, s};
use crate::{hard_terminate, AppState, LockExt};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct RelayConfig {
    pub gpu_type: String,
    pub high_mem: bool,
    pub gdrive_dir: String,
    pub max_session_hours: f64,
    pub backup_interval_minutes: f64,
    pub total_epochs: u32,
    pub reset_done_flag: bool,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            gpu_type: "T4".into(),
            high_mem: true,
            gdrive_dir: "gdrive:colab_relay".into(),
            max_session_hours: 22.5,
            backup_interval_minutes: 30.0,
            total_epochs: 500,
            reset_done_flag: false,
        }
    }
}

#[derive(Clone, Serialize, Debug, Default)]
pub struct RelayStatus {
    pub vm_name: String,
    pub running: bool,
    pub round: u32,
    pub phase: String,
    pub phase_label: String,
    pub started_at: Option<i64>,
    pub round_started_at: Option<i64>,
    pub config: RelayConfig,
    pub logs: Vec<String>,
    pub child_pid: Option<u32>,
}

pub struct RelayHandle {
    pub status: Mutex<RelayStatus>,
    pub cancel: Mutex<CancellationToken>,
    pub child_pid: Mutex<Option<u32>>,
    /// 后台监工任务是否仍在执行（stop 后到任务真正退出之间仍为 true，用于拒绝重复启动）
    pub task_alive: AtomicBool,
    /// 接力启动代次：仅当前代次的任务退出时才允许改写 running 状态
    pub gen: AtomicU64,
}

impl RelayHandle {
    pub fn new(name: &str) -> Self {
        Self {
            status: Mutex::new(RelayStatus {
                vm_name: name.into(),
                phase: "idle".into(),
                phase_label: "未开启接力".into(),
                ..Default::default()
            }),
            cancel: Mutex::new(CancellationToken::new()),
            child_pid: Mutex::new(None),
            task_alive: AtomicBool::new(false),
            gen: AtomicU64::new(0),
        }
    }

    pub fn snapshot(&self) -> RelayStatus {
        let mut st = self.status.l().clone();
        st.child_pid = *self.child_pid.l();
        if st.logs.len() > 160 {
            st.logs = st.logs[st.logs.len() - 160..].to_vec();
        }
        st
    }

    pub fn is_running(&self) -> bool {
        self.status.l().running
    }

    pub fn log(&self, msg: impl AsRef<str>) {
        let ts = chrono::Local::now().format("%H:%M:%S");
        let mut st = self.status.l();
        st.logs.push(format!("[{ts}] {}", msg.as_ref()));
        if st.logs.len() > 600 {
            let cut = st.logs.len() - 600;
            st.logs.drain(..cut);
        }
    }

    pub fn set_phase(&self, phase: &str, label: impl Into<String>) {
        let mut st = self.status.l();
        st.phase = phase.into();
        st.phase_label = label.into();
    }

    pub fn token(&self) -> CancellationToken {
        self.cancel.l().clone()
    }

    /// 停止接力：取消令牌 + 杀掉本机 colab exec 进程树
    pub async fn stop(&self) {
        self.token().cancel();
        let pid = *self.child_pid.l();
        if let Some(pid) = pid {
            #[cfg(windows)]
            {
                let mut k = tokio::process::Command::new("taskkill");
                k.args(["/PID", &pid.to_string(), "/T", "/F"])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null());
                k.creation_flags(0x0800_0000);
                let _ = k.status().await;
            }
            #[cfg(not(windows))]
            let _ = pid;
        }
        let mut st = self.status.l();
        st.running = false;
        st.phase = "stopped".into();
        st.phase_label = "已手动停止接力".into();
    }
}

fn gpu_args(gpu: &str, high_mem: bool) -> Vec<String> {
    let g = gpu.to_uppercase();
    let mut v = vec![];
    match g.as_str() {
        "T4" | "L4" | "G4" | "A100" | "H100" => {
            v.push("--gpu".into());
            v.push(g.clone());
        }
        "V5E1" | "V6E1" => {
            v.push("--tpu".into());
            v.push(g.to_lowercase());
        }
        _ => {}
    }
    if high_mem {
        v.push("--high-mem".into());
    }
    v
}

/// 通过 colab-cli 创建实例（写入 CLI 会话表，保证后续 colab exec/stop 可用）
pub async fn colab_new(
    name: &str,
    gpu: &str,
    high_mem: bool,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    let mut args = s(&["new", "-s", name]);
    args.extend(gpu_args(gpu, high_mem));
    let out = cli::run_streaming(
        &cli::colab_bin(),
        &args,
        None,
        Duration::from_secs(600),
        None,
        None,
        |l| log(l),
    )
    .await?;
    if out.output.contains("Session READY") {
        Ok(())
    } else {
        Err(anyhow!(
            "创建实例失败：{}",
            out.output.trim().lines().last().unwrap_or("未知错误")
        ))
    }
}

async fn done_flag_exists(gdrive_dir: &str) -> bool {
    match cli::run_simple(
        &cli::rclone_bin(),
        &[
            s(&["lsf", "--retries=5", "--low-level-retries=20"]),
            vec![format!("{gdrive_dir}/training_done.flag")],
        ]
        .concat(),
        Duration::from_secs(120),
    )
    .await
    {
        Ok(o) => o.output.lines().any(|l| l.trim() == "training_done.flag"),
        Err(_) => false,
    }
}

async fn sleep_cancellable(secs: u64, cancel: &CancellationToken) -> bool {
    tokio::select! {
        _ = tokio::time::sleep(Duration::from_secs(secs)) => true,
        _ = cancel.cancelled() => false,
    }
}

/// 任务退出守卫：无论正常结束、出错还是 panic，都会清理运行标记（仅当前代次才改写 running）
struct RelayTaskGuard {
    handle: Arc<RelayHandle>,
    gen: u64,
}

impl Drop for RelayTaskGuard {
    fn drop(&mut self) {
        if self.handle.gen.load(Ordering::SeqCst) == self.gen {
            self.handle.status.l().running = false;
            *self.handle.child_pid.l() = None;
        }
        self.handle.task_alive.store(false, Ordering::SeqCst);
    }
}

pub async fn run_relay(app: Arc<AppState>, handle: Arc<RelayHandle>) {
    let my_gen = handle.gen.load(Ordering::SeqCst);
    let _guard = RelayTaskGuard {
        handle: handle.clone(),
        gen: my_gen,
    };
    let cancel = handle.token();
    let result = relay_inner(app.clone(), handle.clone(), cancel.clone()).await;
    if let Err(e) = result {
        if !cancel.is_cancelled() {
            handle.set_phase("error", format!("接力异常：{e}"));
            handle.log(format!("❌ 接力训练发生异常：{e}"));
            // 计费保护：接力异常中止时若实例仍在服务端计费列表中，立即彻底销毁并核验停费，绝不空转计费
            let name = handle.status.l().vm_name.clone();
            if app.find_assignment_by_name(&name).await.is_some() {
                handle.log("🧹 接力已中止，正在销毁残留实例并核验停止计费…");
                let r = hard_terminate(&app, &name, None, false).await;
                handle.log(if r.verified_unassigned {
                    "✅ 计费已停止"
                } else {
                    "⚠️ 销毁未核验，请手动删除"
                });
            }
        }
    }
}

async fn relay_inner(app: Arc<AppState>, h: Arc<RelayHandle>, cancel: CancellationToken) -> Result<()> {
    let name = h.status.l().vm_name.clone();
    let cfg = h.status.l().config.clone();
    let gdrive = cfg.gdrive_dir.clone();
    let max_secs = cfg.max_session_hours * 3600.0;
    let backup_secs = cfg.backup_interval_minutes * 60.0;
    let conf = cli::rclone_conf_path();
    let rclone = cli::rclone_bin();
    let colab = cli::colab_bin();

    h.set_phase("preparing_gdrive", "准备 Google Drive 接力目录");
    h.log(format!("🔑 使用本机 Rclone 授权文件：{}", conf.display()));
    if !conf.exists() {
        return Err(anyhow!("未找到 rclone.conf，请先完成 rclone 授权"));
    }
    let mkdir_out = cli::run_simple(
        &rclone,
        &[
            s(&["mkdir", "--retries=5", "--low-level-retries=20"]),
            vec![gdrive.clone()],
        ]
        .concat(),
        Duration::from_secs(120),
    )
    .await?;
    if mkdir_out.code != 0 {
        return Err(anyhow!(
            "创建网盘目录 {} 失败：{}",
            gdrive,
            mkdir_out.output.lines().last().unwrap_or("rclone mkdir 失败")
        ));
    }
    if cfg.reset_done_flag {
        h.log(format!("🧹 清理网盘旧完成标记 {gdrive}/training_done.flag"));
        let _ = cli::run_simple(
            &rclone,
            &[
                s(&["deletefile", "--retries=5", "--low-level-retries=20"]),
                vec![format!("{gdrive}/training_done.flag")],
            ]
            .concat(),
            Duration::from_secs(120),
        )
        .await;
    }

    let mut round: u32 = 1;
    loop {
        if cancel.is_cancelled() {
            return Ok(());
        }
        if done_flag_exists(&gdrive).await {
            h.log("🎉 检测到网盘完成标记 training_done.flag，全部接力训练已圆满结束！");
            break;
        }

        {
            let mut st = h.status.l();
            st.round = round;
            st.round_started_at = Some(chrono::Utc::now().timestamp());
        }
        h.log("════════════════════════════════════════");
        h.log(format!(
            "🚀 启动第 {round} 棒接力实例 | 硬件 {} ({})",
            cfg.gpu_type,
            if cfg.high_mem { "High-RAM" } else { "Standard" }
        ));

        // 1. 第 1 棒若同名实例仍在服务端计费列表中则直接复用，否则彻底释放旧实例后新建
        let mut reuse = false;
        if round == 1 {
            if app.find_assignment_by_name(&name).await.is_some() {
                reuse = true;
                h.log(format!("♻️ 检测到运行中的实例 [{name}]，第 1 棒直接复用"));
            }
        }
        if !reuse {
            h.set_phase("provisioning", format!("第 {round} 棒：申请云端 {} 实例", cfg.gpu_type));
            if app.find_assignment_by_name(&name).await.is_some() {
                let r = hard_terminate(&app, &name, None, false).await;
                h.log(format!("🧹 释放旧实例：{}", if r.verified_unassigned { "核验通过" } else { "未核验" }));
            }
            if cancel.is_cancelled() {
                return Ok(());
            }
            h.log(format!("☁️ 正在向 Google Colab 申请实例 [{name}]…"));
            let hh = h.clone();
            colab_new(&name, &cfg.gpu_type, cfg.high_mem, &move |l: &str| hh.log(l)).await?;
        }
        if cancel.is_cancelled() {
            return Ok(());
        }

        let a = app
            .find_assignment_by_name(&name)
            .await
            .ok_or_else(|| anyhow!("服务端未找到实例 [{name}]"))?;
        h.log(format!("✅ 实例就绪，Endpoint：{}", a.endpoint));

        // 2. 上传 rclone.conf 与 train.py
        h.set_phase("uploading_auth", format!("第 {round} 棒：上传 rclone.conf 与 train.py"));
        let conf_bytes = std::fs::read(&conf)?;
        app.client.contents_upload(&a, "content/rclone.conf", &conf_bytes).await?;
        let train_path = app.workspace.join("train.py");
        let train_bytes = std::fs::read(&train_path)?;
        app.client.contents_upload(&a, "content/train.py", &train_bytes).await?;
        h.log("📤 rclone.conf 与 train.py 已上传至 /content");

        // 遥测探针
        match crate::deploy_probe_inner(&app, &name).await {
            Ok(_) => h.log("📡 资源/线程遥测探针已部署"),
            Err(e) => h.log(format!("⚠️ 遥测探针部署失败（不影响训练）：{e}")),
        }
        if cancel.is_cancelled() {
            return Ok(());
        }

        // 3. 云端安装 rclone 与依赖
        h.set_phase("installing_env", format!("第 {round} 棒：云端安装 rclone 与依赖"));
        h.log("🔧 正在云端安装 rclone…");
        let install_code = "import shutil, subprocess\nif not shutil.which('rclone'):\n    subprocess.run(['bash','-c','apt-get update -qq && apt-get install -y -qq rclone'], check=True)\nprint('RCLONE_READY:', shutil.which('rclone'))\n";
        let hh = h.clone();
        let out = cli::run_streaming(
            &colab,
            &s(&["exec", "-s", &name, "--timeout", "600"]),
            Some(install_code.to_string()),
            Duration::from_secs(700),
            Some(cancel.clone()),
            Some(&h.child_pid),
            move |l| hh.log(l),
        )
        .await?;
        if out.cancelled {
            return Ok(());
        }
        if !out.output.contains("RCLONE_READY: /") {
            return Err(anyhow!("云端 rclone 安装失败"));
        }

        let req_path = app.workspace.join("requirements.txt");
        let has_reqs = std::fs::read_to_string(&req_path)
            .map(|t| t.lines().any(|l| !l.trim().is_empty() && !l.trim().starts_with('#')))
            .unwrap_or(false);
        if has_reqs {
            h.log("📦 正在安装 requirements.txt 依赖…");
            let hh = h.clone();
            let out = cli::run_streaming(
                &colab,
                &[s(&["install", "-s", &name, "-r"]), vec![req_path.to_string_lossy().to_string()]].concat(),
                None,
                Duration::from_secs(1800),
                Some(cancel.clone()),
                Some(&h.child_pid),
                move |l| hh.log(l),
            )
            .await?;
            if out.cancelled {
                return Ok(());
            }
        }

        // 4. 执行训练（--timeout 86400，流式日志）
        h.set_phase("training", format!("第 {round} 棒：云端训练 · 定时热备中"));
        h.log(format!(
            "⏳ 开始训练：单棒上限 {:.2}h，每 {:.0} 分钟同步至 {gdrive}",
            cfg.max_session_hours, cfg.backup_interval_minutes
        ));
        let args = vec![
            "exec".into(),
            "-s".into(),
            name.clone(),
            "--timeout".into(),
            "86400".into(),
            "--env".into(),
            format!("GDRIVE_DIR={gdrive}"),
            "--env".into(),
            format!("MAX_SESSION_SECONDS={max_secs}"),
            "--env".into(),
            format!("BACKUP_INTERVAL_SECONDS={backup_secs}"),
            "--env".into(),
            format!("TOTAL_EPOCHS={}", cfg.total_epochs),
            "--env".into(),
            format!("RESET_DONE_FLAG={}", if cfg.reset_done_flag && round == 1 { "1" } else { "0" }),
            "-f".into(),
            train_path.to_string_lossy().to_string(),
        ];
        let hh = h.clone();
        let out = cli::run_streaming(
            &colab,
            &args,
            None,
            Duration::from_secs(90_000),
            Some(cancel.clone()),
            Some(&h.child_pid),
            move |l| {
                if !l.trim().is_empty() {
                    hh.log(l)
                }
            },
        )
        .await?;
        if out.cancelled {
            return Ok(());
        }
        let script_errored = out.output.contains("Traceback (most recent call last)");

        // 5. 本棒结束，彻底销毁实例停止计费
        h.set_phase("destroying", format!("第 {round} 棒结束：销毁实例并核验停费"));
        h.log("🧹 本棒结束，正在 unassign 并核验服务端计费列表…");
        let r = hard_terminate(&app, &name, Some(a.endpoint.clone()), false).await;
        if r.verified_unassigned {
            h.log("✅ 本棒实例已彻底销毁，服务端计费已停止");
        } else {
            h.log(format!("⚠️ 销毁未通过核验：{}", r.error.clone().unwrap_or_default()));
            return Err(anyhow!("实例销毁未通过核验，为防止空转计费已中止接力"));
        }

        if script_errored {
            return Err(anyhow!("云端 train.py 执行抛出异常（已安全销毁实例停止计费），请检查接力日志"));
        }

        if done_flag_exists(&gdrive).await {
            h.log("🎉 检测到网盘完成标记，全部接力训练已圆满结束！");
            break;
        }

        round += 1;
        h.set_phase("cooldown", format!("冷却 15 秒后启动第 {round} 棒"));
        h.log(format!("💤 冷却 15 秒后启动第 {round} 棒…"));
        if !sleep_cancellable(15, &cancel).await {
            return Ok(());
        }
    }

    {
        h.set_phase("completed", "全部训练轮次已完成 (100%)");
        // 确保不残留计费实例
        if app.find_assignment_by_name(&name).await.is_some() {
            let _ = hard_terminate(&app, &name, None, false).await;
        }
        let dst = app.workspace.join(format!("checkpoint_{name}.pt"));
        h.log(format!("📥 正在下载最终权重到本机：{}", dst.display()));
        match cli::run_simple(
            &rclone,
            &[
                s(&["copyto", "--retries=5", "--low-level-retries=20"]),
                vec![format!("{gdrive}/checkpoint.pt"), dst.to_string_lossy().to_string()],
                s(&["--multi-thread-streams=16"]),
            ]
            .concat(),
            Duration::from_secs(1800),
        )
        .await
        {
            Ok(o) if o.code == 0 && dst.exists() => {
                h.log("✅ 全部接力训练与权重回收流程执行完毕！");
            }
            Ok(o) => {
                h.log(format!(
                    "⚠️ 最终权重下载返回非零状态（网盘中仍保留最新权重）：{}",
                    o.output.lines().last().unwrap_or("")
                ));
            }
            Err(e) => {
                h.log(format!("⚠️ 最终权重下载失败（网盘中仍保留最新权重）：{e}"));
            }
        }
    }
    Ok(())
}
