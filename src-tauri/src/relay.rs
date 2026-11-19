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

/// 严格校验接力配置：杜绝 rclone `--flag` 参数注入、Windows 盘符误传、路径穿越与非法数值
pub fn validate_relay_config(cfg: &RelayConfig) -> Result<()> {
    let gpu = cfg.gpu_type.trim().to_uppercase();
    if !matches!(
        gpu.as_str(),
        "CPU" | "T4" | "L4" | "G4" | "A100" | "H100" | "V5E1" | "V6E1"
    ) {
        return Err(anyhow!("不支持的硬件加速器型号：{}", cfg.gpu_type));
    }

    let gd = cfg.gdrive_dir.trim();
    if gd.is_empty() || gd.starts_with('-') || gd.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(anyhow!(
            "非法的网盘目录格式 `{}`：必须为 `remote:path` 且不能以 `-` 开头或包含空白字符",
            cfg.gdrive_dir
        ));
    }
    let Some((remote, rel_path)) = gd.split_once(':') else {
        return Err(anyhow!(
            "网盘目录 `{}` 缺少远程名称前缀（示例：`gdrive:colab_relay`）",
            cfg.gdrive_dir
        ));
    };
    if remote.len() < 2
        || remote.starts_with('-')
        || !remote
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(anyhow!(
            "非法的 rclone 远程名称 `{remote}`（长度至少 2 字符，不能使用本地盘符）"
        ));
    }
    let clean_rel = rel_path.trim_matches('/');
    if clean_rel.is_empty()
        || rel_path.starts_with('-')
        || rel_path.contains('\\')
        || rel_path.split('/').any(|seg| seg == "..")
    {
        return Err(anyhow!(
            "非法的网盘子路径 `{rel_path}`：不能为空、不能包含 `..` 或反斜杠"
        ));
    }

    if !cfg.max_session_hours.is_finite() || !(0.05..=48.0).contains(&cfg.max_session_hours) {
        return Err(anyhow!("单棒上限时长必须在 0.05 ~ 48.0 小时之间"));
    }
    if !cfg.backup_interval_minutes.is_finite()
        || !(0.5..=1440.0).contains(&cfg.backup_interval_minutes)
    {
        return Err(anyhow!("热备间隔必须在 0.5 ~ 1440 分钟之间"));
    }
    if cfg.total_epochs == 0 || cfg.total_epochs > 1_000_000 {
        return Err(anyhow!("总训练轮次必须在 1 ~ 1,000,000 之间"));
    }
    Ok(())
}

pub(crate) fn strip_ansi_codes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for seq_ch in chars.by_ref() {
                    if ('@'..='~').contains(&seq_ch) {
                        break;
                    }
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

pub(crate) fn validate_colab_install_output(out: &cli::RunOutput) -> Result<()> {
    if out.timed_out {
        return Err(anyhow!("云端 requirements.txt 依赖安装超时"));
    }
    let clean = cli::strip_colab_cli_banners(&out.output);
    let has_fatal = clean.contains("CalledProcessError")
        || clean.contains("Traceback (most recent call last)");
    if out.code != 0 || !clean.contains("Installation Complete") || has_fatal {
        return Err(anyhow!(
            "云端 requirements.txt 依赖安装失败（exit={}）：{}",
            out.code,
            cli::last_meaningful_cli_line(&clean, "安装异常")
        ));
    }
    Ok(())
}

/// 判定单棒训练子进程退出结果：
/// - `Ok(true)`：训练已全部完成（捕获 `TRAINING_COMPLETED` 或远端存在 `training_done.flag`）
/// - `Ok(false)`：正常触发单棒限时交接（显式输出 `HANDOVER_REQUESTED`），允许开启下一棒
/// - `Err(e)`：子进程异常退出、超时、抛出未捕获异常或未发出交接信号即提前退出，必须熔断接力
pub(crate) fn classify_training_round(out: &cli::RunOutput, remote_done: bool) -> Result<bool> {
    if out.timed_out {
        return Err(anyhow!("云端 train.py 执行超时（已安全销毁实例停止计费），请检查接力日志"));
    }
    let clean = strip_ansi_codes(&out.output);
    let has_fatal_trace = clean.contains("Traceback (most recent call last)")
        || clean.contains("SyntaxError:")
        || clean.contains("IndentationError:")
        || clean.contains("CLOUD_SMOKE_FAILED");
    if out.code != 0 || has_fatal_trace {
        return Err(anyhow!(
            "云端 train.py 执行异常退出（exit={}，已安全销毁实例停止计费），请检查接力日志",
            out.code
        ));
    }
    if clean.contains("TRAINING_COMPLETED") || remote_done {
        return Ok(true);
    }
    if clean.contains("HANDOVER_REQUESTED") {
        return Ok(false);
    }
    Err(anyhow!(
        "云端 train.py 未输出 HANDOVER_REQUESTED 或 TRAINING_COMPLETED 即提前结束（已安全销毁实例停止计费，防止无限重启死循环）"
    ))
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
    /// 记录 provisioning 发起前服务端已有的计费 endpoint 集合，用于在取消分配时精准识别刚诞生的残留孤儿实例
    pub pre_provision_endpoints: Mutex<std::collections::HashSet<String>>,
    /// 标记最近一次 stop 是否发生在 provisioning（colab new）阶段
    pub stopped_during_provisioning: AtomicBool,
    /// 后台监工任务是否仍在执行（stop 后到任务真正退出之间仍为 true，用于拒绝重复启动）
    pub task_alive: AtomicBool,
    /// 手动停止接力时是否保留当前云端实例（仅 `stop_relay(..., destroy_vm=false)` 时置 true）
    pub keep_vm_on_stop: AtomicBool,
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
            pre_provision_endpoints: Mutex::new(std::collections::HashSet::new()),
            stopped_during_provisioning: AtomicBool::new(false),
            task_alive: AtomicBool::new(false),
            keep_vm_on_stop: AtomicBool::new(false),
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

    /// 停止接力：取消令牌 + 杀掉本机当前 colab 子进程树（含 provisioning 阶段的 colab new）
    pub async fn stop(&self) {
        let was_prov = self.status.l().phase == "provisioning";
        self.stopped_during_provisioning
            .store(was_prov, Ordering::SeqCst);
        self.token().cancel();
        let pid = *self.child_pid.l();
        if let Some(pid) = pid {
            cli::kill_pid_tree(pid).await;
        }
        let mut st = self.status.l();
        st.running = false;
        st.phase = "stopped".into();
        st.phase_label = "已手动停止接力".into();
    }
}

/// Google Colab `/tun/m/assign` 仅支持 CPU / T4 / A100 附加 `shape=hm`（`--high-mem`）；
/// G4 / L4 / H100 / V5E1 / V6E1 默认固定高规格内存，若携带 `shape=hm` 会被服务端直接拒绝（HTTP 400）。
pub fn supports_high_mem(gpu: &str) -> bool {
    matches!(
        gpu.trim().to_uppercase().as_str(),
        "" | "NONE" | "CPU" | "T4" | "A100"
    )
}

pub(crate) fn gpu_args(gpu: &str, high_mem: bool) -> Vec<String> {
    let g = gpu.trim().to_uppercase();
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
    if high_mem && supports_high_mem(&g) {
        v.push("--high-mem".into());
    }
    v
}

pub(crate) fn format_colab_new_failure(gpu: &str, raw_output: &str) -> String {
    let cleaned = cli::strip_colab_cli_banners(raw_output);
    let tail = cli::last_meaningful_cli_line(&cleaned, "未知错误");
    let lower = cleaned.to_ascii_lowercase();
    let hw = {
        let u = gpu.trim().to_uppercase();
        if u.is_empty() || u == "NONE" {
            "CPU".to_string()
        } else {
            u
        }
    };
    if lower.contains("503")
        || lower.contains("service unavailable")
        || lower.contains("allocation refused")
        || lower.contains("precondition failed")
        || lower.contains("412")
    {
        format!(
            "云端 [{hw}] 算力当前库存暂时售罄或已达账户并发上限（{tail}），建议切换为 G4 / L4 / T4 后重试"
        )
    } else if lower.contains("401")
        || lower.contains("unauthorized")
        || lower.contains("not logged in")
    {
        format!("Colab 授权凭证已过期或无效（{tail}），请在终端执行 `colab auth login` 重新授权")
    } else {
        format!("创建实例失败：{tail}")
    }
}

/// 构造 `colab [--auth <mode>] new -s <name> [--gpu <G> | --tpu <t>] [--high-mem]` 参数序列：
/// `google-colab-cli v0.8+` 中 `--auth` 注册于根 `@app.callback()`，必须置于子命令 `new` 之前。
pub(crate) fn build_colab_new_args(
    name: &str,
    gpu: &str,
    high_mem: bool,
    validated_auth: Option<&str>,
) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(am) = validated_auth.map(str::trim).filter(|s| !s.is_empty()) {
        args.push("--auth".into());
        args.push(am.to_lowercase());
    }
    args.push("new".into());
    args.push("-s".into());
    args.push(name.to_string());
    args.extend(gpu_args(gpu, high_mem));
    args
}

/// 通过 colab-cli 创建实例（支持硬件内存形态自适配、400 自动降级重试、取消与 PID 树跟踪）
pub async fn colab_new_with_opts(
    name: &str,
    gpu: &str,
    high_mem: bool,
    auth_mode: Option<&str>,
    cancel: Option<CancellationToken>,
    pid_slot: Option<&Mutex<Option<u32>>>,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    let mut validated_auth: Option<String> = None;
    if let Some(m) = auth_mode.map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty()) {
        if m != "oauth2" && m != "adc" {
            return Err(anyhow!("非法的授权模式 `{m}`（仅支持 oauth2 或 adc）"));
        }
        validated_auth = Some(m);
    }

    let effective_hm = high_mem && supports_high_mem(gpu);
    if high_mem && !effective_hm {
        log(&format!(
            "ℹ️ 加速器 {} 采用云端固定内存形态，已自动省略 --high-mem（防止服务端 HTTP 400）",
            gpu.trim().to_uppercase()
        ));
    }

    let mut cur_hm = effective_hm;
    for attempt in 0..2usize {
        let args = build_colab_new_args(name, gpu, cur_hm, validated_auth.as_deref());
        let out = cli::run_streaming(
            &cli::colab_bin(),
            &args,
            None,
            Duration::from_secs(600),
            cancel.clone(),
            pid_slot,
            |l| log(l),
        )
        .await?;
        if out.cancelled {
            return Err(anyhow!("创建实例已取消"));
        }
        if out.timed_out {
            return Err(anyhow!("创建实例超时 (600s)"));
        }
        if out.code == 0 && out.output.contains("Session READY") {
            return Ok(());
        }
        let cleaned = cli::strip_colab_cli_banners(&out.output);
        let lower = cleaned.to_ascii_lowercase();
        let shape_rejected = lower.contains("400")
            || lower.contains("backend rejected accelerator")
            || lower.contains("shape");
        if attempt == 0 && cur_hm && shape_rejected {
            log("⚠️ 服务端拒绝 High-RAM 形态请求（HTTP 400），正在自动降级为标准内存重试…");
            cur_hm = false;
            continue;
        }
        return Err(anyhow!("{}", format_colab_new_failure(gpu, &out.output)));
    }
    Err(anyhow!("创建实例失败"))
}

pub async fn colab_new(
    name: &str,
    gpu: &str,
    high_mem: bool,
    cancel: Option<CancellationToken>,
    pid_slot: Option<&Mutex<Option<u32>>>,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<()> {
    colab_new_with_opts(name, gpu, high_mem, None, cancel, pid_slot, log).await
}

async fn done_flag_exists(gdrive_dir: &str) -> bool {
    match cli::run_simple(
        &cli::rclone_bin(),
        &[
            s(&["lsf", "--retries=5", "--low-level-retries=20", "--"]),
            vec![format!("{gdrive_dir}/training_done.flag")],
        ]
        .concat(),
        Duration::from_secs(120),
    )
    .await
    {
        Ok(o) => o.code == 0 && o.stdout.lines().any(|l| l.trim() == "training_done.flag"),
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
    let name = handle.status.l().vm_name.clone();

    if cancel.is_cancelled() {
        // C-1 防漏保护：若用户在 provisioning 阶段点击停止/销毁，即使 colab new 刚在云端完成分配，
        // 只要不是显式选择“保留实例”，退出前仍复查服务端计费列表并兜底释放残留实例。
        if !handle.keep_vm_on_stop.load(Ordering::SeqCst) && !app.is_deleting(&name) {
            if app.find_assignment_by_name(&name).await.is_some() {
                handle.log("🧹 检测到取消期间残留的云端实例，正在兜底销毁并核验停费…");
                let r = hard_terminate(&app, &name, None, false).await;
                handle.log(if r.verified_unassigned {
                    "✅ 残留实例已释放，计费已停止"
                } else {
                    "⚠️ 残留实例销毁未核验，请在面板中手动检查销毁"
                });
            }
        }
        return;
    }

    if let Err(e) = result {
        handle.set_phase("error", format!("接力异常：{e}"));
        handle.log(format!("❌ 接力训练发生异常：{e}"));
        // 计费保护：接力异常中止时若实例仍在服务端计费列表中，立即彻底销毁并核验停费，绝不空转计费
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

async fn relay_inner(app: Arc<AppState>, h: Arc<RelayHandle>, cancel: CancellationToken) -> Result<()> {
    let name = h.status.l().vm_name.clone();
    let cfg = h.status.l().config.clone();
    validate_relay_config(&cfg)?;
    let gdrive = cfg.gdrive_dir.trim().to_string();
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
            s(&["mkdir", "--retries=5", "--low-level-retries=20", "--"]),
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
                s(&["deletefile", "--retries=5", "--low-level-retries=20", "--"]),
                vec![format!("{gdrive}/training_done.flag")],
            ]
            .concat(),
            Duration::from_secs(120),
        )
        .await;
    }

    let mut round: u32 = 1;
    let mut consecutive_short_rounds: u32 = 0;
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
            if let Ok(existing) = app.client.list_assignments().await {
                *h.pre_provision_endpoints.l() =
                    existing.into_iter().map(|a| a.endpoint).collect();
            }
            h.log(format!("☁️ 正在向 Google Colab 申请实例 [{name}]…"));
            let hh = h.clone();
            colab_new(
                &name,
                &cfg.gpu_type,
                cfg.high_mem,
                Some(cancel.clone()),
                Some(&h.child_pid),
                &move |l: &str| hh.log(l),
            )
            .await?;
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
        if out.timed_out || out.code != 0 || !out.output.contains("RCLONE_READY: /") {
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
            validate_colab_install_output(&out)?;
        }

        // 4. 执行训练（--timeout 86400，流式日志）
        h.set_phase("training", format!("第 {round} 棒：云端训练 · 定时热备中"));
        h.log(format!(
            "⏳ 开始训练：单棒上限 {:.2}h，每 {:.0} 分钟同步至 {gdrive}",
            cfg.max_session_hours, cfg.backup_interval_minutes
        ));
        let exec_started = std::time::Instant::now();
        let args = vec![
            "exec".into(),
            "-s".into(),
            name.clone(),
            "--timeout".into(),
            "86400".into(),
            "--env".into(),
            format!("GDRIVE_DIR={gdrive}"),
            "--env".into(),
            format!("RELAY_ROUND={round}"),
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
        let exec_elapsed = exec_started.elapsed();
        if out.cancelled {
            return Ok(());
        }

        // 5. 本棒结束，无论成功或失败均先彻底销毁实例停止计费
        h.set_phase("destroying", format!("第 {round} 棒结束：销毁实例并核验停费"));
        h.log("🧹 本棒结束，正在 unassign 并核验服务端计费列表…");
        let r = hard_terminate(&app, &name, Some(a.endpoint.clone()), false).await;
        if r.verified_unassigned {
            h.log("✅ 本棒实例已彻底销毁，服务端计费已停止");
        } else {
            h.log(format!("⚠️ 销毁未通过核验：{}", r.error.clone().unwrap_or_default()));
            return Err(anyhow!("实例销毁未通过核验，为防止空转计费已中止接力"));
        }

        let remote_done = done_flag_exists(&gdrive).await;
        let round_completed = classify_training_round(&out, remote_done)?;
        if round_completed {
            h.log("🎉 检测到网盘完成标记，全部接力训练已圆满结束！");
            break;
        }

        // 快速失败/异常换棒熔断器：若单棒配置 > 0.1h 但连续 3 棒均在 60 秒内请求换棒，说明计时配置异常，立即熔断
        if cfg.max_session_hours >= 0.1 && exec_elapsed < Duration::from_secs(60) {
            consecutive_short_rounds += 1;
            if consecutive_short_rounds >= 3 {
                return Err(anyhow!(
                    "连续 3 棒在 60 秒内触发换棒退出，疑似脚本换棒阈值异常，已自动熔断防止频繁开关机计费"
                ));
            }
        } else {
            consecutive_short_rounds = 0;
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
                s(&["copyto", "--multi-thread-streams=16", "--retries=5", "--low-level-retries=20", "--"]),
                vec![format!("{gdrive}/checkpoint.pt"), dst.to_string_lossy().to_string()],
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_out(code: i32, output: &str, timed_out: bool) -> cli::RunOutput {
        cli::RunOutput {
            code,
            output: output.to_string(),
            stdout: output.to_string(),
            cancelled: false,
            timed_out,
        }
    }

    #[test]
    fn test_validate_relay_config_rejects_flag_injection_and_bad_bounds() {
        let valid = RelayConfig {
            gpu_type: "A100".into(),
            high_mem: true,
            gdrive_dir: "gdrive:colab_relay/exp_01".into(),
            max_session_hours: 11.5,
            backup_interval_minutes: 15.0,
            total_epochs: 100,
            reset_done_flag: false,
        };
        assert!(validate_relay_config(&valid).is_ok());

        for bad_gdrive in [
            "--config=/etc/passwd",
            "-vv:foo",
            "C:\\Users\\foo",
            "C:/Users/foo",
            "gdrive:../secret",
            "gdrive:colab_relay/../secret",
            "gdrive:",
            "gdrive:   ",
            "gdrive:colab relay",
            "no_colon_path",
        ] {
            let mut c = valid.clone();
            c.gdrive_dir = bad_gdrive.into();
            assert!(
                validate_relay_config(&c).is_err(),
                "expected {bad_gdrive:?} to be rejected"
            );
        }

        let mut bad_gpu = valid.clone();
        bad_gpu.gpu_type = "RTX4090".into();
        assert!(validate_relay_config(&bad_gpu).is_err());

        let mut bad_hours = valid.clone();
        bad_hours.max_session_hours = f64::NAN;
        assert!(validate_relay_config(&bad_hours).is_err());

        let mut bad_epochs = valid.clone();
        bad_epochs.total_epochs = 0;
        assert!(validate_relay_config(&bad_epochs).is_err());
    }

    #[test]
    fn test_classify_training_round_prevents_infinite_restart_loop() {
        // OOM SIGKILL (137), sys.exit(1), SyntaxError, timeout, or silent exit 0 without marker MUST Err
        assert!(classify_training_round(&mk_out(137, "Killed", false), false).is_err());
        assert!(classify_training_round(&mk_out(1, "HANDOVER_REQUESTED", false), false).is_err());
        assert!(classify_training_round(&mk_out(0, "SyntaxError: invalid syntax", false), false).is_err());
        assert!(classify_training_round(&mk_out(0, "HANDOVER_REQUESTED", true), false).is_err());
        assert!(classify_training_round(&mk_out(0, "Epoch 1 loss=0.5\nbye", false), false).is_err());
        assert!(classify_training_round(&mk_out(0, "CLOUD_SMOKE_FAILED: nan", false), false).is_err());

        // Explicit handover with exit 0 -> Ok(false) (continue to next round)
        assert_eq!(
            classify_training_round(&mk_out(0, "CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\n", false), false)
                .unwrap(),
            false
        );

        // Explicit completion or remote done flag with exit 0 -> Ok(true) (finish relay)
        assert_eq!(
            classify_training_round(&mk_out(0, "TRAINING_COMPLETED\n", false), false).unwrap(),
            true
        );
        assert_eq!(
            classify_training_round(&mk_out(0, "all epochs saved", false), true).unwrap(),
            true
        );
    }

    #[test]
    fn test_strip_ansi_and_colab_install_validation() {
        // IPython wraps exception names in ANSI escapes, e.g. "\x1b[0;31mSyntaxError\x1b[0m: invalid syntax"
        let ansi_syntax_err = "\x1b[0;31mSyntaxError\x1b[0m: invalid syntax (train.py, line 12)";
        assert_eq!(
            strip_ansi_codes(ansi_syntax_err),
            "SyntaxError: invalid syntax (train.py, line 12)"
        );
        assert!(classify_training_round(&mk_out(0, ansi_syntax_err, false), true).is_err());

        // colab install returns exit code 0 even when remote pip/uv raises CalledProcessError
        let failed_install = mk_out(
            0,
            "[colab] Installing packages...\n\x1b[0;31mTraceback (most recent call last)\x1b[0m:\nCalledProcessError: Command '['pip', 'install']' returned non-zero exit status 1.",
            false,
        );
        assert!(validate_colab_install_output(&failed_install).is_err());
        assert!(validate_colab_install_output(&mk_out(0, "[colab] Installing packages...", false)).is_err());
        assert!(validate_colab_install_output(&mk_out(
            0,
            "[colab] Installing packages...\nInstallation Complete (via uv)!\n",
            false
        ))
        .is_ok());
    }

    #[test]
    fn test_supports_high_mem_and_gpu_args_never_pass_hm_for_fixed_shape_accelerators() {
        for ok_hw in ["CPU", "NONE", "", "T4", "A100", "t4", "a100"] {
            assert!(supports_high_mem(ok_hw), "{ok_hw} should allow --high-mem");
        }
        for fixed_hw in ["G4", "L4", "H100", "V5E1", "V6E1", "g4", "h100"] {
            assert!(
                !supports_high_mem(fixed_hw),
                "{fixed_hw} must NOT pass --high-mem (HTTP 400 on /tun/m/assign)"
            );
            let args = gpu_args(fixed_hw, true);
            assert!(
                !args.iter().any(|a| a == "--high-mem"),
                "gpu_args({fixed_hw}, true) must omit --high-mem, got {args:?}"
            );
        }
        assert_eq!(gpu_args("G4", true), vec!["--gpu", "G4"]);
        assert_eq!(gpu_args("H100", true), vec!["--gpu", "H100"]);
        assert_eq!(gpu_args("V6E1", true), vec!["--tpu", "v6e1"]);
        assert_eq!(gpu_args("A100", true), vec!["--gpu", "A100", "--high-mem"]);
        assert_eq!(gpu_args("CPU", true), vec!["--high-mem"]);
    }

    #[test]
    fn test_format_colab_new_failure_actionable_503_and_banner_stripping() {
        let raw_503 = "Error: Failed to create runtime: HTTP 503: Service Unavailable\n[colab] A new version of Colab CLI is available: 0.7.1 -> 0.8.0\n[colab] Run `uv tool upgrade google-colab-cli` to update.\n";
        let msg = format_colab_new_failure("H100", raw_503);
        assert!(msg.contains("H100"));
        assert!(msg.contains("售罄") || msg.contains("上限"));
        assert!(msg.contains("HTTP 503"));
        assert!(!msg.contains("uv tool upgrade"));

        let raw_rich_err = "Usage: colab new [OPTIONS]\n┌─ Error ─────────────────────────────────┐\n│ No such option: --auth                  │\n└─────────────────────────────────────────┘\n";
        let msg_rich = format_colab_new_failure("T4", raw_rich_err);
        assert_eq!(msg_rich, "创建实例失败：No such option: --auth");
    }

    #[test]
    fn test_build_colab_new_args_and_real_v08_click_parser_contract() {
        // 1. Structural ordering: global --auth MUST precede subcommand "new"
        assert_eq!(
            build_colab_new_args("vm-a", "T4", true, Some("oauth2")),
            vec!["--auth", "oauth2", "new", "-s", "vm-a", "--gpu", "T4", "--high-mem"]
        );
        assert_eq!(
            build_colab_new_args("vm-b", "G4", true, Some("adc")),
            vec!["--auth", "adc", "new", "-s", "vm-b", "--gpu", "G4"]
        );
        assert_eq!(
            build_colab_new_args("vm-c", "CPU", false, None),
            vec!["new", "-s", "vm-c"]
        );

        // 2. Real google-colab-cli v0.8.0 Click/Typer parser contract (zero network, zero VM allocation)
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let mut command_matrix: Vec<Vec<String>> = Vec::new();
        for auth in [None, Some("oauth2"), Some("adc")] {
            for hw in ["CPU", "T4", "L4", "G4", "A100", "H100", "V5E1", "V6E1"] {
                for hm in [false, true] {
                    command_matrix.push(build_colab_new_args("vm-smoke", hw, hm, auth));
                }
            }
        }
        // Also verify every other colab CLI subcommand signature invoked across src-tauri
        command_matrix.extend([
            s(&["exec", "-s", "vm-smoke", "--timeout", "600"]),
            s(&[
                "exec",
                "-s",
                "vm-smoke",
                "--timeout",
                "86400",
                "--env",
                "GDRIVE_DIR=gdrive:colab_relay",
                "--env",
                "RELAY_ROUND=1",
                "-f",
                "train.py",
            ]),
            s(&["install", "-s", "vm-smoke", "-r", "requirements.txt"]),
            s(&["stop", "-s", "vm-smoke"]),
            s(&["sessions"]),
            s(&["restart-kernel", "-s", "vm-smoke"]),
            s(&["ssh", "-s", "vm-smoke"]),
            s(&["ssh", "-i", "~/.ssh/colab_pilot_ed25519", "-s", "vm-smoke"]),
        ]);

        let matrix_json = serde_json::to_string(&command_matrix).unwrap();
        let py_script = r#"
import json, sys, typer.main
from colab_cli.cli import app

root = typer.main.get_command(app)

def parse_full(argv):
    ctx = root.make_context("colab", list(argv))
    rem = list(getattr(ctx, "_protected_args", None) or getattr(ctx, "protected_args", ())) + list(ctx.args)
    assert rem, f"missing subcommand in {argv}"
    sub_name, sub_cmd, sub_args = root.resolve_command(ctx, rem)
    assert sub_cmd is not None, f"unknown subcommand {sub_name!r} in {argv}"
    sub_ctx = sub_cmd.make_context(sub_name, list(sub_args), parent=ctx)
    assert not sub_ctx.args, f"unexpected leftover args {sub_ctx.args} for {argv}"

for argv in json.loads(sys.argv[1]):
    parse_full(argv)

# Regression guard: confirm misplaced trailing --auth fails the parser as expected
caught = False
try:
    parse_full(["new", "-s", "vm-bad", "--auth", "oauth2"])
except Exception:
    caught = True
assert caught, "parser should reject trailing --auth after subcommand 'new'"
"#;
        let out = std::process::Command::new(&py)
            .args(["-B", "-c", py_script, &matrix_json])
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .expect("spawn colab_python_bin for Click parser contract validation");
        assert!(
            out.status.success(),
            "colab_cli v0.8.0 Click parser rejected generated args: stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
}
