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

pub const DEFAULT_BARE_GDRIVE_DIR: &str = "gdrive:colab_relay";

pub fn sanitize_vm_segment(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim_matches(['-', '_', '.']);
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn normalize_gdrive_dir(raw: &str) -> String {
    raw.trim().trim_end_matches('/').to_string()
}

/// 当 `gdrive_dir` 为默认裸值 `"gdrive:colab_relay"`（去首尾空白与末尾斜杠后相等）时，
/// 自动按实例名隔离为 `gdrive:colab_relay/<sanitized_vm_name>`，防止多接力任务互相覆盖断点与完成标记。
pub fn resolve_isolated_gdrive_dir(raw: &str, vm_name: &str) -> String {
    let norm = normalize_gdrive_dir(raw);
    if norm == DEFAULT_BARE_GDRIVE_DIR {
        format!("{DEFAULT_BARE_GDRIVE_DIR}/{}", sanitize_vm_segment(vm_name))
    } else {
        norm
    }
}

/// 计算单棒云端 `colab exec --timeout` 秒数与本机流式等待超时：
/// 确保完整覆盖 `max_session_hours * 3600 + 缓冲时间`，同时不超过云端 86400s (24h) 与本机 90000s 安全上限。
pub(crate) fn compute_relay_timeouts(max_session_hours: f64) -> (u64, Duration) {
    let max_secs = (max_session_hours.clamp(0.1, 23.5) * 3600.0).ceil() as u64;
    let exec_timeout_secs = (max_secs + 1800).clamp(600, 86_400);
    let local_wait_secs = (exec_timeout_secs + 600).min(90_000);
    (exec_timeout_secs, Duration::from_secs(local_wait_secs))
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

    if !cfg.max_session_hours.is_finite() || !(0.1..=23.5).contains(&cfg.max_session_hours) {
        return Err(anyhow!(
            "单棒上限时长必须在 0.1 ~ 23.5 小时之间（受限于 Colab 单次最长 24 小时执行期限）"
        ));
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

#[derive(Clone, Serialize, Deserialize, Debug, Default)]
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
    #[serde(default)]
    pub training_completed: bool,
    #[serde(default)]
    pub weights_downloaded: bool,
    #[serde(default)]
    pub download_error: Option<String>,
    #[serde(default)]
    pub local_weights_path: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
}

pub struct RelayHandle {
    pub status: Mutex<RelayStatus>,
    pub cancel: Mutex<CancellationToken>,
    pub child_pid: Mutex<Option<u32>>,
    /// 绑定的本地工作区目录（上传 train.py / requirements.txt 及保存最终权重均使用该目录）
    pub workspace_dir: Mutex<Option<std::path::PathBuf>>,
    /// 会话授权策略（oauth2 / adc），后续 colab 子命令沿用
    pub auth_mode: Mutex<Option<String>>,
    /// 记录 provisioning 发起前服务端已有的计费 endpoint 集合，用于在取消分配时精准识别刚诞生的残留孤儿实例
    pub pre_provision_endpoints: Mutex<std::collections::HashSet<String>>,
    /// 标记 `pre_provision_endpoints` 快照是否无效/尚未在本次 `colab new` 前成功拉取（无效时严禁猜测残留孤儿 #77）
    pub pre_provision_snapshot_invalid: AtomicBool,
    /// 标记最近一次 stop 是否发生在 provisioning（colab new）阶段
    pub stopped_during_provisioning: AtomicBool,
    /// 标记本次接力的 `gdrive_dir` 是否由默认裸值 `gdrive:colab_relay` 自动隔离而来（用于旧断点迁移保护 #80）
    pub auto_isolated_gdrive: AtomicBool,
    /// 用户显式确认的旧版根目录断点处理动作：`Some("migrate")` | `Some("ignore")` | `None` (#80)
    pub legacy_gdrive_action: Mutex<Option<String>>,
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
            workspace_dir: Mutex::new(None),
            auth_mode: Mutex::new(None),
            pre_provision_endpoints: Mutex::new(std::collections::HashSet::new()),
            pre_provision_snapshot_invalid: AtomicBool::new(false),
            stopped_during_provisioning: AtomicBool::new(false),
            auto_isolated_gdrive: AtomicBool::new(false),
            legacy_gdrive_action: Mutex::new(None),
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
        format!("Colab 授权凭证已过期或无效（{tail}），请在终端执行 `colab auth`（或 `colab status`）重新授权")
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

pub(crate) fn is_done_flag_filename(name: &str) -> bool {
    let t = name.trim();
    t.eq_ignore_ascii_case("training_done.flag")
        || t.eq_ignore_ascii_case("training_completed.flag")
}

pub(crate) async fn done_flag_exists_with_conf(
    rclone_bin: &std::path::PathBuf,
    rclone_conf: &std::path::Path,
    gdrive_dir: &str,
) -> bool {
    let mut args = Vec::new();
    if rclone_conf.exists() {
        args.push("--config".into());
        args.push(rclone_conf.to_string_lossy().to_string());
    }
    args.extend(s(&[
        "lsf",
        "--files-only",
        "--retries=5",
        "--low-level-retries=20",
        "--",
    ]));
    args.push(gdrive_dir.trim_end_matches('/').to_string());
    match cli::run_simple(rclone_bin, &args, Duration::from_secs(120)).await {
        Ok(o) => o.code == 0 && o.stdout.lines().any(is_done_flag_filename),
        Err(_) => false,
    }
}

pub(crate) fn find_latest_valid_hf_checkpoint_dir(
    ws_dir: &std::path::Path,
) -> Option<std::path::PathBuf> {
    let hf_root = ws_dir.join("hf_checkpoints");
    let entries = std::fs::read_dir(&hf_root).ok()?;
    let mut best: Option<(u64, std::path::PathBuf)> = None;
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let Some(dir_name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let Some(step_str) = dir_name.strip_prefix("checkpoint-") else {
            continue;
        };
        let Ok(step_num) = step_str.parse::<u64>() else {
            continue;
        };
        let state_json = p.join("trainer_state.json");
        let valid_state = std::fs::read_to_string(&state_json)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| v.is_object())
            .unwrap_or(false);
        if valid_state && best.as_ref().map(|(s, _)| step_num >= *s).unwrap_or(true) {
            best = Some((step_num, p));
        }
    }
    best.map(|(_, p)| p)
}

/// 下载并严格核验最终权重文件（独立维护 `training_completed` 与 `weights_downloaded` 状态，兼容 HF Trainer `hf_checkpoints`）：
/// 1. 先尝试下载 `{gdrive}/checkpoint.pt`，失败或为空时回退尝试 `{gdrive}/checkpoint_prev.pt`；
/// 2. 若远端明确无 `checkpoint.pt` 但存在 `hf_checkpoints/`（通过 `rclone lsf` 区分无文件与鉴权/网络错误），自动回退拉取并校验有效 `checkpoint-*/trainer_state.json`；
/// 3. 必须校验本地目标文件真实存在且字节数 `> 0` 才置 `weights_downloaded = true`、`phase = "completed"`；
/// 4. 若下载均失败或文件缺失/为空，置 `weights_downloaded = false`、`phase = "download_failed"`、`download_error = Some(...)`。
pub(crate) async fn download_and_verify_final_weights(
    h: &RelayHandle,
    rclone_bin: &std::path::PathBuf,
    rclone_conf: &std::path::Path,
    gdrive_dir: &str,
    ws_dir: &std::path::Path,
    vm_name: &str,
) {
    {
        let mut st = h.status.l();
        st.training_completed = true;
        st.phase = "downloading".into();
        st.phase_label = "正在从 Google Drive 下载最终训练权重…".into();
    }
    let _ = std::fs::create_dir_all(ws_dir);
    let dst = ws_dir.join(format!("checkpoint_{vm_name}.pt"));
    let tmp_dst = ws_dir.join(format!(".checkpoint_{vm_name}.pt.downloading"));
    let _ = std::fs::remove_file(&tmp_dst);
    h.log(format!("📥 正在下载最终权重到本机：{}", dst.display()));

    let gdrive_clean = gdrive_dir.trim_end_matches('/');
    let candidates = [
        format!("{gdrive_clean}/checkpoint.pt"),
        format!("{gdrive_clean}/checkpoint_prev.pt"),
    ];
    let mut errors: Vec<String> = Vec::new();
    let mut downloaded_non_empty = false;

    for (idx, remote_src) in candidates.iter().enumerate() {
        let _ = std::fs::remove_file(&tmp_dst);
        let mut args = Vec::new();
        if rclone_conf.exists() {
            args.push("--config".into());
            args.push(rclone_conf.to_string_lossy().to_string());
        }
        args.extend(s(&[
            "copyto",
            "--multi-thread-streams=16",
            "--retries=5",
            "--low-level-retries=20",
            "--",
        ]));
        args.push(remote_src.clone());
        args.push(tmp_dst.to_string_lossy().to_string());

        match cli::run_simple(rclone_bin, &args, Duration::from_secs(1800)).await {
            Ok(o) if o.code == 0 => {
                let valid = std::fs::metadata(&tmp_dst)
                    .map(|m| m.is_file() && m.len() > 0)
                    .unwrap_or(false);
                if valid {
                    downloaded_non_empty = true;
                    break;
                } else {
                    errors.push(format!(
                        "命令 #{} ({remote_src}) 退出码为 0 但本地文件缺失或大小为 0 字节",
                        idx + 1
                    ));
                }
            }
            Ok(o) => {
                let detail = cli::last_meaningful_cli_line(&o.output, "rclone 返回非零状态");
                errors.push(format!(
                    "命令 #{} ({remote_src}) 失败 (exit={}): {detail}",
                    idx + 1,
                    o.code
                ));
            }
            Err(e) => {
                errors.push(format!("命令 #{} ({remote_src}) 执行异常: {e}", idx + 1));
            }
        }
    }

    let mut verified_local_path: Option<std::path::PathBuf> = None;
    if downloaded_non_empty {
        if std::fs::rename(&tmp_dst, &dst).is_err() {
            if let Err(e) = std::fs::copy(&tmp_dst, &dst) {
                errors.push(format!("写入最终权重文件 `{}` 失败: {e}", dst.display()));
            }
            let _ = std::fs::remove_file(&tmp_dst);
        }
        if std::fs::metadata(&dst)
            .map(|m| m.is_file() && m.len() > 0)
            .unwrap_or(false)
        {
            verified_local_path = Some(dst.clone());
        }
    } else {
        let _ = std::fs::remove_file(&tmp_dst);
        if std::fs::metadata(&dst).map(|m| m.len() == 0).unwrap_or(false) {
            let _ = std::fs::remove_file(&dst);
        }

        // 回退检查是否为 HuggingFace Trainer 目录型断点（仅在 lsf 成功确认无 checkpoint.pt 且存在 hf_checkpoints/ 时回退）
        let mut lsf_args = Vec::new();
        if rclone_conf.exists() {
            lsf_args.push("--config".into());
            lsf_args.push(rclone_conf.to_string_lossy().to_string());
        }
        lsf_args.extend(s(&["lsf", "--retries=3", "--low-level-retries=10", "--"]));
        lsf_args.push(gdrive_clean.to_string());
        if let Ok(lsf_out) = cli::run_simple(rclone_bin, &lsf_args, Duration::from_secs(60)).await {
            if lsf_out.code == 0 {
                let has_pt = lsf_out.stdout.lines().any(|l| {
                    let t = l.trim();
                    t == "checkpoint.pt" || t == "checkpoint_prev.pt"
                });
                let has_hf_dir = lsf_out
                    .stdout
                    .lines()
                    .any(|l| l.trim().trim_end_matches('/') == "hf_checkpoints");
                if !has_pt && has_hf_dir {
                    let hf_dst = ws_dir.join("hf_checkpoints");
                    let mut hf_args = Vec::new();
                    if rclone_conf.exists() {
                        hf_args.push("--config".into());
                        hf_args.push(rclone_conf.to_string_lossy().to_string());
                    }
                    hf_args.extend(s(&[
                        "copy",
                        "--multi-thread-streams=16",
                        "--retries=5",
                        "--low-level-retries=20",
                        "--",
                    ]));
                    hf_args.push(format!("{gdrive_clean}/hf_checkpoints"));
                    hf_args.push(hf_dst.to_string_lossy().to_string());
                    match cli::run_simple(rclone_bin, &hf_args, Duration::from_secs(1800)).await {
                        Ok(ho) if ho.code == 0 => {
                            if let Some(best_hf) = find_latest_valid_hf_checkpoint_dir(ws_dir) {
                                verified_local_path = Some(best_hf);
                            } else {
                                errors.push(
                                    "hf_checkpoints 同步退出码为 0，但未找到有效的 checkpoint-*/trainer_state.json"
                                        .into(),
                                );
                            }
                        }
                        Ok(ho) => {
                            let detail = cli::last_meaningful_cli_line(
                                &ho.output,
                                "rclone copy hf_checkpoints 失败",
                            );
                            errors.push(format!(
                                "回退拉取 hf_checkpoints 失败 (exit={}): {detail}",
                                ho.code
                            ));
                        }
                        Err(e) => {
                            errors.push(format!("回退拉取 hf_checkpoints 异常: {e}"));
                        }
                    }
                }
            }
        }
    }

    if let Some(saved_path) = verified_local_path {
        // 顺带同步远端 results/ 目录（最佳努力，不阻塞主权重回收结果）
        let results_dst = ws_dir.join("results");
        let mut res_args = Vec::new();
        if rclone_conf.exists() {
            res_args.push("--config".into());
            res_args.push(rclone_conf.to_string_lossy().to_string());
        }
        res_args.extend(s(&["copy", "--retries=3", "--low-level-retries=10", "--"]));
        res_args.push(format!("{gdrive_clean}/results"));
        res_args.push(results_dst.to_string_lossy().to_string());
        let _ = cli::run_simple(rclone_bin, &res_args, Duration::from_secs(300)).await;

        {
            let mut st = h.status.l();
            st.training_completed = true;
            st.weights_downloaded = true;
            st.download_error = None;
            st.last_error = None;
            st.local_weights_path = Some(saved_path.display().to_string());
            st.phase = "completed".into();
            st.phase_label = "全部训练轮次已完成，最终权重已回收 (100%)".into();
        }
        h.log(format!(
            "✅ 全部接力训练与权重回收流程执行完毕！已保存至 {}",
            saved_path.display()
        ));
    } else {
        let err_summary = if errors.is_empty() {
            format!("未能从 {gdrive_clean} 下载有效非空权重文件")
        } else {
            errors.join("；")
        };
        {
            let mut st = h.status.l();
            st.training_completed = true;
            st.weights_downloaded = false;
            st.download_error = Some(err_summary.clone());
            st.last_error = Some(err_summary.clone());
            st.local_weights_path = None;
            st.phase = "download_failed".into();
            st.phase_label = format!("云端训练已完成，但最终权重下载失败：{err_summary}");
        }
        h.log(format!(
            "❌ 云端训练已完成，但最终权重下载到本地失败（网盘仍保留权重，可点击重试补下载）：{err_summary}"
        ));
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
        let err_str = format!("{e:#}");
        {
            let mut st = handle.status.l();
            st.phase = "error".into();
            st.phase_label = format!("接力异常：{err_str}");
            st.last_error = Some(err_str.clone());
        }
        handle.log(format!("❌ 接力训练发生异常：{err_str}"));
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

pub(crate) fn detect_legacy_gdrive_artifacts_in_lsf(lsf_stdout: &str) -> Vec<String> {
    const FILE_TARGETS: &[&str] = &[
        "checkpoint.pt",
        "checkpoint_prev.pt",
        "training_done.flag",
        "TRAINING_COMPLETED.flag",
    ];
    let mut found = Vec::new();
    for raw in lsf_stdout.lines() {
        let item = raw.trim();
        if item.is_empty() {
            continue;
        }
        let clean = item.trim_end_matches('/');
        if !item.ends_with('/') && FILE_TARGETS.contains(&item) {
            found.push(item.to_string());
        } else if clean == "hf_checkpoints" {
            found.push("hf_checkpoints/".to_string());
        }
    }
    found
}

async fn rclone_lsf_entries_with_conf(
    rclone_bin: &std::path::PathBuf,
    rclone_conf: &std::path::Path,
    remote_dir: &str,
) -> Option<Vec<String>> {
    let mut args = Vec::new();
    if rclone_conf.exists() {
        args.push("--config".into());
        args.push(rclone_conf.to_string_lossy().to_string());
    }
    args.extend(s(&["lsf", "--retries=5", "--low-level-retries=20", "--"]));
    args.push(remote_dir.trim_end_matches('/').to_string());
    let out = cli::run_simple(rclone_bin, &args, Duration::from_secs(90))
        .await
        .ok()?;
    if out.code == 0 {
        return Some(
            out.stdout
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect(),
        );
    }
    let low = out.output.to_ascii_lowercase();
    if low.contains("directory not found")
        || low.contains("object not found")
        || low.contains("file not found")
    {
        return Some(Vec::new());
    }
    None
}

/// Issue #80：当默认裸路径 `gdrive:colab_relay` 自动升级隔离为 `gdrive:colab_relay/<vm>` 时，
/// 若新隔离子目录尚无断点/完成标记，但旧版根目录 `gdrive:colab_relay` 下存在历史断点/完成标记：
/// - 默认在开训前中止并提示确认目录迁移（拒绝静默跳过旧断点从头训练）；
/// - `Some("migrate")`：自动将旧版顶层断点与结果迁移至新隔离目录后续训；
/// - `Some("ignore")`：确认忽略旧版断点并在新隔离目录全新开训。
pub(crate) async fn check_or_migrate_legacy_bare_gdrive_with_conf(
    rclone_bin: &std::path::PathBuf,
    rclone_conf: &std::path::Path,
    isolated_gdrive: &str,
    legacy_action: Option<&str>,
    h: &RelayHandle,
) -> Result<()> {
    let iso_clean = isolated_gdrive.trim_end_matches('/');
    if iso_clean == DEFAULT_BARE_GDRIVE_DIR {
        return Ok(());
    }

    if let Some(iso_entries) = rclone_lsf_entries_with_conf(rclone_bin, rclone_conf, iso_clean).await {
        let joined = iso_entries.join("\n");
        if !detect_legacy_gdrive_artifacts_in_lsf(&joined).is_empty() {
            return Ok(());
        }
    }

    let Some(bare_entries) =
        rclone_lsf_entries_with_conf(rclone_bin, rclone_conf, DEFAULT_BARE_GDRIVE_DIR).await
    else {
        return Ok(());
    };
    let bare_joined = bare_entries.join("\n");
    let legacy_found = detect_legacy_gdrive_artifacts_in_lsf(&bare_joined);
    if legacy_found.is_empty() {
        return Ok(());
    }

    let action = legacy_action.map(str::trim).unwrap_or("");
    if action.eq_ignore_ascii_case("ignore") {
        h.log(format!(
            "ℹ️ 已确认忽略旧版目录 `{DEFAULT_BARE_GDRIVE_DIR}` 中的历史断点（{}），将在新隔离目录 `{iso_clean}` 中全新启动",
            legacy_found.join(", ")
        ));
        return Ok(());
    }

    if !action.eq_ignore_ascii_case("migrate") {
        return Err(anyhow!(
            "检测到旧版默认目录 `{DEFAULT_BARE_GDRIVE_DIR}` 下存在历史断点（{}），而新版会话隔离目录 `{iso_clean}` 尚无断点！为防止跳过旧断点从头训练已中止启动：请在接力面板点击确认「迁移旧断点并续训」或「忽略旧断点全新开始」（或执行 `rclone moveto {DEFAULT_BARE_GDRIVE_DIR}/checkpoint.pt {iso_clean}/checkpoint.pt`）",
            legacy_found.join(", ")
        ));
    }

    let mut mk_args = Vec::new();
    if rclone_conf.exists() {
        mk_args.push("--config".into());
        mk_args.push(rclone_conf.to_string_lossy().to_string());
    }
    mk_args.extend(s(&["mkdir", "--retries=5", "--low-level-retries=20", "--"]));
    mk_args.push(iso_clean.to_string());
    let mk_out = cli::run_simple(rclone_bin, &mk_args, Duration::from_secs(90)).await?;
    if mk_out.code != 0 {
        return Err(anyhow!(
            "创建迁移目标目录 `{iso_clean}` 失败：{}",
            cli::last_meaningful_cli_line(&mk_out.output, "rclone mkdir 失败")
        ));
    }

    let mut migrate_items = legacy_found.clone();
    if bare_entries
        .iter()
        .any(|e| e == "results" || e == "results/")
        && !migrate_items.iter().any(|e| e == "results/")
    {
        migrate_items.push("results/".to_string());
    }

    for item in &migrate_items {
        let clean_item = item.trim_end_matches('/');
        let src = format!("{DEFAULT_BARE_GDRIVE_DIR}/{clean_item}");
        let dst = format!("{iso_clean}/{clean_item}");
        let subcmd = if item.ends_with('/') { "move" } else { "moveto" };
        let mut mv_args = Vec::new();
        if rclone_conf.exists() {
            mv_args.push("--config".into());
            mv_args.push(rclone_conf.to_string_lossy().to_string());
        }
        mv_args.extend(s(&[subcmd, "--retries=5", "--low-level-retries=20", "--"]));
        mv_args.push(src.clone());
        mv_args.push(dst.clone());
        let mv_out = cli::run_simple(rclone_bin, &mv_args, Duration::from_secs(300)).await?;
        if mv_out.code != 0 {
            return Err(anyhow!(
                "迁移旧版断点 `{src}` -> `{dst}` 失败：{}",
                cli::last_meaningful_cli_line(&mv_out.output, "rclone 迁移失败")
            ));
        }
    }

    h.log(format!(
        "✅ 已将旧版目录 `{DEFAULT_BARE_GDRIVE_DIR}` 下的历史断点（{}）迁移至 `{iso_clean}`，继续续训",
        legacy_found.join(", ")
    ));
    Ok(())
}

async fn relay_inner(app: Arc<AppState>, h: Arc<RelayHandle>, cancel: CancellationToken) -> Result<()> {
    let name = h.status.l().vm_name.clone();
    let mut cfg = h.status.l().config.clone();
    let raw_was_bare = normalize_gdrive_dir(&cfg.gdrive_dir) == DEFAULT_BARE_GDRIVE_DIR;
    if raw_was_bare {
        cfg.gdrive_dir = resolve_isolated_gdrive_dir(&cfg.gdrive_dir, &name);
        h.status.l().config.gdrive_dir = cfg.gdrive_dir.clone();
    }
    validate_relay_config(&cfg)?;
    let was_auto_isolated = raw_was_bare || h.auto_isolated_gdrive.load(Ordering::SeqCst);
    let legacy_action = h.legacy_gdrive_action.l().clone();
    let gdrive = cfg.gdrive_dir.trim().trim_end_matches('/').to_string();
    let max_secs = cfg.max_session_hours * 3600.0;
    let backup_secs = cfg.backup_interval_minutes * 60.0;
    let (exec_timeout_secs, host_wait_timeout) = compute_relay_timeouts(cfg.max_session_hours);
    let ws_dir = h
        .workspace_dir
        .l()
        .clone()
        .or_else(|| cli::lookup_bound_workspace_path(&name))
        .unwrap_or_else(|| app.workspace.clone());
    let eff_auth = h
        .auth_mode
        .l()
        .clone()
        .or_else(|| cli::get_session_auth_mode(&name));
    let conf = cli::rclone_conf_path();
    let rclone = cli::rclone_bin();
    let colab = cli::colab_bin();

    h.set_phase("preparing_gdrive", "准备 Google Drive 接力目录");
    h.log(format!("📂 绑定工作区目录：{}", ws_dir.display()));
    h.log(format!("🔑 使用本机 Rclone 授权文件：{}", conf.display()));
    if !conf.exists() {
        return Err(anyhow!("未找到 rclone.conf，请先完成 rclone 授权"));
    }
    if was_auto_isolated {
        check_or_migrate_legacy_bare_gdrive_with_conf(
            &rclone,
            &conf,
            &gdrive,
            legacy_action.as_deref(),
            &h,
        )
        .await?;
        *h.legacy_gdrive_action.l() = None;
    }
    let mkdir_args = vec![
        "--config".into(),
        conf.to_string_lossy().to_string(),
        "mkdir".into(),
        "--retries=5".into(),
        "--low-level-retries=20".into(),
        "--".into(),
        gdrive.clone(),
    ];
    let mkdir_out = cli::run_simple(&rclone, &mkdir_args, Duration::from_secs(120)).await?;
    if mkdir_out.code != 0 {
        return Err(anyhow!(
            "创建网盘目录 {} 失败：{}",
            gdrive,
            mkdir_out.output.lines().last().unwrap_or("rclone mkdir 失败")
        ));
    }
    if cfg.reset_done_flag {
        h.log(format!(
            "🧹 清理网盘旧完成标记 {gdrive}/training_done.flag 与 TRAINING_COMPLETED.flag"
        ));
        for flag_name in ["training_done.flag", "TRAINING_COMPLETED.flag"] {
            let del_args = vec![
                "--config".into(),
                conf.to_string_lossy().to_string(),
                "deletefile".into(),
                "--retries=5".into(),
                "--low-level-retries=20".into(),
                "--".into(),
                format!("{gdrive}/{flag_name}"),
            ];
            let _ = cli::run_simple(&rclone, &del_args, Duration::from_secs(120)).await;
        }
    }

    let mut round: u32 = 1;
    let mut consecutive_short_rounds: u32 = 0;
    loop {
        if cancel.is_cancelled() {
            return Ok(());
        }
        if done_flag_exists_with_conf(&rclone, &conf, &gdrive).await {
            h.status.l().training_completed = true;
            let local_dst = ws_dir.join(format!("checkpoint_{name}.pt"));
        let local_ok = std::fs::metadata(&local_dst)
                .map(|m| m.is_file() && m.len() > 0)
                .unwrap_or(false)
                || find_latest_valid_hf_checkpoint_dir(&ws_dir).is_some();
            if !local_ok {
                h.log("🎉 检测到网盘已完成标记且本机尚未存在最终权重，立即执行补下载回收！");
            } else {
                h.log("🎉 检测到网盘完成标记，全部接力训练已圆满结束，正在核验回收最终权重！");
            }
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
            if app.find_assignment_by_name(&name).await.is_some() {
                h.set_phase("destroying", format!("第 {round} 棒前清理残留同名实例"));
                let r = hard_terminate(&app, &name, None, false).await;
                h.log(format!("🧹 释放旧实例：{}", if r.verified_unassigned { "核验通过" } else { "未核验" }));
            }
            if cancel.is_cancelled() {
                return Ok(());
            }
            match app.client.list_assignments().await {
                Ok(existing) => {
                    *h.pre_provision_endpoints.l() =
                        existing.into_iter().map(|a| a.endpoint).collect();
                    h.pre_provision_snapshot_invalid.store(false, Ordering::SeqCst);
                }
                Err(_) => {
                    h.pre_provision_endpoints.l().clear();
                    h.pre_provision_snapshot_invalid.store(true, Ordering::SeqCst);
                }
            }
            h.set_phase("provisioning", format!("第 {round} 棒：申请云端 {} 实例", cfg.gpu_type));
            h.log(format!("☁️ 正在向 Google Colab 申请实例 [{name}]…"));
            let hh = h.clone();
            let prov_res = colab_new_with_opts(
                &name,
                &cfg.gpu_type,
                cfg.high_mem,
                eff_auth.as_deref(),
                Some(cancel.clone()),
                Some(&h.child_pid),
                &move |l: &str| hh.log(l),
            )
            .await;
            if prov_res.is_ok() {
                h.pre_provision_snapshot_invalid.store(true, Ordering::SeqCst);
                h.set_phase("connecting", format!("第 {round} 棒：连接已分配实例"));
            }
            prov_res?;
        }
        if cancel.is_cancelled() {
            return Ok(());
        }

        let a = app
            .find_assignment_by_name(&name)
            .await
            .ok_or_else(|| anyhow!("服务端未找到实例 [{name}]"))?;
        h.log(format!("✅ 实例就绪，Endpoint：{}", a.endpoint));

        // 2. 上传 rclone.conf、工作区标记与绑定工作区的 train.py
        h.set_phase("uploading_auth", format!("第 {round} 棒：上传 rclone.conf 与 train.py"));
        let conf_bytes = std::fs::read(&conf)?;
        app.client.contents_upload(&a, "content/rclone.conf", &conf_bytes).await?;
        let train_path = ws_dir.join("train.py");
        let train_bytes = std::fs::read(&train_path).map_err(|e| {
            anyhow!(
                "无法从绑定工作区读取训练脚本 `{}`：{e}",
                train_path.display()
            )
        })?;
        app.client.contents_upload(&a, "content/train.py", &train_bytes).await?;
        let ws_meta_json = serde_json::json!({
            "name": ws_dir.file_name().and_then(|s| s.to_str()).unwrap_or(&name),
            "path": ws_dir.display().to_string(),
        });
        if let Ok(ws_bytes) = serde_json::to_vec(&ws_meta_json) {
            let _ = app
                .client
                .contents_upload(&a, "content/.colab_workspace.json", &ws_bytes)
                .await;
        }
        h.log(format!(
            "📤 rclone.conf 与 {} 已上传至 /content",
            train_path.display()
        ));

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
        let rclone_exec_args = cli::build_colab_subcommand_args(
            eff_auth.as_deref(),
            &s(&["exec", "-s", &name, "--timeout", "600"]),
        );
        let out = cli::run_streaming(
            &colab,
            &rclone_exec_args,
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

        let req_path = ws_dir.join("requirements.txt");
        let has_reqs = std::fs::read_to_string(&req_path)
            .map(|t| t.lines().any(|l| !l.trim().is_empty() && !l.trim().starts_with('#')))
            .unwrap_or(false);
        if has_reqs {
            h.log(format!(
                "📦 正在安装 {} 依赖…",
                req_path.display()
            ));
            let hh = h.clone();
            let raw_install_args = [
                s(&["install", "-s", &name, "-r"]),
                vec![req_path.to_string_lossy().to_string()],
            ]
            .concat();
            let install_args =
                cli::build_colab_subcommand_args(eff_auth.as_deref(), &raw_install_args);
            let out = cli::run_streaming(
                &colab,
                &install_args,
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

        // 4. 执行训练（流式日志，超时严格对齐 max_session_hours + 缓冲）
        h.set_phase("training", format!("第 {round} 棒：云端训练 · 定时热备中"));
        h.log(format!(
            "⏳ 开始训练：单棒上限 {:.2}h（exec 超时 {}s），每 {:.0} 分钟同步至 {gdrive}",
            cfg.max_session_hours, exec_timeout_secs, cfg.backup_interval_minutes
        ));
        let exec_started = std::time::Instant::now();
        let raw_train_args = vec![
            "exec".into(),
            "-s".into(),
            name.clone(),
            "--timeout".into(),
            exec_timeout_secs.to_string(),
            "--env".into(),
            format!("COLAB_WORKSPACE={}", ws_dir.display()),
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
        let args = cli::build_colab_subcommand_args(eff_auth.as_deref(), &raw_train_args);
        let hh = h.clone();
        let out = cli::run_streaming(
            &colab,
            &args,
            None,
            host_wait_timeout,
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

        let remote_done = done_flag_exists_with_conf(&rclone, &conf, &gdrive).await;
        let round_completed = classify_training_round(&out, remote_done)?;
        if round_completed {
            h.status.l().training_completed = true;
            h.log("🎉 检测到网盘完成标记，全部接力训练已圆满结束！");
            break;
        }

        // 快速失败/异常换棒熔断器：若单棒配置 >= 0.1h 但连续 3 棒均在 60 秒内请求换棒，说明计时配置异常，立即熔断
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

    // 确保不残留计费实例
    if app.find_assignment_by_name(&name).await.is_some() {
        let _ = hard_terminate(&app, &name, None, false).await;
    }
    download_and_verify_final_weights(&h, &rclone, &conf, &gdrive, &ws_dir, &name).await;
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

    #[test]
    fn test_issue_44_52_54_gdrive_isolation_cpu_preserved_and_23_5h_session_cap() {
        // #44: bare default gdrive:colab_relay auto-isolates by sanitized VM name
        assert_eq!(
            resolve_isolated_gdrive_dir("gdrive:colab_relay", "resnet-a"),
            "gdrive:colab_relay/resnet-a"
        );
        assert_eq!(
            resolve_isolated_gdrive_dir("  gdrive:colab_relay/  ", "vm_02"),
            "gdrive:colab_relay/vm_02"
        );
        assert_eq!(
            resolve_isolated_gdrive_dir("gdrive:colab_relay/custom_shared", "resnet-a"),
            "gdrive:colab_relay/custom_shared"
        );

        // #52: CPU gpu_type is valid and preserved
        let cpu_cfg = RelayConfig {
            gpu_type: "CPU".into(),
            high_mem: false,
            gdrive_dir: "gdrive:colab_relay/cpu-job".into(),
            max_session_hours: 23.5,
            backup_interval_minutes: 15.0,
            total_epochs: 20,
            reset_done_flag: false,
        };
        assert!(validate_relay_config(&cpu_cfg).is_ok());

        // #54: strict 0.1..=23.5h bounds (reject >23.5 such as 24h, 30h, 48h and <=0)
        for bad_h in [0.0, -1.0, 0.05, 23.51, 24.0, 30.0, 48.0] {
            let mut c = cpu_cfg.clone();
            c.max_session_hours = bad_h;
            assert!(
                validate_relay_config(&c).is_err(),
                "max_session_hours={bad_h} must be rejected"
            );
        }
        for ok_h in [0.1, 1.0, 12.0, 22.5, 23.5] {
            let mut c = cpu_cfg.clone();
            c.max_session_hours = ok_h;
            assert!(
                validate_relay_config(&c).is_ok(),
                "max_session_hours={ok_h} must be accepted"
            );
        }

        let (exec_to_max, local_wait_max) = compute_relay_timeouts(23.5);
        assert_eq!(exec_to_max, 86_400);
        assert!(local_wait_max >= Duration::from_secs(87_000));
        assert!(local_wait_max <= Duration::from_secs(90_000));
    }

    #[tokio::test]
    async fn test_issue_53_final_weights_download_verification_and_rerun_recovery() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let py = cli::colab_python_bin();
        if !py.exists() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pilot_relay_dl_{}", uuid::Uuid::new_v4()));
        let cfg_dir = tmp.join("colab-cli");
        let remote_drive = tmp.join("mock_gdrive");
        let ws_dir = tmp.join("custom_ws");
        std::fs::create_dir_all(&cfg_dir).unwrap();
        std::fs::create_dir_all(&remote_drive).unwrap();
        std::fs::create_dir_all(&ws_dir).unwrap();

        let rclone_conf = cfg_dir.join("rclone.conf");
        std::fs::write(&rclone_conf, "[gdrive]\ntype = drive\n").unwrap();

        // Create a cross-platform mock rclone script backed by Python reading MOCK_RCLONE_ROOT & MOCK_RCLONE_MODE
        let rclone_py = tmp.join("mock_rclone.py");
        std::fs::write(
            &rclone_py,
            r#"import os, shutil, sys
from pathlib import Path

args = sys.argv[1:]
if "--config" in args:
    i = args.index("--config")
    args = args[:i] + args[i+2:]
clean = [a for a in args if not a.startswith("-") and a != "--"]
cmd = clean[0] if clean else ""
root = Path(os.environ["MOCK_RCLONE_ROOT"])
mode = os.environ.get("MOCK_RCLONE_MODE", "normal")

def map_remote(spec: str) -> Path:
    _, rel = spec.split(":", 1)
    return root / rel.strip("/")

if mode == "fail_all" and cmd in ("copyto", "copy"):
    sys.stderr.write("ERROR : simulated rclone network failure\n")
    sys.exit(1)

if cmd == "mkdir":
    map_remote(clean[1]).mkdir(parents=True, exist_ok=True)
    sys.exit(0)
elif cmd == "lsf":
    p = map_remote(clean[1])
    if not p.exists():
        sys.exit(0)
    for child in sorted(p.iterdir()):
        sys.stdout.write(child.name + ("/\n" if child.is_dir() else "\n"))
    sys.exit(0)
elif cmd == "copyto":
    src = map_remote(clean[1])
    dst = Path(clean[2])
    if mode == "zero_byte":
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"")
        sys.exit(0)
    if not src.is_file():
        sys.stderr.write(f"directory not found: {src.name}\n")
        sys.exit(3)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    sys.exit(0)
elif cmd == "copy":
    src = map_remote(clean[1])
    dst = Path(clean[2])
    if not src.exists():
        sys.exit(0)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    sys.exit(0)
sys.exit(0)
"#,
        )
        .unwrap();

        #[cfg(windows)]
        let mock_rclone = {
            let cmd_path = tmp.join("mock_rclone.cmd");
            std::fs::write(
                &cmd_path,
                format!("@echo off\r\n\"{}\" \"{}\" %*\r\n", py.display(), rclone_py.display()),
            )
            .unwrap();
            cmd_path
        };
        #[cfg(not(windows))]
        let mock_rclone = {
            use std::os::unix::fs::PermissionsExt;
            let sh_path = tmp.join("mock_rclone.sh");
            std::fs::write(
                &sh_path,
                format!("#!/bin/sh\nexec \"{}\" \"{}\" \"$@\"\n", py.display(), rclone_py.display()),
            )
            .unwrap();
            std::fs::set_permissions(&sh_path, std::fs::Permissions::from_mode(0o755)).unwrap();
            sh_path
        };

        std::env::set_var("MOCK_RCLONE_ROOT", &remote_drive);
        *cli::TEST_CONFIG_DIR.l() = Some(cfg_dir.clone());
        *cli::TEST_RCLONE_BIN.l() = Some(mock_rclone.clone());
        *cli::TEST_RCLONE_CONF.l() = Some(rclone_conf.clone());
        *cli::TEST_COLAB_BIN.l() = Some(tmp.join("nonexistent_colab_must_not_be_called"));

        let job_remote = remote_drive.join("colab_relay").join("dl-vm");
        std::fs::create_dir_all(&job_remote).unwrap();
        std::fs::write(job_remote.join("TRAINING_COMPLETED.flag"), "done\n").unwrap();
        assert!(
            done_flag_exists_with_conf(&mock_rclone, &rclone_conf, "gdrive:colab_relay/dl-vm").await,
            "must detect uppercase TRAINING_COMPLETED.flag"
        );

        // 1. Both download commands fail -> phase MUST be download_failed, weights_downloaded=false
        std::env::set_var("MOCK_RCLONE_MODE", "fail_all");
        let h1 = Arc::new(RelayHandle::new("dl-vm"));
        download_and_verify_final_weights(
            &h1,
            &mock_rclone,
            &rclone_conf,
            "gdrive:colab_relay/dl-vm",
            &ws_dir,
            "dl-vm",
        )
        .await;
        let s1 = h1.snapshot();
        assert!(s1.training_completed);
        assert!(!s1.weights_downloaded);
        assert_eq!(s1.phase, "download_failed");
        assert!(s1.download_error.is_some());
        assert!(s1.local_weights_path.is_none());

        // 2. rclone exits 0 but writes 0-byte file -> still download_failed!
        std::env::set_var("MOCK_RCLONE_MODE", "zero_byte");
        download_and_verify_final_weights(
            &h1,
            &mock_rclone,
            &rclone_conf,
            "gdrive:colab_relay/dl-vm",
            &ws_dir,
            "dl-vm",
        )
        .await;
        let s2 = h1.snapshot();
        assert!(!s2.weights_downloaded);
        assert_eq!(s2.phase, "download_failed");

        // 3. Re-running relay_inner (!reset_done_flag) when remote has TRAINING_COMPLETED.flag and local file is missing:
        //    must perform recovery download into bound ws_dir without allocating a VM, then verify >0 bytes -> completed!
        std::env::set_var("MOCK_RCLONE_MODE", "normal");
        std::fs::write(job_remote.join("checkpoint.pt"), b"PYTORCH_WEIGHTS_PAYLOAD").unwrap();

        let tok_file = tmp.join("token.json");
        std::fs::write(
            &tok_file,
            r#"{"token":"valid-token","expiry":"2099-01-01T00:00:00Z"}"#,
        )
        .unwrap();
        let client = crate::colab::ColabClient::new_for_test("http://127.0.0.1:9".into(), tok_file);
        let default_ws = tmp.join("default_ws_unused");
        std::fs::create_dir_all(&default_ws).unwrap();
        let app = AppState::new_with(client, default_ws.clone());

        let h_rerun = Arc::new(RelayHandle::new("dl-vm"));
        *h_rerun.workspace_dir.l() = Some(ws_dir.clone());
        h_rerun.status.l().config = RelayConfig {
            gpu_type: "CPU".into(),
            high_mem: false,
            gdrive_dir: "gdrive:colab_relay/dl-vm".into(),
            max_session_hours: 12.0,
            backup_interval_minutes: 15.0,
            total_epochs: 10,
            reset_done_flag: false,
        };
        relay_inner(app, h_rerun.clone(), CancellationToken::new())
            .await
            .expect("rerun recovery download should succeed without touching colab new");

        let s3 = h_rerun.snapshot();
        assert!(s3.training_completed);
        assert!(s3.weights_downloaded);
        assert_eq!(s3.phase, "completed");
        assert!(s3.download_error.is_none());
        let expected_ckpt = ws_dir.join("checkpoint_dl-vm.pt");
        assert_eq!(
            std::fs::read(&expected_ckpt).unwrap(),
            b"PYTORCH_WEIGHTS_PAYLOAD"
        );
        assert!(!default_ws.join("checkpoint_dl-vm.pt").exists());

        // 4. HuggingFace Trainer fallback: remote has hf_checkpoints/checkpoint-20/trainer_state.json and no checkpoint.pt
        let hf_remote = remote_drive
            .join("colab_relay")
            .join("hf-vm")
            .join("hf_checkpoints")
            .join("checkpoint-20");
        std::fs::create_dir_all(&hf_remote).unwrap();
        std::fs::write(
            remote_drive.join("colab_relay").join("hf-vm").join("TRAINING_COMPLETED.flag"),
            "done\n",
        )
        .unwrap();
        std::fs::write(hf_remote.join("trainer_state.json"), r#"{"global_step":20}"#).unwrap();
        let hf_ws = tmp.join("hf_ws");
        std::fs::create_dir_all(&hf_ws).unwrap();
        let h_hf = Arc::new(RelayHandle::new("hf-vm"));
        download_and_verify_final_weights(
            &h_hf,
            &mock_rclone,
            &rclone_conf,
            "gdrive:colab_relay/hf-vm",
            &hf_ws,
            "hf-vm",
        )
        .await;
        let s_hf = h_hf.snapshot();
        assert!(s_hf.weights_downloaded);
        assert_eq!(s_hf.phase, "completed");
        assert!(s_hf
            .local_weights_path
            .as_deref()
            .unwrap_or("")
            .contains("checkpoint-20"));

        std::env::remove_var("MOCK_RCLONE_ROOT");
        std::env::remove_var("MOCK_RCLONE_MODE");
        *cli::TEST_CONFIG_DIR.l() = None;
        *cli::TEST_RCLONE_BIN.l() = None;
        *cli::TEST_RCLONE_CONF.l() = None;
        *cli::TEST_COLAB_BIN.l() = None;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_80_legacy_bare_gdrive_checkpoint_requires_confirmation_and_migrates_with_real_rclone() {
        let _guard = cli::TEST_ENV_LOCK.lock().await;
        let real_rclone = cli::rclone_bin();
        if !real_rclone.exists() {
            return;
        }
        let tmp = std::env::temp_dir().join(format!("pilot_issue80_{}", uuid::Uuid::new_v4()));
        let drive_root = tmp.join("drive_storage");
        let legacy_dir = drive_root.join("colab_relay");
        std::fs::create_dir_all(&legacy_dir).unwrap();
        let legacy_ckpt = legacy_dir.join("checkpoint.pt");
        std::fs::write(&legacy_ckpt, b"EPOCH_7_WEIGHTS").unwrap();

        let conf = tmp.join("rclone.conf");
        let drive_norm = drive_root.to_string_lossy().replace('\\', "/");
        std::fs::write(
            &conf,
            format!("[gdrive]\ntype = alias\nremote = {drive_norm}\n"),
        )
        .unwrap();

        let h = RelayHandle::new("legacy-run");

        // 1) Unconfirmed auto-isolated path MUST abort before starting training and explain migration
        let err = check_or_migrate_legacy_bare_gdrive_with_conf(
            &real_rclone,
            &conf,
            "gdrive:colab_relay/legacy-run",
            None,
            &h,
        )
        .await
        .expect_err("unconfirmed legacy root checkpoint must block silent fresh start");
        let err_str = err.to_string();
        assert!(err_str.contains("gdrive:colab_relay"));
        assert!(err_str.contains("gdrive:colab_relay/legacy-run"));
        assert!(legacy_ckpt.exists(), "legacy checkpoint must remain untouched");

        // 2) Confirmed `migrate` moves checkpoint.pt into `gdrive:colab_relay/legacy-run/checkpoint.pt`
        check_or_migrate_legacy_bare_gdrive_with_conf(
            &real_rclone,
            &conf,
            "gdrive:colab_relay/legacy-run",
            Some("migrate"),
            &h,
        )
        .await
        .expect("confirmed migration with real rclone must succeed");
        let migrated_ckpt = legacy_dir.join("legacy-run").join("checkpoint.pt");
        assert_eq!(std::fs::read(&migrated_ckpt).unwrap(), b"EPOCH_7_WEIGHTS");
        assert!(!legacy_ckpt.exists());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
