//! 本机 colab / rclone 命令行与状态读取封装（无控制台窗口、UTF-8、工作区与历史日志解析）

use crate::LockExt;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, Instant, SystemTime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
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

#[cfg(test)]
pub(crate) static TEST_ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[cfg(test)]
pub(crate) static TEST_COLAB_BIN: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) static TEST_CONFIG_DIR: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) static TEST_ADC_PATH: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) static TEST_RCLONE_BIN: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

#[cfg(test)]
pub(crate) static TEST_RCLONE_CONF: std::sync::Mutex<Option<PathBuf>> = std::sync::Mutex::new(None);

pub fn colab_bin() -> PathBuf {
    #[cfg(test)]
    if let Some(p) = TEST_COLAB_BIN.l().clone() {
        return p;
    }
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
    #[cfg(test)]
    if let Some(p) = TEST_RCLONE_BIN.l().clone() {
        return p;
    }
    let p = local_bin("rclone");
    if p.exists() {
        return p;
    }
    which("rclone").unwrap_or(p)
}

pub fn rclone_conf_path() -> PathBuf {
    #[cfg(test)]
    if let Some(p) = TEST_RCLONE_CONF.l().clone() {
        return p;
    }
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

pub fn adc_credentials_path() -> PathBuf {
    #[cfg(test)]
    if let Some(p) = TEST_ADC_PATH.l().clone() {
        return p;
    }
    if let Ok(p) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
        let trimmed = p.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    if let Ok(cfg_dir) = std::env::var("CLOUDSDK_CONFIG") {
        let trimmed = cfg_dir.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed).join("application_default_credentials.json");
        }
    }
    if cfg!(windows) {
        if let Some(appdata) = std::env::var_os("APPDATA") {
            let win_p = PathBuf::from(appdata)
                .join("gcloud")
                .join("application_default_credentials.json");
            if win_p.exists() {
                return win_p;
            }
            let home_p = dirs::home_dir()
                .unwrap_or_default()
                .join(".config")
                .join("gcloud")
                .join("application_default_credentials.json");
            if home_p.exists() {
                return home_p;
            }
            return win_p;
        }
    }
    dirs::home_dir()
        .unwrap_or_default()
        .join(".config")
        .join("gcloud")
        .join("application_default_credentials.json")
}

pub fn is_adc_configured() -> bool {
    let p = adc_credentials_path();
    std::fs::metadata(&p)
        .map(|m| m.is_file() && m.len() > 10)
        .unwrap_or(false)
}

pub fn colab_config_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(p) = TEST_CONFIG_DIR.l().clone() {
        return p;
    }
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

pub fn session_profiles_path() -> PathBuf {
    colab_config_dir().join("pilot_session_profiles.json")
}

pub fn llm_api_keys_path() -> PathBuf {
    colab_config_dir().join("llm_api_keys.json")
}

pub fn ssh_dir() -> PathBuf {
    #[cfg(test)]
    if let Some(p) = TEST_CONFIG_DIR.l().clone() {
        return p.join("test_ssh");
    }
    dirs::home_dir().unwrap_or_default().join(".ssh")
}

pub fn ssh_config_path() -> PathBuf {
    ssh_dir().join("config")
}

pub fn pilot_ssh_private_key_path() -> PathBuf {
    ssh_dir().join("colab_pilot_ed25519")
}

pub fn pilot_ssh_public_key_path() -> PathBuf {
    ssh_dir().join("colab_pilot_ed25519.pub")
}

/// 校验会话别名可安全拼入 `~/.ssh/config` 或命令行标识符（杜绝换行与 SSH 指令注入）
pub fn is_safe_vm_alias(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

pub fn known_hosts_null_device() -> &'static str {
    // OpenSSH 6+（含 Windows OpenSSH 9.x / Git Bash / Unix）均原生支持 `none`，避免 `NUL` 与 `/dev/null` 跨环境冲突
    "none"
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct VmSshInfo {
    pub ready: bool,
    pub host_alias: String,
    pub local_port: u16,
    pub ssh_command: String,
    pub ssh_port_command: String,
    pub colab_ssh_command: String,
    pub private_key_path: String,
    pub public_key_fingerprint: String,
    pub remote_ws_url: String,
    pub user: String,
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct LlmApiAccessInfo {
    pub vm_name: String,
    pub slot: String,
    pub model_name: String,
    pub engine: String,
    pub api_base_url: String,
    pub chat_completions_url: String,
    pub models_url: String,
    pub gateway_base_url: String,
    pub api_key: String,
    pub local_port: u16,
    pub remote_port: u16,
    pub curl_example: String,
    pub python_example: String,
}

fn iso_utc_now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn atomic_write_file(path: &Path, content: &str) -> Result<()> {
    if path.is_dir() {
        return Err(anyhow!(
            "无法写入文件 `{}`：目标路径已存在且是一个目录",
            path.display()
        ));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("创建父目录 `{}` 失败", parent.display()))?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|s| s.to_str()).unwrap_or("state"),
        uuid::Uuid::new_v4().simple()
    ));
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)
            .with_context(|| format!("创建临时文件 `{}` 失败", tmp.display()))?;
        f.write_all(content.as_bytes())
            .with_context(|| format!("写入临时文件 `{}` 失败", tmp.display()))?;
    }
    #[cfg(not(unix))]
    {
        std::fs::write(&tmp, content)
            .with_context(|| format!("写入临时文件 `{}` 失败", tmp.display()))?;
    }
    if std::fs::rename(&tmp, path).is_err() {
        let write_res = std::fs::write(path, content);
        let _ = std::fs::remove_file(&tmp);
        write_res.with_context(|| format!("覆盖写入文件 `{}` 失败", path.display()))?;
    }
    Ok(())
}

/// 收紧 `~/.ssh` 目录及 `~/.ssh/config`（或私钥）权限，防止 Windows 下继承自用户目录的沙箱组
/// （如 `CodexSandboxUsers`）写权限触发 Win32-OpenSSH 报错 `Bad owner or permissions on ~/.ssh/config`。
pub fn ensure_ssh_dir_and_config_acl(ssh_dir_path: &Path, config_path: Option<&Path>) {
    #[cfg(all(windows, not(test)))]
    {
        use std::os::windows::process::CommandExt;
        static DIR_HARDENED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

        let user_principal = match (std::env::var("USERDOMAIN"), std::env::var("USERNAME")) {
            (Ok(dom), Ok(usr)) if !dom.trim().is_empty() && !usr.trim().is_empty() => {
                format!("{}\\{}", dom.trim(), usr.trim())
            }
            (_, Ok(usr)) if !usr.trim().is_empty() => usr.trim().to_string(),
            _ => return,
        };

        let run_icacls = |args: &[String]| {
            let mut cmd = std::process::Command::new("icacls");
            cmd.args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(CREATE_NO_WINDOW);
            let _ = cmd.status();
        };

        if ssh_dir_path.exists() {
            DIR_HARDENED.get_or_init(|| {
                run_icacls(&[
                    ssh_dir_path.to_string_lossy().to_string(),
                    "/inheritance:r".into(),
                    "/grant:r".into(),
                    "*S-1-5-18:(OI)(CI)(F)".into(),
                    "*S-1-5-32-544:(OI)(CI)(F)".into(),
                    format!("{user_principal}:(OI)(CI)(F)"),
                ]);
            });
        }

        if let Some(cfg) = config_path {
            if cfg.exists() {
                run_icacls(&[
                    cfg.to_string_lossy().to_string(),
                    "/inheritance:r".into(),
                    "/grant:r".into(),
                    "*S-1-5-18:(F)".into(),
                    "*S-1-5-32-544:(F)".into(),
                    format!("{user_principal}:(F)"),
                ]);
            }
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if ssh_dir_path.exists() {
            let _ = std::fs::set_permissions(ssh_dir_path, std::fs::Permissions::from_mode(0o700));
        }
        if let Some(cfg) = config_path {
            if cfg.exists() {
                let _ = std::fs::set_permissions(cfg, std::fs::Permissions::from_mode(0o600));
            }
        }
    }

    #[cfg(all(windows, test))]
    {
        let _ = (ssh_dir_path, config_path);
    }
}

/// 生成供本地 SSH 隧道免密登录 Colab 容器使用的 Ed25519 密钥对（缓存结果，避免高频轮询重复检查）
pub fn ensure_ssh_keypair() -> Result<(PathBuf, String)> {
    static KEY_CACHE: std::sync::Mutex<Option<(PathBuf, PathBuf, String)>> =
        std::sync::Mutex::new(None);

    let priv_key = pilot_ssh_private_key_path();
    let pub_key = pilot_ssh_public_key_path();
    {
        let guard = KEY_CACHE.l();
        if let Some((cached_priv, cached_pub, cached_pub_raw)) = guard.as_ref() {
            if cached_priv == &priv_key && cached_priv.exists() && cached_pub.exists() {
                return Ok((priv_key, cached_pub_raw.clone()));
            }
        }
    }

    let dir = ssh_dir();
    let _ = std::fs::create_dir_all(&dir);
    let cfg_path = ssh_config_path();
    ensure_ssh_dir_and_config_acl(&dir, cfg_path.exists().then_some(cfg_path.as_path()));
    if !priv_key.exists() || !pub_key.exists() {
        let _ = std::fs::remove_file(&priv_key);
        let _ = std::fs::remove_file(&pub_key);
        let keygen = which("ssh-keygen").unwrap_or_else(|| PathBuf::from("ssh-keygen"));
        let mut cmd = std::process::Command::new(keygen);
        cmd.args([
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "colab-pilot",
            "-f",
            &priv_key.to_string_lossy(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let st = cmd.status().context("调用 ssh-keygen 生成 SSH 密钥失败")?;
        if !st.success() {
            return Err(anyhow!("ssh-keygen 生成 SSH 密钥返回非零状态"));
        }
        ensure_ssh_dir_and_config_acl(&dir, Some(&priv_key));
    }
    let pub_raw = std::fs::read_to_string(&pub_key)
        .context("读取 colab_pilot_ed25519.pub 失败")?
        .trim()
        .to_string();
    *KEY_CACHE.l() = Some((priv_key.clone(), pub_key, pub_raw.clone()));
    Ok((priv_key, pub_raw))
}

const SSH_CONFIG_BEGIN: &str = "# >>> colab-pilot managed hosts >>>";
const SSH_CONFIG_END: &str = "# <<< colab-pilot managed hosts <<<";

/// 将当前活跃的本地 SSH 桥接端口同步写入 `~/.ssh/config`（使用 `UserKnownHostsFile none`，自动收紧 ACL，内容未变时不写盘）
pub fn sync_ssh_config_hosts(entries: &[(String, u16)]) {
    static LAST_SYNCED_BLOCK: std::sync::Mutex<Option<(PathBuf, String)>> =
        std::sync::Mutex::new(None);

    let Ok((priv_key, _)) = ensure_ssh_keypair() else {
        return;
    };
    let dir = ssh_dir();
    let cfg_path = ssh_config_path();
    ensure_ssh_dir_and_config_acl(&dir, None);
    let priv_str = priv_key.to_string_lossy().replace('\\', "/");
    let null_hosts = known_hosts_null_device();

    let mut sorted: Vec<(String, u16)> = entries
        .iter()
        .filter(|(vm_name, _)| is_safe_vm_alias(vm_name))
        .cloned()
        .collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));

    let mut block = String::new();
    if !sorted.is_empty() {
        block.push_str(SSH_CONFIG_BEGIN);
        block.push('\n');
        for (vm_name, port) in &sorted {
            let alias = format!("colab-{vm_name}");
            block.push_str(&format!(
                "Host {alias}\n    HostName 127.0.0.1\n    Port {port}\n    User root\n    IdentityFile \"{priv_str}\"\n    IdentitiesOnly yes\n    StrictHostKeyChecking no\n    UserKnownHostsFile {null_hosts}\n    ServerAliveInterval 20\n    ServerAliveCountMax 3\n\n"
            ));
        }
        block.push_str(SSH_CONFIG_END);
        block.push('\n');
    }

    {
        let guard = LAST_SYNCED_BLOCK.l();
        if let Some((last_path, last_block)) = guard.as_ref() {
            if last_path == &cfg_path && last_block == &block && (block.is_empty() || cfg_path.exists()) {
                return;
            }
        }
    }

    let existing = std::fs::read_to_string(&cfg_path).unwrap_or_default();
    let cleaned = if let (Some(s_idx), Some(e_idx)) =
        (existing.find(SSH_CONFIG_BEGIN), existing.find(SSH_CONFIG_END))
    {
        if s_idx <= e_idx {
            let after = e_idx + SSH_CONFIG_END.len();
            let tail = existing[after..].trim_start_matches(['\r', '\n']);
            let head = existing[..s_idx].trim_end();
            if head.is_empty() {
                tail.to_string()
            } else if tail.is_empty() {
                format!("{head}\n")
            } else {
                format!("{head}\n\n{tail}")
            }
        } else {
            existing.clone()
        }
    } else {
        existing.clone()
    };

    let merged = if block.is_empty() {
        cleaned
    } else if cleaned.trim().is_empty() {
        block.clone()
    } else {
        format!("{}\n\n{}", cleaned.trim_end(), block)
    };

    if merged != existing {
        let _ = atomic_write_file(&cfg_path, &merged);
        ensure_ssh_dir_and_config_acl(&dir, Some(&cfg_path));
    }
    *LAST_SYNCED_BLOCK.l() = Some((cfg_path, block));
}

static SESSIONS_IO_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 将服务端在线实例同步/去重至 `sessions.json`，兼容 `google-colab-cli` v0.7/v0.8 会话字段规范：
/// - 若 `creating_active == true`（`colab new` 正在分配中），跳过写入新的 `orphan-*` 条目，避免与 CLI 写盘竞态或产生重复孤儿；
/// - 自动清理与已命名会话共享同一 `endpoint` 的陈旧 `orphan-*` 条目，以及已不在 `active_endpoints` 中的过期 `orphan-*` 条目。
#[allow(clippy::too_many_arguments)]
pub fn sync_sessions_with_assignments(
    items: &[(
        String, // session_name
        String, // endpoint
        String, // proxy_url
        String, // proxy_token
        i64,    // token_expires_in_seconds
        String, // accelerator
        String, // variant
        bool,   // high_ram
    )],
    creating_active: bool,
) {
    let _lock = SESSIONS_IO_LOCK.l();
    let path = sessions_path();
    let mut map: HashMap<String, Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let mut dirty = false;

    let active_eps: std::collections::HashSet<&str> =
        items.iter().map(|it| it.1.as_str()).collect();
    let named_eps: std::collections::HashSet<String> = map
        .iter()
        .filter(|(k, _)| !k.starts_with("orphan-"))
        .filter_map(|(_, v)| v.get("endpoint").and_then(|x| x.as_str()).map(|s| s.to_string()))
        .chain(
            items
                .iter()
                .filter(|it| !it.0.starts_with("orphan-"))
                .map(|it| it.1.clone()),
        )
        .collect();

    // 清理已被正式命名会话接管的重复孤儿条目，以及云端已下线的历史孤儿条目
    let stale_orphans: Vec<String> = map
        .iter()
        .filter_map(|(k, v)| {
            if !k.starts_with("orphan-") {
                return None;
            }
            let ep = v.get("endpoint").and_then(|x| x.as_str()).unwrap_or("");
            if ep.is_empty() || named_eps.contains(ep) || !active_eps.contains(ep) {
                Some(k.clone())
            } else {
                None
            }
        })
        .collect();
    for k in stale_orphans {
        map.remove(&k);
        dirty = true;
    }

    for (session_name, endpoint, proxy_url, proxy_token, exp_secs, accelerator, variant, high_ram) in
        items
    {
        if !is_safe_vm_alias(session_name) {
            continue;
        }
        if session_name.starts_with("orphan-") && (creating_active || named_eps.contains(endpoint))
        {
            continue;
        }
        let norm_acc = {
            let u = accelerator.trim().to_uppercase();
            if u.is_empty() || u == "CPU" || u == "NONE" {
                "NONE".to_string()
            } else {
                u
            }
        };
        let norm_var = {
            let u = variant.trim().to_uppercase();
            if u.is_empty() || u == "CPU" || u == "NONE" || u == "DEFAULT" {
                "DEFAULT".to_string()
            } else {
                u
            }
        };
        let shape = if *high_ram { "HIGH_RAM" } else { "STANDARD" };
        let ttl_secs = if *exp_secs > 0 {
            (*exp_secs).min(86400)
        } else {
            3600
        };
        let fresh_exp_iso = (chrono::Utc::now() + chrono::Duration::seconds(ttl_secs))
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);

        if let Some(existing) = map.get_mut(session_name) {
            if let Some(obj) = existing.as_object_mut() {
                let cur_name = obj.get("name").and_then(|x| x.as_str()).unwrap_or("");
                let cur_ep = obj.get("endpoint").and_then(|x| x.as_str()).unwrap_or("");
                let cur_url = obj.get("url").and_then(|x| x.as_str()).unwrap_or("");
                let cur_tok = obj.get("token").and_then(|x| x.as_str()).unwrap_or("");
                let has_exp = obj
                    .get("token_expires_at")
                    .and_then(|x| x.as_str())
                    .map(|s| !s.is_empty())
                    .unwrap_or(false);
                let has_legacy = obj.contains_key("proxy_url")
                    || obj.contains_key("proxy_token")
                    || obj.contains_key("auth_mode")
                    || obj.contains_key("created_at");
                if cur_name != session_name
                    || cur_ep != endpoint
                    || cur_url != proxy_url
                    || cur_tok != proxy_token
                    || !has_exp
                    || has_legacy
                {
                    obj.remove("proxy_url");
                    obj.remove("proxy_token");
                    obj.remove("auth_mode");
                    obj.remove("created_at");
                    obj.insert("name".into(), Value::String(session_name.clone()));
                    obj.insert("endpoint".into(), Value::String(endpoint.clone()));
                    obj.insert("url".into(), Value::String(proxy_url.clone()));
                    obj.insert("token".into(), Value::String(proxy_token.clone()));
                    obj.insert("token_expires_at".into(), Value::String(fresh_exp_iso));
                    obj.insert("accelerator".into(), Value::String(norm_acc));
                    obj.insert("variant".into(), Value::String(norm_var));
                    obj.insert("machine_shape".into(), Value::String(shape.into()));
                    obj.entry("kernel_id").or_insert(Value::Null);
                    obj.entry("session_id").or_insert(Value::Null);
                    obj.entry("last_execution").or_insert(Value::Null);
                    obj.entry("running").or_insert(Value::Null);
                    dirty = true;
                }
            }
        } else {
            map.insert(
                session_name.clone(),
                serde_json::json!({
                    "name": session_name,
                    "token": proxy_token,
                    "url": proxy_url,
                    "endpoint": endpoint,
                    "token_expires_at": fresh_exp_iso,
                    "variant": norm_var,
                    "accelerator": norm_acc,
                    "machine_shape": shape,
                    "kernel_id": Value::Null,
                    "session_id": Value::Null,
                    "last_execution": Value::Null,
                    "running": Value::Null,
                }),
            );
            dirty = true;
        }
    }

    if dirty {
        if let Ok(raw) = serde_json::to_string_pretty(&map) {
            let _ = atomic_write_file(&path, &raw);
        }
    }
}

/// 将静默探针或流式桥接首次建立的 `kernel_id` 回写到 `sessions.json`，防止后续调用反复新建泄露 Kernel
pub fn update_session_kernel_id(endpoint: &str, kernel_id: &str) {
    let kid = kernel_id.trim();
    if endpoint.is_empty() || kid.is_empty() {
        return;
    }
    let _lock = SESSIONS_IO_LOCK.l();
    let path = sessions_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return;
    };
    let Ok(mut map) = serde_json::from_str::<HashMap<String, Value>>(&raw) else {
        return;
    };
    let mut dirty = false;
    for val in map.values_mut() {
        if let Some(obj) = val.as_object_mut() {
            if obj.get("endpoint").and_then(|x| x.as_str()) == Some(endpoint) {
                let existing = obj.get("kernel_id").and_then(|x| x.as_str()).unwrap_or("");
                if existing.is_empty() {
                    obj.insert("kernel_id".into(), Value::String(kid.to_string()));
                    dirty = true;
                }
            }
        }
    }
    if dirty {
        if let Ok(out) = serde_json::to_string_pretty(&map) {
            let _ = atomic_write_file(&path, &out);
        }
    }
}

static WORKSPACES_IO_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn has_workspace_entry(session_name: &str) -> bool {
    let _lock = WORKSPACES_IO_LOCK.l();
    std::fs::read_to_string(workspaces_registry_path())
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, Value>>(&s).ok())
        .map(|m| m.contains_key(session_name))
        .unwrap_or(false)
}

/// 持久化注册实例所属的逻辑工作区与描述（写入 `~/.config/colab-cli/workspaces.json`）
pub fn register_workspace_entry(
    session_name: &str,
    workspace_name: &str,
    workspace_path: Option<&str>,
    description: Option<&str>,
) {
    let _lock = WORKSPACES_IO_LOCK.l();
    let path = workspaces_registry_path();
    let mut map: HashMap<String, Value> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    map.insert(
        session_name.to_string(),
        serde_json::json!({
            "workspace_name": workspace_name,
            "workspace_path": workspace_path,
            "description": description,
            "updated_at": iso_utc_now(),
        }),
    );
    if let Ok(raw) = serde_json::to_string_pretty(&map) {
        let _ = atomic_write_file(&path, &raw);
    }
}

/// 当实例创建失败且会话表中无对应会话时，回滚移除刚预注册的 `workspaces.json` 条目
pub fn remove_workspace_entry(session_name: &str) {
    if session_name.trim().is_empty() {
        return;
    }
    let _lock = WORKSPACES_IO_LOCK.l();
    let path = workspaces_registry_path();
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return;
    };
    let Ok(mut map) = serde_json::from_str::<HashMap<String, Value>>(&raw) else {
        return;
    };
    if map.remove(session_name).is_some() {
        if let Ok(out) = serde_json::to_string_pretty(&map) {
            let _ = atomic_write_file(&path, &out);
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default, Debug)]
pub struct SessionProfile {
    #[serde(default)]
    pub boot_mode: String,
    #[serde(default)]
    pub accelerator: String,
    #[serde(default)]
    pub high_mem: bool,
    #[serde(default)]
    pub workspace_path: Option<String>,
    #[serde(default)]
    pub auth_mode: Option<String>,
    #[serde(default)]
    pub llm_config: Option<crate::CreateVmLlmConfig>,
    #[serde(default)]
    pub relay_config: Option<crate::relay::RelayConfig>,
    #[serde(default)]
    pub updated_at: Option<String>,
}

static PROFILES_IO_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn normalize_auth_mode(auth_mode: Option<&str>) -> Option<String> {
    match auth_mode.map(str::trim).unwrap_or("").to_ascii_lowercase().as_str() {
        "adc" => Some("adc".into()),
        "oauth2" => Some("oauth2".into()),
        _ => None,
    }
}

pub fn read_session_profiles() -> HashMap<String, SessionProfile> {
    let _lock = PROFILES_IO_LOCK.l();
    std::fs::read_to_string(session_profiles_path())
        .ok()
        .and_then(|s| serde_json::from_str::<HashMap<String, SessionProfile>>(&s).ok())
        .unwrap_or_default()
}

pub fn get_session_profile(session_name: &str) -> Option<SessionProfile> {
    read_session_profiles().remove(session_name)
}

pub fn save_session_profile(session_name: &str, profile: &SessionProfile) -> Result<()> {
    if session_name.trim().is_empty() {
        return Ok(());
    }
    let _lock = PROFILES_IO_LOCK.l();
    let path = session_profiles_path();
    let mut map: HashMap<String, SessionProfile> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let mut entry = profile.clone();
    if entry.updated_at.is_none() {
        entry.updated_at = Some(iso_utc_now());
    }
    map.insert(session_name.to_string(), entry);
    let raw = serde_json::to_string_pretty(&map)?;
    atomic_write_file(&path, &raw)?;
    invalidate_recent_sessions_cache();
    Ok(())
}

pub fn get_session_auth_mode(session_name: &str) -> Option<String> {
    get_session_profile(session_name)
        .and_then(|p| normalize_auth_mode(p.auth_mode.as_deref()))
}

pub fn build_colab_subcommand_args(auth_mode: Option<&str>, sub_args: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(sub_args.len() + 2);
    if matches!(normalize_auth_mode(auth_mode).as_deref(), Some("adc")) {
        out.push("--auth".into());
        out.push("adc".into());
    }
    out.extend_from_slice(sub_args);
    out
}

pub fn colab_args_for_session(session_name: &str, sub_args: &[String]) -> Vec<String> {
    let mode = get_session_auth_mode(session_name);
    build_colab_subcommand_args(mode.as_deref(), sub_args)
}

/// 查找某个会话已绑定的本地工作区目录（依次查询 workspaces.json、pilot_session_profiles.json、sessions.json）
pub fn lookup_bound_workspace_path(session_name: &str) -> Option<PathBuf> {
    let name = session_name.trim();
    if name.is_empty() {
        return None;
    }
    if let Ok(raw) = std::fs::read_to_string(workspaces_registry_path()) {
        if let Ok(map) = serde_json::from_str::<HashMap<String, Value>>(&raw) {
            if let Some(p) = map
                .get(name)
                .and_then(|e| e.get("workspace_path").or_else(|| e.get("path")))
                .and_then(|x| x.as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                return Some(PathBuf::from(p));
            }
        }
    }
    if let Some(prof) = get_session_profile(name) {
        if let Some(p) = prof
            .workspace_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Some(PathBuf::from(p));
        }
    }
    let sessions = read_sessions();
    if let Some(sv) = sessions.get(name) {
        if let Some(wp) = sv
            .get("_workspace_path")
            .and_then(|x| x.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            return Some(PathBuf::from(wp));
        }
    }
    None
}

fn generate_random_hex(byte_len: usize) -> String {
    let mut out = String::with_capacity(byte_len * 2 + 32);
    while out.len() < byte_len * 2 {
        out.push_str(&uuid::Uuid::new_v4().simple().to_string());
    }
    out.truncate(byte_len * 2);
    out
}

static API_KEYS_IO_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn read_llm_api_keys_root() -> Value {
    std::fs::read_to_string(llm_api_keys_path())
        .ok()
        .and_then(|s| serde_json::from_str::<Value>(&s).ok())
        .filter(|v| v.is_object())
        .unwrap_or_else(|| serde_json::json!({ "vms": {} }))
}

pub fn lookup_vm_api_key_meta(vm_name: &str) -> Option<(String, Option<u16>)> {
    let root = read_llm_api_keys_root();
    if let Some(entry) = root.get("vms").and_then(|v| v.get(vm_name)) {
        if let Some(k) = entry.get("api_key").and_then(|x| x.as_str()).filter(|s| !s.trim().is_empty()) {
            let lp = entry.get("local_port").and_then(|x| x.as_u64()).map(|n| n as u16);
            return Some((k.to_string(), lp));
        }
    }
    root.get(vm_name)
        .and_then(|x| x.as_str())
        .filter(|s| !s.trim().is_empty())
        .map(|k| (k.to_string(), None))
}

pub fn has_registered_llm_api_key(vm_name: &str) -> bool {
    if vm_name.starts_with("orphan-") {
        return false;
    }
    lookup_vm_api_key_meta(vm_name).is_some()
}

pub fn remove_llm_api_key_entry(vm_name: &str) {
    let _lock = API_KEYS_IO_LOCK.l();
    let path = llm_api_keys_path();
    let mut root = read_llm_api_keys_root();
    let mut dirty = false;
    if let Some(vms_obj) = root.get_mut("vms").and_then(|v| v.as_object_mut()) {
        if vms_obj.remove(vm_name).is_some() {
            dirty = true;
        }
    }
    if let Some(root_obj) = root.as_object_mut() {
        if root_obj.remove(vm_name).is_some() {
            dirty = true;
        }
    }
    if dirty {
        if let Ok(raw) = serde_json::to_string_pretty(&root) {
            let _ = atomic_write_file(&path, &raw);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn try_upsert_llm_api_key_record(
    vm_name: &str,
    slot: &str,
    engine: &str,
    force_new_key: bool,
    custom_key: Option<&str>,
    preferred_local_port: u16,
    remote_port: u16,
    model_name: &str,
    strict_persist: bool,
) -> Result<LlmApiAccessInfo> {
    let _lock = API_KEYS_IO_LOCK.l();
    let mut root = read_llm_api_keys_root();
    if root.get("vms").and_then(|v| v.as_object()).is_none() {
        if let Some(obj) = root.as_object_mut() {
            obj.insert("vms".into(), serde_json::json!({}));
        }
    }
    let clean_vm: String = vm_name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '-' })
        .collect();
    let clean_vm = if clean_vm.is_empty() { "vm".to_string() } else { clean_vm };

    let slot_norm = {
        let s = slot.trim().to_lowercase();
        if s.is_empty() { "standalone".to_string() } else { s }
    };
    let eng_norm = {
        let e = engine.trim();
        if e.is_empty() { "builtin".to_string() } else { e.to_string() }
    };
    let model_norm = {
        let m = model_name.trim();
        if m.is_empty() { "default".to_string() } else { m.to_string() }
    };

    let mut dirty = strict_persist;
    let (existing_key, existing_port) = root
        .get("vms")
        .and_then(|v| v.get(vm_name))
        .map(|entry| {
            let k = entry
                .get("api_key")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let p = entry
                .get("local_port")
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as u16;
            (k, p)
        })
        .unwrap_or_default();

    let key_val = if let Some(ck) = custom_key.map(str::trim).filter(|s| s.len() >= 8) {
        if ck != existing_key {
            dirty = true;
        }
        ck.to_string()
    } else if !force_new_key && !existing_key.is_empty() {
        existing_key.clone()
    } else {
        dirty = true;
        format!("sk-colab-{clean_vm}-{}", generate_random_hex(8))
    };

    let local_port = if preferred_local_port >= 18081 {
        preferred_local_port
    } else if slot_norm == "blue" && vm_name.ends_with("-blue") {
        18081
    } else if slot_norm == "green" && vm_name.ends_with("-green") {
        18082
    } else if existing_port >= 18083 {
        existing_port
    } else {
        let used: std::collections::HashSet<u16> = root
            .get("vms")
            .and_then(|v| v.as_object())
            .map(|m| {
                m.iter()
                    .filter(|(k, _)| k.as_str() != vm_name)
                    .filter_map(|(_, val)| val.get("local_port").and_then(|x| x.as_u64()))
                    .map(|p| p as u16)
                    .collect()
            })
            .unwrap_or_default();
        let mut cand = 18083u16;
        while used.contains(&cand) {
            cand = cand.saturating_add(1);
        }
        dirty = true;
        cand
    };

    if existing_port != local_port {
        dirty = true;
    }

    let api_base_url = format!("http://127.0.0.1:{local_port}/v1");
    let chat_completions_url = format!("{api_base_url}/chat/completions");
    let models_url = format!("{api_base_url}/models");
    let gateway_base_url = "http://127.0.0.1:18080/v1".to_string();

    if root
        .get("gateway_api_key")
        .and_then(|x| x.as_str())
        .map(|s| s.trim().is_empty())
        .unwrap_or(true)
    {
        if let Some(obj) = root.as_object_mut() {
            obj.insert(
                "gateway_api_key".into(),
                Value::String(format!("sk-colab-gateway-{}", generate_random_hex(8))),
            );
            dirty = true;
        }
    }

    if dirty {
        if let Some(vms_obj) = root.get_mut("vms").and_then(|v| v.as_object_mut()) {
            vms_obj.insert(
                vm_name.to_string(),
                serde_json::json!({
                    "vm_name": vm_name,
                    "slot": slot_norm,
                    "api_key": key_val,
                    "local_port": local_port,
                    "remote_port": remote_port,
                    "model_name": model_norm,
                    "engine": eng_norm,
                    "api_base_url": api_base_url,
                    "gateway_base_url": gateway_base_url,
                }),
            );
        }
        let path = llm_api_keys_path();
        let raw = serde_json::to_string_pretty(&root).context("序列化 llm_api_keys.json 失败")?;
        if strict_persist {
            atomic_write_file(&path, &raw)
                .with_context(|| format!("持久化写入 `{}` 失败", path.display()))?;
            let read_back_raw = std::fs::read_to_string(&path)
                .with_context(|| format!("回读校验 `{}` 失败", path.display()))?;
            let read_back_val: Value = serde_json::from_str(&read_back_raw)
                .with_context(|| format!("解析回读 `{}` JSON 失败", path.display()))?;
            let persisted_key = read_back_val
                .get("vms")
                .and_then(|v| v.get(vm_name))
                .and_then(|e| e.get("api_key"))
                .and_then(|x| x.as_str());
            if persisted_key != Some(key_val.as_str()) {
                return Err(anyhow!(
                    "API Key 落盘回读校验失败：`{}` 中未读取到刚写入的新密钥",
                    path.display()
                ));
            }
        } else {
            let _ = atomic_write_file(&path, &raw);
        }
    }

    let curl_example = format!(
        "curl {chat_completions_url} -H \"Content-Type: application/json\" -H \"Authorization: Bearer {key_val}\" -d '{{\"model\":\"{model_norm}\",\"messages\":[{{\"role\":\"user\",\"content\":\"你好\"}}]}}'"
    );
    let python_example = format!(
        "from openai import OpenAI\nclient = OpenAI(base_url=\"{api_base_url}\", api_key=\"{key_val}\")\nresp = client.chat.completions.create(model=\"{model_norm}\", messages=[{{\"role\": \"user\", \"content\": \"你好\"}}])\nprint(resp.choices[0].message.content)"
    );
    Ok(LlmApiAccessInfo {
        vm_name: vm_name.to_string(),
        slot: slot_norm,
        model_name: model_norm,
        engine: eng_norm,
        api_base_url,
        chat_completions_url,
        models_url,
        gateway_base_url,
        api_key: key_val,
        local_port,
        remote_port,
        curl_example,
        python_example,
    })
}

/// 获取或自动签发某台 LLM 实例的本机 OpenAI 兼容 API Key 与调用示例
pub fn resolve_or_create_llm_api_info(
    vm_name: &str,
    slot: &str,
    engine: &str,
    local_port: u16,
    remote_port: u16,
    model_name: &str,
) -> LlmApiAccessInfo {
    try_upsert_llm_api_key_record(
        vm_name,
        slot,
        engine,
        false,
        None,
        local_port,
        remote_port,
        model_name,
        false,
    )
    .unwrap_or_else(|_| LlmApiAccessInfo {
        vm_name: vm_name.to_string(),
        slot: slot.to_string(),
        model_name: model_name.to_string(),
        engine: engine.to_string(),
        local_port,
        remote_port,
        ..Default::default()
    })
}

/// 轮换或自定义某台 LLM 实例的 API Key（强制落盘并回读校验，写入失败返回 Err）
pub fn rotate_llm_api_key_entry(
    vm_name: &str,
    slot: &str,
    engine: &str,
    custom_key: Option<&str>,
    local_port: u16,
    remote_port: u16,
    model_name: &str,
) -> Result<LlmApiAccessInfo> {
    try_upsert_llm_api_key_record(
        vm_name,
        slot,
        engine,
        true,
        custom_key,
        local_port,
        remote_port,
        model_name,
        true,
    )
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
    #[serde(default)]
    pub boot_mode: Option<String>,
    #[serde(default)]
    pub llm_config: Option<crate::CreateVmLlmConfig>,
    #[serde(default)]
    pub relay_config: Option<crate::relay::RelayConfig>,
    #[serde(default)]
    pub profile_saved: bool,
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

/// 读取指定会话的 colab-cli 结构化历史事件（过滤内部探针启动事件，按 mtime + 文件长度缓存避免高频重复解析）
pub fn read_session_events(session_name: &str, max_events: usize) -> (Vec<CliEventView>, Vec<String>) {
    type EventCacheEntry = (Option<SystemTime>, u64, Vec<CliEventView>, Vec<String>);
    static EVENT_CACHE: std::sync::Mutex<Option<HashMap<PathBuf, EventCacheEntry>>> =
        std::sync::Mutex::new(None);

    let path = history_dir().join(format!("{session_name}.jsonl"));
    let meta = match std::fs::metadata(&path) {
        Ok(m) => m,
        Err(_) => return (vec![], vec![]),
    };
    let mtime = meta.modified().ok();
    let flen = meta.len();

    {
        let guard = EVENT_CACHE.l();
        if let Some(map) = guard.as_ref() {
            if let Some((c_mtime, c_len, cached_events, cached_paths)) = map.get(&path) {
                if *c_mtime == mtime && *c_len == flen {
                    let total = cached_events.len();
                    let tail = if total > max_events {
                        cached_events[total - max_events..].iter().rev().cloned().collect()
                    } else {
                        cached_events.iter().rev().cloned().collect()
                    };
                    return (tail, cached_paths.clone());
                }
            }
        }
    }

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
                format!("终端会话 · {session_name}"),
                None,
                None,
                false,
            ),
            "ssh_started" => (
                "SSH 连接".to_string(),
                format!("colab ssh -s {session_name}"),
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
        events[len - max_events..].iter().rev().cloned().collect()
    } else {
        events.iter().rev().cloned().collect()
    };

    {
        let mut guard = EVENT_CACHE.l();
        let map = guard.get_or_insert_with(HashMap::new);
        if map.len() > 128 {
            map.clear();
        }
        map.insert(path, (mtime, flen, events, discovered_paths.clone()));
    }

    (tail, discovered_paths)
}

/// 判断桌面目录名与 Colab 会话名是否满足安全前缀匹配（至少 3 字符且具备 `-`/`_` 词边界）
pub(crate) fn is_workspace_prefix_match(dir_name: &str, session_name: &str) -> bool {
    let dn_low = dir_name.trim().to_lowercase();
    let sn_low = session_name.trim().to_lowercase();
    if dn_low.is_empty() || sn_low.is_empty() {
        return false;
    }
    if dn_low == sn_low {
        return true;
    }
    if dn_low.chars().count() < 3 || sn_low.chars().count() < 3 {
        return false;
    }
    let has_boundary = |longer: &str, shorter: &str| -> bool {
        longer
            .strip_prefix(shorter)
            .and_then(|rest| rest.chars().next())
            .map(|ch| ch == '-' || ch == '_')
            .unwrap_or(false)
    };
    has_boundary(&sn_low, &dn_low) || has_boundary(&dn_low, &sn_low)
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

    // 1b. 检查 pilot_session_profiles.json 中持久化的工作区路径
    if let Some(prof) = get_session_profile(session_name) {
        if let Some(wp) = prof
            .workspace_path
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let name = Path::new(wp)
                .file_name()
                .and_then(|s| s.to_str())
                .filter(|s| !s.is_empty())
                .unwrap_or(session_name)
                .to_string();
            return WorkspaceInfo {
                name,
                path: Some(wp.to_string()),
                remote_cwd,
                source: "会话配置档案 (pilot_session_profiles.json)".into(),
                active_script,
                description: None,
            };
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
                if is_workspace_prefix_match(dn, session_name) {
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

static RECENT_SESSIONS_CACHE: std::sync::Mutex<
    Option<(Instant, PathBuf, Vec<String>, usize, Vec<RecentSessionView>)>,
> = std::sync::Mutex::new(None);

pub fn invalidate_recent_sessions_cache() {
    *RECENT_SESSIONS_CACHE.l() = None;
}

/// 读取最近结束的历史会话摘要（用于仪表盘展示近期工作区执行记录）
/// 结果按 (配置目录, 活跃会话集合, limit) 缓存 30 秒；活跃实例增减或配置更新时立即重新计算
pub fn read_recent_sessions(active_names: &[String], limit: usize) -> Vec<RecentSessionView> {
    let cfg_dir = colab_config_dir();
    let mut key: Vec<String> = active_names.to_vec();
    key.sort();
    if let Some((t, cdir, k, l, v)) = RECENT_SESSIONS_CACHE.l().as_ref() {
        if t.elapsed() < FS_CACHE_TTL && *cdir == cfg_dir && *k == key && *l == limit {
            return v.clone();
        }
    }
    let list = read_recent_sessions_uncached(active_names, limit);
    *RECENT_SESSIONS_CACHE.l() = Some((Instant::now(), cfg_dir, key, limit, list.clone()));
    list
}

fn read_recent_sessions_uncached(active_names: &[String], limit: usize) -> Vec<RecentSessionView> {
    let dir = history_dir();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let saved_profiles = read_session_profiles();
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
        let mut saw_created_event = false;
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
                    saw_created_event = true;
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
        let mut ws = resolve_workspace_for_vm(stem, ep.as_deref(), None, None, &discovered_paths);
        let (boot_mode, llm_config, relay_config, profile_saved) =
            if let Some(prof) = saved_profiles.get(stem) {
                if ws.path.is_none() && prof.workspace_path.is_some() {
                    ws.path = prof.workspace_path.clone();
                    if let Some(ref wp) = ws.path {
                        if let Some(dn) = Path::new(wp)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .filter(|s| !s.is_empty())
                        {
                            ws.name = dn.to_string();
                        }
                    }
                }
                if !saw_created_event && !prof.accelerator.trim().is_empty() {
                    let pa = prof.accelerator.trim().to_uppercase();
                    acc = if pa == "NONE" { "CPU".into() } else { pa };
                    shape = if prof.high_mem {
                        "HIGH_RAM".into()
                    } else {
                        "STANDARD".into()
                    };
                }
                (
                    Some(prof.boot_mode.clone()),
                    prof.llm_config.clone(),
                    prof.relay_config.clone(),
                    true,
                )
            } else {
                (None, None, None, false)
            };
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
            boot_mode,
            llm_config,
            relay_config,
            profile_saved,
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

#[cfg(windows)]
fn terminate_process_tree_win32(root_pid: u32) -> bool {
    #[repr(C)]
    struct ProcessEntry32W {
        dw_size: u32,
        cnt_usage: u32,
        th32_process_id: u32,
        th32_default_heap_id: usize,
        th32_module_id: u32,
        cnt_threads: u32,
        th32_parent_process_id: u32,
        pc_pri_class_base: i32,
        dw_flags: u32,
        sz_exe_file: [u16; 260],
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn CreateToolhelp32Snapshot(dw_flags: u32, th32_process_id: u32) -> isize;
        fn Process32FirstW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn Process32NextW(h_snapshot: isize, lppe: *mut ProcessEntry32W) -> i32;
        fn OpenProcess(desired_access: u32, inherit_handle: i32, process_id: u32) -> isize;
        fn TerminateProcess(h_process: isize, u_exit_code: u32) -> i32;
        fn GetExitCodeProcess(h_process: isize, lp_exit_code: *mut u32) -> i32;
        fn GetLastError() -> u32;
        fn CloseHandle(h_object: isize) -> i32;
    }

    const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;
    const INVALID_HANDLE_VALUE: isize = -1;
    const PROCESS_TERMINATE: u32 = 0x0001;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;
    const STILL_ACTIVE: u32 = 259;
    const ERROR_INVALID_PARAMETER: u32 = 87;

    let self_pid = std::process::id();
    if root_pid == 0 || root_pid == self_pid {
        return false;
    }

    unsafe {
        let snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if snap == INVALID_HANDLE_VALUE || snap == 0 {
            return false;
        }
        let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(256);
        let mut entry: ProcessEntry32W = std::mem::zeroed();
        entry.dw_size = std::mem::size_of::<ProcessEntry32W>() as u32;
        if Process32FirstW(snap, &mut entry) != 0 {
            loop {
                pairs.push((entry.th32_process_id, entry.th32_parent_process_id));
                if Process32NextW(snap, &mut entry) == 0 {
                    break;
                }
            }
        }
        let _ = CloseHandle(snap);

        let root_present = pairs.iter().any(|&(pid, _)| pid == root_pid);
        if !root_present {
            return true;
        }

        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        visited.insert(root_pid);
        order.push(root_pid);
        let mut idx = 0;
        while idx < order.len() {
            let cur = order[idx];
            idx += 1;
            for &(child_pid, parent_pid) in &pairs {
                if parent_pid == cur
                    && child_pid != 0
                    && child_pid != self_pid
                    && visited.insert(child_pid)
                {
                    order.push(child_pid);
                }
            }
        }

        // 先终止子孙进程，再终止根进程，防止父进程退出后子进程逃逸
        let mut root_gone = false;
        for &pid in order.iter().rev() {
            let mut h = OpenProcess(PROCESS_TERMINATE | PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            if h == 0 {
                h = OpenProcess(PROCESS_TERMINATE, 0, pid);
            }
            if h != 0 {
                let term_ok = TerminateProcess(h, 1) != 0;
                if pid == root_pid {
                    if term_ok {
                        root_gone = true;
                    } else {
                        let mut ec: u32 = 0;
                        if GetExitCodeProcess(h, &mut ec) != 0 && ec != STILL_ACTIVE {
                            root_gone = true;
                        }
                    }
                }
                let _ = CloseHandle(h);
            } else if pid == root_pid && GetLastError() == ERROR_INVALID_PARAMETER {
                root_gone = true;
            }
        }
        root_gone
    }
}

/// 按 PID 结束进程树（跨平台封装：Windows 优先采用毫秒级 Toolhelp32 + TerminateProcess，杜绝 taskkill 2s WMI 延迟）
pub async fn kill_pid_tree(pid: u32) {
    #[cfg(windows)]
    {
        if !terminate_process_tree_win32(pid) {
            let mut k = Command::new("taskkill");
            k.args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            k.creation_flags(CREATE_NO_WINDOW);
            let _ = k.status().await;
        }
    }
    #[cfg(not(windows))]
    {
        let mut k = Command::new("kill");
        k.args(["-KILL", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let _ = k.status().await;
    }
}

/// 结束整个进程树（uv trampoline -> python）
pub async fn kill_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        kill_pid_tree(pid).await;
    }
    let _ = child.kill().await;
}

pub struct RunOutput {
    pub code: i32,
    pub output: String,
    pub stdout: String,
    pub cancelled: bool,
    pub timed_out: bool,
}

const MAX_COLLECTED_BYTES: usize = 256 * 1024;
const MAX_LINE_BUFFER_BYTES: usize = 16 * 1024;

pub(crate) fn push_rolling_tail(target: &mut String, line: &str, max_bytes: usize) {
    target.push_str(line);
    target.push('\n');
    if target.len() > max_bytes {
        let mut cut = target.len() - max_bytes;
        while cut < target.len() && !target.is_char_boundary(cut) {
            cut += 1;
        }
        target.drain(..cut);
    }
}

struct LineByteBuffer {
    buf: Vec<u8>,
    last_was_cr: bool,
}

impl LineByteBuffer {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(512),
            last_was_cr: false,
        }
    }

    fn push_bytes<F: FnMut(&str)>(&mut self, bytes: &[u8], mut emit: F) {
        for &b in bytes {
            if b == b'\n' {
                if self.last_was_cr {
                    self.last_was_cr = false;
                    continue;
                }
                let s = String::from_utf8_lossy(&self.buf);
                emit(&s);
                self.buf.clear();
            } else if b == b'\r' {
                self.last_was_cr = true;
                let s = String::from_utf8_lossy(&self.buf);
                emit(&s);
                self.buf.clear();
            } else {
                self.last_was_cr = false;
                self.buf.push(b);
                if self.buf.len() >= MAX_LINE_BUFFER_BYTES {
                    // 若存在未完结的 UTF-8 尾字节（最多 3 字节），保留到下一段，避免人为切断合法多字节字符
                    let split_at = match std::str::from_utf8(&self.buf) {
                        Ok(_) => self.buf.len(),
                        Err(e) if e.error_len().is_none() && e.valid_up_to() > 0 => e.valid_up_to(),
                        Err(_) => self.buf.len(),
                    };
                    let s = String::from_utf8_lossy(&self.buf[..split_at]);
                    emit(&s);
                    self.buf.drain(..split_at);
                }
            }
        }
    }

    fn flush_remaining<F: FnMut(&str)>(&mut self, mut emit: F) {
        if !self.buf.is_empty() {
            let s = String::from_utf8_lossy(&self.buf);
            emit(&s);
            self.buf.clear();
        }
        self.last_was_cr = false;
    }
}

/// 运行命令：可选 stdin、超时、取消，原始字节排空管道（容忍非 UTF-8 输出与 `\r` 进度条，绝不死锁管道）
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
    let deadline_instant = tokio::time::Instant::now() + timeout;

    // Issue #85：stdin 管道写入放入独立异步任务并发推进，使总超时 deadline 与取消令牌完整覆盖输入管道阻塞期，
    // 同时避免子进程先写大量 stdout/stderr 再读 stdin 造成双向管道死锁。
    let stdin_task = match (stdin_data, child.stdin.take()) {
        (Some(data), Some(mut si)) => Some(tokio::spawn(async move {
            let _ = si.write_all(data.as_bytes()).await;
            let _ = si.shutdown().await;
            drop(si);
        })),
        _ => None,
    };

    let mut stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
    let mut out_buf = LineByteBuffer::new();
    let mut err_buf = LineByteBuffer::new();
    let mut out_chunk = [0u8; 4096];
    let mut err_chunk = [0u8; 4096];
    let mut collected = String::new();
    let mut stdout_collected = String::new();
    let mut out_done = false;
    let mut err_done = false;
    let cancel = cancel.unwrap_or_default();
    let deadline = tokio::time::sleep_until(deadline_instant);
    tokio::pin!(deadline);

    let mut cancelled = false;
    let mut timed_out = false;

    while !(out_done && err_done) {
        tokio::select! {
            res = stdout.read(&mut out_chunk), if !out_done => match res {
                Ok(0) | Err(_) => {
                    out_buf.flush_remaining(|l| {
                        on_line(l);
                        push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
                        push_rolling_tail(&mut stdout_collected, l, MAX_COLLECTED_BYTES);
                    });
                    out_done = true;
                }
                Ok(n) => {
                    out_buf.push_bytes(&out_chunk[..n], |l| {
                        on_line(l);
                        push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
                        push_rolling_tail(&mut stdout_collected, l, MAX_COLLECTED_BYTES);
                    });
                }
            },
            res = stderr.read(&mut err_chunk), if !err_done => match res {
                Ok(0) | Err(_) => {
                    err_buf.flush_remaining(|l| {
                        on_line(l);
                        push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
                    });
                    err_done = true;
                }
                Ok(n) => {
                    err_buf.push_bytes(&err_chunk[..n], |l| {
                        on_line(l);
                        push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
                    });
                }
            },
            _ = cancel.cancelled() => { cancelled = true; break; }
            _ = &mut deadline => { timed_out = true; break; }
        }
    }

    if (cancelled || timed_out) && !out_done {
        if let Ok(Ok(n)) =
            tokio::time::timeout(Duration::from_millis(5), stdout.read(&mut out_chunk)).await
        {
            if n > 0 {
                out_buf.push_bytes(&out_chunk[..n], |l| {
                    on_line(l);
                    push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
                    push_rolling_tail(&mut stdout_collected, l, MAX_COLLECTED_BYTES);
                });
            }
        }
        out_buf.flush_remaining(|l| {
            on_line(l);
            push_rolling_tail(&mut collected, l, MAX_COLLECTED_BYTES);
            push_rolling_tail(&mut stdout_collected, l, MAX_COLLECTED_BYTES);
        });
    }

    drop(stdout);
    drop(stderr);
    let status = if cancelled || timed_out {
        if let Some(t) = stdin_task {
            t.abort();
        }
        kill_tree(&mut child).await;
        tokio::time::timeout(Duration::from_secs(5), child.wait())
            .await
            .ok()
            .and_then(|r| r.ok())
    } else {
        // 即使子进程提前关闭了 stdout/stderr 输出流并继续睡眠挂起，child.wait() 仍受总 deadline 与 cancel 约束
        let wait_res = tokio::select! {
            res = child.wait() => Some(res.ok()),
            _ = cancel.cancelled() => {
                cancelled = true;
                None
            }
            _ = tokio::time::timeout_at(deadline_instant, std::future::pending::<()>()) => {
                timed_out = true;
                None
            }
        };
        if let Some(t) = stdin_task {
            t.abort();
        }
        match wait_res {
            Some(st) => st,
            None => {
                kill_tree(&mut child).await;
                tokio::time::timeout(Duration::from_secs(5), child.wait())
                    .await
                    .ok()
                    .and_then(|r| r.ok())
            }
        }
    };
    if let Some(slot) = pid_slot {
        *slot.l() = None;
    }
    Ok(RunOutput {
        code: status.and_then(|s| s.code()).unwrap_or(-1),
        output: collected,
        stdout: stdout_collected,
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

/// 过滤 ANSI 转义与 `google-colab-cli` 版本更新提示横幅，避免升级提示遮蔽真实错误尾行
pub fn strip_colab_cli_banners(raw: &str) -> String {
    let no_ansi = crate::relay::strip_ansi_codes(raw);
    let mut kept = Vec::new();
    for line in no_ansi.lines() {
        let t = line.trim();
        if t.starts_with("[colab] A new version of Colab CLI is available")
            || (t.starts_with("[colab] Run ")
                && (t.contains("google-colab-cli") || t.contains("uv tool upgrade")))
        {
            continue;
        }
        kept.push(line);
    }
    kept.join("\n")
}

fn is_cli_box_border_line(t: &str) -> bool {
    if t.is_empty() {
        return false;
    }
    if (t.starts_with('┌') || t.starts_with('╭')) && (t.ends_with('┐') || t.ends_with('╮')) {
        return true;
    }
    t.chars().all(|c| {
        c.is_whitespace()
            || matches!(
                c,
                '┌' | '┐'
                    | '└'
                    | '┘'
                    | '─'
                    | '━'
                    | '╭'
                    | '╮'
                    | '╰'
                    | '╯'
                    | '═'
                    | '├'
                    | '┤'
                    | '┬'
                    | '┴'
                    | '┼'
                    | '+'
                    | '-'
                    | '='
            )
    })
}

fn strip_cli_box_row(t: &str) -> Option<&str> {
    let mut chars = t.chars();
    let first = chars.next()?;
    let last = chars.next_back()?;
    if matches!(first, '│' | '║') && matches!(last, '│' | '║') {
        let inner = &t[first.len_utf8()..t.len() - last.len_utf8()];
        Some(inner.trim())
    } else {
        None
    }
}

/// 从 CLI 输出中提取最具可读性的最后一条核心报错（剥离 ANSI、升级横幅及 Rich/Typer 方框边框，合并方框内折行正文）
pub fn last_meaningful_cli_line(cleaned: &str, fallback: &str) -> String {
    let stripped = strip_colab_cli_banners(cleaned);
    let lines: Vec<&str> = stripped.lines().map(str::trim).collect();
    let mut idx = lines.len();
    while idx > 0 {
        idx -= 1;
        let t = lines[idx];
        if t.is_empty() || is_cli_box_border_line(t) {
            continue;
        }
        if let Some(inner) = strip_cli_box_row(t) {
            let mut box_parts = Vec::new();
            if !inner.is_empty() {
                box_parts.push(inner);
            }
            while idx > 0 {
                let prev = lines[idx - 1];
                if prev.is_empty() {
                    idx -= 1;
                    continue;
                }
                if is_cli_box_border_line(prev) {
                    break;
                }
                if let Some(prev_inner) = strip_cli_box_row(prev) {
                    if !prev_inner.is_empty() {
                        box_parts.push(prev_inner);
                    }
                    idx -= 1;
                } else {
                    break;
                }
            }
            box_parts.reverse();
            let joined = box_parts.join(" ");
            if !joined.is_empty() {
                return joined;
            }
            continue;
        }
        return t.to_string();
    }
    fallback.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn test_line_byte_buffer_non_utf8_cr_and_unbroken_utf8_boundary() {
        let mut buf = LineByteBuffer::new();
        let mut lines = Vec::new();

        // GBK bytes ("正在") + \r progress + \r\n + UTF-8 across >16KB unbroken line
        buf.push_bytes(b"\xd5\xfd\xd4\xda step1\rstep2\r", |l| lines.push(l.to_string()));
        buf.push_bytes(b"\ndone\n", |l| lines.push(l.to_string()));
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("step1"));
        assert_eq!(lines[1], "step2");
        assert_eq!(lines[2], "done");

        // Unbroken >16KB line with 3-byte UTF-8 ("中" = E4 B8 AD) crossing MAX_LINE_BYTES
        lines.clear();
        let prefix = vec![b'a'; MAX_LINE_BUFFER_BYTES - 1];
        let mut big = prefix;
        big.extend_from_slice("中尾\n".as_bytes());
        buf.push_bytes(&big, |l| lines.push(l.to_string()));
        let joined = lines.join("");
        assert!(joined.ends_with("中尾"));
        assert!(!joined.contains('\u{FFFD}'));
    }

    #[tokio::test]
    async fn test_run_streaming_non_utf8_over_160kb_no_deadlock_and_separate_stdout() {
        let py = colab_python_bin();
        let code = concat!(
            "import sys\n",
            "chunk = b'\\xd5\\xfd\\xd4\\xda_' * 40 + b'ok\\n'\n",
            "for _ in range(900):\n",
            "    sys.stdout.buffer.write(chunk)\n",
            "sys.stdout.buffer.flush()\n",
            "sys.stderr.write('WARN_ONLY_STDERR\\n')\n",
            "sys.stderr.flush()\n",
        );
        let mut callback_count = 0usize;
        let out = run_streaming(
            &py,
            &s(&["-c", code]),
            None,
            Duration::from_secs(15),
            None,
            None,
            |_| callback_count += 1,
        )
        .await
        .expect("run_streaming should succeed on non-UTF-8 >160KB output");

        assert_eq!(out.code, 0);
        assert!(!out.timed_out);
        assert!(!out.cancelled);
        assert!(callback_count >= 901);
        assert!(out.output.len() <= MAX_COLLECTED_BYTES + 256);
        assert!(out.output.contains("WARN_ONLY_STDERR"));
        assert!(!out.stdout.contains("WARN_ONLY_STDERR"));
        assert!(out.stdout.contains("ok"));
    }

    #[tokio::test]
    async fn test_run_streaming_cancel_and_timeout_terminate_child() {
        let py = colab_python_bin();
        let cancel = CancellationToken::new();
        let cancel_trigger = cancel.clone();
        let pid_slot: Mutex<Option<u32>> = Mutex::new(None);

        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(250)).await;
            cancel_trigger.cancel();
        });

        let started = Instant::now();
        let out = run_streaming(
            &py,
            &s(&["-c", "import time; print('READY', flush=True); time.sleep(30)"]),
            None,
            Duration::from_secs(20),
            Some(cancel),
            Some(&pid_slot),
            |_| {},
        )
        .await
        .expect("cancelled run_streaming should return Ok(RunOutput)");

        assert!(out.cancelled);
        assert!(!out.timed_out);
        assert!(started.elapsed() < Duration::from_secs(8));
        assert!(pid_slot.l().is_none());

        let out_to = run_streaming(
            &py,
            &s(&["-c", "import time; time.sleep(30)"]),
            None,
            Duration::from_millis(300),
            None,
            None,
            |_| {},
        )
        .await
        .expect("timed-out run_streaming should return Ok(RunOutput)");
        assert!(out_to.timed_out);
    }

    #[test]
    fn test_workspace_prefix_match_boundaries() {
        assert!(!is_workspace_prefix_match("", "llm-blue"));
        assert!(!is_workspace_prefix_match("ab", "ab-1"));
        assert!(is_workspace_prefix_match("ab", "ab"));
        assert!(!is_workspace_prefix_match("llm", "llmchat"));
        assert!(!is_workspace_prefix_match("dl2", "dl2train"));
        assert!(is_workspace_prefix_match("llm", "llm-blue"));
        assert!(is_workspace_prefix_match("resnet_exp", "resnet_exp_r2"));
        assert!(is_workspace_prefix_match("Qwen-Service-Green", "qwen-service"));
    }

    #[test]
    fn test_strip_colab_cli_banners_preserves_real_error_tail() {
        let noisy = "\x1b[31mError: HTTP 400: Backend rejected accelerator 'G4'\x1b[0m\n[colab] A new version of Colab CLI is available: 0.7.1 -> 0.8.0\n[colab] Run `uv tool upgrade google-colab-cli` to update.\n";
        let cleaned = strip_colab_cli_banners(noisy);
        assert_eq!(
            last_meaningful_cli_line(&cleaned, "unknown"),
            "Error: HTTP 400: Backend rejected accelerator 'G4'"
        );

        // Single-line Typer / Rich error box must extract inner message, NEVER bottom border └────┘
        let rich_single = "Usage: colab new [OPTIONS]\nTry 'colab new -h' for help.\n┌─ Error ─────────────────────────────────────────────────────────────────────┐\n│ No such option: --auth                                                      │\n└─────────────────────────────────────────────────────────────────────────────┘\n";
        assert_eq!(
            last_meaningful_cli_line(rich_single, "unknown"),
            "No such option: --auth"
        );

        // Multi-line wrapped Rich error box with ANSI codes must join wrapped rows cleanly
        let rich_multiline = "\x1b[31m╭─ Error ──────────────────────────────────────────╮\x1b[0m\n│ Invalid value for '--gpu': unsupported value     │\n│ 'FOO'. Supported: T4, L4, G4, H100, A100.        │\n╰──────────────────────────────────────────────────╯\n";
        assert_eq!(
            last_meaningful_cli_line(rich_multiline, "unknown"),
            "Invalid value for '--gpu': unsupported value 'FOO'. Supported: T4, L4, G4, H100, A100."
        );
    }

    #[tokio::test]
    async fn test_sync_sessions_dedups_orphans_and_respects_creating_active() {
        let _env_guard = TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_cli_sync_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        let prev = TEST_CONFIG_DIR.l().clone();
        *TEST_CONFIG_DIR.l() = Some(tmp.clone());

        // Pre-populate stale orphan sharing endpoint with a named session + dead orphan
        std::fs::write(
            sessions_path(),
            serde_json::json!({
                "orphan-abc123-0001": { "endpoint": "m-s-shared01", "proxy_url": "http://127.0.0.1", "proxy_token": "old" },
                "orphan-dead00-9999": { "endpoint": "m-s-gone", "proxy_url": "http://127.0.0.1", "proxy_token": "old" }
            })
            .to_string(),
        )
        .unwrap();

        // 1. While creating_active == true, new orphan-newep-1111 must NOT be written, while named session IS registered
        sync_sessions_with_assignments(
            &[
                (
                    "g4-smoke".into(),
                    "m-s-shared01".into(),
                    "https://proxy.example".into(),
                    "tok-1".into(),
                    3600,
                    "CPU".into(),
                    "CPU".into(),
                    false,
                ),
                (
                    "orphan-newep-1111".into(),
                    "m-s-provisioning".into(),
                    "https://proxy2.example".into(),
                    "tok-2".into(),
                    3600,
                    "G4".into(),
                    "GPU".into(),
                    false,
                ),
            ],
            true,
        );

        let map1 = read_sessions();
        assert!(map1.contains_key("g4-smoke"));
        assert!(!map1.contains_key("orphan-abc123-0001"), "duplicate orphan sharing named endpoint must be removed");
        assert!(!map1.contains_key("orphan-dead00-9999"), "offline orphan must be pruned");
        assert!(!map1.contains_key("orphan-newep-1111"), "orphan must not be registered during colab new");
        assert_eq!(map1["g4-smoke"]["name"].as_str(), Some("g4-smoke"));
        assert_eq!(map1["g4-smoke"]["url"].as_str(), Some("https://proxy.example"));
        assert_eq!(map1["g4-smoke"]["token"].as_str(), Some("tok-1"));
        assert_eq!(map1["g4-smoke"]["endpoint"].as_str(), Some("m-s-shared01"));
        assert_eq!(map1["g4-smoke"]["accelerator"].as_str(), Some("NONE"));
        assert_eq!(map1["g4-smoke"]["variant"].as_str(), Some("DEFAULT"));
        assert!(map1["g4-smoke"].get("proxy_url").is_none());
        assert!(map1["g4-smoke"].get("proxy_token").is_none());
        assert!(map1["g4-smoke"]["token_expires_at"].as_str().unwrap_or("").contains('T'));

        // Verify TTL < 300s is NOT clamped upward to 300s, and existing entry updates canonical url/token while keeping kernel_id
        update_session_kernel_id("m-s-shared01", "kid-keep-01");
        let before_short = chrono::Utc::now();
        sync_sessions_with_assignments(
            &[
                (
                    "g4-smoke".into(),
                    "m-s-shared01".into(),
                    "https://proxy-rot.example".into(),
                    "tok-rot".into(),
                    120,
                    "CPU".into(),
                    "CPU".into(),
                    false,
                ),
                (
                    "orphan-newep-1111".into(),
                    "m-s-browser".into(),
                    "https://proxy2.example".into(),
                    "tok-2".into(),
                    3600,
                    "G4".into(),
                    "GPU".into(),
                    false,
                ),
            ],
            false,
        );
        let map2 = read_sessions();
        assert!(map2.contains_key("orphan-newep-1111"));
        assert_eq!(map2["g4-smoke"]["url"].as_str(), Some("https://proxy-rot.example"));
        assert_eq!(map2["g4-smoke"]["token"].as_str(), Some("tok-rot"));
        assert_eq!(map2["g4-smoke"]["kernel_id"].as_str(), Some("kid-keep-01"));
        let exp_dt = chrono::DateTime::parse_from_rfc3339(
            map2["g4-smoke"]["token_expires_at"].as_str().unwrap(),
        )
        .unwrap()
        .with_timezone(&chrono::Utc);
        let delta_s = (exp_dt - before_short).num_seconds();
        assert!(
            (100..=150).contains(&delta_s),
            "short TTL (120s) must not be clamped to >=300s, got {delta_s}s"
        );
        assert_eq!(map2["orphan-newep-1111"]["name"].as_str(), Some("orphan-newep-1111"));
        assert_eq!(map2["orphan-newep-1111"]["url"].as_str(), Some("https://proxy2.example"));
        assert_eq!(map2["orphan-newep-1111"]["token"].as_str(), Some("tok-2"));
        assert!(map2["orphan-newep-1111"].get("proxy_url").is_none());

        // Steady-state poll must not rewrite sessions.json when nothing changed
        let raw_before = std::fs::read_to_string(sessions_path()).unwrap();
        sync_sessions_with_assignments(
            &[
                (
                    "g4-smoke".into(),
                    "m-s-shared01".into(),
                    "https://proxy-rot.example".into(),
                    "tok-rot".into(),
                    115,
                    "CPU".into(),
                    "CPU".into(),
                    false,
                ),
                (
                    "orphan-newep-1111".into(),
                    "m-s-browser".into(),
                    "https://proxy2.example".into(),
                    "tok-2".into(),
                    3595,
                    "G4".into(),
                    "GPU".into(),
                    false,
                ),
            ],
            false,
        );
        let raw_after = std::fs::read_to_string(sessions_path()).unwrap();
        assert_eq!(raw_before, raw_after, "steady-state poll must leave sessions.json untouched");

        // Real-host google-colab-cli v0.8.0 StateStore roundtrip check when colab python env is installed
        let py = colab_python_bin();
        if py.exists() {
            let out = std::process::Command::new(&py)
                .args([
                    "-c",
                    "import sys; from pathlib import Path; from colab_cli.state import StateStore; s = StateStore(Path(sys.argv[1])).list(); assert 'g4-smoke' in s and 'orphan-newep-1111' in s, f'missing keys: {list(s)}'; assert s['g4-smoke'].url == 'https://proxy-rot.example' and s['g4-smoke'].token == 'tok-rot'",
                    &sessions_path().to_string_lossy(),
                ])
                .env("PYTHONDONTWRITEBYTECODE", "1")
                .output()
                .expect("spawn colab_python_bin for StateStore validation");
            assert!(
                out.status.success(),
                "colab_cli StateStore failed to parse synced sessions.json: stdout={} stderr={}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }

        *TEST_CONFIG_DIR.l() = prev;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_60_run_streaming_enforces_deadline_after_child_closes_stdout_stderr() {
        let py = colab_python_bin();
        if !py.exists() {
            return;
        }
        // Child prints a line, explicitly closes stdout (fd 1) and stderr (fd 2) so pipe readers hit EOF,
        // and then keeps sleeping long past the timeout.
        let code = "import os, sys, time; sys.stdout.write('PIPES_CLOSING\\n'); sys.stdout.flush(); os.close(1); os.close(2); time.sleep(30)";
        let pid_slot: Mutex<Option<u32>> = Mutex::new(None);
        let started = Instant::now();
        let out = run_streaming(
            &py,
            &s(&["-c", code]),
            None,
            Duration::from_millis(400),
            None,
            Some(&pid_slot),
            |_| {},
        )
        .await
        .expect("run_streaming must bound post-EOF child.wait() by deadline");

        assert!(
            out.timed_out,
            "child sleeping after closing stdout/stderr must trigger timed_out=true"
        );
        assert!(
            started.elapsed() < Duration::from_secs(8),
            "post-EOF hang took too long: {:?}",
            started.elapsed()
        );
        assert!(out.stdout.contains("PIPES_CLOSING"));
        assert!(pid_slot.l().is_none());
    }

    #[tokio::test]
    async fn test_issue_59_rotate_llm_api_key_persists_verifies_and_errors_on_unwritable_path() {
        let _env_guard = TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_apikey_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp).unwrap();
        let prev = TEST_CONFIG_DIR.l().clone();
        *TEST_CONFIG_DIR.l() = Some(tmp.clone());

        let ok_info = rotate_llm_api_key_entry(
            "qwen-prod",
            "blue",
            "vllm",
            Some("sk-custom-verified-123456"),
            18081,
            8000,
            "Qwen/Qwen2.5-7B-Instruct",
        )
        .expect("rotate_llm_api_key_entry should succeed on writable config dir");
        assert_eq!(ok_info.api_key, "sk-custom-verified-123456");
        assert_eq!(
            lookup_vm_api_key_meta("qwen-prod"),
            Some(("sk-custom-verified-123456".to_string(), Some(18081)))
        );

        // Replace llm_api_keys.json with a directory so persistence fails
        let key_path = llm_api_keys_path();
        std::fs::remove_file(&key_path).unwrap();
        std::fs::create_dir_all(&key_path).unwrap();

        let err = rotate_llm_api_key_entry(
            "qwen-prod",
            "blue",
            "vllm",
            Some("sk-should-fail-999999"),
            18081,
            8000,
            "Qwen/Qwen2.5-7B-Instruct",
        )
        .expect_err("rotate_llm_api_key_entry must return Err when disk persistence fails");
        assert!(
            err.to_string().contains("失败") || err.to_string().contains("目录"),
            "unexpected error: {err:#}"
        );

        *TEST_CONFIG_DIR.l() = prev;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_43_and_74_adc_detection_subcommand_args_and_session_profile_persistence() {
        let _env_guard = TEST_ENV_LOCK.lock().await;
        let tmp = std::env::temp_dir().join(format!("pilot_profiles_{}", uuid::Uuid::new_v4()));
        let hist = tmp.join("history");
        std::fs::create_dir_all(&hist).unwrap();
        let prev_cfg = TEST_CONFIG_DIR.l().clone();
        let prev_adc = TEST_ADC_PATH.l().clone();
        *TEST_CONFIG_DIR.l() = Some(tmp.clone());

        let fake_adc = tmp.join("application_default_credentials.json");
        *TEST_ADC_PATH.l() = Some(fake_adc.clone());
        assert!(!is_adc_configured());
        std::fs::write(&fake_adc, r#"{"type":"authorized_user"}"#).unwrap();
        assert!(is_adc_configured());

        let saved_llm = crate::CreateVmLlmConfig {
            engine: "vllm".into(),
            model_name: "Qwen/Qwen2.5-14B-Instruct".into(),
            slot: Some("blue".into()),
            gdrive_model_path: None,
            custom_api_key: None,
        };
        save_session_profile(
            "adc-node",
            &SessionProfile {
                boot_mode: "llm".into(),
                accelerator: "L4".into(),
                high_mem: false,
                workspace_path: Some(tmp.display().to_string()),
                auth_mode: Some("adc".into()),
                llm_config: Some(saved_llm),
                relay_config: None,
                updated_at: None,
            },
        )
        .expect("save_session_profile must succeed");

        assert_eq!(get_session_auth_mode("adc-node").as_deref(), Some("adc"));
        assert_eq!(
            colab_args_for_session("adc-node", &s(&["restart-kernel", "-s", "adc-node"])),
            s(&["--auth", "adc", "restart-kernel", "-s", "adc-node"])
        );
        assert_eq!(
            colab_args_for_session("oauth-node", &s(&["stop", "-s", "oauth-node"])),
            s(&["stop", "-s", "oauth-node"])
        );

        // Write history JSONL for both saved `adc-node` and legacy unsaved `llm-legacy`
        std::fs::write(
            hist.join("adc-node.jsonl"),
            r#"{"timestamp":"2026-11-20T00:01:00Z","event_type":"session_created","accelerator":"L4","machine_shape":"STANDARD"}"#,
        )
        .unwrap();
        std::fs::write(
            hist.join("llm-legacy.jsonl"),
            r#"{"timestamp":"2026-11-20T00:00:30Z","event_type":"session_created","accelerator":"T4","machine_shape":"HIGHMEM"}"#,
        )
        .unwrap();
        invalidate_recent_sessions_cache();

        let recent = read_recent_sessions(&[], 10);
        let saved_view = recent
            .iter()
            .find(|r| r.session_name == "adc-node")
            .expect("adc-node in recent");
        assert!(saved_view.profile_saved);
        assert_eq!(saved_view.boot_mode.as_deref(), Some("llm"));
        assert_eq!(
            saved_view
                .llm_config
                .as_ref()
                .map(|c| c.model_name.as_str()),
            Some("Qwen/Qwen2.5-14B-Instruct")
        );

        let legacy_view = recent
            .iter()
            .find(|r| r.session_name == "llm-legacy")
            .expect("llm-legacy in recent");
        assert!(!legacy_view.profile_saved);
        assert!(
            legacy_view.boot_mode.is_none(),
            "unsaved legacy session must NOT guess boot_mode from `llm-` prefix"
        );

        *TEST_ADC_PATH.l() = prev_adc;
        *TEST_CONFIG_DIR.l() = prev_cfg;
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[tokio::test]
    async fn test_issue_85_run_streaming_enforces_timeout_when_32mib_stdin_blocks() {
        let py = colab_python_bin();
        if !py.exists() {
            return;
        }
        // Child sleeps 2.5s without reading a single byte of stdin; send 32 MiB with 150 ms timeout.
        let payload_32mib = "A".repeat(32 * 1024 * 1024);
        let pid_slot: Mutex<Option<u32>> = Mutex::new(None);
        let started = Instant::now();
        let out = run_streaming(
            &py,
            &s(&["-c", "import time; time.sleep(2.5)"]),
            Some(payload_32mib),
            Duration::from_millis(150),
            None,
            Some(&pid_slot),
            |_| {},
        )
        .await
        .expect("run_streaming must enforce timeout even when stdin pipe is blocked");

        let elapsed = started.elapsed();
        assert!(
            out.timed_out,
            "blocked 32 MiB stdin write must trigger timed_out=true (elapsed={elapsed:?})"
        );
        assert!(
            elapsed < Duration::from_millis(1500),
            "command timeout must cover stdin write time instead of waiting 2.5s; actual={elapsed:?}"
        );
        assert!(pid_slot.l().is_none());

        // Bidirectional pipe sanity: child writes 128 KiB to stdout BEFORE reading 4 MiB from stdin within 6s
        let payload_4mib = "B".repeat(4 * 1024 * 1024);
        let out_bi = run_streaming(
            &py,
            &s(&[
                "-c",
                "import sys; sys.stdout.write('X' * 131072 + '\\n'); sys.stdout.flush(); n = len(sys.stdin.buffer.read()); print(f'READ_OK={n}')",
            ]),
            Some(payload_4mib),
            Duration::from_secs(6),
            None,
            None,
            |_| {},
        )
        .await
        .expect("bidirectional stdout + stdin should not deadlock");
        assert!(!out_bi.timed_out);
        assert_eq!(out_bi.code, 0);
        assert!(out_bi.stdout.contains("READ_OK=4194304"));
    }
}
