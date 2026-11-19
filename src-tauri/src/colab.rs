//! Google Colab 服务端 REST 客户端（兼容 google-colab-cli v0.7 token.json 与 v0.8+ 系统密钥环双源 OAuth 凭证）
//!
//! 实现了：
//! - OAuth2 双源凭证读取（优先 `token.json`，回退 OS Keyring `google-colab-cli/oauth_token`）与 `refresh_token` 自动续期回写
//! - `/tun/m/ccu-info`      账户计算单元余额与计费速率
//! - `/tun/m/assignments`   服务端真实计费实例列表（删除核验的唯一可信来源）
//! - `/tun/m/unassign/{ep}` 服务端释放实例（停止计费）
//! - Jupyter Contents API  读写云端 /content 文件（携带 `X-Colab-Runtime-Proxy-Token` 请求头与瞬态重试）

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
#[cfg(test)]
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const COLAB_DOMAIN: &str = "https://colab.research.google.com";
const XSSI_PREFIX: &str = ")]}'\n";
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const KEYRING_SERVICE: &str = "google-colab-cli";
const KEYRING_USERNAME: &str = "oauth_token";

static KEYRING_AUTH_CACHE: std::sync::Mutex<Option<(bool, Instant)>> = std::sync::Mutex::new(None);

#[derive(Deserialize, Serialize, Clone, Debug)]
struct TokenFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    token: Option<String>,
    #[serde(default)]
    refresh_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    token_uri: Option<String>,
    #[serde(default)]
    client_id: String,
    #[serde(default)]
    client_secret: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expiry: Option<String>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenSource {
    File,
    Keyring,
}

fn set_keyring_auth_cache(ok: bool) {
    if let Ok(mut g) = KEYRING_AUTH_CACHE.lock() {
        *g = Some((ok, Instant::now()));
    }
}

fn read_keyring_oauth_json_sync() -> Option<String> {
    use std::io::Read;
    use std::process::{Command, Stdio};

    let py = crate::cli::colab_python_bin();
    let code = format!(
        "import keyring, sys; v = keyring.get_password({:?}, {:?}); sys.stdout.write(v or '')",
        KEYRING_SERVICE, KEYRING_USERNAME
    );
    let mut cmd = Command::new(py);
    cmd.args(["-c", &code])
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("NO_COLOR", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().ok()?;
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut out = String::new();
                if let Some(mut stdout) = child.stdout.take() {
                    let _ = stdout.read_to_string(&mut out);
                }
                let trimmed = out.trim().to_string();
                return if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                };
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => return None,
        }
    }
}

fn write_keyring_oauth_json_sync(json_str: &str) -> Result<()> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let py = crate::cli::colab_python_bin();
    let code = format!(
        "import keyring, sys; d = sys.stdin.read()\nif d.strip(): keyring.set_password({:?}, {:?}, d)",
        KEYRING_SERVICE, KEYRING_USERNAME
    );
    let mut cmd = Command::new(py);
    cmd.args(["-c", &code])
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let mut child = cmd.spawn().context("启动 keyring 写入子进程失败")?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(json_str.as_bytes())?;
    }
    let deadline = Instant::now() + Duration::from_secs(4);
    loop {
        match child.try_wait()? {
            Some(st) if st.success() => return Ok(()),
            Some(st) => return Err(anyhow!("写入系统密钥环失败: {st}")),
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(anyhow!("写入系统密钥环超时"));
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeProxyInfo {
    pub token: String,
    #[serde(default)]
    pub token_expires_in_seconds: i64,
    pub url: String,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Assignment {
    #[serde(default)]
    pub accelerator: String,
    pub endpoint: String,
    #[serde(default)]
    pub variant: Value,
    #[serde(default)]
    pub machine_shape: Value,
    pub runtime_proxy_info: RuntimeProxyInfo,
}

impl Assignment {
    /// NONE -> CPU
    pub fn hardware_label(&self) -> String {
        if self.accelerator.is_empty() || self.accelerator == "NONE" {
            "CPU".into()
        } else {
            self.accelerator.clone()
        }
    }

    pub fn is_high_ram(&self) -> bool {
        match &self.machine_shape {
            Value::Number(n) => n.as_i64() == Some(1),
            Value::String(s) => s == "HIGH_RAM" || s == "1",
            _ => false,
        }
    }

    pub fn variant_label(&self) -> String {
        match &self.variant {
            Value::Number(n) => match n.as_i64() {
                Some(1) => "GPU".into(),
                Some(2) => "TPU".into(),
                _ => "DEFAULT".into(),
            },
            Value::String(s) => s.clone(),
            _ => "DEFAULT".into(),
        }
    }
}

#[derive(Deserialize, Default)]
struct AssignmentList {
    #[serde(default)]
    assignments: Vec<Assignment>,
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct Ccu {
    #[serde(rename(deserialize = "currentBalance"), default)]
    pub balance: f64,
    #[serde(rename(deserialize = "consumptionRateHourly"), default)]
    pub burn_rate_hourly: f64,
    #[serde(rename(deserialize = "assignmentsCount"), default)]
    pub assignments_count: i64,
}

#[derive(Deserialize)]
struct UnassignToken {
    token: String,
}

pub struct ColabClient {
    http: reqwest::Client,
    base_url: String,
    token_file_path: Option<PathBuf>,
    cached: Mutex<Option<(String, chrono::DateTime<chrono::Utc>)>>,
    source_cache: Mutex<Option<(TokenFile, TokenSource, Instant)>>,
    /// 最近一次 refresh_token 续期失败（时间 + 错误信息）：30 秒内直接返回缓存错误，避免高频轮询反复 POST
    refresh_failure: Mutex<Option<(Instant, String)>>,
    #[cfg(test)]
    keyring_override: Option<Arc<std::sync::Mutex<Option<String>>>>,
}

pub fn token_path() -> PathBuf {
    crate::cli::colab_config_dir().join("token.json")
}

/// 判断 Contents API 错误是否由过期的 `runtimeProxyInfo.token`（HTTP 401 / 403）触发
pub fn is_proxy_auth_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("ProxyAuth HTTP 401") || msg.contains("ProxyAuth HTTP 403")
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

pub(crate) fn decode_contents_base64(s: &str) -> Result<String> {
    let cleaned: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    let bytes = base64::engine::general_purpose::STANDARD.decode(cleaned)?;
    Ok(String::from_utf8_lossy(&bytes).to_string())
}

impl ColabClient {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .connect_timeout(Duration::from_secs(8))
            .pool_idle_timeout(Duration::from_secs(90))
            .pool_max_idle_per_host(8)
            .tcp_keepalive(Duration::from_secs(15))
            .tcp_nodelay(true)
            .build()
            .expect("failed to build http client");
        Self {
            http,
            base_url: COLAB_DOMAIN.to_string(),
            token_file_path: None,
            cached: Mutex::new(None),
            source_cache: Mutex::new(None),
            refresh_failure: Mutex::new(None),
            #[cfg(test)]
            keyring_override: None,
        }
    }

    #[cfg(test)]
    pub fn new_for_test(base_url: String, token_file_path: PathBuf) -> Self {
        let http = reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("failed to build test http client");
        Self {
            http,
            base_url: base_url.trim_end_matches('/').to_string(),
            token_file_path: Some(token_file_path),
            cached: Mutex::new(None),
            source_cache: Mutex::new(None),
            refresh_failure: Mutex::new(None),
            keyring_override: None,
        }
    }

    #[cfg(test)]
    pub fn new_for_test_with_keyring(
        base_url: String,
        token_file_path: PathBuf,
        keyring_store: Arc<std::sync::Mutex<Option<String>>>,
    ) -> Self {
        let mut s = Self::new_for_test(base_url, token_file_path);
        s.keyring_override = Some(keyring_store);
        s
    }

    /// 检查本机是否具备可用的 Colab OAuth 凭证：
    /// 1. 优先快速判断 `~/.config/colab-cli/token.json` 是否存在；
    /// 2. 兼容 `google-colab-cli v0.8+` 迁移至系统密钥环后的状态（带 TTL 内存缓存，绝不在 1.5s 高频轮询中重复派生 Python）。
    pub fn is_authorized() -> bool {
        if token_path().exists() {
            return true;
        }
        if let Ok(g) = KEYRING_AUTH_CACHE.lock() {
            if let Some((ok, ts)) = *g {
                let ttl = if ok {
                    Duration::from_secs(120)
                } else {
                    Duration::from_secs(15)
                };
                if ts.elapsed() < ttl {
                    return ok;
                }
            }
        }
        let ok = read_keyring_oauth_json_sync()
            .and_then(|raw| serde_json::from_str::<TokenFile>(&raw).ok())
            .map(|tf| !tf.refresh_token.trim().is_empty())
            .unwrap_or(false);
        set_keyring_auth_cache(ok);
        ok
    }

    pub fn is_client_authorized(&self) -> bool {
        if let Some(ref p) = self.token_file_path {
            if p.exists() {
                return true;
            }
            #[cfg(test)]
            if let Some(ref store) = self.keyring_override {
                return store
                    .lock()
                    .ok()
                    .and_then(|g| g.clone())
                    .and_then(|raw| serde_json::from_str::<TokenFile>(&raw).ok())
                    .map(|tf| !tf.refresh_token.trim().is_empty())
                    .unwrap_or(false);
            }
            return false;
        }
        Self::is_authorized()
    }

    fn effective_token_path(&self) -> PathBuf {
        self.token_file_path.clone().unwrap_or_else(token_path)
    }

    pub async fn invalidate_token(&self) {
        *self.cached.lock().await = None;
        *self.refresh_failure.lock().await = None;
    }

    async fn load_token_file(&self, force_reload: bool) -> Result<(TokenFile, TokenSource)> {
        let path = self.effective_token_path();
        if path.exists() {
            let raw = std::fs::read_to_string(&path).context("读取 token.json 失败")?;
            let tf: TokenFile = serde_json::from_str(&raw).context("token.json 格式无效")?;
            *self.source_cache.lock().await = Some((tf.clone(), TokenSource::File, Instant::now()));
            if self.token_file_path.is_none() {
                set_keyring_auth_cache(true);
            }
            return Ok((tf, TokenSource::File));
        }

        #[cfg(test)]
        if let Some(ref store) = self.keyring_override {
            let raw_opt = store.lock().unwrap().clone();
            let raw = raw_opt.ok_or_else(|| {
                anyhow!("未找到 Colab 授权凭证（测试模拟 token.json 与 keyring 均为空）")
            })?;
            let tf: TokenFile = serde_json::from_str(&raw).context("keyring OAuth JSON 格式无效")?;
            *self.source_cache.lock().await =
                Some((tf.clone(), TokenSource::Keyring, Instant::now()));
            return Ok((tf, TokenSource::Keyring));
        }

        if self.token_file_path.is_some() {
            return Err(anyhow!(
                "未找到 Colab 授权文件 {}，请先运行 `colab auth` 完成授权",
                path.display()
            ));
        }

        if !force_reload {
            if let Some((tf, src, loaded_at)) = self.source_cache.lock().await.as_ref() {
                if *src == TokenSource::Keyring && loaded_at.elapsed() < Duration::from_secs(180) {
                    return Ok((tf.clone(), TokenSource::Keyring));
                }
            }
        }

        let raw_opt = tokio::task::spawn_blocking(read_keyring_oauth_json_sync)
            .await
            .ok()
            .flatten();
        if let Some(raw) = raw_opt {
            let tf: TokenFile =
                serde_json::from_str(&raw).context("系统密钥环中的 Colab OAuth 凭证格式无效")?;
            *self.source_cache.lock().await =
                Some((tf.clone(), TokenSource::Keyring, Instant::now()));
            set_keyring_auth_cache(true);
            return Ok((tf, TokenSource::Keyring));
        }

        set_keyring_auth_cache(false);
        Err(anyhow!(
            "未检测到 Colab 授权凭证（~/.config/colab-cli/token.json 或系统密钥环），请先在终端运行 `colab auth` 或 `colab status` 完成 Google 登录"
        ))
    }

    async fn persist_refreshed_token(
        &self,
        mut tf: TokenFile,
        source: TokenSource,
        new_token: &str,
        exp_utc: chrono::DateTime<chrono::Utc>,
    ) {
        tf.token = Some(new_token.to_string());
        tf.expiry = Some(exp_utc.to_rfc3339_opts(chrono::SecondsFormat::Secs, true));
        *self.source_cache.lock().await = Some((tf.clone(), source, Instant::now()));
        let Ok(json_str) = serde_json::to_string(&tf) else {
            return;
        };

        match source {
            TokenSource::File => {
                // 仅当原 token.json 仍存在时原子回写，若 CLI v0.8+ 已将其迁移删除，则绝不重新生成明文 token.json
                let path = self.effective_token_path();
                if path.exists() {
                    let tmp = path.with_extension("json.tmp");
                    #[cfg(unix)]
                    {
                        use std::io::Write;
                        use std::os::unix::fs::OpenOptionsExt;
                        if let Ok(mut f) = std::fs::OpenOptions::new()
                            .create(true)
                            .write(true)
                            .truncate(true)
                            .mode(0o600)
                            .open(&tmp)
                        {
                            if f.write_all(json_str.as_bytes()).is_ok() {
                                let _ = std::fs::rename(&tmp, &path);
                            }
                        }
                        return;
                    }
                    #[cfg(not(unix))]
                    if std::fs::write(&tmp, &json_str).is_ok() {
                        let _ = std::fs::rename(&tmp, &path);
                    }
                }
            }
            TokenSource::Keyring => {
                #[cfg(test)]
                if let Some(ref store) = self.keyring_override {
                    *store.lock().unwrap() = Some(json_str);
                    return;
                }
                let _ =
                    tokio::task::spawn_blocking(move || write_keyring_oauth_json_sync(&json_str))
                        .await;
                set_keyring_auth_cache(true);
            }
        }
    }

    pub(crate) fn parse_token_expiry_utc(expiry: &str) -> Option<chrono::DateTime<chrono::Utc>> {
        let trimmed = expiry.trim();
        if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(trimmed) {
            return Some(dt.with_timezone(&chrono::Utc));
        }
        let core = trimmed.trim_end_matches('Z');
        let core = core.split('+').next().unwrap_or(core);
        let core = core.split('.').next().unwrap_or(core);
        chrono::NaiveDateTime::parse_from_str(core, "%Y-%m-%dT%H:%M:%S")
            .ok()
            .map(|dt| dt.and_utc())
    }

    async fn access_token_inner(&self, rejected_token: Option<&str>) -> Result<String> {
        let mut guard = self.cached.lock().await;
        if rejected_token.is_none() {
            if let Some((tok, exp)) = guard.as_ref() {
                if chrono::Utc::now() + chrono::Duration::seconds(300) < *exp {
                    return Ok(tok.clone());
                }
            }
        } else {
            *guard = None;
        }

        let (tf, source) = self.load_token_file(rejected_token.is_some()).await?;

        // 1. CLI 已缓存且仍有效的 access_token（若刚被服务端 401 拒绝则跳过静态缓存，直接走 refresh_token 续期）
        if let (Some(tok), Some(expiry)) = (&tf.token, &tf.expiry) {
            if rejected_token != Some(tok.as_str()) {
                if let Some(exp_utc) = Self::parse_token_expiry_utc(expiry) {
                    let remaining = (exp_utc - chrono::Utc::now()).num_seconds();
                    if remaining > 300 {
                        *guard = Some((tok.clone(), exp_utc));
                        return Ok(tok.clone());
                    }
                }
            }
        }

        // 2. 使用 refresh_token 静默续期（失败后 30 秒内直接返回缓存错误，不重复请求）
        const REFRESH_FAIL_BACKOFF: Duration = Duration::from_secs(30);
        if rejected_token.is_none() {
            let fail = self.refresh_failure.lock().await;
            if let Some((at, msg)) = fail.as_ref() {
                if at.elapsed() < REFRESH_FAIL_BACKOFF {
                    return Err(anyhow!("{msg}"));
                }
            }
        }
        match self.refresh_access_token(&tf).await {
            Ok((tok, ttl)) => {
                *self.refresh_failure.lock().await = None;
                let exp_utc = chrono::Utc::now() + chrono::Duration::seconds(ttl as i64);
                *guard = Some((tok.clone(), exp_utc));
                self.persist_refreshed_token(tf, source, &tok, exp_utc).await;
                Ok(tok)
            }
            Err(e) => {
                *self.source_cache.lock().await = None;
                *self.refresh_failure.lock().await = Some((Instant::now(), format!("{e:#}")));
                Err(e)
            }
        }
    }

    async fn access_token(&self) -> Result<String> {
        self.access_token_inner(None).await
    }

    /// 向 Google OAuth 端点 POST refresh_token，返回 (access_token, 有效秒数)
    async fn refresh_access_token(&self, tf: &TokenFile) -> Result<(String, u64)> {
        if tf.refresh_token.trim().is_empty() {
            return Err(anyhow!(
                "Colab OAuth 令牌已过期且缺少 refresh_token，请在终端运行 `colab auth` 或 `colab status` 重新授权"
            ));
        }
        #[derive(Deserialize)]
        struct RefreshResp {
            access_token: String,
            expires_in: Option<u64>,
        }
        let uri = tf.token_uri.clone().unwrap_or_else(|| DEFAULT_TOKEN_URI.into());
        let resp = self
            .http
            .post(&uri)
            .form(&[
                ("client_id", tf.client_id.as_str()),
                ("client_secret", tf.client_secret.as_str()),
                ("refresh_token", tf.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .context("刷新 Google OAuth 令牌失败（网络不可达？）")?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!(
                "刷新 Google OAuth 令牌失败 ({}): {}，请在终端重新运行 `colab status` 登录",
                status,
                truncate(&text, 200)
            ));
        }
        let r: RefreshResp = serde_json::from_str(&text).context("解析 Google OAuth 令牌响应失败")?;
        let ttl = r.expires_in.unwrap_or(3600);
        Ok((r.access_token, ttl))
    }

    async fn tun_request(&self, method: Method, path: &str, xsrf: Option<&str>) -> Result<String> {
        let mut token = self.access_token().await?;
        let url = format!("{}{path}", self.base_url);
        let mut last_err = None;
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            let mut rb = self
                .http
                .request(method.clone(), &url)
                .query(&[("authuser", "0")])
                .bearer_auth(&token)
                .header("Accept", "application/json")
                .header("X-Colab-Client-Agent", "colab-cli");
            if let Some(x) = xsrf {
                rb = rb.header("X-Goog-Colab-Token", x);
            }
            if method == Method::POST {
                rb = rb.body(Vec::<u8>::new());
            }
            match rb.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let text = resp.text().await.unwrap_or_default();
                    if status.is_success() {
                        return Ok(text.strip_prefix(XSSI_PREFIX).unwrap_or(&text).to_string());
                    }
                    if status == StatusCode::UNAUTHORIZED && attempt == 0 {
                        let rejected = token.clone();
                        self.invalidate_token().await;
                        token = self.access_token_inner(Some(&rejected)).await?;
                        last_err = Some(anyhow!("HTTP {} @ {}: {}", status, path, truncate(&text, 240)));
                        continue;
                    }
                    if status.is_server_error() && attempt == 0 {
                        last_err = Some(anyhow!("HTTP {} @ {}: {}", status, path, truncate(&text, 240)));
                        continue;
                    }
                    return Err(anyhow!("HTTP {} @ {}: {}", status, path, truncate(&text, 240)));
                }
                Err(e) => {
                    last_err = Some(anyhow!("请求 {path} 失败: {e}"));
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow!("请求 {path} 失败")))
    }

    pub async fn ccu_info(&self) -> Result<Ccu> {
        let body = self.tun_request(Method::GET, "/tun/m/ccu-info", None).await?;
        Ok(serde_json::from_str(&body)?)
    }

    pub async fn list_assignments(&self) -> Result<Vec<Assignment>> {
        let body = self.tun_request(Method::GET, "/tun/m/assignments", None).await?;
        if body.trim().is_empty() {
            return Ok(vec![]);
        }
        let list: AssignmentList = serde_json::from_str(&body)?;
        Ok(list.assignments)
    }

    /// 服务端释放实例：GET 取 XSRF token，再 POST 真正 unassign
    pub async fn unassign(&self, endpoint: &str) -> Result<()> {
        let path = format!("/tun/m/unassign/{endpoint}");
        let body = self.tun_request(Method::GET, &path, None).await?;
        let t: UnassignToken = serde_json::from_str(&body).context("解析 unassign token 失败")?;
        self.tun_request(Method::POST, &path, Some(&t.token)).await?;
        Ok(())
    }

    fn contents_url(a: &Assignment, path: &str) -> String {
        format!(
            "{}/api/contents/{}",
            a.runtime_proxy_info.url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    /// 读取云端文件文本内容；文件不存在返回 Ok(None)
    pub async fn contents_read_text(&self, a: &Assignment, path: &str) -> Result<Option<String>> {
        let url = Self::contents_url(a, path);
        let mut last_err = None;
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(if cfg!(test) { 20 } else { 300 })).await;
            }
            let send_res = self
                .http
                .get(&url)
                .header(
                    "X-Colab-Runtime-Proxy-Token",
                    a.runtime_proxy_info.token.as_str(),
                )
                .header("X-Colab-Client-Agent", "colab-cli")
                .query(&[
                    ("authuser", "0"),
                    ("colab-runtime-proxy-token", a.runtime_proxy_info.token.as_str()),
                    ("content", "1"),
                ])
                .timeout(Duration::from_secs(12))
                .send()
                .await;
            let resp = match send_res {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(anyhow!("Contents API 请求失败: {e}"));
                    continue;
                }
            };
            let st = resp.status();
            if st == StatusCode::NOT_FOUND {
                return Ok(None);
            }
            if st == StatusCode::UNAUTHORIZED || st == StatusCode::FORBIDDEN {
                return Err(anyhow!("Contents API ProxyAuth HTTP {st}"));
            }
            if st.is_server_error() && attempt == 0 {
                last_err = Some(anyhow!("Contents API HTTP {st}"));
                continue;
            }
            if !st.is_success() {
                return Err(anyhow!("Contents API HTTP {st}"));
            }
            let v: Value = match resp.json().await {
                Ok(val) => val,
                Err(e) => {
                    last_err = Some(anyhow!("Contents API JSON 解析失败: {e}"));
                    continue;
                }
            };
            let content = v.get("content").cloned().unwrap_or(Value::Null);
            let fmt = v.get("format").and_then(|f| f.as_str()).unwrap_or("text");
            return Ok(Some(match content {
                Value::String(s) if fmt == "base64" => decode_contents_base64(&s)?,
                Value::String(s) => s,
                Value::Null => String::new(),
                other => other.to_string(),
            }));
        }
        Err(last_err.unwrap_or_else(|| anyhow!("Contents API 读取超时")))
    }

    /// 以 base64 上传本地字节到云端路径（携带 Proxy Token 请求头并支持容器刚启动期间的瞬态 502/503/504 重试）
    pub async fn contents_upload(&self, a: &Assignment, remote_path: &str, bytes: &[u8]) -> Result<()> {
        let name = remote_path.rsplit('/').next().unwrap_or(remote_path);
        let payload = serde_json::json!({
            "name": name,
            "path": remote_path,
            "type": "file",
            "format": "base64",
            "content": base64::engine::general_purpose::STANDARD.encode(bytes),
            "chunk": 1,
        });
        let url = Self::contents_url(a, remote_path);
        let mut last_err = None;
        for attempt in 0..3usize {
            if attempt > 0 {
                let wait_ms = if cfg!(test) {
                    20
                } else if attempt == 1 {
                    400
                } else {
                    900
                };
                tokio::time::sleep(Duration::from_millis(wait_ms)).await;
            }
            let send_res = self
                .http
                .put(&url)
                .header(
                    "X-Colab-Runtime-Proxy-Token",
                    a.runtime_proxy_info.token.as_str(),
                )
                .header("X-Colab-Client-Agent", "colab-cli")
                .query(&[
                    ("authuser", "0"),
                    ("colab-runtime-proxy-token", a.runtime_proxy_info.token.as_str()),
                ])
                .json(&payload)
                .timeout(Duration::from_secs(120))
                .send()
                .await;
            let resp = match send_res {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(anyhow!("上传 {remote_path} 网络请求失败: {e}"));
                    continue;
                }
            };
            let st = resp.status();
            if st.is_success() {
                return Ok(());
            }
            let t = resp.text().await.unwrap_or_default();
            if st == StatusCode::UNAUTHORIZED || st == StatusCode::FORBIDDEN {
                return Err(anyhow!(
                    "上传 {remote_path} 失败 ProxyAuth HTTP {st}: {}",
                    truncate(&t, 200)
                ));
            }
            if st.is_server_error() && attempt + 1 < 3 {
                last_err = Some(anyhow!(
                    "上传 {remote_path} 失败 HTTP {st}: {}",
                    truncate(&t, 200)
                ));
                continue;
            }
            return Err(anyhow!(
                "上传 {remote_path} 失败 HTTP {st}: {}",
                truncate(&t, 200)
            ));
        }
        Err(last_err.unwrap_or_else(|| anyhow!("上传 {remote_path} 失败")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn test_decode_contents_base64_multiline_whitespace() {
        let orig = r#"{"status":"training","current_epoch":3,"loss":0.42}"#;
        let b64 = base64::engine::general_purpose::STANDARD.encode(orig.as_bytes());
        let wrapped = b64
            .as_bytes()
            .chunks(10)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect::<Vec<_>>()
            .join("\r\n  ");
        let decoded = decode_contents_base64(&wrapped).expect("multiline base64 should decode");
        assert_eq!(decoded, orig);
    }

    #[tokio::test]
    async fn test_tun_request_401_skips_stale_file_token_and_refreshes_oauth() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let auth_hits = Arc::new(AtomicUsize::new(0));
        let refresh_hits = Arc::new(AtomicUsize::new(0));
        let auth_hits_srv = auth_hits.clone();
        let refresh_hits_srv = refresh_hits.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let ah = auth_hits_srv.clone();
                let rh = refresh_hits_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let mut n = 0usize;
                    loop {
                        let Ok(r) = sock.read(&mut buf[n..]).await else {
                            return;
                        };
                        if r == 0 {
                            break;
                        }
                        n += r;
                        if let Some(hdr_end) = buf[..n].windows(4).position(|w| w == b"\r\n\r\n") {
                            let hdr_str = String::from_utf8_lossy(&buf[..hdr_end]);
                            let content_len = hdr_str
                                .lines()
                                .find_map(|l| {
                                    let (k, v) = l.split_once(':')?;
                                    if k.eq_ignore_ascii_case("content-length") {
                                        v.trim().parse::<usize>().ok()
                                    } else {
                                        None
                                    }
                                })
                                .unwrap_or(0);
                            if n >= hdr_end + 4 + content_len {
                                break;
                            }
                        }
                    }
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first_line = req.lines().next().unwrap_or("");
                    let (status, ctype, body) = if first_line.starts_with("POST /oauth2/token") {
                        rh.fetch_add(1, Ordering::SeqCst);
                        (
                            "200 OK",
                            "application/json",
                            r#"{"access_token":"fresh-oauth-token","expires_in":3600}"#.to_string(),
                        )
                    } else if first_line.starts_with("GET /tun/m/assignments") {
                        ah.fetch_add(1, Ordering::SeqCst);
                        let lower = req.to_ascii_lowercase();
                        if lower.contains("authorization: bearer fresh-oauth-token") {
                            (
                                "200 OK",
                                "application/json",
                                ")]}'\n{\"assignments\":[{\"endpoint\":\"m-s-test01\",\"accelerator\":\"T4\",\"runtimeProxyInfo\":{\"token\":\"ptok\",\"url\":\"http://127.0.0.1\"}}]}"
                                    .to_string(),
                            )
                        } else {
                            ("401 Unauthorized", "text/plain", "revoked token".to_string())
                        }
                    } else {
                        ("404 Not Found", "text/plain", "not found".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp_dir = std::env::temp_dir().join(format!("pilot_colab_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let tok_file = tmp_dir.join("token.json");
        let tok_json = serde_json::json!({
            "token": "stale-static-token",
            "expiry": "2099-01-01T00:00:00Z",
            "refresh_token": "rt-test",
            "token_uri": format!("http://{addr}/oauth2/token"),
            "client_id": "cid",
            "client_secret": "csec"
        });
        std::fs::write(&tok_file, serde_json::to_string(&tok_json).unwrap()).unwrap();

        let client = ColabClient::new_for_test(format!("http://{addr}"), tok_file);
        let list = client
            .list_assignments()
            .await
            .expect("401 should trigger OAuth refresh and retry");
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].endpoint, "m-s-test01");
        assert_eq!(refresh_hits.load(Ordering::SeqCst), 1);
        assert_eq!(auth_hits.load(Ordering::SeqCst), 2);

        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_parse_token_expiry_utc_rfc3339_offsets_and_naive() {
        let z = ColabClient::parse_token_expiry_utc("2026-11-11T05:00:00Z").unwrap();
        let frac_z = ColabClient::parse_token_expiry_utc("2026-11-11T05:00:00.123456Z").unwrap();
        let plus8 = ColabClient::parse_token_expiry_utc("2026-11-11T13:00:00+08:00").unwrap();
        let frac_offset =
            ColabClient::parse_token_expiry_utc("2026-11-11T05:00:00.987654+00:00").unwrap();
        let naive = ColabClient::parse_token_expiry_utc("2026-11-11T05:00:00").unwrap();

        assert_eq!(z.timestamp(), plus8.timestamp());
        assert_eq!(z.timestamp(), naive.timestamp());
        assert_eq!(z.timestamp(), frac_z.timestamp());
        assert_eq!(z.timestamp(), frac_offset.timestamp());
        assert!(ColabClient::parse_token_expiry_utc("not-a-timestamp").is_none());
    }

    #[tokio::test]
    async fn test_dual_source_keyring_fallback_and_refresh_writeback_without_creating_file() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 4096];
                    let Ok(n) = sock.read(&mut buf).await else {
                        return;
                    };
                    let req = String::from_utf8_lossy(&buf[..n]);
                    let first_line = req.lines().next().unwrap_or("");
                    let (status, body) = if first_line.starts_with("POST /oauth2/token") {
                        (
                            "200 OK",
                            r#"{"access_token":"keyring-refreshed-tok","expires_in":3600}"#
                                .to_string(),
                        )
                    } else if first_line.starts_with("GET /tun/m/ccu-info") {
                        let lower = req.to_ascii_lowercase();
                        if lower.contains("authorization: bearer keyring-refreshed-tok") {
                            ("200 OK", r#"{"currentBalance":42.5}"#.to_string())
                        } else {
                            ("401 Unauthorized", "expired".to_string())
                        }
                    } else {
                        ("404 Not Found", "not found".to_string())
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp_dir = std::env::temp_dir().join(format!("pilot_kr_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let missing_tok_file = tmp_dir.join("token.json");
        let keyring_store = Arc::new(std::sync::Mutex::new(Some(
            serde_json::json!({
                "token": "expired-kr-tok",
                "expiry": "2020-01-01T00:00:00Z",
                "refresh_token": "rt-keyring",
                "token_uri": format!("http://{addr}/oauth2/token"),
                "client_id": "cid",
                "client_secret": "csec",
                "scopes": ["https://www.googleapis.com/auth/colaboratory"],
                "universe_domain": "googleapis.com"
            })
            .to_string(),
        )));

        let client = ColabClient::new_for_test_with_keyring(
            format!("http://{addr}"),
            missing_tok_file.clone(),
            keyring_store.clone(),
        );
        let ccu = client
            .ccu_info()
            .await
            .expect("missing token.json should fall back to keyring and refresh");
        assert_eq!(ccu.balance, 42.5);
        assert!(
            !missing_tok_file.exists(),
            "keyring refresh must NOT recreate plaintext token.json"
        );
        let updated_raw = keyring_store.lock().unwrap().clone().unwrap();
        let updated_val: Value = serde_json::from_str(&updated_raw).unwrap();
        assert_eq!(updated_val["token"].as_str(), Some("keyring-refreshed-tok"));
        assert_eq!(
            updated_val["universe_domain"].as_str(),
            Some("googleapis.com"),
            "flattened extra OAuth fields must be preserved"
        );

        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[tokio::test]
    async fn test_contents_upload_sends_proxy_headers_and_retries_503() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let put_hits = Arc::new(AtomicUsize::new(0));
        let put_hits_srv = put_hits.clone();

        let srv = tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else {
                    break;
                };
                let hits = put_hits_srv.clone();
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 8192];
                    let mut n = 0usize;
                    loop {
                        let Ok(r) = sock.read(&mut buf[n..]).await else {
                            return;
                        };
                        if r == 0 {
                            break;
                        }
                        n += r;
                        if let Some(hdr_end) = buf[..n].windows(4).position(|w| w == b"\r\n\r\n") {
                            let hdr_str = String::from_utf8_lossy(&buf[..hdr_end]);
                            let content_len = hdr_str
                                .lines()
                                .find_map(|l| {
                                    let (k, v) = l.split_once(':')?;
                                    k.eq_ignore_ascii_case("content-length")
                                        .then(|| v.trim().parse::<usize>().ok())
                                        .flatten()
                                })
                                .unwrap_or(0);
                            if n >= hdr_end + 4 + content_len {
                                break;
                            }
                        }
                    }
                    let req = String::from_utf8_lossy(&buf[..n]).to_ascii_lowercase();
                    let attempt = hits.fetch_add(1, Ordering::SeqCst) + 1;
                    let has_hdr = req.contains("x-colab-runtime-proxy-token: proxy-secret-tok")
                        && req.contains("x-colab-client-agent: colab-cli");
                    let (status, body) = if !has_hdr {
                        ("403 Forbidden", "missing proxy header")
                    } else if attempt == 1 {
                        ("503 Service Unavailable", "container booting")
                    } else {
                        ("201 Created", "{}")
                    };
                    let resp = format!(
                        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                });
            }
        });

        let tmp_dir = std::env::temp_dir().join(format!("pilot_up_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let client = ColabClient::new_for_test(format!("http://{addr}"), tmp_dir.join("tok.json"));
        let a = Assignment {
            accelerator: "G4".into(),
            endpoint: "m-s-g4test".into(),
            variant: Value::String("GPU".into()),
            machine_shape: Value::String("STANDARD".into()),
            runtime_proxy_info: RuntimeProxyInfo {
                token: "proxy-secret-tok".into(),
                token_expires_in_seconds: 3600,
                url: format!("http://{addr}"),
            },
        };

        client
            .contents_upload(&a, "content/.colab_pilot_daemon.py", b"print('ok')")
            .await
            .expect("contents_upload should send X-Colab-Runtime-Proxy-Token and recover from 503");
        assert_eq!(put_hits.load(Ordering::SeqCst), 2);

        srv.abort();
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
