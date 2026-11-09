//! Google Colab 服务端 REST 客户端（与 google-colab-cli 共用 OAuth 凭证 ~/.config/colab-cli/token.json）
//!
//! 实现了：
//! - OAuth2 access_token 读取与 refresh_token 自动续期（仅内存缓存，不改写 CLI 的 token.json）
//! - `/tun/m/ccu-info`      账户计算单元余额与计费速率
//! - `/tun/m/assignments`   服务端真实计费实例列表（删除核验的唯一可信来源）
//! - `/tun/m/unassign/{ep}` 服务端释放实例（停止计费）
//! - Jupyter Contents API  读写云端 /content 文件（遥测快照读取、密钥与脚本上传）

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::{Method, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

const COLAB_DOMAIN: &str = "https://colab.research.google.com";
const XSSI_PREFIX: &str = ")]}'\n";
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";

#[derive(Deserialize)]
struct TokenFile {
    token: Option<String>,
    refresh_token: String,
    token_uri: Option<String>,
    client_id: String,
    client_secret: String,
    expiry: Option<String>,
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
    cached: Mutex<Option<(String, Instant)>>,
}

pub fn token_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".config")
        .join("colab-cli")
        .join("token.json")
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
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
            cached: Mutex::new(None),
        }
    }

    pub fn is_authorized() -> bool {
        token_path().exists()
    }

    async fn access_token(&self) -> Result<String> {
        let mut guard = self.cached.lock().await;
        if let Some((tok, exp)) = guard.as_ref() {
            if Instant::now() + Duration::from_secs(300) < *exp {
                return Ok(tok.clone());
            }
        }

        let raw = std::fs::read_to_string(token_path()).with_context(|| {
            "未找到 Colab 授权文件 ~/.config/colab-cli/token.json，请先在终端运行一次 `colab status` 完成 Google 登录"
        })?;
        let tf: TokenFile = serde_json::from_str(&raw).context("token.json 格式无效")?;

        // 1. CLI 已缓存且仍有效的 access_token
        if let (Some(tok), Some(expiry)) = (&tf.token, &tf.expiry) {
            let core = expiry.trim_end_matches('Z');
            let core = core.split('.').next().unwrap_or(core);
            if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(core, "%Y-%m-%dT%H:%M:%S") {
                let remaining = (dt.and_utc() - chrono::Utc::now()).num_seconds();
                if remaining > 300 {
                    *guard = Some((tok.clone(), Instant::now() + Duration::from_secs(remaining as u64)));
                    return Ok(tok.clone());
                }
            }
        }

        // 2. 使用 refresh_token 静默续期
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
        let r: RefreshResp = serde_json::from_str(&text)?;
        let ttl = r.expires_in.unwrap_or(3600);
        *guard = Some((r.access_token.clone(), Instant::now() + Duration::from_secs(ttl)));
        Ok(r.access_token)
    }

    async fn tun_request(&self, method: Method, path: &str, xsrf: Option<&str>) -> Result<String> {
        let token = self.access_token().await?;
        let url = format!("{COLAB_DOMAIN}{path}");
        let mut last_err = None;
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(Duration::from_millis(350)).await;
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
                tokio::time::sleep(Duration::from_millis(300)).await;
            }
            let send_res = self
                .http
                .get(&url)
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
            if resp.status() == StatusCode::NOT_FOUND {
                return Ok(None);
            }
            if resp.status().is_server_error() && attempt == 0 {
                last_err = Some(anyhow!("Contents API HTTP {}", resp.status()));
                continue;
            }
            if !resp.status().is_success() {
                return Err(anyhow!("Contents API HTTP {}", resp.status()));
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
                Value::String(s) if fmt == "base64" => {
                    let bytes = base64::engine::general_purpose::STANDARD.decode(s.trim())?;
                    String::from_utf8_lossy(&bytes).to_string()
                }
                Value::String(s) => s,
                Value::Null => String::new(),
                other => other.to_string(),
            }));
        }
        Err(last_err.unwrap_or_else(|| anyhow!("Contents API 读取超时")))
    }

    /// 以 base64 上传本地字节到云端路径（与 colab upload 行为一致）
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
        let resp = self
            .http
            .put(Self::contents_url(a, remote_path))
            .query(&[
                ("authuser", "0"),
                ("colab-runtime-proxy-token", a.runtime_proxy_info.token.as_str()),
            ])
            .json(&payload)
            .timeout(Duration::from_secs(120))
            .send()
            .await?;
        if !resp.status().is_success() {
            let st = resp.status();
            let t = resp.text().await.unwrap_or_default();
            return Err(anyhow!("上传 {remote_path} 失败 HTTP {st}: {}", truncate(&t, 200)));
        }
        Ok(())
    }
}
