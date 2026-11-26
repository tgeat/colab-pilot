"""Opaque-box E2E test suite for colab-pilot Python/Shell/Interop issues (#39-#100).

Covers 22 Python/Shell/Interop issues across Tiers 1-4 (29 tests total) with zero
external network calls, isolated HOME/USERPROFILE/RCLONE_CONFIG/PATH/PYTHONPATH,
loopback ThreadingHTTPServer mocks, Git Bash (`C:\\Program Files\\Git\\bin\\bash.exe`),
and temporary `rustc` `std::fs::File::lock()` worker for `#94`.

IMPORTANT: `tests/e2e/__init__.py` MUST NOT exist so `python -m unittest discover -s tests`
remains isolated from `tests/e2e/`.
"""

from __future__ import annotations

import contextlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / ".agents" / "skills"
GATEWAY_SCRIPT = SKILLS_DIR / "colab-llm-bluegreen" / "scripts" / "llm_local_gateway.py"
MANAGER_SCRIPT = SKILLS_DIR / "colab-llm-bluegreen" / "scripts" / "llm_bluegreen_manager.py"
SIDECAR_SCRIPT = SKILLS_DIR / "colab-llm-bluegreen" / "scripts" / "llm_metrics_sidecar.py"
RUNNER_SCRIPT = SKILLS_DIR / "colab-dl-relay" / "scripts" / "dl_relay_runner.py"
ADAPTER_SCRIPT = SKILLS_DIR / "colab-dl-relay" / "scripts" / "colab_relay_adapter.py"
SMOKE_SCRIPT = SKILLS_DIR / "colab-dl-relay" / "scripts" / "smoke_test_dl.py"
TRAIN_SCRIPT = REPO_ROOT / "train.py"
RELAY_SH_SCRIPT = REPO_ROOT / "relay_rclone.sh"
TELEMETRY_SCRIPT = REPO_ROOT / "src-tauri" / "remote" / "telemetry_daemon.py"


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_git_bash() -> Optional[str]:
    if os.name == "nt":
        git_bash = r"C:\Program Files\Git\bin\bash.exe"
        if os.path.exists(git_bash):
            return git_bash
        git_bash_x86 = r"C:\Program Files (x86)\Git\bin\bash.exe"
        if os.path.exists(git_bash_x86):
            return git_bash_x86
    cand = shutil.which("bash")
    if cand and "windowsapps" not in cand.lower():
        return cand
    return None


def find_real_rclone() -> Optional[str]:
    cand = shutil.which("rclone")
    if cand and os.path.exists(cand):
        return cand
    ext = ".exe" if os.name == "nt" else ""
    home_cand = os.path.expanduser(f"~/.local/bin/rclone{ext}")
    if os.path.exists(home_cand):
        return home_cand
    return None


def to_unix_path(p: str) -> str:
    ap = os.path.abspath(p)
    if os.name == "nt" and len(ap) >= 2 and ap[1] == ":":
        drive = ap[0].lower()
        rest = ap[2:].replace("\\", "/")
        return f"/{drive}{rest}"
    return ap.replace("\\", "/")


COLAB_SHIM_PY = r'''import json
import os
import sys
import time

plan_path = os.environ.get("MOCK_COLAB_PLAN", "")
log_path = os.environ.get("MOCK_COLAB_LOG", "")
state_path = os.environ.get("MOCK_COLAB_STATE", "")

plan = {}
if plan_path and os.path.exists(plan_path):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

argv = sys.argv[1:]
sub = argv[0] if argv else ""

stdin_data = ""
if sub == "exec" and "-f" not in argv and not sys.stdin.isatty():
    try:
        stdin_data = sys.stdin.read()
    except Exception:
        stdin_data = ""

def get_flag(flags):
    for fl in flags:
        if fl in argv:
            idx = argv.index(fl)
            if idx + 1 < len(argv):
                return argv[idx + 1]
    return ""

sess = get_flag(["-s", "--session"])
if not sess and sub == "stop" and len(argv) >= 2 and not argv[-1].startswith("-"):
    sess = argv[-1]

is_rclone_probe = (sub == "exec" and "--timeout" in argv and "600" in argv and "RCLONE_READY" in stdin_data)
is_drain_probe = (sub == "exec" and "DRAIN_STATUS" in stdin_data)

# Update stateful counter per subcommand (probes use separate counters so they don't consume exec steps)
state = {}
if state_path and os.path.exists(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {}

if is_rclone_probe:
    entry = {"rc": 0, "stdout": "RCLONE_READY\n", "stderr": ""}
elif is_drain_probe:
    drain_steps = plan.get(f"drain:{sess}") or plan.get("drain") or []
    if drain_steps:
        d_idx = min(int(state.get(f"_drain:{sess}", 0)), len(drain_steps) - 1)
        state[f"_drain:{sess}"] = d_idx + 1
        if state_path:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f)
        entry = drain_steps[d_idx]
    else:
        entry = None
else:
    call_idx = int(state.get(sub, 0))
    state[sub] = call_idx + 1
    key_sess = f"{sub}:{sess}" if sess else sub
    sess_call_idx = int(state.get(key_sess, 0))
    state[key_sess] = sess_call_idx + 1

    if state_path:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    def choose_step(steps):
        if not steps:
            return None
        idx = min(sess_call_idx, len(steps) - 1) if sess and f"{sub}:{sess}" in plan else min(call_idx, len(steps) - 1)
        return steps[idx]

    steps = plan.get(f"{sub}:{sess}") or plan.get(sub) or []
    entry = choose_step(steps) if isinstance(steps, list) else steps

# Track active sessions if enabled
active_set = set(state.get("_active_sessions", []))

if log_path:
    rec = {
        "argv": argv,
        "sub": sub,
        "session": sess,
        "stdin_len": len(stdin_data),
        "stdin_preview": stdin_data[:240],
        "env_round": os.environ.get("RELAY_ROUND", ""),
        "col_retries": os.environ.get("COLAB_CLI_ASSIGN_RETRIES", ""),
        "col_timeout": os.environ.get("COLAB_CLI_REQUEST_TIMEOUT", ""),
    }
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

rc = 0
stdout = ""
stderr = ""
if isinstance(entry, dict):
    rc = int(entry.get("rc", 0))
    stdout = entry.get("stdout", "")
    stderr = entry.get("stderr", "")
    delay = float(entry.get("sleep", 0.0) or 0.0)
    if delay > 0:
        time.sleep(delay)
    # Optional file side effects (e.g. simulating remote done flag or weights)
    for wf in entry.get("write_files", []) or []:
        wp = wf["path"]
        os.makedirs(os.path.dirname(os.path.abspath(wp)), exist_ok=True)
        with open(wp, "w", encoding="utf-8") as f:
            f.write(wf.get("content", ""))
    if entry.get("run_stdin_py"):
        # Execute the uploaded python stdin snippet with rewritten environment
        exec_env = os.environ.copy()
        for k, v in (entry.get("env") or {}).items():
            exec_env[k] = str(v)
        ns = {"__name__": "__main__"}
        # Extract env markers from stdin_data if present
        import re
        m_round = re.search(r'"RELAY_ROUND":\s*"(\d+)"', stdin_data)
        if m_round:
            stdout += f"OBSERVED_RELAY_ROUND={m_round.group(1)}\n"
        m_gd = re.search(r'"GDRIVE_DIR":\s*"([^"]+)"', stdin_data)
        if m_gd:
            stdout += f"OBSERVED_GDRIVE_DIR={m_gd.group(1)}\n"
else:
    if sub == "new":
        if sess:
            active_set.add(sess)
            state["_active_sessions"] = sorted(active_set)
            if state_path:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f)
        stdout = f"Session [{sess}] READY\n"
    elif sub == "stop":
        if sess in active_set:
            active_set.discard(sess)
            state["_active_sessions"] = sorted(active_set)
            if state_path:
                with open(state_path, "w", encoding="utf-8") as f:
                    json.dump(state, f)
        stdout = f"Stopped [{sess}]\n"
    elif sub == "status":
        if sess and sess in active_set:
            stdout = f"[{sess}] active Hardware: T4\n"
        else:
            rc = 1
            stderr = f"No active session: {sess}\n"
    elif sub == "sessions":
        stdout = "\n".join(f"[{s}] active" for s in sorted(active_set))
        if stdout:
            stdout += "\n"
    elif sub == "upload":
        stdout = "Uploaded\n"
    elif sub == "install":
        stdout = "Installation Complete\n"
    elif sub == "exec":
        import re
        m_round = re.search(r'"RELAY_ROUND":\s*"(\d+)"', stdin_data)
        if m_round:
            stdout += f"OBSERVED_RELAY_ROUND={m_round.group(1)}\n"
        m_gd = re.search(r'"GDRIVE_DIR":\s*"([^"]+)"', stdin_data)
        if m_gd:
            stdout += f"OBSERVED_GDRIVE_DIR={m_gd.group(1)}\n"
        if "SLOT_READY" in stdin_data or "llm_metrics_sidecar.py" in stdin_data:
            stdout += 'SLOT_READY {"slot": "blue", "role": "active", "instant_tps": 45.0, "ttft_ms": 60.0, "total_tokens": 16}\n'
        elif "DRAIN_STATUS" in stdin_data:
            stdout += 'DRAIN_STATUS {"active_streams": 0, "active_requests": 0, "queried": true}\n'
        else:
            stdout += "CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n"

if stdout:
    sys.stdout.write(stdout)
    sys.stdout.flush()
if stderr:
    sys.stderr.write(stderr)
    sys.stderr.flush()
sys.exit(rc)
'''


RCLONE_SHIM_PY = r'''import json
import os
import subprocess
import sys

plan_path = os.environ.get("MOCK_RCLONE_PLAN", "")
state_path = os.environ.get("MOCK_RCLONE_STATE", "")
log_path = os.environ.get("MOCK_RCLONE_LOG", "")
real_rclone = os.environ.get("MOCK_REAL_RCLONE", "")

argv = sys.argv[1:]
# Find subcommand (skip flags like --config xxx, --retries=5, etc.)
sub = ""
i = 0
while i < len(argv):
    tok = argv[i]
    if tok == "--config":
        i += 2
        continue
    if tok.startswith("-"):
        i += 1
        continue
    sub = tok
    break

plan = {}
if plan_path and os.path.exists(plan_path):
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

state = {}
if state_path and os.path.exists(state_path):
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception:
        state = {}

call_idx = int(state.get(sub, 0))
state[sub] = call_idx + 1
if state_path:
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)

if log_path:
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(json.dumps({"sub": sub, "argv": argv}, ensure_ascii=False) + "\n")

steps = plan.get(sub)
if isinstance(steps, list) and steps:
    entry = steps[min(call_idx, len(steps) - 1)]
    if isinstance(entry, dict) and not entry.get("delegate_real"):
        if entry.get("stdout"):
            sys.stdout.write(entry["stdout"])
        if entry.get("stderr"):
            sys.stderr.write(entry["stderr"])
        sys.exit(int(entry.get("rc", 0)))

if real_rclone and os.path.exists(real_rclone):
    res = subprocess.run([real_rclone] + argv)
    sys.exit(res.returncode)

sys.exit(0)
'''


FAKE_COLAB_CLI_INIT = ""
FAKE_COLAB_CLI_COMMON_INIT = ""

FAKE_COLAB_CLI_STATE = r'''import json
import os
import types
import urllib.request

def _server_url():
    return os.environ["MOCK_COLAB_SERVER_URL"].rstrip("/")

class _Client:
    def list_assignments(self):
        with urllib.request.urlopen(f"{_server_url()}/tun/m/assignments", timeout=5) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        out = []
        for it in items:
            proxy = types.SimpleNamespace(
                url=it.get("url", ""),
                token=it.get("token", ""),
            )
            out.append(types.SimpleNamespace(
                endpoint=it.get("endpoint", ""),
                runtime_proxy_info=proxy,
            ))
        return out

client = _Client()
'''


FAKE_COLAB_CLI_RUNTIME = r'''import http.client
import json
import os
import urllib.error
import urllib.request

def _server_url():
    return os.environ["MOCK_COLAB_SERVER_URL"].rstrip("/")

class ColabRuntime:
    def __init__(self, url, token, kernel_id=None, session_id=None):
        self.url = (url or "").rstrip("/")
        self.token = token or ""
        self.kernel_id = kernel_id
        self.session_id = session_id
        self._stopped = False

    def _ensure_kernel(self):
        if self.kernel_id:
            return self.kernel_id
        req = urllib.request.Request(
            f"{self.url}/api/kernels",
            data=json.dumps({"name": "python3"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Colab-Runtime-Proxy-Token": self.token,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        self.kernel_id = data.get("id")
        return self.kernel_id

    def execute_code(self, code, output_hook=None, timeout=300):
        self._ensure_kernel()
        # Check if this is gateway's remote proxy snippet targeting a real local sidecar port
        real_sidecar_port = os.environ.get("MOCK_REAL_SIDECAR_PORT", "").strip()
        if "__PILOT_META__:" in code and real_sidecar_port:
            ns = {}
            exec_code = code.replace('http.client.HTTPConnection("127.0.0.1", ', f'http.client.HTTPConnection("127.0.0.1", ')
            import io, contextlib, re
            # Rewrite remote_port in build_remote_proxy_code if requested
            exec_code = re.sub(
                r'HTTPConnection\("127\.0\.0\.1",\s*\d+',
                f'HTTPConnection("127.0.0.1", {int(real_sidecar_port)}',
                exec_code,
            )
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(exec_code, {"__name__": "__main__"})
            txt = buf.getvalue()
            if output_hook and txt:
                output_hook({"output_type": "stream", "text": txt})
            return

        payload = json.dumps({
            "url": self.url,
            "token": self.token,
            "kernel_id": self.kernel_id,
            "code": code,
            "is_pass": code.strip() == "pass",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.url}/mock/execute",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-Colab-Runtime-Proxy-Token": self.token,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/event-stream" in ctype:
                    while True:
                        line = resp.readline()
                        if not line:
                            break
                        if output_hook:
                            output_hook({"output_type": "stream", "text": line.decode("utf-8", "replace")})
                    return
                body = resp.read().decode("utf-8", "replace")
                obj = json.loads(body) if body.strip() else {}
                for out in obj.get("outputs", []):
                    if output_hook:
                        output_hook(out)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {e.code}: {err_body}") from e

    def stop(self):
        self._stopped = True
'''


FAKE_TRANSFORMERS_INIT = r'''class TrainerCallback:
    pass
'''


RUST_LOCK_WORKER_RS = r'''use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process;
use std::thread;
use std::time::Duration;

fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while i < b.len() && b[i].is_ascii_whitespace() {
        i += 1;
    }
    i
}

fn scan_string(b: &[u8], start: usize) -> Result<usize, String> {
    if b.get(start) != Some(&b'"') {
        return Err("expected string quote".into());
    }
    let mut i = start + 1;
    while i < b.len() {
        match b[i] {
            b'\\' => i += 2,
            b'"' => return Ok(i + 1),
            _ => i += 1,
        }
    }
    Err("unterminated string".into())
}

fn scan_value(b: &[u8], start: usize) -> Result<usize, String> {
    let i = skip_ws(b, start);
    if i >= b.len() {
        return Err("unexpected eof".into());
    }
    match b[i] {
        b'"' => scan_string(b, i),
        b'{' | b'[' => {
            let mut depth = 0usize;
            let mut pos = i;
            while pos < b.len() {
                match b[pos] {
                    b'"' => pos = scan_string(b, pos)?,
                    b'{' | b'[' => {
                        depth += 1;
                        pos += 1;
                    }
                    b'}' | b']' => {
                        depth -= 1;
                        pos += 1;
                        if depth == 0 {
                            return Ok(pos);
                        }
                    }
                    _ => pos += 1,
                }
            }
            Err("unterminated container".into())
        }
        _ => {
            let mut pos = i;
            while pos < b.len() && !matches!(b[pos], b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t') {
                pos += 1;
            }
            Ok(pos)
        }
    }
}

fn parse_top_object(raw: &str) -> Result<Vec<(String, String)>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }
    let b = trimmed.as_bytes();
    let mut i = skip_ws(b, 0);
    if b.get(i) != Some(&b'{') {
        return Err("root is not object".into());
    }
    i = skip_ws(b, i + 1);
    let mut entries = Vec::new();
    if b.get(i) == Some(&b'}') {
        return Ok(entries);
    }
    loop {
        i = skip_ws(b, i);
        let key_end = scan_string(b, i)?;
        let key_raw = &trimmed[i + 1..key_end - 1];
        i = skip_ws(b, key_end);
        if b.get(i) != Some(&b':') {
            return Err("expected ':'".into());
        }
        let val_start = skip_ws(b, i + 1);
        let val_end = scan_value(b, val_start)?;
        entries.push((key_raw.to_string(), trimmed[val_start..val_end].to_string()));
        i = skip_ws(b, val_end);
        match b.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => break,
            _ => return Err("expected ',' or '}'".into()),
        }
    }
    Ok(entries)
}

fn update_keep_alive_in_entry(entry_raw: &str, ka_json: &str) -> Result<String, String> {
    let mut fields = parse_top_object(entry_raw)?;
    let mut found = false;
    for (k, v) in fields.iter_mut() {
        if k == "keep_alive" {
            *v = ka_json.to_string();
            found = true;
        }
    }
    if !found {
        fields.push(("keep_alive".to_string(), ka_json.to_string()));
    }
    let body = fields
        .into_iter()
        .map(|(k, v)| format!("\"{}\": {}", k, v))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!("{{{}}}", body))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 8 {
        eprintln!("usage: worker <prof_file> <ready_flag> <hold_ms> <session_name> <enabled> <interval> <kernel_ping>");
        process::exit(2);
    }
    let prof_file = &args[1];
    let ready_flag = &args[2];
    let hold_ms: u64 = args[3].parse().unwrap_or(0);
    let target_session = &args[4];
    let enabled: bool = args[5] == "true";
    let interval: u64 = args[6].parse().unwrap_or(90);
    let kernel_ping: bool = args[7] == "true";

    if let Some(parent) = Path::new(prof_file).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let lock_path = format!("{}.lock", prof_file);
    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .open(&lock_path)
        .expect("open lock file");
    lock_file.lock().expect("acquire OS File::lock");

    if !ready_flag.is_empty() && ready_flag != "-" {
        fs::write(ready_flag, "LOCKED").expect("write ready flag");
    }
    if hold_ms > 0 {
        thread::sleep(Duration::from_millis(hold_ms));
    }

    let raw = if Path::new(prof_file).exists() {
        fs::read_to_string(prof_file).unwrap_or_default()
    } else {
        String::new()
    };
    let mut root_entries = parse_top_object(&raw).expect("parse root profiles json");
    let ka_json = format!(
        "{{\"enabled\": {}, \"interval_seconds\": {}, \"kernel_ping\": {}}}",
        enabled, interval, kernel_ping
    );
    let mut updated = false;
    for (k, v) in root_entries.iter_mut() {
        if k == target_session {
            *v = update_keep_alive_in_entry(v, &ka_json).expect("update entry");
            updated = true;
        }
    }
    if !updated {
        root_entries.push((
            target_session.clone(),
            format!("{{\"keep_alive\": {}}}", ka_json),
        ));
    }
    let serialized = format!(
        "{{{}}}",
        root_entries
            .into_iter()
            .map(|(k, v)| format!("\"{}\": {}", k, v))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let tmp_path = format!("{}.{}.rs.tmp", prof_file, process::id());
    {
        let mut f = fs::File::create(&tmp_path).expect("create tmp");
        f.write_all(serialized.as_bytes()).expect("write tmp");
        f.sync_all().ok();
    }
    fs::rename(&tmp_path, prof_file).expect("rename tmp -> prof_file");
    drop(lock_file);
}
'''


class MockColabGatewayServer:
    """Loopback ThreadingHTTPServer simulating Colab `/tun/m/*` and Jupyter `/api/kernels`."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.assignments: List[Dict[str, Any]] = []
        self.kernels_by_token: Dict[str, List[Dict[str, Any]]] = {}
        self.auth_errors_by_token: Dict[str, int] = {}
        self.llm_states_by_token: Dict[str, Dict[str, Any]] = {}
        self.stream_events: Dict[str, threading.Event] = {}
        self.stream_payloads: Dict[str, str] = {}
        self.keepalive_hits: List[str] = []
        self.kernel_list_hits: List[Dict[str, Any]] = []
        self.kernel_spawn_hits: List[Dict[str, Any]] = []
        self.pass_hits: List[Dict[str, Any]] = []
        self.execute_hits: List[Dict[str, Any]] = []
        self._spawn_counter = 0

        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                pass

            def _send_json(self, code: int, obj: Any) -> None:
                raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()
                self.close_connection = True

            def do_GET(self) -> None:
                path_only = self.path.split("?", 1)[0]
                if path_only == "/tun/m/assignments":
                    with outer.lock:
                        data = list(outer.assignments)
                    self._send_json(200, data)
                    return

                if path_only.startswith("/tun/m/keep-alive/"):
                    ep = path_only.split("/tun/m/keep-alive/", 1)[1]
                    with outer.lock:
                        outer.keepalive_hits.append(ep)
                    self._send_json(200, {"ok": True})
                    return

                tok = self.headers.get("X-Colab-Runtime-Proxy-Token", "")
                if path_only == "/api/kernels":
                    with outer.lock:
                        err_code = outer.auth_errors_by_token.get(tok)
                        outer.kernel_list_hits.append({"token": tok, "ts": time.monotonic()})
                        kernels = list(outer.kernels_by_token.get(tok, []))
                    if err_code:
                        self._send_json(err_code, {"error": "unauthorized"})
                        return
                    self._send_json(200, kernels)
                    return

                if path_only == "/api/contents/content/.colab_pilot_llm_state.json":
                    with outer.lock:
                        st = outer.llm_states_by_token.get(tok)
                    if st is None:
                        self._send_json(404, {"error": "not found"})
                        return
                    self._send_json(200, {"format": "json", "content": st})
                    return

                self._send_json(404, {"error": f"unknown GET {self.path}"})

            def do_POST(self) -> None:
                path_only = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                req = json.loads(raw.decode("utf-8", "replace") or "{}")
                tok = self.headers.get("X-Colab-Runtime-Proxy-Token", "") or req.get("token", "")

                if path_only == "/api/kernels":
                    with outer.lock:
                        outer._spawn_counter += 1
                        new_kid = f"spawned-kid-{outer._spawn_counter}"
                        outer.kernel_spawn_hits.append({"token": tok, "kernel_id": new_kid, "ts": time.monotonic()})
                        outer.kernels_by_token.setdefault(tok, []).append(
                            {"id": new_kid, "name": "python3", "execution_state": "idle"}
                        )
                    self._send_json(201, {"id": new_kid, "name": "python3", "execution_state": "idle"})
                    return

                if path_only == "/mock/execute":
                    with outer.lock:
                        err_code = outer.auth_errors_by_token.get(tok)
                    if err_code:
                        self._send_json(err_code, {"error": "token expired"})
                        return

                    kid = req.get("kernel_id")
                    is_pass = bool(req.get("is_pass"))
                    with outer.lock:
                        outer.execute_hits.append({"token": tok, "kernel_id": kid, "is_pass": is_pass, "ts": time.monotonic()})
                        if is_pass:
                            outer.pass_hits.append({"token": tok, "kernel_id": kid, "ts": time.monotonic()})
                        ev = outer.stream_events.get(tok)
                        custom_sse = outer.stream_payloads.get(tok)

                    if is_pass:
                        self._send_json(200, {"outputs": [{"output_type": "stream", "text": "KA_OK\n"}]})
                        return

                    # Streaming proxy simulation
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    meta = '__PILOT_META__:{"status": 200, "ctype": "text/event-stream"}\n'
                    self.wfile.write(meta.encode("utf-8"))
                    self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                    self.wfile.flush()
                    if ev is not None:
                        ev.wait(timeout=10.0)
                    tail = custom_sse or 'data: {"choices":[{"delta":{"content":" world"}}]}\n\ndata: [DONE]\n\n'
                    self.wfile.write(tail.encode("utf-8"))
                    self.wfile.flush()
                    return

                self._send_json(404, {"error": f"unknown POST {self.path}"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3.0)


class IsolatedPilotEnv:
    """Self-contained zero-network sandbox for subprocess & HTTP E2E tests."""

    def __init__(self, test_case: unittest.TestCase) -> None:
        self.tc = test_case
        self.td = tempfile.TemporaryDirectory(prefix="pilot_e2e_")
        self.root = Path(self.td.name)
        self.home = self.root / "home"
        self.cfg_dir = self.home / ".config" / "colab-cli"
        self.bin_dir = self.root / "bin"
        self.fake_site = self.root / "fake_site"
        self.fake_hf_site = self.root / "fake_hf_site"
        self.drive_root = self.root / "drive_storage"
        self.uv_tool_dir = self.root / "empty_uv_tools"
        self.workspace = self.root / "workspace"
        for d in (
            self.cfg_dir,
            self.bin_dir,
            self.fake_site / "colab_cli" / "common",
            self.fake_hf_site / "transformers",
            self.drive_root,
            self.uv_tool_dir,
            self.workspace,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Seed OAuth token.json so precheck never touches host keyring
        self.write_cfg("token.json", {"token": "mock-oauth", "refresh_token": "mock-refresh"})

        # Rclone alias config mapping gdrive: -> self.drive_root
        self.rclone_conf = self.root / "rclone_alias.conf"
        drive_norm = str(self.drive_root).replace("\\", "/")
        self.rclone_conf.write_text(f"[gdrive]\ntype = alias\nremote = {drive_norm}\n", encoding="utf-8")

        # Write colab & rclone shims
        self.colab_plan = self.root / "colab_plan.json"
        self.colab_state = self.root / "colab_state.json"
        self.colab_log = self.root / "colab_log.jsonl"
        self.colab_shim_py = self.bin_dir / "colab_shim.py"
        self.colab_shim_py.write_text(COLAB_SHIM_PY, encoding="utf-8")

        self.rclone_plan = self.root / "rclone_plan.json"
        self.rclone_state = self.root / "rclone_state.json"
        self.rclone_log = self.root / "rclone_log.jsonl"
        self.rclone_shim_py = self.bin_dir / "rclone_shim.py"
        self.rclone_shim_py.write_text(RCLONE_SHIM_PY, encoding="utf-8")

        self.real_rclone = find_real_rclone() or ""
        self._write_bin_wrappers()

        # Write fake_site/colab_cli & fake_hf_site/transformers
        (self.fake_site / "colab_cli" / "__init__.py").write_text(FAKE_COLAB_CLI_INIT, encoding="utf-8")
        (self.fake_site / "colab_cli" / "common" / "__init__.py").write_text(FAKE_COLAB_CLI_COMMON_INIT, encoding="utf-8")
        (self.fake_site / "colab_cli" / "common" / "state.py").write_text(FAKE_COLAB_CLI_STATE, encoding="utf-8")
        (self.fake_site / "colab_cli" / "runtime.py").write_text(FAKE_COLAB_CLI_RUNTIME, encoding="utf-8")
        (self.fake_hf_site / "transformers" / "__init__.py").write_text(FAKE_TRANSFORMERS_INIT, encoding="utf-8")

        self._procs: List[subprocess.Popen[Any]] = []
        self._servers: List[MockColabGatewayServer] = []

        # Ensure cleanup runs in reverse order: procs -> servers -> tempdir
        self.tc.addCleanup(self.cleanup)

    def _write_bin_wrappers(self) -> None:
        py_exe = sys.executable
        py_unix = to_unix_path(py_exe)
        colab_py_unix = to_unix_path(str(self.colab_shim_py))
        rclone_py_unix = to_unix_path(str(self.rclone_shim_py))

        if os.name == "nt":
            (self.bin_dir / "colab.cmd").write_text(
                f'@echo off\r\n"{py_exe}" -B "{self.colab_shim_py}" %*\r\n',
                encoding="utf-8",
            )
            (self.bin_dir / "rclone.cmd").write_text(
                f'@echo off\r\n"{py_exe}" -B "{self.rclone_shim_py}" %*\r\n',
                encoding="utf-8",
            )

        colab_sh = f'#!/usr/bin/env bash\nexec "{py_unix}" -B "{colab_py_unix}" "$@"\n'
        rclone_sh = f'#!/usr/bin/env bash\nexec "{py_unix}" -B "{rclone_py_unix}" "$@"\n'
        (self.bin_dir / "colab").write_bytes(colab_sh.encode("utf-8"))
        (self.bin_dir / "rclone").write_bytes(rclone_sh.encode("utf-8"))
        os.chmod(self.bin_dir / "colab", 0o755)
        os.chmod(self.bin_dir / "rclone", 0o755)

    def cleanup(self) -> None:
        for p in reversed(self._procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=3.0)
            for stream in (p.stdin, p.stdout, p.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
        for srv in reversed(self._servers):
            try:
                srv.close()
            except Exception:
                pass
        # Also kill any detached child recorded in .colab_dl_relay.pid
        for pid_file in self.root.rglob(".colab_dl_relay.pid"):
            try:
                pid_val = int(pid_file.read_text(encoding="utf-8").strip())
                if pid_val > 0 and pid_val != os.getpid():
                    if os.name == "nt":
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid_val)],
                            capture_output=True,
                            creationflags=0x08000000,
                            timeout=5,
                        )
                    else:
                        os.kill(pid_val, 9)
            except Exception:
                pass
        self.td.cleanup()

    def start_mock_server(self) -> MockColabGatewayServer:
        srv = MockColabGatewayServer()
        self._servers.append(srv)
        return srv

    def write_cfg(self, name: str, data: Any) -> Path:
        p = self.cfg_dir / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def read_cfg(self, name: str) -> Any:
        p = self.cfg_dir / name
        return json.loads(p.read_text(encoding="utf-8"))

    def set_colab_plan(self, plan: Dict[str, Any], reset_state: bool = True) -> None:
        self.colab_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        if reset_state and self.colab_state.exists():
            self.colab_state.unlink()

    def read_colab_log(self) -> List[Dict[str, Any]]:
        if not self.colab_log.exists():
            return []
        return [json.loads(ln) for ln in self.colab_log.read_text(encoding="utf-8").splitlines() if ln.strip()]

    def clear_colab_log(self) -> None:
        if self.colab_log.exists():
            self.colab_log.unlink()

    def set_rclone_plan(self, plan: Dict[str, Any], reset_state: bool = True) -> None:
        self.rclone_plan.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        if reset_state and self.rclone_state.exists():
            self.rclone_state.unlink()

    def build_env(
        self,
        *,
        mock_server: Optional[MockColabGatewayServer] = None,
        use_rclone_shim: bool = False,
        include_fake_site: bool = True,
        include_fake_hf: bool = False,
        extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        env = os.environ.copy()
        path_parts = [str(self.bin_dir)]
        if self.real_rclone:
            path_parts.append(os.path.dirname(self.real_rclone))
        path_parts.append(env.get("PATH", ""))

        py_paths: List[str] = []
        if include_fake_site:
            py_paths.append(str(self.fake_site))
        if include_fake_hf:
            py_paths.append(str(self.fake_hf_site))
        if env.get("PYTHONPATH"):
            py_paths.append(env["PYTHONPATH"])

        env.update({
            "HOME": str(self.home),
            "USERPROFILE": str(self.home),
            "APPDATA": str(self.root / "appdata"),
            "LOCALAPPDATA": str(self.root / "localappdata"),
            "XDG_DATA_HOME": str(self.root / "xdg_data"),
            "UV_TOOL_DIR": str(self.uv_tool_dir),
            "RCLONE_CONFIG": str(self.rclone_conf),
            "PATH": os.pathsep.join(path_parts),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "MOCK_COLAB_PLAN": str(self.colab_plan),
            "MOCK_COLAB_STATE": str(self.colab_state),
            "MOCK_COLAB_LOG": str(self.colab_log),
            "MOCK_RCLONE_PLAN": str(self.rclone_plan),
            "MOCK_RCLONE_STATE": str(self.rclone_state),
            "MOCK_RCLONE_LOG": str(self.rclone_log),
            "MOCK_REAL_RCLONE": "" if use_rclone_shim else self.real_rclone,
        })
        if use_rclone_shim:
            env["MOCK_REAL_RCLONE"] = self.real_rclone
        else:
            # Remove rclone.cmd/rclone shim when using real rclone directly
            for rn in ("rclone.cmd", "rclone"):
                rp = self.bin_dir / rn
                if rp.exists():
                    rp.unlink()

        if py_paths:
            env["PYTHONPATH"] = os.pathsep.join(py_paths)
        if mock_server is not None:
            env["MOCK_COLAB_SERVER_URL"] = mock_server.url
        if extra:
            env.update(extra)
        return env

    def restore_rclone_shim(self) -> None:
        self._write_bin_wrappers()

    def spawn_gateway(
        self,
        mock_server: MockColabGatewayServer,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Tuple[subprocess.Popen[str], int]:
        port = find_free_port()
        env = self.build_env(mock_server=mock_server, extra=extra_env)
        proc = subprocess.Popen(
            [sys.executable, "-B", str(GATEWAY_SCRIPT), "--port", str(port)],
            cwd=str(self.workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._procs.append(proc)
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2)
                raise RuntimeError(f"Gateway exited prematurely ({proc.returncode}): {out}\n{err}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/pilot/gateway", timeout=1.0) as resp:
                    if resp.status == 200:
                        return proc, port
            except Exception:
                time.sleep(0.05)
        raise TimeoutError("Timed out waiting for llm_local_gateway.py")

    def spawn_sidecar(
        self,
        slot: str = "blue",
        role: str = "active",
        model_name: str = "Qwen-Test",
        state_file: Optional[Path] = None,
    ) -> Tuple[subprocess.Popen[str], int, Path]:
        port = find_free_port()
        st_path = state_file or (self.root / f"sidecar_{slot}_{port}.json")
        env = self.build_env(include_fake_site=False)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-B",
                str(SIDECAR_SCRIPT),
                "--port",
                str(port),
                "--slot",
                slot,
                "--role",
                role,
                "--engine",
                "builtin",
                "--model-name",
                model_name,
                "--builtin-engine",
                "--state-file",
                str(st_path),
            ],
            cwd=str(self.workspace),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self._procs.append(proc)
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=2)
                raise RuntimeError(f"Sidecar exited prematurely ({proc.returncode}): {out}\n{err}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/pilot/health", timeout=1.0) as resp:
                    if resp.status == 200:
                        return proc, port, st_path
            except Exception:
                time.sleep(0.05)
        raise TimeoutError("Timed out waiting for llm_metrics_sidecar.py")


def _wait_until(predicate, timeout: float = 4.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _http_json(url: str, method: str = "GET", body: Any = None, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Any]:
    hdrs = dict(headers or {})
    raw_data = None
    if body is not None:
        raw_data = json.dumps(body).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=raw_data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            txt = resp.read().decode("utf-8", "replace")
            return resp.status, json.loads(txt) if txt.strip() else {}
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        e.close()
        try:
            return e.code, json.loads(txt) if txt.strip() else {}
        except Exception:
            return e.code, {"raw": txt}


def _compile_rust_lock_worker(build_dir: Path) -> Optional[Path]:
    rustc = shutil.which("rustc")
    if not rustc:
        return None
    rs_src = build_dir / "rust_lock_worker.rs"
    exe_name = "rust_lock_worker.exe" if os.name == "nt" else "rust_lock_worker"
    exe_path = build_dir / exe_name
    rs_src.write_text(RUST_LOCK_WORKER_RS, encoding="utf-8")
    res = subprocess.run(
        [rustc, str(rs_src), "-O", "-o", str(exe_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if res.returncode == 0 and exe_path.exists():
        return exe_path
    return None


class E2EIssuesTestSuite(unittest.TestCase):
    _rust_td: Optional[tempfile.TemporaryDirectory[str]] = None
    _rust_worker: Optional[Path] = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._rust_td = tempfile.TemporaryDirectory(prefix="pilot_e2e_rust_")
        cls._rust_worker = _compile_rust_lock_worker(Path(cls._rust_td.name))

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._rust_td is not None:
            cls._rust_td.cleanup()
            cls._rust_td = None
            cls._rust_worker = None

    def _require_rclone(self) -> str:
        rclone = find_real_rclone()
        if not rclone:
            self.skipTest("rclone binary not available")
        return rclone

    def _require_git_bash(self) -> str:
        bash = find_git_bash()
        if not bash:
            self.skipTest("Git Bash not available")
        probe = subprocess.run([bash, "--version"], capture_output=True, text=True, timeout=10)
        if probe.returncode != 0:
            self.skipTest("bash --version failed")
        return bash

    def _spawn_lock_worker(
        self,
        env: Dict[str, str],
        prof_file: Path,
        ready_flag: Path,
        hold_ms: int,
        session_name: str,
        enabled: bool,
        interval: int,
        kernel_ping: bool,
    ) -> subprocess.Popen[str]:
        if self._rust_worker is not None and self._rust_worker.exists():
            cmd = [
                str(self._rust_worker),
                str(prof_file),
                str(ready_flag),
                str(hold_ms),
                session_name,
                "true" if enabled else "false",
                str(interval),
                "true" if kernel_ping else "false",
            ]
        else:
            py_fallback = (
                "import errno, json, os, sys, time\n"
                "pf, rf, hm, sn, en, iv, kp = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]=='true', int(sys.argv[6]), sys.argv[7]=='true'\n"
                "os.makedirs(os.path.dirname(os.path.abspath(pf)), exist_ok=True)\n"
                "lf = open(pf + '.lock', 'a+b')\n"
                "try:\n"
                "    if os.name == 'nt':\n"
                "        import msvcrt; lf.seek(0)\n"
                "        while True:\n"
                "            try: msvcrt.locking(lf.fileno(), msvcrt.LK_NBLCK, 1); break\n"
                "            except OSError: time.sleep(0.01)\n"
                "    else:\n"
                "        import fcntl; fcntl.flock(lf.fileno(), fcntl.LOCK_EX)\n"
                "    if rf != '-': open(rf, 'w').write('LOCKED')\n"
                "    if hm > 0: time.sleep(hm / 1000.0)\n"
                "    data = json.load(open(pf, 'r', encoding='utf-8')) if os.path.exists(pf) and open(pf).read().strip() else {}\n"
                "    ent = data.get(sn) if isinstance(data.get(sn), dict) else {}\n"
                "    ent['keep_alive'] = {'enabled': en, 'interval_seconds': iv, 'kernel_ping': kp}\n"
                "    data[sn] = ent\n"
                "    tmp = f'{pf}.{os.getpid()}.pylock.tmp'\n"
                "    json.dump(data, open(tmp, 'w', encoding='utf-8'))\n"
                "    os.replace(tmp, pf)\n"
                "finally:\n"
                "    if os.name == 'nt':\n"
                "        import msvcrt; lf.seek(0); msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)\n"
                "    else:\n"
                "        import fcntl; fcntl.flock(lf.fileno(), fcntl.LOCK_UN)\n"
                "    lf.close()\n"
            )
            cmd = [
                sys.executable,
                "-c",
                py_fallback,
                str(prof_file),
                str(ready_flag),
                str(hold_ms),
                session_name,
                "true" if enabled else "false",
                str(interval),
                "true" if kernel_ping else "false",
            ]
        return subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    # ────────────────────────── TIER 1: Category-Partition (11 tests) ──────────────────────────

    def test_t1_f1_gateway_disabled_ordinary_vm_and_blue_slot_skipped(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {
            "tok-train": [{"id": "k-train", "name": "python3", "execution_state": "idle"}],
            "tok-blue": [{"id": "k-blue", "name": "python3", "execution_state": "idle"}],
            "tok-green": [{"id": "k-green", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "green_session": "llm-green",
            "slots": {
                "blue": {"session_name": "llm-blue", "role": "active"},
                "green": {"session_name": "llm-green", "role": "standby"},
            },
            "phase": "serving",
        })
        env_box.write_cfg("sessions.json", {
            "ordinary-train": {"endpoint": "m-s-train", "url": srv.url, "token": "tok-train", "kernel_id": "k-train"},
            "llm-blue": {"endpoint": "m-s-blue", "url": srv.url, "token": "tok-blue", "kernel_id": "k-blue"},
            "llm-green": {"endpoint": "m-s-green", "url": srv.url, "token": "tok-green", "kernel_id": "k-green"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "ordinary-train": {"boot_mode": "relay", "keep_alive": {"enabled": False, "interval_seconds": 90, "kernel_ping": True}},
            "llm-blue": {"boot_mode": "llm", "keep_alive": {"enabled": False, "interval_seconds": 90, "kernel_ping": True}},
            "llm-green": {"boot_mode": "llm", "keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
        })

        env_box.spawn_gateway(srv)
        ok = _wait_until(lambda: any(h["token"] == "tok-green" for h in srv.pass_hits), timeout=4.0)
        self.assertTrue(ok, "Enabled green slot must receive startup kernel pass")

        time.sleep(0.3)
        with srv.lock:
            self.assertEqual(srv.keepalive_hits, [], "Gateway must never call legacy /tun/m/keep-alive")
            kernel_toks = {h["token"] for h in srv.kernel_list_hits}
            pass_toks = {h["token"] for h in srv.pass_hits}
        self.assertNotIn("tok-train", kernel_toks)
        self.assertNotIn("tok-blue", kernel_toks)
        self.assertNotIn("tok-train", pass_toks)
        self.assertNotIn("tok-blue", pass_toks)
        self.assertIn("tok-green", kernel_toks)
        self.assertIn("tok-green", pass_toks)

    def test_t1_f1_gateway_kernel_ping_false_tunnel_only(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {
            "tok-blue": [{"id": "k-blue", "name": "python3", "execution_state": "idle"}],
            "tok-green": [{"id": "k-green", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "green_session": "llm-green",
            "phase": "serving",
        })
        env_box.write_cfg("sessions.json", {
            "llm-blue": {"endpoint": "m-s-blue", "url": srv.url, "token": "tok-blue", "kernel_id": "k-blue"},
            "llm-green": {"endpoint": "m-s-green", "url": srv.url, "token": "tok-green", "kernel_id": "k-green"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "llm-blue": {"keep_alive": {"enabled": False, "interval_seconds": 90, "kernel_ping": True}},
            "llm-green": {"keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": False}},
        })

        env_box.spawn_gateway(srv)
        ok = _wait_until(lambda: any(h["token"] == "tok-green" for h in srv.kernel_list_hits), timeout=4.0)
        self.assertTrue(ok, "Green slot with kernel_ping=False must still run GET /api/kernels runtime check")
        time.sleep(0.3)

        with srv.lock:
            self.assertEqual(srv.keepalive_hits, [])
            self.assertEqual(srv.pass_hits, [], "kernel_ping=False must NEVER run pass")
            self.assertEqual(srv.kernel_spawn_hits, [], "Must NEVER spawn kernel")
            kernel_toks = {h["token"] for h in srv.kernel_list_hits}
        self.assertEqual(kernel_toks, {"tok-green"})

    def test_t1_f1_gateway_standalone_vm_and_token_refresh_isolation(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.assignments = [
            {"endpoint": "m-s-standalone", "url": srv.url, "token": "fresh-st-tok"},
            {"endpoint": "m-s-paused-blue", "url": srv.url, "token": "fresh-blue-tok"},
        ]
        srv.auth_errors_by_token = {"stale-st-tok": 401}
        srv.kernels_by_token = {
            "fresh-st-tok": [{"id": "k-st-1", "name": "python3", "execution_state": "idle"}],
            "fresh-blue-tok": [{"id": "k-bl-1", "name": "python3", "execution_state": "idle"}],
            "stale-blue-tok": [{"id": "k-bl-1", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "prod",
            "active_slot": "blue",
            "active_session": "custom-blue-node",
            "blue_session": "custom-blue-node",
            "phase": "serving",
        })
        env_box.write_cfg("llm_api_keys.json", {
            "gateway_api_key": "sk-gw-main",
            "vms": {
                "custom-standalone-node": {
                    "vm_name": "custom-standalone-node",
                    "slot": "standalone",
                    "api_key": "sk-standalone-secret",
                    "local_port": find_free_port(),
                    "remote_port": 8000,
                    "model_name": "Qwen-Standalone",
                    "engine": "builtin",
                }
            },
        })
        env_box.write_cfg("sessions.json", {
            "custom-blue-node": {"endpoint": "m-s-paused-blue", "url": srv.url, "token": "stale-blue-tok", "kernel_id": "k-bl-1"},
            "custom-standalone-node": {"endpoint": "m-s-standalone", "url": srv.url, "token": "stale-st-tok", "kernel_id": "k-st-1"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "custom-blue-node": {"keep_alive": {"enabled": False, "interval_seconds": 90, "kernel_ping": True}},
            "custom-standalone-node": {"keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
        })

        _, gw_port = env_box.spawn_gateway(srv)
        ok = _wait_until(lambda: any(h["token"] == "fresh-st-tok" for h in srv.pass_hits), timeout=4.0)
        self.assertTrue(ok, "Standalone VM registered only in llm_api_keys.json must receive keep-alive pass")

        # Verify bearer routing and active chat token refresh isolation
        st_code, gw_info = _http_json(
            f"http://127.0.0.1:{gw_port}/pilot/gateway",
            headers={"Authorization": "Bearer sk-standalone-secret"},
        )
        self.assertEqual(st_code, 200)
        self.assertEqual(gw_info["session_name"], "custom-standalone-node")

        chat_code, chat_resp = _http_json(
            f"http://127.0.0.1:{gw_port}/v1/chat/completions",
            method="POST",
            headers={"Authorization": "Bearer sk-standalone-secret"},
            body={"model": "Qwen-Standalone", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(chat_code, 200)
        self.assertIn("hello", chat_resp["choices"][0]["message"]["content"])

        with srv.lock:
            self.assertEqual(srv.keepalive_hits, [])
            kernel_toks = {h["token"] for h in srv.kernel_list_hits}
            pass_toks = {h["token"] for h in srv.pass_hits}
        self.assertNotIn("stale-blue-tok", kernel_toks)
        self.assertNotIn("fresh-blue-tok", kernel_toks)
        self.assertNotIn("stale-blue-tok", pass_toks)
        self.assertNotIn("fresh-blue-tok", pass_toks)

    def test_t1_f1_gateway_reuses_live_kernel_without_post_spawn(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {
            "tok-blue": [{"id": "kid-live-1", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "phase": "serving",
        })
        env_box.write_cfg("sessions.json", {
            "llm-blue": {"endpoint": "m-s-blue", "url": srv.url, "token": "tok-blue", "kernel_id": "kid-live-1"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "llm-blue": {"keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
        })

        env_box.spawn_gateway(srv)
        ok = _wait_until(lambda: len(srv.pass_hits) >= 1, timeout=4.0)
        self.assertTrue(ok)
        time.sleep(0.2)
        with srv.lock:
            self.assertEqual(srv.kernel_spawn_hits, [], "Keep-alive must never POST /api/kernels when idle kernel exists")
            self.assertEqual([h["kernel_id"] for h in srv.pass_hits], ["kid-live-1"])

    def test_t1_f2_relay_retry_503_keeps_round1_and_401_fails_fast(self) -> None:
        env_box = IsolatedPilotEnv(self)
        train_py = env_box.workspace / "train.py"
        train_py.write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")

        # Subcase A: 503 retry keeps RELAY_ROUND=1
        env_box.set_colab_plan({
            "new": [
                {"rc": 1, "stderr": "HTTP 503 Service Unavailable: capacity full"},
                {"rc": 0, "stdout": "Session READY\n"},
            ],
            "status": [
                {"rc": 1, "stderr": "No active session"},
                {"rc": 1, "stderr": "No active session"},
            ],
            "sessions": [{"rc": 0, "stdout": ""}],
            "upload": [{"rc": 0, "stdout": "Uploaded"}],
            "exec": [{"rc": 0, "run_stdin_py": True, "stdout": "CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n"}],
            "stop": [{"rc": 0, "stdout": "Stopped"}],
        })
        env_box.set_rclone_plan({
            "lsf": [{"rc": 0, "stdout": ""}],
            "copyto": [{"rc": 0, "delegate_real": True}],
            "copy": [{"rc": 0, "delegate_real": True}],
        })
        # Pre-seed final weight in drive storage so download_and_verify succeeds after exec
        job_dir = env_box.drive_root / "colab_relay" / "exp-45"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "checkpoint.pt").write_bytes(b"final-weights-45")

        res_503 = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "exp-45",
                "--workspace",
                str(env_box.workspace),
                "--grab-mode",
                "budget",
                "--grab-retry-seconds",
                "0.05",
                "--reset-done-flag",
            ],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_503.returncode, 0, f"503 retry failed:\n{res_503.stdout}\n{res_503.stderr}")
        logs_a = env_box.read_colab_log()
        new_calls_a = [c for c in logs_a if c["sub"] == "new"]
        self.assertEqual(len(new_calls_a), 2)
        self.assertIn("OBSERVED_RELAY_ROUND=1", res_503.stdout)
        self.assertNotIn("OBSERVED_RELAY_ROUND=2", res_503.stdout)

        # Subcase B: 401 fails fast on first attempt
        env_box.clear_colab_log()
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 1, "stderr": "HTTP 401 Unauthorized: Not logged in"}],
        })
        res_401 = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "exp-45-auth",
                "--workspace",
                str(env_box.workspace),
                "--grab-mode",
                "budget",
                "--grab-retry-seconds",
                "0.05",
                "--reset-done-flag",
            ],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_401.returncode, 1)
        new_calls_b = [c for c in env_box.read_colab_log() if c["sub"] == "new"]
        self.assertEqual(len(new_calls_b), 1, "Fatal 401 error must fail fast without retrying")

    def test_t1_f2_relay_stop_failure_blocks_next_round_alloc(self) -> None:
        env_box = IsolatedPilotEnv(self)
        (env_box.workspace / "train.py").write_text("print('HANDOVER_REQUESTED')\n", encoding="utf-8")

        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": "CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\nTRAIN_EXIT_CODE=0\n"}],
            "stop": [{"rc": 1, "stderr": "HTTP 500 stop failed"}],
        })

        res = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "exp-46",
                "--workspace",
                str(env_box.workspace),
                "--reset-done-flag",
            ],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res.returncode, 1)
        logs = env_box.read_colab_log()
        new_calls = [c for c in logs if c["sub"] == "new"]
        self.assertEqual(len(new_calls), 1, "Must NEVER allocate Round 2 VM when old VM stop fails")
        combined = res.stdout + "\n" + res.stderr
        self.assertIn("严禁开启下一棒实例", combined)
        self.assertNotIn("本棒安全交接完毕", combined)

    def test_t1_f2_hf_trainer_callback_checkpoint_round2_resume(self) -> None:
        self._require_rclone()
        env_box = IsolatedPilotEnv(self)

        # Round 1 subprocess using real rclone alias remote
        r1_out = env_box.workspace / "r1_hf_out"
        ckpt10 = r1_out / "checkpoint-10"
        ckpt10.mkdir(parents=True, exist_ok=True)
        (ckpt10 / "trainer_state.json").write_text(json.dumps({"global_step": 10, "epoch": 1.0}), encoding="utf-8")

        r1_code = (
            "import importlib.util, os, types\n"
            f"spec = importlib.util.spec_from_file_location('adapter', {json.dumps(str(ADAPTER_SCRIPT))})\n"
            "adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)\n"
            "ctx = adapter.ColabRelayContext(\n"
            "    model_name='hf-e2e', framework='huggingface', total_epochs=3,\n"
            "    gdrive_dir='gdrive:colab_relay/hf-47',\n"
            f"    results_dir={json.dumps(str(env_box.workspace / 'r1_results'))},\n"
            f"    checkpoint_path={json.dumps(str(env_box.workspace / 'r1_missing.pt'))},\n"
            f"    hf_checkpoints_dir={json.dumps(str(r1_out))},\n"
            f"    done_flag_path={json.dumps(str(env_box.workspace / 'r1_done.flag'))},\n"
            "    smoke_test=False, round_num=1,\n"
            ")\n"
            f"ctx.state_file = {json.dumps(str(env_box.workspace / 'r1_state.json'))}\n"
            "cb = adapter.make_hf_trainer_callback(ctx)\n"
            f"args = types.SimpleNamespace(output_dir={json.dumps(str(r1_out))})\n"
            "state = types.SimpleNamespace(epoch=1.0, global_step=10, max_steps=30, log_history=[{'loss': 0.28}])\n"
            "ctrl = types.SimpleNamespace(should_save=False, should_training_stop=False)\n"
            "cb.on_epoch_end(args, state, ctrl)\n"
            "assert ctrl.should_save is True\n"
            "cb.on_save(args, state, ctrl)\n"
        )
        res1 = subprocess.run(
            [sys.executable, "-B", "-c", r1_code],
            env=env_box.build_env(include_fake_site=False, include_fake_hf=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res1.returncode, 0, f"HF callback R1 failed:\n{res1.stdout}\n{res1.stderr}")
        self.assertIn("CLOUD_SMOKE_PASSED", res1.stdout)
        remote_ts = env_box.drive_root / "colab_relay" / "hf-47" / "hf_checkpoints" / "checkpoint-10" / "trainer_state.json"
        self.assertTrue(remote_ts.is_file(), "Round 1 HF callback must upload trainer_state.json to gdrive")
        self.assertFalse((env_box.drive_root / "colab_relay" / "hf-47" / "checkpoint.pt").exists())

        # Round 2 subprocess in a clean directory (no checkpoint.pt anywhere)
        r2_out = env_box.workspace / "r2_hf_out"
        r2_code = (
            "import importlib.util, json, os\n"
            f"spec = importlib.util.spec_from_file_location('adapter', {json.dumps(str(ADAPTER_SCRIPT))})\n"
            "adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)\n"
            "ctx = adapter.ColabRelayContext(\n"
            "    model_name='hf-e2e', framework='huggingface', total_epochs=3,\n"
            "    gdrive_dir='gdrive:colab_relay/hf-47',\n"
            f"    results_dir={json.dumps(str(env_box.workspace / 'r2_results'))},\n"
            f"    checkpoint_path={json.dumps(str(env_box.workspace / 'r2_never_exists.pt'))},\n"
            f"    hf_checkpoints_dir={json.dumps(str(r2_out))},\n"
            f"    done_flag_path={json.dumps(str(env_box.workspace / 'r2_done.flag'))},\n"
            "    smoke_test=False, round_num=2,\n"
            ")\n"
            f"ctx.state_file = {json.dumps(str(env_box.workspace / 'r2_state.json'))}\n"
            "ctx.prepare_cloud_workspace()\n"
            "resumed = ctx.resolve_hf_resume_checkpoint()\n"
            "print(json.dumps({'resumed': resumed}))\n"
        )
        res2 = subprocess.run(
            [sys.executable, "-B", "-c", r2_code],
            env=env_box.build_env(include_fake_site=False, include_fake_hf=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res2.returncode, 0, f"HF R2 resume failed:\n{res2.stdout}\n{res2.stderr}")
        out_obj = json.loads(res2.stdout.strip().splitlines()[-1])
        self.assertTrue(out_obj["resumed"].endswith("checkpoint-10"))
        self.assertTrue((r2_out / "checkpoint-10" / "trainer_state.json").is_file())

    def test_t1_f2_bash_relay_rclone_sh_clean_success_and_traceback_breaker(self) -> None:
        bash = self._require_git_bash()
        sh_bytes = RELAY_SH_SCRIPT.read_bytes()
        self.assertNotIn(b"\r", sh_bytes, "relay_rclone.sh must use LF-only line endings")

        # Case A: Clean success pulls checkpoint.pt via real rclone alias
        env_a = IsolatedPilotEnv(self)
        (env_a.workspace / "train.py").write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")
        shutil.copy2(RELAY_SH_SCRIPT, env_a.workspace / "relay_rclone.sh")
        remote_dir = env_a.drive_root / "colab_relay" / "bash-ok"
        remote_dir.mkdir(parents=True, exist_ok=True)
        (remote_dir / "checkpoint.pt").write_bytes(b"bash-final-ckpt")

        env_a.set_colab_plan({
            "exec": [{"rc": 0, "stdout": "TRAINING_COMPLETED\n"}],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        proc_env_a = env_a.build_env(use_rclone_shim=False)
        proc_env_a.update({
            "SESSION": "bash-ok",
            "GDRIVE_DIR": "gdrive:colab_relay/bash-ok",
            "LOCAL_RCLONE_CONF": to_unix_path(str(env_a.rclone_conf)),
        })
        run_a_sh = (
            "#!/usr/bin/env bash\n"
            f'export PATH="{to_unix_path(str(env_a.bin_dir))}:{to_unix_path(os.path.dirname(env_a.real_rclone))}:$PATH"\n'
            "exec bash ./relay_rclone.sh\n"
        )
        (env_a.workspace / "run_a.sh").write_bytes(run_a_sh.encode("utf-8"))
        res_a = subprocess.run(
            [bash, "./run_a.sh"],
            cwd=str(env_a.workspace),
            env=proc_env_a,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        with self.subTest(case="clean_success"):
            self.assertEqual(res_a.returncode, 0, f"Clean relay_rclone.sh failed:\n{res_a.stdout}\n{res_a.stderr}")
            self.assertTrue((env_a.workspace / "checkpoint.pt").is_file())
            self.assertTrue(any(c["sub"] == "stop" for c in env_a.read_colab_log()))

        # Case B: Traceback with colab exec rc=0 triggers circuit breaker and exits 1
        env_b = IsolatedPilotEnv(self)
        (env_b.workspace / "train.py").write_text("raise RuntimeError('oom')\n", encoding="utf-8")
        shutil.copy2(RELAY_SH_SCRIPT, env_b.workspace / "relay_rclone.sh")
        env_b.set_colab_plan({
            "exec": [{"rc": 0, "stdout": "Traceback (most recent call last):\nRuntimeError: CUDA OOM\n"}],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        proc_env_b = env_b.build_env(use_rclone_shim=False)
        proc_env_b.update({
            "SESSION": "bash-tb",
            "GDRIVE_DIR": "gdrive:colab_relay/bash-tb",
            "LOCAL_RCLONE_CONF": to_unix_path(str(env_b.rclone_conf)),
        })
        run_b_sh = (
            "#!/usr/bin/env bash\n"
            f'export PATH="{to_unix_path(str(env_b.bin_dir))}:{to_unix_path(os.path.dirname(env_b.real_rclone))}:$PATH"\n'
            "exec bash ./relay_rclone.sh\n"
        )
        (env_b.workspace / "run_b.sh").write_bytes(run_b_sh.encode("utf-8"))
        res_b = subprocess.run(
            [bash, "./run_b.sh"],
            cwd=str(env_b.workspace),
            env=proc_env_b,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        with self.subTest(case="traceback_circuit_breaker"):
            self.assertEqual(res_b.returncode, 1)
            self.assertIn("熔断止损", res_b.stdout + "\n" + res_b.stderr)
            self.assertTrue(any(c["sub"] == "stop" for c in env_b.read_colab_log()))

    def test_t1_f3_bluegreen_custom_non_blue_session_naming(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        _, gw_port = env_box.spawn_gateway(srv)

        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [
                {"rc": 0, "stdout": 'SLOT_READY {"slot": "blue", "role": "active", "instant_tps": 52.0, "ttft_ms": 70.0, "total_tokens": 12}\n'},
            ],
        })
        res_dep = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "deploy",
                "--session-prefix",
                "prod",
                "--session-name",
                "vm-custom-88aa",
                "--slot",
                "blue",
                "--workspace",
                str(env_box.workspace),
                "--gpu",
                "G4",
                "--engine",
                "builtin",
                "--model-name",
                "Qwen-Custom",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_dep.returncode, 0, f"deploy failed:\n{res_dep.stdout}\n{res_dep.stderr}")

        st_after_dep = env_box.read_cfg("llm_bluegreen_state.json")
        self.assertEqual(st_after_dep["active_session"], "vm-custom-88aa")
        self.assertEqual(st_after_dep["blue_session"], "vm-custom-88aa")
        self.assertEqual(st_after_dep["slots"]["blue"]["session_name"], "vm-custom-88aa")

        # Seed sessions.json with real custom session + unrelated session and verify gateway routing
        env_box.write_cfg("sessions.json", {
            "unrelated-train": {"endpoint": "m-s-rel", "url": srv.url, "token": "t-rel"},
            "vm-custom-88aa": {"endpoint": "m-s-88aa", "url": srv.url, "token": "t-88aa"},
        })
        code_gw, gw_data = _http_json(f"http://127.0.0.1:{gw_port}/pilot/gateway")
        self.assertEqual(code_gw, 200)
        self.assertEqual(gw_data["session_name"], "vm-custom-88aa")

        # Switchover to custom green name `vm-green-99bb` and verify colab stop targets `vm-custom-88aa`
        env_box.clear_colab_log()
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "green", "role": "warming", "instant_tps": 60.0, "ttft_ms": 50.0, "total_tokens": 16}\n'}],
            "drain": [{"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 0, "active_requests": 0, "queried": true}\n'}],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        res_sw = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "switchover",
                "--session-prefix",
                "prod",
                "--new-session-name",
                "vm-green-99bb",
                "--workspace",
                str(env_box.workspace),
                "--drain-timeout",
                "0.2",
                "--drain-poll-interval",
                "0.04",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_sw.returncode, 0, f"switchover failed:\n{res_sw.stdout}\n{res_sw.stderr}")
        stop_calls = [c["session"] for c in env_box.read_colab_log() if c["sub"] == "stop"]
        self.assertEqual(stop_calls, ["vm-custom-88aa"])
        st_after_sw = env_box.read_cfg("llm_bluegreen_state.json")
        self.assertEqual(st_after_sw["active_session"], "vm-green-99bb")
        self.assertEqual(st_after_sw["active_slot"], "green")

    def test_t1_f3_bluegreen_prefix_scoped_stop_preserves_other_sessions(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        gw_proc, _ = env_box.spawn_gateway(srv)
        prod_state = {
            "session_prefix": "production",
            "active_slot": "blue",
            "active_session": "production-blue",
            "blue_session": "production-blue",
            "slots": {"blue": {"session_name": "production-blue", "role": "active"}},
            "phase": "serving",
        }
        env_box.write_cfg("llm_bluegreen_state.json", prod_state)

        # Case 1: stop --session-prefix experiment fails -> production state & gateway stay intact
        env_box.set_colab_plan({
            "stop": [{"rc": 1, "stderr": "Error: stop failed for experiment"}],
        })
        res_fail = subprocess.run(
            [sys.executable, "-B", str(MANAGER_SCRIPT), "stop", "--session-prefix", "experiment"],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_fail.returncode, 1)
        stop_targets = [c["session"] for c in env_box.read_colab_log() if c["sub"] == "stop"]
        self.assertNotIn("production-blue", stop_targets)
        self.assertIn("experiment-blue", stop_targets)
        self.assertIn("experiment-green", stop_targets)
        self.assertIsNone(gw_proc.poll(), "Shared gateway process must NOT be killed")
        self.assertEqual(env_box.read_cfg("llm_bluegreen_state.json")["active_session"], "production-blue")

        # Case 2: stop --session-prefix experiment succeeds -> still preserves production state & gateway
        env_box.clear_colab_log()
        env_box.set_colab_plan({
            "stop": [{"rc": 0, "stdout": "Stopped"}],
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
        })
        res_ok = subprocess.run(
            [sys.executable, "-B", str(MANAGER_SCRIPT), "stop", "--session-prefix", "experiment"],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_ok.returncode, 0)
        self.assertIsNone(gw_proc.poll())
        self.assertEqual(env_box.read_cfg("llm_bluegreen_state.json")["active_session"], "production-blue")

    def test_t1_f3_bluegreen_benchmark_propagates_exec_failure(self) -> None:
        env_box = IsolatedPilotEnv(self)
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "model_name": "Qwen-Test",
        })

        cases = [
            ({"exec": [{"rc": 2, "stderr": "exec error"}]}, [2, 1]),
            ({"exec": [{"rc": 0, "stdout": "Traceback (most recent call last):\nConnectionRefusedError\n"}]}, [1]),
            ({"exec": [{"rc": 0, "stdout": 'BENCH_RESULT {"ok_requests": 0, "bench_tokens": 0, "state": {"total_completion_tokens": 0}}\n'}]}, [1]),
            ({"exec": [{"rc": 0, "stdout": 'BENCH_RESULT {"ok_requests": 2, "bench_tokens": 32, "state": {"total_completion_tokens": 32, "instant_tps": 41.0}}\n'}]}, [0]),
        ]
        for plan, expected_rcs in cases:
            with self.subTest(expected_rcs=expected_rcs):
                env_box.set_colab_plan(plan)
                res = subprocess.run(
                    [sys.executable, "-B", str(MANAGER_SCRIPT), "benchmark", "--session-name", "llm-blue", "--requests", "2"],
                    env=env_box.build_env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                self.assertIn(res.returncode, expected_rcs, f"Unexpected rc={res.returncode}: {res.stdout}\n{res.stderr}")

    # ────────────────────────── TIER 2: Boundary & Corner Cases (12 tests) ──────────────────────────

    def test_t2_f1_gateway_interval_seconds_bva_and_per_instance_cadence(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {
            "tok-fast": [{"id": "k-fast", "name": "python3", "execution_state": "idle"}],
            "tok-slow": [{"id": "k-slow", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "vm-fast",
            "blue_session": "vm-fast",
            "green_session": "vm-slow",
        })
        env_box.write_cfg("sessions.json", {
            "vm-fast": {"endpoint": "m-s-fast", "url": srv.url, "token": "tok-fast", "kernel_id": "k-fast"},
            "vm-slow": {"endpoint": "m-s-slow", "url": srv.url, "token": "tok-slow", "kernel_id": "k-slow"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "vm-29": {"keep_alive": {"enabled": True, "interval_seconds": 29, "kernel_ping": True}},
            "vm-fast": {"keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": True}},
            "vm-90": {"keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
            "vm-slow": {"keep_alive": {"enabled": True, "interval_seconds": 600, "kernel_ping": True}},
            "vm-601": {"keep_alive": {"enabled": True, "interval_seconds": 601, "kernel_ping": True}},
        })

        # 1) Verify real subprocess startup runs initial pass once and respects min 30s cadence (no re-ping at +1s)
        gw_proc, _ = env_box.spawn_gateway(srv)
        ok = _wait_until(
            lambda: {h["token"] for h in srv.pass_hits} == {"tok-fast", "tok-slow"},
            timeout=4.0,
        )
        self.assertTrue(ok)
        time.sleep(0.5)
        with srv.lock:
            self.assertEqual(len(srv.pass_hits), 2, "Must not re-run before 30s interval elapses")
            srv.pass_hits.clear()
            srv.kernel_list_hits.clear()
        gw_proc.terminate()
        gw_proc.wait(timeout=3)

        # 2) Verify BVA clamping (29->30, 30, 90, 600, 601->600) and independent per-instance cadence in isolated subprocess
        cadence_code = (
            "import importlib.util, json\n"
            f"spec = importlib.util.spec_from_file_location('gw', {json.dumps(str(GATEWAY_SCRIPT))})\n"
            "gw = importlib.util.module_from_spec(spec); spec.loader.exec_module(gw)\n"
            "clamped = {k: gw.get_session_keep_alive_policy(k)['interval_seconds'] for k in ('vm-29', 'vm-fast', 'vm-90', 'vm-slow', 'vm-601')}\n"
            "gw.refresh_tokens_and_keepalive(send_kernel_ping=True, respect_interval=True, now=1000.0)\n"
            "gw.refresh_tokens_and_keepalive(send_kernel_ping=True, respect_interval=True, now=1029.0)\n"
            "gw.refresh_tokens_and_keepalive(send_kernel_ping=True, respect_interval=True, now=1030.0)\n"
            "gw.refresh_tokens_and_keepalive(send_kernel_ping=True, respect_interval=True, now=1599.0)\n"
            "gw.refresh_tokens_and_keepalive(send_kernel_ping=True, respect_interval=True, now=1600.0)\n"
            "print(json.dumps(clamped))\n"
        )
        res = subprocess.run(
            [sys.executable, "-B", "-c", cadence_code],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res.returncode, 0, f"Cadence probe failed:\n{res.stdout}\n{res.stderr}")
        clamped = json.loads(res.stdout.strip().splitlines()[-1])
        self.assertEqual(clamped, {"vm-29": 30, "vm-fast": 30, "vm-90": 90, "vm-slow": 600, "vm-601": 600})

        with srv.lock:
            fast_passes = [h for h in srv.pass_hits if h["token"] == "tok-fast"]
            slow_passes = [h for h in srv.pass_hits if h["token"] == "tok-slow"]
        # At t=1000: fast+slow; t=1029: none; t=1030: fast; t=1599: fast (569s since 1030 >= 30s); t=1600: slow (600s since 1000) + wait 1600-1599=1s < 30s so fast skipped!
        self.assertEqual(len(fast_passes), 3, f"Expected 3 fast passes (t=1000,1030,1599), got {fast_passes}")
        self.assertEqual(len(slow_passes), 2, f"Expected 2 slow passes (t=1000,1600), got {slow_passes}")

    def test_t2_f1_gateway_empty_or_retired_kernel_never_spawns(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {"tok-blue": []}
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
        })
        env_box.write_cfg("sessions.json", {
            "llm-blue": {"endpoint": "m-s-blue", "url": srv.url, "token": "tok-blue", "kernel_id": "retired-kid-999"},
        })
        env_box.write_cfg("pilot_session_profiles.json", {
            "llm-blue": {"keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
        })

        gw_proc, _ = env_box.spawn_gateway(srv)
        ok = _wait_until(lambda: len(srv.kernel_list_hits) >= 1, timeout=4.0)
        self.assertTrue(ok)
        time.sleep(0.25)
        with srv.lock:
            self.assertEqual(srv.kernel_spawn_hits, [], "Empty kernel list must NEVER POST /api/kernels")
            self.assertEqual(srv.pass_hits, [], "Empty kernel list must NOT execute pass")
        gw_proc.terminate()
        gw_proc.wait(timeout=3)

        # Now return a DIFFERENT live idle kernel while sessions.json still caches retired-kid-999
        with srv.lock:
            srv.kernels_by_token["tok-blue"] = [{"id": "live-kid-2", "name": "python3", "execution_state": "idle"}]
            srv.kernel_list_hits.clear()
            srv.pass_hits.clear()
            srv.kernel_spawn_hits.clear()

        env_box.spawn_gateway(srv)
        ok2 = _wait_until(lambda: len(srv.pass_hits) >= 1, timeout=4.0)
        self.assertTrue(ok2)
        with srv.lock:
            self.assertEqual(srv.kernel_spawn_hits, [], "Must NEVER POST /api/kernels")
            self.assertEqual([h["kernel_id"] for h in srv.pass_hits], ["live-kid-2"])

    def test_t2_f1_gateway_unregistered_session_defaults_disabled(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        srv.kernels_by_token = {
            "tok-unreg": [{"id": "k-unreg", "name": "python3", "execution_state": "idle"}],
            "tok-ord": [{"id": "k-ord", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "unreg-llm",
            "blue_session": "unreg-llm",
        })
        env_box.write_cfg("sessions.json", {
            "unreg-llm": {"endpoint": "m-s-unreg", "url": srv.url, "token": "tok-unreg", "kernel_id": "k-unreg"},
            "ordinary-vm": {"endpoint": "m-s-ord", "url": srv.url, "token": "tok-ord", "kernel_id": "k-ord"},
        })
        # pilot_session_profiles.json has NO entry for unreg-llm or ordinary-vm
        env_box.write_cfg("pilot_session_profiles.json", {})

        env_box.spawn_gateway(srv)
        time.sleep(0.8)
        with srv.lock:
            self.assertEqual(srv.keepalive_hits, [])
            self.assertEqual(srv.kernel_list_hits, [])
            self.assertEqual(srv.pass_hits, [])
            self.assertEqual(srv.kernel_spawn_hits, [])

    def test_t2_f2_train_py_round2_empty_gdrive_aborts_unless_reset(self) -> None:
        self._require_rclone()
        env_box = IsolatedPilotEnv(self)
        (env_box.drive_root / "colab_relay" / "wiped-round2").mkdir(parents=True, exist_ok=True)

        # Case 1: RELAY_ROUND=2 + empty gdrive + RESET_DONE_FLAG=0 -> must abort
        res_abort = subprocess.run(
            [sys.executable, "-B", str(TRAIN_SCRIPT)],
            cwd=str(env_box.workspace),
            env=env_box.build_env(
                include_fake_site=False,
                extra={
                    "SMOKE_TEST": "0",
                    "RELAY_ROUND": "2",
                    "RESET_DONE_FLAG": "0",
                    "GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                    "RELAY_GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                },
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertNotEqual(res_abort.returncode, 0)
        self.assertIn("checkpoint.pt", res_abort.stdout + "\n" + res_abort.stderr)
        self.assertFalse((env_box.workspace / "checkpoint.pt").exists())

        # Case 2: RESET_DONE_FLAG=1 overrides and runs 1 epoch cleanly
        res_reset = subprocess.run(
            [sys.executable, "-B", str(TRAIN_SCRIPT)],
            cwd=str(env_box.workspace),
            env=env_box.build_env(
                include_fake_site=False,
                extra={
                    "SMOKE_TEST": "0",
                    "RELAY_ROUND": "2",
                    "RESET_DONE_FLAG": "1",
                    "TOTAL_EPOCHS": "1",
                    "GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                    "RELAY_GDRIVE_DIR": "gdrive:colab_relay/wiped-round2",
                },
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        self.assertEqual(res_reset.returncode, 0, f"RESET_DONE_FLAG=1 run failed:\n{res_reset.stdout}\n{res_reset.stderr}")
        self.assertIn("TRAINING_COMPLETED", res_reset.stdout)
        self.assertTrue((env_box.drive_root / "colab_relay" / "wiped-round2" / "training_done.flag").is_file())

    def test_t2_f2_smoke_test_dl_bva_unread_missing_flag_and_tamper(self) -> None:
        self._require_rclone()
        env_box = IsolatedPilotEnv(self)

        fake_unread = env_box.workspace / "fake_unread.py"
        fake_unread.write_text(
            "import json, os\n"
            "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
            "if p == 1:\n"
            "    open(os.environ['CHECKPOINT_PATH'], 'wb').write(b'dummy-bytes')\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
            "else:\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 2, 'loss': 0.5}))\n",
            encoding="utf-8",
        )

        fake_no_flag = env_box.workspace / "fake_no_flag.py"
        fake_no_flag.write_text(
            "import json, os\n"
            "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
            "if p == 1:\n"
            "    open(os.environ['CHECKPOINT_PATH'], 'w', encoding='utf-8').write(json.dumps({'epoch': 1}))\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
            "else:\n"
            "    ck = json.loads(open(os.environ['CHECKPOINT_PATH'], 'r', encoding='utf-8').read())\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': ck['epoch'] + 1, 'loss': 0.4}))\n",
            encoding="utf-8",
        )

        fake_tamper_blind = env_box.workspace / "fake_tamper_blind.py"
        fake_tamper_blind.write_text(
            "import json, os\n"
            "p = int(os.environ.get('SMOKE_PASS', '1'))\n"
            "if p == 1:\n"
            "    open(os.environ['CHECKPOINT_PATH'], 'wb').write(b'anything')\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 1, 'loss': 0.9}))\n"
            "else:\n"
            "    _ = open(os.environ['CHECKPOINT_PATH'], 'rb').read()\n"
            "    open(os.environ['TRAINING_STATE_FILE'], 'w').write(json.dumps({'current_epoch': 2, 'loss': 0.4}))\n"
            "    open(os.environ['DONE_FLAG_PATH'], 'w').write('DONE')\n",
            encoding="utf-8",
        )

        sub_env = env_box.build_env(include_fake_site=False)
        for script_path, expected_rc in (
            (fake_unread, 1),
            (fake_no_flag, 1),
            (fake_tamper_blind, 1),
            (TRAIN_SCRIPT, 0),
        ):
            with self.subTest(script=script_path.name):
                res = subprocess.run(
                    [sys.executable, "-B", str(SMOKE_SCRIPT), "--script", str(script_path), "--skip-rclone-net"],
                    env=sub_env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=45,
                )
                self.assertEqual(res.returncode, expected_rc, f"{script_path.name} output:\n{res.stdout}\n{res.stderr}")

    def test_t2_f2_legacy_gdrive_checkpoint_migration_and_ignore(self) -> None:
        self._require_rclone()
        import torch

        env_box = IsolatedPilotEnv(self)
        (env_box.workspace / "train.py").write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")
        legacy_dir = env_box.drive_root / "colab_relay"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        legacy_ckpt = legacy_dir / "checkpoint.pt"
        torch.save({"epoch": 7, "loss": 0.123}, legacy_ckpt)

        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": "CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n"}],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })

        # 1) Default run without confirmation must abort before colab new
        res_block = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "legacy-80", "--workspace", str(env_box.workspace)],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_block.returncode, 1)
        self.assertEqual(env_box.read_colab_log(), [], "Must abort BEFORE any colab CLI call")
        self.assertTrue(legacy_ckpt.is_file())
        self.assertIn("--migrate-legacy-gdrive", res_block.stdout)

        # 2) --ignore-legacy-gdrive leaves legacy checkpoint untouched and runs in fresh-80 subdir
        fresh_iso_dir = env_box.drive_root / "colab_relay" / "fresh-80"
        fresh_iso_dir.mkdir(parents=True, exist_ok=True)
        (fresh_iso_dir / "checkpoint.pt").write_bytes(b"fresh-weights")
        res_ignore = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "fresh-80",
                "--workspace",
                str(env_box.workspace),
                "--ignore-legacy-gdrive",
            ],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_ignore.returncode, 0, f"ignore run failed:\n{res_ignore.stdout}\n{res_ignore.stderr}")
        self.assertTrue(legacy_ckpt.is_file(), "--ignore-legacy-gdrive must preserve legacy root checkpoint")

        # 3) --migrate-legacy-gdrive moves legacy checkpoint cleanly into gdrive:colab_relay/legacy-80/
        res_mig = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "legacy-80",
                "--workspace",
                str(env_box.workspace),
                "--migrate-legacy-gdrive",
            ],
            env=env_box.build_env(use_rclone_shim=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_mig.returncode, 0, f"migrate run failed:\n{res_mig.stdout}\n{res_mig.stderr}")
        self.assertFalse(legacy_ckpt.exists(), "Legacy root checkpoint should be moved")
        self.assertTrue((env_box.drive_root / "colab_relay" / "legacy-80" / "checkpoint.pt").is_file())

    def test_t2_f2_final_weight_download_failure_and_rerun_backfill(self) -> None:
        self._require_rclone()
        env_box = IsolatedPilotEnv(self)
        (env_box.workspace / "train.py").write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")
        iso_drive = env_box.drive_root / "colab_relay" / "exp-53"
        iso_drive.mkdir(parents=True, exist_ok=True)
        (iso_drive / "checkpoint.pt").write_bytes(b"real-final-model-weights-53")

        # Run 1: training succeeds and writes done flag, but final copyto fails
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{
                "rc": 0,
                "stdout": "CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n",
                "write_files": [{"path": str(iso_drive / "training_done.flag"), "content": "COMPLETED\n"}],
            }],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        env_box.set_rclone_plan({
            "copyto": [{"rc": 1, "stderr": "network reset during copyto"}],
        })

        res1 = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "exp-53", "--workspace", str(env_box.workspace)],
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res1.returncode, 1)
        self.assertNotIn("最终模型已保存至", res1.stdout)
        self.assertFalse((env_box.workspace / "checkpoint_final.pt").exists())
        receipt1 = json.loads((env_box.workspace / ".colab_dl_relay_status.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt1["training_completed"])
        self.assertFalse(receipt1["weights_downloaded"])

        # Run 2: copyto succeeds -> backfills weights without provisioning a new VM
        env_box.clear_colab_log()
        env_box.set_rclone_plan({})
        res2 = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "exp-53", "--workspace", str(env_box.workspace)],
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res2.returncode, 0, f"Backfill rerun failed:\n{res2.stdout}\n{res2.stderr}")
        self.assertEqual(env_box.read_colab_log(), [], "Backfill rerun must NOT allocate or touch colab VMs")
        final_ckpt = env_box.workspace / "checkpoint_final.pt"
        self.assertTrue(final_ckpt.is_file())
        self.assertEqual(final_ckpt.read_bytes(), b"real-final-model-weights-53")
        receipt2 = json.loads((env_box.workspace / ".colab_dl_relay_status.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt2["weights_downloaded"])

    def test_t2_f3_switchover_same_session_name_case_insensitive_rejected(self) -> None:
        env_box = IsolatedPilotEnv(self)
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "vm-shared",
            "blue_session": "vm-shared",
            "green_session": None,
            "slots": {"blue": {"session_name": "vm-shared", "role": "active", "model_name": "Qwen-Test"}},
            "phase": "serving",
            "engine": "builtin",
            "model_name": "Qwen-Test",
            "gpu": "T4",
        })

        for argv in (
            ["switchover", "--workspace", str(env_box.workspace), "--session-name", "vm-shared"],
            ["switchover", "--workspace", str(env_box.workspace), "--session-name", "VM-SHARED"],
            ["switchover", "--workspace", str(env_box.workspace), "--session-prefix", "other", "--session-name", "vm-shared"],
        ):
            with self.subTest(argv=argv):
                env_box.clear_colab_log()
                res = subprocess.run(
                    [sys.executable, "-B", str(MANAGER_SCRIPT)] + argv,
                    env=env_box.build_env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                self.assertEqual(res.returncode, 1)
                self.assertEqual(env_box.read_colab_log(), [], "Must reject before any colab new/exec/stop")
                st = env_box.read_cfg("llm_bluegreen_state.json")
                self.assertEqual(st["phase"], "serving")
                self.assertEqual(st["active_session"], "vm-shared")

    def test_t2_f3_switchover_drain_timeout_keep_and_active_stream_wait(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        _, gw_port = env_box.spawn_gateway(srv)

        base_state = {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "slots": {"blue": {"session_name": "llm-blue", "role": "active", "model_name": "Qwen-Test"}},
            "phase": "serving",
            "engine": "builtin",
            "model_name": "Qwen-Test",
            "gpu": "T4",
        }

        # Case A: active_streams drops 1 -> 0 on 2nd poll, then stops old slot
        env_box.write_cfg("llm_bluegreen_state.json", base_state)
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "green", "role": "warming", "instant_tps": 50.0, "ttft_ms": 60.0, "total_tokens": 10}\n'}],
            "drain": [
                {"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 1, "active_requests": 1, "queried": true}\n'},
                {"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 0, "active_requests": 0, "queried": true}\n'},
            ],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        res_a = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "switchover",
                "--workspace",
                str(env_box.workspace),
                "--drain-timeout",
                "1.5",
                "--drain-poll-interval",
                "0.04",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_a.returncode, 0, f"Drain wait switchover failed:\n{res_a.stdout}\n{res_a.stderr}")
        self.assertEqual([c["session"] for c in env_box.read_colab_log() if c["sub"] == "stop"], ["llm-blue"])

        # Case B: active_streams stays 1 past drain-timeout with --drain-timeout-policy keep
        env_box.clear_colab_log()
        env_box.write_cfg("llm_bluegreen_state.json", base_state)
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "green", "role": "warming", "instant_tps": 50.0, "ttft_ms": 60.0, "total_tokens": 10}\n'}],
            "drain": [{"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 2, "active_requests": 2, "queried": true}\n'}],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        res_b = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "switchover",
                "--workspace",
                str(env_box.workspace),
                "--drain-timeout",
                "0.18",
                "--drain-poll-interval",
                "0.04",
                "--drain-timeout-policy",
                "keep",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_b.returncode, 1)
        self.assertEqual([c for c in env_box.read_colab_log() if c["sub"] == "stop"], [], "keep policy must NOT stop old slot")
        st_b = env_box.read_cfg("llm_bluegreen_state.json")
        self.assertEqual(st_b["active_session"], "llm-green")
        self.assertTrue(st_b["drain_timed_out"])
        self.assertIn("llm-blue", st_b["unreleased_sessions"])

    def test_t2_f3_switchover_old_slot_stop_failure_reports_independent_status(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        _, gw_port = env_box.spawn_gateway(srv)
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "slots": {"blue": {"session_name": "llm-blue", "role": "active", "model_name": "Qwen-Test"}},
            "phase": "serving",
            "engine": "builtin",
            "model_name": "Qwen-Test",
            "gpu": "T4",
        })
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "green", "role": "warming", "instant_tps": 58.0, "ttft_ms": 60.0, "total_tokens": 12}\n'}],
            "drain": [{"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 0, "active_requests": 0, "queried": true}\n'}],
            "stop": [{"rc": 1, "stderr": "HTTP 500 unassign failed"}],
        })

        res = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "switchover",
                "--workspace",
                str(env_box.workspace),
                "--drain-timeout",
                "0.1",
                "--drain-poll-interval",
                "0.03",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res.returncode, 1)
        combined = res.stdout + "\n" + res.stderr
        self.assertNotIn("已彻底释放", combined)
        self.assertIn("已接管在线服务", combined)
        st = env_box.read_cfg("llm_bluegreen_state.json")
        self.assertEqual(st["active_slot"], "green")
        self.assertEqual(st["active_session"], "llm-green")
        self.assertEqual(st["phase"], "serving")
        self.assertFalse(st["old_slot_released"])
        self.assertIn("llm-blue", st["unreleased_sessions"])
        self.assertIn("llm-blue", st["last_error"])

    def test_t2_f3_hardware_compat_cli_matrix_bva(self) -> None:
        env_box = IsolatedPilotEnv(self)
        env_box.set_colab_plan({})

        # Incompatible combinations must exit 1 with zero colab invocations
        rejected = [
            ("CPU", "vllm", False),
            ("CPU", "llama.cpp", False),
            ("CPU", "transformers", False),
            ("CPU", "vllm", True),
            ("CPU", "transformers", True),
            ("V5E1", "vllm", False),
            ("V5E1", "llama.cpp", True),
            ("V6E1", "transformers", False),
        ]
        for gpu, eng, cpu_flag in rejected:
            with self.subTest(kind="rejected", gpu=gpu, engine=eng, cpu_flag=cpu_flag):
                cmd = [
                    sys.executable,
                    "-B",
                    str(MANAGER_SCRIPT),
                    "deploy",
                    "--session-name",
                    f"rej-{gpu.lower()}-{eng}",
                    "--slot",
                    "standalone",
                    "--workspace",
                    str(env_box.workspace),
                    "--gpu",
                    gpu,
                    "--engine",
                    eng,
                ]
                if cpu_flag:
                    cmd.append("--llama-cpp-cpu")
                res = subprocess.run(
                    cmd,
                    env=env_box.build_env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                self.assertEqual(res.returncode, 1)
                self.assertIn("不兼容", res.stderr)
        self.assertEqual(env_box.read_colab_log(), [], "Incompatible deploy presets must never invoke colab CLI")

        # Compatible combinations pass precheck (rc=0)
        allowed = [
            ("CPU", "builtin", False),
            ("CPU", "llama.cpp", True),
            ("T4", "llama.cpp", False),
            ("G4", "vllm", False),
        ]
        for gpu, eng, cpu_flag in allowed:
            with self.subTest(kind="allowed", gpu=gpu, engine=eng, cpu_flag=cpu_flag):
                cmd = [
                    sys.executable,
                    "-B",
                    str(MANAGER_SCRIPT),
                    "precheck",
                    "--gpu",
                    gpu,
                    "--engine",
                    eng,
                ]
                if cpu_flag:
                    cmd.append("--llama-cpp-cpu")
                res = subprocess.run(
                    cmd,
                    env=env_box.build_env(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=15,
                )
                self.assertEqual(res.returncode, 0, f"precheck failed for {gpu}/{eng}:\n{res.stdout}\n{res.stderr}")

    def test_t2_f4_telemetry_daemon_stale_pid_recycled_pid_and_nonce_ack(self) -> None:
        env_box = IsolatedPilotEnv(self)
        proc_root = env_box.root / "proc"
        proc_root.mkdir(parents=True, exist_ok=True)
        tel_file = env_box.root / "telemetry.json"
        pid_file = env_box.root / "daemon.pid"
        ack_file = env_box.root / "probe_ack.json"

        # Spawn a real unrelated child process simulating a recycled PID running train.py
        sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"])
        env_box._procs.append(sleeper)
        s_pid = sleeper.pid
        s_dir = proc_root / str(s_pid)
        s_dir.mkdir(parents=True, exist_ok=True)
        (s_dir / "cmdline").write_bytes(b"python3\x00/content/train.py\x00")
        (s_dir / "stat").write_text(f"{s_pid} (python3) R 1 1 1 0\n", encoding="utf-8")

        # Seed stale PID + stale telemetry sample
        pid_file.write_text(str(s_pid), encoding="utf-8")
        tel_file.write_text(json.dumps({"timestamp": 1700000000.0, "daemon_pid": s_pid, "probe_nonce": "old"}), encoding="utf-8")

        # Failing daemon stub exits immediately without writing telemetry
        fail_daemon = env_box.root / "fail_daemon.py"
        fail_daemon.write_text("import sys; sys.exit(0)\n", encoding="utf-8")

        run_fail_code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('td', {json.dumps(str(TELEMETRY_SCRIPT))})\n"
            "td = importlib.util.module_from_spec(spec); spec.loader.exec_module(td)\n"
            f"td.TELEMETRY_FILE = {json.dumps(str(tel_file))}\n"
            f"td.PID_FILE = {json.dumps(str(pid_file))}\n"
            f"td.PROBE_ACK_FILE = {json.dumps(str(ack_file))}\n"
            f"rc = td.bootstrap_probe(nonce='fresh-1', timeout_sec=0.5, daemon_script={json.dumps(str(fail_daemon))}, proc_root={json.dumps(str(proc_root))})\n"
            "sys.exit(rc)\n"
        )
        res_fail = subprocess.run(
            [sys.executable, "-B", "-c", run_fail_code],
            env=env_box.build_env(include_fake_site=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_fail.returncode, 1)
        self.assertIsNone(sleeper.poll(), "bootstrap_probe must NEVER kill a recycled non-daemon PID")
        self.assertFalse(ack_file.exists())

        # Now run with a live stub daemon that registers itself in proc_root and writes matching nonce & fresh timestamp
        live_daemon = env_box.root / "live_daemon.py"
        live_daemon.write_text(
            "import json, os, sys, time\n"
            "nonce = sys.argv[sys.argv.index('--nonce') + 1]\n"
            "pid = os.getpid()\n"
            f"pdir = os.path.join({json.dumps(str(proc_root))}, str(pid))\n"
            "os.makedirs(pdir, exist_ok=True)\n"
            "open(os.path.join(pdir, 'cmdline'), 'wb').write(f'python3\\x00/content/.colab_pilot_daemon.py\\x00--nonce\\x00{nonce}\\x00'.encode())\n"
            "open(os.path.join(pdir, 'stat'), 'w').write(f'{pid} (python3) S 1 1 1 0\\n')\n"
            f"open({json.dumps(str(pid_file))}, 'w').write(str(pid))\n"
            f"open({json.dumps(str(tel_file))}, 'w').write(json.dumps({{'timestamp': time.time() + 0.02, 'daemon_pid': pid, 'probe_nonce': nonce}}))\n"
            "time.sleep(0.8)\n",
            encoding="utf-8",
        )
        run_ok_code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('td', {json.dumps(str(TELEMETRY_SCRIPT))})\n"
            "td = importlib.util.module_from_spec(spec); spec.loader.exec_module(td)\n"
            f"td.TELEMETRY_FILE = {json.dumps(str(tel_file))}\n"
            f"td.PID_FILE = {json.dumps(str(pid_file))}\n"
            f"td.PROBE_ACK_FILE = {json.dumps(str(ack_file))}\n"
            f"rc = td.bootstrap_probe(nonce='fresh-nonce-55', timeout_sec=3.0, daemon_script={json.dumps(str(live_daemon))}, proc_root={json.dumps(str(proc_root))})\n"
            "sys.exit(rc)\n"
        )
        res_ok = subprocess.run(
            [sys.executable, "-B", "-c", run_ok_code],
            env=env_box.build_env(include_fake_site=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_ok.returncode, 0, f"bootstrap_probe ok failed:\n{res_ok.stdout}\n{res_ok.stderr}")
        self.assertIn("PROBE_OK", res_ok.stdout)
        ack = json.loads(ack_file.read_text(encoding="utf-8"))
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["nonce"], "fresh-nonce-55")
        self.assertEqual(ack["pid"], int(pid_file.read_text(encoding="utf-8").strip()))

    # ────────────────────────── TIER 3: Pairwise Combinations (2 tests) ──────────────────────────

    def test_t3_f1_f3_gateway_kernel_reuse_and_per_vm_busy_during_stream(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        gate_ev = threading.Event()
        srv.stream_events["tok-blue"] = gate_ev
        srv.kernels_by_token = {
            "tok-blue": [{"id": "kid-blue-live", "name": "python3", "execution_state": "idle"}],
            "tok-green": [{"id": "kid-green-live", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "llm",
            "active_slot": "blue",
            "active_session": "llm-blue",
            "blue_session": "llm-blue",
            "green_session": "llm-green",
            "phase": "serving",
        })
        env_box.write_cfg("sessions.json", {
            "llm-blue": {"endpoint": "m-s-blue", "url": srv.url, "token": "tok-blue", "kernel_id": "kid-blue-live"},
            # Deliberately cache retired kernel_id on green; gateway must pick live kid-green-live without POST spawn
            "llm-green": {"endpoint": "m-s-green", "url": srv.url, "token": "tok-green", "kernel_id": "kid-green-retired"},
        })
        # Start with green disabled, then enable both right after opening blue stream
        env_box.write_cfg("pilot_session_profiles.json", {
            "llm-blue": {"keep_alive": {"enabled": False, "interval_seconds": 30, "kernel_ping": True}},
            "llm-green": {"keep_alive": {"enabled": False, "interval_seconds": 30, "kernel_ping": True}},
        })

        _, gw_port = env_box.spawn_gateway(srv)
        time.sleep(0.25)

        stream_result: Dict[str, Any] = {}

        def _run_blue_stream() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", gw_port, timeout=10)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True, "messages": [{"role": "user", "content": "stream"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Colab-Vm": "llm-blue"},
            )
            resp = conn.getresponse()
            stream_result["status"] = resp.status
            stream_result["body"] = resp.read().decode("utf-8", "replace")
            conn.close()

        t_stream = threading.Thread(target=_run_blue_stream, daemon=True)
        t_stream.start()
        try:
            ok_active = _wait_until(
                lambda: _http_json(f"http://127.0.0.1:{gw_port}/pilot/gateway")[1]
                .get("active_requests_by_session", {})
                .get("llm-blue", 0) == 1,
                timeout=4.0,
            )
            self.assertTrue(ok_active, "Gateway must track llm-blue active request")

            # Now enable both blue and green keep-alive while blue is mid-stream
            env_box.write_cfg("pilot_session_profiles.json", {
                "llm-blue": {"keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": True}},
                "llm-green": {"keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": True}},
            })
            # Wait for 5s keep-alive scheduler tick while blue stream remains open
            ok_green = _wait_until(
                lambda: any(h["token"] == "tok-green" for h in srv.pass_hits),
                timeout=7.5,
            )
            self.assertTrue(ok_green, "Idle green slot must still receive pass while blue slot is streaming (#95)")
            with srv.lock:
                blue_passes_during_stream = [h for h in srv.pass_hits if h["token"] == "tok-blue"]
                green_passes = [h for h in srv.pass_hits if h["token"] == "tok-green"]
                spawns = list(srv.kernel_spawn_hits)
            self.assertEqual(blue_passes_during_stream, [], "Busy blue slot must skip pass while streaming")
            self.assertEqual([h["kernel_id"] for h in green_passes], ["kid-green-live"])
            self.assertEqual(spawns, [], "Must NEVER spawn kernel via POST /api/kernels (#93)")
        finally:
            gate_ev.set()
            t_stream.join(timeout=5.0)

        self.assertEqual(stream_result.get("status"), 200)
        self.assertIn("hello", stream_result.get("body", ""))

    def test_t3_f1_f2_cross_process_profiles_lock_rust_and_python_relay(self) -> None:
        env_box = IsolatedPilotEnv(self)
        prof_file = env_box.write_cfg("pilot_session_profiles.json", {
            "vm-user": {
                "boot_mode": "relay",
                "accelerator": "G4",
                "keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True},
            }
        })
        ready_flag = env_box.root / "rust_locked.flag"
        (env_box.workspace / "train.py").write_text("print('TRAINING_COMPLETED')\n", encoding="utf-8")
        # Make relay finish immediately via existing done flag + final weights
        iso_drive = env_box.drive_root / "colab_relay" / "relay-94"
        iso_drive.mkdir(parents=True, exist_ok=True)
        (iso_drive / "training_done.flag").write_text("DONE\n", encoding="utf-8")
        (iso_drive / "checkpoint.pt").write_bytes(b"final-94")

        proc_env = env_box.build_env(use_rclone_shim=False)
        rust_proc = self._spawn_lock_worker(
            proc_env,
            prof_file,
            ready_flag,
            hold_ms=350,
            session_name="vm-user",
            enabled=False,
            interval=600,
            kernel_ping=False,
        )
        env_box._procs.append(rust_proc)
        self.assertTrue(_wait_until(ready_flag.exists, timeout=5.0), "Rust lock worker did not signal lock acquisition")

        # Launch real dl_relay_runner.py subprocess while Rust worker holds .lock
        res_relay = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "relay-94", "--workspace", str(env_box.workspace)],
            env=proc_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        r_out, r_err = rust_proc.communicate(timeout=5)
        self.assertEqual(rust_proc.returncode, 0, f"Rust lock worker failed: {r_out}\n{r_err}")
        self.assertEqual(res_relay.returncode, 0, f"dl_relay_runner failed:\n{res_relay.stdout}\n{res_relay.stderr}")

        final_profiles = env_box.read_cfg("pilot_session_profiles.json")
        self.assertIn("vm-user", final_profiles)
        self.assertIn("relay-94", final_profiles)
        self.assertEqual(final_profiles["vm-user"]["boot_mode"], "relay")
        self.assertEqual(
            final_profiles["vm-user"]["keep_alive"],
            {"enabled": False, "interval_seconds": 600, "kernel_ping": False},
        )
        self.assertEqual(
            final_profiles["relay-94"]["keep_alive"],
            {"enabled": True, "interval_seconds": 90, "kernel_ping": True},
        )

    # ────────────────────────── TIER 4: Real-World Workflows (4 tests) ──────────────────────────

    def test_t4_w1_detached_relay_handover_to_backfill_completion(self) -> None:
        self._require_rclone()
        import torch

        env_box = IsolatedPilotEnv(self)
        parent_dir = env_box.root / "projects"
        proj_dir = parent_dir / "my_proj"
        proj_dir.mkdir(parents=True, exist_ok=True)
        (proj_dir / "train.py").write_text("print('w1')\n", encoding="utf-8")

        # Seed legacy bare gdrive:colab_relay/checkpoint.pt (#80)
        legacy_dir = env_box.drive_root / "colab_relay"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": 4, "loss": 0.25}, legacy_dir / "checkpoint.pt")

        iso_drive = env_box.drive_root / "colab_relay" / "w1-job"

        # Step 1: Unconfirmed detached start is blocked by legacy checkpoint (#80)
        res_block = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "w1-job", "--workspace", "my_proj", "--detach"],
            cwd=str(parent_dir),
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_block.returncode, 1)

        # Step 2: Detached launch with --migrate-legacy-gdrive:
        # Round 1 hits 503 then succeeds with RELAY_ROUND=1 -> handover -> strict stop verify
        # Round 2 succeeds with RELAY_ROUND=2 -> TRAINING_COMPLETED, but copyto fails (#53)
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [
                {"rc": 1, "stderr": "HTTP 503 Service Unavailable"},
                {"rc": 0, "stdout": "Session READY\n"},
                {"rc": 0, "stdout": "Session READY\n"},
            ],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [
                {"rc": 0, "run_stdin_py": True, "stdout": "CLOUD_SMOKE_PASSED\nHANDOVER_REQUESTED\nTRAIN_EXIT_CODE=0\n"},
                {
                    "rc": 0,
                    "run_stdin_py": True,
                    "stdout": "CLOUD_SMOKE_PASSED\nTRAINING_COMPLETED\nTRAIN_EXIT_CODE=0\n",
                    "write_files": [
                        {"path": str(iso_drive / "training_done.flag"), "content": "COMPLETED\n"},
                        {"path": str(iso_drive / "checkpoint.pt"), "content": "w1-final-weights"},
                    ],
                },
            ],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        env_box.set_rclone_plan({
            "copyto": [{"rc": 1, "stderr": "simulated final download failure"}],
        })

        res_det = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "w1-job",
                "--workspace",
                "my_proj",
                "--grab-retry-seconds",
                "0.05",
                "--migrate-legacy-gdrive",
                "--detach",
            ],
            cwd=str(parent_dir),
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_det.returncode, 0, f"Detach parent failed:\n{res_det.stdout}\n{res_det.stderr}")
        self.assertTrue((iso_drive / "checkpoint.pt").is_file(), "Legacy checkpoint must be migrated to w1-job")

        receipt_file = proj_dir / ".colab_dl_relay_status.json"
        self.assertTrue(
            _wait_until(receipt_file.exists, timeout=25.0, interval=0.15),
            "Detached relay child did not finish handover + round 2 within timeout",
        )
        receipt1 = json.loads(receipt_file.read_text(encoding="utf-8"))
        self.assertTrue(receipt1["training_completed"])
        self.assertFalse(receipt1["weights_downloaded"])

        log_txt = (proj_dir / ".colab_dl_relay.log").read_text(encoding="utf-8", errors="replace")
        self.assertIn("OBSERVED_RELAY_ROUND=1", log_txt)
        self.assertIn("OBSERVED_RELAY_ROUND=2", log_txt)
        self.assertIn("OBSERVED_GDRIVE_DIR=gdrive:colab_relay/w1-job", log_txt)

        # Verify intruder with conflicting active lock is rejected in --detach parent
        pid_file = proj_dir / ".colab_dl_relay.pid"
        pid_file.write_text("888888", encoding="utf-8")
        env_box.write_cfg("dl_relay_locks.json", {
            "gdrive:colab_relay/w1-job": {"session": "w1-job", "pid": os.getpid(), "workspace": str(proj_dir)}
        })
        res_intruder = subprocess.run(
            [
                sys.executable,
                "-B",
                str(RUNNER_SCRIPT),
                "--session",
                "intruder-job",
                "--workspace",
                "my_proj",
                "--gdrive-dir",
                "gdrive:colab_relay/w1-job",
                "--detach",
            ],
            cwd=str(parent_dir),
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_intruder.returncode, 1)
        self.assertEqual(pid_file.read_text(encoding="utf-8"), "888888")

        # Step 3: Backfill rerun downloads final weights without allocating another VM
        env_box.clear_colab_log()
        env_box.set_rclone_plan({})
        res_backfill = subprocess.run(
            [sys.executable, "-B", str(RUNNER_SCRIPT), "--session", "w1-job", "--workspace", "my_proj"],
            cwd=str(parent_dir),
            env=env_box.build_env(use_rclone_shim=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_backfill.returncode, 0, f"Backfill failed:\n{res_backfill.stdout}\n{res_backfill.stderr}")
        self.assertEqual(env_box.read_colab_log(), [])
        self.assertEqual((proj_dir / "checkpoint_final.pt").read_text(encoding="utf-8"), "w1-final-weights")

    def test_t4_w2_bluegreen_sidecar_gateway_stream_benchmark_switchover_stop(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        _, sidecar_port, sidecar_state_file = env_box.spawn_sidecar(slot="blue", role="active", model_name="Qwen-W2")
        _, gw_port = env_box.spawn_gateway(srv, extra_env={"MOCK_REAL_SIDECAR_PORT": str(sidecar_port)})

        # 1) Deploy custom-named blue session `custom-alpha`
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "blue", "role": "active", "instant_tps": 46.0, "ttft_ms": 55.0, "total_tokens": 15}\n'}],
        })
        res_dep = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "deploy",
                "--session-prefix",
                "w2",
                "--session-name",
                "custom-alpha",
                "--slot",
                "blue",
                "--workspace",
                str(env_box.workspace),
                "--gpu",
                "G4",
                "--engine",
                "builtin",
                "--model-name",
                "Qwen-W2",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_dep.returncode, 0, f"W2 deploy failed:\n{res_dep.stdout}\n{res_dep.stderr}")

        # Wire sessions.json + paused standby slot + active kernel for gateway
        srv.kernels_by_token = {
            "tok-alpha": [{"id": "k-alpha", "name": "python3", "execution_state": "idle"}],
            "tok-paused": [{"id": "k-paused", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("sessions.json", {
            "custom-alpha": {"endpoint": "m-s-alpha", "url": srv.url, "token": "tok-alpha", "kernel_id": "k-alpha"},
            "paused-other": {"endpoint": "m-s-paused", "url": srv.url, "token": "tok-paused", "kernel_id": "k-paused"},
        })
        profs = env_box.read_cfg("pilot_session_profiles.json")
        profs["paused-other"] = {"keep_alive": {"enabled": False, "interval_seconds": 90, "kernel_ping": True}}
        env_box.write_cfg("pilot_session_profiles.json", profs)

        # 2) Stream chat completion through gateway -> real sidecar -> verify metrics & sidecar state
        c_code, c_body = _http_json(
            f"http://127.0.0.1:{gw_port}/v1/chat/completions",
            method="POST",
            body={"model": "Qwen-W2", "stream": False, "messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertEqual(c_code, 200)
        self.assertIn("BLUE/builtin", c_body["choices"][0]["message"]["content"])
        ok_sidecar = _wait_until(
            lambda: sidecar_state_file.exists()
            and json.loads(sidecar_state_file.read_text(encoding="utf-8")).get("total_completion_tokens", 0) > 0,
            timeout=3.0,
        )
        self.assertTrue(ok_sidecar, "Sidecar state file did not record completion tokens")
        sidecar_snap = json.loads(sidecar_state_file.read_text(encoding="utf-8"))
        self.assertGreater(sidecar_snap["total_completion_tokens"], 0)

        # Update mock server contents endpoint with sidecar metrics and confirm /pilot/metrics
        srv.llm_states_by_token["tok-alpha"] = sidecar_snap
        self.assertTrue(_wait_until(lambda: _http_json(f"http://127.0.0.1:{gw_port}/pilot/metrics")[0] == 200, timeout=5.0))

        # Verify paused-other never got kernel list or pass, and zero POST /api/kernels
        with srv.lock:
            self.assertNotIn("tok-paused", {h["token"] for h in srv.kernel_list_hits})
            self.assertEqual(srv.kernel_spawn_hits, [])

        # 3) Benchmark failure (rc=1) vs success (rc=0)
        env_box.set_colab_plan({"exec": [{"rc": 1, "stderr": "bench crash"}]})
        res_bf = subprocess.run(
            [sys.executable, "-B", str(MANAGER_SCRIPT), "benchmark", "--session-prefix", "w2", "--requests", "1"],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_bf.returncode, 1)

        env_box.set_colab_plan({
            "exec": [{"rc": 0, "stdout": 'BENCH_RESULT {"ok_requests": 1, "bench_tokens": 18, "state": {"total_completion_tokens": 18}}\n'}],
        })
        res_bo = subprocess.run(
            [sys.executable, "-B", str(MANAGER_SCRIPT), "benchmark", "--session-prefix", "w2", "--requests", "1"],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_bo.returncode, 0)

        # 4) Switchover to custom-beta with drain wait, then prefix-scoped stop on other prefix
        env_box.set_colab_plan({
            "status": [{"rc": 1, "stderr": "No active session"}],
            "sessions": [{"rc": 0, "stdout": ""}],
            "new": [{"rc": 0, "stdout": "Session READY\n"}],
            "upload": [{"rc": 0, "stdout": "Uploaded\n"}],
            "exec": [{"rc": 0, "stdout": 'SLOT_READY {"slot": "green", "role": "warming", "instant_tps": 55.0, "ttft_ms": 45.0, "total_tokens": 16}\n'}],
            "drain": [
                {"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 1, "active_requests": 1, "queried": true}\n'},
                {"rc": 0, "stdout": 'DRAIN_STATUS {"active_streams": 0, "active_requests": 0, "queried": true}\n'},
            ],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        res_sw = subprocess.run(
            [
                sys.executable,
                "-B",
                str(MANAGER_SCRIPT),
                "switchover",
                "--session-prefix",
                "w2",
                "--new-session-name",
                "custom-beta",
                "--workspace",
                str(env_box.workspace),
                "--drain-timeout",
                "1.5",
                "--drain-poll-interval",
                "0.04",
                "--gateway-port",
                str(gw_port),
            ],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertEqual(res_sw.returncode, 0, f"W2 switchover failed:\n{res_sw.stdout}\n{res_sw.stderr}")
        self.assertEqual(env_box.read_cfg("llm_bluegreen_state.json")["active_session"], "custom-beta")

        # Stop unrelated prefix preserves w2 state
        res_stop_other = subprocess.run(
            [sys.executable, "-B", str(MANAGER_SCRIPT), "stop", "--session-prefix", "unrelated"],
            env=env_box.build_env(mock_server=srv),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
        self.assertEqual(res_stop_other.returncode, 0)
        self.assertEqual(env_box.read_cfg("llm_bluegreen_state.json")["active_session"], "custom-beta")

    def test_t4_w3_multi_round_bash_and_hf_trainer_relay_workflow(self) -> None:
        self._require_rclone()
        bash = self._require_git_bash()
        env_box = IsolatedPilotEnv(self)

        # 1) Smoke test train.py passes while unread script fails (#63)
        res_smoke_ok = subprocess.run(
            [sys.executable, "-B", str(SMOKE_SCRIPT), "--script", str(TRAIN_SCRIPT), "--skip-rclone-net"],
            env=env_box.build_env(include_fake_site=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        self.assertEqual(res_smoke_ok.returncode, 0)

        # 2) Empty Round 2 remote aborts (#81)
        (env_box.drive_root / "colab_relay" / "w3-empty").mkdir(parents=True, exist_ok=True)
        res_empty = subprocess.run(
            [sys.executable, "-B", str(TRAIN_SCRIPT)],
            cwd=str(env_box.workspace),
            env=env_box.build_env(
                include_fake_site=False,
                extra={"SMOKE_TEST": "0", "RELAY_ROUND": "2", "GDRIVE_DIR": "gdrive:colab_relay/w3-empty", "RELAY_GDRIVE_DIR": "gdrive:colab_relay/w3-empty"},
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        self.assertNotEqual(res_empty.returncode, 0)

        # 3) HF Trainer callback R1 -> R2 resume (#47)
        hf_out1 = env_box.workspace / "hf_r1"
        (hf_out1 / "checkpoint-20").mkdir(parents=True, exist_ok=True)
        (hf_out1 / "checkpoint-20" / "trainer_state.json").write_text('{"global_step": 20, "epoch": 2.0}', encoding="utf-8")
        hf_code = (
            "import importlib.util, types\n"
            f"spec = importlib.util.spec_from_file_location('ad', {json.dumps(str(ADAPTER_SCRIPT))})\n"
            "ad = importlib.util.module_from_spec(spec); spec.loader.exec_module(ad)\n"
            f"ctx1 = ad.ColabRelayContext(framework='huggingface', gdrive_dir='gdrive:colab_relay/w3-hf', results_dir={json.dumps(str(env_box.workspace / 'hf_res1'))}, checkpoint_path={json.dumps(str(env_box.workspace / 'no1.pt'))}, hf_checkpoints_dir={json.dumps(str(hf_out1))}, done_flag_path={json.dumps(str(env_box.workspace / 'd1.flag'))}, smoke_test=False, round_num=1)\n"
            f"ctx1.state_file = {json.dumps(str(env_box.workspace / 's1.json'))}\n"
            "cb = ad.make_hf_trainer_callback(ctx1)\n"
            f"cb.on_save(types.SimpleNamespace(output_dir={json.dumps(str(hf_out1))}), types.SimpleNamespace(epoch=2.0, global_step=20, max_steps=40, log_history=[{{'loss': 0.1}}]), types.SimpleNamespace(should_save=False, should_training_stop=False))\n"
            f"ctx2 = ad.ColabRelayContext(framework='huggingface', gdrive_dir='gdrive:colab_relay/w3-hf', results_dir={json.dumps(str(env_box.workspace / 'hf_res2'))}, checkpoint_path={json.dumps(str(env_box.workspace / 'no2.pt'))}, hf_checkpoints_dir={json.dumps(str(env_box.workspace / 'hf_r2'))}, done_flag_path={json.dumps(str(env_box.workspace / 'd2.flag'))}, smoke_test=False, round_num=2)\n"
            f"ctx2.state_file = {json.dumps(str(env_box.workspace / 's2.json'))}\n"
            "ctx2.prepare_cloud_workspace()\n"
            "res = ctx2.resolve_hf_resume_checkpoint()\n"
            "assert res and res.endswith('checkpoint-20'), res\n"
        )
        res_hf = subprocess.run(
            [sys.executable, "-B", "-c", hf_code],
            env=env_box.build_env(include_fake_site=False, include_fake_hf=True),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
        )
        self.assertEqual(res_hf.returncode, 0, f"HF W3 failed:\n{res_hf.stdout}\n{res_hf.stderr}")

        # 4) Multi-round bash relay_rclone.sh: Round 1 HANDOVER -> Round 2 TRAINING_COMPLETED -> pull checkpoint.pt (#61)
        bash_ws = env_box.workspace / "bash_multi"
        bash_ws.mkdir(parents=True, exist_ok=True)
        (bash_ws / "train.py").write_text("print('w3')\n", encoding="utf-8")
        shutil.copy2(RELAY_SH_SCRIPT, bash_ws / "relay_rclone.sh")
        bash_drive = env_box.drive_root / "colab_relay" / "w3-bash"
        bash_drive.mkdir(parents=True, exist_ok=True)
        (bash_drive / "checkpoint.pt").write_bytes(b"w3-bash-final-ckpt")

        env_box.set_colab_plan({
            "exec": [
                {"rc": 0, "stdout": "HANDOVER_REQUESTED\n"},
                {"rc": 0, "stdout": "TRAINING_COMPLETED\n"},
            ],
            "stop": [{"rc": 0, "stdout": "Stopped\n"}],
        })
        bash_env = env_box.build_env(use_rclone_shim=False)
        bash_env.update({
            "SESSION": "w3-bash",
            "GDRIVE_DIR": "gdrive:colab_relay/w3-bash",
            "LOCAL_RCLONE_CONF": to_unix_path(str(env_box.rclone_conf)),
        })
        run_w3_sh = (
            "#!/usr/bin/env bash\n"
            f'export PATH="{to_unix_path(str(env_box.bin_dir))}:{to_unix_path(os.path.dirname(env_box.real_rclone))}:$PATH"\n'
            "exec bash ./relay_rclone.sh\n"
        )
        (bash_ws / "run_w3.sh").write_bytes(run_w3_sh.encode("utf-8"))
        res_bash = subprocess.run(
            [bash, "./run_w3.sh"],
            cwd=str(bash_ws),
            env=bash_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=45,
        )
        self.assertEqual(res_bash.returncode, 0, f"Multi-round bash relay failed:\n{res_bash.stdout}\n{res_bash.stderr}")
        self.assertEqual((bash_ws / "checkpoint.pt").read_bytes(), b"w3-bash-final-ckpt")
        exec_rounds = [c["stdin_preview"] for c in env_box.read_colab_log() if c["sub"] == "exec"]
        # At least 2 training exec calls (plus 2 rclone check exec calls)
        self.assertGreaterEqual(len(exec_rounds), 2)

    def test_t4_w4_rust_python_concurrent_profile_and_gateway_policy_e2e(self) -> None:
        env_box = IsolatedPilotEnv(self)
        srv = env_box.start_mock_server()
        prof_file = env_box.write_cfg("pilot_session_profiles.json", {
            "vm-paused": {"boot_mode": "llm", "keep_alive": {"enabled": True, "interval_seconds": 90, "kernel_ping": True}},
            "vm-tunnel-only": {"boot_mode": "llm", "keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": False}},
            "vm-active": {"boot_mode": "llm", "keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": True}},
            "vm-busy": {"boot_mode": "llm", "keep_alive": {"enabled": True, "interval_seconds": 30, "kernel_ping": True}},
        })
        proc_env = env_box.build_env(mock_server=srv)

        # 1) Race Rust File::lock() workers + Python dl_relay_runner + llm_bluegreen_manager workers concurrently (#94)
        py_worker = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('wm', sys.argv[1])\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "sys.exit(0 if m.ensure_session_keep_alive_profile(sys.argv[2]) else 2)\n"
        )
        workers: List[subprocess.Popen[str]] = []
        for i in range(3):
            workers.append(subprocess.Popen(
                [sys.executable, "-B", "-c", py_worker, str(RUNNER_SCRIPT), f"w4-dl-{i}"],
                env=proc_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))
            workers.append(subprocess.Popen(
                [sys.executable, "-B", "-c", py_worker, str(MANAGER_SCRIPT), f"w4-llm-{i}"],
                env=proc_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ))
        rust_p = self._spawn_lock_worker(
            proc_env,
            prof_file,
            Path("-"),
            hold_ms=100,
            session_name="vm-paused",
            enabled=False,
            interval=600,
            kernel_ping=False,
        )
        workers.append(rust_p)
        env_box._procs.extend(workers)

        for w in workers:
            out, err = w.communicate(timeout=20)
            self.assertEqual(w.returncode, 0, f"Concurrent profile worker failed: {out}\n{err}")

        profiles = env_box.read_cfg("pilot_session_profiles.json")
        for i in range(3):
            self.assertIn(f"w4-dl-{i}", profiles)
            self.assertIn(f"w4-llm-{i}", profiles)
        self.assertEqual(
            profiles["vm-paused"]["keep_alive"],
            {"enabled": False, "interval_seconds": 600, "kernel_ping": False},
        )

        # 2) Boot real gateway against those concurrent profiles while vm-busy is streaming (#88, #93, #95)
        gate_ev = threading.Event()
        srv.stream_events["tok-busy"] = gate_ev
        srv.kernels_by_token = {
            "tok-paused": [{"id": "k-paused", "name": "python3", "execution_state": "idle"}],
            "tok-tunnel": [{"id": "k-tunnel", "name": "python3", "execution_state": "idle"}],
            "tok-active": [{"id": "k-active-live", "name": "python3", "execution_state": "idle"}],
            "tok-busy": [{"id": "k-busy", "name": "python3", "execution_state": "idle"}],
        }
        env_box.write_cfg("llm_bluegreen_state.json", {
            "session_prefix": "w4",
            "active_slot": "blue",
            "active_session": "vm-active",
            "blue_session": "vm-active",
            "green_session": "vm-tunnel-only",
        })
        env_box.write_cfg("llm_api_keys.json", {
            "vms": {
                "vm-paused": {"vm_name": "vm-paused", "slot": "standalone", "api_key": "sk-p"},
                "vm-busy": {"vm_name": "vm-busy", "slot": "standalone", "api_key": "sk-b"},
            }
        })
        env_box.write_cfg("sessions.json", {
            "vm-paused": {"endpoint": "m-s-p", "url": srv.url, "token": "tok-paused", "kernel_id": "k-paused"},
            "vm-tunnel-only": {"endpoint": "m-s-t", "url": srv.url, "token": "tok-tunnel", "kernel_id": "k-tunnel"},
            "vm-active": {"endpoint": "m-s-a", "url": srv.url, "token": "tok-active", "kernel_id": "k-active-retired"},
            "vm-busy": {"endpoint": "m-s-b", "url": srv.url, "token": "tok-busy", "kernel_id": "k-busy"},
        })

        # Keep vm-active/vm-tunnel-only/vm-busy disabled while we open vm-busy stream, then re-enable
        profiles["vm-tunnel-only"]["keep_alive"]["enabled"] = False
        profiles["vm-active"]["keep_alive"]["enabled"] = False
        profiles["vm-busy"]["keep_alive"]["enabled"] = False
        env_box.write_cfg("pilot_session_profiles.json", profiles)

        _, gw_port = env_box.spawn_gateway(srv)
        time.sleep(0.25)

        busy_res: Dict[str, Any] = {}

        def _busy_stream() -> None:
            conn = http.client.HTTPConnection("127.0.0.1", gw_port, timeout=10)
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps({"stream": True, "messages": [{"role": "user", "content": "busy"}]}).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer sk-b"},
            )
            resp = conn.getresponse()
            busy_res["status"] = resp.status
            busy_res["body"] = resp.read().decode("utf-8", "replace")
            conn.close()

        t_busy = threading.Thread(target=_busy_stream, daemon=True)
        t_busy.start()
        try:
            self.assertTrue(
                _wait_until(
                    lambda: _http_json(f"http://127.0.0.1:{gw_port}/pilot/gateway")[1]
                    .get("active_requests_by_session", {})
                    .get("vm-busy", 0) == 1,
                    timeout=4.0,
                )
            )
            profiles["vm-tunnel-only"]["keep_alive"]["enabled"] = True
            profiles["vm-active"]["keep_alive"]["enabled"] = True
            profiles["vm-busy"]["keep_alive"]["enabled"] = True
            env_box.write_cfg("pilot_session_profiles.json", profiles)

            ok_tick = _wait_until(
                lambda: (
                    any(h["token"] == "tok-active" for h in srv.pass_hits)
                    and any(h["token"] == "tok-tunnel" for h in srv.kernel_list_hits)
                ),
                timeout=7.5,
            )
            self.assertTrue(ok_tick, "Gateway must run tunnel check on vm-tunnel-only and pass on vm-active while vm-busy streams")
            with srv.lock:
                k_toks = {h["token"] for h in srv.kernel_list_hits}
                p_toks = {h["token"] for h in srv.pass_hits}
                active_kids = [h["kernel_id"] for h in srv.pass_hits if h["token"] == "tok-active"]
                spawns = list(srv.kernel_spawn_hits)
            self.assertNotIn("tok-paused", k_toks)
            self.assertNotIn("tok-paused", p_toks)
            self.assertIn("tok-tunnel", k_toks)
            self.assertNotIn("tok-tunnel", p_toks)
            self.assertNotIn("tok-busy", p_toks)
            self.assertEqual(active_kids, ["k-active-live"])
            self.assertEqual(spawns, [])
        finally:
            gate_ev.set()
            t_busy.join(timeout=5.0)
        self.assertEqual(busy_res.get("status"), 200)


if __name__ == "__main__":
    unittest.main()
