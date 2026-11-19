#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 大语言模型多引擎部署与并行蓝绿热切换编排器 (llm_bluegreen_manager.py)
====================================================================================
由 Antigravity `colab-llm-bluegreen` Skill 调用，支持：
1. `precheck`   : 上云前轻量预检（校验边车语法、colab-cli 授权、rclone.conf 与 Google Drive 模型权重路径，不在本机跑假引擎）
2. `deploy`     : 部署首个蓝/绿实例 (-s llm-blue)，拉取 Google Drive 模型权重、启动推理引擎 + 边车，
                  并在 Colab 云端真机执行 Warmup 推理冒烟（若显存 OOM 或测速失败则自动 colab stop 关机止损）
3. `switchover` : 执行 Make-Before-Break（先建后拆）并行蓝绿热切换：
                  保持旧实例在线服务 -> 启动新实例并在云端执行真机 Warmup 冒烟测速 -> 冒烟通过后原子切流
                  -> 排空并销毁旧实例停止双倍计费（若新实例云端冒烟失败，自动销毁新实例，旧实例零中断继续服务）
4. `benchmark`  : 对当前活跃槽位发起流式推理压测，实时刷新 Colab Pilot 上的 Token 输出频率 (tok/s)
5. `stop`       : 停止蓝绿全部实例并释放服务端计费
"""

import argparse
import ast
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLAB_UPDATE_BANNER_RE = re.compile(
    r"^\s*\[colab\]\s*(?:A new version of Colab CLI is available|Run 'colab update'|To silence this check).*$",
    re.IGNORECASE,
)

# Google Colab `/tun/m/assign` 仅提供单一默认内存规格的加速器（传入 `shape=hm` / `--high-mem` 会触发 HTTP 400）
SINGLE_SHAPE_ACCELERATORS = {"G4", "H100", "L4", "V5E1", "V6E1"}
TPU_ACCELERATORS = {"V5E1", "V6E1"}


def strip_colab_cli_banners(text: str) -> str:
    """剥离 google-colab-cli v0.8.0+ 的版本更新提示横幅与 ANSI 控制序列，避免干扰输出解析。"""
    clean = ANSI_ESCAPE_RE.sub("", text or "")
    kept = [ln for ln in clean.splitlines() if not COLAB_UPDATE_BANNER_RE.match(ln)]
    return "\n".join(kept).strip()


def build_colab_new_cmd(colab_bin: str, session_name: str, accelerator: str, high_mem: bool) -> List[str]:
    acc_raw = (accelerator or "T4").strip() or "T4"
    acc_up = acc_raw.upper()
    cmd = [colab_bin, "new", "-s", session_name]
    if acc_up in TPU_ACCELERATORS:
        cmd.extend(["--tpu", acc_raw.lower()])
    elif acc_up != "CPU":
        cmd.extend(["--gpu", acc_raw])
    if high_mem and acc_up not in SINGLE_SHAPE_ACCELERATORS:
        cmd.append("--high-mem")
    return cmd


def is_high_mem_rejected_error(raw_output: str) -> bool:
    low = strip_colab_cli_banners(raw_output).lower()
    return (
        "backend rejected accelerator" in low
        or "400 bad request" in low
        or ("http 400" in low and "assign" in low)
        or "unsupported memory shape" in low
    )


def format_colab_alloc_error(accelerator: str, raw_output: str) -> str:
    cleaned = strip_colab_cli_banners(raw_output) or "未知错误"
    low = cleaned.lower()
    acc_label = (accelerator or "GPU").strip().upper()
    if any(k in low for k in ("503", "service unavailable", "allocation refused", "out of capacity", "stockout")):
        return (
            f"Google Colab 云端 {acc_label} 算力池当前暂时库存不足/排队拥堵 (HTTP 503 Service Unavailable)，"
            f"建议改选 G4 / L4 / T4 或稍后重试。详情: {cleaned}"
        )
    if "toomanyassignments" in low or "412" in low or "precondition failed" in low:
        return (
            f"Google Colab 拒绝分配 {acc_label} 实例 (已达并发会话上限或账户计算单元不足，HTTP 412)。详情: {cleaned}"
        )
    if "backend rejected accelerator" in low or "400" in low:
        return (
            f"Google Colab 后端拒绝 {acc_label} 规格请求 (HTTP 400：请检查订阅等级是否支持该加速器或关闭高内存选项)。详情: {cleaned}"
        )
    return cleaned


def colab_cfg_dir() -> str:
    cfg_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(cfg_dir, exist_ok=True)
    return cfg_dir


def api_keys_path() -> str:
    return os.path.join(colab_cfg_dir(), "llm_api_keys.json")


def upsert_vm_api_key_entry(
    vm_name: str,
    slot: str,
    model_name: str,
    engine: str,
    custom_api_key: str = "",
    gateway_port: int = 18080,
    remote_port: int = 8000,
) -> Dict[str, Any]:
    """注册或更新单台大模型虚拟机在 `llm_api_keys.json` 中的专属端口与 API Key（原子写入）。"""
    path = api_keys_path()
    root: Dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    root = loaded
        except Exception:
            root = {}
    vms_map = root.get("vms")
    if not isinstance(vms_map, dict):
        vms_map = {}
    existing = vms_map.get(vm_name) if isinstance(vms_map.get(vm_name), dict) else {}

    clean_vm = "".join(c if (c.isalnum() or c == "-") else "-" for c in vm_name) or "vm"
    key_val = (custom_api_key or "").strip() or str(existing.get("api_key") or "").strip()
    if not key_val:
        key_val = f"sk-colab-{clean_vm}-{secrets.token_hex(8)}"

    slot_norm = (slot or "standalone").lower()
    local_port = int(existing.get("local_port") or 0)
    if slot_norm == "blue" and vm_name.endswith("-blue"):
        local_port = 18081
    elif slot_norm == "green" and vm_name.endswith("-green"):
        local_port = 18082
    elif local_port < 18083:
        used_ports = {
            int(v.get("local_port"))
            for k, v in vms_map.items()
            if k != vm_name and isinstance(v, dict) and str(v.get("local_port") or "").isdigit()
        }
        cand = 18083
        while cand in used_ports:
            cand += 1
        local_port = cand

    api_base_url = f"http://127.0.0.1:{local_port}/v1"
    gateway_base_url = f"http://127.0.0.1:{gateway_port}/v1"
    entry = {
        "vm_name": vm_name,
        "slot": slot_norm,
        "api_key": key_val,
        "local_port": local_port,
        "remote_port": int(remote_port),
        "model_name": model_name,
        "engine": engine,
        "api_base_url": api_base_url,
        "gateway_base_url": gateway_base_url,
    }
    vms_map[vm_name] = entry
    root["vms"] = vms_map
    if not root.get("gateway_api_key"):
        root["gateway_api_key"] = f"sk-colab-gateway-{secrets.token_hex(8)}"
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(root, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return entry


def find_bin(name: str) -> str:
    cand = shutil.which(name)
    if cand:
        return cand
    ext = ".exe" if os.name == "nt" else ""
    p = os.path.expanduser(f"~/.local/bin/{name}{ext}")
    return p if os.path.exists(p) else name


def find_rclone_conf() -> str:
    if os.environ.get("RCLONE_CONFIG"):
        return os.environ["RCLONE_CONFIG"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        p = os.path.join(appdata, "rclone", "rclone.conf")
        if os.path.exists(p):
            return p
    return os.path.expanduser("~/.config/rclone/rclone.conf")


def bg_state_path() -> str:
    cfg_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "llm_bluegreen_state.json")


def read_bg_state() -> Dict[str, Any]:
    p = bg_state_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_bg_state(data: Dict[str, Any]) -> None:
    p = bg_state_path()
    data["updated_at"] = time.time()
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)


def register_workspace(session_name: str, workspace_dir: str, description: str) -> None:
    cfg_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(cfg_dir, exist_ok=True)
    reg_file = os.path.join(cfg_dir, "workspaces.json")
    data = {}
    if os.path.exists(reg_file):
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    ws_name = os.path.basename(os.path.abspath(workspace_dir))
    data[session_name] = {
        "workspace_name": ws_name,
        "workspace_path": os.path.abspath(workspace_dir),
        "active_script": "llm_metrics_sidecar.py",
        "description": description,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = f"{reg_file}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, reg_file)


def has_colab_oauth_credentials() -> Tuple[bool, str]:
    """检查 colab-cli OAuth 凭证（兼容 v0.7.x `token.json` 与 v0.8.0+ OS Keyring 双存储源，不打印任何明文令牌）。"""
    token_file = os.path.join(colab_cfg_dir(), "token.json")
    if os.path.exists(token_file):
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                tok_obj = json.load(f)
            if isinstance(tok_obj, dict) and (tok_obj.get("token") or tok_obj.get("refresh_token")):
                return True, "token.json"
        except Exception:
            pass
    try:
        import keyring  # type: ignore

        raw_kr = keyring.get_password("google-colab-cli", "oauth_token")
        if raw_kr:
            tok_obj = json.loads(raw_kr)
            if isinstance(tok_obj, dict) and (tok_obj.get("token") or tok_obj.get("refresh_token")):
                return True, "keyring"
    except Exception:
        pass
    py_cli = resolve_colab_cli_python()
    if py_cli and os.path.exists(py_cli) and os.path.abspath(py_cli) != os.path.abspath(sys.executable):
        try:
            probe = subprocess.run(
                [
                    py_cli,
                    "-B",
                    "-c",
                    (
                        "import json, keyring; raw = keyring.get_password('google-colab-cli', 'oauth_token'); "
                        "d = json.loads(raw) if raw else {}; "
                        "print('KEYRING_OK' if isinstance(d, dict) and (d.get('token') or d.get('refresh_token')) else 'NONE')"
                    ),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=8,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if "KEYRING_OK" in (probe.stdout or ""):
                return True, "keyring"
        except Exception:
            pass
    return False, "missing"


def run_cmd(cmd: List[str], timeout: int = 300, stdin_text: Optional[str] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["NO_COLOR"] = "1"
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        env=env,
    )


# ─────────────────────────── 1. 上云前配置与网盘权重预检 (Precheck) ───────────────────────────


def cmd_precheck(args: argparse.Namespace) -> int:
    sidecar_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_metrics_sidecar.py")
    colab_bin = find_bin("colab")
    rclone_bin = find_bin("rclone")
    rclone_conf = find_rclone_conf()

    print("=" * 72)
    print("🔍 [Colab Pilot] 大语言模型上云前配置与网盘权重预检 (Precheck)")
    print("   说明: 大模型真实冒烟 (Cloud Warmup Smoke) 将直接在 Colab GPU 实例上线前执行")
    print("=" * 72)

    print("\n[1/3] 正在校验云端计量边车脚本语法...")
    try:
        with open(sidecar_script, "r", encoding="utf-8") as f:
            src = f.read()
        ast.parse(src, filename=sidecar_script)
        compile(src, sidecar_script, "exec")
        print(f"  ✅ 语法校验通过: {sidecar_script}")
    except Exception as e:
        print(f"  ❌ 语法错误: {e}")
        return 1

    print("\n[2/3] 正在校验 colab-cli 与 rclone 环境...")
    if not os.path.exists(colab_bin) and not shutil.which("colab"):
        print(f"  ❌ 未找到 colab-cli: {colab_bin}")
        return 1
    if not os.path.exists(rclone_conf) or os.path.getsize(rclone_conf) < 20:
        print(f"  ❌ 未找到有效的 rclone.conf: {rclone_conf}")
        return 1
    cred_ok, cred_src = has_colab_oauth_credentials()
    if cred_ok:
        print(f"  ✅ colab-cli ({colab_bin}, 凭证源={cred_src}) 与 rclone.conf 就绪。")
    else:
        print(f"  ⚠️ 未检测到本机 colab-cli OAuth 凭证 (token.json / Keyring)，上云前可能需要先运行 `colab auth login`。")
        print(f"  ✅ colab-cli ({colab_bin}) 与 rclone.conf 可执行环境就绪。")

    if args.gdrive_model_path:
        if args.gdrive_model_path.startswith("-"):
            print(f"  ❌ 非法的 Google Drive 模型路径 (不能以 '-' 开头): {args.gdrive_model_path}")
            return 1
        print(f"\n[3/3] 正在校验 Google Drive 模型权重路径 ({args.gdrive_model_path})...")
        res = run_cmd(
            [
                rclone_bin,
                "--config",
                rclone_conf,
                "--retries=5",
                "--low-level-retries=20",
                "lsf",
                "--",
                args.gdrive_model_path,
            ],
            timeout=90,
        )
        if res.returncode != 0 or not res.stdout.strip():
            print(f"  ❌ Google Drive 模型路径为空或不可访问: {args.gdrive_model_path}")
            print(f"     {res.stderr.strip()[:240]}")
            return 1
        files = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        print(f"  ✅ Google Drive 模型权重路径校验通过（共发现 {len(files)} 个文件/子目录）。")
    else:
        print("\n[3/3] 未指定 Google Drive 权重路径，将直接从 HuggingFace / Ollama 拉取模型。")

    print("\n🎉 [PRECHECK OK] 上云前预检通过！接下来将在 Colab 云端实例上执行真机 Warmup 冒烟与部署。")
    return 0


# ─────────────────────────── 2. 云端单槽位部署与真机 Warmup 冒烟 ───────────────────────────

HF_OPENAI_SERVER_PY = r'''#!/usr/bin/env python3
import json, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoModelForCausalLM, TextIteratorStreamer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

MODEL_ID = sys.argv[1]
print(f"[HF-Engine] 正在从 HuggingFace 高速拉取并加载模型: {MODEL_ID} ...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = None
for cls_name in ("Qwen3_5ForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
    cls = getattr(transformers, cls_name, None)
    if cls is None:
        continue
    try:
        model = cls.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
        break
    except Exception as e:
        print(f"[HF-Engine] {cls_name} 尝试跳过: {e}", flush=True)
if model is None:
    raise RuntimeError(f"无法加载模型: {MODEL_ID}")
model.eval()
vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
print(f"[HF-Engine] 模型 {MODEL_ID} 已成功加载至 GPU 显存！耗时={time.time()-t0:.1f}s, 显存占用={vram_gb:.2f} GB", flush=True)
GPU_LOCK = threading.Lock()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args): pass
    def do_GET(self):
        if self.path.startswith("/health") or self.path.startswith("/v1/models"):
            body = json.dumps({"status": "ok", "model": MODEL_ID}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length > 0 else b"{}"
        req = json.loads(raw.decode("utf-8", errors="ignore") or "{}")
        msgs = req.get("messages") or [{"role": "user", "content": "Hello"}]
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 64)
        try:
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:
            prompt = "\n".join(f"{m.get('role','user')}: {m.get('content','')}" for m in msgs) + "\nassistant: "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(**inputs, streamer=streamer, max_new_tokens=max_new, do_sample=False)

        gen_errors = []

        def _run():
            try:
                with GPU_LOCK, torch.inference_mode():
                    model.generate(**gen_kwargs)
            except Exception as exc:
                gen_errors.append(str(exc))
                print(f"[HF-Engine] generate error: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        self.close_connection = True
        stream_iter = iter(streamer)
        first_piece = None
        for cand in stream_iter:
            if cand:
                first_piece = cand
                break

        if first_piece is None and gen_errors:
            err_body = json.dumps({"error": {"message": "; ".join(gen_errors)}}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err_body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err_body)
            self.wfile.flush()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:10]}"

        def _emit_piece(piece_text):
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": piece_text}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        if first_piece:
            _emit_piece(first_piece)
        for piece in stream_iter:
            if not piece:
                continue
            _emit_piece(piece)
        if gen_errors:
            err_chunk = {"error": {"message": "; ".join(gen_errors)}}
            self.wfile.write(f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
        done_chunk = {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(done_chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
        self.wfile.flush()

ThreadedHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
'''

LLAMA_CPP_RUNNER_PY = r'''#!/usr/bin/env python3
import glob, os, re, shutil, subprocess, sys, time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

MODEL_SPEC = sys.argv[1]
MODEL_DIR = "/content/model"
os.makedirs(MODEL_DIR, exist_ok=True)

llama_server_bin = "/content/llama_bin/llama-b12219/llama-server"
if not os.path.exists(llama_server_bin):
    print("[LlamaCpp-Engine] 正在下载并解压官方预编译 CUDA 版 llama-server (b12219)...", flush=True)
    os.makedirs("/content/llama_bin", exist_ok=True)
    subprocess.run(
        "cd /content/llama_bin && "
        "curl -fsSL https://github.com/ggml-org/llama.cpp/releases/download/b12219/llama-b12219-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz && "
        "curl -fsSL https://github.com/ggml-org/llama.cpp/releases/download/b12219/cudart-llama-b12219-bin-ubuntu-cuda-12.8-x64.tar.gz | tar -xz",
        shell=True,
        check=True,
    )

def find_primary_gguf(root_dir: str):
    all_ggufs = sorted(glob.glob(os.path.join(root_dir, "**", "*.gguf"), recursive=True))
    main_ggufs = [
        p for p in all_ggufs
        if not os.path.basename(p).startswith(("mmproj", "mtp"))
        and not re.search(r"-(?!00001\b)\d{5}-of-\d{5}\.gguf$", os.path.basename(p), re.IGNORECASE)
    ]
    for p in main_ggufs:
        if re.search(r"-00001-of-\d{5}\.gguf$", os.path.basename(p), re.IGNORECASE):
            return p
    return main_ggufs[0] if main_ggufs else None

gguf_path = find_primary_gguf(MODEL_DIR)
if not gguf_path and MODEL_SPEC != MODEL_DIR:
    repo_id = MODEL_SPEC
    subdir = None
    if ":" in MODEL_SPEC and not MODEL_SPEC.startswith(("http://", "https://")):
        repo_id, subdir = MODEL_SPEC.split(":", 1)
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    allow_patterns = ["*.gguf"]
    if subdir:
        allow_patterns = [f"{subdir.rstrip('/')}/*.gguf"]
    else:
        for pref in ("IQ3_S/", "Q4_K_M/", "Q4_0/", "IQ3_XXS/", "IQ2_XS/", "Q2_0/"):
            if any(f.startswith(pref) and f.endswith(".gguf") for f in files):
                allow_patterns = [f"{pref}*.gguf"]
                print(f"[LlamaCpp-Engine] 检测到多档位 GGUF 仓库，自动选择推荐档位: {pref.rstrip('/')}", flush=True)
                break
    print(f"[LlamaCpp-Engine] 正在从 HuggingFace 高速拉取 GGUF 权重: {repo_id} ({allow_patterns}) ...", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=repo_id,
        local_dir=MODEL_DIR,
        allow_patterns=allow_patterns,
    )
    print(f"[LlamaCpp-Engine] GGUF 权重拉取完成！耗时={time.time()-t0:.1f}s", flush=True)
    gguf_path = find_primary_gguf(MODEL_DIR)

if not gguf_path:
    raise RuntimeError(f"未在 {MODEL_DIR} 中找到可用的 .gguf 模型文件")

ctx_size = os.environ.get("LLM_CTX_SIZE", "").strip()
if not ctx_size:
    ctx_size = "32768"
    try:
        vram_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        total_vram_mib = int(vram_out.splitlines()[0].strip())
        if total_vram_mib >= 60000:
            ctx_size = "131072"
    except Exception:
        pass

print(f"[LlamaCpp-Engine] 正在启动 llama-server 加载模型: {gguf_path} (ctx={ctx_size}) ...", flush=True)
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/content/llama_bin/llama-b12219:/content/llama_bin/cudart-llama-b12219-bin-ubuntu-cuda-12.8-x64:" + env.get("LD_LIBRARY_PATH", "")
cmd = [
    llama_server_bin,
    "-m", gguf_path,
    "-a", MODEL_SPEC,
    "-lm", "mmap",
    "--lazy-mode", "on",
    "-ngl", "99",
    "-c", ctx_size,
    "--reasoning-budget", "0",
    "--host", "127.0.0.1",
    "--port", "8001",
]
os.execvpe(llama_server_bin, cmd, env)
'''


def build_cloud_engine_boot_code(
    slot: str,
    role: str,
    engine: str,
    model_name: str,
    gdrive_model_path: str,
    workspace_abs: str,
) -> str:
    """生成在 Colab 实例上拉取模型、启动底层推理引擎 (:8001) 与计量边车 (:8000) 并执行真机 Warmup 冒烟测速的脚本"""
    return f"""
import http.client, json, os, shutil, socket, subprocess, sys, time

os.chdir("/content")
if not shutil.which("rclone"):
    subprocess.run("curl -fsSL https://rclone.org/install.sh | sudo bash", shell=True, check=False)

gdrive_model_path = {json.dumps(gdrive_model_path)}
if gdrive_model_path:
    os.makedirs("/content/model", exist_ok=True)
    print(f"[{slot.upper()}] 正在通过 rclone 从 {{gdrive_model_path}} 拉取模型权重...", flush=True)
    rc = subprocess.run([
        "rclone", "--config", "/content/rclone.conf",
        "--retries=5", "--low-level-retries=20", "--transfers=8",
        "copy", "--", gdrive_model_path, "/content/model"
    ], check=False)
    if rc.returncode != 0:
        raise RuntimeError("CLOUD_WARMUP_FAILED: rclone 拉取 Google Drive 模型权重失败")

# 清理可能残留的旧边车进程
subprocess.run("pkill -f llm_metrics_sidecar.py || true", shell=True)

engine = {json.dumps(engine)}
model_name = {json.dumps(model_name)}
builtin_flag = ["--builtin-engine"]
engine_log = "/content/engine.log"

if engine == "ollama":
    if not shutil.which("ollama"):
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=False)
    env_o = os.environ.copy()
    env_o["OLLAMA_HOST"] = "127.0.0.1:8001"
    subprocess.Popen(["ollama", "serve"], env=env_o, stdout=open(engine_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(3)
    pull_rc = subprocess.run(["ollama", "pull", model_name], env=env_o, check=False)
    if pull_rc.returncode != 0:
        raise RuntimeError(f"CLOUD_WARMUP_FAILED: ollama pull {{model_name}} 失败")
    builtin_flag = []
elif engine == "vllm":
    vllm_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/content/model" if gdrive_model_path else model_name,
        "--served-model-name", model_name,
        "--host", "127.0.0.1", "--port", "8001",
        "--trust-remote-code",
    ]
    low_m = model_name.lower()
    if "qwen3" in low_m or "qwq" in low_m:
        vllm_cmd.extend(["--reasoning-parser", "qwen3"])
    elif "deepseek-r1" in low_m or "deepseek-v3" in low_m:
        vllm_cmd.extend(["--reasoning-parser", "deepseek_r1"])
    subprocess.Popen(vllm_cmd, stdout=open(engine_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    builtin_flag = []
elif engine == "llama.cpp":
    subprocess.run("pkill -f llama-server || true", shell=True)
    subprocess.run("pkill -f llama_cpp_runner.py || true", shell=True)
    subprocess.run(["uv", "pip", "install", "--system", "-q", "hf_transfer", "huggingface_hub"], check=False)
    with open("/content/llama_cpp_runner.py", "w", encoding="utf-8") as f:
        f.write({json.dumps(LLAMA_CPP_RUNNER_PY)})
    env_lc = os.environ.copy()
    env_lc["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    eng_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/llama_cpp_runner.py", "/content/model" if gdrive_model_path else model_name],
        env=env_lc,
        stdout=open(engine_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    builtin_flag = []
elif engine == "transformers":
    subprocess.run(["uv", "pip", "install", "--system", "-q", "flash-linear-attention", "hf_transfer"], check=False)
    with open("/content/hf_openai_server.py", "w", encoding="utf-8") as f:
        f.write({json.dumps(HF_OPENAI_SERVER_PY)})
    env_hf = os.environ.copy()
    env_hf["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    eng_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/hf_openai_server.py", "/content/model" if gdrive_model_path else model_name],
        env=env_hf,
        stdout=open(engine_log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    builtin_flag = []

env_s = os.environ.copy()
env_s["COLAB_WORKSPACE"] = {json.dumps(workspace_abs)}
sidecar_cmd = [
    sys.executable, "/content/llm_metrics_sidecar.py",
    "--port", "8000",
    "--backend-url", "http://127.0.0.1:8001",
    "--slot", {json.dumps(slot)},
    "--role", {json.dumps(role)},
    "--engine", engine,
    "--model-name", model_name,
    "--gdrive-model-path", gdrive_model_path,
] + builtin_flag

subprocess.Popen(
    sidecar_cmd,
    env=env_s,
    stdout=open("/content/llm_sidecar.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

# 等待底层推理引擎 (:8001) 与计量边车 (:8000) 就绪（Colab 云端真机健康探测）
backend_ready = (engine == "builtin")
probe_path = "/health" if engine == "llama.cpp" else "/v1/models"
for i in range(1200):
    if not backend_ready:
        if "eng_proc" in locals() and eng_proc.poll() is not None:
            break
        try:
            c8001 = http.client.HTTPConnection("127.0.0.1", 8001, timeout=3)
            c8001.request("GET", probe_path)
            r8001 = c8001.getresponse()
            _ = r8001.read()
            c8001.close()
            if (r8001.status == 200 if engine == "llama.cpp" else r8001.status < 500):
                backend_ready = True
        except Exception:
            pass
        if not backend_ready and i % 5 == 0 and os.path.exists(engine_log):
            raw_log = open(engine_log, "r", errors="ignore").read().replace("\\r", "\\n").strip().splitlines()
            tail_msg = raw_log[-1] if raw_log else "waiting..."
            du_str = ""
            if os.path.isdir("/content/model"):
                try:
                    du_str = " | model_dir=" + subprocess.check_output(["du", "-sh", "/content/model"], text=True).split()[0]
                except Exception:
                    pass
            print(f"[Engine] {{tail_msg[-160:]}}{{du_str}}", flush=True)
    if backend_ready:
        break
    time.sleep(2.0)

healthy = False
if backend_ready:
    for _ in range(30):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
            conn.request("GET", "/pilot/health")
            r = json.loads(conn.getresponse().read().decode("utf-8"))
            conn.close()
            if r.get("ok"):
                healthy = True
                break
        except Exception:
            pass
        time.sleep(1.0)

if not healthy:
    err_tail = ""
    if os.path.exists(engine_log):
        err_tail = open(engine_log, "r", errors="ignore").read()[-800:]
    raise RuntimeError(f"CLOUD_WARMUP_FAILED: 云端推理引擎未就绪 (可能发生 CUDA OOM 或模型加载报错)\\n{{err_tail}}")

# 在 Colab 云端真机发送 Warmup 流式推理请求，实测真实 Token 输出频率 (tok/s) 与首字延迟 (TTFT)
conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=60)
body = json.dumps({{
    "model": model_name,
    "messages": [{{"role": "user", "content": "Warmup verification check"}}],
    "stream": True,
    "max_tokens": 32,
}})
conn.request("POST", "/v1/chat/completions", body=body, headers={{"Content-Type": "application/json"}})
resp = conn.getresponse()
raw_warmup = resp.read().decode("utf-8", errors="ignore")
conn.close()

if resp.status >= 400:
    raise RuntimeError(f"CLOUD_WARMUP_FAILED: 云端 Warmup 推理请求返回 HTTP {{resp.status}}: {{raw_warmup[:300]}}")

warmup_pieces = []
for _ln in raw_warmup.splitlines():
    _s = _ln.strip()
    if not _s.startswith("data:"):
        continue
    _payload = _s[5:].strip()
    if not _payload or _payload == "[DONE]":
        continue
    try:
        _obj = json.loads(_payload)
    except Exception:
        continue
    if isinstance(_obj, dict):
        if _obj.get("error"):
            raise RuntimeError(f"CLOUD_WARMUP_FAILED: 云端 Warmup 流式生成返回异常: {{_obj['error']}}")
        _choices = _obj.get("choices") or []
        if _choices:
            _delta = _choices[0].get("delta") or {{}}
            _txt = _delta.get("content") or _delta.get("reasoning_content") or _choices[0].get("text") or ""
            if _txt:
                warmup_pieces.append(_txt)

if not "".join(warmup_pieces).strip():
    raise RuntimeError(f"CLOUD_WARMUP_FAILED: 云端 Warmup 未产出有效文本内容: {{raw_warmup[:300]}}")

with open("/content/.colab_pilot_llm_state.json", "r", encoding="utf-8") as f:
    st = json.load(f)

if (st.get("total_completion_tokens") or 0) <= 0:
    raise RuntimeError("CLOUD_WARMUP_FAILED: 云端 Warmup 推理生成的 Token 数为 0")

print("SLOT_READY", json.dumps({{
    "slot": st.get("slot"),
    "role": st.get("role"),
    "instant_tps": st.get("instant_tps"),
    "ttft_ms": st.get("ttft_ms"),
    "total_tokens": st.get("total_completion_tokens"),
}}), flush=True)
"""


def provision_and_warmup_slot(
    session_prefix: str,
    slot: str,
    role: str,
    gpu: str,
    high_mem: bool,
    engine: str,
    model_name: str,
    gdrive_model_path: str,
    workspace_abs: str,
    session_name: Optional[str] = None,
    keep_vm_on_fail: bool = False,
) -> Dict[str, Any]:
    colab_bin = find_bin("colab")
    rclone_conf = find_rclone_conf()
    sidecar_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_metrics_sidecar.py")
    target_session = (session_name or "").strip() or f"{session_prefix}-{slot}"
    acc_up = (gpu or "T4").strip().upper()

    register_workspace(
        session_name=target_session,
        workspace_dir=workspace_abs,
        description=f"LLM [{slot.upper()}/{role.upper()}] {model_name} ({engine})",
    )

    chk = run_cmd([colab_bin, "status", "-s", target_session], timeout=30)
    chk_out = strip_colab_cli_banners((chk.stdout or "") + "\n" + (chk.stderr or ""))
    has_active_session = chk.returncode == 0 and f"[{target_session}]" in chk_out
    hw_matches = (
        keep_vm_on_fail
        or acc_up == "CPU"
        or f"Hardware: {acc_up}" in chk_out
        or f"TPU: {acc_up}" in chk_out
    )
    if has_active_session and hw_matches:
        first_line = next((ln for ln in chk_out.splitlines() if ln.strip()), target_session)
        print(f"♻️ [{slot.upper()}] 复用已就绪的 Colab 实例: {first_line}", flush=True)
    else:
        new_cmd = build_colab_new_cmd(colab_bin, target_session, gpu, high_mem)
        print(f"🚀 [{slot.upper()}] 正在分配 Colab 实例: {' '.join(new_cmd)}", flush=True)
        res_new = run_cmd(new_cmd, timeout=300)
        combined_new = (res_new.stderr or "") + "\n" + (res_new.stdout or "")
        if res_new.returncode != 0 and "--high-mem" in new_cmd and is_high_mem_rejected_error(combined_new):
            fallback_cmd = [c for c in new_cmd if c != "--high-mem"]
            print(
                f"⚠️ [{slot.upper()}] 检测到后端拒绝 --high-mem 规格 (HTTP 400)，自动移除 --high-mem 重试: {' '.join(fallback_cmd)}",
                flush=True,
            )
            res_new = run_cmd(fallback_cmd, timeout=300)
            combined_new = (res_new.stderr or "") + "\n" + (res_new.stdout or "")
        if res_new.returncode != 0 and "already exists" not in combined_new.lower():
            raise RuntimeError(f"创建实例 {target_session} 失败: {format_colab_alloc_error(gpu, combined_new)}")

    try:
        ws_meta_file = os.path.join(workspace_abs, f".colab_workspace_{slot}_{os.getpid()}.json")
        with open(ws_meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": os.path.basename(workspace_abs),
                    "path": workspace_abs,
                    "description": f"LLM Slot [{slot.upper()}] · {model_name} ({engine})",
                },
                f,
                ensure_ascii=False,
            )
        try:
            run_cmd([colab_bin, "upload", "-s", target_session, ws_meta_file, "/content/.colab_workspace.json"], timeout=90)
        finally:
            if os.path.exists(ws_meta_file):
                os.remove(ws_meta_file)

        if os.path.exists(rclone_conf):
            up_rc = run_cmd([colab_bin, "upload", "-s", target_session, rclone_conf, "/content/rclone.conf"], timeout=90)
            if up_rc.returncode != 0 and gdrive_model_path:
                err_rc = strip_colab_cli_banners((up_rc.stderr or "") + "\n" + (up_rc.stdout or ""))
                raise RuntimeError(f"上传 rclone.conf 失败: {err_rc}")
        up_sidecar = run_cmd([colab_bin, "upload", "-s", target_session, sidecar_abs, "/content/llm_metrics_sidecar.py"], timeout=90)
        if up_sidecar.returncode != 0:
            err_sc = strip_colab_cli_banners((up_sidecar.stderr or "") + "\n" + (up_sidecar.stdout or ""))
            raise RuntimeError(f"上传 llm_metrics_sidecar.py 失败: {err_sc}")

        print(f"⚙️ [{slot.upper()}] 正在 Colab 云端真机启动推理引擎 ({engine}) + 计量边车并执行 Warmup 冒烟测速...", flush=True)
        boot_code = build_cloud_engine_boot_code(
            slot=slot,
            role=role,
            engine=engine,
            model_name=model_name,
            gdrive_model_path=gdrive_model_path,
            workspace_abs=workspace_abs,
        )
        env_exec = os.environ.copy()
        env_exec["PYTHONUTF8"] = "1"
        env_exec["PYTHONIOENCODING"] = "utf-8"
        env_exec["NO_COLOR"] = "1"
        p_exec = subprocess.Popen(
            [colab_bin, "exec", "-s", target_session, "--timeout", "2400"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env_exec,
        )
        assert p_exec.stdin is not None and p_exec.stdout is not None
        p_exec.stdin.write(boot_code)
        p_exec.stdin.close()
        timed_out = {"v": False}

        def _kill_stuck_exec() -> None:
            timed_out["v"] = True
            try:
                p_exec.kill()
            except Exception:
                pass

        watchdog = threading.Timer(2460.0, _kill_stuck_exec)
        watchdog.daemon = True
        watchdog.start()
        out_lines: List[str] = []
        try:
            for line in p_exec.stdout:
                if COLAB_UPDATE_BANNER_RE.match(ANSI_ESCAPE_RE.sub("", line)):
                    continue
                out_lines.append(line)
                if any(k in line for k in ("[Engine]", "[HF-Engine]", "[LlamaCpp-Engine]", "SLOT_READY", "CLOUD_WARMUP_FAILED", "Error", "Traceback")):
                    print("  " + line, end="", flush=True)
            p_exec.wait(timeout=30)
        finally:
            watchdog.cancel()
            try:
                p_exec.stdout.close()
            except Exception:
                pass
        out = strip_colab_cli_banners("".join(out_lines))
        if timed_out["v"]:
            raise RuntimeError(f"槽位 {target_session} 云端预热超时 (2460s):\n{out[-1000:]}")
        if p_exec.returncode not in (0, None):
            raise RuntimeError(f"槽位 {target_session} colab exec 异常退出 (code={p_exec.returncode}):\n{out[-1000:]}")
        for line in out.splitlines():
            if "SLOT_READY" in line:
                j_str = line.split("SLOT_READY", 1)[1].strip()
                info = json.loads(j_str)
                print(
                    f"✅ [{slot.upper()}] Colab 云端真机 Warmup 冒烟通过！会话={target_session}, 角色={info.get('role')}, "
                    f"实测速率={info.get('instant_tps')} tok/s, TTFT={info.get('ttft_ms')} ms",
                    flush=True,
                )
                return info
        raise RuntimeError(f"槽位 {target_session} 云端真机 Warmup 冒烟未通过:\n{out[-1000:]}")
    except Exception as e:
        if keep_vm_on_fail:
            print(
                f"⚠️ [保留现有实例] 槽位 {target_session} 启动或预热异常，已按 --keep-vm-on-fail 保留实例不关机！",
                flush=True,
            )
        else:
            print(
                f"❌ [云端真机冒烟失败 / 自动止损] 槽位 {target_session} 启动或预热异常，正在立即执行 `colab stop -s {target_session}` 销毁实例停止计费！",
                flush=True,
            )
            run_cmd([colab_bin, "stop", "-s", target_session], timeout=120)
        raise e


def set_remote_slot_role(session_name: str, new_role: str) -> None:
    colab_bin = find_bin("colab")
    code = f"""
import http.client, json
conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=5)
conn.request("POST", "/pilot/role", body=json.dumps({{"role": {json.dumps(new_role)}}}), headers={{"Content-Type": "application/json"}})
print(conn.getresponse().read().decode("utf-8"))
conn.close()
"""
    run_cmd([colab_bin, "exec", "-s", session_name, "--timeout", "30"], timeout=45, stdin_text=code)


def resolve_colab_cli_python() -> str:
    """跨平台定位安装了 `colab_cli` 的 `uv` 工具虚拟环境 Python 解释器 (H-7)。"""
    candidates: List[str] = []
    uv_tool_dir = os.environ.get("UV_TOOL_DIR")
    if uv_tool_dir:
        candidates.extend(
            [
                os.path.join(uv_tool_dir, "google-colab-cli", "Scripts", "pythonw.exe"),
                os.path.join(uv_tool_dir, "google-colab-cli", "Scripts", "python.exe"),
                os.path.join(uv_tool_dir, "google-colab-cli", "bin", "python"),
            ]
        )
    if os.name == "nt":
        for env_key in ("APPDATA", "LOCALAPPDATA"):
            base = os.environ.get(env_key)
            if base:
                candidates.append(os.path.join(base, "uv", "tools", "google-colab-cli", "Scripts", "pythonw.exe"))
                candidates.append(os.path.join(base, "uv", "tools", "google-colab-cli", "Scripts", "python.exe"))
        candidates.append(os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/Scripts/python.exe"))
    else:
        candidates.append(os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python"))
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            candidates.append(os.path.join(xdg_data, "uv", "tools", "google-colab-cli", "bin", "python"))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return sys.executable


def is_gateway_process_pid(pid: int) -> bool:
    """终止残留 `.pid` 前校验目标进程确为 Python 网关进程，防止 PID 复用误杀无关进程 (H-3)。"""
    if pid <= 4 or pid == os.getpid():
        return False
    try:
        if os.name == "nt":
            res = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=5,
                creationflags=0x08000000,
            )
            out_lower = (res.stdout or "").lower()
            return str(pid) in out_lower and "python" in out_lower
        proc_cmdline = f"/proc/{pid}/cmdline"
        if os.path.exists(proc_cmdline):
            raw = open(proc_cmdline, "rb").read().decode("utf-8", errors="ignore").lower()
            return "python" in raw or "llm_local_gateway" in raw
        res = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm=,args="],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=5,
        )
        out_lower = (res.stdout or "").lower()
        return res.returncode == 0 and ("python" in out_lower or "llm_local_gateway" in out_lower)
    except Exception:
        return False


def ensure_local_gateway(gateway_port: int = 18080) -> None:
    import socket

    try:
        with socket.create_connection(("127.0.0.1", gateway_port), timeout=0.8):
            return
    except Exception:
        pass
    gw_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_local_gateway.py")
    if not os.path.exists(gw_script):
        return
    py_exe = resolve_colab_cli_python()
    kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        # 严格单独使用 CREATE_NO_WINDOW (0x08000000)，绝不叠加 DETACHED_PROCESS
        kwargs["creationflags"] = 0x08000000
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [py_exe, gw_script, "--port", str(gateway_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    for _ in range(15):
        time.sleep(0.25)
        try:
            with socket.create_connection(("127.0.0.1", gateway_port), timeout=0.5):
                return
        except Exception:
            pass


def stop_local_gateway() -> None:
    pid_file = os.path.join(os.path.expanduser("~/.config/colab-cli"), "llm_local_gateway.pid")
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file, "r", encoding="utf-8").read().strip())
            if is_gateway_process_pid(pid):
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        check=False,
                        creationflags=0x08000000,
                    )
                else:
                    os.kill(pid, 15)
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass


def _resolve_deploy_target(args: argparse.Namespace) -> Tuple[str, str, str]:
    """返回 `(effective_prefix, slot, target_session)`，同时兼容 `--session-name` 与 `--session-prefix` + `--slot`。"""
    slot = (getattr(args, "slot", None) or "blue").strip().lower() or "blue"
    explicit_name = (getattr(args, "session_name", None) or "").strip()
    prefix = (getattr(args, "session_prefix", None) or "llm").strip() or "llm"
    if explicit_name:
        if slot in ("blue", "green") and explicit_name.endswith(f"-{slot}") and len(explicit_name) > len(slot) + 1:
            prefix = explicit_name[: -(len(slot) + 1)]
        return prefix, slot, explicit_name
    if slot == "standalone":
        return prefix, slot, prefix
    return prefix, slot, f"{prefix}-{slot}"


def cmd_deploy(args: argparse.Namespace) -> int:
    workspace_abs = os.path.abspath(args.workspace)
    effective_prefix, slot, target_session = _resolve_deploy_target(args)
    keep_vm_on_fail = bool(getattr(args, "keep_vm_on_fail", False))
    custom_api_key = (getattr(args, "api_key", None) or "").strip()
    effective_high_mem = bool(args.high_mem and (args.gpu or "T4").strip().upper() not in SINGLE_SHAPE_ACCELERATORS)

    prev_st = read_bg_state()
    try:
        info = provision_and_warmup_slot(
            session_prefix=effective_prefix,
            slot=slot,
            role="active",
            gpu=args.gpu,
            high_mem=args.high_mem,
            engine=args.engine,
            model_name=args.model_name,
            gdrive_model_path=args.gdrive_model_path,
            workspace_abs=workspace_abs,
            session_name=target_session,
            keep_vm_on_fail=keep_vm_on_fail,
        )
    except Exception as e:
        err_msg = strip_colab_cli_banners(str(e))
        fail_state = dict(prev_st) if isinstance(prev_st, dict) else {}
        fail_state.update(
            {
                "session_prefix": effective_prefix,
                "failed_session": target_session,
                "failed_slot": slot,
                "phase": "failed",
                "last_error": err_msg,
                "engine": args.engine,
                "model_name": args.model_name,
                "gdrive_model_path": args.gdrive_model_path,
                "gpu": args.gpu,
                "high_mem": effective_high_mem,
                "gateway_url": f"http://127.0.0.1:{args.gateway_port}/v1",
            }
        )
        write_bg_state(fail_state)
        print(f"❌ 部署大模型会话 [{target_session}] 失败: {err_msg}", file=sys.stderr, flush=True)
        return 1

    key_entry = upsert_vm_api_key_entry(
        vm_name=target_session,
        slot=slot,
        model_name=args.model_name,
        engine=args.engine,
        custom_api_key=custom_api_key,
        gateway_port=args.gateway_port,
    )

    write_bg_state(
        {
            "session_prefix": effective_prefix,
            "active_slot": slot,
            "active_session": target_session,
            "standby_slot": None,
            "failed_session": None,
            "failed_slot": None,
            "last_error": None,
            "phase": "serving",
            "engine": args.engine,
            "model_name": args.model_name,
            "gdrive_model_path": args.gdrive_model_path,
            "gpu": args.gpu,
            "high_mem": effective_high_mem,
            "local_port": key_entry.get("local_port"),
            "gateway_url": f"http://127.0.0.1:{args.gateway_port}/v1",
            "last_switchover_at": time.time(),
            "warmup_tps": info.get("instant_tps"),
        }
    )
    ensure_local_gateway(args.gateway_port)
    print(f"🎉 会话 [{target_session}] (槽位 {slot.upper()}) 已通过 Colab 云端真机冒烟并上线承载流量！")
    print(
        f"🔗 本机专属端点: {key_entry['api_base_url']} | 统一网关: http://127.0.0.1:{args.gateway_port}/v1"
    )
    return 0


# ─────────────────────────── 3. 并行蓝绿热切换 (Make-Before-Break) ───────────────────────────


def resolve_switchover_settings(args: argparse.Namespace, st: Dict[str, Any]) -> Dict[str, Any]:
    prefix = args.session_prefix if args.session_prefix is not None else st.get("session_prefix", "llm")
    old_slot = (args.from_slot or st.get("active_slot") or "blue").lower()
    if old_slot not in ("blue", "green"):
        old_slot = "blue"
    new_slot = "green" if old_slot == "blue" else "blue"
    engine = args.engine if args.engine is not None else st.get("engine", "builtin")
    model_name = args.model_name if args.model_name is not None else st.get("model_name", "qwen2.5-7b-instruct")
    gdrive_model_path = (
        args.gdrive_model_path if args.gdrive_model_path is not None else st.get("gdrive_model_path", "")
    )
    gpu = args.gpu if args.gpu is not None else st.get("gpu", "T4")
    high_mem = args.high_mem if args.high_mem is not None else bool(st.get("high_mem", True))
    return {
        "session_prefix": prefix,
        "old_slot": old_slot,
        "new_slot": new_slot,
        "engine": engine,
        "model_name": model_name,
        "gdrive_model_path": gdrive_model_path,
        "gpu": gpu,
        "high_mem": high_mem,
    }


def cmd_switchover(args: argparse.Namespace) -> int:
    workspace_abs = os.path.abspath(args.workspace)
    colab_bin = find_bin("colab")
    st = read_bg_state()

    cfg = resolve_switchover_settings(args, st)
    prefix = cfg["session_prefix"]
    old_slot = cfg["old_slot"]
    new_slot = cfg["new_slot"]
    old_session = f"{prefix}-{old_slot}"
    new_session = f"{prefix}-{new_slot}"

    engine = cfg["engine"]
    model_name = cfg["model_name"]
    gdrive_model_path = cfg["gdrive_model_path"]
    gpu = cfg["gpu"]
    high_mem = cfg["high_mem"]

    print("=" * 72)
    print(f"🔄 [Make-Before-Break] 启动大语言模型并行蓝绿热切换: {old_slot.upper()} -> {new_slot.upper()}")
    print(f"   当前承载槽位: {old_session} (保持在线服务不中断)")
    print(f"   新建预热槽位: {new_session} (在 Colab 云端拉取权重并执行真机 Warmup 冒烟测速)")
    print("=" * 72)

    # 阶段 1：写入状态 "warming_standby"，旧实例继续服务，并行启动新实例为 warming 角色
    st.update(
        {
            "session_prefix": prefix,
            "active_slot": old_slot,
            "standby_slot": new_slot,
            "phase": f"warming_{new_slot}",
            "last_error": None,
            "engine": engine,
            "model_name": model_name,
            "gdrive_model_path": gdrive_model_path,
            "gpu": gpu,
            "high_mem": bool(high_mem and (gpu or "T4").strip().upper() not in SINGLE_SHAPE_ACCELERATORS),
        }
    )
    write_bg_state(st)

    try:
        info = provision_and_warmup_slot(
            session_prefix=prefix,
            slot=new_slot,
            role="warming",
            gpu=gpu,
            high_mem=high_mem,
            engine=engine,
            model_name=model_name,
            gdrive_model_path=gdrive_model_path,
            workspace_abs=workspace_abs,
        )
    except Exception as e:
        # 新实例云端冒烟未通过，provision_and_warmup_slot 已自动销毁新实例；恢复旧实例状态继续独立服务
        err_msg = strip_colab_cli_banners(str(e))
        st.update(
            {
                "standby_slot": None,
                "failed_session": new_session,
                "failed_slot": new_slot,
                "last_error": err_msg,
                "phase": "serving_fallback_old",
            }
        )
        write_bg_state(st)
        print(f"⚠️ 新槽位 ({new_session}) 云端冒烟未通过并已销毁，流量保持在原槽位 ({old_session}) 零中断继续服务！原因: {err_msg}")
        return 1

    if args.hold_overlap_seconds > 0:
        print(
            f"⏱️ 双槽位并行在线保持 {args.hold_overlap_seconds}s（可在 Colab Pilot 观察蓝/绿双实例同屏状态）...",
            flush=True,
        )
        time.sleep(args.hold_overlap_seconds)

    # 阶段 2：原子切流（新槽位晋升 active，旧槽位转为 draining）
    print(f"⚡ 正在执行原子切流：将 {new_session} 晋升为 ACTIVE，将 {old_session} 标记为 DRAINING...", flush=True)
    set_remote_slot_role(new_session, "active")
    set_remote_slot_role(old_session, "draining")

    upsert_vm_api_key_entry(
        vm_name=new_session,
        slot=new_slot,
        model_name=model_name,
        engine=engine,
    )

    st.update(
        {
            "active_slot": new_slot,
            "active_session": new_session,
            "standby_slot": None,
            "failed_session": None,
            "failed_slot": None,
            "last_error": None,
            "phase": "serving",
            "last_switchover_at": time.time(),
            "warmup_tps": info.get("instant_tps"),
        }
    )
    write_bg_state(st)

    # 阶段 3：销毁旧槽位停止双倍计费
    if not args.keep_old:
        print(f"🧹 正在销毁旧槽位实例 ({old_session}) 停止双倍计费...", flush=True)
        run_cmd([colab_bin, "stop", "-s", old_session], timeout=120)
        print(f"✅ 旧槽位 ({old_session}) 已彻底释放，当前由 [{new_slot.upper()}] ({new_session}) 独立提供服务！")
    else:
        print(f"ℹ️ 已按 --keep-old 保留旧槽位 ({old_session}) 处于 DRAINING 状态。")

    return 0


# ─────────────────────────── 4. 流式推理压测与停止 ───────────────────────────


def cmd_benchmark(args: argparse.Namespace) -> int:
    colab_bin = find_bin("colab")
    st = read_bg_state()
    prefix = args.session_prefix or st.get("session_prefix", "llm")
    slot = (args.slot or st.get("active_slot") or "blue").lower()
    session_name = (
        getattr(args, "session_name", None)
        or (st.get("active_session") if not args.slot and not args.session_prefix else None)
        or f"{prefix}-{slot}"
    )
    model_name = st.get("model_name", "qwen2.5-7b-instruct")

    print(f"⚡ 正在向 [{slot.upper()}] ({session_name}) 发起 {args.requests} 次流式推理请求，实时刷新 tok/s...")
    code = f"""
import http.client, json, time
for i in range({args.requests}):
    conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=30)
    body = json.dumps({{
        "model": {json.dumps(model_name)},
        "messages": [{{"role": "user", "content": f"Streaming benchmark request #{{i+1}}"}}],
        "stream": True,
        "max_tokens": 36,
    }})
    conn.request("POST", "/v1/chat/completions", body=body, headers={{"Content-Type": "application/json"}})
    _ = conn.getresponse().read()
    conn.close()
    time.sleep(0.4)
with open("/content/.colab_pilot_llm_state.json", "r", encoding="utf-8") as f:
    print("BENCH_RESULT", f.read())
"""
    res = run_cmd([colab_bin, "exec", "-s", session_name, "--timeout", "180"], timeout=200, stdin_text=code)
    out_txt = strip_colab_cli_banners((res.stdout or "") + "\n" + (res.stderr or ""))
    print(out_txt[-800:])
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    colab_bin = find_bin("colab")
    st = read_bg_state()
    prefix = args.session_prefix or st.get("session_prefix", "llm")
    stop_local_gateway()
    targets = [f"{prefix}-blue", f"{prefix}-green"]
    extra_active = st.get("active_session")
    if isinstance(extra_active, str) and extra_active and extra_active not in targets:
        targets.append(extra_active)
    for name in targets:
        print(f"🛑 正在停止并释放会话: {name} ...", flush=True)
        run_cmd([colab_bin, "stop", "-s", name], timeout=120)
    if os.path.exists(bg_state_path()):
        os.remove(bg_state_path())
    print("✅ 蓝绿部署所有云端实例与本机网关隧道均已释放。")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab Pilot 大语言模型多引擎部署与并行蓝绿热切换管理")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_pre = sub.add_parser("precheck", aliases=["smoke"], help="上云前校验语法、colab-cli 与 Google Drive 权重路径")
    p_pre.add_argument("--engine", default="vllm")
    p_pre.add_argument("--model-name", default="qwen2.5-7b-instruct")
    p_pre.add_argument("--gdrive-model-path", default="")

    p_deploy = sub.add_parser("deploy", help="部署初始蓝/绿/独立槽位并在 Colab 云端执行真机 Warmup 冒烟")
    p_deploy.add_argument("--session-prefix", default="llm")
    p_deploy.add_argument("--session-name", default=None, help="显式指定 Colab 会话名称（覆盖 session-prefix + slot）")
    p_deploy.add_argument("--slot", default="blue", choices=["blue", "green", "standalone"])
    p_deploy.add_argument("--workspace", default=".")
    p_deploy.add_argument("--gpu", default="T4")
    p_deploy.add_argument("--high-mem", action=argparse.BooleanOptionalAction, default=True)
    p_deploy.add_argument("--engine", default="builtin", choices=["vllm", "ollama", "llama.cpp", "transformers", "builtin"])
    p_deploy.add_argument("--model-name", default="qwen2.5-7b-instruct")
    p_deploy.add_argument("--gdrive-model-path", default="")
    p_deploy.add_argument("--api-key", default="", help="可选自定义本机网关 API Key")
    p_deploy.add_argument("--keep-vm-on-fail", action="store_true", help="在已有实例上部署失败时保留实例不执行 colab stop")
    p_deploy.add_argument("--gateway-port", type=int, default=18080)

    p_sw = sub.add_parser("switchover", help="执行 Make-Before-Break 并行蓝绿热切换（含云端真机 Warmup 冒烟与失败自动止损）")
    p_sw.add_argument("--session-prefix", default=None)
    p_sw.add_argument("--from-slot", default=None, choices=["blue", "green"])
    p_sw.add_argument("--workspace", default=".")
    p_sw.add_argument("--gpu", default=None)
    p_sw.add_argument("--high-mem", action=argparse.BooleanOptionalAction, default=None)
    p_sw.add_argument("--engine", default=None)
    p_sw.add_argument("--model-name", default=None)
    p_sw.add_argument("--gdrive-model-path", default=None)
    p_sw.add_argument("--hold-overlap-seconds", type=int, default=0, help="切流前保持双槽位同时在线的秒数（便于观察双机面板）")
    p_sw.add_argument("--keep-old", action="store_true", help="切流后暂不销毁旧实例（仅用于调试）")

    p_bench = sub.add_parser("benchmark", help="对活跃槽位发起流式请求压测，测试实时 tok/s 频率")
    p_bench.add_argument("--session-prefix", default=None)
    p_bench.add_argument("--session-name", default=None)
    p_bench.add_argument("--slot", default=None)
    p_bench.add_argument("--requests", type=int, default=5)

    p_stop = sub.add_parser("stop", help="停止并释放蓝绿全部实例")
    p_stop.add_argument("--session-prefix", default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.subcmd in ("precheck", "smoke"):
        return cmd_precheck(args)
    if args.subcmd == "deploy":
        return cmd_deploy(args)
    if args.subcmd == "switchover":
        return cmd_switchover(args)
    if args.subcmd == "benchmark":
        return cmd_benchmark(args)
    if args.subcmd == "stop":
        return cmd_stop(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
