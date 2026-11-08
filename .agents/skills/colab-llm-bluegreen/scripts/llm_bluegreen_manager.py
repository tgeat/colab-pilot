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
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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
    tmp = p + ".tmp"
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
    tmp = reg_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, reg_file)


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
        py_compile.compile(sidecar_script, doraise=True)
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
    print(f"  ✅ colab-cli ({colab_bin}) 与 rclone.conf 就绪。")

    if args.gdrive_model_path:
        print(f"\n[3/3] 正在校验 Google Drive 模型权重路径 ({args.gdrive_model_path})...")
        res = run_cmd(
            [
                rclone_bin,
                "--config",
                rclone_conf,
                "--retries=5",
                "--low-level-retries=20",
                "lsf",
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

        def _run():
            with GPU_LOCK, torch.inference_mode():
                model.generate(**gen_kwargs)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        cid = f"chatcmpl-{uuid.uuid4().hex[:10]}"
        for piece in streamer:
            if not piece:
                continue
            chunk = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
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
        "copy", gdrive_model_path, "/content/model"
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
    subprocess.Popen([
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", "/content/model" if gdrive_model_path else model_name,
        "--served-model-name", model_name,
        "--host", "127.0.0.1", "--port", "8001"
    ], stdout=open(engine_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    builtin_flag = []
elif engine == "llama.cpp":
    subprocess.Popen([
        "/content/llama-server",
        "-m", "/content/model/model.gguf",
        "--host", "127.0.0.1", "--port", "8001"
    ], stdout=open(engine_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
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
for i in range(450):
    if not backend_ready:
        if "eng_proc" in locals() and eng_proc.poll() is not None:
            break
        try:
            c8001 = http.client.HTTPConnection("127.0.0.1", 8001, timeout=3)
            c8001.request("GET", "/v1/models")
            r8001 = c8001.getresponse()
            c8001.close()
            if r8001.status < 500:
                backend_ready = True
        except Exception:
            if i % 5 == 0 and os.path.exists(engine_log):
                tail = open(engine_log, "r", errors="ignore").read().strip().splitlines()
                if tail:
                    print(f"[Engine] {{tail[-1]}}", flush=True)
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
) -> Dict[str, Any]:
    colab_bin = find_bin("colab")
    rclone_conf = find_rclone_conf()
    sidecar_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_metrics_sidecar.py")
    session_name = f"{session_prefix}-{slot}"

    register_workspace(
        session_name=session_name,
        workspace_dir=workspace_abs,
        description=f"LLM [{slot.upper()}/{role.upper()}] {model_name} ({engine})",
    )

    chk = run_cmd([colab_bin, "status", "-s", session_name], timeout=30)
    if chk.returncode == 0 and f"[{session_name}]" in (chk.stdout or "") and f"Hardware: {gpu.upper()}" in (chk.stdout or ""):
        print(f"♻️ [{slot.upper()}] 复用已就绪的 Colab {gpu.upper()} 实例: {chk.stdout.strip().splitlines()[0]}", flush=True)
    else:
        new_cmd = [colab_bin, "new", "-s", session_name, "--gpu", gpu]
        if high_mem and gpu.upper() not in ("G4", "L4", "V5E1", "V6E1"):
            new_cmd.append("--high-mem")
        print(f"🚀 [{slot.upper()}] 正在分配 Colab 实例: {' '.join(new_cmd)}", flush=True)
        res_new = run_cmd(new_cmd, timeout=300)
        if res_new.returncode != 0 and "already exists" not in (res_new.stdout + res_new.stderr).lower():
            raise RuntimeError(f"创建实例 {session_name} 失败: {res_new.stderr or res_new.stdout}")

    try:
        ws_meta_file = os.path.join(workspace_abs, f".colab_workspace_{slot}.json")
        with open(ws_meta_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": os.path.basename(workspace_abs),
                    "path": workspace_abs,
                    "description": f"LLM Blue-Green Slot [{slot.upper()}] · {model_name} ({engine})",
                },
                f,
                ensure_ascii=False,
            )
        try:
            run_cmd([colab_bin, "upload", "-s", session_name, ws_meta_file, "/content/.colab_workspace.json"], timeout=90)
        finally:
            if os.path.exists(ws_meta_file):
                os.remove(ws_meta_file)

        if os.path.exists(rclone_conf):
            run_cmd([colab_bin, "upload", "-s", session_name, rclone_conf, "/content/rclone.conf"], timeout=90)
        run_cmd([colab_bin, "upload", "-s", session_name, sidecar_abs, "/content/llm_metrics_sidecar.py"], timeout=90)

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
            [colab_bin, "exec", "-s", session_name, "--timeout", "900"],
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
        out_lines: List[str] = []
        for line in p_exec.stdout:
            out_lines.append(line)
            if any(k in line for k in ("[Engine]", "[HF-Engine]", "SLOT_READY", "CLOUD_WARMUP_FAILED", "Error", "Traceback")):
                print("  " + line, end="", flush=True)
        p_exec.wait(timeout=960)
        out = "".join(out_lines)
        for line in out.splitlines():
            if "SLOT_READY" in line:
                j_str = line.split("SLOT_READY", 1)[1].strip()
                info = json.loads(j_str)
                print(
                    f"✅ [{slot.upper()}] Colab 云端真机 Warmup 冒烟通过！会话={session_name}, 角色={info.get('role')}, "
                    f"实测速率={info.get('instant_tps')} tok/s, TTFT={info.get('ttft_ms')} ms",
                    flush=True,
                )
                return info
        raise RuntimeError(f"槽位 {session_name} 云端真机 Warmup 冒烟未通过:\n{out[-1000:]}")
    except Exception as e:
        print(
            f"❌ [云端真机冒烟失败 / 自动止损] 槽位 {session_name} 启动或预热异常，正在立即执行 `colab stop -s {session_name}` 销毁实例停止计费！",
            flush=True,
        )
        run_cmd([colab_bin, "stop", "-s", session_name], timeout=120)
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
    kwargs: Dict[str, Any] = {"stdin": subprocess.DEVNULL, "close_fds": True}
    if os.name == "nt":
        kwargs["creationflags"] = 0x08000000 | 0x00000200 | 0x00000008
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        [sys.executable, gw_script, "--port", str(gateway_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )
    time.sleep(2.5)


def stop_local_gateway() -> None:
    pid_file = os.path.join(os.path.expanduser("~/.config/colab-cli"), "llm_local_gateway.pid")
    if os.path.exists(pid_file):
        try:
            pid = int(open(pid_file, "r", encoding="utf-8").read().strip())
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, check=False)
            else:
                os.kill(pid, 15)
        except Exception:
            pass
        try:
            os.remove(pid_file)
        except Exception:
            pass


def cmd_deploy(args: argparse.Namespace) -> int:
    workspace_abs = os.path.abspath(args.workspace)
    slot = args.slot.lower()
    info = provision_and_warmup_slot(
        session_prefix=args.session_prefix,
        slot=slot,
        role="active",
        gpu=args.gpu,
        high_mem=args.high_mem,
        engine=args.engine,
        model_name=args.model_name,
        gdrive_model_path=args.gdrive_model_path,
        workspace_abs=workspace_abs,
    )
    write_bg_state(
        {
            "session_prefix": args.session_prefix,
            "active_slot": slot,
            "standby_slot": None,
            "phase": "serving",
            "engine": args.engine,
            "model_name": args.model_name,
            "gdrive_model_path": args.gdrive_model_path,
            "gpu": args.gpu,
            "high_mem": args.high_mem,
            "gateway_url": f"http://127.0.0.1:{args.gateway_port}/v1",
            "last_switchover_at": time.time(),
            "warmup_tps": info.get("instant_tps"),
        }
    )
    ensure_local_gateway(args.gateway_port)
    print(f"🎉 槽位 [{slot.upper()}] 已通过 Colab 云端真机冒烟并上线承载流量！")
    print(f"🔗 本机固定 OpenAI 网关已就绪: http://127.0.0.1:{args.gateway_port}/v1 (蓝绿热切换时端口保持不变)")
    return 0


# ─────────────────────────── 3. 并行蓝绿热切换 (Make-Before-Break) ───────────────────────────


def cmd_switchover(args: argparse.Namespace) -> int:
    workspace_abs = os.path.abspath(args.workspace)
    colab_bin = find_bin("colab")
    st = read_bg_state()

    prefix = args.session_prefix or st.get("session_prefix", "llm")
    old_slot = (args.from_slot or st.get("active_slot") or "blue").lower()
    new_slot = "green" if old_slot == "blue" else "blue"
    old_session = f"{prefix}-{old_slot}"
    new_session = f"{prefix}-{new_slot}"

    engine = args.engine or st.get("engine", "builtin")
    model_name = args.model_name or st.get("model_name", "qwen2.5-7b-instruct")
    gdrive_model_path = args.gdrive_model_path if args.gdrive_model_path is not None else st.get("gdrive_model_path", "")
    gpu = args.gpu or st.get("gpu", "T4")
    high_mem = args.high_mem

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
            "engine": engine,
            "model_name": model_name,
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
        st.update({"standby_slot": None, "phase": "serving_fallback_old"})
        write_bg_state(st)
        print(f"⚠️ 新槽位 ({new_session}) 云端冒烟未通过并已销毁，流量保持在原槽位 ({old_session}) 零中断继续服务！")
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

    st.update(
        {
            "active_slot": new_slot,
            "standby_slot": None,
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
    session_name = f"{prefix}-{slot}"
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
    print(res.stdout[-800:] if res.stdout else res.stderr[-800:])
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    colab_bin = find_bin("colab")
    st = read_bg_state()
    prefix = args.session_prefix or st.get("session_prefix", "llm")
    stop_local_gateway()
    for s in ("blue", "green"):
        name = f"{prefix}-{s}"
        print(f"🛑 正在停止并释放会话: {name} ...", flush=True)
        run_cmd([colab_bin, "stop", "-s", name], timeout=120)
    if os.path.exists(bg_state_path()):
        os.remove(bg_state_path())
    print("✅ 蓝绿部署所有云端实例与本机网关隧道均已释放。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab Pilot 大语言模型多引擎部署与并行蓝绿热切换管理")
    sub = parser.add_subparsers(dest="subcmd", required=True)

    p_pre = sub.add_parser("precheck", aliases=["smoke"], help="上云前校验语法、colab-cli 与 Google Drive 权重路径")
    p_pre.add_argument("--engine", default="vllm")
    p_pre.add_argument("--model-name", default="qwen2.5-7b-instruct")
    p_pre.add_argument("--gdrive-model-path", default="")

    p_deploy = sub.add_parser("deploy", help="部署初始蓝/绿槽位并在 Colab 云端执行真机 Warmup 冒烟")
    p_deploy.add_argument("--session-prefix", default="llm")
    p_deploy.add_argument("--slot", default="blue", choices=["blue", "green"])
    p_deploy.add_argument("--workspace", default=".")
    p_deploy.add_argument("--gpu", default="T4")
    p_deploy.add_argument("--high-mem", action="store_true", default=True)
    p_deploy.add_argument("--engine", default="builtin", choices=["vllm", "ollama", "llama.cpp", "transformers", "builtin"])
    p_deploy.add_argument("--model-name", default="qwen2.5-7b-instruct")
    p_deploy.add_argument("--gdrive-model-path", default="")
    p_deploy.add_argument("--gateway-port", type=int, default=18080)

    p_sw = sub.add_parser("switchover", help="执行 Make-Before-Break 并行蓝绿热切换（含云端真机 Warmup 冒烟与失败自动止损）")
    p_sw.add_argument("--session-prefix", default="llm")
    p_sw.add_argument("--from-slot", default=None, choices=["blue", "green"])
    p_sw.add_argument("--workspace", default=".")
    p_sw.add_argument("--gpu", default="T4")
    p_sw.add_argument("--high-mem", action="store_true", default=True)
    p_sw.add_argument("--engine", default=None)
    p_sw.add_argument("--model-name", default=None)
    p_sw.add_argument("--gdrive-model-path", default=None)
    p_sw.add_argument("--hold-overlap-seconds", type=int, default=0, help="切流前保持双槽位同时在线的秒数（便于观察双机面板）")
    p_sw.add_argument("--keep-old", action="store_true", help="切流后暂不销毁旧实例（仅用于调试）")

    p_bench = sub.add_parser("benchmark", help="对活跃槽位发起流式请求压测，测试实时 tok/s 频率")
    p_bench.add_argument("--session-prefix", default="llm")
    p_bench.add_argument("--slot", default=None)
    p_bench.add_argument("--requests", type=int, default=5)

    p_stop = sub.add_parser("stop", help="停止并释放蓝绿全部实例")
    p_stop.add_argument("--session-prefix", default="llm")

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
