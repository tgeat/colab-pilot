# -*- coding: utf-8 -*-
"""
Colab Pilot — Google Colab 虚拟机与 24h+ 全自动接力训练可视化管控后端
基于 google-colab-cli + rclone + Tornado 构建
"""

import asyncio
import base64
import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import tornado.ioloop
import tornado.web

from colab_cli.client import (
    Accelerator,
    ColabRequestError,
    PostAssignmentResponse,
    Shape,
    TooManyAssignmentsError,
    Variant,
    resolve_assign_shape,
    shape_display_label,
)
from colab_cli.commands.session import resolve_runtime_options
from colab_cli.common import _apply_proxy_info, state
from colab_cli.contents import ContentsClient
from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(WORKSPACE_DIR, "static")
TRAIN_SCRIPT_PATH = os.path.join(WORKSPACE_DIR, "train.py")
REQUIREMENTS_PATH = os.path.join(WORKSPACE_DIR, "requirements.txt")
SCHEME_DOC_PATH = os.path.join(WORKSPACE_DIR, "RELAY_TRAINING_SCHEME.md")

# 默认可执行文件路径
RCLONE_BIN = shutil.which("rclone") or r"C:\Users\86177\.local\bin\rclone.exe"
COLAB_BIN = shutil.which("colab") or r"C:\Users\86177\.local\bin\colab.exe"

EXECUTOR = ThreadPoolExecutor(max_workers=16, thread_name_prefix="ColabPilotWorker")
STATE_LOCK = threading.RLock()

# 远端虚拟机后台遥测采集守护脚本（写入 /content/.colab_pilot_telemetry.json）
REMOTE_TELEMETRY_DAEMON_CODE = r'''# -*- coding: utf-8 -*-
import json
import os
import shutil
import subprocess
import time

TELEMETRY_FILE = "/content/.colab_pilot_telemetry.json"
TRAINING_STATE_FILE = "/content/.colab_pilot_training_state.json"
PID_FILE = "/content/.colab_pilot_daemon.pid"


def read_cpu_stat():
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = [float(x) for x in line.split()[1:]]
        idle = parts[3] + (parts[4] if len(parts) > 4 else 0.0)
        total = sum(parts)
        return idle, total
    except Exception:
        return 0.0, 0.0


def get_cpu_model():
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "Cloud vCPU"


def get_mem_info():
    info = {}
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    k = parts[0].strip()
                    v = int(parts[1].strip().split()[0]) * 1024
                    info[k] = v
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = max(0, total - avail)
        pct = round((used / total) * 100.0, 1) if total > 0 else 0.0
        return {
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "avail_gb": round(avail / (1024 ** 3), 2),
            "percent": pct,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "avail_gb": 0, "percent": 0}


def get_disk_info():
    try:
        usage = shutil.disk_usage("/content")
        pct = round((usage.used / usage.total) * 100.0, 1) if usage.total > 0 else 0.0
        return {
            "total_gb": round(usage.total / (1024 ** 3), 2),
            "used_gb": round(usage.used / (1024 ** 3), 2),
            "free_gb": round(usage.free / (1024 ** 3), 2),
            "percent": pct,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "percent": 0}


def get_gpu_info():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode != 0 or not res.stdout.strip():
            return {"available": False, "name": "None (CPU Runtime)"}
        line = res.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        def _flt(val):
            try:
                return float(val)
            except Exception:
                return 0.0
        mem_used = _flt(parts[3]) if len(parts) > 3 else 0.0
        mem_total = _flt(parts[4]) if len(parts) > 4 else 0.0
        mem_pct = round((mem_used / mem_total) * 100.0, 1) if mem_total > 0 else 0.0
        return {
            "available": True,
            "name": parts[0] if len(parts) > 0 else "NVIDIA GPU",
            "gpu_util_percent": _flt(parts[1]) if len(parts) > 1 else 0.0,
            "mem_util_percent": _flt(parts[2]) if len(parts) > 2 else 0.0,
            "vram_used_mb": round(mem_used, 1),
            "vram_total_mb": round(mem_total, 1),
            "vram_percent": mem_pct,
            "temperature_c": _flt(parts[5]) if len(parts) > 5 else 0.0,
            "power_draw_w": _flt(parts[6]) if len(parts) > 6 else 0.0,
            "power_limit_w": _flt(parts[7]) if len(parts) > 7 else 0.0,
        }
    except Exception:
        return {"available": False, "name": "None (CPU Runtime)"}


def get_processes_and_threads():
    total_procs = 0
    total_threads = 0
    key_procs = []
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,nlwp,pcpu,pmem,rss,stat,etime,args", "--sort=-pcpu"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        lines = res.stdout.strip().splitlines()
        for line in lines[1:]:
            cols = line.strip().split(None, 7)
            if len(cols) < 8:
                continue
            pid_s, nlwp_s, pcpu_s, pmem_s, rss_s, stat_s, etime_s, cmd_s = cols
            try:
                pid = int(pid_s)
                nlwp = int(nlwp_s)
                pcpu = float(pcpu_s)
                pmem = float(pmem_s)
                rss_mb = round(int(rss_s) / 1024.0, 1)
            except Exception:
                continue
            total_procs += 1
            total_threads += nlwp

            is_interesting = (
                "train.py" in cmd_s
                or "rclone" in cmd_s
                or "ipykernel" in cmd_s
                or "colab_pilot_daemon" in cmd_s
                or pcpu >= 1.0
                or len(key_procs) < 8
            )
            if is_interesting and len(key_procs) < 12 and "ps -eo" not in cmd_s:
                role = "system"
                if "train.py" in cmd_s:
                    role = "train"
                elif "rclone" in cmd_s:
                    role = "rclone"
                elif "ipykernel" in cmd_s:
                    role = "kernel"
                elif "colab_pilot_daemon" in cmd_s:
                    role = "telemetry"
                key_procs.append({
                    "pid": pid,
                    "threads": nlwp,
                    "cpu_percent": pcpu,
                    "mem_percent": pmem,
                    "rss_mb": rss_mb,
                    "state": stat_s,
                    "elapsed": etime_s,
                    "role": role,
                    "command": cmd_s[:160],
                })
    except Exception:
        pass
    return {
        "total_processes": total_procs,
        "total_threads": total_threads,
        "processes": key_procs,
    }


def collect_once(prev_idle, prev_total, cpu_model):
    idle, total = read_cpu_stat()
    d_idle = idle - prev_idle
    d_total = total - prev_total
    cpu_pct = round(max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0)), 1) if d_total > 0 else 0.0

    try:
        load1, load5, load15 = os.getloadavg()
    except Exception:
        load1, load5, load15 = 0.0, 0.0, 0.0

    uptime_sec = 0.0
    try:
        with open("/proc/uptime", "r") as f:
            uptime_sec = float(f.read().split()[0])
    except Exception:
        pass

    training_state = None
    if os.path.exists(TRAINING_STATE_FILE):
        try:
            with open(TRAINING_STATE_FILE, "r", encoding="utf-8") as f:
                training_state = json.load(f)
        except Exception:
            pass

    proc_info = get_processes_and_threads()
    train_proc = next((p for p in proc_info["processes"] if p["role"] == "train"), None)
    rclone_proc = next((p for p in proc_info["processes"] if p["role"] == "rclone"), None)

    payload = {
        "timestamp": time.time(),
        "uptime_seconds": round(uptime_sec, 1),
        "cpu": {
            "model": cpu_model,
            "cores": os.cpu_count() or 2,
            "percent": cpu_pct,
            "load_avg": [round(load1, 2), round(load5, 2), round(load15, 2)],
        },
        "memory": get_mem_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "threads": {
            "total_processes": proc_info["total_processes"],
            "total_threads": proc_info["total_threads"],
            "train_process": train_proc,
            "rclone_process": rclone_proc,
            "processes": proc_info["processes"],
        },
        "files": {
            "has_rclone_conf": os.path.exists("/content/rclone.conf"),
            "has_checkpoint": os.path.exists("/content/checkpoint.pt"),
            "checkpoint_size_mb": round(os.path.getsize("/content/checkpoint.pt") / (1024 * 1024), 2)
            if os.path.exists("/content/checkpoint.pt")
            else 0.0,
            "has_done_flag": os.path.exists("/content/training_done.flag"),
        },
        "training": training_state,
    }

    tmp = TELEMETRY_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, TELEMETRY_FILE)
    return idle, total, payload


def main():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    cpu_model = get_cpu_model()
    idle, total = read_cpu_stat()
    time.sleep(0.25)
    while True:
        try:
            idle, total, _ = collect_once(idle, total, cpu_model)
        except Exception:
            pass
        time.sleep(2.5)


if __name__ == "__main__":
    main()
'''


def get_rclone_conf_path() -> str:
    """获取本机 rclone.conf 路径"""
    try:
        res = subprocess.run(
            [RCLONE_BIN, "config", "file"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        if lines:
            candidate = lines[-1]
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    default_path = os.path.expandvars(r"%APPDATA%\rclone\rclone.conf")
    return default_path


def check_gdrive_relay_status(gdrive_dir: str = "gdrive:colab_relay") -> Dict[str, Any]:
    """检查本机 rclone 授权状态及网盘接力目录文件状态"""
    conf_path = get_rclone_conf_path()
    has_conf = os.path.exists(conf_path) and os.path.getsize(conf_path) > 30
    result = {
        "rclone_installed": os.path.exists(RCLONE_BIN),
        "rclone_conf_path": conf_path,
        "rclone_authorized": has_conf,
        "gdrive_dir": gdrive_dir,
        "has_remote_checkpoint": False,
        "has_done_flag": False,
        "remote_files": [],
    }
    if not has_conf:
        return result
    try:
        res = subprocess.run(
            [RCLONE_BIN, "lsjson", gdrive_dir],
            capture_output=True,
            text=True,
            timeout=8,
        )
        if res.returncode == 0 and res.stdout.strip():
            items = json.loads(res.stdout)
            files = []
            for item in items:
                name = item.get("Name", "")
                files.append({
                    "name": name,
                    "size": item.get("Size", 0),
                    "mod_time": item.get("ModTime", ""),
                })
                if name == "checkpoint.pt":
                    result["has_remote_checkpoint"] = True
                elif name == "training_done.flag":
                    result["has_done_flag"] = True
            result["remote_files"] = files
    except Exception:
        pass
    return result


# 缓存每台虚拟机的最近遥测数据与运行事件
VM_TELEMETRY_CACHE: Dict[str, Dict[str, Any]] = {}
VM_META_CACHE: Dict[str, Dict[str, Any]] = {}


def ensure_remote_telemetry_daemon(session: SessionState) -> Optional[Dict[str, Any]]:
    """
    在云端虚拟机上部署并启动后台遥测守护进程，并返回立即采集的一份快照。
    """
    contents = ContentsClient(session)
    # 1. 上传守护脚本到 /content/.colab_pilot_daemon.py
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as tf:
        tf.write(REMOTE_TELEMETRY_DAEMON_CODE)
        tmp_path = tf.name
    try:
        contents.upload(tmp_path, "content/.colab_pilot_daemon.py")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    # 2. 通过 Kernel 启动守护进程（若尚未运行）并立即采集一次遥测
    bootstrap_code = """
import json, os, subprocess, sys, time
pid_file = "/content/.colab_pilot_daemon.pid"
running = False
if os.path.exists(pid_file):
    try:
        pid = int(open(pid_file).read().strip())
        os.kill(pid, 0)
        running = True
    except Exception:
        running = False
if not running:
    subprocess.Popen(
        [sys.executable, "/content/.colab_pilot_daemon.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.9)
if os.path.exists("/content/.colab_pilot_telemetry.json"):
    print("__TELEMETRY_JSON__" + open("/content/.colab_pilot_telemetry.json", encoding="utf-8").read())
"""
    runtime = ColabRuntime(
        session.url,
        session.token,
        kernel_id=session.kernel_id,
        session_id=session.session_id,
    )
    try:
        outputs = runtime.execute_code(bootstrap_code, timeout=12.0)
        for out in outputs:
            text = out.get("text", "")
            if "__TELEMETRY_JSON__" in text:
                raw = text.split("__TELEMETRY_JSON__", 1)[1].strip()
                data = json.loads(raw)
                VM_TELEMETRY_CACHE[session.name] = data
                return data
    finally:
        runtime.stop()
    return None


def fetch_vm_telemetry(session: SessionState, force_bootstrap: bool = False) -> Dict[str, Any]:
    """
    无阻塞获取云端虚拟机的资源状态、运行状态和线程状态：
    优先通过 Contents REST API 读取 /content/.colab_pilot_telemetry.json（即使 train.py 占满 Kernel 也完全不阻塞）。
    若文件不存在或过期且当前未处于阻塞执行，则自动引导启动后台遥测守护进程。
    """
    contents = ContentsClient(session)
    if not force_bootstrap:
        try:
            data = contents._request("GET", "content/.colab_pilot_telemetry.json", params={"content": "1"})
            content = data.get("content", "")
            if data.get("format") == "base64":
                content = base64.b64decode(content).decode("utf-8")
            elif isinstance(content, dict):
                payload = content
                VM_TELEMETRY_CACHE[session.name] = payload
                return payload
            payload = json.loads(content)
            VM_TELEMETRY_CACHE[session.name] = payload
            # 检查遥测时间戳是否新鲜（< 15 秒）
            if time.time() - payload.get("timestamp", 0) < 15:
                return payload
        except FileNotFoundError:
            pass
        except Exception:
            pass

    # 若当前该会话没有在执行阻塞式命令，则自动启动云端遥测守护进程
    relay_mgr = RELAY_SUPERVISORS.get(session.name)
    is_exec_busy = bool(session.running) or (relay_mgr and relay_mgr.phase == "training")
    if not is_exec_busy or force_bootstrap:
        try:
            snap = ensure_remote_telemetry_daemon(session)
            if snap:
                return snap
        except Exception as e:
            logging.debug(f"Telemetry bootstrap skipped for {session.name}: {e}")

    return VM_TELEMETRY_CACHE.get(session.name, {})


def hard_terminate_vm(name: str, endpoint: Optional[str] = None, stop_relay: bool = True) -> Dict[str, Any]:
    """
    【核心防假性删除机制】彻底销毁 Colab 虚拟机并停止计费：
    1. 若开启了接力训练且 stop_relay=True，立即停止接力监督线程与本地子进程；
    2. 尝试优雅关闭远程 Jupyter Kernel；
    3. 调用 Google Colab 服务端 `/tun/m/unassign/{endpoint}` 释放虚拟机；
    4. 循环调用 `/tun/m/assignments` 进行二次服务端核验，确保云端计费实例列表彻底清零该节点！
    """
    steps_log = []

    if stop_relay and name in RELAY_SUPERVISORS:
        sup = RELAY_SUPERVISORS[name]
        if sup.is_running:
            sup.stop(destroy_vm=False)
            steps_log.append("已停止本机接力训练守护线程与子进程")

    with STATE_LOCK:
        s = state.store.get(name)
        target_endpoint = endpoint or (s.endpoint if s else None)

        # 若本地没有记录，从云端 assignments 查找匹配
        if not target_endpoint:
            try:
                for a in state.client.list_assignments():
                    if a.endpoint == name or name.endswith(a.endpoint[-6:]):
                        target_endpoint = a.endpoint
                        break
            except Exception:
                pass

    # 1. 关闭远程 Kernel
    if s and s.url and s.token:
        try:
            rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id)
            rt.stop(shutdown_kernel=True)
            steps_log.append("已发送远程 Kernel 关停指令")
        except Exception as e:
            steps_log.append(f"Kernel 关停跳过 ({type(e).__name__})")

    # 2. 调用 Google 服务端 unassign 并循环核验 list_assignments
    verified_gone = False
    last_error = None
    remaining_endpoints = []

    for attempt in range(1, 4):
        try:
            if target_endpoint:
                try:
                    state.client.unassign(target_endpoint)
                    steps_log.append(f"第 {attempt} 次调用服务端 unassign({target_endpoint}) 成功")
                except Exception as ue:
                    last_error = str(ue)
                    steps_log.append(f"第 {attempt} 次 unassign 返回: {ue}")

            # 二次核验云端实际计费列表
            assignments = state.client.list_assignments()
            remaining_endpoints = [a.endpoint for a in assignments]
            if not target_endpoint or target_endpoint not in remaining_endpoints:
                verified_gone = True
                steps_log.append("服务端二次核验通过：该实例已从 Google Colab 计费列表中彻底移除")
                break
        except Exception as e:
            last_error = str(e)
            steps_log.append(f"核验异常: {e}")
        time.sleep(1.2)

    # 3. 仅在服务端核验真正释放后，清理本地状态缓存
    if verified_gone:
        with STATE_LOCK:
            state.store.remove(name)
            VM_TELEMETRY_CACHE.pop(name, None)
            VM_META_CACHE.pop(name, None)
        state.history.log_event(
            name,
            "session_terminated",
            {"reason": "hard_terminate_verified", "endpoint": target_endpoint},
        )

    # 4. 获取最新账户计算单元余额与消耗速率作为凭证
    ccu_receipt = None
    try:
        ccu = state.client.get_consumption_user_info()
        ccu_receipt = {
            "balance": round(ccu.paid_compute_units_balance, 2),
            "burn_rate_hourly": round(ccu.consumption_rate_hourly, 2),
            "active_assignments_count": ccu.assignments_count,
        }
    except Exception:
        pass

    return {
        "name": name,
        "endpoint": target_endpoint,
        "verified_unassigned": verified_gone,
        "remaining_server_endpoints": remaining_endpoints,
        "steps": steps_log,
        "ccu_receipt": ccu_receipt,
        "error": None if verified_gone else (last_error or "服务端仍存在该实例，请重试销毁"),
    }


def create_new_vm(
    name: Optional[str] = None,
    accelerator_type: str = "T4",
    high_mem: bool = False,
    init_telemetry: bool = True,
) -> SessionState:
    """
    创建一台新的 Colab 虚拟机，同步写入本地 StateStore，并自动部署后台遥测脚本。
    """
    session_name = (name or "").strip() or f"vm-{uuid.uuid4().hex[:6]}"
    acc_upper = (accelerator_type or "CPU").upper()

    gpu_arg = None
    tpu_arg = None
    if acc_upper in ("T4", "L4", "G4", "A100", "H100"):
        gpu_arg = acc_upper
    elif acc_upper in ("V5E1", "V6E1"):
        tpu_arg = acc_upper.lower()

    variant, accelerator, shape = resolve_runtime_options(
        gpu=gpu_arg, tpu=tpu_arg, high_mem=high_mem
    )

    res = state.client.assign(
        uuid.uuid4(), variant=variant, accelerator=accelerator, shape=shape
    )

    if isinstance(res, PostAssignmentResponse):
        token = res.runtime_proxy_info.token
        url = res.runtime_proxy_info.url
        endpoint = res.endpoint
        expires_at = res.runtime_proxy_info.expires_at()
    else:
        token = (
            res.runtime_proxy_info.token
            if hasattr(res, "runtime_proxy_info")
            else getattr(res, "runtime_proxy_token", "")
        )
        url = res.runtime_proxy_info.url if hasattr(res, "runtime_proxy_info") else ""
        endpoint = res.endpoint
        expires_at = (
            res.runtime_proxy_info.expires_at()
            if hasattr(res, "runtime_proxy_info")
            else None
        )

    s = SessionState(
        name=session_name,
        token=token,
        url=url,
        endpoint=endpoint,
        token_expires_at=expires_at,
        variant=variant.value,
        accelerator=accelerator.value,
        machine_shape=(
            Shape.HIGH_RAM.name if shape == Shape.HIGH_RAM else Shape.STANDARD.name
        ),
    )

    with STATE_LOCK:
        state.store.add(s)
        VM_META_CACHE.setdefault(session_name, {})["created_at"] = time.time()

    state.history.log_event(
        session_name,
        "session_created",
        {
            "endpoint": endpoint,
            "variant": variant.value,
            "accelerator": accelerator.value,
            "machine_shape": s.machine_shape,
        },
    )

    if init_telemetry:
        try:
            ensure_remote_telemetry_daemon(s)
        except Exception as e:
            logging.debug(f"Initial telemetry setup warning for {session_name}: {e}")

    return s


class RelaySupervisor:
    """
    单台虚拟机的 24h+ 全自动跨实例接力训练控制器（严格对应方案中的 relay_rclone.sh + train.py）
    """

    def __init__(self, vm_name: str):
        self.vm_name = vm_name
        self.is_running = False
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.current_proc: Optional[subprocess.Popen] = None
        self.round = 0
        self.phase = "idle"  # idle / preparing_gdrive / provisioning / uploading_auth / installing_env / training / cooldown / completed / stopped / error
        self.phase_label = "未开启接力"
        self.started_at: Optional[float] = None
        self.round_started_at: Optional[float] = None
        self.config: Dict[str, Any] = {
            "gpu_type": "T4",
            "high_mem": True,
            "gdrive_dir": "gdrive:colab_relay",
            "max_session_hours": 22.5,
            "backup_interval_minutes": 30.0,
            "total_epochs": 500,
            "reset_done_flag": False,
        }
        self.logs: List[str] = []
        self._lock = threading.Lock()

    def append_log(self, msg: str):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            self.logs.append(line)
            if len(self.logs) > 400:
                self.logs = self.logs[-400:]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            logs_copy = list(self.logs[-80:])
        thread_alive = bool(self.thread and self.thread.is_alive())
        return {
            "vm_name": self.vm_name,
            "is_running": self.is_running and thread_alive,
            "thread_name": self.thread.name if self.thread else None,
            "thread_ident": self.thread.ident if self.thread else None,
            "thread_native_id": getattr(self.thread, "native_id", None) if self.thread else None,
            "thread_alive": thread_alive,
            "subproc_pid": self.current_proc.pid if (self.current_proc and self.current_proc.poll() is None) else None,
            "round": self.round,
            "phase": self.phase,
            "phase_label": self.phase_label,
            "started_at": self.started_at,
            "round_started_at": self.round_started_at,
            "config": self.config,
            "logs": logs_copy,
        }

    def start(self, config_overrides: Optional[Dict[str, Any]] = None):
        if self.is_running and self.thread and self.thread.is_alive():
            return
        if config_overrides:
            self.config.update(config_overrides)
        self.stop_event.clear()
        self.is_running = True
        self.started_at = time.time()
        self.round = 0
        self.phase = "starting"
        self.phase_label = "正在初始化接力监工线程..."
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"RelaySupervisor-{self.vm_name}",
            daemon=True,
        )
        self.thread.start()

    def stop(self, destroy_vm: bool = False):
        self.append_log("⏹ 收到停止接力训练指令，正在安全中止当前接力流程...")
        self.stop_event.set()
        self.is_running = False
        self.phase = "stopped"
        self.phase_label = "已手动停止接力"

        proc = self.current_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        with STATE_LOCK:
            s = state.store.get(self.vm_name)
            if s and s.running:
                s.running = None
                state.store.add(s)

        if destroy_vm:
            self.append_log("🧹 正在销毁云端实例并释放计费...")
            res = hard_terminate_vm(self.vm_name, stop_relay=False)
            if res.get("verified_unassigned"):
                self.append_log("✅ 云端实例已彻底释放，停止计费核验通过！")

    def _check_done_flag(self, gdrive_dir: str) -> bool:
        try:
            res = subprocess.run(
                [RCLONE_BIN, "lsf", f"{gdrive_dir}/training_done.flag"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return res.stdout.strip() == "training_done.flag"
        except Exception:
            return False

    def _run_loop(self):
        gdrive_dir = self.config.get("gdrive_dir", "gdrive:colab_relay")
        gpu_type = self.config.get("gpu_type", "T4")
        high_mem = bool(self.config.get("high_mem", True))
        max_session_seconds = float(self.config.get("max_session_hours", 22.5)) * 3600.0
        backup_interval_seconds = float(self.config.get("backup_interval_minutes", 30.0)) * 60.0
        total_epochs = int(self.config.get("total_epochs", 500))
        rclone_conf = get_rclone_conf_path()

        try:
            self.phase = "preparing_gdrive"
            self.phase_label = "准备 Google Drive 接力目录"
            self.append_log(f"🔑 使用本机 Rclone 授权文件: {rclone_conf}")
            if not os.path.exists(rclone_conf):
                raise RuntimeError(f"未找到本机 rclone.conf ({rclone_conf})，请先完成 rclone 授权。")

            subprocess.run([RCLONE_BIN, "mkdir", gdrive_dir], check=False, timeout=15)
            if self.config.get("reset_done_flag"):
                self.append_log(f"🧹 清理网盘旧完成标记 {gdrive_dir}/training_done.flag...")
                subprocess.run(
                    [RCLONE_BIN, "deletefile", f"{gdrive_dir}/training_done.flag"],
                    capture_output=True,
                    check=False,
                    timeout=15,
                )

            round_num = 1
            while not self.stop_event.is_set():
                if self._check_done_flag(gdrive_dir):
                    self.phase = "completed"
                    self.phase_label = "全部训练轮次已完成 (100%)"
                    self.append_log("🎉 检测到网盘中的完成标记 training_done.flag，全部接力训练已圆满结束！")
                    break

                self.round = round_num
                self.round_started_at = time.time()
                self.append_log("====================================================")
                self.append_log(f"🚀 启动第 {round_num} 棒接力实例 | 目标硬件: {gpu_type} ({'High-RAM' if high_mem else 'Standard'})")
                self.append_log("====================================================")

                # 1. 第 1 棒如果当前同名虚拟机已在线且可用，直接复用该实例；否则先彻底释放旧实例再开新机
                s = state.store.get(self.vm_name)
                reuse_existing = False
                if round_num == 1 and s:
                    try:
                        server_eps = {a.endpoint for a in state.client.list_assignments()}
                        if s.endpoint in server_eps:
                            reuse_existing = True
                            self.append_log(f"♻️ 检测到现有运行中实例 [{self.vm_name}] ({s.endpoint})，第 1 棒直接复用该实例。")
                    except Exception:
                        pass

                if not reuse_existing:
                    self.phase = "provisioning"
                    self.phase_label = f"第 {round_num} 棒：正在分配云端 GPU 实例 ({gpu_type})"
                    hard_terminate_vm(self.vm_name, stop_relay=False)
                    if self.stop_event.is_set():
                        break
                    self.append_log(f"☁️ 正在向 Google Colab 申请新实例 [{self.vm_name}] ({gpu_type})...")
                    s = create_new_vm(
                        name=self.vm_name,
                        accelerator_type=gpu_type,
                        high_mem=high_mem,
                        init_telemetry=True,
                    )
                    self.append_log(f"✅ 实例 [{self.vm_name}] 分配就绪！Endpoint: {s.endpoint}")

                if self.stop_event.is_set():
                    break

                # 2. 上传本机 rclone.conf 与 train.py 到云端 /content
                self.phase = "uploading_auth"
                self.phase_label = f"第 {round_num} 棒：上传 rclone.conf 密钥与训练脚本"
                self.append_log("📤 正在上传 rclone.conf 免密授权文件与 train.py 到云端 /content...")
                contents = ContentsClient(s)
                contents.upload(rclone_conf, "content/rclone.conf")
                contents.upload(TRAIN_SCRIPT_PATH, "content/train.py")

                # 确保后台遥测守护进程已启动
                try:
                    ensure_remote_telemetry_daemon(s)
                except Exception:
                    pass

                if self.stop_event.is_set():
                    break

                # 3. 在云端安装 rclone 工具及 Python 依赖包
                self.phase = "installing_env"
                self.phase_label = f"第 {round_num} 棒：云端安装 rclone 与依赖环境"
                self.append_log("🔧 正在云端虚拟机安装 rclone 环境...")
                install_rclone_code = (
                    "import shutil, subprocess\n"
                    "if not shutil.which('rclone'):\n"
                    "    subprocess.run(['bash', '-c', 'apt-get update -qq && apt-get install -y -qq rclone'], check=True)\n"
                    "print('RCLONE_READY:', shutil.which('rclone'))\n"
                )
                rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id, session_id=s.session_id)
                try:
                    rt.execute_code(install_rclone_code, timeout=180.0)
                finally:
                    rt.stop()
                self.append_log("✅ 云端 rclone 环境就绪。")

                # 检查 requirements.txt 是否有非注释行
                if os.path.exists(REQUIREMENTS_PATH):
                    with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as rf:
                        pkgs = [
                            line.strip()
                            for line in rf
                            if line.strip() and not line.strip().startswith("#")
                        ]
                    if pkgs:
                        self.append_log(f"📦 正在云端安装 requirements.txt 依赖: {', '.join(pkgs)}...")
                        subprocess.run(
                            [COLAB_BIN, "install", "-s", self.vm_name, "-r", REQUIREMENTS_PATH],
                            capture_output=True,
                            text=True,
                            timeout=600,
                        )

                if self.stop_event.is_set():
                    break

                # 4. 执行训练脚本（--timeout 86400 支持 24h 长耗时运行，实时流式读取输出）
                self.phase = "training"
                self.phase_label = f"第 {round_num} 棒：云端训练与 16 线程定时热备中"
                self.append_log(
                    f"⏳ 开始云端训练（单棒限时 {max_session_seconds/3600:.2f}h，每 {backup_interval_seconds/60:.1f}m 自动同步 {gdrive_dir}）..."
                )

                cmd = [
                    COLAB_BIN,
                    "exec",
                    "-s",
                    self.vm_name,
                    "--timeout",
                    "86400",
                    "--env",
                    f"GDRIVE_DIR={gdrive_dir}",
                    "--env",
                    f"MAX_SESSION_SECONDS={max_session_seconds}",
                    "--env",
                    f"BACKUP_INTERVAL_SECONDS={backup_interval_seconds}",
                    "--env",
                    f"TOTAL_EPOCHS={total_epochs}",
                    "-f",
                    TRAIN_SCRIPT_PATH,
                ]
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"

                self.current_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )

                assert self.current_proc.stdout is not None
                for raw_line in self.current_proc.stdout:
                    line = raw_line.rstrip("\r\n")
                    if line:
                        self.append_log(line)
                    if self.stop_event.is_set():
                        break

                self.current_proc.wait()

                if self.stop_event.is_set():
                    break

                # 5. 本棒结束，彻底销毁当前云端实例释放计费并重置 24 小时周期
                self.phase = "destroying_round_vm"
                self.phase_label = f"第 {round_num} 棒结束：正在彻底销毁实例释放计费"
                self.append_log("🧹 本棒结束，正在调用服务端 unassign 销毁云端实例并核验释放计费...")
                term_res = hard_terminate_vm(self.vm_name, stop_relay=False)
                if term_res.get("verified_unassigned"):
                    self.append_log("✅ 本棒云端实例已彻底销毁，服务端计费已清零。")
                else:
                    self.append_log(f"⚠️ 实例销毁返回警告: {term_res.get('error')}")

                # 检查是否已经完成全部轮次
                if self._check_done_flag(gdrive_dir):
                    self.phase = "completed"
                    self.phase_label = "全部训练轮次已完成 (100%)"
                    self.append_log("🎉 检测到网盘中的完成标记 training_done.flag，全部接力训练已圆满结束！")
                    break

                round_num += 1
                self.phase = "cooldown"
                self.phase_label = f"休息 15 秒后启动第 {round_num} 棒..."
                self.append_log(f"💤 休息 15 秒后检查状态并准备启动第 {round_num} 棒...")
                for _ in range(15):
                    if self.stop_event.is_set():
                        break
                    time.sleep(1)

            # 若全部完成，自动把最终 checkpoint.pt 拉回本机工作目录
            if self.phase == "completed" and not self.stop_event.is_set():
                local_ckpt_out = os.path.join(WORKSPACE_DIR, f"checkpoint_{self.vm_name}.pt")
                self.append_log(f"📥 正在将最终训练好的 checkpoint.pt 下载回本机 ({local_ckpt_out})...")
                subprocess.run(
                    [
                        RCLONE_BIN,
                        "copyto",
                        f"{gdrive_dir}/checkpoint.pt",
                        local_ckpt_out,
                        "--multi-thread-streams=16",
                    ],
                    check=False,
                    timeout=300,
                )
                self.append_log("✅ 全部接力训练与权重回收流程执行完毕！")
        except Exception as e:
            self.phase = "error"
            self.phase_label = f"接力异常: {e}"
            self.append_log(f"❌ 接力训练发生异常: {e}")
            logging.exception("RelaySupervisor error")
        finally:
            self.is_running = False


RELAY_SUPERVISORS: Dict[str, RelaySupervisor] = {}


def get_or_create_supervisor(vm_name: str) -> RelaySupervisor:
    with STATE_LOCK:
        if vm_name not in RELAY_SUPERVISORS:
            RELAY_SUPERVISORS[vm_name] = RelaySupervisor(vm_name)
        return RELAY_SUPERVISORS[vm_name]


def sync_and_build_overview(refresh_telemetry: bool = False) -> Dict[str, Any]:
    """
    同步云端真实计费实例列表与本地会话存储：
    1. 自动发现并纳管云端孤儿实例（防止任何隐藏实例在后台偷偷扣费）；
    2. 清理已在云端释放且未处于接力交接期的过期本地记录；
    3. 汇总每台虚拟机的资源状态、运行状态、线程状态与接力训练状态。
    """
    try:
        ccu = state.client.get_consumption_user_info()
        ccu_info = {
            "balance": round(ccu.paid_compute_units_balance, 2),
            "burn_rate_hourly": round(ccu.consumption_rate_hourly, 2),
            "assignments_count": ccu.assignments_count,
        }
    except Exception as e:
        ccu_info = {
            "balance": None,
            "burn_rate_hourly": None,
            "assignments_count": None,
            "error": str(e),
        }

    try:
        server_assignments = state.client.list_assignments()
    except Exception:
        server_assignments = []

    by_endpoint = {a.endpoint: a for a in server_assignments}

    with STATE_LOCK:
        local_sessions = state.store.list()
        known_endpoints = {s.endpoint: name for name, s in local_sessions.items()}

        # 1. 检查本地会话是否仍在云端计费列表中
        for name, s in list(local_sessions.items()):
            sup = RELAY_SUPERVISORS.get(name)
            in_relay_transition = bool(sup and sup.is_running)
            if s.endpoint in by_endpoint:
                _apply_proxy_info(s, by_endpoint[s.endpoint].runtime_proxy_info)
                state.store.add(s)
            else:
                if not in_relay_transition:
                    state.store.remove(name)
                    local_sessions.pop(name, None)

        # 2. 自动纳管云端存在但本地未记录的孤儿计费实例（杜绝隐形扣费）
        for ep, a in by_endpoint.items():
            if ep not in known_endpoints:
                adopted_name = f"cloud-{ep[-6:].lower()}"
                adopted = SessionState(
                    name=adopted_name,
                    token=a.runtime_proxy_info.token,
                    url=a.runtime_proxy_info.url,
                    endpoint=a.endpoint,
                    token_expires_at=a.runtime_proxy_info.expires_at(),
                    variant=a.variant.name,
                    accelerator=a.accelerator.value,
                    machine_shape=a.machine_shape.name,
                )
                state.store.add(adopted)
                local_sessions[adopted_name] = adopted
                VM_META_CACHE.setdefault(adopted_name, {})["adopted_orphan"] = True

        # 3. 把正在处于接力交接期（如第 N 棒刚销毁、等待 15 秒开第 N+1 棒）的虚拟机也纳入卡片列表
        all_vm_names = list(local_sessions.keys())
        for sup_name, sup in RELAY_SUPERVISORS.items():
            if sup.is_running and sup_name not in all_vm_names:
                all_vm_names.append(sup_name)

    # 并行拉取在线虚拟机的遥测数据
    if refresh_telemetry and local_sessions:
        futures = {
            name: EXECUTOR.submit(fetch_vm_telemetry, s, False)
            for name, s in local_sessions.items()
            if s.endpoint in by_endpoint
        }
        for name, fut in futures.items():
            try:
                fut.result(timeout=6.0)
            except Exception:
                pass

    # 收集本机进程内所有活跃线程状态
    local_threads = []
    for t in threading.enumerate():
        local_threads.append({
            "name": t.name,
            "ident": t.ident,
            "native_id": getattr(t, "native_id", None),
            "daemon": t.daemon,
            "alive": t.is_alive(),
        })

    vms_payload = []
    for name in all_vm_names:
        s = local_sessions.get(name)
        sup = get_or_create_supervisor(name)
        relay_status = sup.get_status()
        is_on_server = bool(s and s.endpoint in by_endpoint)

        telemetry = VM_TELEMETRY_CACHE.get(name, {})
        if s and is_on_server and not telemetry:
            # 异步触发首次遥测拉取
            EXECUTOR.submit(fetch_vm_telemetry, s, False)

        hardware_label = "CPU" if (not s or s.accelerator == "NONE") else s.accelerator
        shape_label = shape_display_label(s.machine_shape if s else "STANDARD")

        if relay_status["is_running"]:
            run_state = f"RELAY ({relay_status['phase_label']})"
            run_badge = "relay"
        elif s and s.running:
            run_state = f"BUSY ({s.running})"
            run_badge = "busy"
        elif is_on_server:
            run_state = "IDLE (在线待命 / 计费中)"
            run_badge = "idle"
        else:
            run_state = "TRANSITION (接力换棒冷却中)"
            run_badge = "transition"

        vms_payload.append({
            "name": name,
            "endpoint": s.endpoint if s else None,
            "variant": s.variant if s else "GPU",
            "accelerator": hardware_label,
            "machine_shape": shape_label,
            "is_billing_on_server": is_on_server,
            "adopted_orphan": VM_META_CACHE.get(name, {}).get("adopted_orphan", False),
            "run_state": run_state,
            "run_badge": run_badge,
            "running_task": s.running if s else None,
            "last_execution": s.last_execution if s else None,
            "token_expires_at": s.token_expires_at.isoformat() if (s and s.token_expires_at) else None,
            "colab_web_url": (
                f"https://colab.research.google.com/ drive/{s.endpoint}".replace(" ", "")
                if s and s.endpoint
                else None
            ),
            "telemetry": telemetry,
            "relay": relay_status,
        })

    return {
        "timestamp": time.time(),
        "ccu": ccu_info,
        "server_active_count": len(server_assignments),
        "server_endpoints": [a.endpoint for a in server_assignments],
        "gdrive": check_gdrive_relay_status(),
        "local_threads": local_threads,
        "vms": vms_payload,
    }


# ==================== Tornado HTTP API Handlers ====================


class BaseAPIHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")

    def write_json(self, data: Any, status: int = 200):
        self.set_status(status)
        self.write(json.dumps(data, ensure_ascii=False, default=str))

    def get_json_body(self) -> Dict[str, Any]:
        if not self.request.body:
            return {}
        try:
            return json.loads(self.request.body.decode("utf-8"))
        except Exception:
            return {}


class OverviewHandler(BaseAPIHandler):
    async def get(self):
        refresh = self.get_argument("refresh_telemetry", "0") == "1"
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(EXECUTOR, sync_and_build_overview, refresh)
        self.write_json(data)


class VMsHandler(BaseAPIHandler):
    async def post(self):
        """右上角 + 新增虚拟机接口"""
        body = self.get_json_body()
        name = (body.get("name") or "").strip() or None
        accelerator = body.get("accelerator", "T4")
        high_mem = bool(body.get("high_mem", False))
        auto_start_relay = bool(body.get("auto_start_relay", False))
        relay_config = body.get("relay_config") or {}

        loop = asyncio.get_running_loop()
        try:
            s = await loop.run_in_executor(
                EXECUTOR, create_new_vm, name, accelerator, high_mem, True
            )
            if auto_start_relay:
                sup = get_or_create_supervisor(s.name)
                relay_cfg = {
                    "gpu_type": accelerator if accelerator != "CPU" else "T4",
                    "high_mem": high_mem,
                    **relay_config,
                }
                sup.start(relay_cfg)
            self.write_json({
                "ok": True,
                "vm": {
                    "name": s.name,
                    "endpoint": s.endpoint,
                    "accelerator": s.accelerator,
                    "machine_shape": s.machine_shape,
                },
            })
        except TooManyAssignmentsError as e:
            self.write_json(
                {
                    "ok": False,
                    "error": "分配被 Google Colab 拒绝（当前活跃实例数已达上限或配额受限）。请先销毁闲置实例或切换 GPU 型号。",
                    "detail": str(e),
                },
                status=400,
            )
        except ColabRequestError as e:
            self.write_json(
                {
                    "ok": False,
                    "error": f"Colab 服务端拒绝了所选硬件 ({accelerator})，可能当前账号无该加速器配额，请尝试 T4 或 CPU。",
                    "detail": str(e),
                },
                status=400,
            )
        except Exception as e:
            self.write_json({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status=500)


class VMDetailHandler(BaseAPIHandler):
    async def delete(self, name: str):
        """彻底删除虚拟机并核验停止计费"""
        endpoint = self.get_argument("endpoint", None)
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(EXECUTOR, hard_terminate_vm, name, endpoint, True)
        status_code = 200 if res.get("verified_unassigned") else 500
        self.write_json({"ok": res.get("verified_unassigned", False), **res}, status=status_code)


class VMProbeHandler(BaseAPIHandler):
    async def post(self, name: str):
        """立即刷新指定虚拟机的资源状态、运行状态与线程状态"""
        body = self.get_json_body()
        force_bootstrap = bool(body.get("force_bootstrap", False))
        with STATE_LOCK:
            s = state.store.get(name)
        if not s:
            self.write_json({"ok": False, "error": f"未找到虚拟机 '{name}'"}, status=404)
            return
        loop = asyncio.get_running_loop()
        telemetry = await loop.run_in_executor(EXECUTOR, fetch_vm_telemetry, s, force_bootstrap)
        self.write_json({"ok": True, "name": name, "telemetry": telemetry})


class VMRelayStartHandler(BaseAPIHandler):
    async def post(self, name: str):
        """开启指定虚拟机的 24h+ 全自动接力训练"""
        body = self.get_json_body()
        with STATE_LOCK:
            s = state.store.get(name)
        default_gpu = "T4"
        default_hm = True
        if s:
            if s.accelerator and s.accelerator != "NONE":
                default_gpu = s.accelerator
            default_hm = s.machine_shape == "HIGH_RAM"

        cfg = {
            "gpu_type": body.get("gpu_type", default_gpu),
            "high_mem": bool(body.get("high_mem", default_hm)),
            "gdrive_dir": body.get("gdrive_dir", "gdrive:colab_relay"),
            "max_session_hours": float(body.get("max_session_hours", 22.5)),
            "backup_interval_minutes": float(body.get("backup_interval_minutes", 30.0)),
            "total_epochs": int(body.get("total_epochs", 500)),
            "reset_done_flag": bool(body.get("reset_done_flag", False)),
        }
        sup = get_or_create_supervisor(name)
        sup.start(cfg)
        self.write_json({"ok": True, "relay": sup.get_status()})


class VMRelayStopHandler(BaseAPIHandler):
    async def post(self, name: str):
        """关闭指定虚拟机的接力训练（可选是否同时销毁虚拟机停止计费）"""
        body = self.get_json_body()
        destroy_vm = bool(body.get("destroy_vm", False))
        sup = get_or_create_supervisor(name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(EXECUTOR, sup.stop, destroy_vm)
        self.write_json({"ok": True, "relay": sup.get_status()})


class FilesConfigHandler(BaseAPIHandler):
    async def get(self):
        """读取本机 train.py、requirements.txt 及方案文档内容"""
        train_code = ""
        req_text = ""
        scheme_text = ""
        if os.path.exists(TRAIN_SCRIPT_PATH):
            with open(TRAIN_SCRIPT_PATH, "r", encoding="utf-8") as f:
                train_code = f.read()
        if os.path.exists(REQUIREMENTS_PATH):
            with open(REQUIREMENTS_PATH, "r", encoding="utf-8") as f:
                req_text = f.read()
        if os.path.exists(SCHEME_DOC_PATH):
            with open(SCHEME_DOC_PATH, "r", encoding="utf-8") as f:
                scheme_text = f.read()
        self.write_json({
            "ok": True,
            "train_py": train_code,
            "requirements_txt": req_text,
            "scheme_md": scheme_text,
            "train_py_path": TRAIN_SCRIPT_PATH,
            "requirements_path": REQUIREMENTS_PATH,
        })

    async def post(self):
        """保存修改后的 train.py 与 requirements.txt"""
        body = self.get_json_body()
        if "train_py" in body:
            with open(TRAIN_SCRIPT_PATH, "w", encoding="utf-8") as f:
                f.write(body["train_py"])
        if "requirements_txt" in body:
            with open(REQUIREMENTS_PATH, "w", encoding="utf-8") as f:
                f.write(body["requirements_txt"])
        self.write_json({"ok": True})


def make_app() -> tornado.web.Application:
    return tornado.web.Application(
        [
            (r"/api/overview", OverviewHandler),
            (r"/api/vms", VMsHandler),
            (r"/api/vms/([^/]+)", VMDetailHandler),
            (r"/api/vms/([^/]+)/probe", VMProbeHandler),
            (r"/api/vms/([^/]+)/relay/start", VMRelayStartHandler),
            (r"/api/vms/([^/]+)/relay/stop", VMRelayStopHandler),
            (r"/api/files", FilesConfigHandler),
            (
                r"/(.*)",
                tornado.web.StaticFileHandler,
                {"path": STATIC_DIR, "default_filename": "index.html"},
            ),
        ],
        debug=False,
    )


def main():
    port = int(os.environ.get("COLAB_PILOT_PORT", "8765"))
    os.makedirs(STATIC_DIR, exist_ok=True)
    app = make_app()
    app.listen(port, address="127.0.0.1")
    print(f"============================================================")
    print(f"🚀 Colab Pilot 可视化管控台已启动: http://127.0.0.1:{port}")
    print(f"📂 工作目录: {WORKSPACE_DIR}")
    print(f"🔑 Rclone 配置: {get_rclone_conf_path()}")
    print(f"============================================================", flush=True)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
