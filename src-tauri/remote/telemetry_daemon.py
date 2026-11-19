# -*- coding: utf-8 -*-
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

TELEMETRY_FILE = "/content/.colab_pilot_telemetry.json"
TRAINING_STATE_FILE = "/content/.colab_pilot_training_state.json"
LLM_STATE_FILE = "/content/.colab_pilot_llm_state.json"
WORKSPACE_META_FILE = "/content/.colab_workspace.json"
PID_FILE = "/content/.colab_pilot_daemon.pid"
PROBE_ACK_FILE = "/content/.colab_pilot_probe_ack.json"


def _finite(obj):
    """递归清洗：非有限浮点数 (NaN / ±Infinity) 转为 None，保证输出严格合法的 JSON。"""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


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


def read_proc_cwd(pid):
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def read_proc_workspace_env(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            raw = f.read()
        for item in raw.split(b"\x00"):
            if item.startswith(b"COLAB_WORKSPACE="):
                return item.split(b"=", 1)[1].decode("utf-8", errors="ignore")
    except Exception:
        pass
    return None


def classify_role(cmd_s):
    low = cmd_s.lower()
    if "colab_pilot_daemon" in low:
        return "telemetry"
    if "rclone" in low:
        return "rclone"
    if "uv pip" in low or "pip install" in low or "apt-get" in low:
        return "install"
    if (
        "vllm" in low
        or "ollama" in low
        or "llama-server" in low
        or "llama_cpp" in low
        or "llm_metrics_sidecar" in low
        or "hf_openai_server" in low
        or "text-generation" in low
        or "sglang" in low
    ):
        return "llm"
    if (
        "train" in low
        or "finetune" in low
        or "torchrun" in low
        or "accelerate" in low
        or "colab_relay_adapter" in low
    ):
        return "train"
    if "ipykernel_launcher" in low or "colab_kernel_launcher" in low:
        return "kernel"
    if "tmux" in low or ("bash" in low and "-i" in low):
        return "shell"
    if ("python" in low or "pytest" in low or "bash " in low or "sh -c" in low) and "/usr/colab" not in low and "jupyter" not in low and "dap_multiplexer" not in low:
        return "exec"
    return "system"


def get_processes_and_threads():
    total_procs = 0
    total_threads = 0
    key_procs = []
    detected_ws_env = None
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
            if "ps -eo" in cmd_s:
                continue
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

            role = classify_role(cmd_s)
            if not detected_ws_env and role in ("train", "llm", "exec", "kernel", "shell"):
                detected_ws_env = read_proc_workspace_env(pid)

            is_interesting = (
                role in ("train", "llm", "exec", "rclone", "install", "kernel", "shell", "telemetry")
                or pcpu >= 0.5
                or len(key_procs) < 10
            )
            if is_interesting and len(key_procs) < 16:
                cwd = read_proc_cwd(pid)
                key_procs.append({
                    "pid": pid,
                    "threads": nlwp,
                    "cpu_percent": pcpu,
                    "mem_percent": pmem,
                    "rss_mb": rss_mb,
                    "state": stat_s,
                    "elapsed": etime_s,
                    "role": role,
                    "cwd": cwd,
                    "command": cmd_s[:200],
                })
    except Exception:
        pass

    # 优先将用户任务进程排在最前面
    priority = {
        "train": 0,
        "llm": 1,
        "exec": 2,
        "install": 3,
        "rclone": 4,
        "kernel": 5,
        "shell": 6,
        "telemetry": 7,
        "system": 8,
    }
    key_procs.sort(key=lambda p: (priority.get(p["role"], 9), -p["cpu_percent"]))

    active_proc = next(
        (p for p in key_procs if p["role"] in ("train", "llm", "exec", "install", "rclone")),
        None,
    )
    if not active_proc:
        # 如果 kernel CPU 占用高，说明正在 notebook / colab exec 中运行代码
        kern = next((p for p in key_procs if p["role"] == "kernel" and p["cpu_percent"] >= 5.0), None)
        active_proc = kern

    return {
        "total_processes": total_procs,
        "total_threads": total_threads,
        "active_process": active_proc,
        "detected_workspace_env": detected_ws_env,
        "processes": key_procs,
    }


def list_content_files():
    entries = []
    try:
        for name in sorted(os.listdir("/content")):
            if name.startswith(".colab_pilot") or name in ("sample_data", ".config"):
                continue
            full = os.path.join("/content", name)
            try:
                st = os.stat(full)
                is_dir = os.path.isdir(full)
                size_mb = round(st.st_size / (1024 * 1024), 2) if not is_dir else 0.0
                entries.append({
                    "name": name,
                    "is_dir": is_dir,
                    "size_mb": size_mb,
                    "mtime": int(st.st_mtime),
                })
            except Exception:
                continue
            if len(entries) >= 12:
                break
    except Exception:
        pass
    return entries


def collect_once(prev_idle, prev_total, cpu_model, probe_nonce=None, daemon_started_at=None):
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

    llm_state = None
    if os.path.exists(LLM_STATE_FILE):
        try:
            with open(LLM_STATE_FILE, "r", encoding="utf-8") as f:
                llm_state = json.load(f)
        except Exception:
            pass

    ws_meta = None
    if os.path.exists(WORKSPACE_META_FILE):
        try:
            with open(WORKSPACE_META_FILE, "r", encoding="utf-8") as f:
                ws_meta = json.load(f)
        except Exception:
            pass

    proc_info = get_processes_and_threads()
    train_proc = next((p for p in proc_info["processes"] if p["role"] in ("train", "exec")), None)
    llm_proc = next((p for p in proc_info["processes"] if p["role"] == "llm"), None)
    rclone_proc = next((p for p in proc_info["processes"] if p["role"] == "rclone"), None)

    has_ckpt = False
    ckpt_size_mb = 0.0
    try:
        if os.path.exists("/content/checkpoint.pt"):
            has_ckpt = True
            ckpt_size_mb = round(os.path.getsize("/content/checkpoint.pt") / (1024 * 1024), 2)
    except OSError:
        has_ckpt = False
        ckpt_size_mb = 0.0

    now_ts = time.time()
    effective_nonce = (
        probe_nonce
        if probe_nonce is not None
        else (os.environ.get("COLAB_PILOT_PROBE_NONCE", "").strip() or None)
    )
    payload = {
        "timestamp": now_ts,
        "daemon_pid": os.getpid(),
        "daemon_started_at": round(daemon_started_at if daemon_started_at is not None else now_ts, 3),
        "probe_nonce": effective_nonce,
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
            "active_process": proc_info["active_process"],
            "train_process": train_proc,
            "llm_process": llm_proc,
            "rclone_process": rclone_proc,
            "processes": proc_info["processes"],
        },
        "files": {
            "has_rclone_conf": os.path.exists("/content/rclone.conf"),
            "has_checkpoint": has_ckpt,
            "checkpoint_size_mb": ckpt_size_mb,
            "has_done_flag": (
                os.path.exists("/content/training_done.flag")
                or os.path.exists("/content/TRAINING_COMPLETED.flag")
            ),
            "has_results_dir": os.path.isdir("/content/results"),
            "entries": list_content_files(),
        },
        "workspace_meta": ws_meta,
        "detected_workspace_env": proc_info["detected_workspace_env"],
        "training": training_state,
        "llm": llm_state,
    }

    tmp = f"{TELEMETRY_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_finite(payload), f, ensure_ascii=False, allow_nan=False)
    os.replace(tmp, TELEMETRY_FILE)
    return idle, total, payload


def is_live_telemetry_daemon_pid(pid, proc_root="/proc"):
    """验证给定 PID 为当前存活的 telemetry_daemon 守护进程（拒绝僵尸进程、已死 PID 或被无关进程复用的 PID）。"""
    try:
        pid_int = int(pid)
    except Exception:
        return False
    if pid_int <= 0:
        return False

    if proc_root and os.path.isdir(proc_root):
        proc_dir = os.path.join(proc_root, str(pid_int))
        if not os.path.isdir(proc_dir):
            return False
        stat_path = os.path.join(proc_dir, "stat")
        if os.path.exists(stat_path):
            try:
                with open(stat_path, "r", encoding="utf-8", errors="ignore") as f:
                    stat_txt = f.read()
                rparen = stat_txt.rfind(")")
                if rparen != -1:
                    after = stat_txt[rparen + 1 :].strip().split()
                    if after and after[0] == "Z":
                        return False
            except Exception:
                pass
        cmdline_path = os.path.join(proc_dir, "cmdline")
        try:
            with open(cmdline_path, "rb") as f:
                raw_cmd = f.read()
            cmd_txt = raw_cmd.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
        except Exception:
            return False
        low = cmd_txt.lower()
        if not low or "--bootstrap" in low:
            return False
        return (".colab_pilot_daemon" in low) or ("telemetry_daemon" in low)

    try:
        os.kill(pid_int, 0)
        return True
    except Exception:
        return False


def _terminate_stale_daemon_if_any(proc_root="/proc"):
    if not os.path.exists(PID_FILE):
        return
    try:
        with open(PID_FILE, "r", encoding="utf-8") as f:
            old_pid = int(f.read().strip())
    except Exception:
        return
    if old_pid <= 0 or old_pid == os.getpid():
        return
    if not is_live_telemetry_daemon_pid(old_pid, proc_root=proc_root):
        return
    try:
        os.kill(old_pid, signal.SIGTERM)
    except Exception:
        return
    deadline = time.time() + 0.6
    while time.time() < deadline:
        if not is_live_telemetry_daemon_pid(old_pid, proc_root=proc_root):
            return
        time.sleep(0.05)
    sig_kill = getattr(signal, "SIGKILL", signal.SIGTERM)
    try:
        os.kill(old_pid, sig_kill)
    except Exception:
        pass


def bootstrap_probe(nonce=None, timeout_sec=6.0, daemon_script=None, proc_root="/proc"):
    """
    安全重装/启动遥测探针：
    1. 若旧 daemon 存活则先行终止（不误杀复用 PID 的无关进程）；
    2. 清理陈旧的 TELEMETRY_FILE / PID_FILE / PROBE_ACK_FILE；
    3. 注入本次部署时间戳 launch_ts 与 deploy_nonce 启动新 daemon；
    4. 严格验证 /proc/<pid> 为存活的 telemetry_daemon 且写出属于本次启动的新样本后才输出 PROBE_OK。
    """
    launch_ts = time.time()
    effective_nonce = (
        (nonce or os.environ.get("COLAB_PILOT_PROBE_NONCE") or "").strip()
        or f"nonce-{int(launch_ts * 1000)}-{os.getpid()}"
    )
    _terminate_stale_daemon_if_any(proc_root=proc_root)
    for stale_path in (TELEMETRY_FILE, PID_FILE, PROBE_ACK_FILE):
        try:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        except OSError:
            pass

    script_path = daemon_script or os.path.abspath(__file__)
    child_env = os.environ.copy()
    child_env["COLAB_PILOT_PROBE_NONCE"] = effective_nonce
    subprocess.Popen(
        [sys.executable, script_path, "--nonce", effective_nonce],
        env=child_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + max(0.5, float(timeout_sec))
    while time.time() < deadline:
        pid_val = None
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, "r", encoding="utf-8") as f:
                    pid_val = int(f.read().strip())
            except Exception:
                pid_val = None

        if pid_val and is_live_telemetry_daemon_pid(pid_val, proc_root=proc_root) and os.path.exists(TELEMETRY_FILE):
            try:
                with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
                    sample = json.load(f)
            except Exception:
                sample = None
            if isinstance(sample, dict):
                sample_ts = sample.get("timestamp")
                sample_pid = sample.get("daemon_pid")
                sample_nonce = sample.get("probe_nonce")
                if (
                    isinstance(sample_ts, (int, float))
                    and float(sample_ts) >= launch_ts
                    and sample_pid == pid_val
                    and sample_nonce == effective_nonce
                ):
                    ack = {
                        "ok": True,
                        "pid": pid_val,
                        "nonce": effective_nonce,
                        "launch_ts": round(launch_ts, 3),
                        "sample_ts": sample_ts,
                    }
                    ack_tmp = f"{PROBE_ACK_FILE}.{os.getpid()}.tmp"
                    try:
                        with open(ack_tmp, "w", encoding="utf-8") as f:
                            json.dump(ack, f, ensure_ascii=False)
                        os.replace(ack_tmp, PROBE_ACK_FILE)
                    except OSError:
                        pass
                    print(f"PROBE_OK started pid={pid_val} nonce={effective_nonce}", flush=True)
                    return 0
        time.sleep(0.15)

    print(f"PROBE_FAIL timeout nonce={effective_nonce}", flush=True)
    return 1


def _acquire_singleton_pid_lock():
    pid_f = open(PID_FILE, "a+", encoding="utf-8")
    try:
        import fcntl

        fcntl.flock(pid_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except ImportError:
        pass
    except OSError:
        pid_f.close()
        return None
    pid_f.seek(0)
    pid_f.truncate()
    pid_f.write(str(os.getpid()))
    pid_f.flush()
    return pid_f


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--nonce", default=None)
    args, _ = parser.parse_known_args(argv)

    if args.bootstrap:
        rc = bootstrap_probe(nonce=args.nonce)
        raise SystemExit(rc)

    lock_fh = _acquire_singleton_pid_lock()
    if lock_fh is None:
        return
    started_at = time.time()
    effective_nonce = (args.nonce or os.environ.get("COLAB_PILOT_PROBE_NONCE") or "").strip() or None
    cpu_model = get_cpu_model()
    idle, total = read_cpu_stat()
    time.sleep(0.15)
    while True:
        try:
            idle, total, _ = collect_once(
                idle,
                total,
                cpu_model,
                probe_nonce=effective_nonce,
                daemon_started_at=started_at,
            )
        except Exception:
            pass
        time.sleep(1.0)


if __name__ == "__main__":
    main()
