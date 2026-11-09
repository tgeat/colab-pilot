# -*- coding: utf-8 -*-
import json
import os
import shutil
import subprocess
import time

TELEMETRY_FILE = "/content/.colab_pilot_telemetry.json"
TRAINING_STATE_FILE = "/content/.colab_pilot_training_state.json"
LLM_STATE_FILE = "/content/.colab_pilot_llm_state.json"
WORKSPACE_META_FILE = "/content/.colab_workspace.json"
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
    if ("python" in low or "pytest" in low or "bash " in low or "sh -c" in low) and "/usr/colab" not in low and "jupyter" not in low and "dap_multiplexer" not in low:
        return "exec"
    if "tmux" in low or ("bash" in low and "-i" in low):
        return "shell"
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
            "active_process": proc_info["active_process"],
            "train_process": train_proc,
            "llm_process": llm_proc,
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
            "has_results_dir": os.path.isdir("/content/results"),
            "entries": list_content_files(),
        },
        "workspace_meta": ws_meta,
        "detected_workspace_env": proc_info["detected_workspace_env"],
        "training": training_state,
        "llm": llm_state,
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
        time.sleep(1.0)


if __name__ == "__main__":
    main()
