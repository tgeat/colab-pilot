#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 深度学习串行接力守护进程 (dl_relay_runner.py)
==========================================================================
由 Antigravity `colab-dl-relay` Skill 在本机 Smoke Test 通过后调用，负责：
1. 在 `~/.config/colab-cli/workspaces.json` 注册当前工作区映射，使 Colab Pilot 立即识别工作区
2. 自动隔离默认裸路径 `gdrive:colab_relay` 为 `gdrive:colab_relay/<sanitized_prefix>`，并检测目录活跃锁冲突
3. 严格区分分配尝试次数与有效接力棒次 `effective_round`，循环驱动 `colab-cli` 执行多轮跨实例串行接力（先存后拆，严格核验停机释放，零双倍计费）
4. 解耦云端训练完成与本地最终权重落盘校验，支持重跑自动补下载
"""

import argparse
import ast
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLAB_UPDATE_BANNER_RE = re.compile(
    r"^\s*\[colab\]\s*(?:A new version of Colab CLI is available|Run 'colab update'|To silence this check).*$",
    re.IGNORECASE,
)

# Google Colab `/tun/m/assign` 仅提供单一默认内存规格的加速器（传入 `shape=hm` / `--high-mem` 会触发 HTTP 400）
SINGLE_SHAPE_ACCELERATORS = {"G4", "H100", "L4", "V5E1", "V6E1"}
TPU_ACCELERATORS = {"V5E1", "V6E1"}

DEFAULT_BARE_GDRIVE_DIR = "gdrive:colab_relay"
DONE_FLAG_FILENAMES = ("training_done.flag", "TRAINING_COMPLETED.flag")
ALLOC_RETRY_SLEEP_SECONDS = 30.0
HANDOVER_SLEEP_SECONDS = 10.0

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def strip_colab_cli_banners(text: str) -> str:
    """剥离 google-colab-cli v0.8.0+ 的版本更新提示横幅与 ANSI 控制序列，避免干扰输出解析。"""
    clean = strip_ansi(text or "")
    kept = [ln for ln in clean.splitlines() if not COLAB_UPDATE_BANNER_RE.match(ln)]
    return "\n".join(kept).strip()


def sanitize_gdrive_segment(raw: str) -> str:
    """将任意会话前缀/名称清洗为安全的 Google Drive 子目录名。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (raw or "").strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-.")
    return cleaned or "default"


def resolve_effective_gdrive_dir(
    raw_gdrive_dir: str,
    session_name: str,
    session_prefix: str = "",
) -> Tuple[str, bool]:
    """
    当 `--gdrive-dir` 为默认裸路径 `gdrive:colab_relay`（未指定子目录）时，
    自动按 `--session-prefix`（优先）或 `--session` 隔离为 `gdrive:colab_relay/<sanitized_prefix>`。
    返回 `(effective_gdrive_dir, was_auto_isolated)`。
    """
    cleaned = (raw_gdrive_dir or "").strip().rstrip("/")
    if not cleaned or cleaned == DEFAULT_BARE_GDRIVE_DIR:
        ident = sanitize_gdrive_segment((session_prefix or "").strip() or session_name)
        return f"{DEFAULT_BARE_GDRIVE_DIR}/{ident}", True
    return cleaned, False


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
            f"Google Colab 云端 {acc_label} 算力池当前暂时库存不足/排队拥堵 (HTTP 503)，"
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


def allocate_colab_instance(
    colab_bin: str,
    session_name: str,
    accelerator: str,
    high_mem: bool,
    timeout: int = 300,
    log_prefix: str = "",
) -> subprocess.CompletedProcess:
    new_cmd = build_colab_new_cmd(colab_bin, session_name, accelerator, high_mem)
    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"{prefix}正在分配 Colab 实例: {' '.join(new_cmd)}", flush=True)
    res_new = run_cmd(new_cmd, timeout=timeout)
    combined = (res_new.stderr or "") + "\n" + (res_new.stdout or "")
    if res_new.returncode != 0 and "--high-mem" in new_cmd and is_high_mem_rejected_error(combined):
        fallback_cmd = [c for c in new_cmd if c != "--high-mem"]
        print(
            f"{prefix}⚠️ 检测到后端拒绝 --high-mem 规格 (HTTP 400)，自动移除 --high-mem 重试分配: {' '.join(fallback_cmd)}",
            flush=True,
        )
        res_new = run_cmd(fallback_cmd, timeout=timeout)
    return res_new


def classify_relay_round_output(
    exec_rc: int,
    output: str,
    has_remote_done_flag: bool = False,
) -> str:
    """
    判定单棒云端执行结果（防御 colab exec 在内核抛出异常/SystemExit 时仍返回 exit 0 的陷阱）：
    返回 "completed" | "handover" | "failed"
    """
    if exec_rc != 0:
        return "failed"
    clean = strip_colab_cli_banners(output)
    if "TRAIN_EXIT_CODE=0" not in clean:
        return "failed"
    if (
        "CLOUD_SMOKE_FAILED" in clean
        or "Traceback (most recent call last)" in clean
        or "SyntaxError:" in clean
        or "IndentationError:" in clean
    ):
        return "failed"
    if "TRAINING_COMPLETED" in clean or has_remote_done_flag:
        return "completed"
    if "HANDOVER_REQUESTED" in clean:
        return "handover"
    return "failed"


def check_python_syntax(filepath: str) -> None:
    with open(filepath, "rb") as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    compile(tree, filename=filepath, mode="exec")


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


def colab_cfg_dir() -> str:
    cfg_dir = os.path.expanduser("~/.config/colab-cli")
    os.makedirs(cfg_dir, exist_ok=True)
    return cfg_dir


def relay_locks_path() -> str:
    return os.path.join(colab_cfg_dir(), "dl_relay_locks.json")


def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x1000, False, int(pid))
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong(0)
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == 259  # STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def acquire_gdrive_dir_lock(
    gdrive_dir: str,
    session_name: str,
    workspace_dir: str,
    allow_shared: bool = False,
) -> Tuple[bool, str]:
    """检查并登记本地活跃接力目录锁，防止两个不同会话误共用同一云盘目录互相覆盖断点与完成标记。"""
    lock_file = relay_locks_path()
    norm_dir = gdrive_dir.strip().rstrip("/")
    locks: Dict[str, Any] = {}
    if os.path.exists(lock_file):
        try:
            with open(lock_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    locks = raw
        except Exception:
            locks = {}

    alive_locks: Dict[str, Any] = {}
    for k, v in locks.items():
        if not isinstance(v, dict):
            continue
        rec_pid = int(v.get("pid") or 0)
        if _is_pid_alive(rec_pid):
            alive_locks[k] = v

    existing = alive_locks.get(norm_dir)
    if isinstance(existing, dict) and existing.get("session") != session_name and not allow_shared:
        other_sess = existing.get("session")
        other_pid = existing.get("pid")
        return (
            False,
            f"云盘目录 `{norm_dir}` 正被另一活跃接力任务 [{other_sess}] (PID={other_pid}) 占用！"
            f"请改用独立的 `--gdrive-dir` / `--session-prefix`，或显式传入 `--allow-shared-gdrive-dir` 确认复用。",
        )

    alive_locks[norm_dir] = {
        "session": session_name,
        "pid": os.getpid(),
        "workspace": os.path.abspath(workspace_dir),
        "updated_at": time.time(),
    }
    tmp = f"{lock_file}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(alive_locks, f, ensure_ascii=False, indent=2)
    os.replace(tmp, lock_file)
    return True, ""


def release_gdrive_dir_lock(gdrive_dir: str, session_name: str) -> None:
    lock_file = relay_locks_path()
    norm_dir = gdrive_dir.strip().rstrip("/")
    if not os.path.exists(lock_file):
        return
    try:
        with open(lock_file, "r", encoding="utf-8") as f:
            locks = json.load(f)
        if not isinstance(locks, dict):
            return
        cur = locks.get(norm_dir)
        if isinstance(cur, dict) and cur.get("session") == session_name and int(cur.get("pid") or 0) == os.getpid():
            locks.pop(norm_dir, None)
            tmp = f"{lock_file}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(locks, f, ensure_ascii=False, indent=2)
            os.replace(tmp, lock_file)
    except Exception:
        pass


def register_workspace(session_name: str, workspace_dir: str, script_name: str, description: str) -> None:
    cfg_dir = colab_cfg_dir()
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
        "active_script": script_name,
        "description": description,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = f"{reg_file}.{os.getpid()}.tmp"
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


def _is_absent_session_text(text: str) -> bool:
    low = strip_colab_cli_banners(text).lower()
    return any(
        k in low
        for k in (
            "no active session",
            "session not found",
            "no session",
            "does not exist",
            "unknown session",
            "not assigned",
            "unassigned",
            "0 active",
        )
    )


def _status_indicates_session_active(session_name: str, rc: int, raw_output: str) -> Optional[bool]:
    """
    解析 `colab status -s <session>` 输出：
    - 返回 False：已明确不存在/已释放
    - 返回 True ：仍在服务端活跃计费列表中
    - 返回 None ：查询本身异常，无法确认已释放
    """
    clean = strip_colab_cli_banners(raw_output)
    low = clean.lower()
    if _is_absent_session_text(clean):
        return False
    if rc == 0:
        if not clean:
            return False
        if f"[{session_name}]" in clean:
            return True
        if session_name.lower() in low and any(
            w in low for w in ("active", "assigned", "connected", "running", "hardware:", "tpu:", "m-s-")
        ):
            return True
        return False
    return None


def _sessions_list_contains_active(session_name: str, rc: int, raw_output: str) -> Optional[bool]:
    clean = strip_colab_cli_banners(raw_output)
    if _is_absent_session_text(clean):
        return False
    if rc != 0:
        return None
    if not clean:
        return False
    for ln in clean.splitlines():
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if any(neg in low for neg in ("no active", "no session", "not found", "stopped", "unassigned", "0 session")):
            continue
        if re.search(rf"(?:^|[\s\[\(]){re.escape(session_name)}(?:$|[\s\]\):])", s):
            return True
    return False


def is_colab_session_active(colab_bin: str, session_name: str) -> Optional[bool]:
    chk = run_cmd([colab_bin, "status", "-s", session_name], timeout=30)
    chk_state = _status_indicates_session_active(
        session_name, chk.returncode, (chk.stdout or "") + "\n" + (chk.stderr or "")
    )
    if chk_state is True:
        return True
    sess_ls = run_cmd([colab_bin, "sessions"], timeout=30)
    ls_state = _sessions_list_contains_active(
        session_name, sess_ls.returncode, (sess_ls.stdout or "") + "\n" + (sess_ls.stderr or "")
    )
    if ls_state is True:
        return True
    if chk_state is False and ls_state is False:
        return False
    if chk_state is False and ls_state is None:
        return False
    return None


def stop_and_verify_session(
    colab_bin: str,
    session_name: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    require_stop_rc_zero: bool = True,
) -> Tuple[bool, str]:
    """
    严格停止并核验旧实例已从服务端释放（Issue #46）：
    1. 检查 `colab stop -s <session>` 返回码；
    2. 通过 `colab status -s <session>` 与 `colab sessions` 核验实例不再处于服务端计费列表中；
    3. 支持失败自动重试；若仍无法确认已释放，返回 `(False, reason)` 阻止申请下一台实例。
    """
    last_err = "未知停机错误"
    attempts = max(1, int(max_retries))
    for attempt in range(1, attempts + 1):
        try:
            res_stop = run_cmd([colab_bin, "stop", "-s", session_name], timeout=120)
        except subprocess.TimeoutExpired:
            last_err = f"colab stop 超时 (attempt {attempt}/{attempts})"
            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue

        stop_out = strip_colab_cli_banners((res_stop.stdout or "") + "\n" + (res_stop.stderr or ""))
        absent_on_stop = _is_absent_session_text(stop_out)
        stop_ok = (res_stop.returncode == 0) or (not require_stop_rc_zero and absent_on_stop)

        if not stop_ok:
            last_err = f"`colab stop -s {session_name}` 返回非零退出码 ({res_stop.returncode}): {stop_out[:240] or '无详细输出'}"
            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue

        active_now = is_colab_session_active(colab_bin, session_name)
        if active_now is False:
            return True, ""
        if active_now is True:
            last_err = f"执行 `colab stop -s {session_name}` 后核验发现会话仍在服务端计费列表中 (attempt {attempt}/{attempts})"
        else:
            last_err = f"执行 `colab stop -s {session_name}` 后无法通过服务端状态接口确认实例已释放 (attempt {attempt}/{attempts})"

        if attempt < attempts and retry_delay > 0:
            time.sleep(retry_delay)

    return False, last_err


def upload_checked(colab_bin: str, session: str, local_path: str, remote_path: str, timeout: int = 120) -> None:
    res = run_cmd([colab_bin, "upload", "-s", session, local_path, remote_path], timeout=timeout)
    if res.returncode != 0:
        err = strip_colab_cli_banners((res.stderr or "") + "\n" + (res.stdout or ""))[:300]
        raise RuntimeError(f"colab upload 失败 ({local_path} -> {remote_path}): {err}")


def list_gdrive_files(rclone_bin: str, rclone_conf: str, gdrive_dir: str) -> Optional[List[str]]:
    cmd = [rclone_bin]
    if rclone_conf and os.path.exists(rclone_conf):
        cmd.extend(["--config", rclone_conf])
    cmd.extend(["--retries=5", "--low-level-retries=20", "lsf", "--files-only", "--", gdrive_dir.rstrip("/")])
    res = run_cmd(cmd, timeout=90)
    if res.returncode == 0:
        return [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
    err_low = ((res.stderr or "") + "\n" + (res.stdout or "")).lower()
    if any(k in err_low for k in ("directory not found", "object not found", "file not found")):
        return []
    return None


def gdrive_has_done_flag(rclone_bin: str, rclone_conf: str, gdrive_dir: str) -> bool:
    files = list_gdrive_files(rclone_bin, rclone_conf, gdrive_dir)
    if files is not None:
        return any(flag in files for flag in DONE_FLAG_FILENAMES)
    for flag in DONE_FLAG_FILENAMES:
        cmd = [rclone_bin]
        if rclone_conf and os.path.exists(rclone_conf):
            cmd.extend(["--config", rclone_conf])
        cmd.extend(["--retries=5", "--low-level-retries=20", "lsf", "--", f"{gdrive_dir.rstrip('/')}/{flag}"])
        res = run_cmd(cmd, timeout=90)
        if res.returncode == 0 and flag in (res.stdout or ""):
            return True
    return False


def check_remote_active_lock(
    rclone_bin: str,
    rclone_conf: str,
    gdrive_dir: str,
    session_name: str,
) -> Tuple[bool, str]:
    """若显式指定的云盘目录存在其他活跃会话写入的 `.relay_active_lock.json`，给出清晰冲突提示。"""
    files = list_gdrive_files(rclone_bin, rclone_conf, gdrive_dir)
    if not files or ".relay_active_lock.json" not in files:
        return True, ""
    cmd = [rclone_bin]
    if rclone_conf and os.path.exists(rclone_conf):
        cmd.extend(["--config", rclone_conf])
    cmd.extend(["--retries=3", "cat", "--", f"{gdrive_dir.rstrip('/')}/.relay_active_lock.json"])
    res = run_cmd(cmd, timeout=45)
    if res.returncode != 0 or not (res.stdout or "").strip():
        return True, ""
    try:
        meta = json.loads(res.stdout)
    except Exception:
        return True, ""
    if not isinstance(meta, dict):
        return True, ""
    owner = str(meta.get("session") or "").strip()
    updated_at = float(meta.get("updated_at") or 0.0)
    if owner and owner != session_name and (time.time() - updated_at) < 86400:
        return (
            False,
            f"云盘目录 `{gdrive_dir}` 中存在会话 [{owner}] 的活跃锁标记 (`.relay_active_lock.json`)！"
            f"为避免覆盖其他任务的断点或完成标记，请更换 `--gdrive-dir` 或显式添加 `--allow-shared-gdrive-dir`。",
        )
    return True, ""


def download_and_verify_final_weights(
    rclone_bin: str,
    rclone_conf: str,
    gdrive_dir: str,
    ckpt_basename: str,
    workspace_abs: str,
) -> Tuple[bool, str, str]:
    """
    将云盘最终权重下载到本地工作区并严格校验文件存在且非空（Issue #53）。
    返回 `(ok, local_ckpt_path, error_message)`。
    """
    stem, ext = os.path.splitext(ckpt_basename)
    local_ckpt = os.path.join(workspace_abs, f"{stem}_final{ext or '.pt'}")
    local_results = os.path.join(workspace_abs, "results")
    os.makedirs(local_results, exist_ok=True)

    if os.path.exists(local_ckpt) and os.path.getsize(local_ckpt) == 0:
        try:
            os.remove(local_ckpt)
        except OSError:
            pass

    base_rclone = [rclone_bin]
    if rclone_conf and os.path.exists(rclone_conf):
        base_rclone.extend(["--config", rclone_conf])
    base_rclone.extend(["--retries=5", "--low-level-retries=20"])

    rc1 = run_cmd(
        base_rclone + ["copyto", "--", f"{gdrive_dir.rstrip('/')}/{ckpt_basename}", local_ckpt],
        timeout=300,
    )
    rc2 = run_cmd(
        base_rclone + ["copy", "--", f"{gdrive_dir.rstrip('/')}/results", local_results],
        timeout=300,
    )
    if rc2.returncode != 0:
        print(f"⚠️ 拉取 results 目录警告: {(rc2.stderr or '').strip()[:240]}", flush=True)

    if rc1.returncode != 0:
        err_detail = strip_ansi((rc1.stderr or "") + "\n" + (rc1.stdout or "")).strip()[:240]
        return False, local_ckpt, f"rclone 下载最终权重返回非零退出码 ({rc1.returncode}): {err_detail or '未知错误'}"

    if not os.path.isfile(local_ckpt) or os.path.getsize(local_ckpt) <= 0:
        return False, local_ckpt, f"本地最终权重文件不存在或大小为 0 字节: {local_ckpt}"

    return True, local_ckpt, ""


def write_local_relay_receipt(
    workspace_abs: str,
    training_completed: bool,
    weights_downloaded: bool,
    local_weights_path: Optional[str],
    download_error: Optional[str] = None,
) -> None:
    receipt_path = os.path.join(workspace_abs, ".colab_dl_relay_status.json")
    payload = {
        "training_completed": training_completed,
        "weights_downloaded": weights_downloaded,
        "local_weights_path": local_weights_path if weights_downloaded else None,
        "download_error": download_error,
        "updated_at": time.time(),
    }
    try:
        tmp = f"{receipt_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, receipt_path)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab 深度学习跨实例串行接力编排器")
    parser.add_argument("--session", "-s", default="", help="colab-cli 会话名称 (建议与工作区同名)")
    parser.add_argument("--session-prefix", default="", help="任务会话前缀（用于默认云盘目录子路径隔离）")
    parser.add_argument("--workspace", default=".", help="本地工作区目录")
    parser.add_argument("--script", default="train.py", help="主训练脚本文件名")
    parser.add_argument("--checkpoint-path", default="checkpoint.pt", help="断点文件名 (默认: checkpoint.pt)")
    parser.add_argument("--extra-files", nargs="*", default=[], help="需额外上传到 /content 的文件或目录")
    parser.add_argument("--gpu", default="T4", help="Colab 加速器类型: T4 / L4 / G4 / A100 / H100 / CPU")
    parser.add_argument(
        "--high-mem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或禁用 High-RAM 规格 (--high-mem / --no-high-mem)",
    )
    parser.add_argument("--gdrive-dir", default=DEFAULT_BARE_GDRIVE_DIR, help="Google Drive 断点与结果保存路径")
    parser.add_argument(
        "--allow-shared-gdrive-dir",
        "--reuse-gdrive-dir",
        dest="allow_shared_gdrive_dir",
        action="store_true",
        help="显式允许与另一会话共用已被占用的云盘目录",
    )
    parser.add_argument("--gdrive-dataset-dir", default="", help="Google Drive 数据集路径 (可选)")
    parser.add_argument("--total-epochs", type=int, default=100, help="总训练轮次")
    parser.add_argument("--max-session-hours", type=float, default=22.5, help="单台实例最长运行小时数 (默认 22.5h)")
    parser.add_argument("--backup-interval-minutes", type=float, default=30.0, help="Google Drive 定时热备间隔 (分钟)")
    parser.add_argument("--reset-done-flag", action="store_true", help="启动第 1 棒时清除旧的完成标记")
    parser.add_argument("--detach", action="store_true", help="在后台独立启动守护进程（关闭 Antigravity 不中断）")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    session_name = (args.session or args.session_prefix or "").strip()
    if not session_name:
        parser.error("必须提供 --session (-s) 或 --session-prefix")
    args.session = session_name

    effective_gdrive_dir, was_auto_isolated = resolve_effective_gdrive_dir(
        args.gdrive_dir,
        session_name=session_name,
        session_prefix=args.session_prefix,
    )
    args.gdrive_dir = effective_gdrive_dir
    if was_auto_isolated:
        print(f"🔒 已将默认云盘路径自动按会话隔离为: {effective_gdrive_dir}", flush=True)

    workspace_abs = os.path.abspath(args.workspace)
    script_abs = os.path.abspath(os.path.join(workspace_abs, args.script))
    adapter_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_relay_adapter.py")

    if not os.path.exists(script_abs):
        print(f"❌ 找不到训练脚本: {script_abs}", flush=True)
        return 1

    try:
        check_python_syntax(script_abs)
    except Exception as e:
        print(f"❌ 上云前语法检查未通过，已阻止分配 Colab 实例: {e}", flush=True)
        return 1

    if args.detach and os.environ.get("DL_RELAY_DETACHED") != "1":
        log_file = os.path.join(workspace_abs, ".colab_dl_relay.log")
        pid_file = os.path.join(workspace_abs, ".colab_dl_relay.pid")
        raw_cli = list(argv) if argv is not None else sys.argv[1:]
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in raw_cli if a != "--detach"]
        env = os.environ.copy()
        env["DL_RELAY_DETACHED"] = "1"
        env["PYTHONUTF8"] = "1"
        kwargs: Dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW only
        else:
            kwargs["start_new_session"] = True
        with open(log_file, "a", encoding="utf-8") as out_f:
            proc = subprocess.Popen(cmd, cwd=workspace_abs, env=env, stdout=out_f, stderr=subprocess.STDOUT, **kwargs)
        with open(pid_file, "w", encoding="utf-8") as pf:
            pf.write(str(proc.pid))
        print(f"✅ 已在后台启动串行接力守护进程 PID={proc.pid}")
        print(f"   日志文件: {log_file}")
        print("   现在即使关闭 Antigravity，后台也会自动完成跨实例接力；打开 Colab Pilot 即可实时查看进度。")
        return 0

    lock_ok, lock_err = acquire_gdrive_dir_lock(
        effective_gdrive_dir,
        session_name=session_name,
        workspace_dir=workspace_abs,
        allow_shared=args.allow_shared_gdrive_dir,
    )
    if not lock_ok:
        print(f"❌ {lock_err}", flush=True)
        return 1

    colab_bin = find_bin("colab")
    rclone_bin = find_bin("rclone")
    rclone_conf = find_rclone_conf()
    ckpt_basename = os.path.basename(args.checkpoint_path)
    stem, ext = os.path.splitext(ckpt_basename)
    local_final_ckpt = os.path.join(workspace_abs, f"{stem}_final{ext or '.pt'}")

    try:
        if not was_auto_isolated and not args.allow_shared_gdrive_dir and not args.reset_done_flag:
            rem_ok, rem_err = check_remote_active_lock(rclone_bin, rclone_conf, effective_gdrive_dir, session_name)
            if not rem_ok:
                print(f"❌ {rem_err}", flush=True)
                return 1

        register_workspace(
            session_name=args.session,
            workspace_dir=workspace_abs,
            script_name=args.script,
            description=f"DL Serial Relay ({args.gpu}) -> {effective_gdrive_dir}",
        )

        if args.reset_done_flag:
            for flag_name in DONE_FLAG_FILENAMES:
                base_rc = [rclone_bin]
                if rclone_conf and os.path.exists(rclone_conf):
                    base_rc.extend(["--config", rclone_conf])
                run_cmd(
                    base_rc
                    + [
                        "--retries=5",
                        "--low-level-retries=20",
                        "deletefile",
                        "--",
                        f"{effective_gdrive_dir.rstrip('/')}/{flag_name}",
                    ],
                    timeout=90,
                )

        effective_round = 1
        alloc_attempt = 0
        fast_handover_streak = 0
        max_sec = int(args.max_session_hours * 3600)
        bk_sec = int(args.backup_interval_minutes * 60)

        while True:
            print(
                f"\n========== [Round #{effective_round}] 串行接力检查 ({args.session} -> {effective_gdrive_dir}) ==========",
                flush=True,
            )

            if (effective_round > 1 or not args.reset_done_flag) and gdrive_has_done_flag(
                rclone_bin, rclone_conf, effective_gdrive_dir
            ):
                if os.path.isfile(local_final_ckpt) and os.path.getsize(local_final_ckpt) > 0:
                    write_local_relay_receipt(workspace_abs, True, True, local_final_ckpt, None)
                    print(
                        f"🎉 检测到 Google Drive 已存在完成标记，且本地最终模型权重已就绪 ({local_final_ckpt})！",
                        flush=True,
                    )
                    return 0

                print(
                    f"📥 检测到 Google Drive 已存在完成标记，但本地最终权重缺失或为空 ({local_final_ckpt})，正在执行补下载...",
                    flush=True,
                )
                dl_ok, dl_path, dl_err = download_and_verify_final_weights(
                    rclone_bin=rclone_bin,
                    rclone_conf=rclone_conf,
                    gdrive_dir=effective_gdrive_dir,
                    ckpt_basename=ckpt_basename,
                    workspace_abs=workspace_abs,
                )
                write_local_relay_receipt(workspace_abs, True, dl_ok, dl_path, None if dl_ok else dl_err)
                if not dl_ok:
                    print(f"❌ 最终模型权重补下载失败（云盘训练已完成，但本地未落盘）: {dl_err}", flush=True)
                    return 1
                print(
                    f"🎉 最终模型已补下载并校验保存至 {dl_path}，训练结果目录：{os.path.join(workspace_abs, 'results')}！",
                    flush=True,
                )
                return 0

            # 若新分配前检测到同名残留活跃实例，必须先停止并确认释放后才能开新机
            if is_colab_session_active(colab_bin, args.session) is True:
                pre_ok, pre_err = stop_and_verify_session(
                    colab_bin,
                    args.session,
                    max_retries=3,
                    retry_delay=2.0,
                    require_stop_rc_zero=True,
                )
                if not pre_ok:
                    print(
                        f"❌ [防重叠计费中止] 启动前检测到残留活跃实例 ({args.session}) 且无法停止释放: {pre_err}",
                        flush=True,
                    )
                    return 1

            res_new = allocate_colab_instance(
                colab_bin=colab_bin,
                session_name=args.session,
                accelerator=args.gpu,
                high_mem=args.high_mem,
                timeout=300,
                log_prefix=f"[Round #{effective_round}]",
            )
            if res_new.returncode != 0:
                alloc_attempt += 1
                err_msg = format_colab_alloc_error(args.gpu, (res_new.stderr or "") + "\n" + (res_new.stdout or ""))
                print(
                    f"[Round #{effective_round} | 分配重试 #{alloc_attempt}] 分配实例失败: {err_msg}，"
                    f"{ALLOC_RETRY_SLEEP_SECONDS:g} 秒后重试（有效棒次保持为 Round #{effective_round}）...",
                    flush=True,
                )
                if ALLOC_RETRY_SLEEP_SECONDS > 0:
                    time.sleep(ALLOC_RETRY_SLEEP_SECONDS)
                continue

            alloc_attempt = 0
            ws_meta_local = os.path.join(workspace_abs, f".colab_workspace.{os.getpid()}.json")
            outcome = "failed"
            stop_ok = False
            stop_err = ""
            try:
                with open(ws_meta_local, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "name": os.path.basename(workspace_abs),
                            "path": workspace_abs,
                            "description": f"深度学习串行接力 Round #{effective_round} ({args.script})",
                        },
                        f,
                        ensure_ascii=False,
                    )

                upload_checked(colab_bin, args.session, ws_meta_local, "/content/.colab_workspace.json", timeout=90)
                upload_checked(colab_bin, args.session, rclone_conf, "/content/rclone.conf", timeout=90)
                if os.path.exists(adapter_abs):
                    upload_checked(colab_bin, args.session, adapter_abs, "/content/colab_relay_adapter.py", timeout=90)
                script_remote = f"/content/{os.path.basename(script_abs)}"
                upload_checked(colab_bin, args.session, script_abs, script_remote, timeout=120)

                req_txt = os.path.join(workspace_abs, "requirements.txt")
                if os.path.exists(req_txt):
                    upload_checked(colab_bin, args.session, req_txt, "/content/requirements.txt", timeout=90)

                for ef in args.extra_files:
                    ef_abs = os.path.abspath(os.path.join(workspace_abs, ef))
                    if os.path.exists(ef_abs):
                        upload_checked(colab_bin, args.session, ef_abs, f"/content/{os.path.basename(ef_abs)}", timeout=180)

                reset_flag = "1" if (effective_round == 1 and args.reset_done_flag) else "0"
                cloud_boot_py = f"""
import os, shutil, subprocess, sys
os.chdir("/content")
if not shutil.which("rclone"):
    subprocess.run("curl -fsSL https://rclone.org/install.sh | sudo bash", shell=True, check=True)
if os.path.exists("/content/requirements.txt") and open("/content/requirements.txt", encoding="utf-8").read().strip():
    subprocess.run(["uv", "pip", "install", "--system", "-r", "/content/requirements.txt"], check=True)

env = os.environ.copy()
env.update({{
    "COLAB_WORKSPACE": {json.dumps(workspace_abs)},
    "GDRIVE_DIR": {json.dumps(effective_gdrive_dir)},
    "RELAY_GDRIVE_DIR": {json.dumps(effective_gdrive_dir)},
    "RELAY_SESSION": {json.dumps(args.session)},
    "GDRIVE_DATASET_DIR": {json.dumps(args.gdrive_dataset_dir)},
    "CHECKPOINT_PATH": {json.dumps(f"/content/{ckpt_basename}")},
    "TOTAL_EPOCHS": "{args.total_epochs}",
    "MAX_SESSION_SECONDS": "{max_sec}",
    "BACKUP_INTERVAL_SECONDS": "{bk_sec}",
    "RELAY_ROUND": "{effective_round}",
    "RESET_DONE_FLAG": "{reset_flag}",
}})
proc = subprocess.Popen(
    [sys.executable, "-u", {json.dumps(script_remote)}],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
proc.stdout.close()
rc = proc.wait()
print(f"TRAIN_EXIT_CODE={{rc}}", flush=True)
if rc != 0:
    sys.exit(rc)
"""
                print(f"[Round #{effective_round}] 云端训练已启动，正在流式监听进度...", flush=True)
                exec_timeout = max(120, max_sec + 120)
                exec_rc = 0
                round_start_ts = time.monotonic()
                try:
                    res_exec = run_cmd(
                        [colab_bin, "exec", "-s", args.session, "--timeout", str(exec_timeout)],
                        timeout=exec_timeout + 30,
                        stdin_text=cloud_boot_py,
                    )
                    exec_rc = res_exec.returncode
                    out_all = (res_exec.stdout or "") + "\n" + (res_exec.stderr or "")
                except subprocess.TimeoutExpired as te:
                    exec_rc = 124
                    out_all = ((te.stdout or "") if isinstance(te.stdout, str) else "") + "\nTIMEOUT_EXPIRED"
                    print(f"[Round #{effective_round}] ⚠️ colab exec 响应超时 ({exec_timeout}s)，准备检查网盘断点...", flush=True)
                round_elapsed = time.monotonic() - round_start_ts
                print(out_all[-1500:], flush=True)

                remote_done = (
                    gdrive_has_done_flag(rclone_bin, rclone_conf, effective_gdrive_dir)
                    if exec_rc == 0 and "TRAIN_EXIT_CODE=0" in strip_ansi(out_all)
                    else False
                )
                outcome = classify_relay_round_output(exec_rc, out_all, has_remote_done_flag=remote_done)

                if outcome == "handover":
                    if round_elapsed < 60.0:
                        fast_handover_streak += 1
                        if fast_handover_streak >= 3:
                            print(
                                f"❌ [自动止损] 连续 {fast_handover_streak} 棒在 60s 内极速换棒 ({round_elapsed:.1f}s)，"
                                f"疑似配置异常，将在释放实例后中止接力循环！",
                                flush=True,
                            )
                            outcome = "fast_handover_abort"
                    else:
                        fast_handover_streak = 0
                elif outcome != "completed":
                    print(
                        f"❌ [自动止损] Round #{effective_round} 未正常发出交接信号或发生执行异常 (exit={exec_rc})，"
                        f"将在释放实例后终止接力以防止死循环消耗计算单元！",
                        flush=True,
                    )
            except Exception as err:
                print(f"❌ [自动止损] Round #{effective_round} 执行异常: {err}", flush=True)
                outcome = "exception"
            finally:
                if os.path.exists(ws_meta_local):
                    try:
                        os.remove(ws_meta_local)
                    except OSError:
                        pass
                print(f"[Round #{effective_round}] 正在释放本棒实例 ({args.session}) 并核验停机状态...", flush=True)
                stop_ok, stop_err = stop_and_verify_session(
                    colab_bin,
                    args.session,
                    max_retries=3,
                    retry_delay=2.0,
                    require_stop_rc_zero=True,
                )

            if outcome == "completed":
                print(f"[Round #{effective_round}] 训练全部完成！正在拉取最终权重与 results/ 到本地工作区...", flush=True)
                dl_ok, dl_path, dl_err = download_and_verify_final_weights(
                    rclone_bin=rclone_bin,
                    rclone_conf=rclone_conf,
                    gdrive_dir=effective_gdrive_dir,
                    ckpt_basename=ckpt_basename,
                    workspace_abs=workspace_abs,
                )
                write_local_relay_receipt(workspace_abs, True, dl_ok, dl_path, None if dl_ok else dl_err)
                if not stop_ok:
                    print(
                        f"❌ [停费核验告警] 云端训练已完成，但无法确认实例 ({args.session}) 已释放停费: {stop_err}",
                        flush=True,
                    )
                if not dl_ok:
                    print(
                        f"❌ [最终权重下载失败] 云端训练已完成，但最终权重未能落盘保存: {dl_err}（可重新运行命令执行补下载）",
                        flush=True,
                    )
                    return 1
                if not stop_ok:
                    return 1
                print(
                    f"🎉 最终模型已保存至 {dl_path}，训练结果目录：{os.path.join(workspace_abs, 'results')}！",
                    flush=True,
                )
                return 0

            if not stop_ok:
                print(
                    f"❌ [防重叠计费熔断] 无法确认旧实例 ({args.session}) 已停止释放 ({stop_err})！"
                    f"严禁开启下一棒实例，请立即执行 `colab status` / `colab stop -s {args.session}` 核验计费状态！",
                    flush=True,
                )
                return 1

            if outcome != "handover":
                return 1

            completed_round = effective_round
            effective_round += 1
            print(
                f"[Round #{completed_round}] 本棒安全交接完毕（旧实例已确认释放），"
                f"{HANDOVER_SLEEP_SECONDS:g} 秒后开启第 {effective_round} 棒...",
                flush=True,
            )
            if HANDOVER_SLEEP_SECONDS > 0:
                time.sleep(HANDOVER_SLEEP_SECONDS)
    finally:
        release_gdrive_dir_lock(effective_gdrive_dir, session_name)


if __name__ == "__main__":
    sys.exit(main())
