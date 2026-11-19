#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 深度学习串行接力守护进程 (dl_relay_runner.py)
==========================================================================
由 Antigravity `colab-dl-relay` Skill 在本机 Smoke Test 通过后调用，负责：
1. 在 `~/.config/colab-cli/workspaces.json` 注册当前工作区映射，使 Colab Pilot 立即识别工作区
2. 循环驱动 `colab-cli` 执行多轮跨实例串行接力（先存后拆，零双倍计费）
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import List, Optional

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
COLAB_UPDATE_BANNER_RE = re.compile(
    r"^\s*\[colab\]\s*(?:A new version of Colab CLI is available|Run 'colab update'|To silence this check).*$",
    re.IGNORECASE,
)

# Google Colab `/tun/m/assign` 仅提供单一默认内存规格的加速器（传入 `shape=hm` / `--high-mem` 会触发 HTTP 400）
SINGLE_SHAPE_ACCELERATORS = {"G4", "H100", "L4", "V5E1", "V6E1"}
TPU_ACCELERATORS = {"V5E1", "V6E1"}


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text or "")


def strip_colab_cli_banners(text: str) -> str:
    """剥离 google-colab-cli v0.8.0+ 的版本更新提示横幅与 ANSI 控制序列，避免干扰输出解析。"""
    clean = strip_ansi(text or "")
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


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


def register_workspace(session_name: str, workspace_dir: str, script_name: str, description: str) -> None:
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


def upload_checked(colab_bin: str, session: str, local_path: str, remote_path: str, timeout: int = 120) -> None:
    res = run_cmd([colab_bin, "upload", "-s", session, local_path, remote_path], timeout=timeout)
    if res.returncode != 0:
        err = strip_colab_cli_banners((res.stderr or "") + "\n" + (res.stdout or ""))[:300]
        raise RuntimeError(f"colab upload 失败 ({local_path} -> {remote_path}): {err}")


def gdrive_has_done_flag(rclone_bin: str, rclone_conf: str, gdrive_dir: str) -> bool:
    res = run_cmd(
        [
            rclone_bin,
            "--config",
            rclone_conf,
            "--retries=5",
            "--low-level-retries=20",
            "lsf",
            "--",
            f"{gdrive_dir.rstrip('/')}/training_done.flag",
        ],
        timeout=90,
    )
    return res.returncode == 0 and "training_done.flag" in res.stdout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab 深度学习跨实例串行接力编排器")
    parser.add_argument("--session", "-s", required=True, help="colab-cli 会话名称 (建议与工作区同名)")
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
    parser.add_argument("--gdrive-dir", default="gdrive:colab_relay", help="Google Drive 断点与结果保存路径")
    parser.add_argument("--gdrive-dataset-dir", default="", help="Google Drive 数据集路径 (可选)")
    parser.add_argument("--total-epochs", type=int, default=100, help="总训练轮次")
    parser.add_argument("--max-session-hours", type=float, default=22.5, help="单台实例最长运行小时数 (默认 22.5h)")
    parser.add_argument("--backup-interval-minutes", type=float, default=30.0, help="Google Drive 定时热备间隔 (分钟)")
    parser.add_argument("--reset-done-flag", action="store_true", help="启动第 1 棒时清除旧的 training_done.flag")
    parser.add_argument("--detach", action="store_true", help="在后台独立启动守护进程（关闭 Antigravity 不中断）")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

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
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:] if a != "--detach"]
        env = os.environ.copy()
        env["DL_RELAY_DETACHED"] = "1"
        env["PYTHONUTF8"] = "1"
        kwargs = {}
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

    colab_bin = find_bin("colab")
    rclone_bin = find_bin("rclone")
    rclone_conf = find_rclone_conf()
    ckpt_basename = os.path.basename(args.checkpoint_path)

    register_workspace(
        session_name=args.session,
        workspace_dir=workspace_abs,
        script_name=args.script,
        description=f"DL Serial Relay ({args.gpu}) -> {args.gdrive_dir}",
    )

    if args.reset_done_flag:
        run_cmd(
            [
                rclone_bin,
                "--config",
                rclone_conf,
                "--retries=5",
                "--low-level-retries=20",
                "deletefile",
                "--",
                f"{args.gdrive_dir.rstrip('/')}/training_done.flag",
            ],
            timeout=90,
        )

    round_num = 0
    fast_handover_streak = 0
    max_sec = int(args.max_session_hours * 3600)
    bk_sec = int(args.backup_interval_minutes * 60)

    while True:
        round_num += 1
        print(f"\n========== [Round #{round_num}] 串行接力检查 ({args.session}) ==========", flush=True)

        if (round_num > 1 or not args.reset_done_flag) and gdrive_has_done_flag(
            rclone_bin, rclone_conf, args.gdrive_dir
        ):
            print("🎉 检测到 Google Drive 已存在 training_done.flag，全部训练已完成！", flush=True)
            break

        run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)

        res_new = allocate_colab_instance(
            colab_bin=colab_bin,
            session_name=args.session,
            accelerator=args.gpu,
            high_mem=args.high_mem,
            timeout=300,
            log_prefix=f"[Round #{round_num}]",
        )
        if res_new.returncode != 0:
            err_msg = format_colab_alloc_error(args.gpu, (res_new.stderr or "") + "\n" + (res_new.stdout or ""))
            print(f"[Round #{round_num}] 分配实例失败: {err_msg}，30 秒后重试...", flush=True)
            time.sleep(30)
            continue

        ws_meta_local = os.path.join(workspace_abs, ".colab_workspace.json")
        try:
            with open(ws_meta_local, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "name": os.path.basename(workspace_abs),
                        "path": workspace_abs,
                        "description": f"深度学习串行接力 Round #{round_num} ({args.script})",
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

            reset_flag = "1" if (round_num == 1 and args.reset_done_flag) else "0"
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
    "GDRIVE_DIR": {json.dumps(args.gdrive_dir)},
    "GDRIVE_DATASET_DIR": {json.dumps(args.gdrive_dataset_dir)},
    "CHECKPOINT_PATH": {json.dumps(f"/content/{ckpt_basename}")},
    "TOTAL_EPOCHS": "{args.total_epochs}",
    "MAX_SESSION_SECONDS": "{max_sec}",
    "BACKUP_INTERVAL_SECONDS": "{bk_sec}",
    "RELAY_ROUND": "{round_num}",
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
            print(f"[Round #{round_num}] 云端训练已启动，正在流式监听进度...", flush=True)
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
                print(f"[Round #{round_num}] ⚠️ colab exec 响应超时 ({exec_timeout}s)，准备检查网盘断点...", flush=True)
            round_elapsed = time.monotonic() - round_start_ts
            print(out_all[-1500:], flush=True)

            remote_done = (
                gdrive_has_done_flag(rclone_bin, rclone_conf, args.gdrive_dir)
                if exec_rc == 0 and "TRAIN_EXIT_CODE=0" in strip_ansi(out_all)
                else False
            )
            outcome = classify_relay_round_output(exec_rc, out_all, has_remote_done_flag=remote_done)
            if outcome == "completed":
                print(f"[Round #{round_num}] 训练全部完成！正在拉取最终权重与 results/ 到本地工作区...", flush=True)
                stem, ext = os.path.splitext(ckpt_basename)
                local_ckpt = os.path.join(workspace_abs, f"{stem}_final{ext or '.pt'}")
                local_results = os.path.join(workspace_abs, "results")
                os.makedirs(local_results, exist_ok=True)
                rc1 = run_cmd(
                    [
                        rclone_bin,
                        "--config",
                        rclone_conf,
                        "--retries=5",
                        "--low-level-retries=20",
                        "copyto",
                        "--",
                        f"{args.gdrive_dir.rstrip('/')}/{ckpt_basename}",
                        local_ckpt,
                    ],
                    timeout=300,
                )
                rc2 = run_cmd(
                    [
                        rclone_bin,
                        "--config",
                        rclone_conf,
                        "--retries=5",
                        "--low-level-retries=20",
                        "copy",
                        "--",
                        f"{args.gdrive_dir.rstrip('/')}/results",
                        local_results,
                    ],
                    timeout=300,
                )
                if rc1.returncode != 0:
                    print(f"⚠️ 拉取最终断点警告: {(rc1.stderr or '').strip()[:240]}", flush=True)
                if rc2.returncode != 0:
                    print(f"⚠️ 拉取 results 目录警告: {(rc2.stderr or '').strip()[:240]}", flush=True)
                print(f"🎉 最终模型已保存至 {local_ckpt}，训练结果目录：{local_results}！", flush=True)
                break

            if outcome != "handover":
                print(
                    f"❌ [自动止损] Round #{round_num} 未正常发出交接信号或发生执行异常 (exit={exec_rc})，"
                    f"将在释放实例后终止接力以防止死循环消耗计算单元！",
                    flush=True,
                )
                return 1

            if round_elapsed < 60.0:
                fast_handover_streak += 1
                if fast_handover_streak >= 3:
                    print(
                        f"❌ [自动止损] 连续 {fast_handover_streak} 棒在 60s 内极速换棒 ({round_elapsed:.1f}s)，"
                        f"疑似配置异常，已中止接力循环！",
                        flush=True,
                    )
                    return 1
            else:
                fast_handover_streak = 0
        except Exception as err:
            print(f"❌ [自动止损] Round #{round_num} 执行异常: {err}", flush=True)
            return 1
        finally:
            if os.path.exists(ws_meta_local):
                try:
                    os.remove(ws_meta_local)
                except OSError:
                    pass
            print(f"[Round #{round_num}] 正在释放本棒实例 ({args.session})...", flush=True)
            run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)

        print(f"[Round #{round_num}] 本棒安全交接完毕，10 秒后开启下一棒...", flush=True)
        time.sleep(10)

    return 0


if __name__ == "__main__":
    sys.exit(main())
