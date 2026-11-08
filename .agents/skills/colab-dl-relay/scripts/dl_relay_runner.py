#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 深度学习串行接力守护进程 (dl_relay_runner.py)
==========================================================================
由 Antigravity `colab-dl-relay` Skill 在本机 Smoke Test 通过后调用，负责：
1. 在 `~/.config/colab-cli/workspaces.json` 注册当前工作区映射，使 Colab Pilot 立即识别工作区
2. 循环驱动 `colab-cli` 执行多轮跨实例串行接力（先存后拆，零双倍计费）：
   - colab new -s <session> --gpu <gpu>
   - colab upload 上传 rclone.conf、colab_relay_adapter.py、.colab_workspace.json 与训练脚本
   - 云端安装 rclone 与依赖，从 Google Drive 拉取数据集与最新 checkpoint.pt
   - 运行训练，实时更新 `/content/.colab_pilot_training_state.json`
   - 满 22.5 小时收到 HANDOVER_REQUESTED -> colab stop 销毁旧机 -> 立即开新机进入下一棒
   - 收到 TRAINING_COMPLETED -> 下载最终权重与 results/ 到本地 -> colab stop 停止计费
"""

import argparse
import json
import os
import py_compile
import shutil
import subprocess
import sys
import time
from typing import List, Optional

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


def gdrive_has_done_flag(rclone_bin: str, rclone_conf: str, gdrive_dir: str) -> bool:
    res = run_cmd(
        [
            rclone_bin,
            "--config",
            rclone_conf,
            "--retries=5",
            "--low-level-retries=20",
            "lsf",
            f"{gdrive_dir.rstrip('/')}/training_done.flag",
        ],
        timeout=90,
    )
    return res.returncode == 0 and "training_done.flag" in res.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab 深度学习跨实例串行接力编排器")
    parser.add_argument("--session", "-s", required=True, help="colab-cli 会话名称 (建议与工作区同名)")
    parser.add_argument("--workspace", default=".", help="本地工作区目录")
    parser.add_argument("--script", default="train.py", help="主训练脚本文件名")
    parser.add_argument("--extra-files", nargs="*", default=[], help="需额外上传到 /content 的文件或目录")
    parser.add_argument("--gpu", default="T4", help="Colab 加速器类型: T4 / L4 / A100 / CPU")
    parser.add_argument("--high-mem", action="store_true", default=True, help="启用 High-RAM 规格")
    parser.add_argument("--gdrive-dir", default="gdrive:colab_relay", help="Google Drive 断点与结果保存路径")
    parser.add_argument("--gdrive-dataset-dir", default="", help="Google Drive 数据集路径 (可选)")
    parser.add_argument("--total-epochs", type=int, default=100, help="总训练轮次")
    parser.add_argument("--max-session-hours", type=float, default=22.5, help="单台实例最长运行小时数 (默认 22.5h)")
    parser.add_argument("--backup-interval-minutes", type=float, default=30.0, help="Google Drive 定时热备间隔 (分钟)")
    parser.add_argument("--reset-done-flag", action="store_true", help="启动第 1 棒时清除旧的 training_done.flag")
    parser.add_argument("--detach", action="store_true", help="在后台独立启动守护进程（关闭 Antigravity 不中断）")
    args = parser.parse_args()

    workspace_abs = os.path.abspath(args.workspace)
    script_abs = os.path.abspath(os.path.join(workspace_abs, args.script))
    adapter_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_relay_adapter.py")

    if not os.path.exists(script_abs):
        print(f"❌ 找不到训练脚本: {script_abs}", flush=True)
        return 1

    # 上云前快速语法检查（< 1s），避免带着语法错误分配 Colab 实例
    try:
        py_compile.compile(script_abs, doraise=True)
    except Exception as e:
        print(f"❌ 上云前语法检查未通过，已阻止分配 Colab 实例: {e}", flush=True)
        return 1

    # 如果请求后台独立守护模式，派生无窗口后台进程并立即返回
    if args.detach and os.environ.get("DL_RELAY_DETACHED") != "1":
        log_file = os.path.join(workspace_abs, ".colab_dl_relay.log")
        pid_file = os.path.join(workspace_abs, ".colab_dl_relay.pid")
        cmd = [sys.executable, os.path.abspath(__file__)] + [a for a in sys.argv[1:] if a != "--detach"]
        env = os.environ.copy()
        env["DL_RELAY_DETACHED"] = "1"
        env["PYTHONUTF8"] = "1"
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        out_f = open(log_file, "a", encoding="utf-8")
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
                f"{args.gdrive_dir.rstrip('/')}/training_done.flag",
            ],
            timeout=90,
        )

    round_num = 0
    max_sec = int(args.max_session_hours * 3600)
    bk_sec = int(args.backup_interval_minutes * 60)

    while True:
        round_num += 1
        print(f"\n========== [Round #{round_num}] 串行接力检查 ({args.session}) ==========", flush=True)

        if round_num > 1 and gdrive_has_done_flag(rclone_bin, rclone_conf, args.gdrive_dir):
            print("🎉 检测到 Google Drive 已存在 training_done.flag，全部训练已完成！", flush=True)
            break

        # 1. 确保同名旧会话已彻底释放（先拆后建，绝不双倍计费）
        run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)

        # 2. 创建新 Colab 实例
        new_cmd = [colab_bin, "new", "-s", args.session, "--gpu", args.gpu]
        if args.high_mem:
            new_cmd.append("--high-mem")
        print(f"[Round #{round_num}] 正在分配 Colab 实例: {' '.join(new_cmd)}", flush=True)
        res_new = run_cmd(new_cmd, timeout=300)
        if res_new.returncode != 0:
            print(f"[Round #{round_num}] 分配实例失败: {res_new.stderr or res_new.stdout}，30 秒后重试...", flush=True)
            time.sleep(30)
            continue

        try:
            # 3. 上传工作区标记、rclone.conf、接力适配器与训练脚本
            ws_meta_local = os.path.join(workspace_abs, ".colab_workspace.json")
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

            run_cmd([colab_bin, "upload", "-s", args.session, ws_meta_local, "/content/.colab_workspace.json"], timeout=90)
            run_cmd([colab_bin, "upload", "-s", args.session, rclone_conf, "/content/rclone.conf"], timeout=90)
            if os.path.exists(adapter_abs):
                run_cmd([colab_bin, "upload", "-s", args.session, adapter_abs, "/content/colab_relay_adapter.py"], timeout=90)
            run_cmd([colab_bin, "upload", "-s", args.session, script_abs, f"/content/{os.path.basename(script_abs)}"], timeout=120)

            req_txt = os.path.join(workspace_abs, "requirements.txt")
            if os.path.exists(req_txt):
                run_cmd([colab_bin, "upload", "-s", args.session, req_txt, "/content/requirements.txt"], timeout=90)

            for ef in args.extra_files:
                ef_abs = os.path.abspath(os.path.join(workspace_abs, ef))
                if os.path.exists(ef_abs):
                    run_cmd([colab_bin, "upload", "-s", args.session, ef_abs, f"/content/{os.path.basename(ef_abs)}"], timeout=180)

            # 4. 云端启动脚本（安装 rclone/依赖 -> 执行训练并通过 IPython sys.stdout 回传实时日志）
            reset_flag = "1" if (round_num == 1 and args.reset_done_flag) else "0"
            cloud_boot_py = f"""
import os, shutil, subprocess, sys
os.chdir("/content")
if not shutil.which("rclone"):
    subprocess.run("curl -fsSL https://rclone.org/install.sh | sudo bash", shell=True, check=False)
if os.path.exists("/content/requirements.txt") and open("/content/requirements.txt").read().strip():
    subprocess.run(["uv", "pip", "install", "--system", "-r", "/content/requirements.txt"], check=False)

env = os.environ.copy()
env.update({{
    "COLAB_WORKSPACE": {json.dumps(workspace_abs)},
    "GDRIVE_DIR": {json.dumps(args.gdrive_dir)},
    "GDRIVE_DATASET_DIR": {json.dumps(args.gdrive_dataset_dir)},
    "TOTAL_EPOCHS": "{args.total_epochs}",
    "MAX_SESSION_SECONDS": "{max_sec}",
    "BACKUP_INTERVAL_SECONDS": "{bk_sec}",
    "RELAY_ROUND": "{round_num}",
    "RESET_DONE_FLAG": "{reset_flag}",
}})
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/{os.path.basename(script_abs)}"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
for line in proc.stdout:
    print(line, end="", flush=True)
rc = proc.wait()
print(f"TRAIN_EXIT_CODE={{rc}}", flush=True)
"""
            print(f"[Round #{round_num}] 云端训练已启动，正在流式监听进度...", flush=True)
            exec_timeout = max(120, max_sec + 120)
            try:
                res_exec = run_cmd(
                    [colab_bin, "exec", "-s", args.session, "--timeout", str(exec_timeout)],
                    timeout=exec_timeout + 30,
                    stdin_text=cloud_boot_py,
                )
                out_all = (res_exec.stdout or "") + "\n" + (res_exec.stderr or "")
            except subprocess.TimeoutExpired as te:
                out_all = ((te.stdout or "") if isinstance(te.stdout, str) else "") + "\nTIMEOUT_EXPIRED"
                print(f"[Round #{round_num}] ⚠️ colab exec 响应超时 ({exec_timeout}s)，准备检查网盘断点或换机重连...", flush=True)
            print(out_all[-1500:], flush=True)

            completed = "TRAINING_COMPLETED" in out_all or gdrive_has_done_flag(rclone_bin, rclone_conf, args.gdrive_dir)
            if completed:
                print(f"[Round #{round_num}] 训练全部完成！正在拉取最终权重与 results/ 到本地工作区...", flush=True)
                local_ckpt = os.path.join(workspace_abs, "checkpoint_final.pt")
                local_results = os.path.join(workspace_abs, "results")
                os.makedirs(local_results, exist_ok=True)
                run_cmd(
                    [
                        rclone_bin,
                        "--config",
                        rclone_conf,
                        "--retries=5",
                        "--low-level-retries=20",
                        "copyto",
                        f"{args.gdrive_dir.rstrip('/')}/checkpoint.pt",
                        local_ckpt,
                    ],
                    timeout=300,
                )
                run_cmd(
                    [
                        rclone_bin,
                        "--config",
                        rclone_conf,
                        "--retries=5",
                        "--low-level-retries=20",
                        "copy",
                        f"{args.gdrive_dir.rstrip('/')}/results",
                        local_results,
                    ],
                    timeout=300,
                )
                run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)
                print(f"🎉 最终模型已保存至 {local_ckpt}，训练结果目录：{local_results}，云端实例已释放！", flush=True)
                break

            # 检查是否未通过 Colab 云端真机首步冒烟（例如 CUDA OOM、缺少依赖、数据集错误或网盘无法写入）
            if "TIMEOUT_EXPIRED" in out_all and "CLOUD_SMOKE_PASSED" not in out_all:
                print(
                    f"⚠️ [自动换机] Round #{round_num} 遇到 Colab WebSocket 通道假死，正在释放实例并换机继续接力...",
                    flush=True,
                )
            elif "CLOUD_SMOKE_PASSED" not in out_all and "HANDOVER_REQUESTED" not in out_all:
                print(
                    f"❌ [自动止损] Round #{round_num} 未通过 Colab 云端真机首步冒烟（可能发生 CUDA OOM、依赖缺失或数据异常）！\n"
                    f"   正在立即执行 `colab stop -s {args.session}` 销毁实例停止计费，避免空挂消耗 CU。",
                    flush=True,
                )
                run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)
                return 1
        finally:
            # 无论正常换棒还是异常中断，本轮结束必ず释放旧实例停止计费
            print(f"[Round #{round_num}] 正在释放本棒实例 ({args.session})...", flush=True)
            run_cmd([colab_bin, "stop", "-s", args.session], timeout=120)

        print(f"[Round #{round_num}] 本棒安全交接完毕，10 秒后开启下一棒...", flush=True)
        time.sleep(10)

    return 0


if __name__ == "__main__":
    sys.exit(main())
