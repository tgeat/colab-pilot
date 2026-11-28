#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Colab Pilot — 深度学习上云前本机 4 阶段冒烟验证工具 (smoke_test_dl.py)
==========================================================================
在将训练脚本放入 Google Colab 之前，强制在本机执行 4 道验证关卡：
1. [语法与依赖编译] 内存级 AST 与字节码编译检查（不污染工作区 __pycache__）
2. [Pass #1 冒烟保存] 跑 1 个 Epoch/Step 后触发模拟换棒，验证前向/反向传播、断点文件落盘及遥测状态写入
3. [Pass #2 断点续训] 基于 Pass #1 断点续训第 2 个 Epoch，通过文件读取审计 + 损坏断点依赖探针严格验证断点被真实读取与加载，
   并校验最终轮完成后确实生成了非空完成标记文件（`training_done.flag` / `TRAINING_COMPLETED.flag`）
4. [Google Drive 连通性] 验证本机 rclone 与 rclone.conf 读写连通性，失败严格返回非零退出码
"""

import argparse
import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

READ_AUDIT_SITECUSTOMIZE = r'''import builtins
import io
import os

_audit_log = os.environ.get("SMOKE_AUDIT_LOG", "").strip()
if _audit_log:
    _orig_open = builtins.open
    _orig_io_open = io.open
    _orig_os_open = os.open

    def _record_read(path_obj):
        try:
            p = os.fspath(path_obj)
            if isinstance(p, bytes):
                p = p.decode("utf-8", errors="ignore")
            abs_p = os.path.abspath(p)
            if os.path.abspath(_audit_log) == abs_p:
                return
            with _orig_open(_audit_log, "a", encoding="utf-8") as af:
                af.write(abs_p + "\n")
        except Exception:
            pass

    def _hooked_open(file, mode="r", *args, **kwargs):
        f = _orig_open(file, mode, *args, **kwargs)
        if isinstance(mode, str) and ("r" in mode or "+" in mode) and not isinstance(file, int):
            _record_read(file)
        return f

    def _hooked_io_open(file, mode="r", *args, **kwargs):
        f = _orig_io_open(file, mode, *args, **kwargs)
        if isinstance(mode, str) and ("r" in mode or "+" in mode) and not isinstance(file, int):
            _record_read(file)
        return f

    def _hooked_os_open(path, flags, *args, **kwargs):
        fd = _orig_os_open(path, flags, *args, **kwargs)
        acc = flags & 3
        if acc in (os.O_RDONLY, os.O_RDWR):
            _record_read(path)
        return fd

    builtins.open = _hooked_open
    io.open = _hooked_io_open
    os.open = _hooked_os_open
'''


def check_python_syntax(filepath: str) -> None:
    with open(filepath, "rb") as f:
        source = f.read()
    tree = ast.parse(source, filename=filepath)
    compile(tree, filename=filepath, mode="exec")


def find_rclone() -> str:
    cand = shutil.which("rclone")
    if cand:
        return cand
    home_bin = os.path.expanduser("~/.local/bin/rclone.exe" if os.name == "nt" else "~/.local/bin/rclone")
    return home_bin if os.path.exists(home_bin) else ""


def find_rclone_conf() -> str:
    if os.environ.get("RCLONE_CONFIG"):
        return os.environ["RCLONE_CONFIG"]
    appdata = os.environ.get("APPDATA")
    if appdata:
        p = os.path.join(appdata, "rclone", "rclone.conf")
        if os.path.exists(p):
            return p
    return os.path.expanduser("~/.config/rclone/rclone.conf")


def _find_produced_checkpoints(tmpdir: str, ckpt_path: str) -> List[str]:
    found: List[str] = []
    if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        found.append(os.path.abspath(ckpt_path))
    for ts in glob.glob(os.path.join(tmpdir, "**", "checkpoint-*", "trainer_state.json"), recursive=True):
        if os.path.isfile(ts) and os.path.getsize(ts) > 0:
            found.append(os.path.abspath(ts))
    return found


def _find_done_flag_file(tmpdir: str, done_path: str) -> Optional[str]:
    candidates = [
        done_path,
        os.path.join(tmpdir, "TRAINING_COMPLETED.flag"),
        os.path.join(tmpdir, "training_done.flag"),
    ]
    for cand in candidates:
        if os.path.isfile(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Colab 深度学习串行接力 — 本机 Smoke Test 验证器")
    parser.add_argument("--script", default="train.py", help="要验证的训练脚本路径 (默认: train.py)")
    parser.add_argument("--python", default=sys.executable, help="用于执行本机 Smoke Test 的 Python 解释器")
    parser.add_argument("--gdrive-dir", default="gdrive:colab_relay", help="Google Drive 断点与结果目录")
    parser.add_argument("--skip-rclone-net", action="store_true", help="仅检查 rclone 配置文件，跳过在线请求")
    args = parser.parse_args(argv)

    script_abs = os.path.abspath(args.script)
    workspace_dir = os.path.dirname(script_abs)
    adapter_abs = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_relay_adapter.py")

    print("=" * 72)
    print("🚀 [Colab Pilot] 深度学习串行接力 — 本机 Smoke Test 4 关验证")
    print(f"   目标脚本: {script_abs}")
    print(f"   工作区  : {workspace_dir}")
    print("=" * 72)

    if not os.path.exists(script_abs):
        print(f"❌ [失败] 找不到训练脚本: {script_abs}")
        return 1

    # ── 第 1 关：语法与模块编译校验（纯内存编译，不产生 __pycache__） ──
    print("\n[1/4] 正在校验 Python 语法与接力适配器...")
    try:
        check_python_syntax(script_abs)
        if os.path.exists(adapter_abs):
            check_python_syntax(adapter_abs)
        print("  ✅ 第 1 关通过：训练脚本与接力适配器语法无误。")
    except Exception as e:
        print(f"  ❌ 第 1 关失败：语法错误 -> {e}")
        return 1

    # 在隔离临时目录中执行 Pass #1 与 Pass #2，通过环境变量驱动（绝不向用户脚本注入未知 CLI 参数）
    with tempfile.TemporaryDirectory(prefix="colab_smoke_") as tmpdir:
        if os.path.exists(adapter_abs):
            shutil.copy2(adapter_abs, os.path.join(tmpdir, "colab_relay_adapter.py"))

        audit_dir = os.path.join(tmpdir, "_smoke_audit")
        os.makedirs(audit_dir, exist_ok=True)
        with open(os.path.join(audit_dir, "sitecustomize.py"), "w", encoding="utf-8") as sf:
            sf.write(READ_AUDIT_SITECUSTOMIZE)

        ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
        done_path = os.path.join(tmpdir, "training_done.flag")
        state_path = os.path.join(tmpdir, ".colab_pilot_training_state.json")
        results_dir = os.path.join(tmpdir, "results")
        audit_log = os.path.join(tmpdir, "_pass2_read_audit.log")

        base_env = os.environ.copy()
        base_env.update({
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": (
                audit_dir
                + os.pathsep
                + workspace_dir
                + os.pathsep
                + tmpdir
                + os.pathsep
                + base_env.get("PYTHONPATH", "")
            ),
            "SMOKE_TEST": "1",
            "SMOKE_SKIP_RCLONE": "1",
            "TOTAL_EPOCHS": "2",
            "CHECKPOINT_PATH": ckpt_path,
            "DONE_FLAG_PATH": done_path,
            "TRAINING_STATE_FILE": state_path,
            "RESULTS_DIR": results_dir,
            "GDRIVE_DIR": args.gdrive_dir,
            "RELAY_GDRIVE_DIR": args.gdrive_dir,
        })

        # ── 第 2 关：Pass #1 (跑第 1 轮 + 断点保存 + 模拟换棒) ──
        print("\n[2/4] 正在执行 Smoke Pass #1（验证前向/反向传播 + Epoch 1 断点保存）...")
        env1 = base_env.copy()
        env1["SMOKE_PASS"] = "1"
        env1["RELAY_ROUND"] = "1"
        t0 = time.time()
        res1 = subprocess.run(
            [args.python, "-B", script_abs],
            cwd=tmpdir,
            env=env1,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if res1.returncode != 0:
            print("  ❌ 第 2 关失败：Pass #1 进程返回非零退出码")
            print(res1.stdout[-1000:])
            print(res1.stderr[-1000:])
            return 1

        produced_ckpts = _find_produced_checkpoints(tmpdir, ckpt_path)
        if not produced_ckpts:
            print(f"  ❌ 第 2 关失败：未生成非空断点文件 {ckpt_path}")
            print(res1.stdout[-800:])
            return 1

        if not os.path.exists(state_path):
            print(f"  ❌ 第 2 关失败：未生成遥测状态文件 {state_path}")
            return 1

        with open(state_path, "r", encoding="utf-8") as f:
            st1 = json.load(f)
        ep1 = st1.get("current_epoch", 0)
        loss1 = st1.get("loss")
        primary_ckpt = produced_ckpts[0]
        print(
            f"  ✅ 第 2 关通过 ({time.time() - t0:.1f}s)：已生成断点 ({os.path.basename(primary_ckpt)}, "
            f"{os.path.getsize(primary_ckpt)} B)，遥测状态 Epoch={ep1}, Loss={loss1}"
        )

        # 确保 Pass #2 前不存在预先残留的完成标记，以严格检验最终轮完成标记由 Pass #2 生成
        for pre_flag in (done_path, os.path.join(tmpdir, "TRAINING_COMPLETED.flag")):
            if os.path.exists(pre_flag):
                os.remove(pre_flag)

        # ── 第 3 关：Pass #2 (强制从 Pass #1 断点续训第 2 轮 + 读取审计 + 损坏探针 + 完成标记校验) ──
        print("\n[3/4] 正在执行 Smoke Pass #2（验证真实读取加载断点续训 Epoch 2 并生成完成标记）...")
        env2 = base_env.copy()
        env2["SMOKE_PASS"] = "2"
        env2["RELAY_ROUND"] = "2"
        env2["SMOKE_AUDIT_LOG"] = audit_log
        t1 = time.time()
        res2 = subprocess.run(
            [args.python, "-B", script_abs],
            cwd=tmpdir,
            env=env2,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        if res2.returncode != 0:
            print("  ❌ 第 3 关失败：Pass #2 续训报错")
            print(res2.stdout[-1000:])
            print(res2.stderr[-1000:])
            return 1

        # (a1) 验证 Pass #2 确实打开并读取了 Pass #1 生成的断点文件
        read_paths = set()
        if os.path.isfile(audit_log):
            with open(audit_log, "r", encoding="utf-8") as af:
                read_paths = {
                    os.path.normcase(os.path.abspath(ln.strip()))
                    for ln in af
                    if ln.strip()
                }
        expected_norm = {os.path.normcase(os.path.abspath(p)) for p in produced_ckpts}
        if not (read_paths & expected_norm):
            print(
                f"  ❌ 第 3 关失败：Pass #2 未读取第一轮产出的断点文件 ({primary_ckpt})！"
                f"请确保第二棒真正加载历史断点而非硬编码跳过。"
            )
            return 1

        # (a2) 损坏断点依赖性探针：若将断点替换为非法二进制垃圾字节后 Pass #2 仍正常返回 0，说明未真正反序列化依赖断点
        tamper_dir = os.path.join(tmpdir, "_tamper_probe")
        os.makedirs(tamper_dir, exist_ok=True)
        if os.path.exists(adapter_abs):
            shutil.copy2(adapter_abs, os.path.join(tamper_dir, "colab_relay_adapter.py"))
        tamper_ckpt = os.path.join(tamper_dir, os.path.basename(ckpt_path))
        with open(tamper_ckpt, "wb") as tf:
            tf.write(b"\x80\xfe\xffCORRUPTED_SMOKE_CHECKPOINT_PROBE")
        for prod_p in produced_ckpts:
            if "checkpoint-" in prod_p and prod_p.endswith("trainer_state.json"):
                rel_hf = os.path.relpath(prod_p, tmpdir)
                tamper_hf_ts = os.path.join(tamper_dir, rel_hf)
                os.makedirs(os.path.dirname(tamper_hf_ts), exist_ok=True)
                with open(tamper_hf_ts, "wb") as hff:
                    hff.write(b"\x80\xfe\xffCORRUPTED_HF_TRAINER_STATE")

        env_tamper = base_env.copy()
        env_tamper.update({
            "SMOKE_PASS": "2",
            "RELAY_ROUND": "2",
            "CHECKPOINT_PATH": tamper_ckpt,
            "DONE_FLAG_PATH": os.path.join(tamper_dir, "training_done.flag"),
            "TRAINING_STATE_FILE": os.path.join(tamper_dir, ".colab_pilot_training_state.json"),
            "RESULTS_DIR": os.path.join(tamper_dir, "results"),
        })
        res_tamper = subprocess.run(
            [args.python, "-B", script_abs],
            cwd=tamper_dir,
            env=env_tamper,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if res_tamper.returncode == 0:
            print(
                "  ❌ 第 3 关失败：将第一轮断点篡改为损坏数据后 Pass #2 仍返回退出码 0，"
                "说明脚本未真正解析或依赖第一轮断点内容！"
            )
            return 1

        # (a3) 验证训练进度递增
        with open(state_path, "r", encoding="utf-8") as f:
            st2 = json.load(f)
        ep2 = st2.get("current_epoch", 0)
        loss2 = st2.get("loss")
        if ep2 <= ep1:
            print(f"  ❌ 第 3 关失败：断点未生效！Pass #1 Epoch={ep1}，Pass #2 Epoch={ep2}（未递增）")
            return 1

        # (b) 验证最终轮完成后确实生成了非空完成标记文件
        verified_done_flag = _find_done_flag_file(tmpdir, done_path)
        if not verified_done_flag:
            print(
                "  ❌ 第 3 关失败：最终轮完成后未生成非空完成标记文件 "
                "(`training_done.flag` / `TRAINING_COMPLETED.flag`)！"
            )
            return 1

        print(
            f"  ✅ 第 3 关通过 ({time.time() - t1:.1f}s)：已核验读取加载断点并从 Epoch {ep1} 续训至 Epoch {ep2}，"
            f"Loss={loss2}，完成标记 ({os.path.basename(verified_done_flag)}) 已生成。"
        )

    # ── 第 4 关：rclone 与 Google Drive 凭证核验 ──
    print("\n[4/4] 正在核验本机 rclone 与 Google Drive 授权状态...")
    rclone_bin = find_rclone()
    rclone_conf = find_rclone_conf()
    if not rclone_bin or not os.path.exists(rclone_bin):
        print("  ❌ 第 4 关失败：未找到 rclone 可执行文件")
        return 1
    if not rclone_conf or not os.path.exists(rclone_conf) or os.path.getsize(rclone_conf) < 20:
        print(f"  ❌ 第 4 关失败：未找到有效的 rclone.conf ({rclone_conf})")
        return 1

    if not args.skip_rclone_net:
        remote_root = args.gdrive_dir.split(":")[0] + ":"
        rc_test = subprocess.run(
            [rclone_bin, "--config", rclone_conf, "--retries=5", "--low-level-retries=20", "about", "--", remote_root],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
        )
        if rc_test.returncode != 0:
            print(f"  ❌ 第 4 关失败：rclone 在线连通性探测返回非零 ({(rc_test.stderr or '').strip()[:240]})")
            return 1
        print(f"  ✅ 第 4 关通过：rclone 与 Google Drive ({args.gdrive_dir}) 读写就绪。")
    else:
        print(f"  ✅ 第 4 关通过：rclone 配置文件 ({rclone_conf}) 已就绪。")

    print("\n🎉 [ALL PASSED] 本机 Smoke Test 4 关全部通过！可以安全上传至 Google Colab 启动串行接力训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
